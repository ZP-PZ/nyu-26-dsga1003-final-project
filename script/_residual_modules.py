"""Shared residual-weighting modules for stage-2 training scripts."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM


def count_lower_triangular_entries(size: int) -> int:
    """Return the number of entries in one lower-triangular square matrix."""

    return size * (size + 1) // 2


class PromptConditionedMLP(nn.Module):
    """A small dense MLP with two hidden layers for prompt conditioning."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc3(hidden_states)
        return hidden_states


class ResidualScalingModelBase(nn.Module):
    """Shared wrapper for frozen Qwen with layer-wise residual scaling hooks."""

    def __init__(self, base_model: AutoModelForCausalLM) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_layers = self.base_model.config.num_hidden_layers
        self.current_layer_scales: torch.Tensor | None = None
        self._hook_handles = []
        self._freeze_base_model()
        self._register_layer_hooks()

    def _freeze_base_model(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    def _register_layer_hooks(self) -> None:
        for layer_index, decoder_layer in enumerate(self.base_model.model.layers):
            handle = decoder_layer.register_forward_hook(
                self._make_layer_hook(layer_index),
            )
            self._hook_handles.append(handle)

    def _make_layer_hook(self, layer_index: int):
        def hook(_module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
            if self.current_layer_scales is None:
                return output

            residual_input = inputs[0]
            layer_scales = self.current_layer_scales
            if layer_scales.dim() == 1:
                scale = layer_scales[layer_index].view(1, 1, 1)
            else:
                scale = layer_scales[:, layer_index].view(layer_scales.shape[0], 1, 1)

            scale = scale.to(device=output.device, dtype=output.dtype)
            delta = output - residual_input
            return residual_input + scale * delta

        return hook

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        full_state = self.state_dict()
        return {
            name: tensor.detach().cpu()
            for name, tensor in full_state.items()
            if name in trainable_names
        }

    def _forward_with_scales(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        layer_scales: torch.Tensor,
    ):
        self.current_layer_scales = layer_scales
        try:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
        finally:
            self.current_layer_scales = None


class StaticWriteStrengthModel(ResidualScalingModelBase):
    """Frozen base model with one learned write-strength scale per decoder layer."""

    def __init__(self, base_model: AutoModelForCausalLM) -> None:
        super().__init__(base_model=base_model)
        self.raw_layer_scales = nn.Parameter(torch.zeros(self.num_layers))

    def get_layer_scales(self) -> torch.Tensor:
        return 1.0 + torch.tanh(self.raw_layer_scales)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        **_: dict,
    ):
        return self._forward_with_scales(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            layer_scales=self.get_layer_scales(),
        )


class PromptConditionedWriteStrengthModel(ResidualScalingModelBase):
    """Frozen base model with prompt-conditioned residual write-strength scales."""

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        mlp_hidden_size: int,
        rms_norm_eps: float | None = None,
    ) -> None:
        super().__init__(base_model=base_model)
        del rms_norm_eps
        self.prompt_representation_cache: dict[str, torch.Tensor] = {}
        self.conditioner = PromptConditionedMLP(
            input_size=self.base_model.config.hidden_size * 2,
            hidden_size=mlp_hidden_size,
            output_size=self.num_layers,
        )

    @staticmethod
    def _pool_prompt_hidden_states(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        last_token_hidden = hidden_states[:, -1, :]
        if attention_mask is None:
            mean_pooled_hidden = hidden_states.mean(dim=1)
        else:
            weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            weighted_sum = (hidden_states * weights).sum(dim=1)
            token_count = weights.sum(dim=1).clamp_min(1.0)
            mean_pooled_hidden = weighted_sum / token_count
        return torch.cat([last_token_hidden, mean_pooled_hidden], dim=-1)

    def get_prompt_representation(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        example_id: list[str] | None = None,
    ) -> torch.Tensor:
        if example_id is None:
            with torch.inference_mode():
                prompt_outputs = self.base_model.model(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    use_cache=False,
                )
            return self._pool_prompt_hidden_states(
                hidden_states=prompt_outputs.last_hidden_state,
                attention_mask=prompt_attention_mask,
            )

        cached_representations: list[torch.Tensor | None] = [None] * len(example_id)
        missing_indices: list[int] = []
        for index, current_example_id in enumerate(example_id):
            cached_representation = self.prompt_representation_cache.get(current_example_id)
            if cached_representation is None:
                missing_indices.append(index)
            else:
                cached_representations[index] = cached_representation

        if missing_indices:
            missing_index_tensor = torch.tensor(
                missing_indices,
                device=prompt_input_ids.device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                prompt_outputs = self.base_model.model(
                    input_ids=prompt_input_ids.index_select(0, missing_index_tensor),
                    attention_mask=(
                        None
                        if prompt_attention_mask is None
                        else prompt_attention_mask.index_select(0, missing_index_tensor)
                    ),
                    use_cache=False,
                )
            missing_representations = self._pool_prompt_hidden_states(
                hidden_states=prompt_outputs.last_hidden_state,
                attention_mask=(
                    None
                    if prompt_attention_mask is None
                    else prompt_attention_mask.index_select(0, missing_index_tensor)
                ),
            ).detach().cpu()
            for local_index, batch_index in enumerate(missing_indices):
                current_example_id = example_id[batch_index]
                cached_representation = missing_representations[local_index].clone()
                self.prompt_representation_cache[current_example_id] = cached_representation
                cached_representations[batch_index] = cached_representation

        stacked_representations = torch.stack(cached_representations, dim=0)
        return stacked_representations.to(
            device=prompt_input_ids.device,
            dtype=self.conditioner.fc1.weight.dtype,
        )

    def get_layer_scales(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        example_id: list[str] | None = None,
    ) -> torch.Tensor:
        prompt_representation = self.get_prompt_representation(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            example_id=example_id,
        )
        raw_scales = self.conditioner(prompt_representation)
        return 1.0 + 0.1 * raw_scales

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        prompt_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        example_id: list[str] | None = None,
        **_: dict,
    ):
        layer_scales = self.get_layer_scales(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            example_id=example_id,
        )
        return self._forward_with_scales(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            layer_scales=layer_scales,
        )


class ResidualReaggregationModelBase(nn.Module):
    """Shared wrapper for frozen Qwen with residual-stream re-aggregation hooks."""

    def __init__(self, base_model: AutoModelForCausalLM) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_layers = self.base_model.config.num_hidden_layers
        self.current_layer_weights: torch.Tensor | None = None
        self.current_embedding_stream: torch.Tensor | None = None
        self.current_layer_writes: list[torch.Tensor] = []
        self._hook_handles = []
        self._freeze_base_model()
        self._register_layer_hooks()

    def _freeze_base_model(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    def _register_layer_hooks(self) -> None:
        for layer_index, decoder_layer in enumerate(self.base_model.model.layers):
            handle = decoder_layer.register_forward_hook(
                self._make_layer_hook(layer_index),
            )
            self._hook_handles.append(handle)

    def _make_layer_hook(self, layer_index: int):
        def hook(_module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
            if self.current_layer_weights is None:
                return output

            residual_input = inputs[0]
            if self.current_embedding_stream is None:
                self.current_embedding_stream = residual_input
                self.current_layer_writes = []

            current_write = output - residual_input
            self.current_layer_writes.append(current_write)
            return self._reaggregate_layer_output(layer_index=layer_index, output=output)

        return hook

    def _reaggregate_layer_output(
        self,
        layer_index: int,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self.current_embedding_stream is None:
            raise RuntimeError("Missing embedding stream during residual re-aggregation.")

        weights = self.current_layer_weights
        if weights is None:
            raise RuntimeError("Missing layer weights during residual re-aggregation.")

        write_tensor = torch.stack(self.current_layer_writes, dim=0)
        if weights.dim() == 2:
            layer_weights = weights[layer_index, : layer_index + 1].view(-1, 1, 1, 1)
        else:
            layer_weights = (
                weights[:, layer_index, : layer_index + 1]
                .transpose(0, 1)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

        layer_weights = layer_weights.to(device=output.device, dtype=output.dtype)
        aggregated_writes = (layer_weights * write_tensor).sum(dim=0)
        return self.current_embedding_stream + aggregated_writes

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        full_state = self.state_dict()
        return {
            name: tensor.detach().cpu()
            for name, tensor in full_state.items()
            if name in trainable_names
        }

    def _forward_with_reaggregation(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        layer_weights: torch.Tensor,
    ):
        self.current_layer_weights = layer_weights
        self.current_embedding_stream = None
        self.current_layer_writes = []
        try:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
        finally:
            self.current_layer_weights = None
            self.current_embedding_stream = None
            self.current_layer_writes = []


class StaticResidualStreamReaggregationModel(ResidualReaggregationModelBase):
    """Frozen base model with one learned write weight per past layer and target layer."""

    def __init__(self, base_model: AutoModelForCausalLM) -> None:
        super().__init__(base_model=base_model)
        self.num_reaggregation_weights = count_lower_triangular_entries(self.num_layers)
        self.raw_lower_triangular_weights = nn.Parameter(
            torch.ones(self.num_reaggregation_weights)
        )

    def get_layer_weights(self) -> torch.Tensor:
        layer_weights = self.raw_lower_triangular_weights.new_zeros(
            self.num_layers,
            self.num_layers,
        )
        row_indices, col_indices = torch.tril_indices(
            row=self.num_layers,
            col=self.num_layers,
            offset=0,
            device=self.raw_lower_triangular_weights.device,
        )
        layer_weights[row_indices, col_indices] = self.raw_lower_triangular_weights
        return layer_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        **_: dict,
    ):
        return self._forward_with_reaggregation(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            layer_weights=self.get_layer_weights(),
        )


class PromptConditionedResidualStreamReaggregationModel(ResidualReaggregationModelBase):
    """Frozen base model with prompt-conditioned residual-stream re-aggregation weights."""

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        mlp_hidden_size: int,
        rms_norm_eps: float | None = None,
    ) -> None:
        super().__init__(base_model=base_model)
        del rms_norm_eps
        self.prompt_representation_cache: dict[str, torch.Tensor] = {}
        self.num_reaggregation_weights = count_lower_triangular_entries(self.num_layers)
        self.conditioner = PromptConditionedMLP(
            input_size=self.base_model.config.hidden_size * 2,
            hidden_size=mlp_hidden_size,
            output_size=self.num_reaggregation_weights,
        )

    @staticmethod
    def _pool_prompt_hidden_states(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        last_token_hidden = hidden_states[:, -1, :]
        if attention_mask is None:
            mean_pooled_hidden = hidden_states.mean(dim=1)
        else:
            weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            weighted_sum = (hidden_states * weights).sum(dim=1)
            token_count = weights.sum(dim=1).clamp_min(1.0)
            mean_pooled_hidden = weighted_sum / token_count
        return torch.cat([last_token_hidden, mean_pooled_hidden], dim=-1)

    def get_prompt_representation(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        example_id: list[str] | None = None,
    ) -> torch.Tensor:
        if example_id is None:
            with torch.inference_mode():
                prompt_outputs = self.base_model.model(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    use_cache=False,
                )
            return self._pool_prompt_hidden_states(
                hidden_states=prompt_outputs.last_hidden_state,
                attention_mask=prompt_attention_mask,
            )

        cached_representations: list[torch.Tensor | None] = [None] * len(example_id)
        missing_indices: list[int] = []
        for index, current_example_id in enumerate(example_id):
            cached_representation = self.prompt_representation_cache.get(current_example_id)
            if cached_representation is None:
                missing_indices.append(index)
            else:
                cached_representations[index] = cached_representation

        if missing_indices:
            missing_index_tensor = torch.tensor(
                missing_indices,
                device=prompt_input_ids.device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                prompt_outputs = self.base_model.model(
                    input_ids=prompt_input_ids.index_select(0, missing_index_tensor),
                    attention_mask=(
                        None
                        if prompt_attention_mask is None
                        else prompt_attention_mask.index_select(0, missing_index_tensor)
                    ),
                    use_cache=False,
                )
            missing_representations = self._pool_prompt_hidden_states(
                hidden_states=prompt_outputs.last_hidden_state,
                attention_mask=(
                    None
                    if prompt_attention_mask is None
                    else prompt_attention_mask.index_select(0, missing_index_tensor)
                ),
            ).detach().cpu()
            for local_index, batch_index in enumerate(missing_indices):
                current_example_id = example_id[batch_index]
                cached_representation = missing_representations[local_index].clone()
                self.prompt_representation_cache[current_example_id] = cached_representation
                cached_representations[batch_index] = cached_representation

        stacked_representations = torch.stack(cached_representations, dim=0)
        return stacked_representations.to(
            device=prompt_input_ids.device,
            dtype=self.conditioner.fc1.weight.dtype,
        )

    def get_layer_weights(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        example_id: list[str] | None = None,
    ) -> torch.Tensor:
        prompt_representation = self.get_prompt_representation(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            example_id=example_id,
        )
        raw_weight_offsets = self.conditioner(prompt_representation)
        batch_size = raw_weight_offsets.shape[0]
        layer_weights = raw_weight_offsets.new_zeros(
            batch_size,
            self.num_layers,
            self.num_layers,
        )
        row_indices, col_indices = torch.tril_indices(
            row=self.num_layers,
            col=self.num_layers,
            offset=0,
            device=raw_weight_offsets.device,
        )
        layer_weights[:, row_indices, col_indices] = 1.0 + 0.1 * raw_weight_offsets
        return layer_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        prompt_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        example_id: list[str] | None = None,
        **_: dict,
    ):
        layer_weights = self.get_layer_weights(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            example_id=example_id,
        )
        return self._forward_with_reaggregation(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            layer_weights=layer_weights,
        )


# Backward-compatible aliases for the original write-strength variants.
StaticResidualModel = StaticWriteStrengthModel
PromptConditionedResidualModel = PromptConditionedWriteStrengthModel
