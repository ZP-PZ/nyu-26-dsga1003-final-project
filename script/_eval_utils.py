"""Shared evaluation helpers for stage-3 scripts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from _residual_modules import (
    PromptConditionedResidualStreamReaggregationModel,
    PromptConditionedWriteStrengthModel,
    StaticResidualStreamReaggregationModel,
    StaticWriteStrengthModel,
)
from _training_utils import (
    configure_runtime,
    create_autocast_context,
    create_train_dataloader,
    detect_device,
    load_frozen_base_model,
    move_batch_to_device,
    select_model_dtype,
)


DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"


def load_checkpoint(checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
    )
    if "trainable_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint missing trainable_state_dict: {checkpoint_path}")
    return checkpoint


def load_trainable_parameters(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    named_parameters = dict(model.named_parameters())
    missing_names: list[str] = []

    with torch.no_grad():
        for name, parameter in named_parameters.items():
            if not parameter.requires_grad:
                continue
            tensor = state_dict.get(name)
            if tensor is None:
                missing_names.append(name)
                continue
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))

    if missing_names:
        raise ValueError(
            "Checkpoint is missing trainable parameters: "
            + ", ".join(missing_names)
        )


def create_eval_dataloader(
    arrow_path: str | Path,
    batch_size: int,
    device: torch.device,
):
    return create_train_dataloader(
        arrow_path=arrow_path,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )


def create_runtime() -> tuple[torch.device, torch.dtype]:
    device = detect_device()
    configure_runtime(device)
    model_dtype = select_model_dtype(device)
    return device, model_dtype


def load_frozen_base_for_eval(
    model_dir: str | Path,
    device: torch.device,
    model_dtype: torch.dtype,
    attn_implementation: str,
):
    model = load_frozen_base_model(
        model_dir=model_dir,
        device=device,
        model_dtype=model_dtype,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return model


def load_static_write_strength_model_for_eval(
    checkpoint_path: str | Path,
    model_dir: str | Path | None,
    device: torch.device,
    model_dtype: torch.dtype,
    attn_implementation: str | None,
):
    checkpoint = load_checkpoint(checkpoint_path)
    train_config = checkpoint["train_config"]
    resolved_model_dir = model_dir or train_config["base_model_dir"]
    resolved_attn = attn_implementation or train_config.get(
        "attn_implementation",
        DEFAULT_ATTN_IMPLEMENTATION,
    )

    base_model = load_frozen_base_model(
        model_dir=resolved_model_dir,
        device=device,
        model_dtype=model_dtype,
        attn_implementation=resolved_attn,
    )
    model = StaticWriteStrengthModel(base_model=base_model).to(device)
    load_trainable_parameters(model, checkpoint["trainable_state_dict"])
    model.eval()
    return model, checkpoint


def load_prompt_conditioned_write_strength_model_for_eval(
    checkpoint_path: str | Path,
    model_dir: str | Path | None,
    device: torch.device,
    model_dtype: torch.dtype,
    attn_implementation: str | None,
):
    checkpoint = load_checkpoint(checkpoint_path)
    train_config = checkpoint["train_config"]
    resolved_model_dir = model_dir or train_config["base_model_dir"]
    resolved_attn = attn_implementation or train_config.get(
        "attn_implementation",
        DEFAULT_ATTN_IMPLEMENTATION,
    )

    base_model = load_frozen_base_model(
        model_dir=resolved_model_dir,
        device=device,
        model_dtype=model_dtype,
        attn_implementation=resolved_attn,
    )
    model = PromptConditionedWriteStrengthModel(
        base_model=base_model,
        mlp_hidden_size=train_config["mlp_hidden_size"],
        rms_norm_eps=train_config["rms_norm_eps"],
    ).to(device)
    load_trainable_parameters(model, checkpoint["trainable_state_dict"])
    model.eval()
    return model, checkpoint


def load_static_reaggregation_model_for_eval(
    checkpoint_path: str | Path,
    model_dir: str | Path | None,
    device: torch.device,
    model_dtype: torch.dtype,
    attn_implementation: str | None,
):
    checkpoint = load_checkpoint(checkpoint_path)
    train_config = checkpoint["train_config"]
    resolved_model_dir = model_dir or train_config["base_model_dir"]
    resolved_attn = attn_implementation or train_config.get(
        "attn_implementation",
        DEFAULT_ATTN_IMPLEMENTATION,
    )

    base_model = load_frozen_base_model(
        model_dir=resolved_model_dir,
        device=device,
        model_dtype=model_dtype,
        attn_implementation=resolved_attn,
    )
    model = StaticResidualStreamReaggregationModel(base_model=base_model).to(device)
    load_trainable_parameters(model, checkpoint["trainable_state_dict"])
    model.eval()
    return model, checkpoint


def load_prompt_conditioned_reaggregation_model_for_eval(
    checkpoint_path: str | Path,
    model_dir: str | Path | None,
    device: torch.device,
    model_dtype: torch.dtype,
    attn_implementation: str | None,
):
    checkpoint = load_checkpoint(checkpoint_path)
    train_config = checkpoint["train_config"]
    resolved_model_dir = model_dir or train_config["base_model_dir"]
    resolved_attn = attn_implementation or train_config.get(
        "attn_implementation",
        DEFAULT_ATTN_IMPLEMENTATION,
    )

    base_model = load_frozen_base_model(
        model_dir=resolved_model_dir,
        device=device,
        model_dtype=model_dtype,
        attn_implementation=resolved_attn,
    )
    model = PromptConditionedResidualStreamReaggregationModel(
        base_model=base_model,
        mlp_hidden_size=train_config["mlp_hidden_size"],
        rms_norm_eps=train_config["rms_norm_eps"],
    ).to(device)
    load_trainable_parameters(model, checkpoint["trainable_state_dict"])
    model.eval()
    return model, checkpoint


def _empty_source_stats() -> dict:
    return {
        "loss_sum": 0.0,
        "token_count": 0,
        "example_count": 0,
    }


def compute_batch_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sources: list[str],
) -> dict:
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()

    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    token_mask = shift_labels.ne(-100)

    overall_loss_sum = float(token_losses[token_mask].sum().item())
    overall_token_count = int(token_mask.sum().item())

    source_metrics: dict[str, dict] = {}
    per_example_loss = (token_losses * token_mask).sum(dim=1)
    per_example_tokens = token_mask.sum(dim=1)

    for index, source in enumerate(sources):
        if source not in source_metrics:
            source_metrics[source] = _empty_source_stats()
        source_metrics[source]["loss_sum"] += float(per_example_loss[index].item())
        source_metrics[source]["token_count"] += int(per_example_tokens[index].item())
        source_metrics[source]["example_count"] += 1

    return {
        "loss_sum": overall_loss_sum,
        "token_count": overall_token_count,
        "source_metrics": source_metrics,
        "example_count": len(sources),
    }


def _finalize_loss_stats(loss_sum: float, token_count: int, example_count: int) -> dict:
    mean_loss = loss_sum / max(token_count, 1)
    perplexity = math.exp(mean_loss) if token_count > 0 else float("inf")
    return {
        "num_examples": example_count,
        "num_target_tokens": token_count,
        "loss": mean_loss,
        "perplexity": perplexity,
    }


def evaluate_model_on_split(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    model_dtype: torch.dtype,
    split_name: str,
) -> dict:
    total_loss_sum = 0.0
    total_token_count = 0
    total_example_count = 0
    per_source: dict[str, dict] = {}

    progress_bar = tqdm(dataloader, desc=f"Eval {split_name}", leave=False)
    with torch.inference_mode():
        for batch in progress_bar:
            batch = move_batch_to_device(batch, device=device)
            with create_autocast_context(device=device, model_dtype=model_dtype):
                outputs = forward_eval_model(model=model, batch=batch)

            batch_metrics = compute_batch_metrics(
                logits=outputs.logits,
                labels=batch["labels"],
                sources=batch["source"],
            )
            total_loss_sum += batch_metrics["loss_sum"]
            total_token_count += batch_metrics["token_count"]
            total_example_count += batch_metrics["example_count"]

            for source_name, source_stats in batch_metrics["source_metrics"].items():
                current_stats = per_source.setdefault(source_name, _empty_source_stats())
                current_stats["loss_sum"] += source_stats["loss_sum"]
                current_stats["token_count"] += source_stats["token_count"]
                current_stats["example_count"] += source_stats["example_count"]

            current_loss = total_loss_sum / max(total_token_count, 1)
            progress_bar.set_postfix(loss=f"{current_loss:.4f}")

    progress_bar.close()

    result = _finalize_loss_stats(
        loss_sum=total_loss_sum,
        token_count=total_token_count,
        example_count=total_example_count,
    )
    result["by_source"] = {
        source_name: _finalize_loss_stats(
            loss_sum=source_stats["loss_sum"],
            token_count=source_stats["token_count"],
            example_count=source_stats["example_count"],
        )
        for source_name, source_stats in per_source.items()
    }
    return result


def evaluate_model(
    model: torch.nn.Module,
    validation_data: str | Path,
    test_data: str | Path,
    batch_size: int,
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict:
    validation_loader = create_eval_dataloader(
        arrow_path=validation_data,
        batch_size=batch_size,
        device=device,
    )
    test_loader = create_eval_dataloader(
        arrow_path=test_data,
        batch_size=batch_size,
        device=device,
    )

    return {
        "validation": evaluate_model_on_split(
            model=model,
            dataloader=validation_loader,
            device=device,
            model_dtype=model_dtype,
            split_name="validation",
        ),
        "test": evaluate_model_on_split(
            model=model,
            dataloader=test_loader,
            device=device,
            model_dtype=model_dtype,
            split_name="test",
        ),
    }


def greedy_generate_continuation(
    model: torch.nn.Module,
    prompt_input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    generated = prompt_input_ids.clone()
    prompt_conditioned_models = (
        PromptConditionedWriteStrengthModel,
        PromptConditionedResidualStreamReaggregationModel,
    )
    static_models = (
        StaticWriteStrengthModel,
        StaticResidualStreamReaggregationModel,
    )

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if isinstance(model, prompt_conditioned_models):
                outputs = model(
                    input_ids=generated,
                    labels=None,
                    prompt_input_ids=prompt_input_ids,
                    example_id=None,
                )
            elif isinstance(model, static_models):
                outputs = model(
                    input_ids=generated,
                    labels=None,
                )
            else:
                outputs = model(
                    input_ids=generated,
                    labels=None,
                    use_cache=False,
                )

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

    return generated


def forward_eval_model(model: torch.nn.Module, batch: dict):
    if isinstance(
        model,
        (
            PromptConditionedWriteStrengthModel,
            PromptConditionedResidualStreamReaggregationModel,
        ),
    ):
        return model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            prompt_input_ids=batch["prompt_input_ids"],
            example_id=batch["example_id"],
        )
    if isinstance(
        model,
        (
            StaticWriteStrengthModel,
            StaticResidualStreamReaggregationModel,
        ),
    ):
        return model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
    return model(
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        use_cache=False,
    )


def save_metrics(output_path: str | Path, metrics: dict) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# Backward-compatible loader aliases.
load_static_model_for_eval = load_static_write_strength_model_for_eval
load_prompt_conditioned_model_for_eval = load_prompt_conditioned_write_strength_model_for_eval
