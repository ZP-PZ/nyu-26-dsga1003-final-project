"""Shared training utilities for stage-2 scripts."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm


class PreparedArrowDataset(Dataset):
    """A simple in-memory dataset backed by one Arrow file."""

    def __init__(self, arrow_path: str | Path) -> None:
        self.arrow_path = Path(arrow_path)
        self.columns = self._load_columns(self.arrow_path)
        self.example_ids = self.columns["example_id"]
        self.source_document_ids = self.columns["source_document_id"]
        self.sources = self.columns["source"]
        self.medical_labels = torch.tensor(self.columns["medical_label"], dtype=torch.long)
        self.prompt_input_ids = torch.tensor(self.columns["prompt_input_ids"], dtype=torch.long)
        self.answer_input_ids = torch.tensor(self.columns["answer_input_ids"], dtype=torch.long)
        self.input_ids = torch.tensor(self.columns["full_input_ids"], dtype=torch.long)
        self.labels = torch.tensor(self.columns["labels"], dtype=torch.long)
        self.prompt_lengths = torch.tensor(self.columns["prompt_length"], dtype=torch.long)
        self.answer_lengths = torch.tensor(self.columns["answer_length"], dtype=torch.long)
        self.full_lengths = torch.tensor(self.columns["full_length"], dtype=torch.long)
        self.start_token_indices = torch.tensor(
            self.columns["start_token_index"],
            dtype=torch.long,
        )
        self.length = len(self.example_ids)

    @staticmethod
    def _load_columns(arrow_path: Path) -> dict[str, list]:
        with pa.memory_map(str(arrow_path), "r") as source:
            reader = pa_ipc.RecordBatchFileReader(source)
            table = reader.read_all()
        return table.to_pydict()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        return {
            "example_id": self.example_ids[index],
            "source_document_id": self.source_document_ids[index],
            "source": self.sources[index],
            "medical_label": self.medical_labels[index],
            "prompt_input_ids": self.prompt_input_ids[index],
            "answer_input_ids": self.answer_input_ids[index],
            "full_input_ids": self.input_ids[index],
            "labels": self.labels[index],
            "prompt_length": self.prompt_lengths[index],
            "answer_length": self.answer_lengths[index],
            "full_length": self.full_lengths[index],
            "start_token_index": self.start_token_indices[index],
        }


def collate_prepared_batch(examples: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    batch = {
        "example_id": [example["example_id"] for example in examples],
        "source_document_id": [example["source_document_id"] for example in examples],
        "source": [example["source"] for example in examples],
        "medical_label": torch.stack([example["medical_label"] for example in examples]),
        "prompt_input_ids": torch.stack([example["prompt_input_ids"] for example in examples]),
        "answer_input_ids": torch.stack([example["answer_input_ids"] for example in examples]),
        "input_ids": torch.stack([example["full_input_ids"] for example in examples]),
        "labels": torch.stack([example["labels"] for example in examples]),
        "prompt_length": torch.stack([example["prompt_length"] for example in examples]),
        "answer_length": torch.stack([example["answer_length"] for example in examples]),
        "full_length": torch.stack([example["full_length"] for example in examples]),
        "start_token_index": torch.stack([example["start_token_index"] for example in examples]),
    }
    return batch


def create_train_dataloader(
    arrow_path: str | Path,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = PreparedArrowDataset(arrow_path=arrow_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_prepared_batch,
        pin_memory=pin_memory,
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved_batch = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[key] = value.to(device, non_blocking=True)
        else:
            moved_batch[key] = value
    return moved_batch


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def select_model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def create_autocast_context(device: torch.device, model_dtype: torch.dtype):
    if device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type=device.type, dtype=model_dtype)
    return nullcontext()


def load_frozen_base_model(
    model_dir: str | Path,
    device: torch.device,
    model_dtype: torch.dtype,
    attn_implementation: str = "sdpa",
):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=model_dtype,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )
    return model.to(device)


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    return torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_dtype: torch.dtype,
    num_epochs: int,
    gradient_clip_norm: float,
    log_every: int,
    gradient_accumulation_steps: int = 1,
    max_train_steps: int | None = None,
) -> tuple[list[dict], list[dict]]:
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive, "
            f"received {gradient_accumulation_steps}"
        )

    step_history: list[dict] = []
    epoch_history: list[dict] = []
    optimizer_step = 0
    seen_batches = 0
    stop_training = False
    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and model_dtype == torch.float16,
    )

    model.train()

    for epoch_index in range(num_epochs):
        running_loss = 0.0
        epoch_optimizer_steps = 0
        epoch_seen_batches = 0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch_index + 1}/{num_epochs}",
            leave=False,
        )
        for batch in progress_bar:
            if max_train_steps is not None and optimizer_step >= max_train_steps:
                stop_training = True
                break

            batch = move_batch_to_device(batch, device=device)

            with create_autocast_context(device=device, model_dtype=model_dtype):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            grad_scaler.scale(loss).backward()
            loss_value = float((loss.detach().cpu()) * gradient_accumulation_steps)
            seen_batches += 1
            epoch_seen_batches += 1
            running_loss += loss_value

            should_step = (seen_batches % gradient_accumulation_steps == 0) or (
                epoch_seen_batches == len(dataloader)
            )
            if should_step:
                if gradient_clip_norm > 0:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)

                optimizer_step += 1
                epoch_optimizer_steps += 1

                step_history.append(
                    {
                        "epoch": epoch_index + 1,
                        "step": optimizer_step,
                        "loss": loss_value,
                    }
                )

                if log_every > 0 and optimizer_step % log_every == 0:
                    print(
                        f"[train] epoch={epoch_index + 1} "
                        f"step={optimizer_step} loss={loss_value:.6f}"
                    )
                progress_bar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    step=optimizer_step,
                )

        progress_bar.close()

        mean_loss = running_loss / max(epoch_seen_batches, 1)
        epoch_history.append(
            {
                "epoch": epoch_index + 1,
                "num_batches": epoch_seen_batches,
                "num_optimizer_steps": epoch_optimizer_steps,
                "mean_loss": mean_loss,
            }
        )
        print(
            f"[epoch] epoch={epoch_index + 1} "
            f"batches={epoch_seen_batches} "
            f"optimizer_steps={epoch_optimizer_steps} "
            f"mean_loss={mean_loss:.6f}"
        )

        if stop_training:
            break

    return step_history, epoch_history


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_training_outputs(
    output_dir: str | Path,
    model: torch.nn.Module,
    train_config: dict,
    step_history: list[dict],
    epoch_history: list[dict],
) -> None:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    torch.save(
        {
            "trainable_state_dict": model.get_trainable_state_dict(),
            "train_config": train_config,
        },
        output_dir / "trained_model.pt",
    )

    (output_dir / "train_loss_history.json").write_text(
        json.dumps(step_history, indent=2),
        encoding="utf-8",
    )
    (output_dir / "train_epoch_history.json").write_text(
        json.dumps(epoch_history, indent=2),
        encoding="utf-8",
    )
    (output_dir / "train_config.json").write_text(
        json.dumps(train_config, indent=2),
        encoding="utf-8",
    )
