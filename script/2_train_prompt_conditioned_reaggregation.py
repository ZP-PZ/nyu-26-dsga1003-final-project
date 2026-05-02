"""Train the prompt-conditioned residual-stream re-aggregation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from _residual_modules import PromptConditionedResidualStreamReaggregationModel
from _training_utils import (
    build_optimizer,
    configure_runtime,
    create_train_dataloader,
    detect_device,
    load_frozen_base_model,
    save_training_outputs,
    select_model_dtype,
    train_model,
)


DEFAULT_BATCH_SIZE = 6
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 5
DEFAULT_NUM_EPOCHS = 1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_LOG_EVERY = 50
DEFAULT_MLP_HIDDEN_SIZE = 512
DEFAULT_RMS_NORM_EPS = 1e-6
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the prompt-conditioned residual-stream re-aggregation model."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--model-dir",
        default=str(repo_root / "artifact" / "model" / "qwen3_0_6b"),
        help="Path to the local frozen Qwen model directory.",
    )
    parser.add_argument(
        "--train-data",
        default=str(repo_root / "artifact" / "data" / "prepared" / "train.arrow"),
        help="Path to the prepared training Arrow file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            repo_root
            / "artifact"
            / "checkpoints"
            / "prompt_conditioned_residual_stream_reaggregation"
        ),
        help="Directory to save the trained prompt-conditioned re-aggregation model and training history.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Per-step micro-batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help=(
            "Number of micro-batches to accumulate before one optimizer step. "
            f"Default: {DEFAULT_GRADIENT_ACCUMULATION_STEPS}"
        ),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"Number of training epochs. Default: {DEFAULT_NUM_EPOCHS}",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate. Default: {DEFAULT_LEARNING_RATE}",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"Weight decay. Default: {DEFAULT_WEIGHT_DECAY}",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_NORM,
        help=f"Gradient clipping norm. Default: {DEFAULT_GRADIENT_CLIP_NORM}",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help=f"Print training loss every N steps. Default: {DEFAULT_LOG_EVERY}",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Optional cap on the total number of training steps.",
    )
    parser.add_argument(
        "--mlp-hidden-size",
        type=int,
        default=DEFAULT_MLP_HIDDEN_SIZE,
        help=f"Hidden size of the prompt-conditioned MLP. Default: {DEFAULT_MLP_HIDDEN_SIZE}",
    )
    parser.add_argument(
        "--rms-norm-eps",
        type=float,
        default=DEFAULT_RMS_NORM_EPS,
        help=f"RMSNorm epsilon used in the MLP. Default: {DEFAULT_RMS_NORM_EPS}",
    )
    parser.add_argument(
        "--attn-implementation",
        default=DEFAULT_ATTN_IMPLEMENTATION,
        help=(
            "Attention backend for the frozen base model. "
            f"Default: {DEFAULT_ATTN_IMPLEMENTATION}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = detect_device()
    configure_runtime(device)
    model_dtype = select_model_dtype(device)
    print(f"[setup] device={device} model_dtype={model_dtype}")
    print(
        "[setup] effective_batch_size="
        f"{args.batch_size * args.gradient_accumulation_steps}"
    )

    dataloader = create_train_dataloader(
        arrow_path=args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    print(f"[data] train_data={args.train_data}")

    base_model = load_frozen_base_model(
        model_dir=args.model_dir,
        device=device,
        model_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
    )
    model = PromptConditionedResidualStreamReaggregationModel(
        base_model=base_model,
        mlp_hidden_size=args.mlp_hidden_size,
        rms_norm_eps=args.rms_norm_eps,
    ).to(device)

    optimizer = build_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    step_history, epoch_history = train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        model_dtype=model_dtype,
        num_epochs=args.num_epochs,
        gradient_clip_norm=args.gradient_clip_norm,
        log_every=args.log_every,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_steps=args.max_train_steps,
    )

    train_config = {
        "model_type": "prompt_conditioned_residual_stream_reaggregation",
        "base_model_dir": args.model_dir,
        "train_data": args.train_data,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "log_every": args.log_every,
        "max_train_steps": args.max_train_steps,
        "mlp_hidden_size": args.mlp_hidden_size,
        "rms_norm_eps": args.rms_norm_eps,
        "attn_implementation": args.attn_implementation,
        "device": str(device),
        "model_dtype": str(model_dtype),
        "num_layers": model.num_layers,
        "num_reaggregation_weights": model.num_reaggregation_weights,
    }
    save_training_outputs(
        output_dir=args.output_dir,
        model=model,
        train_config=train_config,
        step_history=step_history,
        epoch_history=epoch_history,
    )
    print(f"[done] Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
