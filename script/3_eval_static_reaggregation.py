"""Evaluate the trained static re-aggregation model on validation and test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from _eval_utils import (
    DEFAULT_EVAL_BATCH_SIZE,
    create_runtime,
    evaluate_model,
    load_static_reaggregation_model_for_eval,
    save_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained static residual-stream re-aggregation model."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--checkpoint-path",
        default=str(
            repo_root
            / "artifact"
            / "checkpoints"
            / "static_residual_stream_reaggregation"
            / "trained_model.pt"
        ),
        help="Path to the saved static re-aggregation checkpoint.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional override for the local frozen Qwen model directory.",
    )
    parser.add_argument(
        "--validation-data",
        default=str(repo_root / "artifact" / "data" / "prepared" / "validation.arrow"),
        help="Path to the prepared validation Arrow file.",
    )
    parser.add_argument(
        "--test-data",
        default=str(repo_root / "artifact" / "data" / "prepared" / "test.arrow"),
        help="Path to the prepared test Arrow file.",
    )
    parser.add_argument(
        "--output-path",
        default=str(
            repo_root / "result" / "metrics" / "static_residual_stream_reaggregation.json"
        ),
        help="Path to save evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help=f"Evaluation batch size. Default: {DEFAULT_EVAL_BATCH_SIZE}",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional override for the attention backend used by the base model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, model_dtype = create_runtime()
    print(f"[setup] device={device} model_dtype={model_dtype}")

    model, checkpoint = load_static_reaggregation_model_for_eval(
        checkpoint_path=args.checkpoint_path,
        model_dir=args.model_dir,
        device=device,
        model_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
    )

    metrics = evaluate_model(
        model=model,
        validation_data=args.validation_data,
        test_data=args.test_data,
        batch_size=args.batch_size,
        device=device,
        model_dtype=model_dtype,
    )
    metrics["model_type"] = "static_residual_stream_reaggregation"
    metrics["checkpoint_path"] = args.checkpoint_path
    metrics["batch_size"] = args.batch_size
    metrics["device"] = str(device)
    metrics["model_dtype"] = str(model_dtype)
    metrics["train_config"] = checkpoint["train_config"]

    save_metrics(args.output_path, metrics)
    print(f"[done] Saved metrics to: {args.output_path}")


if __name__ == "__main__":
    main()
