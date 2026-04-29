"""Download the base model and datasets used in the final project.

This script keeps the download step explicit and reproducible:
1. Download the frozen base model snapshot from Hugging Face.
2. Download WikiText-2 and one specialized-domain corpus with `datasets`.
3. Save each raw dataset split under `artifact/data/raw/`.

Example
-------
python script/0_download_model_and_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import snapshot_download


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_WIKITEXT_DATASET = "wikitext"
DEFAULT_WIKITEXT_CONFIG = "wikitext-2-raw-v1"
DEFAULT_DOMAIN_DATASET = "TimSchopf/medical_abstracts"
DEFAULT_DOMAIN_CONFIG = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the project model and datasets into local artifact directories."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id to download. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--wikitext-dataset",
        default=DEFAULT_WIKITEXT_DATASET,
        help=f"Dataset name for WikiText-2. Default: {DEFAULT_WIKITEXT_DATASET}",
    )
    parser.add_argument(
        "--wikitext-config",
        default=DEFAULT_WIKITEXT_CONFIG,
        help=f"Dataset config for WikiText-2. Default: {DEFAULT_WIKITEXT_CONFIG}",
    )
    parser.add_argument(
        "--domain-dataset",
        default=DEFAULT_DOMAIN_DATASET,
        help=(
            "Hugging Face dataset name for the specialized-domain corpus. "
            f"Default: {DEFAULT_DOMAIN_DATASET}"
        ),
    )
    parser.add_argument(
        "--domain-config",
        default=DEFAULT_DOMAIN_CONFIG,
        help=(
            "Dataset config for the specialized-domain corpus. "
            f"Default: {DEFAULT_DOMAIN_CONFIG}"
        ),
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Redownload model and datasets even if they already exist locally.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_model(model_id: str, model_dir: Path, force_redownload: bool) -> None:
    if model_dir.exists() and any(model_dir.iterdir()) and not force_redownload:
        print(f"[skip] Model already exists at: {model_dir}")
        return

    print(f"[download] Model: {model_id}")
    ensure_dir(model_dir)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        force_download=force_redownload,
    )
    print(f"[done] Model saved to: {model_dir}")


def download_and_save_dataset(
    dataset_name: str,
    output_dir: Path,
    force_redownload: bool,
    config_name: str | None = None,
    split_name: str | None = None,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force_redownload:
        print(f"[skip] Dataset already exists at: {output_dir}")
        return

    dataset_label = dataset_name
    if config_name:
        dataset_label += f" ({config_name})"
    if split_name:
        dataset_label += f" [{split_name}]"

    print(f"[download] Dataset: {dataset_label}")
    load_kwargs = {"path": dataset_name}
    if config_name is not None:
        load_kwargs["name"] = config_name
    if split_name is not None:
        load_kwargs["split"] = split_name
    if force_redownload:
        load_kwargs["download_mode"] = "force_redownload"

    dataset = load_dataset(**load_kwargs)
    save_dataset(dataset, output_dir, split_name=split_name)
    print(f"[done] Dataset saved to: {output_dir}")


def save_dataset(
    dataset: Dataset | DatasetDict,
    output_dir: Path,
    split_name: str | None = None,
) -> None:
    ensure_dir(output_dir)

    if isinstance(dataset, DatasetDict):
        for current_split_name, split_dataset in dataset.items():
            split_dir = output_dir / current_split_name
            split_dataset.save_to_disk(str(split_dir))
            print(f"  [split] {current_split_name}: {split_dir}")
        return

    final_split_name = split_name or "train"
    split_dir = output_dir / final_split_name
    dataset.save_to_disk(str(split_dir))
    print(f"  [split] {final_split_name}: {split_dir}")


def write_metadata(metadata_path: Path, metadata: dict) -> None:
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] Metadata saved to: {metadata_path}")


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    artifact_dir = repo_root / "artifact"
    model_dir = artifact_dir / "model" / "qwen3_0_6b"
    data_dir = artifact_dir / "data"
    raw_data_dir = data_dir / "raw"
    wikitext_dir = raw_data_dir / "wikitext_2_raw"
    domain_dir = raw_data_dir / "medical_abstracts"
    metadata_path = raw_data_dir / "download_manifest.json"

    ensure_dir(artifact_dir)
    ensure_dir(model_dir.parent)
    ensure_dir(raw_data_dir)

    download_model(
        model_id=args.model_id,
        model_dir=model_dir,
        force_redownload=args.force_redownload,
    )
    download_and_save_dataset(
        dataset_name=args.wikitext_dataset,
        output_dir=wikitext_dir,
        force_redownload=args.force_redownload,
        config_name=args.wikitext_config,
    )
    download_and_save_dataset(
        dataset_name=args.domain_dataset,
        output_dir=domain_dir,
        force_redownload=args.force_redownload,
        config_name=args.domain_config,
    )

    metadata = {
        "model": {
            "model_id": args.model_id,
            "local_path": str(model_dir.relative_to(repo_root)),
        },
        "datasets": {
            "wikitext": {
                "dataset_name": args.wikitext_dataset,
                "config_name": args.wikitext_config,
                "local_path": str(wikitext_dir.relative_to(repo_root)),
            },
            "domain_corpus": {
                "dataset_name": args.domain_dataset,
                "config_name": args.domain_config,
                "local_path": str(domain_dir.relative_to(repo_root)),
            },
        },
    }
    write_metadata(metadata_path, metadata)

    print("[done] All downloads completed.")


if __name__ == "__main__":
    main()
