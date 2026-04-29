"""Prepare the final training dataset from raw corpora.

This script:
1. Loads the raw corpora from `artifact/data/raw/`.
2. Normalizes them into document-level text examples.
3. Creates deterministic train/validation/test splits by source.
4. Tokenizes each document with the local Qwen tokenizer.
5. Builds fixed prompt/answer continuation pairs for model training.
6. Saves only the final training-ready Arrow files under `artifact/data/prepared/`.

Example
-------
python script/1_prepare_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as pa_ipc
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer


DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VALIDATION_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_MIN_TEXT_CHARS = 40
DEFAULT_PROMPT_TOKENS = 96
DEFAULT_ANSWER_TOKENS = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare final prompt/answer continuation data from raw corpora."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for deterministic splitting. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Train split ratio. Default: {DEFAULT_TRAIN_RATIO}",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help=f"Validation split ratio. Default: {DEFAULT_VALIDATION_RATIO}",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Test split ratio. Default: {DEFAULT_TEST_RATIO}",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=DEFAULT_MIN_TEXT_CHARS,
        help=(
            "Drop documents shorter than this many characters after cleaning. "
            f"Default: {DEFAULT_MIN_TEXT_CHARS}"
        ),
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=DEFAULT_PROMPT_TOKENS,
        help=f"Number of prompt tokens per example. Default: {DEFAULT_PROMPT_TOKENS}",
    )
    parser.add_argument(
        "--answer-tokens",
        type=int,
        default=DEFAULT_ANSWER_TOKENS,
        help=f"Number of answer tokens per example. Default: {DEFAULT_ANSWER_TOKENS}",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=None,
        help=(
            "Token step between consecutive windows inside one document. "
            "Default: prompt_tokens + answer_tokens"
        ),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    total_ratio = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Received {args.train_ratio} + {args.validation_ratio} + {args.test_ratio} = {total_ratio}"
        )
    if args.prompt_tokens <= 0:
        raise ValueError(f"prompt_tokens must be positive, received {args.prompt_tokens}")
    if args.answer_tokens <= 0:
        raise ValueError(f"answer_tokens must be positive, received {args.answer_tokens}")
    if args.window_step is not None and args.window_step <= 0:
        raise ValueError(f"window_step must be positive, received {args.window_step}")


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_wikitext_heading(text: str) -> bool:
    return bool(re.fullmatch(r"=+\s*[^=].*?[^=]?\s*=+", text))


def load_split_dataset(split_dir: Path) -> Dataset:
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing dataset split directory: {split_dir}")
    return load_from_disk(str(split_dir))


def load_wikitext_documents(wikitext_dir: Path, min_text_chars: int) -> list[dict]:
    split_names = ["train", "validation", "test"]
    all_lines: list[str] = []

    for split_name in split_names:
        split_dataset = load_split_dataset(wikitext_dir / split_name)
        all_lines.extend(split_dataset["text"])

    documents: list[dict] = []
    current_lines: list[str] = []
    document_index = 0

    for raw_line in all_lines:
        line = normalize_text(raw_line)
        if not line:
            document_index = maybe_add_wikitext_document(
                documents=documents,
                current_lines=current_lines,
                min_text_chars=min_text_chars,
                document_index=document_index,
            )
            current_lines = []
            continue
        current_lines.append(line)

    maybe_add_wikitext_document(
        documents=documents,
        current_lines=current_lines,
        min_text_chars=min_text_chars,
        document_index=document_index,
    )
    return documents


def maybe_add_wikitext_document(
    documents: list[dict],
    current_lines: list[str],
    min_text_chars: int,
    document_index: int,
) -> int:
    if not current_lines:
        return document_index

    text = normalize_text(" ".join(current_lines))
    if len(text) < min_text_chars:
        return document_index
    if is_wikitext_heading(text):
        return document_index

    documents.append(
        {
            "example_id": f"wikitext_{document_index:06d}",
            "source": "wikitext",
            "text": text,
            "medical_label": -1,
        }
    )
    return document_index + 1


def load_medical_documents(medical_dir: Path, min_text_chars: int) -> list[dict]:
    split_names = ["train", "test"]
    documents: list[dict] = []
    document_index = 0

    for split_name in split_names:
        split_dataset = load_split_dataset(medical_dir / split_name)
        for raw_text, label in zip(
            split_dataset["medical_abstract"],
            split_dataset["condition_label"],
        ):
            text = normalize_text(raw_text)
            if len(text) < min_text_chars:
                continue
            documents.append(
                {
                    "example_id": f"medical_{document_index:06d}",
                    "source": "medical",
                    "text": text,
                    "medical_label": int(label),
                }
            )
            document_index += 1

    return documents


def stratified_split_documents(
    records: list[dict],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    grouped_records: dict[str, list[dict]] = {}
    for record in records:
        grouped_records.setdefault(record["source"], []).append(record)

    split_to_records = {
        "train": [],
        "validation": [],
        "test": [],
    }

    rng = random.Random(seed)

    for source_name, source_records in grouped_records.items():
        shuffled_records = list(source_records)
        rng.shuffle(shuffled_records)

        total_count = len(shuffled_records)
        train_count = int(total_count * train_ratio)
        remaining_count = total_count - train_count

        holdout_ratio = validation_ratio + test_ratio
        if holdout_ratio == 0:
            validation_count = 0
        else:
            validation_count = int(remaining_count * (validation_ratio / holdout_ratio))

        if total_count >= 3:
            train_count = max(train_count, 1)
            remaining_count = total_count - train_count
            if validation_ratio > 0 and test_ratio > 0:
                validation_count = max(validation_count, 1)
                validation_count = min(validation_count, remaining_count - 1)
            elif validation_ratio > 0:
                validation_count = remaining_count
            else:
                validation_count = 0

        train_end = train_count
        validation_end = train_count + validation_count

        split_to_records["train"].extend(shuffled_records[:train_end])
        split_to_records["validation"].extend(shuffled_records[train_end:validation_end])
        split_to_records["test"].extend(shuffled_records[validation_end:])

        print(
            f"[document split] {source_name}: "
            f"train={len(shuffled_records[:train_end])}, "
            f"validation={len(shuffled_records[train_end:validation_end])}, "
            f"test={len(shuffled_records[validation_end:])}"
        )

    for split_name, split_records in split_to_records.items():
        rng.shuffle(split_records)
        print(f"[documents] {split_name}: {len(split_records)}")

    return split_to_records


def build_prompt_answer_examples(
    records: list[dict],
    tokenizer: AutoTokenizer,
    prompt_tokens: int,
    answer_tokens: int,
    window_step: int,
) -> list[dict]:
    examples: list[dict] = []
    total_window = prompt_tokens + answer_tokens

    for record in records:
        token_ids = tokenizer(record["text"], add_special_tokens=False)["input_ids"]
        if len(token_ids) < total_window:
            continue

        max_start = len(token_ids) - total_window
        pair_index = 0
        for start in range(0, max_start + 1, window_step):
            prompt_ids = token_ids[start : start + prompt_tokens]
            answer_ids = token_ids[start + prompt_tokens : start + total_window]
            examples.append(
                make_example_record(
                    record=record,
                    prompt_ids=prompt_ids,
                    answer_ids=answer_ids,
                    tokenizer=tokenizer,
                    pair_index=pair_index,
                    start_token_index=start,
                )
            )
            pair_index += 1

    return examples


def make_example_record(
    record: dict,
    prompt_ids: list[int],
    answer_ids: list[int],
    tokenizer: AutoTokenizer,
    pair_index: int,
    start_token_index: int,
) -> dict:
    prompt_text = tokenizer.decode(
        prompt_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    answer_text = tokenizer.decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    full_input_ids = prompt_ids + answer_ids
    labels = ([-100] * len(prompt_ids)) + answer_ids

    return {
        "example_id": f"{record['example_id']}_pair_{pair_index:03d}",
        "source_document_id": record["example_id"],
        "source": record["source"],
        "medical_label": int(record["medical_label"]),
        "prompt_text": prompt_text,
        "answer_text": answer_text,
        "prompt_input_ids": prompt_ids,
        "answer_input_ids": answer_ids,
        "full_input_ids": full_input_ids,
        "labels": labels,
        "prompt_length": len(prompt_ids),
        "answer_length": len(answer_ids),
        "full_length": len(full_input_ids),
        "start_token_index": start_token_index,
    }


def examples_to_columns(examples: list[dict]) -> dict[str, list]:
    return {
        "example_id": [item["example_id"] for item in examples],
        "source_document_id": [item["source_document_id"] for item in examples],
        "source": [item["source"] for item in examples],
        "medical_label": [item["medical_label"] for item in examples],
        "prompt_text": [item["prompt_text"] for item in examples],
        "answer_text": [item["answer_text"] for item in examples],
        "prompt_input_ids": [item["prompt_input_ids"] for item in examples],
        "answer_input_ids": [item["answer_input_ids"] for item in examples],
        "full_input_ids": [item["full_input_ids"] for item in examples],
        "labels": [item["labels"] for item in examples],
        "prompt_length": [item["prompt_length"] for item in examples],
        "answer_length": [item["answer_length"] for item in examples],
        "full_length": [item["full_length"] for item in examples],
        "start_token_index": [item["start_token_index"] for item in examples],
    }


def build_arrow_table(examples: list[dict]) -> pa.Table:
    columns = examples_to_columns(examples)
    schema = pa.schema(
        [
            pa.field("example_id", pa.string()),
            pa.field("source_document_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("medical_label", pa.int64()),
            pa.field("prompt_text", pa.string()),
            pa.field("answer_text", pa.string()),
            pa.field("prompt_input_ids", pa.list_(pa.int32())),
            pa.field("answer_input_ids", pa.list_(pa.int32())),
            pa.field("full_input_ids", pa.list_(pa.int32())),
            pa.field("labels", pa.list_(pa.int32())),
            pa.field("prompt_length", pa.int32()),
            pa.field("answer_length", pa.int32()),
            pa.field("full_length", pa.int32()),
            pa.field("start_token_index", pa.int32()),
        ]
    )
    arrays = [
        pa.array(columns["example_id"], type=pa.string()),
        pa.array(columns["source_document_id"], type=pa.string()),
        pa.array(columns["source"], type=pa.string()),
        pa.array(columns["medical_label"], type=pa.int64()),
        pa.array(columns["prompt_text"], type=pa.string()),
        pa.array(columns["answer_text"], type=pa.string()),
        pa.array(columns["prompt_input_ids"], type=pa.list_(pa.int32())),
        pa.array(columns["answer_input_ids"], type=pa.list_(pa.int32())),
        pa.array(columns["full_input_ids"], type=pa.list_(pa.int32())),
        pa.array(columns["labels"], type=pa.list_(pa.int32())),
        pa.array(columns["prompt_length"], type=pa.int32()),
        pa.array(columns["answer_length"], type=pa.int32()),
        pa.array(columns["full_length"], type=pa.int32()),
        pa.array(columns["start_token_index"], type=pa.int32()),
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def save_examples(examples: list[dict], split_path: Path) -> None:
    ensure_dir(split_path.parent)
    table = build_arrow_table(examples)
    with pa.OSFile(str(split_path), "wb") as sink:
        with pa_ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    print(f"[done] Saved {len(examples)} prompt/answer pairs to: {split_path}")


def build_summary(
    split_to_documents: dict[str, list[dict]],
    split_to_examples: dict[str, list[dict]],
) -> dict:
    summary = {
        "document_splits": {},
        "example_splits": {},
        "document_source_totals": {},
        "example_source_totals": {},
    }

    for split_name, records in split_to_documents.items():
        source_counts: dict[str, int] = {}
        for record in records:
            source_counts[record["source"]] = source_counts.get(record["source"], 0) + 1
            summary["document_source_totals"][record["source"]] = (
                summary["document_source_totals"].get(record["source"], 0) + 1
            )
        summary["document_splits"][split_name] = {
            "num_documents": len(records),
            "source_counts": source_counts,
        }

    for split_name, examples in split_to_examples.items():
        source_counts: dict[str, int] = {}
        for example in examples:
            source_counts[example["source"]] = source_counts.get(example["source"], 0) + 1
            summary["example_source_totals"][example["source"]] = (
                summary["example_source_totals"].get(example["source"], 0) + 1
            )
        summary["example_splits"][split_name] = {
            "num_examples": len(examples),
            "source_counts": source_counts,
        }

    summary["total_documents"] = sum(
        split_info["num_documents"] for split_info in summary["document_splits"].values()
    )
    summary["total_examples"] = sum(
        split_info["num_examples"] for split_info in summary["example_splits"].values()
    )
    return summary


def write_summary(summary_path: Path, summary: dict) -> None:
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] Wrote summary to: {summary_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    total_window = args.prompt_tokens + args.answer_tokens
    window_step = args.window_step or total_window

    repo_root = Path(__file__).resolve().parent.parent
    raw_data_dir = repo_root / "artifact" / "data" / "raw"
    prepared_data_dir = repo_root / "artifact" / "data" / "prepared"
    wikitext_dir = raw_data_dir / "wikitext_2_raw"
    medical_dir = raw_data_dir / "medical_abstracts"
    model_dir = repo_root / "artifact" / "model" / "qwen3_0_6b"
    output_dir = prepared_data_dir

    print("[load] Reading WikiText documents")
    wikitext_documents = load_wikitext_documents(
        wikitext_dir=wikitext_dir,
        min_text_chars=args.min_text_chars,
    )
    print(f"[done] WikiText documents: {len(wikitext_documents)}")

    print("[load] Reading medical abstracts")
    medical_documents = load_medical_documents(
        medical_dir=medical_dir,
        min_text_chars=args.min_text_chars,
    )
    print(f"[done] Medical documents: {len(medical_documents)}")

    all_documents = wikitext_documents + medical_documents
    split_to_documents = stratified_split_documents(
        records=all_documents,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"[load] Tokenizer: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    split_to_examples: dict[str, list[dict]] = {}
    for split_name in ["train", "validation", "test"]:
        print(f"[tokenize] {split_name}")
        examples = build_prompt_answer_examples(
            records=split_to_documents[split_name],
            tokenizer=tokenizer,
            prompt_tokens=args.prompt_tokens,
            answer_tokens=args.answer_tokens,
            window_step=window_step,
        )
        split_to_examples[split_name] = examples
        print(f"[done] {split_name}: {len(examples)} prompt/answer pairs")

    reset_dir(output_dir)
    for split_name, examples in split_to_examples.items():
        save_examples(
            examples=examples,
            split_path=output_dir / f"{split_name}.arrow",
        )

    summary = build_summary(split_to_documents, split_to_examples)
    summary["seed"] = args.seed
    summary["min_text_chars"] = args.min_text_chars
    summary["split_ratios"] = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    summary["prompt_tokens"] = args.prompt_tokens
    summary["answer_tokens"] = args.answer_tokens
    summary["window_step"] = window_step
    summary["min_total_tokens"] = total_window
    write_summary(output_dir / "summary.json", summary)

    print("[done] Final training data preparation completed.")


if __name__ == "__main__":
    main()
