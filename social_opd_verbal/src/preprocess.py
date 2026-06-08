from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data import confidence_target, format_completion, normalize_confidence
from experiment_config import load_named_config, write_metadata
from prompts import build_prompt_from_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess measurement CSVs into prompt JSONL files.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_config", type=Path, default=Path("configs/datasets.json"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=10.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_named_config(args.dataset_config, args.dataset_name)
    output_dir = args.output_dir or Path("data/processed") / args.dataset_name
    preprocess_dataset(
        dataset=dataset,
        output_dir=output_dir,
        tolerance=args.tolerance,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )


def preprocess_dataset(
    dataset: dict[str, Any],
    output_dir: Path,
    tolerance: float = 10.0,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = build_split_rows(dataset, Path(dataset["train"]), "train", tolerance, max_train_samples)
    test_rows = build_split_rows(dataset, Path(dataset["test"]), "test", tolerance, max_test_samples)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_metadata(
        output_dir / "preprocess_metadata.json",
        {
            "dataset": dataset,
            "tolerance": tolerance,
            "train_file": "train.jsonl",
            "test_file": "test.jsonl",
            "train_n": len(train_rows),
            "test_n": len(test_rows),
        },
    )
    print(f"Wrote {len(train_rows)} train rows and {len(test_rows)} test rows to {output_dir}", flush=True)


def build_split_rows(
    dataset: dict[str, Any],
    csv_path: Path,
    split: str,
    tolerance: float,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(csv_path)
    required = {"text", "llm_predict_value", "llm_confidence", "ground_truth"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    if max_samples is not None:
        frame = frame.head(max_samples)
    rows = []
    for row_idx, row in frame.iterrows():
        text = str(row["text"])
        ground_truth = float(row["ground_truth"])
        baseline_score = float(row["llm_predict_value"])
        baseline_confidence = normalize_confidence(float(row["llm_confidence"]))
        payload = {
            "id": f"{dataset['name']}:{split}:{row_idx}",
            "dataset": dataset["name"],
            "split": split,
            "text": text,
            "ground_truth_score": ground_truth,
            "original_llm_score": baseline_score,
            "original_llm_confidence": baseline_confidence,
            "student_prompt": build_prompt_from_dataset(text, dataset, role="student", tolerance=tolerance),
        }
        if split == "train":
            target_confidence = confidence_target(ground_truth, baseline_score, baseline_confidence)
            payload["teacher_prompt"] = build_prompt_from_dataset(text, dataset, role="teacher", tolerance=tolerance)
            payload["completion"] = format_completion(ground_truth, target_confidence)
        rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
