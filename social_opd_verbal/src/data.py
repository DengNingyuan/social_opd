from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class ProcessedJsonlDataset(Dataset):
    def __init__(self, paths: list[Path], max_samples: int | None = None, expected_dataset: str | None = None):
        rows: list[dict[str, Any]] = []
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if expected_dataset is not None and row.get("dataset") != expected_dataset:
                        raise ValueError(
                            f"{path} contains dataset={row.get('dataset')!r}; expected {expected_dataset!r}."
                        )
                    rows.append(row)
        self.rows = rows[:max_samples] if max_samples is not None else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


@dataclass
class OpdCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        student_chosen = []
        teacher_chosen = []
        texts = []
        truths = []
        for row in batch:
            student_chosen.append(self._encode(row["student_prompt"], row["completion"]))
            teacher_chosen.append(self._encode(row["teacher_prompt"], row["completion"]))
            texts.append(row["text"])
            truths.append(float(row["ground_truth_score"]))
        return {
            "student_chosen": self._pad_batch(student_chosen),
            "teacher_chosen": self._pad_batch(teacher_chosen),
            "texts": texts,
            "ground_truth": torch.tensor(truths, dtype=torch.float32),
        }

    def _encode(self, prompt: str, completion: str) -> dict[str, list[int]]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            completion_ids = completion_ids + [self.tokenizer.eos_token_id]
        if len(prompt_ids) + len(completion_ids) > self.max_length:
            completion_budget = min(len(completion_ids), max(1, self.max_length // 4))
            prompt_budget = self.max_length - completion_budget
            prompt_ids = prompt_ids[-prompt_budget:]
            completion_ids = completion_ids[:completion_budget]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def _pad_batch(self, items: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._pad([item["input_ids"] for item in items]),
            "attention_mask": self._pad([item["attention_mask"] for item in items], pad_value=0),
            "labels": self._pad([item["labels"] for item in items], pad_value=-100),
        }

    def _pad(self, sequences: list[list[int]], pad_value: int | None = None) -> torch.Tensor:
        if pad_value is None:
            pad_value = self.tokenizer.pad_token_id
        max_len = max(len(seq) for seq in sequences)
        return torch.tensor([seq + [pad_value] * (max_len - len(seq)) for seq in sequences], dtype=torch.long)


def format_completion(score: float, confidence: float) -> str:
    return json.dumps({"score": int(round(clip_score(score))), "confidence": round(clip_confidence(confidence), 3)}, sort_keys=True)


def confidence_target(ground_truth: float, baseline_score: float, baseline_confidence: float) -> float:
    error_conf = 1.0 - min(abs(ground_truth - baseline_score) / 100.0, 1.0)
    return 0.5 * baseline_confidence + 0.5 * error_conf


def normalize_confidence(value: float) -> float:
    return clip_confidence(value / 100.0 if value > 1.0 else value)


def clip_score(value: float) -> float:
    return float(min(max(value, 0.0), 100.0))


def clip_confidence(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def parse_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]
