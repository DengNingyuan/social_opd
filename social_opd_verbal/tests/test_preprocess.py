from __future__ import annotations

import json

import pandas as pd
import pytest

from preprocess import preprocess_dataset
from data import ProcessedJsonlDataset


def test_preprocess_writes_train_and_test_prompt_jsonl(tmp_path) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    frame = pd.DataFrame(
        [
            {
                "text": "Thanks for your help.",
                "llm_predict_value": 60,
                "llm_confidence": 80,
                "ground_truth": 70,
            }
        ]
    )
    frame.to_csv(train_csv, index=False)
    frame.to_csv(test_csv, index=False)
    dataset = {
        "name": "politeness",
        "train": str(train_csv),
        "test": str(test_csv),
        "attribute_name": "politeness",
        "attribute_definition": "Rate how polite the text is.",
        "teacher_instruction": "Consider data factors, task factors, and annotator factors. Do not output this analysis.",
    }

    preprocess_dataset(dataset, tmp_path / "processed", tolerance=5)

    train_row = json.loads((tmp_path / "processed" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    test_row = json.loads((tmp_path / "processed" / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "politeness: Rate how polite the text is." in train_row["student_prompt"]
    assert "data factors" in train_row["teacher_prompt"]
    assert "completion" in train_row
    assert "student_prompt" in test_row
    assert "teacher_prompt" not in test_row


def test_processed_jsonl_dataset_rejects_mismatched_dataset(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"dataset": "other"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected 'formality'"):
        ProcessedJsonlDataset([path], expected_dataset="formality")
