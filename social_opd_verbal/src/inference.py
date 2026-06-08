from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from data import ProcessedJsonlDataset
from metrics import parse_completion


def run_inference(
    model: torch.nn.Module,
    tokenizer: Any,
    paths: list[Path],
    args: Any,
    device: torch.device,
    expected_dataset: str | None = None,
) -> list[dict[str, Any]]:
    rows = ProcessedJsonlDataset(paths, max_samples=args.max_test_samples, expected_dataset=expected_dataset).rows
    model.eval()
    outputs = []
    for idx, row in enumerate(tqdm(rows, desc="infer", unit="example")):
        prompt = row["student_prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        prediction = parse_completion(completion)
        outputs.append(
            {
                "id": idx,
                "source_id": row.get("id", idx),
                "text": row["text"],
                "ground_truth_score": float(row["ground_truth_score"]),
                "raw_completion": completion,
                "prediction": prediction,
            }
        )
    return outputs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
