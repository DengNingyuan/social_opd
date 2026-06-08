from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_config import load_named_config, timestamp


API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
OUTPUT_DIR = Path("/home/nydeng/social_opd_all/social_opd_geo/data/commercial_api")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call deepseek-chat for three-aspect scoring.")
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset_config", type=Path, default=Path("configs/datasets.json"))
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def build_prompt(text: str, dataset: dict[str, Any]) -> str:
    return f"""BEGIN TEXT
{text}
END TEXT

Attribute: {dataset["attribute_name"]}
Definition: {dataset["attribute_definition"]}

Give a 0-100 score for each aspect below, where higher means that aspect creates more measurement uncertainty:
1. data_factors: uncertainty from the text itself, such as missing context, noise, ambiguity, shortness, or mixed signals.
2. task_factors: uncertainty from the measurement task, such as construct boundaries, scale ambiguity, or nearby score choices.
3. annotator_factors: uncertainty from reasonable annotator disagreement, such as cultural, social, preference, or interpretation differences.

Also give a final 0-100 attribute score for the text.

Return JSON only:
{{
  "data_factors": {{"description": "<brief text>", "score": <integer 0-100>}},
  "task_factors": {{"description": "<brief text>", "score": <integer 0-100>}},
  "annotator_factors": {{"description": "<brief text>", "score": <integer 0-100>}},
  "score": <integer 0-100>
}}
"""


def call_deepseek(prompt: str, api_key: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_response(raw: str) -> dict[str, Any]:
    obj = json.loads(raw)
    return {
        "data_factors": parse_aspect(obj.get("data_factors", {})),
        "task_factors": parse_aspect(obj.get("task_factors", {})),
        "annotator_factors": parse_aspect(obj.get("annotator_factors", {})),
        "score": clamp_score(int(round(float(obj.get("score", 50))))),
    }


def parse_aspect(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        description = str(value.get("description", ""))
        score = int(round(float(value.get("score", 50))))
        return {"description": description, "score": clamp_score(score)}
    return {"description": str(value), "score": 50}


def clamp_score(score: int) -> int:
    return max(0, min(100, score))


def read_rows(csv_path: Path, text_column: str, limit: int | None) -> list[dict[str, Any]]:
    frame = pd.read_csv(csv_path)
    if limit is not None:
        frame = frame.head(limit)
    rows = []
    for idx, row in frame.iterrows():
        item = {"id": f"{csv_path.stem}:{idx}", "text": str(row[text_column])}
        if "ground_truth" in frame.columns:
            item["ground_truth_score"] = float(row["ground_truth"])
        rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY in the environment.")

    dataset = load_named_config(args.dataset_config, args.data_type)
    rows = read_rows(args.csv_path, args.text_column, args.limit)
    outputs = []
    for row in rows:
        raw = call_deepseek(build_prompt(row["text"], dataset), api_key)
        parsed = parse_response(raw)
        outputs.append(
            {
                **row,
                "model": MODEL,
                "raw_response": raw,
                "analysis": {
                    "data_factors": parsed["data_factors"],
                    "task_factors": parsed["task_factors"],
                    "annotator_factors": parsed["annotator_factors"],
                },
                "prediction": {"score": parsed["score"]},
            }
        )

    out_path = args.output_dir / f"{timestamp()}_{args.csv_path.stem}_{args.data_type}_{MODEL}.jsonl"
    write_jsonl(out_path, outputs)
    print(f"Wrote {len(outputs)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
