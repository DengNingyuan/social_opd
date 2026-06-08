from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from data import clip_score
from evaluate import DEFAULT_ECE_BINS, DEFAULT_TOLERANCE, tolerance_correctness, tolerance_ece


def parse_completion(text: str) -> dict[str, float]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            return {
                "score": clip_score(float(payload.get("score", 50.0))),
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    number = re.search(r"-?\d+(?:\.\d+)?", text)
    return {"score": clip_score(float(number.group(0))) if number else 50.0}


def compute_measurement_metrics(
    rows: list[dict[str, Any]],
    tolerance: float = DEFAULT_TOLERANCE,
    bins: int = DEFAULT_ECE_BINS,
) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot compute metrics for an empty prediction set.")
    truth = np.array([float(row["ground_truth_score"]) for row in rows], dtype=float)
    pred = np.array([float(row["prediction"]["score"]) for row in rows], dtype=float)
    conf = np.array([float(row["prediction"]["confidence"]) for row in rows], dtype=float)
    correct = tolerance_correctness(pred, truth, tolerance)
    rho = spearmanr(pred, truth).correlation
    return {
        "n": float(len(rows)),
        "tolerance": float(tolerance),
        "ece_bins": float(bins),
        "accuracy": float(correct.mean()),
        "ece": float(tolerance_ece(conf, correct, bins=bins)),
        "brier": float(((conf - correct) ** 2).mean()),
        "mh": safe_float(rho),
    }


def safe_float(value: float) -> float:
    if value is None or math.isnan(float(value)):
        return 0.0
    return float(value)
