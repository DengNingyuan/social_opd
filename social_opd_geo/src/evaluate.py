from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from schema import DIMENSION_KEYS

DEFAULT_TOLERANCE = 10.0
DEFAULT_ECE_BINS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAUD prediction JSONL.")
    parser.add_argument("--pred_file", type=Path, required=True)
    parser.add_argument("--out_file", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--ece_bins", type=int, default=DEFAULT_ECE_BINS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.pred_file)
    metrics = compute_metrics(rows, tolerance=args.tolerance, ece_bins=args.ece_bins)
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(text + "\n", encoding="utf-8")
    print(text)


def compute_metrics(
    rows: list[dict[str, Any]],
    tolerance: float = DEFAULT_TOLERANCE,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot compute metrics for an empty prediction set.")
    parsed = [parse_prediction(r) for r in rows]
    y_true = np.array([float(r["ground_truth_score"]) for r in rows], dtype=float)
    y_pred = np.array([float(p["score"]) for p in parsed], dtype=float)
    conf = np.array([float(p["confidence"]) for p in parsed], dtype=float)
    has_uncertainty = all("uncertainty_scores" in p for p in parsed)
    if has_uncertainty:
        unc = np.array([[float(p["uncertainty_scores"][k]) for k in DIMENSION_KEYS] for p in parsed], dtype=float)
    else:
        unc = None
    abs_err = np.abs(y_pred - y_true)
    correct = tolerance_correctness(y_pred, y_true, tolerance)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    rho = spearmanr(y_true, y_pred).correlation
    brier = float(((conf - correct) ** 2).mean())
    ece = tolerance_ece(conf, correct, bins=ece_bins)
    metrics = {
        "n": float(len(rows)),
        "tolerance": float(tolerance),
        "ece_bins": float(ece_bins),
        "mae": mae,
        "rmse": rmse,
        "spearman": safe_float(rho),
        "accuracy": float(correct.mean()),
        "tolerance_accuracy": float(correct.mean()),
        "brier": brier,
        "ece": float(ece),
    }
    if unc is not None:
        u_avg = unc.mean(axis=1)
        consistency_corr = spearmanr(1.0 - conf, u_avg).correlation
        violation = np.logical_and(conf >= 0.8, u_avg >= 0.6).mean()
        metrics["confidence_uncertainty_spearman"] = safe_float(consistency_corr)
        metrics["high_conf_high_uncertainty_violation_rate"] = float(violation)
    return metrics


def tolerance_correctness(prediction: np.ndarray, ground_truth: np.ndarray, tolerance: float) -> np.ndarray:
    return (np.abs(prediction - ground_truth) <= tolerance).astype(float)


def tolerance_reliability_curve(confidence: np.ndarray, correct: np.ndarray, bins: int) -> dict[str, np.ndarray]:
    """Aggregate tolerance accuracy and confidence in equal-width observed confidence bins."""
    if bins <= 0:
        raise ValueError("ece_bins must be positive.")
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidence.ndim != 1 or correct.ndim != 1:
        raise ValueError("confidence and correct must be 1D arrays.")
    if len(confidence) != len(correct):
        raise ValueError("confidence and correct must have the same length.")
    if len(confidence) == 0:
        raise ValueError("confidence and correct must not be empty.")

    bin_ids, edges = pd.cut(confidence, bins=bins, labels=False, retbins=True, include_lowest=True)
    bin_ids = np.asarray(bin_ids, dtype=int)
    mean_confidence: list[float] = []
    accuracy: list[float] = []
    counts: list[int] = []
    bin_left: list[float] = []
    bin_right: list[float] = []
    for bin_index in range(bins):
        mask = bin_ids == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_confidence.append(float(confidence[mask].mean()))
        accuracy.append(float(correct[mask].mean()))
        counts.append(count)
        bin_left.append(float(edges[bin_index]))
        bin_right.append(float(edges[bin_index + 1]))
    return {
        "bin_left": np.asarray(bin_left, dtype=float),
        "bin_right": np.asarray(bin_right, dtype=float),
        "mean_confidence": np.asarray(mean_confidence, dtype=float),
        "accuracy": np.asarray(accuracy, dtype=float),
        "count": np.asarray(counts, dtype=int),
    }


def tolerance_ece(confidence: np.ndarray, correct: np.ndarray, bins: int) -> float:
    curve = tolerance_reliability_curve(confidence, correct, bins)
    counts = curve["count"].astype(float)
    total = counts.sum()
    gap = np.abs(curve["accuracy"] - curve["mean_confidence"])
    return float(np.sum((counts / total) * gap))


def parse_prediction(row: dict[str, Any]) -> dict[str, Any]:
    if "prediction" in row and isinstance(row["prediction"], dict):
        return row["prediction"]
    if "completion" in row:
        return json.loads(row["completion"])
    raise KeyError("Each row needs either a prediction object or a completion JSON string.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_float(value: float) -> float:
    if value is None or math.isnan(float(value)):
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
