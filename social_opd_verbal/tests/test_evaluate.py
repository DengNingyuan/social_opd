from __future__ import annotations

import numpy as np
import pytest

from evaluate import compute_metrics, tolerance_ece, tolerance_reliability_curve


def test_compute_metrics_parses_completion_once_shape() -> None:
    rows = [
        {
            "ground_truth_score": 80,
            "completion": (
                '{"score": 75, "confidence": 0.7, "uncertainty_scores": '
                '{"data_context": 0.2, "task_construct": 0.3, "annotator_perspective": 0.4}}'
            ),
        },
        {
            "ground_truth_score": 20,
            "prediction": {
                "score": 40,
                "confidence": 0.2,
                "uncertainty_scores": {
                    "data_context": 0.7,
                    "task_construct": 0.6,
                    "annotator_perspective": 0.5,
                },
            },
        },
    ]

    metrics = compute_metrics(rows, tolerance=10, ece_bins=10)

    assert metrics["n"] == 2
    assert metrics["tolerance"] == 10
    assert metrics["ece_bins"] == 10
    assert metrics["mae"] == pytest.approx(12.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["tolerance_accuracy"] == pytest.approx(0.5)
    assert "confidence_uncertainty_spearman" in metrics


def test_tolerance_ece_uses_ten_equal_confidence_bins() -> None:
    confidence = np.array([0.05, 0.15, 0.95])
    correct = np.array([1.0, 0.0, 1.0])

    ece = tolerance_ece(confidence, correct, bins=10)
    curve = tolerance_reliability_curve(confidence, correct, bins=10)

    assert len(curve["count"]) == 3
    assert curve["bin_left"][0] < 0.05
    assert curve["bin_right"][-1] == pytest.approx(0.95)
    assert ece == pytest.approx((abs(1.0 - 0.05) + abs(0.0 - 0.15) + abs(1.0 - 0.95)) / 3)


def test_compute_metrics_counts_error_equal_to_tolerance_as_correct() -> None:
    rows = [
        {"ground_truth_score": 50, "prediction": {"score": 60, "confidence": 0.9}},
        {"ground_truth_score": 50, "prediction": {"score": 61, "confidence": 0.1}},
    ]

    metrics = compute_metrics(rows)

    assert metrics["tolerance"] == 10
    assert metrics["ece_bins"] == 10
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_compute_metrics_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="empty prediction set"):
        compute_metrics([])
