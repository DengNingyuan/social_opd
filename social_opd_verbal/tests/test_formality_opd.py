from __future__ import annotations

import pytest

from metrics import compute_measurement_metrics, parse_completion
from prompts import build_formality_opd_prompt, build_prompt_from_dataset
from constructs import get_construct
from prompts import build_student_prompt


def test_teacher_formality_prompt_adds_three_angle_instruction() -> None:
    student = build_formality_opd_prompt("Hello there.", role="student")
    teacher = build_formality_opd_prompt("Hello there.", role="teacher")

    assert "Data factors" not in student
    assert "Data factors" in teacher
    assert "Task factors" in teacher
    assert "Annotator factors" in teacher
    assert '{"score": <integer 0-100>, "confidence": <number 0-1>}' in teacher


def test_prompt_can_be_driven_by_dataset_config() -> None:
    dataset = {
        "attribute_name": "politeness",
        "attribute_definition": "Rate how polite the text is.",
        "teacher_instruction": "Consider respectfulness before deciding. Do not output this analysis.",
    }

    student = build_prompt_from_dataset("Thanks for your help.", dataset, role="student", tolerance=5)
    teacher = build_prompt_from_dataset("Thanks for your help.", dataset, role="teacher", tolerance=5)

    assert "politeness: Rate how polite the text is." in student
    assert "formality:" not in student
    assert "Consider respectfulness" in teacher
    assert "+/-5 point tolerance" in teacher


def test_maud_student_prompt_still_returns_diagnostic_prompt() -> None:
    prompt = build_student_prompt(get_construct("formality"), "Hello there.", output_format="maud")

    assert prompt is not None
    assert "Uncertainty dimensions:" in prompt
    assert "uncertainty_scores" in prompt


def test_parse_completion_reads_json_score_and_confidence() -> None:
    parsed = parse_completion('{"score": 73, "confidence": 0.81}')

    assert parsed == {"score": 73.0, "confidence": 0.81}


def test_compute_accuracy_ece_returns_requested_names() -> None:
    rows = [
        {"ground_truth_score": 50, "prediction": {"score": 55, "confidence": 0.8}},
        {"ground_truth_score": 10, "prediction": {"score": 40, "confidence": 0.2}},
    ]

    metrics = compute_measurement_metrics(rows)

    assert metrics["tolerance"] == 10
    assert metrics["ece_bins"] == 10
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert "ece" in metrics
    assert "brier" in metrics
    assert "mh" in metrics


def test_compute_measurement_metrics_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="empty prediction set"):
        compute_measurement_metrics([])
