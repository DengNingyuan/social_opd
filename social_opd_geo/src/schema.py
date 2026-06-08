from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DIMENSION_KEYS = ("data_context", "task_construct", "annotator_perspective")


@dataclass
class DiagnosticLabel:
    uncertainty_scores: dict[str, float]
    uncertainty_evidence: dict[str, str]

    def clipped(self) -> "DiagnosticLabel":
        scores = {k: _clip01(float(self.uncertainty_scores.get(k, 0.5))) for k in DIMENSION_KEYS}
        evidence = {k: str(self.uncertainty_evidence.get(k, "")) for k in DIMENSION_KEYS}
        return DiagnosticLabel(uncertainty_scores=scores, uncertainty_evidence=evidence)


@dataclass
class MAUDExample:
    id: str
    construct: str
    text: str
    ground_truth_score: float
    original_llm_score: float
    target: DiagnosticLabel
    student_prompt: str
    teacher_prompt: str
    completion: str

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["target"] = asdict(self.target)
        return data


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))
