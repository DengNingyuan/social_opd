from __future__ import annotations

import json

from constructs import UNCERTAINTY_DIMENSIONS, ConstructSpec
from schema import DIMENSION_KEYS, DiagnosticLabel


def build_student_prompt(construct: ConstructSpec, text: str, include_evidence: bool = True, output_format: str = "maud") -> str:
    if output_format == "scalar":
        return f"""You are measuring a social-science construct.

Construct: {construct.name}
Definition: {construct.definition}
Scale: 0 means {construct.low_anchor}; 100 means {construct.high_anchor}.

Task: rate the text and report scalar confidence.
Return strict JSON with keys: score, confidence.

Text:
{text}
"""
    evidence_line = (
        "Also include short evidence strings for each uncertainty dimension."
        if include_evidence
        else "Do not include evidence strings."
    )
    dims = "\n".join(f"- {name}: {desc}" for name, desc in UNCERTAINTY_DIMENSIONS.items())
    return f"""You are measuring a social-science construct.

Construct: {construct.name}
Definition: {construct.definition}
Scale: 0 means {construct.low_anchor}; 100 means {construct.high_anchor}.

Uncertainty dimensions:
{dims}

Task: rate the text and report diagnostic uncertainty. {evidence_line}
Return strict JSON with keys: score, confidence, uncertainty_scores, uncertainty_evidence.

Text:
{text}
"""


FORMALITY_DEFINITION = (
    "Rate the linguistic formality of the text. Near 0 means extremely informal, "
    "and near 100 means extremely formal, with intermediate values as appropriate. "
    "Use your intuitive judgment of formality and use the full ratings scale."
)
FORMALITY_TEACHER_INSTRUCTION = (
    "Before choosing the final score and confidence, reason carefully from the following "
    "three perspectives.\n\n"
    "1. Data factors: consider whether the text itself makes the judgment uncertain. "
    "This includes missing context, noisy or very short input, linguistic ambiguity, "
    "unclear tone, or cases where the wording supports multiple reasonable "
    "interpretations.\n\n"
    "2. Task factors: consider whether the measurement task could create uncertainty. "
    "This includes ambiguity in the construct definition, unclear boundaries of the "
    "0-100 scale, difficulty deciding between nearby score levels, or mismatch between "
    "the rubric and the text.\n\n"
    "3. Annotator factors: consider whether reasonable annotators might disagree because "
    "of different preferences, cultural backgrounds, social perspectives, lived "
    "experiences, values, or interpretation styles. Treat this as meaningful perspective "
    "variation, not simply annotation error."
)


def build_measurement_opd_prompt(
    text: str,
    attribute_name: str,
    attribute_definition: str,
    role: str = "student",
    teacher_instruction: str | None = None,
    tolerance: float = 10.0,
) -> str:
    analysis_line = ""
    if role == "teacher" and teacher_instruction:
        analysis_line = teacher_instruction.strip() + "\n"
    return f"""BEGIN TEXT ENTRY
{text}
END TEXT ENTRY
Read the entire text carefully. Do not skim; comprehend the whole text deeply, including subtle style cues.

Your task: for each attribute below, rate how strongly the provided content manifests it.

BEGIN ATTRIBUTES
{attribute_name}: {attribute_definition}
END ATTRIBUTES

BEGIN RATING SCALE
Use integers 0-100 (inclusive). low = absent, high = extreme, mid = moderate.
Use the full range and every increment; do not round to 5s/10s.
Extremes are rare: use near 0 only if truly absent and near 100 only if overwhelming.
Use moderate intermediates (e.g., 19, 67, 32) to account for nuance where applicable.
Aim for the rating that serves as the optimal center for a +/-{tolerance:g} point tolerance interval, ensuring the highest probability of capturing the true intensity.
END RATING SCALE

Method: pick one exact integer. Stick to the 0-100 scale. Double-check before choosing extremes.
Interpret gradations as: absent -> faint -> moderate -> abundant -> extreme.
Do not overlook subtlety; do not default to extremes. Consider the full spectrum, including intermediate gradations.
High accuracy and precision are critical; this needs deep, holistic analysis of content.
{analysis_line}
Also report confidence: the probability from 0 to 1 that your score is within +/-{tolerance:g} points of the true value.

Output JSON only, in this exact format:
{{"score": <integer 0-100>, "confidence": <number 0-1>}}
"""


def build_prompt_from_dataset(text: str, dataset: dict, role: str = "student", tolerance: float = 10.0) -> str:
    return build_measurement_opd_prompt(
        text=text,
        attribute_name=str(dataset["attribute_name"]),
        attribute_definition=str(dataset["attribute_definition"]),
        role=role,
        teacher_instruction=dataset.get("teacher_instruction"),
        tolerance=tolerance,
    )


def build_formality_opd_prompt(text: str, role: str = "student") -> str:
    return build_measurement_opd_prompt(
        text=text,
        attribute_name="formality",
        attribute_definition=FORMALITY_DEFINITION,
        role=role,
        teacher_instruction=FORMALITY_TEACHER_INSTRUCTION,
    )


def build_teacher_prompt(
    construct: ConstructSpec,
    text: str,
    diagnostic: DiagnosticLabel,
    include_evidence: bool = True,
    output_format: str = "maud",
) -> str:
    diag = diagnostic.clipped()
    context = {
        "confidence_hint": round(diag.confidence, 3),
        "uncertainty_scores": {k: round(diag.uncertainty_scores[k], 3) for k in DIMENSION_KEYS},
        "uncertainty_evidence": {k: diag.uncertainty_evidence[k] for k in DIMENSION_KEYS},
    }
    return (
        build_student_prompt(construct, text, include_evidence=include_evidence, output_format=output_format)
        + "\nPrivileged diagnostic context for the teacher:\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\nUse the diagnostic context to produce the same strict JSON output.\n"
    )


def build_completion(
    score: float,
    diagnostic: DiagnosticLabel,
    include_evidence: bool = True,
    output_format: str = "maud",
) -> str:
    diag = diagnostic.clipped()
    if output_format == "scalar":
        return json.dumps(
            {"score": round(float(score), 2), "confidence": round(diag.confidence, 3)},
            ensure_ascii=False,
            sort_keys=True,
        )
    payload = {
        "score": round(float(score), 2),
        "confidence": round(diag.confidence, 3),
        "uncertainty_scores": {k: round(diag.uncertainty_scores[k], 3) for k in DIMENSION_KEYS},
    }
    if include_evidence:
        payload["uncertainty_evidence"] = {k: diag.uncertainty_evidence[k] for k in DIMENSION_KEYS}
    else:
        payload["uncertainty_evidence"] = {}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
