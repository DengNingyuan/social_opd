from __future__ import annotations

import json

import pandas as pd

import scripts.deepseek_three_aspects as deepseek_three_aspects


def test_build_prompt_requests_three_aspect_text_and_scores() -> None:
    dataset = {"attribute_name": "formality", "attribute_definition": "Rate linguistic formality."}

    prompt = deepseek_three_aspects.build_prompt("Hello there.", dataset)

    assert '"data_factors": {"description": "<brief text>", "score": <integer 0-100>}' in prompt
    assert '"task_factors": {"description": "<brief text>", "score": <integer 0-100>}' in prompt
    assert '"annotator_factors": {"description": "<brief text>", "score": <integer 0-100>}' in prompt
    assert '"score": <integer 0-100>' in prompt


def test_parse_response_reads_aspect_descriptions_and_scores() -> None:
    raw = json.dumps(
        {
            "data_factors": {"description": "short but clear", "score": 20},
            "task_factors": {"description": "scale is straightforward", "score": 10},
            "annotator_factors": {"description": "some disagreement possible", "score": 35},
            "score": 62,
        }
    )

    parsed = deepseek_three_aspects.parse_response(raw)

    assert parsed["data_factors"] == {"description": "short but clear", "score": 20}
    assert parsed["task_factors"] == {"description": "scale is straightforward", "score": 10}
    assert parsed["annotator_factors"] == {"description": "some disagreement possible", "score": 35}
    assert parsed["score"] == 62


def test_read_rows_uses_csv_path_and_text_column(tmp_path) -> None:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame([{"text": "Hello.", "ground_truth": 40}]).to_csv(csv_path, index=False)

    rows = deepseek_three_aspects.read_rows(csv_path, text_column="text", limit=None)

    assert rows == [{"id": "sample:0", "text": "Hello.", "ground_truth_score": 40.0}]


def test_default_output_dir_points_to_commercial_api() -> None:
    assert str(deepseek_three_aspects.OUTPUT_DIR) == "/home/nydeng/social_opd_all/social_opd_geo/data/commercial_api"
