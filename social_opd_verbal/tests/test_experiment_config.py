from __future__ import annotations

import pytest

from experiment_config import default_run_name, slugify


def test_slugify_model_name_for_output_path() -> None:
    assert slugify("Qwen/Qwen2.5-3B-Instruct") == "Qwen-Qwen2.5-3B-Instruct"


def test_default_run_name_contains_versions_dataset_and_model() -> None:
    name = default_run_name(
        run_type="train",
        dataset_name="formality",
        loss_version="lossv1",
        teacher_version="teacherv1",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        date="20260605_123000",
    )

    assert name == "20260605_123000_train_lossv1_formality_teacherv1_Qwen-Qwen2.5-3B-Instruct"
