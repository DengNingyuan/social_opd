from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def load_named_config(path: Path, name: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if name not in data:
        available = ", ".join(sorted(data))
        raise KeyError(f"{name!r} not found in {path}. Available: {available}")
    item = dict(data[name])
    item.setdefault("name", name)
    return item


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return value.strip("-") or "model"


def default_run_name(
    run_type: str,
    dataset_name: str,
    loss_version: str,
    teacher_version: str,
    model_name: str,
    date: str | None = None,
) -> str:
    return "_".join(
        [
            date or timestamp(),
            run_type,
            loss_version,
            dataset_name,
            teacher_version,
            slugify(model_name),
        ]
    )


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
