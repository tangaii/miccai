"""Prepare local, unlabeled JSON/JSONL records for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medical_parsing.schema import atomic_write_jsonl, read_records, validate_input_rows


def prepare_records(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    image_root: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(source_path)
    destination = Path(destination_path)
    root = Path(image_root) if image_root is not None else source.resolve().parent
    rows = validate_input_rows(read_records(source), image_root=root, repo_root=source.resolve().parent)
    atomic_write_jsonl(destination, rows)
    return {
        "status": "PASS", "rows": len(rows), "output": str(destination),
        "tasks": {task: sum(row["task_type"] == task for row in rows) for task in sorted({row["task_type"] for row in rows})},
    }


__all__ = ["prepare_records"]
