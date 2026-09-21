#!/usr/bin/env python3
"""Convert labeled Detection answers to the spatial-query target format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medical_parsing.schema import (  # noqa: E402
    TASK_DETECTION,
    canonical_task,
    load_image,
    normalize_image_ref,
    parse_detection_boxes,
    read_records,
    row_image_refs,
)
from medical_parsing.training.detection import (  # noqa: E402
    boxes_to_normalized_targets,
    normalize_detection_uids,
)


def _answer(row: dict[str, Any]) -> Any:
    for key in ("answer", "Answer", "raw_answer", "target", "reference", "label"):
        if key in row:
            return row[key]
    raise ValueError(f"labeled row has no Detection answer: {row.get('uid')}")


def _image_uri(row: dict[str, Any], image_root: Path, source_root: Path) -> str:
    # The canonical public schema is preferred; ImagePath is accepted for
    # source metadata tables that have not yet been normalized.
    if any(key in row for key in ("images", "image_paths", "image", "image_path")):
        refs = row_image_refs(row)
    else:
        value = row.get("ImagePath", row.get("ImageName"))
        if value is None:
            raise ValueError(f"labeled row has no image reference: {row.get('uid')}")
        refs = [str(value)]
    if len(refs) != 1:
        raise ValueError(f"Detection target rows require exactly one image: {row.get('uid')}")
    return normalize_image_ref(refs[0], image_root=image_root, repo_root=source_root)


def prepare_targets(
    input_path: str | Path,
    output_path: str | Path,
    *,
    image_root: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output_path)
    image_base = Path(image_root) if image_root is not None else source.resolve().parent
    targets: list[list[list[float]]] = []
    uids: list[str] = []
    for index, row in enumerate(read_records(source)):
        if "task_type" in row or "task" in row:
            if canonical_task(row.get("task_type", row.get("task"))) != TASK_DETECTION:
                raise ValueError(f"row {row.get('uid', index)} is not a Detection row")
        uri = _image_uri(row, image_base, source.resolve().parent)
        boxes = parse_detection_boxes(_answer(row))
        width, height = load_image(uri).size
        targets.append(boxes_to_normalized_targets(boxes, width, height))
        raw_uid = row.get("uid")
        uid = str(raw_uid).strip() if raw_uid is not None else ""
        uids.append(uid or str(index))
    if not targets:
        raise ValueError("input contains no labeled Detection rows")
    uids = normalize_detection_uids(uids, len(targets), field_name="target uids")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "spatial_query_targets_v1",
        "uids": uids,
        "rows": targets,
        "box_counts": [len(row) for row in targets],
    }
    destination.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return {
        "status": "PASS",
        "rows": len(targets),
        "boxes": sum(len(row) for row in targets),
        "output": str(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare normalized targets for the spatial query decoder."
    )
    parser.add_argument("--input", required=True, type=Path, help="labeled JSON/JSONL")
    parser.add_argument("--output", required=True, type=Path, help="target JSON")
    parser.add_argument("--image-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(prepare_targets(args.input, args.output, image_root=args.image_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
