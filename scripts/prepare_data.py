#!/usr/bin/env python3
"""Normalize a local unlabeled JSON/JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medical_parsing.data.preparation import prepare_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local unlabeled medical-image records.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(prepare_records(args.input, args.output, image_root=args.image_root), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
