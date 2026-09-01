#!/usr/bin/env python3
"""Evaluate canonical predictions against a user-supplied labeled file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medical_parsing.evaluation import evaluate_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate canonical prediction JSONL against labeled reference JSONL.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON summary path")
    args = parser.parse_args()
    result = evaluate_files(args.reference, args.predictions)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
