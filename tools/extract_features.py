#!/usr/bin/env python3
"""Extract shared image-token features for downstream head fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medical_parsing.config import load_config  # noqa: E402
from medical_parsing.models.backbone import extract_image_tokens, load_raw_bundle  # noqa: E402
from medical_parsing.schema import read_records, validate_input_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract [N,256,2560] features from an external base model.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = validate_input_rows(read_records(args.input), image_root=args.input.resolve().parent)
    model, processor, audit = load_raw_bundle(args.base, args.device, config.model)
    tokens = extract_image_tokens(model, processor, rows, args.device, config.model, prompt="image feature")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tokens=tokens, uid=np.asarray([row["uid"] for row in rows]))
    print(json.dumps({"status": "PASS", "rows": len(rows), "shape": list(tokens.shape), "model": audit}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
