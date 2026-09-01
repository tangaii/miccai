#!/usr/bin/env python3
"""Run the public three-branch inference pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medical_parsing.inference import run_inference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run classification, multi-label, and regression inference.")
    parser.add_argument("--input", required=True, type=Path, help="unlabeled JSONL/JSON input")
    parser.add_argument("--output", required=True, type=Path, help="canonical prediction JSONL")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="external fitted assets directory")
    parser.add_argument("--base", type=Path, default=Path(os.environ["MEDICAL_PARSING_BASE"]) if os.environ.get("MEDICAL_PARSING_BASE") else None)
    parser.add_argument("--adapter", type=Path, default=Path(os.environ["MEDICAL_PARSING_ADAPTER"]) if os.environ.get("MEDICAL_PARSING_ADAPTER") else None)
    parser.add_argument("--reg-adapter", type=Path, default=Path(os.environ["MEDICAL_PARSING_REG_ADAPTER"]) if os.environ.get("MEDICAL_PARSING_REG_ADAPTER") else None)
    parser.add_argument("--device", default=os.environ.get("MEDICAL_PARSING_DEVICE", "cuda:0"))
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="validate rows without loading external models/assets")
    args = parser.parse_args()
    audit = run_inference(
        args.input, args.output, base_path=args.base, adapter_path=args.adapter,
        regression_adapter_path=args.reg_adapter, device=args.device,
        config_path=args.config, checkpoint_dir=args.checkpoint_dir,
        audit_path=args.audit_json, dry_run=args.dry_run,
    )
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
