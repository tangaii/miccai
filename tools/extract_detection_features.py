#!/usr/bin/env python3
"""Cache the frozen primary-adapter features consumed by the spatial decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medical_parsing.config import load_config  # noqa: E402
from medical_parsing.module_names import PAPER_MODULES  # noqa: E402
from medical_parsing.models.backbone import clear_model, load_adapter_bundle  # noqa: E402
from medical_parsing.schema import TASK_DETECTION, read_records, validate_input_rows  # noqa: E402
from medical_parsing.tasks.detection import (  # noqa: E402
    SPATIAL_QUERY_FEATURE_BATCH_SIZE,
    SPATIAL_QUERY_IMAGE_SIZE,
    extract_detection_features,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract [N,256,2560] image tokens and [N,2560] query states for the spatial decoder.",
    )
    parser.add_argument("--input", required=True, type=Path, help="unlabeled Detection JSONL/JSON")
    parser.add_argument("--base", required=True, type=Path, help="local MedGemma base directory")
    parser.add_argument("--adapter", required=True, type=Path, help="local primary LoRA adapter directory")
    parser.add_argument("--output", required=True, type=Path, help="output NPZ feature cache")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.model.image_size != SPATIAL_QUERY_IMAGE_SIZE:
        raise ValueError(
            "spatial-query feature extraction requires "
            f"model.image_size={SPATIAL_QUERY_IMAGE_SIZE}; "
            f"got {config.model.image_size}"
        )
    rows = validate_input_rows(
        read_records(args.input),
        image_root=args.input.resolve().parent,
    )
    if any(row["task_type"] != TASK_DETECTION for row in rows):
        raise ValueError("extract_detection_features.py accepts Detection rows only")
    model, processor, model_audit = load_adapter_bundle(
        args.base, args.adapter, args.device, config.model,
    )
    try:
        image_tokens, query_states = extract_detection_features(
            model, processor, rows, args.device, config.model,
            feature_batch_size=SPATIAL_QUERY_FEATURE_BATCH_SIZE,
        )
    finally:
        clear_model(model, processor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    uids = [row["uid"] for row in rows]
    uid_sha256 = hashlib.sha256(
        ("\n".join(uids) + "\n").encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "spatial_query_feature_cache_v1",
        "module": PAPER_MODULES["frozen_spatial_query_decoding"],
        "image_token_shape": list(image_tokens.shape),
        "query_state_shape": list(query_states.shape),
        "dtype": "float16",
        "image_size": SPATIAL_QUERY_IMAGE_SIZE,
        "feature_batch_size": SPATIAL_QUERY_FEATURE_BATCH_SIZE,
        "uid_sha256": uid_sha256,
    }
    np.savez_compressed(
        args.output,
        image_tokens=image_tokens,
        query_states=query_states,
        uid=np.asarray(uids),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    print(json.dumps({
        "status": "PASS",
        "rows": len(rows),
        "image_tokens_shape": list(image_tokens.shape),
        "query_states_shape": list(query_states.shape),
        "dtype": "float16",
        "image_size": SPATIAL_QUERY_IMAGE_SIZE,
        "feature_batch_size": SPATIAL_QUERY_FEATURE_BATCH_SIZE,
        "uid_sha256": uid_sha256,
        "model": model_audit,
        "output": str(args.output),
    }, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
