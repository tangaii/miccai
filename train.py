#!/usr/bin/env python3
"""Dispatch fitting jobs for task heads and external estimators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from medical_parsing.training.classification import train_semantic_heads  # noqa: E402
from medical_parsing.training.multilabel import (  # noqa: E402
    fit_candidate_models,
    fit_probability_models,
    train_multilabel_residual_head,
)
from medical_parsing.training.regression import (  # noqa: E402
    fit_reference,
    fit_visual_regressor,
    train_quantile_head,
)


def _load(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return {key: payload[key] for key in payload.files}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit one public task component from prepared arrays.")
    parser.add_argument("--task", required=True, choices=("classification", "multilabel", "regression"))
    parser.add_argument("--features", required=True, type=Path, help="NPZ feature/target archive")
    parser.add_argument("--labels", type=Path, default=None, help="JSONL labels for semantic classification")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--secondary-output", type=Path, default=None, help="second estimator output for fitting pairs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    arrays = _load(args.features)
    if args.task == "classification":
        if args.labels is None:
            raise ValueError("--labels JSONL is required for classification")
        labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = train_semantic_heads(
            arrays["tokens"], [row["dataset"] for row in labels],
            [row["slot"] for row in labels], [row["label"] for row in labels],
            args.output, seed=args.seed,
        )
    elif args.task == "multilabel":
        if "selector_features" in arrays:
            if args.secondary_output is None:
                raise ValueError("--secondary-output ranker path is required for candidate fitting")
            result = fit_candidate_models(
                arrays["selector_features"], arrays["selector_targets"],
                arrays["ranker_features"], arrays["ranker_targets"],
                args.output, args.secondary_output, ranker_groups=arrays.get("ranker_groups"), seed=args.seed,
            )
        elif "row_features" in arrays:
            result = train_multilabel_residual_head(
                arrays["tokens"], arrays["row_features"], arrays["base_probability"],
                arrays["targets"], args.output, seed=args.seed,
            )
        else:
            result = fit_probability_models(arrays["features"], arrays["targets"], args.output, seed=args.seed)
    else:
        if "visual_features" in arrays and "geometry" in arrays and "uids" in arrays and "groups" in arrays:
            result = fit_reference(
                arrays["visual_features"], arrays["geometry"], arrays["targets"],
                arrays["uids"], arrays["groups"], args.output,
            )
        elif "tokens" in arrays and "geometry" in arrays:
            result = train_quantile_head(arrays["tokens"], arrays["geometry"], arrays["targets"], args.output, seed=args.seed)
        else:
            result = fit_visual_regressor(arrays["visual_features"], arrays["targets"], args.output, seed=args.seed)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
