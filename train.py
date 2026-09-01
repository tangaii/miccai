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
    write_crossfitted_residuals,
)


COMPONENTS = (
    "classification-head",
    "multilabel-selector-ranker",
    "multilabel-probability-models",
    "multilabel-residual-head",
    "regression-visual-estimator",
    "regression-reference",
    "regression-residuals",
    "regression-quantile-head",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return {key: payload[key] for key in payload.files}


def _infer_component(task: str | None, arrays: dict[str, np.ndarray]) -> str:
    if task == "classification":
        return "classification-head"
    if task == "multilabel":
        if "selector_features" in arrays:
            return "multilabel-selector-ranker"
        if "row_features" in arrays:
            return "multilabel-residual-head"
        return "multilabel-probability-models"
    if task == "regression":
        if {"visual_features", "geometry", "uids", "groups"}.issubset(arrays):
            return "regression-reference"
        if {"tokens", "geometry"}.issubset(arrays):
            return "regression-quantile-head"
        return "regression-visual-estimator"
    raise ValueError("--component is required when --task is omitted")


def _require(arrays: dict[str, np.ndarray], *names: str) -> None:
    missing = [name for name in names if name not in arrays]
    if missing:
        raise ValueError(f"feature archive is missing keys for component: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit one named public method component from prepared arrays.")
    parser.add_argument("--task", choices=("classification", "multilabel", "regression"), default=None,
                        help="legacy task selector; component inference remains supported")
    parser.add_argument("--component", choices=COMPONENTS, default=None,
                        help="explicit paper component; preferred over implicit NPZ-key dispatch")
    parser.add_argument("--features", required=True, type=Path, help="NPZ feature/target archive")
    parser.add_argument("--labels", type=Path, default=None, help="JSONL labels for semantic classification")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--secondary-output", type=Path, default=None, help="second estimator output for fitting pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="torch device for neural components")
    args = parser.parse_args()
    arrays = _load(args.features)
    component = args.component or _infer_component(args.task, arrays)
    if args.task is not None:
        expected_prefix = {
            "classification": "classification-",
            "multilabel": "multilabel-",
            "regression": "regression-",
        }[args.task]
        if not component.startswith(expected_prefix):
            raise ValueError(f"--task {args.task!r} is incompatible with --component {component!r}")

    if component == "classification-head":
        if args.labels is None:
            raise ValueError("--labels JSONL is required for classification")
        _require(arrays, "tokens")
        labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = train_semantic_heads(
            arrays["tokens"], [row["dataset"] for row in labels],
            [row["slot"] for row in labels], [row["label"] for row in labels],
            args.output, seed=args.seed, device=args.device,
        )
    elif component == "multilabel-selector-ranker":
        if args.secondary_output is None:
            raise ValueError("--secondary-output ranker path is required for candidate fitting")
        _require(arrays, "selector_features", "selector_targets", "ranker_features", "ranker_targets")
        result = fit_candidate_models(
            arrays["selector_features"], arrays["selector_targets"],
            arrays["ranker_features"], arrays["ranker_targets"],
            args.output, args.secondary_output, ranker_groups=arrays.get("ranker_groups"), seed=args.seed,
        )
    elif component == "multilabel-probability-models":
        _require(arrays, "features", "targets")
        result = fit_probability_models(arrays["features"], arrays["targets"], args.output, seed=args.seed)
    elif component == "multilabel-residual-head":
        _require(arrays, "tokens", "row_features", "base_probability", "targets")
        result = train_multilabel_residual_head(
            arrays["tokens"], arrays["row_features"], arrays["base_probability"],
            arrays["targets"], args.output, seed=args.seed, device=args.device,
        )
    elif component == "regression-reference":
        _require(arrays, "visual_features", "geometry", "targets", "uids", "groups")
        result = fit_reference(
            arrays["visual_features"], arrays["geometry"], arrays["targets"],
            arrays["uids"], arrays["groups"], args.output,
        )
    elif component == "regression-residuals":
        _require(arrays, "uids")
        residual_key = "residuals" if "residuals" in arrays else "residual"
        _require(arrays, residual_key)
        result = write_crossfitted_residuals(arrays["uids"], arrays[residual_key], args.output)
    elif component == "regression-quantile-head":
        _require(arrays, "tokens", "geometry", "targets")
        result = train_quantile_head(
            arrays["tokens"], arrays["geometry"], arrays["targets"], args.output,
            seed=args.seed, device=args.device,
        )
    elif component == "regression-visual-estimator":
        _require(arrays, "visual_features", "targets")
        result = fit_visual_regressor(arrays["visual_features"], arrays["targets"], args.output, seed=args.seed)
    else:  # pragma: no cover - argparse constrains this branch
        raise ValueError(f"unsupported training component: {component}")
    result = dict(result)
    result["component"] = component
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
