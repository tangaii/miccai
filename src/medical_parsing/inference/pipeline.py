"""Public inference entry point and output contract."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

from medical_parsing.config import AssetBundle, RuntimeConfig, load_config
from medical_parsing.models.backbone import configure_environment, set_determinism
from medical_parsing.schema import (
    TASK_CLASSIFICATION,
    TASK_MULTILABEL,
    TASK_REGRESSION,
    atomic_write_jsonl,
    read_records,
    validate_input_rows,
    validate_output_rows,
)
from medical_parsing.tasks.classification import run_classification
from medical_parsing.tasks.multilabel import ATOMS, run_multilabel
from medical_parsing.tasks.regression import run_regression


def _required_asset_names(tasks: set[str]) -> set[str]:
    required: set[str] = set()
    if TASK_CLASSIFICATION in tasks:
        required.update({"classification_routes", "classification_heads"})
    if TASK_MULTILABEL in tasks:
        required.update({
            "multilabel_templates", "multilabel_library", "multilabel_selector",
            "multilabel_ranker", "multilabel_probability_models", "multilabel_residual_head",
        })
    if TASK_REGRESSION in tasks:
        required.update({
            "regression_visual_model", "regression_reference", "regression_residuals",
            "regression_quantile_head",
        })
    return required


def _asset_audit(assets: AssetBundle, tasks: set[str]) -> dict[str, Any]:
    missing = [str(assets.path(name)) for name in sorted(_required_asset_names(tasks)) if not assets.path(name).is_file()]
    return {
        "root": str(assets.root),
        "tasks": sorted(tasks),
        "required_files": len(_required_asset_names(tasks)),
        "missing": missing,
        "status": "PASS" if not missing else "INCOMPLETE",
    }


def run_inference(
    input_path: str | Path,
    output_path: str | Path,
    *,
    base_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
    regression_adapter_path: str | Path | None = None,
    device: str = "cuda:0",
    config_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    audit_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run all task branches represented in an unlabeled input JSONL.

    Base-model and fitted-task assets are intentionally arguments rather than
    package data.  A dry run checks the input and output contract without
    loading external models or checkpoints.
    """

    configure_environment()
    config = load_config(config_path)
    if checkpoint_dir is not None:
        config = RuntimeConfig(
            model=config.model,
            multilabel=config.multilabel,
            regression=config.regression,
            checkpoint_dir=Path(checkpoint_dir),
        )
    source = Path(input_path)
    target = Path(output_path)
    rows = validate_input_rows(read_records(source), image_root=source.resolve().parent)
    tasks = {row["task_type"] for row in rows}
    assets = AssetBundle(config.checkpoint_dir)
    audit: dict[str, Any] = {
        "status": "RUNNING",
        "rows": len(rows),
        "tasks": dict(Counter(row["task_type"] for row in rows)),
        "determinism": set_determinism(config.model.seed),
        "assets": _asset_audit(assets, tasks),
    }
    if dry_run:
        if target.exists():
            raise RuntimeError(f"dry-run refuses to overwrite an existing output: {target}")
        audit.update({"status": "PASS", "mode": "dry-run"})
        if audit_path is not None:
            audit_target = Path(audit_path)
            audit_target.parent.mkdir(parents=True, exist_ok=True)
            audit_target.write_text(json.dumps(audit, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        return audit
    if audit["assets"]["missing"]:
        raise FileNotFoundError(
            "missing external assets:\n" + "\n".join(audit["assets"]["missing"])
        )
    if base_path is None or adapter_path is None:
        raise ValueError("base_path and adapter_path are required for inference")
    if TASK_REGRESSION in tasks and regression_adapter_path is None:
        raise ValueError("regression_adapter_path is required when regression rows are present")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)
    predictions: dict[str, str] = {}
    started = time.monotonic()
    predictions.update(run_classification(
        by_task[TASK_CLASSIFICATION], Path(base_path), Path(adapter_path), device,
        assets, config.model, audit.setdefault("classification", {}),
    ))
    predictions.update(run_multilabel(
        by_task[TASK_MULTILABEL], Path(base_path), Path(adapter_path), device,
        assets, config.model, config.multilabel, audit.setdefault("multilabel", {}),
    ))
    if regression_adapter_path is not None:
        predictions.update(run_regression(
            by_task[TASK_REGRESSION], Path(base_path), Path(adapter_path),
            Path(regression_adapter_path), device, assets, config.model,
            audit.setdefault("regression", {}), regression_config=config.regression,
        ))
    output_rows = [
        {"uid": row["uid"], "task_type": row["task_type"], "prediction": predictions[row["uid"]]}
        for row in rows
    ]
    audit["output_validation"] = validate_output_rows(rows, output_rows, set(ATOMS))
    atomic_write_jsonl(target, output_rows)
    audit.update({"status": "PASS", "runtime_seconds": time.monotonic() - started, "output": str(target)})
    if audit_path is not None:
        audit_target = Path(audit_path)
        audit_target.parent.mkdir(parents=True, exist_ok=True)
        audit_target.write_text(json.dumps(audit, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return audit


__all__ = ["run_inference"]
