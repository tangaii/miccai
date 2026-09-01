"""Small, dependency-light metrics for the three output task types."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from medical_parsing.schema import (
    TASK_CLASSIFICATION,
    TASK_MULTILABEL,
    TASK_REGRESSION,
    canonical_task,
    parse_choices,
    parse_label_set,
    read_records,
)


def _read_any(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("rows", value.get("data", value))
        if not isinstance(value, list):
            raise ValueError(f"expected a list of records in {source}")
        return [dict(item) for item in value]
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reference_value(row: dict[str, Any]) -> Any:
    for key in (
        "answer", "raw_answer", "target", "reference", "label", "ground_truth",
        "gold", "gold_answer", "target_value", "raw_target",
    ):
        if key in row:
            return row[key]
    raise ValueError(f"reference row has no supported label key: {row.get('uid')}")


def _classification_value(value: Any, question: str) -> str:
    text = str(value).strip().upper()
    choices = parse_choices(question)
    legal = {letter for letter, _ in choices}
    if text in legal:
        return text
    normalized = str(value).strip().lower()
    for letter, label in choices:
        if normalized == label.lower():
            return letter
    raise ValueError(f"cannot map classification label {value!r}")


def _f1(precision_n: int, recall_n: int, true_positive: int) -> tuple[float, float, float]:
    precision = true_positive / precision_n if precision_n else 0.0
    recall = true_positive / recall_n if recall_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_rows(
    reference_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if [str(row.get("uid")) for row in reference_rows] != [str(row.get("uid")) for row in prediction_rows]:
        raise ValueError("reference/prediction UID order or coverage mismatch")
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        TASK_CLASSIFICATION: [], TASK_MULTILABEL: [], TASK_REGRESSION: [],
    }
    for reference, prediction in zip(reference_rows, prediction_rows):
        task = canonical_task(reference.get("task_type", reference.get("task")))
        if canonical_task(prediction.get("task_type", prediction.get("task"))) != task:
            raise ValueError(f"task mismatch for {reference.get('uid')}")
        grouped[task].append((reference, prediction))
    result: dict[str, Any] = {"status": "PASS", "rows": len(reference_rows), "tasks": {}}

    cls_pairs = grouped[TASK_CLASSIFICATION]
    if cls_pairs:
        correct = 0
        for reference, prediction in cls_pairs:
            question = str(reference.get("source_question") or reference.get("question") or reference.get("prompt"))
            expected = _classification_value(_reference_value(reference), question)
            actual = _classification_value(prediction["prediction"], question)
            correct += int(expected == actual)
        result["tasks"][TASK_CLASSIFICATION] = {
            "rows": len(cls_pairs), "accuracy": correct / len(cls_pairs),
        }

    mlc_pairs = grouped[TASK_MULTILABEL]
    if mlc_pairs:
        exact = 0
        predicted_total = reference_total = true_positive = 0
        sample_f1: list[float] = []
        for reference, prediction in mlc_pairs:
            expected = parse_label_set(_reference_value(reference))
            actual = parse_label_set(prediction["prediction"])
            exact += int(expected == actual)
            tp = len(expected & actual)
            predicted_total += len(actual)
            reference_total += len(expected)
            true_positive += tp
            _, _, row_f1 = _f1(len(actual), len(expected), tp)
            sample_f1.append(row_f1)
        precision, recall, f1 = _f1(predicted_total, reference_total, true_positive)
        result["tasks"][TASK_MULTILABEL] = {
            "rows": len(mlc_pairs), "exact_match": exact / len(mlc_pairs),
            "micro_precision": precision, "micro_recall": recall, "micro_f1": f1,
            "sample_f1": float(np.mean(sample_f1)),
        }

    reg_pairs = grouped[TASK_REGRESSION]
    if reg_pairs:
        expected = np.asarray([float(_reference_value(reference)) for reference, _ in reg_pairs], dtype=np.float64)
        actual = np.asarray([float(prediction["prediction"]) for _, prediction in reg_pairs], dtype=np.float64)
        error = actual - expected
        result["tasks"][TASK_REGRESSION] = {
            "rows": len(reg_pairs), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
        }
    return result


def evaluate_files(reference_path: str | Path, prediction_path: str | Path) -> dict[str, Any]:
    return evaluate_rows(_read_any(reference_path), _read_any(prediction_path))


__all__ = ["evaluate_files", "evaluate_rows"]
