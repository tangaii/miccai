"""Small, dependency-light metrics for the public output task types."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from medical_parsing.schema import (
    TASK_CLASSIFICATION,
    TASK_DETECTION,
    TASK_MULTILABEL,
    TASK_REGRESSION,
    canonical_task,
    parse_choices,
    parse_detection_boxes,
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
        "gold", "gold_answer", "target_value", "raw_target", "Answer",
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


def detection_iou(
    box_a: tuple[float, float, float, float, str | None],
    box_b: tuple[float, float, float, float, str | None],
) -> float:
    """Compute xyxy IoU for parser-normalized detection boxes."""

    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denominator = area_a + area_b - inter_area
    return 0.0 if denominator <= 0.0 else inter_area / denominator


def greedy_detection_match_counts(
    gold_boxes: list[tuple[float, float, float, float, str | None]],
    predicted_boxes: list[tuple[float, float, float, float, str | None]],
    threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Match boxes by descending IoU using the organizer's one-to-one rule."""

    candidates: list[tuple[float, int, int]] = []
    for gold_index, gold in enumerate(gold_boxes):
        for prediction_index, prediction in enumerate(predicted_boxes):
            if gold[4] and prediction[4] and gold[4] != prediction[4]:
                continue
            overlap = detection_iou(gold, prediction)
            if overlap >= float(threshold):
                candidates.append((overlap, gold_index, prediction_index))
    candidates.sort(reverse=True)
    matched_gold: set[int] = set()
    matched_prediction: set[int] = set()
    true_positive = 0
    for _overlap, gold_index, prediction_index in candidates:
        if gold_index in matched_gold or prediction_index in matched_prediction:
            continue
        matched_gold.add(gold_index)
        matched_prediction.add(prediction_index)
        true_positive += 1
    return (
        true_positive,
        len(predicted_boxes) - true_positive,
        len(gold_boxes) - true_positive,
    )


def detection_metrics(
    gold_values: list[Any],
    predicted_values: list[Any],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return micro precision/recall/F1 and counts at an IoU threshold."""

    if len(gold_values) != len(predicted_values):
        raise ValueError("detection reference/prediction lengths differ")
    true_positive = false_positive = false_negative = 0
    invalid_predictions = 0
    for gold_value, predicted_value in zip(gold_values, predicted_values):
        gold_boxes = parse_detection_boxes(gold_value)
        try:
            predicted_boxes = parse_detection_boxes(predicted_value)
        except ValueError:
            # The organizer-side diagnostic evaluator treats malformed
            # predictions as an empty predicted list.
            predicted_boxes = []
            invalid_predictions += 1
        tp, fp, fn = greedy_detection_match_counts(gold_boxes, predicted_boxes, threshold)
        true_positive += tp
        false_positive += fp
        false_negative += fn
    precision, recall, f1 = _f1(
        true_positive + false_positive,
        true_positive + false_negative,
        true_positive,
    )
    return {
        "iou_threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_score": f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "invalid_predictions": invalid_predictions,
    }


iou = detection_iou
greedy_match_counts = greedy_detection_match_counts


def evaluate_rows(
    reference_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if [str(row.get("uid")) for row in reference_rows] != [str(row.get("uid")) for row in prediction_rows]:
        raise ValueError("reference/prediction UID order or coverage mismatch")
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        TASK_CLASSIFICATION: [], TASK_MULTILABEL: [], TASK_DETECTION: [], TASK_REGRESSION: [],
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
    detection_pairs = grouped[TASK_DETECTION]
    if detection_pairs:
        result["tasks"][TASK_DETECTION] = {
            "rows": len(detection_pairs),
            **detection_metrics(
                [_reference_value(reference) for reference, _ in detection_pairs],
                [prediction["prediction"] for _, prediction in detection_pairs],
                threshold=0.5,
            ),
        }
    return result


def evaluate_files(reference_path: str | Path, prediction_path: str | Path) -> dict[str, Any]:
    return evaluate_rows(_read_any(reference_path), _read_any(prediction_path))


__all__ = [
    "detection_iou", "detection_metrics", "evaluate_files", "evaluate_rows",
    "greedy_detection_match_counts", "greedy_match_counts", "iou",
]
