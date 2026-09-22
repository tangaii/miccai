"""Training utilities for the spatial query decoder."""

from __future__ import annotations

import json
import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import time
from typing import Any

import numpy as np

from medical_parsing.config import DetectionConfig
from medical_parsing.module_names import PAPER_MODULES
from medical_parsing.models.detection_head import (
    SPATIAL_QUERY_ATTENTION_HEADS,
    SPATIAL_QUERY_DECODER_FFN_DIM,
    SPATIAL_QUERY_DECODER_LAYERS,
    SPATIAL_QUERY_DROPOUT,
    SPATIAL_QUERY_HIDDEN_DIM,
    SPATIAL_QUERY_MAX_QUERIES,
    SPATIAL_QUERY_VISION_DIM,
    FrozenSpatialQueryDecoder,
)
from .common import seed_everything


SPATIAL_QUERY_MODULE_NAME = PAPER_MODULES["frozen_spatial_query_decoding"]


def cxcywh_to_xyxy(boxes: Any) -> Any:
    """Convert normalized center-size boxes to corner boxes."""

    cx, cy, width, height = boxes.unbind(-1)
    return __import__("torch").stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=-1,
    )


def generalized_iou(boxes1: Any, boxes2: Any) -> Any:
    """Compute GIoU for broadcastable ``xyxy`` tensors."""

    torch = __import__("torch")
    left_top = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    right_bottom = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp(min=0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp(min=0).prod(dim=-1)
    union = area1 + area2 - intersection
    iou = intersection / union.clamp(min=1e-8)
    enclosure_left = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    enclosure_right = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclosure = (enclosure_right - enclosure_left).clamp(min=0).prod(dim=-1)
    return iou - (enclosure - union) / enclosure.clamp(min=1e-8)


def _target_row(targets: Any) -> list[list[float]]:
    if isinstance(targets, Mapping):
        for key in ("boxes", "bboxes", "targets", "rows"):
            if key in targets:
                return _target_row(targets[key])
        raise ValueError(f"unsupported detection target mapping: {sorted(targets)}")
    if targets is None:
        return []
    if isinstance(targets, np.ndarray):
        targets = targets.tolist()
    if not isinstance(targets, (list, tuple)):
        raise ValueError(f"detection targets must be a list, got {type(targets).__name__}")
    if not targets:
        return []
    rows: list[list[float]] = []
    for box in targets:
        if isinstance(box, np.ndarray):
            box = box.tolist()
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"each normalized detection target must have four values: {box!r}")
        try:
            values = [float(value) for value in box]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"detection target contains a non-numeric value: {box!r}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"detection target contains a non-finite value: {box!r}")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"normalized detection target is outside [0,1]: {box!r}")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError(f"normalized detection target has non-positive size: {box!r}")
        rows.append(values)
    return rows


def normalize_detection_target_rows(targets: Any, expected_rows: int | None = None) -> list[list[list[float]]]:
    """Validate the JSON-safe ``[row][box][cx,cy,w,h]`` training contract."""

    if isinstance(targets, Mapping):
        targets = targets.get("rows", targets.get("targets"))
    if not isinstance(targets, (list, tuple)):
        raise ValueError("detection targets must be a list of per-row box lists")
    normalized = [_target_row(row) for row in targets]
    if expected_rows is not None and len(normalized) != int(expected_rows):
        raise ValueError(
            f"detection target row count mismatch: {len(normalized)} != {expected_rows}"
        )
    return normalized


def _uid_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def normalize_detection_uids(
    values: Any,
    expected_rows: int | None = None,
    *,
    field_name: str = "uids",
) -> list[str]:
    """Normalize and validate the optional row identity vector in a cache."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional, got {array.shape}")
    if expected_rows is not None and len(array) != int(expected_rows):
        raise ValueError(
            f"{field_name} row count mismatch: {len(array)} != {expected_rows}"
        )
    normalized = [_uid_text(value).strip() for value in array.tolist()]
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} contains an empty UID")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} contains duplicate UIDs")
    return normalized


def detection_feature_uids(
    arrays: Mapping[str, Any],
    expected_rows: int,
) -> list[str] | None:
    """Read the optional UID vector from a feature archive.

    Both ``uid`` (the public extractor's spelling) and ``uids`` (the common
    training-cache spelling) are accepted.  If both are present they must
    identify the same ordered rows.
    """

    candidates = [
        normalize_detection_uids(arrays[name], expected_rows, field_name=name)
        for name in ("uid", "uids")
        if name in arrays
    ]
    if not candidates:
        return None
    if len(candidates) == 2 and candidates[0] != candidates[1]:
        raise ValueError("feature archive contains conflicting uid and uids vectors")
    return candidates[0]


def detection_uid_sha256(uids: Sequence[Any]) -> str:
    """Return the ordered-UID digest used by the public feature extractor."""

    normalized = normalize_detection_uids(uids)
    return hashlib.sha256(("\n".join(normalized) + "\n").encode("utf-8")).hexdigest()


def validate_detection_feature_metadata(
    arrays: Mapping[str, Any],
    expected_rows: int,
) -> dict[str, Any]:
    """Validate metadata emitted by ``extract_detection_features.py``.

    Older, externally prepared archives may omit ``metadata`` and remain
    usable; when present, the metadata is treated as a reproducibility lock.
    """

    if "metadata" not in arrays:
        return {}
    value = np.asarray(arrays["metadata"])
    if value.ndim != 0:
        raise ValueError(f"feature metadata must be a scalar JSON string, got {value.shape}")
    try:
        metadata = json.loads(_uid_text(value.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("feature metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("feature metadata must be a JSON object")
    expected = {
        "schema": "spatial_query_feature_cache_v1",
        "module": SPATIAL_QUERY_MODULE_NAME,
        "image_token_shape": [int(expected_rows), 256, SPATIAL_QUERY_VISION_DIM],
        "query_state_shape": [int(expected_rows), SPATIAL_QUERY_VISION_DIM],
        "dtype": "float16",
        "image_size": 896,
        "feature_batch_size": 8,
    }
    mismatches = {
        key: (metadata.get(key), expected_value)
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"spatial-query feature metadata mismatch: {mismatches}")
    image_tokens = np.asarray(arrays.get("image_tokens"))
    query_states = np.asarray(arrays.get("query_states"))
    if image_tokens.dtype != np.float16 or query_states.dtype != np.float16:
        raise ValueError(
            "spatial-query feature metadata declares float16 arrays, but the archive dtype differs"
        )
    if list(image_tokens.shape) != expected["image_token_shape"]:
        raise ValueError(
            "spatial-query image-token array shape disagrees with metadata: "
            f"{image_tokens.shape}"
        )
    if list(query_states.shape) != expected["query_state_shape"]:
        raise ValueError(
            "spatial-query query-state array shape disagrees with metadata: "
            f"{query_states.shape}"
        )
    if "uid_sha256" in metadata:
        uids = detection_feature_uids(arrays, expected_rows)
        if uids is None:
            raise ValueError(
                "spatial-query feature metadata has uid_sha256 but archive has no UID vector"
            )
        actual_uid_sha256 = detection_uid_sha256(uids)
        if metadata["uid_sha256"] != actual_uid_sha256:
            raise ValueError(
                "spatial-query feature UID digest disagrees with metadata: "
                f"{actual_uid_sha256} != {metadata['uid_sha256']}"
            )
    return metadata


def boxes_to_normalized_targets(
    boxes: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> list[list[float]]:
    """Convert original-image ``xyxy`` boxes to spatial-query ``cxcywh`` targets."""

    if int(width) <= 0 or int(height) <= 0:
        raise ValueError(f"image dimensions must be positive, got {(width, height)}")
    normalized: list[list[float]] = []
    for box in boxes:
        if len(box) < 4:
            raise ValueError(f"detection box must contain four coordinates: {box!r}")
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        values = [
            (x1 + x2) / (2.0 * width),
            (y1 + y2) / (2.0 * height),
            max(x2 - x1, 1e-6) / width,
            max(y2 - y1, 1e-6) / height,
        ]
        normalized.append(values)
    return normalize_detection_target_rows([normalized])[0]


def load_detection_targets(
    path: str | Path,
    expected_rows: int | None = None,
    expected_uids: Sequence[Any] | None = None,
) -> list[list[list[float]]]:
    """Read and validate a JSON target file produced by the public tooling."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    target_uids: list[str] | None = None
    target_rows: Any = payload
    if isinstance(payload, Mapping):
        if "uids" in payload:
            target_uids = normalize_detection_uids(
                payload["uids"], expected_rows, field_name="target uids",
            )
        target_rows = payload.get("rows", payload.get("targets"))
    if expected_uids is not None and target_uids is not None:
        normalized_expected = normalize_detection_uids(
            expected_uids, expected_rows, field_name="feature uids",
        )
        if target_uids != normalized_expected:
            first_difference = next(
                (
                    index for index, (expected, actual)
                    in enumerate(zip(normalized_expected, target_uids))
                    if expected != actual
                ),
                0,
            )
            raise ValueError(
                "Detection target/feature UID order mismatch at index "
                f"{first_difference}: {normalized_expected[first_difference]!r} "
                f"!= {target_uids[first_difference]!r}"
            )
    return normalize_detection_target_rows(target_rows, expected_rows=expected_rows)


def infer_query_count(
    targets: Sequence[Sequence[Sequence[float]]],
    *,
    max_queries: int = SPATIAL_QUERY_MAX_QUERIES,
) -> int:
    """Choose K with the linear-interpolated p95 rule."""

    if not 1 <= int(max_queries):
        raise ValueError("max_queries must be positive")
    counts = np.asarray([len(row) for row in targets], dtype=np.float64)
    if counts.size == 0:
        return 1
    p95 = float(np.quantile(counts, 0.95, method="linear"))
    return max(1, min(int(math.ceil(p95)), int(max_queries)))


def matching(
    boxes: Any,
    presence: Any,
    targets: Sequence[Sequence[float]],
) -> tuple[list[tuple[int, int]], list[int]]:
    """Match predictions to targets with the spatial-query Hungarian cost.

    ``presence`` is accepted for API compatibility; the frozen assignment cost
    uses only box geometry.
    """

    del presence
    import torch
    from scipy.optimize import linear_sum_assignment

    if not targets:
        return [], list(range(int(boxes.shape[0])))
    match_boxes = boxes.float()
    target = torch.tensor(targets, dtype=torch.float32, device=boxes.device)
    pred_xyxy = cxcywh_to_xyxy(match_boxes).clamp(0, 1)
    target_xyxy = cxcywh_to_xyxy(target).clamp(0, 1)
    l1 = torch.cdist(match_boxes, target, p=1)
    giou = generalized_iou(pred_xyxy[:, None, :], target_xyxy[None, :, :])
    cost = (l1 + 2.0 * (1.0 - giou)).detach().float().cpu().numpy()
    pred_indices, target_indices = linear_sum_assignment(cost)
    matched = list(zip(pred_indices.tolist(), target_indices.tolist()))
    used = set(pred_indices.tolist())
    unmatched = [index for index in range(int(boxes.shape[0])) if index not in used]
    return matched, unmatched


def detection_loss(
    pred_boxes: Any,
    pred_presence: Any,
    targets: Sequence[Sequence[Sequence[float]]],
    *,
    box_loss_weight: float = 5.0,
    giou_loss_weight: float = 2.0,
    presence_loss_weight: float = 1.0,
) -> Any:
    """Compute the matched spatial-query loss and average over a batch."""

    import torch
    import torch.nn.functional as F

    if pred_boxes.ndim != 3 or pred_boxes.shape[-1] != 4:
        raise ValueError(f"pred_boxes must have shape [B,K,4], got {tuple(pred_boxes.shape)}")
    if pred_presence.shape != pred_boxes.shape[:2]:
        raise ValueError(
            f"pred_presence must have shape {tuple(pred_boxes.shape[:2])}, got {tuple(pred_presence.shape)}"
        )
    if len(targets) != pred_boxes.shape[0]:
        raise ValueError("target and prediction batch lengths differ")

    losses: list[Any] = []
    for index in range(pred_boxes.shape[0]):
        matched, _unmatched = matching(pred_boxes[index], pred_presence[index], targets[index])
        target_presence = torch.zeros_like(pred_presence[index])
        box_loss = pred_boxes[index].new_zeros(())
        giou_loss = pred_boxes[index].new_zeros(())
        if matched:
            target = torch.tensor(
                [targets[index][target_index] for _pred_index, target_index in matched],
                dtype=pred_boxes.dtype,
                device=pred_boxes.device,
            )
            predicted = torch.stack(
                [pred_boxes[index][pred_index] for pred_index, _target_index in matched]
            )
            target_presence[[pred_index for pred_index, _target_index in matched]] = 1.0
            box_loss = F.smooth_l1_loss(predicted, target, reduction="mean")
            giou_loss = (
                1.0 - generalized_iou(cxcywh_to_xyxy(predicted), cxcywh_to_xyxy(target))
            ).mean()
        presence_loss = F.binary_cross_entropy_with_logits(
            pred_presence[index], target_presence, reduction="mean"
        )
        losses.append(
            float(box_loss_weight) * box_loss
            + float(giou_loss_weight) * giou_loss
            + float(presence_loss_weight) * presence_loss
        )
    return torch.stack(losses).mean()


def _enable_determinism(seed: int) -> None:
    # Set this before the first CUDA/cuBLAS operation whenever possible.  It
    # complements torch's deterministic-algorithm switch on supported stacks.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(seed)
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _architecture_metadata() -> dict[str, Any]:
    return {
        "vision_dim": SPATIAL_QUERY_VISION_DIM,
        "hidden": SPATIAL_QUERY_HIDDEN_DIM,
        "heads": SPATIAL_QUERY_ATTENTION_HEADS,
        "dropout": SPATIAL_QUERY_DROPOUT,
        "decoder_layers": SPATIAL_QUERY_DECODER_LAYERS,
        "box_mlp": "256->256->4 sigmoid",
        "presence": "256->1",
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def train_detection_head(
    image_tokens: np.ndarray,
    query_states: np.ndarray,
    targets: Sequence[Sequence[Sequence[float]]],
    output_path: str | Path,
    *,
    epochs: int = DetectionConfig.epochs,
    batch_size: int = DetectionConfig.batch_size,
    learning_rate: float = DetectionConfig.learning_rate,
    weight_decay: float = DetectionConfig.weight_decay,
    warmup_ratio: float = DetectionConfig.warmup_ratio,
    max_grad_norm: float = DetectionConfig.max_grad_norm,
    seed: int = DetectionConfig.seed,
    device: str | None = None,
    max_queries: int = DetectionConfig.max_queries,
    presence_threshold: float = DetectionConfig.presence_threshold,
    box_loss_weight: float = DetectionConfig.box_loss_weight,
    matching_giou_weight: float = DetectionConfig.matching_giou_weight,
    presence_loss_weight: float = DetectionConfig.presence_loss_weight,
) -> dict[str, Any]:
    """Fit and persist the spatial query decoder from frozen feature arrays.

    The feature arrays must be produced by
    :func:`medical_parsing.tasks.detection.extract_detection_features`; the
    target JSON contains normalized ``(cx, cy, width, height)`` boxes.
    """

    import torch

    tokens = np.asarray(image_tokens)
    queries = np.asarray(query_states)
    if tokens.ndim != 3 or tokens.shape[1:] != (256, SPATIAL_QUERY_VISION_DIM):
        raise ValueError(f"image_tokens must have shape [N,256,2560], got {tokens.shape}")
    if queries.ndim != 2 or queries.shape[1:] != (SPATIAL_QUERY_VISION_DIM,):
        raise ValueError(f"query_states must have shape [N,2560], got {queries.shape}")
    if len(tokens) != len(queries):
        raise ValueError("image-token and query-state row counts differ")
    if len(tokens) == 0:
        raise ValueError("at least one detection training row is required")
    if not np.isfinite(tokens).all() or not np.isfinite(queries).all():
        raise ValueError("detection feature arrays contain non-finite values")
    target_rows = normalize_detection_target_rows(targets, expected_rows=len(tokens))
    if not int(epochs) > 0 or not int(batch_size) > 0:
        raise ValueError("epochs and batch_size must be positive")
    if not 0.0 <= float(warmup_ratio) <= 1.0:
        raise ValueError("warmup_ratio must be in [0,1]")
    if not 0.0 <= float(presence_threshold) <= 1.0:
        raise ValueError("presence_threshold must be in [0,1]")
    k = infer_query_count(target_rows, max_queries=max_queries)

    _enable_determinism(int(seed))
    device_name = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device_name.type == "cuda" else torch.float32
    model = FrozenSpatialQueryDecoder(k).to(device=device_name, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay),
    )
    steps_per_epoch = math.ceil(len(tokens) / int(batch_size))
    total_steps = int(epochs) * steps_per_epoch
    warmup_steps = max(1, math.ceil(float(warmup_ratio) * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    started = time.monotonic()
    loss_curve: list[dict[str, Any]] = []
    model.train()
    for epoch in range(int(epochs)):
        order = np.random.default_rng(int(seed) + epoch).permutation(len(tokens))
        epoch_values: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            indices = order[start:start + int(batch_size)]
            image_batch = torch.from_numpy(np.asarray(tokens[indices], dtype=np.float32)).to(
                device=device_name, dtype=dtype,
            )
            query_batch = torch.from_numpy(np.asarray(queries[indices], dtype=np.float32)).to(
                device=device_name, dtype=dtype,
            )
            target_batch = [target_rows[int(index)] for index in indices]
            pred_boxes, pred_presence = model(image_batch, query_batch)
            loss = detection_loss(
                pred_boxes,
                pred_presence,
                target_batch,
                box_loss_weight=box_loss_weight,
                giou_loss_weight=matching_giou_weight,
                presence_loss_weight=presence_loss_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite spatial-query loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()
            scheduler.step()
            epoch_values.append(float(loss.detach().float().cpu()))
            del image_batch, query_batch, pred_boxes, pred_presence, loss
        summary = {
            "epoch": epoch + 1,
            "loss": float(np.mean(epoch_values)),
            "lr": float(scheduler.get_last_lr()[0]),
            "elapsed_seconds": time.monotonic() - started,
        }
        loss_curve.append(summary)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "K": k,
        "architecture": _architecture_metadata(),
    }, destination)
    result = {
        "status": "PASS",
        "schema": "spatial_query_decoder_train_state_v1",
        "module": SPATIAL_QUERY_MODULE_NAME,
        "output": str(destination),
        "output_sha256": _sha256_file(destination),
        "source_rows": len(tokens),
        "K": k,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "AdamW",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "scheduler": "cosine",
        "warmup_ratio": float(warmup_ratio),
        "warmup_steps": warmup_steps,
        "presence_threshold": float(presence_threshold),
        "loss": "5*SmoothL1 + 2*(1-GIoU) + BCE presence",
        "matching": "Hungarian L1(cxcywh)+2*(1-GIoU)",
        "medgemma_gradients": 0,
        "loss_curve": loss_curve,
        "architecture": _architecture_metadata(),
        "elapsed_seconds": time.monotonic() - started,
    }
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


__all__ = [
    "boxes_to_normalized_targets", "cxcywh_to_xyxy", "detection_loss",
    "detection_feature_uids", "detection_uid_sha256", "generalized_iou", "infer_query_count",
    "load_detection_targets", "matching", "normalize_detection_target_rows",
    "normalize_detection_uids", "train_detection_head",
    "validate_detection_feature_metadata",
]
