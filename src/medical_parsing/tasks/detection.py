"""Frozen Spatial Query Decoding implemented with a spatial query decoder.

Detection is intentionally feature-based: the primary LoRA adapter supplies
the image-token memory and the final non-padding decoder state, while the
spatial query decoder predicts normalized center-size boxes and presence logits.
No coordinate text is generated in this path.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np

from medical_parsing.config import AssetBundle, DetectionConfig, ModelConfig
from medical_parsing.module_names import PAPER_MODULES
from medical_parsing.models.backbone import (
    SYSTEM_PROMPT,
    TASK_INSTRUCTIONS,
    apply_chat_template,
    clear_model,
    load_adapter_bundle,
)
from medical_parsing.models.detection_head import (
    FROZEN_SPATIAL_QUERY_ASSET_SHA256,
    FROZEN_SPATIAL_QUERY_ASSET_SIZE_BYTES,
    SPATIAL_QUERY_ATTENTION_HEADS,
    SPATIAL_QUERY_DECODER_FFN_DIM,
    SPATIAL_QUERY_DECODER_LAYERS,
    SPATIAL_QUERY_DROPOUT,
    SPATIAL_QUERY_HIDDEN_DIM,
    SPATIAL_QUERY_PRESENCE_THRESHOLD,
    SPATIAL_QUERY_VISION_DIM,
    load_frozen_spatial_query_decoder,
)
from medical_parsing.schema import (
    TASK_DETECTION,
    load_image,
    parse_detection_boxes,
    prepared_image,
    single_image_ref,
    validate_detection_boxes,
)

SPATIAL_QUERY_IMAGE_SIZE = 896
# The reference runner uses this fixed batch size. MedGemma BF16 kernels can
# produce small batch-dependent changes, so it is part of the spatial-query
# inference contract rather than an ordinary memory tuning knob.
SPATIAL_QUERY_FEATURE_BATCH_SIZE = 8
DETECTION_MODULE_NAME = PAPER_MODULES["frozen_spatial_query_decoding"]


def detection_prompt(row: dict[str, Any]) -> str:
    """Build the frozen spatial-query instruction plus the row question."""

    return TASK_INSTRUCTIONS[TASK_DETECTION] + "\n\n" + str(row["question"])


def detection_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": detection_prompt(row)},
            ],
        },
    ]


def _move_batch(batch: Any, device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(batch).items()
    }


def _feature_batch(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    texts: list[str] = []
    images = []
    for row in rows:
        image = prepared_image(single_image_ref(row), image_size=image_size)
        texts.append(apply_chat_template(processor, detection_messages(row), generation=True))
        images.append(image)
    batch = processor(
        text=texts,
        images=[[image] for image in images],
        return_tensors="pt",
        padding=True,
    )
    model_batch = _move_batch(batch, device)
    pixels = model_batch.get("pixel_values")
    if pixels is None:
        raise RuntimeError("spatial-query feature extraction received no pixel_values")
    feature_pixels = pixels[:, 0] if pixels.ndim == 5 else pixels
    language_output = None
    hidden = None
    with torch.inference_mode():
        image_output = model.get_image_features(
            pixel_values=feature_pixels,
            return_dict=True,
        )
        image_tokens = image_output.pooler_output
        try:
            language_output = model(
                **model_batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        except (TypeError, RuntimeError):
            # A few Transformers/MedGemma combinations accept the squeezed
            # image tensor in the language forward only.  The retry is the
            # same shape-equivalent compatibility path used by the reference
            # spatial-query runner.
            if pixels.ndim != 5:
                raise
            retry_batch = dict(model_batch)
            retry_batch["pixel_values"] = feature_pixels
            language_output = model(
                **retry_batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = getattr(language_output, "hidden_states", None)
        hidden = hidden_states[-1] if hidden_states is not None else getattr(language_output, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError(
                "spatial-query feature extraction returned no final decoder hidden state"
            )
        attention_mask = model_batch.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("spatial-query feature extraction requires attention_mask")
        lengths = attention_mask.sum(dim=1).to(torch.long) - 1
        if int(lengths.min()) < 0 or hidden.shape[1] <= int(lengths.max()):
            raise RuntimeError(
                "spatial-query index exceeds hidden sequence: "
                f"{tuple(hidden.shape)} / {lengths.tolist()}"
            )
        query = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            lengths,
        ]

    image_array = image_tokens.float().cpu().numpy().astype(np.float16)
    query_array = query.float().cpu().numpy().astype(np.float16)
    if image_array.shape != (len(rows), 256, SPATIAL_QUERY_VISION_DIM):
        raise RuntimeError(
            f"unexpected spatial-query image-token shape: {image_array.shape}"
        )
    if query_array.shape != (len(rows), SPATIAL_QUERY_VISION_DIM):
        raise RuntimeError(
            f"unexpected spatial-query query-state shape: {query_array.shape}"
        )
    if not np.isfinite(image_array).all() or not np.isfinite(query_array).all():
        raise RuntimeError("non-finite spatial-query feature value")
    return image_array, query_array


def _iter_detection_features(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: str,
    model_config: ModelConfig,
    *,
    feature_batch_size: int | None = None,
) -> Iterator[tuple[list[dict[str, Any]], np.ndarray, np.ndarray]]:
    batch_size = model_config.feature_batch_size if feature_batch_size is None else int(feature_batch_size)
    if batch_size <= 0:
        raise ValueError(
            f"spatial-query feature batch size must be positive, got {batch_size}"
        )
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start + batch_size]
        image_tokens, query_states = _feature_batch(
            model, processor, subset, device, model_config.image_size,
        )
        yield subset, image_tokens, query_states


def extract_detection_features(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: str,
    model_config: ModelConfig,
    *,
    feature_batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the exact spatial-query feature pair for a list of rows.

    Returns image tokens with shape ``[N, 256, 2560]`` and query states with
    shape ``[N, 2560]``.  Both arrays are finite ``float16`` CPU arrays so
    they can be cached and consumed by the public head trainer.
    """

    if not rows:
        return (
            np.empty((0, 256, SPATIAL_QUERY_VISION_DIM), dtype=np.float16),
            np.empty((0, SPATIAL_QUERY_VISION_DIM), dtype=np.float16),
        )
    token_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    import torch

    for _subset, image_tokens, query_states in _iter_detection_features(
        model, processor, rows, device, model_config,
        feature_batch_size=feature_batch_size,
    ):
        token_parts.append(image_tokens)
        query_parts.append(query_states)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(token_parts, axis=0), np.concatenate(query_parts, axis=0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _presence_probability(logit: Any) -> float:
    import torch

    if isinstance(logit, torch.Tensor):
        return float(torch.sigmoid(logit).float().cpu())
    value = float(logit)
    return float(1.0 / (1.0 + np.exp(-value)))


def decode_detection_outputs(
    normalized_boxes: Any,
    presence_logits: Any,
    image_sizes: list[tuple[int, int]],
    *,
    presence_threshold: float = SPATIAL_QUERY_PRESENCE_THRESHOLD,
) -> list[str]:
    """Convert spatial-query outputs to compact original-image JSON box lists."""

    import torch

    if isinstance(normalized_boxes, torch.Tensor):
        boxes_array = normalized_boxes.float().detach().cpu().numpy()
    else:
        boxes_array = np.asarray(normalized_boxes, dtype=np.float32)
    if isinstance(presence_logits, torch.Tensor):
        logits_array = presence_logits.detach().cpu()
        probabilities = torch.sigmoid(logits_array).float().numpy()
    else:
        logits_array = np.asarray(presence_logits, dtype=np.float32)
        probabilities = 1.0 / (1.0 + np.exp(-logits_array))
    if boxes_array.ndim != 3 or boxes_array.shape[-1] != 4:
        raise ValueError(f"normalized_boxes must have shape [N,K,4], got {boxes_array.shape}")
    if probabilities.shape != boxes_array.shape[:2]:
        raise ValueError(
            f"presence logits must have shape {boxes_array.shape[:2]}, got {probabilities.shape}"
        )
    if len(image_sizes) != boxes_array.shape[0]:
        raise ValueError("image_sizes and prediction batch have different lengths")

    serialized: list[str] = []
    for boxes, scores, (width, height) in zip(boxes_array, probabilities, image_sizes):
        decoded: list[list[float]] = []
        for box, score in zip(boxes, scores):
            if float(score) < float(presence_threshold):
                continue
            cx, cy, box_width, box_height = np.clip(box.astype(np.float64), 0.0, 1.0)
            x1 = float(np.clip(cx - box_width / 2.0, 0.0, 1.0) * width)
            y1 = float(np.clip(cy - box_height / 2.0, 0.0, 1.0) * height)
            x2 = float(np.clip(cx + box_width / 2.0, 0.0, 1.0) * width)
            y2 = float(np.clip(cy + box_height / 2.0, 0.0, 1.0) * height)
            # Keep this epsilon and ordering rule identical to the frozen
            # spatial-query serializer.
            if x2 <= x1:
                x2 = min(float(width), x1 + 1e-6)
            if y2 <= y1:
                y2 = min(float(height), y1 + 1e-6)
            decoded.append([x1, y1, x2, y2])
        serialized.append(json.dumps(decoded, separators=(",", ":")))
    return serialized


def _validate_detection_config(config: DetectionConfig) -> None:
    expected = {
        "vision_dim": (config.vision_dim, SPATIAL_QUERY_VISION_DIM),
        "head_dim": (config.head_dim, SPATIAL_QUERY_HIDDEN_DIM),
        "attention_heads": (config.attention_heads, SPATIAL_QUERY_ATTENTION_HEADS),
        "decoder_layers": (config.decoder_layers, SPATIAL_QUERY_DECODER_LAYERS),
        "decoder_ffn_dim": (config.decoder_ffn_dim, SPATIAL_QUERY_DECODER_FFN_DIM),
        "dropout": (config.dropout, SPATIAL_QUERY_DROPOUT),
    }
    mismatches = {
        key: values for key, values in expected.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"spatial query decoder architecture settings are frozen: {mismatches}")
    if not 0.0 <= config.presence_threshold <= 1.0:
        raise ValueError("detection.presence_threshold must be in [0, 1]")
    if int(config.max_queries) < 1:
        raise ValueError("detection.max_queries must be positive")


def _validate_spatial_query_runtime(model_config: ModelConfig) -> None:
    if int(model_config.image_size) != SPATIAL_QUERY_IMAGE_SIZE:
        raise ValueError(
            "spatial query image preparation is frozen at "
            f"{SPATIAL_QUERY_IMAGE_SIZE}, got {model_config.image_size}"
        )


def run_detection(
    rows: list[dict[str, Any]],
    base_path: Path,
    adapter_path: Path,
    device: str,
    assets: AssetBundle,
    model_config: ModelConfig,
    audit: dict[str, Any],
    detection_config: DetectionConfig | None = None,
) -> dict[str, str]:
    """Run the frozen spatial query decoder over validated Detection rows."""

    if not rows:
        return {}
    import torch

    detection_config = detection_config or DetectionConfig()
    _validate_detection_config(detection_config)
    _validate_spatial_query_runtime(model_config)
    checkpoint_path = assets.path("detection_head")
    started = time.monotonic()
    model = processor = head = None
    invalid: list[dict[str, str]] = []
    output: dict[str, str] = {}
    try:
        model, processor, model_audit = load_adapter_bundle(
            base_path, adapter_path, device, model_config,
        )
        device_type = torch.device(device).type
        head_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
        head, head_audit = load_frozen_spatial_query_decoder(
            checkpoint_path, device=device, dtype=head_dtype,
        )
        if head_audit["K"] > detection_config.max_queries:
            raise RuntimeError(
                f"checkpoint K={head_audit['K']} exceeds detection.max_queries={detection_config.max_queries}"
            )
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(device))

        for subset, image_array, query_array in _iter_detection_features(
            model, processor, rows, device, model_config,
            feature_batch_size=SPATIAL_QUERY_FEATURE_BATCH_SIZE,
        ):
            image_batch = torch.from_numpy(image_array.astype(np.float32)).to(
                device=device, dtype=head_dtype,
            )
            query_batch = torch.from_numpy(query_array.astype(np.float32)).to(
                device=device, dtype=head_dtype,
            )
            with torch.inference_mode():
                pred_boxes, pred_presence = head(image_batch, query_batch)
            image_sizes = [load_image(single_image_ref(row)).size for row in subset]
            serialized = decode_detection_outputs(
                pred_boxes, pred_presence, image_sizes,
                presence_threshold=detection_config.presence_threshold,
            )
            for row, raw in zip(subset, serialized):
                try:
                    validate_detection_boxes(raw)
                except ValueError as exc:
                    invalid.append({
                        "uid": row["uid"],
                        "error": str(exc),
                        "prediction_preview": raw[:240],
                    })
                # Parsing here also catches serializer changes before the
                # pipeline's final output contract is reached.
                parse_detection_boxes(raw)
                output[row["uid"]] = raw
            del image_array, query_array, image_batch, query_batch, pred_boxes, pred_presence
            if device_type == "cuda":
                torch.cuda.empty_cache()

        peak_bytes = int(torch.cuda.max_memory_allocated(torch.device(device))) if device_type == "cuda" else 0
        asset_size = checkpoint_path.stat().st_size
        asset_sha256 = _sha256_file(checkpoint_path)
        audit.update({
            "rows": len(rows),
            "paper_module": DETECTION_MODULE_NAME,
            "route": "frozen spatial query decoder",
            "generation": "none; frozen primary-adapter features plus spatial query decoder",
            "feature_contract": "primary-adapter image pooler_output plus final non-padding decoder input state",
            "serializer": "compact JSON box list compatible with official parse_boxes",
            "valid_outputs": len(rows) - len(invalid),
            "invalid_outputs": invalid,
            "image_size": SPATIAL_QUERY_IMAGE_SIZE,
            "feature_batch_size": SPATIAL_QUERY_FEATURE_BATCH_SIZE,
            "model_load": model_audit,
            "head": {
                **head_audit,
                "asset_size_bytes": asset_size,
                "asset_sha256": asset_sha256,
                "release_asset_match": bool(
                    asset_size == FROZEN_SPATIAL_QUERY_ASSET_SIZE_BYTES
                    and asset_sha256 == FROZEN_SPATIAL_QUERY_ASSET_SHA256
                ),
                "presence_threshold": detection_config.presence_threshold,
            },
            "inference_vram": {
                "peak_gpu_allocated_bytes": peak_bytes,
                "expected_inference_vram_gib": peak_bytes / 1024 ** 3,
                "deployment_target_gib": 24.0,
            },
            "elapsed_seconds": time.monotonic() - started,
        })
    finally:
        if head is not None:
            del head
        if model is not None and processor is not None:
            clear_model(model, processor)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if invalid:
        raise RuntimeError(f"Detection serializer validation failed for {len(invalid)} row(s): {invalid[0]}")
    if set(output) != {row["uid"] for row in rows}:
        raise RuntimeError("Detection output coverage is incomplete")
    return output


make_detection_prompt = detection_prompt
spatial_query_feature_batch = extract_detection_features


__all__ = [
    "SPATIAL_QUERY_FEATURE_BATCH_SIZE", "SPATIAL_QUERY_IMAGE_SIZE",
    "decode_detection_outputs", "detection_messages",
    "detection_prompt", "extract_detection_features", "make_detection_prompt",
    "run_detection", "spatial_query_feature_batch",
]
