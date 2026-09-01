"""Continuous regression branch with retrieval, correction, and quantiles."""

from __future__ import annotations

import gc
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from medical_parsing.config import AssetBundle, ModelConfig, RegressionConfig
from medical_parsing.models.backbone import (
    clear_model,
    extract_image_tokens,
    generated_text,
    load_adapter_bundle,
    load_raw_bundle,
)
from medical_parsing.models.regression_head import make_spatial_quantile_head
from medical_parsing.schema import image_sha, load_image, single_image_ref


def build_adaptive_image_views(image: Any) -> list[Any]:
    """Build aspect-ratio-adaptive views without changing the crop policy.

    Wide and tall images produce the original plus three crops.  Near-square
    images produce the original plus four quadrant crops.
    """

    width, height = image.size
    overlap = .15
    views = [image]
    if width / max(height, 1) >= 1.5:
        crop_width = round(width / (3 - 2 * overlap))
        step = (width - crop_width) / 2
        views.extend(image.crop((x, 0, x + crop_width, height)) for x in (0, round(step), width - crop_width))
    elif height / max(width, 1) >= 1.5:
        crop_height = round(height / (3 - 2 * overlap))
        step = (height - crop_height) / 2
        views.extend(image.crop((0, y, width, y + crop_height)) for y in (0, round(step), height - crop_height))
    else:
        crop_width = round(width / (2 - overlap))
        crop_height = round(height / (2 - overlap))
        views.extend(
            image.crop((x, y, x + crop_width, y + crop_height))
            for x in (0, width - crop_width)
            for y in (0, height - crop_height)
        )
    return views


def geometry_one(uri: str) -> np.ndarray:
    """Compute the fixed 960-dimensional intensity/edge geometry descriptor."""

    from scipy.ndimage import sobel

    image = load_image(uri).convert("L").resize((896, 896), resample=Image.Resampling.BILINEAR)
    intensity = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    gx = np.abs(sobel(intensity, axis=1, mode="reflect")).astype(np.float32)
    gy = np.abs(sobel(intensity, axis=0, mode="reflect")).astype(np.float32)
    parts: list[np.ndarray] = []
    for item in (intensity, gx, gy):
        pooled = item.reshape(16, 56, 16, 56).mean(axis=(1, 3), dtype=np.float64).reshape(-1)
        vertical = item.reshape(32, 28, 896).mean(axis=(1, 2), dtype=np.float64)
        horizontal = item.reshape(896, 32, 28).mean(axis=(0, 2), dtype=np.float64)
        parts.append(np.concatenate([pooled, vertical, horizontal]).astype(np.float32))
    result = np.concatenate(parts).astype(np.float32)
    if result.shape != (960,) or not np.isfinite(result).all():
        raise RuntimeError("invalid regression geometry descriptor")
    return result


def l2(value: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(denominator <= 0):
        raise RuntimeError("zero regression representation norm")
    return value / denominator


def weighted_median(values: np.ndarray, weights: np.ndarray, uids: list[str]) -> float:
    order = np.lexsort((np.asarray(uids).astype(str), np.asarray(values, dtype=np.float64)))
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    index = int(np.searchsorted(np.cumsum(sorted_weights), .5 * sorted_weights.sum(), side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def retrieve_reg(
    query_rep: np.ndarray,
    query_group: str,
    source_rep: np.ndarray,
    source_groups: np.ndarray,
    source_targets: np.ndarray,
    source_uids: list[str],
    neighbors: int = RegressionConfig.retrieval_neighbors,
) -> float:
    allowed = np.flatnonzero(source_groups.astype(str) != str(query_group))
    if len(allowed) < neighbors:
        raise RuntimeError(f"insufficient non-identical regression neighbors: {len(allowed)} < {neighbors}")
    distances = np.maximum(0.0, 1.0 - source_rep[allowed] @ query_rep)
    order = np.argsort(distances, kind="mergesort")[:neighbors]
    selected = allowed[order]
    distances = distances[order]
    return weighted_median(
        source_targets[selected], 1.0 / (distances + 1e-3),
        [source_uids[index] for index in selected],
    )


def fuse_regression_estimates(
    retrieval_blend: np.ndarray,
    residual_median: np.ndarray,
    upper_quantile_estimate: np.ndarray,
    regression_config: RegressionConfig | None = None,
) -> np.ndarray:
    """Apply the frozen residual-correction and upper-quantile fusion."""

    config = regression_config or RegressionConfig()
    corrected = np.clip(
        np.asarray(retrieval_blend) + config.residual_correction_weight * np.asarray(residual_median),
        0.0, 100.0,
    )
    return np.clip(
        config.base_fusion_weight * corrected + config.quantile_fusion_weight * np.asarray(upper_quantile_estimate),
        0.0, 100.0,
    )


def _extract_visual_features(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: str,
    feature_batch_size: int,
) -> np.ndarray:
    import torch

    flat_views: list[Any] = []
    view_spans: list[tuple[int, int]] = []
    for row in rows:
        views = build_adaptive_image_views(load_image(single_image_ref(row)))
        start = len(flat_views)
        flat_views.extend(views)
        view_spans.append((start, len(views)))
    core = getattr(model, "model", model)
    vision = getattr(core, "vision_tower", None)
    projector = getattr(core, "multi_modal_projector", None)
    if vision is None or projector is None:
        raise RuntimeError("raw base vision/projector path unavailable")
    view_parts: list[np.ndarray] = []
    for start in range(0, len(flat_views), feature_batch_size):
        images = flat_views[start:start + feature_batch_size]
        pixels = processor.image_processor(images=images, return_tensors="pt")["pixel_values"]
        if pixels.ndim == 5:
            pixels = pixels[:, 0]
        with torch.inference_mode():
            hidden = vision(pixel_values=pixels.to(device=device, dtype=torch.bfloat16)).last_hidden_state
            projected = projector(hidden)
            pooled = projected.float().mean(dim=1).cpu().numpy().astype(np.float16)
        if pooled.shape != (len(images), 2560):
            raise RuntimeError(f"regression view feature shape mismatch: {pooled.shape}")
        view_parts.append(pooled)
        del pixels, hidden, projected, pooled, images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    all_views = np.concatenate(view_parts, axis=0).astype(np.float16, copy=False)
    visual = np.asarray([
        np.concatenate([
            all_views[start],
            all_views[start + 1:start + count].mean(axis=0),
            all_views[start + 1:start + count].max(axis=0),
        ], axis=0)
        for start, count in view_spans
    ], dtype=np.float16)
    if visual.shape != (len(rows), 7680) or not np.isfinite(visual).all():
        raise RuntimeError(f"regression visual feature contract failed: {visual.shape}")
    return visual


def _first_number(text: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if match is None:
        raise RuntimeError(f"generated regression answer has no numeric value: {text!r}")
    return float(match.group(0))


def run_regression(
    rows: list[dict[str, Any]],
    base_path: Path,
    primary_adapter_path: Path,
    regression_adapter_path: Path,
    device: str,
    assets: AssetBundle,
    model_config: ModelConfig,
    audit: dict[str, Any],
    regression_config: RegressionConfig | None = None,
) -> dict[str, str]:
    if not rows:
        return {}
    import joblib
    import torch

    regression_config = regression_config or RegressionConfig()

    base_model, base_processor, base_audit = load_raw_bundle(base_path, device, model_config)
    visual = _extract_visual_features(
        base_model, base_processor, rows, device, model_config.feature_batch_size,
    )
    clear_model(base_model, base_processor)
    geometry = np.stack([geometry_one(single_image_ref(row)) for row in rows])
    reference = joblib.load(assets.path("regression_reference"))
    if reference.get("visual_f1") is None or reference["visual_f1"].shape[1:] != (7680,):
        raise RuntimeError("regression reference asset is not deployment-complete")
    source_visual = np.asarray(reference["visual_f1"], dtype=np.float32)
    source_geo = np.asarray(reference["geometry"], dtype=np.float32)
    source_targets = np.asarray(reference["targets"], dtype=np.float64)
    source_uids = [str(value) for value in reference["uids"]]
    source_groups = np.asarray(reference["groups"]).astype(str)
    visual_scaler = reference["visual_scaler"]
    visual_pca = reference["visual_pca"]
    geo_scaler = reference["geometry_scaler"]
    geo_pca = reference["geometry_pca"]
    source_v = l2(visual_pca.transform(visual_scaler.transform(source_visual)))
    source_g = l2(geo_pca.transform(geo_scaler.transform(source_geo)))
    source_rep = l2(np.concatenate([.5 * source_v, .5 * source_g], axis=1))
    visual_float32 = visual.astype(np.float32)
    query_v = l2(visual_pca.transform(visual_scaler.transform(visual_float32)))
    query_g = l2(geo_pca.transform(geo_scaler.transform(geometry)))
    query_rep = l2(np.concatenate([.5 * query_v, .5 * query_g], axis=1))

    visual_model = joblib.load(assets.path("regression_visual_model"))
    visual_prediction = np.asarray(
        visual_model["model"].predict(
            visual_model["pca"].transform(visual_model["scaler"].transform(visual_float32)),
        ),
        dtype=np.float64,
    )

    mid_model, mid_processor, mid_audit = load_adapter_bundle(
        base_path, regression_adapter_path, device, model_config,
    )
    generated_numeric_estimate = np.asarray([
        _first_number(generated_text(mid_model, mid_processor, row, device, model_config))
        for row in rows
    ], dtype=np.float64)
    clear_model(mid_model, mid_processor)
    base_fused_estimate = (
        regression_config.generated_fusion_weight * generated_numeric_estimate
        + regression_config.visual_fusion_weight * visual_prediction
    )

    retrieval_estimate = np.asarray([
        retrieve_reg(
            query_rep[index], image_sha(single_image_ref(row)), source_rep,
            source_groups, source_targets, source_uids,
            neighbors=regression_config.retrieval_neighbors,
        )
        for index, row in enumerate(rows)
    ], dtype=np.float64)
    retrieval_refined_estimate = (
        regression_config.base_fusion_weight * base_fused_estimate
        + regression_config.retrieval_fusion_weight * retrieval_estimate
    )
    residual_payload = np.load(assets.path("regression_residuals"), allow_pickle=False)
    residual_uids = np.asarray(residual_payload["uid"]).astype(str)
    residual = np.asarray(residual_payload["residual"], dtype=np.float64)
    if residual_uids.tolist() != source_uids or residual.shape != (len(source_uids),):
        raise RuntimeError("regression residual/reference UID mismatch")
    retrieved_residual_correction = np.asarray([
        retrieve_reg(
            query_rep[index], image_sha(single_image_ref(row)), source_rep,
            source_groups, residual, source_uids,
            neighbors=regression_config.retrieval_neighbors,
        )
        for index, row in enumerate(rows)
    ], dtype=np.float64)
    tokens_model, tokens_processor, primary_adapter_audit = load_adapter_bundle(
        base_path, primary_adapter_path, device, model_config,
    )
    tokens = extract_image_tokens(
        tokens_model, tokens_processor, rows, device, model_config,
        prompt="spatial image feature",
    )
    clear_model(tokens_model, tokens_processor)
    head_payload = torch.load(assets.path("regression_quantile_head"), map_location="cpu", weights_only=False)
    head = make_spatial_quantile_head().to(device)
    head.load_state_dict(head_payload["state_dict"], strict=True)
    head.eval()
    geometry64 = l2(head_payload["geometry_pca"].transform(
        head_payload["geometry_scaler"].transform(geometry),
    ))
    token_tensor = torch.from_numpy(tokens).to(device)
    geometry_tensor = torch.from_numpy(geometry64.astype(np.float32)).to(device)
    with torch.inference_mode():
        enabled = bool(token_tensor.is_cuda)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
            upper_quantile_estimate = head(token_tensor, geometry_tensor).float().cpu().numpy()[:, 2]
    predictions = fuse_regression_estimates(
        retrieval_refined_estimate, retrieved_residual_correction,
        upper_quantile_estimate, regression_config,
    )
    if not np.isfinite(predictions).all():
        raise RuntimeError("non-finite regression output")
    audit.update({
        "rows": len(rows),
        "visual_shape": list(visual.shape),
        "geometry_shape": list(geometry.shape),
        "visual_views": "aspect-ratio-adaptive: original plus three wide/tall crops or four square crops",
        "feature_batch_size": model_config.feature_batch_size,
        "base_model_load": base_audit,
        "regression_adapter_load": mid_audit,
        "primary_adapter_load": primary_adapter_audit,
        "retrieval_neighbors": regression_config.retrieval_neighbors,
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
    })
    del head, token_tensor, geometry_tensor, tokens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {row["uid"]: format(float(value), ".15g") for row, value in zip(rows, predictions)}


tile_views = build_adaptive_image_views
final_regression_blend = fuse_regression_estimates


__all__ = [
    "build_adaptive_image_views", "final_regression_blend", "fuse_regression_estimates",
    "geometry_one", "l2", "retrieve_reg", "run_regression", "tile_views", "weighted_median",
]
