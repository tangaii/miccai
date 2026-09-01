"""Fit regression-side estimators and the spatial quantile head."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from medical_parsing.models.regression_head import SpatialQuantileRefinementHead
from .common import ensure_2d, seed_everything


def _pca64(features: np.ndarray):
    from sklearn.decomposition import PCA

    if len(features) < 64 or features.shape[1] < 64:
        raise ValueError("at least 64 rows and 64 input features are required for the fixed 64-D PCA contract")
    return PCA(n_components=64, svd_solver="randomized", random_state=0)


def fit_visual_regressor(
    visual_features: np.ndarray,
    targets: Sequence[float],
    output_path: str | Path,
    *,
    iterations: int = 700,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit the visual CatBoost model with its persisted scaler and PCA."""

    import joblib
    from catboost import CatBoostRegressor
    from sklearn.preprocessing import StandardScaler

    features = ensure_2d(visual_features, "visual_features").astype(np.float32)
    targets = np.asarray(targets, dtype=np.float64)
    if len(features) != len(targets):
        raise ValueError("visual feature and target lengths do not match")
    seed_everything(seed)
    scaler = StandardScaler().fit(features)
    pca = _pca64(features).fit(scaler.transform(features))
    transformed = pca.transform(scaler.transform(features))
    model = CatBoostRegressor(
        loss_function="RMSE", iterations=iterations, depth=8, learning_rate=.04,
        random_seed=seed, verbose=False,
    )
    model.fit(transformed, targets)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema": "regression-visual-model-v1", "scaler": scaler, "pca": pca, "model": model}, destination)
    return {"status": "PASS", "output": str(destination), "rows": len(features), "components": 64}


def fit_reference(
    visual_features: np.ndarray,
    geometry: np.ndarray,
    targets: Sequence[float],
    uids: Sequence[str],
    groups: Sequence[str],
    output_path: str | Path,
) -> dict[str, Any]:
    """Persist the cross-group retrieval reference and both transforms."""

    import joblib
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    visual = ensure_2d(visual_features, "visual_features").astype(np.float32)
    geometry = ensure_2d(geometry, "geometry").astype(np.float32)
    targets = np.asarray(targets, dtype=np.float64)
    uids = np.asarray(uids).astype(str)
    groups = np.asarray(groups).astype(str)
    if not (len(visual) == len(geometry) == len(targets) == len(uids) == len(groups)):
        raise ValueError("reference arrays must have equal lengths")
    if len(visual) < 64 or len(geometry) < 64:
        raise ValueError("at least 64 reference rows are required for the fixed PCA contract")
    visual_scaler = StandardScaler().fit(visual)
    visual_pca = PCA(n_components=64, svd_solver="randomized", random_state=0).fit(visual_scaler.transform(visual))
    geometry_scaler = StandardScaler().fit(geometry)
    geometry_pca = PCA(n_components=64, svd_solver="randomized", random_state=0).fit(geometry_scaler.transform(geometry))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "schema": "regression-source-reference-v2",
        "visual_f1": visual,
        "geometry": geometry,
        "targets": targets,
        "uids": uids,
        "groups": groups,
        "visual_scaler": visual_scaler,
        "visual_pca": visual_pca,
        "geometry_scaler": geometry_scaler,
        "geometry_pca": geometry_pca,
    }, destination)
    return {"status": "PASS", "output": str(destination), "rows": len(visual), "components": 64}


def write_crossfitted_residuals(
    uids: Sequence[str],
    residuals: Sequence[float],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write the UID-aligned residual table consumed by retrieval correction."""

    uids = np.asarray(uids).astype(str)
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.shape != (len(uids),) or not np.isfinite(residuals).all():
        raise ValueError("residuals must be finite and UID aligned")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, uid=uids, residual=residuals)
    return {"status": "PASS", "output": str(destination), "rows": len(uids)}


def train_quantile_head(
    tokens: np.ndarray,
    geometry: np.ndarray,
    targets: Sequence[float],
    output_path: str | Path,
    *,
    epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    seed: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    """Train the spatial 0.25/0.50/0.75 head and persist geometry transforms."""

    import torch
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    seed_everything(seed)
    tokens = np.asarray(tokens)
    geometry = ensure_2d(geometry, "geometry").astype(np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if tokens.ndim != 3 or tokens.shape[1:] != (256, 2560):
        raise ValueError(f"tokens must have shape [N,256,2560], got {tokens.shape}")
    if len(tokens) != len(geometry) or len(tokens) != len(targets):
        raise ValueError("quantile training arrays must have equal lengths")
    if len(tokens) < 64 or geometry.shape[1] < 64:
        raise ValueError("at least 64 rows and 64 geometry features are required")
    geometry_scaler = StandardScaler().fit(geometry)
    geometry_pca = PCA(n_components=64, svd_solver="randomized", random_state=0).fit(geometry_scaler.transform(geometry))
    geometry64 = geometry_pca.transform(geometry_scaler.transform(geometry)).astype(np.float32)
    device_name = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SpatialQuantileRefinementHead().to(device_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    quantiles = (0.25, 0.50, 0.75)
    for _epoch in range(epochs):
        order = np.arange(len(tokens))
        np.random.shuffle(order)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            token_tensor = torch.from_numpy(tokens[index].astype(np.float32)).to(device_name)
            geometry_tensor = torch.from_numpy(geometry64[index]).to(device_name)
            target_tensor = torch.from_numpy(targets[index]).to(device_name)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(token_tensor, geometry_tensor)
            losses = []
            for q_index, quantile in enumerate(quantiles):
                error = target_tensor - prediction[:, q_index]
                losses.append(torch.maximum((quantile - 1) * error, quantile * error).mean())
            torch.stack(losses).mean().backward()
            optimizer.step()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "regression-quantile-head-v1",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "geometry_scaler": geometry_scaler,
        "geometry_pca": geometry_pca,
    }, destination)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "PASS", "output": str(destination), "rows": len(tokens), "components": 64}


__all__ = ["fit_reference", "fit_visual_regressor", "train_quantile_head", "write_crossfitted_residuals"]
