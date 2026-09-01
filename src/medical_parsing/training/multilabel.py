"""Fit multi-label candidate models and the residual probability head."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from medical_parsing.models.multilabel_head import MultiLabelResidualProbabilityHead
from .common import ensure_2d, seed_everything


def fit_candidate_models(
    selector_features: np.ndarray,
    selector_targets: Sequence[float],
    ranker_features: np.ndarray,
    ranker_targets: Sequence[float],
    selector_output: str | Path,
    ranker_output: str | Path,
    *,
    ranker_groups: Sequence[int] | None = None,
    iterations: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit the candidate selector and listwise ranker from materialized rows."""

    from catboost import CatBoostRanker, CatBoostRegressor

    selector_features = ensure_2d(selector_features, "selector_features").astype(np.float32)
    ranker_features = ensure_2d(ranker_features, "ranker_features").astype(np.float32)
    selector_targets = np.asarray(selector_targets, dtype=np.float32)
    ranker_targets = np.asarray(ranker_targets, dtype=np.float32)
    if len(selector_features) != len(selector_targets) or len(ranker_features) != len(ranker_targets):
        raise ValueError("candidate feature and target lengths do not match")
    seed_everything(seed)
    selector = CatBoostRegressor(
        loss_function="RMSE", iterations=iterations, depth=7, learning_rate=.05,
        random_seed=seed, verbose=False,
    )
    selector.fit(selector_features, selector_targets)
    selector_path = Path(selector_output)
    selector_path.parent.mkdir(parents=True, exist_ok=True)
    selector.save_model(str(selector_path))

    if ranker_groups is None:
        # The caller should normally provide one group per source row.  A
        # single group remains a valid deterministic fallback for a prepared
        # flat table.
        ranker_groups = np.zeros(len(ranker_features), dtype=np.int64)
    ranker_groups = np.asarray(ranker_groups, dtype=np.int64)
    if len(ranker_groups) != len(ranker_features):
        raise ValueError("ranker_groups length does not match ranker features")
    ranker = CatBoostRanker(
        loss_function="YetiRank", iterations=iterations, depth=7, learning_rate=.05,
        random_seed=seed, verbose=False,
    )
    ranker.fit(ranker_features, ranker_targets, group_id=ranker_groups)
    ranker_path = Path(ranker_output)
    ranker_path.parent.mkdir(parents=True, exist_ok=True)
    ranker.save_model(str(ranker_path))
    return {
        "status": "PASS", "selector_output": str(selector_path),
        "ranker_output": str(ranker_path), "selector_rows": len(selector_features),
        "ranker_rows": len(ranker_features),
    }


def fit_probability_models(
    features: np.ndarray,
    targets: np.ndarray,
    output_path: str | Path,
    *,
    iterations: int = 300,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit the forty atom/cardinality probability models.

    Constant columns are stored as scalar probabilities; non-constant columns
    use CatBoost classifiers and expose the same ``predict_proba`` interface
    consumed by inference.
    """

    import joblib
    from catboost import CatBoostClassifier

    features = ensure_2d(features, "features").astype(np.float32)
    targets = np.asarray(targets)
    if targets.shape != (len(features), 10, 4):
        raise ValueError(f"targets must have shape [N,10,4], got {targets.shape}")
    seed_everything(seed)
    models: dict[tuple[int, int], Any] = {}
    for atom_index in range(10):
        for card_index in range(4):
            values = np.asarray(targets[:, atom_index, card_index], dtype=np.float64)
            unique = np.unique(values)
            if len(unique) < 2:
                models[(atom_index, card_index)] = float(np.clip(values.mean(), 1e-4, 1.0 - 1e-4))
                continue
            model = CatBoostClassifier(
                loss_function="Logloss", iterations=iterations, depth=6,
                learning_rate=.05, random_seed=seed, verbose=False,
            )
            model.fit(features, values.astype(np.int64))
            models[(atom_index, card_index)] = model
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, destination)
    return {"status": "PASS", "output": str(destination), "rows": len(features), "models": len(models)}


def train_multilabel_residual_head(
    tokens: np.ndarray,
    row_features: np.ndarray,
    base_probability: np.ndarray,
    targets: np.ndarray,
    output_path: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    seed: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    """Train the residual head over frozen token features and fitted priors."""

    import torch

    seed_everything(seed)
    tokens = np.asarray(tokens)
    row_features = ensure_2d(row_features, "row_features").astype(np.float32)
    base_probability = np.asarray(base_probability, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float32)
    if tokens.ndim != 3 or tokens.shape[1:] != (256, 2560):
        raise ValueError(f"tokens must have shape [N,256,2560], got {tokens.shape}")
    if base_probability.shape != (len(tokens), 10, 4) or targets.shape != (len(tokens), 10, 4):
        raise ValueError("base_probability and targets must have shape [N,10,4]")
    if len(row_features) != len(tokens):
        raise ValueError("row_features length does not match tokens")
    device_name = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiLabelResidualProbabilityHead().to(device_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    mean = row_features.mean(axis=0)
    scale = row_features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized_features = ((row_features - mean) / scale).astype(np.float32)
    base = np.clip(base_probability.reshape(len(tokens), 40), 1e-4, 1.0 - 1e-4)
    base_logits = np.log(base / (1.0 - base)).astype(np.float32)
    target_flat = targets.reshape(len(tokens), 40)
    for _epoch in range(epochs):
        order = np.arange(len(tokens))
        np.random.shuffle(order)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            token_tensor = torch.from_numpy(tokens[index].astype(np.float32)).to(device_name)
            feature_tensor = torch.from_numpy(normalized_features[index]).to(device_name)
            base_tensor = torch.from_numpy(base_logits[index]).to(device_name)
            target_tensor = torch.from_numpy(target_flat[index]).to(device_name)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_tensor, feature_tensor, base_tensor)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_tensor)
            loss.backward()
            optimizer.step()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "multilabel-residual-head-v1",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "row_scaler_mean": mean,
        "row_scaler_scale": scale,
    }, destination)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "PASS", "output": str(destination), "rows": len(tokens)}


__all__ = ["fit_candidate_models", "fit_probability_models", "train_multilabel_residual_head"]
