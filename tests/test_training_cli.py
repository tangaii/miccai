import numpy as np

from train import COMPONENTS, _infer_component


def test_explicit_training_components_cover_public_producers():
    expected = {
        "classification-head", "multilabel-selector-ranker",
        "multilabel-probability-models", "multilabel-residual-head",
        "regression-visual-estimator", "regression-reference",
        "regression-residuals", "regression-quantile-head",
    }
    assert set(COMPONENTS) == expected


def test_legacy_task_dispatch_is_still_deterministic():
    assert _infer_component("classification", {"tokens": np.zeros((1, 1))}) == "classification-head"
    assert _infer_component("multilabel", {"features": np.zeros((1, 1))}) == "multilabel-probability-models"
    assert _infer_component("regression", {"visual_features": np.zeros((1, 1))}) == "regression-visual-estimator"
