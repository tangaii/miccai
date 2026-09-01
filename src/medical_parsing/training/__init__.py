"""Training and fitting utilities for regenerating external assets."""

from .classification import train_semantic_heads
from .adapters import build_lora_model, train_lora_adapter
from .multilabel import fit_candidate_models, fit_probability_models, train_multilabel_residual_head
from .regression import fit_reference, fit_visual_regressor, train_quantile_head, write_crossfitted_residuals

__all__ = [
    "fit_candidate_models", "fit_probability_models", "fit_reference",
    "fit_visual_regressor", "train_multilabel_residual_head", "train_quantile_head",
    "train_semantic_heads", "write_crossfitted_residuals", "build_lora_model", "train_lora_adapter",
]
