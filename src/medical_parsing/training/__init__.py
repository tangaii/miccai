"""Training and fitting utilities for regenerating external assets."""

from .classification import train_semantic_heads
from .adapters import build_lora_model, train_lora_adapter
from .detection import (
    boxes_to_normalized_targets,
    detection_feature_uids,
    detection_uid_sha256,
    detection_loss,
    infer_query_count,
    load_detection_targets,
    matching,
    train_detection_head,
    validate_detection_feature_metadata,
)
from .multilabel import fit_candidate_models, fit_probability_models, train_multilabel_residual_head
from .regression import fit_reference, fit_visual_regressor, train_quantile_head, write_crossfitted_residuals

__all__ = [
    "boxes_to_normalized_targets", "detection_feature_uids", "detection_loss", "detection_uid_sha256", "fit_candidate_models", "fit_probability_models", "fit_reference",
    "fit_visual_regressor", "train_multilabel_residual_head", "train_quantile_head",
    "train_semantic_heads", "train_detection_head", "write_crossfitted_residuals",
    "build_lora_model", "train_lora_adapter", "infer_query_count", "load_detection_targets",
    "matching", "validate_detection_feature_metadata",
]
