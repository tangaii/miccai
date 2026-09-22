"""Small, explicit runtime configuration and external-asset resolution.

The values in the frozen configuration dataclasses are the paper/default
contract.  YAML may override operational values for a controlled run, but
the public defaults remain explicit here rather than being duplicated across
task implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str = "google/medgemma-1.5-4b-it"
    image_size: int = 896
    feature_batch_size: int = 8
    vocab_chunk_size: int = 16
    max_new_tokens: int = 256
    seed: int = 0

    @property
    def token_chunk(self) -> int:
        """Backward-compatible alias for the vocabulary-row chunk size."""

        return self.vocab_chunk_size

    @property
    def scoring_row_batch_size(self) -> int:
        """Backward-compatible alias for the MLC row batch default."""

        return MultilabelConfig.scoring_row_batch_size


@dataclass(frozen=True)
class MultilabelConfig:
    candidate_change_threshold: float = 0.05
    max_replacement_candidates: int = 32
    scoring_row_batch_size: int = 4


@dataclass(frozen=True)
class RegressionConfig:
    """Frozen/default constants for Retrieval-Refined Quantile Regression."""

    retrieval_neighbors: int = 15
    generated_fusion_weight: float = 0.5
    visual_fusion_weight: float = 0.5
    base_fusion_weight: float = 0.75
    retrieval_fusion_weight: float = 0.25
    residual_correction_weight: float = 0.5
    quantile_fusion_weight: float = 0.25
    quantiles: tuple[float, float, float] = (0.25, 0.50, 0.75)
    geometry_pca_components: int = 64


@dataclass(frozen=True)
class DetectionConfig:
    """Frozen spatial-query decoder contract."""

    presence_threshold: float = 0.5
    vision_dim: int = 2560
    head_dim: int = 256
    attention_heads: int = 8
    decoder_layers: int = 2
    decoder_ffn_dim: int = 1024
    dropout: float = 0.1
    max_queries: int = 4
    seed: int = 20260908
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    matching_giou_weight: float = 2.0
    box_loss_weight: float = 5.0
    presence_loss_weight: float = 1.0

    # Short aliases keep the configuration compatible with earlier descriptive
    # component contracts without duplicating configuration keys.
    @property
    def hidden(self) -> int:
        return self.head_dim

    @property
    def heads(self) -> int:
        return self.attention_heads

    @property
    def decoder_feedforward_dim(self) -> int:
        return self.decoder_ffn_dim


@dataclass(frozen=True)
class RuntimeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    multilabel: MultilabelConfig = field(default_factory=MultilabelConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    checkpoint_dir: Path = Path("checkpoints")
    detection: DetectionConfig = field(default_factory=DetectionConfig)


ASSET_FILENAMES: dict[str, str] = {
    "classification_routes": "classification_route_manifest.json",
    "classification_heads": "classification_heads.pt",
    "multilabel_templates": "multilabel_template_map.json",
    "multilabel_library": "multilabel_candidate_library.json",
    "multilabel_selector": "multilabel_candidate_selector.cbm",
    "multilabel_ranker": "multilabel_candidate_ranker.cbm",
    "multilabel_probability_models": "multilabel_probability_models.joblib",
    "multilabel_residual_head": "multilabel_residual_head.pt",
    "regression_visual_model": "regression_visual_model.joblib",
    "regression_reference": "regression_reference.joblib",
    "regression_residuals": "regression_residuals.npz",
    "regression_quantile_head": "regression_quantile_head.pt",
    "detection_head": "spatial_query_decoder.pt",
}


@dataclass(frozen=True)
class AssetBundle:
    root: Path

    def path(self, name: str) -> Path:
        try:
            filename = ASSET_FILENAMES[name]
        except KeyError as exc:
            raise KeyError(f"unknown external asset: {name}") from exc
        return self.root / filename

    def missing(self) -> list[Path]:
        return [path for name in ASSET_FILENAMES for path in [self.path(name)] if not path.is_file()]

    def audit(self) -> dict[str, Any]:
        missing = self.missing()
        return {
            "root": str(self.root),
            "required_files": len(ASSET_FILENAMES),
            "missing": [str(path) for path in missing],
            "status": "PASS" if not missing else "INCOMPLETE",
        }


def _read_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".json", ".jsonl"}:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("PyYAML is required for YAML configuration files") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value or {}


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    """Load a YAML/JSON config while keeping the public surface compact."""

    if path is None:
        default_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        source = default_path if default_path.is_file() else None
        if source is None:
            return RuntimeConfig()
    else:
        source = Path(path)
    raw = _read_mapping(source)
    model_raw = dict(raw.get("model", {}))
    mlc_raw = dict(raw.get("multilabel", {}))
    regression_raw = dict(raw.get("regression", {}))
    detection_raw = dict(raw.get("detection", {}))
    path_raw = dict(raw.get("paths", {}))
    model = ModelConfig(
        name=str(model_raw.get("name", ModelConfig.name)),
        image_size=int(model_raw.get("image_size", ModelConfig.image_size)),
        feature_batch_size=int(model_raw.get("feature_batch_size", ModelConfig.feature_batch_size)),
        vocab_chunk_size=int(model_raw.get("vocab_chunk_size", model_raw.get("token_chunk", ModelConfig.vocab_chunk_size))),
        max_new_tokens=int(model_raw.get("max_new_tokens", ModelConfig.max_new_tokens)),
        seed=int(model_raw.get("seed", ModelConfig.seed)),
    )
    legacy_row_batch_size = model_raw.get("scoring_row_batch_size", MultilabelConfig.scoring_row_batch_size)
    multilabel = MultilabelConfig(
        candidate_change_threshold=float(mlc_raw.get("candidate_change_threshold", MultilabelConfig.candidate_change_threshold)),
        max_replacement_candidates=int(mlc_raw.get("max_replacement_candidates", MultilabelConfig.max_replacement_candidates)),
        scoring_row_batch_size=int(mlc_raw.get("scoring_row_batch_size", legacy_row_batch_size)),
    )
    quantiles_raw = regression_raw.get("quantiles", RegressionConfig.quantiles)
    quantiles = tuple(float(value) for value in quantiles_raw)
    if len(quantiles) != 3:
        raise ValueError("regression.quantiles must contain exactly three values")
    regression = RegressionConfig(
        retrieval_neighbors=int(regression_raw.get("retrieval_neighbors", RegressionConfig.retrieval_neighbors)),
        generated_fusion_weight=float(regression_raw.get("generated_fusion_weight", RegressionConfig.generated_fusion_weight)),
        visual_fusion_weight=float(regression_raw.get("visual_fusion_weight", RegressionConfig.visual_fusion_weight)),
        base_fusion_weight=float(regression_raw.get("base_fusion_weight", RegressionConfig.base_fusion_weight)),
        retrieval_fusion_weight=float(regression_raw.get("retrieval_fusion_weight", RegressionConfig.retrieval_fusion_weight)),
        residual_correction_weight=float(regression_raw.get("residual_correction_weight", RegressionConfig.residual_correction_weight)),
        quantile_fusion_weight=float(regression_raw.get("quantile_fusion_weight", RegressionConfig.quantile_fusion_weight)),
        quantiles=quantiles,  # type: ignore[arg-type]
        geometry_pca_components=int(regression_raw.get("geometry_pca_components", RegressionConfig.geometry_pca_components)),
    )
    detection = DetectionConfig(
        presence_threshold=float(detection_raw.get("presence_threshold", DetectionConfig.presence_threshold)),
        vision_dim=int(detection_raw.get("vision_dim", DetectionConfig.vision_dim)),
        head_dim=int(detection_raw.get("head_dim", detection_raw.get("hidden", DetectionConfig.head_dim))),
        attention_heads=int(detection_raw.get("attention_heads", detection_raw.get("heads", DetectionConfig.attention_heads))),
        decoder_layers=int(detection_raw.get("decoder_layers", DetectionConfig.decoder_layers)),
        decoder_ffn_dim=int(detection_raw.get("decoder_ffn_dim", detection_raw.get("decoder_feedforward_dim", DetectionConfig.decoder_ffn_dim))),
        dropout=float(detection_raw.get("dropout", DetectionConfig.dropout)),
        max_queries=int(detection_raw.get("max_queries", DetectionConfig.max_queries)),
        seed=int(detection_raw.get("seed", DetectionConfig.seed)),
        epochs=int(detection_raw.get("epochs", DetectionConfig.epochs)),
        batch_size=int(detection_raw.get("batch_size", DetectionConfig.batch_size)),
        learning_rate=float(detection_raw.get("learning_rate", DetectionConfig.learning_rate)),
        weight_decay=float(detection_raw.get("weight_decay", DetectionConfig.weight_decay)),
        warmup_ratio=float(detection_raw.get("warmup_ratio", DetectionConfig.warmup_ratio)),
        max_grad_norm=float(detection_raw.get("max_grad_norm", DetectionConfig.max_grad_norm)),
        matching_giou_weight=float(detection_raw.get("matching_giou_weight", DetectionConfig.matching_giou_weight)),
        box_loss_weight=float(detection_raw.get("box_loss_weight", DetectionConfig.box_loss_weight)),
        presence_loss_weight=float(detection_raw.get("presence_loss_weight", DetectionConfig.presence_loss_weight)),
    )
    return RuntimeConfig(
        model=model,
        multilabel=multilabel,
        regression=regression,
        detection=detection,
        checkpoint_dir=Path(path_raw.get("checkpoint_dir", "checkpoints")),
    )


__all__ = [
    "ASSET_FILENAMES", "AssetBundle", "DetectionConfig", "ModelConfig",
    "MultilabelConfig", "RegressionConfig", "RuntimeConfig", "load_config",
]
