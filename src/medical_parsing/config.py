"""Small, explicit runtime configuration and external-asset resolution."""

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
    scoring_row_batch_size: int = 4
    token_chunk: int = 16
    max_new_tokens: int = 256
    seed: int = 0


@dataclass(frozen=True)
class MultilabelConfig:
    candidate_change_threshold: float = 0.05
    max_replacement_candidates: int = 32


@dataclass(frozen=True)
class RuntimeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    multilabel: MultilabelConfig = field(default_factory=MultilabelConfig)
    checkpoint_dir: Path = Path("checkpoints")


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
        return RuntimeConfig()
    source = Path(path)
    raw = _read_mapping(source)
    model_raw = dict(raw.get("model", {}))
    mlc_raw = dict(raw.get("multilabel", {}))
    path_raw = dict(raw.get("paths", {}))
    model = ModelConfig(
        name=str(model_raw.get("name", ModelConfig.name)),
        image_size=int(model_raw.get("image_size", ModelConfig.image_size)),
        feature_batch_size=int(model_raw.get("feature_batch_size", ModelConfig.feature_batch_size)),
        scoring_row_batch_size=int(model_raw.get("scoring_row_batch_size", ModelConfig.scoring_row_batch_size)),
        token_chunk=int(model_raw.get("token_chunk", ModelConfig.token_chunk)),
        max_new_tokens=int(model_raw.get("max_new_tokens", ModelConfig.max_new_tokens)),
        seed=int(model_raw.get("seed", ModelConfig.seed)),
    )
    multilabel = MultilabelConfig(
        candidate_change_threshold=float(mlc_raw.get("candidate_change_threshold", MultilabelConfig.candidate_change_threshold)),
        max_replacement_candidates=int(mlc_raw.get("max_replacement_candidates", MultilabelConfig.max_replacement_candidates)),
    )
    return RuntimeConfig(
        model=model,
        multilabel=multilabel,
        checkpoint_dir=Path(path_raw.get("checkpoint_dir", "checkpoints")),
    )

