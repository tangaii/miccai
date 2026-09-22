"""Spatial query decoder for Frozen Spatial Query Decoding.

The head is deliberately kept in its own module because the public inference
runner loads it from an external checkpoint. The layer names and defaults
below define the released spatial-query state dictionary contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


SPATIAL_QUERY_VISION_DIM = 2560
SPATIAL_QUERY_HIDDEN_DIM = 256
SPATIAL_QUERY_ATTENTION_HEADS = 8
SPATIAL_QUERY_DROPOUT = 0.1
SPATIAL_QUERY_DECODER_LAYERS = 2
SPATIAL_QUERY_DECODER_FFN_DIM = 1024
SPATIAL_QUERY_PRESENCE_THRESHOLD = 0.5
SPATIAL_QUERY_MAX_QUERIES = 4
FROZEN_SPATIAL_QUERY_ASSET_SIZE_BYTES = 6_988_619
FROZEN_SPATIAL_QUERY_ASSET_SHA256 = "1930fe5821a49604bd7dbf5c987664722a7dc3a990fa026d98ffe0d14f92bc9c"


class FrozenSpatialQueryDecoder(nn.Module):
    """Decode normalized boxes and presence logits from frozen MedGemma states."""

    def __init__(
        self,
        k: int = 1,
        *,
        vision_dim: int = SPATIAL_QUERY_VISION_DIM,
        hidden_dim: int = SPATIAL_QUERY_HIDDEN_DIM,
        attention_heads: int = SPATIAL_QUERY_ATTENTION_HEADS,
        dropout: float = SPATIAL_QUERY_DROPOUT,
        decoder_layers: int = SPATIAL_QUERY_DECODER_LAYERS,
        decoder_ffn_dim: int = SPATIAL_QUERY_DECODER_FFN_DIM,
    ) -> None:
        super().__init__()
        if int(k) < 1:
            raise ValueError(f"k must be positive, got {k}")
        self.k = int(k)
        self.vision_projection = nn.Linear(int(vision_dim), int(hidden_dim))
        self.query_projection = nn.Linear(int(vision_dim), int(hidden_dim))
        self.object_queries = nn.Parameter(torch.randn(self.k, int(hidden_dim)) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=int(hidden_dim),
            nhead=int(attention_heads),
            dim_feedforward=int(decoder_ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(decoder_layers))
        self.box_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 4),
        )
        self.presence_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, image_tokens: Any, query_state: Any) -> tuple[Any, Any]:
        """Return ``(sigmoid(cx, cy, w, h), presence_logits)``."""

        memory = self.vision_projection(image_tokens)
        query = self.object_queries.unsqueeze(0).expand(image_tokens.shape[0], -1, -1)
        query = query + self.query_projection(query_state).unsqueeze(1)
        decoded = self.decoder(query, memory)
        boxes = torch.sigmoid(self.box_head(decoded))
        presence = self.presence_head(decoded).squeeze(-1)
        return boxes, presence


def make_frozen_spatial_query_decoder(k: int = 1) -> FrozenSpatialQueryDecoder:
    """Construct the spatial query decoder with the released architecture."""

    return FrozenSpatialQueryDecoder(k)


def _expected_architecture() -> dict[str, Any]:
    return {
        "vision_dim": SPATIAL_QUERY_VISION_DIM,
        "hidden": SPATIAL_QUERY_HIDDEN_DIM,
        "heads": SPATIAL_QUERY_ATTENTION_HEADS,
        "dropout": SPATIAL_QUERY_DROPOUT,
        "decoder_layers": SPATIAL_QUERY_DECODER_LAYERS,
        "box_mlp": "256->256->4 sigmoid",
        "presence": "256->1",
    }


def validate_spatial_query_decoder_checkpoint(payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    """Validate checkpoint metadata and return ``(K, architecture)``."""

    if "state_dict" not in payload or not isinstance(payload["state_dict"], Mapping):
        raise RuntimeError("spatial query decoder checkpoint must contain a state_dict mapping")
    try:
        k = int(payload.get("K", 1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("spatial query decoder checkpoint K is not an integer") from exc
    if not 1 <= k <= SPATIAL_QUERY_MAX_QUERIES:
        raise RuntimeError(
            "spatial query decoder checkpoint K must be in "
            f"[1, {SPATIAL_QUERY_MAX_QUERIES}], got {k}"
        )
    architecture = dict(payload.get("architecture", {}))
    expected = _expected_architecture()
    mismatches = {
        key: (architecture.get(key), value)
        for key, value in expected.items()
        if key in architecture and architecture[key] != value
    }
    if mismatches:
        raise RuntimeError(f"spatial query decoder architecture mismatch: {mismatches}")
    return k, architecture


def load_frozen_spatial_query_decoder(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[FrozenSpatialQueryDecoder, dict[str, Any]]:
    """Load the frozen spatial query decoder with strict state matching.

    ``dtype`` defaults to the checkpoint tensor dtype.  Passing
    ``torch.bfloat16`` reproduces the deployment path used by the frozen
    NVIDIA asset; CPU callers can omit it and retain a portable checkpoint
    dtype when supported by the installed PyTorch build.
    """

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"spatial query decoder checkpoint is not a mapping: {path}")
    k, architecture = validate_spatial_query_decoder_checkpoint(payload)
    state_dict = payload["state_dict"]
    if dtype is None:
        dtype = next(
            (value.dtype for value in state_dict.values() if isinstance(value, torch.Tensor)),
            torch.float32,
        )
    head = make_frozen_spatial_query_decoder(k).to(device=device, dtype=dtype)
    head.load_state_dict(state_dict, strict=True)
    head.eval()
    metadata = {
        "path": str(path),
        "K": k,
        "architecture": architecture or _expected_architecture(),
        "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "dtype": str(dtype),
    }
    return head, metadata


# The shorter names remain available for callers that use the generic
# detection-head terminology; they point to the same implementation.
SpatialDetectionHead = FrozenSpatialQueryDecoder
make_spatial_detection_head = make_frozen_spatial_query_decoder


__all__ = [
    "FROZEN_SPATIAL_QUERY_ASSET_SHA256", "FROZEN_SPATIAL_QUERY_ASSET_SIZE_BYTES",
    "FrozenSpatialQueryDecoder", "SPATIAL_QUERY_ATTENTION_HEADS",
    "SPATIAL_QUERY_DECODER_FFN_DIM", "SPATIAL_QUERY_DECODER_LAYERS",
    "SPATIAL_QUERY_DROPOUT", "SPATIAL_QUERY_HIDDEN_DIM",
    "SPATIAL_QUERY_MAX_QUERIES", "SPATIAL_QUERY_PRESENCE_THRESHOLD",
    "SPATIAL_QUERY_VISION_DIM", "SpatialDetectionHead",
    "load_frozen_spatial_query_decoder", "make_frozen_spatial_query_decoder",
    "make_spatial_detection_head", "validate_spatial_query_decoder_checkpoint",
]
