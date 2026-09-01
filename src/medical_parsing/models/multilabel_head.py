"""Token-conditioned residual probability head for multi-label inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class MultiLabelResidualProbabilityHead(nn.Module):
    """Refine 10-by-4 atom/cardinality probabilities from image tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(2560)
        self.token_project = nn.Sequential(nn.Linear(2560, 128), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(946, 256), nn.GELU(), nn.Dropout(.20),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(.10),
            nn.Linear(128, 40),
        )

    def forward(self, tokens: Any, row_features: Any, base_logits: Any) -> Any:
        projected = self.token_project(self.token_norm(tokens))
        quantiles = torch.quantile(
            projected.float(),
            torch.tensor([.10, .25, .50, .75, .90], device=projected.device, dtype=torch.float32),
            dim=1,
        ).permute(1, 0, 2).reshape(tokens.shape[0], -1)
        distribution = torch.cat([
            quantiles, projected.mean(dim=1), projected.std(dim=1, unbiased=False),
        ], dim=1)
        raw = self.head(torch.cat([distribution, row_features], dim=1))
        return base_logits + 0.50 * torch.tanh(raw)

    def predict(self, tokens: Any, row_features: Any, base_probability: np.ndarray) -> np.ndarray:
        base = np.clip(np.asarray(base_probability, dtype=np.float64).reshape(len(row_features), 40), 1e-4, 1.0 - 1e-4)
        base_logits = torch.from_numpy(np.log(base / (1.0 - base))).to(row_features.device, dtype=torch.float32)
        enabled = bool(tokens.is_cuda)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
                corrected = self(tokens, row_features, base_logits)
            return torch.sigmoid(corrected).float().cpu().numpy().reshape(-1, 10, 4).astype(np.float64)


__all__ = ["MultiLabelResidualProbabilityHead"]
