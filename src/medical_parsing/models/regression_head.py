"""Spatial quantile regression head."""

from __future__ import annotations

import math
from typing import Any


def make_spatial_quantile_head():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SpatialQuantileHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_norm = nn.LayerNorm(2560)
            self.token_project = nn.Sequential(nn.Linear(2560, 128), nn.GELU(), nn.Dropout(.15))
            coordinates = torch.linspace(-1.0, 1.0, 16)
            y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
            self.register_buffer("coordinates", torch.stack([x.reshape(-1), y.reshape(-1)], dim=1), persistent=False)
            self.coordinate_project = nn.Linear(2, 128)
            self.queries = nn.Parameter(torch.empty(4, 128))
            nn.init.normal_(self.queries, mean=0.0, std=.02)
            self.output = nn.Sequential(nn.Linear(704, 128), nn.GELU(), nn.Dropout(.20), nn.Linear(128, 3))

        def forward(self, tokens: Any, geometry64: Any) -> Any:
            projected = self.token_project(self.token_norm(tokens))
            projected = projected + self.coordinate_project(self.coordinates).unsqueeze(0)
            attention_logits = torch.einsum("qd,btd->bqt", self.queries, projected) / math.sqrt(128)
            attention = torch.softmax(attention_logits, dim=-1)
            queried = torch.einsum("bqt,btd->bqd", attention, projected).reshape(tokens.shape[0], -1)
            pooled = projected.mean(dim=1)
            raw = self.output(torch.cat([queried, pooled, geometry64], dim=1))
            median = raw[:, 0]
            return torch.stack([median - F.softplus(raw[:, 1]), median, median + F.softplus(raw[:, 2])], dim=1)

    return SpatialQuantileHead()

