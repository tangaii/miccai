"""Shared, intentionally small training helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_npz(path: str | Path, required: Iterable[str]) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=False)
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"feature archive is missing keys: {missing}")
    return {name: payload[name] for name in payload.files}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_2d(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def seed_everything(seed: int = 0) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
