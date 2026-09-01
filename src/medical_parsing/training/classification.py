"""Fit the semantic classification head pack from cached image tokens."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from medical_parsing.models.classification_head import SLOT_CLASSES, SemanticImageTokenHead
from .common import seed_everything


def _label_index(slot: str, value: Any) -> int:
    classes = SLOT_CLASSES[slot]
    if isinstance(value, (int, np.integer)):
        index = int(value)
    else:
        text = str(value).strip().lower()
        aliases = {
            "yes": "root_yes", "no": "root_no", "standard plane": "standard_plane",
            "non-standard plane": "nonstandard_plane",
        }
        text = aliases.get(text, text)
        index = classes.index(text)
    if not 0 <= index < len(classes):
        raise ValueError(f"label index outside slot vocabulary: {slot} / {value!r}")
    return index


def _fold_indices(size: int, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold

    if size < 2:
        raise ValueError("at least two labeled rows are required")
    splitter = KFold(n_splits=min(folds, size), shuffle=True, random_state=seed)
    result = list(splitter.split(np.arange(size)))
    while len(result) < folds:
        result.append(result[-1])
    return [(np.asarray(train), np.asarray(valid)) for train, valid in result[:folds]]


def train_semantic_heads(
    tokens: np.ndarray,
    sources: Sequence[str],
    slots: Sequence[str],
    labels: Sequence[Any],
    output_path: str | Path,
    *,
    folds: int = 3,
    epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    seed: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    """Train one shared head per fold and write the external head pack.

    ``tokens`` are the shared backbone's ``[N, 256, 2560]`` image-token
    features.  Labels are slot names from :mod:`medical_parsing.models` or
    integer class indices.  The task-specific data split remains outside the
    public repository.
    """

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(seed)
    tokens = np.asarray(tokens)
    if tokens.ndim != 3 or tokens.shape[1:] != (256, 2560):
        raise ValueError(f"tokens must have shape [N,256,2560], got {tokens.shape}")
    if not (len(tokens) == len(sources) == len(slots) == len(labels)):
        raise ValueError("tokens, sources, slots, and labels must have equal length")
    device_name = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fold_payload: dict[str, Any] = {}
    for fold_index, (train_index, _valid_index) in enumerate(_fold_indices(len(tokens), folds, seed)):
        model = SemanticImageTokenHead().to(device_name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        for _epoch in range(epochs):
            order = train_index.copy()
            np.random.shuffle(order)
            for start in range(0, len(order), batch_size):
                batch_indices = order[start:start + batch_size]
                batch_tokens = torch.from_numpy(tokens[batch_indices].astype(np.float32)).to(device_name)
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for local_index, source_index in enumerate(batch_indices):
                    source = str(sources[source_index])
                    slot = str(slots[source_index])
                    logits = model.forward_source(batch_tokens[local_index:local_index + 1], source)[slot]
                    target = torch.tensor([_label_index(slot, labels[source_index])], device=device_name)
                    losses.append(torch.nn.functional.cross_entropy(logits, target))
                if losses:
                    torch.stack(losses).mean().backward()
                    optimizer.step()
        fold_payload[str(fold_index)] = {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    torch.save({"schema": "semantic-classification-heads-v1", "folds": fold_payload}, output)
    return {"status": "PASS", "output": str(output), "folds": folds, "rows": len(tokens)}


__all__ = ["train_semantic_heads"]
