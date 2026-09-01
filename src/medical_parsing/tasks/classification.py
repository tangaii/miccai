"""Disease-diagnosis classification branch with explicit semantic routing."""

from __future__ import annotations

from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from medical_parsing.config import AssetBundle, ModelConfig
from medical_parsing.models.backbone import (
    clear_model,
    generated_text,
    image_features,
    load_adapter_bundle,
    load_raw_bundle,
    prompt_only_generated_text,
)
from medical_parsing.models.classification_head import (
    SLOT_CLASSES,
    load_semantic_heads,
    resolve_slot,
    serialize_concept,
)
from medical_parsing.schema import image_sha, normalize_text, parse_choices, row_image_refs


def load_route_manifest(assets: AssetBundle) -> dict[str, str]:
    payload = json.loads(assets.path("classification_routes").read_text(encoding="utf-8"))
    routes = payload.get("routes", payload)
    if not isinstance(routes, dict):
        raise ValueError("classification route manifest must contain a routes object")
    return {str(key): str(value).lower() for key, value in routes.items()}


def route_for_row(row: dict[str, Any], routes: dict[str, str]) -> str:
    key = f"{row['dataset']}::{normalize_text(row.get('source_question') or row.get('question') or row.get('prompt'))}"
    route = routes.get(key, "fallback")
    if route not in {"semantic", "prompt", "fallback"}:
        raise ValueError(f"unsupported classification route {route!r} for {row['uid']}")
    return route


def parse_generated_classification(raw: str, row: dict[str, Any]) -> str:
    choices = parse_choices(str(row.get("source_question") or row.get("question") or row.get("prompt")))
    text = normalize_text(raw)
    match = re.search(r"\b([a-k])\b", text)
    legal = {letter for letter, _ in choices}
    if match and match.group(1).upper() in legal:
        return match.group(1).upper()
    exact = [(letter, label) for letter, label in choices if text == label or text.startswith(label + " ")]
    if len(exact) == 1:
        return exact[0][0]
    for letter, label in choices:
        if label and label in text:
            return letter
    raise RuntimeError(f"unable to map model classification output for {row['uid']}: {raw!r}")


def run_classification(
    rows: list[dict[str, Any]],
    base_path: Path,
    adapter_path: Path,
    device: str,
    assets: AssetBundle,
    model_config: ModelConfig,
    audit: dict[str, Any],
) -> dict[str, str]:
    if not rows:
        return {}
    routes = load_route_manifest(assets)
    route_values = [route_for_row(row, routes) for row in rows]
    route_counts = Counter(route_values)
    semantic_rows = [row for row, route in zip(rows, route_values) if route == "semantic" and row["dataset"] in {"bone_marrow", "fundus", "iugc"}]
    prompt_rows = [row for row, route in zip(rows, route_values) if route == "prompt"]
    fallback_rows = [row for row, route in zip(rows, route_values) if row not in semantic_rows and route != "prompt"]
    output: dict[str, str] = {}

    if semantic_rows:
        raw_model, raw_processor, _ = load_raw_bundle(base_path, device, model_config)
        tokens = image_features(raw_model, raw_processor, semantic_rows, device, model_config, prompt="semantic image feature")
        clear_model(raw_model, raw_processor)
        import torch

        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(semantic_rows):
            groups[image_sha(row_image_refs(row)[0])].append(index)
        group_keys = list(groups)
        group_tokens = torch.from_numpy(np.stack([tokens[groups[key][0]] for key in group_keys])).to(device)
        token_index = {key: index for index, key in enumerate(group_keys)}
        heads = load_semantic_heads(assets.root, device)
        with torch.inference_mode():
            for row in semantic_rows:
                digest = image_sha(row_image_refs(row)[0])
                slot = resolve_slot(row, parse_choices)
                probabilities = []
                for head in heads:
                    logits = head.forward_source(group_tokens[token_index[digest]:token_index[digest] + 1], row["dataset"])[slot]
                    probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy()[0])
                concept = SLOT_CLASSES[slot][int(np.argmax(np.mean(np.stack(probabilities), axis=0)))]
                output[row["uid"]] = serialize_concept(row, slot, concept, parse_choices, normalize_text)
        del heads, group_tokens, tokens
        gc.collect()
        torch.cuda.empty_cache()

    if prompt_rows:
        model, processor, model_audit = load_adapter_bundle(base_path, adapter_path, device, model_config)
        for row in prompt_rows:
            output[row["uid"]] = parse_generated_classification(prompt_only_generated_text(model, processor, row, device, model_config), row)
        audit["prompt_adapter_load"] = model_audit
        clear_model(model, processor)

    if fallback_rows:
        model, processor, model_audit = load_adapter_bundle(base_path, adapter_path, device, model_config)
        for row in fallback_rows:
            output[row["uid"]] = parse_generated_classification(generated_text(model, processor, row, device, model_config), row)
        audit["fallback_adapter_load"] = model_audit
        clear_model(model, processor)

    if set(output) != {row["uid"] for row in rows}:
        raise RuntimeError("classification route coverage is incomplete")
    audit.update({
        "routes": dict(route_counts),
        "route_decisions": {
            "semantic_head": len(semantic_rows),
            "prompt_generation": len(prompt_rows),
            "fallback_generation": len(fallback_rows),
        },
        "rows": len(rows),
    })
    return output

