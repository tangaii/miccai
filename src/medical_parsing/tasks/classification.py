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
    extract_image_tokens,
    generated_text,
    load_adapter_bundle,
    load_raw_bundle,
    prompt_only_generated_text,
)
from medical_parsing.models.classification_head import (
    SLOT_CLASSES,
    load_semantic_head_ensemble,
    map_semantic_concept_to_option,
    resolve_classification_slot,
)
from medical_parsing.schema import image_sha, normalize_text, parse_choices, single_image_ref


SEMANTIC_HEAD_ROUTE = "semantic_head"
DIRECT_PROMPT_ROUTE = "direct_prompt_generation"
INSTRUCTIONAL_FALLBACK_ROUTE = "instructional_generation_fallback"
SEMANTIC_HEAD_DATASETS = frozenset({"bone_marrow", "fundus", "iugc"})
ROUTE_ALIASES = {
    "semantic": SEMANTIC_HEAD_ROUTE,
    "semantic_head": SEMANTIC_HEAD_ROUTE,
    "prompt": DIRECT_PROMPT_ROUTE,
    "direct_prompt_generation": DIRECT_PROMPT_ROUTE,
    "fallback": INSTRUCTIONAL_FALLBACK_ROUTE,
    "instructional_generation_fallback": INSTRUCTIONAL_FALLBACK_ROUTE,
}


def load_route_manifest(assets: AssetBundle) -> dict[str, str]:
    payload = json.loads(assets.path("classification_routes").read_text(encoding="utf-8"))
    routes = payload.get("routes", payload)
    if not isinstance(routes, dict):
        raise ValueError("classification route manifest must contain a routes object")
    normalized: dict[str, str] = {}
    for key, value in routes.items():
        raw_route = str(value).strip().lower()
        try:
            normalized[str(key)] = ROUTE_ALIASES[raw_route]
        except KeyError as exc:
            raise ValueError(
                f"unsupported classification route {value!r}; expected one of {sorted(ROUTE_ALIASES)}"
            ) from exc
    return normalized


def route_for_row(row: dict[str, Any], routes: dict[str, str]) -> str:
    key = f"{row['dataset']}::{normalize_text(row.get('source_question') or row.get('question') or row.get('prompt'))}"
    route_value = str(routes.get(key, INSTRUCTIONAL_FALLBACK_ROUTE)).strip().lower()
    try:
        return ROUTE_ALIASES[route_value]
    except KeyError as exc:
        raise ValueError(f"unsupported classification route {route_value!r} for {row['uid']}") from exc


def map_generated_classification(raw: str, row: dict[str, Any]) -> str:
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
    semantic_head_rows = [
        row for row, route in zip(rows, route_values)
        if route == SEMANTIC_HEAD_ROUTE and row["dataset"] in SEMANTIC_HEAD_DATASETS
    ]
    prompt_rows = [row for row, route in zip(rows, route_values) if route == DIRECT_PROMPT_ROUTE]
    fallback_rows = [
        row for row, route in zip(rows, route_values)
        if route == INSTRUCTIONAL_FALLBACK_ROUTE
        or (route == SEMANTIC_HEAD_ROUTE and row["dataset"] not in SEMANTIC_HEAD_DATASETS)
    ]
    output: dict[str, str] = {}

    if semantic_head_rows:
        raw_model, raw_processor, _ = load_raw_bundle(base_path, device, model_config)
        tokens = extract_image_tokens(
            raw_model, raw_processor, semantic_head_rows, device, model_config,
            prompt="semantic image feature",
        )
        clear_model(raw_model, raw_processor)
        import torch

        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(semantic_head_rows):
            groups[image_sha(single_image_ref(row))].append(index)
        group_keys = list(groups)
        group_tokens = torch.from_numpy(np.stack([tokens[groups[key][0]] for key in group_keys])).to(device)
        token_index = {key: index for index, key in enumerate(group_keys)}
        heads = load_semantic_head_ensemble(assets.root, device)
        with torch.inference_mode():
            for row in semantic_head_rows:
                digest = image_sha(single_image_ref(row))
                slot = resolve_classification_slot(row, parse_choices)
                probabilities = []
                for head in heads:
                    logits = head.forward_source(group_tokens[token_index[digest]:token_index[digest] + 1], row["dataset"])[slot]
                    probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy()[0])
                concept = SLOT_CLASSES[slot][int(np.argmax(np.mean(np.stack(probabilities), axis=0)))]
                output[row["uid"]] = map_semantic_concept_to_option(
                    row, slot, concept, parse_choices, normalize_text,
                )
        del heads, group_tokens, tokens
        gc.collect()
        torch.cuda.empty_cache()

    if prompt_rows:
        model, processor, model_audit = load_adapter_bundle(base_path, adapter_path, device, model_config)
        for row in prompt_rows:
            output[row["uid"]] = map_generated_classification(
                prompt_only_generated_text(model, processor, row, device, model_config), row,
            )
        audit["prompt_adapter_load"] = model_audit
        clear_model(model, processor)

    if fallback_rows:
        model, processor, model_audit = load_adapter_bundle(base_path, adapter_path, device, model_config)
        for row in fallback_rows:
            output[row["uid"]] = map_generated_classification(
                generated_text(model, processor, row, device, model_config), row,
            )
        audit["fallback_adapter_load"] = model_audit
        clear_model(model, processor)

    if set(output) != {row["uid"] for row in rows}:
        raise RuntimeError("classification route coverage is incomplete")
    audit.update({
        "routes": dict(route_counts),
        "route_decisions": {
            SEMANTIC_HEAD_ROUTE: len(semantic_head_rows),
            DIRECT_PROMPT_ROUTE: len(prompt_rows),
            INSTRUCTIONAL_FALLBACK_ROUTE: len(fallback_rows),
            "semantic_head_ineligible_fallback": sum(
                route == SEMANTIC_HEAD_ROUTE and row["dataset"] not in SEMANTIC_HEAD_DATASETS
                for row, route in zip(rows, route_values)
            ),
        },
        "rows": len(rows),
    })
    return output


def parse_generated_classification(raw: str, row: dict[str, Any]) -> str:
    """Backward-compatible alias for :func:`map_generated_classification`."""

    return map_generated_classification(raw, row)


__all__ = [
    "DIRECT_PROMPT_ROUTE", "INSTRUCTIONAL_FALLBACK_ROUTE", "ROUTE_ALIASES",
    "SEMANTIC_HEAD_DATASETS", "SEMANTIC_HEAD_ROUTE", "load_route_manifest",
    "map_generated_classification", "parse_generated_classification", "route_for_row",
    "run_classification",
]
