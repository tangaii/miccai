"""Structured multi-label prediction and memory-efficient evidence scoring.

The branch deliberately keeps the vocabulary, candidate ordering, feature
layouts, and GFM decoder as small named functions.  This makes the public
implementation auditable and lets the tests compare the memory-efficient
teacher-forced scorer with a reference full-logit implementation.
"""

from __future__ import annotations

from collections import Counter
import gc
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from medical_parsing.config import AssetBundle, ModelConfig, MultilabelConfig
from medical_parsing.models.backbone import (
    DecoderCapabilityError,
    clear_model,
    decoder_hidden_states,
    extract_image_tokens,
    final_logit_softcap,
    generated_text,
    load_adapter_bundle,
    messages_for,
)
from medical_parsing.models.multilabel_head import MultiLabelResidualProbabilityHead
from medical_parsing.schema import (
    image_sha,
    normalize_text,
    parse_label_set,
    prepared_image,
    single_image_ref,
)


ATOMS = [
    "and complications",
    "chronic injury of tooth hard tissues",
    "dental caries",
    "disturbances of eruption of teeth",
    "endodontic treatment",
    "endodontic treatment, restorative treatment, and complications",
    "periodontal diseases",
    "periradicular lesions",
    "pulp diseases",
    "restorative treatment",
]
ATOM_INDEX = {value: index for index, value in enumerate(ATOMS)}
SEMANTIC_LABEL_ATOMS = [
    "dental caries",
    "periradicular lesions",
    "pulp diseases",
    "chronic injury of tooth hard tissues",
    "disturbances of eruption of teeth",
    "periodontal diseases",
    "endodontic treatment, restorative treatment, and complications",
]
AUXILIARY_LABEL_ATOMS = ["and complications", "endodontic treatment", "restorative treatment"]
# Compatibility names retained for serialized feature layouts and downstream
# callers.  The formal names above describe their mathematical roles.
SEMANTIC = SEMANTIC_LABEL_ATOMS
PSEUDO = AUXILIARY_LABEL_ATOMS
CARDINALITIES = (1, 2, 3, 4)
RAW_FORMS = ("SINGLE", "COMMA", "SEMICOLON", "MIXED")


def parse_multilabel(value: Any) -> set[str]:
    """Compatibility alias for the canonical schema label-set parser."""

    return parse_label_set(value)


def set_key(value: Iterable[str]) -> tuple[str, ...]:
    """Return the deterministic atom order used in serialized sets and ties."""

    return tuple(sorted(set(value), key=lambda item: (ATOM_INDEX.get(item, 999), item)))


def serialize_effective(value: set[str]) -> str:
    unknown = set(value) - set(ATOMS)
    if unknown:
        raise ValueError(f"unknown multi-label atoms: {sorted(unknown)}")
    text = "; ".join(set_key(value)) + (";" if value else "")
    if parse_multilabel(text) != set(value):
        raise RuntimeError("multi-label serializer/parser roundtrip failure")
    return text


def raw_form(raw: str) -> str:
    if "," in raw and ";" in raw:
        return "MIXED"
    if ";" in raw:
        return "SEMICOLON"
    if "," in raw:
        return "COMMA"
    return "SINGLE"


def tooth_features(question: str) -> tuple[int, int]:
    match = re.search(r"\btooth\s*(\d{2})\b", question, flags=re.IGNORECASE)
    if not match:
        return -1, -1
    fdi = int(match.group(1))
    return fdi % 10, fdi // 10


def mlc_metadata(
    rows: list[dict[str, Any]],
    initial_generated_text_by_uid: dict[str, str],
    template_map: dict[str, int],
) -> list[dict[str, Any]]:
    counts = Counter(image_sha(single_image_ref(row)) for row in rows)
    values: list[dict[str, Any]] = []
    for row in rows:
        question = str(row.get("source_question") or row.get("question") or row.get("prompt"))
        tooth_index, quadrant = tooth_features(question)
        raw = initial_generated_text_by_uid[row["uid"]]
        values.append({
            "image_sha256": image_sha(single_image_ref(row)),
            "template_id": int(template_map.get(normalize_text(question), -1)),
            "fdi": quadrant * 10 + tooth_index if tooth_index >= 0 else -1,
            "quadrant": quadrant,
            "tooth_index": tooth_index,
            "same_image_count": counts[image_sha(single_image_ref(row))],
            "raw_has_comma": int("," in raw),
            "raw_has_semicolon": int(";" in raw),
            "raw_has_newline": int("\n" in raw),
            "raw_length": len(raw),
            "raw_cardinality": len(parse_multilabel(raw)),
            "raw_form": raw_form(raw),
        })
    return values


def _find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1
    end = len(sequence) - len(pattern) + 1
    for index in range(start, end):
        if sequence[index:index + len(pattern)] == pattern:
            return index
    return -1


def _answer_positions(
    input_ids: Any,
    tokenizer: Any,
    mapping: list[tuple[int, int, list[int]]],
    rows: list[dict[str, Any]],
) -> list[tuple[int, int, list[int], list[int]]]:
    """Find answer tokens and their next-token prediction positions."""

    marker_options = [
        tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False),
        tokenizer.encode("model\n", add_special_tokens=False),
    ]
    ids_cpu = input_ids.detach().cpu().tolist()
    positions: list[tuple[int, int, list[int], list[int]]] = []
    for batch_index, (row_index, answer_index, answer_ids) in enumerate(mapping):
        ids = ids_cpu[batch_index]
        marker = -1
        marker_len = 0
        for candidate in marker_options:
            found = _find_subsequence(ids, candidate)
            if found >= 0 and (marker < 0 or found < marker):
                marker, marker_len = found, len(candidate)
        if marker < 0:
            raise RuntimeError(f"model turn marker not found for {rows[row_index]['uid']}")
        answer_start = _find_subsequence(ids, answer_ids, marker + marker_len)
        if answer_start < 0:
            raise RuntimeError(f"answer token sequence not found for {rows[row_index]['uid']}")
        target_positions = list(range(answer_start - 1, answer_start - 1 + len(answer_ids)))
        if any(position < 0 for position in target_positions):
            raise RuntimeError("invalid answer position alignment")
        positions.append((batch_index, answer_index, target_positions, answer_ids))
    return positions


def _apply_softcap(logits: Any, softcap: float | None) -> Any:
    if softcap is None or softcap <= 0:
        return logits
    return softcap * (logits / softcap).tanh()


def score_logits_full(
    logits: Any,
    input_ids: Any,
    tokenizer: Any,
    mapping: list[tuple[int, int, list[int]]],
    rows: list[dict[str, Any]],
    softcap: float | None = None,
) -> np.ndarray:
    """Reference scorer that materializes vocabulary logits.

    This exists for unit tests and for a compatibility fallback with model
    wrappers that do not expose decoder hidden states.
    """

    import torch

    positions = _answer_positions(input_ids, tokenizer, mapping, rows)
    scores = torch.full((len(rows), len({item[1] for item in mapping})), float("nan"), dtype=torch.float64)
    logits = logits.float()
    if softcap is not None:
        logits = _apply_softcap(logits, softcap)
    logp = torch.log_softmax(logits, dim=-1)
    for batch_index, answer_index, target_positions, answer_ids in positions:
        values = [
            logp[batch_index, position, target_id].item()
            for position, target_id in zip(target_positions, answer_ids)
        ]
        scores[mapping[batch_index][0], answer_index] = float(sum(values) / len(values))
    result = scores.numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite full teacher-forced score matrix")
    return result


def score_hidden_streaming(
    hidden_states: Any,
    lm_head: Any,
    input_ids: Any,
    tokenizer: Any,
    mapping: list[tuple[int, int, list[int]]],
    rows: list[dict[str, Any]],
    vocab_chunk_size: int = 16,
    softcap: float | None = None,
    *,
    token_chunk: int | None = None,
) -> np.ndarray:
    """Score answer tokens while materializing at most ``vocab_chunk_size`` rows.

    The hidden states are already produced by the decoder.  Vocabulary rows
    of the language-model head are visited in chunks, so a batch of four rows
    and ten candidate answers never creates a full ``batch x sequence x
    vocabulary`` tensor.
    """

    import torch
    import torch.nn.functional as F

    if token_chunk is not None:
        if vocab_chunk_size != 16 and vocab_chunk_size != token_chunk:
            raise ValueError("vocab_chunk_size and legacy token_chunk disagree")
        vocab_chunk_size = int(token_chunk)
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    positions = _answer_positions(input_ids, tokenizer, mapping, rows)
    answer_count = max((item[1] for item in mapping), default=-1) + 1
    score_sum = np.full((len(rows), answer_count), np.nan, dtype=np.float64)
    weight = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    target_rows: list[tuple[int, int, int, int]] = []
    for batch_index, answer_index, target_positions, answer_ids in positions:
        for position, target_id in zip(target_positions, answer_ids):
            target_rows.append((batch_index, answer_index, position, int(target_id)))
    if not target_rows:
        raise RuntimeError("no answer tokens found")

    batch_indices = torch.tensor([item[0] for item in target_rows], device=hidden_states.device, dtype=torch.long)
    sequence_indices = torch.tensor([item[2] for item in target_rows], device=hidden_states.device, dtype=torch.long)
    target_ids = torch.tensor([item[3] for item in target_rows], device=hidden_states.device, dtype=torch.long)
    target_hidden = hidden_states[batch_indices, sequence_indices].float()
    denominator = torch.full((len(target_rows),), -float("inf"), device=target_hidden.device, dtype=torch.float32)
    vocab_size = int(weight.shape[0])
    for start in range(0, vocab_size, vocab_chunk_size):
        stop = min(start + vocab_chunk_size, vocab_size)
        chunk_weight = weight[start:stop]
        chunk_bias = None if bias is None else bias[start:stop]
        chunk_logits = F.linear(target_hidden, chunk_weight.float(), None if chunk_bias is None else chunk_bias.float())
        chunk_logits = _apply_softcap(chunk_logits, softcap)
        denominator = torch.logaddexp(denominator, torch.logsumexp(chunk_logits, dim=1))
        del chunk_logits, chunk_weight, chunk_bias
    target_weight = weight.index_select(0, target_ids).float()
    target_bias = None if bias is None else bias.index_select(0, target_ids).float()
    target_logits = (target_hidden * target_weight).sum(dim=1)
    if target_bias is not None:
        target_logits = target_logits + target_bias
    target_logits = _apply_softcap(target_logits, softcap)
    log_prob = target_logits - denominator
    sums: dict[tuple[int, int], list[float]] = {}
    for value, (batch_index, answer_index, _position, _target_id) in zip(log_prob.detach().cpu().tolist(), target_rows):
        row_index = mapping[batch_index][0]
        sums.setdefault((row_index, answer_index), []).append(float(value))
    for (row_index, answer_index), values in sums.items():
        score_sum[row_index, answer_index] = float(np.mean(values))
    if not np.isfinite(score_sum).all():
        raise RuntimeError("non-finite streaming teacher-forced score matrix")
    return score_sum


def _teacher_forced_batch(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: str,
    answers: list[str],
    image_size: int,
) -> tuple[dict[str, Any], list[tuple[int, int, list[int]]]]:
    tokenizer = processor.tokenizer
    texts: list[str] = []
    images: list[list[Any]] = []
    mapping: list[tuple[int, int, list[int]]] = []
    for row_index, row in enumerate(rows):
        for answer_index, answer in enumerate(answers):
            texts.append(messages_for(row, processor, answer=answer))
            images.append([prepared_image(single_image_ref(row), image_size=image_size)])
            mapping.append((row_index, answer_index, tokenizer.encode(answer, add_special_tokens=False)))
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    return batch, mapping


def score_teacher_forced_streaming(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: str,
    answers: list[str],
    model_config: ModelConfig,
) -> tuple[np.ndarray, str]:
    """Return parser-native answer scores and the implementation used."""

    import torch

    batch, mapping = _teacher_forced_batch(
        model, processor, rows, device, answers, model_config.image_size,
    )
    with torch.inference_mode():
        try:
            hidden, lm_head, softcap = decoder_hidden_states(model, batch)
        except DecoderCapabilityError as exc:
            # Only wrappers without the decoder/language head may use the
            # compatibility path.  Scoring/alignment/model failures propagate.
            logits = model(**batch, use_cache=False).logits
            scores = score_logits_full(
                logits, batch["input_ids"], processor.tokenizer, mapping, rows,
                softcap=final_logit_softcap(model),
            )
            implementation = f"full-logit-compatibility-fallback:{type(exc).__name__}"
        else:
            scores = score_hidden_streaming(
                hidden, lm_head, batch["input_ids"], processor.tokenizer, mapping,
                rows, vocab_chunk_size=model_config.vocab_chunk_size, softcap=softcap,
            )
            implementation = "decoder-hidden-state-vocabulary-streaming"
    del batch
    return scores, implementation


def score_teacher_forced_batch(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: str,
    answers: list[str],
    model_config: ModelConfig | None = None,
) -> np.ndarray:
    """Compatibility name for the memory-efficient teacher-forced scorer."""

    config = model_config or ModelConfig()
    return score_teacher_forced_streaming(model, processor, rows, device, answers, config)[0]


def build_probability_model_features(
    scores: np.ndarray,
    initial_generated_set: set[str],
    refined_candidate_set: set[str],
    reranked_candidate_set: set[str],
    row: dict[str, Any],
    same_image_count: int,
    raw: str,
) -> np.ndarray:
    tooth_index, quadrant = tooth_features(str(row.get("source_question") or row.get("question") or row.get("prompt")))
    values = list(np.asarray(scores, dtype=np.float64).reshape(-1))
    values += [float(atom in initial_generated_set) for atom in ATOMS]
    values += [float(atom in refined_candidate_set) for atom in ATOMS]
    values += [float(atom in reranked_candidate_set) for atom in ATOMS]
    values += [
        float(len(initial_generated_set)), float(len(refined_candidate_set)),
        float(len(reranked_candidate_set)),
    ]
    values += [float(tooth_index), float(quadrant), float(same_image_count)]
    values += [float(raw_form(raw) == form) for form in RAW_FORMS]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (50,) or not np.isfinite(result).all():
        raise RuntimeError(f"candidate probability feature shape failure: {result.shape}")
    return result


def build_initial_candidate_table(s0: set[str], bank: list[set[str]], limit: int = 32) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()

    def add(value: set[str], action: str, atom: str | None = None, bank_id: int = -1) -> None:
        frozen = frozenset(value)
        if frozen in seen:
            return
        seen.add(frozen)
        candidates.append({"set": set(value), "action": action, "atom": atom, "bank_id": bank_id})

    add(set(s0), "KEEP")
    for atom in ATOMS:
        if atom not in s0:
            add(set(s0) | {atom}, "ADD", atom)
    for atom in ATOMS:
        if atom in s0:
            add(set(s0) - {atom}, "DROP", atom)
    replacements = []
    for index, target in enumerate(bank):
        difference = len(set(s0) ^ set(target))
        if difference <= 2 and set(target) != set(s0):
            replacements.append((difference, set_key(target), index, target))
    for difference, key, index, target in sorted(replacements, key=lambda value: (value[0], value[1])):
        del difference, key
        if len(candidates) >= limit:
            break
        add(set(target), "REPLACE_WITH", bank_id=index)
    return candidates


def build_candidate_selector_features(scores: dict[str, float], meta: dict[str, Any], s0: set[str], candidate: dict[str, Any]) -> np.ndarray:
    ordered = sorted((float(scores[name]) for name in SEMANTIC_LABEL_ATOMS + AUXILIARY_LABEL_ATOMS), reverse=True)
    values: list[float] = [float(scores[name]) for name in SEMANTIC_LABEL_ATOMS + AUXILIARY_LABEL_ATOMS]
    values += [ordered[0], ordered[1], ordered[0] - ordered[1]]
    values += [float(atom in s0) for atom in ATOMS] + [float(atom in candidate["set"]) for atom in ATOMS]
    values += [float(candidate["action"] == action) for action in ("KEEP", "ADD", "DROP", "REPLACE_WITH")]
    values += [float(candidate.get("atom") == atom) for atom in ATOMS]
    values += [
        float(candidate.get("bank_id", -1)), float(len(candidate["set"])), float(len(s0)),
        float(len(s0 ^ candidate["set"])), float(candidate.get("bank_id", -1) >= 0),
    ]
    values += [float(meta[name]) for name in (
        "template_id", "fdi", "quadrant", "tooth_index", "same_image_count",
        "raw_has_comma", "raw_has_semicolon", "raw_has_newline", "raw_length", "raw_cardinality",
    )]
    values += [float(meta["raw_form"] == form) for form in ("SINGLE", "COMMA", "SEMICOLON")]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (65,):
        raise RuntimeError(f"candidate selector feature shape failure: {result.shape}")
    return result


def select_refined_candidate(candidates: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> int:
    best = 0
    best_delta = -float("inf")
    for index in range(1, len(candidates)):
        delta = float(scores[index]) - float(scores[0])
        tie = (len(candidates[index]["set"]), len(candidates[index]["set"] ^ candidates[0]["set"]))
        best_tie = (len(candidates[best]["set"]), len(candidates[best]["set"] ^ candidates[0]["set"]))
        if delta >= threshold and (
            delta > best_delta + 1e-12
            or (abs(delta - best_delta) <= 1e-12 and tie < best_tie)
        ):
            best, best_delta = index, delta
    return best


def build_reranked_candidates(initial_generated_set: set[str], refined_candidate_set: set[str], bank: list[set[str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()

    def add(value: set[str], action: str, atom: str | None = None, bank_id: int = -1) -> None:
        key = frozenset(value)
        if key not in seen:
            seen.add(key)
            result.append({"set": set(value), "action": action, "atom": atom, "bank_id": bank_id})

    add(set(refined_candidate_set), "KEEP")
    if initial_generated_set != refined_candidate_set:
        add(set(initial_generated_set), "REVERT")
    for atom in ATOMS:
        if atom not in refined_candidate_set:
            add(set(refined_candidate_set) | {atom}, "ADD", atom)
        else:
            add(set(refined_candidate_set) - {atom}, "DROP", atom)
    nearest = []
    for index, target in enumerate(bank):
        difference = len(refined_candidate_set ^ target)
        if difference <= 2 and target != refined_candidate_set:
            nearest.append((difference, set_key(target), index, target))
    for difference, key, index, target in sorted(nearest, key=lambda value: (value[0], value[1])):
        del difference, key
        add(set(target), "NEAREST", bank_id=index)
    return result


def build_candidate_ranker_features(
    scores: dict[str, float],
    meta: dict[str, Any],
    initial_generated_set: set[str],
    refined_candidate_set: set[str],
    candidate: dict[str, Any],
) -> np.ndarray:
    ordered = sorted((float(scores[name]) for name in SEMANTIC_LABEL_ATOMS + AUXILIARY_LABEL_ATOMS), reverse=True)
    values: list[float] = [float(scores[name]) for name in SEMANTIC_LABEL_ATOMS + AUXILIARY_LABEL_ATOMS]
    values += [ordered[0], ordered[1], ordered[0] - ordered[1]]
    values += [float(atom in initial_generated_set) for atom in ATOMS]
    values += [float(atom in refined_candidate_set) for atom in ATOMS]
    values += [float(atom in candidate["set"]) for atom in ATOMS]
    values += [float(candidate["action"] == action) for action in ("KEEP", "REVERT", "ADD", "DROP", "NEAREST")]
    values += [float(candidate.get("atom") == atom) for atom in ATOMS]
    values += [
        float(candidate.get("bank_id", -1)), float(len(candidate["set"])), float(len(refined_candidate_set)),
        float(len(initial_generated_set)), float(len(refined_candidate_set ^ candidate["set"])),
        float(len(refined_candidate_set ^ initial_generated_set)),
        float(len(initial_generated_set ^ candidate["set"])),
    ]
    values += [float(meta[name]) for name in (
        "template_id", "fdi", "quadrant", "tooth_index", "same_image_count",
        "raw_has_comma", "raw_has_semicolon", "raw_has_newline", "raw_length", "raw_cardinality",
    )]
    values += [float(meta["raw_form"] == form) for form in ("SINGLE", "COMMA", "SEMICOLON")]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (78,):
        raise RuntimeError(f"candidate ranker feature shape failure: {result.shape}")
    return result


def load_candidate_library(assets: AssetBundle) -> list[set[str]]:
    payload = json.loads(assets.path("multilabel_library").read_text(encoding="utf-8"))
    bank = [set(item["atoms"]) for item in payload.get("sets", [])]
    if len(bank) != 39 or set().union(*bank) != set(ATOMS):
        raise RuntimeError("multi-label candidate library mismatch")
    return bank


def make_multilabel_model() -> MultiLabelResidualProbabilityHead:
    """Compatibility factory for the paper-named residual probability head."""

    return MultiLabelResidualProbabilityHead()


def gfm_decode(probabilities: np.ndarray) -> set[str]:
    """Decode the fixed cardinality-aware F1 utility used by the method."""

    best_set: set[str] = set()
    best_value = 0.0
    best_cardinality = 0
    best_key: tuple[str, ...] = tuple()
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(ATOMS), len(CARDINALITIES)):
        raise ValueError(f"expected probability shape {(len(ATOMS), len(CARDINALITIES))}, got {values.shape}")
    for prediction_cardinality in range(1, len(ATOMS) + 1):
        atom_values = np.zeros(len(ATOMS), dtype=np.float64)
        for card_index, gold_cardinality in enumerate(CARDINALITIES):
            atom_values += (2.0 / (prediction_cardinality + gold_cardinality)) * values[:, card_index]
        indices = np.argsort(-atom_values, kind="stable")[:prediction_cardinality]
        candidate = {ATOMS[int(index)] for index in indices}
        value = float(atom_values[indices].sum())
        key = set_key(candidate)
        if value > best_value + 1e-15 or (
            abs(value - best_value) <= 1e-15
            and (
                prediction_cardinality < best_cardinality
                or (prediction_cardinality == best_cardinality and key < best_key)
            )
        ):
            best_set, best_value, best_cardinality, best_key = candidate, value, prediction_cardinality, key
    return best_set


def _load_stateful_asset(path: Path, device: str, factory: Any) -> Any:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    model = factory().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, payload


def run_multilabel(
    rows: list[dict[str, Any]],
    base_path: Path,
    adapter_path: Path,
    device: str,
    assets: AssetBundle,
    model_config: ModelConfig,
    mlc_config: MultilabelConfig,
    audit: dict[str, Any],
) -> dict[str, str]:
    if not rows:
        return {}
    import joblib
    import torch
    from catboost import CatBoostRanker, CatBoostRegressor

    template_payload = json.loads(assets.path("multilabel_templates").read_text(encoding="utf-8"))
    template_map = {str(key): int(value) for key, value in template_payload["templates"].items()}
    model, processor, model_audit = load_adapter_bundle(base_path, adapter_path, device, model_config)
    initial_generated_text_by_uid = {
        row["uid"]: generated_text(model, processor, row, device, model_config) for row in rows
    }
    answers = [serialize_effective({atom}) for atom in ATOMS]
    score_by_uid: dict[str, np.ndarray] = {}
    score_impl = ""
    for start in range(0, len(rows), mlc_config.scoring_row_batch_size):
        subset = rows[start:start + mlc_config.scoring_row_batch_size]
        scores, score_impl = score_teacher_forced_streaming(
            model, processor, subset, device, answers, model_config,
        )
        for row, values in zip(subset, scores):
            score_by_uid[row["uid"]] = values
    metas = mlc_metadata(rows, initial_generated_text_by_uid, template_map)
    bank = load_candidate_library(assets)
    selector = CatBoostRegressor()
    selector.load_model(str(assets.path("multilabel_selector")))
    ranker = CatBoostRanker()
    ranker.load_model(str(assets.path("multilabel_ranker")))
    feature_names = getattr(ranker, "feature_names_", None)
    if feature_names and len(feature_names) != 78:
        raise RuntimeError("multi-label candidate ranker feature count mismatch")
    probability_models = joblib.load(assets.path("multilabel_probability_models"))
    residual_head, residual_payload = _load_stateful_asset(
        assets.path("multilabel_residual_head"), device, MultiLabelResidualProbabilityHead,
    )

    feature_rows: list[np.ndarray] = []
    base_probability_rows: list[np.ndarray] = []
    for row, meta in zip(rows, metas):
        uid = row["uid"]
        initial_generated_set = parse_multilabel(initial_generated_text_by_uid[uid])
        if not initial_generated_set.issubset(set(ATOMS)):
            raise RuntimeError(
                f"generated multi-label output contains unknown atoms for {uid}: "
                f"{sorted(initial_generated_set - set(ATOMS))}"
            )
        score_map = {name: float(value) for name, value in zip(ATOMS, score_by_uid[uid])}
        candidates = build_initial_candidate_table(
            initial_generated_set, bank, limit=mlc_config.max_replacement_candidates,
        )
        selector_x = np.stack([
            build_candidate_selector_features(score_map, meta, initial_generated_set, candidate)
            for candidate in candidates
        ])
        selector_scores = np.asarray(selector.predict(selector_x), dtype=np.float64).reshape(-1)
        refined_candidate_set = candidates[
            select_refined_candidate(candidates, selector_scores, mlc_config.candidate_change_threshold)
        ]["set"]
        ranked = build_reranked_candidates(initial_generated_set, refined_candidate_set, bank)
        ranker_x = np.stack([
            build_candidate_ranker_features(
                score_map, meta, initial_generated_set, refined_candidate_set, candidate,
            )
            for candidate in ranked
        ])
        ranker_scores = np.asarray(ranker.predict(ranker_x), dtype=np.float64).reshape(-1)
        reranked_candidate_set = set(ranked[int(np.argmax(ranker_scores))]["set"])
        base_scores = np.zeros((len(ATOMS), len(CARDINALITIES)), dtype=np.float64)
        row_feature = build_probability_model_features(
            score_by_uid[uid], initial_generated_set, refined_candidate_set,
            reranked_candidate_set, row, meta["same_image_count"], initial_generated_text_by_uid[uid],
        )
        for atom_index in range(len(ATOMS)):
            for card_index in range(len(CARDINALITIES)):
                candidate_model = probability_models[(atom_index, card_index)]
                if np.isscalar(candidate_model):
                    base_scores[atom_index, card_index] = float(candidate_model)
                else:
                    base_scores[atom_index, card_index] = float(candidate_model.predict_proba(row_feature[None, :])[0, 1])
        feature_rows.append(row_feature)
        base_probability_rows.append(base_scores.reshape(-1))

    tokens = extract_image_tokens(model, processor, rows, device, model_config, prompt="spatial image feature")
    clear_model(model, processor)
    row_mean = np.asarray(residual_payload["row_scaler_mean"], dtype=np.float64)
    row_scale = np.asarray(residual_payload["row_scaler_scale"], dtype=np.float64)
    row_features = (np.asarray(feature_rows, dtype=np.float64) - row_mean) / row_scale
    probabilities = residual_head.predict(
        torch.from_numpy(tokens).to(device),
        torch.from_numpy(row_features.astype(np.float32)).to(device),
        np.asarray(base_probability_rows, dtype=np.float64),
    )
    outputs = {row["uid"]: serialize_effective(gfm_decode(probability)) for row, probability in zip(rows, probabilities)}
    audit.update({
        "model": model_audit,
        "rows": len(rows),
        "score_row_batch_size": mlc_config.scoring_row_batch_size,
        "score_contract": "row chunks of scoring_row_batch_size times ten parser-native singleton answers",
        "score_implementation": score_impl,
        "vocab_chunk_size": model_config.vocab_chunk_size,
        "changed_from_initial_generation": sum(
            parse_multilabel(initial_generated_text_by_uid[row["uid"]]) != parse_multilabel(outputs[row["uid"]])
            for row in rows
        ),
        "templates_unknown": sum(meta["template_id"] < 0 for meta in metas),
    })
    del residual_head, tokens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


# Compatibility aliases preserve the original public helper surface while the
# formal names above are used by the implementation and paper map.
candidate_probability_feature = build_probability_model_features
candidate_table = build_initial_candidate_table
pnss_features = build_candidate_selector_features
choose_pnss = select_refined_candidate
ranked_candidates = build_reranked_candidates
ranked_features = build_candidate_ranker_features


__all__ = [
    "ATOMS", "AUXILIARY_LABEL_ATOMS", "CARDINALITIES", "SEMANTIC_LABEL_ATOMS",
    "build_candidate_ranker_features", "build_candidate_selector_features",
    "build_initial_candidate_table", "build_probability_model_features",
    "build_reranked_candidates", "gfm_decode", "load_candidate_library",
    "make_multilabel_model", "parse_multilabel", "serialize_effective",
    "score_hidden_streaming", "score_logits_full", "score_teacher_forced_batch",
    "score_teacher_forced_streaming", "select_refined_candidate",
    # Legacy aliases are intentionally retained for serialized feature callers.
    "candidate_probability_feature", "candidate_table", "choose_pnss", "pnss_features",
    "ranked_candidates", "ranked_features", "SEMANTIC", "PSEUDO",
]
