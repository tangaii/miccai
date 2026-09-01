import numpy as np

from medical_parsing.tasks.multilabel import (
    ATOMS,
    apply_tooth_position_aware_correction,
    build_candidate_ranker_features,
    build_candidate_selector_features,
    build_initial_candidate_table,
    build_probability_model_features,
    build_reranked_candidates,
    gfm_decode,
    parse_multilabel,
    select_refined_candidate,
    serialize_effective,
)


def test_multilabel_roundtrip_and_candidate_shapes():
    value = {ATOMS[2], ATOMS[7]}
    assert parse_multilabel(serialize_effective(value)) == value
    bank = [{ATOMS[2]}, {ATOMS[7]}, set(ATOMS[:2])]
    candidates = build_initial_candidate_table(value, bank)
    assert candidates[0]["set"] == value
    ranked = build_reranked_candidates(value, value, bank)
    assert ranked[0]["set"] == value
    row = {"source_question": "report tooth 18", "dataset": "dental"}
    feature = build_probability_model_features(np.zeros((10,), dtype=np.float64), value, value, value, row, 1, "x;")
    assert feature.shape == (50,)

    meta = {
        "template_id": 0, "fdi": 18, "quadrant": 1, "tooth_index": 8,
        "same_image_count": 1, "raw_has_comma": 0, "raw_has_semicolon": 1,
        "raw_has_newline": 0, "raw_length": 2, "raw_cardinality": 1,
        "raw_form": "SEMICOLON",
    }
    scores = {atom: 0.0 for atom in [
        "dental caries", "periradicular lesions", "pulp diseases",
        "chronic injury of tooth hard tissues", "disturbances of eruption of teeth",
        "periodontal diseases", "endodontic treatment, restorative treatment, and complications",
        "and complications", "endodontic treatment", "restorative treatment",
    ]}
    selector_feature = build_candidate_selector_features(scores, meta, value, candidates[0])
    ranker_feature = build_candidate_ranker_features(scores, meta, value, value, ranked[0])
    assert selector_feature.shape == (65,)
    assert ranker_feature.shape == (78,)
    selector_scores = np.zeros(len(candidates), dtype=np.float64)
    selector_scores[0] = 0.0
    if len(candidates) > 1:
        selector_scores[1] = 0.05
    assert select_refined_candidate(candidates, selector_scores, 0.05) in {0, 1}


def test_gfm_decode_is_deterministic():
    probabilities = np.zeros((10, 4), dtype=np.float64)
    probabilities[2, 0] = 0.99
    assert gfm_decode(probabilities) == {ATOMS[2]}


def test_tooth_position_correction_matches_frozen_final_behavior():
    candidates = [
        {"set": {ATOMS[0]}, "action": "KEEP"},
        {"set": {ATOMS[1]}, "action": "ADD"},
    ]

    assert apply_tooth_position_aware_correction(candidates, 1, 8) == {ATOMS[0]}
    assert apply_tooth_position_aware_correction(candidates, 1, 7) == {ATOMS[1]}
    assert apply_tooth_position_aware_correction(candidates, 0, 8) == {ATOMS[0]}
