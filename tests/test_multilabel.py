import numpy as np

from medical_parsing.tasks.multilabel import (
    ATOMS,
    candidate_table,
    gfm_decode,
    parse_multilabel,
    ranked_candidates,
    candidate_probability_feature,
    serialize_effective,
)


def test_multilabel_roundtrip_and_candidate_shapes():
    value = {ATOMS[2], ATOMS[7]}
    assert parse_multilabel(serialize_effective(value)) == value
    bank = [{ATOMS[2]}, {ATOMS[7]}, set(ATOMS[:2])]
    candidates = candidate_table(value, bank)
    assert candidates[0]["set"] == value
    ranked = ranked_candidates(value, value, bank)
    assert ranked[0]["set"] == value
    row = {"source_question": "report tooth 18", "dataset": "dental"}
    feature = candidate_probability_feature(np.zeros((10,), dtype=np.float64), value, value, value, row, 1, "x;")
    assert feature.shape == (50,)


def test_gfm_decode_is_deterministic():
    probabilities = np.zeros((10, 4), dtype=np.float64)
    probabilities[2, 0] = 0.99
    assert gfm_decode(probabilities) == {ATOMS[2]}
