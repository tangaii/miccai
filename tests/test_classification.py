import json

import numpy as np

from medical_parsing.models.classification_head import (
    SLOT_CLASSES,
    SemanticImageTokenHead,
    make_semantic_head,
    map_option_to_semantic_concept,
)
from medical_parsing.tasks.classification import (
    DIRECT_PROMPT_ROUTE,
    INSTRUCTIONAL_FALLBACK_ROUTE,
    SEMANTIC_HEAD_ROUTE,
    load_route_manifest,
    route_for_row,
)


def test_semantic_head_shapes_and_mapping():
    import torch

    model = make_semantic_head().eval()
    tokens = torch.from_numpy(np.zeros((2, 256, 2560), dtype=np.float32))
    with torch.inference_mode():
        outputs = model.forward_source(tokens, "bone_marrow")
    assert outputs["bone_diagnosis"].shape == (2, len(SLOT_CLASSES["bone_diagnosis"]))
    assert isinstance(model, SemanticImageTokenHead)
    assert map_option_to_semantic_concept("iugc", "iugc_standard_plane", "Standard plane", lambda value: str(value).lower()) == "standard_plane"


def test_classification_route_aliases_and_dental_inactive_semantic_contract(tmp_path):
    manifest = tmp_path / "routes.json"
    manifest.write_text(json.dumps({"routes": {"dental::question": "semantic", "dental::prompt": "prompt", "dental::fallback": "fallback"}}), encoding="utf-8")
    from medical_parsing.config import AssetBundle

    assets = AssetBundle(tmp_path)
    (tmp_path / "classification_route_manifest.json").write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    routes = load_route_manifest(assets)
    assert routes["dental::question"] == SEMANTIC_HEAD_ROUTE
    assert route_for_row({"uid": "q", "dataset": "dental", "source_question": "question"}, routes) == SEMANTIC_HEAD_ROUTE
    assert route_for_row({"uid": "p", "dataset": "dental", "source_question": "prompt"}, routes) == DIRECT_PROMPT_ROUTE
    assert route_for_row({"uid": "f", "dataset": "dental", "source_question": "fallback"}, routes) == INSTRUCTIONAL_FALLBACK_ROUTE
