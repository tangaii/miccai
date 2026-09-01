import numpy as np

from medical_parsing.models.classification_head import SLOT_CLASSES, make_semantic_head
from medical_parsing.models.classification_head import semantic_concept


def test_semantic_head_shapes_and_mapping():
    import torch

    model = make_semantic_head().eval()
    tokens = torch.from_numpy(np.zeros((2, 256, 2560), dtype=np.float32))
    with torch.inference_mode():
        outputs = model.forward_source(tokens, "bone_marrow")
    assert outputs["bone_diagnosis"].shape == (2, len(SLOT_CLASSES["bone_diagnosis"]))
    assert semantic_concept("iugc", "iugc_standard_plane", "Standard plane", lambda value: str(value).lower()) == "standard_plane"
