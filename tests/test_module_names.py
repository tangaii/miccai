from medical_parsing.module_names import PAPER_MODULE_KEYS, PAPER_MODULE_NAMES, PAPER_MODULES
from medical_parsing.models.detection_head import (
    FrozenSpatialQueryDecoder,
    load_frozen_spatial_query_decoder,
    make_frozen_spatial_query_decoder,
    SpatialDetectionHead,
    make_spatial_detection_head,
)


def test_paper_module_registry_is_ordered_and_complete():
    assert len(PAPER_MODULE_NAMES) == 5
    assert len(PAPER_MODULE_KEYS) == 5
    assert list(PAPER_MODULES) == list(PAPER_MODULE_KEYS)
    assert list(PAPER_MODULES.values()) == list(PAPER_MODULE_NAMES)
    assert PAPER_MODULE_NAMES == (
        "Shared MedGemma Representation Interface",
        "Task-Routed Classification",
        "Evidence-Guided Set Decoding",
        "Frozen Spatial Query Decoding",
        "Retrieval-Refined Quantile Regression",
    )


def test_detection_public_names_share_one_implementation():
    assert FrozenSpatialQueryDecoder is SpatialDetectionHead
    assert make_frozen_spatial_query_decoder is make_spatial_detection_head
    assert callable(load_frozen_spatial_query_decoder)
