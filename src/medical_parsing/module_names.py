"""Canonical names for the five paper-level method modules.

These names are documentation metadata shared by the public code and the
manuscript. Runtime task names and CLI component identifiers remain stable;
checkpoint filenames and compatibility aliases follow the interfaces documented
by the public release.
"""

PAPER_MODULE_NAMES = (
    "Shared MedGemma Representation Interface",
    "Task-Routed Classification",
    "Evidence-Guided Set Decoding",
    "Frozen Spatial Query Decoding",
    "Retrieval-Refined Quantile Regression",
)

PAPER_MODULE_KEYS = (
    "shared_representation",
    "task_routed_classification",
    "evidence_guided_set_decoding",
    "frozen_spatial_query_decoding",
    "retrieval_refined_quantile_regression",
)

if len(PAPER_MODULE_NAMES) != 5 or len(PAPER_MODULE_KEYS) != len(PAPER_MODULE_NAMES):
    raise RuntimeError("the paper module registry must contain exactly five modules")

PAPER_MODULES = dict(zip(PAPER_MODULE_KEYS, PAPER_MODULE_NAMES))

__all__ = ["PAPER_MODULE_KEYS", "PAPER_MODULE_NAMES", "PAPER_MODULES"]
