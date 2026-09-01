"""Neural model building blocks used by the task branches."""
"""Paper-locatable neural modules used by the three task branches."""

from .classification_head import SemanticImageTokenHead
from .multilabel_head import MultiLabelResidualProbabilityHead
from .regression_head import SpatialQuantileRefinementHead

__all__ = [
    "MultiLabelResidualProbabilityHead", "SemanticImageTokenHead",
    "SpatialQuantileRefinementHead",
]
