"""Intelligence dimensions — 10-pillar model of human wisdom for decision-making."""

from .dimensions import (
    IntelligenceDimension,
    IntelligenceProfile,
    DIMENSION_WEIGHTS,
    AUTO_EXEC_REQUIRED,
    AUTO_EXEC_THRESHOLD,
    infer_profile_from_decision,
)

__all__ = [
    "IntelligenceDimension",
    "IntelligenceProfile",
    "DIMENSION_WEIGHTS",
    "AUTO_EXEC_REQUIRED",
    "AUTO_EXEC_THRESHOLD",
    "infer_profile_from_decision",
]
