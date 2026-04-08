"""Small Language Model matrix — specialized micro-models for the society of minds."""

from .slm import SmallLanguageModel, ModelType
from .matrix import SLMMatrix
from .router import ModelRouter

__all__ = ["SmallLanguageModel", "ModelType", "SLMMatrix", "ModelRouter"]
