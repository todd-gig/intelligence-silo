"""SLM Training Pipeline — converts decision history into trained model weights.

The 6 specialist SLMs start with random weights. This package provides:
1. Data generation: synthetic + historical decision records → training tensors
2. Training loop: per-model training with outcome-supervised labels
3. Evaluation: per-model accuracy metrics against held-out validation set
4. Promotion gates: confidence threshold checks before a model is promoted to production

Usage:
    from core.training import TrainingPipeline
    pipeline = TrainingPipeline()
    pipeline.run(epochs=50, save_checkpoints=True)
"""

from .data_generator import DecisionDataGenerator, TrainingRecord
from .trainer import SLMTrainer, TrainingConfig
from .pipeline import TrainingPipeline
from .evaluator import ModelEvaluator, EvaluationReport
from .promotion import PromotionGate

__all__ = [
    "DecisionDataGenerator",
    "TrainingRecord",
    "SLMTrainer",
    "TrainingConfig",
    "TrainingPipeline",
    "ModelEvaluator",
    "EvaluationReport",
    "PromotionGate",
]
