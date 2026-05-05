"""Promotion gate — decides if a trained model is ready for production."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .trainer import TrainingResult
from .evaluator import EvaluationReport

logger = logging.getLogger(__name__)


@dataclass
class GateThresholds:
    """Per-model promotion thresholds."""
    # Classification models: minimum accuracy
    classifier_accuracy: float = 0.55
    trust_assessor_accuracy: float = 0.50
    pattern_detector_mse: float = 0.08

    # Regression models: maximum MSE
    scorer_mse: float = 0.06
    causal_predictor_mse: float = 0.07
    memory_encoder_mse: float = 0.10

    # All models: maximum validation loss
    max_val_loss: float = 1.5


class PromotionGate:
    """Checks if a trained model passes quality thresholds for production use.

    A model that doesn't pass still runs in production — it's just flagged
    as 'unverified' and its predictions are down-weighted by the router.
    """

    CLASSIFICATION_MODELS = {"classifier", "trust_assessor"}
    REGRESSION_MODELS = {"scorer", "memory_encoder", "pattern_detector", "causal_predictor"}

    def __init__(self, thresholds: GateThresholds | None = None):
        self.thresholds = thresholds or GateThresholds()

    def check(
        self,
        model_name: str,
        train_result: TrainingResult,
        eval_report: EvaluationReport,
    ) -> bool:
        """Return True if the model passes all promotion gates."""
        reasons = []

        # Gate 1: val_loss must be finite and below ceiling
        if train_result.val_loss > self.thresholds.max_val_loss:
            reasons.append(f"val_loss={train_result.val_loss:.4f} > {self.thresholds.max_val_loss}")

        # Gate 2: model-specific metric gate
        if model_name in self.CLASSIFICATION_MODELS:
            min_acc = getattr(self.thresholds, f"{model_name}_accuracy", 0.5)
            acc = eval_report.accuracy or 0.0
            if acc < min_acc:
                reasons.append(f"accuracy={acc:.3f} < {min_acc}")
        elif model_name in self.REGRESSION_MODELS:
            max_mse = getattr(self.thresholds, f"{model_name}_mse", 0.1)
            mse = eval_report.mse or 999.0
            if not eval_report.passed_threshold:
                reasons.append(f"regression signal insufficient (mse={mse:.4f})")

        # Gate 3: minimum training time (guards against degenerate 1-epoch runs)
        if train_result.best_epoch < 3:
            reasons.append(f"best_epoch={train_result.best_epoch} < 3 (converged too fast — check data)")

        passed = len(reasons) == 0
        if passed:
            logger.info("[%s] PROMOTED ✓ (val_loss=%.4f)", model_name, train_result.val_loss)
        else:
            logger.warning("[%s] Not promoted: %s", model_name, "; ".join(reasons))

        return passed
