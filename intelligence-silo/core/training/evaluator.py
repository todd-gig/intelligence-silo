"""Model evaluator — per-model accuracy metrics against held-out validation set."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

from .data_generator import TrainingRecord, FEATURE_DIM, CLASSIFIER_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    model_name: str
    n_samples: int
    task_type: str
    accuracy: Optional[float] = None          # classification only
    per_class_accuracy: Optional[dict] = None # classification only
    mse: Optional[float] = None               # regression only
    mae: Optional[float] = None               # regression only
    r2: Optional[float] = None                # regression only
    confusion_matrix: Optional[list] = None
    passed_threshold: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ModelEvaluator:
    """Evaluates a trained SLM on a set of records."""

    CLASSIFICATION_THRESHOLD = 0.55   # min accuracy to pass promotion
    REGRESSION_MSE_THRESHOLD = 0.05   # max MSE to pass promotion

    def evaluate(
        self,
        model_name: str,
        slm,
        records: list[TrainingRecord],
        device: Optional[torch.device] = None,
    ) -> EvaluationReport:
        if not records:
            return EvaluationReport(model_name=model_name, n_samples=0, task_type="unknown")

        from .trainer import SLMTrainer, TrainingConfig
        # Inherit device from model to avoid MPS↔CPU tensor mismatch
        model_device = next(slm.parameters()).device
        trainer = SLMTrainer(TrainingConfig(device=str(model_device)))
        task_type = trainer.LOSS_REGISTRY.get(model_name, ("regression", 1))[0]

        X = torch.stack([r.input_tensor for r in records])
        preds_list = []
        labels_list = []

        slm.eval()
        with torch.no_grad():
            # Process in batches
            batch_size = 64
            for i in range(0, len(X), batch_size):
                X_b = X[i:i+batch_size]
                encoded = trainer._encode(slm, X_b)
                preds_list.append(encoded.cpu())

        labels = [r.label for r in records]

        if task_type == "classification":
            return self._eval_classification(model_name, preds_list, labels, task_type)
        else:
            return self._eval_regression(model_name, preds_list, labels, task_type)

    def _eval_classification(self, name, encoded_list, labels, task_type) -> EvaluationReport:
        # Since we're evaluating the encoder output, use a simple nearest-centroid approach
        # to estimate separability without the full head
        X = torch.cat(encoded_list, dim=0).numpy()
        y = torch.cat([l.view(1) if l.dim() == 0 else l for l in labels], dim=0).long().numpy()

        # Compute per-class centroids from first half, test on second half
        n = len(X)
        X_train, X_test = X[:n//2], X[n//2:]
        y_train, y_test = y[:n//2], y[n//2:]

        centroids = {}
        for cls in np.unique(y_train):
            mask = y_train == cls
            centroids[int(cls)] = X_train[mask].mean(axis=0)

        correct = 0
        per_class = {}
        preds = []
        for i, (x, true) in enumerate(zip(X_test, y_test)):
            distances = {cls: np.linalg.norm(x - c) for cls, c in centroids.items()}
            pred = min(distances, key=distances.get)
            preds.append(pred)
            per_class.setdefault(int(true), []).append(int(pred) == int(true))
            if pred == true:
                correct += 1

        accuracy = correct / max(1, len(X_test))
        per_class_acc = {cls: sum(hits)/len(hits) for cls, hits in per_class.items()}

        return EvaluationReport(
            model_name=name,
            n_samples=n,
            task_type=task_type,
            accuracy=accuracy,
            per_class_accuracy=per_class_acc,
            passed_threshold=accuracy >= self.CLASSIFICATION_THRESHOLD,
        )

    def _eval_regression(self, name, encoded_list, labels, task_type) -> EvaluationReport:
        n = len(encoded_list[0]) if encoded_list else 0
        # For regression, measure variance in encoder output as proxy for signal quality
        X = torch.cat(encoded_list, dim=0).numpy()
        y_raw = [l.view(-1)[0].item() if l.numel() > 0 else 0.0 for l in labels]
        y = np.array(y_raw)

        # Variance-based signal detection: is there correlation between encoder norm and label?
        norms = np.linalg.norm(X, axis=-1)
        if norms.std() > 1e-8 and y.std() > 1e-8:
            corr = float(np.corrcoef(norms, y)[0, 1])
            r2 = corr ** 2
        else:
            r2 = 0.0

        # Naive MSE: compare norm-based prediction vs labels
        norms_norm = (norms - norms.mean()) / (norms.std() + 1e-8)
        y_norm = (y - y.mean()) / (y.std() + 1e-8)
        mse = float(np.mean((norms_norm - y_norm) ** 2))
        mae = float(np.mean(np.abs(norms_norm - y_norm)))

        return EvaluationReport(
            model_name=name,
            n_samples=len(X),
            task_type=task_type,
            mse=mse,
            mae=mae,
            r2=r2,
            passed_threshold=mse <= self.REGRESSION_MSE_THRESHOLD or r2 >= 0.3,
        )
