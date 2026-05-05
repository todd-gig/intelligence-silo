"""SLM Trainer — per-model training loop with outcome-supervised labels."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from .data_generator import TrainingRecord, FEATURE_DIM, CLASSIFIER_CLASSES, TRUST_TIERS
from ..models.slm import SmallLanguageModel, SLMConfig, ModelType

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters for one training run."""
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_split: float = 0.15
    patience: int = 8           # early stopping patience
    min_delta: float = 1e-4     # minimum improvement to reset patience
    checkpoint_dir: str = "data/checkpoints"
    save_every_n_epochs: int = 10
    device: str = "auto"        # "auto" | "cpu" | "mps" | "cuda"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


@dataclass
class TrainingResult:
    """Outcome of training a single model."""
    model_name: str
    epochs_run: int
    train_loss: float
    val_loss: float
    val_accuracy: Optional[float]  # None for regression tasks
    best_epoch: int
    duration_seconds: float
    promoted: bool = False
    checkpoint_path: str = ""


class SLMTrainer:
    """Trains a single SLM specialist model given a list of TrainingRecords.

    Handles:
    - Adaptive loss functions per model type (CrossEntropy / MSE)
    - Sample weighting
    - Validation split and early stopping
    - Checkpoint saves to disk
    - MPS / CUDA / CPU device detection
    """

    LOSS_REGISTRY = {
        "classifier":      ("classification", CLASSIFIER_CLASSES),
        "trust_assessor":  ("classification", TRUST_TIERS),
        "scorer":          ("regression", 1),
        "memory_encoder":  ("reconstruction", FEATURE_DIM),
        "pattern_detector":("regression", 1),
        "causal_predictor":("regression", 1),
    }

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.device = self.config.resolve_device()
        logger.info("SLMTrainer initialized on device: %s", self.device)

    def train(
        self,
        model_name: str,
        slm: SmallLanguageModel,
        records: list[TrainingRecord],
    ) -> TrainingResult:
        """Train one SLM on the provided records. Returns a TrainingResult."""
        if not records:
            raise ValueError(f"No training records for model '{model_name}'")

        task_type, output_dim = self.LOSS_REGISTRY.get(model_name, ("regression", 1))
        slm = slm.to(self.device)

        # Build dataset
        X, y, w = self._build_tensors(records, task_type)
        dataset = TensorDataset(X, y, w)

        n_val = max(1, int(len(dataset) * self.config.validation_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size)

        # Build output head on top of SLM's encoder
        head = self._build_head(slm.config.hidden_dim, task_type, output_dim).to(self.device)
        params = list(slm.parameters()) + list(head.parameters())
        optimizer = optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        criterion = self._criterion(task_type)

        # Training loop
        best_val_loss = float("inf")
        best_epoch = 0
        patience_count = 0
        checkpoint_path = ""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        val_acc = None

        for epoch in range(1, self.config.epochs + 1):
            # --- Train ---
            slm.train()
            head.train()
            train_loss = 0.0
            for X_b, y_b, w_b in train_loader:
                X_b, y_b, w_b = X_b.to(self.device), y_b.to(self.device), w_b.to(self.device)
                optimizer.zero_grad()
                logits = head(self._encode(slm, X_b))
                loss = self._weighted_loss(criterion, logits, y_b, w_b, task_type)
                loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(1, len(train_loader))

            # --- Validate ---
            slm.eval()
            head.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_b, y_b, w_b in val_loader:
                    X_b, y_b, w_b = X_b.to(self.device), y_b.to(self.device), w_b.to(self.device)
                    logits = head(self._encode(slm, X_b))
                    loss = self._weighted_loss(criterion, logits, y_b, w_b, task_type)
                    val_loss += loss.item()
                    if task_type == "classification":
                        preds = logits.argmax(dim=-1)
                        correct += (preds == y_b).sum().item()
                        total += y_b.size(0)
            val_loss /= max(1, len(val_loader))
            if task_type == "classification" and total > 0:
                val_acc = correct / total

            scheduler.step()

            # Early stopping
            improvement = best_val_loss - val_loss
            if improvement > self.config.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_count = 0
                # Save best checkpoint
                checkpoint_path = str(ckpt_dir / f"{model_name}_best.pt")
                torch.save({
                    "model_state": slm.state_dict(),
                    "head_state": head.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, checkpoint_path)
            else:
                patience_count += 1

            if epoch % self.config.save_every_n_epochs == 0:
                logger.info(
                    "[%s] epoch %d/%d  train=%.4f  val=%.4f  acc=%s  patience=%d",
                    model_name, epoch, self.config.epochs,
                    train_loss, val_loss,
                    f"{val_acc:.3f}" if val_acc is not None else "n/a",
                    patience_count,
                )

            if patience_count >= self.config.patience:
                logger.info("[%s] Early stop at epoch %d (best: epoch %d)", model_name, epoch, best_epoch)
                break

        duration = time.time() - start
        logger.info(
            "[%s] Training complete: best_epoch=%d val_loss=%.4f acc=%s",
            model_name, best_epoch, best_val_loss,
            f"{val_acc:.3f}" if val_acc is not None else "n/a",
        )

        return TrainingResult(
            model_name=model_name,
            epochs_run=best_epoch,
            train_loss=train_loss,
            val_loss=best_val_loss,
            val_accuracy=val_acc,
            best_epoch=best_epoch,
            duration_seconds=duration,
            checkpoint_path=checkpoint_path,
        )

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _encode(self, slm: SmallLanguageModel, x: torch.Tensor) -> torch.Tensor:
        """Run x through the SLM encoder to get a hidden representation."""
        # Always co-locate x with the model — evaluator may pass CPU tensors
        # while the model lives on MPS or CUDA.
        model_device = next(slm.parameters()).device
        x = x.to(model_device)

        hidden = slm.config.hidden_dim
        if not hasattr(self, "_proj"):
            self._proj = {}
        key = (str(model_device), slm.config.name)
        if key not in self._proj:
            self._proj[key] = nn.Linear(FEATURE_DIM, hidden, bias=False).to(model_device)
        proj = self._proj[key]
        embedded = proj(x).unsqueeze(1)  # [batch, 1, hidden_dim]
        for block in slm.blocks:
            embedded = block(embedded)
        return embedded.squeeze(1)  # [batch, hidden_dim]

    def _build_head(self, hidden_dim: int, task_type: str, output_dim: int) -> nn.Module:
        if task_type == "classification":
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        elif task_type == "reconstruction":
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:  # regression
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

    def _criterion(self, task_type: str):
        if task_type == "classification":
            return nn.CrossEntropyLoss(reduction="none")
        return nn.MSELoss(reduction="none")

    def _weighted_loss(self, criterion, logits, y, w, task_type):
        loss = criterion(logits, y)
        if loss.dim() > 1:
            loss = loss.mean(dim=-1)
        return (loss * w).mean()

    def _build_tensors(self, records: list[TrainingRecord], task_type: str):
        X = torch.stack([r.input_tensor for r in records])
        w = torch.tensor([r.weight for r in records], dtype=torch.float32)

        if task_type == "classification":
            y = torch.stack([r.label for r in records]).long()
            if y.dim() > 1:
                y = y.squeeze(-1)
        else:
            y = torch.stack([
                r.label if r.label.dim() > 0 else r.label.unsqueeze(0)
                for r in records
            ])
            if y.dim() == 1:
                y = y.unsqueeze(-1)

        return X, y, w
