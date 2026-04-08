"""Procedural Memory — learned action sequences with confidence-gated auto-execution."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np


@dataclass
class Procedure:
    """A learned action sequence with execution confidence."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    trigger_embedding: np.ndarray | None = None  # [embed_dim]
    action_sequence: list[dict] = field(default_factory=list)
    confidence: float = 0.5
    success_count: int = 0
    failure_count: int = 0
    total_executions: int = 0
    created_at: float = field(default_factory=time.time)
    last_executed: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.success_count / self.total_executions


class ProcedureEncoder(nn.Module):
    """Encodes action sequences into dense trigger embeddings for matching."""

    def __init__(self, action_vocab_size: int = 256, embed_dim: int = 384,
                 hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.action_embed = nn.Embedding(action_vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.project = nn.Linear(hidden_dim, embed_dim)

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of action IDs into a trigger embedding.

        Args:
            action_ids: [batch, seq_len] integer action IDs

        Returns:
            [batch, embed_dim] trigger embedding
        """
        embedded = self.action_embed(action_ids)
        _, (hidden, _) = self.lstm(embedded)
        return self.project(hidden[-1])


class ProceduralMemory:
    """Learned action patterns with confidence-gated execution.

    Procedures are matched by comparing the current context embedding against
    stored trigger embeddings. High-confidence procedures can auto-execute.
    """

    def __init__(self, max_procedures: int = 5000, embedding_dim: int = 384,
                 execution_threshold: float = 0.85, learning_rate: float = 0.001,
                 device: str = "cpu"):
        self.max_procedures = max_procedures
        self.embedding_dim = embedding_dim
        self.execution_threshold = execution_threshold
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.procedures: dict[str, Procedure] = {}
        self.encoder = ProcedureEncoder(embed_dim=embedding_dim).to(self.device)
        self.encoder.eval()

    def learn(self, name: str, trigger_embedding: np.ndarray,
              action_sequence: list[dict], initial_confidence: float = 0.5) -> Procedure:
        """Learn a new procedure from an observed action sequence."""
        trigger = trigger_embedding.flatten().astype(np.float32)

        proc = Procedure(
            name=name,
            trigger_embedding=trigger,
            action_sequence=action_sequence,
            confidence=initial_confidence,
        )

        if len(self.procedures) >= self.max_procedures:
            self._evict()

        self.procedures[proc.id] = proc
        return proc

    def match(self, context_embedding: np.ndarray,
              top_k: int = 3) -> list[tuple[Procedure, float]]:
        """Find procedures matching the current context.

        Returns:
            List of (procedure, match_score) sorted by score descending.
        """
        if not self.procedures:
            return []

        context = context_embedding.flatten().astype(np.float32)
        context_norm = context / (np.linalg.norm(context) + 1e-8)

        results = []
        for proc in self.procedures.values():
            trigger_norm = proc.trigger_embedding / (
                np.linalg.norm(proc.trigger_embedding) + 1e-8
            )
            similarity = float(np.dot(context_norm, trigger_norm))
            # Weight by confidence and success rate
            score = similarity * proc.confidence * (0.5 + 0.5 * proc.success_rate)
            results.append((proc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def should_auto_execute(self, context_embedding: np.ndarray) -> Procedure | None:
        """Check if any procedure should auto-execute given current context.

        Returns the highest-scoring procedure above execution_threshold, or None.
        """
        matches = self.match(context_embedding, top_k=1)
        if matches and matches[0][1] >= self.execution_threshold:
            return matches[0][0]
        return None

    def record_outcome(self, procedure_id: str, success: bool) -> None:
        """Update a procedure's confidence based on execution outcome."""
        proc = self.procedures.get(procedure_id)
        if not proc:
            return

        proc.total_executions += 1
        proc.last_executed = time.time()
        if success:
            proc.success_count += 1
            proc.confidence = min(1.0, proc.confidence + self.learning_rate)
        else:
            proc.failure_count += 1
            proc.confidence = max(0.0, proc.confidence - self.learning_rate * 2)

    def _evict(self) -> None:
        """Remove the least valuable procedure."""
        def value(proc: Procedure) -> float:
            age = time.time() - proc.created_at
            return proc.confidence * proc.success_rate / (1 + 0.0001 * age)

        worst = min(self.procedures, key=lambda k: value(self.procedures[k]))
        del self.procedures[worst]

    @property
    def size(self) -> int:
        return len(self.procedures)

    @property
    def auto_executable_count(self) -> int:
        return sum(
            1 for p in self.procedures.values()
            if p.confidence >= self.execution_threshold
        )

    def snapshot(self) -> dict:
        return {
            "total_procedures": self.size,
            "auto_executable": self.auto_executable_count,
            "avg_confidence": (
                sum(p.confidence for p in self.procedures.values()) / max(self.size, 1)
            ),
            "avg_success_rate": (
                sum(p.success_rate for p in self.procedures.values()) / max(self.size, 1)
            ),
        }
