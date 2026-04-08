"""Model Router — attention-based routing of queries to specialist SLMs."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionMethod(Enum):
    WEIGHTED_VOTE = "weighted_vote"
    CASCADE = "cascade"
    ENSEMBLE = "ensemble"


class RoutingStrategy(Enum):
    ATTENTION = "attention"
    ROUND_ROBIN = "round_robin"
    SPECIALIST = "specialist"


class AttentionRouter(nn.Module):
    """Learns to route queries to the most relevant specialist models.

    Uses a lightweight attention mechanism where each model has a learned
    "expertise embedding" and queries are scored against all experts.
    """

    def __init__(self, query_dim: int, num_models: int, temperature: float = 1.0):
        super().__init__()
        self.num_models = num_models
        self.temperature = temperature
        # Each model gets a learned expertise vector
        self.expertise = nn.Parameter(torch.randn(num_models, query_dim) * 0.02)
        self.query_proj = nn.Linear(query_dim, query_dim)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Compute routing weights for each model.

        Args:
            query: [batch, query_dim]

        Returns:
            [batch, num_models] routing weights (softmax-normalized)
        """
        q = self.query_proj(query)  # [B, D]
        scores = q @ self.expertise.T  # [B, num_models]
        return F.softmax(scores / self.temperature, dim=-1)


class ModelRouter:
    """Routes queries to specialist SLMs and fuses their outputs.

    Supports three routing strategies:
    - attention: learned routing based on query-expertise similarity
    - round_robin: distribute evenly across models
    - specialist: hard routing to single best model

    And three fusion methods:
    - weighted_vote: weight outputs by routing confidence
    - cascade: run models in sequence, stop when confident
    - ensemble: average all outputs equally
    """

    def __init__(self, model_names: list[str], query_dim: int = 384,
                 strategy: str = "attention", fusion: str = "weighted_vote",
                 confidence_floor: float = 0.6, device: str = "cpu"):
        self.model_names = model_names
        self.strategy = RoutingStrategy(strategy)
        self.fusion = FusionMethod(fusion)
        self.confidence_floor = confidence_floor
        self.device = torch.device(device)

        if self.strategy == RoutingStrategy.ATTENTION:
            self.attention_router = AttentionRouter(
                query_dim, len(model_names)
            ).to(self.device)
            self.attention_router.eval()
        else:
            self.attention_router = None

        self._round_robin_idx = 0

    @torch.no_grad()
    def route(self, query_embedding: torch.Tensor) -> list[tuple[str, float]]:
        """Determine which models to invoke and their weights.

        Returns:
            List of (model_name, weight) sorted by weight descending.
            Models below confidence_floor are filtered out (unless none pass).
        """
        if self.strategy == RoutingStrategy.ATTENTION:
            q = query_embedding.to(self.device)
            if q.dim() == 1:
                q = q.unsqueeze(0)
            weights = self.attention_router(q).squeeze(0)
            results = list(zip(self.model_names, weights.cpu().tolist()))
            results.sort(key=lambda x: x[1], reverse=True)

            # Filter by floor, but keep at least one
            filtered = [(n, w) for n, w in results if w >= self.confidence_floor]
            return filtered if filtered else [results[0]]

        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            idx = self._round_robin_idx % len(self.model_names)
            self._round_robin_idx += 1
            return [(self.model_names[idx], 1.0)]

        elif self.strategy == RoutingStrategy.SPECIALIST:
            # Use cosine similarity to pick best single model
            if self.attention_router:
                q = query_embedding.to(self.device).unsqueeze(0)
                weights = self.attention_router(q).squeeze(0)
                best_idx = weights.argmax().item()
                return [(self.model_names[best_idx], 1.0)]
            return [(self.model_names[0], 1.0)]

        return [(self.model_names[0], 1.0)]

    def fuse_outputs(self, outputs: list[tuple[str, float, torch.Tensor]]) -> torch.Tensor:
        """Fuse outputs from multiple models according to fusion strategy.

        Args:
            outputs: List of (model_name, weight, output_tensor)

        Returns:
            Fused output tensor.
        """
        if not outputs:
            raise ValueError("No outputs to fuse")

        if len(outputs) == 1:
            return outputs[0][2]

        if self.fusion == FusionMethod.WEIGHTED_VOTE:
            total_weight = sum(w for _, w, _ in outputs)
            fused = sum(w / total_weight * out for _, w, out in outputs)
            return fused

        elif self.fusion == FusionMethod.CASCADE:
            # Return first output with confidence above floor
            for _, w, out in outputs:
                if w >= self.confidence_floor:
                    return out
            return outputs[0][2]

        elif self.fusion == FusionMethod.ENSEMBLE:
            return torch.stack([out for _, _, out in outputs]).mean(dim=0)

        return outputs[0][2]
