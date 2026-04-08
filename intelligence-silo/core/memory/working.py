"""Working Memory — fast, tensor-backed active context with attention-based relevance."""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemorySlot:
    """Single slot in working memory."""
    key: str
    tensor: torch.Tensor  # [embedding_dim]
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    priority: float = 1.0


class AttentionGate(nn.Module):
    """Multi-head attention gate for relevance scoring between query and memory slots."""

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score relevance of memory slots to a query.

        Args:
            query: [1, embed_dim] — the current focus
            keys: [N, embed_dim] — all active memory slots

        Returns:
            (attended_output, attention_weights)
        """
        q = query.unsqueeze(0) if query.dim() == 2 else query
        k = keys.unsqueeze(0) if keys.dim() == 2 else keys
        attn_out, attn_weights = self.attention(q, k, k)
        return self.norm(attn_out), attn_weights.squeeze(0)


class WorkingMemory:
    """Tensor-backed working memory with attention-gated access and TTL eviction.

    This is the "fast path" — high-bandwidth, low-latency context that the
    orchestrator uses for active reasoning. Evicted slots are candidates for
    episodic consolidation.
    """

    def __init__(self, capacity: int = 128, embed_dim: int = 384,
                 num_heads: int = 4, ttl_seconds: float = 300.0,
                 device: str = "cpu"):
        self.capacity = capacity
        self.embed_dim = embed_dim
        self.ttl = ttl_seconds
        self.device = torch.device(device)
        self.slots: dict[str, MemorySlot] = {}
        self.gate = AttentionGate(embed_dim, num_heads).to(self.device)
        self.gate.eval()

    def store(self, key: str, tensor: torch.Tensor, metadata: dict | None = None,
              priority: float = 1.0) -> MemorySlot:
        """Store a tensor in working memory. Evicts lowest-priority expired slot if full."""
        if tensor.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected dim {self.embed_dim}, got {tensor.shape[-1]}")

        tensor = tensor.detach().to(self.device)
        if tensor.dim() > 1:
            tensor = tensor.squeeze(0)

        # Evict if at capacity
        if len(self.slots) >= self.capacity and key not in self.slots:
            self._evict()

        slot = MemorySlot(
            key=key,
            tensor=tensor,
            metadata=metadata or {},
            priority=priority,
        )
        self.slots[key] = slot
        return slot

    def retrieve(self, key: str) -> MemorySlot | None:
        """Direct key-based retrieval."""
        slot = self.slots.get(key)
        if slot:
            slot.last_accessed = time.time()
            slot.access_count += 1
        return slot

    @torch.no_grad()
    def attend(self, query: torch.Tensor, top_k: int = 8) -> list[tuple[str, float, torch.Tensor]]:
        """Attention-based retrieval — returns top-k most relevant slots.

        Returns:
            List of (key, relevance_score, tensor) sorted by relevance descending.
        """
        if not self.slots:
            return []

        keys = list(self.slots.keys())
        tensors = torch.stack([self.slots[k].tensor for k in keys]).to(self.device)
        q = query.detach().to(self.device)
        if q.dim() == 1:
            q = q.unsqueeze(0)

        _, attn_weights = self.gate(q, tensors)
        # attn_weights: [1, N] or [num_heads, 1, N] — average over heads
        if attn_weights.dim() == 3:
            scores = attn_weights.mean(dim=0).squeeze(0)
        else:
            scores = attn_weights.squeeze(0)

        # Boost by priority and recency
        now = time.time()
        for i, k in enumerate(keys):
            slot = self.slots[k]
            recency = math.exp(-0.01 * (now - slot.last_accessed))
            scores[i] = scores[i] * slot.priority * (0.5 + 0.5 * recency)

        top_indices = scores.topk(min(top_k, len(keys))).indices
        results = []
        for idx in top_indices:
            k = keys[idx.item()]
            slot = self.slots[k]
            slot.last_accessed = now
            slot.access_count += 1
            results.append((k, scores[idx].item(), slot.tensor))

        return results

    def evict_expired(self) -> list[MemorySlot]:
        """Remove all expired slots. Returns evicted slots for consolidation."""
        now = time.time()
        expired = [
            slot for slot in self.slots.values()
            if (now - slot.last_accessed) > self.ttl
        ]
        for slot in expired:
            del self.slots[slot.key]
        return expired

    def _evict(self) -> MemorySlot | None:
        """Evict the lowest-value slot (expired first, then lowest priority * recency)."""
        now = time.time()

        # Prefer expired slots
        expired = [
            (k, s) for k, s in self.slots.items()
            if (now - s.last_accessed) > self.ttl
        ]
        if expired:
            key, slot = min(expired, key=lambda x: x[1].priority)
            del self.slots[key]
            return slot

        # Otherwise evict lowest composite score
        def score(s: MemorySlot) -> float:
            recency = math.exp(-0.01 * (now - s.last_accessed))
            return s.priority * recency * (1 + 0.1 * s.access_count)

        key = min(self.slots, key=lambda k: score(self.slots[k]))
        slot = self.slots.pop(key)
        return slot

    @property
    def active_count(self) -> int:
        return len(self.slots)

    @property
    def utilization(self) -> float:
        return len(self.slots) / self.capacity

    def snapshot(self) -> dict:
        """Serialize current state for persistence."""
        return {
            "slots": {
                k: {
                    "tensor": s.tensor.cpu().tolist(),
                    "metadata": s.metadata,
                    "priority": s.priority,
                    "access_count": s.access_count,
                }
                for k, s in self.slots.items()
            },
            "active_count": self.active_count,
            "utilization": self.utilization,
        }
