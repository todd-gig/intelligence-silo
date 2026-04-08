"""Memory Hierarchy — coordinates all four memory layers with automatic promotion."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import torch
import numpy as np

from .working import WorkingMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .procedural import ProceduralMemory

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for the full memory hierarchy."""
    # Working
    working_capacity: int = 128
    working_ttl: float = 300.0
    working_heads: int = 4
    # Episodic
    episodic_max: int = 10000
    episodic_similarity: float = 0.72
    episodic_consolidation: float = 60.0
    # Semantic
    semantic_max: int = 1_000_000
    semantic_index_type: str = "IVFFlat"
    semantic_nprobe: int = 16
    semantic_path: str | None = None
    # Procedural
    procedural_max: int = 5000
    procedural_threshold: float = 0.85
    procedural_lr: float = 0.001
    # Global
    embedding_dim: int = 384
    device: str = "cpu"


class MemoryHierarchy:
    """Unified interface to the four-layer memory system.

    Flow:
        Input → Working Memory (fast, attention-gated)
              ↓ eviction / consolidation
        Episodic Memory (experience-indexed)
              ↓ high recall / importance
        Semantic Memory (FAISS-indexed knowledge)

        Patterns → Procedural Memory (learned action sequences)

    The hierarchy automatically promotes information upward based on
    access patterns, importance, and consolidation heuristics.
    """

    def __init__(self, config: MemoryConfig | None = None):
        cfg = config or MemoryConfig()
        self.config = cfg

        self.working = WorkingMemory(
            capacity=cfg.working_capacity,
            embed_dim=cfg.embedding_dim,
            num_heads=cfg.working_heads,
            ttl_seconds=cfg.working_ttl,
            device=cfg.device,
        )
        self.episodic = EpisodicMemory(
            max_episodes=cfg.episodic_max,
            embedding_dim=cfg.embedding_dim,
            similarity_threshold=cfg.episodic_similarity,
            consolidation_interval=cfg.episodic_consolidation,
        )
        self.semantic = SemanticMemory(
            embedding_dim=cfg.embedding_dim,
            max_vectors=cfg.semantic_max,
            index_type=cfg.semantic_index_type,
            nprobe=cfg.semantic_nprobe,
            persistence_path=cfg.semantic_path,
        )
        self.procedural = ProceduralMemory(
            max_procedures=cfg.procedural_max,
            embedding_dim=cfg.embedding_dim,
            execution_threshold=cfg.procedural_threshold,
            learning_rate=cfg.procedural_lr,
            device=cfg.device,
        )

        self._last_consolidation = time.time()

    # ── Unified Operations ──────────────────────────────────────────────────

    def encode_and_store(self, key: str, embedding: torch.Tensor,
                         context: dict, priority: float = 1.0) -> None:
        """Store in working memory and record as episode simultaneously."""
        self.working.store(key, embedding, metadata=context, priority=priority)
        self.episodic.record(
            embedding=embedding,
            context=context,
            importance=min(priority / 2.0, 1.0),
        )

    def query(self, embedding: torch.Tensor, top_k: int = 5) -> dict:
        """Query all memory layers and return unified results.

        Returns a dict with results from each layer, sorted by relevance.
        """
        emb_np = embedding.detach().cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding

        results = {
            "working": self.working.attend(embedding, top_k=top_k),
            "episodic": self.episodic.recall(emb_np, top_k=top_k),
            "semantic": self.semantic.search(emb_np, top_k=top_k),
            "procedural": self.procedural.match(emb_np, top_k=min(top_k, 3)),
        }

        return results

    def query_flat(self, embedding: torch.Tensor, top_k: int = 10) -> list[dict]:
        """Query all layers and return a single ranked list."""
        raw = self.query(embedding, top_k=top_k)
        merged = []

        for key, score, tensor in raw["working"]:
            merged.append({
                "source": "working", "key": key, "score": score * 1.2,  # recency boost
                "data": self.working.slots[key].metadata if key in self.working.slots else {},
            })

        for episode, score in raw["episodic"]:
            merged.append({
                "source": "episodic", "key": episode.id, "score": score,
                "data": episode.context,
            })

        for entry, score in raw["semantic"]:
            merged.append({
                "source": "semantic", "key": entry.id, "score": score * 0.9,  # slight decay for old knowledge
                "data": entry.knowledge,
            })

        for proc, score in raw["procedural"]:
            merged.append({
                "source": "procedural", "key": proc.id, "score": score,
                "data": {"name": proc.name, "actions": proc.action_sequence},
            })

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    # ── Consolidation ───────────────────────────────────────────────────────

    def consolidate(self) -> dict:
        """Run a consolidation cycle: promote episodes → semantic, evict working memory.

        Returns stats about what was promoted/evicted.
        """
        stats = {"evicted_working": 0, "promoted_semantic": 0}

        # 1. Evict expired working memory → episodic
        evicted = self.working.evict_expired()
        for slot in evicted:
            emb_np = slot.tensor.detach().cpu().numpy()
            self.episodic.record(
                embedding=emb_np,
                context=slot.metadata,
                importance=slot.priority * 0.5,
            )
            stats["evicted_working"] += 1

        # 2. Promote high-value episodes → semantic
        candidates = self.episodic.get_consolidation_candidates()
        for episode in candidates:
            self.semantic.store(
                embedding=episode.embedding,
                knowledge=episode.context,
                source_episodes=[episode.id],
                category=episode.context.get("category", "general"),
                confidence=min(episode.importance + 0.1 * episode.recall_count, 1.0),
            )
            self.episodic.mark_consolidated(episode.id)
            stats["promoted_semantic"] += 1

        self._last_consolidation = time.time()
        logger.info(
            "Consolidation: evicted %d working → episodic, promoted %d episodic → semantic",
            stats["evicted_working"], stats["promoted_semantic"],
        )
        return stats

    def save(self) -> None:
        """Persist all persistable layers."""
        self.semantic.save()

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Full health report across all memory layers."""
        return {
            "working": {
                "active": self.working.active_count,
                "capacity": self.working.capacity,
                "utilization": f"{self.working.utilization:.1%}",
            },
            "episodic": self.episodic.snapshot(),
            "semantic": self.semantic.snapshot(),
            "procedural": self.procedural.snapshot(),
            "last_consolidation": time.time() - self._last_consolidation,
        }
