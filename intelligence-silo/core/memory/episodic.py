"""Episodic Memory — experience-based recall with temporal context and consolidation."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import torch
import numpy as np


@dataclass
class Episode:
    """A discrete experience stored in episodic memory."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    embedding: np.ndarray | None = None  # [embed_dim] as numpy for FAISS
    context: dict = field(default_factory=dict)
    outcome: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    recall_count: int = 0
    consolidated: bool = False  # promoted to semantic memory


class EpisodicMemory:
    """Experience store with similarity search and consolidation to semantic memory.

    Episodes are stored as embeddings and retrieved via cosine similarity.
    Frequently recalled or high-importance episodes are consolidated into
    semantic memory (long-term knowledge).
    """

    def __init__(self, max_episodes: int = 10000, embedding_dim: int = 384,
                 similarity_threshold: float = 0.72,
                 consolidation_interval: float = 60.0):
        self.max_episodes = max_episodes
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.consolidation_interval = consolidation_interval
        self.episodes: dict[str, Episode] = {}
        self._index_dirty = True
        self._embeddings_cache: np.ndarray | None = None
        self._keys_cache: list[str] = []
        self._last_consolidation = time.time()

    def record(self, embedding: torch.Tensor | np.ndarray, context: dict,
               outcome: dict | None = None, importance: float = 0.5) -> Episode:
        """Record a new episode."""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        embedding = embedding.flatten().astype(np.float32)

        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected dim {self.embedding_dim}, got {embedding.shape[0]}")

        episode = Episode(
            embedding=embedding,
            context=context,
            outcome=outcome or {},
            importance=importance,
        )

        # Evict if full — remove oldest, least-important, unconsolidated
        if len(self.episodes) >= self.max_episodes:
            self._evict()

        self.episodes[episode.id] = episode
        self._index_dirty = True
        return episode

    def recall(self, query: torch.Tensor | np.ndarray, top_k: int = 5,
               min_similarity: float | None = None) -> list[tuple[Episode, float]]:
        """Find most similar episodes to query embedding.

        Returns:
            List of (episode, similarity_score) sorted by similarity descending.
        """
        if not self.episodes:
            return []

        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        query = query.flatten().astype(np.float32)

        threshold = min_similarity or self.similarity_threshold
        self._rebuild_index()

        # Cosine similarity via normalized dot product
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        emb_norms = self._embeddings_cache / (
            np.linalg.norm(self._embeddings_cache, axis=1, keepdims=True) + 1e-8
        )
        similarities = emb_norms @ query_norm

        # Filter by threshold and sort
        valid_indices = np.where(similarities >= threshold)[0]
        if len(valid_indices) == 0:
            # Return top-k regardless if nothing meets threshold
            valid_indices = np.argsort(similarities)[-top_k:]

        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]][:top_k]

        results = []
        for idx in sorted_indices:
            key = self._keys_cache[idx]
            ep = self.episodes[key]
            ep.recall_count += 1
            results.append((ep, float(similarities[idx])))

        return results

    def get_consolidation_candidates(self, min_recalls: int = 3,
                                     min_importance: float = 0.7) -> list[Episode]:
        """Find episodes ready for promotion to semantic memory."""
        return [
            ep for ep in self.episodes.values()
            if not ep.consolidated
            and (ep.recall_count >= min_recalls or ep.importance >= min_importance)
        ]

    def mark_consolidated(self, episode_id: str) -> None:
        """Mark an episode as consolidated into semantic memory."""
        if episode_id in self.episodes:
            self.episodes[episode_id].consolidated = True

    def _rebuild_index(self) -> None:
        """Rebuild the numpy index cache if dirty."""
        if not self._index_dirty and self._embeddings_cache is not None:
            return
        self._keys_cache = list(self.episodes.keys())
        self._embeddings_cache = np.stack(
            [self.episodes[k].embedding for k in self._keys_cache]
        )
        self._index_dirty = False

    def _evict(self) -> None:
        """Remove the least valuable unconsolidated episode."""
        candidates = [
            (k, ep) for k, ep in self.episodes.items()
            if not ep.consolidated
        ]
        if not candidates:
            candidates = list(self.episodes.items())

        # Score: lower is more evictable
        def value(item: tuple[str, Episode]) -> float:
            _, ep = item
            age = time.time() - ep.timestamp
            return ep.importance * (1 + 0.1 * ep.recall_count) / (1 + 0.001 * age)

        evict_key = min(candidates, key=value)[0]
        del self.episodes[evict_key]
        self._index_dirty = True

    @property
    def size(self) -> int:
        return len(self.episodes)

    def snapshot(self) -> dict:
        return {
            "episode_count": self.size,
            "consolidated_count": sum(1 for e in self.episodes.values() if e.consolidated),
            "avg_importance": (
                sum(e.importance for e in self.episodes.values()) / max(self.size, 1)
            ),
        }
