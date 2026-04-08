"""Semantic Memory — long-term knowledge store with FAISS vector indexing."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class SemanticEntry:
    """A consolidated knowledge unit in semantic memory."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    embedding: np.ndarray | None = None
    knowledge: dict = field(default_factory=dict)  # structured knowledge payload
    source_episodes: list[str] = field(default_factory=list)
    category: str = "general"
    confidence: float = 0.8
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


class SemanticMemory:
    """FAISS-indexed long-term knowledge store.

    Knowledge is organized as dense vectors with metadata. Supports:
    - Fast approximate nearest-neighbor search (IVFFlat)
    - Persistence to disk (FAISS index + metadata JSON)
    - Category-filtered retrieval
    - Confidence-weighted results
    """

    def __init__(self, embedding_dim: int = 384, max_vectors: int = 1_000_000,
                 index_type: str = "IVFFlat", nprobe: int = 16,
                 persistence_path: str | None = None):
        self.embedding_dim = embedding_dim
        self.max_vectors = max_vectors
        self.nprobe = nprobe
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.entries: dict[str, SemanticEntry] = {}
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._next_idx = 0

        if HAS_FAISS:
            if index_type == "IVFFlat" and max_vectors > 1000:
                nlist = min(int(max_vectors ** 0.5), 4096)
                quantizer = faiss.IndexFlatIP(embedding_dim)
                self._index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
                self._index.nprobe = nprobe
                self._trained = False
            else:
                self._index = faiss.IndexFlatIP(embedding_dim)
                self._trained = True
        else:
            # Fallback: brute-force numpy
            self._index = None
            self._trained = True
            self._vectors: list[np.ndarray] = []

        if self.persistence_path:
            self._load()

    def store(self, embedding: np.ndarray, knowledge: dict,
              source_episodes: list[str] | None = None,
              category: str = "general", confidence: float = 0.8) -> SemanticEntry:
        """Add a knowledge entry to semantic memory."""
        embedding = embedding.flatten().astype(np.float32)
        # L2 normalize for inner-product search
        norm = np.linalg.norm(embedding) + 1e-8
        embedding = embedding / norm

        entry = SemanticEntry(
            embedding=embedding,
            knowledge=knowledge,
            source_episodes=source_episodes or [],
            category=category,
            confidence=confidence,
        )

        self.entries[entry.id] = entry
        idx = self._next_idx
        self._id_to_idx[entry.id] = idx
        self._idx_to_id[idx] = entry.id
        self._next_idx += 1

        if self._index is not None:
            vec = embedding.reshape(1, -1)
            if not self._trained:
                # Collect vectors for training
                if self._next_idx >= 256:
                    self._train_index()
                    self._index.add(vec)
                # Vectors will be batch-added after training
            else:
                self._index.add(vec)
        else:
            self._vectors.append(embedding)

        return entry

    def search(self, query: np.ndarray, top_k: int = 10,
               category: str | None = None,
               min_confidence: float = 0.0) -> list[tuple[SemanticEntry, float]]:
        """Search semantic memory for relevant knowledge.

        Returns:
            List of (entry, score) sorted by relevance.
        """
        if not self.entries:
            return []

        query = query.flatten().astype(np.float32)
        norm = np.linalg.norm(query) + 1e-8
        query = query / norm

        if self._index is not None and self._trained:
            scores, indices = self._index.search(query.reshape(1, -1), min(top_k * 3, self._next_idx))
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy fallback
            if not self._vectors and self._index is None:
                return []
            vecs = self._vectors if self._index is None else []
            if not vecs:
                return []
            mat = np.stack(vecs)
            scores = mat @ query
            indices = np.argsort(scores)[::-1][:top_k * 3]
            scores = scores[indices]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx not in self._idx_to_id:
                continue
            entry_id = self._idx_to_id[int(idx)]
            entry = self.entries.get(entry_id)
            if entry is None:
                continue
            if category and entry.category != category:
                continue
            if entry.confidence < min_confidence:
                continue
            entry.access_count += 1
            results.append((entry, float(score)))
            if len(results) >= top_k:
                break

        return results

    def _train_index(self) -> None:
        """Train IVF index on accumulated vectors."""
        if self._index is None or self._trained:
            return
        all_vecs = np.stack([
            self.entries[self._idx_to_id[i]].embedding
            for i in range(self._next_idx)
            if i in self._idx_to_id
        ]).astype(np.float32)
        self._index.train(all_vecs)
        self._index.add(all_vecs)
        self._trained = True

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if not self.persistence_path:
            return
        self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta = {}
        for eid, entry in self.entries.items():
            meta[eid] = {
                "knowledge": entry.knowledge,
                "source_episodes": entry.source_episodes,
                "category": entry.category,
                "confidence": entry.confidence,
                "created_at": entry.created_at,
                "access_count": entry.access_count,
            }
        with open(self.persistence_path / "metadata.json", "w") as f:
            json.dump(meta, f)

        # Save FAISS index
        if self._index is not None and HAS_FAISS:
            faiss.write_index(self._index, str(self.persistence_path / "index.faiss"))

        # Save embeddings as numpy backup
        if self.entries:
            ids = list(self.entries.keys())
            embeddings = np.stack([self.entries[eid].embedding for eid in ids])
            np.save(self.persistence_path / "embeddings.npy", embeddings)
            with open(self.persistence_path / "id_order.json", "w") as f:
                json.dump(ids, f)

    def _load(self) -> None:
        """Load persisted state from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        meta_path = self.persistence_path / "metadata.json"
        emb_path = self.persistence_path / "embeddings.npy"
        ids_path = self.persistence_path / "id_order.json"

        if not meta_path.exists() or not emb_path.exists():
            return

        with open(meta_path) as f:
            meta = json.load(f)
        embeddings = np.load(emb_path)
        with open(ids_path) as f:
            ids = json.load(f)

        for i, eid in enumerate(ids):
            if eid in meta:
                m = meta[eid]
                entry = SemanticEntry(
                    id=eid,
                    embedding=embeddings[i].astype(np.float32),
                    knowledge=m["knowledge"],
                    source_episodes=m.get("source_episodes", []),
                    category=m.get("category", "general"),
                    confidence=m.get("confidence", 0.8),
                    created_at=m.get("created_at", 0),
                    access_count=m.get("access_count", 0),
                )
                self.entries[eid] = entry
                self._id_to_idx[eid] = self._next_idx
                self._idx_to_id[self._next_idx] = eid
                self._next_idx += 1

        # Rebuild FAISS index
        if self._index is not None and HAS_FAISS and self.entries:
            index_path = self.persistence_path / "index.faiss"
            if index_path.exists():
                self._index = faiss.read_index(str(index_path))
                self._trained = True
            else:
                self._train_index()

    @property
    def size(self) -> int:
        return len(self.entries)

    def snapshot(self) -> dict:
        return {
            "total_entries": self.size,
            "categories": list(set(e.category for e in self.entries.values())),
            "avg_confidence": (
                sum(e.confidence for e in self.entries.values()) / max(self.size, 1)
            ),
            "faiss_available": HAS_FAISS,
            "index_trained": self._trained,
        }
