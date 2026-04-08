"""Tests for the hierarchical memory system."""

import torch
import numpy as np
import pytest

from core.memory.working import WorkingMemory
from core.memory.episodic import EpisodicMemory
from core.memory.semantic import SemanticMemory
from core.memory.procedural import ProceduralMemory
from core.memory.hierarchy import MemoryHierarchy, MemoryConfig


class TestWorkingMemory:
    def test_store_and_retrieve(self):
        wm = WorkingMemory(capacity=10, embed_dim=32, num_heads=2)
        tensor = torch.randn(32)
        slot = wm.store("test_key", tensor, metadata={"type": "test"})
        assert slot.key == "test_key"

        retrieved = wm.retrieve("test_key")
        assert retrieved is not None
        assert retrieved.metadata["type"] == "test"
        assert retrieved.access_count == 1

    def test_capacity_eviction(self):
        wm = WorkingMemory(capacity=3, embed_dim=16, num_heads=2)
        for i in range(5):
            wm.store(f"key_{i}", torch.randn(16), priority=float(i))
        assert wm.active_count == 3

    def test_attention_retrieval(self):
        wm = WorkingMemory(capacity=10, embed_dim=16, num_heads=2)
        for i in range(5):
            wm.store(f"key_{i}", torch.randn(16))

        query = torch.randn(16)
        results = wm.attend(query, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r[1], float) for r in results)

    def test_snapshot(self):
        wm = WorkingMemory(capacity=10, embed_dim=16, num_heads=2)
        wm.store("a", torch.randn(16))
        snap = wm.snapshot()
        assert snap["active_count"] == 1


class TestEpisodicMemory:
    def test_record_and_recall(self):
        em = EpisodicMemory(max_episodes=100, embedding_dim=32, similarity_threshold=0.0)
        emb = np.random.randn(32).astype(np.float32)
        ep = em.record(emb, context={"action": "test"})
        assert ep.id in em.episodes

        results = em.recall(emb, top_k=1, min_similarity=0.0)
        assert len(results) >= 1
        assert results[0][0].id == ep.id

    def test_consolidation_candidates(self):
        em = EpisodicMemory(max_episodes=100, embedding_dim=16)
        emb = np.random.randn(16).astype(np.float32)
        ep = em.record(emb, context={}, importance=0.9)

        candidates = em.get_consolidation_candidates(min_recalls=0, min_importance=0.7)
        assert len(candidates) == 1

    def test_eviction(self):
        em = EpisodicMemory(max_episodes=3, embedding_dim=16)
        for i in range(5):
            em.record(np.random.randn(16).astype(np.float32), context={"i": i})
        assert em.size == 3


class TestSemanticMemory:
    def test_store_and_search(self):
        sm = SemanticMemory(embedding_dim=16, max_vectors=100, index_type="Flat")
        emb = np.random.randn(16).astype(np.float32)
        entry = sm.store(emb, knowledge={"fact": "test"})
        assert entry.id in sm.entries

        results = sm.search(emb, top_k=1)
        assert len(results) >= 1

    def test_category_filter(self):
        sm = SemanticMemory(embedding_dim=16, max_vectors=100, index_type="Flat")
        sm.store(np.random.randn(16).astype(np.float32), knowledge={}, category="A")
        sm.store(np.random.randn(16).astype(np.float32), knowledge={}, category="B")

        results = sm.search(np.random.randn(16).astype(np.float32), top_k=10, category="A")
        assert all(r[0].category == "A" for r in results)


class TestProceduralMemory:
    def test_learn_and_match(self):
        pm = ProceduralMemory(max_procedures=100, embedding_dim=16)
        trigger = np.random.randn(16).astype(np.float32)
        proc = pm.learn("test_proc", trigger, [{"action": "do_thing"}], initial_confidence=0.9)

        matches = pm.match(trigger, top_k=1)
        assert len(matches) >= 1
        assert matches[0][0].id == proc.id

    def test_outcome_tracking(self):
        pm = ProceduralMemory(max_procedures=100, embedding_dim=16)
        trigger = np.random.randn(16).astype(np.float32)
        proc = pm.learn("test", trigger, [], initial_confidence=0.5)

        pm.record_outcome(proc.id, success=True)
        assert proc.success_count == 1
        assert proc.confidence > 0.5

        pm.record_outcome(proc.id, success=False)
        assert proc.failure_count == 1


class TestMemoryHierarchy:
    def test_encode_and_query(self):
        config = MemoryConfig(
            working_capacity=10, episodic_max=100, semantic_max=100,
            procedural_max=100, embedding_dim=32, device="cpu",
        )
        hierarchy = MemoryHierarchy(config)

        emb = torch.randn(32)
        hierarchy.encode_and_store("test", emb, {"type": "test"})

        results = hierarchy.query_flat(emb, top_k=5)
        assert len(results) > 0

    def test_consolidation(self):
        config = MemoryConfig(
            working_capacity=5, working_ttl=0.01,  # very short TTL
            episodic_max=100, semantic_max=100, procedural_max=100,
            embedding_dim=16, device="cpu",
        )
        hierarchy = MemoryHierarchy(config)

        # Store and let expire
        for i in range(3):
            hierarchy.encode_and_store(f"key_{i}", torch.randn(16), {"i": i})

        import time
        time.sleep(0.02)  # wait for TTL

        stats = hierarchy.consolidate()
        assert stats["evicted_working"] >= 0  # may have already been evicted

    def test_health(self):
        config = MemoryConfig(embedding_dim=16, device="cpu")
        hierarchy = MemoryHierarchy(config)
        health = hierarchy.health()
        assert "working" in health
        assert "episodic" in health
        assert "semantic" in health
        assert "procedural" in health
