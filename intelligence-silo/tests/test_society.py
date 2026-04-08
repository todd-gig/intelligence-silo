"""Tests for the Society of Minds orchestrator."""

import torch
import pytest

from core.models.matrix import SLMMatrix
from core.memory.hierarchy import MemoryHierarchy, MemoryConfig
from core.orchestrator.society import SocietyOfMinds
from core.orchestrator.mind import Mind, MindRole, Signal


def make_test_society() -> SocietyOfMinds:
    """Create a minimal society for testing."""
    configs = [
        {"name": "classifier", "type": "sequence_classification",
         "hidden_dim": 64, "num_layers": 2, "num_heads": 2,
         "vocab_size": 256, "num_classes": 4},
        {"name": "scorer", "type": "regression",
         "hidden_dim": 64, "num_layers": 2, "num_heads": 2,
         "vocab_size": 256, "output_dim": 4},
    ]
    matrix = SLMMatrix(configs, device="cpu")
    memory = MemoryHierarchy(MemoryConfig(embedding_dim=64, device="cpu"))
    return SocietyOfMinds(matrix, memory)


class TestMind:
    def test_perceiver(self):
        mind = Mind("test_perceiver", MindRole.PERCEIVER, ["classifier"])
        signal = Signal(source="test", content={"data": "hello"})
        mind.receive(signal)
        response = mind.process()
        assert response is not None
        assert response.signal_type == "observation"

    def test_critic(self):
        mind = Mind("test_critic", MindRole.CRITIC, ["classifier"])
        signal = Signal(source="analyst", content={}, confidence=0.5, signal_type="proposal")
        mind.receive(signal)
        response = mind.process()
        assert response is not None
        assert response.signal_type == "critique"

    def test_synthesizer(self):
        mind = Mind("test_synth", MindRole.SYNTHESIZER, ["scorer"])
        proposal = Signal(source="a", content={}, confidence=0.8, signal_type="proposal")
        critique = Signal(source="b", content={"approval": True}, signal_type="critique")
        mind.receive(proposal)
        mind.receive(critique)
        response = mind.process()
        assert response is not None
        assert response.signal_type == "verdict"


class TestSocietyOfMinds:
    def test_think_cycle(self):
        society = make_test_society()
        input_data = {"type": "test", "value": 42}
        input_ids = torch.randint(0, 256, (1, 16))
        query_emb = torch.randn(1, 64)

        result = society.think(input_data, input_ids, query_emb)
        assert result.minds_activated > 0
        assert result.cycle_time_ms > 0
        assert len(result.signals) > 0

    def test_deliberation(self):
        society = make_test_society()
        input_data = {"type": "high_stakes_decision"}
        input_ids = torch.randint(0, 256, (1, 16))

        result = society.deliberate(input_data, input_ids)
        assert result.minds_activated > 0

    def test_health(self):
        society = make_test_society()
        health = society.health()
        assert "minds" in health
        assert "total_signals" in health
        assert "matrix" in health
        assert "memory" in health

    def test_default_minds_created(self):
        society = make_test_society()
        roles = {m.role for m in society.minds.values()}
        assert MindRole.PERCEIVER in roles
        assert MindRole.ANALYST in roles
        assert MindRole.CRITIC in roles
        assert MindRole.SYNTHESIZER in roles
        assert MindRole.SENTINEL in roles
