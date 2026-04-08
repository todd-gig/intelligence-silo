"""Tests for the SLM matrix and model components."""

import torch
import pytest

from core.models.slm import SmallLanguageModel, SLMConfig, ModelType
from core.models.matrix import SLMMatrix
from core.models.router import ModelRouter, AttentionRouter


class TestSmallLanguageModel:
    def test_classifier(self):
        cfg = SLMConfig(name="test_cls", model_type=ModelType.SEQUENCE_CLASSIFICATION,
                        hidden_dim=64, num_layers=2, num_heads=2,
                        vocab_size=256, num_classes=8)
        model = SmallLanguageModel(cfg)
        x = torch.randint(0, 256, (2, 16))
        out = model(x)
        assert out.shape == (2, 8)

    def test_regression(self):
        cfg = SLMConfig(name="test_reg", model_type=ModelType.REGRESSION,
                        hidden_dim=64, num_layers=2, num_heads=2,
                        vocab_size=256, output_dim=4)
        model = SmallLanguageModel(cfg)
        x = torch.randint(0, 256, (2, 16))
        out = model(x)
        assert out.shape == (2, 4)

    def test_encoder(self):
        cfg = SLMConfig(name="test_enc", model_type=ModelType.ENCODER,
                        hidden_dim=64, num_layers=2, num_heads=2,
                        vocab_size=256)
        model = SmallLanguageModel(cfg)
        x = torch.randint(0, 256, (2, 16))
        out = model(x)
        assert out.shape == (2, 64)

    def test_generative(self):
        cfg = SLMConfig(name="test_gen", model_type=ModelType.GENERATIVE,
                        hidden_dim=64, num_layers=2, num_heads=2,
                        vocab_size=256)
        model = SmallLanguageModel(cfg)
        x = torch.randint(0, 256, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 256)

    def test_param_count(self):
        cfg = SLMConfig(name="test", model_type=ModelType.ENCODER,
                        hidden_dim=64, num_layers=2, num_heads=2)
        model = SmallLanguageModel(cfg)
        assert model.param_count() > 0
        assert isinstance(model.param_count_human(), str)


class TestModelRouter:
    def test_attention_routing(self):
        router = ModelRouter(
            model_names=["a", "b", "c"],
            query_dim=32,
            strategy="attention",
            confidence_floor=0.0,
        )
        query = torch.randn(32)
        results = router.route(query)
        assert len(results) > 0
        assert all(isinstance(r[1], float) for r in results)

    def test_round_robin(self):
        router = ModelRouter(
            model_names=["a", "b", "c"],
            query_dim=32,
            strategy="round_robin",
        )
        results = [router.route(torch.randn(32)) for _ in range(3)]
        names = [r[0][0] for r in results]
        assert names == ["a", "b", "c"]

    def test_fusion_weighted_vote(self):
        router = ModelRouter(model_names=["a", "b"], query_dim=32)
        outputs = [
            ("a", 0.7, torch.ones(4)),
            ("b", 0.3, torch.zeros(4)),
        ]
        fused = router.fuse_outputs(outputs)
        assert fused.shape == (4,)
        assert fused[0].item() > 0.5  # weighted toward model a


class TestSLMMatrix:
    def test_init_and_infer(self):
        configs = [
            {"name": "cls", "type": "sequence_classification",
             "hidden_dim": 64, "num_layers": 2, "num_heads": 2,
             "vocab_size": 256, "num_classes": 4},
            {"name": "reg", "type": "regression",
             "hidden_dim": 64, "num_layers": 2, "num_heads": 2,
             "vocab_size": 256, "output_dim": 3},
        ]
        matrix = SLMMatrix(configs, device="cpu")

        x = torch.randint(0, 256, (1, 16))
        result = matrix.infer(x)
        assert "fused" in result
        assert "individual" in result
        assert len(result["individual"]) == 2

    def test_targeted_inference(self):
        configs = [
            {"name": "a", "type": "encoder", "hidden_dim": 64,
             "num_layers": 2, "num_heads": 2, "vocab_size": 256},
            {"name": "b", "type": "encoder", "hidden_dim": 64,
             "num_layers": 2, "num_heads": 2, "vocab_size": 256},
        ]
        matrix = SLMMatrix(configs, device="cpu")

        x = torch.randint(0, 256, (1, 16))
        result = matrix.infer(x, target_models=["a"])
        assert "a" in result["individual"]
        assert "b" not in result["individual"]

    def test_performance_report(self):
        configs = [
            {"name": "test", "type": "encoder", "hidden_dim": 64,
             "num_layers": 2, "num_heads": 2, "vocab_size": 256},
        ]
        matrix = SLMMatrix(configs, device="cpu")
        report = matrix.performance_report()
        assert "test" in report
