"""SLM Matrix — manages the ensemble of small language models as a unified intelligence."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from .slm import SmallLanguageModel, SLMConfig, ModelType
from .router import ModelRouter

logger = logging.getLogger(__name__)


class SLMMatrix:
    """Matrix of Small Language Models — the neural backbone of the intelligence silo.

    Each model is a specialist micro-expert. The matrix:
    1. Routes queries to the right expert(s) via learned attention
    2. Runs inference on selected models
    3. Fuses outputs into a unified prediction
    4. Tracks per-model performance for adaptive routing

    Design principle: local inference on small models > remote calls to large models
    for latency-critical, high-frequency decisions.
    """

    def __init__(self, configs: list[dict], router_strategy: str = "attention",
                 fusion_method: str = "weighted_vote", confidence_floor: float = 0.6,
                 device: str = "auto"):
        self.device = self._resolve_device(device)
        self.models: dict[str, SmallLanguageModel] = {}
        self.configs: dict[str, SLMConfig] = {}
        self._performance: dict[str, dict] = {}

        # Build models from config dicts
        for cfg_dict in configs:
            cfg = SLMConfig(
                name=cfg_dict["name"],
                model_type=ModelType(cfg_dict["type"]),
                hidden_dim=cfg_dict.get("hidden_dim", 256),
                num_layers=cfg_dict.get("num_layers", 4),
                num_heads=cfg_dict.get("num_heads", 4),
                vocab_size=cfg_dict.get("vocab_size", 8192),
                num_classes=cfg_dict.get("num_classes", 32),
                output_dim=cfg_dict.get("output_dim", 1),
                role=cfg_dict.get("role", ""),
            )
            model = SmallLanguageModel(cfg).to(self.device)
            model.eval()
            self.models[cfg.name] = model
            self.configs[cfg.name] = cfg
            self._performance[cfg.name] = {
                "calls": 0, "correct": 0, "total_latency_ms": 0.0,
            }
            logger.info(
                "Loaded SLM '%s' (%s) — %s params on %s",
                cfg.name, cfg.model_type.value, model.param_count_human(), self.device,
            )

        # Router
        query_dim = max(c.hidden_dim for c in self.configs.values())
        self.router = ModelRouter(
            model_names=list(self.models.keys()),
            query_dim=query_dim,
            strategy=router_strategy,
            fusion=fusion_method,
            confidence_floor=confidence_floor,
            device=str(self.device),
        )

    @torch.no_grad()
    def infer(self, input_ids: torch.Tensor,
              query_embedding: torch.Tensor | None = None,
              target_models: list[str] | None = None) -> dict:
        """Run inference through the matrix.

        Args:
            input_ids: [batch, seq_len] tokenized input
            query_embedding: optional embedding for routing (if None, routes to all)
            target_models: optional explicit model list (bypasses router)

        Returns:
            {
                "fused": Tensor — fused output from selected models,
                "individual": {model_name: output_tensor},
                "routing": [(model_name, weight)],
            }
        """
        input_ids = input_ids.to(self.device)

        # Determine which models to run
        if target_models:
            routing = [(name, 1.0 / len(target_models)) for name in target_models]
        elif query_embedding is not None:
            routing = self.router.route(query_embedding)
        else:
            routing = [(name, 1.0 / len(self.models)) for name in self.models]

        # Run selected models
        outputs = []
        individual = {}
        for name, weight in routing:
            model = self.models.get(name)
            if model is None:
                continue
            out = model(input_ids)
            individual[name] = out
            outputs.append((name, weight, out))
            self._performance[name]["calls"] += 1

        # Fuse
        fused = self.router.fuse_outputs(outputs) if outputs else None

        return {
            "fused": fused,
            "individual": individual,
            "routing": routing,
        }

    def record_feedback(self, model_name: str, correct: bool) -> None:
        """Record whether a model's prediction was correct (for adaptive routing)."""
        if model_name in self._performance:
            self._performance[model_name]["correct"] += int(correct)

    def save_weights(self, path: str | Path) -> None:
        """Save all model weights as safetensors (or fallback to torch.save)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file
            for name, model in self.models.items():
                tensors = {k: v for k, v in model.state_dict().items()}
                save_file(tensors, path / f"{name}.safetensors")
        except ImportError:
            for name, model in self.models.items():
                torch.save(model.state_dict(), path / f"{name}.pt")
        logger.info("Saved %d model weights to %s", len(self.models), path)

    def load_weights(self, path: str | Path) -> list[str]:
        """Load model weights from safetensors or .pt files."""
        path = Path(path)
        loaded = []
        for name, model in self.models.items():
            sf_file = path / f"{name}.safetensors"
            pt_file = path / f"{name}.pt"
            if sf_file.exists():
                try:
                    from safetensors.torch import load_file
                    tensors = load_file(str(sf_file))
                    model.load_state_dict(tensors)
                except ImportError:
                    continue
            elif pt_file.exists():
                model.load_state_dict(torch.load(pt_file, weights_only=True))
            else:
                continue
            model.eval()
            loaded.append(name)
            logger.info("Loaded weights for '%s'", name)
        return loaded

    def performance_report(self) -> dict:
        """Per-model performance stats."""
        report = {}
        for name, stats in self._performance.items():
            accuracy = (
                stats["correct"] / stats["calls"] if stats["calls"] > 0 else 0.0
            )
            report[name] = {
                "calls": stats["calls"],
                "accuracy": f"{accuracy:.1%}",
                "params": self.models[name].param_count_human(),
                "role": self.configs[name].role,
            }
        return report

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    @property
    def total_params(self) -> int:
        return sum(m.param_count() for m in self.models.values())

    @property
    def total_params_human(self) -> str:
        count = self.total_params
        if count >= 1e6:
            return f"{count / 1e6:.1f}M"
        return f"{count / 1e3:.0f}K"
