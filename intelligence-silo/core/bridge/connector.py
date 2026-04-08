"""Decision Engine Bridge — bidirectional sync between intelligence silo and decision engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx
import yaml
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EngineWeights:
    """Weights imported from the decision engine's engine.yaml."""
    value_weights: dict[str, float] = field(default_factory=dict)
    penalty_weights: dict[str, float] = field(default_factory=dict)
    trust_multiplier: dict[str, float] = field(default_factory=dict)
    rtql_multiplier: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    authority_matrix: dict = field(default_factory=dict)

    def to_tensor(self) -> torch.Tensor:
        """Convert value + penalty weights to a single tensor for neural scoring."""
        values = list(self.value_weights.values())
        penalties = list(self.penalty_weights.values())
        return torch.tensor(values + penalties, dtype=torch.float32)


@dataclass
class NeuralPrediction:
    """A prediction from the intelligence silo to send to the decision engine."""
    decision_id: str = ""
    predicted_scores: dict[str, float] = field(default_factory=dict)
    predicted_trust_tier: str = "T0"
    confidence: float = 0.0
    causal_chain: list[dict] = field(default_factory=list)
    memory_context: list[dict] = field(default_factory=list)
    society_consensus: bool = False


class DecisionEngineBridge:
    """Bidirectional bridge between the intelligence silo and the decision engine.

    Responsibilities:
    1. IMPORT: Pull weights, thresholds, and config from engine.yaml
    2. EXPORT: Push neural predictions to the engine's pipeline
    3. SYNC: Periodically synchronize state
    4. LEARN: Feed engine outcomes back to the neural models

    The bridge operates in two modes:
    - LOCAL: reads engine.yaml directly (same machine)
    - REMOTE: communicates via the decision engine's FastAPI endpoints
    """

    def __init__(self, engine_url: str = "http://localhost:8000",
                 engine_yaml_path: str | None = None,
                 sync_interval: float = 30.0):
        self.engine_url = engine_url.rstrip("/")
        self.engine_yaml_path = engine_yaml_path
        self.sync_interval = sync_interval
        self.weights = EngineWeights()
        self._last_sync = 0.0
        self._client = httpx.AsyncClient(timeout=10.0)

    def load_local_config(self, yaml_path: str) -> EngineWeights:
        """Load weights directly from a local engine.yaml file."""
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        self.weights = EngineWeights(
            value_weights=cfg.get("value_weights", {}),
            penalty_weights=cfg.get("penalty_weights", {}),
            trust_multiplier=cfg.get("trust_multiplier", {}),
            rtql_multiplier=cfg.get("rtql_trust_multiplier", {}),
            thresholds=cfg.get("thresholds", {}),
            authority_matrix=cfg.get("authority_matrix", {}),
        )
        self._last_sync = time.time()
        logger.info(
            "Loaded engine config: %d value weights, %d penalty weights",
            len(self.weights.value_weights), len(self.weights.penalty_weights),
        )
        return self.weights

    async def sync_remote(self) -> EngineWeights | None:
        """Sync weights from the remote decision engine API."""
        try:
            resp = await self._client.get(f"{self.engine_url}/api/config")
            resp.raise_for_status()
            cfg = resp.json()

            self.weights = EngineWeights(
                value_weights=cfg.get("value_weights", {}),
                penalty_weights=cfg.get("penalty_weights", {}),
                trust_multiplier=cfg.get("trust_multiplier", {}),
                rtql_multiplier=cfg.get("rtql_trust_multiplier", {}),
                thresholds=cfg.get("thresholds", {}),
                authority_matrix=cfg.get("authority_matrix", {}),
            )
            self._last_sync = time.time()
            return self.weights

        except httpx.HTTPError as e:
            logger.warning("Failed to sync with decision engine: %s", e)
            return None

    async def push_prediction(self, prediction: NeuralPrediction) -> bool:
        """Push a neural prediction to the decision engine."""
        try:
            payload = {
                "decision_id": prediction.decision_id,
                "neural_scores": prediction.predicted_scores,
                "predicted_trust": prediction.predicted_trust_tier,
                "confidence": prediction.confidence,
                "causal_chain": prediction.causal_chain,
                "society_consensus": prediction.society_consensus,
                "source": "intelligence_silo",
            }
            resp = await self._client.post(
                f"{self.engine_url}/api/neural-prediction",
                json=payload,
            )
            resp.raise_for_status()
            return True

        except httpx.HTTPError as e:
            logger.warning("Failed to push prediction: %s", e)
            return False

    async def pull_outcomes(self, since_timestamp: float = 0) -> list[dict]:
        """Pull execution outcomes from the engine for learning."""
        try:
            resp = await self._client.get(
                f"{self.engine_url}/api/outcomes",
                params={"since": since_timestamp},
            )
            resp.raise_for_status()
            return resp.json().get("outcomes", [])

        except httpx.HTTPError as e:
            logger.warning("Failed to pull outcomes: %s", e)
            return []

    def needs_sync(self) -> bool:
        """Check if we're due for a sync."""
        return (time.time() - self._last_sync) > self.sync_interval

    def weight_tensor(self) -> torch.Tensor:
        """Get engine weights as a tensor for neural model input."""
        return self.weights.to_tensor()

    def interpret_neural_output(self, output: torch.Tensor) -> dict:
        """Map neural model output back to named decision dimensions.

        Assumes output is [12] — 8 value dimensions + 4 penalty dimensions.
        """
        value_names = list(self.weights.value_weights.keys())
        penalty_names = list(self.weights.penalty_weights.keys())
        values = output.detach().cpu().tolist()

        result = {}
        for i, name in enumerate(value_names):
            if i < len(values):
                result[name] = values[i]
        for i, name in enumerate(penalty_names):
            idx = len(value_names) + i
            if idx < len(values):
                result[name] = values[idx]

        return result

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
