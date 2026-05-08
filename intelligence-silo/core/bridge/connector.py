"""Decision Engine weights bridge.

Loads engine.yaml from the decision-engine repo (or from an HTTP endpoint when
running on Cloud Run) and exposes the weights as typed dataclasses so the SLM
matrix can reference canonical scoring parameters.

Priority order for config source:
1. DECISION_ENGINE_CONFIG_PATH env var (absolute path to engine.yaml)
2. HTTP GET to DECISION_ENGINE_URL/config/weights (returns JSON)
3. Fallback: hardcoded defaults matching engine.yaml as of 2026-05-07
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import httpx
    import torch
    import numpy as np
except ImportError:
    httpx = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ── Fallback defaults (mirror engine.yaml exactly as of 2026-05-07) ──────────

_DEFAULT_VALUE_WEIGHTS: dict[str, float] = {
    "revenue_impact": 1.5,
    "cost_efficiency": 1.2,
    "time_leverage": 1.3,
    "strategic_alignment": 2.0,
    "customer_benefit": 1.4,
    "knowledge_creation": 1.1,
    "compounding_potential": 1.8,
    "reversibility": 1.0,
}

_DEFAULT_PENALTY_WEIGHTS: dict[str, float] = {
    "downside_risk": 2.0,
    "execution_drag": 1.2,
    "uncertainty": 1.5,
    "ethical_misalignment": 3.0,
}

_DEFAULT_TRUST_MULTIPLIER: dict[str, float] = {
    "T0": 0.2,
    "T1": 0.5,
    "T2": 0.8,
    "T3": 1.0,
    "T4": 1.2,
}

_DEFAULT_RTQL_TRUST_MULTIPLIER: dict[str, float] = {
    "noise": 0.00,
    "weak_signal": 0.35,
    "echo_signal": 0.50,
    "qualified": 1.00,
    "certification_gap": 0.85,
    "certified": 1.15,
    "research_grade": 1.30,
    "first_principles_candidate": 1.50,
    "axiom_candidate": 1.50,
}

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "value_execute_min": 14.0,
    "value_escalate_min": 8.0,
    "trust_execute_min": 3.5,
    "trust_recommend_min": 2.2,
}


# ── Typed dataclass ───────────────────────────────────────────────────────────

@dataclass
class EngineConfig:
    """Canonical engine weights and thresholds loaded from engine.yaml."""

    value_weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_VALUE_WEIGHTS)
    )
    penalty_weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_PENALTY_WEIGHTS)
    )
    trust_multiplier: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_TRUST_MULTIPLIER)
    )
    rtql_trust_multiplier: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_RTQL_TRUST_MULTIPLIER)
    )
    thresholds: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_THRESHOLDS)
    )
    source: str = "fallback"  # "file" | "http" | "fallback"


# ── Connector ─────────────────────────────────────────────────────────────────

class DecisionEngineConnector:
    """Loads engine.yaml from the decision-engine repo and exposes weights to
    the intelligence silo's SLM matrix as typed, cached properties.

    Source resolution order:
    1. ``DECISION_ENGINE_CONFIG_PATH`` env var — absolute path to engine.yaml
    2. HTTP GET to ``DECISION_ENGINE_URL/config/weights`` — JSON payload
    3. Hardcoded fallback defaults matching engine.yaml as of 2026-05-07

    Example::

        conn = DecisionEngineConnector()
        cfg  = conn.load()
        print(conn.value_weights)        # {'revenue_impact': 1.5, ...}
        print(conn.trust_multiplier)     # {'T0': 0.2, ..., 'T4': 1.2}
        conn.reload()                    # bust cache and re-load
    """

    def __init__(self) -> None:
        self._cache: Optional[EngineConfig] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> EngineConfig:
        """Load config from the highest-priority available source.

        The result is cached in memory.  Call :meth:`reload` to force a fresh
        fetch.  Never raises — falls back to defaults on any error.
        """
        if self._cache is not None:
            return self._cache

        cfg = self._try_file() or self._try_http() or self._fallback()
        self._cache = cfg
        return cfg

    def reload(self) -> EngineConfig:
        """Clear the in-memory cache and re-load from source."""
        self._cache = None
        return self.load()

    # ── Properties (convenience shortcuts) ───────────────────────────────────

    @property
    def value_weights(self) -> dict[str, float]:
        return self.load().value_weights

    @property
    def penalty_weights(self) -> dict[str, float]:
        return self.load().penalty_weights

    @property
    def trust_multiplier(self) -> dict[str, float]:
        return self.load().trust_multiplier

    @property
    def rtql_trust_multiplier(self) -> dict[str, float]:
        return self.load().rtql_trust_multiplier

    @property
    def thresholds(self) -> dict[str, float]:
        return self.load().thresholds

    # ── Private load methods ──────────────────────────────────────────────────

    def _try_file(self) -> Optional[EngineConfig]:
        """Attempt to load from DECISION_ENGINE_CONFIG_PATH env var."""
        path = os.environ.get("DECISION_ENGINE_CONFIG_PATH", "")
        if not path:
            return None
        if not os.path.isfile(path):
            logger.warning(
                "DECISION_ENGINE_CONFIG_PATH set but file not found: %s", path
            )
            return None
        try:
            return self._parse_file(path)
        except Exception as exc:
            logger.warning("Failed to parse engine config file %s: %s", path, exc)
            return None

    def _try_http(self) -> Optional[EngineConfig]:
        """Attempt to load from DECISION_ENGINE_URL/config/weights via HTTP."""
        base_url = os.environ.get("DECISION_ENGINE_URL", "")
        if not base_url:
            return None
        url = base_url.rstrip("/") + "/config/weights"
        try:
            import urllib.request
            import json as _json

            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
            data = _json.loads(raw)
            return self._parse_dict(data, source="http")
        except Exception as exc:
            logger.warning("Failed to fetch engine config from %s: %s", url, exc)
            return None

    def _fallback(self) -> EngineConfig:
        """Return hardcoded defaults matching engine.yaml as of 2026-05-07."""
        logger.info(
            "DecisionEngineConnector: using hardcoded fallback weights "
            "(set DECISION_ENGINE_CONFIG_PATH or DECISION_ENGINE_URL to override)"
        )
        return EngineConfig(source="fallback")

    # ── Parsing helpers ───────────────────────────────────────────────────────

    def _parse_file(self, path: str) -> EngineConfig:
        """Parse a YAML or JSON engine config file."""
        with open(path) as fh:
            raw = fh.read()

        data: dict = {}
        try:
            import yaml  # type: ignore[import]
            data = yaml.safe_load(raw) or {}
        except ImportError:
            import json as _json
            data = _json.loads(raw)

        return self._parse_dict(data, source="file")

    @staticmethod
    def _parse_dict(data: dict, source: str) -> EngineConfig:
        """Build an EngineConfig from a raw parsed dict."""
        return EngineConfig(
            value_weights=dict(
                data.get("value_weights", _DEFAULT_VALUE_WEIGHTS)
            ),
            penalty_weights=dict(
                data.get("penalty_weights", _DEFAULT_PENALTY_WEIGHTS)
            ),
            trust_multiplier=dict(
                data.get("trust_multiplier", _DEFAULT_TRUST_MULTIPLIER)
            ),
            rtql_trust_multiplier=dict(
                data.get("rtql_trust_multiplier", _DEFAULT_RTQL_TRUST_MULTIPLIER)
            ),
            thresholds=dict(
                data.get("thresholds", _DEFAULT_THRESHOLDS)
            ),
            source=source,
        )


# ── Legacy bridge (preserved for backwards-compatibility) ─────────────────────
# The DecisionEngineBridge class below was the original connector; kept intact
# so existing imports of `from .connector import DecisionEngineBridge` continue
# to work.

@dataclass
class EngineWeights:
    """Weights imported from the decision engine's engine.yaml."""
    value_weights: dict[str, float] = field(default_factory=dict)
    penalty_weights: dict[str, float] = field(default_factory=dict)
    trust_multiplier: dict[str, float] = field(default_factory=dict)
    rtql_multiplier: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    authority_matrix: dict = field(default_factory=dict)

    def to_tensor(self):  # type: ignore[return]
        """Convert value + penalty weights to a single tensor for neural scoring."""
        if torch is None:
            raise ImportError("torch is required for to_tensor()")
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
    - LOCAL: reads engine.yaml directly (same machine / dev)
    - REMOTE: communicates with the live SIE API Gateway on Cloud Run

    Default URL is the live SIE production endpoint. Override with:
      DECISION_ENGINE_URL environment variable  (any deployment)
      engine_url constructor parameter          (programmatic)

    SIE Production: https://api-gateway-service-rjmcrtvuzq-uc.a.run.app
    Local dev:      http://localhost:8000
    """

    SIE_PRODUCTION_URL = "https://api-gateway-service-rjmcrtvuzq-uc.a.run.app"

    def __init__(self, engine_url: str | None = None,
                 engine_yaml_path: str | None = None,
                 sync_interval: float = 30.0):
        resolved = engine_url or os.environ.get("DECISION_ENGINE_URL") or self.SIE_PRODUCTION_URL
        self.engine_url = resolved.rstrip("/")
        self.engine_yaml_path = engine_yaml_path
        self.sync_interval = sync_interval
        self.weights = EngineWeights()
        self._last_sync = 0.0
        if httpx is None:
            raise ImportError("httpx is required for DecisionEngineBridge")
        self._client = httpx.AsyncClient(timeout=10.0)

    def load_local_config(self, yaml_path: str) -> EngineWeights:
        """Load weights directly from a local engine.yaml file."""
        try:
            import yaml  # type: ignore[import]
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
        except ImportError:
            import json as _json
            with open(yaml_path) as f:
                cfg = _json.load(f)

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

    async def sync_remote(self) -> Optional[EngineWeights]:
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

        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
            logger.warning("Failed to pull outcomes: %s", e)
            return []

    def needs_sync(self) -> bool:
        """Check if we're due for a sync."""
        return (time.time() - self._last_sync) > self.sync_interval

    def weight_tensor(self):
        """Get engine weights as a tensor for neural model input."""
        return self.weights.to_tensor()

    def interpret_neural_output(self, output) -> dict:
        """Map neural model output back to named decision dimensions.

        Assumes output is [12] — 8 value dimensions + 4 penalty dimensions.
        """
        if torch is None:
            raise ImportError("torch is required for interpret_neural_output()")
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
