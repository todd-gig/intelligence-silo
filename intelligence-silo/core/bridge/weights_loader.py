"""Lightweight weights loader — imports value/penalty/trust weights from
the decision-engine's `config/engine.yaml`.

The existing `connector.DecisionEngineBridge` does the same thing but
imports `httpx` + `torch` + `numpy` at module-load, which makes it heavy
for environments that just need to align the silo's SLM scoring with
the engine's value matrix. This module is the YAML-only path required
by the canonical doctrine claim ("Imports weights from
`decision-engine/config/engine.yaml`") with zero GPU/HTTP dependencies.

Usage:

    from core.bridge.weights_loader import load_decision_weights

    weights = load_decision_weights()  # default path resolution
    print(weights.value_weights["strategic_alignment"])  # 2.0

If the YAML cannot be located or parsed, an empty `EngineWeights` is
returned with `loaded == False` — callers should check before using.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EngineWeights:
    """Decision-engine weights as plain Python data — no tensor, no HTTP.

    Mirrors the connector's EngineWeights field set; consumers that want
    a torch tensor can construct it themselves.
    """
    value_weights: dict[str, float] = field(default_factory=dict)
    penalty_weights: dict[str, float] = field(default_factory=dict)
    trust_multiplier: dict[str, float] = field(default_factory=dict)
    rtql_multiplier: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    authority_matrix: dict = field(default_factory=dict)
    source_path: Optional[str] = None
    loaded: bool = False

    def composite_weight(self, dimension: str) -> float:
        """Return the weight for a value or penalty dimension; 0 if unknown."""
        if dimension in self.value_weights:
            return float(self.value_weights[dimension])
        if dimension in self.penalty_weights:
            return float(self.penalty_weights[dimension])
        return 0.0


# ─────────────────────────────────────────────────────────────────────────


def _candidate_paths() -> list[Path]:
    """Where might engine.yaml live? Try, in order:

    1. $GIGATON_ENGINE_YAML  (explicit override)
    2. Sibling-of-this-repo lookup — `../decision-engine/config/engine.yaml`
       (matches the canonical layout when both repos are cloned side-by-side)
    3. `<silo>/config/engine.yaml` if the operator has copied it locally
    4. `~/Documents/GitHub/decision-engine/config/engine.yaml` (Todd's layout)
    """
    paths: list[Path] = []
    env_path = os.environ.get("GIGATON_ENGINE_YAML")
    if env_path:
        paths.append(Path(env_path))
    here = Path(__file__).resolve()
    silo_root = here.parent.parent.parent  # core/bridge/this -> core/bridge -> core -> silo
    paths.append(silo_root.parent / "decision-engine" / "config" / "engine.yaml")
    paths.append(silo_root / "config" / "engine.yaml")
    paths.append(
        Path.home() / "Documents" / "GitHub" / "decision-engine"
        / "config" / "engine.yaml"
    )
    return paths


def _resolve_yaml_path(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        p = Path(explicit)
        return p if p.exists() else None
    for cand in _candidate_paths():
        if cand.exists():
            return cand
    return None


def load_decision_weights(
    yaml_path: Optional[Path] = None,
) -> EngineWeights:
    """Load decision-engine weights from `engine.yaml`.

    Returns an `EngineWeights` with `loaded=True` if successful; an
    empty `EngineWeights` with `loaded=False` and a logged warning
    otherwise. Never raises — graceful degradation is doctrine.

    The function is idempotent and side-effect-free (does not cache;
    callers cache themselves if desired).
    """
    try:
        import yaml  # local import — pyyaml only needed when called
    except ImportError:
        logger.warning(
            "pyyaml not available — weights bridge cannot load engine.yaml"
        )
        return EngineWeights()

    resolved = _resolve_yaml_path(yaml_path)
    if resolved is None:
        logger.info(
            "engine.yaml not found in any candidate path — "
            "weights_loader returning empty EngineWeights"
        )
        return EngineWeights()

    try:
        with resolved.open() as fh:
            cfg = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to parse %s: %s", resolved, exc)
        return EngineWeights()

    return EngineWeights(
        value_weights=dict(cfg.get("value_weights", {})),
        penalty_weights=dict(cfg.get("penalty_weights", {})),
        trust_multiplier=dict(cfg.get("trust_multiplier", {})),
        rtql_multiplier=dict(cfg.get("rtql_trust_multiplier", {})),
        thresholds=dict(cfg.get("thresholds", {})),
        authority_matrix=dict(cfg.get("authority_matrix", {})),
        source_path=str(resolved),
        loaded=True,
    )


def alignment_score(
    weights: EngineWeights,
    dimension_scores: dict[str, float],
) -> float:
    """Score a SLM output against the engine's value matrix.

    For each (dimension, score) pair, multiply by the engine weight
    (value or penalty); penalty dimensions subtract. Returns the
    weighted net — the silo's local approximation of the engine's
    composite value score.

    This is the alignment hook: the silo's specialist models can call
    `alignment_score(weights, predicted_dims)` to verify their output
    aligns with the engine's value matrix BEFORE pushing predictions
    via the (heavier) `connector.DecisionEngineBridge`.
    """
    if not weights.loaded:
        return 0.0
    total = 0.0
    for dim, score in dimension_scores.items():
        if dim in weights.value_weights:
            total += float(score) * float(weights.value_weights[dim])
        elif dim in weights.penalty_weights:
            total -= float(score) * float(weights.penalty_weights[dim])
    return total
