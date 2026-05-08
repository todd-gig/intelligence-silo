"""Weights loader tests — verifies the silo can ingest engine.yaml
without heavy deps and align against the canonical value matrix.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure repo-root import works without an editable install
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.bridge.weights_loader import (
    EngineWeights,
    alignment_score,
    load_decision_weights,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


SAMPLE_ENGINE_YAML = """\
thresholds:
  value_execute_min: 14.0
  value_escalate_min: 8.0
  trust_execute_min: 3.5
  trust_recommend_min: 2.2

value_weights:
  revenue_impact: 1.5
  cost_efficiency: 1.2
  time_leverage: 1.3
  strategic_alignment: 2.0
  customer_benefit: 1.4
  knowledge_creation: 1.1
  compounding_potential: 1.8
  reversibility: 1.0

penalty_weights:
  downside_risk: 2.0
  execution_drag: 1.2
  uncertainty: 1.5
  ethical_misalignment: 3.0

trust_multiplier:
  T0: 0.2
  T1: 0.5
  T2: 0.8
  T3: 1.0
  T4: 1.2

rtql_trust_multiplier:
  noise: 0.0
  weak_signal: 0.35
  qualified: 1.0
  certified: 1.15
  axiom_candidate: 1.5

authority_matrix:
  D1:
    min_trust: T3
"""


@pytest.fixture
def tmp_engine_yaml() -> Path:
    """Write a temp engine.yaml and return its path."""
    fd, name = tempfile.mkstemp(suffix=".yaml")
    os.write(fd, SAMPLE_ENGINE_YAML.encode())
    os.close(fd)
    return Path(name)


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────


def test_load_with_explicit_path(tmp_engine_yaml: Path) -> None:
    weights = load_decision_weights(tmp_engine_yaml)
    assert weights.loaded
    assert weights.source_path == str(tmp_engine_yaml)
    assert weights.value_weights["strategic_alignment"] == 2.0
    assert weights.penalty_weights["ethical_misalignment"] == 3.0
    assert weights.thresholds["value_execute_min"] == 14.0
    assert weights.trust_multiplier["T3"] == 1.0
    assert weights.rtql_multiplier["axiom_candidate"] == 1.5
    assert weights.authority_matrix["D1"]["min_trust"] == "T3"


def test_load_returns_empty_when_path_missing() -> None:
    """Graceful degradation — no exception, just empty weights."""
    bogus = Path("/tmp/definitely-not-a-real-engine-yaml.yaml")
    weights = load_decision_weights(bogus)
    assert not weights.loaded
    assert weights.value_weights == {}
    assert weights.penalty_weights == {}


def test_load_via_env_var(tmp_engine_yaml: Path, monkeypatch) -> None:
    """`GIGATON_ENGINE_YAML` env override resolves before defaults."""
    monkeypatch.setenv("GIGATON_ENGINE_YAML", str(tmp_engine_yaml))
    weights = load_decision_weights()  # no explicit path
    assert weights.loaded
    assert weights.source_path == str(tmp_engine_yaml)


def test_load_finds_default_layout() -> None:
    """If decision-engine repo is sibling-cloned, default path resolves.

    Skipped if the layout doesn't match — this is an integration-flavored
    smoke test, not a fixture-driven unit test.
    """
    weights = load_decision_weights()
    if not weights.loaded:
        pytest.skip("decision-engine/config/engine.yaml not found in default "
                    "candidate paths — skipping integration smoke test")
    # If we got here, we loaded the real engine.yaml — verify canonical weights
    assert weights.value_weights.get("strategic_alignment") == 2.0
    assert weights.penalty_weights.get("ethical_misalignment") == 3.0


def test_composite_weight_lookup(tmp_engine_yaml: Path) -> None:
    weights = load_decision_weights(tmp_engine_yaml)
    assert weights.composite_weight("strategic_alignment") == 2.0
    assert weights.composite_weight("downside_risk") == 2.0
    assert weights.composite_weight("nonexistent_dimension") == 0.0


def test_alignment_score_uses_value_and_penalty_weights(
    tmp_engine_yaml: Path,
) -> None:
    """alignment_score should multiply value dims by their weights and
    SUBTRACT penalty dims times their weights."""
    weights = load_decision_weights(tmp_engine_yaml)
    score = alignment_score(weights, {
        "strategic_alignment": 5.0,    # 5 * 2.0 = +10
        "compounding_potential": 4.0,  # 4 * 1.8 = +7.2
        "downside_risk": 3.0,          # 3 * 2.0 = -6
        "ethical_misalignment": 1.0,   # 1 * 3.0 = -3
    })
    assert score == pytest.approx(10.0 + 7.2 - 6.0 - 3.0)


def test_alignment_score_returns_zero_when_unloaded() -> None:
    """No weights loaded → score is 0 (caller must handle, not crash)."""
    weights = EngineWeights()  # not loaded
    score = alignment_score(weights, {"strategic_alignment": 5.0})
    assert score == 0.0


def test_unknown_dimension_ignored_in_alignment(tmp_engine_yaml: Path) -> None:
    weights = load_decision_weights(tmp_engine_yaml)
    score = alignment_score(weights, {
        "strategic_alignment": 1.0,
        "totally_made_up_dimension": 99.0,  # should be ignored
    })
    assert score == pytest.approx(2.0)
