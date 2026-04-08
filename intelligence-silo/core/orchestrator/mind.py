"""Mind — an individual agent within the Society of Minds."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import torch
import numpy as np


class MindRole(Enum):
    """Roles a mind can play in the society."""
    PERCEIVER = "perceiver"       # Processes raw input into structured signals
    ANALYST = "analyst"           # Evaluates signals against criteria
    CRITIC = "critic"             # Challenges proposals and finds weaknesses
    SYNTHESIZER = "synthesizer"   # Combines multiple analyses into coherent output
    EXECUTOR = "executor"         # Translates decisions into actionable plans
    MEMORY_KEEPER = "memory_keeper"  # Manages memory consolidation and recall
    SENTINEL = "sentinel"         # Monitors for anomalies and trust violations


@dataclass
class MindState:
    """Current state of a mind."""
    active: bool = True
    confidence: float = 0.5
    last_output: dict = field(default_factory=dict)
    last_active: float = field(default_factory=time.time)
    total_contributions: int = 0
    accepted_contributions: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_contributions == 0:
            return 0.0
        return self.accepted_contributions / self.total_contributions


@dataclass
class Signal:
    """A message passed between minds in the society."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    source: str = ""  # mind name
    target: str | None = None  # None = broadcast
    content: dict = field(default_factory=dict)
    embedding: torch.Tensor | None = None
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    signal_type: str = "observation"  # observation | proposal | critique | verdict


class Mind:
    """An individual agent in the Society of Minds.

    Each mind has:
    - A role (perceiver, analyst, critic, etc.)
    - Access to specific SLMs from the matrix
    - A state tracking its performance
    - The ability to send/receive signals to/from other minds

    Minds collaborate by passing typed signals through the orchestrator.
    """

    def __init__(self, name: str, role: MindRole, slm_names: list[str],
                 priority: float = 1.0):
        self.name = name
        self.role = role
        self.slm_names = slm_names  # which SLMs this mind can use
        self.priority = priority
        self.state = MindState()
        self._inbox: list[Signal] = []

    def receive(self, signal: Signal) -> None:
        """Receive a signal from another mind or the orchestrator."""
        self._inbox.append(signal)

    def process(self, matrix_output: dict | None = None) -> Signal | None:
        """Process inbox signals and optionally matrix output to produce a response.

        Args:
            matrix_output: output from SLM matrix inference (if this mind triggered it)

        Returns:
            A Signal to broadcast, or None if nothing to say.
        """
        if not self._inbox and matrix_output is None:
            return None

        self.state.last_active = time.time()
        self.state.total_contributions += 1

        # Build response based on role
        if self.role == MindRole.PERCEIVER:
            return self._perceive(matrix_output)
        elif self.role == MindRole.ANALYST:
            return self._analyze(matrix_output)
        elif self.role == MindRole.CRITIC:
            return self._critique()
        elif self.role == MindRole.SYNTHESIZER:
            return self._synthesize(matrix_output)
        elif self.role == MindRole.EXECUTOR:
            return self._plan_execution(matrix_output)
        elif self.role == MindRole.MEMORY_KEEPER:
            return self._manage_memory(matrix_output)
        elif self.role == MindRole.SENTINEL:
            return self._monitor()

        return None

    def _perceive(self, matrix_output: dict | None) -> Signal:
        """Perceiver: structure raw input into typed observations."""
        observations = [s.content for s in self._inbox]
        self._inbox.clear()

        content = {
            "role": "perception",
            "observations": observations,
            "model_output": _tensor_to_list(matrix_output) if matrix_output else None,
            "signal_count": len(observations),
        }

        return Signal(
            source=self.name,
            content=content,
            confidence=self.state.confidence,
            signal_type="observation",
        )

    def _analyze(self, matrix_output: dict | None) -> Signal:
        """Analyst: evaluate signals against decision criteria."""
        inputs = [s.content for s in self._inbox]
        self._inbox.clear()

        content = {
            "role": "analysis",
            "inputs_analyzed": len(inputs),
            "model_scores": _tensor_to_list(matrix_output) if matrix_output else None,
            "recommendation": "proceed" if matrix_output else "insufficient_data",
        }

        return Signal(
            source=self.name,
            content=content,
            confidence=self.state.confidence,
            signal_type="proposal",
        )

    def _critique(self) -> Signal:
        """Critic: challenge proposals and find weaknesses."""
        proposals = [s for s in self._inbox if s.signal_type == "proposal"]
        self._inbox.clear()

        concerns = []
        for p in proposals:
            if p.confidence < 0.7:
                concerns.append({
                    "source": p.source,
                    "issue": "low_confidence",
                    "value": p.confidence,
                })

        content = {
            "role": "critique",
            "proposals_reviewed": len(proposals),
            "concerns": concerns,
            "approval": len(concerns) == 0,
        }

        return Signal(
            source=self.name,
            content=content,
            confidence=self.state.confidence,
            signal_type="critique",
        )

    def _synthesize(self, matrix_output: dict | None) -> Signal:
        """Synthesizer: combine multiple inputs into coherent output."""
        all_signals = list(self._inbox)
        self._inbox.clear()

        proposals = [s for s in all_signals if s.signal_type == "proposal"]
        critiques = [s for s in all_signals if s.signal_type == "critique"]

        # Compute consensus
        approved = all(
            c.content.get("approval", False) for c in critiques
        ) if critiques else True

        avg_confidence = (
            sum(s.confidence for s in proposals) / max(len(proposals), 1)
        )

        content = {
            "role": "synthesis",
            "inputs": len(all_signals),
            "consensus": approved,
            "confidence": avg_confidence,
            "model_fusion": _tensor_to_list(matrix_output) if matrix_output else None,
        }

        return Signal(
            source=self.name,
            content=content,
            confidence=avg_confidence,
            signal_type="verdict",
        )

    def _plan_execution(self, matrix_output: dict | None) -> Signal:
        """Executor: translate verdicts into actionable steps."""
        verdicts = [s for s in self._inbox if s.signal_type == "verdict"]
        self._inbox.clear()

        steps = []
        for v in verdicts:
            if v.content.get("consensus", False):
                steps.append({
                    "action": "execute",
                    "confidence": v.confidence,
                    "source_verdict": v.id,
                })

        content = {
            "role": "execution_plan",
            "steps": steps,
            "executable": len(steps) > 0,
        }

        return Signal(
            source=self.name,
            content=content,
            confidence=self.state.confidence,
            signal_type="proposal",
        )

    def _manage_memory(self, matrix_output: dict | None) -> Signal:
        """Memory Keeper: decide what to remember and what to forget."""
        signals = list(self._inbox)
        self._inbox.clear()

        to_store = [
            s for s in signals
            if s.confidence >= 0.6 or s.signal_type == "verdict"
        ]

        content = {
            "role": "memory_management",
            "signals_received": len(signals),
            "signals_to_store": len(to_store),
            "store_ids": [s.id for s in to_store],
        }

        return Signal(
            source=self.name,
            content=content,
            confidence=self.state.confidence,
            signal_type="observation",
        )

    def _monitor(self) -> Signal:
        """Sentinel: watch for anomalies."""
        signals = list(self._inbox)
        self._inbox.clear()

        anomalies = [
            s for s in signals
            if s.confidence < 0.3 or s.content.get("concerns", [])
        ]

        content = {
            "role": "monitoring",
            "signals_checked": len(signals),
            "anomalies_detected": len(anomalies),
            "alert": len(anomalies) > 0,
        }

        return Signal(
            source=self.name,
            content=content,
            confidence=self.state.confidence,
            signal_type="observation",
        )


def _tensor_to_list(data: dict | None) -> dict | None:
    """Convert tensor values in a dict to lists for serialization."""
    if data is None:
        return None
    result = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu().tolist()
        elif isinstance(v, dict):
            result[k] = _tensor_to_list(v)
        else:
            result[k] = v
    return result
