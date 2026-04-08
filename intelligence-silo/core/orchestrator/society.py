"""Society of Minds — multi-agent orchestration engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from .mind import Mind, MindRole, Signal
from ..models.matrix import SLMMatrix
from ..memory.hierarchy import MemoryHierarchy

logger = logging.getLogger(__name__)


@dataclass
class ThinkCycleResult:
    """Result of a single think cycle through the society."""
    signals: list[Signal] = field(default_factory=list)
    verdict: Signal | None = None
    consensus: bool = False
    confidence: float = 0.0
    cycle_time_ms: float = 0.0
    minds_activated: int = 0


class SocietyOfMinds:
    """Orchestrates multiple Minds collaborating on decisions.

    The Society runs in think cycles:
    1. INPUT: Raw signal injected into perceivers
    2. PERCEIVE: Perceivers structure the input
    3. ANALYZE: Analysts score and evaluate
    4. CRITIQUE: Critics challenge proposals
    5. SYNTHESIZE: Synthesizers build consensus
    6. EXECUTE: Executors plan actions (if consensus reached)
    7. REMEMBER: Memory keepers decide what to store

    The orchestrator manages signal routing between minds,
    coordinates with the SLM matrix for neural inference,
    and handles memory consolidation.
    """

    def __init__(self, matrix: SLMMatrix, memory: MemoryHierarchy,
                 max_cycles: int = 3):
        self.matrix = matrix
        self.memory = memory
        self.max_cycles = max_cycles
        self.minds: dict[str, Mind] = {}
        self.signal_history: list[Signal] = []
        self._build_default_society()

    def _build_default_society(self) -> None:
        """Create the default society of minds."""
        model_names = list(self.matrix.models.keys())

        # Create one mind per role, assign relevant SLMs
        self.add_mind(Mind(
            name="perceiver",
            role=MindRole.PERCEIVER,
            slm_names=[n for n in model_names if "encoder" in n or "classifier" in n],
            priority=1.0,
        ))
        self.add_mind(Mind(
            name="analyst",
            role=MindRole.ANALYST,
            slm_names=[n for n in model_names if "scorer" in n or "classifier" in n],
            priority=1.0,
        ))
        self.add_mind(Mind(
            name="critic",
            role=MindRole.CRITIC,
            slm_names=[n for n in model_names if "trust" in n or "pattern" in n],
            priority=1.2,  # critics get slight priority boost
        ))
        self.add_mind(Mind(
            name="synthesizer",
            role=MindRole.SYNTHESIZER,
            slm_names=[n for n in model_names if "causal" in n or "encoder" in n],
            priority=1.0,
        ))
        self.add_mind(Mind(
            name="executor",
            role=MindRole.EXECUTOR,
            slm_names=model_names[:2],
            priority=0.8,
        ))
        self.add_mind(Mind(
            name="memory_keeper",
            role=MindRole.MEMORY_KEEPER,
            slm_names=[n for n in model_names if "encoder" in n or "memory" in n],
            priority=0.9,
        ))
        self.add_mind(Mind(
            name="sentinel",
            role=MindRole.SENTINEL,
            slm_names=[n for n in model_names if "trust" in n or "pattern" in n],
            priority=1.1,
        ))

    def add_mind(self, mind: Mind) -> None:
        """Add a mind to the society."""
        self.minds[mind.name] = mind

    @torch.no_grad()
    def think(self, input_data: dict, input_ids: torch.Tensor | None = None,
              query_embedding: torch.Tensor | None = None) -> ThinkCycleResult:
        """Run a full think cycle through the society.

        Args:
            input_data: structured input to process
            input_ids: optional tokenized input for SLM inference
            query_embedding: optional embedding for memory queries and routing

        Returns:
            ThinkCycleResult with final verdict and all signals.
        """
        start = time.time()
        result = ThinkCycleResult()

        # Get matrix output if we have input_ids
        matrix_output = None
        if input_ids is not None:
            matrix_output = self.matrix.infer(
                input_ids, query_embedding=query_embedding
            )

        # Query memory for relevant context
        memory_context = None
        if query_embedding is not None:
            memory_context = self.memory.query_flat(query_embedding, top_k=5)

        # Phase 1: INJECT input into perceivers
        input_signal = Signal(
            source="orchestrator",
            content={
                "raw_input": input_data,
                "memory_context": memory_context,
            },
            confidence=1.0,
            signal_type="observation",
        )

        # Route through the processing pipeline
        phase_order = [
            MindRole.PERCEIVER,
            MindRole.ANALYST,
            MindRole.CRITIC,
            MindRole.SYNTHESIZER,
            MindRole.EXECUTOR,
            MindRole.MEMORY_KEEPER,
            MindRole.SENTINEL,
        ]

        pending_signals = [input_signal]

        for phase_role in phase_order:
            phase_minds = [m for m in self.minds.values() if m.role == phase_role]

            # Deliver pending signals to this phase's minds
            for mind in phase_minds:
                for signal in pending_signals:
                    if signal.target is None or signal.target == mind.name:
                        mind.receive(signal)

            # Process and collect new signals
            new_signals = []
            for mind in phase_minds:
                # Pass matrix output to minds that can use it
                mind_matrix = matrix_output if mind.slm_names else None
                response = mind.process(matrix_output=mind_matrix)
                if response:
                    new_signals.append(response)
                    result.minds_activated += 1

            result.signals.extend(new_signals)
            self.signal_history.extend(new_signals)
            pending_signals = new_signals

        # Extract verdict (from synthesizer)
        verdicts = [s for s in result.signals if s.signal_type == "verdict"]
        if verdicts:
            result.verdict = verdicts[-1]
            result.consensus = verdicts[-1].content.get("consensus", False)
            result.confidence = verdicts[-1].content.get("confidence", 0.0)

        # Store in memory if we have an embedding
        if query_embedding is not None and result.verdict:
            self.memory.encode_and_store(
                key=f"think_{result.verdict.id}",
                embedding=query_embedding,
                context={
                    "input": input_data,
                    "verdict": result.verdict.content,
                    "consensus": result.consensus,
                    "confidence": result.confidence,
                },
                priority=result.confidence,
            )

        result.cycle_time_ms = (time.time() - start) * 1000
        logger.info(
            "Think cycle: %d minds, consensus=%s, confidence=%.2f, time=%.1fms",
            result.minds_activated, result.consensus, result.confidence,
            result.cycle_time_ms,
        )

        return result

    def deliberate(self, input_data: dict, input_ids: torch.Tensor | None = None,
                   query_embedding: torch.Tensor | None = None) -> ThinkCycleResult:
        """Multi-round deliberation — keeps thinking until consensus or max cycles.

        This is the "slow path" for high-stakes decisions that benefit from
        multiple rounds of critique and refinement.
        """
        best_result = None

        for cycle in range(self.max_cycles):
            result = self.think(input_data, input_ids, query_embedding)

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

            if result.consensus and result.confidence >= 0.7:
                logger.info("Deliberation converged in %d cycles", cycle + 1)
                return result

            # Feed verdict back as input for next cycle
            if result.verdict:
                input_data = {
                    "previous_verdict": result.verdict.content,
                    "previous_confidence": result.confidence,
                    "cycle": cycle + 1,
                    "original_input": input_data,
                }

        logger.info(
            "Deliberation reached max cycles (%d), best confidence: %.2f",
            self.max_cycles, best_result.confidence if best_result else 0.0,
        )
        return best_result or ThinkCycleResult()

    def health(self) -> dict:
        """Society health report."""
        return {
            "minds": {
                name: {
                    "role": mind.role.value,
                    "active": mind.state.active,
                    "confidence": mind.state.confidence,
                    "contributions": mind.state.total_contributions,
                    "acceptance_rate": f"{mind.state.acceptance_rate:.1%}",
                }
                for name, mind in self.minds.items()
            },
            "total_signals": len(self.signal_history),
            "matrix": self.matrix.performance_report(),
            "memory": self.memory.health(),
        }
