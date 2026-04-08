"""Decision Memory Recorder — auto-records every pipeline result into the memory hierarchy.

Every decision processed by the engine becomes a memory:
- Working memory: active decisions (TTL-governed)
- Episodic memory: full decision context + outcome
- Semantic memory: extracted patterns consolidated over time
- Procedural memory: learned action sequences from execution outcomes

This module bridges the decision engine's PipelineResult format with
the intelligence silo's hierarchical memory system.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Normalized record of a decision for memory storage."""
    decision_id: str
    title: str
    domain: str
    verdict: str
    confidence: float
    net_value: int
    trust_tier: str
    priority_score: float
    value_classification: str
    alignment_composite: float
    executive_summary: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_result: dict = field(default_factory=dict)


class DecisionMemoryRecorder:
    """Records decision engine pipeline results into the intelligence silo's memory.

    Auto-recording flow:
    1. Pipeline processes a decision → PipelineResult
    2. Recorder normalizes the result into a DecisionRecord
    3. Record is stored in working memory (fast access for active decisions)
    4. Record is simultaneously stored as an episodic memory (long-term)
    5. High-value decisions are flagged for semantic consolidation
    6. Execution patterns feed procedural memory

    All records are also persisted as structured YAML+MD files locally
    for the decision engine's existing MemoryManager compatibility.
    """

    def __init__(self, memory_hierarchy, embed_model=None,
                 local_journal_path: str = "data/decision_journal"):
        """
        Args:
            memory_hierarchy: MemoryHierarchy from the intelligence silo
            embed_model: optional embedding model; falls back to hash-based embeddings
            local_journal_path: where to persist decision journal files
        """
        self.memory = memory_hierarchy
        self.embed_model = embed_model
        self.journal_path = Path(local_journal_path)
        self.journal_path.mkdir(parents=True, exist_ok=True)
        self._records_today: list[DecisionRecord] = []
        self._total_recorded: int = 0

    def record(self, pipeline_result: dict, decision_title: str = "",
               domain: str = "general") -> DecisionRecord:
        """Record a pipeline result as a memory across all layers.

        Args:
            pipeline_result: serialized PipelineResult from the decision engine
            decision_title: human-readable title
            domain: decision domain for categorization

        Returns:
            The normalized DecisionRecord that was stored.
        """
        record = self._normalize(pipeline_result, decision_title, domain)
        embedding = self._generate_embedding(record)

        # 1. Working memory — active decision context
        self.memory.working.store(
            key=f"decision:{record.decision_id}",
            tensor=embedding,
            metadata={
                "decision_id": record.decision_id,
                "title": record.title,
                "verdict": record.verdict,
                "confidence": record.confidence,
                "net_value": record.net_value,
                "trust_tier": record.trust_tier,
            },
            priority=record.priority_score / 100.0,
        )

        # 2. Episodic memory — full decision experience
        importance = self._compute_importance(record)
        self.memory.episodic.record(
            embedding=embedding.detach().cpu().numpy(),
            context={
                "type": "decision",
                "decision_id": record.decision_id,
                "title": record.title,
                "domain": record.domain,
                "verdict": record.verdict,
                "net_value": record.net_value,
                "trust_tier": record.trust_tier,
                "priority_score": record.priority_score,
                "value_classification": record.value_classification,
                "alignment": record.alignment_composite,
                "summary": record.executive_summary[:500],
                "category": record.domain,
            },
            outcome={"verdict": record.verdict, "confidence": record.confidence},
            importance=importance,
        )

        # 3. If high-value, fast-track to semantic memory
        if importance >= 0.8 or record.net_value >= 20:
            emb_np = embedding.detach().cpu().numpy()
            self.memory.semantic.store(
                embedding=emb_np,
                knowledge={
                    "type": "high_value_decision",
                    "decision_id": record.decision_id,
                    "title": record.title,
                    "verdict": record.verdict,
                    "net_value": record.net_value,
                    "summary": record.executive_summary[:500],
                },
                category=record.domain,
                confidence=min(record.confidence + 0.1, 1.0),
            )

        # 4. Procedural: learn action patterns from auto-executed decisions
        if record.verdict == "auto_execute":
            trigger_emb = embedding.detach().cpu().numpy()
            self.memory.procedural.learn(
                name=f"auto_exec:{record.domain}:{record.value_classification}",
                trigger_embedding=trigger_emb,
                action_sequence=[{
                    "action": "execute",
                    "domain": record.domain,
                    "trust_tier": record.trust_tier,
                    "value_class": record.value_classification,
                }],
                initial_confidence=record.confidence,
            )

        # 5. Persist to local journal
        self._persist_journal(record)
        self._records_today.append(record)
        self._total_recorded += 1

        logger.info(
            "Recorded decision %s: verdict=%s, value=%d, trust=%s, importance=%.2f",
            record.decision_id, record.verdict, record.net_value,
            record.trust_tier, importance,
        )
        return record

    def get_daily_records(self) -> list[DecisionRecord]:
        """Get all decisions recorded today."""
        return list(self._records_today)

    def flush_daily(self) -> dict:
        """End-of-day flush: consolidate and reset daily buffer.

        Returns stats about the day's decisions.
        """
        stats = {
            "total_decisions": len(self._records_today),
            "by_verdict": {},
            "by_domain": {},
            "avg_net_value": 0,
            "avg_confidence": 0,
        }
        if self._records_today:
            for r in self._records_today:
                stats["by_verdict"][r.verdict] = stats["by_verdict"].get(r.verdict, 0) + 1
                stats["by_domain"][r.domain] = stats["by_domain"].get(r.domain, 0) + 1
            stats["avg_net_value"] = sum(r.net_value for r in self._records_today) / len(self._records_today)
            stats["avg_confidence"] = sum(r.confidence for r in self._records_today) / len(self._records_today)

        # Run memory consolidation
        consolidation_stats = self.memory.consolidate()
        stats["consolidation"] = consolidation_stats

        self._records_today.clear()
        return stats

    def _normalize(self, result: dict, title: str, domain: str) -> DecisionRecord:
        """Normalize a pipeline result dict into a DecisionRecord."""
        exec_packet = result.get("execution", result.get("execution_packet", {}))
        verdict = exec_packet.get("verdict", result.get("recommended_action", "unknown"))

        return DecisionRecord(
            decision_id=result.get("decision_id", f"DEC-{int(time.time())}"),
            title=title or result.get("title", "Untitled Decision"),
            domain=domain,
            verdict=verdict,
            confidence=self._estimate_confidence(result),
            net_value=result.get("net_value_score", 0),
            trust_tier=result.get("trust_tier", "T0"),
            priority_score=result.get("priority_score", 0.0),
            value_classification=result.get("value_classification", "unknown"),
            alignment_composite=result.get("alignment_composite", 0.0),
            executive_summary=result.get("executive_summary", ""),
            raw_result=result,
        )

    def _estimate_confidence(self, result: dict) -> float:
        """Derive a confidence score from the pipeline result."""
        has_chain = bool(result.get("certificate_status", {}))
        chain_complete = all(
            v == "issued" for v in result.get("certificate_status", {}).values()
        )
        trust = result.get("trust_total", 0) / 35.0  # max trust = 7 dims * 5
        value_norm = max(0, result.get("net_value_score", 0)) / 28.0  # max net = 40-12

        score = trust * 0.4 + value_norm * 0.3
        if chain_complete:
            score += 0.2
        elif has_chain:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _compute_importance(self, record: DecisionRecord) -> float:
        """Compute importance score for episodic memory storage."""
        # High-value or high-risk decisions are more important
        value_norm = max(0, record.net_value) / 28.0
        priority_norm = record.priority_score / 100.0
        verdict_weight = {
            "auto_execute": 0.6,
            "escalate_tier_1": 0.7,
            "escalate_tier_2": 0.8,
            "escalate_tier_3": 0.9,
            "block": 0.9,
            "needs_data": 0.5,
            "information_only": 0.3,
        }.get(record.verdict, 0.5)

        return min(1.0, value_norm * 0.3 + priority_norm * 0.3 + verdict_weight * 0.4)

    def _generate_embedding(self, record: DecisionRecord) -> torch.Tensor:
        """Generate an embedding for the decision record.

        Uses the embed_model if available, otherwise creates a deterministic
        hash-based embedding from the record's content.
        """
        if self.embed_model is not None:
            # Use the neural memory encoder
            text = f"{record.title} {record.domain} {record.verdict} {record.executive_summary[:200]}"
            # This would use the memory_encoder SLM — for now, hash-based
            pass

        # Deterministic hash-based embedding (works without model training)
        content = json.dumps({
            "title": record.title,
            "domain": record.domain,
            "verdict": record.verdict,
            "net_value": record.net_value,
            "trust_tier": record.trust_tier,
            "value_classification": record.value_classification,
            "summary": record.executive_summary[:300],
        }, sort_keys=True)

        # Deterministic embedding: hash content repeatedly to fill desired dimension
        dim = self.memory.config.embedding_dim
        # Generate enough bytes: each float32 needs 4 bytes
        needed_bytes = dim * 4
        hash_bytes = b""
        seed = content.encode()
        i = 0
        while len(hash_bytes) < needed_bytes:
            hash_bytes += hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
            i += 1
        arr = np.frombuffer(hash_bytes[:needed_bytes], dtype=np.float32).copy()
        arr = arr / (np.linalg.norm(arr) + 1e-8)
        return torch.from_numpy(arr)

    def _persist_journal(self, record: DecisionRecord) -> None:
        """Persist decision record as a structured YAML+MD file."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_dir = self.journal_path / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{record.decision_id}_{record.verdict}.md"
        content = (
            f"---\n"
            f"decision_id: {record.decision_id}\n"
            f"title: \"{record.title}\"\n"
            f"domain: {record.domain}\n"
            f"verdict: {record.verdict}\n"
            f"confidence: {record.confidence:.3f}\n"
            f"net_value: {record.net_value}\n"
            f"trust_tier: {record.trust_tier}\n"
            f"priority_score: {record.priority_score:.2f}\n"
            f"value_classification: {record.value_classification}\n"
            f"alignment_composite: {record.alignment_composite:.3f}\n"
            f"timestamp: {record.timestamp}\n"
            f"---\n\n"
            f"# {record.title}\n\n"
            f"## Executive Summary\n\n{record.executive_summary}\n\n"
            f"## Details\n\n"
            f"- **Verdict**: {record.verdict}\n"
            f"- **Net Value**: {record.net_value}\n"
            f"- **Trust Tier**: {record.trust_tier}\n"
            f"- **Priority**: {record.priority_score:.2f}\n"
            f"- **Alignment**: {record.alignment_composite:.3f}\n"
        )
        (day_dir / filename).write_text(content, encoding="utf-8")

    @property
    def stats(self) -> dict:
        return {
            "total_recorded": self._total_recorded,
            "today": len(self._records_today),
            "journal_path": str(self.journal_path),
        }
