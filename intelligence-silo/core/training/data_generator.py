"""Training data generator for the SLM matrix.

Produces training tensors from two sources:
1. SYNTHETIC: procedurally generated decision records covering the full
   distribution of trust tiers, value scores, verdicts, and alignment states.
   These bootstrap the models before any real historical data is available.

2. HISTORICAL: reads from the decision journal (data/decision_journal/) written
   by DecisionMemoryRecorder, plus any JSON-lines outcome files from the
   decision engine's learning_loop. Converts real decisions + outcomes into
   supervised labels.

Output format per record:
    - input_tensor: [float32, shape=(input_dim,)] — encoded decision features
    - label: model-type-specific target (classification index / regression float)
    - weight: importance weight (higher for rare verdicts and high-priority decisions)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Feature dimensions per model type
FEATURE_DIM = 32  # Input features for all models
CLASSIFIER_CLASSES = 6   # auto_execute, escalate_t1, escalate_t2, escalate_t3, block, needs_data
TRUST_TIERS = 5          # T0–T4
SENTENCE_EMBED_DIM = 384 # Semantic embedding dimension (matches memory hierarchy)


@dataclass
class TrainingRecord:
    """A single training sample for any SLM model."""
    model_name: str              # Which model this record targets
    input_tensor: torch.Tensor   # Encoded decision features
    label: torch.Tensor          # Target label (type depends on model)
    weight: float = 1.0          # Sample importance weight
    decision_id: str = ""
    verdict: str = ""
    domain: str = "general"
    source: str = "synthetic"    # "synthetic" | "historical" | "journal"


class DecisionDataGenerator:
    """Generates training data for the 6 SLM specialist models.

    Models and their training objectives:
    - classifier:       verdict prediction (6-class)
    - scorer:           net value score regression (0–28 float)
    - trust_assessor:   trust tier classification (T0–T4, 5-class)
    - memory_encoder:   embedding quality (cosine similarity vs reference)
    - pattern_detector: binary — is this decision a pattern repeat?
    - causal_predictor: predicted outcome score regression (0–1 float)
    """

    VERDICT_MAP = {
        "auto_execute": 0,
        "escalate_tier_1": 1,
        "escalate_tier_2": 2,
        "escalate_tier_3": 3,
        "block": 4,
        "needs_data": 5,
    }
    VERDICT_NAMES = list(VERDICT_MAP.keys())

    TRUST_MAP = {
        "T0_UNQUALIFIED": 0, "T0": 0,
        "T1_MONITORED": 1, "T1": 1,
        "T2_STANDARD": 2, "T2": 2,
        "T3_TRUSTED": 3, "T3": 3,
        "T4_AUTONOMOUS": 4, "T4": 4,
    }

    def __init__(
        self,
        journal_dir: str = "data/decision_journal",
        outcomes_file: str = "data/learning_loop.jsonl",
        seed: int = 42,
    ):
        self.journal_dir = Path(journal_dir)
        self.outcomes_file = Path(outcomes_file)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────

    def generate(
        self,
        n_synthetic: int = 2000,
        include_historical: bool = True,
    ) -> dict[str, list[TrainingRecord]]:
        """Generate training records for all 6 models.

        Returns a dict keyed by model name, each containing a list of
        TrainingRecord instances ready for DataLoader consumption.
        """
        records: dict[str, list[TrainingRecord]] = {
            "classifier": [],
            "scorer": [],
            "trust_assessor": [],
            "memory_encoder": [],
            "pattern_detector": [],
            "causal_predictor": [],
        }

        # 1. Synthetic samples
        logger.info("Generating %d synthetic training records...", n_synthetic)
        synthetic = self._generate_synthetic(n_synthetic)
        for r in synthetic:
            records[r.model_name].append(r)

        # 2. Historical from journal
        if include_historical:
            hist = self._load_journal_records()
            logger.info("Loaded %d historical records from journal", len(hist))
            for r in hist:
                records[r.model_name].append(r)

            outcomes = self._load_outcome_records()
            logger.info("Loaded %d outcome records from learning loop", len(outcomes))
            for r in outcomes:
                records[r.model_name].append(r)

        # Log per-model counts
        for name, recs in records.items():
            verdict_dist = {}
            for r in recs:
                verdict_dist[r.verdict] = verdict_dist.get(r.verdict, 0) + 1
            logger.info("  %s: %d records %s", name, len(recs), verdict_dist)

        return records

    def encode_decision(self, decision_dict: dict) -> torch.Tensor:
        """Encode a decision dict to a fixed-size feature tensor.

        Feature layout (32 dims):
          [0:8]   value scores (revenue_impact, cost_efficiency, time_leverage,
                  strategic_alignment, customer_human_benefit,
                  knowledge_asset_creation, compounding_potential, reversibility)
          [8:12]  penalty scores (downside_risk, execution_drag, uncertainty,
                  ethical_misalignment)
          [12:19] trust scores (evidence_quality, logic_integrity, outcome_history,
                  context_fit, stakeholder_clarity, risk_containment, auditability)
          [19:22] alignment scores (doctrine, ethos, first_principles)
          [22:25] aggregate signals (net_value_norm, trust_total_norm, priority_norm)
          [25:30] intelligence dimensions composite, knowledge, strategy, experience, information
          [30]    has_missing_data (bool→float)
          [31]    ethical_conflict (bool→float)
        """
        features = np.zeros(FEATURE_DIM, dtype=np.float32)

        # Value scores (0–5 scale, normalize to 0–1)
        vs = decision_dict.get("value_scores", {})
        value_keys = ["revenue_impact","cost_efficiency","time_leverage",
                      "strategic_alignment","customer_human_benefit",
                      "knowledge_asset_creation","compounding_potential","reversibility"]
        for i, k in enumerate(value_keys):
            features[i] = vs.get(k, 0) / 5.0

        # Penalty scores
        penalty_keys = ["downside_risk","execution_drag","uncertainty","ethical_misalignment"]
        for i, k in enumerate(penalty_keys):
            features[8 + i] = vs.get(k, 0) / 5.0

        # Trust scores
        ts = decision_dict.get("trust_scores", {})
        trust_keys = ["evidence_quality","logic_integrity","outcome_history",
                      "context_fit","stakeholder_clarity","risk_containment","auditability"]
        for i, k in enumerate(trust_keys):
            features[12 + i] = ts.get(k, 0) / 5.0

        # Alignment
        al = decision_dict.get("alignment_scores", {})
        features[19] = al.get("doctrine_alignment", 0.0)
        features[20] = al.get("ethos_alignment", 0.0)
        features[21] = al.get("first_principles_alignment", 0.0)

        # Aggregate signals
        net = decision_dict.get("net_value_score", 0)
        features[22] = max(0, min(1, (net + 20) / 48.0))  # -20..+28 → 0..1
        trust_total = decision_dict.get("trust_total", 0)
        features[23] = min(1.0, trust_total / 35.0)
        priority = decision_dict.get("priority_score", 0)
        features[24] = min(1.0, priority / 100.0)

        # Intelligence dimensions
        ip = decision_dict.get("intelligence_profile", {})
        dims = ip.get("dimensions", {}) if isinstance(ip, dict) else {}
        features[25] = ip.get("composite", 0.0) if isinstance(ip, dict) else 0.0
        features[26] = dims.get("knowledge", 0.0)
        features[27] = dims.get("strategy", 0.0)
        features[28] = dims.get("experience", 0.0)
        features[29] = dims.get("information", 0.0)

        # Flags
        features[30] = 1.0 if decision_dict.get("has_missing_data", False) else 0.0
        features[31] = 1.0 if decision_dict.get("ethical_conflict", False) else 0.0

        return torch.tensor(features, dtype=torch.float32)

    # ─────────────────────────────────────────────
    # SYNTHETIC GENERATION
    # ─────────────────────────────────────────────

    def _generate_synthetic(self, n: int) -> list[TrainingRecord]:
        """Generate n synthetic records, balanced across verdict classes."""
        records = []
        per_class = n // CLASSIFIER_CLASSES

        for verdict_idx, verdict in enumerate(self.VERDICT_NAMES):
            for _ in range(per_class):
                d = self._synthetic_decision(verdict)
                feat = self.encode_decision(d)
                records.extend(self._decision_to_records(d, feat, verdict, "synthetic"))

        # Fill remainder randomly
        for _ in range(n - len(records) // 6):
            verdict = random.choice(self.VERDICT_NAMES)
            d = self._synthetic_decision(verdict)
            feat = self.encode_decision(d)
            records.extend(self._decision_to_records(d, feat, verdict, "synthetic"))

        return records

    def _synthetic_decision(self, target_verdict: str) -> dict:
        """Create a synthetic decision dict that should produce a given verdict."""
        # Verdict-conditioned sampling
        if target_verdict == "auto_execute":
            vs = self._sample_value_scores(net_min=12, trust_min=22)
            al = {"doctrine_alignment": rn(0.7, 1.0), "ethos_alignment": rn(0.8, 1.0), "first_principles_alignment": rn(0.7, 1.0)}
            trust_tier = "T3"
        elif target_verdict == "block":
            vs = self._sample_value_scores(net_max=0, force_ethical=True)
            al = {"doctrine_alignment": rn(0.0, 0.3), "ethos_alignment": rn(0.0, 0.3), "first_principles_alignment": rn(0.0, 0.4)}
            trust_tier = "T0"
        elif target_verdict == "needs_data":
            vs = self._sample_value_scores(net_min=3, net_max=15)
            al = {"doctrine_alignment": rn(0.4, 0.8), "ethos_alignment": rn(0.4, 0.8), "first_principles_alignment": rn(0.4, 0.7)}
            trust_tier = random.choice(["T1", "T2"])
        elif target_verdict in ("escalate_tier_1", "escalate_tier_2"):
            vs = self._sample_value_scores(net_min=5, net_max=18)
            al = {"doctrine_alignment": rn(0.5, 0.9), "ethos_alignment": rn(0.5, 0.9), "first_principles_alignment": rn(0.5, 0.8)}
            trust_tier = random.choice(["T1", "T2"])
        else:  # escalate_tier_3
            vs = self._sample_value_scores(net_min=8, net_max=20)
            al = {"doctrine_alignment": rn(0.6, 0.95), "ethos_alignment": rn(0.6, 0.95), "first_principles_alignment": rn(0.6, 0.9)}
            trust_tier = "T2"

        net = self._calc_net(vs)
        trust_total = self._calc_trust_total(vs)
        priority = self._calc_priority(net, trust_total, al)

        return {
            "value_scores": vs,
            "alignment_scores": al,
            "trust_scores": vs,  # reuse for simplicity — trust fields subset
            "net_value_score": net,
            "trust_total": trust_total,
            "trust_tier": trust_tier,
            "priority_score": priority,
            "has_missing_data": target_verdict == "needs_data" and random.random() < 0.7,
            "ethical_conflict": target_verdict == "block" and random.random() < 0.5,
            "intelligence_profile": self._sample_intel_profile(target_verdict),
        }

    def _sample_value_scores(
        self,
        net_min: int = -20,
        net_max: int = 28,
        trust_min: int = 0,
        force_ethical: bool = False,
    ) -> dict:
        while True:
            vs = {
                "revenue_impact": ri(1, 5),
                "cost_efficiency": ri(1, 5),
                "time_leverage": ri(1, 5),
                "strategic_alignment": ri(1, 5),
                "customer_human_benefit": ri(1, 5),
                "knowledge_asset_creation": ri(1, 5),
                "compounding_potential": ri(1, 5),
                "reversibility": ri(1, 5),
                "downside_risk": ri(0, 5),
                "execution_drag": ri(0, 5),
                "uncertainty": ri(0, 5),
                "ethical_misalignment": ri(2, 5) if force_ethical else ri(0, 2),
                # Trust fields (reused)
                "evidence_quality": ri(1, 5),
                "logic_integrity": ri(1, 5),
                "outcome_history": ri(1, 5),
                "context_fit": ri(1, 5),
                "stakeholder_clarity": ri(1, 5),
                "risk_containment": ri(1, 5),
                "auditability": ri(1, 5),
            }
            net = self._calc_net(vs)
            trust_total = sum(vs[k] for k in ["evidence_quality","logic_integrity","outcome_history","context_fit","stakeholder_clarity","risk_containment","auditability"])
            if net_min <= net <= net_max and trust_total >= trust_min:
                return vs

    def _calc_net(self, vs: dict) -> int:
        gross = sum(vs.get(k, 0) for k in [
            "revenue_impact","cost_efficiency","time_leverage","strategic_alignment",
            "customer_human_benefit","knowledge_asset_creation","compounding_potential","reversibility"
        ])
        penalty = sum(vs.get(k, 0) for k in ["downside_risk","execution_drag","uncertainty","ethical_misalignment"])
        return gross - penalty

    def _calc_trust_total(self, vs: dict) -> int:
        return sum(vs.get(k, 0) for k in [
            "evidence_quality","logic_integrity","outcome_history","context_fit",
            "stakeholder_clarity","risk_containment","auditability"
        ])

    def _calc_priority(self, net: int, trust_total: int, al: dict) -> float:
        alignment_composite = sum(al.values()) / len(al)
        return max(0, min(100, (net / 28.0 * 40) + (trust_total / 35.0 * 35) + (alignment_composite * 25)))

    def _sample_intel_profile(self, verdict: str) -> dict:
        quality = {"auto_execute": 0.8, "block": 0.1, "needs_data": 0.4}.get(verdict, 0.5)
        noise = lambda: max(0, min(1, quality + rn(-0.2, 0.2)))
        composite = quality + rn(-0.1, 0.1)
        return {
            "composite": max(0, min(1, composite)),
            "dimensions": {
                "subject": noise(), "training": noise(), "education": noise(),
                "science": noise(), "knowledge": noise(), "strategy": noise(),
                "success": noise(), "tools": noise(), "information": noise(),
                "experience": noise(),
            }
        }

    # ─────────────────────────────────────────────
    # HISTORICAL DATA LOADING
    # ─────────────────────────────────────────────

    def _load_journal_records(self) -> list[TrainingRecord]:
        """Load decision journal Markdown files → TrainingRecord."""
        records = []
        if not self.journal_dir.exists():
            return records

        for md_file in self.journal_dir.rglob("*.md"):
            try:
                content = md_file.read_text()
                d = self._parse_journal_entry(content, md_file.stem)
                if d:
                    feat = self.encode_decision(d)
                    verdict = d.get("verdict", "block")
                    records.extend(self._decision_to_records(d, feat, verdict, "journal"))
            except Exception as e:
                logger.debug("Skipping journal file %s: %s", md_file, e)

        return records

    def _load_outcome_records(self) -> list[TrainingRecord]:
        """Load learning loop JSONL → TrainingRecord (outcome-supervised)."""
        records = []
        if not self.outcomes_file.exists():
            return records

        with open(self.outcomes_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    d = entry.get("decision", {})
                    outcome = entry.get("outcome", {})
                    if not d:
                        continue
                    feat = self.encode_decision(d)
                    verdict = d.get("recommended_action", entry.get("verdict", "block"))
                    recs = self._decision_to_records(d, feat, verdict, "historical")
                    # Upweight records with confirmed outcomes
                    for r in recs:
                        r.weight = 2.0 if outcome.get("confirmed", False) else 1.0
                    records.extend(recs)
                except Exception:
                    pass

        return records

    def _parse_journal_entry(self, content: str, filename: str) -> Optional[dict]:
        """Extract key fields from a journal Markdown entry."""
        import re
        d: dict = {}

        # Extract verdict from filename (format: DEC-xxx_verdict.md)
        m = re.search(r'_(\w+)\.md$', filename)
        if m:
            d["verdict"] = m.group(1)

        # Extract numeric fields from Markdown tables
        patterns = {
            "net_value_score": r"Net Value.*?(-?\d+)",
            "priority_score": r"Priority.*?(\d+\.?\d*)",
            "trust_total": r"Trust Total.*?(\d+)",
            "trust_tier": r"Trust Tier.*?(T\d)",
        }
        for k, pattern in patterns.items():
            m2 = re.search(pattern, content, re.IGNORECASE)
            if m2:
                val = m2.group(1)
                try:
                    d[k] = int(val) if k in ("net_value_score", "trust_total") else float(val) if k == "priority_score" else val
                except ValueError:
                    d[k] = val

        return d if len(d) >= 2 else None

    # ─────────────────────────────────────────────
    # RECORD FACTORY
    # ─────────────────────────────────────────────

    def _decision_to_records(
        self, d: dict, feat: torch.Tensor, verdict: str, source: str
    ) -> list[TrainingRecord]:
        """Create one TrainingRecord per model from a decision dict + feature tensor."""
        records = []
        verdict_idx = self.VERDICT_MAP.get(verdict, 5)
        trust_str = str(d.get("trust_tier", "T0")).upper()
        trust_idx = self.TRUST_MAP.get(trust_str, 0)
        net = float(d.get("net_value_score", 0))
        priority = float(d.get("priority_score", 0)) / 100.0
        domain = d.get("domain", "general")
        dec_id = d.get("decision_id", "")

        # classifier → 6-class verdict
        records.append(TrainingRecord(
            model_name="classifier",
            input_tensor=feat.clone(),
            label=torch.tensor(verdict_idx, dtype=torch.long),
            weight=self._class_weight(verdict_idx),
            decision_id=dec_id, verdict=verdict, domain=domain, source=source,
        ))

        # scorer → net value regression
        records.append(TrainingRecord(
            model_name="scorer",
            input_tensor=feat.clone(),
            label=torch.tensor([(net + 20) / 48.0], dtype=torch.float32),
            weight=1.0,
            decision_id=dec_id, verdict=verdict, domain=domain, source=source,
        ))

        # trust_assessor → 5-class trust tier
        records.append(TrainingRecord(
            model_name="trust_assessor",
            input_tensor=feat.clone(),
            label=torch.tensor(trust_idx, dtype=torch.long),
            weight=1.0,
            decision_id=dec_id, verdict=verdict, domain=domain, source=source,
        ))

        # memory_encoder → embedding (self-supervised: reconstruct feature vector)
        records.append(TrainingRecord(
            model_name="memory_encoder",
            input_tensor=feat.clone(),
            label=feat.clone(),  # reconstruction target
            weight=1.0,
            decision_id=dec_id, verdict=verdict, domain=domain, source=source,
        ))

        # pattern_detector → binary (1 if auto_execute or block — definitive pattern)
        is_pattern = 1 if verdict in ("auto_execute", "block") else 0
        records.append(TrainingRecord(
            model_name="pattern_detector",
            input_tensor=feat.clone(),
            label=torch.tensor([float(is_pattern)], dtype=torch.float32),
            weight=1.5 if is_pattern else 1.0,
            decision_id=dec_id, verdict=verdict, domain=domain, source=source,
        ))

        # causal_predictor → outcome probability (1=good outcome, 0=poor)
        good_outcome = 1.0 if verdict == "auto_execute" else (0.3 if verdict == "block" else 0.6)
        good_outcome += rn(-0.1, 0.1)
        good_outcome = max(0.0, min(1.0, good_outcome))
        records.append(TrainingRecord(
            model_name="causal_predictor",
            input_tensor=feat.clone(),
            label=torch.tensor([good_outcome], dtype=torch.float32),
            weight=2.0 if source in ("historical", "journal") else 1.0,
            decision_id=dec_id, verdict=verdict, domain=domain, source=source,
        ))

        return records

    def _class_weight(self, verdict_idx: int) -> float:
        """Upweight rare classes (block, escalate_tier_3)."""
        weights = [1.0, 1.5, 1.5, 2.0, 2.5, 1.2]  # by verdict index
        return weights[verdict_idx] if verdict_idx < len(weights) else 1.0


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def ri(a: int, b: int) -> int:
    """Random int inclusive."""
    return random.randint(a, b)

def rn(a: float, b: float) -> float:
    """Random float in [a, b]."""
    return random.uniform(a, b)
