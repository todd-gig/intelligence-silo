"""Intelligence Dimensions — the 10 pillars of human wisdom.

These dimensions represent the complete stack of requirements for a human being
to achieve the level of intelligence necessary to make wise, accurate decisions
using developed intelligence systems engineered to accelerate human development.

    SUBJECT     — the domain/field of mastery being applied
    TRAINING    — deliberate practice; the skill acquisition layer
    EDUCATION   — structured knowledge frameworks; how to learn and reason
    SCIENCE     — empirical method; evidence, falsifiability, repeatability
    KNOWLEDGE   — accumulated facts, models, and mental maps of reality
    STRATEGY    — the ability to sequence actions toward a goal across time
    SUCCESS     — pattern recognition of what winning looks like; feedback loops
    TOOLS       — leverage multipliers; systems that extend human capability
    INFORMATION — real-time signal; the raw input that feeds all decisions
    EXPERIENCE  — embodied, tacit knowledge that only time and doing creates

Every decision processed by the engine is scored against these dimensions.
High scores mean the decision is well-grounded in the human intelligence stack.
Low scores flag gaps that should block auto-execution or trigger escalation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class IntelligenceDimension(str, Enum):
    SUBJECT = "subject"
    TRAINING = "training"
    EDUCATION = "education"
    SCIENCE = "science"
    KNOWLEDGE = "knowledge"
    STRATEGY = "strategy"
    SUCCESS = "success"
    TOOLS = "tools"
    INFORMATION = "information"
    EXPERIENCE = "experience"


# Weight of each dimension in the composite intelligence score
# Tuned for decision-making contexts; can be overridden in config
DIMENSION_WEIGHTS: dict[IntelligenceDimension, float] = {
    IntelligenceDimension.SUBJECT:      0.08,
    IntelligenceDimension.TRAINING:     0.09,
    IntelligenceDimension.EDUCATION:    0.08,
    IntelligenceDimension.SCIENCE:      0.10,
    IntelligenceDimension.KNOWLEDGE:    0.12,
    IntelligenceDimension.STRATEGY:     0.13,
    IntelligenceDimension.SUCCESS:      0.10,
    IntelligenceDimension.TOOLS:        0.09,
    IntelligenceDimension.INFORMATION:  0.11,
    IntelligenceDimension.EXPERIENCE:   0.10,
}

# Which dimensions are required for auto-execution (score must be ≥ threshold)
AUTO_EXEC_REQUIRED: set[IntelligenceDimension] = {
    IntelligenceDimension.KNOWLEDGE,
    IntelligenceDimension.STRATEGY,
    IntelligenceDimension.INFORMATION,
    IntelligenceDimension.EXPERIENCE,
}

# Auto-execution threshold per dimension (0–1)
AUTO_EXEC_THRESHOLD = 0.5

# Minimum composite score to allow escalate_tier_1 or lower
ESCALATE_THRESHOLD = 0.3


@dataclass
class IntelligenceProfile:
    """Scores a decision across all 10 intelligence dimensions.

    Each dimension is scored 0.0–1.0:
      0.0 = completely absent / not considered
      0.5 = partially covered
      1.0 = fully grounded in this dimension
    """
    subject: float = 0.0
    training: float = 0.0
    education: float = 0.0
    science: float = 0.0
    knowledge: float = 0.0
    strategy: float = 0.0
    success: float = 0.0
    tools: float = 0.0
    information: float = 0.0
    experience: float = 0.0

    # Optional: which specific subject domain this decision belongs to
    subject_domain: str = ""

    # Optional: which tools are being applied
    active_tools: list[str] = field(default_factory=list)

    def composite(self) -> float:
        """Weighted composite score across all 10 dimensions (0–1)."""
        scores = {
            IntelligenceDimension.SUBJECT: self.subject,
            IntelligenceDimension.TRAINING: self.training,
            IntelligenceDimension.EDUCATION: self.education,
            IntelligenceDimension.SCIENCE: self.science,
            IntelligenceDimension.KNOWLEDGE: self.knowledge,
            IntelligenceDimension.STRATEGY: self.strategy,
            IntelligenceDimension.SUCCESS: self.success,
            IntelligenceDimension.TOOLS: self.tools,
            IntelligenceDimension.INFORMATION: self.information,
            IntelligenceDimension.EXPERIENCE: self.experience,
        }
        return sum(DIMENSION_WEIGHTS[d] * scores[d] for d in IntelligenceDimension)

    def gaps(self, threshold: float = 0.4) -> list[IntelligenceDimension]:
        """Return dimensions scoring below the threshold."""
        scores = self._as_dict()
        return [d for d in IntelligenceDimension if scores[d] < threshold]

    def auto_exec_gaps(self) -> list[IntelligenceDimension]:
        """Return required dimensions that block auto-execution."""
        scores = self._as_dict()
        return [
            d for d in AUTO_EXEC_REQUIRED
            if scores[d] < AUTO_EXEC_THRESHOLD
        ]

    def blocks_auto_exec(self) -> bool:
        return len(self.auto_exec_gaps()) > 0

    def dominant_dimensions(self, top_n: int = 3) -> list[IntelligenceDimension]:
        """Return the top N highest-scoring dimensions."""
        scores = self._as_dict()
        return sorted(IntelligenceDimension, key=lambda d: scores[d], reverse=True)[:top_n]

    def as_memory_tags(self) -> list[str]:
        """Convert to memory tags for categorization in the silo."""
        tags = []
        if self.subject_domain:
            tags.append(f"subject:{self.subject_domain}")
        for d in self.dominant_dimensions(top_n=3):
            tags.append(f"intel:{d.value}")
        if self.active_tools:
            tags.extend(f"tool:{t}" for t in self.active_tools)
        return tags

    def _as_dict(self) -> dict[IntelligenceDimension, float]:
        return {
            IntelligenceDimension.SUBJECT: self.subject,
            IntelligenceDimension.TRAINING: self.training,
            IntelligenceDimension.EDUCATION: self.education,
            IntelligenceDimension.SCIENCE: self.science,
            IntelligenceDimension.KNOWLEDGE: self.knowledge,
            IntelligenceDimension.STRATEGY: self.strategy,
            IntelligenceDimension.SUCCESS: self.success,
            IntelligenceDimension.TOOLS: self.tools,
            IntelligenceDimension.INFORMATION: self.information,
            IntelligenceDimension.EXPERIENCE: self.experience,
        }

    def to_dict(self) -> dict:
        return {
            "composite": round(self.composite(), 4),
            "dimensions": {d.value: round(self._as_dict()[d], 4) for d in IntelligenceDimension},
            "subject_domain": self.subject_domain,
            "active_tools": self.active_tools,
            "gaps": [d.value for d in self.gaps()],
            "auto_exec_gaps": [d.value for d in self.auto_exec_gaps()],
            "memory_tags": self.as_memory_tags(),
        }


def infer_profile_from_decision(decision_dict: dict) -> IntelligenceProfile:
    """Heuristically infer an IntelligenceProfile from a decision's context.

    Used when the operator hasn't explicitly scored the 10 dimensions.
    Derives scores from existing pipeline fields: trust, value, alignment,
    certificate chain completeness, and metadata keywords.
    """
    # Normalize existing pipeline signals
    trust_total = min(1.0, decision_dict.get("trust_total", 0) / 35.0)
    net_value = min(1.0, max(0, decision_dict.get("net_value_score", 0)) / 28.0)
    alignment = min(1.0, max(0, decision_dict.get("alignment_composite", 0.0)))
    priority = min(1.0, decision_dict.get("priority_score", 0.0) / 100.0)

    cert_status = decision_dict.get("certificate_status", {})
    chain_depth = sum(1 for v in cert_status.values() if v == "issued") / max(1, len(cert_status) or 4)

    summary = (decision_dict.get("executive_summary", "") + " " +
               decision_dict.get("title", "")).lower()

    # Keyword signals
    has_data = any(w in summary for w in ["data", "metrics", "analysis", "report", "measured"])
    has_plan = any(w in summary for w in ["plan", "strategy", "roadmap", "sequence", "phase"])
    has_tools = any(w in summary for w in ["tool", "system", "platform", "software", "api", "model"])
    has_experience = any(w in summary for w in ["previous", "history", "pattern", "learned", "past"])
    has_science = any(w in summary for w in ["evidence", "test", "hypothesis", "research", "study"])

    return IntelligenceProfile(
        subject=min(1.0, trust_total * 0.6 + chain_depth * 0.4),
        training=min(1.0, trust_total * 0.5 + priority * 0.5),
        education=min(1.0, alignment * 0.5 + chain_depth * 0.5),
        science=min(1.0, (0.8 if has_science else 0.3) * trust_total + 0.2 * chain_depth),
        knowledge=min(1.0, (0.7 if has_data else 0.4) * trust_total + 0.3 * net_value),
        strategy=min(1.0, (0.8 if has_plan else 0.4) * priority + 0.2 * alignment),
        success=min(1.0, net_value * 0.6 + priority * 0.4),
        tools=min(1.0, (0.9 if has_tools else 0.3) + 0.1 * trust_total),
        information=min(1.0, (0.8 if has_data else 0.4) * chain_depth + 0.2 * trust_total),
        experience=min(1.0, (0.8 if has_experience else 0.3) * trust_total + 0.2 * priority),
        subject_domain=decision_dict.get("domain", "general"),
    )
