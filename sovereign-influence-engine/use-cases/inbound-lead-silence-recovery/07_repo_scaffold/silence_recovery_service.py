from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class LeadSilenceState:
    lead_id: str
    stage: str
    deal_value: float
    days_since_last_touch: int
    previous_attempts: int
    prior_response_rate: float = 0.0
    owner_assigned: bool = True
    recent_open_signal: bool = False
    recent_click_signal: bool = False
    meeting_status: str = "none"
    status: str = "active"

DEFAULT_WEIGHTS = {
    "normalized_deal_value": 0.30,
    "silence_score": 0.30,
    "engagement_decay_score": 0.20,
    "stage_score": 0.20,
}

STAGE_SCORES = {
    "post_demo": 1.0,
    "pricing_sent": 0.9,
    "discovery": 0.7,
    "outreach": 0.5,
    "other": 0.4,
}

def normalize_deal_value(value: float, cap: float = 50000.0) -> float:
    return max(0.0, min(value / cap, 1.0))

def silence_score(days_since_last_touch: int, max_days: int = 14) -> float:
    return max(0.0, min(days_since_last_touch / max_days, 1.0))

def engagement_decay_score(prior_response_rate: float) -> float:
    return 1.0 - max(0.0, min(prior_response_rate, 1.0))

def get_stage_score(stage: str) -> float:
    return STAGE_SCORES.get(stage, STAGE_SCORES["other"])

def evaluate_action(state: LeadSilenceState, high_value_threshold: float = 10000.0) -> Dict[str, Any]:
    if state.status in {"disqualified", "unsubscribed", "bounced", "closed"}:
        return {"selected_action": "do_not_execute", "policy_gate_result": "blocked"}

    if not state.owner_assigned:
        return {"selected_action": "create_task", "policy_gate_result": "approved", "reason": "owner_missing"}

    score = (
        DEFAULT_WEIGHTS["normalized_deal_value"] * normalize_deal_value(state.deal_value)
        + DEFAULT_WEIGHTS["silence_score"] * silence_score(state.days_since_last_touch)
        + DEFAULT_WEIGHTS["engagement_decay_score"] * engagement_decay_score(state.prior_response_rate)
        + DEFAULT_WEIGHTS["stage_score"] * get_stage_score(state.stage)
    )

    if state.days_since_last_touch >= 7 and state.deal_value >= high_value_threshold:
        action = "escalate_to_human"
        authority = "D3"
        gate = "requires_human_review"
    elif state.days_since_last_touch >= 3 and state.previous_attempts < 3:
        action = "send_email"
        authority = "D1"
        gate = "approved"
    elif state.previous_attempts >= 3:
        action = "create_task"
        authority = "D2"
        gate = "approved"
    else:
        action = "do_not_execute"
        authority = "D0"
        gate = "blocked"

    return {
        "selected_action": action,
        "authority_level": authority,
        "policy_gate_result": gate,
        "priority_score": round(score, 4),
    }
