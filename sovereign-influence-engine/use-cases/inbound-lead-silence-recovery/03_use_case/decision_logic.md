# Decision Logic

## Core question
Given a silent lead, what is the next governed action with the highest expected ROI?

## Minimum input features
- days_since_last_touch
- stage
- deal_value
- previous_attempts
- prior_response_rate
- account_tier
- owner_assigned
- recent_open_signal
- recent_click_signal
- meeting_status
- authority ceiling for automation

## Baseline scoring formula
priority_score =
  value_weight * normalized_deal_value +
  silence_weight * silence_score +
  stage_weight * stage_score +
  engagement_weight * engagement_decay_score +
  recency_weight * interaction_recency_score

## Hard decision rules v1
1. if days_since_last_touch >= 3 and previous_attempts < 3 -> send_email
2. if days_since_last_touch >= 7 and deal_value >= high_value_threshold -> escalate_to_human
3. if previous_attempts >= 3 and no reply -> create_task with alternate strategy note
4. if lead is disqualified/unsubscribed/bounced -> do_not_execute
5. if no owner assigned -> create_task for assignment, do not auto-message

## Policy gate
Automation may only execute actions at or below the approved authority level.
Recommended v1 ceiling:
- D1: send email
- D2: create follow-up task / scheduling prompt
- D3+: human approval required

## Calibration rule
Update action and feature weights only within bounded delta after outcome attribution.
