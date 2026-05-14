# Governance Notes

## Authority recommendation
- D1 actions may auto-execute.
- D2 actions may auto-execute if tenant policy allows.
- D3+ actions require human review.

## Audit minimums
Persist:
- input state snapshot
- selected rule path
- score components
- chosen action
- policy gate result
- execution result
- outcome attribution

## Tenant policy knobs
- max daily emails
- allowed channels
- high-value escalation threshold
- dry-run default
- approved template set
