# Risks and Controls

## Risk: over-automation
Control:
- authority ceiling
- feature flags
- max daily action limits
- dry-run mode

## Risk: poor messaging quality
Control:
- small approved template set
- human review for early runs
- log outcomes by template

## Risk: bad input quality
Control:
- trust score floor
- quarantine malformed/incomplete records
- do not execute when owner or contact data is missing

## Risk: system sprawl
Control:
- no expansion beyond this use case until KPI threshold is met

## Risk: calibration drift
Control:
- bounded deltas
- compare cycle-over-cycle performance
- keep baseline weights recoverable
