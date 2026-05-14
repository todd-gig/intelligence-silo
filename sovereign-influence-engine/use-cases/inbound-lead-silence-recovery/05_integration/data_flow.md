# Data Flow

1. Load candidate leads from CRM/contact source.
2. Derive `lead_silence_state`.
3. Evaluate decision rules and compute `follow_up_decision`.
4. Apply authority and policy gates.
5. Execute approved action through adapter.
6. Log execution and action metadata.
7. Observe outcomes from reply/meeting/deal signals.
8. Emit `decision_outcome_event`.
9. Update weights during calibration cycle.

## Required persistence
- lead state snapshot
- decision object
- execution event
- outcome event
- calibration run history
