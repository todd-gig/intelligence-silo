# Architecture Delta Map

## Existing system assumptions
The current Gigaton stack likely already has some or all of:
- event ingestion
- memory or transcript ingestion
- scoring or trust primitives
- service-layer execution concepts
- markdown/system prompt infrastructure
- API and deployment scaffolds from prior bundles

## New minimum additions for this use case
### Must add
- `lead_silence_state` derivation
- `follow_up_decision` object
- decision evaluator service
- email/message execution adapter
- CRM action logging hook
- escalation hook
- outcome capture events
- calibration job

### Should reuse, not rebuild
- auth
- tenanting
- event bus if available
- DB connection and migrations
- observability patterns
- deployment pipeline
- approval/authority primitives where they already exist

## Integration principle
Integrate this as a bounded module inside the existing SIE/decision execution stack, not as a parallel application.
