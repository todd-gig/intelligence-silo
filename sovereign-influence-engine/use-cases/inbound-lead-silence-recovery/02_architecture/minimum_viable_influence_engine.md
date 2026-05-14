# Minimum Viable Influence Engine

## Goal
Deploy the smallest governed execution loop that preserves core SIE properties.

## Required layers
1. Signal ingestion
2. Entity resolution
3. Decision scoring
4. Authority/policy gate
5. Action execution
6. Outcome capture
7. Calibration

## Explicitly excluded from v1
- full ontology expansion
- advanced multi-agent orchestration
- broad dashboarding beyond operational visibility
- generalized cross-domain action engine
- complete memory manifold implementation

## Required architectural behavior
- deterministic enough to audit
- flexible enough to calibrate
- narrow enough to deploy in 72 hours
- compatible with future SIE layers

## Mapping to existing Gigaton primitives
- Trust Matrix + Value Matrix -> scoring and prioritization
- Executive Decision Engine -> policy-gated decision logic
- Decision Execution Engine -> send/escalate/log actions
- EO System -> future enhancement path for adaptive enrichment
- Portable Prompt Contract -> consistent behavior across model-assisted components

## Core loop
1. ingest lead/contact state
2. compute action priority
3. validate against authority thresholds
4. execute next best governed action
5. record event and outcome
6. adjust weights within bounded delta
