# Gigaton SIE First Production Bundle v1

Purpose: give Claude everything needed to design, develop, and integrate the first live production use case of the Sovereign Influence Engine into the existing Gigaton system.

## Use case
Inbound Lead Silence Recovery Engine

## Strategic objective
Deploy the smallest possible governed decision loop that:
1. ingests live lead/contact state,
2. decides the next action,
3. executes that action,
4. captures the outcome,
5. calibrates future decisions.

## Why this exists
The current Gigaton project already defines the Sovereign Influence Engine as a closed-loop decision system with authority, memory, calibration, and execution layers. This bundle narrows that into the first production wedge for fastest ROI and lowest integration friction.

## Recommended Claude upload order
1. `00_START_HERE/claude_bootstrap.md`
2. `00_START_HERE/system_prompt_contract.md`
3. `01_context/source_grounding.md`
4. `01_context/executive_brief.md`
5. `02_architecture/minimum_viable_influence_engine.md`
6. `03_use_case/inbound_lead_silence_recovery_spec.md`
7. `04_schemas/*.json`
8. `05_integration/*.md`
9. `06_execution_plan/*.md`
10. `07_repo_scaffold/*`

## Output expectation for Claude
Claude should use this bundle to:
- map the new use case to the existing SIE architecture,
- identify gaps in the current system,
- generate implementation code and migrations,
- integrate into existing backend/services/event pipelines,
- keep provider-specific logic behind adapters,
- preserve sovereign decision logic, authority, calibration, and auditability.

## Constraint stack
- one production use case only
- deployable within 72 hours
- real execution, not advisory output
- measurable revenue or pipeline impact
- no unnecessary platform expansion before live validation
