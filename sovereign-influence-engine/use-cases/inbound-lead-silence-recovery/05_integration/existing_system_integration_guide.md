# Existing System Integration Guide

## Integration target
This feature should slot into the existing decision and execution layers, not a separate standalone service unless the current architecture forces it.

## Recommended module layout
- `services/sie/silence_recovery/ingest`
- `services/sie/silence_recovery/decision`
- `services/sie/silence_recovery/execution`
- `services/sie/silence_recovery/outcomes`
- `services/sie/silence_recovery/calibration`

## Core integration touchpoints
1. CRM/contact source adapter
2. communication provider adapter
3. activity/task logging adapter
4. event bus or job scheduler
5. DB models/migrations
6. metrics/logging
7. feature flag / tenant policy

## Feature flag recommendation
Add:
- `SIE_SILENCE_RECOVERY_ENABLED`
- `SIE_SILENCE_RECOVERY_DRY_RUN`
- `SIE_SILENCE_RECOVERY_MAX_DAILY_ACTIONS`

## Rollout strategy
- dry run
- observe decision proposals
- enable execution for limited tenant or internal pipeline
- expand only after outcome validation
