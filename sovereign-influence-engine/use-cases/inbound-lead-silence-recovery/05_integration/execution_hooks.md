# Execution Hooks

## Required hooks
### 1. Send email
Endpoint:
`POST /execute/send-email`

Responsibility:
- render approved template
- send through existing provider adapter
- capture provider message id
- emit execution event

### 2. Log CRM action
Endpoint:
`POST /execute/log-action`

Responsibility:
- write activity/note/task to CRM or internal activity store
- attach `decision_id`
- preserve audit trail

### 3. Escalate to human
Endpoint:
`POST /execute/escalate`

Responsibility:
- assign task/alert to owner or queue
- include reason, urgency, and recommended next step
- do not auto-message if policy gate blocks it

### 4. Evaluate scheduled leads
Job:
`evaluate_silent_leads`

Responsibility:
- pull eligible leads on interval
- create decision objects
- execute approved actions
- skip blocked records
- emit metrics

## Adapter principle
Claude should integrate with the existing mail/CRM/tasking adapters if they already exist.
If they do not exist, create thin adapters with no business logic leakage.
