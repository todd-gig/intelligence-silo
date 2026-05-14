# Inbound Lead Silence Recovery Engine Spec

## Objective
Recover silent leads by triggering the highest-ROI next action when a lead has stalled after outreach, demo, pricing, or follow-up.

## Scope
### Included
- post-demo silence
- post-pricing silence
- stalled outbound/inbound thread
- high-value dormant lead reactivation

### Excluded
- full sales automation
- contract generation
- payment workflows
- multi-channel orchestration beyond the first execution hook set

## Actors
- lead/contact
- account/opportunity
- assigned rep
- SIE decision service
- execution adapter
- CRM/activity log
- calibration job

## Trigger conditions
A lead enters evaluation when:
- no reply within threshold window, or
- no meeting booked after defined stage, or
- follow-up sequence halted without resolution

## Candidate actions
- send_email
- send_message
- create_task
- escalate_to_human
- schedule_follow_up

## Success outcomes
- reply_received
- meeting_booked
- opportunity_revived
- deal_closed

## Failure / neutral outcomes
- no_reply
- bounce
- unsubscribed
- meeting_declined
- lead_disqualified

## First deployment recommendation
Start with:
- send_email
- create_task
- escalate_to_human

Add messaging only if the communication infrastructure already exists.
