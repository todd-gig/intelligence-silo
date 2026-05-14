// execution_hooks_example.js

async function sendEmail(executionRequest, mailAdapter) {
  const result = await mailAdapter.send({
    to: executionRequest.payload.email,
    templateId: executionRequest.payload.template_id,
    variables: executionRequest.payload.variables || {},
  });

  return {
    action: "send_email",
    providerMessageId: result.id,
    status: "executed",
  };
}

async function logAction(executionRequest, crmAdapter) {
  return crmAdapter.logActivity({
    entityId: executionRequest.entity_id,
    decisionId: executionRequest.decision_id,
    actionType: executionRequest.action_type,
    payload: executionRequest.payload,
  });
}

async function escalate(executionRequest, taskAdapter) {
  return taskAdapter.createTask({
    entityId: executionRequest.entity_id,
    reason: executionRequest.payload.reason,
    assignedTo: executionRequest.payload.assigned_to,
    decisionId: executionRequest.decision_id,
    urgency: executionRequest.payload.urgency || "high",
  });
}

module.exports = { sendEmail, logAction, escalate };
