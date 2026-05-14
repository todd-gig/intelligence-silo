-- 001_sie_silence_recovery.sql

CREATE TABLE IF NOT EXISTS sie_follow_up_decisions (
  decision_id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL,
  decision_type TEXT NOT NULL,
  priority_score REAL NOT NULL,
  trust_score REAL NOT NULL,
  authority_level TEXT NOT NULL,
  selected_action TEXT NOT NULL,
  policy_gate_result TEXT NOT NULL,
  status TEXT NOT NULL,
  context_json TEXT NOT NULL,
  action_payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  executed_at TEXT
);

CREATE TABLE IF NOT EXISTS sie_decision_outcomes (
  outcome_id TEXT PRIMARY KEY,
  decision_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  outcome_type TEXT NOT NULL,
  revenue_impact REAL,
  time_to_outcome_hours REAL,
  metadata_json TEXT,
  observed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sie_calibration_runs (
  run_id TEXT PRIMARY KEY,
  model_name TEXT NOT NULL,
  weights_before_json TEXT NOT NULL,
  weights_after_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  executed_at TEXT NOT NULL
);
