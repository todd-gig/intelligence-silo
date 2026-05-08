# Intelligence Silo

Neural intelligence node implementing a Society of Minds architecture with hierarchical memory and a matrix of small language models (SLMs).

## Architecture

```
Input → Society of Minds Orchestrator
          ├── Perceiver → structures raw input
          ├── Analyst → scores via SLM matrix
          ├── Critic → challenges proposals
          ├── Synthesizer → builds consensus
          ├── Executor → plans actions
          ├── Memory Keeper → consolidates knowledge
          └── Sentinel → monitors anomalies
              │
              ├── SLM Matrix (6 specialist PyTorch models)
              │   ├── classifier (decision categories)
              │   ├── scorer (value/penalty prediction)
              │   ├── trust_assessor (T0-T4 trust tiers)
              │   ├── memory_encoder (embedding generation)
              │   ├── pattern_detector (recurring patterns)
              │   └── causal_predictor (downstream effects)
              │
              └── Memory Hierarchy
                  ├── Working (tensor-backed, attention-gated, TTL)
                  ├── Episodic (experience-indexed, cosine similarity)
                  ├── Semantic (FAISS vector index, persistent)
                  └── Procedural (learned actions, confidence-gated)
```

## First Principle

**Local computation always has inherent advantages.** Information stored and processed on the local machine will always be faster, more private, and more available than remote alternatives. Google services are fallbacks, not primaries.

## Key Commands

```bash
# Run tests
cd intelligence-silo && python -m pytest tests/ -v

# Show matrix info
python cli.py matrix-info

# Run test think cycle
python cli.py test

# Show health
python cli.py health

# Package for distribution
python cli.py package --name intel-node --signing-key <key>
```

## Integration

- **Decision Engine Bridge**: Imports weights from `decision-engine/config/engine.yaml`, exports neural predictions via FastAPI
- **Google Services**: GCS for shared memory sync, Vertex AI for fallback inference, Firebase for mesh state
- **Node Mesh**: Nodes discover each other via mDNS, share semantic memory via GCS, verify trust via HMAC signatures

## Tech Stack

- PyTorch (SLM matrix, working memory attention gates)
- FAISS (semantic memory vector index)
- FastAPI (node API)
- safetensors (model weight serialization)
- PyInstaller (executable packaging)

---

## Doctrine alignment

Single source of truth for first principles, methodology, and anti-patterns:

- [`decision-engine/drift_sentinel/GIGATON_CANONICAL_FIRST_PRINCIPLES.md`](https://github.com/todd-gig/decision-engine/blob/main/drift_sentinel/GIGATON_CANONICAL_FIRST_PRINCIPLES.md) — 7 non-negotiables, 15 first principles, 8 ethos filters, 17 frameworks, 12 anti-patterns
- [`decision-engine/MASTER_FIRST_PRINCIPLES_REFERENCE.md`](https://github.com/todd-gig/decision-engine/blob/main/MASTER_FIRST_PRINCIPLES_REFERENCE.md) — thresholds, weights, formulas, decision pathways

If this repo's local-first first-principle ever conflicts with the canonical doc, the canonical doc's "Ethos #5: protect sovereignty" frames the decision: this repo's local-first defaults are the canonical guidance for sovereignty, not in conflict with it.

### Doctrine-driven constraints (apply here)

- **CRIT-003** — every prod LLM call must carry `prompt_version` + `schema_version`
- **CRIT-007** — every LLM call must accept `provider` + `model`; SLM matrix calls already are local-only (no provider needed) but Vertex AI fallback (silo.yaml lines 114-125) MUST satisfy CRIT-003 + CRIT-007 if/when wired
- **B-04 open** — silo's claim to import value/penalty weights from `decision-engine/config/engine.yaml` (per "Decision Engine Bridge" above) is not yet implemented; tracked in BETA_2_GAP_LIST
- **Slack is user-level only** — node mesh, sync daemons, and recurring tasks must not post to Slack
