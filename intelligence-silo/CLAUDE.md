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
