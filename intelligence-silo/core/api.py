"""FastAPI server — exposes the intelligence node as a network service for mesh communication.

Environment variables:
    SILO_CONFIG_PATH   Path to silo.yaml (default: config/silo.yaml)
    PORT               Uvicorn listen port when run directly (default: 8080)
"""

from __future__ import annotations

import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ── Request / Response schemas ────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    input_data: dict
    input_ids: list[list[int]] | None = None
    query_embedding: list[float] | None = None
    deliberate: bool = False


class ProcessResponse(BaseModel):
    node_id: str
    consensus: bool
    confidence: float
    verdict: dict | None = None
    minds_activated: int = 0
    cycle_time_ms: float = 0.0


class MemoryStoreRequest(BaseModel):
    key: str
    embedding: list[float]
    context: dict = Field(default_factory=dict)
    priority: float = 1.0


class MemoryQueryRequest(BaseModel):
    embedding: list[float]
    top_k: int = 5


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(config_path: str = "config/silo.yaml") -> FastAPI:
    """Create and fully configure the FastAPI application with an initialized node."""
    from .node import IntelligenceNode

    _app = FastAPI(
        title="Intelligence Silo Node",
        version="0.1.0",
        description=(
            "Society of Minds neural intelligence node — SLM matrix, "
            "hierarchical memory, and multi-agent deliberation."
        ),
    )

    node = IntelligenceNode(config_path=config_path)

    @_app.get("/health", tags=["meta"])
    async def health():
        return node.health()

    @_app.get("/training/status", tags=["training"])
    async def training_status():
        """Return the current retrain trigger status and progress."""
        return node.training_status()

    @_app.post("/process", response_model=ProcessResponse, tags=["inference"])
    async def process(req: ProcessRequest):
        input_ids = None
        if req.input_ids:
            input_ids = torch.tensor(req.input_ids)

        query_emb = None
        if req.query_embedding:
            query_emb = torch.tensor(req.query_embedding).unsqueeze(0)

        if req.deliberate:
            result = node.deliberate(req.input_data, input_ids, query_emb)
        else:
            result = node.process(req.input_data, input_ids, query_emb)

        return ProcessResponse(**result)

    @_app.post("/decisions/record", tags=["decisions"])
    async def record_decision(body: dict):
        """Record a pipeline result into the memory hierarchy."""
        pipeline_result = body.get("pipeline_result", body)
        title = body.get("title", "")
        domain = body.get("domain", "general")
        return node.record_decision(pipeline_result, title=title, domain=domain)

    @_app.post("/memory/store", tags=["memory"])
    async def memory_store(req: MemoryStoreRequest):
        emb = torch.tensor(req.embedding)
        node.memory.encode_and_store(req.key, emb, req.context, req.priority)
        return {"stored": True, "key": req.key}

    @_app.post("/memory/query", tags=["memory"])
    async def memory_query(req: MemoryQueryRequest):
        emb = torch.tensor(req.embedding)
        results = node.memory.query_flat(emb, top_k=req.top_k)
        return {"results": results}

    @_app.post("/memory/consolidate", tags=["memory"])
    async def memory_consolidate():
        stats = node.consolidate_memory()
        return stats

    @_app.get("/matrix/info", tags=["matrix"])
    async def matrix_info():
        return node.matrix.performance_report()

    @_app.get("/society/health", tags=["society"])
    async def society_health():
        return node.society.health()

    return _app


# ── Module-level app — uvicorn loads this directly ────────────────────────────
# create_app() is called here so that `uvicorn core.api:app` gets a fully
# initialized application with all routes registered and the node running.

app = create_app(
    config_path=os.environ.get("SILO_CONFIG_PATH", "config/silo.yaml")
)
