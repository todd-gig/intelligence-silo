"""FastAPI server — exposes the intelligence node as a network service for mesh communication."""

from __future__ import annotations

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Intelligence Silo Node", version="0.1.0")

# Node is initialized at startup (see `create_app`)
_node = None


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


def create_app(config_path: str = "config/silo.yaml") -> FastAPI:
    """Create the FastAPI app with an initialized node."""
    from .node import IntelligenceNode

    global _node
    _node = IntelligenceNode(config_path=config_path)

    @app.get("/health")
    async def health():
        return _node.health()

    @app.post("/process", response_model=ProcessResponse)
    async def process(req: ProcessRequest):
        input_ids = None
        if req.input_ids:
            input_ids = torch.tensor(req.input_ids)

        query_emb = None
        if req.query_embedding:
            query_emb = torch.tensor(req.query_embedding).unsqueeze(0)

        if req.deliberate:
            result = _node.deliberate(req.input_data, input_ids, query_emb)
        else:
            result = _node.process(req.input_data, input_ids, query_emb)

        return ProcessResponse(**result)

    @app.post("/memory/store")
    async def memory_store(req: MemoryStoreRequest):
        emb = torch.tensor(req.embedding)
        _node.memory.encode_and_store(req.key, emb, req.context, req.priority)
        return {"stored": True, "key": req.key}

    @app.post("/memory/query")
    async def memory_query(req: MemoryQueryRequest):
        emb = torch.tensor(req.embedding)
        results = _node.memory.query_flat(emb, top_k=req.top_k)
        return {"results": results}

    @app.post("/memory/consolidate")
    async def memory_consolidate():
        stats = _node.consolidate_memory()
        return stats

    @app.get("/matrix/info")
    async def matrix_info():
        return _node.matrix.performance_report()

    @app.get("/society/health")
    async def society_health():
        return _node.society.health()

    return app
