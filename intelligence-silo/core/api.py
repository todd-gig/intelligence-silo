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


class MemoryIngestRequest(BaseModel):
    """Ingest raw text directly — no pre-embedding required.

    The silo chunks the document, embeds each chunk via sentence-transformers,
    and stores everything in the FAISS semantic index.
    """
    text: str
    source: str = Field(
        default="",
        description="Document identifier — filename, URL, or label. "
                    "Stored with every chunk so search results cite their origin.",
    )
    category: str = Field(
        default="general",
        description="Tag for filtered retrieval (e.g. 'architecture', 'doctrine', 'playbook').",
    )
    author: str = Field(
        default="",
        description="Who produced this content (name or email). Stored in chunk metadata.",
    )
    is_markdown: bool | None = Field(
        default=None,
        description="Force markdown parsing. None = auto-detect from content.",
    )
    priority: float = Field(default=1.0, ge=0.0, le=2.0)


class MemoryIngestResponse(BaseModel):
    chunks_stored: int
    source: str
    category: str
    chunk_ids: list[str]


class MemorySearchRequest(BaseModel):
    """Query shared memory with a plain text question.

    Returns ranked results from the FAISS semantic index with source citations
    so the caller knows which document and section each result came from.
    """
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = Field(
        default=None,
        description="Filter results to a specific category. None = search all.",
    )
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)


class MemorySearchResult(BaseModel):
    text: str
    source: str
    heading: str
    category: str
    score: float
    chunk_id: str


class MemorySearchResponse(BaseModel):
    query: str
    results: list[MemorySearchResult]
    total_found: int


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

    @_app.post("/memory/ingest", response_model=MemoryIngestResponse, tags=["memory"])
    async def memory_ingest(req: MemoryIngestRequest):
        """Ingest a document into shared semantic memory.

        Accepts raw text (markdown or plain). The silo chunks it, embeds each
        chunk using sentence-transformers (all-MiniLM-L6-v2, 384 dim), and
        stores every chunk in the persistent FAISS index.

        Chunks are immediately queryable via POST /memory/search.
        The index is persisted to disk and synced to GCS if configured.
        """
        from .memory.embedder import chunk_document, embed_batch
        import numpy as np

        chunks = chunk_document(req.text, source=req.source, is_markdown=req.is_markdown)
        if not chunks:
            return MemoryIngestResponse(
                chunks_stored=0, source=req.source,
                category=req.category, chunk_ids=[],
            )

        texts = [c["text"] for c in chunks]
        embeddings = embed_batch(texts)  # (N, 384) float32

        chunk_ids = []
        for chunk, emb in zip(chunks, embeddings):
            knowledge = {
                "text": chunk["text"],
                "heading": chunk["heading"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_id"],
                "author": req.author,
            }
            entry = node.memory.semantic.store(
                embedding=emb,
                knowledge=knowledge,
                category=req.category,
                confidence=min(req.priority, 1.0),
            )
            chunk_ids.append(entry.id)

        # Persist index so it survives restarts and syncs to GCS
        node.memory.semantic.save()

        return MemoryIngestResponse(
            chunks_stored=len(chunk_ids),
            source=req.source,
            category=req.category,
            chunk_ids=chunk_ids,
        )

    @_app.post("/memory/search", response_model=MemorySearchResponse, tags=["memory"])
    async def memory_search(req: MemorySearchRequest):
        """Search shared memory with a plain text query.

        Embeds the query and runs approximate nearest-neighbor search over the
        FAISS index. Returns ranked results with source citations (document name,
        section heading, relevance score).

        This is the primary access point for Matt or any node to query
        shared intelligence without needing to pre-compute embeddings.
        """
        from .memory.embedder import embed_text
        import numpy as np

        query_vec = embed_text(req.query)  # (384,) float32

        raw = node.memory.semantic.search(
            query=query_vec,
            top_k=req.top_k * 2,  # over-fetch then filter by min_score
            category=req.category,
        )

        results = []
        for entry, score in raw:
            if score < req.min_score:
                continue
            k = entry.knowledge
            results.append(MemorySearchResult(
                text=k.get("text", ""),
                source=k.get("source", ""),
                heading=k.get("heading", ""),
                category=entry.category,
                score=round(float(score), 4),
                chunk_id=entry.id,
            ))
            if len(results) >= req.top_k:
                break

        return MemorySearchResponse(
            query=req.query,
            results=results,
            total_found=len(results),
        )

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
