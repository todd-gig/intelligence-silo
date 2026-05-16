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


class WeightsResponse(BaseModel):
    """Canonical decision-engine weights exposed so other services (e.g. gigaton-engine)
    can consume the same scoring parameters without parsing engine.yaml themselves."""

    value_weights: dict
    penalty_weights: dict
    trust_multiplier: dict
    rtql_trust_multiplier: dict
    thresholds: dict
    source: str  # "file" | "http" | "fallback"


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

    # ── Source Registry ──────────────────────────────────────────────────────
    # Per Unified Source Registry doctrine (2026-05-14). Single canonical
    # registry of memory sources per user, queried by all downstream engines
    # (HME / PPEME / persona-engine) via SMEN substrate read pattern.
    from . import sources as _sources_module

    _sources_db = os.environ.get(
        "SILO_SOURCES_DB",
        os.path.join(os.path.dirname(__file__), "..", "data", "sources.db"),
    )
    _sources_module.init_db(_sources_db)

    class _SourceAddIn(BaseModel):
        user_id: str = Field(..., min_length=1)
        source_type: str = Field(..., min_length=1)
        status: str = "disconnected"
        consent_state: dict = Field(default_factory=dict)
        provenance_metadata: dict = Field(default_factory=dict)

    class _SourceStatusIn(BaseModel):
        status: str
        error: str | None = None
        mark_synced: bool = False

    class _SourceConsentIn(BaseModel):
        consent_state: dict

    @_app.post("/v1/sources", tags=["sources"], status_code=201)
    async def sources_add(payload: _SourceAddIn):
        from fastapi import HTTPException
        try:
            return _sources_module.add(
                user_id=payload.user_id,
                source_type=payload.source_type,
                status=payload.status,
                consent_state=payload.consent_state,
                provenance_metadata=payload.provenance_metadata,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @_app.get("/v1/sources/coverage/{user_id}", tags=["sources"])
    async def sources_coverage(user_id: str):
        """Summary used by the FE /sources page: connected/total + per-type detail."""
        return _sources_module.coverage_for_user(user_id)

    @_app.get("/v1/sources/{source_id}", tags=["sources"])
    async def sources_get(source_id: str):
        from fastapi import HTTPException
        row = _sources_module.get(source_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"source {source_id!r} not found")
        return row

    @_app.get("/v1/sources", tags=["sources"])
    async def sources_list(
        user_id: str,
        source_type: str | None = None,
        status: str | None = None,
    ):
        items = _sources_module.list_for_user(user_id, source_type=source_type, status=status)
        return {"items": items, "count": len(items)}

    @_app.patch("/v1/sources/{source_id}/status", tags=["sources"])
    async def sources_set_status(source_id: str, payload: _SourceStatusIn):
        from fastapi import HTTPException
        try:
            row = _sources_module.set_status(
                source_id, payload.status,
                error=payload.error, mark_synced=payload.mark_synced,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        if row is None:
            raise HTTPException(status_code=404, detail=f"source {source_id!r} not found")
        return row

    @_app.patch("/v1/sources/{source_id}/consent", tags=["sources"])
    async def sources_update_consent(source_id: str, payload: _SourceConsentIn):
        from fastapi import HTTPException
        row = _sources_module.update_consent(source_id, payload.consent_state)
        if row is None:
            raise HTTPException(status_code=404, detail=f"source {source_id!r} not found")
        return row

    @_app.delete("/v1/sources/{source_id}", tags=["sources"])
    async def sources_remove(source_id: str):
        from fastapi import HTTPException
        if not _sources_module.remove(source_id):
            raise HTTPException(status_code=404, detail=f"source {source_id!r} not found")
        return {"removed": source_id}

    # ── ChatGPT Export Connector (POC) ───────────────────────────────────────
    # Accepts OpenAI's conversations.json from a ChatGPT data export.
    # Walks conversations, windows messages, ingests via node.ingest_memory
    # so embeddings + FAISS storage stay consistent with other memory paths.

    class _ChatGPTExportIn(BaseModel):
        user_id: str = Field(..., min_length=1)
        conversations: list[dict] = Field(default_factory=list)
        source_id: str | None = None
        category: str = "chatgpt_history"
        max_messages_per_chunk: int = 8

    @_app.post("/v1/intelligence/capture/openai-export",
               tags=["sources"], status_code=202)
    async def capture_openai_export(payload: _ChatGPTExportIn):
        from fastapi import HTTPException

        if payload.source_id:
            src = _sources_module.get(payload.source_id)
            if src is None:
                raise HTTPException(status_code=404, detail="source_id not registered")
        else:
            src = _sources_module.add(
                user_id=payload.user_id,
                source_type="chatgpt_export",
                status="syncing",
            )
            _sources_module.set_status(src["source_id"], "syncing")

        chunks_stored = 0
        chunk_ids: list[str] = []
        conversations_processed = 0
        try:
            for conv in payload.conversations:
                title = conv.get("title") or "(untitled)"
                messages = conv.get("messages") or conv.get("mapping") or []
                if isinstance(messages, dict):
                    flat = []
                    for _mid, mv in messages.items():
                        msg = mv.get("message") if isinstance(mv, dict) else None
                        if msg and isinstance(msg, dict):
                            flat.append(msg)
                    messages = flat
                conversations_processed += 1

                window = payload.max_messages_per_chunk
                for i in range(0, len(messages), window):
                    chunk_msgs = messages[i:i + window]
                    text_parts = []
                    for m in chunk_msgs:
                        author = m.get("author") or {}
                        role = author.get("role") if isinstance(author, dict) else m.get("role", "user")
                        content = m.get("content")
                        if isinstance(content, dict):
                            parts = content.get("parts") or []
                            content = "\n".join(str(p) for p in parts)
                        text_parts.append(f"[{role}]\n{content}")
                    chunk_text = (
                        f"## {title} (window {i // window + 1})\n\n"
                        + "\n\n".join(text_parts)
                    )
                    result = node.ingest_memory(
                        text=chunk_text,
                        source=f"chatgpt:{title}",
                        category=payload.category,
                        author=payload.user_id,
                        is_markdown=True,
                        priority=1.0,
                    )
                    for cid in result.get("chunk_ids", []):
                        try:
                            _sources_module.tag_event(cid, src["source_id"])
                        except LookupError:
                            pass
                        chunk_ids.append(cid)
                        chunks_stored += 1

            _sources_module.set_status(src["source_id"], "connected", mark_synced=True)
        except Exception as exc:
            _sources_module.set_status(src["source_id"], "error", error=str(exc))
            raise HTTPException(status_code=500, detail=f"ingest failed: {exc}")

        return {
            "source_id": src["source_id"],
            "user_id": payload.user_id,
            "conversations_processed": conversations_processed,
            "chunks_stored": chunks_stored,
            "chunk_ids": chunk_ids[:50],
            "status": "connected",
        }

    # ── Google Drive Connector (POC) ─────────────────────────────────────────
    # Accepts already-fetched Drive documents (text content + metadata) from
    # the FE. The FE handles Drive auth + content fetch; backend just chunks
    # + ingests via node.ingest_memory so embeddings + FAISS storage stay
    # consistent with every other memory path.
    #
    # Per the Source Registry pattern: each ingested chunk tags back to a
    # source_id so coverage + provenance survive re-ingest.

    class _DriveDocIn(BaseModel):
        title: str = Field(..., min_length=1)
        content: str = Field(..., min_length=1)
        mime_type: str = "text/plain"
        drive_file_id: str | None = None
        modified_at: str | None = None

    class _GoogleDriveCaptureIn(BaseModel):
        user_id: str = Field(..., min_length=1)
        documents: list[_DriveDocIn] = Field(default_factory=list)
        source_id: str | None = None
        category: str = "google_drive"
        max_chars_per_chunk: int = 4000

    @_app.post("/v1/intelligence/capture/google-drive",
               tags=["sources"], status_code=202)
    async def capture_google_drive(payload: _GoogleDriveCaptureIn):
        from fastapi import HTTPException

        if payload.source_id:
            src = _sources_module.get(payload.source_id)
            if src is None:
                raise HTTPException(status_code=404, detail="source_id not registered")
        else:
            src = _sources_module.add(
                user_id=payload.user_id,
                source_type="google_drive",
                status="syncing",
            )
        _sources_module.set_status(src["source_id"], "syncing")

        chunks_stored = 0
        chunk_ids: list[str] = []
        documents_processed = 0
        try:
            for doc in payload.documents:
                documents_processed += 1
                window = max(500, payload.max_chars_per_chunk)
                text = doc.content
                # Char-windowed chunking with paragraph-boundary preference
                pos = 0
                idx = 0
                while pos < len(text):
                    end = min(pos + window, len(text))
                    if end < len(text):
                        # Prefer breaking at last paragraph boundary in window
                        nl = text.rfind("\n\n", pos + window // 2, end)
                        if nl > pos:
                            end = nl
                    chunk_text = (
                        f"## {doc.title} (chunk {idx + 1})\n"
                        f"_mime: {doc.mime_type}"
                        + (f" · drive_file: {doc.drive_file_id}" if doc.drive_file_id else "")
                        + (f" · modified: {doc.modified_at}" if doc.modified_at else "")
                        + "_\n\n"
                        + text[pos:end].strip()
                    )
                    result = node.ingest_memory(
                        text=chunk_text,
                        source=f"google_drive:{doc.title}",
                        category=payload.category,
                        author=payload.user_id,
                        is_markdown=True,
                        priority=1.0,
                    )
                    for cid in result.get("chunk_ids", []):
                        try:
                            _sources_module.tag_event(cid, src["source_id"])
                        except LookupError:
                            pass
                        chunk_ids.append(cid)
                        chunks_stored += 1
                    pos = end
                    idx += 1

            _sources_module.set_status(src["source_id"], "connected", mark_synced=True)
        except Exception as exc:
            _sources_module.set_status(src["source_id"], "error", error=str(exc))
            raise HTTPException(status_code=500, detail=f"ingest failed: {exc}")

        return {
            "source_id": src["source_id"],
            "user_id": payload.user_id,
            "documents_processed": documents_processed,
            "chunks_stored": chunks_stored,
            "chunk_ids": chunk_ids[:50],
            "status": "connected",
        }

    @_app.get("/matrix/info", tags=["matrix"])
    async def matrix_info():
        return node.matrix.performance_report()

    @_app.get("/society/health", tags=["society"])
    async def society_health():
        return node.society.health()

    @_app.get("/config/weights", response_model=WeightsResponse, tags=["config"])
    async def config_weights():
        """Return the canonical decision-engine weights used by this silo.

        The connector resolves weights from (in priority order):
        1. ``DECISION_ENGINE_CONFIG_PATH`` env var — local path to engine.yaml
        2. ``DECISION_ENGINE_URL/config/weights`` — HTTP fetch from remote engine
        3. Hardcoded fallback defaults (engine.yaml as of 2026-05-07)

        The ``source`` field in the response indicates which path was used.
        """
        try:
            from .bridge.connector import DecisionEngineConnector
        except ImportError as exc:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail=f"DecisionEngineConnector unavailable: {exc}",
            )

        connector = DecisionEngineConnector()
        cfg = connector.load()
        return WeightsResponse(
            value_weights=cfg.value_weights,
            penalty_weights=cfg.penalty_weights,
            trust_multiplier=cfg.trust_multiplier,
            rtql_trust_multiplier=cfg.rtql_trust_multiplier,
            thresholds=cfg.thresholds,
            source=cfg.source,
        )

    return _app


# ── Module-level app — uvicorn loads this directly ────────────────────────────
# create_app() is called here so that `uvicorn core.api:app` gets a fully
# initialized application with all routes registered and the node running.

app = create_app(
    config_path=os.environ.get("SILO_CONFIG_PATH", "config/silo.yaml")
)
