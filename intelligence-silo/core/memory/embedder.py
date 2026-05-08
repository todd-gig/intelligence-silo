"""Text embedder — converts plain text into 384-dim vectors for semantic memory.

Uses the same sentence-transformers model that the memory hierarchy expects
(all-MiniLM-L6-v2, 384 dim) so ingest and query vectors are in the same space
as everything else stored in the FAISS index.

Chunking strategy:
  - Markdown documents are split on headers (##, ###) so each chunk is a
    coherent section, not an arbitrary token window.
  - Plain text falls back to sliding-window paragraph splits.
  - Chunks are trimmed to MAX_CHUNK_CHARS to stay within the model's 256-token
    context window comfortably.

Singleton pattern: the model is loaded once on first use and reused across
all ingest/search calls — sentence-transformers load is ~300ms, not something
to repeat per request.
"""

from __future__ import annotations

import re
import textwrap
from typing import Optional

import numpy as np

# Lazy import — avoids paying the load cost at import time
_model = None
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_CHUNK_CHARS = 1_200   # ~300 tokens; well within 256-token window with margin
CHUNK_OVERLAP_CHARS = 120  # overlap so context isn't lost at boundaries


def _get_model():
    """Load the sentence-transformer model on first use (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string. Returns a (384,) float32 array."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.astype(np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns (N, 384) float32 array."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                        batch_size=32)
    return vecs.astype(np.float32)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_markdown(text: str, source: str = "") -> list[dict]:
    """Split a markdown document into section chunks.

    Each chunk is a dict:
        text     — the chunk content (trimmed)
        heading  — the nearest parent heading (or "" if top-level)
        source   — the source identifier passed in (filename, URL, etc.)
        chunk_id — sequential index within this document
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on markdown headers (## or ###, not # which is the doc title)
    header_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    positions = [m.start() for m in header_pattern.finditer(text)]
    positions.append(len(text))

    chunks = []
    chunk_id = 0

    if not positions or positions[0] > 0:
        # Content before first header (preamble)
        preamble = text[: positions[0] if positions else len(text)].strip()
        if preamble:
            for sub in _split_long(preamble):
                chunks.append({
                    "text": sub,
                    "heading": "",
                    "source": source,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

    for i, pos in enumerate(positions[:-1]):
        section_text = text[pos: positions[i + 1]].strip()
        # Extract heading from first line
        first_newline = section_text.find("\n")
        if first_newline == -1:
            heading = section_text.lstrip("#").strip()
            body = ""
        else:
            heading = section_text[:first_newline].lstrip("#").strip()
            body = section_text[first_newline:].strip()

        # If the section body is short enough, keep as one chunk
        combined = f"{heading}\n\n{body}".strip() if body else heading
        for sub in _split_long(combined):
            chunks.append({
                "text": sub,
                "heading": heading,
                "source": source,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    return chunks


def chunk_plaintext(text: str, source: str = "") -> list[dict]:
    """Split plain text into overlapping paragraph chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    chunk_id = 0
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= MAX_CHUNK_CHARS:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append({
                    "text": current,
                    "heading": "",
                    "source": source,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1
            # Keep overlap from end of previous chunk
            overlap_start = max(0, len(current) - CHUNK_OVERLAP_CHARS)
            current = (current[overlap_start:] + "\n\n" + para).strip()

    if current:
        chunks.append({
            "text": current,
            "heading": "",
            "source": source,
            "chunk_id": chunk_id,
        })

    return chunks


def chunk_document(text: str, source: str = "",
                   is_markdown: Optional[bool] = None) -> list[dict]:
    """Auto-detect format and chunk accordingly."""
    if is_markdown is None:
        # Heuristic: markdown if it contains header patterns or code fences
        is_markdown = bool(re.search(r"^#{1,4}\s+\S", text, re.MULTILINE)
                           or "```" in text)
    if is_markdown:
        return chunk_markdown(text, source=source)
    return chunk_plaintext(text, source=source)


def _split_long(text: str) -> list[str]:
    """Split a single block that exceeds MAX_CHUNK_CHARS into overlapping pieces."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        parts.append(text[start:end])
        start = end - CHUNK_OVERLAP_CHARS
    return [p for p in parts if p.strip()]
