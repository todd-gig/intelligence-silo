#!/usr/bin/env python3
"""ingest_docs.py — Ingest markdown documents into the Intelligence Silo's shared memory.

Walks one or more directories (or individual files), chunks each document by section,
embeds each chunk, and stores it in the FAISS semantic index via POST /memory/ingest.

Once ingested, any node that shares the same semantic index (via GCS sync or
direct access) can query the knowledge using POST /memory/search.

Usage:
    # Ingest the full architecture doc directory
    python scripts/ingest_docs.py ../decision-engine/docs/architecture/

    # Ingest a specific file with a category tag
    python scripts/ingest_docs.py ../decision-engine/docs/architecture/MASTER_ARCHITECTURE.md \\
        --category architecture --author todd@gigaton.ai

    # Ingest multiple roots at once
    python scripts/ingest_docs.py \\
        ../decision-engine/docs/ \\
        ../gigaton-engine/docs/ \\
        --category doctrine

    # Dry run — show what would be ingested without calling the API
    python scripts/ingest_docs.py ../decision-engine/docs/ --dry-run

    # Point at a non-default silo endpoint
    python scripts/ingest_docs.py ../decision-engine/docs/ --silo-url https://your-silo.run.app

Environment variables:
    SILO_URL     Base URL of the intelligence silo (default: http://localhost:8080)
    SILO_TOKEN   Bearer token for authenticated Cloud Run endpoints (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_SILO_URL = os.environ.get("SILO_URL", "http://localhost:8080")
SILO_TOKEN = os.environ.get("SILO_TOKEN", "")

# File extensions to ingest
INCLUDE_EXTENSIONS = {".md", ".txt", ".rst"}

# Paths to skip even if they match INCLUDE_EXTENSIONS
SKIP_PATTERNS = {
    "CHANGELOG", "LICENSE", "CONTRIBUTING",
    "node_modules", ".git", "__pycache__", ".venv",
}

# Category mapping: if a path segment matches a key, use that category
PATH_CATEGORY_MAP = {
    "architecture": "architecture",
    "docs": "doctrine",
    "doctrine": "doctrine",
    "playbooks": "playbook",
    "specs": "spec",
    "templates": "template",
    "scripts": "script",
    "operations": "operations",
}


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if SILO_TOKEN:
        h["Authorization"] = f"Bearer {SILO_TOKEN}"
    return h


def ingest_file(
    path: Path,
    silo_url: str,
    category: str,
    author: str,
    verbose: bool = False,
    dry_run: bool = False,
) -> dict:
    """POST a single file to /memory/ingest. Returns the response dict."""
    text = path.read_text(encoding="utf-8", errors="replace")
    source = str(path)

    payload = {
        "text": text,
        "source": source,
        "category": category,
        "author": author,
        "is_markdown": path.suffix == ".md",
        "priority": 1.0,
    }

    if dry_run:
        # Estimate chunk count without calling the API
        from core.memory.embedder import chunk_document
        chunks = chunk_document(text, source=source, is_markdown=payload["is_markdown"])
        print(f"  [DRY RUN] {path.name} → {len(chunks)} chunks, category={category}")
        return {"chunks_stored": len(chunks), "dry_run": True}

    url = f"{silo_url.rstrip('/')}/memory/ingest"
    data = json.dumps(payload).encode()

    req = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            if verbose:
                print(f"  ✓ {path.name} → {result.get('chunks_stored', 0)} chunks stored")
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"  ✗ {path.name} → HTTP {e.code}: {body[:200]}", file=sys.stderr)
        return {"error": e.code}
    except Exception as e:
        print(f"  ✗ {path.name} → {e}", file=sys.stderr)
        return {"error": str(e)}


def check_silo_health(silo_url: str) -> bool:
    """Return True if the silo responds to /health."""
    try:
        req = urllib.request.Request(
            f"{silo_url.rstrip('/')}/health",
            headers=_headers(),
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("status") in ("ok", "healthy", True)
    except Exception as e:
        print(f"Silo health check failed: {e}", file=sys.stderr)
        return False


# ── Path resolution ───────────────────────────────────────────────────────────

def should_skip(path: Path) -> bool:
    """Return True if this file or any parent dir matches a skip pattern."""
    for part in path.parts:
        if part in SKIP_PATTERNS:
            return True
    if path.stem.upper() in SKIP_PATTERNS:
        return True
    return False


def infer_category(path: Path, override: str = "") -> str:
    """Infer a category tag from the path if no override given."""
    if override:
        return override
    for part in reversed(path.parts):
        low = part.lower()
        for key, cat in PATH_CATEGORY_MAP.items():
            if key in low:
                return cat
    return "general"


def collect_files(roots: list[Path]) -> list[Path]:
    """Collect all ingestable files from the given roots."""
    files = []
    for root in roots:
        if root.is_file():
            if root.suffix in INCLUDE_EXTENSIONS and not should_skip(root):
                files.append(root)
        elif root.is_dir():
            for f in sorted(root.rglob("*")):
                if f.is_file() and f.suffix in INCLUDE_EXTENSIONS and not should_skip(f):
                    files.append(f)
        else:
            print(f"Warning: {root} not found, skipping", file=sys.stderr)
    return files


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Intelligence Silo shared memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths", nargs="+", type=Path,
        help="Files or directories to ingest.",
    )
    parser.add_argument(
        "--silo-url", default=DEFAULT_SILO_URL,
        help=f"Silo base URL (default: {DEFAULT_SILO_URL})",
    )
    parser.add_argument(
        "--category", default="",
        help="Override category tag for all ingested files.",
    )
    parser.add_argument(
        "--author", default="",
        help="Author label stored with each chunk (e.g. todd@gigaton.ai).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be ingested without calling the API.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print a line per file as it is processed.",
    )
    args = parser.parse_args()

    # Health check (skip on dry run)
    if not args.dry_run:
        print(f"Checking silo at {args.silo_url} …")
        if not check_silo_health(args.silo_url):
            print("Silo is not reachable. Start it first or set --silo-url.", file=sys.stderr)
            sys.exit(1)
        print("Silo is healthy.\n")

    files = collect_files(args.paths)
    if not files:
        print("No ingestable files found.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(files)} file(s) to ingest.\n")

    total_chunks = 0
    total_errors = 0

    for path in files:
        category = infer_category(path, override=args.category)
        if args.verbose or args.dry_run:
            print(f"→ {path}  [{category}]")

        result = ingest_file(
            path=path,
            silo_url=args.silo_url,
            category=category,
            author=args.author,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )

        if "error" in result:
            total_errors += 1
        else:
            total_chunks += result.get("chunks_stored", 0)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done.")
    print(f"  Files processed : {len(files)}")
    print(f"  Chunks stored   : {total_chunks}")
    if total_errors:
        print(f"  Errors          : {total_errors}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
