"""Source Registry — canonical store of memory sources per user.

Per the Unified Source Registry doctrine (2026-05-14):
  - Every ingest writes ONE row in `sources` + memory events tagged with source_id
  - Every consumer (HME / PPEME / persona-engine) reads from intelligence-silo
    by source_id rather than querying raw sources directly
  - SMEN substrate doctrine §5.18 — sources flow IN to substrate, engines read OUT

Schema (SQLite):
    sources(
        source_id           TEXT PRIMARY KEY (uuid4),
        user_id             TEXT NOT NULL,
        source_type         TEXT NOT NULL (e.g. 'chatgpt_export', 'google_drive', 'gmail',
                                          'github', 'local_files', 'notion', 'clickup'),
        status              TEXT NOT NULL ('disconnected'|'connecting'|'connected'|
                                           'syncing'|'error'|'paused'),
        last_sync_at        TEXT (ISO-8601, nullable),
        consent_state       TEXT (JSON; redaction policy, scope, etc.),
        provenance_metadata TEXT (JSON; provider-specific identifiers),
        error_message       TEXT (nullable, populated on status='error'),
        created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )

This module exposes:
    SourceRegistry.add(user_id, source_type, ...) -> dict
    SourceRegistry.get(source_id) -> dict | None
    SourceRegistry.list(user_id, source_type=None) -> list[dict]
    SourceRegistry.set_status(source_id, status, error=None) -> dict
    SourceRegistry.remove(source_id) -> bool
    SourceRegistry.tag_event(memory_event_id, source_id) -> None
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SOURCE_TYPES = frozenset({
    "chatgpt_export",
    "openai_export",  # alias
    "google_drive",
    "gmail",
    "google_calendar",
    "github",
    "slack",
    "notion",
    "clickup",
    "local_files",
    "browser_history",
    "manual_upload",
})

STATUSES = frozenset({
    "disconnected",
    "connecting",
    "connected",
    "syncing",
    "error",
    "paused",
})


# Per-process lock — SQLite handles concurrent reads but writes need serialization
_db_lock = threading.RLock()
_db_path: Optional[Path] = None


def init_db(path: str | Path) -> None:
    """Create the sources + source_memory_events tables if not present."""
    global _db_path
    _db_path = Path(path)
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                source_id           TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                source_type         TEXT NOT NULL,
                status              TEXT NOT NULL,
                last_sync_at        TEXT,
                consent_state       TEXT NOT NULL DEFAULT '{}',
                provenance_metadata TEXT NOT NULL DEFAULT '{}',
                error_message       TEXT,
                created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_sources_user
                ON sources(user_id, source_type);
            CREATE INDEX IF NOT EXISTS idx_sources_status
                ON sources(user_id, status);

            -- Memory events ingested from a registered source carry source_id.
            -- This join table lets downstream consumers query "give me all
            -- memory events from source X" without scanning the FAISS index.
            CREATE TABLE IF NOT EXISTS source_memory_events (
                memory_event_id     TEXT PRIMARY KEY,
                source_id           TEXT NOT NULL,
                ingested_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_source_events_source
                ON source_memory_events(source_id, ingested_at);
        """)


@contextmanager
def _connection() -> sqlite3.Connection:
    """Thread-safe SQLite connection with row factory."""
    if _db_path is None:
        raise RuntimeError("SourceRegistry not initialized — call init_db(path) first")
    with _db_lock:
        conn = sqlite3.connect(str(_db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert sqlite3.Row → dict, parsing JSON columns."""
    d = dict(row)
    d["consent_state"] = json.loads(d.get("consent_state") or "{}")
    d["provenance_metadata"] = json.loads(d.get("provenance_metadata") or "{}")
    return d


def add(
    *,
    user_id: str,
    source_type: str,
    status: str = "disconnected",
    consent_state: Optional[dict] = None,
    provenance_metadata: Optional[dict] = None,
    source_id: Optional[str] = None,
) -> dict[str, Any]:
    """Register a new source for a user. Returns the persisted row.

    Idempotent on (user_id, source_type) — if a row already exists for this
    pair, returns the existing one rather than creating a duplicate. Callers
    that want a fresh row should remove() first.
    """
    if not user_id:
        raise ValueError("user_id required")
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"source_type {source_type!r} not in {sorted(SOURCE_TYPES)}")
    if status not in STATUSES:
        raise ValueError(f"status {status!r} not in {sorted(STATUSES)}")

    source_id = source_id or str(uuid.uuid4())
    now = datetime.now(tz=timezone.utc).isoformat()
    consent_json = json.dumps(consent_state or {})
    prov_json = json.dumps(provenance_metadata or {})

    with _connection() as conn:
        existing = conn.execute(
            "SELECT * FROM sources WHERE user_id = ? AND source_type = ? LIMIT 1",
            (user_id, source_type),
        ).fetchone()
        if existing:
            return _row_to_dict(existing)
        conn.execute(
            """
            INSERT INTO sources (
                source_id, user_id, source_type, status,
                consent_state, provenance_metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (source_id, user_id, source_type, status,
             consent_json, prov_json, now, now),
        )
        row = conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,),
        ).fetchone()
        return _row_to_dict(row)


def get(source_id: str) -> Optional[dict[str, Any]]:
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None


def list_for_user(
    user_id: str,
    source_type: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM sources WHERE user_id = ?"
    params: list[Any] = [user_id]
    if source_type is not None:
        sql += " AND source_type = ?"
        params.append(source_type)
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY updated_at DESC"
    with _connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]


def set_status(
    source_id: str,
    status: str,
    *,
    error: Optional[str] = None,
    mark_synced: bool = False,
) -> Optional[dict[str, Any]]:
    if status not in STATUSES:
        raise ValueError(f"status {status!r} not in {sorted(STATUSES)}")
    now = datetime.now(tz=timezone.utc).isoformat()
    last_sync = now if mark_synced else None
    with _connection() as conn:
        existing = conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,),
        ).fetchone()
        if not existing:
            return None
        # Only update last_sync_at if mark_synced; otherwise preserve
        if last_sync is not None:
            conn.execute(
                """
                UPDATE sources
                SET status = ?, error_message = ?,
                    last_sync_at = ?, updated_at = ?
                WHERE source_id = ?
                """,
                (status, error, last_sync, now, source_id),
            )
        else:
            conn.execute(
                """
                UPDATE sources
                SET status = ?, error_message = ?, updated_at = ?
                WHERE source_id = ?
                """,
                (status, error, now, source_id),
            )
        row = conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,),
        ).fetchone()
        return _row_to_dict(row)


def update_consent(
    source_id: str,
    consent_state: dict,
) -> Optional[dict[str, Any]]:
    now = datetime.now(tz=timezone.utc).isoformat()
    with _connection() as conn:
        existing = conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,),
        ).fetchone()
        if not existing:
            return None
        conn.execute(
            "UPDATE sources SET consent_state = ?, updated_at = ? WHERE source_id = ?",
            (json.dumps(consent_state), now, source_id),
        )
        row = conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,),
        ).fetchone()
        return _row_to_dict(row)


def remove(source_id: str) -> bool:
    """Delete a source + its memory-event tags (CASCADE). Returns True if removed."""
    with _connection() as conn:
        cur = conn.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))
        return cur.rowcount > 0


def tag_event(memory_event_id: str, source_id: str) -> None:
    """Associate a memory chunk / event with a registered source."""
    with _connection() as conn:
        # Confirm source exists
        existing = conn.execute(
            "SELECT 1 FROM sources WHERE source_id = ?", (source_id,),
        ).fetchone()
        if not existing:
            raise LookupError(f"source_id {source_id!r} not registered")
        conn.execute(
            """
            INSERT OR IGNORE INTO source_memory_events
                (memory_event_id, source_id)
            VALUES (?, ?)
            """,
            (memory_event_id, source_id),
        )


def event_count_for_source(source_id: str) -> int:
    with _connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM source_memory_events WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        return int(row["n"]) if row else 0


def coverage_for_user(user_id: str) -> dict[str, Any]:
    """Summary: { connected_count, total_known_types, last_sync, by_type: [...] }.

    Used by the FE /sources page progress bar + per-type cards.
    """
    sources = list_for_user(user_id)
    by_type = {}
    for src in sources:
        by_type[src["source_type"]] = {
            "source_id": src["source_id"],
            "status": src["status"],
            "last_sync_at": src["last_sync_at"],
            "event_count": event_count_for_source(src["source_id"]),
            "error_message": src.get("error_message"),
        }
    connected = sum(1 for s in sources if s["status"] in {"connected", "syncing"})
    last_sync = max(
        (s.get("last_sync_at") for s in sources if s.get("last_sync_at")),
        default=None,
    )
    return {
        "user_id": user_id,
        "total_sources_known": len(SOURCE_TYPES),
        "registered_count": len(sources),
        "connected_count": connected,
        "last_sync_at": last_sync,
        "by_type": by_type,
    }
