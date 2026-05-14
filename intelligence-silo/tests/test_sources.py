"""Source Registry tests — schema + add/get/list/status/consent/remove/tag_event."""
from __future__ import annotations

import os
import tempfile

import pytest

from core import sources as src


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Each test gets a fresh SQLite file."""
    db = tmp_path / "sources.db"
    src.init_db(str(db))
    yield
    # Reset module-level _db_path so tests don't leak
    src._db_path = None


USER = "user-a"


def test_add_creates_row():
    row = src.add(user_id=USER, source_type="chatgpt_export")
    assert row["user_id"] == USER
    assert row["source_type"] == "chatgpt_export"
    assert row["status"] == "disconnected"
    assert "source_id" in row


def test_add_idempotent_on_user_type():
    r1 = src.add(user_id=USER, source_type="google_drive")
    r2 = src.add(user_id=USER, source_type="google_drive")
    assert r1["source_id"] == r2["source_id"]


def test_add_rejects_unknown_source_type():
    with pytest.raises(ValueError, match="source_type"):
        src.add(user_id=USER, source_type="myspace")


def test_add_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        src.add(user_id=USER, source_type="gmail", status="vibing")


def test_get_returns_none_when_missing():
    assert src.get("nonexistent") is None


def test_get_returns_row():
    row = src.add(user_id=USER, source_type="github")
    fetched = src.get(row["source_id"])
    assert fetched["source_id"] == row["source_id"]
    assert fetched["source_type"] == "github"


def test_list_for_user_returns_all():
    src.add(user_id=USER, source_type="chatgpt_export")
    src.add(user_id=USER, source_type="github")
    src.add(user_id="other-user", source_type="gmail")
    items = src.list_for_user(USER)
    assert len(items) == 2
    types = {i["source_type"] for i in items}
    assert types == {"chatgpt_export", "github"}


def test_list_for_user_filters_by_type():
    src.add(user_id=USER, source_type="chatgpt_export")
    src.add(user_id=USER, source_type="github")
    items = src.list_for_user(USER, source_type="github")
    assert len(items) == 1
    assert items[0]["source_type"] == "github"


def test_list_for_user_filters_by_status():
    r = src.add(user_id=USER, source_type="gmail")
    src.set_status(r["source_id"], "connected", mark_synced=True)
    src.add(user_id=USER, source_type="github")  # stays disconnected
    items = src.list_for_user(USER, status="connected")
    assert len(items) == 1
    assert items[0]["source_type"] == "gmail"


def test_set_status_updates():
    r = src.add(user_id=USER, source_type="chatgpt_export")
    updated = src.set_status(r["source_id"], "syncing")
    assert updated["status"] == "syncing"
    assert updated["last_sync_at"] is None  # mark_synced=False default


def test_set_status_mark_synced_sets_timestamp():
    r = src.add(user_id=USER, source_type="chatgpt_export")
    updated = src.set_status(r["source_id"], "connected", mark_synced=True)
    assert updated["last_sync_at"] is not None
    assert updated["status"] == "connected"


def test_set_status_records_error():
    r = src.add(user_id=USER, source_type="github")
    updated = src.set_status(r["source_id"], "error", error="token expired")
    assert updated["status"] == "error"
    assert updated["error_message"] == "token expired"


def test_set_status_404_returns_none():
    assert src.set_status("ghost", "connected") is None


def test_set_status_rejects_unknown_status():
    r = src.add(user_id=USER, source_type="github")
    with pytest.raises(ValueError, match="status"):
        src.set_status(r["source_id"], "vibing")


def test_update_consent():
    r = src.add(user_id=USER, source_type="gmail")
    updated = src.update_consent(r["source_id"], {"redact_pii": True, "scope": "subject_only"})
    assert updated["consent_state"]["redact_pii"] is True
    assert updated["consent_state"]["scope"] == "subject_only"


def test_remove_deletes_row():
    r = src.add(user_id=USER, source_type="github")
    assert src.remove(r["source_id"]) is True
    assert src.get(r["source_id"]) is None


def test_remove_returns_false_when_missing():
    assert src.remove("ghost") is False


def test_tag_event_associates_memory_chunk():
    r = src.add(user_id=USER, source_type="chatgpt_export")
    src.tag_event("chunk-1", r["source_id"])
    src.tag_event("chunk-2", r["source_id"])
    assert src.event_count_for_source(r["source_id"]) == 2


def test_tag_event_idempotent():
    r = src.add(user_id=USER, source_type="chatgpt_export")
    src.tag_event("chunk-1", r["source_id"])
    src.tag_event("chunk-1", r["source_id"])  # duplicate
    assert src.event_count_for_source(r["source_id"]) == 1


def test_tag_event_unknown_source_raises():
    with pytest.raises(LookupError):
        src.tag_event("chunk-1", "ghost-source-id")


def test_coverage_for_user_summary():
    src.add(user_id=USER, source_type="chatgpt_export")
    r2 = src.add(user_id=USER, source_type="github")
    src.set_status(r2["source_id"], "connected", mark_synced=True)
    cov = src.coverage_for_user(USER)
    assert cov["registered_count"] == 2
    assert cov["connected_count"] == 1
    assert "chatgpt_export" in cov["by_type"]
    assert cov["by_type"]["github"]["status"] == "connected"
    assert cov["total_sources_known"] > 5  # we know 10+ source types
