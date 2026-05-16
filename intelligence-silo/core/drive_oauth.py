"""Google Drive OAuth handler — v0.5 of the google_drive source connector.

Replaces the v0 "manual file upload" flow (still supported as fallback) with a
proper OAuth-pull-from-Drive integration. After a user clicks "Connect Google
Drive" on /sources, the silo holds their access+refresh tokens and can pull
files on demand via /v1/intelligence/sync/google-drive.

Per Source Registry doctrine: ingested chunks tag back to source_id so coverage
+ provenance survive re-ingest. Refresh tokens auto-renew before expiry.

Config (env vars, in priority order):
  1. GOOGLE_DRIVE_OAUTH_CLIENT_JSON — full credentials.json contents (Cloud Run
     pattern: mount Secret Manager secret as env var)
  2. GOOGLE_DRIVE_OAUTH_CLIENT_ID + GOOGLE_DRIVE_OAUTH_CLIENT_SECRET
  3. raise — no OAuth available

Other env:
  GOOGLE_DRIVE_OAUTH_REDIRECT_URI    — must match GCP Console authorized URI
                                       defaults to silo prod callback path
  GOOGLE_DRIVE_OAUTH_FRONTEND_RETURN_URL — where to send user after callback
                                          defaults to gigaton-platform.web.app/sources
  OAUTH_STATE_SECRET                  — HMAC key for CSRF state tokens
                                       defaults to a build-time random (DEV ONLY)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

# Drive scope — read-only access to Drive files the user grants
DRIVE_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/drive.readonly",
]

DEFAULT_REDIRECT_URI = (
    "https://intelligence-silo-rjmcrtvuzq-uc.a.run.app/v1/oauth/drive/callback"
)
DEFAULT_FRONTEND_RETURN_URL = "https://gigaton-platform.web.app/sources?drive_connected=1"

AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

_db_lock = threading.RLock()
_db_path: Optional[Path] = None


# ── Config resolution ────────────────────────────────────────────────────────

def _resolve_client_config() -> dict[str, str]:
    """Load OAuth client_id + client_secret from env vars.

    Supports:
      - GOOGLE_DRIVE_OAUTH_CLIENT_JSON (full credentials.json contents)
      - GOOGLE_DRIVE_OAUTH_CLIENT_ID + GOOGLE_DRIVE_OAUTH_CLIENT_SECRET (discrete)
    """
    blob = os.environ.get("GOOGLE_DRIVE_OAUTH_CLIENT_JSON")
    if blob:
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"GOOGLE_DRIVE_OAUTH_CLIENT_JSON not valid JSON: {exc}")
        # The downloaded credentials.json wraps under "web" or "installed"
        inner = parsed.get("web") or parsed.get("installed") or parsed
        cid = inner.get("client_id")
        secret = inner.get("client_secret")
        if cid and secret:
            return {"client_id": cid, "client_secret": secret}

    cid = os.environ.get("GOOGLE_DRIVE_OAUTH_CLIENT_ID")
    secret = os.environ.get("GOOGLE_DRIVE_OAUTH_CLIENT_SECRET")
    if cid and secret:
        return {"client_id": cid, "client_secret": secret}

    raise RuntimeError(
        "Google Drive OAuth not configured. Set either "
        "GOOGLE_DRIVE_OAUTH_CLIENT_JSON or the pair "
        "GOOGLE_DRIVE_OAUTH_CLIENT_ID + GOOGLE_DRIVE_OAUTH_CLIENT_SECRET."
    )


def _redirect_uri() -> str:
    return os.environ.get("GOOGLE_DRIVE_OAUTH_REDIRECT_URI", DEFAULT_REDIRECT_URI)


def _frontend_return_url() -> str:
    return os.environ.get(
        "GOOGLE_DRIVE_OAUTH_FRONTEND_RETURN_URL", DEFAULT_FRONTEND_RETURN_URL
    )


def _state_secret() -> bytes:
    raw = os.environ.get("OAUTH_STATE_SECRET")
    if not raw:
        # Build-time random — non-persistent. WHY: never use a hardcoded default
        # for HMAC keys, even in dev. Each process gets a fresh key; if you
        # restart the silo, in-flight OAuth flows are invalidated. Acceptable
        # for dev; prod must inject a stable secret.
        raw = secrets.token_urlsafe(32)
        os.environ["OAUTH_STATE_SECRET"] = raw
        logger.warning(
            "OAUTH_STATE_SECRET not set — using ephemeral key. In-flight OAuth "
            "flows will fail after process restart. Inject a stable secret."
        )
    return raw.encode("utf-8")


def is_configured() -> bool:
    """Quick check whether OAuth is usable. Used by /health + status endpoint."""
    try:
        _resolve_client_config()
        return True
    except RuntimeError:
        return False


# ── State token (CSRF) ───────────────────────────────────────────────────────

def _sign_state(user_id: str, nonce: str, ts: int) -> str:
    """HMAC-SHA256 sign the (user_id, nonce, ts) tuple."""
    msg = f"{user_id}|{nonce}|{ts}".encode("utf-8")
    mac = hmac.new(_state_secret(), msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")


def make_state_token(user_id: str) -> str:
    """Build an opaque state token the callback can verify + bind to user_id."""
    nonce = secrets.token_urlsafe(12)
    ts = int(time.time())
    sig = _sign_state(user_id, nonce, ts)
    raw = json.dumps({"u": user_id, "n": nonce, "t": ts, "s": sig})
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def verify_state_token(token: str, max_age_seconds: int = 600) -> Optional[str]:
    """Verify the state token, return user_id if valid, None otherwise."""
    try:
        pad = "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(token + pad).decode("utf-8")
        d = json.loads(decoded)
        user_id = d["u"]
        nonce = d["n"]
        ts = int(d["t"])
        sig = d["s"]
    except (ValueError, KeyError, json.JSONDecodeError):
        return None

    if int(time.time()) - ts > max_age_seconds:
        return None

    expected = _sign_state(user_id, nonce, ts)
    if not hmac.compare_digest(sig, expected):
        return None
    return user_id


# ── Authorization URL ────────────────────────────────────────────────────────

def build_authorization_url(user_id: str) -> dict[str, str]:
    """Return {authorization_url, state} the FE can redirect/popup to."""
    cfg = _resolve_client_config()
    state = make_state_token(user_id)
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": " ".join(DRIVE_SCOPES),
        "state": state,
        "access_type": "offline",        # request refresh_token
        "include_granted_scopes": "true",
        "prompt": "consent",             # force refresh_token even on re-grant
    }
    url = f"{AUTHORIZATION_ENDPOINT}?{urlencode(params)}"
    return {"authorization_url": url, "state": state}


# ── Code exchange ────────────────────────────────────────────────────────────

def exchange_code(code: str) -> dict[str, Any]:
    """Exchange auth code for tokens."""
    cfg = _resolve_client_config()
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(
            TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "redirect_uri": _redirect_uri(),
                "grant_type": "authorization_code",
            },
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Token exchange failed: {resp.status_code} {resp.text}")
    return resp.json()


def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    """Refresh an expired access token. Returns new token payload."""
    cfg = _resolve_client_config()
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(
            TOKEN_ENDPOINT,
            data={
                "refresh_token": refresh_token,
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "grant_type": "refresh_token",
            },
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Token refresh failed: {resp.status_code} {resp.text}")
    return resp.json()


# ── Token storage (SQLite) ───────────────────────────────────────────────────

def init_db(path: str | Path) -> None:
    global _db_path
    _db_path = Path(path)
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS oauth_tokens (
                user_id       TEXT NOT NULL,
                provider      TEXT NOT NULL,
                access_token  TEXT NOT NULL,
                refresh_token TEXT,
                expires_at    INTEGER NOT NULL,
                scopes        TEXT NOT NULL DEFAULT '',
                granted_email TEXT,
                created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, provider)
            );
        """)


@contextmanager
def _connection() -> sqlite3.Connection:
    if _db_path is None:
        raise RuntimeError("drive_oauth not initialized — call init_db(path) first")
    with _db_lock:
        conn = sqlite3.connect(str(_db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def store_tokens(
    user_id: str,
    token_payload: dict[str, Any],
    provider: str = "google_drive",
    granted_email: Optional[str] = None,
) -> dict[str, Any]:
    """Persist tokens for a user. Refresh token preserved if not in payload."""
    access = token_payload.get("access_token")
    if not access:
        raise ValueError("token_payload missing access_token")
    expires_in = int(token_payload.get("expires_in", 3600))
    expires_at = int(time.time()) + expires_in
    refresh = token_payload.get("refresh_token")
    scopes = token_payload.get("scope", " ".join(DRIVE_SCOPES))

    with _connection() as conn:
        existing = conn.execute(
            "SELECT refresh_token FROM oauth_tokens WHERE user_id=? AND provider=?",
            (user_id, provider),
        ).fetchone()
        if existing and not refresh:
            refresh = existing["refresh_token"]
        conn.execute(
            """
            INSERT INTO oauth_tokens
                (user_id, provider, access_token, refresh_token, expires_at, scopes,
                 granted_email, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, provider) DO UPDATE SET
                access_token=excluded.access_token,
                refresh_token=COALESCE(excluded.refresh_token, oauth_tokens.refresh_token),
                expires_at=excluded.expires_at,
                scopes=excluded.scopes,
                granted_email=COALESCE(excluded.granted_email, oauth_tokens.granted_email),
                updated_at=CURRENT_TIMESTAMP
            """,
            (user_id, provider, access, refresh, expires_at, scopes, granted_email),
        )
    return {
        "user_id": user_id,
        "provider": provider,
        "expires_at": expires_at,
        "has_refresh": bool(refresh),
        "scopes": scopes,
        "granted_email": granted_email,
    }


def get_tokens(user_id: str, provider: str = "google_drive") -> Optional[dict[str, Any]]:
    """Fetch stored tokens. Caller responsible for checking expires_at."""
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM oauth_tokens WHERE user_id=? AND provider=?",
            (user_id, provider),
        ).fetchone()
        return dict(row) if row else None


def get_fresh_access_token(user_id: str, provider: str = "google_drive") -> Optional[str]:
    """Return a non-expired access token, refreshing if needed. None if not connected."""
    tokens = get_tokens(user_id, provider)
    if not tokens:
        return None
    if tokens["expires_at"] - int(time.time()) > 60:
        return tokens["access_token"]
    refresh = tokens.get("refresh_token")
    if not refresh:
        logger.warning("No refresh token for user %s provider %s — re-auth needed", user_id, provider)
        return None
    new_payload = refresh_access_token(refresh)
    store_tokens(user_id, new_payload, provider=provider)
    refreshed = get_tokens(user_id, provider)
    return refreshed["access_token"] if refreshed else None


def delete_tokens(user_id: str, provider: str = "google_drive") -> bool:
    """Revoke locally. Doesn't revoke at Google — user must do that in their account."""
    with _connection() as conn:
        cur = conn.execute(
            "DELETE FROM oauth_tokens WHERE user_id=? AND provider=?",
            (user_id, provider),
        )
        return cur.rowcount > 0


# ── Drive API helpers ────────────────────────────────────────────────────────

def list_recent_docs(
    user_id: str,
    page_size: int = 50,
    mime_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """List user's most recently modified Drive files of the given mime types.

    Defaults to Google Docs only (most-text-extractable type). Returns
    minimal metadata; content is fetched per-file via export_doc_text.
    """
    access = get_fresh_access_token(user_id)
    if not access:
        raise RuntimeError("Drive not connected for this user")

    if mime_types is None:
        mime_types = [
            "application/vnd.google-apps.document",  # Google Docs
        ]
    q = " or ".join(f"mimeType='{m}'" for m in mime_types)
    params = {
        "q": f"({q}) and trashed=false",
        "orderBy": "modifiedTime desc",
        "pageSize": page_size,
        "fields": "files(id,name,mimeType,modifiedTime)",
    }
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(
            "https://www.googleapis.com/drive/v3/files",
            headers={"Authorization": f"Bearer {access}"},
            params=params,
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Drive list failed: {resp.status_code} {resp.text}")
    return resp.json().get("files", [])


def export_doc_text(user_id: str, file_id: str) -> str:
    """Export a Google Doc as plain text via Drive API."""
    access = get_fresh_access_token(user_id)
    if not access:
        raise RuntimeError("Drive not connected for this user")
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}/export",
            headers={"Authorization": f"Bearer {access}"},
            params={"mimeType": "text/plain"},
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Drive export failed for {file_id}: {resp.status_code} {resp.text}")
    return resp.text


def fetch_userinfo(access_token: str) -> dict[str, Any]:
    """Get the user's email from the access token (uses openid+email scope)."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        if resp.status_code >= 400:
            return {}
        return resp.json()
    except Exception:
        return {}
