"""GCS Index Sync — durable persistence for the FAISS semantic memory index.

Upload on save, download on cold start. Thread-safe.
Gracefully degrades to local-only when GCS_BUCKET is not set.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Files that make up a complete semantic memory snapshot
_INDEX_FILES = ("index.faiss", "metadata.json", "embeddings.npy", "id_order.json")


class GCSIndexSync:
    """Sync the FAISS semantic memory index to/from Google Cloud Storage.

    Usage
    -----
    Called by SemanticMemory.save() after writing local files::

        sync = GCSIndexSync()
        sync.upload(persistence_path)

    Called by SemanticMemory._load() before reading local files::

        sync = GCSIndexSync()
        sync.sync(persistence_path)   # pulls from GCS if local is missing/stale

    Environment variables
    ---------------------
    GCS_BUCKET      Required. Name of the GCS bucket (e.g. ``gigaton-memory``).
                    If absent the instance is a no-op and all methods return
                    immediately without error.
    GCS_SYNC_PATH   Optional prefix inside the bucket.
                    Default: ``intelligence-silo/shared-memory/``
    """

    def __init__(self) -> None:
        self._bucket_name: str | None = os.environ.get("GCS_BUCKET")
        raw_path = os.environ.get("GCS_SYNC_PATH", "intelligence-silo/shared-memory/")
        self._sync_prefix: str = raw_path.rstrip("/") + "/"
        self._lock = threading.Lock()
        self._client = None  # lazy-initialised

        if not self._bucket_name:
            logger.debug(
                "GCS_BUCKET not set — GCSIndexSync running in local-only mode"
            )

    # ── Public API ──────────────────────────────────────────────────────────

    def upload(self, local_path: Path) -> bool:
        """Upload all index files from *local_path* to GCS.

        Parameters
        ----------
        local_path:
            Directory written by ``SemanticMemory.save()``.

        Returns
        -------
        True on success, False on failure or when GCS is not configured.
        """
        if not self._bucket_name:
            return False

        with self._lock:
            client = self._get_client()
            if client is None:
                return False

            try:
                bucket = client.bucket(self._bucket_name)
                uploaded: list[str] = []
                for filename in _INDEX_FILES:
                    local_file = local_path / filename
                    if not local_file.exists():
                        continue
                    blob_name = f"{self._sync_prefix}{filename}"
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(local_file))
                    uploaded.append(filename)
                    logger.debug("GCS upload: gs://%s/%s", self._bucket_name, blob_name)

                logger.debug(
                    "GCSIndexSync.upload complete — %d files pushed to gs://%s/%s",
                    len(uploaded),
                    self._bucket_name,
                    self._sync_prefix,
                )
                return bool(uploaded)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GCSIndexSync.upload failed (continuing): %s", exc)
                return False

    def download(self, local_path: Path) -> bool:
        """Download all index files from GCS into *local_path*.

        Parameters
        ----------
        local_path:
            Destination directory (created if it does not exist).

        Returns
        -------
        True on success, False on failure or when GCS is not configured.
        """
        if not self._bucket_name:
            return False

        with self._lock:
            client = self._get_client()
            if client is None:
                return False

            try:
                bucket = client.bucket(self._bucket_name)
                local_path.mkdir(parents=True, exist_ok=True)
                downloaded: list[str] = []
                for filename in _INDEX_FILES:
                    blob_name = f"{self._sync_prefix}{filename}"
                    blob = bucket.blob(blob_name)
                    if not blob.exists():
                        logger.debug(
                            "GCS blob not found (skipping): gs://%s/%s",
                            self._bucket_name,
                            blob_name,
                        )
                        continue
                    dest = local_path / filename
                    blob.download_to_filename(str(dest))
                    downloaded.append(filename)
                    logger.debug(
                        "GCS download: gs://%s/%s -> %s",
                        self._bucket_name,
                        blob_name,
                        dest,
                    )

                logger.debug(
                    "GCSIndexSync.download complete — %d files pulled from gs://%s/%s",
                    len(downloaded),
                    self._bucket_name,
                    self._sync_prefix,
                )
                return bool(downloaded)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GCSIndexSync.download failed (continuing): %s", exc)
                return False

    def sync(self, local_path: Path) -> bool:
        """Pull from GCS if local index is missing or GCS copy is newer.

        Intended for cold-start: call this inside ``SemanticMemory._load()``
        before attempting to read local files.

        Logic
        -----
        1. If the local ``metadata.json`` is absent → always download.
        2. If GCS copy is newer (by ``updated`` timestamp) → download.
        3. Otherwise → local is current, skip download.

        Parameters
        ----------
        local_path:
            Persistence directory passed to ``SemanticMemory``.

        Returns
        -------
        True if files were downloaded, False otherwise.
        """
        if not self._bucket_name:
            return False

        client = self._get_client()
        if client is None:
            return False

        local_meta = local_path / "metadata.json"
        if not local_meta.exists():
            logger.debug(
                "Local index absent — pulling from GCS gs://%s/%s",
                self._bucket_name,
                self._sync_prefix,
            )
            return self.download(local_path)

        # Compare timestamps without holding the lock — lock is acquired inside
        # download() only if we decide to pull.
        try:
            bucket = client.bucket(self._bucket_name)
            blob = bucket.blob(f"{self._sync_prefix}metadata.json")
            blob.reload()
            gcs_updated = blob.updated  # datetime (UTC, tz-aware)
            if gcs_updated is None:
                logger.debug("GCS blob has no updated timestamp — skipping pull")
                return False

            local_mtime = local_meta.stat().st_mtime
            import datetime
            local_dt = datetime.datetime.fromtimestamp(local_mtime, tz=timezone.utc)

            if gcs_updated > local_dt:
                logger.debug(
                    "GCS index is newer (gcs=%s local=%s) — pulling",
                    gcs_updated.isoformat(),
                    local_dt.isoformat(),
                )
                return self.download(local_path)
            else:
                logger.debug(
                    "Local index is current (gcs=%s local=%s) — no pull needed",
                    gcs_updated.isoformat(),
                    local_dt.isoformat(),
                )
                return False
        except Exception as exc:  # noqa: BLE001
            logger.debug("GCSIndexSync.sync timestamp check failed: %s", exc)
            return False

    # ── Internals ───────────────────────────────────────────────────────────

    def _get_client(self):
        """Return a cached ``google.cloud.storage.Client``, or None on failure."""
        if self._client is not None:
            return self._client
        try:
            from google.cloud import storage  # type: ignore[import]

            self._client = storage.Client()
            logger.debug("GCS client initialised")
            return self._client
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not initialise GCS client (offline mode): %s", exc
            )
            return None
