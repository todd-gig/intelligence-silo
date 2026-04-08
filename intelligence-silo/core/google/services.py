"""Google Services — shared memory via GCS, fallback inference via Vertex AI, state sync via Firebase."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GoogleConfig:
    enabled: bool = False
    project_id: str | None = None
    gcs_bucket: str | None = None
    gcs_sync_path: str = "intelligence-silo/shared-memory"
    vertex_region: str = "us-central1"
    vertex_fallback_model: str = "gemini-2.0-flash"
    firebase_enabled: bool = False
    firebase_collection: str = "silo_state"


class GoogleServices:
    """Integration layer for Google Cloud services.

    Three responsibilities:
    1. GCS: Shared memory sync — nodes upload/download semantic memory indices
       so the mesh shares long-term knowledge without direct P2P.
    2. Vertex AI: Fallback inference — when local SLMs don't have sufficient
       confidence, fall back to a hosted model (Gemini) for a second opinion.
    3. Firebase: State sync — real-time node state broadcasting for mesh awareness.

    All services are optional and gracefully degrade when unavailable.
    """

    def __init__(self, config: GoogleConfig):
        self.config = config
        self._gcs_client = None
        self._vertex_client = None
        self._firebase_db = None

        if config.enabled:
            self._init_services()

    def _init_services(self) -> None:
        """Initialize Google service clients."""
        # GCS
        if self.config.gcs_bucket:
            try:
                from google.cloud import storage
                self._gcs_client = storage.Client(project=self.config.project_id)
                logger.info("GCS client initialized, bucket: %s", self.config.gcs_bucket)
            except Exception as e:
                logger.warning("GCS init failed (will work offline): %s", e)

        # Vertex AI
        try:
            from google.cloud import aiplatform
            aiplatform.init(
                project=self.config.project_id,
                location=self.config.vertex_region,
            )
            self._vertex_client = aiplatform
            logger.info("Vertex AI initialized, region: %s", self.config.vertex_region)
        except Exception as e:
            logger.warning("Vertex AI init failed (local-only mode): %s", e)

        # Firebase
        if self.config.firebase_enabled:
            try:
                import firebase_admin
                from firebase_admin import firestore
                if not firebase_admin._apps:
                    firebase_admin.initialize_app()
                self._firebase_db = firestore.client()
                logger.info("Firebase initialized, collection: %s", self.config.firebase_collection)
            except Exception as e:
                logger.warning("Firebase init failed: %s", e)

    # ── GCS: Shared Memory ──────────────────────────────────────────────────

    def upload_memory_index(self, local_path: Path, node_id: str) -> bool:
        """Upload a semantic memory index to GCS for mesh sharing."""
        if not self._gcs_client or not self.config.gcs_bucket:
            return False

        try:
            bucket = self._gcs_client.bucket(self.config.gcs_bucket)
            for file in local_path.iterdir():
                if file.is_file():
                    blob_path = f"{self.config.gcs_sync_path}/{node_id}/{file.name}"
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(str(file))
            logger.info("Uploaded memory index for node %s", node_id)
            return True
        except Exception as e:
            logger.error("Memory upload failed: %s", e)
            return False

    def download_memory_index(self, node_id: str, local_path: Path) -> bool:
        """Download a peer node's memory index from GCS."""
        if not self._gcs_client or not self.config.gcs_bucket:
            return False

        try:
            bucket = self._gcs_client.bucket(self.config.gcs_bucket)
            prefix = f"{self.config.gcs_sync_path}/{node_id}/"
            blobs = bucket.list_blobs(prefix=prefix)
            local_path.mkdir(parents=True, exist_ok=True)
            for blob in blobs:
                filename = blob.name.split("/")[-1]
                blob.download_to_filename(str(local_path / filename))
            logger.info("Downloaded memory index for node %s", node_id)
            return True
        except Exception as e:
            logger.error("Memory download failed: %s", e)
            return False

    def list_peer_nodes(self) -> list[str]:
        """List all node IDs that have uploaded memory indices."""
        if not self._gcs_client or not self.config.gcs_bucket:
            return []

        try:
            bucket = self._gcs_client.bucket(self.config.gcs_bucket)
            prefix = f"{self.config.gcs_sync_path}/"
            blobs = bucket.list_blobs(prefix=prefix, delimiter="/")
            # Extract node IDs from prefixes
            nodes = set()
            for prefix in blobs.prefixes:
                parts = prefix.rstrip("/").split("/")
                if parts:
                    nodes.add(parts[-1])
            return sorted(nodes)
        except Exception as e:
            logger.error("Peer listing failed: %s", e)
            return []

    # ── Vertex AI: Fallback Inference ───────────────────────────────────────

    async def fallback_inference(self, prompt: str, context: dict | None = None) -> dict:
        """Call Vertex AI Gemini when local models lack confidence.

        This is the "phone a friend" mechanism — local-first, but with
        access to a powerful hosted model when needed.
        """
        if not self._vertex_client:
            return {"error": "vertex_ai_unavailable", "response": None}

        try:
            from vertexai.generative_models import GenerativeModel

            model = GenerativeModel(self.config.vertex_fallback_model)
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {json.dumps(context)}\n\n{prompt}"

            response = model.generate_content(full_prompt)
            return {
                "response": response.text,
                "model": self.config.vertex_fallback_model,
                "source": "vertex_ai",
            }
        except Exception as e:
            logger.error("Vertex AI inference failed: %s", e)
            return {"error": str(e), "response": None}

    # ── Firebase: State Sync ────────────────────────────────────────────────

    def broadcast_state(self, node_id: str, state: dict) -> bool:
        """Broadcast this node's state to Firebase for mesh awareness."""
        if not self._firebase_db:
            return False

        try:
            doc_ref = self._firebase_db.collection(
                self.config.firebase_collection
            ).document(node_id)
            doc_ref.set(state, merge=True)
            return True
        except Exception as e:
            logger.error("State broadcast failed: %s", e)
            return False

    def get_mesh_state(self) -> dict[str, dict]:
        """Get the state of all nodes in the mesh."""
        if not self._firebase_db:
            return {}

        try:
            docs = self._firebase_db.collection(
                self.config.firebase_collection
            ).stream()
            return {doc.id: doc.to_dict() for doc in docs}
        except Exception as e:
            logger.error("Mesh state retrieval failed: %s", e)
            return {}

    # ── Health ──────────────────────────────────────────────────────────────

    @property
    def available_services(self) -> list[str]:
        services = []
        if self._gcs_client:
            services.append("gcs")
        if self._vertex_client:
            services.append("vertex_ai")
        if self._firebase_db:
            services.append("firebase")
        return services

    def health(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "services": self.available_services,
            "project_id": self.config.project_id,
        }
