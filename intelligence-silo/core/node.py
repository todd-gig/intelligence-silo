"""Node — the top-level runtime that ties everything together."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import yaml
import torch

from .memory.hierarchy import MemoryHierarchy, MemoryConfig
from .models.matrix import SLMMatrix
from .orchestrator.society import SocietyOfMinds
from .bridge.connector import DecisionEngineBridge
from .google.services import GoogleServices, GoogleConfig
from .integration.recorder import DecisionMemoryRecorder
from .integration.sync_daemon import SyncDaemon
from .vault.vault import SecureVault
from .vault.backup import GitBackupManager

logger = logging.getLogger(__name__)


class IntelligenceNode:
    """The Intelligence Silo — a self-contained neural intelligence node.

    Composes all subsystems:
    - Memory Hierarchy (working/episodic/semantic/procedural)
    - SLM Matrix (6 specialist micro-models)
    - Society of Minds (7-role multi-agent orchestrator)
    - Decision Engine Bridge (bidirectional sync)
    - Decision Memory Recorder (auto-records all pipeline results)
    - Secure Vault (encrypted local credential storage)
    - Git Backup Manager (private repo failsafe)
    - Sync Daemon (daily consolidation + midnight full sync)
    - Google Services (GCS, Vertex AI, Firebase)

    Each node can run standalone or as part of a mesh.
    """

    def __init__(self, config_path: str | Path = "config/silo.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.node_id = self.config.get("node", {}).get("id") or uuid.uuid4().hex[:12]
        self.node_name = self.config.get("node", {}).get("name", "primary")
        self.device = self._resolve_device()

        logger.info("Initializing Intelligence Node: %s (%s) on %s",
                     self.node_name, self.node_id, self.device)

        # Build subsystems
        self.memory = self._build_memory()
        self.matrix = self._build_matrix()
        self.society = SocietyOfMinds(self.matrix, self.memory)
        self.bridge = self._build_bridge()
        self.google = self._build_google()

        # Integration layer
        self.vault = SecureVault()
        self.recorder = DecisionMemoryRecorder(
            memory_hierarchy=self.memory,
            local_journal_path=str(Path("data") / "decision_journal"),
        )
        self.backup = GitBackupManager(
            local_data_root=Path("."),
            remote_repo=self.config.get("backup", {}).get("remote_repo"),
        )
        self.daemon = SyncDaemon(
            memory_hierarchy=self.memory,
            recorder=self.recorder,
            vault=self.vault,
            backup_manager=self.backup,
            consolidation_interval=60.0,
            important_interval=300.0,
            sleep_sync_hour=0,  # midnight — day boundary
        )

        self._running = False

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        logger.warning("Config not found at %s, using defaults", self.config_path)
        return {}

    def _resolve_device(self) -> str:
        dev = self.config.get("node", {}).get("device", "auto")
        if dev == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return dev

    def _build_memory(self) -> MemoryHierarchy:
        mcfg = self.config.get("memory", {})
        return MemoryHierarchy(MemoryConfig(
            working_capacity=mcfg.get("working", {}).get("capacity", 128),
            working_ttl=mcfg.get("working", {}).get("ttl_seconds", 300),
            working_heads=mcfg.get("working", {}).get("attention_heads", 4),
            episodic_max=mcfg.get("episodic", {}).get("max_episodes", 10000),
            episodic_similarity=mcfg.get("episodic", {}).get("similarity_threshold", 0.72),
            episodic_consolidation=mcfg.get("episodic", {}).get("consolidation_interval", 60),
            semantic_max=mcfg.get("semantic", {}).get("max_vectors", 1_000_000),
            semantic_index_type=mcfg.get("semantic", {}).get("index_type", "IVFFlat"),
            semantic_nprobe=mcfg.get("semantic", {}).get("nprobe", 16),
            semantic_path=mcfg.get("semantic", {}).get("persistence_path"),
            procedural_max=mcfg.get("procedural", {}).get("max_procedures", 5000),
            procedural_threshold=mcfg.get("procedural", {}).get("execution_threshold", 0.85),
            procedural_lr=mcfg.get("procedural", {}).get("learning_rate", 0.001),
            embedding_dim=mcfg.get("semantic", {}).get("embedding_dim", 384),
            device=self.device,
        ))

    def _build_matrix(self) -> SLMMatrix:
        slm_cfg = self.config.get("slm_matrix", {})
        models = slm_cfg.get("models", [])
        if not models:
            models = [
                {"name": "classifier", "type": "sequence_classification",
                 "hidden_dim": 256, "num_layers": 4, "num_heads": 4,
                 "vocab_size": 8192, "num_classes": 32},
                {"name": "scorer", "type": "regression",
                 "hidden_dim": 256, "num_layers": 4, "num_heads": 4,
                 "vocab_size": 8192, "output_dim": 12},
                {"name": "trust_assessor", "type": "sequence_classification",
                 "hidden_dim": 192, "num_layers": 3, "num_heads": 4,
                 "vocab_size": 8192, "num_classes": 5},
                {"name": "memory_encoder", "type": "encoder",
                 "hidden_dim": 384, "num_layers": 6, "num_heads": 6,
                 "vocab_size": 8192},
                {"name": "pattern_detector", "type": "sequence_classification",
                 "hidden_dim": 256, "num_layers": 4, "num_heads": 4,
                 "vocab_size": 8192, "num_classes": 64},
                {"name": "causal_predictor", "type": "regression",
                 "hidden_dim": 384, "num_layers": 6, "num_heads": 6,
                 "vocab_size": 8192, "output_dim": 16},
            ]

        router_cfg = slm_cfg.get("router", {})
        return SLMMatrix(
            configs=models,
            router_strategy=router_cfg.get("strategy", "attention"),
            fusion_method=router_cfg.get("fusion_method", "weighted_vote"),
            confidence_floor=router_cfg.get("confidence_floor", 0.6),
            device=self.device,
        )

    def _build_bridge(self) -> DecisionEngineBridge:
        bcfg = self.config.get("bridge", {})
        bridge = DecisionEngineBridge(
            engine_url=bcfg.get("decision_engine_url", "http://localhost:8000"),
            sync_interval=bcfg.get("sync_interval", 30),
        )
        engine_yaml = Path("decision-engine/config/engine.yaml")
        if engine_yaml.exists():
            bridge.load_local_config(str(engine_yaml))
        return bridge

    def _build_google(self) -> GoogleServices:
        gcfg = self.config.get("google", {})
        return GoogleServices(GoogleConfig(
            enabled=gcfg.get("enabled", False),
            project_id=gcfg.get("project_id"),
            gcs_bucket=gcfg.get("gcs", {}).get("bucket"),
            gcs_sync_path=gcfg.get("gcs", {}).get("sync_path", "intelligence-silo/shared-memory"),
            vertex_region=gcfg.get("vertex_ai", {}).get("region", "us-central1"),
            vertex_fallback_model=gcfg.get("vertex_ai", {}).get("fallback_model", "gemini-2.0-flash"),
            firebase_enabled=gcfg.get("firebase", {}).get("enabled", False),
            firebase_collection=gcfg.get("firebase", {}).get("collection", "silo_state"),
        ))

    # ── Core Operations ─────────────────────────────────────────────────────

    @torch.no_grad()
    def process(self, input_data: dict, input_ids: torch.Tensor | None = None,
                query_embedding: torch.Tensor | None = None) -> dict:
        """Process input through the full intelligence pipeline.

        Single-cycle fast path for latency-sensitive decisions.
        """
        result = self.society.think(input_data, input_ids, query_embedding)
        return {
            "node_id": self.node_id,
            "consensus": result.consensus,
            "confidence": result.confidence,
            "verdict": result.verdict.content if result.verdict else None,
            "minds_activated": result.minds_activated,
            "cycle_time_ms": result.cycle_time_ms,
        }

    @torch.no_grad()
    def deliberate(self, input_data: dict, input_ids: torch.Tensor | None = None,
                   query_embedding: torch.Tensor | None = None) -> dict:
        """Multi-round deliberation for high-stakes decisions."""
        result = self.society.deliberate(input_data, input_ids, query_embedding)
        return {
            "node_id": self.node_id,
            "consensus": result.consensus,
            "confidence": result.confidence,
            "verdict": result.verdict.content if result.verdict else None,
            "minds_activated": result.minds_activated,
            "cycle_time_ms": result.cycle_time_ms,
        }

    def record_decision(self, pipeline_result: dict, title: str = "",
                        domain: str = "general") -> dict:
        """Record a decision engine pipeline result into memory.

        This is the primary integration point — call this after every
        process_decision() in the decision engine pipeline.
        """
        record = self.recorder.record(pipeline_result, title, domain)

        # Queue high-value decisions for immediate remote backup
        if record.net_value >= 20 or record.verdict in ("block", "escalate_tier_2", "escalate_tier_3"):
            journal_path = self.recorder.journal_path / datetime.now().strftime("%Y-%m-%d")
            for f in journal_path.iterdir() if journal_path.exists() else []:
                if record.decision_id in f.name:
                    self.backup.queue_important(f)

        return {
            "recorded": True,
            "decision_id": record.decision_id,
            "verdict": record.verdict,
            "importance": self.recorder._compute_importance(record),
            "memory_layers": ["working", "episodic"] + (
                ["semantic"] if record.net_value >= 20 else []
            ) + (["procedural"] if record.verdict == "auto_execute" else []),
        }

    def consolidate_memory(self) -> dict:
        """Run memory consolidation cycle."""
        return self.memory.consolidate()

    # ── Vault Operations ────────────────────────────────────────────────────

    def store_credential(self, key: str, value: str, service: str = "") -> dict:
        """Store a credential in the encrypted vault."""
        entry = self.vault.store(key, value, service=service)
        # Immediately sync vault to remote
        blob = self.vault.get_encrypted_blob()
        if blob and self.backup.remote_repo:
            self.backup.sync_vault(blob)
        return {"stored": True, "key": key, "service": service}

    def get_credential(self, key: str) -> str | None:
        """Retrieve a credential from the vault."""
        return self.vault.get(key)

    def import_env_credentials(self, keys: list[str]) -> dict:
        """Import credentials from environment variables into the vault."""
        count = self.vault.store_env_keys(keys)
        return {"imported": count, "keys": keys}

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the node's background tasks."""
        self._running = True
        logger.info("Node %s started", self.node_id)

        # Ensure backup repo exists
        self.backup.ensure_remote_repo()

        # Start sync daemon (handles consolidation + backup loops)
        await self.daemon.start()

        # Background bridge sync
        if self.bridge:
            asyncio.create_task(self._bridge_sync_loop())

    async def stop(self) -> None:
        """Gracefully stop the node."""
        self._running = False

        # Stop daemon (runs final sync)
        await self.daemon.stop()

        # Save all persistent state
        self.memory.save()
        await self.bridge.close()
        logger.info("Node %s stopped", self.node_id)

    async def _bridge_sync_loop(self) -> None:
        while self._running:
            if self.bridge.needs_sync():
                await self.bridge.sync_remote()
            await asyncio.sleep(self.bridge.sync_interval)

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Full node health report."""
        return {
            "node": {
                "id": self.node_id,
                "name": self.node_name,
                "device": self.device,
                "running": self._running,
            },
            "matrix": {
                "total_params": self.matrix.total_params_human,
                "models": self.matrix.performance_report(),
            },
            "society": self.society.health(),
            "memory": self.memory.health(),
            "vault": self.vault.health(),
            "recorder": self.recorder.stats,
            "daemon": self.daemon.health(),
            "google": self.google.health(),
        }


# Needed for record_decision's date check
from datetime import datetime
