"""Automated Retrain Trigger — fires the training pipeline when enough decisions accumulate.

Every decision recorded by DecisionMemoryRecorder calls trigger.tick().
When the accumulated count crosses the threshold AND the minimum cooldown
has elapsed since the last run, training fires in a background thread.

State persists across restarts in data/training_state.json so the counter
is never reset by a process restart.

Config (silo.yaml → training:):
    min_decisions:      50       Decisions since last train before triggering
    min_interval_hours: 6        Cooldown between runs (prevents thrash)
    n_synthetic:        1000     Synthetic samples added to real data each run
    epochs:             60       Max epochs per triggered run
    device:             auto     mps | cuda | cpu | auto

Status is always queryable without blocking:
    trigger.status()  →  {
        decisions_since_last_train: int,
        threshold: int,
        progress_pct: float,        # 0–100
        decisions_until_trigger: int,
        last_train_at: str | None,
        last_train_result: dict | None,
        training_active: bool,
        total_decisions_ever: int,
        runs_completed: int,
    }
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainState:
    """Persisted counter state."""
    decisions_since_last_train: int = 0
    total_decisions_ever: int = 0
    runs_completed: int = 0
    last_train_at: Optional[str] = None          # ISO-8601
    last_train_duration_s: float = 0.0
    last_train_promoted: list[str] = field(default_factory=list)
    last_train_failed: list[str] = field(default_factory=list)
    last_train_accuracies: dict[str, float] = field(default_factory=dict)
    # Rolling domain distribution of accumulated decisions
    domain_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TriggerConfig:
    """Runtime config, loaded from silo.yaml training: block."""
    min_decisions: int = 50          # decisions since last run before triggering
    min_interval_hours: float = 6.0  # cooldown between runs
    n_synthetic: int = 1000          # synthetic samples added each run
    epochs: int = 60                 # max epochs per triggered run
    device: str = "auto"
    checkpoint_dir: str = "data/checkpoints"
    state_file: str = "data/training_state.json"


class RetrainTrigger:
    """Watches the decision counter and fires background retraining at threshold.

    Usage:
        trigger = RetrainTrigger(config)
        trigger.tick(decision_id="DEC-001", domain="sales", verdict="auto_execute")
        print(trigger.status())

    The trigger is non-blocking — training runs in a daemon thread and the
    pipeline continues uninterrupted. Only one training run can be active at
    a time; if a run is already underway, tick() is a no-op until it finishes.
    """

    def __init__(
        self,
        config: Optional[TriggerConfig] = None,
        journal_dir: str = "data/decision_journal",
        outcomes_file: str = "data/learning_loop.jsonl",
    ):
        self.config = config or TriggerConfig()
        self.journal_dir = journal_dir
        self.outcomes_file = outcomes_file

        self._state = self._load_state()
        self._lock = threading.Lock()
        self._training_thread: Optional[threading.Thread] = None
        self._training_active = False

        logger.info(
            "RetrainTrigger initialized: threshold=%d decisions, cooldown=%gh, "
            "current_count=%d, runs_completed=%d",
            self.config.min_decisions,
            self.config.min_interval_hours,
            self._state.decisions_since_last_train,
            self._state.runs_completed,
        )

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────

    def tick(
        self,
        decision_id: str = "",
        domain: str = "general",
        verdict: str = "",
    ) -> bool:
        """Record one decision. Returns True if training was triggered.

        Called by DecisionMemoryRecorder after every successful record().
        Thread-safe — can be called from any thread.
        """
        with self._lock:
            self._state.decisions_since_last_train += 1
            self._state.total_decisions_ever += 1
            self._state.domain_counts[domain] = self._state.domain_counts.get(domain, 0) + 1

        self._persist_state()

        if self._should_trigger():
            self._fire()
            return True
        return False

    def status(self) -> dict:
        """Return the current trigger status. Never blocks."""
        s = self._state
        pct = min(100.0, s.decisions_since_last_train / max(1, self.config.min_decisions) * 100)
        until = max(0, self.config.min_decisions - s.decisions_since_last_train)

        cooldown_remaining = self._cooldown_remaining_seconds()

        return {
            "decisions_since_last_train": s.decisions_since_last_train,
            "threshold": self.config.min_decisions,
            "progress_pct": round(pct, 1),
            "decisions_until_trigger": until,
            "cooldown_remaining_seconds": int(cooldown_remaining),
            "training_active": self._training_active,
            "last_train_at": s.last_train_at,
            "last_train_duration_s": s.last_train_duration_s,
            "last_train_promoted": s.last_train_promoted,
            "last_train_accuracies": s.last_train_accuracies,
            "total_decisions_ever": s.total_decisions_ever,
            "runs_completed": s.runs_completed,
            "domain_distribution": s.domain_counts,
            "ready_to_trigger": self._should_trigger(),
        }

    def force_trigger(self) -> bool:
        """Manually trigger a retrain regardless of threshold/cooldown.

        Returns False if a run is already active.
        """
        if self._training_active:
            logger.warning("force_trigger called but training already active — skipping")
            return False
        logger.info("Manual retrain triggered")
        self._fire()
        return True

    def reset_counter(self) -> None:
        """Reset the decision counter without running training (for testing)."""
        with self._lock:
            self._state.decisions_since_last_train = 0
        self._persist_state()

    # ─────────────────────────────────────────────
    # TRIGGER LOGIC
    # ─────────────────────────────────────────────

    def _should_trigger(self) -> bool:
        if self._training_active:
            return False
        if self._state.decisions_since_last_train < self.config.min_decisions:
            return False
        if self._cooldown_remaining_seconds() > 0:
            logger.debug(
                "Threshold met (%d decisions) but cooldown active (%.0fs remaining)",
                self._state.decisions_since_last_train,
                self._cooldown_remaining_seconds(),
            )
            return False
        return True

    def _cooldown_remaining_seconds(self) -> float:
        if not self._state.last_train_at:
            return 0.0
        try:
            last_dt = datetime.fromisoformat(self._state.last_train_at)
            elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
            cooldown = self.config.min_interval_hours * 3600
            return max(0.0, cooldown - elapsed)
        except Exception:
            return 0.0

    def _fire(self) -> None:
        """Launch training in a background daemon thread."""
        self._training_active = True
        snapshot_count = self._state.decisions_since_last_train

        logger.info(
            "AUTO-RETRAIN TRIGGERED: %d decisions accumulated (threshold=%d), run #%d",
            snapshot_count,
            self.config.min_decisions,
            self._state.runs_completed + 1,
        )

        t = threading.Thread(
            target=self._run_training,
            args=(snapshot_count,),
            daemon=True,
            name=f"retrain-run-{self._state.runs_completed + 1}",
        )
        self._training_thread = t
        t.start()

    def _run_training(self, decisions_at_trigger: int) -> None:
        """Training thread body — runs the full pipeline and updates state."""
        start = time.time()
        try:
            from .pipeline import TrainingPipeline
            from .trainer import TrainingConfig

            cfg = TrainingConfig(
                epochs=self.config.epochs,
                batch_size=64,
                learning_rate=1e-3,
                patience=10,
                device=self.config.device,
                checkpoint_dir=self.config.checkpoint_dir,
            )
            pipeline = TrainingPipeline(
                checkpoint_dir=self.config.checkpoint_dir,
                journal_dir=self.journal_dir,
                outcomes_file=self.outcomes_file,
                config=cfg,
            )

            logger.info(
                "Retrain run #%d starting: %d synthetic + historical data from %s",
                self._state.runs_completed + 1,
                self.config.n_synthetic,
                self.journal_dir,
            )

            result = pipeline.run(
                epochs=self.config.epochs,
                n_synthetic=self.config.n_synthetic,
                models=None,
                save_report=True,
            )

            duration = time.time() - start

            # Collect per-model accuracy from results
            accuracies = {}
            for name, r in result.model_results.items():
                if r.val_accuracy is not None:
                    accuracies[name] = round(r.val_accuracy, 4)
                else:
                    accuracies[name] = round(1.0 - min(1.0, r.val_loss), 4)

            with self._lock:
                self._state.runs_completed += 1
                self._state.decisions_since_last_train = 0  # reset counter
                self._state.last_train_at = datetime.now(timezone.utc).isoformat()
                self._state.last_train_duration_s = round(duration, 1)
                self._state.last_train_promoted = result.promoted
                self._state.last_train_failed = result.failed
                self._state.last_train_accuracies = accuracies

            self._persist_state()

            logger.info(
                "Retrain run #%d COMPLETE in %.0fs: promoted=%s failed=%s",
                self._state.runs_completed,
                duration,
                result.promoted,
                result.failed,
            )

        except Exception as exc:
            logger.exception("Retrain run failed: %s", exc)
        finally:
            self._training_active = False

    # ─────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────

    def _load_state(self) -> TrainState:
        path = Path(self.config.state_file)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                state = TrainState.from_dict(data)
                logger.debug("Loaded training state: %d decisions since last train", state.decisions_since_last_train)
                return state
            except Exception as e:
                logger.warning("Could not load training state (%s) — starting fresh", e)
        return TrainState()

    def _persist_state(self) -> None:
        path = Path(self.config.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(self._state.to_dict(), indent=2))
        except Exception as e:
            logger.warning("Could not persist training state: %s", e)
