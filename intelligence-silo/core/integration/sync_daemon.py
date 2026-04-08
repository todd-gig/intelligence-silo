"""Sync Daemon — automated daily memory consolidation and remote backup.

Schedule:
- CONTINUOUS: Every decision auto-records to local memory (via DecisionMemoryRecorder)
- IMPORTANT: High-value decisions sync to remote within minutes
- DAILY (2:00 AM): Full consolidation + full remote sync during system sleep

The daemon runs as a background asyncio task within the intelligence node.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Default sleep-time sync hour (24h format, local time)
# "System sleep time" = the boundary between one calendar day and the next = midnight
SLEEP_SYNC_HOUR = 0  # 00:00 — midnight, the day boundary


class SyncDaemon:
    """Background daemon that manages automated memory sync and backup.

    Three loops running concurrently:
    1. Consolidation loop (every 60s): promotes episodic → semantic memory
    2. Important sync loop (every 5min): pushes high-value items to remote
    3. Daily full sync (at 2:00 AM): complete backup during system sleep
    """

    def __init__(self, memory_hierarchy, recorder, vault, backup_manager,
                 consolidation_interval: float = 60.0,
                 important_interval: float = 300.0,
                 sleep_sync_hour: int = SLEEP_SYNC_HOUR):
        self.memory = memory_hierarchy
        self.recorder = recorder
        self.vault = vault
        self.backup = backup_manager
        self.consolidation_interval = consolidation_interval
        self.important_interval = important_interval
        self.sleep_sync_hour = sleep_sync_hour
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._last_daily_sync: str = ""
        self._stats = {
            "consolidation_runs": 0,
            "important_syncs": 0,
            "full_syncs": 0,
            "total_memories_promoted": 0,
        }

    async def start(self) -> None:
        """Start all background sync loops."""
        self._running = True
        logger.info("SyncDaemon starting — consolidation=%ds, important=%ds, daily=%d:00",
                     self.consolidation_interval, self.important_interval, self.sleep_sync_hour)

        self._tasks = [
            asyncio.create_task(self._consolidation_loop()),
            asyncio.create_task(self._important_sync_loop()),
            asyncio.create_task(self._daily_sync_loop()),
        ]

    async def stop(self) -> None:
        """Stop all background loops and run a final sync."""
        self._running = False
        for task in self._tasks:
            task.cancel()

        # Final sync before shutdown
        logger.info("SyncDaemon stopping — running final sync")
        self._run_consolidation()
        self._run_important_sync()

    async def _consolidation_loop(self) -> None:
        """Consolidate memory every interval: evict working → episodic → semantic."""
        while self._running:
            try:
                await asyncio.sleep(self.consolidation_interval)
                self._run_consolidation()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Consolidation error: %s", e)

    async def _important_sync_loop(self) -> None:
        """Sync important items to remote every few minutes."""
        while self._running:
            try:
                await asyncio.sleep(self.important_interval)
                self._run_important_sync()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Important sync error: %s", e)

    async def _daily_sync_loop(self) -> None:
        """Full sync at the designated sleep hour."""
        while self._running:
            try:
                now = datetime.now()
                # Calculate next sync time
                next_sync = now.replace(
                    hour=self.sleep_sync_hour, minute=0, second=0, microsecond=0
                )
                if now >= next_sync:
                    next_sync += timedelta(days=1)

                wait_seconds = (next_sync - now).total_seconds()
                logger.info("Daily sync scheduled in %.0f seconds (at %s)",
                           wait_seconds, next_sync.strftime("%H:%M"))

                await asyncio.sleep(wait_seconds)
                self._run_full_sync()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Daily sync error: %s", e)
                await asyncio.sleep(3600)  # retry in 1h

    # ── Sync Operations ─────────────────────────────────────────────────────

    def _run_consolidation(self) -> None:
        """Run memory consolidation cycle."""
        stats = self.memory.consolidate()
        self._stats["consolidation_runs"] += 1
        self._stats["total_memories_promoted"] += stats.get("promoted_semantic", 0)

        if stats.get("promoted_semantic", 0) > 0 or stats.get("evicted_working", 0) > 0:
            logger.info(
                "Consolidation: evicted=%d, promoted=%d",
                stats.get("evicted_working", 0),
                stats.get("promoted_semantic", 0),
            )

    def _run_important_sync(self) -> None:
        """Sync important items to remote backup."""
        if not self.backup.remote_repo:
            return

        # Sync vault if it has changed
        vault_blob = self.vault.get_encrypted_blob()
        if vault_blob:
            self.backup.sync_vault(vault_blob)

        # Sync queued important files
        result = self.backup.sync_important()
        if result.get("synced", 0) > 0:
            self._stats["important_syncs"] += 1
            logger.info("Important sync: %d files", result["synced"])

    def _run_full_sync(self) -> None:
        """Full daily sync — everything goes to remote."""
        logger.info("Starting full daily sync...")

        # 1. Flush daily decision records
        daily_stats = self.recorder.flush_daily()
        logger.info("Daily flush: %d decisions", daily_stats.get("total_decisions", 0))

        # 2. Persist semantic memory to disk
        self.memory.save()

        # 3. Full remote sync
        if self.backup.remote_repo:
            result = self.backup.sync_full()
            logger.info("Full sync: %d files synced", result.get("synced", 0))

        self._stats["full_syncs"] += 1
        self._last_daily_sync = datetime.now(timezone.utc).isoformat()
        logger.info("Full daily sync complete")

    # ── Manual triggers ─────────────────────────────────────────────────────

    def trigger_consolidation(self) -> dict:
        """Manually trigger a consolidation cycle."""
        self._run_consolidation()
        return {"status": "completed", "stats": self._stats}

    def trigger_important_sync(self) -> dict:
        """Manually trigger an important sync."""
        self._run_important_sync()
        return {"status": "completed", "stats": self._stats}

    def trigger_full_sync(self) -> dict:
        """Manually trigger a full sync (equivalent to sleep-time sync)."""
        self._run_full_sync()
        return {"status": "completed", "stats": self._stats}

    def health(self) -> dict:
        return {
            "running": self._running,
            "last_daily_sync": self._last_daily_sync,
            "stats": self._stats,
            "backup": self.backup.health(),
            "memory": self.memory.health(),
        }
