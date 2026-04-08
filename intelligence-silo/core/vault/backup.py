"""Git Backup Manager — syncs encrypted vault + important memories to a private GitHub repo.

Strategy:
- ALL information is saved locally first (always available, always fast)
- IMPORTANT information (high-value decisions, credentials vault) syncs to remote immediately
- ALL information syncs during "system sleep time" (daily scheduled sync)

The remote backup is a PRIVATE GitHub repository containing:
1. Encrypted credentials vault (useless without machine-bound decryption key)
2. Semantic memory index (knowledge consolidated from decisions)
3. Decision journal (daily decision records)
4. Learning loop outcomes
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SyncManifest:
    """Tracks what has been synced and when."""
    last_sync: str = ""
    last_important_sync: str = ""
    last_full_sync: str = ""
    files_synced: int = 0
    sync_errors: list[str] = field(default_factory=list)


class GitBackupManager:
    """Manages remote backup of intelligence silo data to a private GitHub repo.

    Three sync modes:
    1. IMMEDIATE: High-value data syncs right away (vault changes, critical decisions)
    2. IMPORTANT: Flagged items sync on next check-in (high-value decisions, learning outcomes)
    3. FULL: Everything syncs during system sleep (daily 2am cycle)

    The repo structure:
    ```
    backup-repo/
    ├── vault/
    │   └── credentials.vault  (encrypted blob — machine-bound)
    ├── memory/
    │   ├── semantic/          (FAISS index + metadata)
    │   └── journal/           (daily decision records)
    ├── config/
    │   └── silo.yaml          (non-sensitive config)
    └── manifest.json          (sync metadata)
    ```
    """

    def __init__(self, local_data_root: Path, remote_repo: str | None = None,
                 gh_binary: str = "/opt/homebrew/bin/gh"):
        self.local_root = local_data_root
        self.remote_repo = remote_repo
        self.gh = gh_binary
        self.backup_dir = Path.home() / ".intelligence-silo" / "backup-staging"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()
        self._important_queue: list[Path] = []

    def ensure_remote_repo(self, repo_name: str = "intelligence-silo-backup") -> str:
        """Create the private backup repo if it doesn't exist."""
        if self.remote_repo:
            return self.remote_repo

        try:
            # Check if repo exists
            result = subprocess.run(
                [self.gh, "repo", "view", f"todd-gig/{repo_name}", "--json", "name"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                self.remote_repo = f"todd-gig/{repo_name}"
                return self.remote_repo

            # Create private repo
            result = subprocess.run(
                [self.gh, "repo", "create", f"todd-gig/{repo_name}",
                 "--private",
                 "--description", "Encrypted backup for intelligence silo — vault + memory + journal"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                self.remote_repo = f"todd-gig/{repo_name}"
                logger.info("Created private backup repo: %s", self.remote_repo)

                # Add collaborator
                subprocess.run(
                    [self.gh, "api", f"repos/todd-gig/{repo_name}/collaborators/bella-byte",
                     "-X", "PUT", "-f", "permission=push"],
                    capture_output=True, text=True, timeout=15,
                )

                # Initialize with README
                self._init_backup_repo(repo_name)
                return self.remote_repo

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error("Failed to ensure backup repo: %s", e)

        return ""

    def _init_backup_repo(self, repo_name: str) -> None:
        """Initialize the backup repo with structure."""
        staging = self.backup_dir / "init"
        staging.mkdir(parents=True, exist_ok=True)

        # Clone, add structure, push
        try:
            subprocess.run(
                ["git", "clone", f"https://github.com/todd-gig/{repo_name}.git",
                 str(staging / repo_name)],
                capture_output=True, text=True, timeout=30,
            )
            repo_dir = staging / repo_name
            for d in ["vault", "memory/semantic", "memory/journal", "config"]:
                (repo_dir / d).mkdir(parents=True, exist_ok=True)
                (repo_dir / d / ".gitkeep").touch()

            # .gitignore to prevent accidental sensitive leaks
            (repo_dir / ".gitignore").write_text(
                "# Never commit unencrypted secrets\n"
                "*.key\n*.pem\n*.env\n.salt\n.machine_id\n"
            )

            subprocess.run(
                ["git", "-C", str(repo_dir), "add", "-A"],
                capture_output=True, timeout=10,
            )
            subprocess.run(
                ["git", "-C", str(repo_dir), "commit", "-m",
                 "init: backup repo structure for intelligence silo"],
                capture_output=True, timeout=10,
            )
            subprocess.run(
                ["git", "-C", str(repo_dir), "push", "origin", "main"],
                capture_output=True, timeout=30,
            )
        except Exception as e:
            logger.error("Backup repo init failed: %s", e)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # ── Sync Operations ─────────────────────────────────────────────────────

    def sync_vault(self, encrypted_blob: bytes) -> bool:
        """IMMEDIATE sync: push encrypted vault to remote."""
        if not self.remote_repo:
            return False

        vault_path = self.backup_dir / "vault" / "credentials.vault"
        vault_path.parent.mkdir(parents=True, exist_ok=True)
        vault_path.write_bytes(encrypted_blob)

        return self._push_file(
            local_path=vault_path,
            remote_path="vault/credentials.vault",
            commit_msg="vault: encrypted credential update",
        )

    def queue_important(self, local_path: Path) -> None:
        """Queue a file for IMPORTANT sync (next check-in)."""
        self._important_queue.append(local_path)

    def sync_important(self) -> dict:
        """Sync all queued important files to remote."""
        if not self.remote_repo or not self._important_queue:
            return {"synced": 0}

        synced = 0
        errors = []
        for path in self._important_queue:
            if path.exists():
                rel = path.relative_to(self.local_root) if path.is_relative_to(self.local_root) else Path(path.name)
                remote_path = f"memory/{rel}"
                if self._push_file(path, remote_path, f"sync: {path.name}"):
                    synced += 1
                else:
                    errors.append(str(path))

        self._important_queue.clear()
        self.manifest.last_important_sync = datetime.now(timezone.utc).isoformat()
        self._save_manifest()
        return {"synced": synced, "errors": errors}

    def sync_full(self) -> dict:
        """FULL sync: everything during system sleep time.

        Syncs:
        1. Encrypted vault
        2. Semantic memory index
        3. Decision journal
        4. Config (non-sensitive)
        """
        if not self.remote_repo:
            return {"error": "no_remote_repo"}

        stats = {"synced": 0, "errors": []}

        # Collect all syncable files
        sync_targets = []

        # Vault
        vault_file = Path.home() / ".intelligence-silo" / "vault" / "credentials.vault"
        if vault_file.exists():
            sync_targets.append((vault_file, "vault/credentials.vault"))

        # Semantic memory
        sem_path = self.local_root / "data" / "semantic_index"
        if sem_path.exists():
            for f in sem_path.iterdir():
                if f.is_file():
                    sync_targets.append((f, f"memory/semantic/{f.name}"))

        # Decision journal
        journal_path = self.local_root / "data" / "decision_journal"
        if journal_path.exists():
            for day_dir in sorted(journal_path.iterdir()):
                if day_dir.is_dir():
                    for f in day_dir.iterdir():
                        if f.is_file():
                            sync_targets.append((
                                f, f"memory/journal/{day_dir.name}/{f.name}"
                            ))

        # Config
        config_path = self.local_root / "config" / "silo.yaml"
        if config_path.exists():
            sync_targets.append((config_path, "config/silo.yaml"))

        # Batch push via git
        if sync_targets:
            success = self._batch_push(sync_targets, "sync: full daily backup")
            stats["synced"] = len(sync_targets) if success else 0
            if not success:
                stats["errors"].append("batch_push_failed")

        self.manifest.last_full_sync = datetime.now(timezone.utc).isoformat()
        self.manifest.files_synced += stats["synced"]
        self._save_manifest()

        logger.info("Full sync complete: %d files", stats["synced"])
        return stats

    def _push_file(self, local_path: Path, remote_path: str,
                   commit_msg: str) -> bool:
        """Push a single file to the remote backup repo via GitHub API."""
        try:
            content = local_path.read_bytes()
            import base64
            encoded = base64.b64encode(content).decode()

            # Get existing file SHA (for updates)
            sha = self._get_file_sha(remote_path)

            cmd = [
                self.gh, "api",
                f"repos/{self.remote_repo}/contents/{remote_path}",
                "-X", "PUT",
                "-f", f"message={commit_msg}",
                "-f", f"content={encoded}",
            ]
            if sha:
                cmd.extend(["-f", f"sha={sha}"])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0

        except Exception as e:
            logger.error("Push failed for %s: %s", remote_path, e)
            return False

    def _batch_push(self, targets: list[tuple[Path, str]], commit_msg: str) -> bool:
        """Push multiple files via git clone/commit/push."""
        staging = self.backup_dir / "batch"
        try:
            shutil.rmtree(staging, ignore_errors=True)

            # Clone
            result = subprocess.run(
                ["git", "clone", "--depth", "1",
                 f"https://github.com/{self.remote_repo}.git",
                 str(staging)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                return False

            # Copy files
            for local_path, remote_path in targets:
                dest = staging / remote_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_path, dest)

            # Commit and push
            subprocess.run(["git", "-C", str(staging), "add", "-A"],
                          capture_output=True, timeout=10)
            result = subprocess.run(
                ["git", "-C", str(staging), "commit", "-m", commit_msg],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return True  # nothing to commit = already synced

            result = subprocess.run(
                ["git", "-C", str(staging), "push", "origin", "main"],
                capture_output=True, text=True, timeout=60,
            )
            return result.returncode == 0

        except Exception as e:
            logger.error("Batch push failed: %s", e)
            return False
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _get_file_sha(self, remote_path: str) -> str | None:
        """Get the SHA of an existing file in the remote repo."""
        try:
            result = subprocess.run(
                [self.gh, "api", f"repos/{self.remote_repo}/contents/{remote_path}",
                 "--jq", ".sha"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return None

    # ── Manifest ────────────────────────────────────────────────────────────

    def _load_manifest(self) -> SyncManifest:
        manifest_path = self.backup_dir / "manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text())
                return SyncManifest(**data)
            except Exception:
                pass
        return SyncManifest()

    def _save_manifest(self) -> None:
        manifest_path = self.backup_dir / "manifest.json"
        data = {
            "last_sync": self.manifest.last_sync,
            "last_important_sync": self.manifest.last_important_sync,
            "last_full_sync": self.manifest.last_full_sync,
            "files_synced": self.manifest.files_synced,
            "sync_errors": self.manifest.sync_errors[-20:],
        }
        manifest_path.write_text(json.dumps(data, indent=2))

    def health(self) -> dict:
        return {
            "remote_repo": self.remote_repo,
            "last_important_sync": self.manifest.last_important_sync,
            "last_full_sync": self.manifest.last_full_sync,
            "files_synced_total": self.manifest.files_synced,
            "important_queue": len(self._important_queue),
        }
