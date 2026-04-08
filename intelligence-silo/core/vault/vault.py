"""Secure Vault — encrypted local storage for tokens, API keys, and sensitive configuration.

ALL credentials live locally on-device, encrypted at rest with Fernet (AES-128-CBC + HMAC).
The vault key is derived from a machine-specific secret via PBKDF2.

Principle: information stored locally always has inherent advantages —
faster access, no network dependency, no third-party trust required.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

VAULT_DIR = Path.home() / ".intelligence-silo" / "vault"
VAULT_FILE = VAULT_DIR / "credentials.vault"
SALT_FILE = VAULT_DIR / ".salt"
MACHINE_ID_FILE = VAULT_DIR / ".machine_id"


@dataclass
class VaultEntry:
    """A single credential stored in the vault."""
    key: str
    value: str  # the actual secret
    service: str = ""
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed: str = ""
    access_count: int = 0


class SecureVault:
    """Encrypted credential store — local-first, machine-bound.

    The vault:
    1. Derives an encryption key from a machine-specific identifier + salt via PBKDF2
    2. Encrypts all credentials with Fernet (AES-128-CBC + HMAC-SHA256)
    3. Stores encrypted blob locally at ~/.intelligence-silo/vault/credentials.vault
    4. Provides CRUD operations for credentials
    5. Tracks access patterns for audit

    Machine binding: the encryption key is derived from the machine's UUID,
    so the vault can only be decrypted on the same machine. The remote backup
    (via GitBackupManager) stores the encrypted blob — it's useless without
    the machine key.
    """

    def __init__(self, vault_dir: Path | None = None, passphrase: str | None = None):
        self.vault_dir = vault_dir or VAULT_DIR
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.vault_file = self.vault_dir / "credentials.vault"
        self._entries: dict[str, VaultEntry] = {}
        self._fernet = self._init_encryption(passphrase)
        self._load()

    def _init_encryption(self, passphrase: str | None = None) -> Fernet:
        """Initialize Fernet encryption with machine-derived key."""
        salt = self._get_or_create_salt()
        machine_secret = passphrase or self._get_machine_id()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(machine_secret.encode()))
        return Fernet(key)

    def _get_or_create_salt(self) -> bytes:
        """Get or create the vault salt."""
        salt_file = self.vault_dir / ".salt"
        if salt_file.exists():
            return salt_file.read_bytes()
        salt = os.urandom(32)
        salt_file.write_bytes(salt)
        salt_file.chmod(0o600)
        return salt

    def _get_machine_id(self) -> str:
        """Get a machine-specific identifier for key derivation."""
        mid_file = self.vault_dir / ".machine_id"
        if mid_file.exists():
            return mid_file.read_text().strip()

        # Derive from system info
        node = uuid.getnode()
        system = platform.system()
        machine = platform.machine()
        raw = f"{node}:{system}:{machine}:{os.getlogin()}"
        mid = hashlib.sha256(raw.encode()).hexdigest()

        mid_file.write_text(mid)
        mid_file.chmod(0o600)
        return mid

    # ── CRUD ────────────────────────────────────────────────────────────────

    def store(self, key: str, value: str, service: str = "",
              description: str = "") -> VaultEntry:
        """Store or update a credential."""
        entry = VaultEntry(
            key=key,
            value=value,
            service=service,
            description=description,
        )
        self._entries[key] = entry
        self._save()
        logger.info("Vault: stored credential '%s' (service: %s)", key, service)
        return entry

    def get(self, key: str) -> str | None:
        """Retrieve a credential value. Returns None if not found."""
        entry = self._entries.get(key)
        if entry:
            entry.last_accessed = datetime.now(timezone.utc).isoformat()
            entry.access_count += 1
            self._save()
            return entry.value
        return None

    def get_entry(self, key: str) -> VaultEntry | None:
        """Retrieve full entry metadata."""
        return self._entries.get(key)

    def delete(self, key: str) -> bool:
        """Remove a credential from the vault."""
        if key in self._entries:
            del self._entries[key]
            self._save()
            logger.info("Vault: deleted credential '%s'", key)
            return True
        return False

    def list_keys(self) -> list[dict]:
        """List all stored keys (without values)."""
        return [
            {
                "key": e.key,
                "service": e.service,
                "description": e.description,
                "created_at": e.created_at,
                "access_count": e.access_count,
            }
            for e in self._entries.values()
        ]

    def has(self, key: str) -> bool:
        return key in self._entries

    # ── Batch operations ────────────────────────────────────────────────────

    def store_env_keys(self, env_keys: list[str]) -> int:
        """Import credentials from environment variables."""
        stored = 0
        for key in env_keys:
            value = os.environ.get(key)
            if value:
                self.store(key, value, service="environment", description=f"Imported from ${key}")
                stored += 1
        return stored

    def export_to_env(self, keys: list[str] | None = None) -> dict[str, str]:
        """Export credentials as environment variables (in-memory only, not written to shell)."""
        result = {}
        targets = keys or list(self._entries.keys())
        for key in targets:
            value = self.get(key)
            if value:
                result[key] = value
        return result

    # ── Persistence ─────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Encrypt and save the vault to disk."""
        data = {}
        for key, entry in self._entries.items():
            data[key] = {
                "value": entry.value,
                "service": entry.service,
                "description": entry.description,
                "created_at": entry.created_at,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
            }
        plaintext = json.dumps(data).encode()
        encrypted = self._fernet.encrypt(plaintext)
        self.vault_file.write_bytes(encrypted)
        self.vault_file.chmod(0o600)

    def _load(self) -> None:
        """Load and decrypt the vault from disk."""
        if not self.vault_file.exists():
            return
        try:
            encrypted = self.vault_file.read_bytes()
            plaintext = self._fernet.decrypt(encrypted)
            data = json.loads(plaintext.decode())
            for key, meta in data.items():
                self._entries[key] = VaultEntry(
                    key=key,
                    value=meta["value"],
                    service=meta.get("service", ""),
                    description=meta.get("description", ""),
                    created_at=meta.get("created_at", ""),
                    last_accessed=meta.get("last_accessed", ""),
                    access_count=meta.get("access_count", 0),
                )
            logger.info("Vault: loaded %d credentials", len(self._entries))
        except Exception as e:
            logger.error("Vault: failed to decrypt — %s", e)

    def get_encrypted_blob(self) -> bytes | None:
        """Get the raw encrypted vault blob for remote backup."""
        if self.vault_file.exists():
            return self.vault_file.read_bytes()
        return None

    @property
    def size(self) -> int:
        return len(self._entries)

    def health(self) -> dict:
        return {
            "credentials_stored": self.size,
            "vault_path": str(self.vault_file),
            "vault_exists": self.vault_file.exists(),
            "services": list(set(e.service for e in self._entries.values() if e.service)),
        }
