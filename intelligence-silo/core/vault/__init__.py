"""Secure Vault — encrypted local token/credential storage with remote backup."""

from .vault import SecureVault
from .backup import GitBackupManager

__all__ = ["SecureVault", "GitBackupManager"]
