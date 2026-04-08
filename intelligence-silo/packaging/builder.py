"""Node Builder — packages the intelligence silo as a distributable executable."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NodeManifest:
    """Manifest describing a packaged node — shipped alongside the executable."""
    node_name: str = "intel-node"
    version: str = "0.1.0"
    created_at: float = field(default_factory=time.time)
    includes_models: bool = True
    model_names: list[str] = field(default_factory=list)
    model_checksums: dict[str, str] = field(default_factory=dict)
    config_checksum: str = ""
    signing_signature: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "node_name": self.node_name,
            "version": self.version,
            "created_at": self.created_at,
            "includes_models": self.includes_models,
            "model_names": self.model_names,
            "model_checksums": self.model_checksums,
            "config_checksum": self.config_checksum,
            "signing_signature": self.signing_signature,
        }, indent=2)

    @classmethod
    def from_json(cls, data: str) -> NodeManifest:
        d = json.loads(data)
        return cls(**d)


class NodeBuilder:
    """Packages the intelligence silo into a distributable executable.

    The builder:
    1. Collects model weights (safetensors)
    2. Bundles configuration
    3. Creates a PyInstaller spec for single-file executable
    4. Signs the package with HMAC for trust verification
    5. Outputs: executable + manifest + model weights bundle

    The resulting package can be deployed to any node in the mesh.
    """

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.build_dir = self.project_root / "dist" / "build"
        self.output_dir = self.project_root / "dist" / "release"

    def build(self, name: str = "intel-node", include_models: bool = True,
              signing_key: str | None = None, model_weights_dir: Path | None = None) -> Path:
        """Build a distributable node package.

        Args:
            name: executable name
            include_models: whether to bundle model weights
            signing_key: HMAC key for trust signing
            model_weights_dir: path to trained model weights

        Returns:
            Path to the output directory containing the package.
        """
        logger.info("Building node package: %s", name)

        # Clean and create dirs
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        manifest = NodeManifest(node_name=name)

        # 1. Copy config
        config_src = self.project_root / "config" / "silo.yaml"
        config_dst = self.output_dir / "config" / "silo.yaml"
        config_dst.parent.mkdir(parents=True, exist_ok=True)
        if config_src.exists():
            shutil.copy2(config_src, config_dst)
            manifest.config_checksum = self._checksum(config_src)

        # 2. Copy model weights
        if include_models and model_weights_dir and model_weights_dir.exists():
            weights_dst = self.output_dir / "weights"
            weights_dst.mkdir(parents=True, exist_ok=True)
            for weight_file in model_weights_dir.glob("*.safetensors"):
                shutil.copy2(weight_file, weights_dst / weight_file.name)
                model_name = weight_file.stem
                manifest.model_names.append(model_name)
                manifest.model_checksums[model_name] = self._checksum(weight_file)
            manifest.includes_models = True
            logger.info("Bundled %d model weight files", len(manifest.model_names))

        # 3. Generate PyInstaller spec
        spec_content = self._generate_spec(name)
        spec_path = self.build_dir / f"{name}.spec"
        spec_path.write_text(spec_content)

        # 4. Run PyInstaller
        exe_path = self._run_pyinstaller(spec_path, name)

        # 5. Sign manifest
        if signing_key:
            manifest_json = manifest.to_json()
            manifest.signing_signature = hmac.new(
                signing_key.encode(), manifest_json.encode(), hashlib.sha256
            ).hexdigest()

        # 6. Write manifest
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(manifest.to_json())

        logger.info("Node package built: %s", self.output_dir)
        return self.output_dir

    def verify_package(self, package_dir: Path, signing_key: str) -> bool:
        """Verify a node package's integrity using its HMAC signature."""
        manifest_path = package_dir / "manifest.json"
        if not manifest_path.exists():
            return False

        manifest = NodeManifest.from_json(manifest_path.read_text())
        stored_sig = manifest.signing_signature

        # Recompute without signature
        manifest.signing_signature = ""
        expected_sig = hmac.new(
            signing_key.encode(), manifest.to_json().encode(), hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(stored_sig, expected_sig)

    def _generate_spec(self, name: str) -> str:
        """Generate a PyInstaller spec file."""
        entry_point = self.project_root / "core" / "__init__.py"
        return f"""# -*- mode: python ; coding: utf-8 -*-
# Intelligence Silo Node — PyInstaller Spec
# Auto-generated by NodeBuilder

a = Analysis(
    ['{entry_point}'],
    pathex=['{self.project_root}'],
    binaries=[],
    datas=[
        ('{self.project_root / "config"}', 'config'),
    ],
    hiddenimports=[
        'torch', 'numpy', 'yaml', 'faiss',
        'safetensors', 'httpx', 'pydantic',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter', 'PIL'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='{name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=True,
    target_arch=None,
    codesign_identity=None,
)
"""

    def _run_pyinstaller(self, spec_path: Path, name: str) -> Path | None:
        """Run PyInstaller to build the executable."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "PyInstaller",
                 "--distpath", str(self.output_dir),
                 "--workpath", str(self.build_dir),
                 "--clean", "--noconfirm",
                 str(spec_path)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                exe_path = self.output_dir / name
                logger.info("PyInstaller build successful: %s", exe_path)
                return exe_path
            else:
                logger.warning("PyInstaller failed: %s", result.stderr[:500])
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("PyInstaller not available or timed out: %s", e)
            return None

    @staticmethod
    def _checksum(path: Path) -> str:
        """SHA256 checksum of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
