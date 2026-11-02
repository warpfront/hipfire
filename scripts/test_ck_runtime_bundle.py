#!/usr/bin/env python3
"""No-GPU contract tests for the generic CK runtime bundle tools."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGER = ROOT / "scripts" / "package-ck-runtime.sh"
INSTALLER = ROOT / "scripts" / "install-ck-runtime.sh"


def run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script), *args], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


class CkRuntimeBundleTest(unittest.TestCase):
    def test_help_requires_no_gpu_or_sdk(self) -> None:
        for script in (PACKAGER, INSTALLER):
            with self.subTest(script=script.name):
                result = run(script, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Usage:", result.stdout)

    def test_installer_requires_exactly_one_source(self) -> None:
        result = run(INSTALLER)
        self.assertEqual(result.returncode, 2)
        self.assertIn("exactly one", result.stderr)

    def test_remote_install_requires_checksum(self) -> None:
        result = run(INSTALLER, "--url", "https://invalid.example/runtime.tar.gz")
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --sha256", result.stderr)

    def test_packager_rejects_unknown_arch_before_artifact_access(self) -> None:
        result = run(PACKAGER, "--gpu-arch", "gfx9999", "--allow-dirty")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported GPU arch", result.stderr)

    def test_packager_rejects_path_like_version(self) -> None:
        result = run(PACKAGER, "--version", "../../escape", "--allow-dirty")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe bundle version", result.stderr)


if __name__ == "__main__":
    unittest.main()
