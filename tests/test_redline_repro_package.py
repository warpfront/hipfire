#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""CPU-only capture tests; synthetic objects do not verify ELF or loader acceptance."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARCH = "gfx1201"


class RedlineReproPackageCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="redline-repro-package-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        for directory in ("scripts", "registry"):
            (self.repo / directory).mkdir(parents=True)
        self.script = self.repo / "scripts" / "redline-repro-package.sh"
        shutil.copyfile(REPO / "scripts" / self.script.name, self.script)
        shutil.copyfile(
            REPO / "registry" / "redline-golden-v1.json",
            self.repo / "registry" / "redline-golden-v1.json",
        )
        self.out = self.root / "output"
        self.env = os.environ.copy()
        self.env.pop("HIPFIRE_KERNEL_CACHE", None)
        # Never invoke GPU utilities or host device compilers. Capture tolerates
        # unavailable tools; these stubs only isolate environment collection.
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for command in (
            "hipcc", "amdclang++", "amdclang", "clang++", "clang",
            "rocminfo", "rocm-smi", "amd-smi", "git",
        ):
            stub = bin_dir / command
            stub.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            stub.chmod(0o755)
        self.env["PATH"] = str(bin_dir) + os.pathsep + self.env.get("PATH", os.defpath)
        self.env["TMPDIR"] = str(self.root)

    def write_triple(self, directory: Path, module: str = "synthetic_module") -> dict[str, bytes]:
        directory.mkdir(parents=True, exist_ok=True)
        blob = b"synthetic packaging fixture, not an ELF code object\x00\xff\n"
        files = {
            module + ".hsaco": blob,
            module + ".hash": b"synthetic-source-and-flags-hash\n",
            module + ".index.json": (
                json.dumps({
                    "module": module,
                    "symbols": ["synthetic_kernel"],
                    "object_sha256": hashlib.sha256(blob).hexdigest(),
                    "synthetic_packaging_fixture": True,
                }, indent=2) + "\n"
            ).encode(),
        }
        for name, contents in files.items():
            (directory / name).write_bytes(contents)
        return files

    def capture(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.script), "capture", "--arch", ARCH, "--out", str(self.out)],
            cwd=self.repo,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def assert_archive_triple(self, expected: dict[str, bytes]) -> None:
        result = self.capture()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        archives = list(self.out.glob("repro-gfx1201-*.tar.gz"))
        self.assertEqual(len(archives), 1)
        with tarfile.open(archives[0], "r:gz") as archive:
            packaged = {
                member.name.removeprefix("./"): member
                for member in archive.getmembers()
                if member.isfile()
            }
            prefix = f"hsaco/{ARCH}/"
            self.assertEqual(
                {name for name in packaged if name.startswith(prefix)},
                {prefix + name for name in expected},
            )
            for name, contents in expected.items():
                with self.subTest(file=name):
                    stream = archive.extractfile(packaged[prefix + name])
                    self.assertIsNotNone(stream)
                    with stream:
                        self.assertEqual(stream.read(), contents)
            print(
                f"{self.id()} archive triple:\n"
                + "\n".join(sorted(prefix + name for name in expected)),
                flush=True,
            )

    def test_capture_preserves_installed_object_and_metadata(self) -> None:
        expected = self.write_triple(self.repo / "kernels" / "compiled" / ARCH)
        self.assert_archive_triple(expected)

    def test_explicit_cache_selects_alternative_indexed_directory(self) -> None:
        self.write_triple(self.repo / "kernels" / "compiled" / ARCH, "not_selected")
        cache = self.root / "alternative-cache"
        expected = self.write_triple(cache / ARCH, "selected")
        self.env["HIPFIRE_KERNEL_CACHE"] = str(cache)
        self.assert_archive_triple(expected)

    def test_missing_index_fails_without_archive(self) -> None:
        directory = self.repo / "kernels" / "compiled" / ARCH
        self.write_triple(directory)
        (directory / "synthetic_module.index.json").unlink()
        result = self.capture()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("missing indexed-object metadata", result.stderr)
        self.assertEqual(list(self.out.glob("*.tar.gz")), [])

    def test_missing_hash_fails_without_archive(self) -> None:
        directory = self.repo / "kernels" / "compiled" / ARCH
        self.write_triple(directory)
        (directory / "synthetic_module.hash").unlink()
        result = self.capture()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("missing hash sidecar", result.stderr)
        self.assertEqual(list(self.out.glob("*.tar.gz")), [])


if __name__ == "__main__":
    unittest.main()
