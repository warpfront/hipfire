#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""scripts/check-conflict-markers.py on throwaway files."""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "check-conflict-markers.py"


def _run(*paths: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, paths)], capture_output=True, text=True)


def test_diff3_block_is_flagged(tmp_path: Path) -> None:
    f = tmp_path / "CHANGELOG.md"
    f.write_text("a\n<<<<<<< HEAD\nx\n||||||| parent of 1234\ny\n=======\nz\n>>>>>>> 5678 (msg)\n")
    r = _run(f)
    assert r.returncode == 1
    assert [l.split(":")[1] for l in r.stdout.splitlines()[:-1]] == ["2", "4", "6", "8"]


def test_stray_base_marker_alone_is_flagged(tmp_path: Path) -> None:
    f = tmp_path / "CHANGELOG.md"
    f.write_text("## v1\n- a\n||||||| parent of 9a343bca2e (x)\n- b\n")
    assert _run(f).returncode == 1


def test_setext_underline_and_text_are_clean(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("Title\n=======\n\nnot a <<<<<<< marker mid-line\n")
    assert _run(f).returncode == 0


def test_binary_is_skipped(tmp_path: Path) -> None:
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\0\1<<<<<<< HEAD\n")
    assert _run(f).returncode == 0
