# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""pytest helpers for hw-gate select tests: path constants only."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SELECT_PY = Path(__file__).resolve().parents[1] / "bucket_select.py"


def run_select(stdin_text: str, *extra_args: str) -> subprocess.CompletedProcess:
    """Run bucket_select.py with stdin_text piped to stdin, return CompletedProcess."""
    cmd = [sys.executable, str(SELECT_PY), *extra_args]
    return subprocess.run(
        cmd,
        input=stdin_text.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
