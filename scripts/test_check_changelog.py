#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""scripts/check-changelog.py against a throwaway git repo."""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "check-changelog.py"

RELEASED = "# Changelog\n\n## v1.1.0 — new\n- b\n\n## v1.0.0 — old\n- a\n"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## v1.0.0 — old\n- a\n")
    _git(tmp_path, "add", "CHANGELOG.md")
    _git(tmp_path, "commit", "-qm", "1.0.0")
    _git(tmp_path, "tag", "v1.0.0")
    (tmp_path / "CHANGELOG.md").write_text(RELEASED)
    _git(tmp_path, "commit", "-qam", "1.1.0")
    _git(tmp_path, "tag", "v1.1.0")
    return tmp_path


def _run(repo: Path, text: str) -> int:
    (repo / "CHANGELOG.md").write_text(text)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo)], capture_output=True
    ).returncode


def test_unchanged_released_sections_pass(repo):
    assert _run(repo, RELEASED) == 0


def test_new_top_section_passes(repo):
    text = RELEASED.replace("## v1.1.0", "## Unreleased\n- c\n\n## v1.2.0 — next\n- d\n\n## v1.1.0")
    assert _run(repo, text) == 0


def test_edit_inside_latest_released_section_fails(repo):
    assert _run(repo, RELEASED.replace("- b\n", "- b amended\n")) == 1


def test_appending_to_older_released_section_fails(repo):
    assert _run(repo, RELEASED + "- late note\n") == 1


def test_removing_a_released_section_fails(repo):
    assert _run(repo, "# Changelog\n\n## v1.1.0 — new\n- b\n\n") == 1


def test_untagged_version_section_is_editable(repo):
    text = RELEASED.replace("## v1.1.0", "## v1.2.0 — next\n- d\n\n## v1.1.0")
    _run(repo, text)
    assert _run(repo, text.replace("- d\n", "- d edited\n")) == 0
