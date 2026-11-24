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


def test_second_unreleased_heading_fails(repo):
    top = "# Changelog\n\n## Unreleased\n- c\n\n## Unreleased\n- d\n\n"
    assert _run(repo, RELEASED.replace("# Changelog\n\n", top)) == 1


def test_duplicated_unreleased_entry_fails(repo):
    top = "# Changelog\n\n## Unreleased\n\n### Fixes\n- c\n\n### Docs\n- c\n\n"
    assert _run(repo, RELEASED.replace("# Changelog\n\n", top)) == 1


def test_one_unreleased_section_with_distinct_entries_passes(repo):
    top = "# Changelog\n\n## Unreleased\n- c\n\n### Fixes\n- d\n  - c\n\n"
    assert _run(repo, RELEASED.replace("# Changelog\n\n", top)) == 0


def _frag(repo: Path, name: str, text: str) -> int:
    d = repo / "changelog.d"
    d.mkdir(exist_ok=True)
    (d / name).write_text(text)
    return _run(repo, RELEASED)


def test_valid_fragment_passes(repo):
    assert _frag(repo, "x.md", "### Fixes\n\n- new thing\n") == 0


def test_fragment_without_heading_fails(repo):
    assert _frag(repo, "x.md", "- new thing\n") == 1


def test_fragment_without_bullet_fails(repo):
    assert _frag(repo, "x.md", "### Fixes\n\nprose\n") == 1


def test_fragment_repeating_existing_bullet_fails(repo):
    top = "# Changelog\n\n## Unreleased\n\n### Fixes\n- c\n\n"
    (repo / "changelog.d").mkdir()
    (repo / "changelog.d" / "x.md").write_text("### Docs\n\n- c\n")
    assert _run(repo, RELEASED.replace("# Changelog\n\n", top)) == 1


def test_fragments_repeating_each_other_fail(repo):
    _frag(repo, "a.md", "### Fixes\n\n- same\n")
    assert _frag(repo, "b.md", "### Docs\n\n- same\n") == 1


def test_readme_is_not_a_fragment(repo):
    assert _frag(repo, "README.md", "no heading here\n") == 0


def test_refuse_direct_flags_added_unreleased_lines(repo):
    _git(repo, "branch", "base")
    top = "# Changelog\n\n## Unreleased\n- c\n\n"
    (repo / "CHANGELOG.md").write_text(RELEASED.replace("# Changelog\n\n", top))
    run = lambda *a: subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo), *a], capture_output=True
    ).returncode
    assert run() == 0
    assert run("--refuse-direct", "base") == 1
    (repo / "CHANGELOG.md").write_text(RELEASED)
    assert run("--refuse-direct", "base") == 0


ASSEMBLE = Path(__file__).resolve().parent / "changelog-assemble.py"


def _assemble(repo: Path, *a: str):
    return subprocess.run(
        [sys.executable, str(ASSEMBLE), "--repo", str(repo), *a], capture_output=True, text=True
    )


def test_assemble_check_is_stable_and_groups_by_heading(repo):
    top = "# Changelog\n\n## Unreleased\n\n### Fixes\n- old\n\n"
    _frag(repo, "a.md", "### Fixes\n\n- one\n  more\n")
    _frag(repo, "b.md", "### Docs\n\n- two\n")
    (repo / "CHANGELOG.md").write_text(RELEASED.replace("# Changelog\n\n", top))
    first = _assemble(repo, "--check").stdout
    assert first == _assemble(repo, "--check").stdout
    assert first.index("- old") < first.index("- one") < first.index("### Docs") < first.index("- two")
    assert first.index("- two") < first.index("## v1.1.0")
    assert _assemble(repo).returncode == 0
    assert (repo / "CHANGELOG.md").read_text() == first
    assert not (repo / "changelog.d" / "a.md").exists()
    assert _assemble(repo, "--check").stdout == first
    assert _run(repo, first) == 0
