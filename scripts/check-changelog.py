#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Released CHANGELOG sections are immutable.

Every `## vX.Y.Z...` section whose tag `vX.Y.Z` exists must be byte-identical
to the same section in the newest release tag's CHANGELOG.md, which for that
tag's own section is the text it shipped with. (Comparing older sections to the
newest tag rather than their own tag grandfathers edits made before it.) New
notes go in a new top section (e.g. `## Unreleased`), never in a shipped one.

Usage: check-changelog.py [--repo DIR] [--changelog PATH]
Exit 0 = clean, 1 = a released section changed, 2 = usage/git error.
Needs tags (CI checks out with fetch-depth: 0).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HEADING = re.compile(r"^## (v\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?)(?=[\s(]|$)")


def sections(text: str) -> dict[str, str]:
    """Version -> section bytes, heading line through the line before the next `## `."""
    out: dict[str, str] = {}
    current = None
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            m = HEADING.match(line)
            current = m.group(1) if m else None
            if current is not None:
                out.setdefault(current, "")
        if current is not None:
            out[current] += line
    return out


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def first_diff_line(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a.splitlines(), b.splitlines()), 1):
        if x != y:
            return i
    return min(len(a.splitlines()), len(b.splitlines())) + 1


def check(repo: Path, changelog: Path) -> list[str]:
    tags = git(repo, "tag", "--list", "v*", "--sort=-v:refname")
    if tags.returncode != 0:
        raise RuntimeError(tags.stderr.strip())
    tagged = tags.stdout.split()
    reference = None
    for tag in tagged:
        shown = git(repo, "show", f"refs/tags/{tag}:CHANGELOG.md")
        if shown.returncode == 0:
            reference = tag, sections(shown.stdout)
            break
    if reference is None:
        return []
    ref_tag, released = reference
    current = sections(changelog.read_text())
    errors = []
    for version in sorted(set(tagged) & set(released)):
        body = current.get(version)
        if body is None:
            errors.append(f"{version}: released section was removed (present at {ref_tag})")
        elif body != released[version]:
            line = first_diff_line(body, released[version])
            errors.append(
                f"{version}: released section differs from {ref_tag}:CHANGELOG.md "
                f"(first difference at section line {line}); "
                f"put new notes in a new top section"
            )
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo", type=Path, default=root)
    ap.add_argument("--changelog", type=Path, default=None)
    args = ap.parse_args()
    changelog = args.changelog or args.repo / "CHANGELOG.md"
    try:
        errors = check(args.repo, changelog)
    except (OSError, RuntimeError) as e:
        print(f"check-changelog: {e}", file=sys.stderr)
        return 2
    for e in errors:
        print(f"check-changelog: {e}", file=sys.stderr)
    if errors:
        return 1
    print("check-changelog: released sections match their tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
