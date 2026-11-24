#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Released CHANGELOG sections are immutable.

Every `## vX.Y.Z...` section whose tag `vX.Y.Z` exists must be byte-identical
to the same section in the newest release tag's CHANGELOG.md, which for that
tag's own section is the text it shipped with. (Comparing older sections to the
newest tag rather than their own tag grandfathers edits made before it.) New
notes go in a new top section (e.g. `## Unreleased`), never in a shipped one.

The region above the newest `## vX.Y.Z...` heading must hold at most one
`## Unreleased` heading and no repeated bullet line (a `merge=union` merge of
stacked branches pastes both sides and leaves exactly those).

New entries go in `changelog.d/<slug>.md` fragments (see changelog.d/README.md):
each needs a `### ` first line, at least one bullet, and no bullet already in
the Unreleased region or in another fragment. `--refuse-direct BASE` also fails
a CHANGELOG.md that adds lines above the newest release relative to BASE
(off by default while both forms are accepted).

Usage: check-changelog.py [--repo DIR] [--changelog PATH] [--refuse-direct BASE]
Exit 0 = clean, 1 = a released section changed, the unreleased region is
duplicated or a fragment is bad, 2 = usage/git error. Needs tags (CI checks out with fetch-depth: 0).
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
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")


def first_diff_line(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a.splitlines(), b.splitlines()), 1):
        if x != y:
            return i
    return min(len(a.splitlines()), len(b.splitlines())) + 1


def unreleased_errors(text: str) -> list[str]:
    """Duplicate `## Unreleased` headings or bullet lines above the newest release."""
    errors = []
    region = []
    for line in text.splitlines():
        if HEADING.match(line):
            break
        region.append(line)
    headings = [i for i, line in enumerate(region, 1) if line.rstrip() == "## Unreleased"]
    if len(headings) > 1:
        errors.append(
            f"{len(headings)} `## Unreleased` headings above the newest release "
            f"(lines {', '.join(map(str, headings))}); keep one top section"
        )
    seen: dict[str, int] = {}
    for i, line in enumerate(region, 1):
        if not line.lstrip().startswith("- "):
            continue
        if line in seen:
            errors.append(
                f"line {i}: duplicated entry (first at line {seen[line]}): {line.strip()[:80]}"
            )
        else:
            seen[line] = i
    return errors


def region_lines(text: str) -> list[str]:
    region = []
    for line in text.splitlines():
        if HEADING.match(line):
            break
        region.append(line)
    return region


def fragment_errors(repo: Path, text: str) -> list[str]:
    errors = []
    seen = {l: "CHANGELOG.md" for l in region_lines(text) if l.lstrip().startswith("- ")}
    for path in sorted((repo / "changelog.d").glob("*.md")):
        if path.name == "README.md":
            continue
        head, _, body = path.read_text(encoding="utf-8").partition("\n")
        name = f"changelog.d/{path.name}"
        if not head.startswith("### ") or not head[4:].strip():
            errors.append(f"{name}: first line must be a `### ` section heading")
        bullets = [l for l in body.splitlines() if l.startswith("- ")]
        if not bullets:
            errors.append(f"{name}: no `- ` entry after the heading")
        for l in bullets:
            if l in seen:
                errors.append(f"{name}: entry already present in {seen[l]}: {l[:80]}")
            else:
                seen[l] = name
    return errors


def direct_edit_errors(repo: Path, changelog: Path, base: str) -> list[str]:
    old = git(repo, "show", f"{base}:CHANGELOG.md")
    if old.returncode != 0:
        raise RuntimeError(old.stderr.strip())
    before = set(region_lines(old.stdout))
    added = [l for l in region_lines(changelog.read_text(encoding="utf-8")) if l.strip() and l not in before]
    if not added:
        return []
    return [
        f"{len(added)} line(s) added to the Unreleased region of CHANGELOG.md "
        f"(first: {added[0][:60]}); add a changelog.d/ fragment instead"
    ]


def check(repo: Path, changelog: Path, refuse_base: str | None = None) -> list[str]:
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
    errors = unreleased_errors(changelog.read_text()) + fragment_errors(
        repo, changelog.read_text(encoding="utf-8")
    )
    if refuse_base:
        errors += direct_edit_errors(repo, changelog, refuse_base)
    if reference is None:
        return errors
    ref_tag, released = reference
    current = sections(changelog.read_text(encoding="utf-8"))
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
    ap.add_argument("--refuse-direct", metavar="BASE", default=None)
    args = ap.parse_args()
    changelog = args.changelog or args.repo / "CHANGELOG.md"
    try:
        errors = check(args.repo, changelog, args.refuse_direct)
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
