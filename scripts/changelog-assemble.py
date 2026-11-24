#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Fold changelog.d/*.md fragments into the `## Unreleased` section.

A fragment's first line is its `### ` heading; the rest is bullet text. Files
are taken in name order. Bullets go to the end of an existing identical `###`
block in Unreleased, else under a new `###` block appended to the section.

Usage: changelog-assemble.py [--repo DIR] [--check]
  --check   print the resulting CHANGELOG.md to stdout; change nothing.
Without --check the CHANGELOG is rewritten and the fragments are deleted.
With no fragments the output is the input, byte for byte.
"""
import argparse
import sys
from pathlib import Path


def fragments(directory: Path) -> list[tuple[Path, str, str]]:
    out = []
    for path in sorted(directory.glob("*.md")):
        if path.name == "README.md":
            continue
        head, _, body = path.read_text(encoding="utf-8").partition("\n")
        if not head.startswith("### ") or not head[4:].strip():
            raise ValueError(f"{path.name}: first line must be a `### ` heading")
        body = body.strip("\n")
        if not body.strip():
            raise ValueError(f"{path.name}: no entry text after the heading")
        out.append((path, head.rstrip(), body))
    return out


def assemble(text: str, frags: list[tuple[Path, str, str]]) -> str:
    if not frags:
        return text
    lines = text.splitlines(keepends=True)
    try:
        start = next(i for i, l in enumerate(lines) if l.rstrip() == "## Unreleased")
    except StopIteration:
        raise ValueError("CHANGELOG.md has no `## Unreleased` section")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    section = [l.rstrip("\n") for l in lines[start:end]]
    while section and not section[-1].strip():
        section.pop()
    grouped: dict[str, list[str]] = {}
    for _, head, body in frags:
        grouped.setdefault(head, []).append(body)
    for head, bodies in grouped.items():
        block = []
        for b in bodies:
            block += b.split("\n")
        idx = [i for i, l in enumerate(section) if l.rstrip() == head]
        if idx:
            at = next((i for i in range(idx[-1] + 1, len(section)) if section[i].startswith("#")), len(section))
            while at > idx[-1] + 1 and not section[at - 1].strip():
                at -= 1
            section[at:at] = block
        else:
            section += ["", head, ""] + block
    new = [l + "\n" for l in section] + ["\n"] if end < len(lines) else [l + "\n" for l in section]
    return "".join(lines[:start] + new + lines[end:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    changelog = args.repo / "CHANGELOG.md"
    try:
        frags = fragments(args.repo / "changelog.d")
        result = assemble(changelog.read_text(encoding="utf-8"), frags)
    except (OSError, ValueError) as e:
        print(f"changelog-assemble: {e}", file=sys.stderr)
        return 2
    if args.check:
        sys.stdout.write(result)
        return 0
    changelog.write_text(result, encoding="utf-8")
    for path, _, _ in frags:
        path.unlink()
    print(f"changelog-assemble: folded {len(frags)} fragment(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
