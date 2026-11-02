#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate and drift-check per-crate map.md files.

Each crate may carry a map.md at its crate root. The mechanical half of the
map — source module list with each module's exported symbols, direct and
reverse dependencies — lives between explicit markers:

    <!-- crate-map:generated:begin -->
    ...
    <!-- crate-map:generated:end -->

and is regenerated from the tree by this script. Everything outside the
markers is hand-written judgement (purpose, status, gotchas) and is preserved
byte-for-byte on regeneration.

The block deliberately carries no volatile stats (line/item/test counts) and
no totals: it changes only when a file is added/removed, an exported symbol is
added/removed/renamed, or a dependency edge changes. Every list (exports are a
sub-list under their module) is sorted alphabetically with one entry per line,
so two branches that each add an entry merge without conflict unless the
entries are adjacent.

Usage:
    scripts/check-crate-maps.py <crate>...        generate/refresh those maps
    scripts/check-crate-maps.py --check [crate...]  fail on drift or broken links
        (no crate named = every map.md present under crates/)

--check validates two things per map.md:
  1. The generated block matches the tree (modules from actual src/**/*.rs
     files, exports, dependencies, reverse dependencies, link hrefs).
  2. Every local markdown link in the WHOLE file -- inside and outside the
     markers, even when the generated block is otherwise fresh -- resolves to
     an existing path. Inline links, images and reference definitions are
     checked; targets are relative to the crate directory (a leading `/` is
     relative to the repo root), `#fragment` / `?query` suffixes are stripped
     before the lookup, and fragment-only links, external schemes (https:,
     mailto:, ...) and `//host` links are ignored. Fenced code, inline code and
     HTML comments are not scanned. Anchors themselves are not verified.

Exit codes in --check mode (mirrors scripts/check-env-docs.py):
    0 - every checked map matches the tree and all local links resolve
    1 - drift or a broken local link found (details printed, one line per
        finding, with the map.md line number for link problems)
    2 - usage error or a named crate has no map.md
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
CRATES = ROOT / "crates"

BEGIN = "<!-- crate-map:generated:begin -->"
END = "<!-- crate-map:generated:end -->"

# Leading whitespace is REQUIRED to be optional here. Anchoring `^pub` at column
# zero silently reported 0 public items for `rdna-compute/src/gemm.rs`, a file with
# 296 of them -- every one is a method inside an `impl Gpu` block, so every one is
# indented. Any crate whose API is methods rather than free functions was
# under-counted the same way, and the drift check passed anyway because it
# regenerates the map with the same counter: a self-consistent wrong measurement.
#
# Caveat this cannot resolve by regex: a `pub fn` inside a private `mod` is not
# actually public API, so this over-counts in that case. Over-counting a few is
# strictly better than reporting 296 as 0.
PUB_ITEM = re.compile(
    r"^\s*pub\s+(?:unsafe\s+)?(?:async\s+)?(?:default\s+)?"
    r"(fn|struct|enum|trait|type|const|static|mod|use)\s+([A-Za-z_][A-Za-z0-9_]*)"
)

# Local-link validation. Code is masked first so examples such as `[x](y)` in
# prose or fences are not mistaken for real links.
FENCED_CODE = re.compile(
    r"^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*\1[`~]*[ \t]*$|\Z)",
    re.MULTILINE | re.DOTALL,
)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)
# `[text](target "title")`, also images; target may be `<...>`-wrapped.
INLINE_LINK = re.compile(
    r"\[(?:[^\]\\]|\\.)*\]\(\s*(<[^>\n]*>|[^)\s]*)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
REF_DEFINITION = re.compile(r"^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*(<[^>\n]*>|\S+)", re.MULTILINE)
URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def public_items(path: Path) -> list[str]:
    """Unique `pub` item names, alphabetical (case-insensitive, ties by case)."""
    names = {
        m.group(2)
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if (m := PUB_ITEM.match(line))
    }
    return sorted(names, key=lambda n: (n.casefold(), n))


def module_inventory(crate: Path) -> list[tuple[str, list[str]]]:
    """(relative path, sorted exported names) for every src/**/*.rs."""
    src = crate / "src"
    if not src.is_dir():
        return []
    return [
        (rs.relative_to(crate).as_posix(), public_items(rs))
        for rs in sorted(src.rglob("*.rs"), key=lambda p: p.relative_to(crate).as_posix())
        if rs.is_file()
    ]


def parse_manifest(crate: Path) -> dict[str, list[str]]:
    """Dependencies by kind: path / external / dev / build (crate-local path
    deps are recognised by a `path = "../` value)."""
    deps: dict[str, list[str]] = {"path": [], "external": [], "dev": [], "build": []}
    manifest = crate / "Cargo.toml"
    if not manifest.is_file():
        return deps
    section = ""
    for raw in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        m = re.match(r"\[([^\]]+)\]", line)
        if m:
            section = m.group(1).strip().strip('"')
            continue
        m = re.match(r"([A-Za-z0-9_-]+)(?:\.workspace)?\s*=\s*(.+)", line)
        if not m:
            continue
        name, value = m.group(1), m.group(2)
        base = section
        if base.startswith("target."):
            base = base.rsplit(".", 1)[-1]
        is_path = 'path = "../' in value or 'path="../' in value
        if base == "dependencies":
            deps["path" if is_path else "external"].append(name)
        elif base == "dev-dependencies":
            deps["dev"].append(name)
        elif base == "build-dependencies":
            deps["build"].append(name)
    for key in deps:
        deps[key] = sorted(set(deps[key]))
    return deps


def workspace_crates() -> list[Path]:
    return sorted(p.parent for p in CRATES.glob("*/Cargo.toml"))


def reverse_dependencies(crate: Path) -> list[str]:
    target = crate.name
    rev = []
    for other in workspace_crates():
        if other == crate:
            continue
        manifest = (other / "Cargo.toml").read_text(encoding="utf-8", errors="ignore")
        if f'path = "../{target}"' in manifest or f'path="../{target}"' in manifest:
            rev.append(other.name)
    return sorted(rev)


def bullet_list(names: list[str]) -> list[str]:
    return [f"- `{n}`" for n in names] or ["—"]


def generated_block(crate: Path) -> str:
    deps = parse_manifest(crate)
    lines = [
        BEGIN,
        "",
        "_Generated by `scripts/check-crate-maps.py` from the tree — do not edit "
        "inside the markers._",
        "",
        "### Modules and exported symbols",
        "",
    ]
    modules = module_inventory(crate)
    for rel, names in modules:
        lines.append(f"- [`{rel}`]({rel})")
        lines += [f"  - `{n}`" for n in names]
    if not modules:
        lines.append("—")
    lines += ["", "### Dependencies (from `Cargo.toml`)", ""]
    for kind in ("path", "external", "dev", "build"):
        lines += [f"#### {kind}", "", *bullet_list(deps[kind]), ""]
    lines += [
        "### Reverse dependencies",
        "",
        "Workspace crates with a path dependency on this crate:",
        "",
        *bullet_list(reverse_dependencies(crate)),
        "",
        END,
    ]
    return "\n".join(lines)


def parse_block_sections(block: str) -> dict[str, set[str]]:
    """First backticked name of every top-level `- ` line, keyed by its `### ` section."""
    out: dict[str, set[str]] = {}
    section = ""
    for line in block.splitlines():
        if line.startswith("### "):
            section = line[4:]
            out[section] = set()
        elif section and line.startswith("- "):
            m = re.search(r"`([^`]+)`", line)
            if m:
                out[section].add(m.group(1))
    return out


def parse_block_modules(block: str) -> dict[str, set[str]]:
    """Module path -> its exported-symbol sub-list."""
    out: dict[str, set[str]] = {}
    rel = None
    for line in block.splitlines():
        if m := re.match(r"- \[`(src/[^`]+)`\]", line):
            rel = m.group(1)
            out[rel] = set()
        elif rel and (m := re.match(r"  - `([^`]+)`", line)):
            out[rel].add(m.group(1))
        elif not line.startswith("  "):
            rel = None
    return out


def parse_block_hrefs(block: str) -> dict[str, str]:
    """Module path -> the href its generated bullet links to."""
    return dict(re.findall(r"^- \[`(src/[^`]+)`\]\(([^)]*)\)$", block, re.MULTILINE))


def mask_non_link_text(text: str) -> str:
    """Blank out fenced code, HTML comments and inline code spans, keeping
    every offset and newline so line numbers of the remaining text survive."""

    def blank(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    for pattern in (FENCED_CODE, HTML_COMMENT, INLINE_CODE):
        text = pattern.sub(blank, text)
    return text


def local_link_path(target: str) -> str | None:
    """Filesystem part of a link target, or None when it is not a local file
    link (external scheme, protocol-relative, or fragment/query-only)."""
    target = target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if not target or target.startswith("//") or URL_SCHEME.match(target):
        return None
    path = re.split(r"[#?]", target, maxsplit=1)[0]
    return unquote(path) or None


def check_links(crate: Path, text: str) -> list[str]:
    """Every local markdown link in the whole map.md (inside and outside the
    generated markers) must resolve to an existing path. Targets are relative
    to the crate directory; a leading `/` is relative to the repo root."""
    problems: list[str] = []
    masked = mask_non_link_text(text)
    for pattern in (INLINE_LINK, REF_DEFINITION):
        for m in pattern.finditer(masked):
            target = m.group(1)
            path = local_link_path(target)
            if path is None:
                continue
            resolved = ROOT / path.lstrip("/") if path.startswith("/") else crate / path
            if resolved.exists():
                continue
            line = text.count("\n", 0, m.start(1)) + 1
            shown = os.path.normpath(resolved).replace(os.sep, "/")
            problems.append(
                f"{crate.name}: map.md:{line}: link target does not exist: "
                f"{target.strip()} (resolved to {shown})"
            )
    return problems


def check_crate(crate: Path) -> list[str]:
    map_path = crate / "map.md"
    if not map_path.is_file():
        return [f"{crate.name}: no map.md"]
    text = map_path.read_text(encoding="utf-8", errors="ignore")
    if BEGIN not in text or END not in text:
        return [f"{crate.name}: map.md is missing the generated-block markers"]
    link_problems = check_links(crate, text)
    old_block = text[text.index(BEGIN):text.index(END) + len(END)]
    fresh = generated_block(crate)
    if old_block == fresh:
        return link_problems

    problems: list[str] = []
    old = parse_block_sections(old_block)
    old_modules = parse_block_modules(old_block)
    new_modules = {rel: set(names) for rel, names in module_inventory(crate)}
    for rel in sorted(new_modules.keys() - old_modules.keys()):
        problems.append(f"{crate.name}: module exists with no map entry: {rel}")
    for rel in sorted(old_modules.keys() - new_modules.keys()):
        problems.append(f"{crate.name}: map lists a module that no longer exists: {rel}")
    for rel in sorted(new_modules.keys() & old_modules.keys()):
        for name in sorted(new_modules[rel] - old_modules[rel]):
            problems.append(f"{crate.name}: export missing from map: {rel}::{name}")
        for name in sorted(old_modules[rel] - new_modules[rel]):
            problems.append(f"{crate.name}: map lists a removed export: {rel}::{name}")

    old_deps = old.get("Dependencies (from `Cargo.toml`)", set())
    new_deps = set().union(*parse_manifest(crate).values())
    for name in sorted(old_deps - new_deps):
        problems.append(f"{crate.name}: stale declared dependency in map: {name}")
    for name in sorted(new_deps - old_deps):
        problems.append(f"{crate.name}: dependency missing from map: {name}")

    old_rev, new_rev = old.get("Reverse dependencies", set()), set(reverse_dependencies(crate))
    for name in sorted(old_rev - new_rev):
        problems.append(f"{crate.name}: stale reverse dependency in map: {name}")
    for name in sorted(new_rev - old_rev):
        problems.append(f"{crate.name}: reverse dependency missing from map: {name}")

    for rel, href in sorted(parse_block_hrefs(old_block).items()):
        if href != rel and rel in new_modules:
            problems.append(
                f"{crate.name}: map link for {rel} points at {href}, expected {rel}"
            )

    if not problems:
        problems.append(
            f"{crate.name}: generated block is stale — "
            f"run scripts/check-crate-maps.py {crate.name}"
        )
    return problems + link_problems


def generate_crate(crate: Path) -> bool:
    map_path = crate / "map.md"
    block = generated_block(crate)
    if map_path.is_file():
        text = map_path.read_text(encoding="utf-8", errors="ignore")
        if BEGIN not in text or END not in text:
            print(f"{crate.name}: map.md exists but has no generated-block markers", file=sys.stderr)
            return False
        new_text = text[:text.index(BEGIN)] + block + text[text.index(END) + len(END):]
        if new_text != text:
            map_path.write_text(new_text, encoding="utf-8")
            print(f"{crate.name}: refreshed generated block")
        else:
            print(f"{crate.name}: already up to date")
        return True

    skeleton = f"""# {crate.name} — map

> **Status:** `production` / `research` / `legacy` — pick exactly one
> (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** see the layering table in
> [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) — do not restate it here.

## Purpose

<!-- hand-written: what this crate owns and why it exists. Prefer pointing at
     the `//!` docs in `src/lib.rs` over copying prose. -->

## Gotchas

<!-- hand-written: what not to do here; traps a reader will hit. -->

## Crate map

{block}
"""
    map_path.write_text(skeleton, encoding="utf-8")
    print(f"{crate.name}: wrote new map.md skeleton with generated block")
    return True


def main(argv: list[str]) -> int:
    check = "--check" in argv
    names = [a for a in argv if not a.startswith("--")]

    if check:
        if names:
            crates = []
            for name in names:
                crate = CRATES / name
                if not (crate / "map.md").is_file():
                    print(f"{name}: no map.md", file=sys.stderr)
                    return 2
                crates.append(crate)
        else:
            crates = sorted(p.parent for p in CRATES.glob("*/map.md"))
            if not crates:
                print("crate-maps: no map.md files present; nothing to check")
                return 0
        problems: list[str] = []
        for crate in crates:
            problems.extend(check_crate(crate))
        without = len(list(CRATES.glob("*/Cargo.toml"))) - len(crates)
        if problems:
            print("crate maps have drifted from the tree or contain broken links:")
            for p in problems:
                print(f"  {p}")
            return 1
        print(
            f"crate-maps: {len(crates)} map(s) match the tree "
            f"({without} crates without a map, not enforced)"
        )
        return 0

    if not names:
        print(__doc__, file=sys.stderr)
        return 2
    ok = True
    for name in names:
        crate = CRATES / name
        if not crate.is_dir():
            print(f"{name}: no such crate under crates/", file=sys.stderr)
            ok = False
            continue
        ok = generate_crate(crate) and ok
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
