#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Every Scratch growth is preceded by Gpu::invalidate_for_scratch_growth.

`grow_scratch_buffer` (crates/rdna-compute/src/scratch.rs) syncs and frees the
old slot, assuming no captured graph or retained replay still embeds its
pointer. The contract: a caller outside scratch.rs that invokes a `Scratch`
method which can grow a slot must call `invalidate_for_scratch_growth` earlier
in the same fn (normally behind `scratch_will_grow`).

This finds every `ScratchState` method that reaches `grow_scratch_buffer` /
`grow_scratch_slot` (transitively through `self.<method>(`), then every
`.scratch.<method>(` call under crates/ outside scratch.rs, and fails if the
enclosing fn has no earlier `invalidate_for_scratch_growth`.
Structural ratchet like the other check-* scripts. Exit 0 clean, 1 violations.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "crates" / "rdna-compute" / "src" / "scratch.rs"
PRIMITIVES = {"grow_scratch_buffer", "grow_scratch_slot"}
FN = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+(\w+)", re.M)


def strip_comments(src: str) -> str:
    # Keep offsets/lines: blank out // comments (no string in this code base
    # carries `//` inside a call we care about).
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), src)


def fn_spans(src: str):
    """Yield (name, start, body_end) for every fn with a body."""
    for m in FN.finditer(src):
        brace = src.find("{", m.end())
        semi = src.find(";", m.end())
        if brace < 0 or (0 <= semi < brace):
            continue  # trait/extern declaration
        depth, i = 0, brace
        while i < len(src):
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        yield m.group(1), m.start(), i


def growers() -> set[str]:
    src = strip_comments(SCRATCH.read_text())
    bodies = {name: src[start:end] for name, start, end in fn_spans(src)}
    grow = set(PRIMITIVES)
    changed = True
    while changed:
        changed = False
        for name, body in bodies.items():
            if name in grow:
                continue
            calls = set(re.findall(r"\b(\w+)\s*\(", body[body.find("{"):]))
            if calls & grow:
                grow.add(name)
                changed = True
    return grow - PRIMITIVES


def violations(grow: set[str]) -> list[str]:
    call = re.compile(r"\.scratch\s*\.\s*(" + "|".join(sorted(grow)) + r")\s*\(")
    out = []
    for path in sorted((ROOT / "crates").rglob("*.rs")):
        if path == SCRATCH or "/target/" in str(path):
            continue
        src = strip_comments(path.read_text(errors="replace"))
        if ".scratch" not in src:
            continue
        spans = list(fn_spans(src))
        for m in call.finditer(src):
            enclosing = [s for s in spans if s[1] <= m.start() <= s[2]]
            if not enclosing:
                continue
            name, start, _ = max(enclosing, key=lambda s: s[1])  # innermost
            if "invalidate_for_scratch_growth" not in src[start:m.start()]:
                line = src.count("\n", 0, m.start()) + 1
                rel = path.relative_to(ROOT)
                out.append(f"{rel}:{line}: fn {name} calls Scratch::{m.group(1)} "
                           f"(can grow) without invalidate_for_scratch_growth first")
    return out


def main() -> int:
    grow = growers()
    bad = violations(grow)
    for v in bad:
        print(f"check-scratch-growth: {v}", file=sys.stderr)
    if bad:
        return 1
    print(f"check-scratch-growth: {len(grow)} growing Scratch methods, every caller invalidates first")
    return 0


if __name__ == "__main__":
    sys.exit(main())
