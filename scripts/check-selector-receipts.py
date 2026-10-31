#!/usr/bin/env python3
"""Row-predicate selector constants must carry a covering selector receipt.

Scans crates/*/src/**/*.rs for `const NAME: <int type> = <int>;` where NAME looks
like a row-count predicate (MIN_N, MAX_ROWS, ...). Each match needs a
`// selector-receipt: <repo-relative path>` comment within the 5 lines above it
pointing at a `pm.selector_receipt.v1` JSON whose `constant`/`value` match, with
`covers == true` and every existing `records[].path` matching its `sha256`; OR
the `(path, NAME)` pair is grandfathered in scripts/selector-receipts-baseline.txt
(one `path NAME` per line). A baseline line whose constant no longer exists
fails, which keeps the list exact.

Constants in PENDING are awaiting a receipt from a card job: reported as
"awaiting receipt", not failures. Remove the entry when the receipt lands.

Usage:
  check-selector-receipts.py [--root DIR]    check (exit 1 on violations)
  check-selector-receipts.py --list          print matched constants
  check-selector-receipts.py --write-baseline  regenerate the baseline from the tree
  check-selector-receipts.py --self-test
"""
import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

SCHEMA = "pm.selector_receipt.v1"
BASELINE_REL = "scripts/selector-receipts-baseline.txt"
# (path, NAME) pairs awaiting a receipt from a card job. BF16_TILED_ROWS_MIN_N carries the floor of
# the removed F32 selector over to the BF16-weight kernels (same tiled shape, bit-identical outputs);
# its tiled-vs-batched crossover is not re-measured for BF16 yet.
PENDING: set = {("crates/hipfire-dispatch/src/pipeline/stream_mix.rs", "BF16_TILED_ROWS_MIN_N")}
NAME_RE = re.compile(
    r"(^|_)(MIN|MAX)_(N|M|ROWS|TOKENS|BATCH|T)($|_)"
    r"|ROWS_MIN|MIN_ROWS|MIN_TOKENS|ROWS_MAX|MAX_ROWS"
)
CONST_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?const\s+([A-Z][A-Z0-9_]*)\s*:\s*"
    r"(usize|u32|u64|i32)\s*=\s*([0-9][0-9_]*)\s*;"
)
ANNOT_RE = re.compile(r"//\s*selector-receipt:\s*(\S+)")


def scan(root: Path):
    """Yield (relpath, name, value, line_no, lines) for every matching constant."""
    out = []
    for f in sorted(root.glob("crates/*/src/**/*.rs")):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        rel = f.relative_to(root).as_posix()
        for i, line in enumerate(lines):
            m = CONST_RE.match(line)
            if m and NAME_RE.search(m.group(1)):
                out.append((rel, m.group(1), int(m.group(3).replace("_", "")), i + 1, lines))
    return out


def load_baseline(root: Path):
    p = root / BASELINE_REL
    pairs = set()
    if p.exists():
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                parts = ln.split()
                if len(parts) == 2:
                    pairs.add((parts[0], parts[1]))
    return pairs


def check_receipt(root: Path, rel_receipt: str, name: str, value: int):
    p = root / rel_receipt
    try:
        r = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        return f"receipt {rel_receipt} unreadable: {e}"
    if r.get("schema") != SCHEMA:
        return f"receipt {rel_receipt} schema != {SCHEMA}"
    if r.get("constant") != name:
        return f"receipt {rel_receipt} constant {r.get('constant')!r} != {name}"
    if r.get("value") != value:
        return f"receipt {rel_receipt} value {r.get('value')!r} != {value}"
    if r.get("covers") is not True:
        return f"receipt {rel_receipt} does not cover the value (covers != true)"
    for rec in r.get("records") or []:
        rp = root / rec.get("path", "")
        if rp.is_file():
            h = hashlib.sha256(rp.read_bytes()).hexdigest()
            if h != rec.get("sha256"):
                return f"receipt {rel_receipt} record {rec.get('path')} sha256 mismatch"
    return None


def run_check(root: Path, pending=PENDING):
    """Return (violations, awaiting)."""
    baseline = load_baseline(root)
    found = scan(root)
    present = {(rel, name) for rel, name, *_ in found}
    violations, awaiting = [], []
    for rel, name, value, ln, lines in found:
        key = (rel, name)
        if key in baseline:
            continue
        window = lines[max(0, ln - 6) : ln - 1]
        receipt = None
        for w in reversed(window):
            m = ANNOT_RE.search(w)
            if m:
                receipt = m.group(1)
                break
        if receipt is None:
            if key in pending:
                awaiting.append(f"{rel}:{ln}: {name} = {value}: awaiting receipt")
            else:
                violations.append(
                    f"{rel}:{ln}: {name} = {value}: no `// selector-receipt:` annotation "
                    f"and not in {BASELINE_REL}"
                )
            continue
        err = check_receipt(root, receipt, name, value)
        if err:
            violations.append(f"{rel}:{ln}: {name}: {err}")
    for pair in sorted(baseline - present):
        violations.append(f"{BASELINE_REL}: stale baseline line `{pair[0]} {pair[1]}` (constant no longer exists)")
    return violations, awaiting


def report(root: Path):
    v, a = run_check(root)
    for ln in a:
        print(ln)
    for ln in v:
        print(ln)
    return 1 if v else 0


def write_baseline(root: Path):
    pairs = sorted(
        {(rel, name) for rel, name, *_ in scan(root)} - PENDING
    )
    body = (
        "# Grandfathered row-predicate constants without a selector receipt.\n"
        "# One `path NAME` per line. Generated by scripts/check-selector-receipts.py\n"
        "# --write-baseline; do NOT add lines for new constants: add a receipt instead.\n"
    ) + "".join(f"{p} {n}\n" for p, n in pairs)
    (root / BASELINE_REL).write_text(body)
    print(f"wrote {len(pairs)} baseline entries to {BASELINE_REL}")


def self_test():
    def mk(td, const_block, baseline="", receipt=None):
        root = Path(td)
        (root / "crates/x/src").mkdir(parents=True)
        (root / "scripts").mkdir()
        (root / "crates/x/src/a.rs").write_text(const_block)
        (root / BASELINE_REL).write_text(baseline)
        if receipt is not None:
            (root / "receipts").mkdir()
            (root / "receipts/R.json").write_text(json.dumps(receipt))
        return root

    def rcpt(covers):
        return {"schema": SCHEMA, "constant": "FOO_MIN_N", "value": 7, "kind": "min",
                "covers": covers, "records": []}

    plain = "const FOO_MIN_N: usize = 7;\n"
    annot = "// selector-receipt: receipts/R.json\nconst FOO_MIN_N: usize = 7;\n"
    ok = True

    def expect(label, root, want_fail):
        nonlocal ok
        v, _ = run_check(root, pending=set())
        good = bool(v) == want_fail
        ok &= good
        print(f"self-test {label}: {'ok' if good else 'FAIL'}" + (f" ({v[0]})" if v else ""))

    with tempfile.TemporaryDirectory() as td:
        expect("(a) new constant without receipt fails", mk(td, plain), True)
    with tempfile.TemporaryDirectory() as td:
        expect("(b) covering receipt passes", mk(td, annot, receipt=rcpt(True)), False)
    with tempfile.TemporaryDirectory() as td:
        expect("(c) non-covering receipt fails", mk(td, annot, receipt=rcpt(False)), True)
    with tempfile.TemporaryDirectory() as td:
        expect("(d) stale baseline line fails",
               mk(td, "const OTHER: usize = 1;\n", baseline="crates/x/src/a.rs FOO_MIN_N\n"), True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if a.self_test:
        return self_test()
    if a.list:
        for rel, name, value, ln, _ in scan(root):
            print(f"{rel}:{ln} {name} = {value}")
        return 0
    if a.write_baseline:
        write_baseline(root)
        return 0
    return report(root)


if __name__ == "__main__":
    sys.exit(main())
