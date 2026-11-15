#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Check fixtures/pins.toml, the fixture and quality pins behind AGENTS.md.

    check-pins.py [--root DIR]                 schema, cross-refs, prompt md5s,
                                               mirrored copies, `pin:` citations
    check-pins.py --self-test                  plant each defect, expect a failure
    check-pins.py verify KIND/ID PATH          digest (and size) of PATH vs a pin
    check-pins.py get KIND/ID FIELD            print one field (for shell callers)

Pins are cited from docs and scripts as `pin:<kind>/<id>`; every citation in
AGENTS.md, CLAUDE.md, CONTRIBUTING.md, README.md, docs/, .agents/ and scripts/
must name an existing pin. A `current` kld pin must name an evaluator (build
commit + binary md5) and a receipt. Exit 0 = clean, 1 = violation or digest
mismatch, 2 = usage error.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS_REL = "fixtures/pins.toml"
SCHEMA = "hipfire.pins"
VERSION = 1
STATUSES = {"current", "previous", "historical", "retired"}
CITE_RE = re.compile(r"pin:([a-z]+)/([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
# (path, suffix): a top-level file, or every file under a directory whose name
# ends with suffix ("" = any file).
CITE_SPECS = [("AGENTS.md", None), ("CLAUDE.md", None), ("CONTRIBUTING.md", None),
              ("README.md", None), ("docs", ".md"), (".agents", ".md"), ("scripts", "")]
HEX = {"sha256": 64, "md5": 32, "kldseq_sha256": 64, "binary_md5": 32,
       "binary_sha256": 64, "sha256_before_qt54_retag": 64}
PREFIX = {"control_kldseq_sha256_prefix", "ple_source_rows"}
COMMON = {"status": str, "date": str, "pinned_in": str, "role": str, "note": str,
          "receipt": str, "mirrored_in": list}
FILE_ID = {"file": str, "hf_repo": str, "registry_tag": str, "aliases": list,
           "size_bytes": int, "sha256": str, "md5": str}
KINDS: dict[str, tuple[set[str], dict[str, type]]] = {
    "model": ({"status", "date", "file"}, {
        **COMMON, **FILE_ID, "recipe": str, "min_build": str, "validated_on": str,
        "ple_source_rows": str, "download_bytes_with_ple_q4": int,
        "download_bytes_with_ple_q8": int, "sha256_before_qt54_retag": str}),
    "draft": ({"status", "date", "file"}, {
        **COMMON, **FILE_ID, "pairs_with": str, "size_mb": int}),
    "sidecar": ({"status", "date", "file", "sha256"}, {
        **COMMON, **FILE_ID, "binds_to": str}),
    "reference": ({"status", "date", "file", "sha256", "corpus"}, {
        **COMMON, **FILE_ID, "corpus": str, "shape": str, "teacher": str}),
    "prompt": ({"status", "date", "path", "md5"}, {
        **COMMON, "path": str, "md5": str}),
    "evaluator": ({"status", "date", "tool", "commit", "binary_md5"}, {
        **COMMON, "tool": str, "commit": str, "branch": str, "binary_md5": str,
        "binary_sha256": str, "build_arch": list, "toolchain": str}),
    "kld": ({"status", "date", "arch", "route", "kv_flags", "corpus", "kld",
             "kldseq_sha256", "model", "reference"}, {
        **COMMON, "arch": str, "route": str, "kv_flags": str, "corpus": str,
        "kld": float, "kldseq_sha256": str, "model": str, "reference": str,
        "evaluator": str, "toolchain": str, "control_env": str,
        "control_kld": float, "control_kldseq_sha256_prefix": str}),
}
# Fields that name another pin: field -> kind.
XREF = {"pairs_with": "model", "binds_to": "model", "model": "model",
        "reference": "reference", "evaluator": "evaluator"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


def load(root: Path) -> dict:
    with open(root / PINS_REL, "rb") as f:
        return tomllib.load(f)


def pins_of(data: dict) -> dict[str, dict[str, dict]]:
    return {k: data.get(k, {}) for k in KINDS}


def check_schema(data: dict, root: Path) -> list[str]:
    errs = []
    if data.get("schema") != SCHEMA or data.get("version") != VERSION:
        errs.append("%s: want schema = %r, version = %d" % (PINS_REL, SCHEMA, VERSION))
    for key in data:
        if key not in KINDS and key not in ("schema", "version"):
            errs.append("unknown top-level key %r" % key)
    pins = pins_of(data)
    for kind, table in pins.items():
        required, allowed = KINDS[kind]
        if not isinstance(table, dict):
            errs.append("%s: not a table" % kind)
            continue
        for pid, pin in table.items():
            where = "%s/%s" % (kind, pid)
            if not isinstance(pin, dict):
                errs.append("%s: not a table" % where)
                continue
            for field in sorted(required - pin.keys()):
                errs.append("%s: missing %s" % (where, field))
            if kind in ("model", "draft") and "sha256" not in pin and "md5" not in pin:
                errs.append("%s: needs sha256 or md5" % where)
            for field, value in pin.items():
                want = allowed.get(field)
                if want is None:
                    errs.append("%s: unknown field %s" % (where, field))
                    continue
                if want is float and isinstance(value, int) and not isinstance(value, bool):
                    value = float(value)
                if not isinstance(value, want) or isinstance(value, bool):
                    errs.append("%s: %s must be %s" % (where, field, want.__name__))
                    continue
                if field in HEX and not re.fullmatch("[0-9a-f]{%d}" % HEX[field], value):
                    errs.append("%s: %s is not %d lowercase hex" % (where, field, HEX[field]))
                if field in PREFIX and not re.fullmatch("[0-9a-f]{8}", value):
                    errs.append("%s: %s is not an 8-hex prefix" % (where, field))
                if field in ("pinned_in", "commit", "min_build", "validated_on") \
                        and not COMMIT_RE.match(value):
                    errs.append("%s: %s is not a commit sha" % (where, field))
                if field == "date" and not DATE_RE.match(value):
                    errs.append("%s: date is not YYYY-MM-DD" % where)
                if field == "status" and value not in STATUSES:
                    errs.append("%s: status %r not in %s" % (where, value, sorted(STATUSES)))
                if field == "size_bytes" and value <= 0:
                    errs.append("%s: size_bytes must be positive" % where)
                if field in ("mirrored_in", "aliases", "build_arch") and \
                        not all(isinstance(x, str) and x for x in value):
                    errs.append("%s: %s must be a list of strings" % (where, field))
                if field in XREF and value not in pins[XREF[field]]:
                    errs.append("%s: %s names missing pin %s/%s" % (where, field, XREF[field], value))
            if kind == "kld" and pin.get("status") == "current":
                if "evaluator" not in pin or "receipt" not in pin:
                    errs.append("%s: a current kld pin needs evaluator and receipt" % where)
            if kind == "evaluator" and pin.get("status") == "current" and "receipt" not in pin:
                errs.append("%s: a current evaluator needs a receipt" % where)
    return errs


def check_files(data: dict, root: Path) -> list[str]:
    errs = []
    for kind, table in pins_of(data).items():
        for pid, pin in table.items():
            where = "%s/%s" % (kind, pid)
            if kind == "prompt" and isinstance(pin.get("path"), str):
                path = root / pin["path"]
                if not path.is_file():
                    errs.append("%s: %s does not exist" % (where, pin["path"]))
                elif hashlib.md5(path.read_bytes()).hexdigest() != pin.get("md5"):
                    errs.append("%s: md5 of %s != %s" % (where, pin["path"], pin.get("md5")))
            digest = pin.get("sha256") or pin.get("md5")
            for rel in pin.get("mirrored_in", []) if isinstance(pin.get("mirrored_in"), list) else []:
                path = root / rel
                if not path.is_file():
                    errs.append("%s: mirrored_in %s does not exist" % (where, rel))
                elif digest and digest not in path.read_text(encoding="utf-8", errors="replace"):
                    errs.append("%s: mirrored_in %s does not carry %s" % (where, rel, digest))
    return errs


def cite_files(root: Path) -> list[Path]:
    """Tracked and untracked-unignored files in a git checkout, else all files."""
    top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    tracked = None
    if top.returncode == 0 and Path(top.stdout.strip()).resolve() == root.resolve():
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached",
                              "--others", "--exclude-standard"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            tracked = {p for p in out.stdout.split("\0") if p}
    files = []
    for path, suffix in CITE_SPECS:
        base = root / path
        if suffix is None:
            candidates = [base]
        else:
            candidates = sorted(p for p in base.rglob("*") if p.name.endswith(suffix))
        for p in candidates:
            if p.is_file() and (tracked is None or p.relative_to(root).as_posix() in tracked):
                files.append(p)
    return files


def check_citations(data: dict, root: Path) -> list[str]:
    errs = []
    pins = pins_of(data)
    for path in cite_files(root):
        if path.name == "check-pins.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for m in CITE_RE.finditer(line):
                kind, pid = m.group(1), m.group(2)
                if kind not in KINDS or pid not in pins[kind]:
                    errs.append("%s:%d: cites missing pin %s/%s"
                                % (path.relative_to(root), n, kind, pid))
    return errs


def check(root: Path) -> list[str]:
    try:
        data = load(root)
    except (OSError, tomllib.TOMLDecodeError) as e:
        return ["%s: %s" % (PINS_REL, e)]
    return check_schema(data, root) + check_files(data, root) + check_citations(data, root)


def lookup(root: Path, ref: str) -> tuple[str, str, dict]:
    kind, _, pid = ref.partition("/")
    pin = pins_of(load(root)).get(kind, {}).get(pid)
    if pin is None:
        raise KeyError(ref)
    return kind, pid, pin


def verify(root: Path, ref: str, path: str) -> int:
    kind, pid, pin = lookup(root, ref)
    real = os.path.realpath(os.path.expanduser(path))
    if not os.path.isfile(real):
        print("FAIL: %s -> %s is not a file" % (path, real), file=sys.stderr)
        return 1
    size = os.path.getsize(real)
    if "size_bytes" in pin and size != pin["size_bytes"]:
        other = [k for k, p in all_pins(root) if p.get("size_bytes") == size]
        hint = " (size of %s)" % ", ".join(other) if other else ""
        print("FAIL: %s -> %s size %d != pinned %d%s" % (path, real, size, pin["size_bytes"], hint),
              file=sys.stderr)
        return 1
    field = "kldseq_sha256" if kind == "kld" else ("sha256" if "sha256" in pin else "md5")
    algo = hashlib.md5 if field == "md5" else hashlib.sha256
    h = algo()
    with open(real, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != pin[field]:
        other = [k for k, p in all_pins(root) if got in (p.get("sha256"), p.get("md5"),
                                                         p.get("kldseq_sha256"),
                                                         p.get("sha256_before_qt54_retag"))]
        hint = " (matches %s)" % ", ".join(other) if other else ""
        print("FAIL: %s -> %s %s %s%s != pin:%s/%s %s"
              % (path, real, field.replace("kldseq_", ""), got, hint, kind, pid, pin[field]),
              file=sys.stderr)
        return 1
    print("OK: %s -> %s (%d B, %s verified against pin:%s/%s)" % (path, real, size, field, kind, pid))
    return 0


def all_pins(root: Path):
    for kind, table in pins_of(load(root)).items():
        for pid, pin in table.items():
            yield "pin:%s/%s" % (kind, pid), pin


def self_test() -> int:
    """Plant each defect in a scratch copy and require check() to report it."""
    scratch = tempfile.mkdtemp(prefix="check-pins-", dir=str(ROOT.parent))
    n = 0
    try:
        base = Path(scratch)
        (base / "fixtures").mkdir()
        (base / "docs").mkdir()
        (base / "benchmarks").mkdir()
        (base / "benchmarks/p.txt").write_bytes(b"prompt\n")
        md5 = hashlib.md5(b"prompt\n").hexdigest()
        (base / "mirror.json").write_text('{"sha256": "%s"}\n' % ("a" * 64))
        good = """schema = "hipfire.pins"
version = 1
[model."m"]
status = "current"
date = "2026-11-15"
file = "m.hfq"
size_bytes = 10
sha256 = "%s"
mirrored_in = ["mirror.json"]
[reference."r"]
status = "current"
date = "2026-11-15"
file = "r.bin"
sha256 = "%s"
corpus = "WT2"
[evaluator."e"]
status = "current"
date = "2026-11-15"
tool = "eval_hipfire"
commit = "40c52c783a"
binary_md5 = "%s"
receipt = "/receipts/e"
[prompt."p"]
status = "current"
date = "2026-11-15"
path = "benchmarks/p.txt"
md5 = "%s"
[kld."k"]
status = "current"
date = "2026-11-15"
arch = "gfx1201"
route = "A4"
kv_flags = "--kv-mode fp8"
corpus = "WT2"
kld = 0.07
kldseq_sha256 = "%s"
model = "m"
reference = "r"
evaluator = "e"
receipt = "/receipts/k"
""" % ("a" * 64, "b" * 64, "c" * 32, md5, "d" * 64)
        doc_ok = "See `pin:model/m` and pin:kld/k.\n"

        def case(name, toml_text, doc, want_bad, mutate=None):
            nonlocal n
            (base / PINS_REL).write_text(toml_text)
            (base / "docs/x.md").write_text(doc)
            (base / "benchmarks/p.txt").write_bytes(b"prompt\n")
            if mutate:
                mutate()
            errs = check(base)
            n += 1
            assert bool(errs) == want_bad, "case %r: %r" % (name, errs)

        case("clean", good, doc_ok, False)
        case("bad hash", good.replace("a" * 64, "A" * 64, 1), doc_ok, True)
        case("unknown field", good + 'typo_field = "x"\n', doc_ok, True)
        case("dangling model xref", good.replace('model = "m"', 'model = "nope"'), doc_ok, True)
        case("dangling citation", good, doc_ok + "and pin:model/missing.\n", True)
        case("unknown kind citation", good, "pin:bogus/m\n", True)
        case("current kld without evaluator", good.replace('evaluator = "e"\n', ""), doc_ok, True)
        case("bad status", good.replace('status = "current"', 'status = "live"', 1), doc_ok, True)
        case("bad date", good.replace('date = "2026-11-15"', 'date = "15 Nov"', 1), doc_ok, True)
        case("kld not a float", good.replace("kld = 0.07", 'kld = "0.07"'), doc_ok, True)
        case("prompt md5 drift", good, doc_ok, True,
             lambda: (base / "benchmarks/p.txt").write_bytes(b"prompt \n"))
        case("mirror drift", good.replace("a" * 64, "e" * 64, 1), doc_ok, True)
        case("wrong schema", good.replace('"hipfire.pins"', '"other"'), doc_ok, True)
        print("check-pins: self-test OK (%d checks)" % n)
        return 0
    except AssertionError as e:
        print("check-pins: self-test FAILED: %s" % e, file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv: list[str]) -> int:
    root = ROOT
    if len(argv) >= 2 and argv[0] == "--root":
        root, argv = Path(argv[1]).resolve(), argv[2:]
    if argv == ["--self-test"]:
        return self_test()
    if argv and argv[0] in ("verify", "get"):
        if len(argv) != 3:
            print(__doc__, file=sys.stderr)
            return 2
        try:
            if argv[0] == "verify":
                return verify(root, argv[1], argv[2])
            _, _, pin = lookup(root, argv[1])
            value = pin[argv[2]]
        except KeyError as e:
            print("check-pins: no such pin or field: %s" % e, file=sys.stderr)
            return 2
        print(" ".join(value) if isinstance(value, list) else value)
        return 0
    if argv:
        print(__doc__, file=sys.stderr)
        return 2
    errs = check(root)
    for e in errs:
        print(e)
    if errs:
        print("check-pins: %d violation(s) in %s" % (len(errs), PINS_REL), file=sys.stderr)
        return 1
    n = sum(len(t) for t in pins_of(load(root)).values())
    print("check-pins: %d pins OK" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
