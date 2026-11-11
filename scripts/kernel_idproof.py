#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Did any shipped kernel object change between two commits? No GPU needed.

    kernel_idproof.py BASE_REF NEW_REF [--arch gfx1100 ...]
                      [--require-identical GLOB ...] [--json OUT] [--work DIR]

Each ref is built with scripts/build-kernel-pack.sh --commit REF: `git archive`
of the exact commit, fresh HOME, no HIPFIRE_* overrides. Per arch, every
packaged module (`<name>.hsaco` / `.hash` / `.index.json`) is classified
identical / changed / removed / added by sha256. A fifth class,
`nondeterministic`, is a module whose `.index.json` is identical but for
`object_sha256` (same source, compile key, flags, scheduler profile and
toolchain) and whose `.hash` matches, yet whose object bytes differ: the
compiler turned identical inputs into different code (hipcc's post-RA
scheduler on `qsa_bf16_scores`, gfx1201), so nothing in the commits caused it.
It is listed by name and never fails --require-identical.
The pack's manifest.json is not compared (it names the commit).

Exit 0: done, and no module matching --require-identical changed or vanished.
Exit 1: a --require-identical module changed or vanished.
Exit 2: build or usage error.
Needs git, cargo and hipcc.
"""
import argparse
import collections
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "scripts" / "build-kernel-pack.sh"
EXTS = (".hsaco", ".hash", ".index.json")
DEFAULT_ARCHS = ("gfx1100", "gfx1151", "gfx1201")
# Not a file: sha256 of the module's index.json without `object_sha256`, i.e. every
# input of the compile. Equal inputs with different object bytes is toolchain noise.
INPUTS = ".inputs"


def module_digests(files: dict[str, str]) -> dict[str, dict[str, str]]:
    """{file name: sha256} -> {module: {ext: sha256}}."""
    mods: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for name, sha in files.items():
        for ext in EXTS:
            if name.endswith(ext):
                mods[name[: -len(ext)]][ext] = sha
                break
        else:
            mods[name]["?"] = sha
    return dict(mods)


def index_inputs(index_json: bytes) -> str:
    """sha256 of an index.json's compile inputs: every field but `object_sha256`."""
    index = json.loads(index_json)
    index.pop("object_sha256", None)
    return hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()


def classify(base: dict, new: dict) -> dict[str, list[str]]:
    """Module digests of two builds -> {identical, changed, nondeterministic, removed, added}.

    `changed` and `nondeterministic` entries name the differing parts:
    `foo (.hash,.hsaco)`. A module is `nondeterministic` when its compile inputs
    (`INPUTS`) are present and equal in both builds and only the object (and
    the `object_sha256` it records) differs; anything else that differs is `changed`.
    """
    out: dict[str, list[str]] = {
        "identical": [], "changed": [], "nondeterministic": [], "removed": [], "added": [],
    }
    for m in sorted(set(base) | set(new)):
        if m not in new:
            out["removed"].append(m)
        elif m not in base:
            out["added"].append(m)
        elif base[m] == new[m]:
            out["identical"].append(m)
        else:
            parts = sorted(
                e for e in set(base[m]) | set(new[m])
                if e != INPUTS and base[m].get(e) != new[m].get(e)
            )
            noise = (
                INPUTS in base[m]
                and base[m].get(INPUTS) == new[m].get(INPUTS)
                and ".hsaco" in parts
                and set(parts) <= {".hsaco", ".index.json"}
            )
            out["nondeterministic" if noise else "changed"].append(f"{m} ({','.join(parts)})")
    return out


def violations(result: dict[str, list[str]], globs: list[str]) -> list[str]:
    """Modules matching a --require-identical glob that changed or vanished."""
    hits = []
    for kind in ("changed", "removed"):
        for entry in result[kind]:
            module = entry.split(" ", 1)[0]
            if any(fnmatch.fnmatchcase(module, g) for g in globs):
                hits.append(f"{kind}: {entry}")
    return hits


def resolve(ref: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def build(commit: str, archs: list[str], out: Path, target: Path) -> dict[str, dict]:
    """Build packs for one commit; return {arch: module digests}."""
    tag = f"idproof-{commit[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(PACK), "--tag", tag, "--commit", commit, "--out", str(out),
         "--target-dir", str(target), *archs],
        check=True, stdout=sys.stderr,
        # build-kernel-pack's scratch (sources + objects) stays in the work dir.
        env={**os.environ, "TMPDIR": str(out)},
    )
    digests = {}
    for arch in archs:
        files, inputs = {}, {}
        with tarfile.open(out / f"hipfire-kernels-{tag}-{arch}.tar.gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.startswith(f"{arch}/"):
                    data = tar.extractfile(member).read()
                    name = member.name[len(arch) + 1:]
                    files[name] = hashlib.sha256(data).hexdigest()
                    if name.endswith(".index.json"):
                        inputs[name[: -len(".index.json")]] = index_inputs(data)
        digests[arch] = module_digests(files)
        for module, sha in inputs.items():
            digests[arch][module][INPUTS] = sha
    return digests


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--arch", nargs="+", default=list(DEFAULT_ARCHS))
    ap.add_argument("--require-identical", nargs="+", default=[], metavar="GLOB")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--work", type=Path, help="build dir (default: a fresh dir under target/)")
    args = ap.parse_args()

    try:
        base, new = resolve(args.base), resolve(args.new)
    except subprocess.CalledProcessError as e:
        print(f"kernel_idproof: {e.stderr.strip()}", file=sys.stderr)
        return 2
    if args.work:
        work = args.work
        work.mkdir(parents=True, exist_ok=True)
    else:
        (ROOT / "target").mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="kernel-idproof.", dir=ROOT / "target"))
    try:
        # One cargo target per label: git-archive sources carry commit mtimes
        # older than the other label's outputs, so a shared target can reuse
        # stale artifacts and silently report IDENTICAL.
        built = {
            label: build(commit, args.arch, work / label, work / f"cargo-target-{label}")
            for label, commit in (("base", base), ("new", new))
        }
    except (subprocess.CalledProcessError, OSError, tarfile.TarError) as e:
        print(f"kernel_idproof: build failed: {e}", file=sys.stderr)
        return 2

    report = {"base": base, "new": new, "archs": {}}
    bad = []
    print(f"kernel_idproof: base {base[:12]} ({args.base}) vs new {new[:12]} ({args.new})")
    for arch in args.arch:
        result = classify(built["base"][arch], built["new"][arch])
        report["archs"][arch] = result
        counts = ", ".join(f"{k} {len(v)}" for k, v in result.items())
        verdict = "IDENTICAL" if not (result["changed"] or result["removed"] or result["added"]) else "CHANGED"
        print(f"== {arch}: {verdict}: {counts}")
        for kind in ("changed", "removed", "added"):
            for entry in result[kind]:
                print(f"   {kind}: {entry}")
        for entry in result["nondeterministic"]:
            print(f"   toolchain-nondeterministic (identical compile inputs, different bytes): {entry}")
            module = entry.split(" ", 1)[0]
            if any(fnmatch.fnmatchcase(module, g) for g in args.require_identical):
                print(f"kernel_idproof: {arch} required-identical module {module} is toolchain-nondeterministic only", file=sys.stderr)
        bad += [f"{arch} {v}" for v in violations(result, args.require_identical)]
    report["require_identical_violations"] = bad
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    for v in bad:
        print(f"kernel_idproof: required-identical module {v}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
