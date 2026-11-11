#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Generate the Route-A PeaceMaker twins of `kernels/src/moe_combine_ordered.hip` (gfx1200, gfx1201).

Explicit invocation only: no test, build script or runtime path runs this. Per arch it

  1. compiles the source with the runtime's `core_hipcc_flags`
     (`--genco --offload-arch=<arch> -O3 --no-offload-compress -fuse-cuid=none
     -mcode-object-version=6`, `rdna-compute/src/compiler.rs`) into the anchor
     `kernels/moe_combine_ordered_hipcc_<arch>.co`;
  2. seals BOTH entries of the module with `peacemaker-isa`'s `hip_twin` example (lift,
     `emit::checked` byte identity, hazards, kernarg ABI == the anchor's, bundle, sha-pin) into one
     `kernels/moe_combine_ordered_pm_<arch>.hxaco` (+ `.hxaco.sha256`):
       moe_down_combine_grouped_top10_bf16in_ordered
       moe_down_combine_grouped_top10_bf16in_ordered_zinit
  3. writes `kernels/moe_combine_ordered_pm_<arch>.twin.json` (source sha256, compiler provenance,
     portable flags, both seal reports incl. the kernarg ABI, per-symbol code-object metadata
     `max_flat_workgroup_size`).

Tests (`crates/peacemaker-isa/tests/hip_twin_moe_combine_ordered.rs`) only consume the committed files.

    scripts/pm-moe-combine-ordered-twin.py [--arch gfx1200|gfx1201|all] [--rocm-path DIR]
        [--build-wrapper 'WRAPPER ARGS'] [--seal-bin PATH] [--out-dir DIR]
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODULE = "moe_combine_ordered"
SYMBOLS = (
    "moe_down_combine_grouped_top10_bf16in_ordered",
    "moe_down_combine_grouped_top10_bf16in_ordered_zinit",
)
SOURCE_REL = "kernels/src/moe_combine_ordered.hip"
ARCHES = ("gfx1200", "gfx1201")

def core_hipcc_flags(arch):
    """Mirror of `rdna-compute` `core_hipcc_flags`."""
    return [
        "--genco",
        f"--offload-arch={arch}",
        "-O3",
        "--no-offload-compress",
        "-fuse-cuid=none",
        "-mcode-object-version=6",
    ]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def die(msg):
    print(f"pm-moe-combine-ordered-twin: {msg}", file=sys.stderr)
    sys.exit(1)


def find_rocm(arg):
    cands = [arg, os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH"), "/opt/rocm"]
    for c in cands:
        if c and (Path(c) / "bin" / "hipcc").is_file():
            return Path(c)
    hipcc = shutil.which("hipcc")
    if hipcc:
        return Path(hipcc).resolve().parent.parent
    die("no ROCm found: pass --rocm-path or set ROCM_PATH (needs bin/hipcc)")


def run(cmd, cwd=None, env=None, capture=True):
    p = subprocess.run(
        cmd, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if p.returncode != 0:
        die(f"command failed ({p.returncode}): {shlex.join(cmd)}\n{p.stderr or ''}")
    return p.stdout


_ABS_PATH = re.compile(r"(^|\s)/[\w.+-]+/")


def compiler_version(hipcc, env):
    """`hipcc --version` minus every line that names a host path (InstalledDir, config file)."""
    out = run([str(hipcc), "--version"], env=env)
    keep = []
    for line in out.splitlines():
        if line.startswith(("InstalledDir", "Configuration file")) or _ABS_PATH.search(line):
            continue
        if line.strip():
            keep.append(line.rstrip())
    return keep



def seal_cmd(args, root):
    if args.seal_bin:
        return [str(Path(args.seal_bin).resolve())]
    wrapper = shlex.split(args.build_wrapper) if args.build_wrapper else []
    return [*wrapper, "cargo", "run", "--release", "-p", "peacemaker-isa",
            "--features", "lab", "--example", "hip_twin", "--"]



def find_readelf(rocm):
    for c in (rocm / "lib" / "llvm" / "bin" / "llvm-readelf", rocm / "llvm" / "bin" / "llvm-readelf",
              rocm / "bin" / "llvm-readelf"):
        if c.is_file():
            return c
    found = shutil.which("llvm-readelf")
    return Path(found) if found else None


def code_object_metadata(readelf, anchor, tmp):
    """`.max_flat_workgroup_size` per sealed symbol from the anchor's amdhsa note metadata."""
    if readelf is None:
        die("llvm-readelf not found (needed to record max_flat_workgroup_size)")
    # The anchor is a clang offload bundle (uncompressed): the device ELF starts at its only ELF magic.
    blob = anchor.read_bytes()
    at = blob.find(b"\x7fELF")
    if at < 0:
        die(f"{anchor.name}: no device ELF in the anchor")
    elf = Path(tmp) / f"{anchor.stem}.elf"
    elf.write_bytes(blob[at:])
    text = run([str(readelf), "--notes", str(elf)])
    lines = text.splitlines()
    out = {}
    for sym in SYMBOLS:
        name_at = [i for i, l in enumerate(lines) if re.fullmatch(rf"\s*\.name:\s*{re.escape(sym)}", l)]
        if len(name_at) != 1:
            die(f"{anchor.name}: expected one kernel metadata entry named {sym}, found {len(name_at)}")
        flat = None
        for j in range(name_at[0] - 1, -1, -1):
            m = re.match(r"\s*\.max_flat_workgroup_size:\s*(\d+)", lines[j])
            if m:
                flat = int(m.group(1))
                break
            if re.match(r"\s*\.name:", lines[j]):
                break
        if flat is None:
            die(f"{anchor.name}: no .max_flat_workgroup_size before {sym}")
        out[sym] = {"max_flat_workgroup_size": flat}
    return out


def build_arch(args, root, hipcc, env, arch, tmp, readelf):
    out_dir = args.out_dir
    co = out_dir / f"{MODULE}_hipcc_{arch}.co"
    hxaco = out_dir / f"{MODULE}_pm_{arch}.hxaco"
    manifest = out_dir / f"{MODULE}_pm_{arch}.twin.json"
    flags = core_hipcc_flags(arch)

    tmp_co = Path(tmp) / f"anchor_{arch}.co"
    run([str(hipcc), *flags, "-o", str(tmp_co), SOURCE_REL], cwd=root, env=env)
    shutil.copyfile(tmp_co, co)

    report_path = Path(tmp) / f"report_{arch}.json"
    kernels = []
    for sym in SYMBOLS:
        kernels += ["--kernel", sym, "--abi-ref", str(co), "--abi-ref-symbol", sym]
    run([*seal_cmd(args, root), "--in", str(co), "--arch", arch, "--out", str(hxaco),
         "--report", str(report_path), *kernels], cwd=root, env=env)
    reports = json.loads(report_path.read_text())
    if not isinstance(reports, list) or [r["kernel"] for r in reports] != list(SYMBOLS):
        die(f"{arch}: seal report is not one entry per symbol in order {SYMBOLS}")

    rel = lambda p: p.relative_to(root).as_posix() if p.is_relative_to(root) else p.name
    doc = {
        "module": f"{MODULE}_pm_{arch}",
        "arch": arch,
        "symbols": list(SYMBOLS),
        "source": {
            "path": SOURCE_REL,
            "bytes": (root / SOURCE_REL).stat().st_size,
            "sha256": sha256_file(root / SOURCE_REL),
        },
        "compiler": {
            "version": compiler_version(hipcc, env),
            "flags": flags,
            "invocation": "hipcc <flags> -o <anchor.co> " + SOURCE_REL,
        },
        "anchor": {"file": co.name, "bytes": co.stat().st_size, "sha256": sha256_file(co)},
        "hxaco": {"file": hxaco.name, "bytes": hxaco.stat().st_size, "sha256": sha256_file(hxaco)},
        "code_object_metadata": code_object_metadata(readelf, co, tmp),
        "seal": {
            "tool": "peacemaker-isa example hip_twin (Route A: lift, emit::checked, bundle)",
            "reports": reports,
        },
    }
    # Supplemental receipts (ledger/M7 OPEN diagnostics, gpu_oracle) concern one exact
    # hxaco: carried over only when the existing manifest is for the same hxaco and the receipt's
    # own input hash matches, otherwise omitted so a CPU audit fails closed.
    new_sha = doc["hxaco"]["sha256"]
    if manifest.exists():
        try:
            old = json.loads(manifest.read_text())
        except ValueError:
            old = {}
        if old.get("hxaco", {}).get("sha256") == new_sha:
            for key in ("m7", "ledger_cfg", "gpu_oracle"):
                rec = old.get(key)
                if isinstance(rec, dict) and rec.get("input_hxaco_sha256") == new_sha:
                    doc[key] = rec
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"{arch}: wrote {rel(co)} {rel(hxaco)} {rel(manifest)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", default="all", choices=(*ARCHES, "all"))
    ap.add_argument("--rocm-path", help="ROCm root (default: $ROCM_PATH, $HIP_PATH, /opt/rocm, PATH)")
    ap.add_argument("--build-wrapper", default=None,
                    help="command prefix for the cargo invocation, e.g. a build-slot script")
    ap.add_argument("--seal-bin", help="prebuilt hip_twin example binary (skips cargo)")
    ap.add_argument("--out-dir", type=Path, help="artifact directory (default: <repo>/kernels)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    if not (root / SOURCE_REL).is_file():
        die(f"{SOURCE_REL} not found under {root}")
    args.out_dir = (args.out_dir or root / "kernels").resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rocm = find_rocm(args.rocm_path)
    hipcc = rocm / "bin" / "hipcc"
    readelf = find_readelf(rocm)
    env = dict(os.environ, ROCM_PATH=str(rocm))

    arches = ARCHES if args.arch == "all" else (args.arch,)
    work = Path(os.environ.get("TMPDIR") or root / "target") / "pm-moe-combine-ordered-twin"
    work.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(work)
    with tempfile.TemporaryDirectory(prefix="run-", dir=work) as tmp:
        for arch in arches:
            build_arch(args, root, hipcc, env, arch, tmp, readelf)


if __name__ == "__main__":
    main()
