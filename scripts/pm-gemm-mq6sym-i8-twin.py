#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Generate the Route-A PeaceMaker twin of the F4 `gemm_mq6sym_i8_set_*_x4_gfx1201` entries.

Explicit invocation only: no test, build script or runtime path runs this. It

  1. builds the runtime source text `GEMM_MQ6SYM_I8_GFX1201_SRC`
     (`rdna-compute/src/kernels.rs`): `#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\\n`
     + `kernels/src/block_i8_128_quant.hip` + `kernels/src/gemm_mq6sym_i8.gfx1201.hip`;
  2. compiles it with the runtime's `core_hipcc_flags` (`--genco --offload-arch=gfx1201 -O3
     --no-offload-compress -fuse-cuid=none -mcode-object-version=6`) into the anchor
     `kernels/gemm_mq6sym_i8_hipcc_gfx1201.co`;
  3. seals the four x4 entries with `peacemaker-isa`'s `hip_twin` example (lift,
     `emit::checked` whole-ELF byte identity, kernarg ABI vs the anchor, bundle, sha-pin) into
     `kernels/gemm_mq6sym_i8_pm_gfx1201.hxaco` (+ `.hxaco.sha256`);
  4. writes `kernels/gemm_mq6sym_i8_pm_gfx1201.twin.json` (source + prelude hashes, compiler
     provenance, portable flags, one seal report per entry).

The twin is proof only: dispatch keeps using the registry HIP source.

    scripts/pm-gemm-mq6sym-i8-twin.py [--rocm-path DIR] [--build-wrapper 'WRAPPER ARGS']
        [--seal-bin PATH] [--out-dir DIR]
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

ARCH = "gfx1201"
MODULE = "gemm_mq6sym_i8_pm_gfx1201"
ANCHOR = "gemm_mq6sym_i8_hipcc_gfx1201.co"
SYMBOLS = (
    "gemm_mq6sym_i8_set_m64n16_x4_gfx1201",
    "gemm_mq6sym_i8_set_m32n32_x4_gfx1201",
    "gemm_mq6sym_i8_set_m64n48_x4_gfx1201",
    "gemm_mq6sym_i8_set_m64n16y_x4_gfx1201",
)
PRELUDE_DEFINE = "#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n"
PRELUDE_REL = "kernels/src/block_i8_128_quant.hip"
SOURCE_REL = "kernels/src/gemm_mq6sym_i8.gfx1201.hip"


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


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def die(msg):
    print(f"pm-gemm-mq6sym-i8-twin: {msg}", file=sys.stderr)
    sys.exit(1)


def find_rocm(arg):
    for c in (arg, os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH"), "/opt/rocm"):
        if c and (Path(c) / "bin" / "hipcc").is_file():
            return Path(c)
    hipcc = shutil.which("hipcc")
    if hipcc:
        return Path(hipcc).resolve().parent.parent
    die("no ROCm found: pass --rocm-path or set ROCM_PATH (needs bin/hipcc)")


def run(cmd, cwd=None, env=None):
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        die(f"command failed ({p.returncode}): {shlex.join(cmd)}\n{p.stderr or ''}")
    return p.stdout


_ABS_PATH = re.compile(r"(^|\s)/[\w.+-]+/")


def compiler_version(hipcc, env):
    """`hipcc --version` minus every line that names a host path."""
    keep = []
    for line in run([str(hipcc), "--version"], env=env).splitlines():
        if line.startswith(("InstalledDir", "Configuration file")) or _ABS_PATH.search(line):
            continue
        if line.strip():
            keep.append(line.rstrip())
    return keep


def seal_cmd(args):
    if args.seal_bin:
        return [str(Path(args.seal_bin).resolve())]
    wrapper = shlex.split(args.build_wrapper) if args.build_wrapper else []
    return [*wrapper, "cargo", "run", "--release", "-p", "peacemaker-isa",
            "--features", "lab", "--example", "hip_twin", "--"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rocm-path", help="ROCm root (default: $ROCM_PATH, $HIP_PATH, /opt/rocm, PATH)")
    ap.add_argument("--build-wrapper", default=None,
                    help="command prefix for the cargo invocation, e.g. a build-slot script")
    ap.add_argument("--seal-bin", help="prebuilt hip_twin example binary (skips cargo)")
    ap.add_argument("--out-dir", type=Path, help="artifact directory (default: <repo>/kernels)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    for rel in (PRELUDE_REL, SOURCE_REL):
        if not (root / rel).is_file():
            die(f"{rel} not found under {root}")
    out_dir = (args.out_dir or root / "kernels").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rocm = find_rocm(args.rocm_path)
    hipcc = rocm / "bin" / "hipcc"
    env = dict(os.environ, ROCM_PATH=str(rocm))

    prelude = (root / PRELUDE_REL).read_bytes()
    source = (root / SOURCE_REL).read_bytes()
    combined = PRELUDE_DEFINE.encode() + prelude + source
    flags = core_hipcc_flags(ARCH)

    work = root / "target" / "pm-gemm-mq6sym-i8-twin"
    work.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(work)
    co = out_dir / ANCHOR
    hxaco = out_dir / f"{MODULE}.hxaco"
    manifest = out_dir / f"{MODULE}.twin.json"
    with tempfile.TemporaryDirectory(prefix="run-", dir=work) as tmp:
        src = Path(tmp) / "gemm_mq6sym_i8_gfx1201.hip"
        src.write_bytes(combined)
        tmp_co = Path(tmp) / "anchor.co"
        run([str(hipcc), *flags, "-o", str(tmp_co), str(src)], cwd=root, env=env)
        shutil.copyfile(tmp_co, co)

        report_path = Path(tmp) / "report.json"
        cmd = [*seal_cmd(args), "--in", str(co), "--arch", ARCH, "--out", str(hxaco),
               "--report", str(report_path)]
        for sym in SYMBOLS:
            cmd += ["--kernel", sym, "--abi-ref", str(co), "--abi-ref-symbol", sym]
        run(cmd, cwd=root, env=env)
        reports = json.loads(report_path.read_text())
    if not isinstance(reports, list) or [r["kernel"] for r in reports] != list(SYMBOLS):
        die("hip_twin did not return one report per requested entry, in order")

    doc = {
        "module": MODULE,
        "arch": ARCH,
        "symbols": list(SYMBOLS),
        "source": {
            "path": SOURCE_REL,
            "bytes": len(source),
            "sha256": sha256_bytes(source),
        },
        "prelude": {
            "define": PRELUDE_DEFINE.rstrip("\n"),
            "path": PRELUDE_REL,
            "bytes": len(prelude),
            "sha256": sha256_bytes(prelude),
            "combined_bytes": len(combined),
            "combined_sha256": sha256_bytes(combined),
            "runtime_const": "rdna-compute kernels::GEMM_MQ6SYM_I8_GFX1201_SRC",
        },
        "compiler": {
            "version": compiler_version(hipcc, env),
            "flags": flags,
            "invocation": "hipcc <flags> -o <anchor.co> <define + prelude + source>",
        },
        "anchor": {"file": co.name, "bytes": co.stat().st_size,
                   "sha256": sha256_bytes(co.read_bytes())},
        "hxaco": {"file": hxaco.name, "bytes": hxaco.stat().st_size,
                  "sha256": sha256_bytes(hxaco.read_bytes())},
        "seal": {
            "tool": "peacemaker-isa example hip_twin (Route A: lift, emit::checked, bundle)",
            "reports": reports,
        },
    }
    doc["strict_cfg_ledger"] = {
        "status": "OPEN",
        "proven": False,
        "scope": ("DS-store-source WAR findings are UNPROVEN original-LLVM obligations; byte identity "
                  "and zero blocking hip_twin hazards do not discharge them"),
        "evidence": "crates/peacemaker-isa/tests/ledger_cfg_pins/allowlist.rs",
    }
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {co.name} {hxaco.name} {manifest.name}")


if __name__ == "__main__":
    main()
