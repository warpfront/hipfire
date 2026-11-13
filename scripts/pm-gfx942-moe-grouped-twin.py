#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Generate the Route-A PeaceMaker twin of the gfx942 FP8 grouped MoE kernels.

Explicit invocation only: no test, build script or runtime path runs this. It

  1. compiles `kernels/src/gemm_fp8b128_moe_grouped_bf16mfma.gfx942.hip` with the runtime's
     `core_hipcc_flags` (`--genco --offload-arch=gfx942 -O3 --no-offload-compress
     -fuse-cuid=none -mcode-object-version=6`, `rdna-compute/src/compiler.rs`) into the anchor
     `kernels/gemm_fp8b128_moe_grouped_bf16mfma_hipcc_gfx942.co`. The production object only:
     `FP8MOE_TEST_REFERENCE` is NOT defined, so the diagnostic `_reference` entries are absent;
  2. seals it with `peacemaker-isa`'s `hip_twin` example (gfx942 Route A: typed lift,
     checked byte identity of the whole device ELF, bundle, sha-pin) into
     `kernels/gemm_fp8b128_moe_grouped_bf16mfma_pm_gfx942.hxaco` (+ `.hxaco.sha256`), one
     bundle holding `gemm_fp8b128_moe_grouped_bf16mfma`, `gemm_fp8b128_moe_grouped_bf16mfma_xbf16`
     and `gemm_fp8b128_moe_tile_desc_pad16` (the whole object is one checked ELF);
  3. writes `kernels/gemm_fp8b128_moe_grouped_bf16mfma_pm_gfx942.twin.json` (source sha256,
     compiler provenance, portable flags, the full seal report of each sealed symbol). An
     existing `gpu_oracle` receipt is carried over only while it is bound to both new image
     hashes.

Tests (`crates/peacemaker-isa/tests/hip_twin_gfx942_moe_grouped.rs`) only consume the committed
files.

    scripts/pm-gfx942-moe-grouped-twin.py [--rocm-path DIR]
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

ARCH = "gfx942"
SYMBOLS = (
    "gemm_fp8b128_moe_grouped_bf16mfma",
    "gemm_fp8b128_moe_grouped_bf16mfma_xbf16",
    "gemm_fp8b128_moe_tile_desc_pad16",
)
SOURCE_REL = "kernels/src/gemm_fp8b128_moe_grouped_bf16mfma.gfx942.hip"
STEM = "gemm_fp8b128_moe_grouped_bf16mfma"


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
    print(f"pm-gfx942-moe-grouped-twin: {msg}", file=sys.stderr)
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


def run(cmd, cwd=None, env=None):
    p = subprocess.run(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if p.returncode != 0:
        die(f"{' '.join(map(str, cmd))} failed ({p.returncode}):\n{p.stderr}")
    return p.stdout


_ABS_PATH = re.compile(r"(^|\s)/[\w.+-]+/")


def compiler_version(hipcc, env):
    """`hipcc --version` minus every line that names a host path (InstalledDir, config file)."""
    out = run([str(hipcc), "--version"], env=env)
    return [ln for ln in out.splitlines() if ln.strip() and not _ABS_PATH.search(ln)]


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
    if not (root / SOURCE_REL).is_file():
        die(f"{SOURCE_REL} not found under {root}")
    args.out_dir = (args.out_dir or root / "kernels").resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rocm = find_rocm(args.rocm_path)
    hipcc = rocm / "bin" / "hipcc"
    env = dict(os.environ, ROCM_PATH=str(rocm))

    work = root / "target" / "pm-gfx942-moe-grouped-twin"
    work.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(work)

    co = args.out_dir / f"{STEM}_hipcc_{ARCH}.co"
    hxaco = args.out_dir / f"{STEM}_pm_{ARCH}.hxaco"
    manifest = args.out_dir / f"{STEM}_pm_{ARCH}.twin.json"
    flags = core_hipcc_flags(ARCH)

    with tempfile.TemporaryDirectory(prefix="run-", dir=work) as tmp:
        tmp_co = Path(tmp) / f"anchor_{ARCH}.co"
        run([str(hipcc), *flags, "-o", str(tmp_co), SOURCE_REL], cwd=root, env=env)
        shutil.copyfile(tmp_co, co)

        report_path = Path(tmp) / f"report_{ARCH}.json"
        kernel_args = [a for s in SYMBOLS for a in ("--kernel", s)]
        run([*seal_cmd(args), "--in", str(co), "--arch", ARCH, "--out", str(hxaco),
             "--report", str(report_path), *kernel_args], cwd=root, env=env)
        reports = json.loads(report_path.read_text())
        if not isinstance(reports, list) or [r.get("kernel") for r in reports] != list(SYMBOLS):
            die(f"seal report is not the per-symbol array for {list(SYMBOLS)}")

        doc = {
            "module": f"{STEM}_pm_{ARCH}",
            "arch": ARCH,
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
            "hxaco": {"file": hxaco.name, "bytes": hxaco.stat().st_size,
                      "sha256": sha256_file(hxaco)},
            "seal": {
                "tool": "peacemaker-isa example hip_twin (gfx942 Route A: typed lift, "
                        "full device-ELF identity, bundle)",
                "reports": reports,
            },
        }
        # A hardware receipt qualifies both exact input files, not just the PM image.
        # Keep it only while both image hashes match; otherwise the CPU audit fails closed.
        new_sha = doc["hxaco"]["sha256"]
        if manifest.exists():
            try:
                old = json.loads(manifest.read_text())
            except ValueError:
                old = {}
            rec = old.get("gpu_oracle")
            if (
                old.get("hxaco", {}).get("sha256") == new_sha
                and isinstance(rec, dict)
                and rec.get("input_hxaco_sha256") == new_sha
                and rec.get("hipcc_image_sha256") == doc["anchor"]["sha256"]
            ):
                doc["gpu_oracle"] = rec
        manifest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    def rel(p):
        return p.relative_to(root).as_posix() if p.is_relative_to(root) else p.name

    print(f"{ARCH}: wrote {rel(co)} {rel(hxaco)} {rel(manifest)}")


if __name__ == "__main__":
    main()
