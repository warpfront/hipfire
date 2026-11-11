#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Generate the Route-A PeaceMaker twin of the gfx942 Qwen4 MoE glue kernels.

Explicit invocation only: no test, build script or runtime path runs this. It

  1. compiles `kernels/src/qwen4_g942_glue_moe.hip` (or the file given with `--source PATH`,
     e.g. one owned by another worktree) with the runtime's `core_hipcc_flags`
     (`--genco --offload-arch=gfx942 -O3 --no-offload-compress -fuse-cuid=none
     -mcode-object-version=6`, `rdna-compute/src/compiler.rs`) into the anchor
     `kernels/qwen4_g942_glue_moe_hipcc_gfx942.co`. The source bytes must match the pinned
     SOURCE_SHA256 and the compiled anchor must match the pinned ANCHOR_SHA256 (fail closed);
  2. seals the whole module with `peacemaker-isa`'s `hip_twin` example (gfx942 Route A: typed
     lift, checked byte identity of the whole device ELF, bundle, sha-pin) into
     `kernels/qwen4_g942_glue_moe_pm_gfx942.hxaco` (+ `.hxaco.sha256`), one bundle holding
     `qwen4_g942_moe_group_scatter_desc`, `qwen4_g942_moe_unscatter_act` and
     `qwen4_g942_moe_combine_shared_add`;
  3. writes `kernels/qwen4_g942_glue_moe_pm_gfx942.twin.json` (source sha256, compiler
     provenance, portable flags, the full seal report of each sealed symbol). No host path is
     recorded: the source is named by its repo-relative path only. An existing `gpu_oracle`
     receipt is carried over only while it is bound to the exact hxaco, hipcc image and both
     device-ELF hashes.

Tests (`crates/peacemaker-isa/tests/hip_twin_gfx942_glue_moe.rs`) only consume the committed
files.

    scripts/pm-gfx942-glue-moe-twin.py [--source PATH] [--rocm-path DIR]
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
    "qwen4_g942_moe_group_scatter_desc",
    "qwen4_g942_moe_unscatter_act",
    "qwen4_g942_moe_combine_shared_add",
)
SOURCE_REL = "kernels/src/qwen4_g942_glue_moe.hip"
STEM = "qwen4_g942_glue_moe"
SOURCE_SHA256 = "0f2e37e06e386ceaf9c95defeda3e7cd7d73355bf4367fb96fb39108ca0eb2bc"
ANCHOR_SHA256 = "e3304e667978ca5c8b32c9cd6b2d8a1580a1d6121e9327bd1bfd36afe39ff4c1"


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
    print(f"pm-gfx942-glue-moe-twin: {msg}", file=sys.stderr)
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
    ap.add_argument("--source", type=Path,
                    help=f"glue source (default: <repo>/{SOURCE_REL}; required when absent)")
    ap.add_argument("--rocm-path", help="ROCm root (default: $ROCM_PATH, $HIP_PATH, /opt/rocm, PATH)")
    ap.add_argument("--build-wrapper", default=None,
                    help="command prefix for the cargo invocation, e.g. a build-slot script")
    ap.add_argument("--seal-bin", help="prebuilt hip_twin example binary (skips cargo)")
    ap.add_argument("--out-dir", type=Path, help="artifact directory (default: <repo>/kernels)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.source is not None:
        source = args.source.resolve()
    elif (root / SOURCE_REL).is_file():
        source = root / SOURCE_REL
    else:
        die(f"{SOURCE_REL} not found under {root}: pass --source PATH")
    if not source.is_file():
        die(f"source {source} is not a file")
    src_sha = sha256_file(source)
    if src_sha != SOURCE_SHA256:
        die(f"source sha256 {src_sha} != pinned {SOURCE_SHA256}")
    args.out_dir = (args.out_dir or root / "kernels").resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rocm = find_rocm(args.rocm_path)
    hipcc = rocm / "bin" / "hipcc"
    env = dict(os.environ, ROCM_PATH=str(rocm))

    work = root / "target" / "pm-gfx942-glue-moe-twin"
    work.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(work)

    co = args.out_dir / f"{STEM}_hipcc_{ARCH}.co"
    hxaco = args.out_dir / f"{STEM}_pm_{ARCH}.hxaco"
    sha_file = args.out_dir / f"{STEM}_pm_{ARCH}.hxaco.sha256"
    manifest = args.out_dir / f"{STEM}_pm_{ARCH}.twin.json"
    flags = core_hipcc_flags(ARCH)

    with tempfile.TemporaryDirectory(prefix="run-", dir=work) as tmp:
        tmp_co = Path(tmp) / f"anchor_{ARCH}.co"
        run([str(hipcc), *flags, "-o", str(tmp_co), str(source)], cwd=root, env=env)
        anchor_sha = sha256_file(tmp_co)
        if anchor_sha != ANCHOR_SHA256:
            die(f"compiled anchor sha256 {anchor_sha} != pinned {ANCHOR_SHA256}")
        shutil.copyfile(tmp_co, co)

        report_path = Path(tmp) / f"report_{ARCH}.json"
        kernel_args = [a for s in SYMBOLS for a in ("--kernel", s)]
        run([*seal_cmd(args), "--in", str(co), "--arch", ARCH, "--out", str(hxaco),
             "--report", str(report_path), *kernel_args], cwd=root, env=env)
        reports = json.loads(report_path.read_text())
        if not isinstance(reports, list) or [r.get("kernel") for r in reports] != list(SYMBOLS):
            die(f"seal report is not the per-symbol array for {list(SYMBOLS)}")

        new_sha = sha256_file(hxaco)
        sha_file.write_text(new_sha + "\n")
        doc = {
            "module": f"{STEM}_pm_{ARCH}",
            "arch": ARCH,
            "symbols": list(SYMBOLS),
            "source": {
                "path": SOURCE_REL,
                "bytes": source.stat().st_size,
                "sha256": src_sha,
            },
            "compiler": {
                "version": compiler_version(hipcc, env),
                "flags": flags,
                "invocation": "hipcc <flags> -o <anchor.co> " + SOURCE_REL,
            },
            "anchor": {"file": co.name, "bytes": co.stat().st_size, "sha256": sha256_file(co)},
            "hxaco": {"file": hxaco.name, "bytes": hxaco.stat().st_size, "sha256": new_sha},
            "seal": {
                "tool": "peacemaker-isa example hip_twin (gfx942 Route A: typed lift, "
                        "full device-ELF identity, bundle)",
                "reports": reports,
            },
        }
        # A hardware receipt qualifies both exact input files and the device ELF they share.
        # Keep it only while every bound hash matches; otherwise the CPU audit fails closed.
        elf_sha = reports[0].get("elf_sha256")
        if manifest.exists():
            try:
                old = json.loads(manifest.read_text())
            except ValueError:
                old = {}
            rec = old.get("gpu_oracle")
            if (
                isinstance(rec, dict)
                and old.get("hxaco", {}).get("sha256") == new_sha
                and rec.get("input_hxaco_sha256") == new_sha
                and rec.get("hipcc_image_sha256") == doc["anchor"]["sha256"]
                and rec.get("pm_device_elf_sha256") == elf_sha
                and rec.get("hipcc_device_elf_sha256") == elf_sha
            ):
                doc["gpu_oracle"] = rec
        manifest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    def rel(p):
        return p.relative_to(root).as_posix() if p.is_relative_to(root) else p.name

    print(f"{ARCH}: wrote {rel(co)} {rel(hxaco)} {rel(sha_file)} {rel(manifest)}")


if __name__ == "__main__":
    main()
