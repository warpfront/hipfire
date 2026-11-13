#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Seal the immutable owner-provided gfx942 qsa-row anchor, without recompiling it.

The source and complete input bundle are pinned to owner commit dc33131250.
The owner compiled the module from the exact source bytes under the runtime
basename qwen4_g942_qsa_row_attention.gfx942.hip with the recorded flags below. The
committed anchor is the reproducible input, not a substitute compiler result.

Explicit invocation only; tests consume committed artifacts. Supply either
--seal-bin (prebuilt hip_twin) or --build-wrapper (cargo build-slot prefix).
No runtime dispatch, kernel source, launcher, or anchor instruction is changed.
"""
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

ARCH = "gfx942"
STEM = "qwen4_g942_qsa_row_attention"
SOURCE_REL = "kernels/src/qwen4_g942_qsa_row_attention.gfx942.hip"
OWNER_COMMIT = "dc33131250"
SOURCE_SHA = "e95e554a7f939feaa2d9a80393edb9a4d1df0596149afb3b4cc54e9599a06b6e"
ANCHOR_SHA = "bbc9fbcfe6a2cd5ebc830fc82ffc9a6a360d7efe3c15c5824d93eda5513e4d9c"
ELF_SHA = "1caa4bd326be42faeab2f09fc43ff7a5b46a63e3a29cbe0f552bd44a88443b0a"
SYMBOLS = (
    "qwen4_g942_qsa_row_f32",
    "qwen4_g942_qsa_row_fp8_fnuz",
)
FLAGS = [
    "--genco", "--offload-arch=gfx942", "-O3", "--no-offload-compress",
    "-fuse-cuid=none", "-mcode-object-version=6",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    compiler = parser.add_mutually_exclusive_group(required=True)
    compiler.add_argument("--seal-bin", type=Path, help="prebuilt hip_twin example")
    compiler.add_argument("--build-wrapper", help="cargo command prefix, e.g. build-slot.sh")
    parser.add_argument("--anchor", type=Path, help="exact pinned input bundle (default: committed anchor)")
    parser.add_argument("--out-dir", type=Path, help="artifact directory (default: kernels)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source = root / SOURCE_REL
    anchor = (args.anchor or root / "kernels" / f"{STEM}_hipcc_{ARCH}.co").resolve()
    output = (args.out_dir or root / "kernels").resolve()
    if digest(source) != SOURCE_SHA:
        parser.error(f"{SOURCE_REL} differs from approved owner {OWNER_COMMIT}; obtain a new final anchor")
    if digest(anchor) != ANCHOR_SHA:
        parser.error("anchor differs from the approved full owner bundle")
    output.mkdir(parents=True, exist_ok=True)
    co = output / f"{STEM}_hipcc_{ARCH}.co"
    hxaco = output / f"{STEM}_pm_{ARCH}.hxaco"
    manifest = output / f"{STEM}_pm_{ARCH}.twin.json"
    if co.resolve() != anchor:
        shutil.copyfile(anchor, co)
    if args.seal_bin:
        command = [str(args.seal_bin.resolve())]
    else:
        command = [*shlex.split(args.build_wrapper), "cargo", "run", "--release", "-p",
                   "peacemaker-isa", "--features", "lab", "--example", "hip_twin", "--"]
    work = root / "target" / "pm-gfx942-qsa-row-twin"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="seal-", dir=work) as temp:
        report = Path(temp) / "report.json"
        kernel_args = [arg for symbol in SYMBOLS for arg in ("--kernel", symbol)]
        subprocess.run([*command, "--in", str(co), "--arch", ARCH, "--out", str(hxaco),
                        "--report", str(report), *kernel_args], cwd=root, check=True)
        reports = json.loads(report.read_text())
    if isinstance(reports, dict):
        reports = [reports]
    if not isinstance(reports, list) or [r.get("kernel") for r in reports] != list(SYMBOLS):
        raise RuntimeError("seal report does not cover the two qsa-row entries")
    for report in reports:
        hazards = report["hazards"]
        if (report["elf_sha256"] != ELF_SHA or hazards["checked"] is not True
                or any(hazards[k] != 0 for k in ("findings", "blocking", "bypasses"))
                or any(hazards["hazard"][k] != 0 for k in ("inserted_nops", "rewritten_instructions", "violations"))):
            raise RuntimeError("not an exact, fully certified owner image")
    if digest(source) != SOURCE_SHA or digest(co) != ANCHOR_SHA:
        raise RuntimeError("pinned inputs changed while sealing")
    doc = {
        "module": f"{STEM}_pm_{ARCH}", "arch": ARCH, "symbols": list(SYMBOLS),
        "source": {"path": SOURCE_REL, "bytes": source.stat().st_size,
                   "sha256": SOURCE_SHA, "owner_commit": OWNER_COMMIT},
        "compiler": {
            "flags": FLAGS, "runtime_source_basename": "qwen4_g942_qsa_row_attention.gfx942.hip",
            "provenance": "Owner's immutable final anchor (dc33131250); this recipe does not recompile it",
            "invocation": "hipcc <flags> -o <anchor.co> <source_dir>/qwen4_g942_qsa_row_attention.gfx942.hip",
        },
        "anchor": {"file": co.name, "bytes": co.stat().st_size,
                   "sha256": ANCHOR_SHA, "device_elf_sha256": ELF_SHA},
        "hxaco": {"file": hxaco.name, "bytes": hxaco.stat().st_size, "sha256": digest(hxaco)},
        "seal": {"tool": "peacemaker-isa example hip_twin (gfx942 Route A: typed lift, full device-ELF identity, bundle)",
                 "reports": reports},
    }
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        receipt = previous.get("gpu_oracle")
        if (isinstance(receipt, dict)
                and receipt.get("input_hxaco_sha256") == doc["hxaco"]["sha256"]
                and receipt.get("hipcc_image_sha256") == ANCHOR_SHA
                and receipt.get("pm_device_elf_sha256") == ELF_SHA
                and receipt.get("hipcc_device_elf_sha256") == ELF_SHA):
            doc["gpu_oracle"] = receipt
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"{ARCH}: sealed the exact two qsa-row entries into {hxaco}")


if __name__ == "__main__":
    main()
