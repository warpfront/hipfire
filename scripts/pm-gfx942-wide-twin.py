#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Seal the immutable owner-provided gfx942 P1b wide anchor, without recompiling it.

The source and complete input bundle are pinned to owner commit 17183b9b89.
The owner compiled the module from the exact source bytes under the runtime
basename gemm_bf16_mfma_wide_gfx942.hip with the recorded flags below. The
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
STEM = "gemm_bf16_mfma_wide_gfx942"
SOURCE_REL = "kernels/src/gemm_bf16_mfma_wide.gfx942.hip"
OWNER_COMMIT = "17183b9b89"
SOURCE_SHA = "49f356151cf2fac0c31e6e965bf7cf7e5a9968e2c8970e90f691e4acac512567"
ANCHOR_SHA = "5ecf282d0a3129cc9cfc0d4268677bc1b67e11454931818ba808a33a30061bcc"
ELF_SHA = "141276895e3dd53cb51746e7fb8f761d2e29929dc36147c719d0fe8355dec1e6"
SYMBOLS = (
    "gemm_bf16_mfma_gfx942_wide64",
    "gemm_bf16_mfma_gfx942_wide64_m32",
    "gemm_bf16_mfma_gfx942_wide64_m64",
)
FLAGS = [
    "--genco", "--offload-arch=gfx942", "-O3", "--no-offload-compress",
    "-fuse-cuid=none", "-mcode-object-version=6", "--rocm-path=<rocm>",
    "--hip-path=<rocm>", "-I<rocm>/include", "-fmacro-prefix-map=<source_dir>/=",
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
    work = root / "target" / "pm-gfx942-wide-twin"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="seal-", dir=work) as temp:
        report = Path(temp) / "report.json"
        kernel_args = [arg for symbol in SYMBOLS for arg in ("--kernel", symbol)]
        subprocess.run([*command, "--in", str(co), "--arch", ARCH, "--out", str(hxaco),
                        "--report", str(report), *kernel_args], cwd=root, check=True)
        reports = json.loads(report.read_text())
    if not isinstance(reports, list) or [r.get("kernel") for r in reports] != list(SYMBOLS):
        raise RuntimeError("seal report does not cover all three entries in order")
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
            "flags": FLAGS, "runtime_source_basename": STEM + ".hip",
            "provenance": "Owner's immutable final anchor (17183b9b89); this recipe does not recompile it",
            "invocation": "hipcc <flags> -o <anchor.co> <source_dir>/" + STEM + ".hip",
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
    print(f"{ARCH}: sealed all three exact P1b entries into {hxaco}")


if __name__ == "__main__":
    main()
