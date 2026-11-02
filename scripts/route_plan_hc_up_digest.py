#!/usr/bin/env python3
"""Refresh the `hyper_read_up_wmma_gfx1201` rows (ar + native_mtp) of the three gfx1201 Flash-Next
route plans: the symbol list (the retiled `_t128` entry and the persistent `_v2` entry the
default route can launch) and the sha256 of kernels/src/hyper_read_up_wmma.gfx1201.hip (the
module's effective source). Re-runnable after any edit of that .hip.
Run from the repo root: python3 scripts/route_plan_hc_up_digest.py"""
import glob, hashlib, pathlib

MODULE = "hyper_read_up_wmma_gfx1201"
SYMBOLS = ",".join(
    sorted(["hyper_read_up_wmma_bf16_gfx1201_t128", "hyper_read_up_wmma_bf16_gfx1201_v2"])
)
digest = hashlib.sha256(
    pathlib.Path("kernels/src/hyper_read_up_wmma.gfx1201.hip").read_bytes()
).hexdigest()
for path in sorted(glob.glob("crates/rdna-compute/src/route_plans/*.gfx1201.tsv")):
    out, seen = [], set()
    for line in pathlib.Path(path).read_text().splitlines(keepends=True):
        f = line.rstrip("\n").split("\t")
        if len(f) == 5 and f[1] == "hip" and f[2] == MODULE:
            line = "\t".join([f[0], "hip", MODULE, SYMBOLS, digest]) + "\n"
            seen.add(f[0])
        out.append(line)
    pathlib.Path(path).write_text("".join(out))
    print(path, sorted(seen), digest)
