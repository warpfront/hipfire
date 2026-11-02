#!/usr/bin/env python3
"""Refresh the `qwen4_hc_down_fused` rows (ar + native_mtp) of the three gfx1201 Flash-Next
route plans with the sha256 of kernels/src/qwen4_hc_down_fused.hip (the module's effective
source). Inserts the rows before `qwen4_hc_streams_init_from_embed_batched` when absent.
Run from the repo root: python3 scripts/route_plan_hc_down_digest.py"""
import glob, hashlib, pathlib

MODULE, SYMBOL = "qwen4_hc_down_fused", "qwen4_hc_down_fused_b64_k32"
digest = hashlib.sha256(pathlib.Path(f"kernels/src/{MODULE}.hip").read_bytes()).hexdigest()
for path in sorted(glob.glob("crates/rdna-compute/src/route_plans/*.gfx1201.tsv")):
    out, seen = [], set()
    for line in pathlib.Path(path).read_text().splitlines(keepends=True):
        f = line.rstrip("\n").split("\t")
        if len(f) == 5 and f[1] == "hip" and f[2] == MODULE:
            line = "\t".join([f[0], "hip", MODULE, SYMBOL, digest]) + "\n"
            seen.add(f[0])
        elif len(f) == 5 and f[1] == "hip" and f[2] == "qwen4_hc_streams_init_from_embed_batched" and f[0] not in seen:
            out.append("\t".join([f[0], "hip", MODULE, SYMBOL, digest]) + "\n")
            seen.add(f[0])
        out.append(line)
    pathlib.Path(path).write_text("".join(out))
    print(path, sorted(seen), digest)
