#!/usr/bin/env python3
"""Census one steady K128 step of the shipped iu4 full-set kernel."""
from collections import Counter
from pathlib import Path
import sys

lines = Path(sys.argv[1]).read_text().splitlines()

def first(prefix, start=0):
    return next(i for i, line in enumerate(lines[start:], start) if line.startswith(prefix))

def logical_valu(segment):
    out = []
    for line in segment:
        op = line.strip()
        if op.startswith("v_") and not op.startswith("v_wmma"):
            out.extend(part.strip().split()[0] for part in op.split("::"))
    return out

kernel = first("gemm_mq4g256v2_residual_mmq_iu4_full_set:")
fold_label = first(".LBB3_9:", kernel)
body_label = first(".LBB3_10:", fold_label)
first_only_label = first("; %bb.13:", body_label)
body_end = first("\ts_cbranch_vccnz .LBB3_9", body_label)
first_only_end = first("\ts_branch .LBB3_9", first_only_label)

# The loop enters LBB3_10 once, then alternates LBB3_9 (fold prior tile) and
# LBB3_10 (next tile). The bb.13 setup runs only before the first fold.
fold = lines[fold_label + 1 : body_label]
body = lines[body_label + 1 : body_end + 1]
first_only = lines[first_only_label : first_only_end + 1]
fold_valu = logical_valu(fold)
body_valu = logical_valu(body)
first_valu = logical_valu(first_only)
wmma = sum("v_wmma_i32_16x16x32_iu4" in line for line in body)
b64 = sum("ds_load_2addr" in line and "_b64" in line for line in body)
b32 = sum("ds_load_2addr" in line and "_b32" in line for line in body)
g64 = sum("global_load_b64" in line for line in body)
g32 = sum("global_load_b32" in line for line in body)

assert (len(fold_valu), len(body_valu), len(first_valu), wmma) == (192, 43, 10, 32)
assert (b64, b32, g64, g32) == (12, 10, 8, 2)
print(f"steady WMMA/K128: {wmma}")
print(f"steady logical VALU/K128: {len(body_valu) + len(fold_valu)} "
      f"(body {len(body_valu)}, fold {len(fold_valu)})")
print(f"steady logical VALU/WMMA: {(len(body_valu) + len(fold_valu)) / wmma:.5f}")
print(f"first-step-only logical VALU: {len(first_valu)}")
print("fold VALU classes:", dict(sorted(Counter(fold_valu).items())))
print(f"operand LDS reads/WMMA: {2 * b64 / wmma:.5f} "
      f"({b64} encoded dual-address b64 loads)")
print(f"fold-metadata LDS reads/WMMA: {2 * b32 / wmma:.5f} "
      f"({b32} encoded dual-address b32 loads)")
print(f"all logical LDS reads/WMMA: {2 * (b64 + b32) / wmma:.5f}")
print(f"global loads/K128/lane: {g64 + g32} ({g64} b64 payload, {g32} b32 metadata)")
