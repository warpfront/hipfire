#!/usr/bin/env python3
"""Gate-3 byte-identity check: fixture vs study artifact (slice A).

Compares tensor index (names/qt/shape/gs/len), byte-compares every
non-qt44 payload and every AWQ sidecar, and reports qt44 census +
header/code relationship. Exits nonzero on any non-qt44/sidecar mismatch.
Usage: check_bytes.py <fixture.hfq> <artifact.hfq> [label]
"""
import sys

sys.path.insert(0, "/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a")
from hfq_util import Hfq, QT44

FIX = sys.argv[1]
ART = sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else ART

fx = Hfq(FIX)
ax = Hfq(ART)
print(f"fixture: {FIX}\n  arch={fx.arch} tensors={len(fx.tensors)}")
print(f"artifact: {ART}\n  arch={ax.arch} tensors={len(ax.tensors)}")
ok = True
if fx.arch != ax.arch:
    print("MISMATCH arch"); ok = False

fn = fx.by_name()
an = ax.by_name()
if set(fn) != set(an):
    print(f"MISMATCH tensor-name sets: fixture-only={len(set(fn)-set(an))} artifact-only={len(set(an)-set(fn))}")
    for n in sorted(set(fn) ^ set(an))[:20]:
        print("   ", n)
    ok = False
else:
    print(f"tensor names identical ({len(fn)})")
    if list(fn) != list(an):
        print("NOTE: order differs (harmless)")

n_qt44 = n_side = n_other = 0
bad_other = []
bad_side = []
for name, ft in fn.items():
    at = an[name]
    for k in ("qt", "shape", "gs", "len"):
        if ft[k] != at[k]:
            print(f"MISMATCH {name} {k}: {ft[k]} vs {at[k]}"); ok = False
    if ft["qt"] == QT44:
        n_qt44 += 1
        continue
    fb = fx.payload(ft)
    ab = ax.payload(at)
    if name.endswith(".awq_scale.weight"):
        n_side += 1
        if bytes(fb) != bytes(ab):
            bad_side.append(name)
    else:
        n_other += 1
        if bytes(fb) != bytes(ab):
            bad_other.append(name)

print(f"qt44 tensors: {n_qt44}")
print(f"AWQ sidecars: {n_side}, mismatched: {len(bad_side)}")
for n in bad_side[:10]:
    print("   SIDE-MISMATCH", n)
print(f"other non-qt44: {n_other}, mismatched: {len(bad_other)}")
for n in bad_other[:10]:
    print("   OTHER-MISMATCH", n)
if bad_side or bad_other:
    ok = False

# qt44 header relationship: same scales? (C0 keeps base scales; zeros/codes move)
import numpy as np

same_sc = same_z = same_q = tot = 0
for name, ft in fn.items():
    if ft["qt"] != QT44:
        continue
    at = an[name]
    fb = fx.payload(ft)
    ab = ax.payload(at)
    if len(fb) != len(ab):
        print(f"MISMATCH qt44 len {name}"); ok = False; continue
    ng = len(fb) // 136
    tot += ng
    a = np.frombuffer(fb, dtype=np.uint8).reshape(ng, 136)
    b = np.frombuffer(ab, dtype=np.uint8).reshape(ng, 136)
    same_sc += int(((a[:, 0:2] == b[:, 0:2]) & (a[:, 4:6] == b[:, 4:6])).sum())
    same_z += int(((a[:, 2:4] == b[:, 2:4]) & (a[:, 6:8] == b[:, 6:8])).sum())
    same_q += int((a[:, 8:] == b[:, 8:]).sum())
print(f"qt44 groups: {tot}; headers: scales-identical {same_sc} ({100*same_sc/tot:.2f}%), "
      f"zeros-identical {same_z} ({100*same_z/tot:.2f}%), payload-bytes-identical {same_q} ({100*same_q/(tot*128):.2f}%)")

print("metadata fixture:", {k: fx.meta.get(k) for k in ("hipfire_base_format", "hipfire_product_tier")})
print("metadata artifact:", {k: ax.meta.get(k) for k in ("hipfire_base_format", "hipfire_product_tier", "hipfire_mq4e8_study")})
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
