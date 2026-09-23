#!/usr/bin/env python3
"""VOPD-aware fold census for the iu4 gfx12 kernel.

Usage: vopd_census.py <disasm.txt> <symbol> [<symbol2> ...]

Per symbol body:
  - fold dual halves: v_dual_(mul|fmac|add|fmamk|fmaak)_f32 occurrences
    (each half = 1 scalar fold op)
  - fold singletons: plain v_(cvt_f32_i32|cvt_f32_u32|mul|fmac|fma|add)_f32
  - all v_dual_ lines (any kind) for context
  - FOLD packets = singleton lines + lines containing >=1 fold half
  - FOLD scalar ops = singleton ops + fold halves
"""
import re, sys

path = sys.argv[1]
syms = sys.argv[2:]
lines = open(path).read().splitlines()

SUFF = r'(?:_e32|_e64)?(?![0-9a-z_])'
FOLD_SINGLE = re.compile(
    r'(?<![0-9a-z_])v_(cvt_f32_i32|cvt_f32_u32|mul_f32|fmac_f32|fma_f32|add_f32)'
    + SUFF)
FOLD_HALF = re.compile(
    r'v_dual_(mul_f32|fmac_f32|add_f32|fmamk_f32|fmaak_f32)' + SUFF)
ANY_DUAL = re.compile(r'(?<![0-9a-z_])v_dual_[a-z0-9_]+')
WMMA = re.compile(r'(?<![0-9a-z_])v_wmma_')


def carve(sym):
    start = next(i for i, l in enumerate(lines) if "<%s>:" % sym in l)
    end = next((i for i, l in enumerate(lines)
                if i > start and re.match(r'^[0-9a-f]+ <.*>:', l)), len(lines))
    return lines[start:end]


for sym in syms:
    body = carve(sym)
    code = [l.strip().split('//')[0] for l in body if re.match(r'\s+[a-z]', l)]
    n_dual_lines = sum(1 for s in code if ANY_DUAL.search(s))
    halves = []
    for s in code:
        halves.extend(m.group(1) for m in FOLD_HALF.finditer(s))
    hkind = {}
    for h in halves:
        hkind[h] = hkind.get(h, 0) + 1
    singles = []
    for s in code:
        if ANY_DUAL.search(s):
            continue
        singles.extend(m.group(1) for m in FOLD_SINGLE.finditer(s))
    skind = {}
    for op in singles:
        skind[op] = skind.get(op, 0) + 1
    fold_lines = sum(1 for s in code
                     if FOLD_HALF.search(s)
                     or (FOLD_SINGLE.search(s) and not ANY_DUAL.search(s)))
    wmma = sum(1 for s in code if WMMA.search(s))
    print("== %s ==" % sym)
    print("  instr lines       = %d" % len(code))
    print("  any v_dual_ lines = %d" % n_dual_lines)
    print("  fold dual halves  = %d %s" % (len(halves), hkind))
    print("  fold singletons   = %d %s" % (len(singles), skind))
    print("  FOLD packets      = %d   (singleton lines + lines w/ fold half)"
          % fold_lines)
    print("  FOLD scalar ops   = %d   (singletons + fold halves)"
          % (len(singles) + len(halves)))
    print("  wmma instr        = %d" % wmma)
