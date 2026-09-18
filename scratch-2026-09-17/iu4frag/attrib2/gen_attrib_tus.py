#!/usr/bin/env python3
"""Generate 8 kernel-only ablation TUs from the current iu4 kernel (timing only).

Usage: gen_attrib_tus.py <kernel.hip> <prelude.hip> <outdir>
Writes tu-1-full.hip .. tu-8-singlebuf.hip (prelude + variant kernel).
Variants (all timing-only unless noted):
  1 full       - unmodified copy (reference).
  2 no-stage   - all staging loads/stores after the prologue guarded by
                 if(false); barriers + w_lds re-points stay.
  3 no-fold    - the per-half fold nest guarded by if(false).
  4 no-compute - the 4 iu4_bundle calls guarded by if(false).
  5 no-store   - the coalesced store nest guarded by if(false).
  6 no-barrier - every __syncthreads() replaced by ((void)0).
  7 sync-stage - prefetch/publish flow replaced by direct synchronous
                 staging (same op counts, zero overlap; see NOTES below).
  8 singlebuf  - W1 re-point replaced by W0 (slab1 publishes to W0).
"""
import re, sys

kpath, ppath, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
pre = open(ppath).read()
ker = open(kpath).read()

def brace_end(src, open_idx):
    d = 0
    for i in range(open_idx, len(src)):
        if src[i] == '{': d += 1
        elif src[i] == '}':
            d -= 1
            if d == 0: return i
    raise ValueError("unbalanced")

def guard_region(src, start_anchor, end_anchor_incl, tag):
    i = src.index(start_anchor)
    j = src.index(end_anchor_incl, i) + len(end_anchor_incl)
    # extend j past trailing newline
    if j < len(src) and src[j] == '\n': j += 1
    seg = src[i:j]
    return src[:i] + f"if (false) {{ // ATTRIB-{tag}\n" + seg + "} // ATTRIB-" + tag + "\n" + src[j:]

def guard_lines_with(src, substr, tag):
    out = []
    for ln in src.splitlines(keepends=True):
        if substr in ln and 'ATTRIB' not in ln:
            indent = ln[:len(ln) - len(ln.lstrip())]
            out.append(f"{indent}if (false) // ATTRIB-{tag}\n{indent}    {ln.lstrip()}")
        else:
            out.append(ln)
    return ''.join(out)

# --- variant 2: no-stage ---
v2 = ker
# A: slab-1 prefetch loop
v2 = guard_region(v2,
    "                #pragma unroll\n                for (int r = 0; r < 2; r++) {\n                    const u32x2_t av = *(const u32x2_t*)(xqB1 + st_avoff[r]);",
    "                    W_pf[r] = *(const u32x2_t*)(wB1 + st_wvoff[r]);\n                }",
    "no-stage-prefetch")
# B: publish slab1 + refill next A0
v2 = guard_region(v2,
    "                #pragma unroll\n                for (int r = 0; r < 2; r++) {\n                    *(u32x2_t*)(a_lds + st_ldsoff[r]) = A_pf[r];",
    "                        A_pf[r] = st_aok[r] ? av : z64;\n                    }\n                }",
    "no-stage-publish")
# C: next-block W0 staging loop
v2 = guard_region(v2,
    "                    #pragma unroll\n                    for (int r = 0; r < 2; r++) {\n                        *(u32x2_t*)(w_lds + st_ldsoff[r]) = *(const u32x2_t*)(wB0n + st_wvoff[r]);",
    "                        *(u32x2_t*)(w_lds + st_ldsoff[r]) = *(const u32x2_t*)(wB0n + st_wvoff[r]);\n                    }",
    "no-stage-w0")
b2a_anchor = "                    #pragma unroll\n                    for (int r = 0; r < 2; r++) {\n                        *(u32x2_t*)(a_lds + st_ldsoff[r]) = A_pf[r];"
assert v2.count(b2a_anchor) == 1, v2.count(b2a_anchor)
v2 = guard_region(v2, b2a_anchor,
    "                        *(u32x2_t*)(a_lds + st_ldsoff[r]) = A_pf[r];\n                    }",
    "no-stage-b2a")

# --- variant 3: no-fold ---
fold_anchor = "                #pragma unroll\n                for (int rg = 0; rg < 2; rg++) {\n                    float sc_row[8];"
assert ker.count(fold_anchor) == 1
fi = ker.index(fold_anchor)
fj = brace_end(ker, ker.index('{', fi)) + 1
v3 = ker[:fi] + "if (false) { // ATTRIB-no-fold\n" + ker[fi:fj] + "} // ATTRIB-no-fold\n" + ker[fj:]

# --- variant 4: no-compute (+ opaque cacc init: defined runtime-varying
# values via a volatile load, so folds cannot constant-collapse and no
# LLVM-undef/poison flows anywhere; Sol review) ---
v4 = ker
for sb_first in ["iu4_bundle<0, true>(w_lds, a_lds, frW, frA, cacc, zc);",
                 "iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);",
                 "iu4_bundle<0, false>(w_lds, a_lds, frW, frA, cacc, zc);"]:
    n = v4.count(sb_first)
    assert n >= 1, sb_first
    v4 = v4.replace(sb_first, "if (false) " + sb_first + " // ATTRIB-no-compute")
v4_opaque = """
    { // ATTRIB-nocompute-opaque (no WMMA anywhere below).
        int opaque = ((const volatile int*)A)[tid & 15];
        #pragma unroll
        for (int inb = 0; inb < 4; inb++)
            #pragma unroll
            for (int irg = 0; irg < 2; irg++)
                cacc[inb][irg] = (int32x8_t){opaque, opaque + 1, opaque + 2, opaque + 3, opaque + 4, opaque + 5, opaque + 6, opaque + 7};
    }"""
assert v4.count("    int32x8_t cacc[4][2];") == 1
v4 = v4.replace("    int32x8_t cacc[4][2];",
    "    int32x8_t cacc[4][2];" + v4_opaque, 1)

# --- variant 5: no-store ---
store_anchor = "    float* slots = (float*)LDS;"
assert ker.count(store_anchor) == 1
si = ker.index(store_anchor)
sj = brace_end(ker, ker.index('{', ker.index('for (int rg = 0; rg < 2; rg++) {', si))) + 1
v5 = ker[:si] + "if (false) { // ATTRIB-no-store\n    " + ker[si:sj] + "} // ATTRIB-no-store\n" + ker[sj:]
# --- variant 6: no-barrier ---
nbar = ker.count("__syncthreads();")
assert nbar > 0, "no barriers found"
v6 = ker.replace("__syncthreads();", "((void)0); // ATTRIB-no-barrier (was __syncthreads)")

# --- variant 7: sync staging ---
# Prefetch deleted; slab1 is staged directly AFTER slab-0 compute (sunk
# below the overlap window, just before B0), next-A0 directly next to the
# next-W0 staging. Same loads/stores/barriers, zero overlap.
v7 = ker
old_prefetch = """                const char* __restrict__ xqB1 = (const char*)(Xq + (long long)kb * N) + 8 + 32;
                const char* __restrict__ wB1 = A + (long long)g * 136 + 8 + (long long)h * 64 + 32;
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    const u32x2_t av = *(const u32x2_t*)(xqB1 + st_avoff[r]);
                    A_pf[r] = st_aok[r] ? av : z64;
                    W_pf[r] = *(const u32x2_t*)(wB1 + st_wvoff[r]);
                }"""
assert v7.count(old_prefetch) == 1
v7 = v7.replace(old_prefetch, "                // ATTRIB-sync: no prefetch (slab1 staged directly below).")
# sunk direct-stage before B0
b0_anchor = """                iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);
                // B0:"""
assert v7.count(b0_anchor) == 1
sunk_stage1 = """                iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);
                // ATTRIB-sync: slab1 staged directly (no overlap with compute above).
                w_lds = LDS + IU4_W1_OFF;
                {
                const char* __restrict__ xqB1 = (const char*)(Xq + (long long)kb * N) + 8 + 32;
                const char* __restrict__ wB1 = A + (long long)g * 136 + 8 + (long long)h * 64 + 32;
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    const u32x2_t av = *(const u32x2_t*)(xqB1 + st_avoff[r]);
                    *(u32x2_t*)(a_lds + st_ldsoff[r]) = st_aok[r] ? av : z64;
                    *(u32x2_t*)(w_lds + st_ldsoff[r]) = *(const u32x2_t*)(wB1 + st_wvoff[r]);
                }
                }
                // B0:"""
v7 = v7.replace(b0_anchor, sunk_stage1)
# B: publish + refill deleted (already staged); keep nothing (re-point moved)
old_publish = """                w_lds = LDS + IU4_W1_OFF;
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    *(u32x2_t*)(a_lds + st_ldsoff[r]) = A_pf[r];
                    *(u32x2_t*)(w_lds + st_ldsoff[r]) = W_pf[r];
                }
                if (!last) {
                    // Next block's A slab 0 (from Xq; only kb_n0 needed).
                    const int kb_n0 = kb + 1;
                    const char* __restrict__ xqBn0 = (const char*)(Xq + (long long)kb_n0 * N) + 8;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        const u32x2_t av = *(const u32x2_t*)(xqBn0 + st_avoff[r]);
                        A_pf[r] = st_aok[r] ? av : z64;
                    }
                }"""
assert v7.count(old_publish) == 1, v7.count(old_publish)
v7 = v7.replace(old_publish, "                // ATTRIB-sync: slab1 + next-A0 already staged directly; no publish/refill.")
# C: after W0 staging loop, add direct next-A0 staging (uses kb_n0 bases)
old_w0 = """                    w_lds = LDS + IU4_W0_OFF;
                    const char* __restrict__ wB0n = A + (long long)g_n * 136 + 8 + (long long)h_n * 64;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        *(u32x2_t*)(w_lds + st_ldsoff[r]) = *(const u32x2_t*)(wB0n + st_wvoff[r]);
                    }"""
new_w0 = """                    w_lds = LDS + IU4_W0_OFF;
                    const char* __restrict__ wB0n = A + (long long)g_n * 136 + 8 + (long long)h_n * 64;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        *(u32x2_t*)(w_lds + st_ldsoff[r]) = *(const u32x2_t*)(wB0n + st_wvoff[r]);
                    }
                    if (!last) {
                    // ATTRIB-sync: next block's A slab 0 staged directly.
                    const int kb_n0 = kb + 1;
                    const char* __restrict__ xqBn0 = (const char*)(Xq + (long long)kb_n0 * N) + 8;
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        const u32x2_t av = *(const u32x2_t*)(xqBn0 + st_avoff[r]);
                        *(u32x2_t*)(a_lds + st_ldsoff[r]) = st_aok[r] ? av : z64;
                    }
                    }"""
old_b2a = """                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        *(u32x2_t*)(a_lds + st_ldsoff[r]) = A_pf[r];
                    }"""
assert v7.count(old_w0) == 1
v7 = v7.replace(old_w0, new_w0)
assert v7.count(old_b2a) == 1
v7 = v7.replace(old_b2a, "                    // ATTRIB-sync: no B2a publish (staged directly).")
# volatile sinks: keep the WMMA chain live where the consumer is guarded.
# (Sol DCE review: checksums over ALL elements; single-element sinks let
# most WMMA chains die. v3 sink sits inside the trip since cacc restarts
# from zero every block; v5 sink sits after the loops since acc carries.)
v3 = v3.replace("} // ATTRIB-no-fold\n",
    "} // ATTRIB-no-fold\n    { int attrib_sink = 0;\n#pragma unroll\n    for (int snb = 0; snb < 4; snb++)\n#pragma unroll\n    for (int srg = 0; srg < 2; srg++)\n#pragma unroll\n    for (int sj = 0; sj < 8; sj++)\n    attrib_sink ^= ((const int*)&cacc[snb][srg])[sj];\n    ((volatile int*)Y)[tid] = attrib_sink; } // ATTRIB-sink3\n")
v5 = v5.replace("if (false) { // ATTRIB-no-store\n",
    "{ float attrib_sinkf = 0.0f;\n#pragma unroll\n    for (int snb = 0; snb < 4; snb++)\n#pragma unroll\n    for (int srg = 0; srg < 2; srg++)\n#pragma unroll\n    for (int sj = 0; sj < 8; sj++)\n    attrib_sinkf += acc[snb][srg][sj];\n    ((volatile float*)Y)[tid] = attrib_sinkf; } // ATTRIB-sink5\nif (false) { // ATTRIB-no-store\n")
assert ker.count("w_lds = LDS + IU4_W1_OFF;") == 1
v8 = ker.replace("w_lds = LDS + IU4_W1_OFF;", "w_lds = LDS + IU4_W0_OFF; // ATTRIB-singlebuf (was W1)")

variants = {"1-full": ker, "2-nostage": v2, "3-nofold": v3, "4-nocompute": v4,
            "5-nostore": v5, "6-nobarrier": v6, "7-syncstage": v7, "8-singlebuf": v8}
import os
os.makedirs(outdir, exist_ok=True)
for tag, k in variants.items():
    open(f"{outdir}/tu-{tag}.hip", "w").write(pre + k)
    print(f"wrote tu-{tag}.hip barriers={k.count('__syncthreads();')} bundles={k.count('iu4_bundle<')}")
