# Stage B2 receipt — fp8 QK leg on route N (f16 PV via temporary V-expand)

Base: `311a03698` (B1b). Plan §3 (sq folding), §4 (scale ownership), §14.3.
GPU: ordinal 1 (ab1). Device work in wt-fa2a warm tree (release).

## Change (one file: `kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip`)

New `fa2_stageb_nbody<PARTIAL>` (separate template; f16 objects untouched):

- Init: Ofr/m/l as before; `qb` (Q-code row base) + per-tile `qs_tile`
  (`sq[query]*scale_attn`; sq at codes+batch*24*256; OOR query → sq=1).
- Bounds/tile loop/completion identical to the f16 body (same 258-records).
- Fill: B1b loops (proven identical textually), but ROLLED t-loops in the
  body copy (unrolled scheduling overlap held ~80 VGPR live across copies;
  rolling serializes; dynamic work identical + latches). Oracle copy kept
  unrolled.
- QK fused per subtile (one sacc, not four): per (sub,dg,u) stream Q frag
  (b64, OOR lanes clamp `qb` to row 0 — valid; mask discards them downstream,
  proven safe) + K frag (b64, base VGPR + immediate: `kbn+dc*128`, the
  slice-A idiom; dynamic indexing cost ~100 packets in index math + spill
  reloads) + `wmma_f32_16x16x16_fp8_fp8` (64 dynamic per full tile).
  Per-dg scheduling barrier kept.
- Score: `(sacc[j]*qs_tile)*sc[j]` at the old scale position; `sc[8]` =
  OWNED keys' sk via `fa2_scale_n` (never ml), shared file with sv.
- PV still f16 (temporary, B3 deletes): V-expand at operand load (HW
  `cvt_pk_f32_fp8` ×4 + `sv` mul + f16 pack; bit-exact Q0 values since f32
  mul commutes). P/softmax/O bookkeeping unchanged.
- Direct/partial entries call the body (dump retired; dump template kept
  dead for B3-oracle reuse — templates instantiate nothing unreferenced).
- New typedefs `v2i_t`/`float2_t` (types only, no object impact).

VGPR journey (all direct-entry numbers): split-sacc 256+37spill →
fused-sub-outer 254/0 → +streamed/no-arrays 252 → +rolled-fill-tloops
208/7waves → +static-K-index/clamp/rolled-dg 209/7waves. Two rules learned:
(1) unrolled slot copies overlap in scheduling (~80 VGPR) — roll them;
(2) dynamic LDS indexing defeats the addr folder and spills (~100
packets) — base VGPR + immediates, the slice-A idiom.

## Gate 1 — K0/K3/K8-Q0 objects bit-identical to 311a03698: PASS

Same-filename `--genco` object `cmp` ×3.

## Gate 2 — resources (K8+FP8 TU): PASS

direct 209 / partial 209 / fp8-preconvert 56 / merge 17 VGPR; 0 spill V/S,
0 scratch; 7 waves/SIMD (exceeds the 6-wave class).

## Gate 3 — ISA (direct entry): PASS

- 64 fp8-QK WMMA dynamic per full tile (4 static × 16 trips: 4 subs × 4
  dg, both rolled; 16 f16 PV WMMA static likewise).
- Zero fp8→f16 conversions anywhere (64 `cvt_pk_f32_fp8` all in PV-expand,
  0 in the QK-dg span; 16 f16→f32 sk/sv widens, allowed).
- QK packets per KT64: dg-iter 26 static (4 Q-b64 + 2 K-pair-b64 + 4 WMMA
  + addr/waits/latch) × 16 trips = **416 vs the 472 cap** (12% margin).
  K loads use `ds_load_2addr_stride64_b64` pairs where adjacent.
- LDS still 32768 (no scale plane, no second tile); QK barriers kept.

## Gate 4 — layer-35 score/O compare (real taps): PASS

Throwaway `crates/rdna-compute/examples/tmp_b2_qk_compare.rs` (UNTRACKED):
native KV via production writer, f32 Q, batch 384 × 4224 rows; S3 Q0
launcher (control) + S3 fp8 launcher (B2), vs `o_kern_final.f32` (arm1).

- Q0-vs-arm1: 1.012 / 4.78e-3 / tail 2.01e-2 (control reproduces Screen
  arm3's band 0.796/1.94e-2 within writer-encode systematics; vehicle valid).
- B2-vs-arm1: **1.913 / 7.07e-3 / tail 3.08e-2** ≈ Screen arm5 (stage-b
  route-N host model: 1.915/3.21e-2). No nonfinite.
- B2-vs-Q0 (QK-leg isolation; PV bit-exact by construction):
  1.915 / 4.84e-3 / tail 1.84e-2.
- No NaN/structural garbage ⇒ WMMA orientation, sk ownership (owned keys,
  never ml), qs folding position, and V-expand all correct. Bit-identical
  across all post-compare restructures (static-index/clamp/roll are pure).

## Gate 5 — 1-chunk eval on the route: PARTIAL (dispatch gap, not kernel)

`eval_hipfire` with the flag on still routes prefill FA2 through the Q0
launcher (0.031193, kldseq md5 identical to Q0) — nothing in production
calls S3's stage-b launcher yet (only throwaways); wiring eval→stage-b is
dispatch (S6) scope, not this file. Route proof instead: bare symbols
`attention_fp8_e4m3_fa2_gqa_gfx1201.*.hsaco` (+preconvert via S3's
pre-launch) JIT-compiled and launched on device through the committed S3
launcher with correct numerics (above); flag-off fail-loud preserved
(HostGate-tested).

## Explicitly NOT gated here (B3/B4)

fp8 PV leg (V-expand is temporary), preconvert numerics beyond S3 smoke,
partial/merge production paths, any timing claim.
