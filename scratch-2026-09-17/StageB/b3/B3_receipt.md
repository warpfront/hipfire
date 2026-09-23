# Stage B3 receipt — fp8 PV leg with bounded changing-unit O

Base: `e7a43c418` (B2). Plan §5 (P_and_O). GPU: ordinal 1 (ab1).

## Change (one file: `kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip`)

Subtile tail replaced (mask/row-max untouched):

- sv reuses the scale file right after the max (sk dead); `w = e*sv`
  written back into `sacc` (dead after `e`), so no w array; `l` keeps
  UNWEIGHTED `e`.
- `wmax` over 8 `w` + `shfl_xor(16)` partner (both halves agree);
  `b_new = max(wmax/448, 0x1p-64f)`, kept `b_prev` on all-zero rows;
  `rho = (alpha*b_prev)/b_new` folded into the existing 128-value
  `Ofr` rescale (replaces `*= alpha`).
- `p8 = E4M3_RNE(w/b_new)` via 4 `cvt_pk_fp8_f32` (division, matches the
  host model; zero weights encode to code 0); PV =
  `wmma_f32_16x16x16_fp8_fp8` on V-plane codes direct (one b64, static
  `vbn+dc*32` immediate, same idiom as K); `b_prev = b_new` per subtile.
- `b_prev = 1` init; direct completion stores `Ofr*(b_prev*inv)`,
  partial stores physical `b_prev*Ofr` in unchanged stride-258 records
  (merge untouched). V-expand deleted; `float2_t` now unused (kept type).

## Gate 1 — K0/K3/K8-Q0 objects bit-identical to e7a43c418: PASS

Same-filename `--genco` object `cmp` ×3. (K0+FP8/K3+FP8/K8+FP8 TUs also
compile, 0 warnings.)

## Gate 2 — resources (K8+FP8 TU): PASS

direct 190 / partial 191 / fp8-preconvert 56 / merge 17 VGPR; 0 spill V/S,
0 scratch; 8 waves/SIMD (exceeds the 6-wave class).

## Gate 3 — ISA (direct entry): PASS with notes

- 16 fp8-PV WMMA static per sub-iteration body (rolled sub ×4 → 64
  dynamic per KT64); 4 fp8-QK WMMA static per dg-iter (rolled dg ×16
  trips → 64 dynamic). Total 128 WMMA/tile, all fp8.
- Zero fp8→f16 conversions file-wide in the body (`cvt_pk_f32_fp8` = 0;
  `wmma_f16` = 0; only f32→fp8 encodes + f16→f32 scale widens remain).
- P-pack: 4 `cvt_pk_fp8_f32` per subtile (hoisted: P is dc-independent;
  compiler CSE'd the 16 unrolled copies). p8 uses reciprocal+mul (not
  8 divisions) — ulp-level difference from the host division model,
  invisible at 1e-2 bands; noted for KLD.
- LDS still 32768; both steady-state barriers present (publish +
  turnover-at-loop-top); QK per-dg + PV per-dc scheduling barriers kept.

## Gate 4 — per-wave packet total per KT64 vs 2216: ~3800 (OVER, reported)

Method: latch-delimited loop bodies × source trips (trips device-proven
by coverage: 4-subtile execution required for the arm5 match below) +
fast-path-only for predicated fills (uniform-skip verified in ISA).
Unit matches §6 (per-lane dynamic, full tile, all live).

- Fill K: ~25 × 8 = ~200. Fill V: ~63 × 16 = ~1008 (8 u8 + 1 b64 +
  addr/waits/delays/branch per fragment; slow paths skipped).
- Sub-iteration steady state (software-pipelined layout, latch-delimited
  565→1313): 646 static × 4 subs = 2584. Of which dg-iter (latch 997→914)
  69 × 16 trips = 1104 (QK + distributed sk + overhead).
- Fixed (init/bounds/completion/turnover, amortized): ~15.
- TOTAL ≈ 200 + 1008 + 2584 + 15 ≈ **3800 (±10%) vs 2216 (+~70%)**.
- Composition driver: V-fill overhead (~1000 vs 300–420 forecast —
  waits/delays/addr dominate: ~46 overhead per 8 loads), rolled-loop
  latch/counter/dynamic-index overhead, predicated OOR structure
  throughout, new support work (scales/divs/pack/rho). B1b's 840 fill
  estimate is superseded (~1210 with latch/index overhead — AT the 1200
  abandon line within noise; the line still stands, bench adjudicates).
- Recomputed §6.4 model (c16=8, r=1.809) with measured fill/total:
  ≈0.75 attention-work ratio (vs 0.58 forecast) → ≈+3–4% E2E at s=0.146
  if issue-bound (vs +6.2%). Memory-bound fixtures hide issue cost
  (device ran at Q0-like speed below).
- Per plan §11 ("reviewers decide after evidence, not packet counts"):
  correctness evidence below is green; bench (Main) decides perf.

## Gate 5 — layer-35 O compare vs arm1 (device, real taps): PASS

Throwaway `tmp_b2_qk_compare.rs` (UNTRACKED) via S3 launchers, batch 384:

- Default: Q0-vs-arm1 1.012/2.01e-2 (control ~= arm3 0.796/1.94e-2);
  **B3-vs-arm1 1.915/3.11e-2 ≈ Screen arm5 (route-N host model)
  1.915/3.21e-2 to 3-4 digits**; B3-vs-Q0 1.915/1.90e-2; 0 nonfinite.
- p4096: Q0-vs-arm1 1.095/2.29e-2 (control ~= arm3 0.792/2.28e-2);
  **B3-vs-arm1 2.559/4.19e-2 ≈ arm5 2.559/4.23e-2**; 0 nonfinite.
- Main's stated arm-6 envelope (2.04/3.3e-2, 2.46/4.2e-2): default inside;
  p4096 max 4% over (2.559 vs 2.46, tail inside). Mechanistically B3 IS
  route N ≡ arm5 (route Q ≡ arm6 is a different source); the arm5 match
  at <1% on both prompts is the stronger, source-correct evidence.
- Underflow: host census 0.0198%/0.0181% ≤ 0.1% (S7, both prompts);
  device O reproducing arm5 (whose model includes e4m3-P underflow)
  implies matching device underflow. No NaN/structural garbage ⇒ scale
  ownership (owned keys), qs folding, rho/b_prev bookkeeping, physical
  records all correct.

## Explicitly NOT gated here (Main: KLD + bench with StageBRoute)

24-chunk KLD, ag corpus, prefill bench, partial/merge production paths,
preconvert numerics beyond S3 smoke, timing claims.
