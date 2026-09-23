# AddEpilogue: gfx1151 V2B ADD epilogue (stack-1 port)

**Verdict: LAND (exact).** Kill switch `HIPFIRE_V2B_ADDEPI=0`.
Production commit `96cd40e27` on stack-1 `8397738d4`; this directory holds the
evidence. Halo pp8192 rocprof, same binaries with the lever off (A) vs on (B):
**−106.8 ms GPU (−1.36%)**, wall 7918.9 → 7810.6 ms (**1034.5 → 1048.8 t/s**).
WT2 c24 q8/q8 is **byte-identical** off vs on (0.075973).

## Lever

The V2B ADD epilogue's residual read-modify-write stalls on Halo DRAM because
every CTA in a round reaches the epilogue together. Two exact changes:

- **Fold (out-projections, DN + FA, 64 per pass).** When the next residual
  reader is the FFN RMSNorm on the IU4 route (`residual_fold_consumer_ready`),
  the out-projection arms `pbs.gate_ffn_batch` as a delta buffer. That buffer
  is dead from the out-projection until the gate/up GEMM runs after the norm.
  The V2B GEMM then writes SET into it and records `residual_fold_pending`.
  `fused_rmsnorm_mq_rotate_awq_i4_fold` computes RN(x + delta) in both phases,
  writes x back, and emits the same block_i4_128 as the plain norm. The GEMM
  no longer reads the residual; the norm pays for an extra delta read and an
  x write instead.
- **Touch (all other V2B ADDs, here w_down K17408).** `_add_touch` loads one
  dword per 64-byte line of the tile's residual over epochs E−16..E−9, ahead
  of the staging packet, so the epilogue reads the residual from the
  memory-side cache instead of DRAM. The output bytes match `_add`.
- **Safety.** A pending fold is landed with `add_inplace_f32` (same RN add) by
  any other reader or writer before it acts:
  - the non-IU4 norm route (`try_iu4_rmsnorm_prepared`);
  - the f16 gate/up routes;
  - every IU4 GEMM entry;
  - the arm call itself;
  - the F1-lite gate/up entry. This one is new in the port: F1's `h` is
    `gate_ffn_batch`, the same buffer as the delta.
- **Scope.** The change lives inside the V2B block (`arch == "gfx1151"`). On
  gfx1100 the arm is never taken and every flush is a no-op, so gfx1100 is
  unchanged by construction and got no proof run.
- **Kernel ISA.** The V2B body is templated `<EPI, TOUCH>`. The set, add and
  gate_up_silu entries disassemble byte-identically to stack-1 on gfx1151 and
  gfx1100; only the address offsets differ.

## Evidence (Halo, both cards quiet, Stacker2 lease)

| check | result |
|---|---|
| pp8192 trace A/B (`trace/`, `trace/gfx1151-8192-delta.txt`) | GPU 7868.4 → 7761.6 ms; wall 7918.9 → 7810.6 ms |
| out-proj ADD → SET + fold | ADD 9.570 ms → SET ≈7.10 ms; norm 1.173 → 2.711 ms; ≈ −0.93 ms/call × 64 |
| w_down ADD → touch | 21.795 → 21.073 ms; −0.72 ms/call × 64 |
| WT2 c24 q8/q8 off/on (`gfx1151/wt2/`) | 0.075973 / 0.075973, kldseq sha256 79bbff9e… identical. The fold norm was JIT-compiled only in the B home, so it fired (n_ctx 2048 is V2B-eligible) |
| J2: JIT objects on (B cache) vs off (A cache) | 346,030,080 values compared, 0 differing (ADD touch, ADD, SET+fold residual, fold i4, base-norm i4; K6144/K17408, N8192) |
| J1: JIT objects vs hipcc stack-1 reference | 0 differing |
| P1: hipcc objects from this tree vs hipcc stack-1 reference | 0 differing |

`logs/superseded/J1-*`: the first hipcc reference was built without the
runtime's gfx11 define `-DIU4_A4_CANDIDATES=2` (`feature_flags.rs`), so
`block_i4_128_quant.hip` fell back to 8 candidates. About 22% of the i4 bytes
then differed, equally for the base and fold norms, while the residuals
matched. That was a defect in the reference build, not in the lever.
`build_ref.sh` and `build_prod.sh` now pass the define.

Standalone JIT-object timing (median of 7 with a cache flush before each,
F/R order):

| N | shape | ADD + norm | SET + fold norm | change | touch vs ADD |
|---|---|---|---|---|---|
| 8192 | K6144 (out-proj) | 10.96 / 11.00 ms | 9.79 / 10.12 ms | −0.88 to −1.17 ms | 9.00 vs 9.64 ms |
| 8192 | K17408 (w_down) | – | – | – | 20.50 vs 20.65 ms (F), 20.92 vs 21.60 ms (R) |

**Short-prefill caveat (hipcc objects, N512).** At N512 the fold loses 0.03 to
0.04 ms per out-projection (−4 to −5% of the pair). That is about +2 ms per
512-token chunk, roughly 0.4% of pp512. Touch is neutral at N512
(0.576 vs 0.576 to 0.593 ms). The crossover between N512 and N8192 has not
been measured. A batch threshold on `residual_fold_consumer_ready` is a
possible follow-up; any threshold is still exact.

Pre-crash run of the same lever on the 2af base (`precrash-2af-gfx1151-8192-delta.txt`):
GPU −72.0 ms (−0.88%), wall 8188.3 → 8119.2 ms. That standalone oracle also
passed (0 differing).

## Reproduce

On hipx, in a worktree at this commit with release binaries in
`target/release`, run `scratch-addepi/proof.sh trace jit standalone wt2`.
