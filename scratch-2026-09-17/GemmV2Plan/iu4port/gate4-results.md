# iu4port gate 4 — TIME N=512, same-session s2bt8 control (ordinal 1)
# iu4 v2 (this branch): tmp_iu4_gfx12_oracle TIME=1, 3 runs (run1 in gate4-time.log)
# s2bt8 control: wt-gemmv2 tmp_gemm_v2_oracle --device 0 --quick, V2=0,
#   built with redirected CARGO_TARGET_DIR (read-only source), out dirs fp8ctl-n512[-quick].
# Both isolated-GEMM medians (pack excluded both sides).

## Medians (us/call)
| row | shape (M,K,N) | iu4 v2 r1 | iu4 v2 r2 | iu4 v2 r3 | s2bt8 same-session | hist s2bt8 |
|---|---|---|---|---|---|---|
| gate | 17408,5120,512 set | 733.4 | 739.9 | 740.8 | 1970.8 (gate_up, 2 matrices) | 1804.7 |
| gate add | 17408,5120,512 add | 741.3 | 747.3 | 748.1 | — | — |
| residual | 5120,17408,512 add | 715.6 | 716.1 | 720.7 | 1007.2 | 676.8 |
| qkvza | 16480,5120,512 set | 711.1 | 712.1 | 714.4 | 892.0 | 825.7 |
| qkv | 14336,5120,512 set | 608.8 | 614.2 | 610.8 | 821.0 | 770.1 |

Stability: all iu4 rows within ~1.0% across 3 runs (< 1.5% noise bar).

## Ratios vs same-session s2bt8 (admission ≤ 0.50 every row)
| row | ratio | verdict |
|---|---|---|
| gate_up (raw: 1 iu4 matrix vs 2-matrix fp8 fused) | 737/1971 = 0.37 | misleading (see below) |
| gate_up per-matrix (2x iu4 calls vs fused fp8) | 1474/1971 = 0.75 | FAIL |
| residual | 718/1007 = 0.71 | FAIL |
| qkvza | 713/892 = 0.80 | FAIL |
| qkv | 610/821 = 0.74 | FAIL |

Vs iu4 weight-tile baseline (same session, 1122/1247/1244/1006/917): 0.58-0.71x —
the staged tile is a large, real win over v1. Vs fp8 s2bt8: first iu4 build to beat
the incumbent on any row on this card (per-matrix 0.71-0.80), but no row clears 0.50.

Caveats: iu4 shapes use the residual-form symbol on all shapes (no fused gate_up/
qkvza/qkv iu4 kernels exist); fp8 rows are family-fused kernels. Pack/quantize cost
excluded both sides (fp8 pack med 28.6 us; iu4 int4 quantize not timed here).
Effective throughput ~121-128 TOPS (~23% of the 539.7 K32 peak).

Gate 4 verdict: FAIL vs the 0.50 admission target. No gate-5 phase (per assignment).
Sweep projection (BN=256/BM=64, same wave shape): traffic 367→~280 MB on residual
(~0.76x time → ~0.54 ratio), still short of 0.50 — recorded, not executed;
proceed only on explicit approval.
