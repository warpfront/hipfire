# Raster + cpol sweep — KILL (no ship)

Worktree `wt-raster` (branch `gfx1201-gemm-raster` off `mq4-lloyd` 1f947d0f9).
GPU ordinal 3 / ab3. Kernel file reverted to HEAD; this dir is the full record.
Parametrized kernel preserved as `kernel_tested_base0.hip` (defaults G0/cpol0 =
HEAD behavior; oracle PASS).

## Verdict

* Raster: real oracle-clean −10.7% gate-GEMM win (G8T0), residual flat — but
  end-to-end transfer is +0.3…+1.0% (steady state), below the pp8192 ≥ +1.5% ×
  both-pairs ship bar for BOTH g4t0 and g16t0. NO SHIP.
* Cpol: all KILL. A-load variants neutral (±0.7%, one +1.4% regression);
  EVERY W-load buffer-policy variant catastrophic (+51% … +312%), including
  the U5 aux-20 recipe (+95%): it does not replicate on the double-buffered
  kernel. Default global loads stay.
* Decode: bimodal system noise in P1 (29.3 vs 36.5 clusters, cold start);
  P2–P4 decode paired within 0.1% on all arms.

## Rig table (kx3_foldfree_probe, median of 20, A/B/A vs base.bin)

baseline: gate 6647–6690 us, residual 3007–3040 us (noise ±0.07%)

| variant | oracle 9/9 | gate us (Δ%) | residual us (Δ%) |
|---|---|---|---|
| G2T1 | PASS | 6254.8 (−5.93%) | 2982.4 (−1.03%) |
| G4T1 | PASS | 6145.4 (−7.57%) | 3020.0 (+0.28%) |
| G8T1 | PASS | 6077.5 (−8.54%) | 3034.5 (+0.75%) |
| G16T1 | PASS | 6052.6 (−8.99%) | 3032.0 (+0.68%) |
| G2T0 | PASS | 5976.0 (−10.03%) | 2992.9 (−0.83%) |
| G4T0 | PASS | 5958.9 (−10.43%) | 2992.5 (−0.74%) |
| G8T0 | PASS | 5951.6 (−10.71%) | 3022.4 (+0.08%) |
| G16T0 | PASS | 5946.3 (−10.66%) | 3021.3 (+0.06%) |
| A_RT+DEV(16) | PASS | 6640.3 (−0.27%) | 2997.0 (−0.61%) |
| A_NT(1) | PASS | 6705.2 (+0.36%) | 3004.5 (−0.71%) |
| A_NT+DEV(17) | PASS | 6768.2 (+1.42%) | 3021.8 (−0.37%) |
| A_NT_RT+DEV(20) | PASS | 6659.9 (−0.32%) | 3007.6 (−0.74%) |
| W_RT+DEV(16) | PASS | 10136 (+51.8%) | 4583 (+51.2%) |
| W_NT(1) | PASS | 25895 (+288%) | 12421 (+310%) |
| W_NT+DEV(17) | PASS | 25889 (+288%) | 12463 (+312%) |
| W_NT_RT+DEV(20) | PASS | 12958 (+94.6%) | 6332 (+109%) |
| W_HT+DEV(18) | PASS | 10186 (+52.9%) | 4582 (+51.5%) |
| W_NT_HT+DEV(22) | PASS | 12954 (+94.4%) | 6416 (+113%) |

T=1: G consecutive token-tiles share a row window (W-slab reuse).
T=0: G consecutive row-tiles share a token window (A-slab reuse). G16T0
residual (40 row tiles % 16) correctly falls back to direct map (measures
baseline — proves the divisibility guard). Swizzle confirmed live in ISA
(modulo-via-RCP + guard branch in full_add prologue); G2T1 needed the guard
after oracle caught n80cols mis-coverage pre-guard.

## Paired matrix (`hipfire bench qwen3.8:27b-mq4-xt --matrix
## --pp 512,2048,8192,32768 --ctx 128 --tg 64 --spec off --runs 3
## --warmups 1 --kv-mode fp8 --json`, per-arm HIPFIRE_DAEMON_BIN)

arms/md5: daemon_base 6ccbc121 / daemon_g4t0 c3255556 / daemon_g16t0 78ffeb54
(hipfire_base 9e01d8eb / hipfire_g4t0 dcb85d8e / hipfire_g16t0 bdb4b824)

| pair | pp512 | pp2048 | pp8192 | pp32768 | decode paired |
|---|---|---|---|---|---|
| P1 g4t0 | +1.19% | +1.80% | +1.51% | +1.49% | NO (+24%, cold) |
| P3 g4t0 | +0.46% | +0.42% | +0.31% | +0.53% | yes (−0.09%) |
| P2 g16t0 | −0.05% | −0.39% | −0.43% | −0.12% | yes (−0.02%) |
| P4 g16t0 | +1.05% | +1.17% | +1.04% | −0.22% | yes (−0.09%) |

baselines pp8192 across pairs: 2360.8 / 2385.3 / 2381.5 / 2354.8 (±1% drift —
larger than candidate effects). Raw JSON: p1/p2/p3/p4_{base,g4t0,g16t0}.json.

## Paired ttft (fixed 8 KB prompt, max-tokens 8, runs 3)

* base 1092.7ms vs g4t0 1064.6ms (−2.57%, first-run cold) 
* base 1071.4 vs g16t0 1073.5 (+0.20%)
* base 1074.6 vs g4t0 1073.3 (−0.12%)
* base 1075.8 vs g16t0 1074.7 (−0.10%)
Steady state: flat. Log: ttft.pairs.log.

## Why the kernel win doesn't transfer

Rig gate GEMM ≈ 6.6 ms; end-to-end pp8192 prefill ≈ 3.4 s over 64 layers ×
(gate+up+down+qkv+o+attention). Gate GEMMs are ~0.4% of prefill → −10.7%
predicts +0.4–1.0% end-to-end. Observed exactly that. The ship bar (+1.5%)
is unreachable for any GEMM-only lever at this fraction — arithmetic, not noise.

## Launch-site map (for any future re-landing)

The remap lived in the shared gfx12 body
`gemm_mq4g256v2_residual_mmq_iu4_gfx12_body`, so ALL gfx1201 iu4 entries
inherit it: generic `gemm_mq4g256v2_residual_mmq_iu4`, `full_add`, `full_set`
via `gemm_mq4g256v2_mmq_prequant_iu4` (gemm.rs:19572) and its
set/add wrappers. Routes covered: gate_up (`..._iu4_prepared`:31075), qkv
(`gemm_qkv_hfq4g256_wmma_gfx12_mq4v2`:29581), qkvza (10164/29111/29290),
residual/down + small-tail (`..._small_tail_set_iu4`:29074). Untouched by
design: gfx1151/gfx1100 col/occ3/lf16 symbols (separate entries), DS/metadata
and Y traffic (always plain global), fp8 kernels.

## Aux map (empirical, gfx1201 buffer_load, for the record)

0=default(RT) 1=NT 2=HT 3=LU 4=NT_RT 5=RT_NT; +16=SCOPE_DEV; +32=nv.
E.g. 16=DEV, 17=NT+DEV, 18=HT+DEV, 19=LU+DEV, 20=NT_RT+DEV, 22=NT_HT+DEV.
No SYS/SE scope encoding exists for buffer loads.
