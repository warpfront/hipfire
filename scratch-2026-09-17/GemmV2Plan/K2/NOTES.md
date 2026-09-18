# K2: v2 tile limiters — coalesced store + staging overlap (gfx1201, 2026-09-18)

Branch `gfx12-fp8-v2-tile`, worktree `wt-gemmv2`. Only file changed:
`kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip` (V2_TILE branch only;
non-V2 kernels and the shared expand helper untouched — all diff hunks within
HEAD lines 391-567, current-file V2 region 207-649). No host/prelude changes;
dynamic LDS stays 24576 in all four `gemm.rs` v2 launch arms.

## Method

Throwaway harness `crates/saddle-lab/examples/tmp_gemm_v2_k2bench.rs` (DELETED
before commit; no Cargo.toml change — autodiscovery) drives the four canonical
N=512 shapes through the PRODUCTION launchers (same four calls as the oracle's
incumbent path): gate_up 17408/17408 K5120, qkv 12288/1024/1024 K5120, qkvza
10240/6144/48/48 K5120, residual 5120 K17408. V2 selected purely by
`HIPFIRE_GFX12_MQ4V2_FP8_V2=1`. Synth weights/X identical salts to
`tmp_gemm_v2_oracle` (disjoint-halves per split, `synth_x(0xBEEF)`). Per family:
2 untimed launches (fnv1a64 hash + determinism) + 11 timed launches under
`rdna_compute::profile` (median). Box: ordinal 0 (`ROCR_VISIBLE_DEVICES=0`,
`HOME=/home/kaden/.hipfire-homes/ab0`,
`HIPFIRE_KERNEL_CACHE=<ab0home>/.hipfire_kernels` — literal path; the runtime
appends `/gfx1201`. NOTE: `$HOME` inside the same `export` line expands to the
OLD home; early runs wrote blobs to `/home/kaden/.hipfire_kernels` — harmless,
content-keyed, hashes matched the validator's V/ pins exactly).

Baselines reproduce the validator: v2 hashes `2adf7620c1d722e1 /
09726b10ba1eb1b2 / 082a7b24def26f75 / 373cb087fb519b9e` identical to
`wt-lloyd .../GemmV2Plan/V/oracle-n512-v2/results.json`.

## Gate 1 — bit-identity (all pass)

Every variant below is `det=true` AND hash-equal to the v2 baseline it was
measured against. Canonical N=512 plus N-tails (N=256 exact-1-tile, N=320
half-empty second batch tile) plus the qkvza row-tail (16480 = 257.5×BN64)
and 48-row beta/alpha source-boundary crossings:

| fam | N=512 | N=256 | N=320 |
| gate_up | 2adf7620c1d722e1 | a9535728ff0174c1 | b02a9dd0ab2c91b5 |
| qkv | 09726b10ba1eb1b2 | d20c97dbe2188241 | d2ec9220f240eff2 |
| qkvza | 082a7b24def26f75 | c76fb47144786358 | 34b8568668b8440c |
| residual | 373cb087fb519b9e | dba99241d9e569bb | 96baa3225992466e |

(base / +store / +overlap all three produce these hashes at every N.)

## Change 2 — coalesced epilogue store (COMMIT 1)

Old: lane (ml,kg) stored T[nb][wf][j] directly → one instruction touched 16
output rows at 4 B each. New: per nb, lanes dump 16 values into an LDS tile
(row ml, skewed cols `(wf*16+8*kg+j+ml)&31` — without the skew all 16 ml lanes
hit one bank), barrier, then 32 lanes stream 16 token rows with lane tid
owning column tid: every global store is one contiguous 128 B segment.
Per-element arithmetic unchanged (`y = a[t]*T`, one residual add) → only visit
order changes. A/W/S/SZ dead post-fold; tile = 8×16×32×4 B = 16 KiB at LDS
base, no host change.

Ablation finding (temp `#define`, reverted): gate_up no-drain 2483 vs full
2478 — the old "store phase" was frontend/issue-bound (address/branch soup),
not DRAM-bound; the rewrite removes that overhead. Residual no-drain saved
~197 us (RMW traffic is real there).

## Change 1 — staging/compute overlap (COMMIT 2)

Spec asked for BK32 double-buffer (a) or A-direct (b). Measured instead the
overlap mechanism directly with a cheaper, strictly-better variant: prologue
stages slab 0; each iteration issues slab s+1's A/W GLOBAL LOADS into registers
(`A_pf` 8×v2i, `Wpk_pf` 2×u32) BEFORE slab s's compute (pinned by
`sched_barrier`), publishes (expand + LDS write) after compute under an added
publish barrier. Single LDS buffer retained; compute section byte-identical;
K-loop 0-VALU contract kept.

- (a) is bounded above by this probe: identical overlap (same thread executes
  staging either way — overlap hides latency, never execution), plus 2× slabs,
  2× barriers, stride-40 bank re-enumeration, 27 KiB LDS, host LDS change.
- (b) rejected analytically: A is ~74% of staging bytes (1.4 GB of 1.9 GB per
  gate_up launch); per-wave A-direct doubles it (two tok_part-sharing waves
  reload separately) — the staged plane's 16x token-fragment reuse (plan §2)
  is destroyed for no LDS saving that matters.

Race found & fixed during probe: publish without barrier lets fast waves
clobber slab s under stragglers (det=false, faster garbage) — publish barrier
added; det=true restored.

ISA verified (gate_up probe blob `ec33fab06ae33f09`, unbundled+objdump):
8×`global_load_b64` prefetch-A + pack b32s sit between the prologue barrier
and the first `v_wmma_f32_16x16x16` (line ~409-447 vs first WMMA line 482);
64 static WMMAs, 22 barriers. Prologue A row was compiler-widened to
4×`global_load_b128` — follow-up: widen prefetch loads the same way in source.

## Gate 2 — metadata (all pass, 0 spill/scratch throughout)

| kernel | variant | VGPR | SGPR | spill | scratch | LDS |
| gate_up v2 | base | 179 | 38 | 0/0 | 0 | 24576 |
| gate_up v2 | +store | 179 | 38 | 0/0 | 0 | 24576 |
| gate_up v2 | +overlap | 192 | 33 | 0/0 | 0 | 24576 |
| qkv v2 | +overlap | 196 | 38 | 0/0 | 0 | 24576 |
| qkvza v2 | +overlap | 196 | 38 | 0/0 | 0 | 24576 |
| residual v2 | +overlap | 192 | 29 | 0/0 | 0 | 24576 |

wave32 everywhere. 24 KiB → 2 WG/CU retained (no host change). VGPR 192-196:
within the 256 hard ceiling with 0 spills (qkv/qkvza 4 over the 192 target —
allocator choice around the live prefetch registers; no action).
Grid unchanged (e.g. gate_up [544,2]) → occupancy limited by LDS/WG count,
not block count.

## Gate 3 — TIME, same session, N=512 medians over 11 reps (us)

| fam | incumbent s2bt8 | v2 base | +store | +overlap | ratio Over/Inc | ≤0.60? |
| gate_up | 1964.3 | 2678.2 | 2440.3 | 1998.5 | 1.017 | NO |
| qkv | 767.4 | 1105.1 | 938.5 | 759.1 | 0.989 | NO |
| qkvza | 896.3 | 1292.2 | 1212.0 | 960.2 | 1.071 | NO |
| residual | 995.0 | 1489.8 | 1486.6 | 1228.9 | 1.235 | NO |

Per-change contribution (vs base): store −8.9%/−15.1%/−6.2%/−0.2%;
overlap −18.1%/−19.1%/−20.8%/−17.4% (vs +store). Combined vs base:
−25.4%/−31.3%/−25.7%/−17.5%.

ADMISSION VERDICT: FAIL — best row qkv 0.989×, worst residual 1.235×.
Per stop rules (plan §“Stop an arm… failure of any named 0.6× TIME row”): STOP
after the two commits. No KLD / interleave / decode / serve phase (gated on
admission). Physics, not tuning: staging moves ~1.9 GB through L2 per gate_up
launch (each A byte staged ~544×, once per row-tile WG) — execution/issue
bound at ~1.3 TB/s. Overlap hides its latency (~2% per l2tiny) plus the
restructure bonus measured here, but cannot un-execute it. Reaching 0.60×
needs staging-WORK reduction (wider vector loads — the compiler already
demonstrates 4×b128 prologue coalescing; source-level 16 B A loads/publish —
or fewer row-tiles sharing tokens, i.e. larger BN, pruned for registers).
Proposed next step (not this ticket): source-level 16 B global A loads in
prologue+prefetch, keep 8 B LDS writes (stride-72 16 B LDS alignment fails on
odd rows), re-measure.

## Source variants (md5 of the TU file)

- base (HEAD f9bfe6d98): 698428413acf35ca7e4d056bec82f37b (= attrib pristine)
- +store (commit 1): 1c7da1cb1a61a5aef7f1ae80ffc84129
- +overlap (commit 2): d76c11a59fb892db43bc3986e8d3dd7f

Raw: `final-incumbent/base/store/probe.{json,log}`, `base-n{256,320,512}.json`,
`probe-n{256,320}.json`, `store-skew-v2.{json,log}`.
