# gfx1101 container A/B — empirical verification of the gfx1100 decode-campaign branch

**Date:** 2026-10-09 · **Lifecycle:** historical
**Branch:** `perf/gfx1100-decode-campaign` @ `aed5f93e1` + review fixes
(fmt-only; binaries byte-identical)
**GPU:** gfx1101 — Radeon RX 7700 XT 16 GiB (Navi 32), iGPU-attached desktop
on the host's other render node; compute dGPU otherwise idle.
**Toolchain:** docker `hipfire-build:rocm10` (ROCm 10.0.0 userspace, HIP 7.15
hipcc), fresh container per process.
**Fixture:** `qwen3.5-4b.mq4` md5 `712b69f8cf1016081cfa507c4d50e33d`
(= the 2026-08 campaign fixture). AR, KV q8, `--spec off`, legacy KV backend
(gfx1101 has no certified VMM path), synthetic matrix bench
`--pp 64,2048 --ctx 64,2048 --tg 128 --runs 5 --warmups 3`, fresh process per
invocation, arms interleaved M,B0,B1,B1F × 3 rounds.

**Binaries:** master `0f999cb4dc` (hipfire `2dd5d7bd…`, daemon `765fa313…`);
branch (hipfire `56ba8a8b…`, daemon `2916b3f8…`).

**Arms:** M = master · B0 = branch default · B1 = +`HIPFIRE_GFX1101_GFX1100_CAMPAIGN=1`
(four gfx1100-certified fusions admitted on gfx1101) · B1F = B1
+`HIPFIRE_FA_KVWRITE_FOLD=1` · B0D = B0 +`HIPFIRE_BT2_DISABLE=1` (kill-switch
isolation) · B3 = B1 +`HIPFIRE_QKVZA_FUSEDNORM=1` · H3 = B1
+`HIPFIRE_LM_HEAD_HFQ3=1`.

## Results (medians of per-process medians)

| metric | M | B0 | B1 | B1F | B0D |
|---|---|---|---|---|---|
| pp64 prefill tok/s | 1539.2 | 2081.8 | 2100.5 | 2085.4 | 1551.4 |
| pp2048 prefill tok/s | 2535.1 | 2537.0 | 2538.2 | 2535.2 | — |
| tg128@64 decode tok/s | 109.73 | 109.58 | 115.12 | 115.75 | — |
| tg128@2048 decode tok/s | 108.50 | 108.44 | 113.84 | 114.51 | — |

H3: tg64 130.63 (+13.5% vs B1), tg2048 128.95, pp64 2151.5.
B3: identical to B1 within noise (115.06 vs 115.12) — see verdict 5.

## Verdicts per branch claim

1. **Default-path decode neutrality: CONFIRMED.** B0 vs M: −0.14% @64,
   −0.06% @2048 (noise). The branch adds no default-on decode-path lever on
   gfx1101; every decode lever is opt-in or gfx1100-gated.
2. **BT2 prefill: WITHDRAWN 2026-10-09.** The measurement below was taken
   before the v1-only prune; the BT2 family has since been removed because its
   kernels are HFQ4-G256-only (no Magnum V2 dtype can reach them) and the
   surviving V2 GEMM families already ship their own batch-tiled arms. Kept
   here as the dated record of what was seen at the time:
   B0 vs M = **+35.3% pp64**; the kill switch collapsed it (B0D 1551.4 ≈ M
   1539.2); flat at pp2048 (+0.07%). The 4B/gfx1100 counterpart and the
   removal rationale are in
   [2026-10-09-gfx1100-campaign-reverify.md](2026-10-09-gfx1100-campaign-reverify.md).
3. **Four gfx1100 fusions, +5.7% decode (gfx1100 fixture): direction and
   magnitude hold on gfx1101 via the experimental gate.** B1 vs B0 = +4.91%
   @64, +4.92% @2048 (process medians 115.08–115.43, tight). NOT a
   certification: `probe_fa_prep` on gfx1101 shows 1-ulp diffs in the rope
   sin-branch (dims 32-63; 102/4096 fa_q, 34/1024 fa_k) — the gfx1100 FMA
   contraction fix does not pin gfx1101 codegen. The
   `HIPFIRE_GFX1101_GFX1100_CAMPAIGN` gate stays experimental/diagnostic.
4. **fa-kvwrite fold (+0.5–1% on gfx1100): engaged, bit-exact, ~+0.2% on
   gfx1101 — treat +0.5–1% as gfx1100-fixture-bound.** Paired interleaved
   rounds: r1 +0.70%, r2 −0.41%, r3 +0.27%; pooled n=15 mean +0.19%,
   10/15 wins, both contexts. `probe_fa_kvwrite` PASS (fold outputs + both
   cache rows bitwise identical) on gfx1101.
5. **qkvza fusednorm negative oracle: REMOVED 2026-10-09.** Measured −26%
   on gfx1100 and inert off-gfx1100 by design (the gate was `is_gfx1100()`-only).
   The oracle and its probe were discarded with the rest of the v1-only
   surface, because the gate also required `MQ4G256 | HFQ4G256` and is
   therefore unreachable from any Magnum V2 dtype.
6. **lm_head HFQ3 (research-only): +13.5% decode on gfx1101** (larger than
   gfx1100's ~+10% — the 7700 XT is more bandwidth-starved), pp2048 +1.3%.
   Output is lossy-divergent as documented: greedy factual/code/poem outputs
   remain coherent and correct but differ from the Q8_0 head's tokens.
7. **Greedy text parity: byte-identical across M ≡ B0 ≡ B1 ≡ B1F** on three
   prompts × up to 96 tokens (57/96/96 tokens, thinking off, temp 0) —
   despite the fa_prep ulp diffs, no argmax flip occurred in these samples.
   H3 differs (expected, lossy).
8. **test_kernels:** 17 passed / 0 failed / 1 skipped (gfx1151-only route)
   in-container on gfx1101.
9. **Serve battery (VALIDATION route, B1F arm):** 5/5 genre turns coherent
   (code/reason/factual/prose/instruct), runaway=2 (length-cap at temp=1.0,
   matching the gfx1100 battery's shape), empty=0, attractor=0, serve-path
   decode 113.9–116.1 tok/s. daemon md5 `2916b3f8…`.

## Scope notes

- Absolute tok/s here are gfx1101 numbers (~110 AR decode vs gfx1100's ~210:
- bandwidth ratio), not comparable to the gfx1100 campaign records; only the
  within-host A/B deltas are evidence.
- Raw JSON, parity outputs and probe logs: `/tmp/hipfire-ab/out/`,
  `/tmp/hipfire-ab/parity/` (discovery pointers; medians recorded above).
