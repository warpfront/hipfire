# gfx1151 batched-prefill gap vs halogen — lever space after the cheap tier (max brainstorm)

Date: 2026-09-15. Worktree `/home/kaden/ClaudeCode/warpfront/wt-lloyd` at
`cadd315a1`, read-only. **No GPU program was run; no source was edited.**
Every magnitude below is an **[estimate]** unless it is a caller-supplied
measurement. Not a plan of record; a Sol/Muse seat converts survivors into
specs. Predecessors: `docs/plans/2026-09-15-halo-prefill-levers.md`
(brainstorm 1) and `docs/plans/2026-09-15-halo-prefill-plan.md` (plan of
record with the A2/B1/C2 admissions and the C1/F STATUS blocks).

## 0. Denominators (HTTP-measured walls, current head)

| prompt | hipfire | wall | halogen | target wall | gap | gap % |
|---|---:|---:|---:|---:|---:|---:|
| pp512 | 600 t/s | 853.3 ms | 620 | 825.8 ms | **27.5 ms** | 3.2% |
| pp2048 | 582 t/s | 3519 ms | 710 | 2884.5 ms | **634.5 ms** | 18.0% |
| pp8192 | 530 t/s | 15.46 s | — | — | — | — |
| pp32768 | 395 t/s | 82.96 s | 566 | 57.89 s | **25.06 s** | 30.2% |

Per-chunk (N=512) ledger from the caller's current pp512 profile (~837 ms
GPU): IU4 `full_set_occ3` 272×1690 µs = 459.7 ms + `full_add_occ3`
128×1640 µs = 209.9 ms → **IU4 669.6 ms (80%)**; `gated_delta_net_q8_fast`
48×~1260 µs ≈ 58.6 ms (7%); FA2 16×512 µs = 8.2 ms (1%); `quantize_int4`
64×154 µs = 9.9 ms; `fused_silu_mul_mq_rotate` 64×457 µs = 29.2 ms;
`conv1d_silu_split` 48×237 µs = 11.4 ms; other ≈ 50 ms. Linear buckets scale
×4 (pp2048) and ×64 (pp32768); FA2 scales with the causal prefix: 7.0 ms/call
at mean ctx 4352 (pp8192 profile) → ≈ 0.13 s at pp2048, ≈ 27–29 s at pp32768.

pp32768 attribution [estimate]: IU4 42.9 s (52%), FA2 27–29 s (33–35%), GDN
3.75 s, other 6.4 s, host/launch residual ≈ 2 s.

### 0.1 Two conflations in the inherited ledger that change the ceilings

1. **"IU4 at 45% of int4 peak" treated the per-symbol average as the gate/up
   call.** 1690 µs is the mean over 272 set calls whose MAC counts range
   5.2 M (k/v) to 89 M (gate/up). Fit the profile with block time = groups ×
   g and rounds = ⌈blocks/80⌉ (§1.1): set total 459.7 ms = g × 26,880 →
   **g ≈ 17.1 µs per 256-K group per block**; add total 209.9 ms = g ×
   11,776 → g ≈ 17.8 µs. The two families agree, so the model holds and
   gate/up is ≈ 2.4 ms/call, k/v ≈ 0.34 ms. Blended: 24.8 T int4-ops in
   669.6 ms = **37 TOPS = 31% of the nominal 118.8 TOPS** (40 CU × 2.9 GHz ×
   1024 ops/clk), 36% if the sustained clock is 2.5 GHz. Per SIMD: 1024
   WMMAs per 17.5 µs = **49.6 clk per iu4 WMMA**. Whether the WMMA itself
   costs 16 or 32 of those clocks on gfx1151 is *unmeasured* (§1.0) and is
   the single fact that sizes the whole GEMM lane.
2. **Gate F is not blocked by the kernel.** The removed F1 oracle
   (`git show 203864992:crates/hipfire-runtime/examples/tmp_fa2_n1024_oracle.rs`)
   uploaded `positions` through `upload_raw` → `DType::Raw`
   (`crates/rdna-compute/src/dispatch.rs:3607-3611`), whose element size is
   1 byte (`:458`), then sliced the second arm with
   `d_pos_b.sub_offset(HALF, HALF)` — `sub_offset` multiplies by the dtype
   size (`:251-266`), so the second N=512 launch read `positions[128..640)`
   for rows 512..1023. That explains all three STATUS numbers: max_abs 0.34
   at L=1024 (causal bound off by 384 of ≤1024 keys), 1.5e-2 at 8192, 3.8e-3
   at 32768 (384 of ~32K keys), and the "two 512s faster at L=1024" timing
   (the second launch did ~half its work). The kernel has no batch-dependent
   path: `q_base = blockIdx.x*8`, per-WG causal bounds from `positions[]`,
   `qok = qr < batch_size` (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:163-206,544-548`).
   The only 512 limits are host gates (`crates/rdna-compute/src/attention.rs:3274-3275,3889,4054`).
   The L=32768 timing (98.5 vs 106.4 ms, −7%) is nearly fair (384/32K
   positions wrong) and is the honest per-launch prize. **F is re-opened at
   the cost of a one-hour oracle fix** (§4.2).

Found while reading, not a lever: the FA2 launcher's capture-only kernarg
blob omits `out_ptr` (`crates/rdna-compute/src/attention.rs:3972-3982`
pushes q,k,v,pos,…; `params` at `:3941-3952` has out). Dormant because the
route rejects capture (`:3269-3270`), but it will bite any prefill-capture
experiment.

## 1. IU4 GEMM lane (80% of pp512, 52% of pp32768)

Kernel facts (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`):
128×128 tile, 8 wave32s as 4 row-groups × 2 column-halves, wave tile 32×64
= `sum[64]` f32 (`:43-46,172-185,236`); one i32 accumulator chain per
(j0,n) tile, 4 `t` steps × 2 WMMAs, `#pragma nounroll` (`:186-199`); fold
per 128-K half (`:200-210`); loader: 16 raw nibble dwords + 1 header dword
per thread per group, header replicated 4× per ksc (`:137-159`); serialized
kb loop with four `__syncthreads` per group (`:238-262`); LDS
(128×18+128×42)×4 = 30,720 B (`:30,51`), **190 VGPR, 0 spills** (`:50`);
entries `__launch_bounds__(256, 2|3)` (`:291-297`). Host: occ3 symbols
selected for full shapes (`crates/rdna-compute/src/gemm.rs:19065-19070`),
grid `[⌈M/128⌉, ⌈N/128⌉]`, block `[32,8]`, LDS mirror (`:19093-19109`).
Residency: 2 blocks/CU (LDS 2×30.7 KB ≤ 64 KB; VGPR 1536/190 → 8
waves/SIMD = exactly 2 blocks). Both limits bind simultaneously — that is
why every in-place latency-hiding attempt (A1) blew up: there is no slack
in either budget at this tile shape.

### 1.0 A0 — calibrate the iu4 WMMA rate and sustained clock on gfx1151 (run first)

**Mechanism.** Nothing in-tree measured `V_WMMA_I32_16X16X16_IU4`
throughput on RDNA3.5 (grep of `docs/` finds only the gfx906 dp4a/dot8
microbench precedent, `docs/plans/gfx906-mmq-prd.md:76-78`). The "45%" and
this document's "31%" both assume 1024 ops/clk/CU at 2.9 GHz. XTX data
proves ≥1024 ops/clk on gfx1100 (588 µs mean set call at 96 CU / 2.5 GHz
would exceed peak otherwise), but not on gfx1151, and Halo's clock under a
40-CU WMMA load is unknown.

**Experiment (2 h, composer, diagnostic only).** A ~30-line HIP microkernel:
40+ CU-saturating grid, 8 waves/SIMD, a loop of register-only iu4 WMMAs in
(a) one dependent chain, (b) 4 independent chains, (c) 4 chains
interleaved with 2 `ds_read_b64` per WMMA from a conflict-free LDS layout;
time with `s_memrealtime` inside the kernel and hipEvents outside; report
**WMMAs per µs per CU** (clock-independent) plus `pp_dpm_sclk` during the
run. Output: the real peak, the dependent-chain penalty, and the co-issue
cost of the current consumer's LDS:WMMA ratio.

**Decision.** With g = 17.5 µs per group per block (§0.1):

| hypothesis | WMMA clk | WMMA share of 49.6 clk | non-WMMA VALU [est.] | stalls (latency/barrier/LDS) | GEMM headroom |
|---|---:|---:|---:|---:|---:|
| H1: 16 clk | 32% | ≈ 14% (fold 256 + loader ≈100 + addr ≈100 VALU per 64 WMMAs) | **≈ 54%** | ×0.5–0.6 reachable by a latency-hiding redesign |
| H2: 32 clk | 64% | ≈ 14% | ≈ 21% | ×0.8–0.85 at best |

Under H2 the GEMM lane is worth ≈ −100 to −130 ms pp512 / −0.4 to −0.5 s
pp2048 / −6 to −8 s pp32768 at its ceiling and pp2048 needs the middle
tier as well; under H1 the same redesign is worth 2× that. A0 also settles
whether the XTX-vs-Halo "extra 1.2–1.4×" is memory or clock. Not on any kill
list: nobody measured it.

### 1.1 Grid occupancy per shape (80 resident blocks: 2/CU × 40 CU)

Block duration = groups × 17.5 µs; rounds = ⌈blocks/80⌉; tail loss =
(⌈w⌉−w)/⌈w⌉ [model, §0.1-validated to 4%].

| projection | M×K | blocks | rounds | tail loss | calls/chunk | est. µs/call | loss ms/chunk |
|---|---|---:|---:|---:|---:|---:|---:|
| gate, up | 17408×5120 | 136×4 = 544 | 6.8 → 7 | 2.9% | 128 | 2,390 | 8.8 |
| FA q | 12288×5120 | 384 | 4.8 → 5 | 4.0% | 16 | 1,710 | 1.1 |
| LA qkv | 10240×5120 | 320 | 4.0 | 0 | 48 | 1,370 | 0 |
| LA z | 6144×5120 | 192 | 2.4 → 3 | 20% | 48 | 1,030 | 9.9 |
| down (add) | 5120×17408 | 160 | 2.0 | 0 | 64 | 2,380 | 0 |
| wo/o (add) | 5120×6144 | 160 | 2.0 | 0 | 64 | 840 | 0 |
| FA k, v | 1024×5120 | 32 | 0.4 → 1 | 60% | 32 | 342 | 6.6 |

Structural tail ≈ **26 ms/chunk = 3.9% of IU4** [estimate]. Recoverable
cheaply: z as an M-tile-64 twin (384 blocks, 4.8 → 5 rounds at half
duration: loss 20% → 4%, −8 ms) and q+k+v in one launch (M=14336, 448
blocks, 5.6 → 6 rounds: replaces 4%+60%+60% with 7%, −6 ms). Gate/up's 7th
round (64/80) is not fixable by tiling; only a 3-blocks/CU design (§1.3) or
a persistent grid changes it. **Lever T0 (tails): −14 ms pp512, −56 ms
pp2048, −0.9 s pp32768 [estimate]; composer; 1 day; bit-exact (same block
work, different grid); kill if the z twin is <15% faster on the z shape.**
This is brainstorm-1 A3, deferred by the plan; the numbers above are its
sizing.

### 1.2 What the kernel loses per WMMA (the 49.6-clk budget)

Per wave per 128-K half (64 WMMAs):

- **Issue/LDS:** 128 `ds_read_b64` (2 per WMMA, `:193-196`) + ~64–128 VALU
  of address arithmetic because `t` is `nounroll` and (n, j0) are outer
  (`:185-199`); A fragments are reloaded 4× (once per j0), B fragments 2×
  (once per n). LDS *bandwidth* is not the wall: a conflict-free b64 read
  across 16 unique rows is 1 clk of the 128 B/clk LDS pipe → 4 clk per 16
  clk (25%) under H1. LDS *latency* is: each `t` step waits ~100+ clk for 4
  reads before 2 WMMAs, and only inter-wave interleaving covers it.
- **WMMA dependency:** one accumulator chain per wave; the 8 WMMAs of a
  128-K half are serially dependent (`:197-198`). A0(a)-vs-(b) measures the
  penalty; the ISA doc's back-to-back-same-C exception makes it plausibly
  small, but it means a wave can never have two WMMAs in flight.
- **Barrier/latency:** all 8 waves load, then all 8 compute; four
  `s_barrier` per group (`:247,249,258,260`); the X loads (17 dwords/thread,
  DRAM) and Y loads (9 dwords/thread, L2) are exposed once per phase as the
  *max* over 8 waves' latencies, and the co-resident block is in phase
  (same start, same work), so it rarely fills the gap. Under H1 this is the
  bulk of the 54% stall term.
- **DRAM traffic:** grid x = row tiles fastest (`gemm.rs:19108`,
  `row0 = blockIdx.x*128` `:225`), so the four column tiles re-stream every
  weight tile: 4 × 47.3 MB per gate/up call = 189 MB, + 35.7 MB f32 output
  → ≈ 94 GB/s over 2.39 ms [estimate], ~40% of practical LPDDR5X. Not
  bandwidth-bound, but the loads are bursty (80 blocks issue in phase).
- **VALU sharing:** fold (`:204-210`: cvt + 2 FMA + half2 unpack per
  element, 64 elements) ≈ 256 VALU, loader ≈ 100, addresses ≈ 100 → ≈ 450
  VALU per 1024 (H1) WMMA clk. Fixed by the arithmetic contract (bit-exact
  fold order); only the address term is reducible.

Attribution experiments (each 1–2 h, throwaway twins, numerically invalid):
**no-global-load twin** (skip loads after group 0, compute on stale LDS) —
its time is the compute+barrier floor; the difference to production is the
exposed memory term. **No-fold twin** — measures the fold's VALU share.
Together with A0 they replace every "[estimate]" in the table above with a
number before anyone writes a kernel.

### 1.3 Tile/wave/LDS redesign options with budgets

Rules: VGPR cap = 1536 / (waves per SIMD); LDS 64 KB/CU; 2 blocks/CU needs
≤32 KB and ≤192 VGPR; 3 blocks/CU needs ≤21.3 KB and ≤128 VGPR; 1 block/CU
allows ≤64 KB and ≤384 VGPR at 8 waves (≤307 at 10 waves).

| option | wave tile / waves | VGPR est. | LDS | residency | LDS reads/WMMA | verdict |
|---|---|---:|---:|---|---:|---|
| T1 current | 32×64 / 8 | 190 (measured) | 30,720 | 2/CU | 2.0 | baseline |
| T2 register-blocked consumer (t-outer, 8 live i32 chains, 2A+4B frags) | 32×64 / 8 | 190+56+4 ≈ 250 | 30,720 | 6 waves/SIMD → **1 block/CU** | 0.75 | **dead** unless the ISA shows ≥60 VGPR of compiler fat; the 4-chain variant (≈214) also exceeds 192 |
| T3 M-tile 64 (2×4 waves, 32×32) | 32×32 / 8 | ≈ 190−32−10 ≈ 148 [est.] | (64×42+128×18)×4 = 19,968 | 3/CU **only if ≤128 VGPR** (else 2/CU, no gain) | 2.0 | conditional: +50% waves/CU to hide the phase stall; Y traffic per MAC ×1.5 (L2); gate: profile JSON VGPR ≤128, else kill |
| T4 N-tile 256 | 32×128 / 8 | ≥ 254 | 40,960 | 1/CU | 2.0 | dead: halves waves, buys only L2 reuse that A5 gets for free |
| T5 4 waves × 2 slabs (64×64 per wave) | 64×64 / 4 | ≈ 250–280 | 30,720 | 6 waves/SIMD → 12 waves/CU = **3 blocks of 4**, LDS caps at 2 → 8 waves/CU | 1.0 (2 chains) | dead-ish: half the waves per CU, same phase problem, no loader |
| **W2 warp-specialized, double-plane** (§1.4) | 8 compute 32×64 + 2 loader | compute ≈ 198 (2 chains), loader ≈ 110 staging | **61,440** (2 × 30,720) | **1 block/CU by design**, 10 waves = 5/SIMD, cap 307 | 1.5 | the only option that removes the phase stall inside the budgets |

Free but non-lever: the header slots are 8 words per row but the consumer
reads only index `4*h` ∈ {0,4} (`:206` vs `:155-157`), so
`IU4_TILE_X_K` could be 34 (bank stride 34r mod 32 = 2r, still
conflict-free) and save 4 KB/block. It does not unlock a second Y plane at
2 blocks/CU (26.6+9.2 = 35.8 KB > 32 KB), so it only matters inside W2/T3.

### 1.4 W2 — warp-specialized IU4 with two LDS planes (the A-lane redesign)

**Mechanism.** Accept 1 block/CU. Block = 8 compute waves (the current
consumer and `sum[64]`, unchanged fold order → bit-exact) + 2 loader waves.
LDS = two X strips (2 × 21,504) + two Y-half strips (2 × 9,216) = 61,440 B.
Steady state for group kb: compute reads X(kb) from X-strip kb&1 and
Y(kb,h0) from Y-strip 0; meanwhile the loaders, which hold no `sum[]`, have
already fetched X(kb+1) (17.4 KB) and Y(kb,h1) (9.2 KB) into VGPRs
(≈104 staging dwords per lane across the 64 loader lanes) and `ds_write`
them into the other X strip and Y-strip 1 — strips freed at the previous
barrier. One `s_barrier` per half (two per group instead of four,
`:247,249,258,260`) then certifies both "readers done with the old strip"
and "writers done with the new one"; compute waves reach it late because
the loaders finished long before, so the exposed term collapses from a
DRAM round trip plus wave skew (≈2000+ clk/group today) to ds_write +
barrier skew (≈300 clk/group) [estimate]. Register-block the consumer to
2 i32 chains (n=0,1 share B fragments: 6 b64 per 4 WMMAs); the VGPR cap is
307 at 5 waves/SIMD, so the +8 VGPR is affordable here and not at T1.
Per-SIMD WMMA feeding with 4 compute waves: each wave's `t` step issues 6
reads then 4 WMMAs (64 clk of pipe under H1) — 4 waves × 64 = 256 clk of
pipe per ≈150–200 clk of per-wave iteration → fed [estimate]. Weight
bytes per MAC unchanged; combine with A5 (§1.5) for DRAM.

**Ceiling.** H1: GEMM ×0.55–0.70 → **−200 to −300 ms pp512, −0.8 to −1.2 s
pp2048, −13 to −19 s pp32768**; H2: ×0.80–0.85 → −100 to −135 ms / −0.4 to
−0.54 s / −6.4 to −8.6 s [estimate]. Either clears pp512; H1 alone clears
pp2048.

**Kill ladder (3–5 days, Sol spec).** (0) A0 says H1 or H2 and the
no-load twin shows ≥25% exposed memory term — else stop here. (1) Day-1
skeleton (roles + planes, no correctness): JIT log must show VGPR ≤ 300,
scratch 0, LDS 61,440, and the profile JSON one block/CU; a spill or a
VGPR count that forces <5 waves/SIMD kills it (the A1 lesson, enforced
before any math). (2) Dominant-shape microbench (M=17408, K=5120, N=512,
100 iterations interleaved) must beat T1 by ≥15%. (3) Bit-exact
differential on the §13.4 shape set; then in-model weighted IU4 ≥15%.
Not on the kill list: A1 was in-place VGPR staging at 2 blocks/CU; "iu8
register-resident" was the iu8 kernel; no record of a producer/consumer
wave split exists in `docs/` (grep `warp.special|loader wave|round-2`);
if "MMQ round-2 ×3" contained one, that record supersedes this.

### 1.5 A5 — column-adjacent grid order (weights once per call)

**Mechanism.** Swap grid to `[⌈N/128⌉, ⌈M/128⌉]` and `row0 = blockIdx.y*128,
col0 = blockIdx.x*128` (`gemm.rs:19100-19108`, kernel `:225-226`). The four
column tiles of one weight band become consecutive blocks, co-resident in
the same round, so each 17.4 KB weight tile is fetched once (L2 2 MB,
`crates/rdna-compute/src/profiler.rs:67-74`) instead of four times: 189 MB →
47 MB per gate/up call. Cost: four Y slices hot instead of one (1.47 MB at
K=5120; **5 MB at K=17408** for down, over the 2 MB L2 — the strided
alternative is a 2×2 swizzle). Bit-exact: per-block work and per-element
writer unchanged. Also the prerequisite for chunk ≥1024 to pay on the GEMM
side (§5.2): at N=2048 with this order, weight bytes per token fall 16×
relative to today.

**Ceiling.** The kernel is ~40% of practical bandwidth, so this attacks
burst latency, not bandwidth: **2–10% of IU4 = −13 to −67 ms pp512, −54 to
−270 ms pp2048, −0.9 to −4.3 s pp32768** [estimate, low confidence on
magnitude, high on cost]. **Kill (1 h, composer):** `full_set_occ3` on
gate/up and `full_add_occ3` on down (the K=17408 Y-footprint case) before
and after; <2% on the weighted bucket or any down regression → revert (or
try the 2×2 swizzle once, then drop). Brainstorm-1 A5 was deferred, never
tried.

### 1.6 Prefill-only lower-precision weight copy (halogen's trick) — dead by dominance

**What halogen actually does.** `docs/QUANT.md` in `peonist-ai/halogen-server`:
"The single aggressive technique (W4A4, int4 *activations*) is fenced to
prefill … all 400 planes active"; decode streams FP8 rows + NVFP4 FFN
(23.5 GB) while the file is 35.9 GB — the ~12.4 GB difference is a
prefill-only **int4 weight copy** for tensors whose decode copy is FP8.
That is exactly hipfire's iu4 route (`block_i4_128` activations, 4.25-bpw
MQ4V2 weights fed as raw nibbles, `:5-8,121-142`), so halogen's prefill
MAC precision equals hipfire's; halogen's decode copy is *wider*, not its
prefill copy narrower. The 512→2048 slope (620→710 t/s, +15%) while
hipfire is flat (600→582) is therefore not quantization; it is chunk/tile
amortization (§5.2).

**W2/W3 sidecar for hipfire.** MACs: unchanged — `wmma_i32_16x16x16_iu4` is
the narrowest WMMA on gfx11; a 2/3-bit copy must be expanded to nibbles in
the loader (~3 VALU per output dword, +50 VALU/thread/group, +2.5% of the
H1 VALU budget). Bytes: 47.3 → 24–35 MB per gate/up call (−30 to −50%).
VRAM: +8–10 GB (fits). Quality: the KV cache and Q8-EF GDN state that decode
consumes are produced by the lower-precision copy; hipfire's WT2 KLD gate is
a teacher-forced (prefill) measurement, so the penalty lands 1:1 in the
existing ship gate (mq3/mq2 tiers exist as `DType::MQ3G256V2`/`MQ2G256V2`,
`dispatch.rs:435-447`, but their KLD on this model is not in front of me).
**Verdict:** it only ever buys the same "weight bytes" term that A5 (4×,
bit-exact, 1 h) and chunk 2048 (another 4×) buy with zero quality cost;
even in the best case the ceiling is the memory term (≤10% of IU4). Int4
with smaller groups (g128) adds header bytes and changes nothing in the
MAC — a quality lever, not a speed lever. Drop; re-open only if A5 proves a
large byte term *and* chunk ≥1024 is refused.

## 2. FA2 attention lane (1% of pp512, 33–35% of pp32768)

Kernel (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip`): WG = 1 KV head × 8
positions = 48 rows on 3 compute waves + 1 fill-only wave (`:7-16,138`);
KT32, LDS 32,768 (`:85-87,131`), grid `[⌈batch/8⌉,4]`
(`attention.rs:3929,3967-3969`); B1 landed (even swizzle + b64 loads,
`:51-57`, −4..−5%), clobbers kept (`:406,496`).

### 2.0 Where the time is (source-derived; confirms with two 1-h twins)

Tile time is context-independent: at pp512 (KV 557 KB, L2-resident) the
16 calls × 512 µs with 256 WGs on 80 slots give ≈ 12–19 µs per WG-tile; at
pp8192 (mean 136 tiles/WG, 3.2 rounds) 7.0 ms → ≈ 16 µs per WG-tile
[estimate]. Compute per wave per tile ≈ 64 f16 WMMAs (2048 clk at 32
clk) + ≈ 1240 VALU (512 of them the per-dc f32→f16 Q reconversion,
`:389-402`; 128 the rescale `:462-464`) + 256 LDS reads + 128 Q float4
loads ≈ 3.5–5 µs per SIMD [estimate]. The rest is the **fill**: both K
loops carry `#pragma unroll 1` on the 8-dword code loop (`:279-285`) and
the V loop likewise (`:326-331`), each iteration being load → wait → cvt →
LDS store. Per thread per tile that is 2 K slots × (1 scale + 8 serialized
dword loads) + 1 V pair × (2 scale + 8 × 2 loads) ≈ **27 serialized memory
round trips**, ≈ 1 outstanding load per thread. At L2-hit latency (~600–900
clk) that alone is 5.6–8.4 µs per WG-tile; the two co-resident WGs
overlap it only partially. CHANGELOG.md:11's Halo/XTX FA2 ratio (1870/810 =
2.3× ≈ CU × clock ratio) says the kernel is equally latency-bound on both
parts, i.e. not DRAM-bound — consistent with a per-thread latency chain.

**Experiments (composer, 2 h total, numerically invalid twins):** dump the
ISA and count `s_waitcnt vmcnt(0)` inside the fill (expect ≈27 per tile);
time a fill-only twin (`compute=false` for all waves) and a compute-only
twin (fill once, loop on stale planes) at (512, 7680, 8192). Their sum vs
production gives the overlap; the fill-only time is B6's denominator.

### 2.1 B6 — fill memory-level parallelism (the FA2 lever)

**Mechanism.** (a) Remove `#pragma unroll 1` at `:279` and `:326` so the 8
code loads per slot issue back-to-back (VGPR +8..16; kernel is
`__launch_bounds__(128,1)` with 2 WGs/CU → 4 waves/SIMD → 384-VGPR budget,
0 spills today). (b) Better: load all 36 dwords a thread needs for the tile
(both K slots, both V blocks; 34-B blocks alternate 2-mod-4 alignment,
`__builtin_memcpy` already emits unaligned dwords) before any conversion —
one wait per tile instead of 27. (c) Best, the real B4: issue tile t+1's 36
raw dwords into registers during tile t's compute (all four waves keep
their own share; +36 VGPR per wave), convert and store after the turnover
barrier `:502`. All three are bit-exact: same bytes, same conversions
(`:286-296,336-341`), same swizzled store addresses.

**Ceiling.** If the fill is 60–75% of tile time [estimate], (a)/(b) cut
serialized round trips 27 → 3–5 and attention falls ×0.4–0.6: **−3 to −5 ms
pp512, −50 to −80 ms pp2048, −11 to −16 s pp32768** [estimate, medium
confidence]. (c) hides what remains of the fill: a further ×0.8. Cross-check:
halogen's 32K wall implies its attention ≤ 15–18 s vs hipfire's 27–29 s, so
a 1.7–2× attention gain is exactly the size of the visible gap.

**Kill (composer; ISA check 30 min, twin 2 h, oracle rebuild 3 h).** Rebuild
the seven-shape/two-K-mode oracle (crib: the removed
`tmp_fa2_n1024_oracle.rs` with positions uploaded as F32-dtype bits or
sliced by `HALF*4` on the Raw tensor). Gate: ISA `vmcnt(0)` count in the
fill ≤6 per tile, 0 spills, bit-identical output and post-launch Q on all
shapes, `attention_q8_0_fa2_gqa_gfx11` at pp8192 ≥ 20% faster (B1's 4–5%
is the noise floor for "the fill was not it"); else revert and record
"fill latency is hidden by the co-resident WG" — which then makes (c)
pointless too. Not on the kill list: B1 touched the compute-side LDS reads,
B2 the compiler clobbers, KT64 the tile width; nobody touched the fill's
issue structure.

### 2.2 FA2 compute-side, after B6 measures (composer, bit-exact, 1 day)

- **B3 pre-convert Q to f16 once per chunk**: deletes 512 of ≈1240 VALU
  per tile and halves Q L1 traffic (`:389-402` reloads and reconverts four
  float4 per dc per subtile). f32→f16 RN is the same rounding the kernel
  applies now → bit-identical operands. ≈ −25 to −35% of the *post-B6*
  compute term [estimate]. Plan §16 deferred it pending a ≥10% attribution;
  the twin in §2.0 provides that number.
- **Rescale skip when alpha == 1.0 everywhere in the wave** (`__ballot` of
  `alpha != 1.0f`, wave-uniform branch around `:462-464`): `x*1.0f == x`
  bit-exactly; at long context the running max stabilizes and most subtiles
  skip 128 VALU. ≈ −5 to −10% of compute [estimate].
- **LPT grid order** (map `blockIdx.x → ⌈batch/8⌉−1−x` in the launcher or
  kernel): the longest-causal WGs are dispatched last today; reversing
  trims the tail at short context only (≈ −1 ms pp512). Trivial; bank it
  with B6.
- **f16 accumulation / split-KV / KT64:** killed before; nothing new here.

### 2.3 >512 rows (lever 4 answer)

Nothing in the kernel breaks: see §0.1 (2). Cost to fix: one-hour oracle
fix, then host gates (`attention.rs:3274-3275,3889,4054`) to 1024/2048. The
real per-launch prize at 32K is the caller-measured −7% (≈ −1.9 s at
pp32768) plus what §4.2 lists for the rest of the chunk.

## 3. GDN lane (7% of pp512 and pp2048, 4.5% of pp32768)

Shipping: `gated_delta_net_q8_fast` launched `[48, 32, 1]` × 32 threads
(`crates/rdna-compute/src/norm.rs:3016-3018,3064-3068`) with the *plain*
source (`kernels.rs:6311-6312`); each wave owns 4 state rows and walks 512
tokens serially (`kernels/src/gated_delta_net_q8_fast.hip:145-256`): per
token fresh global loads of q/k/v/gate/beta (`:146-151,165-169`), and per
row two 5-step `__shfl_down` trees + one `__shfl` broadcast
(`:180-182,201-202`) = ds_bpermute chains. 1536 single-wave WGs vs 1280
slots (16 waves/SIMD at 64 VGPR, 2 KB LDS) → 1.2 rounds. Requant once per
launch (`:258-296`). The DPP tree that preserves this exact summation
order exists (`:51-84`, lane-select bug from
`docs/perf-checkpoints/2026-08-05-mq2r-a3b-lowbit-campaign.md:223-230` is
fixed in the current source `:61-62`) but is compiled **only into the
decode compact symbols** (`kernels.rs:6345-6363`). The F32 chunked oracle
kernel is scalar loops over global re-reads with host-serialized chunks
(`kernels/src/gated_delta_net_f32_chunked.hip:12-17,177-186,192-205,226-255,262-272`)
and was correctly rejected at C1.0 (10.5× slower).

### 3.1 G1 — the sequential kernel's constant factors (composer, bit-exact, ≤1 day)

Three independent, individually gated changes to the batched launch:

1. **DPP reductions** for the prefill compile: a `GATED_DELTA_NET_Q8_FAST_DPP_GFX1151_SRC`
   twin (`#define HIPFIRE_GFX1151_GDN_DPP_REDUCE 1`, same symbol otherwise)
   selected in `gated_delta_net_q8_batch_seq` on exact gfx1151. Replaces
   ≈11 bpermute round trips per row per token (~50 clk each) with DPP VALU
   ops; the comment at `:52-56` claims order preservation, so lane 0's sum
   is bit-identical — the oracle proves it.
2. **Software prefetch** of token t+1's q/k/v/gate/beta (≈14 VGPR) inside
   the loop; LLVM does not modulo-schedule this loop, so the ~800-clk
   L2 latency sits on the per-token chain today.
3. **TILE_ROWS=8** (already supported, `:215-248`): 768 waves → 1 round
   instead of 1.2; per-token chain grows (8 rows) unless the compiler
   interleaves rows — an A/B, not a design.

**Ceiling.** Per-token chain ≈ 800 (loads) + 4 rows × ~600 (two bpermute
trees + math) ≈ 3200 clk today [estimate]; DPP + prefetch → ≈ 4 × 150 +
overlap ≈ 600–1000 clk → GDN ×0.3–0.5: **−30 to −40 ms pp512, −120 to −160
ms pp2048, −1.9 to −2.6 s pp32768** [estimate, medium confidence]. This
alone clears pp512. **Kill:** extend `crates/rdna-compute/examples/gdn_chunk_parity.rs`
(it already has a shipping Q8-fast timed arm from C1.0) with the twin;
require bitwise-identical output + Q8 state + EF + scales at T=512, 48
heads, both regimes; then median ≥25% faster; else drop that item. Not on
the kill list: the plan's §16 explicitly declared the DPP/R8 family
"decode-only, not a batched-prefill competitor" — it was never tried in
the batched launch.

### 3.2 G2 — chunked GDN kernel that could beat q8_fast (Sol spec, non-bit-exact)

**Design.** Grid `[48 heads × 8 state-row slabs]` = 384 WGs of 128 threads;
each WG owns 16 rows of the 128×128 state for one head, kept **f32 in
registers** (16 floats/lane) for the entire 512-token launch; on-device
loop over 512/C chunks (C=32). Per chunk: stage K and V(slab rows) for the
chunk in LDS (K f32 16 KB, V-slab 2 KB), Q read from L2 in place;
w = S_slab@K^T (65k MAC, f32 VALU), KK^T and QK^T (32×32×128 each,
recomputed per slab — redundant ×8 but 131k MAC; f16 WMMA optional,
`wmma_f32_16x16x16_f16`: 32 WMMAs each), the unit-lower solve over the
slab's 16 columns (32 sequential steps, LDS round trips), out = expG·(S@Q^T)
+ Mkq·δ, S ← γS + Σ decay_j δ_j⊗k_j (65k MAC). LDS ≈ 16+2+8 (L,Mkq)+2 (δ)
≈ 28 KB → 2 WGs/CU. Prologue dequantizes the slab from Q8 (`:135-143`
arithmetic), epilogue folds EF and requantizes exactly as `:260-296` — one
persistent commit per outer chunk, the shipping cadence, no bridge
launches. Algebra: the proven `gdn_chunked_f32` decomposition
(`gated_delta_net_f32_chunked.hip:26-37`). Summation order differs from
the sequential kernel → **not bit-exact; ships default-OFF behind
`HIPFIRE_GFX1151_GDN_CHUNKED`**, KLD ≤ baseline + 5e-4, C1.1 tolerances.

**Ceiling.** ≈ 10k clk per chunk per WG (≈3.2k FMA/lane + 5k solve +
barriers) × 16 chunks ≈ 55 µs + L2 latency stalls ≈ 80–100 µs per WG, 4.8
rounds → **≈ 400–500 µs vs 1260** [estimate] → −35 to −45 ms pp512, −140
to −180 ms pp2048, −2.2 to −2.9 s pp32768. WMMA is not needed to clear the
gate; it is an optimization for the CxC parts once the f32 version works.
**After G1 lands, G2's incremental value is ≈ −15 to −25 ms pp512** — rank
it accordingly.

**Kill ladder (2–4 days).** Stage 1 (1 day): multi-WG on-device-loop f32
kernel, no Q8, no WMMA; parity <1e-4 vs `gated_delta_net_f32_batch_seq`
through `gdn_chunk_parity`; median at T=512/48 heads must be ≤ 700 µs
(leaves room for the Q8 prologue/epilogue and the 25% admission bar), else
kill. Stage 2 (1 day): in-kernel Q8/EF prologue/epilogue, C1.1 oracle.
Stage 3: in-model, KLD, battery.

## 4. Structural levers (lever 5)

### 4.1 S1 — two-chunk, two-stream layer-lockstep prefill pipeline (Sol spec)

**Mechanism.** The outer loop runs every layer of chunk c before chunk c+1
(`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1419-1504`); all
launches go to one `active_stream` (`crates/rdna-compute/src/dispatch.rs:665`).
The only cross-chunk dependencies are per layer: chunk c+1's GDN at layer l
needs chunk c's committed state at layer l; chunk c+1's FA at layer l needs
chunk c's KV rows at layer l. Run chunk c+1 on a second stream one layer
behind chunk c with two events per layer (`hip-bridge/src/ffi.rs:1476,1517`
exist), two PBS scratch sets, and no kernel changes. The GPU then always
has a MAC-bound GEMM (2 blocks/CU, 30.7 KB LDS) available to run under the
other chunk's latency-bound kernels: GDN (1.26 ms, mostly idle SIMDs), FA2
(32 KB LDS → one FA2 WG + one GEMM block per CU fit in 64 KB), conv1d,
quantize, silu, rmsnorm, and each other's grid tails. Bit-exact and
deterministic: same kernels, same inputs, ordering by events; no atomics.
Zero effect at pp512 (one chunk).

**Ceiling.** Bounded per layer by min(GEMM, non-GEMM) and by co-scheduling
efficiency (unknown; GEMM at 1 block/CU beside an FA2 WG runs slower).
pp2048: non-GEMM ≈ 0.84 s of 3.52 → **−150 to −400 ms**; pp32768: FA2+GDN+
other ≈ 38 s beside 43 s of GEMM → **−5 to −15 s** [estimate, low
confidence]. Composes with B6 (a faster FA2 is easier to hide fully).

**Kill (3 h, composer, timing-only microbench).** Launch gate/up IU4
(M=17408,K=5120,N=512) on stream A and FA2 (N=512, ctx 16K) on stream B
concurrently, 20 iterations; compare wall vs the serial sum. ≥0.9× → no
useful co-scheduling, drop S1; ≤0.7× → write the spec. Repeat with q8_fast
on B. Not on the kill list: "prefill graph capture" (≤2%) is a different
mechanism; nothing multi-stream was tried.

### 4.2 S2 — re-open Gate F: chunk 1024, then 2048 (host-only; Sol-lite)

Prerequisites after the oracle fix (§0.1): host gates to 1024/2048; PBS
capacity at 2048 (f32 [2048×17408] intermediates ≈ 143 MB each, fine on
96 GB); GDN cadence decision — keep two 512-token `q8_fast` launches per
1024 to preserve the Q8-EF commit contract (plan §11.4), or accept fewer
commits as an explicit numerics change with the KLD/battery gate (fewer
requants is *less* noise, but decode-visible state changes). Prize
[estimate]: FA2 −6..−7% per launch (−1.9 s at 32K); GEMM tails (z 192→384
blocks, k/v 32→64, gate/up 7→14 rounds) ≈ −0.6 s; launch count halves
(≈ −0.5 s); with A5, weight bytes per token ÷2 (÷8 at 2048 — the
halogen-slope test). **pp2048: −40 to −80 ms at 1024; unknown at 2048 until
A5's byte sensitivity is measured; pp32768: −2.5 to −3.5 s.** Kill: after
the oracle passes, `HIPFIRE_PREFILL_MAX_BATCH=1024` with the gates lifted
must improve pp2048/8192/32768 by ≥3% ABBA (plan §11 gate), else restore.

### 4.3 S3 — gate/up + SiLU/FWHT fusion into the GEMM epilogue (Sol spec)

**Mechanism.** gate and up are separate M=17408 launches whose f32 outputs
(2 × 35.7 MB) exist only to be read by `fused_silu_mul_mq_rotate` (29.2
ms/chunk, near its byte roofline after C2 drops the f32 store). Interleave
gate/up weight rows at load time (prefill-only copy: 2 × 47 MB × 64 layers
= 6 GB, or a stride-aware decode GEMV), so one block holds gate and up
sums for the same rows and its epilogue computes silu(g)·u; the 256-wide
FWHT over the intermediate dim then needs a 256-row block (X strip 43 KB +
Y 9.2 KB = 52 KB, single-plane, 1 block/CU — incompatible with W2's double
plane, so it competes with W2 rather than composing) or a two-phase
rotate. Emit the `block_i4_128`
sidecar from the epilogue. Bit-exact only if the silu/FWHT/quant arithmetic
order is preserved verbatim from `fused_silu_mul_mq_rotate.hip`.
**Ceiling:** silu 29 ms + 71 MB of GEMM output writes/reads ≈ **−30 to −40
ms pp512, −0.12 to −0.16 s pp2048, −1.9 to −2.6 s pp32768** [estimate].
Cost: 5+ days, touches weight layout. Rank below W2/B6/G1 and decide only
after W2's verdict: the two cannot share a tile, and W2's prize is larger.

### 4.4 Producer/quantizer chain after C2, KV-write fusion, conv1d

64 remaining `quantize_int4` calls (9.9 ms/chunk, 1.2%) consume the gated
attention-output and GDN-output epilogues; fusing them saves ≤ −7 ms pp512
[estimate]. conv1d (11.4 ms) is at ~220 GB/s with no fusion partner. The
k/v → rope → Q8 KV write chain sits inside the ≈50 ms "other" bucket; even
fully fused it is <1%. None can move a row; fold into a composer's spare
day only after the levers above are measured. Under S1 all of these hide
under the other chunk's GEMM anyway.

## 5. Ranking (ceiling × confidence), owners, and where the rows land

| # | lever | pp512 | pp2048 | pp32768 | conf. | cost | numerics | owner |
|---|---|---:|---:|---:|---|---|---|---|
| 0 | A0 WMMA-rate calibration (+ no-load / no-fold twins) | sizes 1–3 | | | — | 0.5 day | diagnostic | composer |
| 1 | **B6** FA2 fill MLP (a/b), then (c) | −3..−5 ms | −50..−80 ms | **−11..−16 s** | med | 1 day | bit-exact | composer |
| 2 | **G1** GDN DPP + prefetch + R8 A/B | **−30..−40 ms** | −120..−160 ms | −1.9..−2.6 s | med | 1 day | bit-exact (oracle-proven) | composer |
| 3 | **W2** warp-specialized double-plane IU4 | −100..−300 ms | −0.4..−1.2 s | −6..−19 s | low-med (H1/H2 fork) | 3–5 days | bit-exact | Sol |
| 4 | A5 grid order weights-once | −13..−67 ms | −54..−270 ms | −0.9..−4.3 s | low | 1 h | bit-exact | composer |
| 5 | S1 two-stream chunk pipeline | 0 | −150..−400 ms | −5..−15 s | low | 3 h screen, then 4+ days | bit-exact | Sol |
| 6 | S2 chunk 1024/2048 (F re-opened) | 0 | −40..−80 ms (+A5 upside) | −2.5..−3.5 s | med | 1 h oracle + 1–2 days | bit-exact (cadence kept) | Sol-lite |
| 7 | T0 tails (z M64 twin, qkv merge) | −14 ms | −56 ms | −0.9 s | high | 1 day | bit-exact | composer |
| 8 | B3/alpha-skip/LPT (post-B6 compute) | −1..−2 ms | −15..−30 ms | −3..−5 s | med | 1 day | bit-exact | composer |
| 9 | G2 chunked GDN kernel | −15..−25 ms (post-G1) | −60..−100 ms | −1..−1.5 s | med | 2–4 days | **non-bit-exact, default-OFF `HIPFIRE_GFX1151_GDN_CHUNKED`** | Sol |
| 10 | S3 gate/up+SiLU fusion | −30..−40 ms | −0.12..−0.16 s | −1.9..−2.6 s | low | 5+ days | bit-exact if order kept | Sol |
| 11 | T3 M-tile-64 at 3 blocks/CU | −30..−130 ms | | | low (VGPR ≤128 gate) | 1–2 days | bit-exact | Sol-lite |

**Drop:** W2/W3 prefill weight copy (§1.6, dominated); T2 register-blocked
consumer at 2 blocks/CU (VGPR); T4/T5 (residency); LDS double-buffering at
2 blocks/CU (does not fit); header-slot trim as a standalone; prefill
graph capture (≤2% and the FA2 blob bug); A1 retries; KT64, split-KV, f16
accumulation, s_setprio, NPU, B=16 arms (kill list).

### 5.1 Row reachability (honest)

- **pp512 (−27.5 ms):** cleared by G1 alone, or by A5 ≥ 4%, or by T0 + B6 +
  A5 at their floors. The most likely first clear is G1 + T0 in one
  composer week. No Sol spec required.
- **pp2048 (−634 ms):** composer tier at ceilings: B6 −80 + G1 −160 + T0 −56
  + A5 −270 + S2 −80 ≈ −650 at the *top* of every range, ≈ −300 at the
  middle. **Not reachable without one of: W2 (needs A0 = H1 for the full
  −0.8..−1.2 s; H2 gives −0.4..−0.5 s which still requires the composer
  tier near ceiling), or S1 delivering ≥ −250 ms.** Order: A0 → A5 → (W2 or
  S1 screen) decides the row.
- **pp32768 (−25.1 s):** B6 −11..−16 + S2 −3 + G1 −2 + T0 −1 + A5 −1..−4 =
  −18..−26 s; clearing needs **B6 at ≥ −13 s AND (S1 ≥ −5 s OR W2 under
  either hypothesis)**. If B6's fill-only twin shows the fill under 40% of
  tile time, the 32K row is not reachable from attention and the finding is
  "quadratic exact attention at 12% of fp16 peak plus the 40-CU int4
  denominator" — unless S1's co-scheduling screen passes, which is then the
  only remaining path.

### 5.2 First week, in order

1. A0 + no-load/no-fold twins (day 1) — fixes every estimate in §1.
2. A5 grid swap (1 h) and the F1 oracle fix + rerun (1 h) — both are
   one-line changes with row-relevant answers.
3. B6 ISA check + twin + oracle (day 2).
4. G1 DPP twin + oracle (day 3), T0 (day 4).
5. S1 co-scheduling screen (3 h) — its result and A0's decide whether the
   Sol spec written next is W2 or S1.

Everything above is bit-exact except G2 (named switch) and the optional
S2 cadence change; arch-gate all of it to exact gfx1151 (`gfx1100` shares
the IU4 and FA2 sources: `gemm.rs:19049`, `attention.rs:3265-3268`), and
measure nothing except with fresh-process ABBA per
`docs/methodology/perf-benchmarking.md`.
