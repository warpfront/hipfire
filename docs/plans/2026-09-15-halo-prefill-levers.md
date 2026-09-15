# gfx1151 (Strix Halo) batched-prefill lever space — Qwen3.8-27B MQ4V2-xt

Date: 2026-09-15. Worktree `/home/kaden/ClaudeCode/warpfront/wt-lloyd`, read-only.
**No GPU program was run; no source was edited.** Every magnitude is an
**[estimate]** derived from the caller-supplied profile/bench numbers plus
arithmetic on cited source; nothing here is a measurement. Not a plan of
record — a Muse/Sol seat converts survivors into specs.

In-flight work this document must not collide with: `Liu4K1` landed an
additive iu4 **LUT twin** (`HIPFIRE_MMQ_IU4_LUT`, `*_lloyd` symbols) of
`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip` — same LDS budget,
grid, block, occ3 bounds; only the fragment decode differs. A-tier findings
below transfer to that twin (its extra VALU sits in the consumer, so the
bank/pipeline analysis shifts slightly in its favor to re-verify).

## 0. Denominators (what a lever can at most attack)

Per-chunk N=512 everywhere on gfx1151: `prefill_max_batch_for_model` returns
512 for gfx1151 + all-MQ4V2 (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:455-458`),
so pp2048 = 4 chunks and pp32768 = 64 chunks, each at N=512.

Measured totals (caller-supplied): pp512 = 878 ms GPU / ~898 ms wall (570 t/s);
pp2048 = 3.70 s (554 t/s); pp32768 = 86.7 s (378 t/s). Halogen bar: 620 t/s
pp512 (−53 ms needed), 710 t/s pp2048 (−815 ms), 566 t/s pp32768 (−28.8 s).

Bucket budget [estimate, scaled linearly/quadratically from the pp512 and
pp8192 profiles; sums cross-check within ~5% of measured totals]:

| bucket | pp512 | pp2048 | pp32768 | scaling |
|---|---:|---:|---:|---|
| iu4 GEMMs (set 272×1729 µs + add 128×1685 µs per chunk) | 686 ms | 2.74 s | 43.9 s | linear |
| FA2 attention (16 calls/chunk; 512 µs at ctx≤512, 7255 µs avg at 8K) | 8 ms | 0.15 s | ~29.8 s | quadratic |
| gated_delta_net_q8_batch_seq (48×1340 µs per chunk) | 64 ms | 0.26 s | 4.1 s | linear |
| quantize_int4_mmq_ds128 (256×154 µs per chunk) | 39 ms | 0.16 s | 2.5 s | linear |
| fused_silu_mul_mq_rotate_batched (64×457 µs per chunk) | 29 ms | 0.12 s | 1.9 s | linear |
| conv1d_silu_split_f32_n (48×237 µs per chunk) | 11 ms | 0.05 s | 0.7 s | linear |
| tail (rmsnorm/rope/embed/launch gaps) | ~40 ms | ~0.2 s | ~3.8 s | linear |

Model shape ground truth (verified `config.rs:1339-1350`, `load.rs:3500`,
`load.rs:3544-3549`): hidden 5120, intermediate 17408, 64 layers = 48 LA + 16 FA;
LA qkv M=10240, z M=6144, beta/alpha M=48; FA q M=12288 (48×128×2 gated),
k/v M=1024; gate/up M=17408; down K=17408 M=5120; wo/o M=5120 K=6144.
272 occ3 sets/chunk = 96 qkv+z + 128 gate/up + 48 q/k/v; 128 adds = 64 down +
48 wo + 16 o. ✓ matches the profile counts exactly.

Peak yardsticks [estimate, caller-consistent]: int4 WMMA 40 CU × 2.9 GHz ×
1024 ops/clk = 1.19e14 ops/s; fp16 WMMA 512 FLOP/clk/CU = 5.94e13 FLOP/s
(with this peak the measured 7255 µs FA2 call at mean ctx ≈4608 = 13.5%,
consistent with the caller's "~12% of fp16 peak").

## A. iu4 MMQ GEMM (44% of int4 peak on the dominant shape; 30% blended)

Kernel facts (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`):
tile 128×128 (`:43-44`), block 256 = 8 wave32 (`:45-46`), LDS 31744 B
(`:30`) → **2 blocks/CU on a 64 KB LDS; 80 blocks resident on 40 CUs**.
The kb loop (`:321-345`) is strictly serialized: load weights→LDS
(`load_iu4_tile`, `:207-243`), `__syncthreads` (`:330`), dot, sync (`:332`),
load Y half 2, sync (`:341`), dot, sync (`:343`) — **no double buffering
anywhere**; four barriers per 256-K group, 20 groups at K=5120.
Weight loads are raw nibble words, no expand VALU (`:204-225`).
Consumer (`vec_dot_i4_x128`, `:247-296`): per 16-K t-step, two 8-byte LDS
reads per operand side (`:276-279`) feeding two
`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32` (`:280-281`).

Grid math at N=512 (`gemm.rs:18562-18578`: `row_tiles=ceil(M/128)`,
`batch_tiles=512/128=4`, block [32,8]; occ3 select `gemm.rs:18534-18539`):

| call | M×K | blocks | waves of 80 | last-wave fill | ideal µs/call* |
|---|---|---:|---:|---:|---:|
| gate, up | 17408×5120 | 136×4 = 544 | 6.8 → 7 | 64/80 = 80% | 768 |
| FA q | 12288×5120 | 96×4 = 384 | 4.8 → 5 | 64/80 = 80% | 542 |
| LA qkv | 10240×5120 | 80×4 = 320 | 4.0 | 100% | 452 |
| LA z | 6144×5120 | 48×4 = 192 | 2.4 → 3 | 32/80 = 40% | 271 |
| down | 5120×17408 | 40×4 = 160 | 2.0 | 100% | 768 |
| wo / o | 5120×6144 | 40×4 = 160 | 2.0 | 100% | 271 |
| FA k, v | 1024×5120 | 8×4 = 32 | 0.4 → 1 | 32/80 = 40% | 45 |
| beta, alpha | 48×5120 | — | diverted to MW4 f16 small tail (`gemm.rs:27987-28004`) | — | — |

\* ideal = 2·M·K·512 / 1.19e14 [estimate]. Structural tail loss (fraction of
each call's duration = (⌈w⌉−w)/⌈w⌉) totals **3.2% of ideal-time-weighted GEMM
work**; against measured times ≈ 1–3% [estimate].

Clock-and-CU-scaled XTX comparison: same gate/up shape on XTX is 588 µs
(96 CU, ~2.5 GHz); Halo-scaled expectation is 588×(96/40)×(2.5/2.9) ≈ 1216 µs;
measured 1729 µs → Halo carries an extra **~1.42×** beyond scaling [estimate].
Per-block WMMA-issue floor at 8 clk/WMMA/SIMD32-pair accounting is ≈770 µs
for this shape on 40 CUs → the kernel spends ~56% of its time NOT issuing
WMMA on Halo vs ~37% on XTX [estimate]. That delta is the A-tier prize.

### A1 — Software-pipeline the kb loop (prefetch kb+1 during compute)

**Mechanism:** gfx11 has no cp.async; do global→VGPR loads for iteration
kb+1 *before* `vec_dot_i4_x128` of kb, then VGPR→LDS at the sync point.
This overlaps the ~35.8 KB/block/kb global stream (17.4 KB weights `:220-225`
+ 2×9.2 KB activations `:324-329`) with 1024 iu4 WMMAs of compute, directly
attacking the LPDDR5X latency exposure that the XTX (GDDR6) mostly hides.
Register cost ≈ 26 VGPRs/thread for the staged tiles on top of `sum[64]`
(`:319`) — fits the occ3 budget (~168 VGPR cap at `__launch_bounds__(256,3)`,
`:374-380`) [estimate].
**Ceiling:** closing Halo→XTX-efficiency parity (2.25× → 1.58× overhead
[estimate]) ⇒ GEMM ×0.70: **−205 ms pp512, −820 ms pp2048, −13.1 s pp32768**.
Clears the pp512 bar alone; with B-tier partial wins clears pp2048.
**Kill experiment:** throwaway `#if HIPFIRE_IU4_PIPE1` variant + single-shape
microbench (M=17408 K=5120 N=512, 100-iteration median) vs base; abandon if
<10% on the shape. Cost ≈ 1 day. Step 0: read VGPR/scratch from the JIT
build log to confirm no spill (occupancy-AB methodology,
`docs/plans/hfq4g256-occupancy-ab-results.md:57-70`).
**Kill-list check:** not iu8-register-resident (different structure), not an
occupancy knob, not an env arm. Untried.

### A2 — X-tile LDS stride 44 → 42 (kill the 2-way bank conflict; *shrinks* LDS)

**Mechanism:** consumer A-side reads are `int32x2` LDS loads at dword offset
`lane_row×44 + h*16 + t*4` (`:274-279`), 16 lanes. Bank = (44·lane + c) mod 32
= (12·lane + c) mod 32 → lanes ℓ and ℓ+8 collide on every b64 load
(12·8 = 96 ≡ 0 mod 32) → **every A-side consumer LDS load is 2-way
bank-conflicted**. Row stride must stay even (b64 alignment) and ≥40 words
(32 nibble + 8 header, `:47-50`); stride 42 gives (10·lane) mod 32 — all 16
lanes distinct, conflict-free — and *drops* LDS to 30720 B, preserving the
<32 KB dynamic-LDS property and 2 blocks/CU. (46 also works mathematically
but lands exactly on 32768×2 = 65536 — exact-fit risk; 42 is the pick.)
Y-side stride 18 is already conflict-free ((18·lane) mod 32 distinct).
B-side observation: WMMA operand lanes r and r+16 load byte-identical
fragments (`:26-28`), i.e. half the LDS reads are duplicates — inherent to
the gfx11 WMMA lane map, not fixable without a different fragment layout.
**Ceiling:** A-side LDS issue halves; LDS is *not* the primary limiter here
(~1.7–3.5 K clk/kb/SIMD vs ~8.2 K WMMA clk [estimate]) → **≤3–6% of GEMM:
20–40 ms pp512, 80–160 ms pp2048, 1.3–2.6 s pp32768** [estimate].
**Kill experiment:** one-const change (`IU4_TILE_X_K` `:47` + mirror
`gemm.rs:18568`) + in-model `profile_prefill_qwen35`
(`saddle-lab/examples/profile_prefill_qwen35.rs`) A/B on the `full_set_occ3`
symbol timer. ≈1 hour. If ≤1% on the symbol, drop.
**Kill-list check:** "MMQ round-2 x3" was structure rewrites; nobody touched
the 44-word pad. Untried.

### A3 — Grid-tail recovery: z (M=6144) half-height tile; k/v launch merge

**Mechanism:** two cheap shapes pay the 40%-fill tail from the table above:
z (192 blocks, 3rd wave 40% full, 48 calls/chunk) and k/v (32 blocks, 40% of
one wave, 32 calls/chunk). Options: (a) M-tile-64 variant for M=6144
(LDS (64×18+128×42)×4 ≈ 26.2 KB → 2 blocks/CU, 96×4 = 384 blocks = 4.8
waves of 80 at half block duration → tail 4% instead of 20%); (b) merge
q/k/v into ONE launch (M=14336 → 112×4 = 448 blocks = 5.6 waves) — needs a
multi-source selector, the exact mechanism the gfx12 fused kernels and the
in-flight gfx11 MMQ-LUT launcher family already use for per-slab LUT select
(precedent: `docs/plans/2026-09-15-lloyd-gfx11-mmq-lut.md:140-155`).
**Ceiling:** structural-loss math above: **7–21 ms pp512, 27–85 ms pp2048,
0.4–1.3 s pp32768** [estimate; range spans ideal-vs measured-time weighting].
**Kill experiment:** (a) is a template param + one launcher arm — microbench
the z shape; abandon if <15% on that call. (b) defer unless (a) lands AND the
profile still shows k/v tails; the merge is a bigger diff for a smaller slice.
**Kill-list check:** not an occupancy/env retread; tile-height variants for
the *F16* residual path won on gfx1100 (MW4/MW8,
`docs/perf-checkpoints/2026-08-23-qwen38-gfx1100-mq4v2-residual-multi-wave.md:260-266`)
but were never applied to the iu4 kernel or to gfx1151.

### A4 — occ2-vs-occ3 re-test on gfx1151 (cheap screen, likely nil)

**Mechanism:** occ3 (`gemm.rs:18536-18537`) requests 3 blocks/CU but LDS
caps residency at 2 (31744×3 = 95232 > 65536); the only real effect is the
VGPR budget (~168 vs ~256). #766's 450→581 win bundled occ3 with the iu4
route itself (CHANGELOG.md:8); whether the occ2 symbols (`:377-378`) were
ever A/B'd *on gfx1151* at this head is unrecorded in what I read.
**Ceiling:** 0–3% of GEMM (0–20 ms pp512) [estimate].
**Kill experiment:** swap the two symbol strings at `gemm.rs:18536-18537`,
profile. 30 min. **First** verify from #766's ledger that the Halo arm never
A/B'd occ2 — if it did, this lever is dead on arrival.
**Kill-list check:** this *is* #766-adjacent; included only as an
unrecorded-arm check, not a re-proposal.

### A5 — Column-fastest grid order for L2 weight reuse

**Mechanism:** grid is `[row_tiles, batch_tiles]` (`gemm.rs:18577`) with
`row0 = blockIdx.x*128` (`:308-309`) — the 4 column tiles sharing one weight
row-band are 136 blocks apart in launch order, never co-resident, so each
weight tile is fetched up to 4× per call (per-call global traffic ≈ 390 MB
for gate/up: ~190 MB weights×4 + ~200 MB activation re-reads [estimate]).
Swapping grid dims (and `:308-309`) puts the 4 siblings adjacent → one DRAM
fetch, 3 L2 hits. DRAM weight floor per chunk: 59 ms once vs ≤236 ms at 4×
[estimate] — mostly hidden under compute today, but it also feeds the
latency stalls A1 attacks.
**Ceiling:** **0–5% of GEMM** at pp512 (0–35 ms); up to −700 ms pp2048,
−2.4 s pp32768 if the re-reads are partially exposed at larger footprints
[estimate; wide because L2 size/hit behavior on gfx1151 is unmeasured here].
**Kill experiment:** one-line grid swap + profile A/B. ≈1 h. If <1%, drop.
**Kill-list check:** launch-order swizzling is not on the list.

### A6 — Considered and dropped

- **wave64** — the WMMA builtins are `_w32` (`:280-281`); a wave64 body is a
  new kernel, and the only wave64 knob in-tree is decode-side
  (`developer.gfx1151_gate_up_wave64`, docs/CONFIG.md:176). Drop.
- **M-tile-256 iu4** — LDS (128×18+256×44)×4 ≈ 54 KB → 1 block/CU, halves
  residency and exposes every barrier. Drop.
- **Y-store vectorization** (`:360-368`, stride-M f32 stores) — 64 KB/block
  of stores vs ~390 MB of loads; noise. Drop.

## B. FA2 prefill attention (12% of fp16 peak; ~51% of pp32768)

Kernel facts (`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip`): one WG = 1 KV
head × 8 positions = 48 rows on 3 compute waves + 1 helper wave
(`:7-17`); KT32 pinned — KT64 **lost the ABBA and was deleted**
(`crates/rdna-compute/src/attention.rs:3919-3927`); grid
`[ceil(batch/8), 4]` (`attention.rs:3929`), `__launch_bounds__(128,1)`
(`:502`), LDS 32768 B → 2 WGs/CU; batch ≤ 512 gate (`attention.rs:3889-3896`);
0 spills measured on Halo (CHANGELOG.md:11).
Per 16-key subtile per wave: QK = 16 dc × (8 dword LDS reads for the K
fragment + 4 global float4 for Q + 1 WMMA + **one full-memory compiler
fence**, `:364-391`), softmax + 8-shuffle P reconstruction (`:397-442`),
PV = 16 dc × (8 dword reads + 1 WMMA + fence, `:455-472`). That is **256 LDS
instructions against 32 WMMAs per subtile** — 8:1 — plus 32 memory clobbers.
The gfx12 origin (`attention_q8_0_fa2_gqa.gfx1201.hip:22-33`) needs no P
interleave and no pair reductions (lane owns the whole query row; "zero
shuffles anywhere"); gfx11 pays lane-pair duplication (2× operand reads,
`:26-28`) plus reconstruction VALU. gfx1201 also has split-KV
partial/merge (`gfx1201.hip:14-15,74-93`); gfx11 is direct-only
(`gfx11.hip:17`).

### B1 — b64 LDS loads + even-granular swizzle (the big one)

**Mechanism:** each half16 K/V fragment is 8 *consecutive* dwords at
`(dc*8+i+rot) & 127` (`:364-372`, `:455-467`) with rotation
`rot = S(k&15) = 8*(c>>2)+(c&3)` (`:51-55`, `:99-102`) — odd for odd k, so
only dword loads are used. Redefine the swizzle at 2-dword granularity
(`rot' = 2·perm(k&15)`, a permutation of even residues 0..30): 16 lanes × 2
banks = all 32 banks exactly once → conflict-free `ds_read_b64`, halving
256→128 LDS instructions per subtile. (b128 is impossible conflict-free:
16 lanes × 4 banks = 64 > 32 bank-slots — state this so nobody retries it.)
Producer stores (`:248-249`, `:289-291`, `:335`) rotate identically — same
change both sides, bit-exact (pure address permutation of identical bytes).
**Ceiling:** if LDS-issue is ~40–50% of subtile time [estimate], attention
×0.75–0.8: **−2 ms pp512, −30–45 ms pp2048, −6.0–7.5 s pp32768** [estimate].
**Kill experiment:** patch swizzle + loads, run the existing 7-shape oracle
(CHANGELOG.md:11 cites it) for bit-identity, then profile pp8192; abandon if
<10% on the attention symbol. ≈1–2 days.
**Kill-list check:** FA2 levers are untouched by the GEMM kill list; KT64
(the adjacent knob) is already dead and this is not it.

### B2 — Delete the two per-dc-loop memory clobbers

**Mechanism:** `asm volatile("" ::: "memory")` at `:390` and `:471` runs 32×
per subtile per wave to bound register hoisting. It forbids the compiler
from batching the dword loads across dc iterations — on the
higher-latency APU this serializes exactly the loads B1 is trying to widen.
If VGPRs spike when removed, the scoped alternative is
`__builtin_amdgcn_sched_barrier()` (no memory semantics).
**Ceiling:** **5–15% of attention** (0.4–1.2 ms pp512, 7–22 ms pp2048,
1.5–4.5 s pp32768) [estimate]; multiplicative with B1, not additive.
**Kill experiment:** delete both lines, read VGPR/scratch from the JIT log
(gate: 0 spills, same as CHANGELOG.md:11's acceptance), 7-shape oracle,
profile. ≈1 h. If spills appear and sched_barrier doesn't contain them, drop.
**Kill-list check:** untouched area.

### B3 — Pre-convert Q to f16 once per chunk (bit-exact)

**Mechanism:** Q is f32 in global and converted per subtile per lane
(`:375-385`: 4 float4 + 16 cvt per dc, i.e. ~256 VALU converts + 64 loads
per subtile per lane), re-read from L1 every subtile by design (`:143-148`).
A pre-pass emitting `q_f16` once per chunk loads 32 B instead of 64 B per
fragment and deletes all per-subtile converts. f32→f16 RN is the same
rounding the kernel applies today, so operands are bit-identical.
**Ceiling:** **5–10% of attention** [estimate].
**Kill experiment:** q_f16 scratch + loader arm behind a new symbol; oracle
bit-identity must hold exactly (any diff = rounding bug, abandon). ≈1 day.
**Kill-list check:** not the killed FP8/FP16 *weight* experiments; this is an
activation layout change with a bit-exactness proof obligation.

### B4 — Helper-wave prefetch pipeline (fill/compute overlap)

**Mechanism:** wave 3 only fills (`:132`, `:208-338`); during the two
compute subtiles it idles. Restructure so wave 3 stages the *next* KT tile's
K/V into VGPRs while waves 0–2 compute, committing to LDS at the turnover
sync (`:477`). Fill is KT×8 K slots + KT×2 V pairs of byte-gather dequant
(`:256-338`) — a large fraction of tile time at long ctx [estimate].
**Ceiling:** up to **~25% of attention at long ctx**: ~7 s at pp32768
[estimate; weakest confidence in this doc — fill/compute split unmeasured].
**Kill experiment:** instrument first (two counters or a fill-only variant
timed standalone at ctx 8192); if fill <20% of tile time, drop before
building anything. ≈2–4 days if it survives.
**Kill-list check:** the gfx12 `HIPFIRE_FLASH_PREFILL_PREFETCH_V` knob
(docs/env-vars.md:152) is the incumbent WMMA kernel, not this FA2 path.

### B5 — Considered and dropped

- **KT64** — deleted after losing the ABBA on both gfx1100 and gfx1151
  (`attention.rs:3919-3921`). Dead; do not re-propose.
- **Split-KV (partial+merge) port** — prefill grid is already 256 WGs at
  pp512 (3.2 waves of 80) and 16K WGs at pp32768; parallelism is not the
  deficit, total quadratic work is. Helps decode/small-batch, not this target.
- **f16-accumulator WMMA** — 2× WMMA rate but breaks the f32-accumulator
  bit-exactness contract (`:57-58`) and the kernel is LDS/issue-bound, not
  WMMA-bound. Drop.

## C. Middle tier (linear: launches and bytes)

### C1 — Chunked-parallel Q8-EF GDN (the validated, unbuilt kernel)

**Mechanism:** `gated_delta_net_q8_batch_seq` walks 512 tokens serially per
launch (token loop `gated_delta_net_q8.hip:78-112`; grid `[48, 32]`, block 32
= one wave, `norm.rs:3064-3067`) at ~39 GB/s effective [estimate] — pure
latency serialization, 1.8 TFLOP/s ≈ 1.5% of peak. The chunked form is
**math-validated at 1.3e-15** with the Q8 error-feedback boundary protocol
proven (`docs/plans/chunked-gdn.md:3-8,125-155`); the F32-state variant
already exists in-tree and is dispatched under `HIPFIRE_GDN_CHUNKED`
(`prefill.rs:5194-5207`, `norm.rs:63-81`); the Q8-EF HIP kernel is the named
gap ("what remains is the HIP kernel", chunked-gdn.md:8).
**Ceiling:** GDN 64 ms pp512 / 0.26 s pp2048 / 4.1 s pp32768 → chunked at
CS=64 costs ~25 MB state re-reads + trivial GEMM per layer [estimate] ⇒
**−40 to −48 ms pp512, −170 to −200 ms pp2048, −2.7 to −3.1 s pp32768**
[estimate]. Also cheapens spec-verify (out of scope here but real).
**Kill experiment:** the plan's own staging (chunked-gdn.md:193-205):
F32-kernel parity gate < 1e-4 vs `gated_delta_net_f32_batch_seq`, then Q8-EF
rel-err in the 7e-3 band, then in-model A/B. A parity failure at stage 1
ends it cheap.
**Cost:** the only multi-day kernel program in this document.
**Kill-list check:** not on it; the sequential kernel is "sequential by
construction" (`docs/plans/gfx906_prefill_kernels.md:213-216`) — nobody
failed building this, it just isn't built.

### C2 — Fuse int4 quantization into its producers (bytes + launches)

**Mechanism:** the naive form of "quantize once per layer-input" is
**already taken** — `ensure_int4_mmq_x` is called once per family input and
shared across qkv/z/beta/alpha (`gemm.rs:27984`), q/k/v (`gemm.rs:28526`),
gate/up (`gemm.rs:29634`). Do not re-propose that. The residual lever:
(a) `quantize_int4_mmq_ds128` re-reads the f32 tensor its producer just
wrote — 256 calls/chunk, ~12–37 MB read each, at ~78 GB/s effective
[estimate, from profile bytes/time], well under the ~234 GB/s the silu
kernel achieves on the same machine; (b) `fused_silu_mul_mq_rotate_batched`
writes 35.7 MB of f32 (down-proj input) that *only* the quantizer reads
(bytes formula `gemv.rs:2949`; grid `[K/256, N]` block 32 `gemv.rs:2956-2959`).
Fuse: silu workgroups own exactly one 256-group = two 128-K quant groups per
token (`fused_silu_mul_mq_rotate.hip:32-46`) — emit two `block_i4_128`
directly from registers beside the f32 store (or drop the f32 store where
the contract allows; the f16-emit sibling precedent is
`fused_silu_mul_mq_rotate_f16.gfx1100.hip:12-16`). Same for the rmsnorm
producer feeding qkvza/qkv/gate_up (K=5120, 128 calls/chunk). Bit-exact by
construction if the quantizer consumes the identical f32 register value it
would have reloaded (the MSE-clip math at `iu4.gfx11.hip:96-117` is
unchanged). o-proj/wo inputs (FA2/GDN epilogues, 64 calls/chunk) are the
messier second tranche — scatter-written outputs; defer.
**Ceiling:** removes 192 of 256 quantize calls (~30 of 39 ms pp512) and ~32%
of silu write bytes (~9 ms) ⇒ **−30 to −39 ms pp512, −120 to −155 ms pp2048,
−1.9 to −2.5 s pp32768** [estimate].
**Kill experiment:** one-layer differential: run silu→quantize→down chain,
assert the fused variant's `block_i4_128` buffer is bit-identical
(`to_bits`), then profile. If not bit-identical, the fusion premise is wrong
— abandon. ≈1–2 days per producer.
**Kill-list check:** fusion of *existing* kernels, no new numerics path.

### C3 — Why is quantize at ~78 GB/s? (probe before building C2)

**Mechanism:** per-lane critical path is amax-reduce + 8 MSE candidates ×
(4 rint/clamp/FMA + 5-shuffle reduce) + final reduce (`iu4.gfx11.hip:91-129`)
over grid `[(K+1023)/1024, N]`, block 256 (`scratch.rs:1261-1277`). Issue
math says ~7 µs/call of ALU [estimate]; measured 154 µs — the gap is
unattributed (latency chain? write-allocate? grid-y token-major ordering?).
**Ceiling:** if ALU/latency-bound, a restructure (fewer shuffle trees via
DPP — the `HIPFIRE_GFX1151_GDN_DPP_REDUCE` pattern at
`gated_delta_net_q8_fast.hip:51-84` is the in-tree precedent — or
candidate-parallel layout) could halve the *remaining* quantize time even
after C2: **−5 to −20 ms pp512, −20 to −80 ms pp2048, −0.3 to −1.3 s
pp32768** [estimate].
**Kill experiment:** throwaway 1-candidate build (numerics break; diag
only). Time collapses → ALU-bound, build the restructure. Time unchanged →
BW/latency-bound, fold findings into C2 and stop. ≈1 h.
**Kill-list check:** nobody profiled this kernel; it was written for the
oracle-matching contract, not speed.

### C4 — Prefill chunk > 512 on gfx1151 (`HIPFIRE_PREFILL_MAX_BATCH`)

**Mechanism:** knob exists (`prefill.rs:418-422`), gfx1151 default 512
(`prefill.rs:457`). N=1024 per chunk: halves all per-chunk launch counts,
fixes the z tail (48×8 = 384 blocks = 4.8 waves vs 2.4→3), halves KV
re-read passes per token. **Blocker:** the FA2 launcher refuses batch > 512
(`attention.rs:3889-3896`); exceeding it silently routes attention back to
the incumbent that FA2 beat by 1.8× at 8K on Halo (CHANGELOG.md:11) — so
this lever is *negative* until the FA2 envelope is extended (kernel has no
structural 512 limit; the gate is the validated-envelope claim). The
2026-08-23 chunk>256 rejection
(`docs/perf-checkpoints/2026-08-23-qwen38-gfx1100-prefill-screen.md:178-181`)
was gfx1100 / F16-WMMA era — pre-X128, pre-iu4, pre-FA2 — and does **not**
cover this stack; gfx1151's 512 default was itself a later measurement.
**Ceiling:** launch amortization + z-tail + tail-kernel fixed costs:
**0 pp512 (single chunk regardless), −40 to −75 ms pp2048, −0.9 to −1.7 s
pp32768** [estimate].
**Kill experiment:** `HIPFIRE_PREFILL_MAX_BATCH=1024` + local gate relaxation
to 1024 (batch-envelope claim extends; oracle re-run mandatory), bench
pp2048/pp32768. ≈2 h. If <1%, drop.
**Kill-list check:** the rejected screen is a different arch/stack; stated
above so nobody cites it against this.

### C5 — Considered and dropped

- **conv1d_silu_split_f32_n** (48×237 µs) — ~220 GB/s effective [estimate],
  near the BW roof; no fusion partner (its consumers are the GDN and the
  qkv GEMM, both structurally elsewhere). Drop.
- **Prefill-chunk graph capture** — GPU busy ≈ 878/898 ms at pp512 ⇒ ≤2%
  ceiling [estimate]; GDN stochastic-rounding frame counter is host-advanced
  (`norm.rs:3031-3035`) so capture changes dither vs eager (EF deterministic
  mode is capture-safe; stochastic is not). Real work, tiny prize. Drop.

## Ranking (ceiling × confidence)

1. **A1 pipeline the kb loop** — only lever that single-handedly clears a
   halogen row (pp512: −205 ms vs −53 needed; pp2048: −820 ms vs −815
   needed) [estimate]. Confidence: medium — the latency-exposure diagnosis
   is an inference from the serialized loop + the Halo/XTX residual, not a
   measurement. The 1-day microbench gates it.
2. **B1+B2 FA2 LDS width + fence removal** — largest pp32768 lever
   (−7.5 to −12 s combined [estimate]), cheap kill gates, bit-exact by
   construction. Confidence: medium (8:1 LDS:WMMA instruction ratio is
   counted from source; the time fraction is the inference).
3. **C1 chunked Q8-EF GDN** — −2.7 to −3.1 s at pp32768, −40+ ms everywhere
   [estimate]; math pre-validated; the cost is days, not risk. Confidence:
   high on direction, medium on magnitude (state re-read traffic).
4. **C2 producer-fused quantize** — −30 to −39 ms pp512, ~4% everywhere,
   bit-exact by construction. Confidence: high on bytes, medium on the 78
   GB/s attribution (C3 probe first — it's 1 h).
5. **A2 LDS stride 42** — do it in the first hour regardless of rank: the
   A/B is nearly free and the sign of the result also informs the LUT twin
   K1 just landed.
6. **A3 z-tile / k/v-merge** — small, real, structural; fold into whichever
   A-tier kernel diff survives.
7. **C4 chunk 1024** — contingent on extending the FA2 batch envelope;
   revisit after B-tier lands.

**Drop without further discussion:** A4 (only if #766's Halo arm already
A/B'd occ2 — check first, else run the 30-min screen), A6 (wave64 / tall
tiles / Y-stores), B5 (KT64, split-KV, f16 accumulators), C5 (conv1d,
capture). **A5** is the honest borderline: keep only because it's a 1-h
experiment; expect nil.

**Where the bar actually lands:** pp512 clears with A1 alone, or with
A2+A3+C2+C1 combined (~−100 to −130 ms vs −53 needed) if A1 dies in its
microbench. pp2048 needs A1 *or* nearly everything else at ceiling.
pp32768 (−28.8 s needed) does **not** clear on any single lever: the
honest combination is A1 (−13.1) + B1/B2 (−7.5 to −12) + C1 (−2.9) + C2
(−2.2) ≈ −25.7 to −30.2 s [all estimates] — i.e. the 32K row is achievable
only if both A1 and B-tier land near their ceilings, and if either dies,
the correct answer for 32K is "none of these clear the bar" and the
denominator itself (quadratic exact attention at 40 CUs) is the finding to
report back.

## Verification inventory (what each kill experiment needs that already exists)

- In-model per-symbol profiler: `saddle-lab/examples/profile_prefill_qwen35.rs`
  (produced the numbers this doc is sized against).
- FA2 7-shape oracle + "0 spills" acceptance precedent: CHANGELOG.md:11.
- Bit-identity gates for GEMM/LUT work: LUT GEMV differential precedent in
  `docs/plans/2026-09-15-lloyd-fp8-prefill-gap.md:292-302`.
- GDN parity harnesses + numpy refs: `docs/plans/chunked-gdn.md:214-226`.
- VGPR/spill measurement method: `docs/plans/hfq4g256-occupancy-ab-results.md:57-70`.
