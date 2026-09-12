// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

// Parameterised LDS-staged WMMA GEMM: F16 weight × F16 input → F32 output,
// fused bias. gfx1100+ only (RDNA3 / RDNA3.5 wave32 WMMA, 16×16×16 tiles).
//
// Same operand contract as gemm_f16_x_f16_wmma_lds, plus a row pitch on the two
// inputs (see ROW PITCH below):
//   A    : [M, lda] F16 row-major  (weight),      lda >= K
//   X    : [B, ldx] F16 row-major  (activations), ldx >= K
//   Y    : [B, M] F32 row-major
//   Bias : [M]    F32, added to every row of Y when has_bias != 0
//   Y[b, m] = bias[m] + sum_k A[m, k] * X[b, k]
// Only k < K of each input row is ever addressed; lda == ldx == K is the packed
// contract and reproduces the old kernel exactly.
//
// PITCH PRECONDITION: lda and ldx must each be a MULTIPLE OF 16 elements.
// `wlds_src` hands the staging loop a `half16_t*`, a 32-byte vector load, whose
// address is `base + row·ld + k0 + kt·16` elements. `base` is 256-byte aligned
// and every other term is a multiple of 16, so the load is 32-byte aligned iff
// `ld % 16 == 0`. K is already a multiple of KS ∈ {32, 64}, so the packed
// contract satisfies this automatically and only a PAD that is not a multiple
// of 16 can break it. The Rust entry points assert it; do not relax that
// without also giving `wlds_src` an unaligned load path.
//
// ROW PITCH — the largest single effect in this file
// --------------------------------------------------
// A block stages BM rows of A and BN rows of X per K stage, concurrently, each
// row a short contiguous run (KT half16 = 32·KT bytes) at a stride of the row
// pitch. When that pitch is a multiple of 1024 BYTES, all BM + BN of those
// concurrent runs land on the same small set of DRAM channels and camp there.
// Every FLUX.1-dev K is such a pitch: 3072, 12288 and 15360 halves are 6144,
// 24576 and 30720 bytes, all multiples of 1024.
//
// MEASURED, gfx1150, bench_gemm_wide_lds, WARM=1 REPS=5, lock held, `pgrep
// rustc` empty, tile 256×256/64×64 k64_p, M = 3072 B = 4608. Sweeping K alone
// (the FLOP count moves with K, so read the TFLOP/s column, not the ms one):
//
//   K       pitch B   pitch = 2^v·odd   TFLOP/s   % of the 14.8 peak
//   3008    6016      v = 7             11.53     78 %
//   3072    6144      v = 11             7.35     50 %   <- FLUX
//   3136    6272      v = 7             10.71     72 %
//   3200    6400      v = 8             12.56     85 %
//   3328    6656      v = 9             10.06     68 %
//   3584    7168      v = 10             7.13     48 %
//   12224   24448     v = 7             13.26     90 %
//   12288   24576     v = 13             7.52     51 %   <- FLUX
//   12352   24704     v = 7             12.16     82 %
//   15360   30720     v = 11             7.63     52 %   <- FLUX
//
// The cliff is at v >= 10, i.e. a pitch that is a multiple of 1024 B, i.e. a K
// that is a multiple of 512 halves. Below it the kernel runs at 68-90 % of WMMA
// peak; on it, at 48-52 %. Every FLUX K sits on it, and K ± 64 does not.
//
// THE MECHANISM IS ARCHITECTURAL, THE THRESHOLD IS NOT. v >= 10 is gfx1150's
// number and it is a function of that part's channel interleave. CONTROLLER,
// gfx1151, same sweep, same tile:
//
//   K       pitch B   v      ms      TFLOP/s   % of peak
//   12224   24448     7      10.28   33.66     66 %
//   12288   24576     13     21.27   16.36     32 %   <- FLUX, deep in it
//   12352   24704     7      10.43   33.53     66 %
//   12416   24832     8      12.85   27.35     54 %
//   15360   30720     11     18.34   23.71     47 %   <- FLUX, shallower
//   15424   30848     7      12.28   35.57     70 %
//
// Same shape of curve, different depths: gfx1151 falls off a cliff at v = 13
// (K = 12288 loses 2.06× against its neighbours) and only part-way at v = 11
// (K = 15360, 1.50×), where gfx1150 treats v = 11 and v = 13 alike. That is why
// gfx1151's per-step win is 1.24× and gfx1150's is 1.53×: on gfx1151 only one
// of the three FLUX K values is deep in the cliff. v = 7 is the best cell on
// both parts, and PAD = 64 puts all three FLUX K there — that is now a
// measurement on both archs, not an inference. Re-run this sweep with
// bench_gemm_wide_lds `CUSTOM=` on any NEW arch before choosing its pad.
//
// The fix is `lda`/`ldx`: the same K, summed in the same order, with the rows
// laid out 64 elements further apart. The parity gate covers it (suite 4, pad
// filled with poison so an over-read cannot pass), and it is bit-exact — the
// pitch moves where the operands live and changes nothing numerically.
//
// MEASURED with K held fixed and only the pitch moved, gfx1150,
// bench_gemm_wide_lds WARM=3 REPS=9, lock held, `pgrep rustc` empty throughout,
// five interleaved fresh-process triples. Per-denoise-step total over the four
// census shapes (K = 3072, 3072, 15360, 3072), seconds:
//
//   entry                 PAD  runs                        med    vs PAD=0
//   256×256/64×64 k64       0  7.61 7.55 7.14 6.60 6.57    7.14
//                          64  5.05 4.98 4.68 4.63 4.60    4.68   1.53×
//                         128  4.69 5.29 4.32 4.34 4.27    4.34   1.65×
//   256×256/64×64 k64_p     0  6.77 6.44 5.95 5.56 5.52    5.95
//                          64  4.39 4.47 3.84 3.82 3.88    3.88   1.53×
//                         128  4.32 5.15 3.74 3.78 3.74    3.78   1.57×
//   128×256/32×64 k64       0  6.50 5.97 6.08 5.98 6.07    6.07
//                          64  4.45 4.85 4.87 4.55 4.93    4.85   1.25×
//                         128  4.61 5.09 4.78 4.75 4.78    4.78   1.27×
//   128×256/32×64 k64_p     0  5.71 5.91 5.84 5.81 5.70    5.81
//                          64  4.54 4.63 4.52 4.53 4.71    4.54   1.28×
//                         128  4.42 4.60 4.52 4.61 4.59    4.59   1.27×
//
// CONTROLLER, gfx1151, same bench, PAD=0 / 64 / 128, EXTRA=1 WARM=3 REPS=9,
// tile 256×256/64×64 k64_p. Per-denoise-step total 1.44 / 1.16 / 1.16 s, i.e.
// 1.24× for PAD=64 and nothing further from PAD=128. Per shape, ms:
//
//   shape (M×K×B)                    PAD=0    PAD=64   PAD=128
//   3072×3072×4608  single/qkv        2.78     2.58     2.59
//   12288×3072×4608 single/mlp_in    14.47    10.31    10.15
//   3072×15360×4608 single/linear2   14.20    11.70    11.61
//   3072×3072×512   double/txt_qkv    0.47     0.46     0.46
//   3072×12288×4608 single/mlp_out   18.83     9.42     9.38   36 % -> 73 %
//   21504×3072×4608 qkv_mlp_fused    24.97    18.32    17.79
//
// The win concentrates on K = 12288 (single/mlp_out, 2.00× on its own) — which
// is exactly the shape gfx1151's K sweep puts deepest in the cliff, and the one
// the FLUX per-family profile names as the largest single line. K = 3072 barely
// moves there, so gfx1151's per-step 1.24× is smaller than gfx1150's 1.53×
// while the mechanism is the same.
//
// PAD = 64 takes every FLUX K off the cliff (3136, 12352, 15424 are all
// 2^6·odd, so their pitches are 2^7·odd); PAD = 128 lands them on 2^8·odd and
// is worth a further 0-8 % on the wide tile on gfx1150 and nothing at all on
// gfx1151, so PAD = 64 is the pad to wire.
//
// READING THESE TABLES: the same cell measured in different sessions of this
// file spreads by about ±3 % (e.g. the gfx1150 PAD=64 256×256/64×64 k64 census
// total reads 4.59, 4.68 and 4.72 s in three different runs above). Compare
// arms WITHIN one table, never a number in one table against a number in
// another.
//
// The win is larger on the wider tile, which is what the mechanism predicts: a
// 256×256 block stages 512 concurrent rows per stage against 384 for 128×256,
// so it has more concurrent requests to camp onto the same channels. The
// pipelined form does not substitute for it — PIPE hides *latency*, and this is
// *throughput* — but the two compose: 256×256 goes 7.14 → 3.78 s, 1.89×, with
// both.
//
// The tightest form of the same A/B — ONE tile per process (so the timing loop
// never alternates kernels) and EXTRA=1 (two more shapes ahead of the census
// ones, i.e. a longer warm) — gives a spread of a few per mille and the same
// answer:
//
//   arm     per-step total, three fresh processes   med    ratio
//   PAD=0   5.36 5.39 5.38                          5.38
//   PAD=64  3.28 3.45 3.49                          3.45   1.56×
//
// and in that configuration the kernel reaches 81-95 % of the 14.8 TFLOP/s
// register-resident WMMA peak on gfx1150, against the 35-48 % this file's
// opening measurement recorded:
//
//   shape (M×K×B)                    ms      TFLOP/s   % of WMMA peak
//   3072×3072×4608  single/qkv        7.03   12.38     84 %
//   12288×3072×4608 single/mlp_in    28.15   12.36     84 %
//   3072×15360×4608 single/linear2   32.94   13.20     89 %
//   3072×3072×512   double/txt_qkv    0.79   12.30     83 %
//   3072×12288×4608 single/mlp_out   24.82   14.02     95 %
//   21504×3072×4608 qkv_mlp_fused    50.88   11.97     81 %
//
// WHICH OPERAND NEEDS THE PAD: almost entirely A, the weight. Same protocol,
// PADA and PADX varied independently, four fresh-process runs per arm,
// per-denoise-step total over the four census shapes:
//
//   arm                    256×256 k64_p runs         med    ratio
//   packed                 5.97 5.62 5.51 5.63        5.63
//   pad A (weight) only    3.93 3.95 4.09 4.08        4.02   1.40×
//   pad X (activation)     4.83 4.92 4.97 4.87        4.90   1.15×
//   pad both               3.66 3.71 3.78 3.78        3.75   1.50×
//
//   arm                    256×256 k64 runs           med    ratio
//   packed                 7.00 6.41 6.53 6.55        6.54
//   pad A (weight) only    4.92 5.05 5.25 5.16        5.11   1.28×
//   pad X (activation)     6.32 6.40 6.91 6.51        6.46   1.01×
//   pad both               4.43 4.46 4.67 4.59        4.53   1.44×
//
// CONTROLLER, gfx1151, same split: PADA=64 alone 1.17 s, PADX=64 alone 1.25 s,
// against 1.44 s packed and 1.16 s both — and on the shape that carries the win
// there, single/mlp_in, 10.36 ms with the weight padded against 12.18 ms with
// the activation padded. Same verdict: the weight pad is the whole lever and
// the activation pad adds ~1 %.
//
// That is what the SWAPXY model at the top of this file predicts: with grid.x
// along M the resident blocks share one X slab, which stays cached, while A is
// STREAMED once per column pass. The streamed operand is the one whose
// concurrent row reads have to hit DRAM, so it is the one that camps.
//
// COST: one extra `int` register pair in the kernel, and rows·pad·2 bytes per
// padded matrix (2 % of the weight at PAD = 64, K = 3072). What it does NOT do
// is help the caller for free — the FLUX weights have to be *stored* at the
// padded pitch, which is `upload_f16_bits` / the streamed `stream_into` path /
// `split_linear2` in crates/hipfire-arch-diffusion/src/flux_gpu.rs. Until they
// are, every product call passes lda = ldx = K and lands on the cliff. Per the
// table above, padding the weights alone is worth 1.28-1.40× of the 1.44-1.50×,
// and the activation allocations can stay packed.
//
// STALE PRE-COMPILED .hsaco HAZARD: this change altered the kernel ABI (two
// new `int` kernargs, `lda` and `ldx`) at the same time as the host weight
// layout (weights now stored at pitch K + 64, not packed at K). The two only
// agree when they ship together. A pre-compiled `.hsaco` blob built before
// this change, loaded against post-change host code that uploads padded
// weights, reads every row at stride K from a buffer laid out at stride
// K + 64 — silently wrong output, not a crash, and not the same failure mode
// as ordinary kernel drift. The JIT source-hash cache guards every normal
// build path (a source change invalidates the cached binary automatically);
// this only bites a packaged precompiled-`.hsaco` distribution, and any such
// set built before this change must be rebuilt, not just re-deployed.
//
// WHY THIS KERNEL EXISTS
// ----------------------
// gemm_f16_x_f16_wmma_lds hard-codes one point in a four-dimensional tuning
// space: block macro-tile 128×128, wave register tile 32×64, K stage 64,
// grid.x along M. Each axis controls a different, independent thing, and the
// only way to find out which one binds is to vary them one at a time:
//
//   BM × BN   block macro-tile.  Sets DRAM intensity BM·BN/(BM+BN) FLOP/byte:
//             over the whole K a block reads (BM+BN)·K halves to issue
//             2·BM·BN·K FLOP.
//   WP × WQ   wave register tile, in 16×16 WMMA subtiles. Sets LDS intensity
//             8·WP·WQ/(WP+WQ) FLOP/byte: a wave issues WP·WQ WMMA (8192 FLOP
//             each) per k-substep after WP+WQ fragment loads of 1024 B.
//             Accumulators cost 8·WP·WQ VGPRs per lane.
//   KS        K elements staged per barrier pair. Sets LDS footprint
//             (BM+BN)·KS·2 B and the barrier rate; does NOT affect either
//             intensity.
//   SWAPXY    which grid axis is fastest-varying. With grid.x along M the
//             concurrently-resident blocks share an X slab and stream distinct
//             A; swapping reverses which operand gets the in-flight reuse.
//
// MEASURED (bench_gemm_wide_lds, median of 5, two fresh processes per arch,
// lock held, every cell warmed separately) across the four FLUX.1-dev hot
// shapes, summed at their real per-step call counts:
//
//   - **Neither intensity binds.** On gfx1150, doubling DRAM intensity
//     (128×128 → 256×256, 64 → 128 FLOP/byte) moved the step total by 3 %
//     while the implied DRAM rate fell from 88 to 46 GB/s against a 67.5 GB/s
//     copy roof; raising LDS intensity 1.5× (32×64 → 64×64 wave tile) moved it
//     by −1 % while the implied LDS rate fell from 0.53 to 0.34 TB/s. Every
//     variant sat at 5.1–7.1 TFLOP/s, 35–48 % of the 14.8 TFLOP/s
//     register-resident WMMA peak. Time tracked neither intensity.
//   - **What moved on the iGPUs was BN**, and the SWAPXY control says why: with
//     grid.x along M the resident blocks share one n_block, so the X slab stays
//     in cache while A is streamed once per B/BN column pass. 128×256 with
//     grid.x on M and 256×128 with grid.x on B are the same physical
//     configuration under axis relabelling, and they measured identically
//     (gfx1150 6.16 vs 6.18 s; gfx1151 1.67 vs 1.67 s). In both, the *cached*
//     operand gets the 256-row slab and the *streamed* one gets 128.
//   - **BN = 512 falls off a cliff** on the iGPUs (0.54–0.57× on gfx1150,
//     0.57× on gfx1151): a 512×K half slab is 3 MB at K = 3072 and no longer
//     fits their L2. It is only neutral-to-positive on gfx1100, which has
//     96 MB of Infinity Cache behind its L2.
//   - **gfx1100 behaves differently on every axis.** There the wave tile is
//     worth 1.25× on its own and KS = 32 another 1.15×, taking the kernel to
//     86 % of WMMA peak on the two large shapes — while that same tile is a
//     14 % loss on gfx1150 and a 27 % loss on gfx1151.
//
// Per-step totals for those four shapes vs the shipped 128×128 / 32×64 tiling:
//
//   arch      shipped   best tile                    best    speedup
//   gfx1150   7.33 s    128×256 / 32×64 k64          6.16 s   1.17×
//   gfx1151   2.11 s    256×256 / 64×64 k64          1.66 s   1.28×
//   gfx1100   0.69 s    128×256 / 64×64 k32          0.49 s   1.41×
//
// The winner is therefore selected per arch in Gpu::lds_tile_for, not fixed
// here. A large BN needs a shallow KS to stay inside the 64 KB per-workgroup
// LDS limit: (BM+BN)·KS·2 ≤ 65536.
//
// REJECTED, on record: wave tile 96×64 and 64×96 (LDS intensity 19.2) compile
// for gfx1150 at vgpr 256 with 411 and 4 vgpr spills and 784 / 20 B of scratch.
// That is the wall the earlier mb4/mb8 attempts hit. Zero spills is a hard
// requirement, so they are not instantiated. Reaching that intensity needs
// register pressure cut elsewhere first — the compiler currently hoists a whole
// K-stage of global loads into registers across the barrier.
//
// REJECTED, on record: KS = 128 anywhere useful. The LDS budget is
// (BM + BN)·KS·2 ≤ 65536, so KS = 128 needs BM + BN ≤ 256:
//
//   tile          (BM+BN)·128·2   verdict
//   256×256        131072         2× over
//   256×128         98304         1.5× over
//   128×256         98304         1.5× over
//   128×128         65536         fits — EXACTLY on the limit
//
// The only fit is 128×128, which is the fallback tile, not the wide one this
// was aimed at, and putting it at KS = 128 doubles its LDS from 32 to 64 KB and
// so takes it from two resident workgroups per CU to one — strictly worse
// occupancy for a barrier count this file has already measured as not binding
// (KS = 32 vs 64 on the same tile moved gfx1150 by < 3 %). Halving BN to keep
// the footprint is hypothesis 1 below, and that is a 28-30 % loss on its own.
// Nothing fits where it would help, so KS = 128 is not implemented and the
// KS ∈ {32, 64} static_assert stands.
//
// REJECTED, on record: wave tile 64×128 (WP = 4, WQ = 8, LDS intensity 21.3).
// 8·WP·WQ = 256 accumulator VGPRs per lane before anything else is live, i.e.
// the entire gfx11 wave32 budget. It compiles, and the damage is exactly that:
//
//   entry (uninstantiated probe)   arch      VGPR   spill   scratch B/lane
//   256×256 / 64×128 k64           gfx1150   256    776     1092
//   256×256 / 64×128 k64           gfx1151   256    776     1092
//   256×128 / 64×128 k64           gfx1150   256    777     1160
//
// Zero spills is a hard requirement, so it is not instantiated. The asymmetric
// 32×128 (WP = 2, WQ = 8) DOES compile clean — 247 VGPRs, zero spills, zero
// scratch, occupancy 4 on gfx1150 and gfx1151 — but it is dominated rather than
// promising: 8·WP·WQ = 128 accumulators, the same as the 64×64 wave tile, at an
// LDS intensity of 12.8 against 64×64's 16.0. Strictly more LDS traffic for the
// same register cost, so it was not measured.
//
// REJECTED ON BOTH ARCHS, DELETED, on record: two workgroups per CU by halving
// BN — 256×128 / 64×64 at KS = 32 (24 KB of LDS, 256 threads, genuinely two
// resident workgroups) and at KS = 64 (48 KB, one workgroup but half the
// barrier count per K element). Both compiled clean (k32 191 VGPRs, k32_p
// 207-215, k64_p 241, zero spills on all three archs), and both lose:
//
//   entry                  gfx1150 med   vs incumbent   gfx1151 (controller)
//   256×256/64×64 k64_p    3.82 s        1.00           1.11 s   1.00
//   256×128/64×64 k64_p    5.31 s        0.72           1.34 s   0.83
//   256×128/64×64 k32      5.44 s        0.70           2.78 s   0.40
//   256×128/64×64 k32_p    5.36 s        0.71           2.12 s   0.52
//
// (gfx1150: TILES=64_64 PAD=64 WARM=3 REPS=9, five fresh-process runs, all
// tiles timed in one process. gfx1151: controller, same command.) The result is
// flat across the stage depth on gfx1150 and worse at KS = 32 on gfx1151, so
// the second resident workgroup is not merely unhelpful — it is a cost. Same
// conclusion as the SWAPXY control above: what a block wins is which operand
// gets the 256-row cached slab, not how many blocks are resident. Halving BN
// halves the reuse of the cached operand and occupancy does not pay that back.
//
// SOFTWARE PIPELINING (PIPE)
// --------------------------
// The default main loop is load → barrier → WMMA → barrier. Every tile is
// launched at `__launch_bounds__(NT, 1)`, and the widest ones (256×256 k64 =
// 64 KB of LDS) can only ever have one workgroup resident per CU, so there is
// no second workgroup to cover the global-load latency: the wave stalls at the
// top of every stage waiting for DRAM, then runs the WMMA with the memory
// pipe idle.
//
// PIPE = true issues stage k0 + KS's global loads into the per-thread staging
// registers *between* the leading barrier and the WMMA loop of stage k0, and
// writes them to LDS at the top of the next iteration. The loads are then in
// flight across the whole WMMA block. Same two barriers per stage, same number
// of loads, and the WMMA loop itself is `wlds_wmma_stage` on both paths — the
// K summation order is untouched and the two are bit-exact against each other,
// which is what the parity gate asserts.
//
// It costs (A_PASS + X_PASS) half16 = 8·(A_PASS + X_PASS) VGPRs of extra live
// range on top of the 8·WP·WQ accumulators. That is 24-32 VGPRs on the four
// selectable tiles and 64 on 128×128/64×64 (NT = 128, so each thread stages
// four vectors of each operand), which is why the pipelined entries are
// instantiated per tile rather than everywhere: a spill is a hard no. The
// per-tile register and occupancy numbers are in the instantiation block at
// the bottom of this file.
//
// MEASURED, gfx1150 (bench_gemm_wide_lds, REPS=5 WARM=1, lock held, five runs
// — three full-table and two tile-pair-only — each pipelined tile timed in the
// same process as its own base). Per-denoise-step total over the four census
// shapes, pipelined ÷ plain:
//
//   tile                  runs                        median
//   128×256 / 32×64 k64   1.12 1.01 1.06 0.98 1.03    1.03
//   128×128 / 32×64 k64   1.03 1.02 1.02 0.98 1.01    1.02
//   256×256 / 64×64 k64   1.19 1.15 1.15 1.13 1.17    1.15
//   128×256 / 64×64 k32   1.06 1.00 1.02 0.99 1.13    1.02
//
// Only 256×256/64×64 is outside the ±3 % session band — and it is the tile
// gfx1151 selects, the only one whose 64 KB of LDS pins it to one workgroup
// per CU, i.e. exactly the configuration with nothing else to hide the load
// latency. Its five runs span 1.13-1.19× and never dip below 1.13. The other
// three are free rather than harmful: no run of any of them was worse than
// 0.98×.
//
// gfx1151, the arch this was written for, was measured end-to-end on the real
// FLUX.1-dev checkpoint (gpu_flux_real_throughput, f32 activation path):
// 3.950 / 3.921 s per denoise step pipelined vs 4.202 s plain, i.e. **1.07×**
// (t_double 76.2 vs 78.3 ms, t_single 65.1-65.8 vs 71.4 ms). gfx1100 measured
// +2.6 % on its selected tile (89.7 vs 87.4 TFLOP/s), inside the band.
//
// So the pipelined form is ON by default for gfx1151 only. Which archs default
// to it is decided on the Rust side by the Gpu::LDS_PIPE_ON table, overridden
// per process by HIPFIRE_FLUX_GEMM_PIPE (0 = plain, 1 = pipelined).
//
// REJECTED ON BOTH ARCHS, DELETED, on record: the SECOND level of the same
// idea — register-double-buffering the A fragments inside wlds_wmma_stage, so
// substep kt+1's ds_load_b128s issue before substep kt's WMMA block. PIPE hides
// the GLOBAL load behind a whole stage; this would have hidden the LDS load
// behind one substep. It was bit-exact (the WMMA sequence was unchanged, only
// the loads moved; 792/792 parity cells on both arms) and it cost nothing in
// registers on the wide tile (227 -> 228 VGPRs, no occupancy step on any arch).
//
// It bought nothing. gfx1150, PAD=64 WARM=3 REPS=9, five interleaved
// fresh-process pairs, per-denoise-step total over the four census shapes:
//
//   entry                  off     on      ratio
//   256×256/64×64 k64      4.72    4.66    1.013×
//   256×256/64×64 k64_p    3.94    3.90    1.010×
//   128×256/32×64 k64      4.36    4.53    0.96×
//   128×256/32×64 k64_p    4.35    4.75    0.92×
//
// and (controller, gfx1151, PAD=64, TILES=256_256_64_64_k64) 1.21 -> 1.18 s
// plain and 1.13 -> 1.11 s pipelined, +1.8-2.5 %, inside the band. On the
// narrow tiles it is a real loss on gfx1150, where the extra live range costs
// an occupancy step (126 -> 144 VGPRs, 8 -> 7 waves per SIMD).
//
// The LDS read is not what the wave is waiting on: at a 64×64 wave tile the
// stage issues 8 fragment loads per 16 WMMA, and 16 WMMA is long enough that
// the compiler's own scheduling already covers them. Do not re-try this without
// first showing that ds_load latency is on the critical path.
//
// LDS LAYOUT — tile-major [kt][row][ki], the same reasoning as the 128×128
// kernel: 16 lanes each read a 32-byte half16 from a different row at the same
// kt, so tile-major makes those 16 reads one contiguous 512-byte run — the
// conflict-free case — where row-major would put them all on one bank. What
// this layout does NOT get right is the staging write; see the bank note under
// STAGING below for the measurement and for the two rejected fixes.
//
// STAGING — the A tile is BM·KT half16 vectors and the X tile BN·KT, spread
// over NT threads by a predicated strided loop, so no divisibility relation
// between the tile and the thread count is required. Consecutive tid walk the
// KT subtiles of one row, so each group of KT lanes covers 32·KT contiguous
// bytes of a single source row.
//
// LDS BANK BEHAVIOUR OF THAT STAGING WRITE — measured, and left alone
// ---------------------------------------------------------------------------
// The staging write IS bank-conflicted, by a factor of four, and removing the
// conflict is worth up to 9 % on the plain main loop and nothing on the
// pipelined one. Both attempts at removing it are below as rejected, with the
// numbers, so the next person does not re-derive the analysis.
//
// The analysis. LDS moves 128 B per clock (32 banks × 4 B), so a `ds_*_b128` is
// conflict-free when each group of 8 lanes covers 8 distinct 16-byte bank quads
// — 8 distinct values of `byte_addr / 16 & 7`. A half16 is two such quads and
// compiles to two `ds_*_b128`. In the `[kt][row][16]` layout a quad index is
// `(2·row + c) & 7`, and `kt·ROWS·2 ≡ 0 (mod 8)` on every instantiated tile, so
// the kt axis contributes NOTHING to the bank:
//
//   * fragment read (`wlds_wmma_stage`), kt fixed, 16 consecutive rows: 8 lanes
//     span 8 rows, 256 contiguous bytes — the ideal pattern.
//   * staging write (`wlds_dst`), `row = v / KT`, `kt = v % KT`: 8 consecutive
//     lanes span only TWO rows against KT kt-slabs that alias in bank space, so
//     they cover 2 quads instead of 8 — a 4-way conflict, 4× the clocks.
//
// Fixing it needs the quad to depend on kt, i.e. `q = (row & 7) ^ (KTS·kt)`
// with `KTS = 8/KT`, applied to the 16-byte chunk index and inverted in the
// fragment read (a permutation of the 16 chunks of each aligned 8-row block, so
// the LDS footprint is unchanged — which is required, 256×256 k64 sits exactly
// on the 64 KB limit and cannot be padded).
//
// REJECTED (A), full swizzle `q = (row & 7) ^ (KTS·kt)`. Bit-exact (494/494
// wide-LDS cells, 912/912 epilogue cells), zero spills on gfx1150/1151/1100,
// and on 256×256/64×64 k64_p the same 64 `ds_load_b128` + 8 `ds_store_b128`
// per stage as the baseline — only the addresses move. gfx1150,
// bench_gemm_wide_lds REPS=5 WARM=1, lock held, `pgrep rustc` empty, matched
// fresh-process runs per arm, per-denoise-step total over the four census
// shapes:
//
//   entry                arm   runs                       med    ratio
//   256×256/64×64 k64    base  6.49 6.06 6.05 6.47 6.54   6.47
//                        swiz  5.99 5.82 5.78 5.96 5.95   5.95   1.087×
//   256×256/64×64 k64_p  base  5.87 5.49 5.50 5.51 5.52   5.51
//                        swiz  5.46 5.49 5.37 5.45 5.41   5.45   1.011×
//   128×256/32×64 k64    base  5.84 6.09 5.94              5.94
//                        swiz  5.96 6.10 5.86              5.96   0.997×
//   128×256/32×64 k64_p  base  5.76 5.82 5.77              5.77
//                        swiz  7.47 7.39 7.45              7.45   0.775×
//
// Two results, and they are the reason this is not landed:
//
//   1. **The win is real but the pipelined loop already has it.** +8.7 % on the
//      plain main loop, +1.1 % on the pipelined one. PIPE hoists the LDS write
//      out of the stall path by issuing it from registers a stage early, so it
//      and the swizzle attack the SAME stall and do not compose: pipelined ÷
//      plain falls from 1.16× to 1.09× once the swizzle is in. gfx1151 — the
//      arch that selects this tile — ships PIPE on, so it would collect the
//      1 %, not the 9 %.
//   2. **It costs an occupancy cliff on another selectable tile.** The swizzle
//      makes the fragment address kt-dependent; the compiler hoists one address
//      per kt per slab out of the K loop and widens its in-flight fragment
//      window from 6 to 8 fragments to match. On 128×256/32×64 k64_p that is
//      126 → 139 VGPRs, i.e. over the 128-VGPR / 8-waves-per-SIMD cliff (gfx11
//      wave32 has 1024 VGPRs per SIMD), occupancy 8 → 7, and **−23 %** on the
//      tile gfx1150 and gfx1151 select for the narrow shapes. Every other entry
//      moved by ≤ 6 VGPRs and no other occupancy changed.
//
//   Also tried, on that second point:
//     * XOR instead of a modular add, so the per-lane term factors out of the
//       kt dependence and the kt/c/subtile part rides in the `ds_*_b128` offset
//       immediate. Identical resource profile — 139 VGPRs. The hoisting is the
//       scheduler's choice, not an addressing-mode limit.
//     * `__launch_bounds__(NT, 8)` on the 32×64 wave tiles to force the
//       allocator back under 128. It rematerialises, but then SPILLS: 8 VGPRs /
//       20 B scratch on 128×256/32×64 k64_p and 32 VGPRs / 68 B on
//       128×128/32×64 k64_p. Zero spills is a hard requirement.
//
// REJECTED (A′), 1-bit swizzle `q = (row & 7) ^ (4·(kt & 1))`. Keeps the read
// conflict-free and halves the write conflict (4-way → 2-way) at KS = 64; at
// KS = 32 it is identical to the full swizzle. Costs one address register per
// slab instead of four, so nothing crosses an occupancy cliff — every entry
// within ±7 VGPRs of baseline, no occupancy change, zero spills, bit-exact.
// But the win shrinks with the conflict. gfx1150 per-step totals, five clean
// full-table runs against three baseline full-table runs:
//
//   entry                    baseline med   A′ med   ratio
//   256×256/64×64 k64        6.47           6.23     1.039×
//   256×256/64×64 k64_p      5.57           5.52     1.009×
//   128×256/32×64 k64        5.94           5.89     1.008×
//   128×256/32×64 k64_p      5.94           5.99     0.992×
//   128×128/32×64 k64        7.19           7.29     0.986×
//   128×128/32×64 k64_p      7.17           7.33     0.978×
//   128×256/64×64 k32        8.45           8.69     0.972×
//   128×256/64×64 k32_p      8.36           8.23     1.016×
//   128×128/64×64 k64        7.25           7.37     0.984×
//
// +3.9 % on one plain entry, ±2 % everywhere else — under the ≥ 3 % bar on the
// tile that was the target, and slower on three selectable entries. Not landed.
//
// REJECTED (B), write-side remap `row = v % ROWS`, `kt = v / ROWS`, so
// consecutive lanes write consecutive rows (a 2-way write, down from 4-way)
// with the LDS layout untouched. Bit-exact, but it decoalesces the GLOBAL read:
// each lane then pulls one 32-byte vector at a K·2-byte stride where before
// each group of KT lanes pulled 32·KT contiguous bytes. Neither iGPU cache
// hides it. gfx1150 per-step totals, three runs per arm:
//
//   entry                 baseline med   remapped runs        med    ratio
//   256×256/64×64 k64     6.47           7.08 6.96 6.95       6.96   0.930×
//   256×256/64×64 k64_p   5.51           5.87 5.88 5.82       5.87   0.939×
//
// −7 % on both forms. The write conflict is worth less than the global-read
// coalescing, which is the same ordering the 128×128 kernel already assumed.
//
// If this is picked up again: the only shape of it that is still open is a
// PIPE-conditional layout (swizzle the plain loop, leave the pipelined one
// alone). It is bit-exact-safe — the layout does not change any value, and
// PIPE and non-PIPE are separate instantiations — but it doubles the layout
// surface of this file to buy a tile-dependent 0-9 % on the archs that default
// to the plain loop, and nothing on gfx1151. It was not worth that here.
//
// CLOCK AND POWER, and why gfx1150 numbers are only ever ratios
// -------------------------------------------------------------
// gfx1150 does NOT hold its top boost through a sweep. Sampled during a six-
// shape run of bench_gemm_wide_lds (PAD=64, TILES=256_256_64_64_k64_p),
// /sys/class/drm/card1/device/hwmon/hwmon*/{freq1_input,power1_average,
// temp1_input} every 0.25-0.5 s:
//
//   sclk    1851-2403 MHz, against a 2900 MHz top pp_dpm_sclk level
//   PPT     28-35 W
//   edge    64-79 °C
//
// i.e. 64-83 % of the DPM ceiling, power-limited rather than idle. Two
// consequences: absolute TFLOP/s on this part is a moving target set by the
// power budget, so only same-session ratios mean anything here; and the
// %peak column is against a 14.8 TFLOP/s reference measured under the same
// budget, not against a datasheet number.
//
// SAMPLING GOTCHA: `freq1_input` is backed by the driver's metrics table and
// only refreshes on the order of a second. A tight read loop returns a stale
// 600 MHz for tens of thousands of samples in a row (measured: 35198/35198).
// Sample at 0.25 s or slower.
//
// ON gfx1151 THIS METHOD DOES NOT WORK AT ALL. The same file under
// /sys/class/drm/card2/device/ (card0 there is the discrete 7900 XT) read
// 603-608 MHz throughout a sweep that was delivering 37 TFLOP/s — physically
// impossible, so the counter is not wired to the real sclk on that APU.
// Treat gfx1151 clock behaviour as UNMEASURED by this route (controller); if it
// ever matters, use a rocprof/SMU-side counter, not sysfs.
//
// PRECONDITION: K % KS == 0, and KS ∈ {32, 64}. Every FLUX.1-dev K is a
// multiple of 64. The Rust wrapper enforces it and falls back to the 16-step
// kernel otherwise.
//
// M and B tails are handled by clamping OOB rows to the last valid row and
// discarding at the store, per the house pattern. No thread early-returns after
// the block-uniform origin check — every thread must reach both
// __syncthreads() of every stage.
//
// K ORDER is ascending in 16-element WMMA substeps, identical to both
// gemm_f16_x_f16_wmma and gemm_f16_x_f16_wmma_lds, so the F32 accumulation
// order matches and results are bit-exact against them regardless of KS.
//
// Grid:  SWAPXY ? [ceil(B/BN), ceil(M/BM)] : [ceil(M/BM), ceil(B/BN)]
// Block: (BM/(16·WP)) · (BN/(16·WQ)) · 32

// EPILOGUE VARIANTS (EPI, a compile-time bitmask)
// ----------------------------------------------
// The FLUX MMDiT blocks never want the raw F32 product: every GEMM is followed
// by some fixed elementwise pass (a cast to F16 for the next GEMM's activation,
// a GELU, a gated residual accumulation, or a sum of the two halves of a split
// linear2). Each of those is a separate kernel that re-reads and re-writes the
// whole B×M matrix — at B = 4608, M = 12288 that is 226 MB out and 226 MB back
// per pass, on a part with 90 GB/s of DRAM. Folding them into the store turns
// the pass into zero extra traffic (ADDIN/GATED read one extra matrix; OUT_F16
// halves the write).
//
//   EPI_OUT_F16 (1) : store _Float16 (RNE) instead of F32.
//   EPI_GELU    (2) : GELU-tanh, the exact expression from gelu_tanh.hip.
//   EPI_ADDIN   (4) : acc += C[b, m], C an F32 [B, M] matrix.
//   EPI_GATED   (8) : Y[b, m] = R[b, m] + gate[m] · value, R F32 [B, M],
//                     gate F32 [M].
//
// The evaluation order is fixed and is what the parity gate reproduces:
//
//   v = acc                       (the WMMA F32 accumulator)
//   v += C[b, m]                  if EPI_ADDIN
//   v += has_bias ? Bias[m] : 0   (always — same expression as EPI = 0)
//   v  = gelu_tanh(v)             if EPI_GELU
//   v  = fma(gate[m], v, R[b, m]) if EPI_GATED
//   store v (RNE to F16 if EPI_OUT_F16)
//
// EPI = 0 reduces to `v[e] + bv` — the original store, unchanged and bit-exact.
//
// ALIASING: R is allowed to alias Y (the in-place gated residual update Task 5
// needs). It is race-free: element (b, m) is read from R and written to Y by
// the SAME thread, at the same flat index, and no other thread touches it — the
// load feeds the stored value, so no compiler reordering can break the
// dependency. C may alias Y for the same reason. None of Y, C, R, Gate is
// marked __restrict__, so the aliasing is also well-defined and not merely
// benign. A and X keep theirs; they are read-only and never alias the output.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define EPI_OUT_F16 1
#define EPI_GELU    2
#define EPI_ADDIN   4
#define EPI_GATED   8

// COALESCED EPILOGUE (CST)
// ------------------------
// The natural epilogue is transposed with respect to memory: a lane owns a
// fixed B column (`ml`) and walks M, so its stores are 4 B apart with a stride
// of 4·M. That costs little when the only global access is the store, but
// EPI_ADDIN and EPI_GATED add one or two full [B, M] *reads* in the same
// pattern, and those are where it bites.
//
// MEASURED, gfx1150, bench_gemm_epilogue, three interleaved fresh-process pairs
// per arm (HIPFIRE_LDS_EPI_DIRECT=1 is the direct arm), median of the medians,
// tile 128×256/32×64 k64. Staged / direct:
//
//   shape (M×K×B)             _o16   _o16g   _gr    _gra   _a
//   3072×3072×4608            1.06   1.07    1.08   1.22   1.02
//   12288×3072×4608           1.06   1.06    1.03   1.20   1.01
//   3072×7680×4608            1.08   1.07    1.05   1.15   1.05
//   3072×3072×512             1.02   1.04    1.07   1.22   1.04
//
// Staged is faster in all 20 cells, against a ±3-5 % within-arm spread, and the
// win scales with how many [B, M] matrices the epilogue touches: +20 % where it
// touches three (_gra: add-in, residual, output), +2-8 % where it touches one.
// A single-run A/B suggested far more (up to 1.63×); it was measuring the
// machine's DPM state, not the kernel. Interleave.
//
// So the accumulators go through LDS first: one 16-wide B slab of the wave's
// patch at a time, written from the WMMA layout, read back with lane = m. Every
// global access in the epilogue then runs along m — 32 consecutive floats per
// lane group — for the loads and the store alike.
//
// It is free in LDS: the K staging buffers are dead by the epilogue, so the
// slab reuses them, and the kernel's LDS footprint is unchanged. That is also
// what limits it — a tile whose slab does not fit inside its own K staging
// buffer keeps the direct path rather than growing LDS and losing occupancy
// (`WLDS_CST_FITS` decides, per instantiation, at compile time). Of the five
// instantiated tiles the three with a 32×64 wave tile and 128×128/64×64 fit;
// 256×256/64×64 (69632 B of slab vs 65536 B of staging) and 128×256/64×64 k32
// (34816 vs 24576) do not.
//
// The 17-float row stride is the bank-conflict pad: reads are
// `stage[(m0 + lane) * 17 + b]`, and 17 is coprime with the 32 LDS banks, so
// the 32 lanes hit 32 distinct banks. Writes take one 2-way conflict.
//
// Set to 0 to A/B the direct epilogue (both are bit-identical; only the access
// pattern differs).
#define WLDS_STAGE_EPILOGUE 1

// Bytes of LDS the K staging needs, and the slab the coalesced epilogue would
// need: one 16-column slab of every wave's WROWS×16 patch, padded to 17.
#define WLDS_KSTAGE_BYTES(BM, BN, KS) ((KS) / 16 * ((BM) + (BN)) * 16 * 2)
#define WLDS_SLAB_BYTES(BM, BN, WP, WQ)                                        \
    ((BM) / (16 * (WP)) * ((BN) / (16 * (WQ))) * (16 * (WP)) * 17 * 4)
#define WLDS_CST_FITS(BM, BN, WP, WQ, KS)                                      \
    (WLDS_STAGE_EPILOGUE &&                                                    \
     WLDS_SLAB_BYTES(BM, BN, WP, WQ) <= WLDS_KSTAGE_BYTES(BM, BN, KS))

// Character-for-character the body of gelu_tanh_f32 (kernels/src/gelu_tanh.hip)
// so the fused epilogue is bit-exact against that kernel applied to the EPI = 0
// output — which is what the parity gate asserts, and what makes Task 5's
// deletion of the separate GELU pass a no-op numerically.
__device__ __forceinline__ float wlds_gelu_tanh(float v) {
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanhf(inner));
}

// --- Staging address helpers ------------------------------------------------
// The A and the X staging loop differ only in which slab they read, which row
// bound clamps them and which LDS extent they write, so both go through these
// two. Factoring them out is not cosmetic: the pipelined main loop below issues
// the global load and the LDS write in *different* loop iterations, and the two
// must agree on the address to the character.
//
// `v` is the flat half16-vector index inside one K stage: `v / KT` is the row
// of the slab, `v % KT` the 16-element k-substep inside that row.
//
// `ld` is the ROW PITCH in elements, which is NOT K: see the ROW PITCH note at
// the top. Only `k0 + kt*16 < K` is ever addressed, so anything past K in a row
// is never read and may hold garbage.
template <int KT>
__device__ __forceinline__ const half16_t* wlds_src(
    const _Float16* __restrict__ S, int rows, int ld, int row_block, int k0, int v
) {
    const int row = v / KT;
    const int kt  = v % KT;
    // Clamp OOB rows to the last valid one. They are computed but discarded at
    // the store; they must never read out of bounds.
    const int r    = row_block + row;
    const int safe = (r < rows) ? r : (rows - 1);
    return (const half16_t*)(S + (long long)safe * ld + k0 + kt * 16);
}

// ROWS is BM for the A slab and BN for the X slab: LDS is tile-major
// [kt][row][ki].
template <int ROWS, int KT>
__device__ __forceinline__ half16_t* wlds_dst(_Float16* lds, int v) {
    const int row = v / KT;
    const int kt  = v % KT;
    return (half16_t*)(&lds[(kt * ROWS + row) * 16]);
}

// One K stage of WMMA out of LDS. This is the ONLY place the accumulation order
// is written down, so the pipelined and the non-pipelined main loop provably
// sum K in the same order and stay bit-exact against each other.
template <int BM, int BN, int WP, int WQ, int KT>
__device__ __forceinline__ void wlds_wmma_stage(
    const _Float16* a_lds, const _Float16* x_lds,
    int wave_m, int wave_n, int ml, float8_t* acc
) {
    constexpr int WROWS = 16 * WP;
    constexpr int WCOLS = 16 * WQ;
    #pragma unroll
    for (int kt = 0; kt < KT; kt++) {
        // All WP A fragments stay live and are reused across every B subtile;
        // one B fragment is live at a time. That is what makes the LDS traffic
        // WP + WQ loads per WP·WQ WMMA.
        half16_t a[WP];
        #pragma unroll
        for (int i = 0; i < WP; i++) {
            a[i] = *(const half16_t*)(
                &a_lds[(kt * BM + wave_m * WROWS + i * 16 + ml) * 16]);
        }
        #pragma unroll
        for (int j = 0; j < WQ; j++) {
            const half16_t b = *(const half16_t*)(
                &x_lds[(kt * BN + wave_n * WCOLS + j * 16 + ml) * 16]);
            #pragma unroll
            for (int i = 0; i < WP; i++) {
                acc[i * WQ + j] =
                    __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b, acc[i * WQ + j]);
            }
        }
    }
}

// BM, BN : block macro-tile (M rows × B columns)
// WP, WQ : wave register tile, in 16×16 WMMA subtiles (M × B)
// KS     : K elements staged per barrier pair
// SWAPXY : true → grid.x indexes B, grid.y indexes M
// EPI    : epilogue bitmask, see above. 0 is the plain F32 + bias store.
// CST    : stage the epilogue through LDS so every global access runs along m.
// PIPE   : software-pipeline the main loop — see the PIPE note above.
template <int BM, int BN, int WP, int WQ, int KS, bool SWAPXY, int EPI, bool CST,
          bool PIPE>
__device__ __forceinline__ void gemm_f16_wmma_lds_tile(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ X,
    void*                        Y,   // float* , or _Float16* when EPI_OUT_F16
    const float*    __restrict__ Bias,
    int M, int K, int B, int has_bias,
    // Row pitch of A and X in ELEMENTS. `lda == K` and `ldx == K` reproduce the
    // packed row-major contract exactly; anything larger pads each row. See the
    // ROW PITCH note at the top of this file.
    int lda, int ldx,
    const float* C,                   // [B, M] add-in, EPI_ADDIN only
    const float* R,                   // [B, M] residual, EPI_GATED only
    const float* Gate,                // [M]    gate,     EPI_GATED only
    // NONE of the three LDS pointers is __restrict__. Under CST they are three
    // names for one __shared__ array: `stage` reuses the K staging storage once
    // the K loop is done. The reuse is correct — the K loop's trailing
    // __syncthreads() is a memory clobber, and this is the house pattern for
    // epilogue LDS reuse — but this function is __forceinline__, so a restrict
    // qualifier would hand LLVM a noalias scope at the call site covering
    // pointers that provably alias, and correctness would rest on the barrier
    // alone. Dropping it costs nothing: the 15 EPI = 0 entries compile to
    // instruction-identical ISA either way (checked on gfx1150), because the
    // K-loop accesses are address-space-disambiguated LDS.
    _Float16* a_lds,                // [KT][BM][16]
    _Float16* x_lds,                // [KT][BN][16]
    float*    stage                 // CST only: aliases a_lds/x_lds, dead by now
) {
    constexpr int KT     = KS / 16;             // WMMA k-substeps per stage
    constexpr int WROWS  = 16 * WP;             // M rows per wave
    constexpr int WCOLS  = 16 * WQ;             // B columns per wave
    constexpr int WAVE_M = BM / WROWS;          // waves along M
    constexpr int WAVE_N = BN / WCOLS;          // waves along B
    constexpr int NT     = WAVE_M * WAVE_N * 32;
    constexpr int A_VEC  = BM * KT;             // half16 vectors of A per stage
    constexpr int X_VEC  = BN * KT;             // half16 vectors of X per stage
    constexpr int A_PASS = (A_VEC + NT - 1) / NT;
    constexpr int X_PASS = (X_VEC + NT - 1) / NT;

    static_assert(KS == 32 || KS == 64, "KS must be 32 or 64");
    static_assert(BM % WROWS == 0, "BM must tile by the wave row extent");
    static_assert(BN % WCOLS == 0, "BN must tile by the wave column extent");
    static_assert(NT <= 1024, "block exceeds the 1024-thread limit");
    static_assert((BM + BN) * KS * 2 <= 65536, "LDS exceeds 64 KB per workgroup");

    const int m_block = (SWAPXY ? blockIdx.y : blockIdx.x) * BM;
    const int n_block = (SWAPXY ? blockIdx.x : blockIdx.y) * BN;

    // Block-uniform origin check — legal before any barrier because every
    // thread in the block agrees on it.
    if (m_block >= M || n_block >= B) return;

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int wave_id = tid >> 5;
    const int ml      = lane & 15;   // fragment row; lanes 16..31 mirror 0..15

    const int wave_m = wave_id % WAVE_M;
    const int wave_n = wave_id / WAVE_M;
    const int m_wave = m_block + wave_m * WROWS;
    const int n_wave = n_block + wave_n * WCOLS;

    float8_t acc[WP * WQ];
    #pragma unroll
    for (int i = 0; i < WP * WQ; i++) {
        acc[i] = float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // Staging registers, live only on the pipelined path. `PIPE ? N : 1` keeps
    // the array from existing at all in the non-pipelined instantiations.
    half16_t a_reg[PIPE ? A_PASS : 1];
    half16_t x_reg[PIPE ? X_PASS : 1];

    if constexpr (PIPE) {
        // Prologue: stage 0's global loads, issued before the loop so the
        // steady state always has one stage in flight.
        #pragma unroll
        for (int p = 0; p < A_PASS; p++) {
            const int v = tid + p * NT;
            if (A_VEC % NT != 0 && v >= A_VEC) {
                continue;
            }
            a_reg[p] = *wlds_src<KT>(A, M, lda, m_block, 0, v);
        }
        #pragma unroll
        for (int p = 0; p < X_PASS; p++) {
            const int v = tid + p * NT;
            if (X_VEC % NT != 0 && v >= X_VEC) {
                continue;
            }
            x_reg[p] = *wlds_src<KT>(X, B, ldx, n_block, 0, v);
        }

        for (int k0 = 0; k0 < K; k0 += KS) {
            // Retire the stage that was prefetched during the previous
            // iteration's WMMA. Nothing reads LDS between the trailing barrier
            // of that iteration and the leading barrier below, so this is safe.
            #pragma unroll
            for (int p = 0; p < A_PASS; p++) {
                const int v = tid + p * NT;
                if (A_VEC % NT != 0 && v >= A_VEC) {
                    continue;
                }
                *wlds_dst<BM, KT>(a_lds, v) = a_reg[p];
            }
            #pragma unroll
            for (int p = 0; p < X_PASS; p++) {
                const int v = tid + p * NT;
                if (X_VEC % NT != 0 && v >= X_VEC) {
                    continue;
                }
                *wlds_dst<BN, KT>(x_lds, v) = x_reg[p];
            }

            __syncthreads();

            // Issue stage k0 + KS NOW, before the WMMA. The loads are
            // independent of everything the WMMA loop touches, so their DRAM
            // latency is covered by this stage's math instead of stalling the
            // wave at the top of the next iteration. The guard is block-uniform
            // and keeps the last iteration from reading past the end of A/X.
            const int k1 = k0 + KS;
            if (k1 < K) {
                #pragma unroll
                for (int p = 0; p < A_PASS; p++) {
                    const int v = tid + p * NT;
                    if (A_VEC % NT != 0 && v >= A_VEC) {
                        continue;
                    }
                    a_reg[p] = *wlds_src<KT>(A, M, lda, m_block, k1, v);
                }
                #pragma unroll
                for (int p = 0; p < X_PASS; p++) {
                    const int v = tid + p * NT;
                    if (X_VEC % NT != 0 && v >= X_VEC) {
                        continue;
                    }
                    x_reg[p] = *wlds_src<KT>(X, B, ldx, n_block, k1, v);
                }
            }

            wlds_wmma_stage<BM, BN, WP, WQ, KT>(a_lds, x_lds, wave_m, wave_n, ml, acc);

            __syncthreads();
        }
    } else {
        for (int k0 = 0; k0 < K; k0 += KS) {
            #pragma unroll
            for (int p = 0; p < A_PASS; p++) {
                const int v = tid + p * NT;
                if (A_VEC % NT != 0 && v >= A_VEC) {
                    continue;
                }
                *wlds_dst<BM, KT>(a_lds, v) = *wlds_src<KT>(A, M, lda, m_block, k0, v);
            }
            #pragma unroll
            for (int p = 0; p < X_PASS; p++) {
                const int v = tid + p * NT;
                if (X_VEC % NT != 0 && v >= X_VEC) {
                    continue;
                }
                *wlds_dst<BN, KT>(x_lds, v) = *wlds_src<KT>(X, B, ldx, n_block, k0, v);
            }

            __syncthreads();

            wlds_wmma_stage<BM, BN, WP, WQ, KT>(a_lds, x_lds, wave_m, wave_n, ml, acc);

            __syncthreads();
        }
    }

    // Epilogue. RDNA3 wave32 WMMA: acc[e] = C[2*e + (lane>>4)][lane & 15],
    // where the C row is the M index (from the A fragment) and the C column is
    // the B index (from the B fragment).
    const int row_half = lane >> 4;

    if (CST) {
        // Coalesced epilogue — see the CST note at the top. One 16-column B
        // slab at a time: write the slab from the WMMA layout into this wave's
        // private region of the (now dead) K staging buffer, then read it back
        // with lane = m so every global access below runs along m.
        constexpr int STG = 17;                     // pad: coprime with 32 banks
        // The readback below hands one M row to each of the 32 lanes, so the
        // wave's M extent must be a whole number of lane sweeps.
        static_assert(!CST || WROWS % 32 == 0,
                      "the staged epilogue needs WROWS to be a multiple of 32");
        static_assert(!CST || WLDS_SLAB_BYTES(BM, BN, WP, WQ) <=
                                  WLDS_KSTAGE_BYTES(BM, BN, KS),
                      "the epilogue slab must fit inside the K staging buffer");
        float* slab = stage + wave_id * WROWS * STG;
        for (int j = 0; j < WQ; j++) {
            // Round j's writes must not overtake round j-1's reads.
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < WP; i++) {
                const float8_t v = acc[i * WQ + j];
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    slab[(i * 16 + 2 * e + row_half) * STG + ml] = v[e];
                }
            }
            __syncthreads();
            // 32 lanes per WROWS rows of M; WROWS is 32 or 64.
            #pragma unroll
            for (int mc = 0; mc < WROWS; mc += 32) {
                const int m_local = mc + lane;
                const int out_m = m_wave + m_local;
                if (out_m >= M) {
                    continue;
                }
                const float bv = has_bias ? Bias[out_m] : 0.0f;
                const float gv = (EPI & EPI_GATED) ? Gate[out_m] : 0.0f;
                for (int bl = 0; bl < 16; bl++) {
                    const int out_b = n_wave + j * 16 + bl;
                    if (out_b >= B) {
                        break;
                    }
                    const long long idx = (long long)out_b * M + out_m;
                    float o = slab[m_local * STG + bl];
                    if (EPI & EPI_ADDIN) {
                        o += C[idx];
                    }
                    o += bv;
                    if (EPI & EPI_GELU) {
                        o = wlds_gelu_tanh(o);
                    }
                    if (EPI & EPI_GATED) {
                        o = __builtin_fmaf(gv, o, R[idx]);
                    }
                    if (EPI & EPI_OUT_F16) {
                        ((_Float16*)Y)[idx] = (_Float16)o;
                    } else {
                        ((float*)Y)[idx] = o;
                    }
                }
            }
        }
        return;
    }

    #pragma unroll
    for (int i = 0; i < WP; i++) {
        #pragma unroll
        for (int j = 0; j < WQ; j++) {
            const int out_b = n_wave + j * 16 + ml;
            if (out_b >= B) {
                continue;
            }
            const float8_t v = acc[i * WQ + j];
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const int out_m = m_wave + i * 16 + 2 * e + row_half;
                if (out_m < M) {
                    const long long idx = (long long)out_b * M + out_m;
                    float o = v[e];
                    if (EPI & EPI_ADDIN) {
                        o += C[idx];
                    }
                    const float bv = has_bias ? Bias[out_m] : 0.0f;
                    o += bv;
                    if (EPI & EPI_GELU) {
                        o = wlds_gelu_tanh(o);
                    }
                    if (EPI & EPI_GATED) {
                        // Explicit fma: the Rust-side parity reference uses
                        // f32::mul_add, so the contraction is pinned on both
                        // sides instead of left to -ffp-contract.
                        o = __builtin_fmaf(Gate[out_m], o, R[idx]);
                    }
                    if (EPI & EPI_OUT_F16) {
                        ((_Float16*)Y)[idx] = (_Float16)o;
                    } else {
                        ((float*)Y)[idx] = o;
                    }
                }
            }
        }
    }
}

// Emit one entry point. The name carries the full parameter vector so the Rust
// side can build it by format!(), and so a perf table row identifies its kernel
// unambiguously: gemm_wmma_lds_<BM>_<BN>_<16·WP>_<16·WQ>_k<KS>[_sw][_p].
// PIPE is the trailing `_p` — see the SOFTWARE PIPELINING note above.
#define WLDS_KERNEL(NAME, BM, BN, WP, WQ, KS, SWAPXY, PIPE)                    \
    __launch_bounds__((BM / (16 * WP)) * (BN / (16 * WQ)) * 32, 1)             \
    extern "C" __global__ void NAME(                                           \
        const _Float16* __restrict__ A,                                        \
        const _Float16* __restrict__ X,                                        \
        float*          __restrict__ Y,                                        \
        const float*    __restrict__ Bias,                                     \
        int M, int K, int B, int has_bias, int lda, int ldx                    \
    ) {                                                                        \
        __shared__ __align__(32) _Float16 a_lds[(KS / 16) * (BM) * 16];        \
        __shared__ __align__(32) _Float16 x_lds[(KS / 16) * (BN) * 16];        \
        gemm_f16_wmma_lds_tile<BM, BN, WP, WQ, KS, SWAPXY, 0, false, PIPE>(    \
            A, X, Y, Bias, M, K, B, has_bias, lda, ldx,                        \
            nullptr, nullptr, nullptr, a_lds, x_lds, nullptr);                 \
    }

// Same, with a fused epilogue. Three pointers longer; `Y` is void* because its
// element type is EPI-dependent. Name = the EPI = 0 name plus the epilogue
// suffix, which is what LdsTile::entry_epi builds on the Rust side.
#define WLDS_KERNEL_EPI(NAME, BM, BN, WP, WQ, KS, SWAPXY, EPI, PIPE)           \
    __launch_bounds__((BM / (16 * WP)) * (BN / (16 * WQ)) * 32, 1)             \
    extern "C" __global__ void NAME(                                           \
        const _Float16* __restrict__ A,                                        \
        const _Float16* __restrict__ X,                                        \
        void*                        Y,                                        \
        const float*    __restrict__ Bias,                                     \
        int M, int K, int B, int has_bias,                                     \
        const float* C, const float* R, const float* Gate,                     \
        int lda, int ldx                                                       \
    ) {                                                                        \
        /* One buffer: the epilogue slab reuses the K staging, which is dead    \
           by then, so LDS is exactly what the EPI = 0 entry uses. */           \
        __shared__ __align__(32) char lds_raw[WLDS_KSTAGE_BYTES(BM, BN, KS)];  \
        _Float16* a_lds = (_Float16*)lds_raw;                                  \
        _Float16* x_lds = a_lds + (KS / 16) * (BM) * 16;                       \
        gemm_f16_wmma_lds_tile<BM, BN, WP, WQ, KS, SWAPXY, EPI,                \
                               WLDS_CST_FITS(BM, BN, WP, WQ, KS), PIPE>(       \
            A, X, Y, Bias, M, K, B, has_bias, lda, ldx, C, R, Gate,            \
            a_lds, x_lds, (float*)lds_raw);                                    \
    }

// The five combinations Task 5 dispatches, for one tile. Nothing else is
// instantiated: each entry is a full kernel in this translation unit, and the
// JIT compiles the whole file per module — the cross product (16 masks × every
// tile) would cost minutes of hipcc for entries no caller names.
#define WLDS_EPI_SET(BASE, BM, BN, WP, WQ, KS, SWAPXY, PIPE)                   \
    WLDS_KERNEL_EPI(BASE##_o16,  BM, BN, WP, WQ, KS, SWAPXY, EPI_OUT_F16, PIPE)\
    WLDS_KERNEL_EPI(BASE##_o16g, BM, BN, WP, WQ, KS, SWAPXY,                   \
                    EPI_OUT_F16 | EPI_GELU, PIPE)                              \
    WLDS_KERNEL_EPI(BASE##_gr,   BM, BN, WP, WQ, KS, SWAPXY, EPI_GATED, PIPE)  \
    WLDS_KERNEL_EPI(BASE##_gra,  BM, BN, WP, WQ, KS, SWAPXY,                   \
                    EPI_GATED | EPI_ADDIN, PIPE)                               \
    WLDS_KERNEL_EPI(BASE##_a,    BM, BN, WP, WQ, KS, SWAPXY, EPI_ADDIN, PIPE)

// --- KS = 64, wave tile 32×64 (I_lds 10.67) — the shipped inner loop --------
WLDS_KERNEL(gemm_wmma_lds_128_128_32_64_k64,    128, 128, 2, 4, 64, false, false)
WLDS_KERNEL(gemm_wmma_lds_256_128_32_64_k64,    256, 128, 2, 4, 64, false, false)
WLDS_KERNEL(gemm_wmma_lds_128_256_32_64_k64,    128, 256, 2, 4, 64, false, false)
WLDS_KERNEL(gemm_wmma_lds_256_256_32_64_k64,    256, 256, 2, 4, 64, false, false)

// --- KS = 64, wave tile 64×64 (I_lds 16.00), 128 accumulator VGPRs ---------
WLDS_KERNEL(gemm_wmma_lds_128_128_64_64_k64,    128, 128, 4, 4, 64, false, false)
WLDS_KERNEL(gemm_wmma_lds_256_128_64_64_k64,    256, 128, 4, 4, 64, false, false)
WLDS_KERNEL(gemm_wmma_lds_128_256_64_64_k64,    128, 256, 4, 4, 64, false, false)
WLDS_KERNEL(gemm_wmma_lds_256_256_64_64_k64,    256, 256, 4, 4, 64, false, false)

// --- KS = 32: halves the LDS footprint so BN can go past 256 ---------------
// 128×256 at KS 32 is the control that separates "shallower stage" from
// "wider BN"; the 512-wide tiles are the actual test of the locality model.
WLDS_KERNEL(gemm_wmma_lds_128_256_64_64_k32,    128, 256, 4, 4, 32, false, false)
WLDS_KERNEL(gemm_wmma_lds_64_512_64_64_k32,      64, 512, 4, 4, 32, false, false)
WLDS_KERNEL(gemm_wmma_lds_128_512_64_64_k32,    128, 512, 4, 4, 32, false, false)
WLDS_KERNEL(gemm_wmma_lds_128_512_32_64_k32,    128, 512, 2, 4, 32, false, false)

// --- Swapped grid axes: grid.x along B instead of M -------------------------
// If the win from a wide BN really comes from which operand the resident blocks
// share, swapping the axes should reverse the preference and make BM the axis
// that matters. This is the direct test of that model.
WLDS_KERNEL(gemm_wmma_lds_128_256_64_64_k64_sw, 128, 256, 4, 4, 64, true, false)
WLDS_KERNEL(gemm_wmma_lds_256_128_64_64_k64_sw, 256, 128, 4, 4, 64, true, false)
WLDS_KERNEL(gemm_wmma_lds_128_512_64_64_k32_sw, 128, 512, 4, 4, 32, true, false)

// --- Software-pipelined main loop (PIPE, the `_p` suffix) -------------------
// Four of the five tiles Gpu::lds_tile_for can return get a pipelined twin:
// each costs one more kernel here (six more with its epilogue set), and this
// file is one translation unit the JIT compiles as a whole.
// Gpu::LDS_TILE_VARIANTS and Gpu::LDS_EPI_TILES carry the same four, and
// `epi_tiles_cover_selector` asserts it.
//
// MEASURED, hipcc -Rpass-analysis=kernel-resource-usage, gfx1150. VGPR/
// occupancy(waves per SIMD) of the EPI = 0 entry, base → pipelined:
//
//   tile                  base       pipelined   verdict
//   128×256 / 32×64 k64   126 / 8    126 / 8     free
//   128×128 / 32×64 k64   121 / 8    138 / 7     4 → 3 workgroups per WGP
//   256×256 / 64×64 k64   190 / 5    227 / 4     free — a 512-thread block on
//                                                a WGP needs exactly 4, and
//                                                its 64 KB of LDS already
//                                                pinned it to one workgroup
//   128×256 / 64×64 k32   190 / 5    214 / 4     free — 256 threads = 2 waves
//                                                per SIMD, so both round to 2
//                                                workgroups
//
// REJECTED, on record: 128×128 / 64×64 k64. Its block is 128 threads, so each
// thread stages A_PASS = X_PASS = 4 vectors of each operand — 64 extra VGPRs on
// top of 128 accumulators — and it compiles at vgpr 254 with **16 VGPR spills**
// (8 on the epilogue entries). Zero spills is a hard requirement, so that tile
// keeps the non-pipelined loop only; it is the gfx1100 fallback for shapes too
// small for 128×256, where the K loop is short anyway.
WLDS_KERNEL(gemm_wmma_lds_128_256_32_64_k64_p,  128, 256, 2, 4, 64, false, true)
WLDS_KERNEL(gemm_wmma_lds_128_128_32_64_k64_p,  128, 128, 2, 4, 64, false, true)
WLDS_KERNEL(gemm_wmma_lds_256_256_64_64_k64_p,  256, 256, 4, 4, 64, false, true)
WLDS_KERNEL(gemm_wmma_lds_128_256_64_64_k32_p,  128, 256, 4, 4, 32, false, true)

// --- Fused-epilogue instantiations ------------------------------------------
// Only the five tiles Gpu::lds_tile_for can return (the three per-arch
// preference chains plus the LDS_TILE_FALLBACK), times the five combinations
// the FLUX single/double blocks use, times the two main-loop forms.
// Gpu::LDS_EPI_TILES mirrors this list and a unit test asserts the two agree,
// so a new entry in a preference chain without a matching instantiation here
// fails `cargo test --lib`, not at runtime.
WLDS_EPI_SET(gemm_wmma_lds_128_256_32_64_k64, 128, 256, 2, 4, 64, false, false)  // gfx1150/1151/portable
WLDS_EPI_SET(gemm_wmma_lds_128_128_32_64_k64, 128, 128, 2, 4, 64, false, false)  // fallback, all archs
WLDS_EPI_SET(gemm_wmma_lds_256_256_64_64_k64, 256, 256, 4, 4, 64, false, false)  // gfx1151
WLDS_EPI_SET(gemm_wmma_lds_128_256_64_64_k32, 128, 256, 4, 4, 32, false, false)  // gfx1100
WLDS_EPI_SET(gemm_wmma_lds_128_128_64_64_k64, 128, 128, 4, 4, 64, false, false)  // gfx1100 fallback

WLDS_EPI_SET(gemm_wmma_lds_128_256_32_64_k64_p, 128, 256, 2, 4, 64, false, true)
WLDS_EPI_SET(gemm_wmma_lds_128_128_32_64_k64_p, 128, 128, 2, 4, 64, false, true)
WLDS_EPI_SET(gemm_wmma_lds_256_256_64_64_k64_p, 256, 256, 4, 4, 64, false, true)
WLDS_EPI_SET(gemm_wmma_lds_128_256_64_64_k32_p, 128, 256, 4, 4, 32, false, true)
