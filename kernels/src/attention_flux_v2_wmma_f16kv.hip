// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.
//
// Non-causal flash attention specialised to the FLUX.1-dev MMDiT shape:
// head_dim == 128, K/V f16, no mask, no KV cache.
//
// Third generation. `vt` fixed the PV B-fragment (transpose V while staging);
// `vtk` additionally staged K. Both were then measured as *barrier- and
// gather-bound* rather than LDS- or DRAM-bound, at ~38 % of the f16 WMMA peak
// on gfx1151. Per 64-key tile, per wave, `vtk` issues
//
//     64 WMMA, 8 __syncthreads, ~136 ds_read_b128,
//     64 scalar global_load_u16 (the V transpose), 32 ds_write_b16 (S)
//
// — i.e. one barrier per 8 WMMA, and the 64 scalar V gathers sit immediately
// before a barrier, so their latency is drained by the whole workgroup rather
// than overlapped. This kernel attacks exactly those three numbers and leaves
// the arithmetic identical:
//
//   1. **8 barriers -> 2.** K and V get separate LDS regions instead of
//      aliasing one, so the whole 64-key tile is staged in a single
//      write phase: `barrier; load+write K and V; barrier; QK + softmax + PV`.
//      All of the tile's global loads are issued back to back before the
//      barrier, so one memory latency is exposed per tile instead of four
//      serialised ones. S needs no barrier at all — a wave reads back only its
//      own 16 rows — exactly as in `vt`/`vtk`.
//   2. **128 query rows per workgroup** (8 waves of 16 rows) instead of 64.
//      The staged K/V tile is shared by twice as many query rows, so the
//      staging traffic *and* the barrier count per unit of WMMA work halve
//      again. Per query row this kernel stages a quarter of what `vtk` does.
//   3. **64 scalar `global_load_u16` -> 16 `global_load_dword`.** The V
//      transpose is done with `v_perm_b32` on register pairs, not by gathering
//      one half at a time. Lane `l` loads the dword pair `V[k][2l, 2l+1]` and
//      `V[k][2l+64, 2l+65]` for its 8 keys (a fully coalesced 128 B per
//      instruction), then two `__builtin_amdgcn_perm` per key-pair interleave
//      two keys' halves at a fixed `d`. The LDS store stays `ds_write_b128`.
//
// Everything else is carried over verbatim from `vtk`: Q fragments resident in
// registers, online softmax max/sum in registers with 4 `shfl_xor` per group of
// 8 rows, S written to LDS once already exponentiated, f32 accumulation, and
// the `{Q dtype} x {out dtype}` template surface over {float, _Float16}.
//
// Layout notes (LDS banking, 32 banks x 4 B; "floor" = the cycles a wave32
// access of that width needs even with zero conflicts):
//   * `Kl` row stride 136 halves (272 B): 16 lanes read 16 rows 68 dwords
//     apart, gcd(68, 32) = 4, so the 16 b128 reads start on 8 distinct banks
//     and cover all 32 exactly twice — the 4-cycle floor.
//   * `Vt` row stride 72 halves (144 B): same gcd(36, 32) = 4 property on the
//     read side. On the *write* side lane `l` owns rows 2l and 2l+1 (and +64),
//     i.e. a 72-dword lane stride, 72 mod 32 = 8, so the four `ds_write_b128`
//     per tile run at 8 cycles against a floor of 4. That is 16 extra LDS
//     cycles per wave-tile against ~544 spent on the fragment reads, and the
//     alternatives all cost more: a 16 B-aligned row stride is necessarily a
//     multiple of 4 dwords, so *any* mapping that gives a lane adjacent rows
//     collides, and the XOR swizzle that fixes it reorders the two 8-key
//     groups inside a 16-key B-fragment (the k-order would then differ per
//     lane, which a WMMA cannot express).
//   * `Slds` row stride 72 halves (144 B): gcd(36, 32) = 4 again.
//
// Grid  = [n_heads, ceil(B/128)], block = [256] (8 waves), dynamic LDS 54272 B
// (53 KB -> 1 workgroup/CU). Occupancy is 8 waves/CU against `vtk`'s 12, and
// **LDS is what caps it, not registers**: a second resident workgroup would
// need 2 x 54272 = 108544 B against the 65536 B a CU has. Registers have room
// to spare — two 8-wave workgroups is 16 waves over 4 SIMDs, i.e. 2 waves/SIMD,
// and a gfx11 SIMD holds 1536 VGPRs, so 6 wave32 at the 256-VGPR ceiling would
// fit. Only cutting LDS to <= 32 KB would buy the second workgroup, and there
// is no room to: K[64][136] + Vt[128][72] + S[128][72] is already the minimum
// this tile geometry needs (K rows are head_dim long, so `AV2_K_STRIDE` cannot
// go below 128 + pad).
//
// So the trade is a third of the wave slots for a quarter of the staging and a
// quarter of the barriers. It pays on every arch measured so far — 1.68x on
// gfx1150, 1.17-1.22x on gfx1151, 1.18x on gfx1100 — but it is measured, never
// assumed: the `vt`/`vtk` ranking inverts between gfx1150 and gfx1151. See
// `Gpu::attention_flux_best_f16kv_f32` for the routing table.
//
// Register budget: 256 VGPRs (the wave32 architectural ceiling), 0 SGPR
// spills, and 1-2 VGPRs parked in scratch — one `scratch_store_b32` in the tile
// loop's *preheader* and one reload at its exit, with zero scratch traffic
// inside the loop. Several structural choices below (the softmax scalars, the
// split perm stores, the un-hoisted staging addresses) exist only to hold that
// line; each is marked where it occurs. **Re-check this on any toolchain bump**
// (`hipcc --genco -Rpass-analysis=kernel-resource-usage`, then `hipcc -S` and
// confirm no `scratch_` inside the tile loop body): the kernel sits exactly at
// the ceiling, so a scheduling change of one register is the difference between
// this and real spill traffic in the hot loop.

// ── Wave 2 attempts: what was tried on top of this kernel, and what it cost ──
//
// A full session (2026-09-04) tried to move this kernel from ~45% of the f16
// peak toward 60% by attacking the *softmax*. Five hypotheses were implemented
// and measured on gfx1150; one was worth ~1%, three lost badly, one was
// arithmetically impossible. None shipped, and the working variant (a copy of
// this file called `v3`) was deleted rather than carried — 1.6% that nothing
// dispatches to does not pay for a 595-line clone with no drift detector.
// The measurements are the deliverable, and they are here so the next author
// reads them before rebuilding any of it.
//
// Everything below was measured with the interleaved `AB=` mode of
// `crates/rdna-compute/examples/bench_attention_flux_vt.rs`: a clock soak,
// then N rounds of one timed launch of each variant back to back, order
// alternating between rounds, medians reported per order. **Use that mode for
// any A/B on this kernel.** The block-per-variant sweep in the same example
// hands whichever kernel runs first a several-percent head start on a small
// part — the same v2 cell reads 33.9 ms cold and 42.9 ms warm on gfx1150,
// which is larger than every delta below. Never compare an absolute number
// from one session against one from another.
//
//   * **exp2 with a pre-scaled log2(e): +0.95% (gfx1150), +1.57% (gfx1151).
//     Correct, kept nowhere.** This kernel folds the softmax `scale` into the
//     WMMA accumulator and then calls `__expf` twice per tile per row group.
//     `__expf` is not one instruction: it is `v_mul_f32` by `log2(e)` then
//     `v_exp_f32`. Folding `log2(e)` into the *same* `v_fma` that already
//     applies `scale`, and swapping both `__expf` for `__builtin_amdgcn_exp2f`,
//     makes both exponentials a bare `v_exp_f32` — the running max, the
//     rescale exponent and the score then all live in the log2 domain. It is a
//     three-line change and it removes 40 `v_mul_f32` per wave per 64-key
//     tile (8 in the `alpha` rescale, 32 in the exponential loop), confirmed
//     by `hipcc -S --offload-arch=gfx1150`: `v_mul_f32` in the tile-loop body
//     falls 176 -> 136 with `v_exp_f32` unchanged at 40 and
//     `v_wmma_f32_16x16x16_f16_w32` unchanged at 64.
//
//     Not bit-exact — `exp2(x * log2e)` and `exp(x)` round differently in the
//     last place — but numerically free: at the FLUX shape it matched this
//     kernel's own error against the v5 reference to four significant figures
//     (v2 rel_inf 4.035e-4 / rel_l2 3.037e-4, variant 4.034e-4 / 3.037e-4,
//     ratio 1.000x on both; variant-vs-v2 directly 1.796e-5 / 5.884e-6).
//     If you re-derive this, fold `log2(e)` into `scale` *inside* the kernel,
//     not into Q on the host: pre-scaling Q double-rounds and breaks the
//     bit-exactness of the f16-Q entry against the f32-Q one.
//
//     gfx1150, five interleaved runs, medians:
//         v2 40.96 -> 40.76 ms  +0.49%      v2 42.89 -> 42.50 ms  +0.91%
//         v2 33.94 -> 33.62 ms  +0.95%      v2 35.08 -> 34.67 ms  +1.18%
//         v2 36.38 -> 35.82 ms  +1.55%
//     gfx1151, one interleaved run: v2 11.56 -> 11.37 ms, +1.57%.
//     Positive everywhere, and a fifth of the 5% a route change asks for.
//
//     **The ISA says why, and that is the useful part.** gfx11 VOPD packs two
//     VALU ops into one issue slot. Removing those 40 multiplies dropped the
//     tile body's dual-issue pairs from 136 to 110 while total issue slots
//     fell only 1495 -> 1487. Eight slots out of 1495 is 0.5% — the measured
//     result to three digits. An instruction count is not a cost model on this
//     part: the softmax multiplies were riding in VOPD slots the scheduler had
//     spare, so deleting them frees nothing.
//
//   * **Conditional accumulator rescale: -17.5% and -19.7%. Rejected.**
//     This kernel unconditionally executes `alpha = exp(m_run - mnew)`,
//     `l_run *= alpha` and 64 `Of[dc][j] *= alpha` per wave per tile, and after
//     the first few tiles `alpha` is exactly 1.0f for most rows, so all 73 of
//     those VALU ops are identities. Guarding them with a wave-uniform
//     `if (__any(alpha != 1.0f))` is correct — skipping is bit-exact, since
//     `x * 1.0f == x` — and even *improves* the register picture: the branch
//     splits the live ranges enough to go from 256 VGPRs with 1-2 spilled and
//     8-12 B/lane of scratch down to **220 VGPRs, 0 spills, 0 scratch**. It is
//     still a large loss: gfx1150 interleaved, 40.64 -> 47.76 ms (-17.5%) and
//     40.51 -> 48.51 ms (-19.7%).
//
//     The cause is the branch, not the restructuring it needed. Keeping the
//     two-pass form (rescale factors computed in one loop, applied in the next)
//     but forcing the branch always-taken measures 42.89 -> 43.13 ms, -0.55%,
//     inside noise — and puts the registers straight back to 256 with a spill.
//     So the split is free and the `s_cbranch` costs ~18%: with two waves per
//     SIMD there is nothing to fill the gap, and the branch also stops the
//     scheduler interleaving the rescale with the exponential loop and the
//     `ds_write_b16`s that follow. **Do not re-try this as a per-row
//     `v_cndmask`** — a select does not remove the multiply, which is the only
//     thing the branch was there to remove.
//
//   * **`__shfl_xor` -> DPP row reductions: -27.0% and -27.2%. Rejected.**
//     `__shfl_xor` lowers to `ds_bpermute_b32`, which occupies the LDS
//     crossbar — the same pipe as the `ds_load_b128`s this kernel is bound by.
//     There are 64 per wave per tile (8 rows x 4 masks, across the max and the
//     sum reduction), a quarter of the tile's 240 LDS operations, so moving
//     them to the VALU is the one lever the cost model below actually points
//     at.
//
//     It works and it is bit-exact. DPP16 has no control for xor 4 or xor 8,
//     but the butterfly **{1, 2, 7, 15}** is all DPP — `quad_perm:[1,0,3,2]`,
//     `quad_perm:[2,3,0,1]`, `row_half_mirror`, `row_mirror` — and builds the
//     *same* reduction tree: after the two `quad_perm` steps every lane holds
//     its quad's result, and pairing quads by ^7 then ^15 pairs exactly the
//     same quads in the same order as ^4 then ^8, so even the non-associative
//     sum is bit-identical (parity confirmed: 240 checks ALL PASS, unchanged
//     error). Registers improve to **217 VGPRs, 0 spills, 0 scratch**.
//
//     And it is 27% slower: gfx1150 interleaved, 38.90 -> 49.40 ms and
//     37.27 -> 47.40 ms. All 64 `ds_bpermute_b32` are gone; the ISA shows what
//     replaced them. `v_mov_b32_dpp` does not fuse into its consumer, so each
//     stage stays two instructions; DPP operands carry a VALU hazard the
//     scheduler pays for with `s_delay_alu` (102 -> 178 in the tile body); and
//     **DPP instructions cannot go in a VOPD slot**, so the 8 independent row
//     chains lose their dual issue. Net: 1487 -> 1563 issue slots plus 76 more
//     delay slots, to save 64 LDS-crossbar ops.
//
//     The lesson generalises past this kernel: on gfx11 a cross-lane op through
//     the LDS crossbar is *cheaper* than the DPP sequence that avoids it,
//     because the crossbar runs in parallel with the VALU while DPP competes
//     for the issue slot and forfeits VOPD packing. Do not "optimise"
//     `__shfl_xor` here again without re-measuring.
//
//   * **128 keys per tile: does not fit. Not buildable.** The three LDS regions
//     are sized by the tile: `K[NT][head_dim + pad]`, `Vt[head_dim][NT + pad]`,
//     `S[MT][NT + pad]`. Every stride must be a multiple of 8 halves for
//     `ds_read_b128` / `ds_write_b128`, so the smallest legal padded stride at
//     NT = 128 is 136 halves:
//         K  128 x 136 x 2 = 34816 B
//         Vt 128 x 136 x 2 = 34816 B
//         S  128 x 136 x 2 = 34816 B   -> 104448 B, against a CU's 65536 B.
//     Dropping to MT = 64 only shrinks S to 17408 B, for 87040 B — still 1.33x
//     over. Splitting the K staging into two 64-key halves does not help: S
//     alone must hold 128 columns for all MT rows, so S + Vt is already
//     69632 B. There is no arrangement of a 128-key tile that fits. The
//     register side is independently fatal: the score accumulator `sa` is
//     `NT/16` f32x8 fragments, 32 VGPRs at NT = 64 and 64 at NT = 128, on a
//     kernel already at the 256-VGPR wave32 ceiling.
//
//   * **LDS <= 32 KB: unreachable at this geometry; the reachable 36864 B is
//     worth zero.** The suggested route was to stream K through registers
//     straight into WMMA B fragments so only V^T and S stay in LDS. It does not
//     work for *this* decomposition: the QK^T B fragment is indexed `n = key`,
//     so lane `h` of a wave needs the full head_dim of key `c*16 + h`, and
//     every wave needs all 64 keys of the tile, not the 8 it stages. Holding
//     the tile in registers is 64 x 128 halves / 32 lanes = 128 VGPRs per lane;
//     having each wave re-load the tile from global instead is exactly what
//     `vt` does, and this kernel beats `vt` by 1.85x on gfx1150.
//
//     What *is* reachable was built and measured rather than argued about:
//     aliasing S onto K, dead once phase A ends, at the cost of a third barrier
//     per tile — max(17408, 18432) + 18432 = **36864 B**, a 32% cut. It passes
//     parity and it is worth nothing: gfx1150 interleaved, +1.58%, +0.36%,
//     +0.77% across three runs, the same band as the exp2 change alone that was
//     also in that build, so the LDS cut and the extra barrier both measure as
//     zero. That is the expected result, and the reason 32768 B was the target
//     rather than "less than now": 36864 B is still over half of 65536 B, so
//     the second resident workgroup that would double the 8 waves/CU never
//     appears, and without it a smaller footprint buys nothing.
//     **Anything short of 32768 B is not worth building.**
//
// ── Where the time actually goes ────────────────────────────────────────────
//
// Four independent results above point the same way, so record the model.
// Counted from the gfx1150 ISA of this kernel's f32/f32 entry, one pass of the
// tile loop per wave is
//
//     64  v_wmma_f32_16x16x16_f16_w32
//    136  ds_load_b128        (K and V^T B fragments, S A fragments)
//     64  ds_bpermute_b32     (the 8 x 4 __shfl_xor in the two row reductions)
//     40  ds_store_*          (S, plus the staging writes)
//     56  global_load_*
//   1495  issue slots in total, 136 of them VOPD pairs.
//
// The LDS pipeline therefore sees 240 operations per wave per tile, and the
// `ds_load_b128`s alone are ~544 cycles at their 4-cycle floor — against
// roughly 2048 cycles of WMMA issue per SIMD for the same tile at two waves per
// SIMD. Those two numbers are the same order, which is exactly what "45% of the
// f16 peak" means, and it is why every *arithmetic* saving above measured as
// zero: the VALU is not the queue with the line in front of it.
//
// The levers that remain are all on the LDS side:
//   * **fewer fragment reads per WMMA** — each B fragment is read once and used
//     once, which a second query row-block per wave would halve. At head_dim
//     128 that is 320+ VGPRs, so it is a different decomposition, not a tweak;
//   * **hiding the per-tile global-load latency** by prefetching tile t+1's K/V
//     into registers during tile t's compute.
// Neither is a softmax change. That is the substantive conclusion of wave 2:
// **60% of peak is not reachable by touching the softmax.**

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(4))) half4_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef unsigned int __attribute__((ext_vector_type(4))) uint4_t;

#define AV2_HD 128
#define AV2_DC 8    // AV2_HD / 16
#define AV2_MT 128  // query rows per block (8 waves x 16)
#define AV2_NT 64   // keys per online-softmax tile
#define AV2_NC 4    // AV2_NT / 16
#define AV2_WAVES 8
#define AV2_KPW (AV2_NT / AV2_WAVES)  // keys staged per wave = 8
#define AV2_VT_STRIDE 72              // halves; 144 B, 16 B aligned
#define AV2_K_STRIDE 136              // halves; 272 B, 16 B aligned
#define AV2_S_STRIDE 72               // halves; 144 B, 16 B aligned
#define AV2_K_HALVES (AV2_NT * AV2_K_STRIDE)   // 8704
#define AV2_VT_HALVES (AV2_HD * AV2_VT_STRIDE)  // 9216
#define AV2_S_HALVES (AV2_MT * AV2_S_STRIDE)    // 9216
#define AV2_LDS_BYTES ((AV2_K_HALVES + AV2_VT_HALVES + AV2_S_HALVES) * 2)  // 54272

// 16-byte-aligned pair load; guarantees ds_read_b128 / global_load_dwordx4
// rather than letting the compiler fall back to narrow scalar reads.
__device__ __forceinline__ half16_t av2_ld16(const _Float16 *p) {
    const half8_t *q = (const half8_t *)p;
    half8_t a = q[0];
    half8_t b = q[1];
    half16_t r;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        r[i] = a[i];
        r[i + 8] = b[i];
    }
    return r;
}

// Epilogue store, specialised on the out dtype. KEEP IN SYNC with the copies in
// attention_flux_vt_wmma_f16kv_f32.hip / attention_flux_vtk_wmma_f16kv_f32.hip
// — the empty asm in the `_Float16` specialisation is LOAD-BEARING, not a
// leftover. It pins the f32 product in a VGPR so the narrowing stays a separate
// RNE rounding; without it the backend folds `fptrunc (fmul a, b)` into one
// `v_fma_mixlo_f16` and the f16 entry stops being bit-identical to "f32 entry +
// RNE cast". See attention_flux_vt_wmma_f16kv_f32.hip for the full note.
template <typename OT>
struct av2_store;

template <>
struct av2_store<float> {
    __device__ __forceinline__ static void put(float *p, float v) { *p = v; }
};

template <>
struct av2_store<_Float16> {
    __device__ __forceinline__ static void put(_Float16 *p, float v) {
        asm("" : "+v"(v));
        *p = (_Float16)v;
    }
};

template <typename QT, typename OT>
__device__ __forceinline__ void
av2_body(const QT *__restrict__ q, const _Float16 *__restrict__ k,
         const _Float16 *__restrict__ v, OT *__restrict__ out, int B, int L,
         int n_heads, int n_kv_heads, int head_dim, float scale, int qmajor) {
    // See the base kernel: qmajor puts the query tile on the fast-varying grid
    // axis so the resident workgroups span one head's K/V, not all of them.
    const int head = qmajor ? blockIdx.y : blockIdx.x;
    const int q_start = (qmajor ? blockIdx.x : blockIdx.y) * AV2_MT;
    const int tid = threadIdx.x;

    if (head >= n_heads || q_start >= B) return;
    if (head_dim != AV2_HD) return;

    const int rep = n_heads / n_kv_heads;
    const int kv_head = head / rep;
    const int q_stride = n_heads * AV2_HD;
    const int kv_stride = n_kv_heads * AV2_HD;

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int h = lane & 15;   // WMMA a-row / b-col selector
    const int hi = lane >> 4;  // WMMA c-row parity
    const int rb = wave * 16;  // this wave's query-row base inside the tile

    // half8_t element type so the dynamic-LDS base is 16 B aligned, which is
    // what ds_read_b128 / ds_write_b128 require. Three disjoint regions: K and
    // V^T are both live inside one barrier region (phase A reads K while
    // another wave may already be in phase C reading V^T), so unlike `vtk`
    // they cannot alias.
    extern __shared__ half8_t av2_smem[];
    _Float16 *Kl = (_Float16 *)av2_smem;        // [AV2_NT][AV2_K_STRIDE]
    _Float16 *Vt = Kl + AV2_K_HALVES;           // [AV2_HD][AV2_VT_STRIDE]
    _Float16 *Slds = Vt + AV2_VT_HALVES;        // [AV2_MT][AV2_S_STRIDE]

    // 32-bit index arithmetic throughout: the largest tensor this kernel is
    // used on is far below 2^31 elements, and size_t offsets cost a
    // v_mad_i64_i32 + v_lshlrev_b64 + carry-add per address.
    const _Float16 *kbase = k + kv_head * AV2_HD;
    const _Float16 *vbase = v + kv_head * AV2_HD;

    // ---- Q fragments, resident for the whole kernel (8 x half16 = 64 VGPR) --
    // Rows past B are clamped, not predicated: their outputs are dropped in the
    // epilogue and softmax is row-independent, so a duplicate row is harmless
    // and costs no divergent branch.
    half16_t Qf[AV2_DC];
    {
        const int gq = min(q_start + rb + h, B - 1);
        const QT *qr = q + gq * q_stride + head * AV2_HD;
#pragma unroll
        for (int dc = 0; dc < AV2_DC; ++dc) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                Qf[dc][j] = (_Float16)qr[dc * 16 + j];
            }
        }
    }

    float8_t Of[AV2_DC];
#pragma unroll
    for (int dc = 0; dc < AV2_DC; ++dc) {
        Of[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // Running softmax state. Lane (h, hi) owns rows rb + 2*j + hi, j = 0..7 —
    // exactly the rows its WMMA accumulator holds. All 16 h-lanes of a
    // half-wave hold a redundant copy of the same row's state.
    float m_run[8], l_run[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        m_run[j] = -INFINITY;
        l_run[j] = 0.0f;
    }

    // This wave's 8 keys inside the 64-key tile. The staging *addresses* are
    // deliberately NOT hoisted out of the loop: they are two VALU ops each, and
    // a hoisted pointer stays live across the WMMA phases, where the wave is
    // already at the 256-VGPR wave32 ceiling. Recomputing is free; spilling
    // into scratch inside the tile loop is not.
    const int kk0 = wave * AV2_KPW;

    for (int kt = 0; kt < L; kt += AV2_NT) {
        // ---------------- Stage K and V^T for the whole tile ----------------
        // One barrier before (every wave has finished reading the previous
        // tile's K and V^T) and one after — the whole 64-key tile is staged in
        // between, so the workgroup drains at most one memory latency per tile
        // instead of the four `vtk` serialises across its four staging chunks.
        __syncthreads();
        {
            const int kbeg = kt + kk0;
            // K row-major: lane covers d = 4*lane..+3, one ds_write_b64 per key
            // with the 32 lanes' 2-dword writes spread evenly over the banks.
            // Load and store are written as one loop on purpose: the scheduler
            // is free to hoist all eight `global_load_dwordx2` above the first
            // `ds_write_b64` (LDS and global cannot alias), but it is *not*
            // forced to hold eight half4 live if that would spill.
            {
                const _Float16 *src = kbase + 4 * lane;
                _Float16 *dst = Kl + kk0 * AV2_K_STRIDE + 4 * lane;
#pragma unroll
                for (int r = 0; r < AV2_KPW; ++r) {
                    *(half4_t *)(dst + r * AV2_K_STRIDE) = *(
                        const half4_t *)(src + min(kbeg + r, L - 1) * kv_stride);
                }
            }
            // V transposed, two rows of V^T per d-block. Lane `l` loads the
            // dword `V[key][2l, 2l+1]` (a coalesced 128 B across the wave) for
            // its eight keys, then `perm(b, a, sel)` permutes bytes of the
            // 64-bit pool {b:7..4, a:3..0}: 0x05040100 takes a's low half then
            // b's low half (two keys at the even d), 0x07060302 takes both high
            // halves (the odd d). One register pair yields two `d` rows of two
            // keys — no LDS round trip and no scalar gather.
            //
            // The two d-blocks are staged one after the other rather than
            // together: eight live dwords instead of sixteen is what keeps the
            // tile-staging block inside the 256-VGPR wave32 budget. Staging
            // both at once compiled to 16 spilled VGPRs and 68 B/lane of
            // scratch instead of the 1-2 / 8 B this form costs.
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const _Float16 *src = vbase + 2 * lane + 64 * blk;
                unsigned int vr[AV2_KPW];
#pragma unroll
                for (int r = 0; r < AV2_KPW; ++r) {
                    vr[r] = *(const unsigned int *)(src +
                                                    min(kbeg + r, L - 1) *
                                                        kv_stride);
                }
                _Float16 *dst =
                    Vt + (2 * lane + 64 * blk) * AV2_VT_STRIDE + kk0;
                // Even and odd `d` rows are permuted and stored one at a time:
                // four live result dwords instead of eight, which is what keeps
                // this kernel off scratch.
#pragma unroll
                for (int par = 0; par < 2; ++par) {
                    const unsigned int sel = par ? 0x07060302u : 0x05040100u;
                    uint4_t o;
#pragma unroll
                    for (int t = 0; t < AV2_KPW / 2; ++t) {
                        o[t] = __builtin_amdgcn_perm(vr[2 * t + 1], vr[2 * t],
                                                     sel);
                    }
                    *(uint4_t *)(dst + par * AV2_VT_STRIDE) = o;
                }
            }
        }
        __syncthreads();

        // ---------------- Phase A: S = scale * Q K^T ----------------
        float8_t sa[AV2_NC];
#pragma unroll
        for (int c = 0; c < AV2_NC; ++c) {
            sa[c] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
#pragma unroll
        for (int dc = 0; dc < AV2_DC; ++dc) {
#pragma unroll
            for (int c = 0; c < AV2_NC; ++c) {
                const half16_t breg =
                    av2_ld16(Kl + (c * 16 + h) * AV2_K_STRIDE + dc * 16);
                sa[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(Qf[dc], breg,
                                                                  sa[c]);
            }
        }

        // ---------------- Phase B: online softmax, in registers -------------
        // Fold the softmax scale and the column mask into the accumulator once
        // (in place) so neither the row max nor the exponential repeats it. The
        // mask is a scalar per `c`, not a live array: this wave is already at
        // the 256-VGPR ceiling.
#pragma unroll
        for (int c = 0; c < AV2_NC; ++c) {
            const float cmask = ((kt + c * 16 + h) < L) ? 0.0f : -INFINITY;
#pragma unroll
            for (int j = 0; j < 8; ++j) sa[c][j] = sa[c][j] * scale + cmask;
        }

        // Row max, then the rescale. The row reduction is over the 16 lanes that
        // share `hi`; masks 1/2/4/8 stay inside the half-wave, and the eight `j`
        // chains are independent so the unrolled form still has ILP.
        //
        // `tm` and `alpha` are scalars inside the `j` loop, not arrays, and the
        // O rescale happens here rather than after the exponential loop: both
        // are register-pressure choices, not algorithmic ones. This wave sits at
        // the 256-VGPR wave32 ceiling, and holding either as an 8-wide live
        // range spills.
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float tm = sa[0][j];
#pragma unroll
            for (int c = 1; c < AV2_NC; ++c) tm = fmaxf(tm, sa[c][j]);
            tm = fmaxf(tm, __shfl_xor(tm, 1));
            tm = fmaxf(tm, __shfl_xor(tm, 2));
            tm = fmaxf(tm, __shfl_xor(tm, 4));
            tm = fmaxf(tm, __shfl_xor(tm, 8));
            const float mnew = fmaxf(m_run[j], tm);
            const float alpha =
                (m_run[j] == -INFINITY) ? 0.0f : __expf(m_run[j] - mnew);
            m_run[j] = mnew;
            l_run[j] *= alpha;
#pragma unroll
            for (int dc = 0; dc < AV2_DC; ++dc) Of[dc][j] *= alpha;
        }

        float rs[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rs[j] = 0.0f;
#pragma unroll
        for (int c = 0; c < AV2_NC; ++c) {
            _Float16 *srow = Slds + (rb + hi) * AV2_S_STRIDE + c * 16 + h;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float e = __expf(sa[c][j] - m_run[j]);
                rs[j] += e;
                srow[(2 * j) * AV2_S_STRIDE] = (_Float16)e;
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            rs[j] += __shfl_xor(rs[j], 1);
            rs[j] += __shfl_xor(rs[j], 2);
            rs[j] += __shfl_xor(rs[j], 4);
            rs[j] += __shfl_xor(rs[j], 8);
            l_run[j] += rs[j];
        }

        // ---------------- Phase C: O += P V, V transposed in LDS ------------
        // No barrier: `Slds` rows [rb, rb+16) are private to this wave, and the
        // whole 64-key V^T tile was staged before the barrier above.
#pragma unroll
        for (int c = 0; c < AV2_NC; ++c) {
            const half16_t areg =
                av2_ld16(Slds + (rb + h) * AV2_S_STRIDE + c * 16);
#pragma unroll
            for (int dc = 0; dc < AV2_DC; ++dc) {
                const half16_t breg =
                    av2_ld16(Vt + (dc * 16 + h) * AV2_VT_STRIDE + c * 16);
                Of[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(areg, breg,
                                                                   Of[dc]);
            }
        }
    }

    // ---------------- Epilogue ----------------
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float lv = l_run[j];
        l_run[j] = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
    }
    // j outer so the `gq < B` predicate is evaluated 8 times, not 64.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int gq = q_start + rb + 2 * j + hi;
        if (gq < B) {
            OT *orow = out + gq * q_stride + head * AV2_HD + h;
#pragma unroll
            for (int dc = 0; dc < AV2_DC; ++dc) {
                av2_store<OT>::put(&orow[dc * 16], Of[dc][j] * l_run[j]);
            }
        }
    }
}

// One entry per {Q dtype} x {out dtype}; see the base `vt` kernel. Same kernarg
// layout as `vt` / `vtk`, so the launcher differs only in block size, grid
// divisor and dynamic LDS.
#define AV2_ENTRY(SUFFIX, QT, OT)                                             \
    extern "C" __global__ void __launch_bounds__(256, 1)                      \
        attention_flux_v2_wmma_f16kv##SUFFIX(                                 \
            const QT *__restrict__ q, const _Float16 *__restrict__ k,         \
            const _Float16 *__restrict__ v, OT *__restrict__ out, int B,      \
            int L, int n_heads, int n_kv_heads, int head_dim, float scale,    \
            int qmajor) {                                                     \
        av2_body<QT, OT>(q, k, v, out, B, L, n_heads, n_kv_heads, head_dim,    \
                         scale, qmajor);                                      \
    }

AV2_ENTRY(, float, float)
AV2_ENTRY(_qf16, _Float16, float)
AV2_ENTRY(_of16, float, _Float16)
AV2_ENTRY(_qf16_of16, _Float16, _Float16)
