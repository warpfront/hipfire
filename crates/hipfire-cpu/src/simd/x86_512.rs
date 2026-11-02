// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! AVX-512F row dots for the two MQ4 V2 formats the D2 host-slot expert path
//! reads: qt 44 `Mq4G256V2` (gate/up, `k = 2560`) and qt 53 `Mq4G128V2` (down,
//! `k = 640`).
//!
//! This tier is **bit-identical** to the AVX2 kernels in [`super::x86`], not
//! merely close to them — it is a wider execution of the *same* arithmetic, so
//! swapping tiers never changes a logit. [`super::x86::codes_dot_sum`] keeps two
//! independent 8-lane FMA chains: `dot0` takes the even chunks of a range, `dot1`
//! the odd ones, and `hsum256(dot0 + dot1)` joins them. A chunk pair's sixteen
//! activations are contiguous, so one `zmm` carries both chains at once:
//! lanes 0‥7 *are* `dot0` (the even chunk) and lanes 8‥15 *are* `dot1` (the odd
//! chunk). Every lane then executes exactly the FMA sequence its AVX2 chain
//! executes, in the same order, on the same operands (an FMA is the same fused,
//! single-rounded operation at either width). An odd tail chunk belongs to
//! chain 0 only, so it is applied to lanes 0‥7 under a mask. The join takes the
//! two `ymm` halves of the accumulator, adds them lane-wise (what
//! `_mm256_add_ps(dot0, dot1)` does), and reuses the same `hsum256` tree; the
//! affine association above it is the AVX2 kernels' own.
//!
//! Nothing here reassociates, reorders a reduction, widens a precision or
//! changes an operand: the only new thing is how many lanes one instruction
//! carries. The codes are unpacked with AVX-512 VBMI: one 8-byte payload word
//! is broadcast to every qword lane and `vpmultishiftqb` picks each lane's two
//! nibbles straight into dword lanes ([`dec16`]), so 16 codes cost a
//! broadcast load, one multishift, one mask and one int->float convert.
//!
//! The `Σx` side follows suit: `!CACHED` runs the two `Σx` chains in a second
//! `zmm` with the identical lane mapping, and the cached variants take the same
//! per-128 sums [`super::x86::activation_sums_v2`] prepares for the AVX2 tier.

use super::x86::{f16_pair, f16_quad, hsum256};
use core::arch::x86_64::*;

/// Mask selecting lanes 0‥7 — chain 0's half of the accumulator.
const CHAIN0: __mmask16 = 0x00ff;

/// `vpmultishiftqb` control: qword lane `q` selects the byte at bit `8q`
/// (code `2q` in its low nibble) into its low dword and the byte at bit `8q + 4`
/// (code `2q + 1`) into its high dword.
#[inline]
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
unsafe fn dec_ctrl() -> __m512i {
    let c = |q: i64| (8 * q) | ((8 * q + 4) << 32);
    _mm512_setr_epi64(c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7))
}

/// Sixteen 4-bit codes — the eight payload bytes at `p`, code `2j` in byte `j`'s
/// low nibble and `2j + 1` in its high nibble (the qt 13/44/53 nibble order
/// [`super::x86`] decodes) — one per dword lane, in code order, as `f32`.
///
/// # Safety
///
/// Requires AVX-512F/VBMI; `8` readable bytes at `p`.
#[inline]
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
unsafe fn dec16(p: *const u8, ctrl: __m512i, mask: __m512i) -> __m512 {
    let word = _mm512_set1_epi64((p as *const i64).read_unaligned());
    let fields = _mm512_multishift_epi64_epi8(ctrl, word);
    _mm512_cvtepi32_ps(_mm512_and_si512(fields, mask))
}

/// The two `Σ` chains' join: `hsum256(chain0 + chain1)` with chain 0 in lanes
/// 0‥7 and chain 1 in lanes 8‥15 — the AVX2 kernels' exact reduction tree.
#[inline]
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
unsafe fn join_chains(acc: __m512) -> f32 {
    let chain0 = _mm512_castps512_ps256(acc);
    let chain1 = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(acc)));
    hsum256(_mm256_add_ps(chain0, chain1))
}

/// `(Σ c·x, Σ x)` over the `first..last` 8-lane chunks of 4-bit codes that
/// start `PAYLOAD` bytes into the group: [`super::x86::codes_dot_sum`]`::<4,
/// PAYLOAD, CACHED>` bit for bit, for any `first..last`.
///
/// Two chunk pairs (four chunks, 16 payload bytes, 32 activations) are decoded
/// per step; each pair is one FMA into the shared accumulator, in chunk order.
///
/// # Safety
///
/// Requires AVX-512F (+ AVX2/FMA/F16C), `4` readable payload bytes per chunk of
/// `first..last` at `gptr + PAYLOAD`, and `8` readable `f32` per chunk at `xg`.
/// Memory is touched only within those chunks (the odd tail uses a 4-byte
/// payload read and a masked activation load).
#[inline]
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
pub(super) unsafe fn codes_dot_sum512<const PAYLOAD: usize, const CACHED: bool>(
    gptr: *const u8,
    xg: *const f32,
    first: usize,
    last: usize,
    cached_sum: f32,
) -> (f32, f32) {
    let (ctrl, mask) = (dec_ctrl(), _mm512_set1_epi32(0xf));
    let mut dot = _mm512_setzero_ps();
    let mut sum = _mm512_setzero_ps();
    let mut idx = first;
    while idx + 3 < last {
        let p = gptr.add(PAYLOAD + idx * 4);
        let (c0, c1) = (dec16(p, ctrl, mask), dec16(p.add(8), ctrl, mask));
        let x0 = _mm512_loadu_ps(xg.add(idx * 8));
        let x1 = _mm512_loadu_ps(xg.add(idx * 8 + 16));
        dot = _mm512_fmadd_ps(c0, x0, dot);
        dot = _mm512_fmadd_ps(c1, x1, dot);
        if !CACHED {
            sum = _mm512_add_ps(sum, x0);
            sum = _mm512_add_ps(sum, x1);
        }
        idx += 4;
    }
    if idx + 1 < last {
        let c = dec16(gptr.add(PAYLOAD + idx * 4), ctrl, mask);
        let x = _mm512_loadu_ps(xg.add(idx * 8));
        dot = _mm512_fmadd_ps(c, x, dot);
        if !CACHED {
            sum = _mm512_add_ps(sum, x);
        }
        idx += 2;
    }
    if idx < last {
        // Odd tail: chain 0 only, i.e. lanes 0‥7. Lanes 8‥15 keep their value.
        // Only four payload bytes are readable: the upper dword is zero, and
        // lanes 8‥15 (which would read it) are masked off below.
        let word = (gptr.add(PAYLOAD + idx * 4) as *const u32).read_unaligned();
        let fields = _mm512_multishift_epi64_epi8(ctrl, _mm512_set1_epi64(word as i64));
        let c = _mm512_cvtepi32_ps(_mm512_and_si512(fields, mask));
        let x = _mm512_maskz_loadu_ps(CHAIN0, xg.add(idx * 8));
        dot = _mm512_mask3_fmadd_ps(c, x, dot, CHAIN0);
        if !CACHED {
            sum = _mm512_mask_add_ps(sum, CHAIN0, sum, x);
        }
    }
    let dot = join_chains(dot);
    let sum = if CACHED { cached_sum } else { join_chains(sum) };
    (dot, sum)
}

/// Two independent 128-element blocks (64 payload bytes at `pa` / `pb`, 128
/// activations at `xa` / `xb`) dotted at once: each is [`codes_dot_sum512`]'s
/// `CACHED` body over `0..16` chunks — two FMAs per four-chunk step into its
/// own accumulator, then [`join_chains`] — with the two chains' instructions
/// interleaved so one block's FMA latency hides behind the other's decode.
/// Neither accumulator sees the other's operands, so each result is
/// bit-identical to the block run alone.
///
/// # Safety
///
/// Requires AVX-512F/VBMI (+ AVX2/FMA/F16C), `64` readable payload bytes at
/// each of `pa` / `pb` and `128` readable `f32` at each of `xa` / `xb`.
#[inline]
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
unsafe fn block_pair_dot512(
    pa: *const u8,
    xa: *const f32,
    pb: *const u8,
    xb: *const f32,
) -> (f32, f32) {
    let (ctrl, mask) = (dec_ctrl(), _mm512_set1_epi32(0xf));
    let mut da = _mm512_setzero_ps();
    let mut db = _mm512_setzero_ps();
    for s in 0..4 {
        let (a0, a1) = (
            dec16(pa.add(16 * s), ctrl, mask),
            dec16(pa.add(16 * s + 8), ctrl, mask),
        );
        let (b0, b1) = (
            dec16(pb.add(16 * s), ctrl, mask),
            dec16(pb.add(16 * s + 8), ctrl, mask),
        );
        da = _mm512_fmadd_ps(a0, _mm512_loadu_ps(xa.add(32 * s)), da);
        db = _mm512_fmadd_ps(b0, _mm512_loadu_ps(xb.add(32 * s)), db);
        da = _mm512_fmadd_ps(a1, _mm512_loadu_ps(xa.add(32 * s + 16)), da);
        db = _mm512_fmadd_ps(b1, _mm512_loadu_ps(xb.add(32 * s + 16)), db);
    }
    (join_chains(da), join_chains(db))
}

/// One 256-element `Mq4G256V2` group (`[s0 z0 s1 z1]` `fp16`, then 128 B of
/// nibbles): [`super::x86`]'s `v2_group_dot::<4>` bit for bit. `sums` is read
/// (`sums[0]`, `sums[1]`) only when `CACHED`.
#[inline]
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
unsafe fn v2_group_dot512<const CACHED: bool>(
    gptr: *const u8,
    xg: *const f32,
    sums: *const f32,
) -> f32 {
    let h = f16_quad(gptr);
    if CACHED {
        let (d0, d1) = block_pair_dot512(gptr.add(8), xg, gptr.add(8 + 64), xg.add(128));
        return (h[0] * d0 + h[1] * *sums) + (h[2] * d1 + h[3] * *sums.add(1));
    }
    let (d0, s0) = codes_dot_sum512::<8, false>(gptr, xg, 0, 16, 0.0);
    let (d1, s1) = codes_dot_sum512::<8, false>(gptr, xg, 16, 32, 0.0);
    (h[0] * d0 + h[1] * s0) + (h[2] * d1 + h[3] * s1)
}

/// One 128-element `Mq4G128V2` group (`[scale zero]` `fp16`, then 64 B of
/// nibbles): [`super::x86`]'s `uniform_group_dot::<4, 4, 16, _>` bit for bit.
#[inline]
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
unsafe fn g128_group_dot512<const CACHED: bool>(
    gptr: *const u8,
    xg: *const f32,
    cached_sum: f32,
) -> f32 {
    let (scale, zero) = f16_pair(gptr);
    let (dot, sum) = codes_dot_sum512::<4, CACHED>(gptr, xg, 0, 16, cached_sum);
    scale * dot + zero * sum
}

/// qt 44 — `Mq4G256V2` row dot, 256 elements / 136 B per group. Bit-equal to
/// [`super::x86::mq4g256v2_row_dot`].
///
/// # Safety
///
/// Requires AVX-512F, AVX2, FMA and F16C; `k % 256 == 0`; `(k / 256) * 136`
/// readable bytes at `row`; `k` readable `f32` at `x`.
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
pub(super) unsafe fn mq4g256v2_row_dot(row: *const u8, k: usize, x: *const f32) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 256 {
        acc += v2_group_dot512::<false>(row.add(g * 136), x.add(g * 256), core::ptr::null());
    }
    acc
}

/// qt 44 with the per-128 `Σx` from [`super::x86::activation_sums_v2`]
/// (`sums[2g]`, `sums[2g + 1]` for group `g`). Bit-equal to
/// [`mq4g256v2_row_dot`].
///
/// # Safety
///
/// As [`mq4g256v2_row_dot`], plus `k / 128` readable `f32` at `sums`.
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
pub(super) unsafe fn mq4g256v2_row_dot_cached(
    row: *const u8,
    k: usize,
    x: *const f32,
    sums: *const f32,
) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 256 {
        acc += v2_group_dot512::<true>(row.add(g * 136), x.add(g * 256), sums.add(2 * g));
    }
    acc
}

/// qt 53 — `Mq4G128V2` row dot, 128 elements / 68 B per group. Bit-equal to
/// [`super::x86::mq4g128v2_row_dot`].
///
/// # Safety
///
/// Requires AVX-512F, AVX2, FMA and F16C; `k % 128 == 0`; `(k / 128) * 68`
/// readable bytes at `row`; `k` readable `f32` at `x`.
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
pub(super) unsafe fn mq4g128v2_row_dot(row: *const u8, k: usize, x: *const f32) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 128 {
        acc += g128_group_dot512::<false>(row.add(g * 68), x.add(g * 128), 0.0);
    }
    acc
}

/// qt 53 with the per-128 `Σx` from [`super::x86::activation_sums_v2`]
/// (`sums[g]` for group `g`). Bit-equal to [`mq4g128v2_row_dot`].
///
/// # Safety
///
/// As [`mq4g128v2_row_dot`], plus `k / 128` readable `f32` at `sums`.
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
pub(super) unsafe fn mq4g128v2_row_dot_cached(
    row: *const u8,
    k: usize,
    x: *const f32,
    sums: *const f32,
) -> f32 {
    let groups = k / 128;
    let mut acc = 0.0f32;
    let mut g = 0;
    // Two groups at a time (independent blocks, interleaved); the affine terms
    // still join `acc` in group order.
    while g + 1 < groups {
        let (p0, p1) = (row.add(g * 68), row.add((g + 1) * 68));
        let (d0, d1) =
            block_pair_dot512(p0.add(4), x.add(g * 128), p1.add(4), x.add((g + 1) * 128));
        let (q0, q1) = (f16_quad(p0), f16_quad(p1));
        acc += q0[0] * d0 + q0[1] * *sums.add(g);
        acc += q1[0] * d1 + q1[1] * *sums.add(g + 1);
        g += 2;
    }
    if g < groups {
        acc += g128_group_dot512::<true>(row.add(g * 68), x.add(g * 128), *sums.add(g));
    }
    acc
}

/// Widest token group the AVX-512 multi-token kernels carry: one `zmm` dot
/// accumulator per token (both chains share it) plus the two shared code
/// vectors and an activation temporary, far inside the 32 `zmm` registers.
pub(super) const MULTI_TOKENS_AVX512: usize = 8;

/// The sixteen 8-lane chunks (128 activations) of one V2 half / `Mq4G128V2`
/// group at payload `p`, dotted with `T` activation rows `x[t] + off`, each
/// four-chunk payload decoded **once** for all `T` tokens.
///
/// Per token this is [`codes_dot_sum512`]'s `CACHED` body over `first..last =
/// 0..16` verbatim (two FMAs per four-chunk step into the token's own `zmm`,
/// then [`join_chains`]), so every result is bit-identical to the single-token
/// kernel's; tokens never mix.
///
/// # Safety
///
/// Requires AVX-512F (+ AVX2/FMA/F16C), `64` readable payload bytes at `p` and
/// `128` readable `f32` at `x[t].add(off)` for every `t < T`.
#[inline]
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
unsafe fn block_dots_multi512<const T: usize>(
    p: *const u8,
    x: &[*const f32; T],
    off: usize,
) -> [f32; T] {
    let (ctrl, mask) = (dec_ctrl(), _mm512_set1_epi32(0xf));
    let mut dot = [_mm512_setzero_ps(); T];
    let mut idx = 0;
    while idx < 16 {
        let q = p.add(idx * 4);
        let (c0, c1) = (dec16(q, ctrl, mask), dec16(q.add(8), ctrl, mask));
        for t in 0..T {
            let xb = x[t].add(off + idx * 8);
            dot[t] = _mm512_fmadd_ps(c0, _mm512_loadu_ps(xb), dot[t]);
            dot[t] = _mm512_fmadd_ps(c1, _mm512_loadu_ps(xb.add(16)), dot[t]);
        }
        idx += 4;
    }
    core::array::from_fn(|t| join_chains(dot[t]))
}

/// [`mq4g256v2_row_dot_cached`] for `T` activation rows at once (`x[t]`,
/// `sums[t]`), each weight chunk decoded once. Result `t` is bit-equal to
/// `mq4g256v2_row_dot_cached(row, k, x[t], sums[t])`.
///
/// # Safety
///
/// As [`mq4g256v2_row_dot_cached`] for every token.
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
pub(super) unsafe fn mq4g256v2_row_dot_cached_multi<const T: usize>(
    row: *const u8,
    k: usize,
    x: &[*const f32; T],
    sums: &[*const f32; T],
) -> [f32; T] {
    let mut acc = [0.0f32; T];
    for g in 0..k / 256 {
        let gptr = row.add(g * 136);
        let h = f16_quad(gptr);
        let d0 = block_dots_multi512::<T>(gptr.add(8), x, g * 256);
        let d1 = block_dots_multi512::<T>(gptr.add(8 + 64), x, g * 256 + 128);
        for t in 0..T {
            let (s0, s1) = (*sums[t].add(2 * g), *sums[t].add(2 * g + 1));
            acc[t] += (h[0] * d0[t] + h[1] * s0) + (h[2] * d1[t] + h[3] * s1);
        }
    }
    acc
}

/// [`mq4g128v2_row_dot_cached`] for `T` activation rows at once. Result `t` is
/// bit-equal to `mq4g128v2_row_dot_cached(row, k, x[t], sums[t])`.
///
/// # Safety
///
/// As [`mq4g128v2_row_dot_cached`] for every token.
#[target_feature(enable = "avx512f,avx512vbmi,avx2,fma,f16c")]
pub(super) unsafe fn mq4g128v2_row_dot_cached_multi<const T: usize>(
    row: *const u8,
    k: usize,
    x: &[*const f32; T],
    sums: &[*const f32; T],
) -> [f32; T] {
    let mut acc = [0.0f32; T];
    for g in 0..k / 128 {
        let gptr = row.add(g * 68);
        let (scale, zero) = f16_pair(gptr);
        let d = block_dots_multi512::<T>(gptr.add(4), x, g * 128);
        for t in 0..T {
            acc[t] += scale * d[t] + zero * *sums[t].add(g);
        }
    }
    acc
}
