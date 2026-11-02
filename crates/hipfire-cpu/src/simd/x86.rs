// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! AVX2 + FMA row dots for the formats the CPU offload path spills.
//!
//! Every kernel here computes the same thing — one weight row dotted with a
//! pre-rotated activation — and differs only in how a group's metadata and
//! payload bytes are laid out. So this file is a table of layouts over one
//! decode core rather than a kernel per format:
//!
//! * [`codes8`] — eight `BITS`-wide codes (the chunk of payload one vector
//!   lane group consumes) as `i32` lanes. `BITS` bytes hold eight codes at every
//!   width the formats use, so nibbles, the 2/3/5/6-bit cross-byte packs and
//!   BQ1's sign bits are all the same broadcast-and-shift.
//! * [`codes_dot_sum`] — `(Σ c·x, Σ x)` over a run of chunks.
//! * a per-group header: one `(scale, zero)` from an `f32` pair (the flat HFQ/MQ
//!   families) or an `fp16` pair (qt 45), a per-128 `fp16` quad (the V2 family),
//!   or a `fp16` codebook (the Lloyd family).
//!
//! A group is then `scale·Σ(c·x) + zero·Σx` (affine) or `Σ(cb[c]·x)`
//! (codebook) — mathematically identical to the scalar decode and the reason a
//! group costs a handful of vector FMAs plus two horizontal sums instead of 256
//! scalar multiply-adds. It is deliberately *not* bit-identical to the scalar
//! path: [`super::tests`] bounds the difference relative, and the offload
//! feature's contract is llama.cpp-level coherence, not device parity.
//!
//! Rotation is the caller's business — the activation arrives already
//! FWHT-rotated for the rotated formats (see [`crate::quant`]) — so one kernel
//! serves a rotated format and its byte-identical unrotated sibling.
//!
//! `Mq4G256` (qt 13) keeps its own nibble path ([`mq4_group_dot`]), decoding
//! 32 codes per 16-byte load instead of 8 per 4-byte load.

use core::arch::x86_64::*;

/// Horizontal sum of eight `f32` lanes.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn hsum256(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// The `BITS` bytes at `p` as the little-endian integer they encode: eight
/// `BITS`-wide codes are exactly `BITS` bytes, so a chunk is one packed field.
///
/// Read *exactly* those bytes. A wider load would be one instruction and would
/// run up to three bytes past the group's end on the group's last chunk, which
/// the caller's `row` slice licenses only for the rows that are not the
/// tensor's last.
#[inline]
unsafe fn load_le<const BITS: usize>(p: *const u8) -> u64 {
    match BITS {
        1 => *p as u64,
        2 => (p as *const u16).read_unaligned() as u64,
        4 => (p as *const u32).read_unaligned() as u64,
        5 => (p as *const u32).read_unaligned() as u64 | ((*p.add(4) as u64) << 32),
        _ => unreachable!("load_le handles only 1-, 2-, 4- and 5-bit chunks"),
    }
}

/// Shift counts `[OFF·BITS, …, (OFF+3)·BITS]` for the 64-bit-lane path.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn shifts64<const BITS: usize, const OFF: usize>() -> __m256i {
    let b = BITS as i64;
    let o = (OFF * BITS) as i64;
    _mm256_setr_epi64x(o, o + b, o + 2 * b, o + 3 * b)
}

/// Eight `BITS`-wide codes from the chunk at `p`, in element order, as `i32`
/// lanes (each already masked to its width).
///
/// One broadcast of the chunk plus one variable shift per code. An 8-code chunk
/// is `8·BITS` bits, so widths up to four fit a 32-bit lane and a single shift.
/// Three-bit chunks retain the preceding byte and discard it in the vector
/// shift. Six-bit chunks broadcast two prefix bytes and their six payload bytes,
/// then shuffle two 24-bit halves into the corresponding 128-bit lanes.
/// Five-bit packs need 64-bit lanes, compacted back into eight i32 lanes.
///
/// # Safety
///
/// Requires AVX2 and `BITS` readable payload bytes at `p`. For 3-bit packs,
/// `p.sub(1)` must also be readable; for 6-bit packs, `p.sub(2)` must be
/// readable. These prefix bytes belong to the same allocation's group header
/// or earlier payload, never before the allocation's first byte.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn codes8<const BITS: usize>(p: *const u8) -> __m256i {
    if BITS == 6 {
        // One qword ends exactly at this chunk's last payload byte. Repeat
        // [prefix, code bytes] for each four-code half; the shifts below discard
        // its prefix. Both extra bytes are within the group header or payload.
        let packed = _mm256_set1_epi64x((p.sub(2) as *const i64).read_unaligned());
        let bytes = _mm256_setr_epi8(
            1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4,
            5, 6, 7,
        );
        let v = _mm256_shuffle_epi8(packed, bytes);
        let shifts = _mm256_setr_epi32(8, 14, 20, 26, 8, 14, 20, 26);
        let mask = _mm256_set1_epi32(0x3f);
        _mm256_and_si256(_mm256_srlv_epi32(v, shifts), mask)
    } else if BITS == 3 {
        // Fold the prefix-byte removal into the existing vector shifts. This
        // enables a memory broadcast instead of a scalar load/shift/transfer.
        let v = _mm256_set1_epi32((p.sub(1) as *const i32).read_unaligned());
        let shifts = _mm256_setr_epi32(8, 11, 14, 17, 20, 23, 26, 29);
        _mm256_and_si256(_mm256_srlv_epi32(v, shifts), _mm256_set1_epi32(7))
    } else if BITS <= 4 {
        let v = _mm256_set1_epi32(load_le::<BITS>(p) as i32);
        let b = BITS as i32;
        let shifts = _mm256_setr_epi32(0, b, 2 * b, 3 * b, 4 * b, 5 * b, 6 * b, 7 * b);
        let mask = _mm256_set1_epi32(((1u32 << BITS) - 1) as i32);
        _mm256_and_si256(_mm256_srlv_epi32(v, shifts), mask)
    } else {
        let v = _mm256_set1_epi64x(load_le::<BITS>(p) as i64);
        let mask = _mm256_set1_epi64x(((1u64 << BITS) - 1) as i64);
        let lo = _mm256_and_si256(_mm256_srlv_epi64(v, shifts64::<BITS, 0>()), mask);
        let hi = _mm256_and_si256(_mm256_srlv_epi64(v, shifts64::<BITS, 4>()), mask);
        // Each 64-bit lane now holds one code in its low dword; take dwords
        // 0, 2, 4 and 6 of each half and join them.
        let pick = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let lo = _mm256_permutevar8x32_epi32(lo, pick);
        let hi = _mm256_permutevar8x32_epi32(hi, pick);
        _mm256_set_m128i(_mm256_castsi256_si128(hi), _mm256_castsi256_si128(lo))
    }
}

/// `(Σ c·x, Σ x)` over the `first..last` chunks of eight `BITS`-wide codes that
/// start `PAYLOAD` bytes into the group.
///
/// Both sums are vector accumulators, so a group pays two horizontal sums
/// rather than one per chunk.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn codes_dot_sum<const BITS: usize, const PAYLOAD: usize, const CACHED: bool>(
    gptr: *const u8,
    xg: *const f32,
    first: usize,
    last: usize,
    cached_sum: f32,
) -> (f32, f32) {
    const { assert!((BITS != 3 || PAYLOAD >= 1) && (BITS != 6 || PAYLOAD >= 2)) };
    // Two independent chains per sum, stepped two chunks at a time. A single
    // chain serializes on the `fma`'s four-cycle latency while the decode is
    // several uops per chunk, so a same-chain reuse every *other* iteration
    // (~5+ cycles apart at this decode's throughput) already covers that
    // latency; a wider fold would only add register pressure. `first..last` is
    // arbitrary, so the odd tail folds into the first chain.
    //
    // With `CACHED` the caller has already reduced this range's `Σx`
    // ([`super::prepare_activation_sums`]) and only the dot chain runs; the
    // cached value is returned in the second slot. The dot FMAs and, when
    // `!CACHED`, the sum vectors and their reduction order are identical in
    // both modes.
    let mut dot0 = _mm256_setzero_ps();
    let mut dot1 = _mm256_setzero_ps();
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut idx = first;
    while idx + 1 < last {
        let j = idx + 1;
        let x0 = _mm256_loadu_ps(xg.add(idx * 8));
        let c0 = _mm256_cvtepi32_ps(codes8::<BITS>(gptr.add(PAYLOAD + idx * BITS)));
        let x1 = _mm256_loadu_ps(xg.add(j * 8));
        let c1 = _mm256_cvtepi32_ps(codes8::<BITS>(gptr.add(PAYLOAD + j * BITS)));
        dot0 = _mm256_fmadd_ps(c0, x0, dot0);
        dot1 = _mm256_fmadd_ps(c1, x1, dot1);
        if !CACHED {
            sum0 = _mm256_add_ps(sum0, x0);
            sum1 = _mm256_add_ps(sum1, x1);
        }
        idx += 2;
    }
    if idx < last {
        let x = _mm256_loadu_ps(xg.add(idx * 8));
        let c = _mm256_cvtepi32_ps(codes8::<BITS>(gptr.add(PAYLOAD + idx * BITS)));
        dot0 = _mm256_fmadd_ps(c, x, dot0);
        if !CACHED {
            sum0 = _mm256_add_ps(sum0, x);
        }
    }
    let dot = hsum256(_mm256_add_ps(dot0, dot1));
    let sum = if CACHED {
        cached_sum
    } else {
        hsum256(_mm256_add_ps(sum0, sum1))
    };
    (dot, sum)
}

/// One group with a single `(scale, zero)` over all `CHUNKS` of its chunks:
/// `scale·Σ(c·x) + zero·Σx`.
///
/// The header is read by the caller, which is what lets one group body serve an
/// `f32` header, an `fp16` pair and TQ2/BQ1's *derived* `(d, -d)` / `(2d, -d)`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn uniform_group_dot<
    const BITS: usize,
    const PAYLOAD: usize,
    const CHUNKS: usize,
    const CACHED: bool,
>(
    gptr: *const u8,
    xg: *const f32,
    scale: f32,
    zero: f32,
    cached_sum: f32,
) -> f32 {
    let (dot, sum) = codes_dot_sum::<BITS, PAYLOAD, CACHED>(gptr, xg, 0, CHUNKS, cached_sum);
    scale * dot + zero * sum
}

/// One 256-element V2 group: `[s0 z0 s1 z1]` `fp16` (one pair per 128
/// elements) then `256/BITS` payload bytes.
///
/// Each half is scored separately and then affinely corrected, so a half's
/// header costs two vector FMAs rather than an `fma` per element.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn v2_group_dot<const BITS: usize>(gptr: *const u8, xg: *const f32) -> f32 {
    let h = f16_quad(gptr);
    let (d0, s0) = codes_dot_sum::<BITS, 8, false>(gptr, xg, 0, 16, 0.0);
    let (d1, s1) = codes_dot_sum::<BITS, 8, false>(gptr, xg, 16, 32, 0.0);
    (h[0] * d0 + h[1] * s0) + (h[2] * d1 + h[3] * s1)
}

/// [`v2_group_dot`] for a V2 group whose two per-128 `Σx` are already reduced
/// (`s0`, `s1`): the dot chains and the affine association are unchanged, so
/// the result equals the uncached group bit for bit.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn v2_group_dot_cached<const BITS: usize>(
    gptr: *const u8,
    xg: *const f32,
    s0: f32,
    s1: f32,
) -> f32 {
    let h = f16_quad(gptr);
    let (d0, _) = codes_dot_sum::<BITS, 8, true>(gptr, xg, 0, 16, s0);
    let (d1, _) = codes_dot_sum::<BITS, 8, true>(gptr, xg, 16, 32, s1);
    (h[0] * d0 + h[1] * s0) + (h[2] * d1 + h[3] * s1)
}

/// The `CB`-entry `fp16` codebook at the group's start, widened to `f32` lanes:
/// `lo` holds the first (and, for the 4- and 8-entry books, only) eight entries
/// and `hi` the second eight of a 16-entry book, zero otherwise. The caller
/// loads this once per group rather than once per chunk.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn codebook_tables<const CB: usize>(gptr: *const u8) -> (__m256, __m256) {
    if CB == 16 {
        (
            _mm256_cvtph_ps(_mm_loadu_si128(gptr as *const __m128i)),
            _mm256_cvtph_ps(_mm_loadu_si128(gptr.add(16) as *const __m128i)),
        )
    } else if CB == 8 {
        (
            _mm256_cvtph_ps(_mm_loadu_si128(gptr as *const __m128i)),
            _mm256_setzero_ps(),
        )
    } else {
        (
            _mm256_insertf128_ps(
                _mm256_setzero_ps(),
                _mm_cvtph_ps(_mm_loadl_epi64(gptr as *const __m128i)),
                0,
            ),
            _mm256_setzero_ps(),
        )
    }
}

/// One lane's codebook value for each index in `sel`, against a table already
/// widened by [`codebook_tables`].
///
/// A 4- or 8-entry book is one `vpermd`; a 16-entry book is two plus a blend on
/// the index's top bit. `vpermd` only looks at an index's low three bits, which
/// is what makes the two-table form work — both lookups are valid and the blend
/// picks the half.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn codebook_lookup<const CB: usize>(lo: __m256, hi: __m256, sel: __m256i) -> __m256 {
    if CB == 16 {
        let top = _mm256_slli_epi32(_mm256_and_si256(sel, _mm256_set1_epi32(8)), 28);
        _mm256_blendv_ps(
            _mm256_permutevar8x32_ps(lo, sel),
            _mm256_permutevar8x32_ps(hi, sel),
            _mm256_castsi256_ps(top),
        )
    } else {
        _mm256_permutevar8x32_ps(lo, sel)
    }
}

/// One group of a Lloyd-codebook format: `CB` `fp16` entries at the group's
/// start, then `PAYLOAD` bytes of `BITS`-wide indices.
///
/// There is no affine term — the codebook *is* the decode — so the group needs
/// one dot accumulator, split into two interleaved chains as in
/// [`codes_dot_sum`] to keep the `fma` latency from serializing. The table is
/// widened once, not once per chunk.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn codebook_group_dot<const BITS: usize, const CB: usize, const PAYLOAD: usize>(
    gptr: *const u8,
    xg: *const f32,
) -> f32 {
    const { assert!((BITS != 3 || PAYLOAD >= 1) && (BITS != 6 || PAYLOAD >= 2)) };
    let (cb_lo, cb_hi) = codebook_tables::<CB>(gptr);
    let mut dot0 = _mm256_setzero_ps();
    let mut dot1 = _mm256_setzero_ps();
    // 32 chunks (256 elements) per group, even, so the pair step leaves no tail.
    let mut idx = 0;
    while idx < 32 {
        let j = idx + 1;
        let sel0 = codes8::<BITS>(gptr.add(PAYLOAD + idx * BITS));
        let sel1 = codes8::<BITS>(gptr.add(PAYLOAD + j * BITS));
        dot0 = _mm256_fmadd_ps(
            codebook_lookup::<CB>(cb_lo, cb_hi, sel0),
            _mm256_loadu_ps(xg.add(idx * 8)),
            dot0,
        );
        dot1 = _mm256_fmadd_ps(
            codebook_lookup::<CB>(cb_lo, cb_hi, sel1),
            _mm256_loadu_ps(xg.add(j * 8)),
            dot1,
        );
        idx += 2;
    }
    hsum256(_mm256_add_ps(dot0, dot1))
}

/// `(scale, zero)` from an `f32` header — the flat HFQ/MQ families, 8 B.
#[inline]
unsafe fn f32_pair(p: *const u8) -> (f32, f32) {
    (f32_at(p), f32_at(p.add(4)))
}

/// `(scale, zero)` from an `fp16` pair — qt 45's packed header, and the `d` of
/// the TQ2/BQ1 pair (whose zero the caller derives).
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn f16_pair(p: *const u8) -> (f32, f32) {
    let v = f16_quad(p);
    (v[0], v[1])
}

/// The V2 family's per-128 header, `[s0 z0 s1 z1]`, widened with one
/// `vcvtph2ps`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn f16_quad(p: *const u8) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    _mm_storeu_ps(
        out.as_mut_ptr(),
        _mm_cvtph_ps(_mm_loadl_epi64(p as *const __m128i)),
    );
    out
}

/// One `f32` from the little-endian bytes at `p`.
#[inline]
unsafe fn f32_at(p: *const u8) -> f32 {
    f32::from_le_bytes(*(p as *const [u8; 4]))
}

/// The row driver every format shares: `Σ_g group(row + g·GB, x + g·GE)`.
///
/// A macro rather than a generic function because the group body is a
/// `#[target_feature]` function and its features do not propagate into a
/// closure or through a function pointer, so the call has to be monomorphized
/// in place.
macro_rules! row_dot {
    ($(#[$meta:meta])* $name:ident, $group:expr, $ge:literal, $gb:literal, $feat:literal) => {
        $(#[$meta])*
        ///
        /// # Safety
        ///
        /// Requires the features in this function's own `target_feature` list,
        /// `k % $ge == 0`, at least `(k / $ge) * $gb` readable bytes at `row`
        /// and `k` readable `f32` at `x`. Callers reach it through
        /// [`super::row_dot_avx2`], which resolves the CPU features first.
        #[target_feature(enable = $feat)]
        pub unsafe fn $name(row: *const u8, k: usize, x: *const f32) -> f32 {
            let mut acc = 0.0f32;
            for g in 0..k / $ge {
                acc += ($group)(row.add(g * $gb), x.add(g * $ge));
            }
            acc
        }
    };
}

/// A format whose group is one `(scale, zero)` header over `BITS`-wide codes:
/// the group body is [`uniform_group_dot`] with the header read by `$hdr`.
///
/// `{$ge / 8}` is the group's chunk count — a chunk is eight codes, so it is
/// the element count over eight.
macro_rules! affine_row_dot {
    ($(#[$meta:meta])* $name:ident, $group:ident, $hdr:ident, $bits:literal, $payload:literal, $ge:literal, $gb:literal, $feat:literal) => {
        #[inline]
        #[target_feature(enable = $feat)]
        unsafe fn $group(gptr: *const u8, xg: *const f32) -> f32 {
            let (scale, zero) = $hdr(gptr);
            uniform_group_dot::<$bits, $payload, { $ge / 8 }, false>(gptr, xg, scale, zero, 0.0)
        }
        row_dot!($(#[$meta])* $name, $group, $ge, $gb, $feat);
    };
}

row_dot!(
    /// qt 49 — `Mq3G256V2`: 3-bit cross-byte packs, per-128 `fp16` header.
    mq3g256v2_row_dot,
    v2_group_dot::<3>,
    256,
    104,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 44 — `Mq4G256V2`: the `mq4v2` SKU's format. Nibble payload (qt 13's
    /// geometry) under qt 49's V2 `fp16` header.
    mq4g256v2_row_dot,
    v2_group_dot::<4>,
    256,
    136,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 47 — `Mq6G256V2`: 6-bit packs under the V2 header.
    mq6g256v2_row_dot,
    v2_group_dot::<6>,
    256,
    200,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 48 — `Mq5G256V2`: 5-bit packs under the V2 header.
    mq5g256v2_row_dot,
    v2_group_dot::<5>,
    256,
    168,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 50 — `Mq2G256V2`: 2-bit packs under the V2 header (quality-rejected
    /// for product, still decodes).
    mq2g256v2_row_dot,
    v2_group_dot::<2>,
    256,
    72,
    "avx2,fma,f16c"
);

affine_row_dot!(
    /// qt 53 — `Mq4G128V2`: row-local 128-element groups, 68 B each:
    /// `[f16 scale][f16 zero]` then 64 B of paired nibbles (the same nibble
    /// order as qt 13/44). One `(scale, zero)` covers the whole group, so the
    /// body is [`uniform_group_dot`] over its 16 eight-code chunks with the
    /// payload at byte 4 — the same decode core as every other 4-bit kernel.
    mq4g128v2_row_dot,
    mq4g128v2_group,
    f16_pair,
    4,
    4,
    128,
    68,
    "avx2,fma,f16c"
);

/// qt 44 — `Mq4G256V2` row dot reusing the per-128 `Σx` prepared by
/// [`activation_sums_v2`] (`sums[2g]`, `sums[2g + 1]` for group `g`). Bit-equal
/// to [`mq4g256v2_row_dot`].
///
/// # Safety
///
/// As [`mq4g256v2_row_dot`], plus `k / 128` readable `f32` at `sums`.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn mq4g256v2_row_dot_cached(
    row: *const u8,
    k: usize,
    x: *const f32,
    sums: *const f32,
) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 256 {
        acc += v2_group_dot_cached::<4>(
            row.add(g * 136),
            x.add(g * 256),
            *sums.add(2 * g),
            *sums.add(2 * g + 1),
        );
    }
    acc
}

/// qt 53 — `Mq4G128V2` row dot reusing the per-128 `Σx` prepared by
/// [`activation_sums_v2`] (`sums[g]` for group `g`). Bit-equal to
/// [`mq4g128v2_row_dot`].
///
/// # Safety
///
/// As [`mq4g128v2_row_dot`], plus `k / 128` readable `f32` at `sums`.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn mq4g128v2_row_dot_cached(
    row: *const u8,
    k: usize,
    x: *const f32,
    sums: *const f32,
) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 128 {
        let gptr = row.add(g * 68);
        let (scale, zero) = f16_pair(gptr);
        acc += uniform_group_dot::<4, 4, 16, true>(gptr, x.add(g * 128), scale, zero, *sums.add(g));
    }
    acc
}

/// Widest token group the AVX2 multi-token kernels carry: two `ymm` dot
/// accumulators per token (eight) plus the two shared code vectors and one
/// activation temporary stay within the sixteen `ymm` registers.
pub(super) const MULTI_TOKENS_AVX2: usize = 4;

/// The sixteen 8-lane chunks (128 activations) of one V2 half / `Mq4G128V2`
/// group at payload `p`, dotted with `T` activation rows `x[t] + off`, each
/// chunk's nibbles decoded **once** for all `T` tokens.
///
/// Per token this is [`codes_dot_sum`]'s `CACHED` body over `first..last = 0..16`
/// verbatim: chunk `2i` FMAs into `dot0`, chunk `2i + 1` into `dot1`, then
/// `hsum256(dot0 + dot1)`. Tokens never mix, so each result is bit-identical to
/// the single-token kernel's.
///
/// # Safety
///
/// Requires AVX2/FMA, `64` readable payload bytes at `p` and `128` readable
/// `f32` at `x[t].add(off)` for every `t < T`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_dots_multi<const T: usize>(
    p: *const u8,
    x: &[*const f32; T],
    off: usize,
) -> [f32; T] {
    let mut dot0 = [_mm256_setzero_ps(); T];
    let mut dot1 = [_mm256_setzero_ps(); T];
    let mut idx = 0;
    while idx + 1 < 16 {
        let c0 = _mm256_cvtepi32_ps(codes8::<4>(p.add(idx * 4)));
        let c1 = _mm256_cvtepi32_ps(codes8::<4>(p.add((idx + 1) * 4)));
        for t in 0..T {
            let xb = x[t].add(off + idx * 8);
            dot0[t] = _mm256_fmadd_ps(c0, _mm256_loadu_ps(xb), dot0[t]);
            dot1[t] = _mm256_fmadd_ps(c1, _mm256_loadu_ps(xb.add(8)), dot1[t]);
        }
        idx += 2;
    }
    core::array::from_fn(|t| hsum256(_mm256_add_ps(dot0[t], dot1[t])))
}

/// [`mq4g256v2_row_dot_cached`] for `T` activation rows at once (`x[t]`,
/// `sums[t]`), each weight chunk decoded once. Result `t` is bit-equal to
/// `mq4g256v2_row_dot_cached(row, k, x[t], sums[t])`.
///
/// # Safety
///
/// As [`mq4g256v2_row_dot_cached`] for every token.
#[target_feature(enable = "avx2,fma,f16c")]
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
        let d0 = block_dots_multi::<T>(gptr.add(8), x, g * 256);
        let d1 = block_dots_multi::<T>(gptr.add(8 + 64), x, g * 256 + 128);
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
#[target_feature(enable = "avx2,fma,f16c")]
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
        let d = block_dots_multi::<T>(gptr.add(4), x, g * 128);
        for t in 0..T {
            acc[t] += scale * d[t] + zero * *sums[t].add(g);
        }
    }
    acc
}

/// Per-128-element `Σx` in [`codes_dot_sum`]'s exact order over a 16-chunk
/// range: even 8-lane chunks into `sum0`, odd into `sum1`, then one
/// `hsum256(sum0 + sum1)`. A V2 group's two halves and `Mq4G128V2`'s group each
/// reduce exactly this way, so one prepared value serves both formats.
/// `out[b]` is block `b`'s sum, `b < blocks`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn activation_sums_v2(x: *const f32, out: *mut f32, blocks: usize) {
    for b in 0..blocks {
        let xb = x.add(b * 128);
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut idx = 0;
        while idx + 1 < 16 {
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(xb.add(idx * 8)));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(xb.add((idx + 1) * 8)));
            idx += 2;
        }
        *out.add(b) = hsum256(_mm256_add_ps(sum0, sum1));
    }
}

/// One 256-element `Mq4G256` (qt 13) group: `[f32 scale][f32 zero]` then 128 B
/// of paired nibbles, element `i` taking the low nibble of byte `i/2` for even
/// `i` and the high nibble for odd `i`. The activation arrives already
/// FWHT-rotated, so this kernel only decodes and multiplies.
///
/// `_mm_unpacklo_epi8` of the low/high nibble planes yields `[c0, c1, c2, …]` in
/// element order (byte `j` contributes its low nibble as `c[2j]` and its high
/// nibble as `c[2j+1]`), so no shuffle of the activation is needed.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn mq4_group_dot<const CACHED: bool>(
    gptr: *const u8,
    xg: *const f32,
    cached_sum: f32,
) -> f32 {
    let scale = f32_at(gptr);
    let zero = f32_at(gptr.add(4));
    let mask = _mm_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_ps();
    let mut acc_x = _mm256_setzero_ps();
    for blk in 0..8usize {
        let bytes = _mm_loadu_si128(gptr.add(8 + blk * 16) as *const __m128i);
        let lo = _mm_and_si128(bytes, mask);
        let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        let c_lo = _mm_unpacklo_epi8(lo, hi);
        let c_hi = _mm_unpackhi_epi8(lo, hi);
        let xb = xg.add(blk * 32);
        let c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(c_lo));
        let c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(c_lo, 8)));
        let c2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(c_hi));
        let c3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(c_hi, 8)));
        let x0 = _mm256_loadu_ps(xb);
        let x1 = _mm256_loadu_ps(xb.add(8));
        let x2 = _mm256_loadu_ps(xb.add(16));
        let x3 = _mm256_loadu_ps(xb.add(24));
        acc = _mm256_fmadd_ps(c0, x0, acc);
        acc = _mm256_fmadd_ps(c1, x1, acc);
        acc = _mm256_fmadd_ps(c2, x2, acc);
        acc = _mm256_fmadd_ps(c3, x3, acc);
        if !CACHED {
            acc_x = _mm256_add_ps(
                acc_x,
                _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3)),
            );
        }
    }
    let sum_x = if CACHED { cached_sum } else { hsum256(acc_x) };
    scale * hsum256(acc) + zero * sum_x
}

/// qt 13 — `Mq4G256` row dot with the original `f32` header and no cached sums,
/// 256 elements / 136 B per group.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mq4g256_row_dot(row: *const u8, k: usize, x: *const f32) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 256 {
        acc += mq4_group_dot::<false>(row.add(g * 136), x.add(g * 256), 0.0);
    }
    acc
}

/// qt 13 — `Mq4G256` row dot reusing the per-group `Σx` for
/// [`super::ActivationSumKind::Mq4Quartets`] prepared by
/// [`super::prepare_activation_sums`]. Same group accumulation and dot FMAs as
/// [`mq4g256_row_dot`].
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn mq4g256_row_dot_cached(
    row: *const u8,
    k: usize,
    x: *const f32,
    sums: *const f32,
) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 256 {
        acc += mq4_group_dot::<true>(row.add(g * 136), x.add(g * 256), *sums.add(g));
    }
    acc
}

/// qt 15 — `Mq6G256` row dot reusing the per-group `Σx` for
/// [`super::ActivationSumKind::Mq6Pairs`] prepared by
/// [`super::prepare_activation_sums`]: the shared `codes8` decode and
/// `f32_pair` header, 256 elements / 200 B per group.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn mq6g256_row_dot_cached(
    row: *const u8,
    k: usize,
    x: *const f32,
    sums: *const f32,
) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 256 {
        let gptr = row.add(g * 200);
        let (scale, zero) = f32_pair(gptr);
        acc += uniform_group_dot::<6, 8, 32, true>(gptr, x.add(g * 256), scale, zero, *sums.add(g));
    }
    acc
}

/// Per-256-element-group `Σx` in [`mq4_group_dot`]'s exact order: eight 32-wide
/// blocks accumulated as `acc_x += ((x0+x1)+(x2+x3))` over four AVX vectors,
/// then one [`hsum256`]. `out[g]` is group `g`'s sum.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn activation_sums_mq4(x: *const f32, out: *mut f32, groups: usize) {
    for g in 0..groups {
        let xg = x.add(g * 256);
        let mut acc_x = _mm256_setzero_ps();
        for blk in 0..8usize {
            let xb = xg.add(blk * 32);
            let x0 = _mm256_loadu_ps(xb);
            let x1 = _mm256_loadu_ps(xb.add(8));
            let x2 = _mm256_loadu_ps(xb.add(16));
            let x3 = _mm256_loadu_ps(xb.add(24));
            acc_x = _mm256_add_ps(
                acc_x,
                _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3)),
            );
        }
        *out.add(g) = hsum256(acc_x);
    }
}

/// Per-256-element-group `Σx` in [`codes_dot_sum`]'s exact order: even 8-lane
/// chunks into `sum0`, odd into `sum1`, then one `hsum256(sum0 + sum1)`.
/// `out[g]` is group `g`'s sum.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn activation_sums_mq6(x: *const f32, out: *mut f32, groups: usize) {
    for g in 0..groups {
        let xg = x.add(g * 256);
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut idx = 0;
        while idx + 1 < 32 {
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(xg.add(idx * 8)));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(xg.add((idx + 1) * 8)));
            idx += 2;
        }
        *out.add(g) = hsum256(_mm256_add_ps(sum0, sum1));
    }
}

// ── one f32 (scale, zero) header per 256-element group ───────────────────────
//
// The qt 13 shape with a different payload width, plus its own unrotated twins:
// the kernel only sees bytes, so `Hfq4G256` (qt 6) and `Mq4G256` (qt 13) are the
// same geometry, and the same holds for 8/15, 11/17 and 9/18. Rotation is
// applied to the activation by the caller, never here.

affine_row_dot!(
    /// qt 6 — `Hfq4G256`: nibbles, natural basis. Byte-identical geometry to
    /// qt 13, minus the FWHT rotation the caller applies.
    hfq4g256_row_dot,
    hfq4g256_group,
    f32_pair,
    4,
    8,
    256,
    136,
    "avx2,fma"
);

affine_row_dot!(
    /// qt 8 — `Hfq6G256`: 6-bit packs, natural basis (qt 15's geometry).
    hfq6g256_row_dot,
    hfq6g256_group,
    f32_pair,
    6,
    8,
    256,
    200,
    "avx2,fma"
);

affine_row_dot!(
    /// qt 15 — `Mq6G256`: 6-bit cross-byte packs, FWHT-rotated. Ships in the
    /// qwen3.8 dense ladder.
    mq6g256_row_dot,
    mq6g256_group,
    f32_pair,
    6,
    8,
    256,
    200,
    "avx2,fma"
);

affine_row_dot!(
    /// qt 31 — `Mq5G256`: 5-bit packs, FWHT-rotated. Arch-loaded by qwen35, so
    /// it reaches the CPU only when its layer is spilled.
    mq5g256_row_dot,
    mq5g256_group,
    f32_pair,
    5,
    8,
    256,
    168,
    "avx2,fma"
);

affine_row_dot!(
    /// qt 11 — `Hfq3G256`: 3-bit cross-byte packs, natural basis (qt 17's
    /// geometry).
    hfq3g256_row_dot,
    hfq3g256_group,
    f32_pair,
    3,
    8,
    256,
    104,
    "avx2,fma"
);

affine_row_dot!(
    /// qt 17 — `Mq3G256`: the uniform 3-bit tier some checkpoints carry, as
    /// opposed to qt 20's Lloyd-Max codebook.
    mq3g256_row_dot,
    mq3g256_group,
    f32_pair,
    3,
    8,
    256,
    104,
    "avx2,fma"
);

affine_row_dot!(
    /// qt 9 — `Hfq2G256`: 2-bit packs, natural basis (qt 18's geometry).
    hfq2g256_row_dot,
    hfq2g256_group,
    f32_pair,
    2,
    8,
    256,
    72,
    "avx2,fma"
);

affine_row_dot!(
    /// qt 18 — `Mq2G256`: FWHT-rotated 2-bit packs.
    mq2g256_row_dot,
    mq2g256_group,
    f32_pair,
    2,
    8,
    256,
    72,
    "avx2,fma"
);

// ── one f32 header per 128-element block ─────────────────────────────────────

affine_row_dot!(
    /// qt 7 — `Hfq4G128`: 128-weight blocks, nibbles. The block formats are what
    /// the blocking tensors (norms, the embedding) use.
    hfq4g128_row_dot,
    hfq4g128_group,
    f32_pair,
    4,
    8,
    128,
    72,
    "avx2,fma"
);

affine_row_dot!(
    /// qt 12 — `Hfq3G128`: 128-weight blocks, 3-bit cross-byte packs.
    hfq3g128_row_dot,
    hfq3g128_group,
    f32_pair,
    3,
    8,
    128,
    56,
    "avx2,fma"
);

affine_row_dot!(
    /// qt 10 — `Hfq2G128`: 128-weight blocks, 2-bit packs.
    hfq2g128_row_dot,
    hfq2g128_group,
    f32_pair,
    2,
    8,
    128,
    40,
    "avx2,fma"
);

// ── an fp16 header, or an fp16 `d`, per group ────────────────────────────────

affine_row_dot!(
    /// qt 45 — `Mq4CG256`: nibbles under a *packed* `fp16` `[scale][zero]`
    /// dword (plus 4 B of padding), i.e. qt 13's payload with qt 49's header
    /// width.
    mq4cg256_row_dot,
    mq4cg256_group,
    f16_pair,
    4,
    8,
    256,
    136,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 40 — `Tq2G128`: 2-bit ternary codes over a 128-element block, decoded
    /// as `(code - 1)·d` — the affine form with `scale = d` and `zero = -d`, so
    /// no separate kernel is needed.
    tq2g128_row_dot,
    tq2g128_group,
    128,
    34,
    "avx2,fma,f16c"
);

/// qt 40's group: one fp16 `d` then 16 chunks of 2-bit codes.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tq2g128_group(gptr: *const u8, xg: *const f32) -> f32 {
    let (d, _) = f16_pair(gptr);
    uniform_group_dot::<2, 2, 16, false>(gptr, xg, d, -d, 0.0)
}

row_dot!(
    /// qt 41 — `Bq1G128`: sign bits over a 128-element block, decoded as
    /// `bit ? +d : -d`. That is `2d·bit - d` — the affine form with
    /// `scale = 2d` and `zero = -d` over one-bit codes.
    bq1g128_row_dot,
    bq1g128_group,
    128,
    18,
    "avx2,fma,f16c"
);

/// qt 41's group: one fp16 `d` then 16 chunks of 8 sign bits.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn bq1g128_group(gptr: *const u8, xg: *const f32) -> f32 {
    let (d, _) = f16_pair(gptr);
    uniform_group_dot::<1, 2, 16, false>(gptr, xg, 2.0 * d, -d, 0.0)
}

// ── a per-group fp16 codebook instead of an affine header ────────────────────
//
// The Lloyd-Max tier: the group's `CB` codebook entries sit where the affine
// header would be, and the payload indexes them, so there is no scale/zero to
// apply.

row_dot!(
    /// qt 19 — `Mq2G256Lloyd`: 2-bit indices into a 4-entry `fp16` codebook.
    mq2g256lloyd_row_dot,
    codebook_group_dot::<2, 4, 8>,
    256,
    72,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 51 — `Mq2G256LloydU`: byte-identical to qt 19 and deliberately *not*
    /// rotated (it carries native-ternary checkpoints losslessly), which is
    /// again a caller-side difference, not a kernel one.
    mq2g256lloydu_row_dot,
    codebook_group_dot::<2, 4, 8>,
    256,
    72,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 20 — `Mq3G256Lloyd`: 3-bit indices into an 8-entry `fp16` codebook.
    /// This is what the registry's `qwen3.5:2b-mq3` and the other `-mq3` tags
    /// carry (measured, not assumed).
    mq3g256lloyd_row_dot,
    codebook_group_dot::<3, 8, 16>,
    256,
    112,
    "avx2,fma,f16c"
);

row_dot!(
    /// qt 30 — `Mq4G256Lloyd`: nibble indices into a 16-entry `fp16` codebook.
    mq4g256lloyd_row_dot,
    codebook_group_dot::<4, 16, 32>,
    256,
    160,
    "avx2,fma,f16c"
);

// ── element formats: no codes to unpack, only weights to widen ───────────────
//
// These carry one weight per element rather than a group header, so the group
// is just a run of elements and the kernel is a plain widening dot. `F32` is the
// only one that needs no widening at all, and `Bf16` the only one that needs no
// F16C (a 16-bit shift is the whole conversion).

/// Eight `fp16` elements at `p`, widened.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn widen8_f16(p: *const u8) -> __m256 {
    _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
}

/// Eight `bf16` elements at `p`: the zero-extended halfwords shifted into a
/// float's high half — `bf16` is an f32 truncated to 16 bits, so scaling is the
/// conversion.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn widen8_bf16(p: *const u8) -> __m256 {
    let v = _mm256_cvtepu16_epi32(_mm_loadu_si128(p as *const __m128i));
    _mm256_castsi256_ps(_mm256_slli_epi32(v, 16))
}

/// Eight `f32` elements at `p`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn widen8_f32(p: *const u8) -> __m256 {
    _mm256_loadu_ps(p as *const f32)
}

/// A row of an element format: `256`-element groups, `$elem` bytes per element
/// and no header, with `$lane` widening eight elements at a time.
macro_rules! dense_row_dot {
    ($(#[$meta:meta])* $name:ident, $elem:literal, $feat:literal, $lane:ident) => {
        $(#[$meta])*
        ///
        /// # Safety
        ///
        /// Same contract as the `row_dot!` drivers: the features above,
        /// `k % 256 == 0`, `(k / 256) * 256 * $elem` readable bytes at `row`
        /// and `k` readable `f32` at `x`.
        #[target_feature(enable = $feat)]
        pub unsafe fn $name(row: *const u8, k: usize, x: *const f32) -> f32 {
            let mut acc = 0.0f32;
            for g in 0..k / 256 {
                let wp = row.add(g * 256 * $elem);
                let xg = x.add(g * 256);
                let mut dot = _mm256_setzero_ps();
                for c in 0..32 {
                    let xv = _mm256_loadu_ps(xg.add(c * 8));
                    dot = _mm256_fmadd_ps($lane(wp.add(c * 8 * $elem)), xv, dot);
                }
                acc += hsum256(dot);
            }
            acc
        }
    };
}

dense_row_dot!(
    /// qt 1 — `F16`: plain `fp16` weights, 2 B per element.
    f16_row_dot,
    2,
    "avx2,fma,f16c",
    widen8_f16
);

dense_row_dot!(
    /// qt 16 — `Bf16`: plain `bf16`, the one format whose kernel needs no F16C.
    bf16_row_dot,
    2,
    "avx2,fma",
    widen8_bf16
);

dense_row_dot!(
    /// qt 2 — `F32`: the weights are already the kernel's own type, so there is
    /// no decode at all — only the accumulation order differs from scalar.
    f32_row_dot,
    4,
    "avx2,fma",
    widen8_f32
);

row_dot!(
    /// qt 3 — `Q8F16` (`DType::Q8_0`): a 32-element block of one `fp16` scale
    /// then 32 `i8` codes. The only signed-code format, and the only block
    /// smaller than 128 elements.
    q8f16_row_dot,
    q8f16_group,
    32,
    34,
    "avx2,fma,f16c"
);

/// qt 3's group: one `fp16` scale, then four chunks of eight signed bytes.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q8f16_group(gptr: *const u8, xg: *const f32) -> f32 {
    let (scale, _) = f16_pair(gptr);
    let mut dot = _mm256_setzero_ps();
    for c in 0..4 {
        let bytes = _mm_loadl_epi64(gptr.add(2 + c * 8) as *const __m128i);
        let v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
        dot = _mm256_fmadd_ps(v, _mm256_loadu_ps(xg.add(c * 8)), dot);
    }
    scale * hsum256(dot)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prefix bytes must not leak into codes, including a chunk ending at the
    /// packed buffer's last byte. Every possible code occupies every lane.
    #[target_feature(enable = "avx2")]
    unsafe fn check_prefixed_codes<const BITS: usize>() {
        let mask = (1u32 << BITS) - 1;
        let prefix_bytes = if BITS == 6 { 2 } else { 1 };
        let mut packed = vec![0u8; BITS + prefix_bytes];
        for phase in 0..=mask {
            let expected: [u32; 8] = core::array::from_fn(|lane| (phase + lane as u32 * 13) & mask);
            let word = expected
                .iter()
                .enumerate()
                .fold(0u64, |word, (lane, &code)| {
                    word | ((code as u64) << (lane * BITS))
                });
            packed[prefix_bytes..].copy_from_slice(&word.to_le_bytes()[..BITS]);
            for prefix in 0..=u8::MAX {
                packed[..prefix_bytes].fill(prefix);
                let mut got = [0u32; 8];
                _mm256_storeu_si256(
                    got.as_mut_ptr() as *mut __m256i,
                    codes8::<BITS>(packed.as_ptr().add(prefix_bytes)),
                );
                assert_eq!(got, expected, "bits={BITS} phase={phase} prefix={prefix}");
            }
        }
    }

    #[test]
    fn three_and_six_bit_codes_ignore_prefix_bytes() {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                check_prefixed_codes::<3>();
                check_prefixed_codes::<6>();
            }
        }
    }
}

/// AoS E8 rows: one fp16 row scale, then 17-byte blocks of 32 weights.
///
/// # Safety
/// Requires AVX2/FMA/F16C, `16 + (k / 32) * 17` readable bytes at `row`,
/// `k` readable f32 values at `x`, and `k % 256 == 0`.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn mfp4g32_e8_row_dot(row: *const u8, k: usize, x: *const f32) -> f32 {
    let row_scale = f16_pair(row).0;
    let shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    let mask = _mm256_set1_epi32(15);
    let center = _mm256_set1_epi32(7);
    let mut acc = 0.0f32;
    for group in 0..k / 256 {
        let mut dot = _mm256_setzero_ps();
        for block in 0..8 {
            let at = row.add(16 + (group * 8 + block) * 17);
            let scale = row_scale * crate::quant::mfp4_e8_scale(*at) * 0.88;
            let scale = _mm256_set1_ps(scale);
            for word in 0..4 {
                // Exactly four codeword bytes, including at the final row end.
                let index = (at.add(1 + word * 4) as *const u32).read_unaligned();
                let parity = (index
                    ^ (index >> 4)
                    ^ (index >> 8)
                    ^ (index >> 12)
                    ^ (index >> 16)
                    ^ (index >> 20)
                    ^ (index >> 24))
                    & 1;
                let last = ((index >> 27) & 14) | parity;
                let nibbles = _mm256_and_si256(
                    _mm256_srlv_epi32(_mm256_set1_epi32(index as i32), shifts),
                    mask,
                );
                let nibbles = _mm256_blend_epi32(nibbles, _mm256_set1_epi32(last as i32), 0x80);
                let coordinates = _mm256_add_ps(
                    _mm256_cvtepi32_ps(_mm256_sub_epi32(nibbles, center)),
                    _mm256_set1_ps((index >> 31) as f32 * 0.5),
                );
                // Round the decoded weight before its FMA, as in the scalar
                // decoder. Folding scale into x changes the format's bits.
                let weights = _mm256_mul_ps(scale, coordinates);
                let xp = x.add(group * 256 + block * 32 + word * 8);
                dot = _mm256_fmadd_ps(weights, _mm256_loadu_ps(xp), dot);
            }
        }
        acc += hsum256(dot);
    }
    acc
}

// ---- planar slab formats (qt 131/132/135/136) ----------------------------------
//
// Planar tensors keep a row's codes and its f16 scales in separate planes, so
// these kernels take two pointers: the row's codes (exactly `row_bytes` of them)
// and the row's `k / 128` little-endian f16 scales (`2 * (k / 128)` bytes). Both
// stay inside those ranges: every load below is exactly the bytes it consumes,
// and the one W3 load that would otherwise run one byte past a row's codes is
// read as three scalar bytes.

use crate::quant::{E8P_ABS2, E8P_ROW_PARITY};

/// 32-byte alignment wrapper: each 8 x f32 table row is then one aligned half
/// cache line (`vmovaps` / memory-operand legal).
#[repr(C, align(32))]
struct Aligned32<T>(T);

/// E8P magnitude table, 8 KiB: row `idx`, lane `k` is `2·E8P_ABS2[idx][k]`, with
/// lane 7 pre-negated on odd-parity rows (the stored half of lane 7's sign
/// rule, as in `e8p_i4_reference::build_lut`). Built at compile time.
static E8P_MAGNITUDE: Aligned32<[[f32; 8]; 256]> = Aligned32({
    let mut t = [[0.0f32; 8]; 256];
    let mut i = 0;
    while i < 256 {
        let mut k = 0;
        while k < 8 {
            let m = 2.0 * E8P_ABS2[i][k] as f32;
            t[i][k] = if k == 7 && E8P_ROW_PARITY[i] == 1 {
                -m
            } else {
                m
            };
            k += 1;
        }
        i += 1;
    }
    t
});

/// E8P sign-flip table, 8 KiB: row `hi`, lane `k` is `-0.0` (a pure sign-bit
/// XOR mask) iff the lane is flipped — bit `k` of `sign7 = hi & 0x7f` for
/// `k < 7`, and for lane 7 iff `popcount(sign7)` is odd (the other half of the
/// parity rule, as in `e8p_i4_reference::build_mask`). Indexed by the *whole*
/// high byte of a wire word, so the shift bit (`hi >> 7`) selects a duplicate of
/// the `sign7` row and is ignored: the lookup needs no mask. XOR with
/// [`E8P_MAGNITUDE`] yields the signed lane `±2|g|`; the shift offset is folded
/// out of the lane loop entirely (see [`off8`]).
static E8P_SIGN: Aligned32<[[f32; 8]; 256]> = Aligned32({
    let mut t = [[0.0f32; 8]; 256];
    let mut s = 0usize;
    while s < 256 {
        let s7 = s & 0x7f;
        let mut k = 0;
        while k < 8 {
            let flip = if k < 7 {
                (s7 >> k) & 1 == 1
            } else {
                (s7 as u32).count_ones() & 1 == 1
            };
            t[s][k] = if flip { -0.0 } else { 0.0 };
            k += 1;
        }
        s += 1;
    }
    t
});

/// One eight-lane `f32` vector of an E8P wire word's signed magnitudes
/// `mag[idx] ^ sign[hi]` — the word's `v` values *without* the shift offset
/// (`v = this + off`, `off = -1` iff the shift bit, else `+1`).
///
/// The word is one little-endian `u16` load (x86 is little-endian): `idx` is its
/// low byte and the whole high byte indexes [`E8P_SIGN`].
///
/// # Safety
///
/// Requires AVX2 and the two word bytes at `p`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn wm16(p: *const u8) -> __m256 {
    let w = (p as *const u16).read_unaligned() as usize;
    _mm256_xor_ps(
        _mm256_load_ps(E8P_MAGNITUDE.0[w & 0xff].as_ptr()),
        _mm256_load_ps(E8P_SIGN.0[w >> 8].as_ptr()),
    )
}

/// `off_w · S_w` for the eight words at `cp` against their eight activation word
/// sums at `s`: `off = -1` iff the word's bit 15, applied as a sign-bit XOR of
/// `S_w` (exact: `-1 · S = -S`, `+1 · S = S`).
///
/// # Safety
///
/// Requires AVX2, 16 readable bytes at `cp` and 8 readable `f32` at `s`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn off8(cp: *const u8, s: *const f32) -> __m256 {
    let w = _mm256_cvtepu16_epi32(_mm_loadu_si128(cp as *const __m128i));
    let sign_bit = _mm256_and_si256(_mm256_slli_epi32(w, 16), _mm256_set1_epi32(i32::MIN));
    _mm256_xor_ps(_mm256_loadu_ps(s), _mm256_castsi256_ps(sign_bit))
}

/// The sums `S_w` of the eight 8-lane chunks of `x[0..64]` (lane `i` = the sum of
/// `x[8i..8i+8]`), in one fixed `hadd`/permute order. Both the prepared sums
/// ([`activation_sums_e8p`]) and the inline path of [`e8p_row_dot`] use exactly
/// this function, which is what makes the two paths bit-identical.
///
/// # Safety
///
/// Requires AVX2 and 64 readable `f32` at `x`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn chunk_sums8(x: *const f32) -> __m256 {
    let c = |i: usize| _mm256_loadu_ps(x.add(8 * i));
    let h01 = _mm256_hadd_ps(c(0), c(1));
    let h23 = _mm256_hadd_ps(c(2), c(3));
    let h45 = _mm256_hadd_ps(c(4), c(5));
    let h67 = _mm256_hadd_ps(c(6), c(7));
    // [c0l c1l c2l c3l | c0h c1h c2h c3h], then likewise for chunks 4..7.
    let q0 = _mm256_hadd_ps(h01, h23);
    let q1 = _mm256_hadd_ps(h45, h67);
    let lo = _mm256_permute2f128_ps(q0, q1, 0x20);
    let hi = _mm256_permute2f128_ps(q0, q1, 0x31);
    _mm256_add_ps(lo, hi)
}

/// Per-word activation sums for [`e8p_row_dot_cached`] /
/// [`e8p_row_pair_dot_cached`]: `out[w] = Σ x[8w..8w+8]` for `w < len`, in
/// [`chunk_sums8`] order.
///
/// # Safety
///
/// Requires AVX2; `len % 8 == 0` (`len = k / 8`, `k % 64 == 0`); `8 * len`
/// readable `f32` at `x` and `len` writable `f32` at `out`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn activation_sums_e8p(x: *const f32, out: *mut f32, len: usize) {
    for c in 0..len / 8 {
        _mm256_storeu_ps(out.add(8 * c), chunk_sums8(x.add(64 * c)));
    }
}

/// One 128-group of an E8P row, folded: `Σ_lanes v·x` as
/// `((a0+a1)+(a2+a3)) + (off8(lo)+off8(hi))` with `a0..a3` the four FMA
/// accumulators over the group's 16 signed-magnitude words ([`wm16`]) and `sp`
/// the group's 16 word sums.
///
/// # Safety
///
/// Requires AVX2 and FMA; 32 readable bytes at `cp`, 128 `f32` at `xp`, 16 `f32`
/// at `sp`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn e8p_group(cp: *const u8, xp: *const f32, sp: *const f32) -> __m256 {
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    for w in 0..4 {
        let c = cp.add(w * 8);
        let xw = xp.add(w * 32);
        a0 = _mm256_fmadd_ps(wm16(c), _mm256_loadu_ps(xw), a0);
        a1 = _mm256_fmadd_ps(wm16(c.add(2)), _mm256_loadu_ps(xw.add(8)), a1);
        a2 = _mm256_fmadd_ps(wm16(c.add(4)), _mm256_loadu_ps(xw.add(16)), a2);
        a3 = _mm256_fmadd_ps(wm16(c.add(6)), _mm256_loadu_ps(xw.add(24)), a3);
    }
    let off = _mm256_add_ps(off8(cp, sp), off8(cp.add(16), sp.add(8)));
    _mm256_add_ps(
        _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)),
        off,
    )
}

/// The f16 at `p` widened to f32 and broadcast to all eight lanes.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn f16_broadcast(p: *const u8) -> __m256 {
    let bits = (p as *const u16).read_unaligned();
    _mm256_cvtph_ps(_mm_set1_epi16(bits as i16))
}

/// E8P-I4 planar row (qt 135 / 136): `Σ_g (0.25·f32(d_g)) · Σ_w v_w · x_w`,
/// with each word's shift offset folded out of the lane loop.
///
/// A word's lanes are `v = s + off` with `s = mag[idx] ^ sign[hi]` (the signed
/// magnitudes, [`wm16`]) and `off = -1` iff the shift bit, else `+1`, so
/// `Σ_lanes v·x = Σ_lanes s·x + off·S_w` with `S_w` the word's eight-lane
/// activation sum ([`chunk_sums8`]). Per 128-group ([`e8p_group`]): 16 words
/// FMA'd into four independent accumulators (breaking the FMA latency chain),
/// plus the two [`off8`] vectors, folded into the row accumulator with the
/// group's `0.25·f32(d)` — one fold per 128 weights, one horizontal sum per row.
/// `0.25·f32(d)` is exact, so this is the scalar decode's `(f32(d)·0.25)·v`
/// weight up to summation order.
///
/// This is the unprepared entry: each group's 16 word sums are computed inline
/// into a small aligned stack buffer with [`chunk_sums8`]. It is bit-identical to
/// [`e8p_row_dot_cached`] over [`activation_sums_e8p`].
///
/// # Safety
///
/// Requires AVX2, FMA and F16C; `k % 128 == 0`; `k / 4` readable bytes at
/// `codes`, `2 * (k / 128)` at `scales`, and `k` readable `f32` at `x`.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn e8p_row_dot(
    codes: *const u8,
    scales: *const u8,
    k: usize,
    x: *const f32,
) -> f32 {
    let quarter = _mm256_set1_ps(0.25);
    let mut row = _mm256_setzero_ps();
    let mut sums = Aligned32([0.0f32; 16]);
    for g in 0..k / 128 {
        let xp = x.add(g * 128);
        _mm256_store_ps(sums.0.as_mut_ptr(), chunk_sums8(xp));
        _mm256_store_ps(sums.0.as_mut_ptr().add(8), chunk_sums8(xp.add(64)));
        let group = e8p_group(codes.add(g * 32), xp, sums.0.as_ptr());
        let scale = _mm256_mul_ps(f16_broadcast(scales.add(2 * g)), quarter);
        row = _mm256_fmadd_ps(group, scale, row);
    }
    hsum256(row)
}

/// [`e8p_row_dot`] over word sums prepared by [`activation_sums_e8p`]
/// (`k / 8` of them): the same group helper, scale fold and reduction, with the
/// inline [`chunk_sums8`] replaced by the cached value it would have produced —
/// bit-identical to the unprepared row.
///
/// # Safety
///
/// As [`e8p_row_dot`], plus `k / 8` readable `f32` at `sums`.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn e8p_row_dot_cached(
    codes: *const u8,
    scales: *const u8,
    k: usize,
    x: *const f32,
    sums: *const f32,
) -> f32 {
    let quarter = _mm256_set1_ps(0.25);
    let mut row = _mm256_setzero_ps();
    for g in 0..k / 128 {
        let group = e8p_group(codes.add(g * 32), x.add(g * 128), sums.add(g * 16));
        let scale = _mm256_mul_ps(f16_broadcast(scales.add(2 * g)), quarter);
        row = _mm256_fmadd_ps(group, scale, row);
    }
    hsum256(row)
}

/// Two E8P rows against the same activation and prepared word sums at once: each
/// activation load is shared by both rows' FMAs. Each row's operation order is
/// exactly [`e8p_row_dot_cached`]'s, so `[r0, r1]` is bit-identical to two
/// single-row calls.
///
/// # Safety
///
/// As [`e8p_row_dot_cached`], for both rows: `k / 4` readable bytes at `c0` and
/// at `c1`, `2 * (k / 128)` at `s0` and at `s1` (the two rows may be anywhere,
/// they need not be adjacent), `k` readable `f32` at `x`, `k / 8` at `sums`.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn e8p_row_pair_dot_cached(
    c0: *const u8,
    s0: *const u8,
    c1: *const u8,
    s1: *const u8,
    k: usize,
    x: *const f32,
    sums: *const f32,
) -> [f32; 2] {
    let codes = [c0, c1];
    let scales = [s0, s1];
    let quarter = _mm256_set1_ps(0.25);
    let mut row = [_mm256_setzero_ps(); 2];
    for g in 0..k / 128 {
        let xp = x.add(g * 128);
        let sp = sums.add(g * 16);
        let mut a = [[_mm256_setzero_ps(); 4]; 2];
        for w in 0..4 {
            let xw = xp.add(w * 32);
            let x0 = _mm256_loadu_ps(xw);
            let x1 = _mm256_loadu_ps(xw.add(8));
            let x2 = _mm256_loadu_ps(xw.add(16));
            let x3 = _mm256_loadu_ps(xw.add(24));
            for r in 0..2 {
                let c = codes[r].add(g * 32 + w * 8);
                a[r][0] = _mm256_fmadd_ps(wm16(c), x0, a[r][0]);
                a[r][1] = _mm256_fmadd_ps(wm16(c.add(2)), x1, a[r][1]);
                a[r][2] = _mm256_fmadd_ps(wm16(c.add(4)), x2, a[r][2]);
                a[r][3] = _mm256_fmadd_ps(wm16(c.add(6)), x3, a[r][3]);
            }
        }
        for r in 0..2 {
            let cp = codes[r].add(g * 32);
            let off = _mm256_add_ps(off8(cp, sp), off8(cp.add(16), sp.add(8)));
            let group = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(a[r][0], a[r][1]),
                    _mm256_add_ps(a[r][2], a[r][3]),
                ),
                off,
            );
            let scale = _mm256_mul_ps(f16_broadcast(scales[r].add(2 * g)), quarter);
            row[r] = _mm256_fmadd_ps(group, scale, row[r]);
        }
    }
    [hsum256(row[0]), hsum256(row[1])]
}

/// Eight W3 codes centered by `3.5` as `f32` lanes: `c - 3.5 = (2c - 7) / 2`
/// (exact), in element order.
///
/// One broadcast of a 32-bit read at the chunk start plus one variable shift
/// per code. The 4-byte read straddles one byte into the next chunk, which the
/// shifts (at most 21) and the 3-bit mask discard. `SAFE` reads exactly the
/// chunk's three bytes instead, for the last chunk of a row, whose fourth byte
/// would lie outside the row's codes slice.
///
/// # Safety
///
/// Requires AVX2 and 3 readable bytes at `p` (4 when `!SAFE`).
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn w3_centered<const SAFE: bool>(p: *const u8) -> __m256 {
    let word = if SAFE {
        *p as i32 | (*p.add(1) as i32) << 8 | (*p.add(2) as i32) << 16
    } else {
        (p as *const i32).read_unaligned()
    };
    let shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    let codes = _mm256_and_si256(
        _mm256_srlv_epi32(_mm256_set1_epi32(word), shifts),
        _mm256_set1_epi32(7),
    );
    _mm256_sub_ps(_mm256_cvtepi32_ps(codes), _mm256_set1_ps(3.5))
}

/// `Σ (c - 3.5)·x` over one W3 128-group (48 code bytes at `cp`, 16 chunks), as
/// an eight-lane vector over two accumulator chains. `LAST` marks the row's last
/// group, whose last chunk is read without the extra fourth byte.
///
/// # Safety
///
/// Requires AVX2 and FMA; 48 readable bytes at `cp` (when `!LAST` the last chunk
/// also reads `cp + 48`, the next group's first byte, which exists because this
/// is not the row's last group), and 128 readable `f32` at `xp`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn w3_group_dot<const LAST: bool>(cp: *const u8, xp: *const f32) -> __m256 {
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    for c in 0..7 {
        a0 = _mm256_fmadd_ps(
            w3_centered::<false>(cp.add(6 * c)),
            _mm256_loadu_ps(xp.add(16 * c)),
            a0,
        );
        a1 = _mm256_fmadd_ps(
            w3_centered::<false>(cp.add(6 * c + 3)),
            _mm256_loadu_ps(xp.add(16 * c + 8)),
            a1,
        );
    }
    a0 = _mm256_fmadd_ps(
        w3_centered::<false>(cp.add(42)),
        _mm256_loadu_ps(xp.add(112)),
        a0,
    );
    a1 = _mm256_fmadd_ps(
        w3_centered::<LAST>(cp.add(45)),
        _mm256_loadu_ps(xp.add(120)),
        a1,
    );
    _mm256_add_ps(a0, a1)
}

/// MQ3-I4 (W3) planar row (qt 131 / 132): `Σ_g 2·f32(d_g) · Σ (c - 3.5)·x`, i.e.
/// the scalar decode's `f32(d)·(2c - 7)` weight up to summation order (both
/// `c - 3.5` and `2·f32(d)` are exact). One scale fold per 128 weights, one
/// horizontal sum per row.
///
/// # Safety
///
/// Requires AVX2, FMA and F16C; `k % 128 == 0`; `3k/8` readable bytes at
/// `codes`, `2 * (k / 128)` at `scales`, and `k` readable `f32` at `x`. Reads
/// stay within those ranges (see [`w3_centered`]).
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn w3_row_dot(
    codes: *const u8,
    scales: *const u8,
    k: usize,
    x: *const f32,
) -> f32 {
    let groups = k / 128;
    if groups == 0 {
        return 0.0;
    }
    let mut row = _mm256_setzero_ps();
    for g in 0..groups - 1 {
        let d = f16_broadcast(scales.add(2 * g));
        row = _mm256_fmadd_ps(
            w3_group_dot::<false>(codes.add(g * 48), x.add(g * 128)),
            _mm256_add_ps(d, d),
            row,
        );
    }
    let g = groups - 1;
    let d = f16_broadcast(scales.add(2 * g));
    row = _mm256_fmadd_ps(
        w3_group_dot::<true>(codes.add(g * 48), x.add(g * 128)),
        _mm256_add_ps(d, d),
        row,
    );
    hsum256(row)
}
