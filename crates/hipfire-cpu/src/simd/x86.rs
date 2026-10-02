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
//! `Mq4G256` (qt 13) keeps its own hand-unrolled nibble path ([`mq4_group_dot`]):
//! it predates this core, decodes 32 codes per 16-byte load instead of 8 per
//! 4-byte load, and its numbers are the ones the recorded qt 13 measurements
//! were taken with, so it is left alone rather than re-derived from the table.

use core::arch::x86_64::*;

/// Horizontal sum of eight `f32` lanes.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(v: __m256) -> f32 {
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
        // Three bytes from one 32-bit load of the byte *before* the chunk plus
        // the chunk, shifted down. In bounds for every chunk: a group's payload
        // always starts at least one header byte in.
        3 => ((p.sub(1) as *const u32).read_unaligned() >> 8) as u64,
        4 => (p as *const u32).read_unaligned() as u64,
        5 => (p as *const u32).read_unaligned() as u64 | ((*p.add(4) as u64) << 32),
        6 => {
            (p as *const u32).read_unaligned() as u64
                | (((p.add(4) as *const u16).read_unaligned() as u64) << 32)
        }
        _ => unreachable!("no format packs codes wider than six bits"),
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
/// is `8·BITS` bits, so widths up to four fit a 32-bit lane and a single shift;
/// the 5- and 6-bit packs are 40 and 48 bits and need the field in 64-bit lanes,
/// where each lane carries one code and the four low dwords of each half are
/// compacted back into eight lanes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn codes8<const BITS: usize>(p: *const u8) -> __m256i {
    if BITS <= 4 {
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
unsafe fn codes_dot_sum<const BITS: usize, const PAYLOAD: usize>(
    gptr: *const u8,
    xg: *const f32,
    first: usize,
    last: usize,
) -> (f32, f32) {
    let mut dot = _mm256_setzero_ps();
    let mut sum = _mm256_setzero_ps();
    for idx in first..last {
        let xv = _mm256_loadu_ps(xg.add(idx * 8));
        let c = _mm256_cvtepi32_ps(codes8::<BITS>(gptr.add(PAYLOAD + idx * BITS)));
        dot = _mm256_fmadd_ps(c, xv, dot);
        sum = _mm256_add_ps(sum, xv);
    }
    (hsum256(dot), hsum256(sum))
}

/// One group with a single `(scale, zero)` over all `CHUNKS` of its chunks:
/// `scale·Σ(c·x) + zero·Σx`.
///
/// The header is read by the caller, which is what lets one group body serve an
/// `f32` header, an `fp16` pair and TQ2/BQ1's *derived* `(d, -d)` / `(2d, -d)`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn uniform_group_dot<const BITS: usize, const PAYLOAD: usize, const CHUNKS: usize>(
    gptr: *const u8,
    xg: *const f32,
    scale: f32,
    zero: f32,
) -> f32 {
    let (dot, sum) = codes_dot_sum::<BITS, PAYLOAD>(gptr, xg, 0, CHUNKS);
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
    let (d0, s0) = codes_dot_sum::<BITS, 8>(gptr, xg, 0, 16);
    let (d1, s1) = codes_dot_sum::<BITS, 8>(gptr, xg, 16, 32);
    (h[0] * d0 + h[1] * s0) + (h[2] * d1 + h[3] * s1)
}

/// The `CB`-entry `fp16` codebook at the group's start, indexed per lane by
/// `sel`.
///
/// A 4- or 8-entry book is one `vpermd`; a 16-entry book is two plus a blend on
/// the index's top bit. `vpermd` only looks at an index's low three bits, which
/// is what makes the two-table form work — both lookups are valid and the blend
/// picks the half.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn codebook_lookup<const CB: usize>(gptr: *const u8, sel: __m256i) -> __m256 {
    if CB == 16 {
        let lo = _mm256_cvtph_ps(_mm_loadu_si128(gptr as *const __m128i));
        let hi = _mm256_cvtph_ps(_mm_loadu_si128(gptr.add(16) as *const __m128i));
        let top = _mm256_slli_epi32(_mm256_and_si256(sel, _mm256_set1_epi32(8)), 28);
        _mm256_blendv_ps(
            _mm256_permutevar8x32_ps(lo, sel),
            _mm256_permutevar8x32_ps(hi, sel),
            _mm256_castsi256_ps(top),
        )
    } else if CB == 8 {
        let table = _mm256_cvtph_ps(_mm_loadu_si128(gptr as *const __m128i));
        _mm256_permutevar8x32_ps(table, sel)
    } else {
        let table = _mm256_insertf128_ps(
            _mm256_setzero_ps(),
            _mm_cvtph_ps(_mm_loadl_epi64(gptr as *const __m128i)),
            0,
        );
        _mm256_permutevar8x32_ps(table, sel)
    }
}

/// One group of a Lloyd-codebook format: `CB` `fp16` entries at the group's
/// start, then `PAYLOAD` bytes of `BITS`-wide indices.
///
/// There is no affine term — the codebook *is* the decode — so this needs one
/// accumulator, not two.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn codebook_group_dot<const BITS: usize, const CB: usize, const PAYLOAD: usize>(
    gptr: *const u8,
    xg: *const f32,
) -> f32 {
    let mut dot = _mm256_setzero_ps();
    for idx in 0..32 {
        let sel = codes8::<BITS>(gptr.add(PAYLOAD + idx * BITS));
        let vals = codebook_lookup::<CB>(gptr, sel);
        dot = _mm256_fmadd_ps(vals, _mm256_loadu_ps(xg.add(idx * 8)), dot);
    }
    hsum256(dot)
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
unsafe fn f16_pair(p: *const u8) -> (f32, f32) {
    let v = f16_quad(p);
    (v[0], v[1])
}

/// The V2 family's per-128 header, `[s0 z0 s1 z1]`, widened with one
/// `vcvtph2ps`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn f16_quad(p: *const u8) -> [f32; 4] {
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
            uniform_group_dot::<$bits, $payload, { $ge / 8 }>(gptr, xg, scale, zero)
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
unsafe fn mq4_group_dot(gptr: *const u8, xg: *const f32) -> f32 {
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
        acc_x = _mm256_add_ps(
            acc_x,
            _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3)),
        );
    }
    scale * hsum256(acc) + zero * hsum256(acc_x)
}

row_dot!(
    /// qt 13 — `Mq4G256`: nibbles under the original `f32` header. The
    /// hand-unrolled nibble group above, at 32 codes per 16-byte load — see the
    /// module docs for why it is not the `codes8` path.
    mq4g256_row_dot,
    mq4_group_dot,
    256,
    136,
    "avx2,fma"
);

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
    uniform_group_dot::<2, 2, 16>(gptr, xg, d, -d)
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
    uniform_group_dot::<1, 2, 16>(gptr, xg, 2.0 * d, -d)
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
