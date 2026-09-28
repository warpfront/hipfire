// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! IQ-family dequantization kernels, ported from llama.cpp
//! `ggml/src/ggml-quants.c` (Apache-2.0 / MIT, see NOTICE).
//!
//! Why this module exists: the upstream GGUF ingest path only understood the
//! legacy Q-quants plus BF16/F16/F32, so mixed-precision IQ checkpoints (the
//! GSQ-RCO / ISTA-DASLab releases, and anything else llama.cpp produces at
//! sub-3-bit) could not be converted without a lossy F16 round-trip through
//! `llama-quantize`. Every kernel here is a line-by-line translation of the
//! corresponding `dequantize_row_iq*` so the byte layout stays reproduced
//! exactly; the grid tables live in [`crate::iq_tables`] and are likewise
//! machine-extracted rather than retyped.
//!
//! Porting discipline (keep this list when re-syncing with upstream):
//!   1. Table contents come from `tools/extract_iq_tables.py` — never hand-edit.
//!   2. Field offsets below are `sizeof`-derived from the `block_iq*` structs in
//!      `ggml-common.h`; they are stated per kernel so a future upstream field
//!      addition is a visible diff instead of a silent misread.
//!   3. Grid entries are packed integers exactly as upstream declares them. The
//!      C code reinterprets `grid + idx` as a byte pointer, which is little-endian
//!      on every supported host, so the translation uses `to_le_bytes()`.
//!   4. `x[i].d` is a `ggml_half`, i.e. little-endian f16 in the first two bytes.
//!
//! Not ported: `iq1s_grid_gpu` (a Metal/CUDA-precomputed variant of the same
//! grid; the CPU path never reads it) and the `_4_4` / `_8_8` SIMD reorderings.

use hipfire_quantize::float16::f16_to_f32;

use crate::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL,
};

/// Elements per K-quant super-block (llama.cpp `QK_K`).
pub(crate) const QK_K: usize = 256;
/// Elements per IQ4_NL block (llama.cpp `QK4_NL`).
pub(crate) const QK4_NL: usize = 32;
/// Grid-shift delta for IQ1_S / IQ1_M (llama.cpp `IQ1S_DELTA`).
const IQ1S_DELTA: f32 = 0.125;

// ── byte-level helpers ──────────────────────────────────────────────────────
//
// Each mirrors one C idiom. Keeping them separate keeps the kernel bodies
// readable as direct transcriptions of `ggml-quants.c`.

/// `GGML_FP16_TO_FP32(x[i].d)` — block scale in the first two bytes.
#[inline(always)]
fn scale_of(blk: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]))
}

#[inline(always)]
fn rd_u16(blk: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([blk[off], blk[off + 1]])
}

#[inline(always)]
fn rd_u32(blk: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([blk[off], blk[off + 1], blk[off + 2], blk[off + 3]])
}

/// `signs & kmask_iq2xs[j] ? -1.f : 1.f`
#[inline(always)]
fn sign_of(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Reinterpret a packed grid entry as `N` bytes, mirroring `(const uint8_t *)`.
#[inline(always)]
fn grid_bytes_u64(v: u64) -> [u8; 8] {
    v.to_le_bytes()
}

#[inline(always)]
fn grid_bytes_u32(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

/// Reinterpret a packed grid entry as `N` *signed* bytes, mirroring
/// `(const int8_t *)`. Used by the IQ1 grids, whose values are in -1..=1.
#[inline(always)]
fn grid_i8_u64(v: u64) -> [i8; 8] {
    let b = v.to_le_bytes();
    [
        b[0] as i8,
        b[1] as i8,
        b[2] as i8,
        b[3] as i8,
        b[4] as i8,
        b[5] as i8,
        b[6] as i8,
        b[7] as i8,
    ]
}

// ── IQ2_XXS — 2.0625 bpw, 66 B/block ────────────────────────────────────────
//
// block_iq2_xxs { ggml_half d; uint16_t qs[QK_K/8]; }
//   d  @ 0..2   qs @ 2..66
//
// `qs` is `uint16_t*` upstream, so `qs + 4*ib32` advances 8 bytes: two u32s.
// The low u32 is the grid-index byte vector (aux8[0..4]); the high u32 carries
// four 7-bit sign selectors in its top nibble region plus a 4-bit sub-scale.
pub(crate) fn dequantize_iq2_xxs(src: &[u8], out: &mut [f32]) {
    const B: usize = 66;
    let nb = out.len() / QK_K;
    for i in 0..nb {
        let blk = &src[i * B..];
        let d = scale_of(blk);
        let mut o = i * QK_K;
        for ib32 in 0..QK_K / 32 {
            let off = 2 + 8 * ib32;
            let grid_idx = rd_u32(blk, off);
            let sign_word = rd_u32(blk, off + 4);
            let db = d * (0.5 + (sign_word >> 28) as f32) * 0.25;
            let aux8 = grid_idx.to_le_bytes();
            for l in 0..4 {
                let grid = grid_bytes_u64(IQ2XXS_GRID[aux8[l] as usize]);
                let signs = KSIGNS_IQ2XS[((sign_word >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    out[o + j] = db * grid[j] as f32 * sign_of(signs, j);
                }
                o += 8;
            }
        }
    }
}

// ── IQ2_XS — 2.3125 bpw, 74 B/block ─────────────────────────────────────────
//
// block_iq2_xs { ggml_half d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; }
//   d @ 0..2   qs @ 2..66 (u16[32])   scales @ 66..74
//
// Each u16 qs entry packs a 9-bit grid index and a 7-bit sign selector.
// `scales[ib32]` supplies one 4-bit sub-scale per 16-element half.
pub(crate) fn dequantize_iq2_xs(src: &[u8], out: &mut [f32]) {
    const B: usize = 74;
    let nb = out.len() / QK_K;
    for i in 0..nb {
        let blk = &src[i * B..];
        let d = scale_of(blk);
        let mut o = i * QK_K;
        for ib32 in 0..QK_K / 32 {
            let sc = blk[66 + ib32];
            let db = [
                d * (0.5 + (sc & 0xf) as f32) * 0.25,
                d * (0.5 + (sc >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let q = rd_u16(blk, 2 + 2 * (4 * ib32 + l));
                let grid = grid_bytes_u64(IQ2XS_GRID[(q & 511) as usize]);
                let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                for j in 0..8 {
                    out[o + j] = db[l / 2] * grid[j] as f32 * sign_of(signs, j);
                }
                o += 8;
            }
        }
    }
}

// ── IQ2_S — 2.5625 bpw, 82 B/block ──────────────────────────────────────────
//
// block_iq2_s { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32];
//               uint8_t scales[QK_K/32]; }
//   d @ 0..2   qs @ 2..66   qh @ 66..74   scales @ 74..82
//
// Upstream folds the sign bytes into the *tail* of the `qs` array:
// `const uint8_t * signs = qs + QK_K/8;` — i.e. absolute offset 2 + 32 = 34.
// Both `qs` and `signs` are running pointers advanced by 4 per ib32.
pub(crate) fn dequantize_iq2_s(src: &[u8], out: &mut [f32]) {
    const B: usize = 82;
    const QS: usize = 2;
    const SIGNS: usize = QS + QK_K / 8; // 34
    const QH: usize = 66;
    const SCALES: usize = 74;
    let nb = out.len() / QK_K;
    for i in 0..nb {
        let blk = &src[i * B..];
        let d = scale_of(blk);
        let mut o = i * QK_K;
        for ib32 in 0..QK_K / 32 {
            let sc = blk[SCALES + ib32];
            let db = [
                d * (0.5 + (sc & 0xf) as f32) * 0.25,
                d * (0.5 + (sc >> 4) as f32) * 0.25,
            ];
            let qh = blk[QH + ib32] as u32;
            for l in 0..4 {
                let idx = (blk[QS + 4 * ib32 + l] as u32 | ((qh << (8 - 2 * l)) & 0x300)) as usize;
                let grid = grid_bytes_u64(IQ2S_GRID[idx]);
                let signs = blk[SIGNS + 4 * ib32 + l];
                for j in 0..8 {
                    out[o + j] = db[l / 2] * grid[j] as f32 * sign_of(signs, j);
                }
                o += 8;
            }
        }
    }
}

// ── IQ3_XXS — 3.0625 bpw, 98 B/block ────────────────────────────────────────
//
// block_iq3_xxs { ggml_half d; uint8_t qs[3*QK_K/8]; }
//   d @ 0..2   qs @ 2..98
//
// `scales_and_signs = qs + QK_K/4` → absolute 2 + 64 = 66. Each u32 there holds
// a 4-bit sub-scale in the top nibble and four 7-bit sign selectors. `qs` is a
// running pointer advanced by 8 per ib32; each 8-byte group feeds two grids.
pub(crate) fn dequantize_iq3_xxs(src: &[u8], out: &mut [f32]) {
    const B: usize = 98;
    const QS: usize = 2;
    const SCALES_AND_SIGNS: usize = QS + QK_K / 4; // 66
    let nb = out.len() / QK_K;
    for i in 0..nb {
        let blk = &src[i * B..];
        let d = scale_of(blk);
        let mut o = i * QK_K;
        for ib32 in 0..QK_K / 32 {
            let aux = rd_u32(blk, SCALES_AND_SIGNS + 4 * ib32);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = grid_bytes_u32(IQ3XXS_GRID[blk[QS + 8 * ib32 + 2 * l] as usize]);
                let g2 = grid_bytes_u32(IQ3XXS_GRID[blk[QS + 8 * ib32 + 2 * l + 1] as usize]);
                for j in 0..4 {
                    out[o + j] = db * g1[j] as f32 * sign_of(signs, j);
                    out[o + j + 4] = db * g2[j] as f32 * sign_of(signs, j + 4);
                }
                o += 8;
            }
        }
    }
}

// ── IQ3_S — 3.3125 bpw, 110 B/block ─────────────────────────────────────────
//
// block_iq3_s { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32];
//               uint8_t signs[QK_K/8]; uint8_t scales[IQ3S_N_SCALE=QK_K/64]; }
//   d @ 0..2   qs @ 2..66   qh @ 66..74   signs @ 74..106   scales @ 106..110
//
// Upstream walks ib32 in steps of 2 and unrolls the two halves by hand: the
// first half consumes `qh[0]` at db1, then `qs`/`signs` advance, then the second
// half consumes `qh[1]` at db2, and only afterwards does `qh += 2`. So step
// `ib32` reads qh[0] = qh_base[ib32] and qh[1] = qh_base[ib32 + 1]; `qs`
// advances 8 per half (16 per step) and `signs` 4 per half (8 per step).
// Both are kept as running cursors here; `qh` is indexed absolutely.
pub(crate) fn dequantize_iq3_s(src: &[u8], out: &mut [f32]) {
    const B: usize = 110;
    const QS: usize = 2;
    const QH: usize = 66;
    const SIGNS: usize = 74;
    const SCALES: usize = 106;
    let nb = out.len() / QK_K;
    for i in 0..nb {
        let blk = &src[i * B..];
        let d = scale_of(blk);
        let mut o = i * QK_K;
        let mut qs = QS;
        let mut sg = SIGNS;
        for ib32 in (0..QK_K / 32).step_by(2) {
            let sc = blk[SCALES + ib32 / 2];
            let db = [
                d * (1.0 + 2.0 * (sc & 0xf) as f32),
                d * (1.0 + 2.0 * (sc >> 4) as f32),
            ];
            for half in 0..2 {
                // qh_base has advanced by `ib32` at this point (qh += 2 per step).
                let qh = blk[QH + ib32 + half] as u32;
                for l in 0..4 {
                    let i1 = (blk[qs + 2 * l] as u32 | ((qh << (8 - 2 * l)) & 256)) as usize;
                    let i2 = (blk[qs + 2 * l + 1] as u32 | ((qh << (7 - 2 * l)) & 256)) as usize;
                    let g1 = grid_bytes_u32(IQ3S_GRID[i1]);
                    let g2 = grid_bytes_u32(IQ3S_GRID[i2]);
                    let signs = blk[sg + l];
                    for j in 0..4 {
                        out[o + j] = db[half] * g1[j] as f32 * sign_of(signs, j);
                        out[o + j + 4] = db[half] * g2[j] as f32 * sign_of(signs, j + 4);
                    }
                    o += 8;
                }
                qs += 8;
                sg += 4;
            }
        }
    }
}

// ── IQ1_S — 1.5625 bpw, 50 B/block ──────────────────────────────────────────
//
// block_iq1_s { ggml_half d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; }
//   d @ 0..2   qs @ 2..34   qh @ 34..50 (u16[8])
//
// IQ1 stores `grid[j] + delta` with delta = ±IQ1S_DELTA selected per 32-block
// by the high bit of qh[ib]. `qs` is a running pointer advanced by 4 per ib.
pub(crate) fn dequantize_iq1_s(src: &[u8], out: &mut [f32]) {
    const B: usize = 50;
    const QS: usize = 2;
    const QH: usize = 34;
    let nb = out.len() / QK_K;
    for i in 0..nb {
        let blk = &src[i * B..];
        let d = scale_of(blk);
        let mut o = i * QK_K;
        for ib in 0..QK_K / 32 {
            let qh = rd_u16(blk, QH + 2 * ib);
            let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
            let delta = if qh & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for l in 0..4 {
                let idx =
                    blk[QS + 4 * ib + l] as usize | (((qh >> (3 * l)) & 7) as usize) << 8;
                let grid = grid_i8_u64(IQ1S_GRID[idx]);
                for j in 0..8 {
                    out[o + j] = dl * (grid[j] as f32 + delta);
                }
                o += 8;
            }
        }
    }
}

// ── IQ1_M — 1.75 bpw, 56 B/block ────────────────────────────────────────────
//
// block_iq1_m { uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16];
//               uint8_t scales[QK_K/32]; }
//   qs @ 0..32   qh @ 32..48   scales @ 48..56
//
// Note the ABSENCE of a leading `d` field: the block scale is not stored as a
// plain half. Upstream reconstructs a 16-bit value by gathering one nibble out
// of each of the four u16 views of `scales` and reinterpreting that as an f16
// (`iq1m_scale_t` is a union of `uint16_t` and `ggml_half`). Getting this
// gather wrong shifts every scale in the block, so it is spelled out below.
//
// `qs` and `qh` are running pointers advanced by 4 and 2 per ib respectively.
pub(crate) fn dequantize_iq1_m(src: &[u8], out: &mut [f32]) {
    const B: usize = 56;
    const QS: usize = 0;
    const QH: usize = 32;
    const SCALES: usize = 48;
    let nb = out.len() / QK_K;
    for i in 0..nb {
        let blk = &src[i * B..];
        let sc = [
            rd_u16(blk, SCALES),
            rd_u16(blk, SCALES + 2),
            rd_u16(blk, SCALES + 4),
            rd_u16(blk, SCALES + 6),
        ];
        let d = f16_to_f32(
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000),
        );
        let mut o = i * QK_K;
        for ib in 0..QK_K / 32 {
            let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) as f32 + 1.0);
            let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as f32 + 1.0);
            let qh0 = blk[QH + 2 * ib];
            let qh1 = blk[QH + 2 * ib + 1];
            let q = [
                blk[QS + 4 * ib] as u32,
                blk[QS + 4 * ib + 1] as u32,
                blk[QS + 4 * ib + 2] as u32,
                blk[QS + 4 * ib + 3] as u32,
            ];
            let idx = [
                q[0] | ((qh0 as u32) << 8) & 0x700,
                q[1] | ((qh0 as u32) << 4) & 0x700,
                q[2] | ((qh1 as u32) << 8) & 0x700,
                q[3] | ((qh1 as u32) << 4) & 0x700,
            ];
            let delta = [
                if qh0 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                if qh0 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                if qh1 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                if qh1 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
            ];
            for l in 0..2 {
                let grid = grid_i8_u64(IQ1S_GRID[idx[l] as usize]);
                for j in 0..8 {
                    out[o + j] = dl1 * (grid[j] as f32 + delta[l]);
                }
                o += 8;
            }
            for l in 2..4 {
                let grid = grid_i8_u64(IQ1S_GRID[idx[l] as usize]);
                for j in 0..8 {
                    out[o + j] = dl2 * (grid[j] as f32 + delta[l]);
                }
                o += 8;
            }
        }
    }
}

// ── IQ4_NL — 4.5 bpw, 18 B/block (QK4_NL = 32) ──────────────────────────────
//
// block_iq4_nl { ggml_half d; uint8_t qs[QK4_NL/2]; }
//   d @ 0..2   qs @ 2..18
//
// The only IQ variant whose block is 32 elements rather than a 256 super-block.
// Note the interleave: the low nibbles fill the first 16 outputs, the high
// nibbles the next 16 — not a straight nibble-pair expansion.
pub(crate) fn dequantize_iq4_nl(src: &[u8], out: &mut [f32]) {
    const B: usize = 18;
    const QS: usize = 2;
    let nb = out.len() / QK4_NL;
    for i in 0..nb {
        let blk = &src[i * B..];
        let d = scale_of(blk);
        let o = i * QK4_NL;
        for j in 0..QK4_NL / 2 {
            out[o + j] = d * KVALUES_IQ4NL[(blk[QS + j] & 0xf) as usize] as f32;
            out[o + j + QK4_NL / 2] = d * KVALUES_IQ4NL[(blk[QS + j] >> 4) as usize] as f32;
        }
    }
}

// ── IQ4_XS — 4.25 bpw, 136 B/block ──────────────────────────────────────────
//
// block_iq4_xs { ggml_half d; uint16_t scales_h; uint8_t scales_l[QK_K/64];
//                uint8_t qs[QK_K/2]; }
//   d @ 0..2   scales_h @ 2..4   scales_l @ 4..8   qs @ 8..136
//
// The 6-bit signed sub-scale is split across `scales_l` (low 4 bits, one nibble
// per 32-block) and `scales_h` (high 2 bits), then biased by -32.
pub(crate) fn dequantize_iq4_xs(src: &[u8], out: &mut [f32]) {
    const B: usize = 136;
    const SCALES_H: usize = 2;
    const SCALES_L: usize = 4;
    const QS: usize = 8;
    let nb = out.len() / QK_K;
    for i in 0..nb {
        let blk = &src[i * B..];
        let d = scale_of(blk);
        let scales_h = rd_u16(blk, SCALES_H);
        let mut o = i * QK_K;
        for ib in 0..QK_K / 32 {
            let lo = ((blk[SCALES_L + ib / 2] >> (4 * (ib % 2))) & 0xf) as i32;
            let hi = ((scales_h >> (2 * ib)) & 3) as i32;
            let dl = d * ((lo | (hi << 4)) - 32) as f32;
            let q = &blk[QS + 16 * ib..];
            for j in 0..16 {
                out[o + j] = dl * KVALUES_IQ4NL[(q[j] & 0xf) as usize] as f32;
                out[o + j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
            }
            o += 32;
        }
    }
}

// ── Dispatch ────────────────────────────────────────────────────────────────

/// One supported IQ GGML type, with its block geometry and kernel.
///
/// Kept separate from `gguf_input::GgmlType` so the type-table plumbing in that
/// module stays a flat match: it maps a GGML id to this enum and then only has
/// to forward `block_elems` / `block_bytes` / `dequantize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IqKind {
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq1S,
    Iq1M,
    Iq4Nl,
    Iq4Xs,
}

impl IqKind {
    /// GGML type id → IQ kind. `None` for ids outside the IQ family.
    pub(crate) fn from_ggml_id(v: u32) -> Option<Self> {
        match v {
            16 => Some(Self::Iq2Xxs),
            17 => Some(Self::Iq2Xs),
            18 => Some(Self::Iq3Xxs),
            19 => Some(Self::Iq1S),
            20 => Some(Self::Iq4Nl),
            21 => Some(Self::Iq3S),
            22 => Some(Self::Iq2S),
            23 => Some(Self::Iq4Xs),
            29 => Some(Self::Iq1M),
            _ => None,
        }
    }

    /// Elements per block. Only IQ4_NL differs from `QK_K`.
    pub(crate) fn block_elems(self) -> usize {
        match self {
            Self::Iq4Nl => QK4_NL,
            _ => QK_K,
        }
    }

    /// Bytes per block — `sizeof(block_iq*)` from `ggml-common.h`.
    pub(crate) fn block_bytes(self) -> usize {
        match self {
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq2S => 82,
            Self::Iq3Xxs => 98,
            Self::Iq3S => 110,
            Self::Iq1S => 50,
            Self::Iq1M => 56,
            Self::Iq4Nl => 18,
            Self::Iq4Xs => 136,
        }
    }

    /// Total bytes for `n` elements, matching `ggml_type_size` rounding.
    pub(crate) fn tensor_bytes(self, n: usize) -> usize {
        let be = self.block_elems();
        n.div_ceil(be) * self.block_bytes()
    }

    /// Dequantize `src` into `out`. `src.len()` must be at least
    /// `tensor_bytes(out.len())` and `out.len()` a multiple of `block_elems()`.
    pub(crate) fn dequantize(self, src: &[u8], out: &mut [f32]) {
        let be = self.block_elems();
        debug_assert_eq!(out.len() % be, 0, "output not block-aligned");
        debug_assert!(
            src.len() >= self.tensor_bytes(out.len()),
            "source shorter than block geometry implies"
        );
        match self {
            Self::Iq2Xxs => dequantize_iq2_xxs(src, out),
            Self::Iq2Xs => dequantize_iq2_xs(src, out),
            Self::Iq2S => dequantize_iq2_s(src, out),
            Self::Iq3Xxs => dequantize_iq3_xxs(src, out),
            Self::Iq3S => dequantize_iq3_s(src, out),
            Self::Iq1S => dequantize_iq1_s(src, out),
            Self::Iq1M => dequantize_iq1_m(src, out),
            Self::Iq4Nl => dequantize_iq4_nl(src, out),
            Self::Iq4Xs => dequantize_iq4_xs(src, out),
        }
    }
}

/// Encode a known IQ3_S block and assert the kernel reproduces it.
///
/// This is a *self-consistency* check on the transcription (offsets, pointer
/// advance, endianness) built from hand-computed values rather than a
/// round-trip — IQ is lossy, so there is nothing to round-trip to. Numerical
/// parity against llama.cpp is verified separately in `tests/` by comparing
/// against an F16 reference produced by `llama-quantize`.
#[cfg(test)]
mod tests {
    use super::*;

    /// Build one synthetic IQ4_NL block: d = 1.0, all qs nibbles = 0.
    /// Expect every output = kvalues_iq4nl[0] * 1.0 = -127.0.
    #[test]
    fn iq4_nl_all_zero_nibbles_hit_first_value() {
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
        let mut out = vec![0.0f32; QK4_NL];
        IqKind::Iq4Nl.dequantize(&blk, &mut out);
        for (j, v) in out.iter().enumerate() {
            assert_eq!(*v, -127.0, "out[{j}]");
        }
    }

    /// IQ1_S: a block whose qh encodes dl = d*1 (scale code 0) and delta = +0.125,
    /// with all grid indices pointing at iq1s_grid[0] = all-(-1) bytes.
    /// Expect every output = d * (-1 + 0.125).
    #[test]
    fn iq1_s_delta_and_scale_pickup() {
        let d = 1.0f32;
        let mut blk = vec![0u8; 50];
        blk[0..2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        // qh[ib] = 0 → (qh >> 12) & 7 == 0 → dl = d * 1; high bit clear → +delta.
        // qs bytes stay 0 → grid index = 0 for every l.
        let mut out = vec![0.0f32; QK_K];
        IqKind::Iq1S.dequantize(&blk, &mut out);
        let g0 = grid_i8_u64(IQ1S_GRID[0]);
        for (o, v) in out.iter().enumerate() {
            let expect = d * (g0[o % 8] as f32 + IQ1S_DELTA);
            assert!((v - expect).abs() < 1e-6, "out[{o}] = {v}, want {expect}");
        }
    }

    /// IQ4_XS sub-scale is 6-bit signed with a -32 bias: ls = 32 must give
    /// dl = d * 0 = 0 regardless of the stored values.
    #[test]
    fn iq4_xs_zero_scale_biases_to_zero() {
        let mut blk = vec![0u8; 136];
        blk[0..2].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
        // scales_l nibbles = 0 for every ib. scales_h holds the high 2 bits of
        // EACH of the eight 6-bit sub-scales, two bits per ib, so reaching
        // ls = 32 on all of them needs 0b10 in every 2-bit slot → 0xAAAA.
        // (Setting scales_h = 2 would only fix ib = 0 and leave the rest at
        // ls = 0, i.e. dl = -32 * d.)
        blk[2..4].copy_from_slice(&0xAAAAu16.to_le_bytes());
        // Give qs non-zero nibbles so a wrong bias would show up as non-zero.
        for b in &mut blk[8..136] {
            *b = 0xff;
        }
        let mut out = vec![0.0f32; QK_K];
        IqKind::Iq4Xs.dequantize(&blk, &mut out);
        for (o, v) in out.iter().enumerate() {
            assert_eq!(*v, 0.0, "out[{o}]");
        }
    }

    /// Block geometry must match `sizeof(block_iq*)` / `sizeof(block_iq4_nl)`.
    #[test]
    fn block_geometry_matches_upstream_structs() {
        let cases = [
            (IqKind::Iq2Xxs, 256, 66),
            (IqKind::Iq2Xs, 256, 74),
            (IqKind::Iq2S, 256, 82),
            (IqKind::Iq3Xxs, 256, 98),
            (IqKind::Iq3S, 256, 110),
            (IqKind::Iq1S, 256, 50),
            (IqKind::Iq1M, 256, 56),
            (IqKind::Iq4Nl, 32, 18),
            (IqKind::Iq4Xs, 256, 136),
        ];
        for (k, elems, bytes) in cases {
            assert_eq!(k.block_elems(), elems, "{k:?} elems");
            assert_eq!(k.block_bytes(), bytes, "{k:?} bytes");
        }
    }

    /// Round-trip a float through f16 bits without pulling in a converter.
    fn f32_to_f16_bits(v: f32) -> u16 {
        // Test-only: exact for the values used above (all representable).
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        if exp == 0 {
            return sign;
        }
        let e = exp - 127 + 15;
        assert!((1..=30).contains(&e), "test helper only handles normals");
        let mant = ((b >> 13) & 0x3ff) as u16;
        sign | ((e as u16) << 10) | mant
    }
}
