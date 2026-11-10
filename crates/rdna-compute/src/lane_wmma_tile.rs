// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane-WMMA prefill tile GEMM: a staged-tile twin of the [`crate::lane_wmma`] A1 body for the
//! sub-512-row prefill (BF16 weights, f32 activations, 1..=2 BF16 split terms).
//!
//! Every output word is BITWISE what [`crate::lane_wmma::lane_wmma_gemm`] writes at the same
//! `split_k` (see the contract in `kernels/src/qwen4_lwmma_tile.gfx1201.hip`). The difference is
//! the schedule: one block owns a `BM x BN` output tile and walks K once, staging the f32 X tile
//! from global ONCE per block, splitting it ONCE into BF16 term planes in LDS (the incumbent
//! re-reads and re-splits X for every 16-column block), with the four quarter chains and the
//! split-K chunks folded in registers (one launch, no partial scratch, no counters).
//!
//! Kernels: `kernels/src/qwen4_lwmma_tile.gfx1201.hip`. This module reads no environment variable.

use crate::lane_wmma::{
    err, i32_of, lane_wmma_prefill_plan, need_bytes, need_f32, u32_of, LaneWmmaLaunch,
    LaneWmmaWeight, LANE_WMMA_PREFILL_SPLIT_ROWS,
};
use crate::dispatch::ReplayLaunchBindings;
use crate::{Gpu, GpuTensor};
use hip_bridge::{HipResult, KernargBlob};

/// The tile kernel source (one module for every `lwmma_tile_bf16_*` entry).
pub const LANE_WMMA_TILE_SRC: &str =
    include_str!("../../../kernels/src/qwen4_lwmma_tile.gfx1201.hip");

const LANE_WMMA_TILE_MODULE: &str = "qwen4_lwmma_tile";

/// One compiled tile configuration: the block owns `bm x bn` outputs, `wm x wn` waves per wave
/// group; `qg` wave groups per block (1: one group walks all of K; 4: group q owns quarter q of
/// every split-K chunk, for latency-bound small grids).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneWmmaTile {
    pub bm: usize,
    pub bn: usize,
    pub wm: usize,
    pub wn: usize,
    pub qg: usize,
}

impl LaneWmmaTile {
    /// Threads per block.
    pub fn threads(&self) -> usize {
        32 * self.wm * self.wn * self.qg
    }

    /// `b<BM>x<BN>w<WM>x<WN>` plus `g4` for the quarter-group variants.
    pub fn name(&self) -> String {
        let g = if self.qg == 4 { "g4" } else { "" };
        format!("b{}x{}w{}x{}{g}", self.bm, self.bn, self.wm, self.wn)
    }

    /// Whether the kernel file instantiates this tile for `terms` activation terms (the wider
    /// quarter-group tiles only fit the 64 KB LDS with one term).
    pub fn instantiated(&self, terms: usize) -> bool {
        LANE_WMMA_TILES.contains(self) && (self.qg == 1 || terms == 1 || (self.bm, self.bn) == (16, 32))
    }

    fn entry(&self, terms: usize) -> String {
        format!("lwmma_tile_bf16_t{terms}_{}", self.name())
    }
}

const fn tile(bm: usize, bn: usize, wm: usize, wn: usize) -> LaneWmmaTile {
    LaneWmmaTile { bm, bn, wm, wn, qg: 1 }
}

const fn tile_g4(bm: usize, bn: usize, wm: usize, wn: usize) -> LaneWmmaTile {
    LaneWmmaTile { bm, bn, wm, wn, qg: 4 }
}

/// Every tile configuration the kernel file instantiates (see [`LaneWmmaTile::instantiated`]).
pub const LANE_WMMA_TILES: &[LaneWmmaTile] = &[
    tile(64, 64, 4, 2),
    tile(64, 128, 2, 4),
    tile(128, 64, 4, 2),
    tile(128, 128, 4, 4),
    tile(32, 64, 2, 2),
    tile(32, 128, 2, 4),
    tile(64, 64, 2, 2),
    tile(64, 128, 4, 2),
    tile(32, 128, 2, 2),
    tile(64, 64, 4, 1),
    tile(32, 64, 2, 1),
    tile(128, 64, 8, 1),
    tile(64, 256, 4, 4),
    tile_g4(16, 32, 1, 1),
    tile_g4(16, 64, 1, 2),
    tile_g4(32, 32, 2, 1),
];

/// Row buckets of the pick table: up to 300 rows, up to 800 rows, above.
fn rows_bucket(rows: usize) -> usize {
    match rows {
        0..=300 => 0,
        301..=800 => 1,
        _ => 2,
    }
}

/// Tuned picks per served `(m, k)` (and `terms`, 0 = any) and row bucket (T3 bench on card C,
/// 2026-11-10: the fastest configuration of the sweep at 149 / 596 / 1192 rows). The quarter-group
/// tiles win only the 149-row bucket of the 320 x 10240 shapes, where the serial K walk of the
/// plain tiles is latency bound.
const TILE_PICKS: &[(usize, usize, usize, [LaneWmmaTile; 3])] = &[
    (320, 10240, 1, [tile_g4(32, 32, 2, 1), tile(32, 64, 2, 2), tile(64, 64, 4, 1)]),
    (320, 10240, 2, [tile_g4(16, 32, 1, 1), tile(32, 64, 2, 2), tile(64, 64, 4, 1)]),
    (512, 2560, 0, [tile(32, 64, 2, 2), tile(32, 64, 2, 2), tile(64, 64, 4, 2)]),
    (640, 2560, 0, [tile(32, 64, 2, 2), tile(32, 128, 2, 2), tile(64, 128, 4, 2)]),
    (2560, 640, 0, [tile(64, 128, 4, 2), tile(64, 128, 2, 4), tile(64, 128, 2, 4)]),
    (2560, 2560, 0, [tile(64, 128, 2, 4), tile(32, 128, 2, 2), tile(64, 128, 4, 2)]),
    (10240, 2560, 0, [tile(64, 128, 4, 2), tile(64, 128, 4, 2), tile(64, 128, 4, 2)]),
];

/// The tile configuration [`lane_wmma_tile_gemm`] runs for `(m, k, rows, terms)`: the tuned table
/// for the served shapes, else a shape-generic rule (every configuration is bitwise the same).
pub fn lane_wmma_tile_pick(m: usize, k: usize, rows: usize, terms: usize) -> LaneWmmaTile {
    let bucket = rows_bucket(rows);
    if let Some((_, _, _, t)) = TILE_PICKS
        .iter()
        .find(|(pm, pk, pt, _)| (*pm, *pk) == (m, k) && (*pt == 0 || *pt == terms))
    {
        return t[bucket];
    }
    match (bucket, m % 128 == 0, m % 64 == 0) {
        (0, _, _) => tile(32, 64, 2, 2),
        (_, true, _) => tile(64, 128, 4, 2),
        (_, false, true) => tile(64, 64, 4, 2),
        _ => tile(32, 64, 2, 2),
    }
}

/// [`lane_wmma_tile_gemm`] on an explicit tile configuration (the bench sweeps these).
#[allow(clippy::too_many_arguments)]
pub fn lane_wmma_tile_gemm_cfg(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
    terms: usize,
    hc_act: Option<f32>,
    split_k: usize,
    t: LaneWmmaTile,
) -> HipResult<LaneWmmaLaunch> {
    if m == 0 || rows == 0 {
        return Err(err("tile: m and rows must be non-zero"));
    }
    if !(1..=2).contains(&terms) {
        return Err(err(format!("tile: terms {terms} not in 1..=2")));
    }
    if split_k == 0 || k == 0 || k % (128 * split_k) != 0 {
        return Err(err(format!(
            "tile: k {k} must be a non-zero multiple of 128 * split_k {split_k}"
        )));
    }
    if !t.instantiated(terms) {
        return Err(err(format!(
            "tile: configuration {t:?} is not instantiated for terms {terms}"
        )));
    }
    need_bytes(w, m * k * 2, "tile bf16 weight")?;
    need_f32(x, rows * k, "tile x")?;
    need_f32(y, rows * m, "tile y")?;
    let grid = [
        u32_of(m.div_ceil(t.bn), "grid x")?,
        u32_of(rows.div_ceil(t.bm), "grid y")?,
        1,
    ];
    let block = [u32_of(t.threads(), "block")?, 1, 1];
    let kernel = t.entry(terms);
    let (scale, act) = match hc_act {
        Some(s) => (s, 1),
        None => (0.0f32, 0),
    };
    gpu.ensure_kernel_public(LANE_WMMA_TILE_MODULE, LANE_WMMA_TILE_SRC, &kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(w.buf.as_ptr() as *const _);
    args.push_ptr(x.buf.as_ptr());
    args.push_ptr(y.buf.as_ptr());
    args.push_i32(i32_of(m, "m")?);
    args.push_i32(i32_of(k, "k")?);
    args.push_i32(i32_of(rows, "rows")?);
    args.push_i32(i32_of(split_k, "split_k")?);
    args.push_f32(scale);
    args.push_i32(act);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        &kernel,
        grid,
        block,
        0,
        args.as_mut_slice(),
        ReplayLaunchBindings::NONE,
    )?;
    Ok(LaneWmmaLaunch {
        kernel,
        grid,
        block,
    })
}

/// `y[rows, m] = epi(x[rows, k] . w[m, k]^T)` for a BF16 weight `w` and an F32 `x`, bitwise what
/// [`crate::lane_wmma::lane_wmma_gemm`] writes at the same `split_k` (`LaneWmmaCfg { nn: 1,
/// split_k }`): `terms` BF16 split terms (1 | 2), `hc_act` the optional HC epilogue scale,
/// `k % (128 * split_k) == 0`. One launch over all rows; no scratch.
#[allow(clippy::too_many_arguments)]
pub fn lane_wmma_tile_gemm(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
    terms: usize,
    hc_act: Option<f32>,
    split_k: usize,
) -> HipResult<LaneWmmaLaunch> {
    let t = lane_wmma_tile_pick(m, k, rows, terms);
    lane_wmma_tile_gemm_cfg(gpu, w, x, y, m, k, rows, terms, hc_act, split_k, t)
}

impl Gpu {
    /// [`Gpu::lane_wmma_gemm_prefill`] on the tile kernel: the split-K pinned by
    /// [`lane_wmma_prefill_plan`] at [`LANE_WMMA_PREFILL_SPLIT_ROWS`] for ALL rows in ONE launch
    /// (no row chunking, no partial scratch, no counters). Bitwise the incumbent's outputs.
    /// The weight must be BF16.
    #[allow(clippy::too_many_arguments)]
    pub fn lane_wmma_tile_gemm_prefill(
        &mut self,
        w: &LaneWmmaWeight<'_>,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        rows: usize,
        terms: usize,
        hc_act: Option<f32>,
    ) -> HipResult<()> {
        let LaneWmmaWeight::Bf16(wt) = w else {
            return Err(err("tile: the tile prefill GEMM takes a BF16 weight"));
        };
        let (split_k, _) = lane_wmma_prefill_plan(m, k, rows, LANE_WMMA_PREFILL_SPLIT_ROWS);
        lane_wmma_tile_gemm(self, wt, x, y, m, k, rows, terms, hc_act, split_k).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pick_is_an_instantiated_tile() {
        for (m, k, _, picks) in TILE_PICKS {
            for t in picks {
                assert!(t.instantiated(1), "{m}x{k}: {t:?} not instantiated");
            }
        }
        for m in [320, 512, 640, 2560, 10240, 96, 1000] {
            for k in [640, 2560, 10240] {
                for rows in [1, 65, 149, 300, 301, 596, 800, 801, 1192, 2048] {
                    for terms in [1, 2] {
                        let t = lane_wmma_tile_pick(m, k, rows, terms);
                        assert!(t.instantiated(terms), "m {m} k {k} rows {rows} t {terms}: {t:?}");
                    }
                }
            }
        }
    }
}
