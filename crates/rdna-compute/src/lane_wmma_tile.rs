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

use crate::dispatch::ReplayLaunchBindings;
use crate::lane_wmma::{
    err, i32_of, lane_wmma_prefill_plan, need_bytes, need_f32, u32_of, LaneWmmaLaunch,
    LaneWmmaWeight, LANE_WMMA_PREFILL_SPLIT_ROWS,
};
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

    /// Whether the kernel file exports this tile for `terms` activation terms.
    pub fn instantiated(&self, terms: usize) -> bool {
        lane_wmma_tile_set(terms).contains(self)
    }

    fn entry(&self, terms: usize) -> String {
        format!("lwmma_tile_bf16_t{terms}_{}", self.name())
    }
}

const fn tile(bm: usize, bn: usize, wm: usize, wn: usize) -> LaneWmmaTile {
    LaneWmmaTile {
        bm,
        bn,
        wm,
        wn,
        qg: 1,
    }
}

const fn tile_g4(bm: usize, bn: usize, wm: usize, wn: usize) -> LaneWmmaTile {
    LaneWmmaTile {
        bm,
        bn,
        wm,
        wn,
        qg: 4,
    }
}

/// The tiles the kernel file exports for one activation term (the HC down): exactly the ones
/// [`lane_wmma_tile_pick`] can select, nothing else is built.
const TILES_T1: &[LaneWmmaTile] = &[
    tile_g4(32, 32, 2, 1),
    tile(32, 64, 2, 2),
    tile(64, 64, 4, 1),
];

/// The tiles the kernel file exports for two activation terms (dense BF16 projections).
const TILES_T2: &[LaneWmmaTile] = &[
    tile_g4(16, 32, 1, 1),
    tile(32, 64, 2, 2),
    tile(64, 64, 4, 1),
    tile(64, 64, 4, 2),
    tile(32, 128, 2, 2),
    tile(64, 128, 4, 2),
    tile(64, 128, 2, 4),
];

/// Every tile exported for `terms` (1 | 2; empty otherwise).
pub fn lane_wmma_tile_set(terms: usize) -> &'static [LaneWmmaTile] {
    match terms {
        1 => TILES_T1,
        2 => TILES_T2,
        _ => &[],
    }
}

/// Tuned picks per served `(m, k, terms)`: ascending `(max_rows, pick)` ranges, the last one open.
/// `None` keeps the shipped lwmma kernel (bitwise the same) where every tile is slower. Fine-grid
/// bench on card C, 2026-11-10 (fabric j8842, 65..1192 rows, min of two passes): the HC down
/// (320 x 10240, one term) loses to the shipped kernel up to about 360 rows (its serial K walk is
/// latency bound at few rows); every other shape wins at every row count (0.29x..0.87x).
const TILE_PICKS: &[(usize, usize, usize, &[(usize, Option<LaneWmmaTile>)])] = &[
    (
        320,
        10240,
        1,
        &[
            (360, None),
            (420, Some(tile_g4(32, 32, 2, 1))),
            (1100, Some(tile(32, 64, 2, 2))),
            (usize::MAX, Some(tile(64, 64, 4, 1))),
        ],
    ),
    (
        320,
        10240,
        2,
        &[
            (200, Some(tile_g4(16, 32, 1, 1))),
            (1100, Some(tile(32, 64, 2, 2))),
            (usize::MAX, Some(tile(64, 64, 4, 1))),
        ],
    ),
    (
        512,
        2560,
        2,
        &[
            (140, Some(tile_g4(16, 32, 1, 1))),
            (420, Some(tile(32, 64, 2, 2))),
            (530, Some(tile(64, 64, 4, 2))),
            (750, Some(tile(32, 64, 2, 2))),
            (usize::MAX, Some(tile(64, 128, 4, 2))),
        ],
    ),
    (
        640,
        2560,
        2,
        &[
            (110, Some(tile_g4(16, 32, 1, 1))),
            (280, Some(tile(32, 64, 2, 2))),
            (470, Some(tile(32, 128, 2, 2))),
            (650, Some(tile(32, 64, 2, 2))),
            (usize::MAX, Some(tile(64, 128, 2, 4))),
        ],
    ),
    (
        2560,
        640,
        2,
        &[
            (90, Some(tile(32, 64, 2, 2))),
            (140, Some(tile(32, 128, 2, 2))),
            (usize::MAX, Some(tile(64, 128, 2, 4))),
        ],
    ),
    (
        2560,
        2560,
        2,
        &[
            (110, Some(tile(32, 128, 2, 2))),
            (140, Some(tile(32, 64, 2, 2))),
            (usize::MAX, Some(tile(64, 128, 2, 4))),
        ],
    ),
    (10240, 2560, 2, &[(usize::MAX, Some(tile(64, 128, 4, 2)))]),
    // The trunk-sharded shared expert (rank 0 keeps 384 of the 640 rows / columns, rank 1 256) and
    // the shared selector (one row): the per-lane GEMM sites of the served wave. Bench on card E,
    // 2026-11-10 (fabric, 65..1192 rows): every pick is within 7 % of the best tile at every bench
    // row, `None` where the shipped kernel is as fast or faster.
    (
        384,
        2560,
        2,
        &[
            (170, Some(tile_g4(16, 32, 1, 1))),
            (1096, Some(tile(32, 64, 2, 2))),
            (usize::MAX, Some(tile(64, 128, 4, 2))),
        ],
    ),
    (
        256,
        2560,
        2,
        &[
            (138, Some(tile_g4(16, 32, 1, 1))),
            (170, None),
            (278, Some(tile_g4(16, 32, 1, 1))),
            (usize::MAX, Some(tile(32, 64, 2, 2))),
        ],
    ),
    (
        2560,
        384,
        2,
        &[
            (138, Some(tile(32, 64, 2, 2))),
            (usize::MAX, Some(tile(64, 128, 4, 2))),
        ],
    ),
    (
        2560,
        256,
        2,
        &[
            (170, Some(tile(32, 64, 2, 2))),
            (usize::MAX, Some(tile(64, 128, 4, 2))),
        ],
    ),
    (1, 2560, 2, &[(278, None), (usize::MAX, Some(tile_g4(16, 32, 1, 1)))]),
];

/// The tile configuration [`lane_wmma_tile_gemm`] runs for `(m, k, rows, terms)`: the tuned table
/// for the served shapes, `None` for any other shape or a row range where the shipped lwmma kernel
/// is faster (the caller stays on it; it is bitwise the same).
pub fn lane_wmma_tile_pick(m: usize, k: usize, rows: usize, terms: usize) -> Option<LaneWmmaTile> {
    let (_, _, _, ranges) = TILE_PICKS
        .iter()
        .find(|(pm, pk, pt, _)| (*pm, *pk, *pt) == (m, k, terms))?;
    ranges
        .iter()
        .find(|(max_rows, _)| rows <= *max_rows)
        .and_then(|(_, t)| *t)
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
    let t = lane_wmma_tile_pick(m, k, rows, terms)
        .ok_or_else(|| err(format!("tile: no tuned tile for m {m} k {k} rows {rows} terms {terms}")))?;
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

    /// The product entry of the sub-512-row prefill lane-WMMA GEMMs above 64 rows: the tile kernel
    /// ([`Gpu::lane_wmma_tile_gemm_prefill`]) for a BF16 weight of a tuned `(m, k, terms)` shape
    /// while `HIPFIRE_QWEN4_PREFILL_LWMMA_TILE` is on (default), else the shipped
    /// [`Gpu::lane_wmma_gemm_prefill`]. Every output word is the same either way.
    #[allow(clippy::too_many_arguments)]
    pub fn lane_wmma_prefill_gemm(
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
        if self.flags.qwen4_prefill_lwmma_tile_enabled()
            && matches!(w, LaneWmmaWeight::Bf16(_))
            && lane_wmma_tile_pick(m, k, rows, terms).is_some()
        {
            return self.lane_wmma_tile_gemm_prefill(w, x, y, m, k, rows, terms, hc_act);
        }
        self.lane_wmma_gemm_prefill(w, x, y, m, k, rows, terms, hc_act)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every selectable tile is exported, and every exported tile is selectable (the module builds
    /// and twins nothing else).
    #[test]
    fn exports_are_exactly_the_selectable_tiles() {
        let mut selected: Vec<(usize, LaneWmmaTile)> = Vec::new();
        for (m, k, terms, ranges) in TILE_PICKS {
            for t in ranges.iter().filter_map(|(_, t)| t.as_ref()) {
                assert!(
                    t.instantiated(*terms),
                    "{m}x{k} t{terms}: {t:?} not exported"
                );
                if !selected.contains(&(*terms, *t)) {
                    selected.push((*terms, *t));
                }
            }
        }
        for terms in [1, 2] {
            for t in lane_wmma_tile_set(terms) {
                assert!(
                    selected.contains(&(terms, *t)),
                    "t{terms}: {t:?} exported but never picked"
                );
            }
        }
        for m in [1, 256, 320, 384, 512, 640, 2560, 10240, 96, 1000] {
            for k in [256, 384, 640, 2560, 10240] {
                for rows in [1, 65, 149, 300, 301, 596, 800, 801, 1192, 2048] {
                    for terms in [1, 2] {
                        if let Some(t) = lane_wmma_tile_pick(m, k, rows, terms) {
                            assert!(
                                t.instantiated(terms),
                                "m {m} k {k} rows {rows} t {terms}: {t:?}"
                            );
                        }
                    }
                }
            }
        }
        assert!(lane_wmma_tile_pick(96, 2560, 300, 2).is_none());
        assert!(lane_wmma_tile_pick(512, 2560, 300, 1).is_none());
        assert!(lane_wmma_tile_pick(320, 10240, 149, 1).is_none());
        assert!(lane_wmma_tile_pick(320, 10240, 596, 1).is_some());
        // The trunk-sharded shared expert and the selector: tiled in the wave, the shipped kernel
        // where it is as fast.
        assert_eq!(lane_wmma_tile_pick(384, 2560, 1050, 2), Some(tile(32, 64, 2, 2)));
        assert_eq!(lane_wmma_tile_pick(384, 2560, 1192, 2), Some(tile(64, 128, 4, 2)));
        assert_eq!(lane_wmma_tile_pick(256, 2560, 1050, 2), Some(tile(32, 64, 2, 2)));
        assert_eq!(lane_wmma_tile_pick(256, 2560, 149, 2), None);
        assert_eq!(lane_wmma_tile_pick(2560, 384, 1050, 2), Some(tile(64, 128, 4, 2)));
        assert_eq!(lane_wmma_tile_pick(2560, 256, 149, 2), Some(tile(32, 64, 2, 2)));
        assert_eq!(lane_wmma_tile_pick(1, 2560, 1050, 2), Some(tile_g4(16, 32, 1, 1)));
        assert_eq!(lane_wmma_tile_pick(1, 2560, 256, 2), None);
    }
}
