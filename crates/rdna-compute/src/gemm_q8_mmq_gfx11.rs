// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tiled W8A8 Q8_0 prefill GEMM for gfx11 (`kernels/src/gemm_q8_0_mmq_i8.gfx11.hip`).
//!
//! `Y[n, m] = sum_b fma-fold(RN(wscale[m, b] * d_x[n, b]) * dot_b(m, n))` on
//! `v_wmma_i32_16x16x16_iu8`: the Q8_0 weights are read in place and the activations are the Q8_1
//! `block_q8_1_mmq` plane that [`Gpu::ensure_q8_1_mmq_x`] produces. The kernels are BIT-IDENTICAL
//! to the 64x64 [`Gpu::gemm_q8_0_mmq_4w_gfx1151`] (same per-32-K fold order, exact integer dots), so
//! the tile menu and [`q8_mmq_pick_tile`] are performance choices only
//! (`tests/gemm_q8_0_mmq_gfx11_tiles.rs` asserts the identity on every tile and tail).
//!
//! * [`Gpu::gemm_q8_0_mmq_gfx11_batched`] = producer + GEMM, the shape of
//!   `gemm_q8_0_mmq_4w_gfx1151`.
//! * [`Gpu::q8_mmq_quantize_x`] + [`Gpu::gemm_q8_0_mmq_gfx11_gemm`] split the two so one
//!   quantization serves every projection that shares an activation (RDA q/k/v/g, MLA
//!   q_a/kv_a/g); [`Gpu::q8_mmq_acts_current`] tells whether a later Q8_1 quantizer overwrote it.
//!
//! Admission: gfx1100 and gfx1151 (gfx1151 builds it for the guard and the kernel test; its
//! dispatch route stays on the 64x64 kernel), `k % 128 == 0`, F32 `x`/`y`, 4-byte aligned weight
//! base. `HIPFIRE_Q8_MMQ_GFX11_TILES=0` keeps the 64x64 kernel everywhere (kill switch of the
//! default).

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

const MODULE: &str = "gemm_q8_0_mmq_i8_gfx11";

/// Output tile of the gfx11 W8A8 kernels (`BM x BN` = weight rows x tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q8MmqTile {
    T64x64,
    T64x128,
    T128x128,
}

impl Q8MmqTile {
    pub const ALL: [Q8MmqTile; 3] = [Q8MmqTile::T64x64, Q8MmqTile::T64x128, Q8MmqTile::T128x128];

    pub fn label(self) -> &'static str {
        match self {
            Q8MmqTile::T64x64 => "64x64",
            Q8MmqTile::T64x128 => "64x128",
            Q8MmqTile::T128x128 => "128x128",
        }
    }

    /// `(BM, BN)`.
    pub fn dims(self) -> (usize, usize) {
        match self {
            Q8MmqTile::T64x64 => (64, 64),
            Q8MmqTile::T64x128 => (64, 128),
            Q8MmqTile::T128x128 => (128, 128),
        }
    }

    /// Entry symbol; `bf16` selects the `_bf16` twin whose epilogue rounds every output to BF16.
    fn symbol(self, bf16: bool) -> &'static str {
        match (self, bf16) {
            (Q8MmqTile::T64x64, false) => "gemm_q8_0_mmq_i8_gfx11_t64x64",
            (Q8MmqTile::T64x128, false) => "gemm_q8_0_mmq_i8_gfx11_t64x128",
            (Q8MmqTile::T128x128, false) => "gemm_q8_0_mmq_i8_gfx11_t128x128",
            (Q8MmqTile::T64x64, true) => "gemm_q8_0_mmq_i8_gfx11_t64x64_bf16",
            (Q8MmqTile::T64x128, true) => "gemm_q8_0_mmq_i8_gfx11_t64x128_bf16",
            (Q8MmqTile::T128x128, true) => "gemm_q8_0_mmq_i8_gfx11_t128x128_bf16",
        }
    }
}

/// `HIPFIRE_Q8_MMQ_GFX11_TILES=0` keeps the 64x64 kernel on every gfx11 Q8_0 call (the tiles are
/// bit-identical to it, so the default is on). Read once.
fn tiles_default_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_bool("HIPFIRE_Q8_MMQ_GFX11_TILES", true)
    });
    *ON
}

/// Q8_1 activations in the Gpu's `q8_1_mmq_x_scratch`. The plane stays valid until the next writer
/// of that scratch (any Q8_1 quantizer, a growth); [`Gpu::q8_mmq_acts_current`] tells whether that
/// happened, and the GEMM refuses stale activations.
#[derive(Clone, Copy, Debug)]
pub struct Q8MmqActivations {
    xq: *mut c_void,
    n: usize,
    k: usize,
    generation: u64,
}

impl Q8MmqActivations {
    pub fn rows(&self) -> usize {
        self.n
    }
    pub fn k(&self) -> usize {
        self.k
    }
}

fn q8_bytes(m: usize, k: usize) -> usize {
    m * (k / 32) * 34
}

/// Row count from which the 128x128 tile is considered (node bench, 7900 XTX, N = 2048 / 4096).
const WIDE_TILE_ROWS: usize = 2048;
/// CTAs a 128x128 grid must fill to take the wide tile (>= 1024: 5120x2560 at 4096 rows = 1280,
/// 12032x2560 = 3008; narrower grids lose to 64x128, 5120x2560 at 2048 rows = 640: 822 vs 837 us).
const WIDE_TILE_CTAS: usize = 1024;
/// CTAs a 64x128 grid must fill to beat 64x64 (576x2560 @ 4096 rows = 288: 191.5 vs 204.1 us;
/// 5120x2560 @ 512 = 320: 219.9 vs 228.2 us; 2560x5120 @ 512 = 160: 262.3 vs 239.1 us; 576 / 1088 @
/// 512 rows = 36 / 68: 45.5 vs 37.8 / 78.6 vs 54.1 us; 1088 @ 2048 = 272: 193.0 vs 195.9 us, a tie).
const T64X128_CTAS: usize = 288;

/// Tile for a `(m, n)` call (pure). The thresholds are the measured crossovers of the Raon Q8
/// shapes at 512 / 768 / 1024 / 2048 / 4096 rows on the 96-CU 7900 XTX (receipt
/// `raon/pfx/xq8/ev/xtx-gb2/job.out`, 3 tile kernels + the landed 64x64 kernel, weights rotated out
/// of cache): 128x128 for wide calls, 64x128 when it still fills enough CTAs for the 48 WGPs, else
/// 64x64 (small M or few rows, where 64x128 starves the grid).
pub fn q8_mmq_pick_tile(m: usize, n: usize) -> Q8MmqTile {
    if n >= WIDE_TILE_ROWS && m.div_ceil(128) * n.div_ceil(128) >= WIDE_TILE_CTAS {
        Q8MmqTile::T128x128
    } else if m.div_ceil(64) * n.div_ceil(128) >= T64X128_CTAS {
        Q8MmqTile::T64x128
    } else {
        Q8MmqTile::T64x64
    }
}

/// Arch/shape half of the tiled route's admission (pure apart from the kill switch): gfx11, the
/// switch on, `k % 128 == 0`. The weight base alignment is [`Gpu::q8_mmq_gfx11_supported`]'s.
pub fn q8_mmq_gfx11_admits(arch: &str, k: usize) -> bool {
    tiles_default_enabled() && matches!(arch, "gfx1100" | "gfx1151") && k > 0 && k % 128 == 0
}

impl Gpu {
    /// Whether the tiled gfx11 W8A8 route can run for this device and a weight with `k` columns at
    /// `w_ptr` (the kernel's weight dword loads need a 4-byte aligned base).
    pub fn q8_mmq_gfx11_supported(&self, k: usize, w_ptr: *const c_void) -> bool {
        q8_mmq_gfx11_admits(&self.arch, k) && (w_ptr as usize) % 4 == 0
    }

    /// Whether `acts` still describes the Q8_1 scratch contents (no Q8_1 writer ran since).
    pub fn q8_mmq_acts_current(&self, acts: &Q8MmqActivations) -> bool {
        acts.generation == self.scratch.q8_1_mmq_x_generation
            && self
                .scratch
                .q8_1_mmq_x_scratch
                .as_ref()
                .is_some_and(|b| b.as_ptr() == acts.xq)
    }

    /// Quantize F32 `x [n, k]` (row-major) into the Q8_1 scratch (`quantize_q8_1_mmq_ds4`).
    pub fn q8_mmq_quantize_x(
        &mut self,
        x: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<Q8MmqActivations> {
        self.bind_thread()?;
        if !matches!(self.arch.as_str(), "gfx1100" | "gfx1151") || k == 0 || k % 128 != 0 || n == 0
        {
            return Err(HipError::new(
                1,
                &format!(
                    "q8_mmq_quantize_x: needs gfx11, k % 128 == 0, n > 0 (arch {}, k {k}, n {n})",
                    self.arch
                ),
            ));
        }
        if x.dtype != DType::F32 || x.buf.size() < n * k * 4 {
            return Err(HipError::new(
                1,
                "q8_mmq_quantize_x: x must be F32 with at least n*k elements",
            ));
        }
        let xq = self.ensure_q8_1_mmq_x(x, n, k)?;
        Ok(Q8MmqActivations {
            xq,
            n,
            k,
            generation: self.scratch.q8_1_mmq_x_generation,
        })
    }

    /// Tiled GEMM on already-quantized activations, tile chosen by [`q8_mmq_pick_tile`].
    /// `Y[n, m] = X[n, k] @ A_q8[m, k]^T`.
    pub fn gemm_q8_0_mmq_gfx11_gemm(
        &mut self,
        w_q8: &GpuTensor,
        acts: &Q8MmqActivations,
        y: &GpuTensor,
        m: usize,
    ) -> HipResult<()> {
        let tile = q8_mmq_pick_tile(m, acts.n);
        self.gemm_q8_0_mmq_gfx11_gemm_tile(w_q8, acts, y, m, tile)
    }

    /// As [`Self::gemm_q8_0_mmq_gfx11_gemm`] with the BF16 epilogue: every output is rounded
    /// RN-even to BF16 before its store (`q8g11_bf16_rn`, the operation of
    /// `bf16_round_trip_f32_strided`; non-finite values pass through). Bit-identical to the plain
    /// GEMM followed by that round kernel.
    pub fn gemm_q8_0_mmq_gfx11_gemm_bf16(
        &mut self,
        w_q8: &GpuTensor,
        acts: &Q8MmqActivations,
        y: &GpuTensor,
        m: usize,
    ) -> HipResult<()> {
        let tile = q8_mmq_pick_tile(m, acts.n);
        self.gemm_q8_0_mmq_gfx11_gemm_tile_bf16(w_q8, acts, y, m, tile)
    }

    /// As [`Self::gemm_q8_0_mmq_gfx11_gemm`] with an explicit tile (bench / tests).
    pub fn gemm_q8_0_mmq_gfx11_gemm_tile(
        &mut self,
        w_q8: &GpuTensor,
        acts: &Q8MmqActivations,
        y: &GpuTensor,
        m: usize,
        tile: Q8MmqTile,
    ) -> HipResult<()> {
        self.gemm_q8_0_mmq_gfx11_launch(w_q8, acts, y, m, tile, false)
    }

    /// As [`Self::gemm_q8_0_mmq_gfx11_gemm_bf16`] with an explicit tile (bench / tests).
    pub fn gemm_q8_0_mmq_gfx11_gemm_tile_bf16(
        &mut self,
        w_q8: &GpuTensor,
        acts: &Q8MmqActivations,
        y: &GpuTensor,
        m: usize,
        tile: Q8MmqTile,
    ) -> HipResult<()> {
        self.gemm_q8_0_mmq_gfx11_launch(w_q8, acts, y, m, tile, true)
    }

    fn gemm_q8_0_mmq_gfx11_launch(
        &mut self,
        w_q8: &GpuTensor,
        acts: &Q8MmqActivations,
        y: &GpuTensor,
        m: usize,
        tile: Q8MmqTile,
        bf16: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (n, k) = (acts.n, acts.k);
        if !self.q8_mmq_acts_current(acts) {
            return Err(HipError::new(
                1,
                "gemm_q8_0_mmq_gfx11_gemm: activations were overwritten by a later Q8_1 quantizer",
            ));
        }
        if !matches!(self.arch.as_str(), "gfx1100" | "gfx1151")
            || m == 0
            || n == 0
            || k % 128 != 0
            || (w_q8.buf.as_ptr() as usize) % 4 != 0
        {
            return Err(HipError::new(
                1,
                &format!(
                    "gemm_q8_0_mmq_gfx11_gemm: needs gfx11, k % 128 == 0, m,n > 0, 4-byte aligned weights (arch {}, m {m}, k {k}, n {n})",
                    self.arch
                ),
            ));
        }
        if w_q8.buf.size() < q8_bytes(m, k) || y.dtype != DType::F32 || y.buf.size() < n * m * 4 {
            return Err(HipError::new(
                1,
                "gemm_q8_0_mmq_gfx11_gemm: weight bytes < m*k/32*34 or y is not F32 [n, m]",
            ));
        }
        let (bm, bn) = tile.dims();
        let grid_y = n.div_ceil(bn);
        if grid_y > 65535 {
            return Err(HipError::new(
                1,
                "gemm_q8_0_mmq_gfx11_gemm: n too large for the launch grid",
            ));
        }
        let symbol = tile.symbol(bf16);
        self.ensure_kernel(MODULE, kernels::GEMM_Q8_0_MMQ_I8_GFX11_SRC, symbol)?;

        let wp = w_q8.buf.as_ptr();
        let xp = acts.xq;
        let yp = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];
        let grid = [m.div_ceil(bm) as u32, grid_y as u32, 1];
        let bytes = q8_bytes(m, k) + n * (k / 128) * 144 + n * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", symbol, bytes);
        let result = self.launch_maybe_blob(
            symbol,
            grid,
            [(2 * bm) as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Producer + GEMM: `Y[n, m] = X[n, k] @ A_q8[m, k]^T` with the activation quantized to Q8_1.
    /// Same shape contract as [`Gpu::gemm_q8_0_mmq_4w_gfx1151`], bit-identical result.
    pub fn gemm_q8_0_mmq_gfx11_batched(
        &mut self,
        w_q8: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        let acts = self.q8_mmq_quantize_x(x, n, k)?;
        self.gemm_q8_0_mmq_gfx11_gemm(w_q8, &acts, y, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_table_matches_the_receipt_regimes() {
        // 512-row chunk (the XTX prefill chunk): M=5120 fills 320 CTAs at 64x128, the rest take 64x64.
        assert_eq!(q8_mmq_pick_tile(5120, 512), Q8MmqTile::T64x128);
        assert_eq!(q8_mmq_pick_tile(2560, 512), Q8MmqTile::T64x64);
        assert_eq!(q8_mmq_pick_tile(576, 512), Q8MmqTile::T64x64);
        assert_eq!(q8_mmq_pick_tile(1088, 512), Q8MmqTile::T64x64);
        // wide calls
        assert_eq!(q8_mmq_pick_tile(5120, 4096), Q8MmqTile::T128x128);
        assert_eq!(q8_mmq_pick_tile(576, 4096), Q8MmqTile::T64x128);
        // tiny M never takes a wide tile
        assert_eq!(q8_mmq_pick_tile(40, 4096), Q8MmqTile::T64x64);
        // every row count picks a valid tile
        for n in [1usize, 63, 65, 511, 513, 4097] {
            for m in [1usize, 40, 576, 5120] {
                let (bm, bn) = q8_mmq_pick_tile(m, n).dims();
                assert!(bm > 0 && bn > 0);
            }
        }
    }
}
