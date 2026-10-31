// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! W8A8 Q8_0 prefill GEMM for gfx12 (gfx1200/gfx1201).
//!
//! `Y[n, m] = sum_b (wscale[m, b] * xscale[n, b]) * sum_{i<32} W8[m, 32b+i] * X8[n, 32b+i]`
//! on `v_wmma_i32_16x16x16_iu8` with F32 output (`kernels/src/gemm_q8_0_mmq_i8.gfx12.hip`).
//! Weights are the artifact's Q8_0 (34-byte blocks) read in place; activations are
//! quantized per row per 32-K block (`quantize_x_q8_blk32`: `d = RN(amax/127)`,
//! `q = clamp(rint(x/d), ±127)`) into an int8 `[n, k]` plane plus F32 scales
//! `[k/32, n]`. This replaces the F16-activation `gemm_q8_0_wmma_gfx12` path
//! (`gemm_q8_0_batched_chunked`) for attention-class projections, so it is a
//! quality-changing route: the caller gates it per site (KLD).
//!
//! * [`Gpu::gemm_q8_0_a8_batched`] = producer + GEMM, the shape of
//!   `gemm_q8_0_batched_chunked`.
//! * [`Gpu::q8_a8_quantize_x`] + [`Gpu::gemm_q8_0_a8_gemm`] split the two so one
//!   quantization serves every projection that shares an activation
//!   (RDA q/k/v/g, ...).
//!
//! Admission: exact RDNA4, `k % 32 == 0`, F32 `x`/`y`. No environment variable.

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

const MODULE: &str = "gemm_q8_0_mmq_i8_gfx12";
const QUANT_SYMBOL: &str = "quantize_x_q8_blk32";

/// Output tile of the gfx12 W8A8 kernels (`BM x BN` = weight rows x tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q8A8Tile {
    T128x128,
    T128x64,
    T64x64,
}

impl Q8A8Tile {
    pub const ALL: [Q8A8Tile; 3] = [Q8A8Tile::T128x128, Q8A8Tile::T128x64, Q8A8Tile::T64x64];

    pub fn label(self) -> &'static str {
        match self {
            Q8A8Tile::T128x128 => "128x128",
            Q8A8Tile::T128x64 => "128x64",
            Q8A8Tile::T64x64 => "64x64",
        }
    }

    /// `(BM, BN)`.
    pub fn dims(self) -> (usize, usize) {
        match self {
            Q8A8Tile::T128x128 => (128, 128),
            Q8A8Tile::T128x64 => (128, 64),
            Q8A8Tile::T64x64 => (64, 64),
        }
    }

    fn symbol(self, fast: bool) -> &'static str {
        match (self, fast) {
            (Q8A8Tile::T128x128, true) => "gemm_q8_0_mmq_i8_t128x128",
            (Q8A8Tile::T128x128, false) => "gemm_q8_0_mmq_i8_t128x128_slow",
            (Q8A8Tile::T128x64, true) => "gemm_q8_0_mmq_i8_t128x64",
            (Q8A8Tile::T128x64, false) => "gemm_q8_0_mmq_i8_t128x64_slow",
            (Q8A8Tile::T64x64, true) => "gemm_q8_0_mmq_i8_t64x64",
            (Q8A8Tile::T64x64, false) => "gemm_q8_0_mmq_i8_t64x64_slow",
        }
    }
}

/// Quantized activations in the Gpu's W8A8 scratch. The pointers stay valid
/// until the next [`Gpu::q8_a8_quantize_x`] (which overwrites them) or a
/// scratch growth.
#[derive(Clone, Copy, Debug)]
pub struct Q8A8Activations {
    xq: *mut c_void,
    xs: *mut c_void,
    n: usize,
    k: usize,
}

impl Q8A8Activations {
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

impl Gpu {
    /// Whether the W8A8 Q8_0 route can run for this device and `k`.
    pub fn q8_a8_supported(&self, k: usize) -> bool {
        self.arch_caps.is_rdna4() && k > 0 && k % 32 == 0
    }

    /// Tile for a `(m, n)` call: the biggest tile whose grid still holds at
    /// least `Q8A8_FILL_BLOCKS` CTAs (two resident CTAs on each of 64 CUs);
    /// below that even the 64x64 tile is the best of the three. Measured on
    /// R9700: 128x128 wins every wide shape, 64x64 only the narrow
    /// `M = 576, N = 512` class.
    pub fn q8_a8_pick_tile(&self, m: usize, n: usize) -> Q8A8Tile {
        const Q8A8_FILL_BLOCKS: usize = 128;
        for tile in [Q8A8Tile::T128x128, Q8A8Tile::T128x64] {
            let (bm, bn) = tile.dims();
            if m.div_ceil(bm) * n.div_ceil(bn) >= Q8A8_FILL_BLOCKS {
                return tile;
            }
        }
        Q8A8Tile::T64x64
    }

    /// Quantize F32 `x [n, k]` (row-major) into the W8A8 scratch.
    pub fn q8_a8_quantize_x(
        &mut self,
        x: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<Q8A8Activations> {
        self.bind_thread()?;
        if !self.q8_a8_supported(k) || n == 0 {
            return Err(HipError::new(
                1,
                &format!(
                    "q8_a8_quantize_x: needs gfx12, k % 32 == 0, n > 0 (arch {}, k {k}, n {n})",
                    self.arch
                ),
            ));
        }
        if x.dtype != DType::F32 || x.buf.size() < n * k * 4 {
            return Err(HipError::new(
                1,
                "q8_a8_quantize_x: x must be F32 with at least n*k elements",
            ));
        }
        self.ensure_kernel(MODULE, kernels::GEMM_Q8_0_MMQ_I8_GFX12_SRC, QUANT_SYMBOL)?;
        let (_, needed) = crate::scratch::q8_a8_x_layout(k, n);
        if crate::scratch::scratch_will_grow(
            self.scratch.q8_a8_x_scratch_bytes,
            self.scratch.q8_a8_x_scratch.is_some(),
            needed,
        ) {
            self.invalidate_for_scratch_growth();
        }
        let (xq, xs) = self.scratch.reserve_q8_a8_x(&self.hip, k, n)?;

        let xp = x.buf.as_ptr();
        let k_val = k as i32;
        let n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &xs as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];
        let groups = (n * (k / 4)).div_ceil(256);
        let grid = [
            u32::try_from(groups).map_err(|_| {
                HipError::new(1, "q8_a8_quantize_x: n*k too large for a 1-D launch")
            })?,
            1,
            1,
        ];
        let timer = crate::profile::begin_timer(&self.hip, "quantize", QUANT_SYMBOL, n * k * 5);
        let result =
            self.launch_maybe_blob(QUANT_SYMBOL, grid, [256, 1, 1], 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xq);
                b.push_ptr(xs);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;
        Ok(Q8A8Activations { xq, xs, n, k })
    }

    /// W8A8 GEMM on already-quantized activations, tile chosen by
    /// [`Self::q8_a8_pick_tile`]. `Y[n, m] = X[n, k] @ A_q8[m, k]^T`.
    pub fn gemm_q8_0_a8_gemm(
        &mut self,
        w_q8: &GpuTensor,
        acts: &Q8A8Activations,
        y: &GpuTensor,
        m: usize,
    ) -> HipResult<()> {
        let tile = self.q8_a8_pick_tile(m, acts.n);
        self.gemm_q8_0_a8_gemm_tile(w_q8, acts, y, m, tile)
    }

    /// As [`Self::gemm_q8_0_a8_gemm`] with an explicit tile (bench / tests).
    pub fn gemm_q8_0_a8_gemm_tile(
        &mut self,
        w_q8: &GpuTensor,
        acts: &Q8A8Activations,
        y: &GpuTensor,
        m: usize,
        tile: Q8A8Tile,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (n, k) = (acts.n, acts.k);
        if !self.q8_a8_supported(k) || m == 0 || n == 0 {
            return Err(HipError::new(
                1,
                &format!(
                    "gemm_q8_0_a8_gemm: needs gfx12, k % 32 == 0, m,n > 0 (arch {}, m {m}, k {k}, n {n})",
                    self.arch
                ),
            ));
        }
        if w_q8.buf.size() < q8_bytes(m, k) || y.dtype != DType::F32 || y.buf.size() < n * m * 4 {
            return Err(HipError::new(
                1,
                "gemm_q8_0_a8_gemm: weight bytes < m*k/32*34 or y is not F32 [n, m]",
            ));
        }
        // FAST kernels read the 68-byte K64 slab with 4-byte loads: they need
        // an even number of Q8_0 blocks per row and a 4-byte aligned base.
        let fast = k % 64 == 0 && (w_q8.buf.as_ptr() as usize) % 4 == 0;
        let symbol = tile.symbol(fast);
        self.ensure_kernel(MODULE, kernels::GEMM_Q8_0_MMQ_I8_GFX12_SRC, symbol)?;

        let wp = w_q8.buf.as_ptr();
        let xq = acts.xq;
        let xs = acts.xs;
        let yp = y.buf.as_ptr();
        let m_val = m as i32;
        let k_val = k as i32;
        let n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &xs as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
        ];
        let (bm, bn) = tile.dims();
        let grid = [m.div_ceil(bm) as u32, n.div_ceil(bn) as u32, 1];
        let bytes = q8_bytes(m, k) + n * k + n * (k / 32) * 4 + n * m * 4;
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
                b.push_ptr(xq);
                b.push_ptr(xs);
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

    /// Producer + GEMM: `Y[n, m] = X[n, k] @ A_q8[m, k]^T` with the activation
    /// quantized to int8 per row per 32-K block. Same shape contract as
    /// [`Gpu::gemm_q8_0_batched_chunked`] plus F32 `x`.
    pub fn gemm_q8_0_a8_batched(
        &mut self,
        w_q8: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        let acts = self.q8_a8_quantize_x(x, n, k)?;
        self.gemm_q8_0_a8_gemm(w_q8, &acts, y, m)
    }
}
