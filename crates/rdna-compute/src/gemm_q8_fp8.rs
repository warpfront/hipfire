// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! W8-as-E4M3 x A-E4M3 Q8_0 prefill GEMM for gfx12 (gfx1200/gfx1201): the FP8 twin of
//! [`crate::gemm_q8_a8`].
//!
//! `Y[n, m] = sum_b (wscale[m, b] * xscale[n, b]) * sum_{i<32} e4m3(W8[m, 32b+i]) * X8[n, 32b+i]`
//! on `v_wmma_f32_16x16x16_fp8_fp8` with F32 output
//! (`kernels/src/gemm_q8_0_mmq_fp8.gfx12.hip`). Weights are the artifact's Q8_0
//! (34-byte blocks) read in place and converted int8 -> E4M3 (RNE) once per element at LDS
//! staging, which is lossy for `|q| > 16` (3 mantissa bits): that is this tier's contract.
//! Activations are quantized per row per 32-K block by `quantize_x_fp8_blk32`
//! (`d = RN(amax / 448)`, `code = e4m3_rne_sat(x / d)`) into an E4M3 `[n, k]` plane plus F32
//! scales `[k/32, n]` in the same scratch sidecar layout as the int8 route, so the two
//! routes share (and overwrite) one scratch: quantized activations of one route are invalid
//! after the other route's quantize call.
//!
//! * [`Gpu::gemm_q8_0_fp8_batched`] = producer + GEMM, the shape of
//!   [`Gpu::gemm_q8_0_a8_batched`].
//! * [`Gpu::q8_fp8_quantize_x`] + [`Gpu::gemm_q8_0_fp8_gemm`] split the two so one
//!   quantization serves every projection that shares an activation.
//!
//! Admission: exact RDNA4, `k % 32 == 0`, F32 `x`/`y`. No environment variable. Quality
//! changing: the caller gates it per site.

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::gemm_q8_a8::Q8A8Tile;
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

const MODULE: &str = "gemm_q8_0_mmq_fp8_gfx12";
const QUANT_SYMBOL: &str = "quantize_x_fp8_blk32";

fn tile_symbol(tile: Q8A8Tile, fast: bool) -> &'static str {
    match (tile, fast) {
        (Q8A8Tile::T128x128, true) => "gemm_q8_0_mmq_fp8_t128x128",
        (Q8A8Tile::T128x128, false) => "gemm_q8_0_mmq_fp8_t128x128_slow",
        (Q8A8Tile::T128x64, true) => "gemm_q8_0_mmq_fp8_t128x64",
        (Q8A8Tile::T128x64, false) => "gemm_q8_0_mmq_fp8_t128x64_slow",
        (Q8A8Tile::T64x64, true) => "gemm_q8_0_mmq_fp8_t64x64",
        (Q8A8Tile::T64x64, false) => "gemm_q8_0_mmq_fp8_t64x64_slow",
    }
}

/// E4M3 activations in the Gpu's W8A8 scratch (shared with the int8 route). The pointers
/// stay valid until the next [`Gpu::q8_fp8_quantize_x`] / [`Gpu::q8_a8_quantize_x`]
/// (which overwrite them) or a scratch growth.
#[derive(Clone, Copy, Debug)]
pub struct Q8Fp8Activations {
    xq: *mut c_void,
    xs: *mut c_void,
    n: usize,
    k: usize,
}

impl Q8Fp8Activations {
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
    /// Whether the FP8 Q8_0 route can run for this device and `k`.
    pub fn q8_fp8_supported(&self, k: usize) -> bool {
        self.arch_caps.is_rdna4() && k > 0 && k % 32 == 0
    }

    /// Tile for a `(m, n)` call: the int8 route's picker (same tile set and CTA geometry).
    pub fn q8_fp8_pick_tile(&self, m: usize, n: usize) -> Q8A8Tile {
        self.q8_a8_pick_tile(m, n)
    }

    /// Quantize F32 `x [n, k]` (row-major) into the shared W8A8 scratch as E4M3 + scales.
    pub fn q8_fp8_quantize_x(
        &mut self,
        x: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<Q8Fp8Activations> {
        self.bind_thread()?;
        if !self.q8_fp8_supported(k) || n == 0 {
            return Err(HipError::new(
                1,
                &format!(
                    "q8_fp8_quantize_x: needs gfx12, k % 32 == 0, n > 0 (arch {}, k {k}, n {n})",
                    self.arch
                ),
            ));
        }
        if x.dtype != DType::F32 || x.buf.size() < n * k * 4 {
            return Err(HipError::new(
                1,
                "q8_fp8_quantize_x: x must be F32 with at least n*k elements",
            ));
        }
        self.ensure_kernel(MODULE, kernels::GEMM_Q8_0_MMQ_FP8_GFX12_SRC, QUANT_SYMBOL)?;
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
                HipError::new(1, "q8_fp8_quantize_x: n*k too large for a 1-D launch")
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
        Ok(Q8Fp8Activations { xq, xs, n, k })
    }

    /// FP8 GEMM on already-quantized activations, tile chosen by
    /// [`Self::q8_fp8_pick_tile`]. `Y[n, m] = X[n, k] @ A_q8[m, k]^T`.
    pub fn gemm_q8_0_fp8_gemm(
        &mut self,
        w_q8: &GpuTensor,
        acts: &Q8Fp8Activations,
        y: &GpuTensor,
        m: usize,
    ) -> HipResult<()> {
        let tile = self.q8_fp8_pick_tile(m, acts.n);
        self.gemm_q8_0_fp8_gemm_tile(w_q8, acts, y, m, tile)
    }

    /// As [`Self::gemm_q8_0_fp8_gemm`] with an explicit tile (bench / tests).
    pub fn gemm_q8_0_fp8_gemm_tile(
        &mut self,
        w_q8: &GpuTensor,
        acts: &Q8Fp8Activations,
        y: &GpuTensor,
        m: usize,
        tile: Q8A8Tile,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (n, k) = (acts.n, acts.k);
        if !self.q8_fp8_supported(k) || m == 0 || n == 0 {
            return Err(HipError::new(
                1,
                &format!(
                    "gemm_q8_0_fp8_gemm: needs gfx12, k % 32 == 0, m,n > 0 (arch {}, m {m}, k {k}, n {n})",
                    self.arch
                ),
            ));
        }
        if w_q8.buf.size() < q8_bytes(m, k) || y.dtype != DType::F32 || y.buf.size() < n * m * 4 {
            return Err(HipError::new(
                1,
                "gemm_q8_0_fp8_gemm: weight bytes < m*k/32*34 or y is not F32 [n, m]",
            ));
        }
        // FAST kernels read the 68-byte K64 slab with 4-byte loads: they need
        // an even number of Q8_0 blocks per row and a 4-byte aligned base.
        let fast = k % 64 == 0 && (w_q8.buf.as_ptr() as usize) % 4 == 0;
        let symbol = tile_symbol(tile, fast);
        self.ensure_kernel(MODULE, kernels::GEMM_Q8_0_MMQ_FP8_GFX12_SRC, symbol)?;

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
    /// quantized to E4M3 per row per 32-K block and the int8 weights read as E4M3. Same shape
    /// contract as [`Gpu::gemm_q8_0_a8_batched`].
    pub fn gemm_q8_0_fp8_batched(
        &mut self,
        w_q8: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        let acts = self.q8_fp8_quantize_x(x, n, k)?;
        self.gemm_q8_0_fp8_gemm(w_q8, &acts, y, m)
    }
}
