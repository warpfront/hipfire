// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Generic per-key-decay gated delta net (serial T>=1). Owner: slice R.
//!
//! * `gated_delta_net_vector_serial_f32`: gated delta rule with per-head or
//!   per-key-channel log decay; N>=1 tokens in ONE launch, FP32 `[H,V,K]` state
//!   kept in registers across tokens (no tape, no per-token launches).
//! * `causal_conv_depthwise_silu_rows_f32`: depthwise causal short conv over N
//!   rows with explicit raw history, optional SiLU, optional BF16 rounding.
//! * `gated_rmsnorm_sigmoid_rows_f32`: per-head RMSNorm times sigmoid(gate).
//! * `raon_rda_prepare_f32`: checkpoint-specific RDA gate/normalization prep.
//!
//! Operand planes are FP32; "BF16 rounding" means the stored FP32 value is
//! rounded to the nearest-even BF16 value (the planes stay FP32).

use crate::dispatch::{Gpu, GpuTensor};
use crate::profile;
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

pub const GATED_DELTA_NET_VECTOR_SERIAL_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_net_vector_serial.hip");
pub const CAUSAL_CONV_DEPTHWISE_ROWS_SRC: &str =
    include_str!("../../../kernels/src/causal_conv_depthwise_rows.hip");
pub const GATED_RMSNORM_SIGMOID_ROWS_SRC: &str =
    include_str!("../../../kernels/src/gated_rmsnorm_sigmoid_rows.hip");
pub const RAON_RDA_PREPARE_SRC: &str = include_str!("../../../kernels/src/raon_rda_prepare.hip");

/// Head dimension (K == V) the serial kernel is specialized for.
pub const GATED_DELTA_VECTOR_HEAD_DIM: usize = 128;
/// Largest conv kernel size supported by the depthwise rows kernel.
pub const CAUSAL_CONV_ROWS_KMAX: usize = 8;
const CONV_ROWS_PER_CHUNK: usize = 64;

fn refuse(msg: String) -> HipError {
    HipError::new(0, &msg)
}

/// Check that `t` (F32) holds at least `(rows-1)*row_stride + extent` elements.
fn need(t: &GpuTensor, what: &str, rows: usize, row_stride: usize, extent: usize) -> HipResult<()> {
    if rows == 0 {
        return Ok(());
    }
    let elems = (rows - 1)
        .checked_mul(row_stride)
        .and_then(|x| x.checked_add(extent))
        .ok_or_else(|| refuse(format!("{what}: extent overflow")))?;
    if t.buf.size() < elems * 4 {
        return Err(refuse(format!(
            "{what}: buffer holds {} f32 elements, launch needs {elems}",
            t.buf.size() / 4
        )));
    }
    Ok(())
}

fn i32_of(v: usize, what: &str) -> HipResult<i32> {
    i32::try_from(v).map_err(|_| refuse(format!("{what}: {v} exceeds i32")))
}

/// Geometry + operands of one `gated_delta_net_vector_serial_f32` launch.
/// All strides are in f32 elements. Query/key are `[N, Hqk, K]`, value/output
/// `[N, H, V]`, beta `[N, H]`; log decay is `[N, H, K]` (per key channel) or
/// `[N, H]` (per head). State is FP32 value-major (key contiguous):
/// element `(slot, h, v, k)` at `slot*state_slot_stride + h*state_head_stride + v*state_row_stride + k`.
pub struct GatedDeltaVectorLaunch<'a> {
    pub query: &'a GpuTensor,
    pub key: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub log_decay: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub output: &'a GpuTensor,
    /// Optional device `i32` slot index; overrides `slot` when present.
    pub slot_idx: Option<&'a GpuTensor>,
    /// Optional device `i32` token count (<= `n_tokens`); graph-capture safe.
    pub n_dev: Option<&'a GpuTensor>,
    pub n_tokens: usize,
    pub n_heads: usize,
    /// Value heads per q/k head (1 = no repetition).
    pub qk_group: usize,
    pub head_dim: usize,
    pub decay_per_key: bool,
    /// Round the written output to BF16 (value stays FP32).
    pub round_out_bf16: bool,
    pub slot: usize,
    pub q_row_stride: usize,
    pub k_row_stride: usize,
    pub v_row_stride: usize,
    pub decay_row_stride: usize,
    pub beta_row_stride: usize,
    pub out_row_stride: usize,
    pub state_head_stride: usize,
    pub state_row_stride: usize,
    pub state_slot_stride: usize,
}

impl Gpu {
    /// One launch for `n_tokens >= 1` (or device count) tokens. Kernel args, in
    /// order: query,key,value,log_decay,beta,state,output (ptr), slot_idx,n_dev
    /// (ptr, nullable), n_tokens,n_heads,qk_group,decay_per_key,round_out_bf16,
    /// slot,q_row_stride,k_row_stride,v_row_stride,decay_row_stride,
    /// beta_row_stride,out_row_stride,state_head_stride,state_row_stride (i32),
    /// state_slot_stride (u64).
    pub fn gated_delta_net_vector_serial_f32(
        &mut self,
        p: &GatedDeltaVectorLaunch<'_>,
    ) -> HipResult<()> {
        let d = GATED_DELTA_VECTOR_HEAD_DIM;
        if p.head_dim != d {
            return Err(refuse(format!(
                "gated_delta_net_vector_serial_f32: head_dim {} unsupported (specialized for {d})",
                p.head_dim
            )));
        }
        if p.n_heads == 0 || p.qk_group == 0 || p.n_heads % p.qk_group != 0 {
            return Err(refuse(format!(
                "gated_delta_net_vector_serial_f32: n_heads {} not divisible by qk_group {}",
                p.n_heads, p.qk_group
            )));
        }
        if p.state_row_stride < d || p.state_head_stride < d * p.state_row_stride {
            return Err(refuse(
                "gated_delta_net_vector_serial_f32: state strides overlap rows/heads".into(),
            ));
        }
        if p.n_tokens == 0 && p.n_dev.is_none() {
            return Ok(());
        }
        let n = p.n_tokens;
        let hq = p.n_heads / p.qk_group;
        need(p.query, "query", n, p.q_row_stride, hq * d)?;
        need(p.key, "key", n, p.k_row_stride, hq * d)?;
        need(p.value, "value", n, p.v_row_stride, p.n_heads * d)?;
        let decay_extent = if p.decay_per_key {
            p.n_heads * d
        } else {
            p.n_heads
        };
        need(
            p.log_decay,
            "log_decay",
            n,
            p.decay_row_stride,
            decay_extent,
        )?;
        need(p.beta, "beta", n, p.beta_row_stride, p.n_heads)?;
        need(p.output, "output", n, p.out_row_stride, p.n_heads * d)?;
        let slots_needed = if p.slot_idx.is_some() { 1 } else { p.slot + 1 };
        // Only the baked slot can be bounds-checked on the host.
        need(
            p.state,
            "state",
            slots_needed,
            p.state_slot_stride,
            (p.n_heads - 1) * p.state_head_stride + (d - 1) * p.state_row_stride + d,
        )?;

        self.bind_thread()?;
        // gfx1100 / gfx1201 PM twin (`gated_delta_vector_pm`): same kernargs, grid and block.
        let twin = self.gated_delta_vector_pm_rda_admitted(p.head_dim);
        let func = if let Some(twin) = twin {
            self.ensure_embedded_kernel(twin.module, twin.image, twin.rda_symbol)?;
            twin.rda_symbol
        } else {
            self.ensure_kernel(
                "gated_delta_net_vector_serial",
                GATED_DELTA_NET_VECTOR_SERIAL_SRC,
                "gated_delta_net_vector_serial_f32",
            )?;
            "gated_delta_net_vector_serial_f32"
        };
        let qp = p.query.buf.as_ptr();
        let kp = p.key.buf.as_ptr();
        let vp = p.value.buf.as_ptr();
        let lp = p.log_decay.buf.as_ptr();
        let bp = p.beta.buf.as_ptr();
        let sp = p.state.buf.as_ptr();
        let op = p.output.buf.as_ptr();
        let sip: *const c_void = p.slot_idx.map_or(std::ptr::null(), |t| t.buf.as_ptr());
        let ndp: *const c_void = p.n_dev.map_or(std::ptr::null(), |t| t.buf.as_ptr());
        let n_tokens = i32_of(p.n_tokens, "n_tokens")?;
        let n_heads = i32_of(p.n_heads, "n_heads")?;
        let qk_group = i32_of(p.qk_group, "qk_group")?;
        let per_key = p.decay_per_key as i32;
        let round = p.round_out_bf16 as i32;
        let slot = i32_of(p.slot, "slot")?;
        let qrs = i32_of(p.q_row_stride, "q_row_stride")?;
        let krs = i32_of(p.k_row_stride, "k_row_stride")?;
        let vrs = i32_of(p.v_row_stride, "v_row_stride")?;
        let drs = i32_of(p.decay_row_stride, "decay_row_stride")?;
        let brs = i32_of(p.beta_row_stride, "beta_row_stride")?;
        let ors = i32_of(p.out_row_stride, "out_row_stride")?;
        let shs = i32_of(p.state_head_stride, "state_head_stride")?;
        let srs = i32_of(p.state_row_stride, "state_row_stride")?;
        let sss = p.state_slot_stride as u64;
        let mut params: Vec<*mut c_void> = vec![
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &lp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &sip as *const _ as *mut c_void,
            &ndp as *const _ as *mut c_void,
            &n_tokens as *const _ as *mut c_void,
            &n_heads as *const _ as *mut c_void,
            &qk_group as *const _ as *mut c_void,
            &per_key as *const _ as *mut c_void,
            &round as *const _ as *mut c_void,
            &slot as *const _ as *mut c_void,
            &qrs as *const _ as *mut c_void,
            &krs as *const _ as *mut c_void,
            &vrs as *const _ as *mut c_void,
            &drs as *const _ as *mut c_void,
            &brs as *const _ as *mut c_void,
            &ors as *const _ as *mut c_void,
            &shs as *const _ as *mut c_void,
            &srs as *const _ as *mut c_void,
            &sss as *const _ as *mut c_void,
        ];
        // GDV_TV = 8 value rows per workgroup (same grid for the PM twin).
        let grid = [p.n_heads as u32, (d / 8) as u32, 1];
        let bytes = p.n_heads * d * d * 4 * 2;
        let timer = profile::begin_timer(
            &self.hip,
            "gated_delta",
            "gated_delta_net_vector_serial_f32",
            bytes,
        );
        let result = self.launch_maybe_blob(func, grid, [32, 1, 1], 0, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_ptr(qp);
            b.push_ptr(kp);
            b.push_ptr(vp);
            b.push_ptr(lp);
            b.push_ptr(bp);
            b.push_ptr(sp);
            b.push_ptr(op);
            b.push_ptr(sip);
            b.push_ptr(ndp);
            b.push_i32(n_tokens);
            b.push_i32(n_heads);
            b.push_i32(qk_group);
            b.push_i32(per_key);
            b.push_i32(round);
            b.push_i32(slot);
            b.push_i32(qrs);
            b.push_i32(krs);
            b.push_i32(vrs);
            b.push_i32(drs);
            b.push_i32(brs);
            b.push_i32(ors);
            b.push_i32(shs);
            b.push_i32(srs);
            b.push_u64(sss);
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

/// Geometry of one `causal_conv_depthwise_silu_rows_f32` launch (f32 elements).
/// Weights are `[channel, tap]` (`w_ch_stride`, `w_tap_stride`); history is
/// `[K-1, channel]` oldest -> newest with `hist_tap_stride` between taps.
#[derive(Clone, Copy, Debug)]
pub struct CausalConvRowsGeom {
    pub n_rows: usize,
    pub n_channels: usize,
    pub ksize: usize,
    pub in_row_stride: usize,
    pub out_row_stride: usize,
    pub w_ch_stride: usize,
    pub w_tap_stride: usize,
    pub hist_tap_stride: usize,
    pub apply_silu: bool,
    pub round_bf16: bool,
}

impl Gpu {
    /// Depthwise causal conv over `n_rows` rows with in-kernel history update.
    /// `hist_in` and `hist_out` may be the same tensor (in-place update; the
    /// launcher then uses one row chunk) or distinct. Kernel args: out,in,
    /// weight,hist_in,hist_out (ptr); n_rows,n_channels,ksize,rows_per_chunk,
    /// in_row_stride,out_row_stride,w_ch_stride,w_tap_stride,hist_tap_stride,
    /// apply_silu,round_bf16 (i32).
    pub fn causal_conv_depthwise_silu_rows_f32(
        &mut self,
        out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        hist_in: &GpuTensor,
        hist_out: &GpuTensor,
        g: &CausalConvRowsGeom,
    ) -> HipResult<()> {
        if g.ksize < 2 || g.ksize > CAUSAL_CONV_ROWS_KMAX {
            return Err(refuse(format!(
                "causal_conv_depthwise_silu_rows_f32: ksize {} outside 2..={CAUSAL_CONV_ROWS_KMAX}",
                g.ksize
            )));
        }
        if g.n_rows == 0 {
            return Ok(());
        }
        let km1 = g.ksize - 1;
        need(input, "conv input", g.n_rows, g.in_row_stride, g.n_channels)?;
        need(out, "conv output", g.n_rows, g.out_row_stride, g.n_channels)?;
        need(
            weight,
            "conv weight",
            g.n_channels,
            g.w_ch_stride,
            (g.ksize - 1) * g.w_tap_stride + 1,
        )?;
        need(
            hist_in,
            "conv hist_in",
            km1,
            g.hist_tap_stride,
            g.n_channels,
        )?;
        need(
            hist_out,
            "conv hist_out",
            km1,
            g.hist_tap_stride,
            g.n_channels,
        )?;

        self.bind_thread()?;
        // gfx1100 / gfx1201 PM twin (`gated_delta_vector_pm`): same kernargs and block; row slice 0
        // alone touches the history, so it is correct in place and not (gfx1201 admits in place only).
        let in_place = hist_in.buf.as_ptr() == hist_out.buf.as_ptr();
        let twin =
            self.gated_delta_vector_pm_conv_admitted(crate::gated_delta_vector_pm::ConvShape {
                ksize: g.ksize,
                in_place,
            });
        let func = if let Some(twin) = twin {
            self.ensure_embedded_kernel(twin.module, twin.image, twin.conv_symbol)?;
            twin.conv_symbol
        } else {
            self.ensure_kernel(
                "causal_conv_depthwise_rows",
                CAUSAL_CONV_DEPTHWISE_ROWS_SRC,
                "causal_conv_depthwise_silu_rows_f32",
            )?;
            "causal_conv_depthwise_silu_rows_f32"
        };
        let op = out.buf.as_ptr();
        let ip = input.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let hip_ = hist_in.buf.as_ptr();
        let hop = hist_out.buf.as_ptr();
        let rows_per_chunk = if in_place {
            g.n_rows
        } else {
            CONV_ROWS_PER_CHUNK
        };
        let chunks = (g.n_rows + rows_per_chunk - 1) / rows_per_chunk;
        let n_rows = i32_of(g.n_rows, "n_rows")?;
        let n_ch = i32_of(g.n_channels, "n_channels")?;
        let ks = i32_of(g.ksize, "ksize")?;
        let rpc = i32_of(rows_per_chunk, "rows_per_chunk")?;
        let irs = i32_of(g.in_row_stride, "in_row_stride")?;
        let ors = i32_of(g.out_row_stride, "out_row_stride")?;
        let wcs = i32_of(g.w_ch_stride, "w_ch_stride")?;
        let wts = i32_of(g.w_tap_stride, "w_tap_stride")?;
        let hts = i32_of(g.hist_tap_stride, "hist_tap_stride")?;
        let silu = g.apply_silu as i32;
        let round = g.round_bf16 as i32;
        let mut params: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &hip_ as *const _ as *mut c_void,
            &hop as *const _ as *mut c_void,
            &n_rows as *const _ as *mut c_void,
            &n_ch as *const _ as *mut c_void,
            &ks as *const _ as *mut c_void,
            &rpc as *const _ as *mut c_void,
            &irs as *const _ as *mut c_void,
            &ors as *const _ as *mut c_void,
            &wcs as *const _ as *mut c_void,
            &wts as *const _ as *mut c_void,
            &hts as *const _ as *mut c_void,
            &silu as *const _ as *mut c_void,
            &round as *const _ as *mut c_void,
        ];
        let block = 128u32;
        let grid = if twin.is_some() {
            // 32 rows per workgroup regardless of `rows_per_chunk`.
            crate::gated_delta_vector_pm::conv_grid(g.n_channels, g.n_rows)
        } else {
            [
                ((g.n_channels as u32) + block - 1) / block,
                chunks as u32,
                1,
            ]
        };
        let bytes = g.n_rows * g.n_channels * 4 * 2;
        let timer = profile::begin_timer(
            &self.hip,
            "conv",
            "causal_conv_depthwise_silu_rows_f32",
            bytes,
        );
        let result = self.launch_maybe_blob(func, grid, [block, 1, 1], 0, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_ptr(op);
            b.push_ptr(ip);
            b.push_ptr(wp);
            b.push_ptr(hip_);
            b.push_ptr(hop);
            for v in [n_rows, n_ch, ks, rpc, irs, ors, wcs, wts, hts, silu, round] {
                b.push_i32(v);
            }
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

/// Geometry of `gated_rmsnorm_sigmoid_rows_f32` (f32 elements).
#[derive(Clone, Copy, Debug)]
pub struct GatedRmsnormRowsGeom {
    pub n_rows: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub x_row_stride: usize,
    pub gate_row_stride: usize,
    pub out_row_stride: usize,
    pub eps: f32,
    pub round_bf16: bool,
}

impl Gpu {
    /// `out = bf16?( x * rsqrt(mean(x^2)+eps) * gamma * sigmoid(gate) )` per head.
    /// Kernel args: out,x,gate,gamma (ptr); head_dim,n_heads,x_row_stride,
    /// gate_row_stride,out_row_stride (i32); eps (f32); round_bf16 (i32).
    pub fn gated_rmsnorm_sigmoid_rows_f32(
        &mut self,
        out: &GpuTensor,
        x: &GpuTensor,
        gate: &GpuTensor,
        gamma: &GpuTensor,
        g: &GatedRmsnormRowsGeom,
    ) -> HipResult<()> {
        if g.head_dim == 0 || g.head_dim > 1024 || !g.head_dim.is_power_of_two() {
            return Err(refuse(format!(
                "gated_rmsnorm_sigmoid_rows_f32: head_dim {} must be a power of two <= 1024",
                g.head_dim
            )));
        }
        if g.n_rows == 0 {
            return Ok(());
        }
        let ext = g.n_heads * g.head_dim;
        need(x, "rmsnorm x", g.n_rows, g.x_row_stride, ext)?;
        need(gate, "rmsnorm gate", g.n_rows, g.gate_row_stride, ext)?;
        need(out, "rmsnorm out", g.n_rows, g.out_row_stride, ext)?;
        need(gamma, "rmsnorm gamma", 1, 0, g.head_dim)?;

        self.bind_thread()?;
        self.ensure_kernel(
            "gated_rmsnorm_sigmoid_rows",
            GATED_RMSNORM_SIGMOID_ROWS_SRC,
            "gated_rmsnorm_sigmoid_rows_f32",
        )?;
        let op = out.buf.as_ptr();
        let xp = x.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let mp = gamma.buf.as_ptr();
        let hd = i32_of(g.head_dim, "head_dim")?;
        let nh = i32_of(g.n_heads, "n_heads")?;
        let xrs = i32_of(g.x_row_stride, "x_row_stride")?;
        let grs = i32_of(g.gate_row_stride, "gate_row_stride")?;
        let ors = i32_of(g.out_row_stride, "out_row_stride")?;
        let eps = g.eps;
        let round = g.round_bf16 as i32;
        let mut params: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &mp as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &xrs as *const _ as *mut c_void,
            &grs as *const _ as *mut c_void,
            &ors as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &round as *const _ as *mut c_void,
        ];
        let bytes = g.n_rows * ext * 4 * 3;
        let timer =
            profile::begin_timer(&self.hip, "norm", "gated_rmsnorm_sigmoid_rows_f32", bytes);
        let result = self.launch_maybe_blob(
            "gated_rmsnorm_sigmoid_rows_f32",
            [g.n_heads as u32, g.n_rows as u32, 1],
            [g.head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(op);
                b.push_ptr(xp);
                b.push_ptr(gp);
                b.push_ptr(mp);
                for v in [hd, nh, xrs, grs, ors] {
                    b.push_i32(v);
                }
                b.push_f32(eps);
                b.push_i32(round);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

/// Operands + geometry of `raon_rda_prepare_f32` (f32 elements).
pub struct RdaPrepareLaunch<'a> {
    pub q_out: &'a GpuTensor,
    pub k_out: &'a GpuTensor,
    pub vprime_out: &'a GpuTensor,
    pub log_alpha_out: &'a GpuTensor,
    pub beta_out: &'a GpuTensor,
    pub qconv: &'a GpuTensor,
    pub kconv: &'a GpuTensor,
    pub vconv: &'a GpuTensor,
    pub f: &'a GpuTensor,
    pub b: &'a GpuTensor,
    pub delta: &'a GpuTensor,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub n_rows: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub conv_row_stride: usize,
    pub f_row_stride: usize,
    pub bd_row_stride: usize,
    pub out_row_stride: usize,
    pub beta_row_stride: usize,
    pub q_scale: f32,
    pub norm_eps: f32,
    pub lower_bound: f32,
    /// Model-dtype BF16 boundaries (true) or FP32 model dtype (false).
    pub round_bf16: bool,
}

impl Gpu {
    /// Raon RDA q/k normalization, decay gate, beta and write-ratio prep over
    /// `n_rows` rows in one launch. Kernel args: q_out,k_out,vprime_out,
    /// log_alpha_out,beta_out,qconv,kconv,vconv,f,b,delta,a_log,dt_bias (ptr);
    /// head_dim,n_heads,conv_row_stride,f_row_stride,bd_row_stride,
    /// out_row_stride,beta_row_stride (i32); q_scale,norm_eps,lower_bound (f32);
    /// round_bf16 (i32).
    pub fn raon_rda_prepare_f32(&mut self, p: &RdaPrepareLaunch<'_>) -> HipResult<()> {
        if p.head_dim == 0 || p.head_dim > 256 || !p.head_dim.is_power_of_two() {
            return Err(refuse(format!(
                "raon_rda_prepare_f32: head_dim {} must be a power of two <= 256",
                p.head_dim
            )));
        }
        if p.n_rows == 0 {
            return Ok(());
        }
        let ext = p.n_heads * p.head_dim;
        let n = p.n_rows;
        need(p.qconv, "qconv", n, p.conv_row_stride, ext)?;
        need(p.kconv, "kconv", n, p.conv_row_stride, ext)?;
        need(p.vconv, "vconv", n, p.conv_row_stride, ext)?;
        need(p.f, "f", n, p.f_row_stride, ext)?;
        need(p.b, "b", n, p.bd_row_stride, p.n_heads)?;
        need(p.delta, "delta", n, p.bd_row_stride, p.n_heads)?;
        need(p.a_log, "a_log", 1, 0, p.n_heads)?;
        need(p.dt_bias, "dt_bias", 1, 0, ext)?;
        need(p.q_out, "q_out", n, p.out_row_stride, ext)?;
        need(p.k_out, "k_out", n, p.out_row_stride, ext)?;
        need(p.vprime_out, "vprime_out", n, p.out_row_stride, ext)?;
        need(p.log_alpha_out, "log_alpha_out", n, p.out_row_stride, ext)?;
        need(p.beta_out, "beta_out", n, p.beta_row_stride, p.n_heads)?;

        self.bind_thread()?;
        self.ensure_kernel(
            "raon_rda_prepare",
            RAON_RDA_PREPARE_SRC,
            "raon_rda_prepare_f32",
        )?;
        let ptrs = [
            p.q_out.buf.as_ptr(),
            p.k_out.buf.as_ptr(),
            p.vprime_out.buf.as_ptr(),
            p.log_alpha_out.buf.as_ptr(),
            p.beta_out.buf.as_ptr(),
            p.qconv.buf.as_ptr(),
            p.kconv.buf.as_ptr(),
            p.vconv.buf.as_ptr(),
            p.f.buf.as_ptr(),
            p.b.buf.as_ptr(),
            p.delta.buf.as_ptr(),
            p.a_log.buf.as_ptr(),
            p.dt_bias.buf.as_ptr(),
        ];
        let ints = [
            i32_of(p.head_dim, "head_dim")?,
            i32_of(p.n_heads, "n_heads")?,
            i32_of(p.conv_row_stride, "conv_row_stride")?,
            i32_of(p.f_row_stride, "f_row_stride")?,
            i32_of(p.bd_row_stride, "bd_row_stride")?,
            i32_of(p.out_row_stride, "out_row_stride")?,
            i32_of(p.beta_row_stride, "beta_row_stride")?,
        ];
        let floats = [p.q_scale, p.norm_eps, p.lower_bound];
        let round = p.round_bf16 as i32;
        let mut params: Vec<*mut c_void> = Vec::with_capacity(24);
        for v in &ptrs {
            params.push(v as *const _ as *mut c_void);
        }
        for v in &ints {
            params.push(v as *const _ as *mut c_void);
        }
        for v in &floats {
            params.push(v as *const _ as *mut c_void);
        }
        params.push(&round as *const _ as *mut c_void);
        let bytes = p.n_rows * ext * 4 * 8;
        let timer = profile::begin_timer(&self.hip, "gated_delta", "raon_rda_prepare_f32", bytes);
        let result = self.launch_maybe_blob(
            "raon_rda_prepare_f32",
            [p.n_heads as u32, p.n_rows as u32, 1],
            [p.head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                for v in ptrs {
                    b.push_ptr(v);
                }
                for v in ints {
                    b.push_i32(v);
                }
                for v in floats {
                    b.push_f32(v);
                }
                b.push_i32(round);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
