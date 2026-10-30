// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Generic N-stream broadcast / route / aggregate / mix / mean launchers.
//!
//! A multi-stream residual keeps `n` copies ("streams") of a `width`-wide row.
//! A site reads the OLD streams, derives controls, collapses the streams to one
//! branch input (`aggregate`), and after the branch writes the NEW streams
//! (`postmix`) into a separate ping-pong buffer. The final hidden state is the
//! `mean` of the streams. Nothing here names a model: stream count, width,
//! epsilons and rounding are arguments.
//!
//! Streams are F32 planes that carry model-dtype values. With
//! [`StreamRounding::Bf16`] every stage rounds its output to bfloat16
//! (round-to-nearest-even) at the same points a bf16 reference does; with
//! [`StreamRounding::F32`] no rounding happens (explicit F32 oracle mode).
//!
//! Chart policy `HelmertBounded` ([`Gpu::stream_mix_route_helmert`]): controls
//! `pre`, `post` and a bounded stream-to-stream map `res` come from a selector
//! projection `z_raw[r] = x[r] . phi^T` (ordinary GEMM, done by the caller) and
//! a Helmert-basis chart, see `kernels/src/stream_mix_route_helmert.hip`.
//! Kernels are batched over rows (grid by row), so `rows == 1` is the decode
//! case. Launchers use `bind_thread` + `launch_maybe_blob` with owned kernarg
//! blobs and read no device value on the host, so they are graph-capture safe.

use std::ffi::c_void;

use crate::dispatch::{launch_params_blob, DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};

pub const STREAM_MIX_BROADCAST_SRC: &str =
    include_str!("../../../kernels/src/stream_mix_broadcast.hip");
pub const STREAM_MIX_ROUTE_HELMERT_SRC: &str =
    include_str!("../../../kernels/src/stream_mix_route_helmert.hip");
pub const STREAM_MIX_AGGREGATE_SRC: &str =
    include_str!("../../../kernels/src/stream_mix_aggregate.hip");
pub const STREAM_MIX_MIX_SRC: &str = include_str!("../../../kernels/src/stream_mix_mix.hip");
pub const STREAM_MIX_MEAN_SRC: &str = include_str!("../../../kernels/src/stream_mix_mean.hip");

/// Largest stream count the kernels support.
pub const STREAM_MIX_MAX_STREAMS: usize = 8;
/// Largest row width: aggregate/mean keep one rounded row in LDS.
pub const STREAM_MIX_MAX_WIDTH: usize = 16_000;

const FUNC_BROADCAST: &str = "stream_mix_broadcast";
const FUNC_ROUTE_HELMERT: &str = "stream_mix_route_helmert";
const FUNC_AGGREGATE: &str = "stream_mix_aggregate";
const FUNC_POSTMIX: &str = "stream_mix_postmix";
const FUNC_MEAN: &str = "stream_mix_mean";

/// Where stages round their outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamRounding {
    /// bfloat16 round-to-nearest-even at every stage output (and the selector).
    Bf16,
    /// No rounding (F32 oracle mode).
    F32,
}

impl StreamRounding {
    fn flag(self) -> i32 {
        matches!(self, StreamRounding::Bf16) as i32
    }
}

/// Plain RMSNorm folded into an aggregate/mean launch: `x * rsqrt(mean(x^2) +
/// eps) * gamma`, no `1 + gamma`. `weight` is `[width]`, F32 or BF16.
#[derive(Clone, Copy)]
pub struct StreamGamma<'a> {
    pub weight: &'a GpuTensor,
    pub eps: f32,
}

/// Operands of [`Gpu::stream_mix_route_helmert`].
pub struct StreamRouteHelmert<'a> {
    /// Old streams `[rows, n * width]` (read).
    pub streams: &'a GpuTensor,
    /// Raw selector projection `[rows, 2n + (n-1)^2]` (read).
    pub z_raw: &'a GpuTensor,
    /// `[alpha_pre, alpha_post, alpha_z, eta, b_pre[n], b_post[n], b_z[(n-1)^2]]`
    /// F32 (read).
    pub params: &'a GpuTensor,
    /// `[rows, n]` (written).
    pub pre: &'a GpuTensor,
    /// `[rows, n]` (written).
    pub post: &'a GpuTensor,
    /// `[rows, n, n]` destination/source (written).
    pub res: &'a GpuTensor,
    pub rows: usize,
    pub n: usize,
    pub width: usize,
    pub post_factor: f32,
    pub selector_eps: f32,
    pub rounding: StreamRounding,
}

/// Operands of [`Gpu::stream_mix_aggregate`].
pub struct StreamAggregate<'a> {
    /// Old streams `[rows, n, width]` (read).
    pub streams: &'a GpuTensor,
    /// `[rows, n]` (read).
    pub pre: &'a GpuTensor,
    /// Optional norm applied to the rounded aggregate.
    pub norm: Option<StreamGamma<'a>>,
    /// `[rows, width]` (written).
    pub out: &'a GpuTensor,
    pub rows: usize,
    pub n: usize,
    pub width: usize,
    pub rounding: StreamRounding,
}

/// Operands of [`Gpu::stream_mix_postmix`].
pub struct StreamPostmix<'a> {
    /// Old streams `[rows, n, width]` (read).
    pub streams: &'a GpuTensor,
    /// Branch output `[rows, width]` (read).
    pub branch: &'a GpuTensor,
    /// `[rows, n]` (read).
    pub post: &'a GpuTensor,
    /// `[rows, n, n]` destination/source (read).
    pub res: &'a GpuTensor,
    /// New streams `[rows, n, width]`; must not alias `streams` (written).
    pub out: &'a GpuTensor,
    pub rows: usize,
    pub n: usize,
    pub width: usize,
    pub rounding: StreamRounding,
}

/// Operands of [`Gpu::stream_mix_mean`].
pub struct StreamMean<'a> {
    /// Streams `[rows, n, width]` (read).
    pub streams: &'a GpuTensor,
    /// Optional norm applied to the rounded mean.
    pub norm: Option<StreamGamma<'a>>,
    /// `[rows, width]` (written).
    pub out: &'a GpuTensor,
    pub rows: usize,
    pub n: usize,
    pub width: usize,
    pub rounding: StreamRounding,
}

fn err(msg: impl Into<String>) -> HipError {
    HipError::new(1, &msg.into())
}

fn f32_tensor(t: &GpuTensor, elems: usize, what: &str) -> HipResult<()> {
    if t.dtype != DType::F32 {
        return Err(err(format!("stream_mix: {what} must be F32")));
    }
    need(t, elems * 4, what)
}

fn need(t: &GpuTensor, bytes: usize, what: &str) -> HipResult<()> {
    if t.buf.size() < bytes {
        return Err(err(format!(
            "stream_mix: {what} holds {} bytes, needs {bytes}",
            t.buf.size()
        )));
    }
    Ok(())
}

fn disjoint(a: &GpuTensor, a_bytes: usize, b: &GpuTensor, b_bytes: usize) -> bool {
    let (ap, bp) = (a.buf.as_ptr() as usize, b.buf.as_ptr() as usize);
    ap + a_bytes <= bp || bp + b_bytes <= ap
}

fn check_geometry(what: &str, rows: usize, n: usize, width: usize) -> HipResult<()> {
    if rows == 0 {
        return Err(err(format!("{what}: rows == 0")));
    }
    if n == 0 || n > STREAM_MIX_MAX_STREAMS {
        return Err(err(format!(
            "{what}: stream count {n} outside 1..={STREAM_MIX_MAX_STREAMS}"
        )));
    }
    if width == 0 || width > STREAM_MIX_MAX_WIDTH {
        return Err(err(format!(
            "{what}: width {width} outside 1..={STREAM_MIX_MAX_WIDTH}"
        )));
    }
    if rows > u32::MAX as usize || rows * n * width > i32::MAX as usize {
        return Err(err(format!(
            "{what}: row count / extent overflows i32 index"
        )));
    }
    Ok(())
}

/// Validated `(gamma pointer, kind, eps, has_norm)`.
fn norm_args(
    what: &str,
    norm: &Option<StreamGamma<'_>>,
    width: usize,
) -> HipResult<(*mut c_void, i32, f32, i32)> {
    match norm {
        None => Ok((std::ptr::null_mut(), 0, 0.0, 0)),
        Some(g) => {
            let (kind, bytes) = match g.weight.dtype {
                DType::F32 => (0, width * 4),
                DType::BF16 => (1, width * 2),
                other => {
                    return Err(err(format!(
                        "{what}: norm gamma dtype {other:?} unsupported (F32 or BF16 only)"
                    )))
                }
            };
            need(g.weight, bytes, "norm gamma")?;
            Ok((g.weight.buf.as_ptr(), kind, g.eps, 1))
        }
    }
}

impl Gpu {
    /// `dst[r, s, :] = round(src[r, :])` for every stream `s < n_streams`.
    /// `src` is `[rows, src_stride]` F32 (`src_stride >= width`).
    pub fn stream_mix_broadcast(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        rows: usize,
        n_streams: usize,
        width: usize,
        src_stride: usize,
        rounding: StreamRounding,
    ) -> HipResult<()> {
        check_geometry("stream_mix_broadcast", rows, n_streams, width)?;
        if src_stride < width {
            return Err(err("stream_mix_broadcast: src_stride < width"));
        }
        f32_tensor(src, (rows - 1) * src_stride + width, "broadcast src")?;
        f32_tensor(dst, rows * n_streams * width, "broadcast dst")?;
        if !disjoint(src, src.buf.size(), dst, dst.buf.size()) {
            return Err(err("stream_mix_broadcast: src and dst alias"));
        }
        self.bind_thread()?;
        self.ensure_kernel(FUNC_BROADCAST, STREAM_MIX_BROADCAST_SRC, FUNC_BROADCAST)?;
        let sp = src.buf.as_ptr();
        let dp = dst.buf.as_ptr();
        let ns = n_streams as i32;
        let wd = width as i32;
        let ss = src_stride as i32;
        let bf = rounding.flag();
        let mut params = [
            &sp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &ns as *const _ as *mut c_void,
            &wd as *const _ as *mut c_void,
            &ss as *const _ as *mut c_void,
            &bf as *const _ as *mut c_void,
        ];
        launch_params_blob!(
            self,
            FUNC_BROADCAST,
            [(width as u32).div_ceil(256), rows as u32, 1],
            [256, 1, 1],
            0,
            params; sp, dp, ns, wd, ss, bf
        )
    }

    /// Selector finalize + Helmert chart for `rows` rows; see the kernel header.
    pub fn stream_mix_route_helmert(&mut self, p: &StreamRouteHelmert<'_>) -> HipResult<()> {
        const W: &str = "stream_mix_route_helmert";
        check_geometry(W, p.rows, p.n, p.width)?;
        if p.n < 2 {
            return Err(err(format!("{W}: the Helmert chart needs n >= 2 streams")));
        }
        let nm = p.n - 1;
        let zdim = 2 * p.n + nm * nm;
        let x_dim = p.n * p.width;
        f32_tensor(p.streams, p.rows * x_dim, "streams")?;
        f32_tensor(p.z_raw, p.rows * zdim, "z_raw")?;
        f32_tensor(p.params, 4 + zdim, "params")?;
        f32_tensor(p.pre, p.rows * p.n, "pre")?;
        f32_tensor(p.post, p.rows * p.n, "post")?;
        f32_tensor(p.res, p.rows * p.n * p.n, "res")?;
        self.bind_thread()?;
        self.ensure_kernel(
            FUNC_ROUTE_HELMERT,
            STREAM_MIX_ROUTE_HELMERT_SRC,
            FUNC_ROUTE_HELMERT,
        )?;
        let xp = p.streams.buf.as_ptr();
        let zp = p.z_raw.buf.as_ptr();
        let qp = p.params.buf.as_ptr();
        let pre = p.pre.buf.as_ptr();
        let post = p.post.buf.as_ptr();
        let res = p.res.buf.as_ptr();
        let nn = p.n as i32;
        let xd = x_dim as i32;
        let pf = p.post_factor;
        let se = p.selector_eps;
        let bf = p.rounding.flag();
        let mut params = [
            &xp as *const _ as *mut c_void,
            &zp as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &pre as *const _ as *mut c_void,
            &post as *const _ as *mut c_void,
            &res as *const _ as *mut c_void,
            &nn as *const _ as *mut c_void,
            &xd as *const _ as *mut c_void,
            &pf as *const _ as *mut c_void,
            &se as *const _ as *mut c_void,
            &bf as *const _ as *mut c_void,
        ];
        launch_params_blob!(
            self,
            FUNC_ROUTE_HELMERT,
            [p.rows as u32, 1, 1],
            [256, 1, 1],
            0,
            params; xp, zp, qp, pre, post, res, nn, xd, pf, se, bf
        )
    }

    /// `out = [norm](round(sum_s pre[s] * streams[s]))` per row.
    pub fn stream_mix_aggregate(&mut self, p: &StreamAggregate<'_>) -> HipResult<()> {
        const W: &str = "stream_mix_aggregate";
        check_geometry(W, p.rows, p.n, p.width)?;
        f32_tensor(p.streams, p.rows * p.n * p.width, "streams")?;
        f32_tensor(p.pre, p.rows * p.n, "pre")?;
        f32_tensor(p.out, p.rows * p.width, "out")?;
        if !disjoint(p.streams, p.streams.buf.size(), p.out, p.out.buf.size()) {
            return Err(err(format!("{W}: streams and out alias")));
        }
        let (gp, gk, eps, has_norm) = norm_args(W, &p.norm, p.width)?;
        self.bind_thread()?;
        self.ensure_kernel(FUNC_AGGREGATE, STREAM_MIX_AGGREGATE_SRC, FUNC_AGGREGATE)?;
        let xp = p.streams.buf.as_ptr();
        let pp = p.pre.buf.as_ptr();
        let op = p.out.buf.as_ptr();
        let nn = p.n as i32;
        let wd = p.width as i32;
        let bf = p.rounding.flag();
        let mut params = [
            &xp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nn as *const _ as *mut c_void,
            &wd as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &bf as *const _ as *mut c_void,
            &has_norm as *const _ as *mut c_void,
            &gk as *const _ as *mut c_void,
        ];
        launch_params_blob!(
            self,
            FUNC_AGGREGATE,
            [p.rows as u32, 1, 1],
            [256, 1, 1],
            (p.width * 4) as u32,
            params; xp, pp, gp, op, nn, wd, eps, bf, has_norm, gk
        )
    }

    /// `out[i] = round(sum_j res[i, j] * streams[j] + post[i] * branch)`.
    pub fn stream_mix_postmix(&mut self, p: &StreamPostmix<'_>) -> HipResult<()> {
        const W: &str = "stream_mix_postmix";
        check_geometry(W, p.rows, p.n, p.width)?;
        let sbytes = p.rows * p.n * p.width * 4;
        f32_tensor(p.streams, p.rows * p.n * p.width, "streams")?;
        f32_tensor(p.branch, p.rows * p.width, "branch")?;
        f32_tensor(p.post, p.rows * p.n, "post")?;
        f32_tensor(p.res, p.rows * p.n * p.n, "res")?;
        f32_tensor(p.out, p.rows * p.n * p.width, "out")?;
        // Every output stream reads every old stream: ping-pong, never in place.
        if !disjoint(p.streams, sbytes, p.out, sbytes) {
            return Err(err(format!("{W}: out aliases the old streams")));
        }
        self.bind_thread()?;
        self.ensure_kernel(FUNC_POSTMIX, STREAM_MIX_MIX_SRC, FUNC_POSTMIX)?;
        let xp = p.streams.buf.as_ptr();
        let bp = p.branch.buf.as_ptr();
        let pp = p.post.buf.as_ptr();
        let rp = p.res.buf.as_ptr();
        let op = p.out.buf.as_ptr();
        let nn = p.n as i32;
        let wd = p.width as i32;
        let bf = p.rounding.flag();
        let mut params = [
            &xp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nn as *const _ as *mut c_void,
            &wd as *const _ as *mut c_void,
            &bf as *const _ as *mut c_void,
        ];
        launch_params_blob!(
            self,
            FUNC_POSTMIX,
            [(p.width as u32).div_ceil(256), p.rows as u32, 1],
            [256, 1, 1],
            0,
            params; xp, bp, pp, rp, op, nn, wd, bf
        )
    }

    /// `out = [norm](round(mean_s streams[s]))` per row.
    pub fn stream_mix_mean(&mut self, p: &StreamMean<'_>) -> HipResult<()> {
        const W: &str = "stream_mix_mean";
        check_geometry(W, p.rows, p.n, p.width)?;
        f32_tensor(p.streams, p.rows * p.n * p.width, "streams")?;
        f32_tensor(p.out, p.rows * p.width, "out")?;
        if !disjoint(p.streams, p.streams.buf.size(), p.out, p.out.buf.size()) {
            return Err(err(format!("{W}: streams and out alias")));
        }
        let (gp, gk, eps, has_norm) = norm_args(W, &p.norm, p.width)?;
        self.bind_thread()?;
        self.ensure_kernel(FUNC_MEAN, STREAM_MIX_MEAN_SRC, FUNC_MEAN)?;
        let xp = p.streams.buf.as_ptr();
        let op = p.out.buf.as_ptr();
        let nn = p.n as i32;
        let wd = p.width as i32;
        let bf = p.rounding.flag();
        let mut params = [
            &xp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nn as *const _ as *mut c_void,
            &wd as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &bf as *const _ as *mut c_void,
            &has_norm as *const _ as *mut c_void,
            &gk as *const _ as *mut c_void,
        ];
        launch_params_blob!(
            self,
            FUNC_MEAN,
            [p.rows as u32, 1, 1],
            [256, 1, 1],
            (p.width * 4) as u32,
            params; xp, gp, op, nn, wd, eps, bf, has_norm, gk
        )
    }
}

impl Gpu {
    /// JITs every stream-mix kernel up front so a later graph capture never
    /// compiles inside the capture window.
    pub fn stream_mix_preload(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(FUNC_BROADCAST, STREAM_MIX_BROADCAST_SRC, FUNC_BROADCAST)?;
        self.ensure_kernel(
            FUNC_ROUTE_HELMERT,
            STREAM_MIX_ROUTE_HELMERT_SRC,
            FUNC_ROUTE_HELMERT,
        )?;
        self.ensure_kernel(FUNC_AGGREGATE, STREAM_MIX_AGGREGATE_SRC, FUNC_AGGREGATE)?;
        self.ensure_kernel(FUNC_POSTMIX, STREAM_MIX_MIX_SRC, FUNC_POSTMIX)?;
        self.ensure_kernel(FUNC_MEAN, STREAM_MIX_MEAN_SRC, FUNC_MEAN)
    }
}

/// CPU reference of the same math (scalar F32, same operation order as the
/// kernels). Used by unit tests and by diagnostic oracles; not a launch path.
pub mod reference {
    use super::StreamRounding;

    /// bfloat16 round-to-nearest-even (NaN preserved), result as F32.
    pub fn round_bf16(x: f32) -> f32 {
        let u = x.to_bits();
        if (u & 0x7fff_ffff) > 0x7f80_0000 {
            return x;
        }
        let u = u.wrapping_add(0x7fff + ((u >> 16) & 1));
        f32::from_bits(u & 0xffff_0000)
    }

    fn rnd(x: f32, r: StreamRounding) -> f32 {
        match r {
            StreamRounding::Bf16 => round_bf16(x),
            StreamRounding::F32 => x,
        }
    }

    pub fn sigmoid(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    /// Orthonormal Helmert basis `Q[i][kk]`, `n x (n-1)`.
    pub fn helmert(i: usize, kk: usize) -> f32 {
        let k = kk + 1;
        let s = 1.0 / ((k * (k + 1)) as f32).sqrt();
        if i < k {
            s
        } else if i == k {
            -(k as f32) * s
        } else {
            0.0
        }
    }

    /// Chart output for one row: `res[n*n]` from the `(n-1)^2` chart logits
    /// `t_z` (already `alpha_z * z + b_z`) and `eta`.
    pub fn helmert_chart(n: usize, t_z: &[f32], eta: f32) -> Vec<f32> {
        let nm = n - 1;
        assert_eq!(t_z.len(), nm * nm);
        let inv_n = 1.0 / n as f32;
        let mut e = vec![0.0f32; n * n];
        let mut b = vec![0.0f32; n * n];
        let mut rho = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0f32;
                for k in 0..nm {
                    let mut u = 0.0f32;
                    for m in 0..nm {
                        u += helmert(i, m) * t_z[m * nm + k];
                    }
                    acc += u * helmert(j, k);
                }
                e[i * n + j] = acc;
                b[i * n + j] = 0.5 * ((if i == j { 1.0 } else { 0.0 }) + inv_n);
                rho = rho.max(-acc / b[i * n + j]);
            }
        }
        let g = sigmoid(eta);
        (0..n * n)
            .map(|t| {
                let (i, j) = (t / n, t % n);
                let phi = b[t] + e[t] / (1.0 + rho);
                (1.0 - g) * (if i == j { 1.0 } else { 0.0 }) + g * phi
            })
            .collect()
    }

    /// `(pre[n], post[n], res[n*n])` of one row from the raw selector output
    /// `z_raw[2n + (n-1)^2]` and the old row `x[n*width]`; `params` as in
    /// [`super::StreamRouteHelmert::params`].
    pub fn route_helmert(
        n: usize,
        x: &[f32],
        z_raw: &[f32],
        params: &[f32],
        post_factor: f32,
        selector_eps: f32,
        rounding: StreamRounding,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let zdim = 2 * n + (n - 1) * (n - 1);
        assert_eq!(z_raw.len(), zdim);
        assert_eq!(params.len(), 4 + zdim);
        let mut ss = 0.0f32;
        for v in x {
            ss += v * v;
        }
        let inv = 1.0 / (ss / x.len() as f32 + selector_eps).sqrt();
        let (a_pre, a_post, a_z, eta) = (params[0], params[1], params[2], params[3]);
        let bias = &params[4..];
        let z: Vec<f32> = z_raw.iter().map(|&v| rnd(v, rounding) * inv).collect();
        let pre = (0..n).map(|i| sigmoid(a_pre * z[i] + bias[i])).collect();
        let post = (0..n)
            .map(|i| post_factor * sigmoid(a_post * z[n + i] + bias[n + i]))
            .collect();
        let t_z: Vec<f32> = (2 * n..zdim).map(|t| a_z * z[t] + bias[t]).collect();
        (pre, post, helmert_chart(n, &t_z, eta))
    }

    /// One-row aggregate: `round(sum_s pre[s] * x[s])` then optional norm.
    pub fn aggregate(
        n: usize,
        width: usize,
        x: &[f32],
        pre: &[f32],
        norm: Option<(&[f32], f32)>,
        rounding: StreamRounding,
    ) -> Vec<f32> {
        let agg: Vec<f32> = (0..width)
            .map(|c| {
                let mut a = 0.0f32;
                for s in 0..n {
                    a += pre[s] * x[s * width + c];
                }
                rnd(a, rounding)
            })
            .collect();
        rms_norm(&agg, norm, rounding)
    }

    /// One-row postmix.
    pub fn postmix(
        n: usize,
        width: usize,
        x: &[f32],
        branch: &[f32],
        post: &[f32],
        res: &[f32],
        rounding: StreamRounding,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n * width];
        for i in 0..n {
            for c in 0..width {
                let mut a = 0.0f32;
                for j in 0..n {
                    a += res[i * n + j] * x[j * width + c];
                }
                a += post[i] * branch[c];
                out[i * width + c] = rnd(a, rounding);
            }
        }
        out
    }

    /// One-row mean then optional norm.
    pub fn mean(
        n: usize,
        width: usize,
        x: &[f32],
        norm: Option<(&[f32], f32)>,
        rounding: StreamRounding,
    ) -> Vec<f32> {
        let m: Vec<f32> = (0..width)
            .map(|c| {
                let mut a = 0.0f32;
                for s in 0..n {
                    a += x[s * width + c];
                }
                rnd(a / n as f32, rounding)
            })
            .collect();
        rms_norm(&m, norm, rounding)
    }

    fn rms_norm(v: &[f32], norm: Option<(&[f32], f32)>, rounding: StreamRounding) -> Vec<f32> {
        match norm {
            None => v.to_vec(),
            Some((gamma, eps)) => {
                let mut ss = 0.0f32;
                for x in v {
                    ss += x * x;
                }
                let r = 1.0 / (ss / v.len() as f32 + eps).sqrt();
                v.iter()
                    .zip(gamma)
                    .map(|(x, g)| rnd(x * r * g, rounding))
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reference::*;
    use super::StreamRounding;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(round_bf16(1.0), 1.0);
        // 1 + 2^-8 is the halfway point between bf16 neighbours 1 and 1 + 2^-7: ties to even (1.0).
        assert_eq!(round_bf16(1.0 + 1.0 / 256.0), 1.0);
        // 1 + 3*2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: ties to even (1 + 2^-6).
        assert_eq!(round_bf16(1.0 + 3.0 / 256.0), 1.0 + 1.0 / 64.0);
        assert!(round_bf16(f32::NAN).is_nan());
        assert_eq!(round_bf16(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn helmert_basis_is_orthonormal_and_sum_free() {
        for n in 2..=8usize {
            for a in 0..n - 1 {
                let col_sum: f32 = (0..n).map(|i| helmert(i, a)).sum();
                assert!(close(col_sum, 0.0, 1e-6), "n={n} col {a} sum {col_sum}");
                for b in 0..n - 1 {
                    let dot: f32 = (0..n).map(|i| helmert(i, a) * helmert(i, b)).sum();
                    let want = if a == b { 1.0 } else { 0.0 };
                    assert!(close(dot, want, 1e-6), "n={n} Q^TQ[{a},{b}]={dot}");
                }
            }
        }
    }

    #[test]
    fn zero_chart_gives_blended_identity_and_b() {
        // Z = 0 -> E = 0, rho = 0, Phi = B; res = (1-g) I + g B.
        // n=4, eta=0 -> g = 0.5: diag .5 + .5*.625 = .8125, off .5*.125 = .0625.
        let res = helmert_chart(4, &[0.0; 9], 0.0);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 0.8125 } else { 0.0625 };
                assert!(
                    close(res[i * 4 + j], want, 1e-7),
                    "[{i},{j}]={}",
                    res[i * 4 + j]
                );
            }
        }
    }

    #[test]
    fn chart_rows_and_columns_sum_to_one_and_stay_nonnegative() {
        // Doubly stochastic for any Z, eta (Q^T 1 = 0, B rows sum 1), nonnegative via rho.
        let mut seed = 1u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 8.0
        };
        for n in [2usize, 3, 4, 5, 8] {
            for eta in [-20.0f32, -3.0, 0.0, 3.0, 20.0] {
                for _ in 0..20 {
                    let t: Vec<f32> = (0..(n - 1) * (n - 1)).map(|_| rnd() * 4.0).collect();
                    let res = helmert_chart(n, &t, eta);
                    for i in 0..n {
                        let rs: f32 = (0..n).map(|j| res[i * n + j]).sum();
                        let cs: f32 = (0..n).map(|j| res[j * n + i]).sum();
                        assert!(
                            close(rs, 1.0, 2e-5) && close(cs, 1.0, 2e-5),
                            "n={n} eta={eta}"
                        );
                        for j in 0..n {
                            assert!(
                                res[i * n + j] >= -2e-6,
                                "n={n} eta={eta} neg {}",
                                res[i * n + j]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn chart_rho_positive_scales_error_and_touches_boundary() {
        // n = 2: Q = [1,-1]/sqrt2, nm = 1, Z scalar z -> E = z * Q Q^T = z/2 * [[1,-1],[-1,1]].
        // B = [[.75,.25],[.25,.75]]. -E/B max: off-diagonal entries give z/2/.25 = 2z when z>0.
        // rho = 2z (z=1.5 -> 3); Phi = B + E/(1+rho): off-diag = .25 - .75/4 = .0625 ... checked below.
        let z = 1.5f32;
        let res = helmert_chart(2, &[z], 0.0); // g = .5
        let e = z / 2.0;
        let rho = (e / 0.25).max(0.0);
        assert!(close(rho, 3.0, 1e-6));
        let phi_off = 0.25 - e / (1.0 + rho);
        let phi_diag = 0.75 + e / (1.0 + rho);
        assert!(close(res[1], 0.5 * phi_off, 1e-6) && close(res[2], 0.5 * phi_off, 1e-6));
        assert!(close(res[0], 0.5 + 0.5 * phi_diag, 1e-6));
        // Z = z has -E/B >= 0 only for the z>0 off-diagonal: with the sign flipped the diagonal binds instead.
        let neg = helmert_chart(2, &[-z], 0.0);
        let rho_n = (z / 2.0 / 0.75f32).max(0.0);
        let phi_d = 0.75 - (z / 2.0) / (1.0 + rho_n);
        assert!(close(neg[0], 0.5 + 0.5 * phi_d, 1e-6));
    }

    #[test]
    fn eta_extremes_saturate_to_identity_or_phi() {
        let t = [0.3f32, -1.0, 2.0, 0.5, 0.1, -0.7, 1.1, 0.0, -0.2];
        let ident = helmert_chart(4, &t, -40.0);
        let phi = helmert_chart(4, &t, 40.0);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(close(ident[i * 4 + j], want, 1e-6));
            }
        }
        // g = 1: result is Phi itself; rows still sum to one.
        let s: f32 = (0..4).map(|j| phi[j]).sum();
        assert!(close(s, 1.0, 2e-6));
    }

    #[test]
    fn orientation_is_destination_source() {
        // A non-symmetric Z must give non-symmetric res; the transpose of Z gives the transposed res.
        let t = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut tt = [0.0f32; 9];
        for a in 0..3 {
            for b in 0..3 {
                tt[a * 3 + b] = t[b * 3 + a];
            }
        }
        let r = helmert_chart(4, &t, 3.0);
        let rt = helmert_chart(4, &tt, 3.0);
        let mut asym = 0.0f32;
        for i in 0..4 {
            for j in 0..4 {
                asym = asym.max((r[i * 4 + j] - r[j * 4 + i]).abs());
                assert!(close(r[i * 4 + j], rt[j * 4 + i], 1e-6));
            }
        }
        assert!(asym > 1e-3, "res unexpectedly symmetric");
    }

    #[test]
    fn route_postmix_hand_case() {
        // n=2, width=1, x=[1,3]; z_raw all zero, params alphas 1, biases 0, eta=0.
        // pre = sigmoid(0) = .5; post = 2*.5 = 1; Z = 0 -> res = [[.875,.125],[.125,.875]]*... g=.5:
        // B(n=2) diag .75 off .25 -> res diag .5+.5*.75 = .875, off .125.
        let n = 2;
        let params = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x = [1.0f32, 3.0];
        let (pre, post, res) =
            route_helmert(n, &x, &[0.0; 5], &params, 2.0, 1e-5, StreamRounding::F32);
        assert_eq!(pre, vec![0.5, 0.5]);
        assert_eq!(post, vec![1.0, 1.0]);
        assert!(close(res[0], 0.875, 1e-7) && close(res[1], 0.125, 1e-7));
        let agg = aggregate(n, 1, &x, &pre, None, StreamRounding::F32);
        assert_eq!(agg, vec![2.0]);
        let out = postmix(n, 1, &x, &[10.0], &post, &res, StreamRounding::F32);
        // out0 = .875*1 + .125*3 + 1*10 = 11.25; out1 = .125*1 + .875*3 + 10 = 12.75
        assert!(close(out[0], 11.25, 1e-6) && close(out[1], 12.75, 1e-6));
        let m = mean(n, 1, &out, None, StreamRounding::F32);
        assert!(close(m[0], 12.0, 1e-6));
    }

    #[test]
    fn norm_epilogue_uses_plain_gamma() {
        let g = [2.0f32, 2.0];
        let y = mean(1, 2, &[3.0, 4.0], Some((&g, 0.0)), StreamRounding::F32);
        // rms = sqrt((9+16)/2); y = x/rms * 2
        let rms = (12.5f32).sqrt();
        assert!(close(y[0], 3.0 / rms * 2.0, 1e-6) && close(y[1], 4.0 / rms * 2.0, 1e-6));
    }
}

/// Golden vectors produced by running the reference `RaonMHC` (route /
/// aggregate / mix, float32) on inputs from the generator below; the chart
/// is exercised with rho = 0 (zero chart) and rho in ~0.3..270.
#[cfg(test)]
mod golden_tests {
    use super::reference::*;
    use super::StreamRounding;

    struct Golden {
        n: usize,
        width: usize,
        rows: usize,
        seed: u32,
        a_z: f32,
        eta: f32,
        bz_scale: f32,
        pre: &'static [f32],
        post: &'static [f32],
        res: &'static [f32],
        agg: &'static [f32],
        normed: &'static [f32],
        out: &'static [f32],
        mean: &'static [f32],
    }

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self, scale: f32) -> f32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (((self.0 >> 8) as f32) / 16_777_216.0 - 0.5) * scale
        }
        fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
            (0..len).map(|_| self.next(scale)).collect()
        }
    }

    fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{what} length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * (1.0 + w.abs()),
                "{what}[{i}]: got {g}, reference {w}"
            );
        }
    }

    #[test]
    fn matches_reference_module_float32() {
        let goldens = [
            Golden {
                n: 4,
                width: 8,
                rows: 2,
                seed: 12345,
                a_z: 2.5,
                eta: 0.4,
                bz_scale: 2.0,
                pre: &[
                    0.378416598,
                    0.799479604,
                    0.882300913,
                    0.284599751,
                    0.0415635332,
                    0.199565321,
                    0.588936448,
                    0.863234639,
                ],
                post: &[
                    1.39600372,
                    0.0127347484,
                    1.74672437,
                    1.85150421,
                    0.381473303,
                    1.79569304,
                    1.02933466,
                    1.87206078,
                ],
                res: &[
                    0.808278561,
                    0.0802802891,
                    0.0496070497,
                    0.0618340783,
                    0.127046913,
                    0.760576069,
                    0.065072611,
                    0.0473044254,
                    0.0640964508,
                    0.0831044391,
                    0.781112313,
                    0.0716867745,
                    0.000578045321,
                    0.0760392323,
                    0.104208,
                    0.819174707,
                    0.795141518,
                    0.0335397869,
                    0.109932475,
                    0.0613862202,
                    0.00698909396,
                    0.846220016,
                    0.0733171105,
                    0.0734737888,
                    0.0764374435,
                    0.119280778,
                    0.715519905,
                    0.0887618661,
                    0.121431954,
                    0.000959433033,
                    0.101230502,
                    0.776378095,
                ],
                agg: &[
                    3.14728165,
                    -1.59769452,
                    -1.64558446,
                    1.18035448,
                    0.515774965,
                    2.82984877,
                    1.35902596,
                    0.45998162,
                    1.50457859,
                    -1.78824461,
                    1.50581002,
                    0.826935768,
                    -0.155681789,
                    -0.656420231,
                    -0.433212578,
                    1.12787533,
                ],
                normed: &[
                    1.63718927,
                    -1.01936829,
                    -0.752044141,
                    0.409126669,
                    0.190216869,
                    1.75605905,
                    0.834143519,
                    0.161574304,
                    1.26294363,
                    -1.84106922,
                    1.11045027,
                    0.462511927,
                    -0.0926472172,
                    -0.657299995,
                    -0.42906183,
                    0.639291108,
                ],
                out: &[
                    0.106670916,
                    -0.946727335,
                    1.63971972,
                    0.224889934,
                    -1.80226743,
                    0.911176026,
                    2.6093061,
                    -1.28991425,
                    1.3237884,
                    -1.21316063,
                    -0.0270662419,
                    1.4074012,
                    -0.247913063,
                    0.824262559,
                    0.478750646,
                    -0.960435152,
                    0.261191249,
                    -0.0116856238,
                    -0.320364475,
                    -1.34479177,
                    -1.7480371,
                    2.02236414,
                    1.40158796,
                    1.50951707,
                    -0.546395957,
                    -1.39841342,
                    0.165710926,
                    -0.92676121,
                    -1.41210246,
                    2.19716954,
                    2.84409809,
                    1.34005392,
                    0.47072503,
                    0.174456388,
                    1.52598143,
                    -1.31710815,
                    -0.575255036,
                    -0.136119545,
                    -1.27055717,
                    0.768433332,
                    -1.88375008,
                    2.27486229,
                    2.43932223,
                    -2.41051817,
                    2.33597636,
                    -1.12869966,
                    -3.26426697,
                    2.514781,
                    0.467301309,
                    0.876407862,
                    1.24165177,
                    -1.32567668,
                    1.44587278,
                    -2.12857533,
                    -1.93866706,
                    1.22771525,
                    0.618186891,
                    0.749504805,
                    2.18331575,
                    -0.211963296,
                    0.971446514,
                    -2.45389223,
                    -1.87389171,
                    1.89713609,
                ],
                mean: &[
                    1.18288708,
                    -0.89372623,
                    -0.511289179,
                    0.549386263,
                    0.325893641,
                    1.18647289,
                    0.988779426,
                    0.0766115785,
                    0.319898605,
                    -0.532023191,
                    1.0895766,
                    -0.385590017,
                    -0.115740001,
                    0.0978093743,
                    -0.703840137,
                    0.747959018,
                ],
            },
            Golden {
                n: 4,
                width: 8,
                rows: 3,
                seed: 777,
                a_z: 9.0,
                eta: -1.2,
                bz_scale: 6.0,
                pre: &[
                    0.217935145,
                    0.256431729,
                    0.556805372,
                    0.524394095,
                    0.591208756,
                    0.734887183,
                    0.781780124,
                    0.810873449,
                    0.896743536,
                    0.857250214,
                    0.102128781,
                    0.976763964,
                ],
                post: &[
                    1.97440612,
                    1.92158473,
                    0.779334307,
                    1.28772748,
                    0.06881845,
                    1.72079659,
                    0.457235694,
                    0.321682155,
                    1.04689133,
                    0.714643419,
                    0.0445469134,
                    0.182125166,
                ],
                res: &[
                    0.943294525,
                    0.028017031,
                    0.0227979738,
                    0.00589042157,
                    0.00010774938,
                    0.931906879,
                    0.0446935892,
                    0.023291748,
                    0.0291008633,
                    0.00991173927,
                    0.91938144,
                    0.0416059494,
                    0.0274968147,
                    0.0301643144,
                    0.0131269852,
                    0.929211855,
                    0.912673116,
                    0.0533210263,
                    0.00359083782,
                    0.0304150004,
                    0.0143076219,
                    0.929710865,
                    0.0412679389,
                    0.0147135379,
                    0.000136278119,
                    0.0145715959,
                    0.911294699,
                    0.0739974082,
                    0.0728829652,
                    0.00239647646,
                    0.0438465029,
                    0.880874038,
                    0.917743623,
                    0.012439901,
                    0.0643679649,
                    0.00544851553,
                    0.0572675541,
                    0.916612208,
                    0.00604822813,
                    0.0200719722,
                    0.000242463269,
                    0.0373804159,
                    0.917889237,
                    0.0444879048,
                    0.024746364,
                    0.0335674398,
                    0.0116945822,
                    0.929991603,
                ],
                agg: &[
                    1.50137353,
                    1.37622976,
                    -0.909692347,
                    -1.85665131,
                    1.88234079,
                    0.758179188,
                    -0.826021552,
                    0.814059377,
                    -1.37987232,
                    0.234280467,
                    -1.5744487,
                    -0.877586484,
                    0.946462512,
                    2.77613187,
                    -0.759091973,
                    -1.96817517,
                    -3.75566816,
                    -1.5244081,
                    -0.39922595,
                    0.949755669,
                    -1.43600082,
                    3.14435339,
                    3.65296698,
                    -2.15598583,
                ],
                normed: &[
                    1.22030807,
                    1.24362314,
                    -0.566876531,
                    -1.72303641,
                    1.77727282,
                    0.543457091,
                    -0.775612235,
                    0.578691363,
                    -0.978022397,
                    0.184613362,
                    -0.85556227,
                    -0.710204124,
                    0.779270887,
                    1.7352525,
                    -0.621551096,
                    -1.2200675,
                    -1.65165997,
                    -0.745335579,
                    -0.134606317,
                    0.476901531,
                    -0.733606637,
                    1.21948791,
                    1.85588789,
                    -0.829257786,
                ],
                out: &[
                    -0.201051176,
                    1.26927412,
                    2.17079329,
                    1.06126809,
                    3.33967352,
                    -0.0917614102,
                    1.49709153,
                    -2.28692198,
                    2.31926537,
                    2.32978058,
                    0.138644695,
                    -0.325113177,
                    3.26832652,
                    1.43658018,
                    0.329769731,
                    -2.20276046,
                    1.48209071,
                    2.18054771,
                    0.649913073,
                    -0.600594461,
                    2.33506942,
                    -0.250926286,
                    1.29483998,
                    -0.0282606483,
                    2.01896524,
                    1.63702106,
                    0.379888535,
                    -0.0941919088,
                    3.06233835,
                    0.459570706,
                    -0.900069177,
                    -0.666352689,
                    0.311805248,
                    1.6097312,
                    -0.945427418,
                    -0.0786917359,
                    -1.19720137,
                    0.466493458,
                    -0.594644666,
                    0.35375309,
                    -2.96755266,
                    1.94033623,
                    -2.09019613,
                    0.329352617,
                    -0.361600041,
                    2.72494745,
                    2.80886793,
                    1.06963813,
                    -1.73107874,
                    0.300765336,
                    0.267947942,
                    1.59045076,
                    0.489187837,
                    1.8022635,
                    -1.02292156,
                    0.243167102,
                    0.457608819,
                    -1.36195457,
                    -1.75676513,
                    -0.984069586,
                    0.711804926,
                    0.938074708,
                    0.457659781,
                    -1.01670277,
                    -1.51119399,
                    -0.345314443,
                    0.361012787,
                    0.142928779,
                    -1.21289456,
                    -0.0796774626,
                    -1.14415085,
                    0.342464924,
                    -0.663234651,
                    -0.546800494,
                    0.86738503,
                    1.6715008,
                    0.165271237,
                    -0.131803751,
                    0.946529686,
                    -1.70487094,
                    0.149653986,
                    -1.81131935,
                    0.838590026,
                    -1.2080822,
                    -1.6054244,
                    -1.39167893,
                    0.350619376,
                    1.37737775,
                    -1.28133249,
                    -1.88975477,
                    -1.82942319,
                    0.393900275,
                    0.410033762,
                    0.807243109,
                    1.67445827,
                    -1.10450208,
                ],
                mean: &[
                    0.68718791,
                    0.673134565,
                    -0.620866835,
                    -1.17331791,
                    1.08936143,
                    0.630682826,
                    -0.546436191,
                    0.32781601,
                    -0.44427681,
                    0.211635351,
                    -0.550730348,
                    -0.286095381,
                    0.218149185,
                    0.921362281,
                    -0.277322888,
                    -0.604754627,
                    -0.970681071,
                    -0.792279482,
                    0.161362469,
                    -0.0397686958,
                    -0.806974709,
                    0.521737933,
                    1.05495059,
                    -0.233824193,
                ],
            },
            Golden {
                n: 3,
                width: 6,
                rows: 2,
                seed: 4242,
                a_z: 4.0,
                eta: 2.2,
                bz_scale: 3.0,
                pre: &[
                    0.469519854,
                    0.133475244,
                    0.210633248,
                    0.272467405,
                    0.711936414,
                    0.199169531,
                ],
                post: &[
                    1.92742085,
                    1.35964179,
                    1.95628345,
                    1.03980422,
                    0.278999448,
                    0.701378107,
                ],
                res: &[
                    0.810260057,
                    0.111238964,
                    0.0785009563,
                    0.00254762452,
                    0.73926264,
                    0.258189738,
                    0.187192291,
                    0.149498403,
                    0.663309336,
                    0.435182959,
                    0.0482073128,
                    0.516609728,
                    0.224748835,
                    0.765747547,
                    0.00950363558,
                    0.340068221,
                    0.186045155,
                    0.473886639,
                ],
                agg: &[
                    0.909330845,
                    0.862175405,
                    0.490297496,
                    0.253704369,
                    0.672205389,
                    -0.502939284,
                    -0.847711623,
                    -0.775278986,
                    -0.449140131,
                    0.943365276,
                    0.750931442,
                    -1.11962056,
                ],
                normed: &[
                    1.68474114,
                    1.09275103,
                    0.607933164,
                    0.429710835,
                    0.722293079,
                    -0.593182683,
                    -1.22672176,
                    -0.767485678,
                    -0.434975386,
                    1.24800098,
                    0.630229056,
                    -1.03140783,
                ],
                out: &[
                    3.33896542,
                    1.45697212,
                    -1.45483947,
                    0.855312467,
                    4.16964245,
                    0.478695333,
                    1.89940763,
                    0.263319016,
                    -1.03456509,
                    -1.63385403,
                    1.19627094,
                    0.862766147,
                    3.64270496,
                    1.82187629,
                    -3.40447378,
                    -0.81574589,
                    2.54115081,
                    1.07060599,
                    0.123226658,
                    0.991765857,
                    -0.157000542,
                    1.89255834,
                    1.02777123,
                    0.45317322,
                    -0.823739171,
                    -0.238167703,
                    -1.09742403,
                    1.24458706,
                    1.08939922,
                    -1.29700828,
                    -0.10965465,
                    0.438880146,
                    -0.126790404,
                    1.57063258,
                    0.749045253,
                    0.0741640329,
                ],
                mean: &[
                    0.922434747,
                    0.791223526,
                    0.421340466,
                    -0.43620348,
                    0.170867369,
                    -0.38654986,
                    -0.424372762,
                    -0.605155706,
                    0.290887117,
                    0.587668419,
                    0.479927868,
                    -0.292387724,
                ],
            },
            Golden {
                n: 4,
                width: 8,
                rows: 2,
                seed: 99,
                a_z: 0.02,
                eta: 0.0,
                bz_scale: 0.05,
                pre: &[
                    0.448961288,
                    0.0236460958,
                    0.766795278,
                    0.400038004,
                    0.431921214,
                    0.244790673,
                    0.521183312,
                    0.00825026818,
                ],
                post: &[
                    1.61947644,
                    0.0493777059,
                    0.658491671,
                    0.389941722,
                    0.110042654,
                    1.92855442,
                    1.99742436,
                    0.936429441,
                ],
                res: &[
                    0.824137568,
                    0.0523306169,
                    0.0574113727,
                    0.0661204383,
                    0.0572936274,
                    0.785259366,
                    0.0808200985,
                    0.0766269341,
                    0.0641834065,
                    0.0732666552,
                    0.813853025,
                    0.0486968793,
                    0.0543853939,
                    0.089143388,
                    0.0479154736,
                    0.808555722,
                    0.848221064,
                    0.0527512357,
                    0.0539243706,
                    0.0451032855,
                    0.0554237515,
                    0.784947753,
                    0.0708838627,
                    0.0887446553,
                    0.0471314564,
                    0.0887138471,
                    0.814940989,
                    0.0492137112,
                    0.0492236875,
                    0.0735871792,
                    0.0602507852,
                    0.816938341,
                ],
                agg: &[
                    0.441320002,
                    0.255412161,
                    -1.57581198,
                    -1.62454188,
                    -1.35981607,
                    1.19758749,
                    0.0411334857,
                    1.64547157,
                    0.385120213,
                    -0.0385340787,
                    0.594139636,
                    -0.37104404,
                    -0.616261303,
                    -0.221569315,
                    -0.00101112248,
                    -0.293822378,
                ],
                normed: &[
                    0.192729056,
                    0.243888125,
                    -1.01867843,
                    -1.99718559,
                    -0.913537502,
                    1.09168255,
                    0.0347124413,
                    1.53179562,
                    0.527632058,
                    -0.115434408,
                    1.20493186,
                    -1.43104804,
                    -1.29882991,
                    -0.633636117,
                    -0.00267691538,
                    -0.858097315,
                ],
                out: &[
                    -0.490446866,
                    -0.297446012,
                    0.867720604,
                    -2.62398386,
                    -0.172598928,
                    -1.48407841,
                    -2.72681642,
                    0.165469766,
                    1.07380056,
                    -0.211436361,
                    -1.63910079,
                    -1.33489609,
                    -0.803396583,
                    1.34428227,
                    -0.988729596,
                    0.122217685,
                    1.3392514,
                    1.14921844,
                    0.542475343,
                    -1.20946133,
                    -1.35719621,
                    0.8711555,
                    -0.0632094145,
                    0.672472358,
                    -1.41670704,
                    1.13056207,
                    -1.07718694,
                    -0.965471923,
                    -0.740394354,
                    0.680619597,
                    -0.30405122,
                    -0.322686851,
                    -0.584382892,
                    0.687366724,
                    0.478805304,
                    -0.86243093,
                    -1.61556768,
                    -0.659279108,
                    1.04441786,
                    1.3744607,
                    2.54376173,
                    0.112771153,
                    -1.32292902,
                    -0.0276374817,
                    -1.37408161,
                    0.98410511,
                    -1.68911064,
                    -1.55011725,
                    3.20564508,
                    0.925886452,
                    -0.643099964,
                    -1.88652205,
                    -0.740313053,
                    -0.315470994,
                    -3.51919889,
                    -1.72920609,
                    2.25161052,
                    1.42830968,
                    -0.162156671,
                    0.852460146,
                    0.283948362,
                    1.33998656,
                    -1.57883084,
                    -0.60518837,
                ],
                mean: &[
                    0.21415168,
                    -0.0778741837,
                    -1.34409404,
                    -1.12853873,
                    -0.704065919,
                    0.803502142,
                    -0.40871805,
                    0.669302583,
                    0.574731529,
                    0.14370662,
                    0.393069088,
                    0.473366737,
                    -0.158547103,
                    0.460294604,
                    0.014272809,
                    -0.32304132,
                ],
            },
            Golden {
                n: 4,
                width: 8,
                rows: 2,
                seed: 55,
                a_z: 0.0,
                eta: 0.7,
                bz_scale: 0.0,
                pre: &[
                    0.246410787,
                    0.54431361,
                    0.235094815,
                    0.919570506,
                    0.110849574,
                    0.0416835174,
                    0.250887543,
                    0.922190368,
                ],
                post: &[
                    0.919738889,
                    0.0183273014,
                    1.09275281,
                    0.649870217,
                    0.189598158,
                    0.373407483,
                    0.524890721,
                    1.91238332,
                ],
                res: &[
                    0.749429584,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.749429584,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.749429584,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.749429584,
                    0.749429584,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.749429584,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.749429584,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.0835234672,
                    0.749429584,
                ],
                agg: &[
                    1.89832187,
                    -2.39804077,
                    -1.40437341,
                    1.5407449,
                    -1.18341923,
                    1.91441822,
                    0.513446808,
                    -1.57232678,
                    -1.02117944,
                    -1.5897783,
                    1.44314921,
                    -1.06928408,
                    -0.925854445,
                    1.41060627,
                    -1.10564899,
                    -1.08306336,
                ],
                normed: &[
                    1.56487894,
                    -1.03418279,
                    -0.993220508,
                    1.31696582,
                    -1.01065576,
                    1.04004014,
                    0.323613435,
                    -0.580495656,
                    -1.12501407,
                    -0.916267574,
                    1.36401558,
                    -1.22146738,
                    -1.05670154,
                    1.02415133,
                    -0.931308031,
                    -0.534385979,
                ],
                out: &[
                    -0.482328415,
                    -0.703600705,
                    0.476818144,
                    1.37207794,
                    0.56655705,
                    0.785577238,
                    -1.1447866,
                    -1.62909031,
                    1.45570493,
                    -1.11466658,
                    -0.760612726,
                    1.51592577,
                    0.268646717,
                    0.829382181,
                    0.176461145,
                    -1.24759388,
                    -1.40873194,
                    1.46916962,
                    1.02992773,
                    1.93252265,
                    0.478199333,
                    -0.00613094866,
                    -0.0892922133,
                    -2.78194356,
                    0.134305358,
                    -1.06451285,
                    0.250831723,
                    1.43311131,
                    -0.522029936,
                    1.23584783,
                    0.673189819,
                    -1.59520197,
                    -0.303854644,
                    0.00266804546,
                    -0.684461653,
                    0.00215056539,
                    -0.112513267,
                    0.820779085,
                    -0.381494194,
                    1.50847912,
                    -0.0304244757,
                    0.133822337,
                    1.63189578,
                    0.0771710128,
                    1.23328876,
                    0.879969835,
                    0.313114941,
                    1.98203635,
                    -0.566402495,
                    -0.0359834731,
                    0.722669303,
                    1.3325274,
                    0.178230584,
                    -0.553857028,
                    0.0366954803,
                    1.57358766,
                    1.52850163,
                    -0.196898818,
                    4.07132912,
                    0.191013217,
                    -0.933837175,
                    0.149047494,
                    1.86847436,
                    1.30012941,
                ],
                mean: &[
                    0.690827727,
                    -0.775579751,
                    -0.721412718,
                    0.650178552,
                    -0.392724931,
                    0.667224944,
                    -0.231639028,
                    -0.815910697,
                    -0.707220674,
                    -0.438872814,
                    0.34222424,
                    -0.0909160376,
                    0.185738087,
                    0.772555709,
                    -0.589222372,
                    0.718839765,
                ],
            },
        ];
        for g in goldens.iter() {
            let (n, c, rows) = (g.n, g.width, g.rows);
            let nm2 = (n - 1) * (n - 1);
            let zdim = 2 * n + nm2;
            let cols = n * c;
            let mut r = Lcg(g.seed);
            let phi = r.vec(zdim * cols, 1.5);
            let x = r.vec(rows * cols, 4.0);
            let b_pre = r.vec(n, 1.0);
            let b_post = r.vec(n, 1.0);
            let b_z = r.vec(nm2, g.bz_scale);
            let gamma: Vec<f32> = r.vec(c, 1.0).into_iter().map(|v| 1.0 + v).collect();
            let branch = r.vec(rows * c, 3.0);
            let mut params = vec![0.7f32, 1.3, g.a_z, g.eta];
            params.extend(&b_pre);
            params.extend(&b_post);
            params.extend(&b_z);
            let (mut pre, mut post, mut res) = (vec![], vec![], vec![]);
            let (mut agg, mut normed, mut out, mut mean) = (vec![], vec![], vec![], vec![]);
            for row in 0..rows {
                let xr = &x[row * cols..(row + 1) * cols];
                // selector projection (the GEMM stage): z_raw = phi . x
                let z_raw: Vec<f32> = (0..zdim)
                    .map(|k| (0..cols).map(|d| phi[k * cols + d] * xr[d]).sum())
                    .collect();
                let (p, q, h) =
                    route_helmert(n, xr, &z_raw, &params, 2.0, 1e-5, StreamRounding::F32);
                agg.extend(aggregate(n, c, xr, &p, None, StreamRounding::F32));
                normed.extend(aggregate(
                    n,
                    c,
                    xr,
                    &p,
                    Some((&gamma, 1e-5)),
                    StreamRounding::F32,
                ));
                // reference `normed` here is agg * rsqrt(mean(agg^2)+eps) * gamma
                out.extend(postmix(
                    n,
                    c,
                    xr,
                    &branch[row * c..(row + 1) * c],
                    &q,
                    &h,
                    StreamRounding::F32,
                ));
                mean.extend(mean_of(n, c, xr));
                pre.extend(p);
                post.extend(q);
                res.extend(h);
            }
            assert_close("pre", &pre, g.pre, 2e-5);
            assert_close("post", &post, g.post, 2e-5);
            assert_close("res", &res, g.res, 5e-5);
            assert_close("agg", &agg, g.agg, 1e-4);
            assert_close("normed", &normed, g.normed, 1e-4);
            assert_close("x_out", &out, g.out, 1e-4);
            assert_close("mean", &mean, g.mean, 1e-5);
        }
    }

    fn mean_of(n: usize, width: usize, x: &[f32]) -> Vec<f32> {
        mean(n, width, x, None, StreamRounding::F32)
    }
}
