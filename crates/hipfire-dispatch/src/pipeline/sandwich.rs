// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Sandwich-norm decoder layer operations.
//!
//! A sandwich-norm decoder normalizes each sublayer's input AND its output
//! before the residual add (`x = residual + post_norm(f(pre_norm(x)))`).
//! Architecture crates declare one attention and one MLP operation per layer
//! (plus an optional per-layer-input branch and layer scale) and bind their
//! resident weights, KV storage and fixed scratch; every dtype-, tier-, arch-
//! and fusion-gated route choice lives here.
//!
//! Executors take `rows`: `rows == 1` is single-token decode, `rows > 1` a
//! batch of consecutive positions (prefill or speculative verify) laid out
//! row-major. The bodies are the former Gemma 4 eager decode and batched
//! verify arms moved verbatim, so every route issues the same launches.

use crate::context::DispatchCtx;
use crate::families::attention::{AttentionFamily, AttnParams};
use crate::families::fused_qkv::{FusedQkvFamily, FusedQkvParams};
use crate::families::gemm::{GemmFamily, GemmParams};
use crate::families::gemv::{GemvFamily, GemvParams, WeightRef};
use crate::families::kv_tier::{KvTierInputs, KvTierPlan};
use crate::types::{DispatchError, GemvVariant, KernelKey};
use hip_bridge::DeviceBuffer;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::LazyLock;

fn hip(e: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

fn hip_dbg(e: impl std::fmt::Debug) -> DispatchError {
    DispatchError::Hip(format!("{e:?}"))
}

/// Greedy EAGLE (`HIPFIRE_GEMMA4_EAGLE=1`) turns off the fusions that are not
/// byte-identical to their unfused form and keeps batched Q8 GEMMs on the
/// unchunked scalar kernel, so batched verify and single-row decode share one
/// arithmetic.
static EAGLE_STRICT: LazyLock<bool> = LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_GEMMA4_EAGLE")
        .ok()
        .as_deref()
        == Some("1")
});

static GEMV: LazyLock<GemvFamily> = LazyLock::new(GemvFamily::new);
static GEMM: LazyLock<GemmFamily> = LazyLock::new(GemmFamily::new);
static FUSED_QKV: LazyLock<FusedQkvFamily> = LazyLock::new(FusedQkvFamily::new);
static ATTENTION: LazyLock<AttentionFamily> = LazyLock::new(AttentionFamily::new);

/// Arches whose batched sandwich prefill admits the exact fused and wide
/// routes below. Elsewhere batched rows take the per-row fallbacks.
fn batched_fusion_arch(arch: &str) -> bool {
    matches!(arch, "gfx1100" | "gfx1201")
}

/// True while a calibration collector is armed. Routes that never
/// materialize a projection's unrotated input (fused norm+FWHT) step aside so
/// every projection reaches its `maybe_capture_activation` tap.
fn calibrating(gpu: &Gpu) -> bool {
    gpu.active_capture.is_some()
}

/// `y = W x` for one row. Taps the calibration collector with the unrotated
/// input.
fn gemv(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &WeightRef,
    x: &GpuTensor,
    y: &GpuTensor,
) -> Result<(), DispatchError> {
    if w.dtype == DType::BF16 {
        return gemm_rows(gpu, ctx, w, x, y, x, 1);
    }
    gpu.maybe_capture_activation(w.buf, x, 1, w.k);
    GEMV.run_auto(ctx, gpu, w, x, y)
}

fn gemv_prerotated(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &WeightRef,
    x_rot: &GpuTensor,
    y: &GpuTensor,
) -> Result<(), DispatchError> {
    GEMV.run(
        ctx,
        gpu,
        &GemvParams {
            w,
            x: x_rot,
            y,
            variant: GemvVariant::Prerotated,
            residual: None,
            gate: None,
            up: None,
        },
    )
}

/// FWHT rotation of `rows` rows of `x` for the MQ weight `w` (AWQ-aware).
fn rotate_x_mq_rows(
    gpu: &mut Gpu,
    w: &WeightRef,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    rows: usize,
) -> Result<(), DispatchError> {
    match w.awq_scale {
        Some(awq) => gpu.rotate_x_mq_awq_batched(x, awq, x_rot, w.k, rows),
        None => gpu.rotate_x_mq_batched(x, x_rot, w.k, rows),
    }
    .map_err(hip)
}

/// `y[rows, m] = x[rows, k] · Wᵀ`. MagnumQuant weights rotate `x` into
/// `x_rot` (`[rows, k]`) first. Formats without a batched kernel run one
/// GEMV per row. Taps the calibration collector with the unrotated input.
pub fn gemm_rows(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &WeightRef,
    x: &GpuTensor,
    y: &GpuTensor,
    x_rot: &GpuTensor,
    rows: usize,
) -> Result<(), DispatchError> {
    if !supports_batched_projection(w.dtype) {
        for row in 0..rows {
            let x_row = x.sub_offset(row * w.k, w.k);
            let y_row = y.sub_offset(row * w.m, w.m);
            gemv(gpu, ctx, w, &x_row, &y_row)?;
        }
        return Ok(());
    }
    gpu.maybe_capture_activation(w.buf, x, rows, w.k);
    match w.dtype {
        DType::F32 => gpu
            .gemm_f32_batched(w.buf, x, y, w.m, w.k, rows)
            .map_err(hip),
        DType::Q8_0 if *EAGLE_STRICT => {
            // The scalar kernel holds at most 64 rows.
            for off in (0..rows).step_by(64) {
                let n = (rows - off).min(64);
                gpu.gemm_q8_0_batched(
                    w.buf,
                    &x.sub_offset(off * w.k, n * w.k),
                    &y.sub_offset(off * w.m, n * w.m),
                    w.m,
                    w.k,
                    n,
                )
                .map_err(hip)?;
            }
            Ok(())
        }
        // gfx1151: the 4-warp 64x64 tile reads each weight tile once per 64
        // rows (the 16x16 kernel re-reads it per 16 rows); a sub-64 row tail
        // takes the 16x16 kernel.
        DType::Q8_0 if gpu.arch_caps.is_gfx1151() && w.m % 64 == 0 && rows >= 64 => {
            let full = rows / 64 * 64;
            gpu.gemm_q8_0_wmma_4w(w.buf, &x.sub_offset(0, full * w.k), y, w.m, w.k, full)
                .map_err(hip)?;
            if full < rows {
                let tail = rows - full;
                gpu.gemm_q8_0_wmma(
                    w.buf,
                    &x.sub_offset(full * w.k, tail * w.k),
                    &y.sub_offset(full * w.m, tail * w.m),
                    w.m,
                    w.k,
                    tail,
                )
                .map_err(hip)?;
            }
            Ok(())
        }
        DType::Q8_0 => gpu
            .gemm_q8_0_batched_chunked(w.buf, x, y, w.m, w.k, rows)
            .map_err(hip),
        DType::MQ4G256 | DType::HFQ4G256 => {
            rotate_x_mq_rows(gpu, w, x, x_rot, rows)?;
            gpu.gemm_hfq4g256_batched_lmhead(w.buf, x_rot, y, w.m, w.k, rows)
                .map_err(hip)
        }
        DType::MQ6G256 => {
            rotate_x_mq_rows(gpu, w, x, x_rot, rows)?;
            gpu.gemm_mq6g256_batched_lmhead(w.buf, x_rot, y, w.m, w.k, rows)
                .map_err(hip)
        }
        DType::MQ4G256V2 => {
            rotate_x_mq_rows(gpu, w, x, x_rot, rows)?;
            gpu.gemm_mq4g256v2(w.buf, x_rot, y, w.m, w.k, rows)
                .map_err(hip)
        }
        // BF16 teachers (calibration): stage the F32 input to BF16 for the
        // MFMA GEMM, which resolves only where the arch has one.
        // ponytail: per-call staging buffer; hold one in scratch if BF16
        // calibration ever becomes throughput-bound.
        DType::BF16 => {
            let n = rows * w.k;
            let staged = gpu.alloc_tensor(&[n], DType::BF16).map_err(hip)?;
            let run = gpu
                .convert_f32_to_bf16(x, &staged, n)
                .map_err(hip)
                .and_then(|_| {
                    GEMM.run_key(
                        KernelKey::GemmBf16Mfma,
                        ctx,
                        gpu,
                        &GemmParams {
                            w,
                            x: &staged,
                            y,
                            batch_size: rows,
                        },
                    )
                });
            gpu.free_tensor(staged).map_err(hip)?;
            run
        }
        other => unreachable!("{other:?} is not a batched projection format"),
    }
}

/// Weight formats [`gemm_rows`] can project with `rows > 1`.
pub fn supports_batched_projection(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32
            | DType::BF16
            | DType::Q8_0
            | DType::MQ4G256
            | DType::HFQ4G256
            | DType::MQ6G256
            | DType::MQ4G256V2
    )
}

fn fused_projection(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    key: KernelKey,
    weights: &[&WeightRef],
    x: &GpuTensor,
    outputs: &[&GpuTensor],
    rows: usize,
) -> Result<(), DispatchError> {
    let bufs: Vec<&GpuTensor> = weights.iter().map(|w| w.buf).collect();
    let m: Vec<usize> = weights.iter().map(|w| w.m).collect();
    FUSED_QKV.run(
        ctx,
        gpu,
        &FusedQkvParams {
            kind: key,
            weights: &bufs,
            x,
            outputs,
            m: &m,
            k: weights[0].k,
            rot_scratch: &[],
            batch_size: (rows > 1).then_some(rows),
        },
    )
}

/// gfx1100 opt-in: Q8 projections sharing one input run as one fused WMMA GEMM.
fn q8_fused_prefill(gpu: &Gpu, weights: &[&WeightRef]) -> bool {
    gpu.arch == "gfx1100"
        && gpu.flags.gemma4_q8_fused_prefill
        && weights
            .iter()
            .all(|w| w.dtype == DType::Q8_0 && w.k == weights[0].k)
        && weights[0].k.is_multiple_of(32)
}

fn scale(gpu: &mut Gpu, x: &GpuTensor, factor: f32) -> Result<(), DispatchError> {
    rdna_compute::tensor_ops::scale_f32(
        gpu,
        &rdna_compute::tensor_ops::ScaleF32 {
            values: x,
            scale: factor,
        },
    )
    .map_err(hip)
}

/// Residual stream and the fixed scratch every sandwich sublayer shares.
/// With `rows > 1` each tensor holds `rows` row-major rows.
#[derive(Clone, Copy)]
pub struct SandwichStream<'a> {
    pub rows: usize,
    pub hidden: usize,
    pub eps: f32,
    /// Residual stream; updated in place.
    pub x: &'a GpuTensor,
    /// Holds `x` at sublayer entry.
    pub residual: &'a GpuTensor,
    /// Normed activation / sublayer output scratch.
    pub normed: &'a GpuTensor,
}

impl SandwichStream<'_> {
    fn len(&self) -> usize {
        self.rows * self.hidden
    }

    /// Copies are kernel launches so hipGraph capture and the retained-replay
    /// recorder both see them (a D2D memcpy is invisible to Redline PM4).
    fn save_residual(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        gpu.copy_f32_buffer(self.residual, self.x, self.len())
            .map_err(hip)
    }

    fn restore_residual(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        gpu.copy_f32_buffer(self.x, self.residual, self.len())
            .map_err(hip)
    }

    fn norm(
        &self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        w: &GpuTensor,
        out: &GpuTensor,
    ) -> Result<(), DispatchError> {
        if self.rows == 1 {
            gpu.rmsnorm_f32(x, w, out, self.eps)
        } else {
            gpu.rmsnorm_batched(x, w, out, self.rows, self.hidden, self.eps)
        }
        .map_err(hip)
    }

    /// `x = residual + post_norm(out)`; `out` may alias `normed`.
    fn post_norm_residual(
        &self,
        gpu: &mut Gpu,
        out: &GpuTensor,
        post_norm: &GpuTensor,
    ) -> Result<(), DispatchError> {
        if self.rows == 1 && !*EAGLE_STRICT {
            return gpu
                .rmsnorm_residual_add_f32(out, post_norm, self.residual, self.x, self.eps)
                .map_err(hip);
        }
        self.norm(gpu, out, post_norm, self.normed)?;
        self.restore_residual(gpu)?;
        gpu.add_inplace_f32(self.x, self.normed).map_err(hip)
    }
}

/// Rotary position embedding applied to Q and K.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeKind {
    /// Every channel pair of the head rotates.
    RotateHalf,
    /// The first `rot_pairs` pairs of each half rotate; the rest pass through.
    PartialHalved { rot_pairs: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    pub kind: RopeKind,
    pub theta: f32,
}

impl Rope {
    fn rot_pairs(&self, head_dim: usize) -> usize {
        match self.kind {
            RopeKind::RotateHalf => head_dim / 2,
            RopeKind::PartialHalved { rot_pairs } => rot_pairs,
        }
    }
}

/// The K/V cache one attention sublayer reads. A sublayer with its own
/// [`KvProjection`] writes the current rows first; a query-only sublayer reads
/// another layer's populated cache (KV sharing, draft heads).
pub struct SandwichKv<'a> {
    pub tier: KvTierInputs,
    pub k_cache: &'a GpuTensor,
    pub v_cache: &'a GpuTensor,
    pub physical_cap: usize,
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
    /// Sliding window; `0` = full causal.
    pub window: usize,
}

/// K/V projections of an attention sublayer that writes its own cache rows.
pub struct KvProjection<'a> {
    pub wk: WeightRef<'a>,
    /// `None` takes V from the pre-norm K (K=V).
    pub wv: Option<WeightRef<'a>>,
    pub k_norm: &'a GpuTensor,
    /// Weight-less V RMSNorm, bound as a ones vector of `head_dim`.
    pub v_norm: Option<&'a GpuTensor>,
}

/// Attention sublayer:
/// `x = residual + post_norm(o(attend(rope(q_norm(q)), rope(k_norm(k)), v_norm(v))))`
/// over `input_norm(x)`.
///
/// `kv_proj == None` projects only Q and attends a cache it does not write.
/// Row `r` sits at absolute position `position + r`.
pub struct SandwichAttentionOp<'a> {
    pub stream: SandwichStream<'a>,
    pub position: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub input_norm: &'a GpuTensor,
    pub wq: WeightRef<'a>,
    pub q_norm: &'a GpuTensor,
    pub kv_proj: Option<KvProjection<'a>>,
    /// Q multiplier applied after q_norm (sets the softmax scale against the
    /// kernel's `1/sqrt(head_dim)`).
    pub q_scale: f32,
    pub rope: Rope,
    pub kv: SandwichKv<'a>,
    /// Device position scalar (`rows == 1`).
    pub pos_buf: &'a DeviceBuffer,
    /// Per-row i32 positions (`rows > 1`).
    pub positions: Option<&'a GpuTensor>,
    pub wo: WeightRef<'a>,
    pub post_norm: &'a GpuTensor,
    /// FWHT rotation scratch, `[rows, max(hidden, n_heads*head_dim)]`.
    pub x_rot: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    pub flash_partials: &'a GpuTensor,
}

impl SandwichAttentionOp<'_> {
    fn wk(&self) -> Option<&WeightRef<'_>> {
        self.kv_proj.as_ref().map(|p| &p.wk)
    }

    fn wv(&self) -> Option<&WeightRef<'_>> {
        self.kv_proj.as_ref().and_then(|p| p.wv.as_ref())
    }

    fn projections(&self) -> [Option<(&WeightRef<'_>, &GpuTensor)>; 3] {
        [
            Some((&self.wq, self.q)),
            self.wk().map(|w| (w, self.k)),
            self.wv().map(|w| (w, self.v)),
        ]
    }

    fn kv_len(&self) -> usize {
        self.stream.rows * self.n_kv_heads * self.head_dim
    }

    /// `input_norm(x)` → Q, K, V projections (V from K when K=V).
    fn project(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        let s = &self.stream;
        if s.rows > 1 {
            return self.project_rows(gpu, ctx);
        }
        let all_mq4 = self
            .projections()
            .iter()
            .flatten()
            .all(|(w, _)| w.dtype == DType::MQ4G256 && w.awq_scale.is_none());
        if all_mq4 && !calibrating(gpu) {
            gpu.fused_rmsnorm_rotate_mq(s.x, self.input_norm, self.x_rot, s.hidden, s.eps)
                .map_err(hip)?;
            for (w, out) in self.projections().iter().flatten() {
                gemv_prerotated(gpu, ctx, w, self.x_rot, out)?;
            }
        } else {
            s.norm(gpu, s.x, self.input_norm, s.normed)?;
            match self.wk() {
                Some(wk) if self.wq.dtype == DType::Q8_0 && wk.dtype == DType::Q8_0 => {
                    fused_projection(
                        gpu,
                        ctx,
                        KernelKey::FusedGateUpQ8_0,
                        &[&self.wq, wk],
                        s.normed,
                        &[self.q, self.k],
                        1,
                    )?;
                }
                Some(wk) => {
                    gemv(gpu, ctx, &self.wq, s.normed, self.q)?;
                    gemv(gpu, ctx, wk, s.normed, self.k)?;
                }
                None => gemv(gpu, ctx, &self.wq, s.normed, self.q)?,
            }
            if let Some(wv) = self.wv() {
                gemv(gpu, ctx, wv, s.normed, self.v)?;
            }
        }
        if self.wk().is_some() && self.wv().is_none() {
            gpu.copy_f32_buffer(self.v, self.k, self.kv_len())
                .map_err(hip)?;
        }
        Ok(())
    }

    fn project_rows(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        let s = &self.stream;
        let rows = s.rows;
        s.norm(gpu, s.x, self.input_norm, s.normed)?;
        let mut v_projected = false;
        match (self.wk(), self.wv()) {
            (Some(wk), Some(wv)) if q8_fused_prefill(gpu, &[&self.wq, wk, wv]) => {
                fused_projection(
                    gpu,
                    ctx,
                    KernelKey::FusedQkvQ8_0,
                    &[&self.wq, wk, wv],
                    s.normed,
                    &[self.q, self.k, self.v],
                    rows,
                )?;
                v_projected = true;
            }
            (Some(wk), None) if q8_fused_prefill(gpu, &[&self.wq, wk]) => {
                fused_projection(
                    gpu,
                    ctx,
                    KernelKey::FusedGateUpQ8_0,
                    &[&self.wq, wk],
                    s.normed,
                    &[self.q, self.k],
                    rows,
                )?;
            }
            (wk, wv) => {
                gemm_rows(gpu, ctx, &self.wq, s.normed, self.q, self.x_rot, rows)?;
                if let Some(wk) = wk {
                    gemm_rows(gpu, ctx, wk, s.normed, self.k, self.x_rot, rows)?;
                }
                if let (Some(_), Some(wv)) = (wk, wv) {
                    gemm_rows(gpu, ctx, wv, s.normed, self.v, self.x_rot, rows)?;
                    v_projected = true;
                }
            }
        }
        if self.wk().is_some() && !v_projected {
            gpu.copy_f32_buffer(self.v, self.k, self.kv_len())
                .map_err(hip)?;
        }
        Ok(())
    }

    /// Per-head q_norm/k_norm, weight-less v_norm, Q prescale, RoPE.
    fn position_encode(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let eps = self.stream.eps;
        let rows = self.stream.rows;
        let (n_heads, n_kv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);
        if let Some(ones) = self.kv_proj.as_ref().and_then(|p| p.v_norm) {
            gpu.rmsnorm_batched(self.v, ones, self.v, rows * n_kv, hd, eps)
                .map_err(hip)?;
        }
        if let (1, Some(p), true) = (rows, &self.kv_proj, !*EAGLE_STRICT) {
            return gpu
                .fused_gemma4_qk_norm_rope_f32(
                    self.q,
                    self.k,
                    self.q_norm,
                    p.k_norm,
                    self.pos_buf,
                    n_heads,
                    n_kv,
                    hd,
                    self.rope.rot_pairs(hd),
                    self.q_scale,
                    self.rope.theta,
                    eps,
                )
                .map_err(hip);
        }
        gpu.rmsnorm_batched(self.q, self.q_norm, self.q, rows * n_heads, hd, eps)
            .map_err(hip)?;
        if let Some(p) = &self.kv_proj {
            gpu.rmsnorm_batched(self.k, p.k_norm, self.k, rows * n_kv, hd, eps)
                .map_err(hip)?;
        }
        scale(gpu, self.q, self.q_scale)?;
        let rope_kv = if self.kv_proj.is_some() { n_kv } else { 0 };
        let theta = self.rope.theta;
        match (self.rope.kind, rows) {
            (RopeKind::RotateHalf, 1) => {
                gpu.rope_f32(self.q, self.k, self.pos_buf, n_heads, rope_kv, hd, theta)
            }
            (RopeKind::PartialHalved { rot_pairs }, 1) => gpu.rope_partial_halved_f32(
                self.q,
                self.k,
                self.pos_buf,
                n_heads,
                rope_kv,
                hd,
                rot_pairs,
                theta,
            ),
            (RopeKind::RotateHalf, _) => gpu.rope_batched_f32(
                self.q,
                self.k,
                self.positions()?,
                n_heads,
                rope_kv,
                hd,
                theta,
                rows,
            ),
            (RopeKind::PartialHalved { rot_pairs }, _) => gpu.rope_partial_halved_f32_batched(
                self.q,
                self.k,
                self.positions()?,
                n_heads,
                rope_kv,
                hd,
                rot_pairs,
                theta,
                rows,
            ),
        }
        .map_err(hip)
    }

    fn positions(&self) -> Result<&GpuTensor, DispatchError> {
        self.positions.ok_or_else(|| {
            DispatchError::Hip("sandwich attention: rows > 1 needs per-row positions".into())
        })
    }

    fn attend(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        if self.stream.rows > 1 {
            return self.attend_rows(gpu);
        }
        let kv = &self.kv;
        let plan = KvTierPlan::derive(KvTierInputs {
            pos: self.position,
            batch_size: 1,
            q8_windowed: true,
            window: kv.window as i32,
            ..kv.tier
        })
        .map_err(hip)?;
        let io = AttnParams {
            q: self.q,
            k: self.k,
            v: self.v,
            k_cache: kv.k_cache,
            v_cache: kv.v_cache,
            k_scales: None,
            v_scales: None,
            pos_buf: self.pos_buf,
            pos: self.position,
            positions: None,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            physical_cap: kv.physical_cap,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(self.flash_partials),
            givens_cos: kv.givens_cos,
            givens_sin: kv.givens_sin,
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output_awq_scale: None,
            output: self.attn_out,
        };
        if self.kv_proj.is_some() {
            ATTENTION.run_attention(ctx, gpu, &plan, &io)
        } else {
            ATTENTION.run_attend_only(ctx, gpu, &plan, &io)
        }
    }

    /// Causal attention for `rows` consecutive positions over a Q8 cache.
    fn attend_rows(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let kv = &self.kv;
        if !kv.tier.quant_q8 {
            return Err(DispatchError::Hip(
                "sandwich attention: rows > 1 needs a Q8 KV cache".into(),
            ));
        }
        let rows = self.stream.rows;
        let positions = self.positions()?;
        let (n_heads, n_kv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);
        let seq_len = self.position + rows;
        if self.kv_proj.is_some() {
            gpu.kv_cache_write_q8_0_batched(kv.k_cache, self.k, positions, n_kv, hd, rows)
                .map_err(hip)?;
            gpu.kv_cache_write_q8_0_batched(kv.v_cache, self.v, positions, n_kv, hd, rows)
                .map_err(hip)?;
        }
        // Prefill is not tree verification: the native causal kernels, and the
        // windowed tile kernel for sliding layers. gfx1151 takes the
        // GQA-shared WMMA flash kernel for every layer.
        if gpu.attention_q8_0_prefill_gqa_wmma_admitted(n_heads, n_kv, hd) {
            gpu.attention_q8_0_prefill_gqa_wmma(
                self.q,
                kv.k_cache,
                kv.v_cache,
                self.attn_out,
                positions,
                n_heads,
                n_kv,
                hd,
                rows,
                kv.window as i32,
            )
        } else if kv.window > 0 {
            gpu.attention_flash_q8_0_batched_masked_windowed(
                self.q,
                kv.k_cache,
                kv.v_cache,
                self.attn_out,
                positions,
                n_heads,
                n_kv,
                hd,
                kv.physical_cap,
                seq_len,
                rows,
                self.flash_partials,
                None,
                0,
                0,
                kv.window as i32,
            )
        } else if seq_len > 8_192 {
            gpu.attention_flash_q8_0_batched_masked(
                self.q,
                kv.k_cache,
                kv.v_cache,
                self.attn_out,
                positions,
                n_heads,
                n_kv,
                hd,
                kv.physical_cap,
                seq_len,
                rows,
                self.flash_partials,
                None,
                0,
                0,
            )
        } else {
            gpu.attention_q8_0_kv_batched_masked(
                self.q,
                kv.k_cache,
                kv.v_cache,
                self.attn_out,
                positions,
                n_heads,
                n_kv,
                hd,
                kv.physical_cap,
                seq_len,
                rows,
                None,
                0,
                0,
            )
        }
        .map_err(hip)
    }
}

pub fn execute_sandwich_attention(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichAttentionOp<'_>,
) -> Result<(), DispatchError> {
    let s = op.stream;
    s.save_residual(gpu)?;
    op.project(gpu, ctx)?;
    op.position_encode(gpu)?;
    op.attend(gpu, ctx)?;
    if s.rows == 1 {
        gemv(gpu, ctx, &op.wo, op.attn_out, s.normed)?;
    } else {
        gemm_rows(gpu, ctx, &op.wo, op.attn_out, s.normed, op.x_rot, s.rows)?;
    }
    s.post_norm_residual(gpu, s.normed, op.post_norm)
}

/// Gated MLP activation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    /// `gelu_pytorch_tanh(gate) * up`.
    GeluTanh,
}

/// MLP sublayer: `x = residual + post_norm(down(act(gate(n)) * up(n)))`,
/// `n = pre_norm(x)`.
pub struct SandwichMlpOp<'a> {
    pub stream: SandwichStream<'a>,
    pub pre_norm: &'a GpuTensor,
    pub w_gate: WeightRef<'a>,
    pub w_up: WeightRef<'a>,
    pub w_down: WeightRef<'a>,
    pub activation: Activation,
    pub hidden_dim: usize,
    pub post_norm: &'a GpuTensor,
    /// FWHT rotation scratch, `[rows, max(hidden, hidden_dim)]`.
    pub x_rot: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub out: &'a GpuTensor,
}

pub fn execute_sandwich_mlp(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichMlpOp<'_>,
) -> Result<(), DispatchError> {
    dense_mlp(gpu, ctx, op)?;
    op.stream.post_norm_residual(gpu, op.out, op.post_norm)
}

/// Residual save, then `out = down(act(gate(n)) * up(n))`, `n = pre_norm(x)`.
fn dense_mlp(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichMlpOp<'_>,
) -> Result<(), DispatchError> {
    let s = op.stream;
    let rows = s.rows;
    s.save_residual(gpu)?;
    if rows > 1 {
        s.norm(gpu, s.x, op.pre_norm, s.normed)?;
        if q8_fused_prefill(gpu, &[&op.w_gate, &op.w_up]) {
            fused_projection(
                gpu,
                ctx,
                KernelKey::FusedGateUpQ8_0,
                &[&op.w_gate, &op.w_up],
                s.normed,
                &[op.gate, op.up],
                rows,
            )?;
        } else {
            gemm_rows(gpu, ctx, &op.w_gate, s.normed, op.gate, op.x_rot, rows)?;
            gemm_rows(gpu, ctx, &op.w_up, s.normed, op.up, op.x_rot, rows)?;
        }
    } else if op.w_gate.dtype == DType::MQ4G256
        && op.w_up.dtype == DType::MQ4G256
        && !calibrating(gpu)
    {
        gpu.fused_rmsnorm_rotate_mq(s.x, op.pre_norm, op.x_rot, s.hidden, s.eps)
            .map_err(hip)?;
        gpu.fused_gate_up_hfq4g256(
            op.w_gate.buf,
            op.w_up.buf,
            op.x_rot,
            op.gate,
            op.up,
            op.w_gate.m,
            op.w_up.m,
            op.w_gate.k,
        )
        .map_err(hip)?;
    } else {
        s.norm(gpu, s.x, op.pre_norm, s.normed)?;
        gemv(gpu, ctx, &op.w_gate, s.normed, op.gate)?;
        gemv(gpu, ctx, &op.w_up, s.normed, op.up)?;
    }
    match op.activation {
        Activation::GeluTanh => {
            gpu.gelu_tanh_f32(op.gate, op.act, rows * op.hidden_dim)
                .map_err(hip)?;
            gpu.mul_f32(op.act, op.up, op.act).map_err(hip)?;
        }
    }
    if rows == 1 {
        gemv(gpu, ctx, &op.w_down, op.act, op.out)?;
    } else {
        gemm_rows(gpu, ctx, &op.w_down, op.act, op.out, op.x_rot, rows)?;
    }
    Ok(())
}

/// Routed-expert half of a parallel dense+MoE block.
pub struct RoutedExperts<'a> {
    /// Norm of the residual stream feeding the experts.
    pub pre_norm: &'a GpuTensor,
    /// Router input norm weight; the normed input is scaled by
    /// `router_input_scale` before the router projection.
    pub router_norm: &'a GpuTensor,
    pub router_input_scale: f32,
    pub router: WeightRef<'a>,
    pub n_experts: usize,
    pub top_k: usize,
    /// Per-expert FFN width.
    pub hidden_dim: usize,
    pub gate_up_dtype: DType,
    pub down_dtype: DType,
    /// `[n_experts]` device pointers to each expert's fused gate/up weight.
    pub gate_up_ptrs: &'a GpuTensor,
    /// `[n_experts]` device pointers to each expert's down weight.
    pub down_ptrs: &'a GpuTensor,
    /// `[n_experts]` learned scale applied to each expert's output.
    pub per_expert_scale: &'a GpuTensor,
    /// Norm of the combined expert output.
    pub post_norm: &'a GpuTensor,
}

impl RoutedExperts<'_> {
    /// Expert formats with an indexed device-resident kernel pair.
    pub fn supports(gate_up: DType, down: DType) -> bool {
        matches!(
            gate_up,
            DType::MQ4G256
                | DType::MQ4G256V2
                | DType::MQ6G256
                | DType::HFQ4G256
                | DType::HFQ6G256
                | DType::Q8_0
        ) && matches!(down, DType::Q8_0 | DType::HFQ4G128)
    }

    /// Formats whose batched rows take the grouped (sorted-by-expert) gate/up
    /// GEMM: each expert's weights are read once per batch instead of once
    /// per routed row.
    pub fn grouped_rows(gate_up: DType, down: DType) -> bool {
        matches!(gate_up, DType::MQ4G256V2 | DType::MQ6G256) && down == DType::HFQ4G128
    }
}

/// Batched-rows scratch of the grouped routed-expert route (`rows` rows,
/// `slots = rows * top_k`, `m_total = slots + n_experts * MOE_GROUPED_BLOCK_M`).
/// Index buffers hold i32.
#[derive(Clone, Copy)]
pub struct RoutedBatch<'a> {
    /// `[rows, hidden]`: router input, then `pre_norm(residual)`.
    pub input: &'a GpuTensor,
    /// `[rows, hidden]` FWHT rotation (router `x_rot` scratch first).
    pub input_rot: &'a GpuTensor,
    /// `[rows, n_experts]`.
    pub router_logits: &'a GpuTensor,
    /// `[slots]` each.
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    /// `[n_experts]`, `[n_experts + 1]`.
    pub counts: &'a GpuTensor,
    pub offsets: &'a GpuTensor,
    /// `[m_total]`, `[m_total / MOE_GROUPED_BLOCK_M]`.
    pub sorted_slots: &'a GpuTensor,
    pub tile_ids: &'a GpuTensor,
    /// `[m_total, max(2 * hidden_dim, hidden)]`: grouped gate/up rows, then
    /// grouped down rows.
    pub gate_up_grouped: &'a GpuTensor,
    /// `[slots, hidden_dim]` each; `gate` ends as the activation.
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    /// `[rows, hidden]` combined expert output.
    pub out: &'a GpuTensor,
}

impl RoutedBatch<'_> {
    /// Grouped rows (with padding) for `rows` rows: a whole number of
    /// 16-slot tiles, because the grouped kernels launch `ceil(m_total / 16)`
    /// tiles and read one tile id each.
    pub fn m_total(rows: usize, top_k: usize, n_experts: usize) -> usize {
        let block = crate::families::moe::MOE_GROUPED_BLOCK_M;
        (rows * top_k).next_multiple_of(block) + n_experts * block
    }
}

/// Scratch of one routed-expert pass.
#[derive(Clone, Copy)]
pub struct RoutedScratch<'a> {
    /// `pre_norm(residual)`, and its FWHT rotation for MagnumQuant experts.
    pub input: &'a GpuTensor,
    pub input_rot: &'a GpuTensor,
    pub router_in: &'a GpuTensor,
    pub router_logits: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    /// `[top_k, hidden_dim]` each.
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub act: &'a GpuTensor,
    /// Weighted sum of the selected experts.
    pub out: &'a GpuTensor,
    /// `dense_post_norm(dense MLP output)`.
    pub dense_normed: &'a GpuTensor,
    /// Grouped-route scratch for batched rows; `None` routes row by row.
    pub batch: Option<RoutedBatch<'a>>,
}

/// Parallel dense+MoE MLP sublayer:
/// `x = residual + post_norm(dense_post_norm(mlp(n)) + experts.post_norm(moe(r)))`
/// with `n = mlp.pre_norm(x)` and `r = experts.pre_norm(x)`; the router reads
/// `router_norm(x) * router_input_scale`. `mlp.post_norm` is the block's final
/// post-norm.
pub struct ParallelMoeMlpOp<'a> {
    pub mlp: SandwichMlpOp<'a>,
    pub dense_post_norm: &'a GpuTensor,
    pub experts: RoutedExperts<'a>,
    pub scratch: RoutedScratch<'a>,
}

pub fn execute_parallel_moe_mlp(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &ParallelMoeMlpOp<'_>,
) -> Result<(), DispatchError> {
    let mlp = &op.mlp;
    let s = mlp.stream;
    let e = &op.experts;
    let r = &op.scratch;
    if e.top_k != 8 {
        return Err(DispatchError::Hip(format!(
            "parallel MoE MLP: top_k={} unsupported (indexed kernels are k=8)",
            e.top_k
        )));
    }
    if !RoutedExperts::supports(e.gate_up_dtype, e.down_dtype) {
        return Err(DispatchError::Hip(format!(
            "parallel MoE MLP: expert formats gate_up={:?} down={:?} have no indexed kernel",
            e.gate_up_dtype, e.down_dtype
        )));
    }
    dense_mlp(gpu, ctx, mlp)?;
    if s.rows == 1 {
        s.norm(gpu, mlp.out, op.dense_post_norm, r.dense_normed)?;
        routed_row(gpu, ctx, e, r, s.residual, s.hidden, s.eps)?;
        gpu.add_f32(r.dense_normed, r.out, s.normed).map_err(hip)?;
        s.norm(gpu, s.normed, mlp.post_norm, s.normed)?;
    } else {
        s.norm(gpu, mlp.out, op.dense_post_norm, mlp.out)?;
        match r.batch.filter(|_| {
            gpu.arch_caps.is_gfx1151() && RoutedExperts::grouped_rows(e.gate_up_dtype, e.down_dtype)
        }) {
            Some(b) => {
                routed_rows_grouped(gpu, ctx, e, &b, s)?;
                gpu.add_inplace_f32(mlp.out, b.out).map_err(hip)?;
            }
            // The indexed expert kernels are single-row: route row by row
            // and accumulate into the normed dense output.
            None => {
                for row in 0..s.rows {
                    let residual = s.residual.sub_offset(row * s.hidden, s.hidden);
                    routed_row(gpu, ctx, e, r, &residual, s.hidden, s.eps)?;
                    let out = mlp.out.sub_offset(row * s.hidden, s.hidden);
                    gpu.add_inplace_f32(&out, r.out).map_err(hip)?;
                }
            }
        }
        s.norm(gpu, mlp.out, mlp.post_norm, s.normed)?;
    }
    s.restore_residual(gpu)?;
    gpu.add_inplace_f32(s.x, s.normed).map_err(hip)
}

/// `b.out = experts.post_norm(moe(pre_norm(residual)))` for all `s.rows`
/// rows: batched router, then the routed slots sorted by expert so each
/// expert's gate/up and down weights are read once per 16-slot tile.
fn routed_rows_grouped(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    e: &RoutedExperts<'_>,
    b: &RoutedBatch<'_>,
    s: SandwichStream<'_>,
) -> Result<(), DispatchError> {
    use crate::families::moe::MOE_GROUPED_BLOCK_M;
    let (rows, hidden, mi, k) = (s.rows, s.hidden, e.hidden_dim, e.top_k);
    gpu.rmsnorm_batched(s.residual, e.router_norm, b.input, rows, hidden, s.eps)
        .map_err(hip)?;
    scale(gpu, b.input, e.router_input_scale)?;
    // Expert selection is tie-sensitive: the F16-input Q8 WMMA GEMM moves
    // router logits by ~1e-3, which flips near-tied experts and costs
    // prefill KL 0.16 -> 1.19 on 26B-A4B. The batched Q8 GEMV keeps F32
    // activations and shares each weight load across eight rows.
    if e.router.dtype == DType::Q8_0 {
        gpu.gemm_q8_0_batched_wide_exact(
            e.router.buf,
            b.input,
            b.router_logits,
            e.router.m,
            e.router.k,
            rows,
        )
        .map_err(hip)?;
    } else {
        gemm_rows(
            gpu,
            ctx,
            &e.router,
            b.input,
            b.router_logits,
            b.input_rot,
            rows,
        )?;
    }
    gpu.moe_softmax_topk_renorm_k8_batched(
        b.router_logits,
        b.topk_indices,
        b.topk_weights,
        e.n_experts,
        true,
        rows,
    )
    .map_err(hip)?;

    gpu.rmsnorm_batched(s.residual, e.pre_norm, b.input, rows, hidden, s.eps)
        .map_err(hip)?;
    gpu.rotate_x_mq_batched(b.input, b.input_rot, hidden, rows)
        .map_err(hip)?;
    let m_total = RoutedBatch::m_total(rows, k, e.n_experts);
    gpu.moe_scatter_fused_k8(
        b.topk_indices,
        b.counts,
        b.offsets,
        b.sorted_slots,
        b.tile_ids,
        b.inverse_perm,
        rows * k,
        e.n_experts,
        m_total,
        MOE_GROUPED_BLOCK_M,
    )
    .map_err(hip)?;
    super::dispatch_grouped_gemm(
        gpu,
        e.gate_up_dtype,
        None,
        e.gate_up_ptrs,
        b.tile_ids,
        b.sorted_slots,
        b.input_rot,
        b.gate_up_grouped,
        2 * mi,
        hidden,
        k,
        m_total,
        rows,
        false,
        false,
        false,
    )?;
    gpu.moe_gate_up_unscatter_k8(
        b.gate_up_grouped,
        b.sorted_slots,
        b.gate,
        b.up,
        mi,
        k,
        m_total,
    )
    .map_err(hip)?;
    gpu.gelu_tanh_f32(b.gate, b.gate, rows * k * mi)
        .map_err(hip)?;
    gpu.mul_f32(b.gate, b.up, b.gate).map_err(hip)?;
    gpu.gemv_hfq4g128_moe_down_grouped(
        e.down_ptrs,
        b.tile_ids,
        b.sorted_slots,
        e.per_expert_scale,
        b.gate,
        b.gate_up_grouped,
        hidden,
        mi,
        m_total,
    )
    .map_err(hip)?;
    gpu.zero_f32(b.out).map_err(hip)?;
    gpu.moe_down_combine_grouped_k8(
        b.gate_up_grouped,
        b.inverse_perm,
        b.topk_weights,
        b.out,
        hidden,
        k,
        rows,
    )
    .map_err(hip)?;
    gpu.rmsnorm_batched(b.out, e.post_norm, b.out, rows, hidden, s.eps)
        .map_err(hip)
}

/// `r.out = experts.post_norm(moe(pre_norm(residual)))` for one row; the
/// router reads `router_norm(residual) * router_input_scale`.
fn routed_row(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    e: &RoutedExperts<'_>,
    r: &RoutedScratch<'_>,
    residual: &GpuTensor,
    hidden: usize,
    eps: f32,
) -> Result<(), DispatchError> {
    gpu.rmsnorm_f32(residual, e.pre_norm, r.input, eps)
        .map_err(hip)?;
    gpu.rmsnorm_f32(residual, e.router_norm, r.router_in, eps)
        .map_err(hip)?;
    scale(gpu, r.router_in, e.router_input_scale)?;
    gemv(gpu, ctx, &e.router, r.router_in, r.router_logits)?;
    gpu.moe_softmax_topk_renorm_k8(
        r.router_logits,
        r.topk_indices,
        r.topk_weights,
        e.n_experts,
        true,
    )
    .map_err(hip)?;

    // Q8 down accumulates atomically; HFQ4G128 down assigns.
    gpu.zero_f32(r.out).map_err(hip)?;
    let (mi, k) = (e.hidden_dim, e.top_k);
    match e.gate_up_dtype {
        DType::MQ4G256 => {
            gpu.rotate_x_mq(r.input, r.input_rot, hidden).map_err(hip)?;
            gpu.gemv_mq4g256_moe_gate_up_k8_indexed(
                e.gate_up_ptrs,
                r.topk_indices,
                r.input_rot,
                r.gate,
                r.up,
                2 * mi,
                hidden,
            )
            .map_err(hip)?;
        }
        // FWHT-rotated input; MQ6G256 reads it with the HFQ6G256 kernel.
        DType::MQ4G256V2 | DType::MQ6G256 => {
            gpu.rotate_x_mq(r.input, r.input_rot, hidden).map_err(hip)?;
            crate::pipeline::run_uniform_moe_gate_up(
                gpu,
                e.gate_up_dtype,
                e.gate_up_ptrs,
                r.topk_indices,
                r.input_rot,
                r.gate,
                r.up,
                2 * mi,
                hidden,
                k,
            )?
        }
        DType::HFQ4G256 | DType::HFQ6G256 => crate::pipeline::run_uniform_moe_gate_up(
            gpu,
            e.gate_up_dtype,
            e.gate_up_ptrs,
            r.topk_indices,
            r.input,
            r.gate,
            r.up,
            2 * mi,
            hidden,
            k,
        )?,
        _ => gpu
            .gemv_q8_0_moe_gate_up_k8_indexed(
                e.gate_up_ptrs,
                r.topk_indices,
                r.input,
                r.gate,
                r.up,
                2 * mi,
                hidden,
            )
            .map_err(hip)?,
    }
    gpu.gelu_tanh_f32(r.gate, r.act, k * mi).map_err(hip)?;
    gpu.mul_f32(r.act, r.up, r.act).map_err(hip)?;
    if e.down_dtype == DType::Q8_0 {
        gpu.gemv_q8_0_moe_down_residual_scaled_k8_indexed(
            e.down_ptrs,
            r.topk_indices,
            r.topk_weights,
            e.per_expert_scale,
            r.act,
            r.out,
            hidden,
            mi,
        )
    } else {
        gpu.gemv_hfq4g128_moe_down_residual_scaled_k8_indexed(
            e.down_ptrs,
            r.topk_indices,
            r.topk_weights,
            e.per_expert_scale,
            r.act,
            r.out,
            hidden,
            mi,
        )
    }
    .map_err(hip)?;
    gpu.rmsnorm_f32(r.out, e.post_norm, r.out, eps).map_err(hip)
}

/// Per-layer-input branch:
/// `x = residual + post_norm(proj(gelu(gate(x)) * inputs[row, layer]))`.
/// `inputs` holds `layer_width`-wide slices for `n_layers` layers per row.
pub struct PerLayerInputOp<'a> {
    pub stream: SandwichStream<'a>,
    pub layer: usize,
    pub layer_width: usize,
    pub n_layers: usize,
    pub inputs: &'a GpuTensor,
    pub w_gate: WeightRef<'a>,
    pub w_proj: WeightRef<'a>,
    pub post_norm: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub out: &'a GpuTensor,
}

/// Q8 projections whose batched GEMM reproduces the single-row GEMV exactly.
fn exact_wide_q8_gemm(dtype: DType, k: usize) -> bool {
    dtype == DType::Q8_0 && k > 0 && k <= 1536 && k.is_multiple_of(32)
}

pub fn execute_per_layer_input(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &PerLayerInputOp<'_>,
) -> Result<(), DispatchError> {
    let s = op.stream;
    let (rows, width, hidden) = (s.rows, op.layer_width, s.hidden);
    let packed = op.n_layers * width;
    s.save_residual(gpu)?;
    if rows == 1 {
        let layer_input = op.inputs.sub_offset(op.layer * width, width);
        gemv(gpu, ctx, &op.w_gate, s.x, op.gate)?;
        gpu.gelu_tanh_f32(op.gate, op.act, width).map_err(hip)?;
        gpu.mul_f32(op.act, &layer_input, op.act).map_err(hip)?;
        gemv(gpu, ctx, &op.w_proj, op.act, op.out)?;
    } else {
        let exact = batched_fusion_arch(&gpu.arch) && gpu.flags.gemma4_ple_branch_batched_prefill;
        let wide = |gpu: &mut Gpu, w: &WeightRef, x: &GpuTensor, y: &GpuTensor| {
            GEMM.run_key(
                KernelKey::GemmQ8_0BatchedWideExact,
                ctx,
                gpu,
                &GemmParams {
                    w,
                    x,
                    y,
                    batch_size: rows,
                },
            )
        };
        if exact && exact_wide_q8_gemm(op.w_gate.dtype, op.w_gate.k) {
            wide(gpu, &op.w_gate, s.x, op.gate)?;
        } else {
            for row in 0..rows {
                let x_row = s.x.sub_offset(row * hidden, hidden);
                let gate_row = op.gate.sub_offset(row * width, width);
                gemv(gpu, ctx, &op.w_gate, &x_row, &gate_row)?;
            }
        }
        if batched_fusion_arch(&gpu.arch) && gpu.flags.gemma4_ple_activation_fused_prefill {
            gpu.gemma4_ple_gelu_mul_strided_f32(
                op.gate, op.inputs, op.act, rows, width, packed, op.layer,
            )
            .map_err(hip_dbg)?;
        } else {
            gpu.gelu_tanh_f32(op.gate, op.act, rows * width)
                .map_err(hip)?;
            for row in 0..rows {
                let act_row = op.act.sub_offset(row * width, width);
                let layer_input = op.inputs.sub_offset(row * packed + op.layer * width, width);
                gpu.mul_f32(&act_row, &layer_input, &act_row).map_err(hip)?;
            }
        }
        if exact && exact_wide_q8_gemm(op.w_proj.dtype, op.w_proj.k) {
            wide(gpu, &op.w_proj, op.act, op.out)?;
        } else {
            for row in 0..rows {
                let act_row = op.act.sub_offset(row * width, width);
                let out_row = op.out.sub_offset(row * hidden, hidden);
                gemv(gpu, ctx, &op.w_proj, &act_row, &out_row)?;
            }
        }
    }
    s.norm(gpu, op.out, op.post_norm, s.normed)?;
    s.restore_residual(gpu)?;
    gpu.add_inplace_f32(s.x, s.normed).map_err(hip)
}

/// In-place `x *= factor`.
pub struct ScaleOp<'a> {
    pub x: &'a GpuTensor,
    pub factor: f32,
}

pub fn execute_scale(gpu: &mut Gpu, op: &ScaleOp<'_>) -> Result<(), DispatchError> {
    scale(gpu, op.x, op.factor)
}

/// In-place final-logit soft cap: `x = tanh(x / cap) * cap`.
pub struct SoftcapOp<'a> {
    pub logits: &'a GpuTensor,
    pub n: usize,
    pub cap: f32,
}

pub fn execute_softcap(gpu: &mut Gpu, op: &SoftcapOp<'_>) -> Result<(), DispatchError> {
    gpu.logit_softcap_f32(op.logits, op.n, op.cap).map_err(hip)
}

#[cfg(test)]
mod tests {
    use super::{exact_wide_q8_gemm, supports_batched_projection};
    use rdna_compute::DType;

    #[test]
    fn batched_projection_formats_are_explicit() {
        for dtype in [
            DType::F32,
            DType::Q8_0,
            DType::MQ4G256,
            DType::HFQ4G256,
            DType::MQ6G256,
            DType::MQ4G256V2,
        ] {
            assert!(supports_batched_projection(dtype));
        }
        for dtype in [
            DType::HFQ4G128,
            DType::HFQ6G256,
            DType::HFQ2G256,
            DType::HFQ3G256,
            DType::MQ3G256,
        ] {
            assert!(!supports_batched_projection(dtype));
        }
    }

    #[test]
    fn exact_wide_q8_gemm_matches_the_wide_gemv_boundary() {
        assert!(exact_wide_q8_gemm(DType::Q8_0, 1536));
        assert!(!exact_wide_q8_gemm(DType::Q8_0, 1535));
        assert!(!exact_wide_q8_gemm(DType::Q8_0, 1537));
        assert!(!exact_wide_q8_gemm(DType::Q8_0, 0));
        assert!(!exact_wide_q8_gemm(DType::F32, 256));
    }
}
