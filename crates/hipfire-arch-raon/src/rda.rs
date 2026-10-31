// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon RDA binding (`raon_rda_prepare`) onto generic GatedDeltaOp. Owner: slice R.
//!
//! Composition per RDA layer (PLAN §1.5), N rows in one pass, no per-token
//! launches:
//!
//! ```text
//! (caller, projection API)  q_raw=Wq x  k_raw=Wk x  v_raw=Wv x  z=Wg x
//!                           b=Wb x  delta=Wrda_w x  f=Wf_b(Wf_a x)
//! 3x causal_conv_depthwise_silu_rows_f32   (K=4, no bias, SiLU, BF16 round, history update)
//! raon_rda_prepare_f32                     (q/k norm, log_alpha, beta, vprime)
//! execute_gated_delta                      (GatedDeltaOp, PerKeyChannel, y BF16)
//! gated_rmsnorm_sigmoid_rows_f32           (per-head RMSNorm * sigmoid(z), one BF16 cast)
//! (caller)                  branch_output = Wo gated
//! ```
//!
//! All activations are FP32 planes holding BF16-rounded values (the projection
//! API must deliver BF16-rounded projections). Projections and the output
//! projection stay with the caller (slice A projection API) so this module has
//! no weight-codec dependency.
//!
//! Conv history is stored raw/pre-conv, `[K-1, P]`, oldest -> newest.

use hipfire_dispatch::ops::delta_net::{
    execute_gated_delta, DecayAxis, GatedDeltaOp, GatedDeltaStrides,
};
use rdna_compute::gated_delta_vector::{
    rda_conv_prepare_rows_per_wave, CausalConvRowsGeom, GatedRmsnormRowsGeom, RdaConvPrepareLaunch,
    RdaPrepareLaunch, GATED_DELTA_VECTOR_HEAD_DIM,
};
use rdna_compute::gated_delta_vector_chunk::{
    chunked_enabled, GatedDeltaVectorChunkScratch, GATED_DELTA_VECTOR_CHUNK_MIN_ROWS,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Fixed RDA geometry and constants of the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RdaGeometry {
    pub n_heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    /// Per-head output RMSNorm epsilon.
    pub rms_eps: f32,
    /// q/k L2 normalization clamp.
    pub norm_eps: f32,
    /// `log_alpha = lower_bound * sigmoid(u)`.
    pub gate_lower_bound: f32,
    /// Model dtype boundaries: true = BF16 rounding (checkpoint), false = FP32
    /// model dtype (F32-mode oracle).
    pub round_bf16: bool,
}

impl RdaGeometry {
    /// Raon-1-mini-preview: 40 heads x 128, conv K=4, lower bound -5.
    pub fn raon() -> Self {
        Self {
            n_heads: 40,
            head_dim: 128,
            conv_kernel: 4,
            rms_eps: 1e-5,
            norm_eps: 1e-6,
            gate_lower_bound: -5.0,
            round_bf16: true,
        }
    }

    /// Projection width `P = n_heads * head_dim`.
    pub fn proj(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Elements of one slot's recurrent state `[H, V, K]`.
    pub fn state_elems(&self) -> usize {
        self.n_heads * self.head_dim * self.head_dim
    }
}

/// Immutable per-layer weights (F32 device tensors; BF16-origin values widened).
pub struct RaonRdaLayer<'a> {
    /// Depthwise conv weights `[P, K]` for q, k, v.
    pub conv_w_q: &'a GpuTensor,
    pub conv_w_k: &'a GpuTensor,
    pub conv_w_v: &'a GpuTensor,
    /// `[H]` F32.
    pub a_log: &'a GpuTensor,
    /// `[H*K]` F32 (`[H, K]` row-major).
    pub dt_bias: &'a GpuTensor,
    /// Shared per-head RMSNorm gamma `[K]` F32.
    pub o_norm_gamma: &'a GpuTensor,
}

/// Mutable per-slot layer state: FP32 `[H,V,K]` recurrent matrix and the three
/// raw conv histories `[K-1, P]` (oldest -> newest).
pub struct RdaLayerState<'a> {
    pub state: &'a GpuTensor,
    pub hist_q: &'a GpuTensor,
    pub hist_k: &'a GpuTensor,
    pub hist_v: &'a GpuTensor,
    /// Slot index into `state` (multiples of `state_slot_stride`); 0 for a
    /// per-slot view.
    pub slot: usize,
    /// Elements between slots in `state`; `geom.state_elems()` for dense.
    pub state_slot_stride: usize,
}

/// Projected activations, row-major, BF16-rounded FP32 planes.
pub struct RdaProjected<'a> {
    pub q_raw: &'a GpuTensor,
    pub k_raw: &'a GpuTensor,
    pub v_raw: &'a GpuTensor,
    /// Output gate projection `z = Wg x` `[N, P]`.
    pub z: &'a GpuTensor,
    /// `b = Wb x` `[N, H]`.
    pub b: &'a GpuTensor,
    /// `delta = Wrda_w x` `[N, H]`.
    pub delta: &'a GpuTensor,
    /// Decay `f = Wf_b(Wf_a x)` `[N, P]`.
    pub f: &'a GpuTensor,
}

/// Bounded scratch for up to `max_rows` rows.
pub struct RdaScratch {
    pub max_rows: usize,
    pub qconv: GpuTensor,
    pub kconv: GpuTensor,
    pub vconv: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub vprime: GpuTensor,
    pub log_alpha: GpuTensor,
    pub beta: GpuTensor,
    /// BF16-rounded recurrence output (`out = cast_model(y)`) `[N, P]`.
    pub y: GpuTensor,
    /// Chunked-RDA workspace, allocated unless `HIPFIRE_GATED_DELTA_VECTOR_CHUNKED=0`, on a
    /// device that has the chunked kernel; `None` keeps the serial recurrence.
    pub chunk: Option<GatedDeltaVectorChunkScratch>,
}

/// Rows per chunked-RDA scratch segment (a longer prefill chunk runs as several segments).
pub const RDA_CHUNK_SCRATCH_ROWS: usize = 512;

fn chunk_scratch_rows(max_rows: usize) -> usize {
    max_rows.min(RDA_CHUNK_SCRATCH_ROWS)
}

impl RdaScratch {
    pub fn new(gpu: &mut Gpu, geom: &RdaGeometry, max_rows: usize) -> Result<Self, String> {
        let p = geom.proj();
        let plane = |gpu: &mut Gpu, cols: usize| {
            gpu.alloc_tensor(&[max_rows, cols], DType::F32)
                .map_err(|e| format!("rda scratch alloc: {e:?}"))
        };
        Ok(Self {
            max_rows,
            qconv: plane(gpu, p)?,
            kconv: plane(gpu, p)?,
            vconv: plane(gpu, p)?,
            q: plane(gpu, p)?,
            k: plane(gpu, p)?,
            vprime: plane(gpu, p)?,
            log_alpha: plane(gpu, p)?,
            beta: plane(gpu, geom.n_heads)?,
            y: plane(gpu, p)?,
            chunk: if chunked_enabled() && gpu.gated_delta_vector_chunked_arch_supported() {
                Some(
                    gpu.alloc_gated_delta_vector_chunk_scratch(
                        chunk_scratch_rows(max_rows),
                        geom.n_heads,
                    )
                    .map_err(|e| format!("rda chunk scratch alloc: {e:?}"))?,
                )
            } else {
                None
            },
        })
    }

    /// Device bytes [`Self::new`] allocates for `max_rows` rows (pool allocations
    /// are exact, minimum 256 B each). `chunked_arch`: the device has the chunked kernel
    /// ([`Gpu::gated_delta_vector_chunked_arch_supported`]), so [`Self::new`] allocates its workspace.
    pub fn required_bytes(geom: &RdaGeometry, max_rows: usize, chunked_arch: bool) -> usize {
        let p = geom.proj();
        // qconv, kconv, vconv, q, k, vprime, log_alpha, y: `[rows, P]`; beta: `[rows, heads]`.
        let mut elems = [max_rows * p; 8].to_vec();
        elems.push(max_rows * geom.n_heads);
        let planes: usize = elems.into_iter().map(|e| (e * 4).max(256)).sum();
        // Charged exactly when `new` allocates it: flag on and an arch with the kernel.
        let chunk = if chunked_enabled() && chunked_arch {
            GatedDeltaVectorChunkScratch::required_bytes(chunk_scratch_rows(max_rows), geom.n_heads)
        } else {
            0
        };
        planes + chunk
    }
}

/// Per-layer recurrence-input tape: non-owning `[max_rows, *]` plane views the
/// verify forward writes INSTEAD of the scratch planes, so a partial accept can
/// rerun the recurrence of the kept prefix ([`crate::rollback`]). `q`, `k`,
/// `vprime`, `log_alpha` are `raon_rda_prepare`'s outputs (`[rows, P]`) and
/// `beta` its `[rows, H]` output, exactly what [`execute_gated_delta`] consumes;
/// `q_raw`/`k_raw`/`v_raw` are the pre-conv projections (`[rows, P]`) whose last
/// rows rebuild the conv history. Every plane is row-contiguous with a row
/// stride equal to its width.
pub struct RdaTape {
    pub q_raw: GpuTensor,
    pub k_raw: GpuTensor,
    pub v_raw: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub vprime: GpuTensor,
    pub log_alpha: GpuTensor,
    pub beta: GpuTensor,
}

/// The recurrence of one RDA layer: [`execute_gated_delta`] over `n` rows of
/// `q, k, vprime, log_alpha, beta` (contiguous planes, row stride = plane
/// width) from and into `st.state`, `output` receiving the BF16-rounded
/// `[n, P]` result. The ONE launch shared by [`rda_forward_rows_taped`] and the
/// rollback replay, so both run the same geometry, strides and slot.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rda_recurrence(
    gpu: &mut Gpu,
    geom: &RdaGeometry,
    st: &RdaLayerState<'_>,
    q: &GpuTensor,
    k: &GpuTensor,
    vprime: &GpuTensor,
    log_alpha: &GpuTensor,
    beta: &GpuTensor,
    output: &GpuTensor,
    n: usize,
    chunk: Option<&GatedDeltaVectorChunkScratch>,
) -> Result<(), String> {
    let h = geom.n_heads;
    let kd = geom.head_dim;
    let mut strides = GatedDeltaStrides::contiguous(h, 1, kd, kd, DecayAxis::PerKeyChannel);
    strides.state_slot = st.state_slot_stride;
    execute_gated_delta(
        gpu,
        &GatedDeltaOp {
            query: q,
            key: k,
            value: vprime,
            log_decay: log_alpha,
            beta,
            state: st.state,
            output,
            slot_idx: None,
            n_dev: None,
            n_tokens: n,
            n_heads: h,
            qk_group: 1,
            key_dim: kd,
            value_dim: kd,
            decay_axis: DecayAxis::PerKeyChannel,
            round_output_bf16: geom.round_bf16,
            slot: st.slot,
            strides,
            chunk_scratch: chunk,
        },
    )
    .map_err(|e| e.to_string())
}

/// Run one RDA layer over `n` rows. Writes `gated [N, P]` (input of the output
/// projection) and updates recurrent state + conv histories in place.
/// A zero-row call is a no-op (history/state unchanged).
#[allow(clippy::too_many_arguments)]
pub fn rda_forward_rows(
    gpu: &mut Gpu,
    geom: &RdaGeometry,
    layer: &RaonRdaLayer<'_>,
    st: &RdaLayerState<'_>,
    x: &RdaProjected<'_>,
    scratch: &RdaScratch,
    gated: &GpuTensor,
    n: usize,
) -> Result<(), String> {
    rda_forward_rows_taped(gpu, geom, layer, st, x, scratch, gated, n, None)
}

/// [`rda_forward_rows`] with an optional [`RdaTape`]: when present,
/// `raon_rda_prepare` writes `q, k, vprime, log_alpha, beta` into the tape's
/// planes and the recurrence reads them from there (the launch sequence is
/// identical, only the pointers differ). The caller supplies the tape's
/// `q_raw`/`k_raw`/`v_raw` planes as `x.q_raw`/`x.k_raw`/`x.v_raw`.
#[allow(clippy::too_many_arguments)]
pub fn rda_forward_rows_taped(
    gpu: &mut Gpu,
    geom: &RdaGeometry,
    layer: &RaonRdaLayer<'_>,
    st: &RdaLayerState<'_>,
    x: &RdaProjected<'_>,
    scratch: &RdaScratch,
    gated: &GpuTensor,
    n: usize,
    tape: Option<&RdaTape>,
) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    if n > scratch.max_rows {
        return Err(format!(
            "rda_forward_rows: {n} rows exceed scratch capacity {}",
            scratch.max_rows
        ));
    }
    let p = geom.proj();
    let h = geom.n_heads;
    let kd = geom.head_dim;

    let (q, k, vprime, log_alpha, beta) = match tape {
        Some(t) => (&t.q, &t.k, &t.vprime, &t.log_alpha, &t.beta),
        None => (
            &scratch.q,
            &scratch.k,
            &scratch.vprime,
            &scratch.log_alpha,
            &scratch.beta,
        ),
    };

    // Wave-per-tile conv+prepare / gated norm: exact gfx1201, ordinary prefill rows, no tape.
    let wave = tape.is_none()
        && n >= RDA_WAVE_MIN_ROWS
        && fused_prep_enabled()
        && gpu.rda_wave_kernels_arch_supported()
        && kd == GATED_DELTA_VECTOR_HEAD_DIM
        && geom.conv_kernel == 4
        && p % 4 == 0;
    // Segmented producer -> core -> norm: each segment's planes are produced right before the chunked core
    // consumes them and the norm right after, so the segment's working set stays in the last-level cache.
    let seg_rows = match scratch.chunk.as_ref() {
        Some(c) if tape.is_none() && segment_enabled() => c.segment_rows().min(RDA_SEGMENT_ROWS),
        _ => usize::MAX,
    };

    if n <= seg_rows {
        let io = RowsIo {
            q_raw: x.q_raw,
            k_raw: x.k_raw,
            v_raw: x.v_raw,
            z: x.z,
            b: x.b,
            delta: x.delta,
            f: x.f,
            gated,
            qconv: &scratch.qconv,
            kconv: &scratch.kconv,
            vconv: &scratch.vconv,
            q,
            k,
            vprime,
            log_alpha,
            beta,
            y: &scratch.y,
        };
        return rda_rows(gpu, geom, layer, st, &io, scratch.chunk.as_ref(), n, wave);
    }

    // Segments reuse the first `seg_rows` rows of the scratch planes (bounded by `max_rows >= seg_rows`).
    let mut row0 = 0usize;
    while row0 < n {
        let mut len = (n - row0).min(seg_rows);
        // A trailing range under the chunked core's admission floor would run the serial kernel (different
        // bits): fold it into this segment (the core splits a longer launch at its own capacity).
        if n - row0 - len > 0 && n - row0 - len < GATED_DELTA_VECTOR_CHUNK_MIN_ROWS {
            len = n - row0;
        }
        let rows_p = |t: &GpuTensor| t.sub_offset(row0 * p, len * p);
        let rows_h = |t: &GpuTensor| t.sub_offset(row0 * h, len * h);
        let head_p = |t: &GpuTensor| t.sub_offset(0, len * p);
        let (q_raw, k_raw, v_raw, z, b, delta, f, gated_seg) = (
            rows_p(x.q_raw),
            rows_p(x.k_raw),
            rows_p(x.v_raw),
            rows_p(x.z),
            rows_h(x.b),
            rows_h(x.delta),
            rows_p(x.f),
            rows_p(gated),
        );
        let (qconv, kconv, vconv, q_s, k_s, vp_s, la_s, y_s) = (
            head_p(&scratch.qconv),
            head_p(&scratch.kconv),
            head_p(&scratch.vconv),
            head_p(q),
            head_p(k),
            head_p(vprime),
            head_p(log_alpha),
            head_p(&scratch.y),
        );
        let beta_s = beta.sub_offset(0, len * h);
        let io = RowsIo {
            q_raw: &q_raw,
            k_raw: &k_raw,
            v_raw: &v_raw,
            z: &z,
            b: &b,
            delta: &delta,
            f: &f,
            gated: &gated_seg,
            qconv: &qconv,
            kconv: &kconv,
            vconv: &vconv,
            q: &q_s,
            k: &k_s,
            vprime: &vp_s,
            log_alpha: &la_s,
            beta: &beta_s,
            y: &y_s,
        };
        rda_rows(gpu, geom, layer, st, &io, scratch.chunk.as_ref(), len, wave)?;
        row0 += len;
    }
    Ok(())
}

/// Smallest row count the wave-per-tile conv+prepare and gated-norm kernels are used for (ordinary prefill;
/// decode and verify rows keep the incumbent launches).
pub const RDA_WAVE_MIN_ROWS: usize = 64;

/// Rows per segment of the segmented RDA layer (a whole number of C64 chunks, at most the chunked core's own
/// segment). 192-256 rows measured best (R9700, 4096-row layer: 3.60-3.65 ms vs 3.84 ms at 512 and 4.84 ms
/// unsegmented/unfused); at 256 a segment's producer planes, `y` and the core workspace stay cache-resident.
pub const RDA_SEGMENT_ROWS: usize = 256;

/// `HIPFIRE_RAON_RDA_FUSED_PREP` (developer, exact gfx1201): the three RDA convs + `raon_rda_prepare` run as one
/// launch (no conv planes) and the gated norm as one wave per (row, head); both bit-identical to the
/// incumbent launches. `=0` restores them.
fn fused_prep_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| hipfire_config::developer_bool("HIPFIRE_RAON_RDA_FUSED_PREP", true))
}

/// `HIPFIRE_RAON_RDA_SEGMENT` (developer): with the chunked core, an ordinary prefill chunk runs its RDA layer
/// as consecutive core-segment-sized row ranges (producer, core, gated norm per range) so the planes the core
/// consumes are still cache-resident. Bit-identical (segments are whole C64 chunks). `=0` restores one pass.
fn segment_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| hipfire_config::developer_bool("HIPFIRE_RAON_RDA_SEGMENT", true))
}

/// Tensors of one row range of an RDA layer (all row-contiguous, row stride = plane width).
struct RowsIo<'a> {
    q_raw: &'a GpuTensor,
    k_raw: &'a GpuTensor,
    v_raw: &'a GpuTensor,
    z: &'a GpuTensor,
    b: &'a GpuTensor,
    delta: &'a GpuTensor,
    f: &'a GpuTensor,
    gated: &'a GpuTensor,
    qconv: &'a GpuTensor,
    kconv: &'a GpuTensor,
    vconv: &'a GpuTensor,
    q: &'a GpuTensor,
    k: &'a GpuTensor,
    vprime: &'a GpuTensor,
    log_alpha: &'a GpuTensor,
    beta: &'a GpuTensor,
    y: &'a GpuTensor,
}

/// conv(q,k,v) -> prepare -> recurrence -> gated norm over `n` rows of `io` (the conv histories and the
/// recurrent state advance by exactly these rows, so consecutive ranges chain).
#[allow(clippy::too_many_arguments)]
fn rda_rows(
    gpu: &mut Gpu,
    geom: &RdaGeometry,
    layer: &RaonRdaLayer<'_>,
    st: &RdaLayerState<'_>,
    io: &RowsIo<'_>,
    chunk: Option<&GatedDeltaVectorChunkScratch>,
    n: usize,
    wave: bool,
) -> Result<(), String> {
    let p = geom.proj();
    let h = geom.n_heads;
    let kd = geom.head_dim;
    let err = |what: &str, e: String| format!("rda_forward_rows {what}: {e}");
    let q_scale = (kd as f64).powf(-0.5) as f32;

    if wave {
        gpu.raon_rda_conv_prepare_w128_f32(&RdaConvPrepareLaunch {
            q_out: io.q,
            k_out: io.k,
            vprime_out: io.vprime,
            log_alpha_out: io.log_alpha,
            beta_out: io.beta,
            q_raw: io.q_raw,
            k_raw: io.k_raw,
            v_raw: io.v_raw,
            conv_w_q: layer.conv_w_q,
            conv_w_k: layer.conv_w_k,
            conv_w_v: layer.conv_w_v,
            hist_q: st.hist_q,
            hist_k: st.hist_k,
            hist_v: st.hist_v,
            f: io.f,
            b: io.b,
            delta: io.delta,
            a_log: layer.a_log,
            dt_bias: layer.dt_bias,
            n_rows: n,
            n_heads: h,
            head_dim: kd,
            conv_kernel: geom.conv_kernel,
            rows_per_wave: rda_conv_prepare_rows_per_wave(n),
            raw_row_stride: p,
            f_row_stride: p,
            bd_row_stride: h,
            out_row_stride: p,
            beta_row_stride: h,
            hist_tap_stride: p,
            q_scale,
            norm_eps: geom.norm_eps,
            lower_bound: geom.gate_lower_bound,
            round_bf16: geom.round_bf16,
        })
        .map_err(|e| err("conv+prepare", format!("{e:?}")))?;
    } else {
        // Three independent depthwise convs: raw -> SiLU -> BF16, history in place.
        for (input, weight, hist, out, what) in [
            (io.q_raw, layer.conv_w_q, st.hist_q, io.qconv, "conv q"),
            (io.k_raw, layer.conv_w_k, st.hist_k, io.kconv, "conv k"),
            (io.v_raw, layer.conv_w_v, st.hist_v, io.vconv, "conv v"),
        ] {
            let g = CausalConvRowsGeom {
                n_rows: n,
                n_channels: p,
                ksize: geom.conv_kernel,
                in_row_stride: p,
                out_row_stride: p,
                w_ch_stride: geom.conv_kernel,
                w_tap_stride: 1,
                hist_tap_stride: p,
                apply_silu: true,
                round_bf16: geom.round_bf16,
            };
            gpu.causal_conv_depthwise_silu_rows_f32(out, input, weight, hist, hist, &g)
                .map_err(|e| err(what, format!("{e:?}")))?;
        }

        gpu.raon_rda_prepare_f32(&RdaPrepareLaunch {
            q_out: io.q,
            k_out: io.k,
            vprime_out: io.vprime,
            log_alpha_out: io.log_alpha,
            beta_out: io.beta,
            qconv: io.qconv,
            kconv: io.kconv,
            vconv: io.vconv,
            f: io.f,
            b: io.b,
            delta: io.delta,
            a_log: layer.a_log,
            dt_bias: layer.dt_bias,
            n_rows: n,
            n_heads: h,
            head_dim: kd,
            conv_row_stride: p,
            f_row_stride: p,
            bd_row_stride: h,
            out_row_stride: p,
            beta_row_stride: h,
            q_scale,
            norm_eps: geom.norm_eps,
            lower_bound: geom.gate_lower_bound,
            round_bf16: geom.round_bf16,
        })
        .map_err(|e| err("prepare", format!("{e:?}")))?;
    }

    rda_recurrence(
        gpu,
        geom,
        st,
        io.q,
        io.k,
        io.vprime,
        io.log_alpha,
        io.beta,
        io.y,
        n,
        chunk,
    )
    .map_err(|e| err("recurrence", e))?;

    let norm = GatedRmsnormRowsGeom {
        n_rows: n,
        n_heads: h,
        head_dim: kd,
        x_row_stride: p,
        gate_row_stride: p,
        out_row_stride: p,
        eps: geom.rms_eps,
        round_bf16: geom.round_bf16,
    };
    if wave {
        gpu.gated_rmsnorm_sigmoid_rows_w128_f32(io.gated, io.y, io.z, layer.o_norm_gamma, &norm)
    } else {
        gpu.gated_rmsnorm_sigmoid_rows_f32(io.gated, io.y, io.z, layer.o_norm_gamma, &norm)
    }
    .map_err(|e| err("output gate", format!("{e:?}")))
}
