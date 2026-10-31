// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon-1-mini forward composition (`forward_token` / `forward_rows`). Owner: slice I.
//!
//! One layer schedule serves decode (`rows == 1`) and bounded prefill chunks
//! (`rows == N`): every stage is a module call over `rows` rows, so a chunk is
//! genuinely batched (projections, mHC, RDA, MLA, dense/MoE) and a single token
//! is the `rows == 1` instance of the same route.
//!
//! ```text
//! embed rows -> broadcast to the 4 streams (PLAN §1.2)
//! for layer in 0..L:
//!     attention site:  route(OLD streams) -> aggregate+RMS(input_layernorm)
//!                      -> mixer (RDA | MLA) -> postmix(OLD streams, branch)
//!     FFN site:        route(OLD streams) -> aggregate+RMS(post_attention_layernorm)
//!                      -> dense (layer 0) | MoE -> postmix
//! head: mean over streams -> RMS(model.norm) -> lm_head (BF16-rounded), last row only
//! ```
//!
//! All activations are F32 planes holding BF16-rounded values: every GEMM output
//! is rounded through `projection::OutRound::Bf16`; the stream-mix stages, the
//! RDA module, the latent-attention module and the MoE module apply their own
//! documented boundaries.
//!
//! Position bookkeeping: the MLA kernels read the slot's *device* `seq_len`
//! (no host-baked lengths), RDA is position-free (no RoPE anywhere). The
//! counter is advanced exactly once per call, after every layer consumed it.
//!
//! Decode graph: `forward_body` is device-only (the token ids are staged into
//! `scratch.token_ids` by the host first), so the `rows == 1` instance is
//! recorded into a hipGraph by [`decode_body`] and replayed per token by
//! `RaonBundle::decode_step`; [`decode_binding`] names every baked address and
//! extent so a graph is rebuilt exactly when one changes.
//!
//! Failure semantics: an `Err` after the first layer may leave RDA state and
//! latent rows partially updated while `n_tokens` is unchanged; the caller must
//! `reset` the slot before reusing it.
//!
//! The final streams of the last call always live in `scratch.stream_a`
//! (50 ping-pong sites: an even count), which is what [`readout_logits`] reads.

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::latent_attention::{
    execute_latent_attention, ExpandedMirror as DMirror, LatentAttentionOp, LatentCache,
    LatentGeometry, RowRange,
};
use hipfire_dispatch::pipeline::stream_mix::{
    execute_stream_mix, StreamChart, StreamMixOp, StreamMixStage, StreamNorm,
};
use hipfire_runtime::llama::EmbeddingFormat;
use rdna_compute::{Gpu, GpuTensor};

use crate::config::{LayerKind, RaonConfig};
use crate::mhc::{broadcast_embedding, final_mean_norm, RaonMhcShape, RaonMhcWork};
use crate::mla::{raon_mla_scale, MlaRoute, RaonMlaWork, RAON_MLA_EPS};
use crate::moe::{
    forward_dense, forward_moe, RaonDenseRefs, RaonDenseScratch, RaonMoeDims, RaonMoeLayerRefs,
    RaonMoeScratch,
};
use crate::projection::{project, OutRound};
use crate::rda::{
    rda_forward_rows_taped, RaonRdaLayer, RdaGeometry, RdaLayerState, RdaProjected, RdaScratch,
    RdaTape,
};
use crate::state::{RaonPrefillScratch, RaonState, MLA_TILE_ROWS};
use crate::weights::{
    MhcSiteWeights, MlaWeights, RaonFfn, RaonLayerWeights, RaonMixer, RaonWeights, RdaWeights,
};

// ───────────────────────── module scratch ─────────────────────────

/// Module-owned work buffers of the forward (mHC controls, RDA, MoE, dense
/// FFN), sized for `rows` rows like the [`RaonPrefillScratch`] they accompany.
/// Owned once per bundle; reused by every layer and every call.
pub struct RaonWork {
    rows: usize,
    mhc_shape: RaonMhcShape,
    mhc: RaonMhcWork,
    rda_geom: RdaGeometry,
    rda: RdaScratch,
    moe_dims: RaonMoeDims,
    moe: RaonMoeScratch,
    dense: RaonDenseScratch,
    /// MLA route controls, split-KV workspace and absorbed-route operands.
    pub mla: RaonMlaWork,
}

impl RaonWork {
    /// Select the activation tier of the grouped routed-expert GEMMs (bundle load, once).
    pub fn set_routed_tier(&mut self, tier: hipfire_dispatch::pipeline::GroupedMoeActTier) {
        self.moe_dims.routed_tier = tier;
    }

    /// Module geometries every work buffer is derived from.
    fn shapes(cfg: &RaonConfig) -> Result<(RaonMhcShape, RdaGeometry, RaonMoeDims), String> {
        let mhc_shape = RaonMhcShape {
            streams: cfg.mhc_streams,
            width: cfg.hidden,
            post_factor: cfg.mhc_post_factor,
            selector_eps: cfg.mhc_selector_eps,
            ..RaonMhcShape::raon_mini()
        };
        let lower_bound = cfg.rda_gate_lower_bound.ok_or_else(|| {
            "raon1_mini: RDA gate without a lower bound (-softplus form) is not implemented"
                .to_string()
        })?;
        let rda_geom = RdaGeometry {
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim,
            conv_kernel: cfg.rda_conv_kernel,
            rms_eps: cfg.rms_eps,
            norm_eps: cfg.qk_l2_eps,
            gate_lower_bound: lower_bound,
            round_bf16: true,
        };
        let moe_dims = RaonMoeDims {
            hidden: cfg.hidden,
            n_experts: cfg.n_experts,
            top_k: cfg.top_k,
            logical_mi: cfg.moe_inter,
            down_k: cfg.moe_down_k,
            dense_mi: cfg.dense_inter,
            gate_beta: cfg.situ_beta,
            value_beta: cfg.situ_linear_beta,
            router_eps: cfg.router_eps,
            router_scale: cfg.routed_scale,
            routed_tier: hipfire_dispatch::pipeline::GroupedMoeActTier::F16,
        };
        moe_dims.validate()?;
        Ok((mhc_shape, rda_geom, moe_dims))
    }

    /// Device bytes of the work for `rows` rows once the bundle is assembled
    /// (every module buffer plus the packed absorbed MLA weights): what the
    /// prefill chunk admission charges against free memory.
    pub fn required_bytes(cfg: &RaonConfig, rows: usize) -> Result<usize, String> {
        if rows == 0 {
            return Err("raon work: rows == 0".into());
        }
        let (mhc_shape, rda_geom, moe_dims) = Self::shapes(cfg)?;
        Ok(RaonMhcWork::required_bytes(&mhc_shape, rows)
            + RdaScratch::required_bytes(&rda_geom, rows)
            + RaonMoeScratch::required_bytes(&moe_dims, rows)?
            + RaonDenseScratch::required_bytes(&moe_dims, rows)?
            + RaonMlaWork::required_bytes(cfg, rows)?)
    }

    /// Allocate the module scratch for at most `rows` rows.
    pub fn new(gpu: &mut Gpu, cfg: &RaonConfig, rows: usize) -> Result<Self, String> {
        if rows == 0 {
            return Err("raon work: rows == 0".into());
        }
        let (mhc_shape, rda_geom, moe_dims) = Self::shapes(cfg)?;
        let mhc = RaonMhcWork::new(gpu, &mhc_shape, rows)?;
        let rda = match RdaScratch::new(gpu, &rda_geom, rows) {
            Ok(r) => r,
            Err(e) => {
                let _ = mhc.free(gpu);
                return Err(e);
            }
        };
        let moe = match RaonMoeScratch::new(gpu, &moe_dims, rows) {
            Ok(m) => m,
            Err(e) => {
                let _ = mhc.free(gpu);
                free_rda_scratch(gpu, rda);
                return Err(e);
            }
        };
        let dense = match RaonDenseScratch::new(gpu, &moe_dims, rows) {
            Ok(d) => d,
            Err(e) => {
                let _ = mhc.free(gpu);
                free_rda_scratch(gpu, rda);
                moe.free(gpu);
                return Err(e);
            }
        };
        let mla = match RaonMlaWork::new(gpu, cfg, rows) {
            Ok(m) => m,
            Err(e) => {
                let _ = mhc.free(gpu);
                free_rda_scratch(gpu, rda);
                moe.free(gpu);
                dense.free(gpu);
                return Err(e);
            }
        };
        Ok(Self {
            rows,
            mhc_shape,
            mhc,
            rda_geom,
            rda,
            moe_dims,
            moe,
            dense,
            mla,
        })
    }

    /// Rows this scratch supports.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// RDA geometry (state elements per slot, head count, width): the rollback
    /// replay launches the recurrence with the verify's own geometry.
    pub(crate) fn rda_geom(&self) -> &RdaGeometry {
        &self.rda_geom
    }

    /// The `[rows, P]` recurrence-output scratch: the rollback replay discards
    /// its recurrence output here (only the state update matters).
    pub(crate) fn rda_discard(&self) -> &GpuTensor {
        &self.rda.y
    }

    /// Return every device buffer.
    pub fn free(self, gpu: &mut Gpu) {
        if let Err(e) = self.mhc.free(gpu) {
            eprintln!("raon1_mini: mhc work free: {e}");
        }
        free_rda_scratch(gpu, self.rda);
        self.moe.free(gpu);
        self.dense.free(gpu);
        self.mla.free(gpu);
    }
}

fn free_rda_scratch(gpu: &mut Gpu, r: RdaScratch) {
    for t in [
        r.qconv,
        r.kconv,
        r.vconv,
        r.q,
        r.k,
        r.vprime,
        r.log_alpha,
        r.beta,
        r.y,
    ] {
        let _ = gpu.free_tensor(t);
    }
}

// ───────────────────────── taps ─────────────────────────

/// A point in the layer schedule at which the stream state can be observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapSite {
    /// After the embedding broadcast.
    Embedding,
    /// After the attention-site postmix of this layer.
    Attn(usize),
    /// After the FFN-site postmix of this layer.
    Ffn(usize),
}

/// Observer of the stream state (`[rows, streams, hidden]` F32 holding BF16
/// values). Oracle only: a tap downloads and is not graph-capturable.
pub type LayerTap<'a> =
    &'a mut dyn FnMut(&mut Gpu, TapSite, &GpuTensor, usize) -> Result<(), String>;

fn fire(
    tap: &mut Option<LayerTap<'_>>,
    gpu: &mut Gpu,
    site: TapSite,
    streams: &GpuTensor,
    rows: usize,
) -> Result<(), String> {
    if let Some(t) = tap.as_mut() {
        (**t)(gpu, site, streams, rows)?;
    }
    Ok(())
}

/// Device-side capture of the post-FFN mHC carrier for the DFlash2 drafter.
///
/// After the FFN site of every listed layer (the buffer `fire(.., TapSite::Ffn(l), ..)`
/// sees: the layer's OUTPUT, i.e. the input of layer `l + 1`) the forward copies that
/// layer's `[rows, streams, hidden]` rows into `dst` with one strided device copy, no
/// download. The drafter's feature of a row is the concatenation, over the listed
/// layers in order, of the four streams of that layer, stream-major
/// (`[s0|s1|s2|s3]`, each `hidden` wide): `layers.len() * streams * hidden` F32 per
/// row, layer `layers[k]` at column offset `k * streams * hidden`.
///
/// This is the representation KRAFTON's DFlash2 draft for Raon-1-mini-preview was
/// trained against (design notes `raon/dflash2/PLAN.md` §2.1): the vLLM fork's
/// `raon1_mini/model.py` appends `s.flatten(1)` (`[T, 4*H]`, "tap-major,
/// stream-major") of the carrier after `layer(positions, s)` for every layer in
/// `aux_hidden_state_layers`, and the llama.cpp port's `raon1.cpp` hands the
/// drafter `t_layer_inp[il]`, "the output of layer il-1 after its MLP hc_post,
/// stream-major per token", concatenated per token with layer k at offset
/// `k * n_embd_tgt`. The raw (un-normalized, pre-final-mean) residual is kept.
///
/// No capture means no extra launch: the decode graph and the AR prefill never
/// pass one.
#[derive(Clone, Copy)]
pub struct DflashCapture<'a> {
    /// Zero-based layers to capture: non-empty, strictly ascending, below the
    /// layer count ([`validate_extract_layers`]).
    pub layers: &'a [usize],
    /// F32 `[>= row0 + rows, layers.len() * streams * hidden]` destination.
    pub dst: &'a GpuTensor,
    /// Destination row of the call's first row.
    pub row0: usize,
}

/// DFlash extract layers must be non-empty, strictly ascending and below
/// `n_layers`: one capture slot per layer, in the drafter's `fc` input order.
pub fn validate_extract_layers(layers: &[usize], n_layers: usize) -> Result<(), String> {
    if layers.is_empty() {
        return Err("raon1_mini: DFlash extract layer list is empty".into());
    }
    if layers.windows(2).any(|pair| pair[0] >= pair[1]) || layers.iter().any(|&l| l >= n_layers) {
        return Err(format!(
            "raon1_mini: DFlash extract layers {layers:?} must be strictly ascending and below \
             the {n_layers} layers"
        ));
    }
    Ok(())
}

/// `(dst_row_stride, dst_col_offset)` in F32 elements of the strided copy that
/// lands layer slot `k` of a `layers`-layer capture of `width`-wide stream rows
/// at destination row `row0`.
fn capture_geometry(layers: usize, width: usize, row0: usize, k: usize) -> (usize, usize) {
    let stride = layers * width;
    (stride, row0 * stride + k * width)
}

impl DflashCapture<'_> {
    fn check(&self, cfg: &RaonConfig, rows: usize) -> Result<(), String> {
        validate_extract_layers(self.layers, cfg.n_layers)?;
        let width = cfg.mhc_streams * cfg.hidden;
        let need = (self.row0 + rows) * self.layers.len() * width;
        if self.dst.dtype != rdna_compute::DType::F32 || self.dst.numel() < need {
            return Err(format!(
                "raon1_mini: DFlash capture buffer holds {} F32 elements, rows {}..{} of {} layers \
                 need {need}",
                self.dst.numel(),
                self.row0,
                self.row0 + rows,
                self.layers.len()
            ));
        }
        Ok(())
    }

    /// Identity words of the capture a recorded graph baked in: the layer
    /// list, the destination buffer, its first row and its extent.
    pub(crate) fn binding_words(&self, out: &mut Vec<u64>) {
        out.push(self.layers.len() as u64);
        out.extend(self.layers.iter().map(|&l| l as u64));
        out.push(self.dst.buf.as_ptr() as u64);
        out.push(self.dst.numel() as u64);
        out.push(self.row0 as u64);
    }
}

/// Copy the `rows` stream rows of `streams` into capture slot `k`.
fn capture_streams(
    gpu: &mut Gpu,
    cfg: &RaonConfig,
    cap: &DflashCapture<'_>,
    k: usize,
    streams: &GpuTensor,
    rows: usize,
) -> Result<(), String> {
    let width = cfg.mhc_streams * cfg.hidden;
    let (dst_stride, dst_col) = capture_geometry(cap.layers.len(), width, cap.row0, k);
    gpu.copy_rows_strided_f32(
        streams, cap.dst, rows, width, width, dst_stride, dst_col, None,
    )
    .map_err(|e| {
        format!(
            "raon1_mini: DFlash capture of layer {}: {e:?}",
            cap.layers[k]
        )
    })
}

/// Per-RDA-layer recurrence-input ring of an armed verify (rollback source,
/// [`crate::rollback`]). One F32 tensor per RDA layer ordinal holding, each as
/// a row-contiguous `[max_rows, *]` plane and in this order, `q_raw`, `k_raw`,
/// `v_raw`, `q`, `k`, `vprime`, `log_alpha` (`P = heads * head_dim` wide) and
/// `beta` (`heads` wide): `max_rows * (7 P + heads)` elements per layer.
///
/// Armed, the forward points its RDA projections (raw q/k/v) and
/// `raon_rda_prepare`'s outputs at these planes instead of the scratch ones: the
/// launch sequence is the unarmed one (no extra copy), only the pointers differ.
/// The addresses never move after allocation, so an armed verify is
/// graph-capture safe.
#[derive(Clone, Copy)]
pub struct RdaRing<'a> {
    /// One tensor per RDA layer ordinal.
    pub layers: &'a [GpuTensor],
    /// Rows each plane holds.
    pub max_rows: usize,
}

impl RdaRing<'_> {
    /// Planes of `max_rows * P` elements per layer (the `beta` plane follows them).
    pub const WIDE_PLANES: usize = 7;

    /// F32 elements of one layer's ring for `max_rows` rows.
    pub fn layer_elems(max_rows: usize, proj: usize, heads: usize) -> usize {
        max_rows * (Self::WIDE_PLANES * proj + heads)
    }

    /// Non-owning plane views of RDA layer `ri`.
    pub(crate) fn tape(&self, ri: usize, geom: &RdaGeometry) -> RdaTape {
        let plane = self.max_rows * geom.proj();
        let t = &self.layers[ri];
        RdaTape {
            q_raw: t.sub_offset(0, plane),
            k_raw: t.sub_offset(plane, plane),
            v_raw: t.sub_offset(2 * plane, plane),
            q: t.sub_offset(3 * plane, plane),
            k: t.sub_offset(4 * plane, plane),
            vprime: t.sub_offset(5 * plane, plane),
            log_alpha: t.sub_offset(6 * plane, plane),
            beta: t.sub_offset(7 * plane, self.max_rows * geom.n_heads),
        }
    }

    pub(crate) fn check(
        &self,
        geom: &RdaGeometry,
        n_rda: usize,
        rows: usize,
    ) -> Result<(), String> {
        let need = Self::layer_elems(self.max_rows, geom.proj(), geom.n_heads);
        if self.layers.len() != n_rda || rows > self.max_rows {
            return Err(format!(
                "raon1_mini: RDA ring holds {} layers x {} rows, call needs {n_rda} layers x {rows} rows",
                self.layers.len(),
                self.max_rows
            ));
        }
        if let Some(t) = self
            .layers
            .iter()
            .find(|t| t.dtype != rdna_compute::DType::F32 || t.numel() < need)
        {
            return Err(format!(
                "raon1_mini: RDA ring layer holds {} {:?} elements, {need} F32 needed",
                t.numel(),
                t.dtype
            ));
        }
        Ok(())
    }

    /// Every buffer address and extent the armed forward bakes into a launch
    /// (graph-key words).
    pub(crate) fn binding_words(&self, out: &mut Vec<u64>) {
        out.push(self.max_rows as u64);
        out.push(self.layers.len() as u64);
        for t in self.layers {
            out.push(t.buf.as_ptr() as usize as u64);
            out.push(t.numel() as u64);
        }
    }
}

// ───────────────────────── mHC site bindings ─────────────────────────

fn mix_op<'a>(shape: &RaonMhcShape, rows: usize, stage: StreamMixStage<'a>) -> StreamMixOp<'a> {
    StreamMixOp {
        rows,
        streams: shape.streams,
        width: shape.width,
        rounding: shape.rounding,
        stage,
    }
}

fn run_mix(gpu: &mut Gpu, op: &StreamMixOp<'_>) -> Result<(), String> {
    execute_stream_mix(gpu, op).map_err(|e| e.to_string())
}

/// Controls of one site from the OLD streams (selector/params are the loaded
/// `[17, n*hidden]` F32 matrix and the F32 parameter vector of the site).
fn mhc_route(
    gpu: &mut Gpu,
    work: &RaonWork,
    site: &MhcSiteWeights,
    rows: usize,
    streams: &GpuTensor,
) -> Result<(), String> {
    let shape = &work.mhc_shape;
    let op = mix_op(
        shape,
        rows,
        StreamMixStage::Route {
            streams,
            chart: StreamChart::HelmertBounded {
                selector: &site.selector.buf,
                params: &site.params,
                post_factor: shape.post_factor,
                selector_eps: shape.selector_eps,
            },
            selector_out: work.mhc.selector_out(),
            controls: work.mhc.controls(),
        },
    );
    run_mix(gpu, &op)
}

fn mhc_aggregate(
    gpu: &mut Gpu,
    work: &RaonWork,
    rows: usize,
    streams: &GpuTensor,
    norm: StreamNorm<'_>,
    out: &GpuTensor,
) -> Result<(), String> {
    let op = mix_op(
        &work.mhc_shape,
        rows,
        StreamMixStage::Aggregate {
            streams,
            controls: work.mhc.controls(),
            norm: Some(norm),
            out,
        },
    );
    run_mix(gpu, &op)
}

fn mhc_postmix(
    gpu: &mut Gpu,
    work: &RaonWork,
    rows: usize,
    streams: &GpuTensor,
    branch: &GpuTensor,
    streams_out: &GpuTensor,
) -> Result<(), String> {
    let op = mix_op(
        &work.mhc_shape,
        rows,
        StreamMixStage::Postmix {
            streams,
            branch,
            controls: work.mhc.controls(),
            streams_out,
        },
    );
    run_mix(gpu, &op)
}

// ───────────────────────── mixers ─────────────────────────

/// RDA layer: projections (BF16-rounded) -> `rda_forward_rows` -> `o` projection.
/// Reads `scratch.branch_in`, writes `scratch.branch_out`. With a `ring` the raw
/// q/k/v projections and `raon_rda_prepare`'s outputs land in the ring's planes
/// (same launches, different pointers).
#[allow(clippy::too_many_arguments)]
fn rda_mixer(
    gpu: &mut Gpu,
    w: &RdaWeights,
    ri: usize,
    s: &RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    rows: usize,
    ring: Option<&RdaRing<'_>>,
) -> Result<(), String> {
    let x = &scratch.branch_in;
    let b = OutRound::Bf16;
    let tape = ring.map(|r| r.tape(ri, &work.rda_geom));
    let (q_raw, k_raw, v_raw) = match &tape {
        Some(t) => (&t.q_raw, &t.k_raw, &t.v_raw),
        None => (&scratch.rda_q_raw, &scratch.rda_k_raw, &scratch.rda_v_raw),
    };
    project(gpu, &w.q_proj, x, q_raw, rows, b)?;
    project(gpu, &w.k_proj, x, k_raw, rows, b)?;
    project(gpu, &w.v_proj, x, v_raw, rows, b)?;
    project(gpu, &w.g_proj, x, &scratch.rda_z, rows, b)?;
    project(gpu, &w.b_proj, x, &scratch.rda_b, rows, b)?;
    project(gpu, &w.rda_w_proj, x, &scratch.rda_delta, rows, b)?;
    project(gpu, &w.f_a_proj, x, &scratch.rda_f_lat, rows, b)?;
    project(
        gpu,
        &w.f_b_proj,
        &scratch.rda_f_lat,
        &scratch.rda_f,
        rows,
        b,
    )?;

    let layer = RaonRdaLayer {
        conv_w_q: &w.q_conv,
        conv_w_k: &w.k_conv,
        conv_w_v: &w.v_conv,
        a_log: &w.a_log,
        dt_bias: &w.dt_bias,
        o_norm_gamma: &w.o_norm,
    };
    let st = RdaLayerState {
        state: &s.rda[ri],
        hist_q: &s.conv_q[ri],
        hist_k: &s.conv_k[ri],
        hist_v: &s.conv_v[ri],
        slot: 0,
        state_slot_stride: work.rda_geom.state_elems(),
    };
    let proj = RdaProjected {
        q_raw,
        k_raw,
        v_raw,
        z: &scratch.rda_z,
        b: &scratch.rda_b,
        delta: &scratch.rda_delta,
        f: &scratch.rda_f,
    };
    rda_forward_rows_taped(
        gpu,
        &work.rda_geom,
        &layer,
        &st,
        &proj,
        &work.rda,
        &scratch.rda_gated,
        rows,
        tape.as_ref(),
    )?;
    project(
        gpu,
        &w.o_proj,
        &scratch.rda_gated,
        &scratch.branch_out,
        rows,
        b,
    )
}

fn latent_geometry(cfg: &RaonConfig) -> LatentGeometry {
    LatentGeometry {
        n_heads: cfg.n_heads,
        nope: cfg.qk_nope,
        tail: cfg.qk_rope,
        v_dim: cfg.v_head,
        latent: cfg.kv_lora,
    }
}

fn run_latent(gpu: &mut Gpu, op: LatentAttentionOp<'_>) -> Result<(), String> {
    execute_latent_attention(gpu, &op).map_err(|e| e.to_string())
}

/// Expand cached latent rows `range` of one MLA layer into the K/V mirror
/// through the layer's own (quantized) `kv_b` projection:
/// gather -> `kv_b` projection (exact F32) -> BF16 scatter. `range` spans at most
/// [`MLA_TILE_ROWS`] rows.
#[allow(clippy::too_many_arguments)]
fn mla_expand(
    gpu: &mut Gpu,
    cfg: &RaonConfig,
    w: &MlaWeights,
    scratch: &RaonPrefillScratch,
    cache: LatentCache<'_>,
    mirror: DMirror<'_>,
    range: RowRange,
    count: usize,
) -> Result<(), String> {
    let geometry = latent_geometry(cfg);
    run_latent(
        gpu,
        LatentAttentionOp::GatherLatent {
            geometry,
            cache,
            range,
            out: &scratch.mla_latent_plane,
        },
    )?;
    project(
        gpu,
        &w.kv_b_proj,
        &scratch.mla_latent_plane,
        &scratch.mla_kv_tile,
        count,
        OutRound::Exact,
    )?;
    run_latent(
        gpu,
        LatentAttentionOp::Decompress {
            geometry,
            cache,
            mirror,
            expansion: hipfire_dispatch::pipeline::latent_attention::KvExpansion::External {
                projected: &scratch.mla_kv_tile,
            },
            range,
        },
    )
}

/// Route of a call that leaves `end` rows cached (pure: policy x admission
/// budget, no device state).
pub fn mla_route(work: &RaonWork, s: &RaonState, end: usize) -> MlaRoute {
    work.mla.policy().route(end, s.mirror_budget_rows())
}

/// MLA layer: PLAN §1.6 over the expanded mirror (reference or tiled
/// kernels) or, beyond the mirror budget with `HIPFIRE_RAON_MLA_ABSORB=1`,
/// over the latent cache alone. Reads `scratch.branch_in`, writes
/// `scratch.branch_out`.
#[allow(clippy::too_many_arguments)]
fn mla_mixer(
    gpu: &mut Gpu,
    cfg: &RaonConfig,
    w: &MlaWeights,
    mi: usize,
    s: &RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    rows: usize,
    start: usize,
) -> Result<(), String> {
    let x = &scratch.branch_in;
    let b = OutRound::Bf16;
    let geometry = latent_geometry(cfg);
    let route = mla_route(work, s, start + rows);
    let dmirror = match route {
        MlaRoute::Decompressed => {
            let mirror = s
                .expanded_mirror()
                .ok_or("raon1_mini: MLA attention needs a backed expanded K/V mirror")?;
            Some(DMirror {
                knope: &mirror.knope[mi],
                vmir: &mirror.vmir[mi],
                capacity_rows: mirror.rows,
                q8: mirror.q8,
            })
        }
        MlaRoute::Absorbed => None,
    };
    let cache = LatentCache {
        rows: &s.mla_cache[mi],
        capacity_rows: s.capacity_rows(),
        pos: &s.seq_len,
        host_pos: Some(start),
    };

    project(gpu, &w.q_a_proj, x, &scratch.mla_qa, rows, b)?;
    project(gpu, &w.kv_a_proj, x, &scratch.mla_kv_a, rows, b)?;
    project(gpu, &w.g_proj, x, &scratch.mla_g, rows, b)?;

    run_latent(
        gpu,
        LatentAttentionOp::NormRows {
            x: &scratch.mla_qa,
            gamma: &w.q_a_norm,
            out: &scratch.mla_qa_norm,
            rows,
            width: cfg.q_lora,
            eps: RAON_MLA_EPS,
        },
    )?;
    project(
        gpu,
        &w.q_b_proj,
        &scratch.mla_qa_norm,
        &scratch.mla_q,
        rows,
        b,
    )?;
    run_latent(
        gpu,
        LatentAttentionOp::SplitKv {
            geometry,
            cache,
            kv_a: &scratch.mla_kv_a,
            gamma: &w.kv_a_norm,
            rows,
            eps: RAON_MLA_EPS,
        },
    )?;
    if let Some(dm) = dmirror {
        if rows <= MLA_TILE_ROWS {
            mla_expand(
                gpu,
                cfg,
                w,
                scratch,
                cache,
                dm,
                RowRange::NewRows { count: rows },
                rows,
            )?;
        } else {
            // A chunk wider than the decompression tile: expand the new rows
            // tile by tile at their absolute positions (`start` is the host
            // copy of the device position `NewRows` would read). Prefill only,
            // never captured.
            let mut done = 0;
            while done < rows {
                let count = MLA_TILE_ROWS.min(rows - done);
                mla_expand(
                    gpu,
                    cfg,
                    w,
                    scratch,
                    cache,
                    dm,
                    RowRange::Absolute {
                        start: start + done,
                        count,
                    },
                    count,
                )?;
                done += count;
            }
        }
    }
    run_latent(
        gpu,
        LatentAttentionOp::Attention {
            geometry,
            cache,
            mirror: dmirror,
            q: &scratch.mla_q,
            y: &scratch.mla_attn,
            rows,
            scale: raon_mla_scale(&geometry),
            route: work.mla.attention_route(route, mi)?,
        },
    )?;
    run_latent(
        gpu,
        LatentAttentionOp::Gate {
            y: &scratch.mla_attn,
            g: &scratch.mla_g,
            out: &scratch.mla_attn,
            elements: rows * cfg.n_heads * cfg.v_head,
        },
    )?;
    project(
        gpu,
        &w.o_proj,
        &scratch.mla_attn,
        &scratch.branch_out,
        rows,
        b,
    )
}

// ───────────────────────── FFN ─────────────────────────

fn ffn_block(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    work: &RaonWork,
    layer: &RaonLayerWeights,
    scratch: &RaonPrefillScratch,
    rows: usize,
) -> Result<(), String> {
    match &layer.ffn {
        RaonFfn::Dense(d) => {
            let (gate, up, down) = d.projections(rows);
            let refs = RaonDenseRefs::from_parts(gate, up, down)?;
            forward_dense(
                gpu,
                ctx,
                &work.moe_dims,
                &refs,
                rows,
                &scratch.branch_in,
                &work.dense,
                &scratch.ffn_out,
            )
        }
        RaonFfn::Moe(m) => {
            let (shared_gate, shared_up, shared_down) = m.shared.projections(rows);
            let refs = RaonMoeLayerRefs::from_parts(
                &m.router,
                &m.e_bias,
                shared_gate,
                shared_up,
                shared_down,
                &m.expert_gate_up_ptrs,
                &m.expert_down_ptrs,
                m.gate_up_awq.as_ref(),
                m.expert_down_awq_ptrs.as_ref(),
            )
            .with_iu4_symmetric(m.iu4_symmetric);
            forward_moe(
                gpu,
                ctx,
                &work.moe_dims,
                &refs,
                rows,
                &scratch.branch_in,
                &work.moe,
                &scratch.ffn_out,
            )
        }
    }
}

// ───────────────────────── mirror admission ─────────────────────────

/// Make sure `s` can attend over `end` rows: map the latent caches ahead and,
/// on the decompressed route, back the expanded K/V mirror (in place, VMM) and
/// bring it level with the live prefix (rows `[valid, start)` are expanded
/// from the latent cache through each layer's `kv_b`; rows of the call itself
/// are expanded by the MLA mixers). A context beyond the mirror budget is
/// refused with the maximum supported context unless the absorbed route is
/// enabled, which needs no mirror. Outside capture only.
fn ensure_context(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    gpu: &mut Gpu,
    end: usize,
) -> Result<(), String> {
    s.ensure_capacity(gpu, end)?;
    if cfg.n_mla() == 0 {
        return Ok(());
    }
    let start = s.n_tokens();
    // Rows at or beyond `start` are about to be rewritten.
    s.clamp_mirror_valid_rows(start);
    match mla_route(work, s, end) {
        MlaRoute::Absorbed => {
            if !work.mla.absorbed_ready() {
                return Err(
                    "raon1_mini: absorbed MLA route selected but its packed weights are not built"
                        .into(),
                );
            }
            Ok(())
        }
        MlaRoute::Decompressed => {
            s.ensure_mirror_rows(gpu, end)?;
            let valid = s.mirror_valid_rows();
            if valid < start {
                let mirror = s.expanded_mirror().expect("mirror just backed");
                for (mi, layer_idx) in cfg.mla_layers().into_iter().enumerate() {
                    let RaonMixer::Mla(mw) = &w.layers[layer_idx].mixer else {
                        return Err(format!("raon1_mini: layer {layer_idx} is not an MLA layer"));
                    };
                    let cache = LatentCache {
                        rows: &s.mla_cache[mi],
                        capacity_rows: s.capacity_rows(),
                        pos: &s.seq_len,
                        host_pos: None,
                    };
                    let dmirror = DMirror {
                        knope: &mirror.knope[mi],
                        vmir: &mirror.vmir[mi],
                        capacity_rows: mirror.rows,
                        q8: mirror.q8,
                    };
                    let mut done = valid;
                    while done < start {
                        let count = MLA_TILE_ROWS.min(scratch.rows).min(start - done);
                        mla_expand(
                            gpu,
                            cfg,
                            mw,
                            scratch,
                            cache,
                            dmirror,
                            RowRange::Absolute { start: done, count },
                            count,
                        )?;
                        done += count;
                    }
                }
            }
            s.set_mirror_valid_rows(end);
            Ok(())
        }
    }
}

// ───────────────────────── embedding / head ─────────────────────────

/// Validate `tokens` and upload them to the slot's stable device id buffer.
/// Host side of the embedding: never part of a captured region (a captured
/// graph reads the same buffer, which the caller refreshes before each replay).
fn stage_tokens(
    cfg: &RaonConfig,
    scratch: &RaonPrefillScratch,
    gpu: &mut Gpu,
    tokens: &[u32],
) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for &t in tokens {
        if t as usize >= cfg.vocab {
            return Err(format!(
                "raon1_mini: token id {t} outside vocab {}",
                cfg.vocab
            ));
        }
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    gpu.memcpy_htod_auto(&scratch.token_ids.buf, &bytes)
        .map_err(|e| format!("raon1_mini: upload token ids: {e:?}"))
}

/// Embedding lookup of the first `rows` staged ids into `out` (`[rows, hidden]`
/// F32): the raw table rows, no scaling. Device work only; capture-safe.
pub(crate) fn embed_lookup(
    cfg: &RaonConfig,
    w: &RaonWeights,
    scratch: &RaonPrefillScratch,
    gpu: &mut Gpu,
    rows: usize,
    out: &GpuTensor,
) -> Result<(), String> {
    let res = match w.embed_format {
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8_batched(&w.embed, out, &scratch.token_ids, rows, cfg.hidden)
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup_f32_batched(&w.embed, out, &scratch.token_ids, rows, cfg.hidden)
        }
        other => {
            return Err(format!(
                "raon1_mini: embedding format {other:?} is not routed (expected Q8F16 or F32)"
            ));
        }
    };
    res.map_err(|e| format!("raon1_mini: embedding lookup: {e:?}"))
}

/// Embedding lookup of the first `rows` staged ids + stream broadcast. Device
/// work only (reads `scratch.token_ids`); capture-safe.
fn embed_staged(
    cfg: &RaonConfig,
    w: &RaonWeights,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    gpu: &mut Gpu,
    rows: usize,
) -> Result<(), String> {
    embed_lookup(cfg, w, scratch, gpu, rows, &scratch.branch_out)?;
    broadcast_embedding(
        gpu,
        &work.mhc_shape,
        rows,
        &scratch.branch_out,
        cfg.hidden,
        &scratch.stream_a,
    )
}

/// Final mean + RMSNorm + lm_head for rows `row0..row0+count` of
/// `scratch.stream_a` into `out` (`[count, vocab]` F32, BF16-valued).
/// Uses `scratch.branch_in` as the normalized-hidden staging plane.
pub fn readout_logits(
    cfg: &RaonConfig,
    w: &RaonWeights,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    row0: usize,
    count: usize,
    out: &GpuTensor,
) -> Result<(), String> {
    if count == 0 || row0 + count > scratch.rows || count > work.rows {
        return Err(format!(
            "raon1_mini: readout of rows {row0}..{} outside the {} scratch rows",
            row0 + count,
            scratch.rows
        ));
    }
    let width = cfg.mhc_streams * cfg.hidden;
    let view = scratch.stream_a.sub_offset(row0 * width, count * width);
    let norm = StreamNorm {
        gamma: &w.final_norm,
        eps: cfg.rms_eps,
    };
    final_mean_norm(gpu, &work.mhc_shape, count, &view, norm, &scratch.branch_in)?;
    project(
        gpu,
        &w.lm_head,
        &scratch.branch_in,
        out,
        count,
        OutRound::Bf16,
    )
}

// ───────────────────────── the forward ─────────────────────────

#[allow(clippy::too_many_arguments)]
fn forward_impl(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    want_logits: bool,
    mut tap: Option<LayerTap<'_>>,
    capture: Option<&DflashCapture<'_>>,
    ring: Option<&RdaRing<'_>>,
) -> Result<(), String> {
    let rows = tokens.len();
    if rows == 0 || rows > scratch.rows || rows > work.rows {
        return Err(format!(
            "raon1_mini: {rows} rows outside 1..={} (scratch rows {}, work rows {})",
            scratch.rows.min(work.rows),
            scratch.rows,
            work.rows
        ));
    }
    if w.layers.len() != cfg.n_layers {
        return Err(format!(
            "raon1_mini: weights hold {} layers, config {}",
            w.layers.len(),
            cfg.n_layers
        ));
    }
    if let Some(cap) = capture {
        cap.check(cfg, rows)?;
    }
    if let Some(r) = ring {
        r.check(&work.rda_geom, cfg.n_rda(), rows)?;
    }
    let start = start_pos as usize;
    if start != s.n_tokens() {
        return Err(format!(
            "raon1_mini: position {start} does not match the slot's consumed tokens {}",
            s.n_tokens()
        ));
    }
    let end = start + rows;
    ensure_context(cfg, w, s, scratch, work, gpu, end)?;
    stage_tokens(cfg, scratch, gpu, tokens)?;
    forward_body(
        cfg,
        w,
        s,
        scratch,
        work,
        gpu,
        rows,
        start,
        want_logits,
        &mut tap,
        capture,
        ring,
    )?;
    // One advance per call, after every MLA layer consumed the old length.
    s.set_n_tokens(gpu, end)
}

/// Admit `end` rows of context for a decode step: map the latent caches ahead
/// and back the expanded K/V mirror (in place: mirror and latent addresses
/// never move, but the mapped-row clamp and the route are baked into a
/// captured graph, so the caller re-derives [`decode_binding`] afterwards).
/// A context beyond the mirror budget is refused with the maximum supported
/// context unless the absorbed route is enabled. Outside capture only.
pub fn admit_decode_context(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    gpu: &mut Gpu,
    end: usize,
) -> Result<(), String> {
    ensure_context(cfg, w, s, scratch, work, gpu, end)
}

/// Upload the one decode token to the stable device id buffer (host work,
/// outside any captured region; refreshed before every replay).
pub fn stage_decode_token(
    cfg: &RaonConfig,
    scratch: &RaonPrefillScratch,
    gpu: &mut Gpu,
    token: u32,
) -> Result<(), String> {
    stage_tokens(cfg, scratch, gpu, &[token])
}

/// Stream-ordered [`stage_decode_token`] for the graph decode route: the id is
/// the immediate of one asynchronous memset on the decode stream
/// ([`Gpu::write_u32_stream_ordered`]), so the token reaches the stable id
/// buffer through the queue that then replays the graph. The plain staging is
/// a blocking copy on the legacy null stream: before every replay the runtime
/// synchronizes the null queue with the decode queue, and on R9700 that hop
/// doubles the per-token time whenever the two queues share a command-processor
/// pipe or another process holds a context on the card.
pub fn stage_decode_token_on_stream(
    cfg: &RaonConfig,
    scratch: &RaonPrefillScratch,
    gpu: &Gpu,
    token: u32,
) -> Result<(), String> {
    if token as usize >= cfg.vocab {
        return Err(format!(
            "raon1_mini: token id {token} outside vocab {}",
            cfg.vocab
        ));
    }
    gpu.write_u32_stream_ordered(&scratch.token_ids.buf, token)
        .map_err(|e| format!("raon1_mini: stage decode token on stream: {e:?}"))
}

/// Device-only decode step (`rows == 1`, logits wanted, no tap): embedding of
/// the staged token id, all layers, head, then a device-side `seq_len += 1`
/// after every kernel that reads it. Issues kernel launches that read and write
/// stable device buffers only (token id, `seq_len`, state, scratch), so it is
/// recordable into a hipGraph and a replay advances the length with no host
/// write. The host mirror (`RaonState::note_host_tokens`) is the caller's.
pub fn decode_body(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    gpu: &mut Gpu,
) -> Result<(), String> {
    forward_body(
        cfg,
        w,
        s,
        scratch,
        work,
        gpu,
        1,
        s.n_tokens(),
        true,
        &mut None,
        None,
        None,
    )?;
    gpu.device_counter_add_i32(&s.seq_len, 1)
        .map_err(|e| format!("raon1_mini: device seq_len advance: {e:?}"))
}

/// Device-only verify body: `rows` rows from the slot's current position, no
/// logits, optional DFlash capture and RDA ring, then a device-side
/// `seq_len += rows` (graph-recordable like [`decode_body`]; the host mirror is
/// the caller's). The ring arms the rollback replay ([`RdaRing`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_body(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    gpu: &mut Gpu,
    rows: usize,
    capture: Option<&DflashCapture<'_>>,
    ring: Option<&RdaRing<'_>>,
) -> Result<(), String> {
    if let Some(cap) = capture {
        cap.check(cfg, rows)?;
    }
    if let Some(r) = ring {
        r.check(&work.rda_geom, cfg.n_rda(), rows)?;
    }
    forward_body(
        cfg,
        w,
        s,
        scratch,
        work,
        gpu,
        rows,
        s.n_tokens(),
        false,
        &mut None,
        capture,
        ring,
    )?;
    gpu.device_counter_add_i32(&s.seq_len, rows as i32)
        .map_err(|e| format!("raon1_mini: device seq_len advance: {e:?}"))
}

/// Identity of every device address and baked extent the decode body reads:
/// weights, state slab, latent caches and their mapped capacity, expanded
/// mirror and its row stride, scratch and work planes. A captured graph is only
/// valid for the binding it was recorded with. Never zero.
pub fn decode_binding(
    w: &RaonWeights,
    s: &RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
) -> u64 {
    rows_binding(w, s, scratch, work, 1, &[])
}

/// [`decode_binding`] of a body over `rows` rows (the MLA route is the one of
/// the context `s.n_tokens() + rows`), with the caller's `extra` identity words
/// (row count, capture/readout buffers, ...) mixed in last.
pub(crate) fn rows_binding(
    w: &RaonWeights,
    s: &RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    rows: usize,
    extra: &[u64],
) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x100_0000_01b3);
    mix(w.embed.buf.as_ptr() as u64);
    mix(w.lm_head.buf.buf.as_ptr() as u64);
    mix(s.seq_len.buf.as_ptr() as u64);
    for t in s
        .rda
        .iter()
        .chain(&s.conv_q)
        .chain(&s.conv_k)
        .chain(&s.conv_v)
        .chain(&s.mla_cache)
    {
        mix(t.buf.as_ptr() as u64);
    }
    mix(s.capacity_rows() as u64);
    if let Some(m) = s.expanded_mirror() {
        mix(m.rows as u64);
        for t in m.knope.iter().chain(&m.vmir) {
            mix(t.buf.as_ptr() as u64);
        }
    }
    // Route (decompressed vs absorbed), split-KV workspace and absorbed operands.
    mix(mla_route(work, s, s.n_tokens() + rows) as u64);
    for a in work.mla.binding_key() {
        mix(a as u64);
    }
    mix(scratch.token_ids.buf.as_ptr() as u64);
    mix(scratch.logits.buf.as_ptr() as u64);
    mix(scratch.stream_a.buf.as_ptr() as u64);
    mix(scratch.stream_b.buf.as_ptr() as u64);
    for &e in extra {
        mix(e);
    }
    h | 1
}

/// Launch sequence of one forward over `rows` staged rows: embedding, layer
/// schedule, head (last row). Pure device work apart from the optional `tap`
/// observer (which downloads and is therefore never used under capture).
#[allow(clippy::too_many_arguments)]
fn forward_body(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    gpu: &mut Gpu,
    rows: usize,
    start: usize,
    want_logits: bool,
    tap: &mut Option<LayerTap<'_>>,
    capture: Option<&DflashCapture<'_>>,
    ring: Option<&RdaRing<'_>>,
) -> Result<(), String> {
    embed_staged(cfg, w, scratch, work, gpu, rows)?;
    fire(tap, gpu, TapSite::Embedding, &scratch.stream_a, rows)?;

    let mut ctx = DispatchCtx::new(gpu);
    ctx.batch_size = rows;
    let eps = cfg.rms_eps;

    // Ping-pong: `cur` holds the OLD streams of a site, `oth` receives the new.
    let mut cur_is_a = true;
    for (l, layer) in w.layers.iter().enumerate() {
        let kind = cfg.layer_kinds[l];

        // ── attention site ──
        {
            let (cur, oth) = if cur_is_a {
                (&scratch.stream_a, &scratch.stream_b)
            } else {
                (&scratch.stream_b, &scratch.stream_a)
            };
            mhc_route(gpu, work, &layer.attn_mhc, rows, cur)?;
            mhc_aggregate(
                gpu,
                work,
                rows,
                cur,
                StreamNorm {
                    gamma: &layer.input_norm,
                    eps,
                },
                &scratch.branch_in,
            )?;
            match (kind, &layer.mixer) {
                (LayerKind::Rda, RaonMixer::Rda(rw)) => {
                    let ri = cfg
                        .rda_index(l)
                        .ok_or_else(|| format!("raon1_mini: layer {l} has no RDA slot"))?;
                    rda_mixer(gpu, rw, ri, s, scratch, work, rows, ring)
                        .map_err(|e| format!("raon1_mini L{l} rda: {e}"))?;
                }
                (LayerKind::Mla, RaonMixer::Mla(mw)) => {
                    let mi = cfg
                        .mla_index(l)
                        .ok_or_else(|| format!("raon1_mini: layer {l} has no MLA slot"))?;
                    mla_mixer(gpu, cfg, mw, mi, s, scratch, work, rows, start)
                        .map_err(|e| format!("raon1_mini L{l} mla: {e}"))?;
                }
                _ => {
                    return Err(format!(
                        "raon1_mini: layer {l} weights do not match its configured mixer"
                    ))
                }
            }
            mhc_postmix(gpu, work, rows, cur, &scratch.branch_out, oth)?;
        }
        cur_is_a = !cur_is_a;
        fire(
            tap,
            gpu,
            TapSite::Attn(l),
            if cur_is_a {
                &scratch.stream_a
            } else {
                &scratch.stream_b
            },
            rows,
        )?;

        // ── FFN site ──
        {
            let (cur, oth) = if cur_is_a {
                (&scratch.stream_a, &scratch.stream_b)
            } else {
                (&scratch.stream_b, &scratch.stream_a)
            };
            mhc_route(gpu, work, &layer.ffn_mhc, rows, cur)?;
            mhc_aggregate(
                gpu,
                work,
                rows,
                cur,
                StreamNorm {
                    gamma: &layer.post_attn_norm,
                    eps,
                },
                &scratch.branch_in,
            )?;
            ffn_block(gpu, &ctx, work, layer, scratch, rows)
                .map_err(|e| format!("raon1_mini L{l} ffn: {e}"))?;
            mhc_postmix(gpu, work, rows, cur, &scratch.ffn_out, oth)?;
        }
        cur_is_a = !cur_is_a;
        fire(
            tap,
            gpu,
            TapSite::Ffn(l),
            if cur_is_a {
                &scratch.stream_a
            } else {
                &scratch.stream_b
            },
            rows,
        )?;
        if let Some(cap) = capture {
            if let Some(k) = cap.layers.iter().position(|&id| id == l) {
                let streams = if cur_is_a {
                    &scratch.stream_a
                } else {
                    &scratch.stream_b
                };
                capture_streams(gpu, cfg, cap, k, streams, rows)?;
            }
        }
    }
    debug_assert!(
        cur_is_a,
        "an even number of sites leaves the result in stream_a"
    );
    if !cur_is_a {
        return Err(
            "raon1_mini: odd number of mHC sites (layer count) — result not in stream_a".into(),
        );
    }

    if want_logits {
        let last = rows - 1;
        let width = cfg.mhc_streams * cfg.hidden;
        let view = scratch.stream_a.sub_offset(last * width, width);
        let norm = StreamNorm {
            gamma: &w.final_norm,
            eps,
        };
        final_mean_norm(gpu, &work.mhc_shape, 1, &view, norm, &scratch.final_norm)?;
        project(
            gpu,
            &w.lm_head,
            &scratch.final_norm,
            &scratch.logits,
            1,
            OutRound::Bf16,
        )?;
    }
    Ok(())
}

/// One token at position `pos` (must equal `s.n_tokens()`); advances the slot by
/// one. With `want_logits`, `scratch.logits` holds this token's `[vocab]` logits.
#[allow(clippy::too_many_arguments)]
pub fn forward_token(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    token: u32,
    pos: u32,
    want_logits: bool,
) -> Result<(), String> {
    forward_token_tapped(cfg, w, s, scratch, work, gpu, token, pos, want_logits, None)
}

/// [`forward_token`] with an optional stream observer.
#[allow(clippy::too_many_arguments)]
pub fn forward_token_tapped(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    token: u32,
    pos: u32,
    want_logits: bool,
    tap: Option<LayerTap<'_>>,
) -> Result<(), String> {
    forward_impl(
        cfg,
        w,
        s,
        scratch,
        work,
        gpu,
        &[token],
        pos,
        want_logits,
        tap,
        None,
        None,
    )
}

/// `tokens.len()` (1..=`scratch.rows`) consecutive tokens starting at
/// `start_pos` (must equal `s.n_tokens()`) through one batched pass: every
/// projection, mHC site, RDA/MLA layer and FFN runs once over all rows. With
/// `want_logits`, `scratch.logits` holds the LAST row's logits. Advances the
/// slot by `tokens.len()`.
#[allow(clippy::too_many_arguments)]
pub fn forward_rows(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    want_logits: bool,
) -> Result<(), String> {
    forward_rows_tapped(
        cfg,
        w,
        s,
        scratch,
        work,
        gpu,
        tokens,
        start_pos,
        want_logits,
        None,
    )
}

/// [`forward_rows`] with an optional stream observer.
#[allow(clippy::too_many_arguments)]
pub fn forward_rows_tapped(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    want_logits: bool,
    tap: Option<LayerTap<'_>>,
) -> Result<(), String> {
    forward_impl(
        cfg,
        w,
        s,
        scratch,
        work,
        gpu,
        tokens,
        start_pos,
        want_logits,
        tap,
        None,
        None,
    )
}

/// [`forward_rows`] with the DFlash2 device capture ([`DflashCapture`]: the
/// post-FFN carrier of the listed layers lands in `capture.dst` rows
/// `capture.row0..capture.row0 + tokens.len()`, one strided copy per listed
/// layer) and the rollback ring ([`RdaRing`]) armed (the RDA recurrence inputs
/// of every row land in `ring`; same launch sequence). This is the verify
/// forward (`crate::rollback`); `capture` is present when the drafter wants
/// this window's features.
#[allow(clippy::too_many_arguments)]
pub fn forward_rows_verify(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    want_logits: bool,
    capture: Option<&DflashCapture<'_>>,
    ring: Option<&RdaRing<'_>>,
) -> Result<(), String> {
    forward_impl(
        cfg,
        w,
        s,
        scratch,
        work,
        gpu,
        tokens,
        start_pos,
        want_logits,
        None,
        capture,
        ring,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_layers_must_be_ascending_and_in_range() {
        assert!(validate_extract_layers(&[1, 6, 11, 16, 21], 25).is_ok());
        assert!(validate_extract_layers(&[24], 25).is_ok());
        assert!(validate_extract_layers(&[], 25).is_err());
        assert!(validate_extract_layers(&[6, 1], 25).is_err());
        assert!(validate_extract_layers(&[1, 1], 25).is_err());
        assert!(validate_extract_layers(&[1, 25], 25).is_err());
    }

    #[test]
    fn capture_geometry_is_layer_major_within_a_row() {
        let width = 4 * 2560;
        let (stride, col) = capture_geometry(5, width, 0, 0);
        assert_eq!((stride, col), (51200, 0));
        let (stride, col) = capture_geometry(5, width, 0, 3);
        assert_eq!((stride, col), (51200, 3 * 10240));
        // A later call row offsets by whole destination rows.
        let (stride, col) = capture_geometry(5, width, 7, 4);
        assert_eq!((stride, col), (51200, 7 * 51200 + 4 * 10240));
    }
}
