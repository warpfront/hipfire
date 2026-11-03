// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon-1-mini forward composition (`forward_token` / `forward_rows`). Owner: slice I.
//!
//! One layer schedule serves decode (`rows == 1`) and bounded prefill chunks
//! (`rows == N`): every site is a typed step program ([`crate::program`]) over
//! `rows` rows, so a chunk is genuinely batched (projections, mHC, RDA, MLA,
//! dense/MoE) and a single token is the `rows == 1` instance of the same
//! schedule. This module sequences the sites, stages host input, admits the
//! context and fires the oracle taps between sites; it launches nothing itself.
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
//! crosses `Step::BoundaryProject`'s BF16 boundary; the stream-mix, RDA,
//! latent-attention and MoE ops apply their own documented boundaries.
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
    ExpandedMirror as DMirror, LatentCache, RowRange,
};
use hipfire_dispatch::pipeline::rda::{RdaGeometry, RdaScratch, RdaTape};
use hipfire_dispatch::pipeline::{execute_steps, RowsRoute, Step};
use rdna_compute::{Gpu, GpuTensor};

use crate::config::{LayerKind, RaonConfig};
use crate::mhc::{RaonMhcShape, RaonMhcWork};
use crate::mla::{MlaRoute, RaonMlaWork};
use crate::moe::{RaonDenseScratch, RaonMoeDims, RaonMoeScratch};
use crate::precision::{RaonPrecisionSite, RaonSiteTiers};
use crate::program;
use crate::state::{RaonPrefillScratch, RaonState, MLA_TILE_ROWS};
use crate::weights::{RaonMixer, RaonWeights};

// ───────────────────────── module scratch ─────────────────────────

/// Module-owned work buffers of the forward (mHC controls, RDA, MoE, dense
/// FFN), sized for `rows` rows like the [`RaonPrefillScratch`] they accompany.
/// Owned once per bundle; reused by every layer and every call.
pub struct RaonWork {
    pub(crate) rows: usize,
    pub(crate) mhc_shape: RaonMhcShape,
    pub(crate) mhc: RaonMhcWork,
    pub(crate) rda_geom: RdaGeometry,
    pub(crate) rda: RdaScratch,
    pub(crate) moe_dims: RaonMoeDims,
    pub(crate) moe: RaonMoeScratch,
    pub(crate) dense: RaonDenseScratch,
    /// MLA route controls, split-KV workspace and absorbed-route operands.
    pub mla: RaonMlaWork,
    /// Effective prefill activation tier of every site, resolved once at load
    /// ([`crate::precision`]).
    pub(crate) site_tiers: RaonSiteTiers,
}

impl RaonWork {
    /// Install the load-resolved effective prefill tier of every site (bundle load, once). The
    /// FFN sites are carried to the MoE/dense programs through the dims; the mixer sites are
    /// read by the projections.
    pub fn set_site_tiers(&mut self, tiers: RaonSiteTiers) {
        self.site_tiers = tiers;
        self.moe_dims.tiers = tiers;
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
            tiers: RaonSiteTiers::ALL_F16,
        };
        moe_dims.validate()?;
        Ok((mhc_shape, rda_geom, moe_dims))
    }

    /// Device bytes of the work for `rows` rows once the bundle is assembled
    /// (every module buffer plus the packed absorbed MLA weights): what the
    /// prefill chunk admission charges against free memory. `chunked_rda_arch`: the device has
    /// the chunked RDA kernel ([`Gpu::gated_delta_vector_chunked_arch_supported`]).
    pub fn required_bytes(
        cfg: &RaonConfig,
        rows: usize,
        chunked_rda_arch: bool,
    ) -> Result<usize, String> {
        if rows == 0 {
            return Err("raon work: rows == 0".into());
        }
        let (mhc_shape, rda_geom, moe_dims) = Self::shapes(cfg)?;
        Ok(RaonMhcWork::required_bytes(&mhc_shape, rows)
            + RdaScratch::required_bytes(&rda_geom, rows, chunked_rda_arch)
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
            site_tiers: RaonSiteTiers::ALL_F16,
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
    if let Some(c) = r.chunk {
        let _ = gpu.free_gated_delta_vector_chunk_scratch(c);
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
    /// An intermediate buffer of this layer (census taps): `Stage(layer, name)`
    /// with a `[rows, width]` F32 view (`width = numel / rows`). The head uses
    /// layer `n_layers`. Fired only when a tap is armed; never under capture.
    Stage(usize, &'static str),
}

/// Observer of the stream state (`[rows, streams, hidden]` F32 holding BF16
/// values). Oracle only: a tap downloads and is not graph-capturable. With BF16
/// stream planes (`HIPFIRE_RAON_MHC_BF16_STREAMS=1`) the forward widens them
/// exactly into a temporary F32 tensor first, so a tap always sees F32.
pub type LayerTap<'a> =
    &'a mut dyn FnMut(&mut Gpu, TapSite, &GpuTensor, usize) -> Result<(), String>;

fn fire(
    tap: &mut Option<LayerTap<'_>>,
    gpu: &mut Gpu,
    site: TapSite,
    streams: &GpuTensor,
    rows: usize,
    width: usize,
) -> Result<(), String> {
    let Some(t) = tap.as_mut() else {
        return Ok(());
    };
    if streams.dtype != rdna_compute::DType::BF16 {
        return (**t)(gpu, site, streams, rows);
    }
    let wide = gpu
        .alloc_tensor(&[rows, width], rdna_compute::DType::F32)
        .map_err(|e| format!("raon1_mini: tap staging alloc: {e:?}"))?;
    let widened = {
        let widen = [program::capture_streams(
            rows, width, streams, &wide, width, 0,
        )];
        let ctx = DispatchCtx::new(gpu);
        run(gpu, &ctx, &widen).map_err(|e| format!("raon1_mini: tap widen: {e}"))
    };
    let r = widened.and_then(|()| (**t)(gpu, site, &wide, rows));
    let _ = gpu.free_tensor(wide);
    r
}

/// Run one step program (validated first, then launched in order).
pub(crate) fn run(gpu: &mut Gpu, ctx: &DispatchCtx, steps: &[Step<'_>]) -> Result<(), String> {
    execute_steps(gpu, ctx, steps).map_err(|e| e.to_string())
}

/// Fire a stage tap with the first `rows` rows of a `[cap_rows, width]` plane
/// (`width = numel / cap_rows`) as a `[rows, width]` view.
fn fire_stage_scratch(
    tap: &mut Option<LayerTap<'_>>,
    gpu: &mut Gpu,
    layer: usize,
    name: &'static str,
    t: &GpuTensor,
    rows: usize,
    cap_rows: usize,
) -> Result<(), String> {
    if tap.is_none() {
        return Ok(());
    }
    fire_stage_width(tap, gpu, layer, name, t, rows, t.numel() / cap_rows)
}

/// [`fire_stage_scratch`] with an explicit row width (ring planes).
fn fire_stage_width(
    tap: &mut Option<LayerTap<'_>>,
    gpu: &mut Gpu,
    layer: usize,
    name: &'static str,
    t: &GpuTensor,
    rows: usize,
    width: usize,
) -> Result<(), String> {
    if tap.is_none() {
        return Ok(());
    }
    let view = t.sub_offset(0, rows * width);
    fire(tap, gpu, TapSite::Stage(layer, name), &view, rows, width)
}

/// Census taps of an mHC site's controls (`prefix` is `attn` or `ffn`).
fn stage_route(
    tap: &mut Option<LayerTap<'_>>,
    gpu: &mut Gpu,
    work: &RaonWork,
    l: usize,
    rows: usize,
    prefix: &'static str,
) -> Result<(), String> {
    if tap.is_none() {
        return Ok(());
    }
    let names: [&'static str; 4] = if prefix == "attn" {
        [
            "attn.route.sel",
            "attn.route.pre",
            "attn.route.post",
            "attn.route.res",
        ]
    } else {
        [
            "ffn.route.sel",
            "ffn.route.pre",
            "ffn.route.post",
            "ffn.route.res",
        ]
    };
    let c = work.mhc.controls();
    let planes = [work.mhc.selector_out(), c.pre, c.post, c.res];
    for (name, t) in names.into_iter().zip(planes) {
        fire_stage_scratch(tap, gpu, l, name, t, rows, work.rows)?;
    }
    Ok(())
}

/// Census taps of an RDA layer's intermediates (after `rda_mixer`): the
/// projections, the conv / prepare / recurrence outputs and the gated plane.
/// With an armed ring the raw projections and prepare outputs live in the ring
/// planes, not the scratch ones.
#[allow(clippy::too_many_arguments)]
fn stage_rda(
    tap: &mut Option<LayerTap<'_>>,
    gpu: &mut Gpu,
    l: usize,
    rows: usize,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    ring: Option<&RdaRing<'_>>,
    ri: usize,
) -> Result<(), String> {
    if tap.is_none() {
        return Ok(());
    }
    let proj = work.rda_geom.proj();
    let heads = work.rda_geom.n_heads;
    let cap = scratch.rows;
    let rcap = work.rda.max_rows;
    let tape = ring.map(|r| r.tape(ri, &work.rda_geom));
    match &tape {
        Some(t) => {
            for (name, p, w) in [
                ("rda.q_raw", &t.q_raw, proj),
                ("rda.k_raw", &t.k_raw, proj),
                ("rda.v_raw", &t.v_raw, proj),
            ] {
                fire_stage_width(tap, gpu, l, name, p, rows, w)?;
            }
        }
        None => {
            fire_stage_scratch(tap, gpu, l, "rda.q_raw", &scratch.rda_q_raw, rows, cap)?;
            fire_stage_scratch(tap, gpu, l, "rda.k_raw", &scratch.rda_k_raw, rows, cap)?;
            fire_stage_scratch(tap, gpu, l, "rda.v_raw", &scratch.rda_v_raw, rows, cap)?;
        }
    }
    for (name, t) in [
        ("rda.z", &scratch.rda_z),
        ("rda.b", &scratch.rda_b),
        ("rda.delta", &scratch.rda_delta),
        ("rda.f_lat", &scratch.rda_f_lat),
        ("rda.f", &scratch.rda_f),
    ] {
        fire_stage_scratch(tap, gpu, l, name, t, rows, cap)?;
    }
    fire_stage_scratch(tap, gpu, l, "rda.qconv", &work.rda.qconv, rows, rcap)?;
    fire_stage_scratch(tap, gpu, l, "rda.kconv", &work.rda.kconv, rows, rcap)?;
    fire_stage_scratch(tap, gpu, l, "rda.vconv", &work.rda.vconv, rows, rcap)?;
    match &tape {
        Some(t) => {
            for (name, p, w) in [
                ("rda.q", &t.q, proj),
                ("rda.k", &t.k, proj),
                ("rda.vprime", &t.vprime, proj),
                ("rda.log_alpha", &t.log_alpha, proj),
                ("rda.beta", &t.beta, heads),
            ] {
                fire_stage_width(tap, gpu, l, name, p, rows, w)?;
            }
        }
        None => {
            for (name, p) in [
                ("rda.q", &work.rda.q),
                ("rda.k", &work.rda.k),
                ("rda.vprime", &work.rda.vprime),
                ("rda.log_alpha", &work.rda.log_alpha),
                ("rda.beta", &work.rda.beta),
            ] {
                fire_stage_scratch(tap, gpu, l, name, p, rows, rcap)?;
            }
        }
    }
    fire_stage_scratch(tap, gpu, l, "rda.y", &work.rda.y, rows, rcap)?;
    fire_stage_scratch(tap, gpu, l, "rda.gated", &scratch.rda_gated, rows, cap)
}

/// Census taps of an MLA layer's intermediates (after `mla_mixer`): the
/// projections, the normed q_a, q, and the gated attention output.
fn stage_mla(
    tap: &mut Option<LayerTap<'_>>,
    gpu: &mut Gpu,
    l: usize,
    rows: usize,
    scratch: &RaonPrefillScratch,
) -> Result<(), String> {
    if tap.is_none() {
        return Ok(());
    }
    let cap = scratch.rows;
    for (name, t) in [
        ("mla.qa", &scratch.mla_qa),
        ("mla.kv_a", &scratch.mla_kv_a),
        ("mla.g", &scratch.mla_g),
        ("mla.qa_norm", &scratch.mla_qa_norm),
        ("mla.q", &scratch.mla_q),
        ("mla.attn_gated", &scratch.mla_attn),
    ] {
        fire_stage_scratch(tap, gpu, l, name, t, rows, cap)?;
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
    ctx: &DispatchCtx,
    cfg: &RaonConfig,
    cap: &DflashCapture<'_>,
    k: usize,
    streams: &GpuTensor,
    rows: usize,
) -> Result<(), String> {
    let width = cfg.mhc_streams * cfg.hidden;
    let (dst_stride, dst_col) = capture_geometry(cap.layers.len(), width, cap.row0, k);
    let copy = [program::capture_streams(
        rows, width, streams, cap.dst, dst_stride, dst_col,
    )];
    run(gpu, ctx, &copy)
        .map_err(|e| format!("raon1_mini: DFlash capture of layer {}: {e}", cap.layers[k]))
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

/// Route of a call that leaves `end` rows cached (pure: policy x admission
/// budget, no device state).
pub fn mla_route(work: &RaonWork, s: &RaonState, end: usize) -> MlaRoute {
    work.mla.policy().route(end, s.mirror_budget_rows())
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
                let arch = gpu.arch.clone();
                let ctx = DispatchCtx::new(gpu);
                let tier = work.site_tiers.get(RaonPrecisionSite::MlaKv);
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
                        let expand = program::mla_expand_steps(
                            &arch,
                            cfg,
                            mw,
                            scratch,
                            cache,
                            dmirror,
                            RowRange::Absolute { start: done, count },
                            count,
                            tier,
                        );
                        run(gpu, &ctx, &expand)?;
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
    // Ordered on the decode stream when there is one (no null-stream hop); blocking otherwise.
    gpu.htod_stream_ordered(&scratch.token_ids.buf, 0, &bytes)
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
    let lookup = [program::embed_step(
        cfg,
        w.embed_table()?,
        scratch,
        rows,
        out,
    )];
    let ctx = DispatchCtx::new(gpu);
    run(gpu, &ctx, &lookup).map_err(|e| format!("raon1_mini: embedding lookup: {e}"))
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
    let head = program::head_steps(
        work,
        w.lm_head_weight()?,
        w.final_norm_weight()?,
        cfg.rms_eps,
        count,
        program::rows_route(&gpu.arch, count),
        &view,
        &scratch.branch_in,
        out,
    );
    let ctx = DispatchCtx::new(gpu);
    run(gpu, &ctx, &head)
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
    require_whole_band(w, "token/row forward")?;
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
        w.band.is_last(),
        &mut None,
        None,
        None,
    )?;
    advance(gpu, s, 1)
}

/// Device-side `seq_len += rows` once every stage of the call read the length.
fn advance(gpu: &mut Gpu, s: &RaonState, rows: usize) -> Result<(), String> {
    let ctx = DispatchCtx::new(gpu);
    run(gpu, &ctx, &[program::advance_step(s, rows)])
        .map_err(|e| format!("raon1_mini: device seq_len advance: {e}"))
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
    require_whole_band(w, "verify forward")?;
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
    advance(gpu, s, rows)
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
    mix(w.embed.as_ref().map_or(0, |t| t.buf.as_ptr() as u64));
    mix(w.lm_head.as_ref().map_or(0, |t| t.buf.buf.as_ptr() as u64));
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
    let arch = gpu.arch.clone();
    let route = program::rows_route(&arch, rows);
    let mut ctx = DispatchCtx::new(gpu);
    ctx.batch_size = rows;
    let ctx = ctx;
    // A band past the first receives its streams in `scratch.stream_a` (rows
    // `[0, rows)`) from the previous band instead of embedding.
    if w.band.is_first() {
        let embed = program::embed_steps(
            cfg,
            w.embed_table()?,
            scratch,
            work,
            rows,
            &scratch.branch_out,
            &scratch.stream_a,
        );
        run(gpu, &ctx, &embed)?;
        fire(
            tap,
            gpu,
            TapSite::Embedding,
            &scratch.stream_a,
            rows,
            cfg.mhc_streams * cfg.hidden,
        )?;
    }

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
            let site = [program::site_route(
                work,
                &layer.attn_mhc,
                rows,
                cur,
                &layer.input_norm,
                eps,
                &scratch.branch_in,
            )];
            run(gpu, &ctx, &site)?;
            // The fused stage still writes the selector and control planes, so the census taps
            // them after it (one stage, same planes as the unfused route).
            stage_route(tap, gpu, work, l, rows, "attn")?;
            fire_stage_scratch(
                tap,
                gpu,
                l,
                "attn.agg",
                &scratch.branch_in,
                rows,
                scratch.rows,
            )?;
            match (kind, &layer.mixer) {
                (LayerKind::Rda, RaonMixer::Rda(rw)) => {
                    let ri = cfg
                        .rda_index(l)
                        .ok_or_else(|| format!("raon1_mini: layer {l} has no RDA slot"))?;
                    let tape: Option<RdaTape> = ring.map(|r| r.tape(ri, &work.rda_geom));
                    let mixer =
                        program::rda_steps(rw, ri, s, scratch, work, rows, route, tape.as_ref());
                    run(gpu, &ctx, &mixer).map_err(|e| format!("raon1_mini L{l} rda: {e}"))?;
                    stage_rda(tap, gpu, l, rows, scratch, work, ring, ri)?;
                }
                (LayerKind::Mla, RaonMixer::Mla(mw)) => {
                    let mi = cfg
                        .mla_index(l)
                        .ok_or_else(|| format!("raon1_mini: layer {l} has no MLA slot"))?;
                    let mixer = program::mla_steps(
                        &arch,
                        cfg,
                        mw,
                        mi,
                        s,
                        scratch,
                        work,
                        rows,
                        start,
                        mla_route(work, s, start + rows),
                    )
                    .and_then(|mixer| run(gpu, &ctx, &mixer))
                    .map_err(|e| format!("raon1_mini L{l} mla: {e}"));
                    mixer?;
                    stage_mla(tap, gpu, l, rows, scratch)?;
                }
                _ => {
                    return Err(format!(
                        "raon1_mini: layer {l} weights do not match its configured mixer"
                    ))
                }
            }
            fire_stage_scratch(
                tap,
                gpu,
                l,
                "attn.out",
                &scratch.branch_out,
                rows,
                scratch.rows,
            )?;
            let post = [program::site_postmix(
                work,
                rows,
                cur,
                &scratch.branch_out,
                oth,
            )];
            run(gpu, &ctx, &post)?;
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
            cfg.mhc_streams * cfg.hidden,
        )?;

        // ── FFN site ──
        {
            let (cur, oth) = if cur_is_a {
                (&scratch.stream_a, &scratch.stream_b)
            } else {
                (&scratch.stream_b, &scratch.stream_a)
            };
            let site = [program::site_route(
                work,
                &layer.ffn_mhc,
                rows,
                cur,
                &layer.post_attn_norm,
                eps,
                &scratch.branch_in,
            )];
            run(gpu, &ctx, &site)?;
            stage_route(tap, gpu, work, l, rows, "ffn")?;
            // Developer capture (`HIPFIRE_MOE_ROUTE_DUMP` / `HIPFIRE_MOE_ACT_CAPTURE`):
            // names this layer and its first row's position; a no-op when both are unset.
            hipfire_dispatch::pipeline::moe_capture::set_site(l, start);
            fire_stage_scratch(
                tap,
                gpu,
                l,
                "ffn.agg",
                &scratch.branch_in,
                rows,
                scratch.rows,
            )?;
            program::ffn_step(&ctx, work, layer, scratch, rows, route)
                .and_then(|ffn| run(gpu, &ctx, &[ffn]))
                .map_err(|e| format!("raon1_mini L{l} ffn: {e}"))?;
            fire_stage_scratch(tap, gpu, l, "ffn.out", &scratch.ffn_out, rows, scratch.rows)?;
            let post = [program::site_postmix(
                work,
                rows,
                cur,
                &scratch.ffn_out,
                oth,
            )];
            run(gpu, &ctx, &post)?;
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
            cfg.mhc_streams * cfg.hidden,
        )?;
        if let Some(cap) = capture {
            if let Some(k) = cap.layers.iter().position(|&id| id == l) {
                let streams = if cur_is_a {
                    &scratch.stream_a
                } else {
                    &scratch.stream_b
                };
                capture_streams(gpu, &ctx, cfg, cap, k, streams, rows)?;
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
        let head = program::head_steps(
            work,
            w.lm_head_weight()?,
            w.final_norm_weight()?,
            eps,
            1,
            RowsRoute::Decode,
            &view,
            &scratch.final_norm,
            &scratch.logits,
        );
        run(gpu, &ctx, &head)?;
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
    forward_rows_verify_tapped(
        cfg,
        w,
        s,
        scratch,
        work,
        gpu,
        tokens,
        start_pos,
        want_logits,
        capture,
        ring,
        None,
    )
}

/// [`forward_rows_verify`] with an optional observer (census: the same verify
/// launch sequence, every stage tapped; not graph-capturable).
#[allow(clippy::too_many_arguments)]
pub fn forward_rows_verify_tapped(
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
        capture,
        ring,
    )
}

// ───────────────────────── band forward ─────────────────────────
//
// A band ([`crate::band::RaonBand`]) runs its layers only: the first band
// embeds staged token ids into `scratch.stream_a`, every other band expects the
// previous band's streams already in `scratch.stream_a` rows `[0, rows)` (the
// edge, [`crate::band::edge_view`]); the last band reads the logits out. Each
// band keeps its own `seq_len` / `n_tokens` and advances them once per call.
//
// The host work (context admission / map-ahead, token staging) is split from
// the device-only body so a mesh executor can run the body per rank with no
// host wait in it.

/// `Err` unless `w` covers the whole model: whole-model logits, speculation
/// (DFlash capture, verify, rollback) and session snapshots are not defined
/// under a partial band.
pub(crate) fn require_whole_band(w: &RaonWeights, what: &str) -> Result<(), String> {
    if w.band.is_whole() {
        Ok(())
    } else {
        Err(format!(
            "raon1_mini: {what} is not supported on a partial band ({}): the whole-model \
             forward, DFlash/speculation, rollback and session snapshots need the whole model \
             on one device; drive the band with the band forward",
            w.band.label()
        ))
    }
}

/// What one band call consumes.
#[derive(Clone, Copy, Debug)]
pub enum BandInput<'a> {
    /// The first band: token ids to embed (`1..=scratch rows`).
    Tokens(&'a [u32]),
    /// A later band: `rows` stream rows already in `scratch.stream_a`.
    Streams { rows: usize },
}

impl BandInput<'_> {
    /// Rows of the call.
    pub fn rows(&self) -> usize {
        match self {
            BandInput::Tokens(t) => t.len(),
            BandInput::Streams { rows } => *rows,
        }
    }
}

/// Host phase of a band call: validate the input against the band, admit the
/// context (map-ahead of the band's latent caches and expanded mirror) for
/// `n_tokens + rows`, and stage the token ids on the first band. Returns the
/// rows of the call. Outside capture only.
#[allow(clippy::too_many_arguments)]
pub fn band_admit(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    gpu: &mut Gpu,
    input: BandInput<'_>,
) -> Result<usize, String> {
    let rows = input.rows();
    if rows == 0 || rows > scratch.rows || rows > work.rows {
        return Err(format!(
            "raon1_mini: {rows} rows outside 1..={} (scratch rows {}, work rows {})",
            scratch.rows.min(work.rows),
            scratch.rows,
            work.rows
        ));
    }
    if w.layers.len() != cfg.n_layers || w.layers.len() != w.band.len() {
        return Err(format!(
            "raon1_mini: weights hold {} layers, config view {}, {}",
            w.layers.len(),
            cfg.n_layers,
            w.band.label()
        ));
    }
    match input {
        BandInput::Tokens(_) if !w.band.is_first() => {
            return Err(format!(
                "raon1_mini: token input on {}: only the first band embeds",
                w.band.label()
            ))
        }
        BandInput::Streams { .. } if w.band.is_first() => {
            return Err(format!(
                "raon1_mini: stream input on {}: the first band takes tokens",
                w.band.label()
            ))
        }
        _ => {}
    }
    ensure_context(cfg, w, s, scratch, work, gpu, s.n_tokens() + rows)?;
    if let BandInput::Tokens(tokens) = input {
        stage_tokens(cfg, scratch, gpu, tokens)?;
    }
    Ok(rows)
}

/// Device-only body of a band call over `rows` rows from the slot's current
/// position: (first band) embedding of the staged ids, the band's layers,
/// (last band, `want_logits`) final norm + `lm_head` of the last row into
/// `scratch.logits`, then a device-side `seq_len += rows`. Issues launches over
/// stable device buffers only (no host wait), so it is recordable into a
/// hipGraph at `rows == 1` exactly like [`decode_body`]. The host mirror
/// ([`RaonState::note_host_tokens`]) is the caller's.
#[allow(clippy::too_many_arguments)]
pub fn band_body(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    gpu: &mut Gpu,
    rows: usize,
    want_logits: bool,
) -> Result<(), String> {
    if want_logits && !w.band.is_last() {
        return Err(format!(
            "raon1_mini: logits requested on {}: only the last band reads them out",
            w.band.label()
        ));
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
        want_logits,
        &mut None,
        None,
        None,
    )?;
    advance(gpu, s, rows)
}

/// One band call: [`band_admit`], [`band_body`], then the host mirror advance.
/// The slot advances by the call's rows exactly once.
#[allow(clippy::too_many_arguments)]
pub fn forward_band(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    input: BandInput<'_>,
    want_logits: bool,
) -> Result<(), String> {
    if want_logits && !w.band.is_last() {
        return Err(format!(
            "raon1_mini: logits requested on {}: only the last band reads them out",
            w.band.label()
        ));
    }
    let rows = band_admit(cfg, w, s, scratch, work, gpu, input)?;
    band_body(cfg, w, s, scratch, work, gpu, rows, want_logits)?;
    s.note_host_tokens(rows);
    Ok(())
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
