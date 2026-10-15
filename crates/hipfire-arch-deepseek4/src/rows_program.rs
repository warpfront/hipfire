// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative DeepSeek-V4 decoder layer over a window of rows: batched
//! prefill, speculative verify and the MTP layer block.
//!
//! A window is `rows` consecutive positions `start_pos..` held in a
//! [`PrefillBatchScratch`]. Each layer is one step program of
//! `hipfire_dispatch::pipeline::mla` steps, one `Step` variant per operation:
//!
//! ```text
//! attention: HcPre → QLoraPrepare → PackedProjections (gfx1151 B3 pack) → QLoraProject
//!            → KvJoint → ApplyTailRope → SwaStage
//!            ratio 0:  SwaAttention
//!            ratio >0: CompressorProject → KvCompress (batched window, or one
//!                      decode KvCompress per row) → SparseIndex → SparseAttention
//!            → ApplyTailRope⁻¹ → OLora → SwaRingWrite → HcPost
//! FFN:       HcPre → ExpertFfn → HcPost  [→ StreamMeanCapture on DSpark layers]
//! ```
//!
//! This module binds weights, per-layer caches and the window scratch and
//! resolves the route knobs; kernels and their order live in dispatch.

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::moe::{MoeBiasAwarePrefillParams, MoePrefillRouting};
use hipfire_dispatch::pipeline::mla::compressor::CompressorRope;
use hipfire_dispatch::pipeline::mla::attention::{
    CompressedCache, DirectTopk, IdentityGatherOp, IdentityGatherRoute, OLoraOp, OLoraRoute,
    SparseAttentionOp, SparseAttentionRoute, SwaAttentionOp, SwaAttentionRoute, SwaRingWriteOp,
    SwaRingWriteRoute,
};
use hipfire_dispatch::pipeline::mla::compressor::{
    CompressorProjectOp, CompressorProjectRoute, CompressorProjection,
};
use hipfire_dispatch::pipeline::mla::rows::{
    AttentionInputPackB3, BlockStageOp, EmbedStreamsOp, RingKvOp, StreamMeanCaptureOp, SwaStageOp,
};
use hipfire_dispatch::pipeline::mla::RowsProjection;
use hipfire_dispatch::pipeline::mla::{attention, compressor, expert_ffn, hc, project, query_kv};
use hipfire_dispatch::pipeline::{execute_steps, Step};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::backend::Mq2rBackend;
use crate::config_cache;
use crate::deepseek4::{CompressorCachePlacement, DeepseekV4LayerWeights};
use crate::forward::{
    compressor_op, dense_activation_dump_enabled, dispatch_err, dspark_requires_typed_device_copy,
    dump_buf, dump_dense_activation_if_enabled, dump_dense_activations_if_enabled, e8_soa_kernel,
    ensure_compressor, ensure_final_norm_compute, final_norm_compute_op, layer_rope_params,
    PrefillBatchScratch, POS_SLOTS_PER_LAYER,
};
use crate::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};

/// One window of rows.
#[derive(Clone, Copy)]
pub(crate) struct RowsWindow<'a> {
    pub pbs: &'a PrefillBatchScratch,
    pub start_pos: u32,
    pub rows: usize,
    /// Every position-dependent count comes from the device arrays uploaded
    /// at the window boundary (a captured verify graph replays at any depth).
    pub capture_safe: bool,
    /// The window's tokens (hash-routed layers read them; empty when no
    /// layer of the program hash-routes).
    pub tokens: &'a [u32],
}

impl<'a> RowsWindow<'a> {
    /// `rows` positions from `start_pos`; `uploaded_counts` reads every
    /// position-dependent count from the window's device arrays (the
    /// capture-safe verify body).
    pub(crate) fn new(
        pbs: &'a PrefillBatchScratch,
        start_pos: u32,
        rows: usize,
        tokens: &'a [u32],
        uploaded_counts: bool,
    ) -> Self {
        Self {
            pbs,
            start_pos,
            rows,
            capture_safe: uploaded_counts,
            tokens,
        }
    }
}

/// Each row's stripes of the window's position and attention-state arrays:
/// the decode slots of the per-row compressor fallback.
pub(crate) struct RowSlots {
    slots: Vec<(GpuTensor, GpuTensor)>,
}

impl RowSlots {
    pub(crate) fn new(cfg: &DeepseekV4Config, pbs: &PrefillBatchScratch, rows: usize) -> Self {
        let pos_stripe = (cfg.num_hidden_layers + 1) * POS_SLOTS_PER_LAYER;
        let attn_stripe = 10;
        let slots = (0..rows)
            .map(|b| {
                (
                    pbs.pos_array_device_batch
                        .sub_offset(b * pos_stripe, pos_stripe),
                    pbs.attn_state_buf_batch
                        .sub_offset(b * attn_stripe, attn_stripe),
                )
            })
            .collect();
        Self { slots }
    }
}

/// What a step produces, for the diagnostics that observe a program.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Embed,
    HcPreAttn,
    QLora,
    KvJoint,
    TailRope,
    SwaStaging,
    SwaAttention,
    InverseRope,
    OLora,
    AttentionEnd,
    HcPostAttn,
    HcPreFfn,
    Ffn,
    HcPostFfn,
    Other,
}

pub(crate) struct RowsProgram<'a> {
    steps: Vec<Step<'a>>,
    stages: Vec<Stage>,
}

impl<'a> RowsProgram<'a> {
    pub(crate) fn new() -> Self {
        Self {
            steps: Vec::with_capacity(24),
            stages: Vec::with_capacity(24),
        }
    }

    fn push(&mut self, stage: Stage, step: Step<'a>) {
        self.steps.push(step);
        self.stages.push(stage);
    }
}

/// Route knobs of the batched projections for this model on `gpu`.
pub(crate) fn rows_projection(gpu: &Gpu, backend: Mq2rBackend) -> RowsProjection {
    let flag = |name: &str| hipfire_config::developer_var(name).as_deref() != Ok("0");
    RowsProjection {
        e8_soa: e8_soa_kernel(&gpu.arch, backend),
        e8_b2: config_cache::e8_prefill_b2_on(&gpu.arch, backend.is_gfx1151()),
        e8_b4: config_cache::e8_prefill_b4_on(&gpu.arch, backend.is_gfx1151()),
        e8_batched_gemv_max: config_cache::e8_batched_gemv_max_batch(),
        gfx1201_e8: backend.is_gfx1201(),
        gfx1201_wide: flag("HIPFIRE_DEEPSEEK4_GFX1201_E8_WIDE"),
        e8_pair_b3: flag("HIPFIRE_DEEPSEEK4_E8_BATCHED_PAIR_B3"),
        hfq4_wmma: flag("HIPFIRE_DEEPSEEK4_HFQ4_WMMA"),
        f32_trace: hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_F32_TRACE").is_ok(),
    }
}

fn req<'a>(
    t: &'a Option<GpuTensor>,
    what: &str,
    layer_idx: usize,
) -> Result<&'a GpuTensor, String> {
    t.as_ref()
        .ok_or_else(|| format!("layer {layer_idx} {what} missing"))
}

fn tail_rope(cfg: &DeepseekV4Config, layer: &DeepseekV4LayerWeights) -> CompressorRope {
    let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
        layer_rope_params(cfg, layer.compress_ratio);
    CompressorRope {
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        corr_low,
        corr_high,
        rope_dim: cfg.qk_rope_head_dim,
    }
}

/// Whether the compressor projections run as F16 WMMA on the F16 weight
/// copies (all of the layer's flavors have them).
fn compressor_f16_wmma(layer: &DeepseekV4LayerWeights) -> bool {
    let on = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_COMP_F16_WMMA")
        .map(|s| s != "0")
        .unwrap_or(true);
    let have_idx_f16 = layer.compress_ratio != 4
        || (layer.indexer_compressor_wkv_f16.is_some()
            && layer.indexer_compressor_wgate_f16.is_some());
    on && layer.compressor_wkv_f16.is_some() && layer.compressor_wgate_f16.is_some() && have_idx_f16
}

/// The windowed compressor commit can run batched: a chunk starting on a
/// window boundary, or one that ends before the next event (the common
/// DSpark verify case on ratio-128 layers).
fn compressor_window_batched(start_pos: u32, rows: usize, ratio: usize) -> bool {
    let slot_base = start_pos as usize % ratio;
    slot_base == 0 || rows < ratio - slot_base
}

/// Allocates (once) the per-layer caches a window of `layer_idx` binds.
pub(crate) fn ensure_rows_layer(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let (n_kv, head_dim, win) = (cfg.num_key_value_heads, cfg.head_dim, cfg.sliding_window);
    let ratio = weights.resolve_layer(layer_idx).compress_ratio;
    let attn = &mut state._attention[layer_idx];
    let zeros = |gpu: &mut Gpu, slot: &mut Option<GpuTensor>, shape: &[usize], what: &str| {
        if slot.is_none() {
            *slot = Some(
                gpu.zeros(shape, DType::F32)
                    .map_err(|e| format!("alloc {what} l{layer_idx}: {e:?}"))?,
            );
        }
        Ok::<(), String>(())
    };
    zeros(gpu, &mut attn.swa_k, &[n_kv, head_dim, win], "swa_k")?;
    zeros(gpu, &mut attn.swa_v, &[n_kv, head_dim, win], "swa_v")?;
    if ratio > 0 {
        zeros(
            gpu,
            &mut attn.gathered_k,
            &[n_kv, head_dim, cfg.index_topk],
            "gathered_k",
        )?;
        ensure_compressor(cfg, weights, state, gpu, layer_idx, false)?;
        ensure_compressor(cfg, weights, state, gpu, layer_idx, true)?;
    }
    Ok(())
}

/// Allocates (once) the DSpark capture buffers a window of `max_batch` rows
/// writes.
pub(crate) fn ensure_stream_mean_capture(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    max_batch: usize,
) -> Result<(), String> {
    if state.dspark_cap_ones.is_none() {
        let quarter = vec![0.25f32; max_batch * cfg.hc_mult];
        state.dspark_cap_ones = Some(
            gpu.upload_f32(&quarter, &[max_batch, cfg.hc_mult])
                .map_err(|e| format!("dspark_capture alloc cap_ones: {e:?}"))?,
        );
    }
    if state.dspark_caps.is_none() {
        let n_targets = state.dspark_target_layers.len();
        state.dspark_caps = Some(
            gpu.zeros(&[max_batch, n_targets, cfg.hidden_size], DType::F32)
                .map_err(|e| format!("dspark_capture alloc caps: {e:?}"))?,
        );
    }
    Ok(())
}

/// The window's embedding lookup and HC stream broadcast.
pub(crate) fn embed_rows_program<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    window: RowsWindow<'a>,
) -> Result<RowsProgram<'a>, String> {
    let pbs = window.pbs;
    let token_embd = weights
        .token_embd
        .as_ref()
        .ok_or_else(|| "forward_prefill_batch_chunk: token_embd not uploaded".to_string())?;
    let mut program = RowsProgram::new();
    program.push(
        Stage::Embed,
        Step::EmbedStreams(EmbedStreamsOp {
            token_embd,
            tokens: &pbs.tokens,
            embed: &pbs.embed_batch,
            streams: &pbs.streams_batch,
            hidden: cfg.hidden_size,
            hc_mult: cfg.hc_mult,
            rows: window.rows,
        }),
    );
    Ok(program)
}

/// Both sublayers of `layer_idx` over the window, mixing back into
/// `pbs.streams_batch`; on a DSpark target layer the stream mean is captured
/// into `capture_slot`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rows_layer_program<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    window: RowsWindow<'a>,
    slots: &'a RowSlots,
    capture_slot: Option<usize>,
) -> Result<RowsProgram<'a>, String> {
    let mut program = RowsProgram::new();
    let projection = rows_projection(gpu, weights.mq2r_backend);
    let (pbs, rows) = (window.pbs, window.rows);
    let layer = weights.resolve_layer(layer_idx);
    push_attention(
        &mut program,
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        window,
        slots,
        projection,
    )?;
    push_mix(&mut program, cfg, pbs, gpu, false, rows);
    push_ffn(
        &mut program,
        cfg,
        layer,
        pbs,
        gpu,
        layer_idx,
        window,
        layer_idx < cfg.num_hash_layers,
        projection,
        weights.mq2r_backend,
    )?;
    push_mix(&mut program, cfg, pbs, gpu, true, rows);
    if let Some(slot) = capture_slot {
        program.push(
            Stage::Other,
            Step::StreamMeanCapture(StreamMeanCaptureOp {
                quarter: state.dspark_cap_ones.as_ref().unwrap(),
                streams: &pbs.streams_batch,
                tmp: &pbs.tmp_batch,
                caps: state.dspark_caps.as_ref().unwrap(),
                hidden: cfg.hidden_size,
                n_slots: state.dspark_target_layers.len(),
                slot,
                typed_copy: dspark_requires_typed_device_copy(gpu),
                rows,
                layer: layer_idx,
            }),
        );
    }
    Ok(program)
}

/// The HC mix of the attention (`ffn == false`) or FFN output into
/// `pbs.streams_batch`.
pub(crate) fn push_mix<'a>(
    program: &mut RowsProgram<'a>,
    cfg: &DeepseekV4Config,
    pbs: &'a PrefillBatchScratch,
    gpu: &Gpu,
    ffn: bool,
    rows: usize,
) {
    let (stage, branch) = if ffn {
        (Stage::HcPostFfn, &pbs.ffn_out_batch)
    } else {
        (Stage::HcPostAttn, &pbs.attn_out_batch)
    };
    program.push(
        stage,
        hc_post(
            cfg,
            pbs,
            branch,
            rows,
            dspark_requires_typed_device_copy(gpu),
        ),
    );
}

/// FFN HC pre and the shared + routed expert FFN into `pbs.ffn_out_batch`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_ffn<'a>(
    program: &mut RowsProgram<'a>,
    cfg: &DeepseekV4Config,
    layer: &'a DeepseekV4LayerWeights,
    pbs: &'a PrefillBatchScratch,
    gpu: &Gpu,
    layer_idx: usize,
    window: RowsWindow<'a>,
    hash_routing: bool,
    projection: RowsProjection,
    backend: Mq2rBackend,
) -> Result<(), String> {
    program.push(
        Stage::HcPreFfn,
        hc_pre(cfg, layer, pbs, gpu, layer_idx, false, window.rows)?,
    );
    program.push(
        Stage::Ffn,
        expert_ffn_rows(
            cfg,
            layer,
            pbs,
            layer_idx,
            window,
            hash_routing,
            projection,
            backend,
        )?,
    );
    Ok(())
}

fn hc_pre<'a>(
    cfg: &DeepseekV4Config,
    layer: &'a DeepseekV4LayerWeights,
    pbs: &'a PrefillBatchScratch,
    gpu: &Gpu,
    layer_idx: usize,
    is_attn: bool,
    rows: usize,
) -> Result<Step<'a>, String> {
    let (hc_fn, hc_base, hc_scale) = if is_attn {
        (&layer.hc_attn_fn, &layer.hc_attn_base, &layer.hc_attn_scale)
    } else {
        (&layer.hc_ffn_fn, &layer.hc_ffn_base, &layer.hc_ffn_scale)
    };
    Ok(Step::HcPre(hc::HcPreOp {
        rows,
        streams: &pbs.streams_batch,
        hc_fn: req(hc_fn, "hc fn", layer_idx)?,
        hc_base: req(hc_base, "hc base", layer_idx)?,
        hc_scale: req(hc_scale, "hc scale", layer_idx)?,
        hc_c: &pbs.hc_c_batch,
        hc_x_in: &pbs.hc_x_in_batch,
        layer: layer_idx,
        hidden: cfg.hidden_size,
        x_dim: cfg.hidden_size * cfg.hc_mult,
        hc_eps: cfg.hc_eps,
        post_scale: hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_POST_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.5),
        sinkhorn_iters: cfg.hc_sinkhorn_iters as i32,
        route: hc::HcPreRoute::Batched {
            pre: &pbs.hc_pre_batch,
            post: &pbs.hc_post_batch,
            comb: &pbs.hc_comb_batch,
            // gfx1201 1,024-row chunk only; every other shape keeps the
            // 24-workgroup route.
            fused24: gpu.arch_caps.is_gfx1201()
                && rows == 1024
                && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_GFX1201_HC_FUSED24")
                    .as_deref()
                    != Ok("0"),
        },
    }))
}

fn hc_post<'a>(
    cfg: &DeepseekV4Config,
    pbs: &'a PrefillBatchScratch,
    branch: &'a GpuTensor,
    rows: usize,
    typed_copy: bool,
) -> Step<'a> {
    Step::HcPost(hc::HcPostOp {
        rows,
        branch: hc::HcMixBranch::Transform(branch),
        streams_in: &pbs.streams_batch,
        streams_out: &pbs.streams_out_batch,
        writeback: Some(&pbs.streams_batch),
        hidden: cfg.hidden_size,
        copy_bytes: rows * cfg.hc_mult * cfg.hidden_size * 4,
        route: hc::HcPostRoute::Batched {
            comb: &pbs.hc_comb_batch,
            post: &pbs.hc_post_batch,
            typed_copy,
        },
    })
}

/// The gfx1151 B3 pack of every E8 projection of the attention input, when
/// admitted (three rows, trunk layer, every packed weight E8 SoA, compressor
/// not on the F16 route).
fn attention_input_pack<'a>(
    cfg: &DeepseekV4Config,
    layer: &'a DeepseekV4LayerWeights,
    pbs: &'a PrefillBatchScratch,
    gpu: &Gpu,
    layer_idx: usize,
    rows: usize,
) -> Option<AttentionInputPackB3<'a>> {
    let enabled = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_E8_ATTN_PACK_B3")
        .ok()
        .as_deref()
        != Some("0");
    if !enabled
        || gpu.arch != "gfx1151"
        || layer_idx >= cfg.num_hidden_layers
        || rows != 3
        || !cfg.hidden_size.is_multiple_of(256)
    {
        return None;
    }
    let wq_a = layer.wq_a.as_ref()?;
    let mut weights = [wq_a; 7];
    let mut outputs = [&pbs.q_lat_batch; 7];
    let mut out_rows = [0usize; 7];
    out_rows[0] = cfg.q_lora_rank;
    weights[1] = layer.wkv.as_ref()?;
    outputs[1] = &pbs.kv_batch;
    out_rows[1] = cfg.num_key_value_heads * cfg.head_dim;
    let ratio = layer.compress_ratio as usize;
    if ratio > 0 {
        if compressor_f16_wmma(layer) {
            return None;
        }
        let main_rows = if ratio == 4 {
            2 * cfg.head_dim
        } else {
            cfg.head_dim
        };
        weights[2] = layer.compressor_wkv.as_ref()?;
        outputs[2] = &pbs.comp_main_kv_batch;
        out_rows[2] = main_rows;
        weights[3] = layer.compressor_wgate.as_ref()?;
        outputs[3] = &pbs.comp_main_score_batch;
        out_rows[3] = main_rows;
        if ratio == 4 {
            weights[4] = layer.indexer_weights_proj.as_ref()?;
            outputs[4] = &pbs.idx_w_batch;
            out_rows[4] = cfg.index_n_heads;
            weights[5] = layer.indexer_compressor_wkv.as_ref()?;
            outputs[5] = &pbs.comp_idx_kv_batch;
            out_rows[5] = 2 * cfg.index_head_dim;
            weights[6] = layer.indexer_compressor_wgate.as_ref()?;
            outputs[6] = &pbs.comp_idx_score_batch;
            out_rows[6] = 2 * cfg.index_head_dim;
        }
    }
    if weights
        .iter()
        .zip(out_rows)
        .any(|(w, r)| r != 0 && w.dtype != DType::MFP4G32E8SOA)
    {
        return None;
    }
    Some(AttentionInputPackB3 {
        weights,
        outputs,
        rows: out_rows,
    })
}

/// Attention input of the window: HC pre, Q-LoRA (+ B3 pack), joint KV and
/// tail RoPE. Returns whether the B3 pack produced the compressor / indexer
/// projections too.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_attention_input<'a>(
    program: &mut RowsProgram<'a>,
    cfg: &DeepseekV4Config,
    layer: &'a DeepseekV4LayerWeights,
    pbs: &'a PrefillBatchScratch,
    gpu: &Gpu,
    layer_idx: usize,
    rows: usize,
    projection: RowsProjection,
) -> Result<bool, String> {
    let (hidden, n_heads, head_dim) = (cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim);
    let n_kv = cfg.num_key_value_heads;
    let eps = cfg.rms_norm_eps;
    program.push(
        Stage::HcPreAttn,
        hc_pre(cfg, layer, pbs, gpu, layer_idx, true, rows)?,
    );
    let pack = attention_input_pack(cfg, layer, pbs, gpu, layer_idx, rows);
    let packed = pack.is_some();
    let attn_norm = req(&layer.attn_norm, "attn_norm", layer_idx)?;
    let q_norm = req(&layer.q_norm, "q_norm", layer_idx)?;
    let wq_a = req(&layer.wq_a, "wq_a", layer_idx)?;
    let wq_b = req(&layer.wq_b, "wq_b", layer_idx)?;
    program.push(
        Stage::QLora,
        Step::QLoraPrepare(query_kv::QLoraPrepareOp {
            hc_x_in: &pbs.hc_x_in_batch,
            attn_norm,
            tmp: &pbs.tmp_batch,
            tmp_plain: &pbs.tmp_plain_batch,
            hidden,
            eps,
            rotate: crate::forward::weight_needs_fwht(wq_a),
            rows,
            layer: layer_idx,
            route: query_kv::QLoraPrepareRoute::Batched,
        }),
    );
    if let Some(pack) = pack {
        program.push(
            Stage::QLora,
            Step::PackedProjections(query_kv::PackedProjectionsOp {
                x: &pbs.tmp_batch,
                k: hidden,
                rows,
                layer: layer_idx,
                route: query_kv::PackedProjectionsRoute::Batched { pack },
            }),
        );
    }
    program.push(
        Stage::QLora,
        Step::QLoraProject(query_kv::QLoraProjectOp {
            wq_a,
            wq_b,
            q_norm,
            tmp: &pbs.tmp_batch,
            tmp_plain: &pbs.tmp_plain_batch,
            q_lat: &pbs.q_lat_batch,
            q_lat_rot: &pbs.q_lat_rot_batch,
            q: &pbs.q_batch,
            q_head_ones: &pbs.q_head_ones,
            hidden,
            q_lora_rank: cfg.q_lora_rank,
            heads: n_heads,
            head_dim,
            eps,
            wq_a_preprojected: packed,
            wq_b_needs_fwht: crate::forward::weight_needs_fwht(wq_b),
            rows,
            layer: layer_idx,
            route: query_kv::QLoraProjectRoute::Batched {
                x_f16: &pbs.wmma_x_scratch_f16,
                projection,
            },
        }),
    );
    program.push(
        Stage::KvJoint,
        Step::KvJoint(query_kv::KvJointOp {
            wkv: req(&layer.wkv, "wkv", layer_idx)?,
            kv_norm: req(&layer.kv_norm, "kv_norm", layer_idx)?,
            tmp: &pbs.tmp_batch,
            tmp_plain: &pbs.tmp_plain_batch,
            kv: &pbs.kv_batch,
            kv_dim: n_kv * head_dim,
            hidden,
            eps,
            preprojected: packed,
            rows,
            layer: layer_idx,
            route: query_kv::KvJointRoute::Batched {
                x_f16: &pbs.wmma_x_scratch_f16,
                projection,
            },
        }),
    );
    program.push(
        Stage::TailRope,
        Step::ApplyTailRope(query_kv::ApplyTailRopeOp {
            q: &pbs.q_batch,
            kv: &pbs.kv_batch,
            pos: &pbs.positions,
            heads: n_heads,
            kv_heads: n_kv,
            head_dim,
            rope: tail_rope(cfg, layer),
            inverse: false,
            rows,
            layer: layer_idx,
            route: query_kv::ApplyTailRopeRoute::Batched,
        }),
    );
    Ok(packed)
}

/// Inverse tail RoPE of the attention output, then the grouped O-LoRA into
/// `pbs.attn_out_batch`; `rotate` forces both FWHT rotations (otherwise
/// each follows its weight's format).
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_o_lora<'a>(
    program: &mut RowsProgram<'a>,
    cfg: &DeepseekV4Config,
    layer: &'a DeepseekV4LayerWeights,
    pbs: &'a PrefillBatchScratch,
    layer_idx: usize,
    rows: usize,
    rotate: bool,
    q8_multirow: i32,
    projection: RowsProjection,
) -> Result<(), String> {
    let (n_heads, head_dim) = (cfg.num_attention_heads, cfg.head_dim);
    let wo_a = req(&layer.wo_a, "wo_a", layer_idx)?;
    let wo_b = req(&layer.wo_b, "wo_b", layer_idx)?;
    program.push(
        Stage::InverseRope,
        Step::ApplyTailRope(query_kv::ApplyTailRopeOp {
            q: &pbs.attn_out_raw_batch,
            kv: &pbs.attn_out_raw_batch,
            pos: &pbs.positions,
            heads: n_heads,
            kv_heads: 0,
            head_dim,
            rope: tail_rope(cfg, layer),
            inverse: true,
            rows,
            layer: layer_idx,
            route: query_kv::ApplyTailRopeRoute::Batched,
        }),
    );
    let rotated = |w: &GpuTensor| rotate || crate::forward::weight_needs_fwht(w);
    program.push(
        Stage::OLora,
        Step::OLora(OLoraOp {
            wo_a,
            wo_b,
            raw: &pbs.attn_out_raw_batch,
            raw_rot: &pbs.attn_out_raw_rot_batch,
            wo_a_out: &pbs.wo_a_out_batch,
            wo_a_out_rot: &pbs.wo_a_out_rot_batch,
            out: &pbs.attn_out_batch,
            n_groups: cfg.o_groups,
            o_lora_rank: cfg.o_lora_rank,
            per_group_in: (n_heads / cfg.o_groups) * head_dim,
            hidden: cfg.hidden_size,
            rotate_raw: rotated(wo_a),
            rotate_mid: rotated(wo_b),
            rows,
            layer: layer_idx,
            route: OLoraRoute::Batched {
                x_f16: &pbs.wmma_x_scratch_f16,
                q8_multirow,
                projection,
            },
        }),
    );
    Ok(())
}

/// The attention sublayer of a trunk (or MTP) layer over the window, up to
/// `pbs.attn_out_batch` and the SWA ring advance; the HC mix is separate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_attention<'a>(
    program: &mut RowsProgram<'a>,
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    window: RowsWindow<'a>,
    slots: &'a RowSlots,
    projection: RowsProjection,
) -> Result<(), String> {
    let RowsWindow {
        pbs,
        start_pos,
        rows,
        capture_safe,
        ..
    } = window;
    let layer = weights.resolve_layer(layer_idx);
    let ratio = layer.compress_ratio as usize;
    let (n_heads, head_dim) = (cfg.num_attention_heads, cfg.head_dim);
    let (n_kv, win) = (cfg.num_key_value_heads, cfg.sliding_window);
    let packed = push_attention_input(program, cfg, layer, pbs, gpu, layer_idx, rows, projection)?;
    let attn = &state._attention[layer_idx];
    let swa_k = attn.swa_k.as_ref().unwrap();
    let swa_v = attn.swa_v.as_ref().unwrap();
    program.push(
        Stage::SwaStaging,
        Step::SwaStage(SwaStageOp {
            swa_k,
            kv: &pbs.kv_batch,
            staged: &pbs.swa_staged_batch,
            positions: &pbs.positions,
            start_pos,
            win,
            head_dim,
            capture_safe,
            rows,
            layer: layer_idx,
        }),
    );
    let sink = req(&layer.attn_sink, "attn_sink", layer_idx)?;
    if ratio == 0 {
        program.push(
            Stage::SwaAttention,
            Step::SwaAttention(SwaAttentionOp {
                q: &pbs.q_batch,
                sink,
                out: &pbs.attn_out_raw_batch,
                n_heads,
                head_dim,
                n_groups: cfg.o_groups,
                win,
                rows,
                layer: layer_idx,
                route: SwaAttentionRoute::Batched {
                    staged: &pbs.swa_staged_batch,
                    n_valid: &pbs.n_valid_swa_arr,
                    start_pos,
                    per_position: hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ATTN_PER_POS")
                        .as_deref()
                        == Ok("1"),
                },
            }),
        );
    } else {
        compressed_attention_rows(
            program, cfg, layer, state, gpu, layer_idx, window, slots, packed, projection, sink,
            weights,
        )?;
    }
    // The compressed route rotates both O-LoRA inputs whatever the format.
    let q8_multirow = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_WO_MULTIROW")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&r| r == 2 || r == 4)
        .unwrap_or(0);
    push_o_lora(
        program,
        cfg,
        layer,
        pbs,
        layer_idx,
        rows,
        ratio > 0,
        q8_multirow,
        projection,
    )?;
    program.push(
        Stage::AttentionEnd,
        Step::SwaRingWrite(SwaRingWriteOp {
            kv: &pbs.kv_batch,
            swa_k,
            swa_v,
            n_kv,
            head_dim,
            win,
            rows,
            layer: layer_idx,
            route: SwaRingWriteRoute::Batched {
                positions: &pbs.positions,
                start_pos,
                capture_safe,
            },
        }),
    );
    Ok(())
}

/// Compressor, indexer and joint sparse attention of a compressed layer.
#[allow(clippy::too_many_arguments)]
fn compressed_attention_rows<'a>(
    program: &mut RowsProgram<'a>,
    cfg: &DeepseekV4Config,
    layer: &'a DeepseekV4LayerWeights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    window: RowsWindow<'a>,
    slots: &'a RowSlots,
    packed: bool,
    projection: RowsProjection,
    sink: &'a GpuTensor,
    weights: &'a DeepseekV4Weights,
) -> Result<(), String> {
    let RowsWindow {
        pbs,
        start_pos,
        rows,
        capture_safe,
        ..
    } = window;
    let ratio = layer.compress_ratio as usize;
    let (hidden, n_heads, head_dim) = (cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim);
    let (win, topk) = (cfg.sliding_window, cfg.index_topk);
    let l_state = &state._indexer[layer_idx];
    // The direct-attention ABI takes `n_compressed` as a scalar kernarg, which
    // a replayed graph would bake in: the capture route gathers instead.
    let direct = !capture_safe
        && ratio == 4
        && state.compressor_cache_dtype == DType::F32
        && matches!(
            state.compressor_cache_placement,
            CompressorCachePlacement::Replicated
        )
        && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT")
            .map(|s| s != "0")
            .unwrap_or(gpu.arch == "gfx1151" && rows >= 64);

    // Compressor projections of the window (the B3 pack may have run them).
    if !packed {
        let f16 = compressor_f16_wmma(layer);
        let (wkv, wgate, iwkv, iwgate) = if f16 {
            (
                &layer.compressor_wkv_f16,
                &layer.compressor_wgate_f16,
                &layer.indexer_compressor_wkv_f16,
                &layer.indexer_compressor_wgate_f16,
            )
        } else {
            (
                &layer.compressor_wkv,
                &layer.compressor_wgate,
                &layer.indexer_compressor_wkv,
                &layer.indexer_compressor_wgate,
            )
        };
        let main_proj = if ratio == 4 { 2 * head_dim } else { head_dim };
        program.push(
            Stage::Other,
            Step::CompressorProject(CompressorProjectOp {
                main: CompressorProjection {
                    wkv: req(wkv, "comp_wkv", layer_idx)?,
                    wgate: req(wgate, "comp_wgate", layer_idx)?,
                    kv: &pbs.comp_main_kv_batch,
                    score: &pbs.comp_main_score_batch,
                    proj_dim: main_proj,
                },
                indexer: if ratio == 4 {
                    Some(CompressorProjection {
                        wkv: req(iwkv, "idx_comp_wkv", layer_idx)?,
                        wgate: req(iwgate, "idx_comp_wgate", layer_idx)?,
                        kv: &pbs.comp_idx_kv_batch,
                        score: &pbs.comp_idx_score_batch,
                        proj_dim: 2 * cfg.index_head_dim,
                    })
                } else {
                    None
                },
                tmp: &pbs.tmp_batch,
                tmp_plain: &pbs.tmp_plain_batch,
                hidden,
                rows,
                layer: layer_idx,
                route: CompressorProjectRoute::Batched {
                    f16: f16.then_some((&pbs.tmp_batch_f16, &pbs.tmp_plain_batch_f16)),
                    x_f16: &pbs.wmma_x_scratch_f16,
                    projection,
                },
            }),
        );
    }

    // Compressor commits: the whole window batched, or one decode step per
    // row (capture route, or an unaligned window that closes an event).
    let flavors: &[bool] = if ratio == 4 { &[false, true] } else { &[false] };
    if !capture_safe && compressor_window_batched(start_pos, rows, ratio) {
        for &is_indexer in flavors {
            program.push(
                Stage::Other,
                Step::KvCompress(kv_compress_rows(
                    cfg, layer, state, pbs, layer_idx, window, is_indexer,
                )?),
            );
        }
    } else {
        for (b, (pos_slots, attn_slots)) in slots.slots.iter().take(rows).enumerate() {
            for &is_indexer in flavors {
                let (kv, score) = if is_indexer {
                    (&pbs.comp_idx_kv_batch, &pbs.comp_idx_score_batch)
                } else {
                    (&pbs.comp_main_kv_batch, &pbs.comp_main_score_batch)
                };
                let op = compressor_op(
                    cfg,
                    weights,
                    state,
                    gpu,
                    layer_idx,
                    start_pos + b as u32,
                    is_indexer,
                    Some((kv, score, b)),
                    capture_safe,
                    Some((pos_slots, attn_slots)),
                )?
                .ok_or_else(|| format!("compressor does not run on layer {layer_idx}"))?;
                program.steps.push(Step::KvCompress(op));
                program.stages.push(Stage::Other);
            }
        }
    }

    // Indexer + gather (ratio 4) or in-order gather (ratio 128).
    let capacity = pbs.idx_score_capacity;
    let main_cache_t = req(&l_state.main_kv_cache, "main_kv_cache", layer_idx)?;
    let main_cache = compressed_rows(state, main_cache_t, &l_state.main_kv_cache_shards);
    let main_cache_f16 = main_cache_t.dtype == DType::F16;
    let mut n_active = vec![0i32; rows];
    // Whether the window gathers compressed rows into the staging buffer.
    let gathers;
    let (index_step, n_compressed_direct) = if ratio == 4 {
        let n_per_row: Vec<i32> = (0..rows)
            .map(|b| ((start_pos as usize + b + 1) / ratio).min(capacity) as i32)
            .collect();
        let n_max = *n_per_row.iter().max().unwrap_or(&0) as usize;
        if n_max > 0 {
            for b in 0..rows {
                n_active[b] = topk.min(n_per_row[b] as usize) as i32;
            }
        }
        let cache_t = req(&l_state.indexer_kv_cache, "indexer_kv_cache", layer_idx)?;
        let cache = compressed_rows(state, cache_t, &l_state.indexer_kv_cache_shards);
        gathers = n_max > 0 && !direct;
        if let (CompressedCache::Sharded { world, .. }, true) = (cache, n_max > 0) {
            if cache_t.dtype == DType::F16 {
                return Err(format!(
                    "F16 compressor cache does not support block-cyclic batched score l{layer_idx}"
                ));
            }
            if l_state.cache_shard_count != world {
                return Err(format!(
                    "batched indexer shard table missing l{layer_idx}: have {}, want {world}",
                    l_state.cache_shard_count
                ));
            }
        }
        let (h_idx, d_idx) = (cfg.index_n_heads, cfg.index_head_dim);
        let op = attention::SparseIndexOp {
            rows,
            layer: layer_idx,
            wq_b: req(&layer.indexer_wq_b, "idx wq_b", layer_idx)?,
            weights_proj: req(&layer.indexer_weights_proj, "idx weights_proj", layer_idx)?,
            q_lat: &pbs.q_lat_batch,
            q_lat_rot: &pbs.q_lat_rot_batch,
            tmp: &pbs.tmp_batch,
            tmp_plain: &pbs.tmp_plain_batch,
            q_idx: &pbs.idx_q_batch,
            idx_weights: &pbs.idx_w_batch,
            scores: &pbs.idx_scores_batch,
            topk_indices: &pbs.idx_topk_indices_batch,
            pos: &pbs.positions,
            weights_preprojected: packed,
            heads: h_idx,
            head_dim: d_idx,
            top_k_max: topk,
            q_lora_rank: cfg.q_lora_rank,
            hidden,
            rope_dim: cfg.qk_rope_head_dim,
            rope_theta: cfg.compress_rope_theta,
            route: attention::SparseIndexRoute::Batched {
                cache,
                wmma: h_idx == 64
                    && d_idx == 128
                    && (gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12"))
                    && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_INDEXER_WMMA")
                        .map(|s| s != "0")
                        .unwrap_or(true),
                x_f16: &pbs.wmma_x_scratch_f16,
                n_compressed: if capture_safe {
                    &pbs.n_compressed_4_arr
                } else {
                    &pbs.n_active_topk_arr
                },
                n_per_row,
                n_max,
                capacity,
                direct,
                capture_safe,
                projection,
                main_cache,
                // The LDS-transpose (coalesced-store) gather: gfx1201, and gfx1151
                // where it measured 9.9-11.9x the scattered-store kernel,
                // byte-identical.
                tiled_gather: (gpu.arch.eq_ignore_ascii_case("gfx1201")
                    && hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_GFX1201_TOPK_GATHER_TILED")
                        .as_deref()
                        != Ok("0"))
                    || (gpu.arch.eq_ignore_ascii_case("gfx1151")
                        && hipfire_config::developer_var("HIPFIRE_DS4_GATHER_TILED").as_deref()
                            != Ok("0")),
                staged: &pbs.topk_staged_batch,
                gather_head_dim: head_dim,
            },
        };
        (Step::SparseIndex(op), n_max)
    } else {
        let n = ((start_pos as usize + rows) / ratio)
            .min(capacity)
            .min(topk);
        let n_gather = if capture_safe { topk } else { n };
        gathers = n_gather > 0;
        if n_gather > 0 {
            for b in 0..rows {
                n_active[b] = ((start_pos as usize + b + 1) / ratio)
                    .min(capacity)
                    .min(topk) as i32;
            }
        }
        (
            Step::IdentityGather(IdentityGatherOp {
                cache: main_cache,
                staged: &pbs.topk_staged_batch,
                head_dim,
                top_k_max: topk,
                rows,
                layer: layer_idx,
                route: IdentityGatherRoute::Batched { n: n_gather },
            }),
            0,
        )
    };
    if gathers && main_cache_f16 && matches!(main_cache, CompressedCache::Sharded { .. }) {
        return Err(format!(
            "F16 compressor cache does not support block-cyclic batched gather l{layer_idx}"
        ));
    }
    program.push(Stage::Other, index_step);

    let wmma_flag = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DSA_WMMA").as_deref()
        != Ok("0")
        && gpu.arch_caps.has_wmma();
    let wmma = if direct {
        // The portable WMMA symbol is not a viable gfx1201 lowering (351 ms
        // against 0.8 ms on the TP3 rank-2 shape).
        wmma_flag && !gpu.arch_caps.is_gfx1201() && n_heads % 16 == 0 && head_dim % 16 == 0
    } else {
        wmma_flag && (gpu.arch_caps.is_gfx1201() || n_heads % 16 == 0) && head_dim % 16 == 0
    };
    program.push(
        Stage::Other,
        Step::SparseAttention(SparseAttentionOp {
            q: &pbs.q_batch,
            sink,
            out: &pbs.attn_out_raw_batch,
            n_heads,
            head_dim,
            win,
            topk,
            rows,
            layer: layer_idx,
            route: SparseAttentionRoute::Batched {
                staged_swa: &pbs.swa_staged_batch,
                staged_topk: &pbs.topk_staged_batch,
                n_valid: &pbs.n_valid_swa_arr,
                n_active: if !capture_safe {
                    &pbs.n_active_topk_arr
                } else if ratio == 4 {
                    &pbs.n_active_topk_4_arr
                } else {
                    &pbs.n_active_topk_128_arr
                },
                n_active_host: n_active,
                direct: direct.then_some(DirectTopk {
                    cache: main_cache_t,
                    indices: &pbs.idx_topk_indices_batch,
                    n_compressed: n_compressed_direct,
                }),
                wmma,
                capture_safe,
            },
        }),
    );
    Ok(())
}

fn compressed_rows<'a>(
    state: &DeepseekV4State,
    cache: &'a GpuTensor,
    shards: &'a [usize; 4],
) -> CompressedCache<'a> {
    match state.compressor_cache_placement {
        CompressorCachePlacement::BlockCyclic(shard) => CompressedCache::Sharded {
            shards,
            world: shard.world(),
            block_rows: shard.block_rows(),
        },
        _ if cache.dtype == DType::F16 => CompressedCache::F16(cache),
        _ => CompressedCache::F32(cache),
    }
}

/// The batched compressor commit of one flavor over the window.
fn kv_compress_rows<'a>(
    cfg: &DeepseekV4Config,
    layer: &'a DeepseekV4LayerWeights,
    state: &'a DeepseekV4State,
    pbs: &'a PrefillBatchScratch,
    layer_idx: usize,
    window: RowsWindow<'a>,
    is_indexer: bool,
) -> Result<compressor::KvCompressOp<'a>, String> {
    let RowsWindow {
        start_pos, rows, ..
    } = window;
    let ratio = layer.compress_ratio as usize;
    let head_dim = if is_indexer {
        cfg.index_head_dim
    } else {
        cfg.head_dim
    };
    let l_state = &state._indexer[layer_idx];
    let (kv, score, norm, ape, kv_state, score_state, kv_cache) = if is_indexer {
        (
            &pbs.comp_idx_kv_batch,
            &pbs.comp_idx_score_batch,
            req(&layer.indexer_compressor_norm, "idx_comp_norm", layer_idx)?,
            req(&layer.indexer_compressor_ape, "idx_comp_ape", layer_idx)?,
            &l_state.indexer_kv_state,
            &l_state.indexer_score_state,
            &l_state.indexer_kv_cache,
        )
    } else {
        (
            &pbs.comp_main_kv_batch,
            &pbs.comp_main_score_batch,
            req(&layer.compressor_norm, "comp_norm", layer_idx)?,
            req(&layer.compressor_ape, "comp_ape", layer_idx)?,
            &l_state.main_kv_state,
            &l_state.main_score_state,
            &l_state.main_kv_cache,
        )
    };
    let max_compressed = state.compressor_capacity.active_rows();
    let slot_base = start_pos as usize % ratio;
    let first_event = if slot_base == 0 {
        ratio - 1
    } else {
        ratio - slot_base - 1
    };
    let n_events = if first_event < rows {
        (rows - first_event).div_ceil(ratio)
    } else {
        0
    };
    let first_row = start_pos as usize / ratio;
    let events = if first_row + n_events > max_compressed {
        max_compressed.saturating_sub(first_row)
    } else {
        n_events
    };
    if slot_base == 0 && events > 0 {
        if let CompressorCachePlacement::BlockCyclic(shard) = state.compressor_cache_placement {
            let last = first_row + events - 1;
            if shard.owner(first_row) != shard.owner(last) {
                return Err(format!(
                    "compressor chunk crosses a TP cache ownership block: l{layer_idx}, start_slot={first_row}, events={events}, block_rows={}",
                    shard.block_rows()
                ));
            }
        }
    }
    // "start" (the default, as the reference) rotates each committed row at
    // the start of its window; "mid" / "end" stay for diagnostic A/B.
    let mode = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_COMP_ROPE_POS")
        .ok()
        .unwrap_or_else(|| "start".to_string());
    let event_positions = (0..events)
        .map(|k| {
            let pos = first_event + k * ratio + start_pos as usize;
            let window_start = pos / ratio * ratio;
            if is_indexer {
                window_start as i32
            } else {
                match mode.as_str() {
                    "end" => pos as i32,
                    "mid" => (window_start + ratio / 2) as i32,
                    _ => window_start as i32,
                }
            }
        })
        .collect();
    let rope = if is_indexer {
        CompressorRope {
            freq_base: cfg.compress_rope_theta,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            corr_low: 0.0,
            corr_high: 0.0,
            rope_dim: cfg.qk_rope_head_dim,
        }
    } else {
        tail_rope(cfg, layer)
    };
    Ok(compressor::KvCompressOp {
        rows,
        ape,
        norm,
        kv_state: kv_state.as_ref().unwrap(),
        score_state: score_state.as_ref().unwrap(),
        kv_cache: kv_cache.as_ref().unwrap(),
        staging: Some(&pbs.comp_cache_batch_f32),
        rope,
        ratio,
        head_dim,
        eps: cfg.rms_norm_eps,
        indexer: is_indexer,
        layer: layer_idx,
        route: compressor::KvCompressRoute::Batched {
            kv,
            score,
            comp_positions: &pbs.comp_positions,
            start_pos,
            events,
            local_row: state.compressor_cache_placement.global_to_local(first_row),
            event_positions,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn expert_ffn_rows<'a>(
    cfg: &DeepseekV4Config,
    layer: &'a DeepseekV4LayerWeights,
    pbs: &'a PrefillBatchScratch,
    layer_idx: usize,
    window: RowsWindow<'a>,
    hash_routing: bool,
    projection: RowsProjection,
    backend: Mq2rBackend,
) -> Result<Step<'a>, String> {
    let rows = window.rows;
    let shared_w1 = layer.shared_w1.as_ref().unwrap();
    let shared_w2 = layer.shared_w2.as_ref().unwrap();
    let shared_w3 = layer.shared_w3.as_ref().unwrap();
    let im = cfg.moe_intermediate_size;
    let shared_im = if layer.shared_tp_size > 1 {
        layer.shared_intermediate_count
    } else {
        im
    };
    if shared_im == 0 || shared_im > im {
        return Err(format!(
            "ffn_batched l{layer_idx}: invalid shared TP width {shared_im} (global {im}, tp={})",
            layer.shared_tp_size
        ));
    }
    let fwht = crate::forward::weight_needs_fwht;
    // The routed experts read the rotated input, so it stays live with MoE on.
    let rotate_in = config_cache::moe_on() || fwht(shared_w1) || fwht(shared_w3);
    let do_routed = hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_MOE")
        .ok()
        .as_deref()
        != Some("0")
        && layer.expert_gate_up_blob.is_some()
        && layer.expert_w2_blob.is_some()
        // Hash layers without the static tid2eid table fall back to the
        // shared expert alone.
        && !(hash_routing && layer.tid2eid_host.is_empty());
    let (router, routed) = if do_routed {
        let routing = if hash_routing {
            if window.tokens.len() < rows {
                return Err(format!(
                    "ffn_batched l{layer_idx}: tokens len {} < batch_size {rows}",
                    window.tokens.len(),
                ));
            }
            let tid2eid = layer.tid2eid_dev.as_ref().ok_or_else(|| {
                format!(
                    "ffn_batched hash l{layer_idx}: tid2eid_dev missing (pre-FP4 quant skipped \
                     tid2eid; HFQ load_weights should still populate the device buffer)"
                )
            })?;
            MoePrefillRouting::Hash {
                tid2eid,
                tokens: &pbs.tokens,
            }
        } else {
            MoePrefillRouting::BiasAware {
                gate_bias: req(&layer.gate_bias, "gate.bias", layer_idx)?,
            }
        };
        (
            Some(expert_ffn::ExpertFfnRouter {
                gate: req(&layer.gate_weight, "gate.weight", layer_idx)?,
                scores: &pbs.moe_scores_batch,
                n_exp: cfg.n_routed_experts,
            }),
            Some(MoeBiasAwarePrefillParams {
                hidden: cfg.hidden_size,
                mi: im,
                n_exp: cfg.n_routed_experts,
                k_top: cfg.num_experts_per_tok,
                batch_size: rows,
                route_scale: config_cache::route_scale(cfg.routed_scaling_factor, cfg.mq2r),
                swiglu_limit: cfg.swiglu_limit,
                uses_atomic_moe_down: backend.uses_atomic_moe_down(),
                layer_idx,
                routing,
                scores: &pbs.moe_scores_batch,
                topk_indices: &pbs.moe_topk_indices_batch,
                topk_weights: &pbs.moe_topk_weights_batch,
                expert_gate_up_ptrs: layer.expert_gate_up_ptrs.as_ref().unwrap(),
                expert_down_ptrs: layer.expert_w2_ptrs.as_ref().unwrap(),
                x_rot: &pbs.ffn_x_rot_batch,
                ffn_out: &pbs.ffn_out_batch,
                expert_token_counts: &pbs.moe_expert_token_counts,
                expert_offsets: &pbs.moe_expert_offsets,
                sorted_slot_index: &pbs.moe_sorted_slot_index,
                expert_tile_ids: &pbs.moe_expert_tile_ids,
                inverse_perm: &pbs.moe_inverse_perm,
                y_gate_up_grouped: &pbs.moe_y_gate_up_grouped,
                y_down_grouped: &pbs.moe_y_down_grouped,
                gate_batch: &pbs.moe_gate_batch,
                up_batch: &pbs.moe_up_batch,
                rot_batch: &pbs.moe_rot_batch,
                down_expert_outputs: &pbs.moe_down_expert_outputs,
            }),
        )
    } else {
        (None, None)
    };
    Ok(Step::ExpertFfn(expert_ffn::ExpertFfnOp {
        x_in: &pbs.hc_x_in_batch,
        norm: layer.ffn_norm.as_ref().unwrap(),
        x_rot: &pbs.ffn_x_rot_batch,
        x_plain: &pbs.ffn_x_plain_batch,
        shared_w1,
        shared_w2,
        shared_w3,
        shared_gate: &pbs.ffn_shared_gate_batch,
        shared_up: &pbs.ffn_shared_up_batch,
        shared_rot: &pbs.ffn_shared_rot_batch,
        out: &pbs.ffn_out_batch,
        router,
        hidden: cfg.hidden_size,
        shared_im,
        swiglu_limit: cfg.swiglu_limit,
        eps: cfg.rms_norm_eps,
        rotate: rotate_in,
        rows,
        layer: layer_idx,
        route: expert_ffn::ExpertFfnRoute::Batched {
            x_f16: &pbs.wmma_x_scratch_f16,
            projection,
            routed,
        },
    }))
}

/// Developer captures observe a rows program step by step.
fn diagnostics_on() -> Result<bool, String> {
    use std::sync::LazyLock;
    static ENV: LazyLock<bool> = LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DUMP_STATE").is_ok()
            || hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ATTN_TWIN").as_deref() == Ok("1")
            || hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT").as_deref()
                == Ok("1")
    });
    Ok(*ENV || dense_activation_dump_enabled()?)
}

/// What a rows program computes, for its developer captures.
#[derive(Clone, Copy)]
pub(crate) enum Observed<'a> {
    Embed,
    /// A trunk (or MTP) decoder layer.
    Layer(usize, &'a DeepseekV4LayerWeights),
    /// A DSpark drafter stage: only the dense calibration captures.
    DrafterStage(usize, &'a DeepseekV4LayerWeights),
}

/// Runs `program`; with developer captures on, one step at a time with each
/// capture after the step that produced its tensor.
pub(crate) fn run_rows(
    program: &RowsProgram<'_>,
    cfg: &DeepseekV4Config,
    gpu: &mut Gpu,
    window: RowsWindow<'_>,
    observed: Observed<'_>,
) -> Result<(), String> {
    let ctx = DispatchCtx::new(gpu);
    if !diagnostics_on()? {
        return execute_steps(gpu, &ctx, &program.steps).map_err(dispatch_err);
    }
    for (step, stage) in program.steps.iter().zip(&program.stages) {
        execute_steps(gpu, &ctx, std::slice::from_ref(step)).map_err(dispatch_err)?;
        observe(cfg, gpu, window, observed, step, *stage)?;
    }
    Ok(())
}

fn observe(
    cfg: &DeepseekV4Config,
    gpu: &mut Gpu,
    window: RowsWindow<'_>,
    observed: Observed<'_>,
    step: &Step<'_>,
    stage: Stage,
) -> Result<(), String> {
    let pbs = window.pbs;
    let rows = window.rows;
    let (l, layer, trunk) = match observed {
        Observed::Embed => {
            dump_buf(gpu, "01_embed", &pbs.embed_batch);
            dump_buf(gpu, "02_hc_streams_init", &pbs.streams_batch);
            return Ok(());
        }
        Observed::Layer(l, layer) => (l, layer, true),
        Observed::DrafterStage(l, layer) => (l, layer, false),
    };
    let l0 = trunk && l == 0;
    let (hidden, n_heads, head_dim) = (cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim);
    let dense = dense_activation_dump_enabled()?;
    match stage {
        Stage::HcPreAttn if l0 => dump_buf(gpu, "03_l0_mhc_pre_attn_hc_x_in", &pbs.hc_x_in_batch),
        Stage::QLora if matches!(step, Step::QLoraProject(_)) => {
            if dense {
                let active = pbs.tmp_plain_batch.sub_offset(0, rows * hidden);
                let mut names = vec![
                    format!("layers.{l}.attn.wq_a.weight"),
                    format!("layers.{l}.attn.wkv.weight"),
                ];
                if layer.compress_ratio != 0 {
                    names.push(format!("layers.{l}.attn.compressor.wkv.weight"));
                    names.push(format!("layers.{l}.attn.compressor.wgate.weight"));
                }
                if layer.compress_ratio == 4 {
                    names.push(format!("layers.{l}.attn.indexer.weights_proj.weight"));
                    names.push(format!("layers.{l}.attn.indexer.compressor.wkv.weight"));
                    names.push(format!("layers.{l}.attn.indexer.compressor.wgate.weight"));
                }
                dump_dense_activations_if_enabled(gpu, &names, &active, hidden)?;
                let q_rank = cfg.q_lora_rank;
                let active = pbs.q_lat_batch.sub_offset(0, rows * q_rank);
                let mut names = vec![format!("layers.{l}.attn.wq_b.weight")];
                if layer.compress_ratio == 4 {
                    names.push(format!("layers.{l}.attn.indexer.wq_b.weight"));
                }
                dump_dense_activations_if_enabled(gpu, &names, &active, q_rank)?;
            }
            if l0 {
                dump_buf(gpu, "04_l0_q_lora_q_batch", &pbs.q_batch);
            }
        }
        Stage::KvJoint if l0 => dump_buf(gpu, "05_l0_kv_joint_kv_batch", &pbs.kv_batch),
        Stage::TailRope if l0 => {
            dump_buf(gpu, "06_l0_tail_rope_q_batch", &pbs.q_batch);
            dump_buf(gpu, "06_l0_tail_rope_kv_batch", &pbs.kv_batch);
        }
        Stage::SwaStaging if l0 && layer.compress_ratio == 0 => {
            dump_buf(gpu, "06a_l0_swa_staged", &pbs.swa_staged_batch)
        }
        Stage::SwaAttention if l0 => {
            dump_buf(gpu, "06b_l0_attn_swa_raw", &pbs.attn_out_raw_batch);
            if let Step::SwaAttention(op) = step {
                attention_debug_reruns(gpu, op)?;
            }
        }
        Stage::InverseRope if l0 && layer.compress_ratio == 0 => {
            dump_buf(gpu, "06c_l0_inv_rope_raw", &pbs.attn_out_raw_batch)
        }
        Stage::OLora => {
            if dense && trunk {
                let per_group_in = (n_heads / cfg.o_groups) * head_dim;
                let active = pbs
                    .attn_out_raw_batch
                    .sub_offset(0, rows * n_heads * head_dim);
                dump_dense_activation_if_enabled(
                    gpu,
                    &format!("layers.{l}.attn.wo_a.weight"),
                    &active,
                    per_group_in,
                )?;
                let groups_o_lora = cfg.o_groups * cfg.o_lora_rank;
                let active = pbs.wo_a_out_batch.sub_offset(0, rows * groups_o_lora);
                dump_dense_activation_if_enabled(
                    gpu,
                    &format!("layers.{l}.attn.wo_b.weight"),
                    &active,
                    groups_o_lora,
                )?;
            }
            if l0 && layer.compress_ratio == 0 {
                dump_buf(gpu, "06d_l0_attn_raw_rot", &pbs.attn_out_raw_rot_batch);
                dump_buf(gpu, "06e_l0_wo_a_out", &pbs.wo_a_out_batch);
                dump_buf(gpu, "06f_l0_wo_a_out_rot", &pbs.wo_a_out_rot_batch);
            }
        }
        Stage::AttentionEnd if l0 => dump_buf(gpu, "07_l0_attn_out_batch", &pbs.attn_out_batch),
        Stage::HcPostAttn if l0 => dump_buf(gpu, "08_l0_hc_attn_mix_streams", &pbs.streams_batch),
        Stage::HcPreFfn if l0 => dump_buf(gpu, "09_l0_mhc_pre_ffn_hc_x_in", &pbs.hc_x_in_batch),
        Stage::Ffn => {
            if dense {
                let active = pbs.ffn_x_plain_batch.sub_offset(0, rows * hidden);
                let names = [
                    format!("layers.{l}.ffn.shared_experts.w1.weight"),
                    format!("layers.{l}.ffn.shared_experts.w3.weight"),
                    format!("layers.{l}.ffn.gate.weight"),
                ];
                dump_dense_activations_if_enabled(gpu, &names, &active, hidden)?;
                if let Step::ExpertFfn(op) = step {
                    let active = pbs.ffn_shared_gate_batch.sub_offset(0, rows * op.shared_im);
                    dump_dense_activation_if_enabled(
                        gpu,
                        &format!("layers.{l}.ffn.shared_experts.w2.weight"),
                        &active,
                        op.shared_im,
                    )?;
                }
            }
            if l0 {
                dump_buf(gpu, "10_l0_ffn_out", &pbs.ffn_out_batch);
            }
        }
        Stage::HcPostFfn if trunk && l <= 3 => {
            dump_buf(gpu, &format!("11_l{l}_end_streams"), &pbs.streams_batch)
        }
        _ => {}
    }
    Ok(())
}

/// Developer re-runs of the layer-0 sliding-window attention kernel after its
/// step: `HIPFIRE_DEEPSEEK4_ATTN_TWIN=1` (same kernel again) and
/// `HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT=1` (debug variant writing the softmax
/// max / sum).
fn attention_debug_reruns(gpu: &mut Gpu, op: &SwaAttentionOp<'_>) -> Result<(), String> {
    let SwaAttentionRoute::Batched {
        staged, n_valid, ..
    } = op.route
    else {
        return Ok(());
    };
    let (h, hd, g, win, b) = (
        op.n_heads as i32,
        op.head_dim as i32,
        op.n_groups as i32,
        op.win as i32,
        op.rows as i32,
    );
    if hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ATTN_TWIN").as_deref() == Ok("1") {
        gpu.deepseek4_attn_swa_batched(
            op.q, staged, staged, op.sink, n_valid, op.out, h, hd, g, win, b,
        )
        .map_err(|e| format!("deepseek4_attn_swa_batched twin l0: {e:?}"))?;
        dump_buf(gpu, "06b2_l0_attn_swa_raw_twin", op.out);
    }
    if hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT").as_deref() == Ok("1") {
        let debug_max = gpu
            .alloc_tensor(&[op.rows, op.n_heads], DType::F32)
            .map_err(|e| format!("alloc debug_max: {e:?}"))?;
        let debug_sumexp = gpu
            .alloc_tensor(&[op.rows, op.n_heads], DType::F32)
            .map_err(|e| format!("alloc debug_sumexp: {e:?}"))?;
        gpu.deepseek4_attn_swa_batched_debug(
            op.q,
            staged,
            staged,
            op.sink,
            n_valid,
            op.out,
            &debug_max,
            &debug_sumexp,
            h,
            hd,
            g,
            win,
            b,
        )
        .map_err(|e| format!("deepseek4_attn_swa_batched_debug l0: {e:?}"))?;
        dump_buf(gpu, "06b_dbg_max", &debug_max);
        dump_buf(gpu, "06b_dbg_sumexp", &debug_sumexp);
        dump_buf(gpu, "06b_dbg_attn_out", op.out);
    }
    Ok(())
}

/// Exact-size views of the window scratch a DSpark stage writes its context
/// KV rows through (the compressor / FFN scratch is idle there).
pub(crate) struct DrafterViews {
    x_rot: GpuTensor,
    kv: GpuTensor,
    position: GpuTensor,
    block_kv: GpuTensor,
}

impl DrafterViews {
    pub(crate) fn new(cfg: &DeepseekV4Config, pbs: &PrefillBatchScratch, block: usize) -> Self {
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        Self {
            x_rot: pbs.ffn_x_rot_batch.sub_offset(0, cfg.hidden_size),
            kv: pbs.comp_main_kv_batch.sub_offset(0, kv_dim),
            position: pbs.comp_positions.sub_offset(0, 1),
            block_kv: pbs.kv_batch.sub_offset(0, block * kv_dim),
        }
    }
}

/// One DSpark drafter stage over its `block` rows: the trunk's attention
/// input, each context row's latent KV into the stage ring, bidirectional
/// attention over [ring window | block KV] (`staged`), O-LoRA, HC mix, then
/// the score-routed FFN and its mix. The drafter runs the portable route.
#[allow(clippy::too_many_arguments)]
pub(crate) fn drafter_stage_program<'a>(
    cfg: &DeepseekV4Config,
    layer: &'a DeepseekV4LayerWeights,
    pbs: &'a PrefillBatchScratch,
    views: &'a DrafterViews,
    gpu: &Gpu,
    stage: usize,
    block: usize,
    context: &'a GpuTensor,
    context_positions: Vec<u32>,
    ring: &'a GpuTensor,
    staged: &'a GpuTensor,
    n_committed: usize,
) -> Result<RowsProgram<'a>, String> {
    let backend = Mq2rBackend::Portable;
    let projection = rows_projection(gpu, backend);
    let (n_kv, head_dim, win) = (cfg.num_key_value_heads, cfg.head_dim, cfg.sliding_window);
    let stage_w = win + block;
    let mut program = RowsProgram::new();
    push_attention_input(&mut program, cfg, layer, pbs, gpu, stage, block, projection)?;
    program.push(
        Stage::Other,
        Step::RingKv(RingKvOp {
            wkv: req(&layer.wkv, "wkv", stage)?,
            kv_norm: req(&layer.kv_norm, "kv_norm", stage)?,
            x: context,
            x_rot: &views.x_rot,
            kv: &views.kv,
            position: &views.position,
            ring,
            positions: context_positions,
            hidden: cfg.hidden_size,
            n_kv,
            head_dim,
            win,
            eps: cfg.rms_norm_eps,
            rope: tail_rope(cfg, layer),
            e8_soa: e8_soa_kernel(&gpu.arch, backend),
            layer: stage,
        }),
    );
    program.push(
        Stage::Other,
        Step::BlockStage(BlockStageOp {
            ring,
            block_kv: &views.block_kv,
            staged,
            win,
            kv_dim: n_kv * head_dim,
            head_dim,
            n_committed,
            block,
            stage_w,
            layer: stage,
        }),
    );
    program.push(
        Stage::Other,
        Step::SwaAttention(SwaAttentionOp {
            q: &pbs.q_batch,
            sink: req(&layer.attn_sink, "attn_sink", stage)?,
            out: &pbs.attn_out_raw_batch,
            n_heads: cfg.num_attention_heads,
            head_dim,
            n_groups: cfg.o_groups,
            win: stage_w,
            rows: block,
            layer: stage,
            route: SwaAttentionRoute::Batched {
                staged,
                n_valid: &pbs.n_valid_swa_arr,
                start_pos: 0,
                per_position: false,
            },
        }),
    );
    push_o_lora(
        &mut program,
        cfg,
        layer,
        pbs,
        stage,
        block,
        false,
        0,
        projection,
    )?;
    push_mix(&mut program, cfg, pbs, gpu, false, block);
    let window = RowsWindow::new(pbs, 0, block, &[], false);
    push_ffn(
        &mut program,
        cfg,
        layer,
        pbs,
        gpu,
        stage,
        window,
        false,
        projection,
        backend,
    )?;
    push_mix(&mut program, cfg, pbs, gpu, true, block);
    Ok(program)
}

/// Head prologue (head HC combine, final RMSNorm) of each of the window's
/// `rows` rows staged into `state.head_norm_batch`, then with `logits` one
/// batched lm_head projection into `state.head_logits_batch` (Q8 / F16 / F32
/// heads: the plain normed rows are the projection input).
pub(crate) fn run_head_rows(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    rows: usize,
    logits: bool,
) -> Result<(), String> {
    ensure_final_norm_compute(cfg, weights, state, gpu)?;
    let state = &*state;
    let hidden = cfg.hidden_size;
    let norm_rows = state.head_norm_batch.as_ref().unwrap();
    let mut steps = vec![Step::HcHead(final_norm_compute_op(
        cfg,
        weights,
        state,
        &pbs.streams_batch,
        rows,
        hc::HcHeadRoute::Batched {
            rows_out: norm_rows,
        },
    )?)];
    if logits {
        steps.push(Step::PreparedProjection(project::PreparedProjectionOp {
            rows,
            weight: weights
                .head
                .as_ref()
                .ok_or_else(|| "head not uploaded".to_string())?,
            x_rotated: norm_rows,
            x_plain: norm_rows,
            y: state.head_logits_batch.as_ref().unwrap(),
            m: cfg.vocab_size,
            k: hidden,
            route: project::PreparedProjectionRoute::Batched {
                x_f16: state.head_x_f16.as_ref(),
                projection: rows_projection(gpu, weights.mq2r_backend),
            },
        }));
    }
    let ctx = DispatchCtx::new(gpu);
    execute_steps(gpu, &ctx, &steps).map_err(dispatch_err)
}

#[cfg(test)]
mod tests {
    use super::compressor_window_batched;

    #[test]
    fn compressor_small_unaligned_no_event_chunks_stay_batched() {
        assert!(compressor_window_batched(129, 4, 128));
        assert!(compressor_window_batched(253, 2, 128));
        assert!(!compressor_window_batched(253, 3, 128));

        assert!(compressor_window_batched(8, 9, 4));
        assert!(compressor_window_batched(9, 2, 4));
        assert!(!compressor_window_batched(9, 3, 4));
    }
}
