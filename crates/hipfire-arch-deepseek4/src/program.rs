// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative DeepSeek-V4 decoder-layer program (one decode row).
//!
//! A layer is two hyper-connected sublayers over the `[hc_mult, hidden]`
//! residual streams `X`:
//!
//! ```text
//! attention: x  = HcPre(X)                       (Sinkhorn mHC control, input map)
//!            n  = rmsnorm(x, attn_norm)           (plain + FWHT-rotated)
//!            q  = rmsnorm_heads(wq_b · rmsnorm(wq_a · n, q_norm))
//!            kv = rmsnorm(wkv · n, kv_norm);  tail YaRN RoPE(q, kv)
//!            ratio > 0:  compressor(n) appends a pooled row every `ratio` tokens
//!            ratio == 4: indexer compressor(n); indexer picks the top-k rows
//!            a  = attend(q; SWA ring ∪ compressed rows ∪ sink); inverse RoPE
//!            X' = HcPost(X, wo_b · grouped wo_a · a)
//! FFN:       x  = HcPre(X');  X" = HcPost(X', shared(x) + routed_top6(x))
//! head:      logits = head · rmsnorm(HcHead(X), output_norm)
//! ```
//!
//! This module binds the model's resident weights, per-layer caches and
//! scratch to typed `hipfire_dispatch::pipeline::mla` steps; kernels, their
//! order and every route choice live in dispatch. With HC ping-pong the
//! attention mix writes the second stream buffer and the FFN mix writes back,
//! so a layer ends in the buffer it started in and needs no swap.

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::mla::hc::HcMixBranch;
use hipfire_dispatch::pipeline::{execute_steps, Step};
use rdna_compute::{Gpu, GpuTensor};

use crate::config_cache;
use crate::forward::{
    apply_tail_rope_op, attention_dumps, attention_input_pack_op, attention_ops, compressor_op,
    compressor_project_op, dispatch_err, dump_indexer_state, dump_stage_norm,
    ensure_apply_tail_rope, ensure_attention, ensure_attention_input_pack, ensure_compressor,
    ensure_expert_ffn, ensure_indexer, ensure_kv_joint, ensure_mhc_pre, ensure_q_lora_prepare,
    expert_ffn_dumps, expert_ffn_step, hc_mix_op, indexer_compressor_pack_op, indexer_filled_rows,
    indexer_op, kv_joint_op, mhc_pre_op, q_lora_prepare_dumps, q_lora_prepare_op,
    q_lora_project_dumps, q_lora_project_op, OloraSchedule,
};
use crate::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};

/// What a step produces, for the diagnostics that observe a program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    HcPreAttn,
    QueryNorm,
    QueryProject,
    Index,
    Attention,
    HcPostAttn,
    HcPreFfn,
    Ffn,
    HcPostFfn,
    Other,
}

/// The steps of one or both sublayers of a layer.
pub(crate) struct LayerProgram<'a> {
    steps: Vec<Step<'a>>,
    stages: Vec<Stage>,
}

impl<'a> LayerProgram<'a> {
    pub(crate) fn new() -> Self {
        Self {
            steps: Vec::with_capacity(20),
            stages: Vec::with_capacity(20),
        }
    }

    fn push(&mut self, stage: Stage, step: Step<'a>) {
        self.steps.push(step);
        self.stages.push(stage);
    }
}

/// The HC mix that closes a sublayer, when it runs in the same program.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mix {
    /// The sublayer output stays for an external reduction (EP / TP); the
    /// owner runs the mix afterwards.
    Deferred,
    /// The program mixes the sublayer output into the streams.
    Fold,
}

/// Allocates (once) everything one decode row of `layer_idx` binds, and
/// points `state.pos_buf` at the layer's position slot.
pub(crate) fn ensure_layer(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    ensure_attention_sublayer(cfg, weights, state, gpu, layer_idx)?;
    ensure_expert_ffn(cfg, weights, state, gpu, layer_idx)
}

pub(crate) fn ensure_attention_sublayer(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    ensure_mhc_pre(cfg, weights, state, gpu)?;
    ensure_q_lora_prepare(cfg, state, gpu)?;
    ensure_attention_input_pack(cfg, weights, state, gpu, layer_idx)?;
    ensure_kv_joint(cfg, state, gpu)?;
    ensure_apply_tail_rope(state, layer_idx)?;
    ensure_compressor(cfg, weights, state, gpu, layer_idx, false)?;
    ensure_compressor(cfg, weights, state, gpu, layer_idx, true)?;
    if weights.resolve_layer(layer_idx).compress_ratio == 4 {
        ensure_indexer(cfg, weights, state, gpu, layer_idx)?;
    }
    ensure_attention(cfg, weights, state, gpu, layer_idx)
}

/// Both sublayers of `layer_idx` for one decode row at `position` whose
/// token is `token_id` (hash-routed layers read it), mixing back into
/// `state.residual_streams`.
pub(crate) fn layer_program<'a>(
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    position: u32,
    token_id: u32,
    skip_ffn: bool,
) -> Result<LayerProgram<'a>, String> {
    let pingpong = config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r);
    let streams = state.residual_streams.as_ref().unwrap();
    let mut program = LayerProgram::new();
    let attn_out = attention_steps(
        &mut program,
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        position,
        streams,
        Mix::Fold,
        OloraSchedule::Default,
    )?;
    // Without ping-pong the mix lands back in `streams`.
    let ffn_streams = if pingpong { attn_out } else { streams };
    ffn_steps(
        &mut program,
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        token_id,
        skip_ffn,
        ffn_streams,
        None,
        Mix::Fold,
    )?;
    Ok(program)
}

/// The attention sublayer reading `streams`. Returns the buffer the HC mix
/// writes (`streams` itself when `mix` is deferred).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_steps<'a>(
    program: &mut LayerProgram<'a>,
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    position: u32,
    streams: &'a GpuTensor,
    mix: Mix,
    olora: OloraSchedule,
) -> Result<&'a GpuTensor, String> {
    let ratio = weights.resolve_layer(layer_idx).compress_ratio;
    program.push(
        Stage::HcPreAttn,
        Step::HcPre(mhc_pre_op(
            cfg, weights, state, gpu, layer_idx, true, streams,
        )),
    );
    program.push(
        Stage::QueryNorm,
        Step::QLoraPrepare(q_lora_prepare_op(cfg, weights, state, layer_idx)?),
    );
    // Exact gfx1201: the projections that share the normalized input run as
    // one mixed-row launch; their own steps then skip the projection.
    let pack = attention_input_pack_op(cfg, weights, state, gpu, layer_idx)?;
    let packed = pack.is_some();
    if let Some(pack) = pack {
        program.push(Stage::Other, Step::PackedProjections(pack));
    }
    program.push(
        Stage::QueryProject,
        Step::QLoraProject(q_lora_project_op(
            cfg, weights, state, gpu, layer_idx, packed,
        )?),
    );
    program.push(
        Stage::Other,
        Step::KvJoint(kv_joint_op(cfg, weights, state, gpu, layer_idx, packed)?),
    );
    program.push(
        Stage::Other,
        Step::ApplyTailRope(apply_tail_rope_op(cfg, weights, state, layer_idx)),
    );
    if ratio > 0 {
        push_compressor(
            program, cfg, weights, state, gpu, layer_idx, position, false, packed,
        )?;
        if ratio == 4 {
            let indexer_pack = if packed {
                indexer_compressor_pack_op(cfg, weights, state, gpu, layer_idx)?
            } else {
                None
            };
            let indexer_packed = indexer_pack.is_some();
            if let Some(pack) = indexer_pack {
                program.push(Stage::Other, Step::PackedProjections(pack));
            }
            push_compressor(
                program,
                cfg,
                weights,
                state,
                gpu,
                layer_idx,
                position,
                true,
                indexer_packed,
            )?;
            program.push(
                Stage::Index,
                Step::SparseIndex(indexer_op(cfg, weights, state, gpu, layer_idx, packed)?),
            );
        }
    }
    // The attention dumps read the O-LoRA inputs after the last step.
    let attention = attention_ops(cfg, weights, state, gpu, layer_idx, olora)?;
    let last = attention.len() - 1;
    for (i, step) in attention.into_iter().enumerate() {
        program.push(
            if i == last {
                Stage::Attention
            } else {
                Stage::Other
            },
            step,
        );
    }
    if mix == Mix::Deferred {
        return Ok(streams);
    }
    let op = hc_mix_op(
        cfg,
        state,
        config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r),
        streams,
        HcMixBranch::Transform(state.attn_out.as_ref().unwrap()),
        "hc_mix_4stream layer",
        "d2d hc_mix → streams",
    );
    let out = op.streams_out;
    program.push(Stage::HcPostAttn, Step::HcPost(op));
    Ok(out)
}

/// One compressor flavor of a decode row: its `wkv` / `wgate` projections
/// (unless a packed projection already wrote the `kv` / `score` scratch), then
/// the compressor over them.
#[allow(clippy::too_many_arguments)]
fn push_compressor<'a>(
    program: &mut LayerProgram<'a>,
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    position: u32,
    is_indexer: bool,
    preprojected: bool,
) -> Result<(), String> {
    let missing = || format!("compressor does not run on layer {layer_idx}");
    if !preprojected {
        let x = state.tmp.as_ref().unwrap();
        let op = compressor_project_op(cfg, weights, state, gpu, layer_idx, x, is_indexer)?
            .ok_or_else(missing)?;
        program.push(Stage::Other, Step::CompressorProject(op));
    }
    let op = compressor_op(
        cfg, weights, state, gpu, layer_idx, position, is_indexer, None,
        /*state_buffer_driven=*/ true,
    )?
    .ok_or_else(missing)?;
    program.push(Stage::Other, Step::KvCompress(op));
    Ok(())
}

/// The FFN sublayer reading `streams`; `routed_out` redirects the routed
/// experts into an EP partial (the shared expert stays in `ffn_out`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ffn_steps<'a>(
    program: &mut LayerProgram<'a>,
    cfg: &DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a DeepseekV4State,
    gpu: &Gpu,
    layer_idx: usize,
    token_id: u32,
    skip_ffn: bool,
    streams: &'a GpuTensor,
    routed_out: Option<&'a GpuTensor>,
    mix: Mix,
) -> Result<(), String> {
    program.push(
        Stage::HcPreFfn,
        Step::HcPre(mhc_pre_op(
            cfg, weights, state, gpu, layer_idx, false, streams,
        )),
    );
    let fold = mix == Mix::Fold;
    program.push(
        Stage::Ffn,
        expert_ffn_step(
            cfg,
            weights,
            state,
            gpu,
            layer_idx,
            token_id,
            skip_ffn,
            routed_out,
            routed_out.is_none() && fold,
        )?,
    );
    if fold {
        program.push(
            Stage::HcPostFfn,
            Step::HcPost(hc_mix_op(
                cfg,
                state,
                config_cache::hc_pingpong_on(&gpu.arch, cfg.mq2r),
                streams,
                HcMixBranch::Transform(state.ffn_out.as_ref().unwrap()),
                "hc_mix_4stream ffn",
                "d2d hc_ffn_mix → streams",
            )),
        );
    }
    Ok(())
}

/// Developer captures observe a program step by step; otherwise it runs as
/// one list.
fn diagnostics_on() -> Result<bool, String> {
    use std::sync::LazyLock;
    static ENV: LazyLock<bool> = LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_STAGE_NORM")
            .ok()
            .as_deref()
            == Some("1")
            || hipfire_config::developer_var("HIPFIRE_DEEPSEEK4_DUMP_INDEXER").is_ok()
            || hipfire_config::developer_var("HIPFIRE_DS4_ROUTE_DUMP").is_ok()
    });
    Ok(*ENV || crate::forward::dense_activation_dump_enabled()?)
}

/// Runs `program`; with developer captures on, one step at a time with each
/// capture after the step that produced its tensor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    program: &LayerProgram<'_>,
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    position: u32,
    skip_ffn: bool,
) -> Result<(), String> {
    let ctx = DispatchCtx::new(gpu);
    if !diagnostics_on()? {
        return execute_steps(gpu, &ctx, &program.steps).map_err(dispatch_err);
    }
    for (step, stage) in program.steps.iter().zip(&program.stages) {
        execute_steps(gpu, &ctx, std::slice::from_ref(step)).map_err(dispatch_err)?;
        observe(
            cfg, weights, state, gpu, step, *stage, layer_idx, position, skip_ffn,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn observe(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &DeepseekV4State,
    gpu: &mut Gpu,
    step: &Step<'_>,
    stage: Stage,
    layer_idx: usize,
    position: u32,
    skip_ffn: bool,
) -> Result<(), String> {
    let stage_norm =
        |gpu: &mut Gpu, tag: &str, t: &GpuTensor| dump_stage_norm(gpu, tag, t, layer_idx, position);
    match stage {
        Stage::HcPreAttn => stage_norm(gpu, "hc_x_in_attn", state.hc_x_in.as_ref().unwrap()),
        Stage::QueryNorm => q_lora_prepare_dumps(cfg, weights, state, gpu, layer_idx)?,
        Stage::QueryProject => {
            if let Step::QLoraProject(op) = step {
                q_lora_project_dumps(weights, state, gpu, op)?;
            }
        }
        Stage::Index => dump_indexer_state(
            gpu,
            state,
            layer_idx,
            position,
            indexer_filled_rows(state, position),
        ),
        Stage::Attention => {
            attention_dumps(cfg, state, gpu, layer_idx)?;
            stage_norm(gpu, "attn_out", state.attn_out.as_ref().unwrap());
        }
        Stage::HcPostAttn => stage_norm(gpu, "hc_post_attn", mixed(step)),
        Stage::HcPreFfn => stage_norm(gpu, "hc_x_in_ffn", state.hc_x_in.as_ref().unwrap()),
        Stage::Ffn => {
            if !skip_ffn {
                expert_ffn_dumps(cfg, weights, state, gpu, layer_idx)?;
            }
            stage_norm(gpu, "ffn_out", state.ffn_out.as_ref().unwrap());
        }
        Stage::HcPostFfn => stage_norm(gpu, "hc_post_ffn", mixed(step)),
        Stage::Other => {}
    }
    Ok(())
}

/// The streams an HC mix step leaves its result in.
fn mixed<'s>(step: &Step<'s>) -> &'s GpuTensor {
    match step {
        Step::HcPost(op) => op.writeback.unwrap_or(op.streams_out),
        _ => unreachable!("HC post stage tags an HcPost step"),
    }
}
