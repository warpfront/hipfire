// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative Cohere2-MoE (North-Mini-Code) decoder-layer program.
//!
//! Every layer is the parallel block
//! `n = norm(h); h += o(attend(rope(qkv(n)))); h += ffn(n)`: GQA attention
//! over the Q8 KV cache (sliding layers windowed, global layers full causal),
//! GPT-J interleaved RoPE on sliding layers and the rotated dense prefix, NoPE
//! elsewhere; the FFN is a dense SwiGLU on the prefix and the sealed sigmoid
//! MoE (no renorm, no bias, no shared expert) after it. This module only binds
//! resident weights, KV state and scratch to typed `hipfire_dispatch` steps;
//! route selection, fusion and the executor bodies live in dispatch.
//!
//! One program serves decode and batched prefill ([`Route`]); the routes
//! differ only in what they bind: per-row GEMVs with fused residuals and the
//! decode MoE seal, or `rows` projections on one [`ProjectLadder`] with a
//! separate residual add, per-row RoPE positions and the prefill MoE seal.

use crate::cohere2moe::{Cohere2MoeLayerWeights, Cohere2MoeWeights, Ffn, MoeFfn};
use crate::config::{AttnKind, Cohere2MoeConfig};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::{GivensRef, WeightRef};
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::families::moe::{
    MoeDtypes, MoeEpMode, MoeNormalization, MoeParams, MoePrefillParams, MoePrefillPrelude,
    MoeQ8RouterPolicy, MoeRecipe,
};
use hipfire_dispatch::pipeline::sandwich::{Rope, RopeKind};
use hipfire_dispatch::pipeline::sealed_moe::{
    seal_decode, seal_prefill, BoundMoeExperts, PrefillRouteMode,
};
use hipfire_dispatch::pipeline::{GemvInput, ProjectLadder, ProjectOp, Step};
use hipfire_dispatch::types::RotationPlan;
use hipfire_runtime::llama::{KvCache, KvCacheExt, WeightTensor};
use rdna_compute::{DType, GpuTensor};

/// Routed-MoE scratch of a prefill chunk (sizes as `forward_batch` allocates).
pub(crate) struct MoeRows<'a> {
    pub router_logits: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub rot: &'a GpuTensor,
    pub down_expanded: &'a GpuTensor,
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    pub y_gate_up_grouped: &'a GpuTensor,
    pub y_down_grouped: &'a GpuTensor,
    pub m_total_max: usize,
}

/// Decode MoE scratch (the `Cohere2MoeState` buffers).
pub(crate) struct MoeRow<'a> {
    pub router_logits: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub gate_up: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub down: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    pub down_expanded: &'a GpuTensor,
}

/// Which forward a program runs.
pub(crate) enum Route<'a> {
    /// One row at `position` (the device scalar in `Bufs::pos_buf`).
    Decode { position: usize, moe: MoeRow<'a> },
    /// `rows` consecutive positions `positions` (I32) ending at `max_ctx`.
    Prefill {
        rows: usize,
        positions: &'a GpuTensor,
        max_ctx: usize,
        ladder: ProjectLadder,
        /// The projections' ladder scratch, `[rows, max K]` (F16 for
        /// `GemmQ8Fp16`); also the Q8 router's FP16 WMMA scratch.
        scratch: &'a GpuTensor,
        moe: MoeRows<'a>,
    },
}

/// Activations of one program, each `[rows, ·]`.
pub(crate) struct Bufs<'a> {
    pub h: &'a GpuTensor,
    /// The shared pre-norm, read by both branches.
    pub normed: &'a GpuTensor,
    pub fa_q: &'a GpuTensor,
    pub fa_k: &'a GpuTensor,
    pub fa_v: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    /// Prefill residual-projection output before its add (unread on decode,
    /// whose residual GEMVs accumulate in place).
    pub proj_out: &'a GpuTensor,
    pub dense_gate: &'a GpuTensor,
    pub dense_up: &'a GpuTensor,
    pub dense_act: &'a GpuTensor,
    pub pos_buf: &'a hip_bridge::DeviceBuffer,
    pub kv: &'a KvCache,
    pub flash_partials: &'a GpuTensor,
    pub route: Route<'a>,
}

/// The GEMV form `weight_gemv` launched: no AWQ or row stride; a PARO weight
/// keeps its Givens sidecar (`weight_gemv` rotates ParoQ4 through `w.paro`).
fn plain_ref(w: &WeightTensor) -> WeightRef<'_> {
    WeightRef {
        row_stride: 0,
        awq_scale: None,
        ..w.dispatch_ref()
    }
}

impl<'a> Bufs<'a> {
    fn rows(&self) -> usize {
        match self.route {
            Route::Decode { .. } => 1,
            Route::Prefill { rows, .. } => rows,
        }
    }

    /// `out = w · input`.
    fn project(&self, w: &'a WeightRef<'a>, input: &'a GpuTensor, out: &'a GpuTensor) -> Step<'a> {
        match self.route {
            Route::Decode { .. } => Step::Gemv {
                w,
                input: GemvInput::Raw(input),
                out,
            },
            Route::Prefill {
                rows,
                ladder,
                scratch,
                ..
            } => Step::Project(ProjectOp {
                weight: *w,
                input,
                output: out,
                rows,
                rotation: Some(scratch),
                ladder,
                residual: None,
                grouped: None,
            }),
        }
    }

    /// `h += w · input`: the residual GEMV on decode; on prefill the
    /// projection lands in `proj_out` and is added.
    fn project_residual(&self, w: &'a WeightRef<'a>, input: &'a GpuTensor) -> Step<'a> {
        match self.route {
            Route::Decode { .. } => Step::GemvResidual {
                w,
                input: GemvInput::Raw(input),
                residual: self.h,
                out: self.h,
            },
            Route::Prefill {
                rows,
                ladder,
                scratch,
                ..
            } => Step::Project(ProjectOp {
                weight: *w,
                input,
                output: self.proj_out,
                rows,
                rotation: Some(scratch),
                ladder,
                residual: Some(self.h),
                grouped: None,
            }),
        }
    }
}

/// Resident weight refs of one layer, borrowed by its steps.
pub(crate) struct LayerRefs<'a> {
    layer: &'a Cohere2MoeLayerWeights,
    /// q, k, v, o.
    attn: [WeightRef<'a>; 4],
    /// Dense prefix gate, up, down.
    dense: Option<[WeightRef<'a>; 3]>,
}

impl<'a> LayerRefs<'a> {
    pub fn new(layer: &'a Cohere2MoeLayerWeights) -> Self {
        let attn = [&layer.wq, &layer.wk, &layer.wv, &layer.wo].map(plain_ref);
        let dense = match &layer.ffn {
            Ffn::Dense(d) => Some([&d.gate, &d.up, &d.down].map(plain_ref)),
            Ffn::Moe(_) => None,
        };
        Self { layer, attn, dense }
    }

    /// Parallel-block norm and attention branch:
    /// `normed = norm(h); h += o(attend(rope(qkv(normed))))`.
    pub fn attention_steps(
        &'a self,
        cfg: &Cohere2MoeConfig,
        b: &'a Bufs<'a>,
        layer_idx: usize,
    ) -> Result<Vec<Step<'a>>, String> {
        let [wq, wk, wv, wo] = &self.attn;
        let (n_heads, n_kv, head_dim) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        // RoPE on sliding layers and on the rotated dense prefix (force_rope:
        // North layer 0 is full_attention but rotated); other global layers
        // are NoPE. Cohere2 pairs adjacent channels over the full head.
        let rotated = self.layer.attn_kind == AttnKind::Sliding
            || (layer_idx < cfg.first_k_dense_replace
                && cfg.prefix_dense_sliding_window_pattern == 1);
        let rope = Rope {
            kind: if rotated {
                RopeKind::Interleaved { rot_dim: head_dim }
            } else {
                RopeKind::NoPe
            },
            theta: cfg.rope_theta,
        };
        // Sliding layers clip to the last `sliding_window` keys; global layers
        // are full causal. The windowed Q8 kernels are taken unconditionally:
        // the non-windowed keys would attend the whole context past the window.
        let window = if self.layer.attn_kind == AttnKind::Sliding {
            cfg.sliding_window as i32
        } else {
            0
        };
        let (pos, positions, batch_size, max_ctx_len) = match b.route {
            Route::Decode { position, .. } => (position, None, 1, 0),
            Route::Prefill {
                rows,
                positions,
                max_ctx,
                ..
            } => (0, Some(positions), rows, max_ctx),
        };
        let plan = KvTierPlan::derive(KvTierInputs {
            pos,
            q8_windowed: true,
            window,
            batch_size,
            ..b.kv.tier_inputs()
        })
        .map_err(|e| format!("kv tier: {e}"))?;
        Ok(vec![
            Step::RmsnormAutomatic {
                x: b.h,
                norm_weight: &self.layer.input_norm,
                x_plain: b.normed,
                out: b.normed,
                awq_scale: None,
                k: cfg.hidden_size,
                eps: cfg.rms_norm_eps,
                rotation: RotationPlan::None,
                rows: b.rows(),
            },
            b.project(wq, b.normed, b.fa_q),
            b.project(wk, b.normed, b.fa_k),
            b.project(wv, b.normed, b.fa_v),
            Step::Rope {
                q: b.fa_q,
                k: b.fa_k,
                pos_buf: b.pos_buf,
                positions,
                n_heads,
                n_kv_heads: n_kv,
                head_dim,
                rope,
            },
            Step::Attend {
                plan,
                io: AttnParams {
                    q: b.fa_q,
                    k: b.fa_k,
                    v: b.fa_v,
                    k_cache: &b.kv.k_gpu[layer_idx],
                    v_cache: &b.kv.v_gpu[layer_idx],
                    k_scales: None,
                    v_scales: None,
                    pos_buf: b.pos_buf,
                    pos,
                    positions,
                    n_heads,
                    n_kv_heads: n_kv,
                    head_dim,
                    physical_cap: b.kv.physical_cap,
                    batch_size,
                    max_ctx_len,
                    flash_partials: Some(b.flash_partials),
                    givens_cos: None,
                    givens_sin: None,
                    tree_bias: None,
                    block_start: 0,
                    block_cols: 0,
                    output_gate: None,
                    output_awq_scale: None,
                    output: b.attn_out,
                },
            },
            b.project_residual(wo, b.attn_out),
        ])
    }

    /// FFN branch over the same `normed`: `h += ffn(normed)`.
    pub fn ffn_steps(
        &'a self,
        cfg: &Cohere2MoeConfig,
        ctx: &'a DispatchCtx,
        b: &'a Bufs<'a>,
        layer_idx: usize,
    ) -> Result<Vec<Step<'a>>, String> {
        let m = match (&self.layer.ffn, &self.dense) {
            (Ffn::Dense(_), Some([gate, up, down])) => {
                return Ok(vec![
                    b.project(gate, b.normed, b.dense_gate),
                    b.project(up, b.normed, b.dense_up),
                    Step::SiluMul {
                        gate: b.dense_gate,
                        up: b.dense_up,
                        out: b.dense_act,
                    },
                    b.project_residual(down, b.dense_act),
                ]);
            }
            (Ffn::Moe(m), _) => m,
            (Ffn::Dense(_), None) => unreachable!("dense FFN refs are bound in LayerRefs::new"),
        };
        let bound = BoundMoeExperts::from_cache(&m.sealed.table, &m.sealed.cache)
            .map_err(|e| format!("bind experts: {e:?}"))?;
        let dtypes = moe_dtypes(m)?;
        let paro = |w: &'a WeightTensor| {
            w.paro.as_ref().map(|paro| GivensRef {
                pairs: &paro.pairs,
                theta: &paro.theta,
                scales: &paro.channel_scales,
                krot: paro.krot as usize,
            })
        };
        let (gate_up, down) = (&m.experts[0].gate_up, &m.experts[0].down);
        let call = match &b.route {
            Route::Decode { moe, .. } => seal_decode(
                bound,
                ctx,
                MoeParams {
                    dtypes,
                    mq2_lloyd: None,
                    recipe: MoeRecipe::SigmoidRoutedNoShared,
                    route_policy: None,
                    normalization: MoeNormalization::Provided,
                    batch_size: 1,
                    hidden: cfg.hidden_size,
                    mi: cfg.moe_intermediate_size,
                    k: cfg.num_experts_per_tok,
                    n_exp: cfg.num_experts,
                    norm_topk_prob: cfg.norm_topk_prob,
                    selection_bias: None,
                    x_rot_prerotated: false,
                    defer_routed_combine: false,
                    cpu_deferred_experts: false,
                    ep_mode: MoeEpMode::None,
                    layer_idx: layer_idx as u16,
                    x_norm: b.normed,
                    x_residual: b.h,
                    routed_out: None,
                    skip_shared: false,
                    router: m.router.dispatch_ref(),
                    shared: None,
                    expert_gate_up_ptrs: &m.expert_gate_up_ptrs,
                    expert_down_ptrs: &m.expert_down_ptrs,
                    expert_ptrs_host: None,
                    expert_down_awq_ptrs: None,
                    expert_dtype_tags: None,
                    routed_gate_up_k: gate_up.k,
                    routed_down_m: down.m,
                    routed_down_k: down.k,
                    routed_experts: m,
                    routed_gate_up_paro: paro(gate_up),
                    routed_down_paro: paro(down),
                    router_logits: moe.router_logits,
                    x_rot_local: moe.x_rot,
                    gate_up_buf: moe.gate_up,
                    ffn_hidden: moe.act,
                    ffn_out: moe.down,
                    gate_batch: moe.gate_batch,
                    up_batch: moe.up_batch,
                    rot_batch: moe.rot_batch,
                    topk_indices: moe.topk_indices,
                    topk_weights: moe.topk_weights,
                    down_expanded: moe.down_expanded,
                    ep_row_gather: None,
                },
            )
            .map_err(|e| format!("seal decode: {e:?}"))?,
            Route::Prefill {
                rows,
                ladder,
                scratch,
                moe,
                ..
            } => seal_prefill(
                bound,
                ctx,
                MoePrefillParams {
                    dtypes,
                    recipe: MoeRecipe::SigmoidRoutedNoShared,
                    route_policy: None,
                    prelude: MoePrefillPrelude {
                        normalization: MoeNormalization::Provided,
                        router: m.router.dispatch_ref(),
                        router_logits: moe.router_logits,
                        // The preallocated 2-D score view is the exact
                        // router-logits allocation, not a reshaped copy.
                        router_scores: moe.router_logits,
                        norm_topk_prob: cfg.norm_topk_prob,
                        route: PrefillRouteMode::Replicated,
                        shared: None,
                        q8_router_policy: MoeQ8RouterPolicy::FreshFp16Wmma {
                            enabled: *ladder == ProjectLadder::GemmQ8Fp16,
                            scratch,
                        },
                    },
                    batch_size: *rows,
                    mi: cfg.moe_intermediate_size,
                    down_m: cfg.hidden_size,
                    down_k: cfg.moe_intermediate_size,
                    gate_up_k: cfg.hidden_size,
                    k_top: cfg.num_experts_per_tok,
                    n_exp: cfg.num_experts,
                    m_total_max: moe.m_total_max,
                    force_mq4_grouped_fp16: false,
                    cpu_deferred_experts: false,
                    topk_indices: moe.topk_indices,
                    topk_weights: moe.topk_weights,
                    x_batch: b.h,
                    x_norm_batch: b.normed,
                    x_rot_batch: moe.x_rot,
                    expert_gate_up_ptrs: &m.expert_gate_up_ptrs,
                    expert_down_ptrs: &m.expert_down_ptrs,
                    expert_stage_ptrs: None,
                    routed_experts: m,
                    expert_down_awq_ptrs: None,
                    expert_dtype_tags: None,
                    gate_batch: moe.gate,
                    up_batch: moe.up,
                    rot_batch: moe.rot,
                    down_expanded: moe.down_expanded,
                    expert_token_counts: moe.expert_token_counts,
                    expert_offsets: moe.expert_offsets,
                    sorted_slot_index: moe.sorted_slot_index,
                    expert_tile_ids: moe.expert_tile_ids,
                    inverse_perm: moe.inverse_perm,
                    y_gate_up_grouped: moe.y_gate_up_grouped,
                    y_down_grouped: moe.y_down_grouped,
                    paro_gate_up: paro(gate_up),
                    paro_down: paro(down),
                    down_awq_scale: None,
                    routed_out: None,
                    ep_row_gather: None,
                    shared_down_out: None,
                    ep_layer_idx: 0,
                    ep_tile_exchange: None,
                    expert_iu4: None,
                    mq2_lloyd: None,
                },
            )
            .map_err(|e| format!("seal prefill: {e:?}"))?,
        };
        Ok(vec![Step::Moe(call)])
    }
}

/// Expert formats of one sealed layer; per-expert tiers only when they vary.
fn moe_dtypes(m: &MoeFfn) -> Result<MoeDtypes<'_>, String> {
    let gate_up = m.sealed.gate_up_dtypes.as_ref();
    let down = m.sealed.down_dtypes.as_ref();
    let varies = |tiers: &[DType]| {
        tiers
            .first()
            .is_some_and(|&first| tiers.iter().skip(1).any(|&dtype| dtype != first))
    };
    let (gate_up_mixed, down_mixed) = (varies(gate_up), varies(down));
    Ok(MoeDtypes {
        router: m.router.gpu_dtype,
        shared: None,
        experts_all_gate_up_mq4: gate_up.iter().all(|&dtype| dtype == DType::MQ4G256),
        routed_gate_up: *gate_up.first().ok_or("no gate/up expert dtype")?,
        routed_down: *down.first().ok_or("no down expert dtype")?,
        routed_has_mixed_experts: gate_up_mixed || down_mixed,
        has_paro_shared: m.paro_shared.is_some(),
        per_expert_gate_up: gate_up_mixed.then_some(gate_up),
        per_expert_down: down_mixed.then_some(down),
    })
}

/// Final norm of `x` (one row) + tied LM head into `logits`.
pub(crate) fn head_steps<'a>(
    cfg: &Cohere2MoeConfig,
    weights: &'a Cohere2MoeWeights,
    lm_head: &'a WeightRef<'a>,
    x: &'a GpuTensor,
    normed: &'a GpuTensor,
    logits: &'a GpuTensor,
) -> [Step<'a>; 2] {
    [
        Step::RmsnormAutomatic {
            x,
            norm_weight: &weights.final_norm,
            x_plain: normed,
            out: normed,
            awq_scale: None,
            k: cfg.hidden_size,
            eps: cfg.rms_norm_eps,
            rotation: RotationPlan::None,
            rows: 1,
        },
        Step::Gemv {
            w: lm_head,
            input: GemvInput::Raw(normed),
            out: logits,
        },
    ]
}

/// The LM head's GEMV form.
pub(crate) fn lm_head_ref(weights: &Cohere2MoeWeights) -> WeightRef<'_> {
    plain_ref(&weights.lm_head)
}
