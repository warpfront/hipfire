// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative Maple-Preview decoder-layer program.
//!
//! Every layer is the pre-norm block
//! `h += o(attend(rope(qk_norm(qkv(norm(h)))))); h += moe(norm2(h))`: QK-norm
//! per head BEFORE the partial half-split RoPE (sliding layers; the full
//! layers are NoPE), GQA attention over the KV cache (sliding layers windowed,
//! full layers full causal), then the sealed softmax top-8 renormalized MoE
//! over unrotated MQ2-Lloyd (qt51) experts with the clamped SwiGLU and no
//! shared expert. This module only binds resident weights, KV state and
//! scratch to typed `hipfire_dispatch` steps; route selection and the executor
//! bodies live in dispatch.
//!
//! One program serves decode and batched prefill ([`Route`]). Decode binds
//! per-row GEMVs with fused residuals and the decode MoE seal; prefill binds
//! the qt51 projections on [`ProjectLadder::GroupedLloydF16`] (the grouped MoE
//! WMMA GEMM as one dense expert) with a separate residual add, per-row RoPE
//! positions and the prefill MoE seal.

use crate::config::{MapleConfig, MapleLayerType};
use crate::maple::{MapleLayerWeights, MapleMoeFfn, MapleWeights};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::families::moe::{
    MoeDtypes, MoeEpMode, MoeMq2Lloyd, MoeNormalization, MoeParams, MoePrefillParams,
    MoePrefillPrelude, MoeQ8RouterPolicy, MoeRecipe,
};
use hipfire_dispatch::pipeline::sandwich::{Rope, RopeKind};
use hipfire_dispatch::pipeline::sealed_moe::{
    seal_decode, seal_prefill, BoundMoeExperts, PrefillRouteMode,
};
use hipfire_dispatch::pipeline::{DenseGroupedTables, GemvInput, ProjectLadder, ProjectOp, Step};
use hipfire_dispatch::types::RotationPlan;
use hipfire_runtime::llama::{KvCache, KvCacheExt, WeightTensor};
use rdna_compute::{DType, GpuTensor};

/// Decode MoE scratch (the `MapleState` row buffers).
pub(crate) struct MoeRow<'a> {
    pub router_logits: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    /// The clamped SwiGLU rows the down GEMV reads.
    pub act: &'a GpuTensor,
    pub down_expanded: &'a GpuTensor,
    /// Capacity-only scratch of the sealed record (`[hidden]`, `[2 * mi]`,
    /// `[mi]`, `[hidden]`): the natural-basis routed-only route never reads
    /// or writes them, so the caller lends scratch that is dead in the MoE.
    pub unread: [&'a GpuTensor; 4],
}

/// Routed-MoE scratch of a prefill chunk (views of exactly its rows).
pub(crate) struct MoeRows<'a> {
    pub router_logits: &'a GpuTensor,
    /// `[rows, n_exp]` softmax output, distinct from the logits.
    pub router_scores: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    /// F16 input basis, `[rows, hidden]`.
    pub x_f16: &'a GpuTensor,
    /// Clamped SwiGLU rows `[rows * k_top, mi]` and their F16 copy.
    pub act: &'a GpuTensor,
    pub act_f16: &'a GpuTensor,
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_perm: &'a GpuTensor,
    pub y_gate_up_grouped: &'a GpuTensor,
    pub y_down_grouped: &'a GpuTensor,
    pub m_total: usize,
}

/// Which forward a program runs.
pub(crate) enum Route<'a> {
    /// One row at `position` (the device scalar in `Bufs::pos_buf`).
    Decode { position: usize, moe: MoeRow<'a> },
    /// `rows` consecutive positions `positions` (I32) ending at `last_pos`.
    Prefill {
        rows: usize,
        positions: &'a GpuTensor,
        last_pos: usize,
        /// Sliding layers attend full causal (`HIPFIRE_MAPLE_FORCE_FULL_CAUSAL`).
        full_causal: bool,
        /// Identity slot index / all-zero tile ids of the dense projections.
        dense_slot_index: &'a GpuTensor,
        dense_tile_ids: &'a GpuTensor,
        /// F16 projection inputs: `[rows, hidden]` and `[rows, q_dim]`.
        hidden_f16: &'a GpuTensor,
        attn_f16: &'a GpuTensor,
        moe: MoeRows<'a>,
    },
}

/// Activations of one program, each `[rows, ·]`.
pub(crate) struct Bufs<'a> {
    pub h: &'a GpuTensor,
    pub normed: &'a GpuTensor,
    pub fa_q: &'a GpuTensor,
    pub fa_k: &'a GpuTensor,
    pub fa_v: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    /// Prefill o-projection output before its add (unread on decode, whose
    /// residual GEMV accumulates in place).
    pub proj_out: &'a GpuTensor,
    pub pos_buf: &'a hip_bridge::DeviceBuffer,
    pub kv: &'a KvCache,
    pub flash_partials: &'a GpuTensor,
    pub route: Route<'a>,
}

/// The GEMV form `weight_gemv` launched: no PARO rotation, AWQ or row stride.
fn plain_ref(w: &WeightTensor) -> WeightRef<'_> {
    WeightRef {
        row_stride: 0,
        rotation: None,
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

    /// `out = w · input`, or `h += w · input` when `residual` (the
    /// o-projection, whose prefill input is the attention output; q/k/v read
    /// the hidden-width norm).
    fn project(
        &self,
        w: &'a WeightRef<'a>,
        expert_ptrs: &'a GpuTensor,
        input: &'a GpuTensor,
        out: &'a GpuTensor,
        residual: bool,
    ) -> Step<'a> {
        match self.route {
            Route::Decode { .. } if residual => Step::GemvResidual {
                w,
                input: GemvInput::Raw(input),
                residual: self.h,
                out: self.h,
            },
            Route::Decode { .. } => Step::Gemv {
                w,
                input: GemvInput::Raw(input),
                out,
            },
            Route::Prefill {
                rows,
                dense_slot_index,
                dense_tile_ids,
                hidden_f16,
                attn_f16,
                ..
            } => Step::Project(ProjectOp {
                weight: *w,
                input,
                output: out,
                rows,
                rotation: Some(if residual { attn_f16 } else { hidden_f16 }),
                ladder: ProjectLadder::GroupedLloydF16,
                residual: residual.then_some(self.h),
                grouped: Some(DenseGroupedTables {
                    expert_ptrs,
                    tile_ids: dense_tile_ids,
                    slot_index: dense_slot_index,
                }),
            }),
        }
    }
}

/// Resident weight refs of one layer, borrowed by its steps.
pub(crate) struct LayerRefs<'a> {
    layer: &'a MapleLayerWeights,
    /// q, k, v, o.
    attn: [WeightRef<'a>; 4],
    /// The prefill router: the F16 mirror (`None` disables batched prefill).
    router_f16: Option<WeightRef<'a>>,
}

impl<'a> LayerRefs<'a> {
    pub fn new(layer: &'a MapleLayerWeights) -> Self {
        let attn = [&layer.wq, &layer.wk, &layer.wv, &layer.wo].map(plain_ref);
        let router_f16 = layer.moe.router_f16.as_ref().map(|buf| WeightRef {
            buf,
            dtype: DType::F16,
            m: layer.moe.router.m,
            k: layer.moe.router.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        });
        Self {
            layer,
            attn,
            router_f16,
        }
    }

    /// Attention branch: `h += o(attend(rope(qk_norm(qkv(norm(h))))))`.
    pub fn attention_steps(
        &'a self,
        cfg: &MapleConfig,
        b: &'a Bufs<'a>,
        layer_idx: usize,
    ) -> Result<Vec<Step<'a>>, String> {
        let [wq, wk, wv, wo] = &self.attn;
        let ptrs = &self.layer.attn_ptr_tables;
        let (n_heads, n_kv, head_dim) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let rows = b.rows();
        // Partial half-split RoPE (pairs within the rotated prefix, frequency
        // denominator `rotary_dim`) on sliding layers; full layers are NoPE.
        let rope = Rope {
            kind: if cfg.applies_rope(layer_idx) {
                RopeKind::PrefixHalfSplit {
                    rot_dim: cfg.rotary_dim(),
                }
            } else {
                RopeKind::NoPe
            },
            theta: cfg.rope_theta,
        };
        let (pos, positions, batch_size, max_ctx_len, full_causal) = match b.route {
            Route::Decode { position, .. } => (position, None, 1, 0, false),
            Route::Prefill {
                rows,
                positions,
                last_pos,
                full_causal,
                ..
            } => (last_pos, Some(positions), rows, last_pos + 1, full_causal),
        };
        // Sliding layers clip to the last `sliding_window` keys; full layers
        // are full causal. The windowed Q8 kernels are taken unconditionally:
        // the non-windowed keys would attend the whole context past the window.
        // Under `--kv-mode bf16` the Bf16 tier is windowed in both shapes.
        let window = if cfg.layer_type(layer_idx) == MapleLayerType::Sliding && !full_causal {
            cfg.sliding_window as i32
        } else {
            0
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
                rows,
            },
            b.project(wq, &ptrs.wq, b.normed, b.fa_q, false),
            b.project(wk, &ptrs.wk, b.normed, b.fa_k, false),
            b.project(wv, &ptrs.wv, b.normed, b.fa_v, false),
            // QK-norm BEFORE RoPE: both are in place, so the order is the semantics.
            Step::QkNorm {
                x: b.fa_q,
                weight: &self.layer.q_norm,
                n_groups: n_heads * rows,
                head_dim,
                eps: cfg.rms_norm_eps,
            },
            Step::QkNorm {
                x: b.fa_k,
                weight: &self.layer.k_norm,
                n_groups: n_kv * rows,
                head_dim,
                eps: cfg.rms_norm_eps,
            },
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
            b.project(wo, &ptrs.wo, b.attn_out, b.proj_out, true),
        ])
    }

    /// MoE branch over the post-attention residual: `h += moe(norm2(h))`.
    pub fn moe_steps(
        &'a self,
        cfg: &MapleConfig,
        ctx: &'a DispatchCtx,
        b: &'a Bufs<'a>,
        layer_idx: usize,
    ) -> Result<Vec<Step<'a>>, String> {
        let m: &'a MapleMoeFfn = &self.layer.moe;
        let bound = BoundMoeExperts::from_cache(&m.table, &m.cache)
            .map_err(|e| format!("bind experts: {e:?}"))?;
        let expert = m.experts.first().ok_or("MoE layer has no experts")?;
        let dtypes = |router: DType| MoeDtypes {
            router,
            shared: None,
            experts_all_gate_up_mq4: false,
            routed_gate_up: expert.gate_up.gpu_dtype,
            routed_down: expert.down.gpu_dtype,
            routed_has_mixed_experts: false,
            has_paro_shared: false,
            per_expert_gate_up: None,
            per_expert_down: None,
        };
        // Clamped SwiGLU `silu(min(gate, L)) * clamp(up, -L, L)`; the expanded
        // down + k8 combine (HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC=0: the
        // self-combining residual down).
        let lloyd = MoeMq2Lloyd {
            swiglu_limit: cfg.swiglu_clamp,
            uses_atomic_down: false,
            native: None,
            nonowned_gate_up_dummy: None,
        };
        let norm = Step::RmsnormAutomatic {
            x: b.h,
            norm_weight: &self.layer.post_attn_norm,
            x_plain: b.normed,
            out: b.normed,
            awq_scale: None,
            k: cfg.hidden_size,
            eps: cfg.rms_norm_eps,
            rotation: RotationPlan::None,
            rows: b.rows(),
        };
        let call = match &b.route {
            Route::Decode { moe, .. } => seal_decode(
                bound,
                ctx,
                MoeParams {
                    dtypes: dtypes(m.router.gpu_dtype),
                    mq2_lloyd: Some(lloyd),
                    recipe: MoeRecipe::SoftmaxRoutedNoShared,
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
                    router: plain_ref(&m.router),
                    shared: None,
                    expert_gate_up_ptrs: &m.expert_gate_up_ptrs,
                    expert_down_ptrs: &m.expert_down_ptrs,
                    expert_ptrs_host: None,
                    expert_down_awq_ptrs: None,
                    expert_dtype_tags: None,
                    routed_gate_up_k: expert.gate_up.k,
                    routed_down_m: expert.down.m,
                    routed_down_k: expert.down.k,
                    routed_experts: m,
                    routed_gate_up_paro: None,
                    routed_down_paro: None,
                    router_logits: moe.router_logits,
                    x_rot_local: moe.unread[0],
                    gate_up_buf: moe.unread[1],
                    ffn_hidden: moe.unread[2],
                    ffn_out: moe.unread[3],
                    gate_batch: moe.gate,
                    up_batch: moe.up,
                    rot_batch: moe.act,
                    topk_indices: moe.topk_indices,
                    topk_weights: moe.topk_weights,
                    down_expanded: moe.down_expanded,
                    ep_row_gather: None,
                },
            )
            .map_err(|e| format!("seal decode: {e:?}"))?,
            Route::Prefill { rows, moe, .. } => seal_prefill(
                bound,
                ctx,
                MoePrefillParams {
                    dtypes: dtypes(DType::F16),
                    recipe: MoeRecipe::SoftmaxRoutedNoShared,
                    route_policy: None,
                    prelude: MoePrefillPrelude {
                        normalization: MoeNormalization::Provided,
                        router: self
                            .router_f16
                            .ok_or("router has no F16 mirror: batched MoE unavailable")?,
                        router_logits: moe.router_logits,
                        router_scores: moe.router_scores,
                        norm_topk_prob: cfg.norm_topk_prob,
                        route: PrefillRouteMode::Replicated,
                        shared: None,
                        q8_router_policy: MoeQ8RouterPolicy::DispatcherEntry,
                    },
                    batch_size: *rows,
                    mi: cfg.moe_intermediate_size,
                    down_m: cfg.hidden_size,
                    down_k: cfg.moe_intermediate_size,
                    gate_up_k: cfg.hidden_size,
                    k_top: cfg.num_experts_per_tok,
                    n_exp: cfg.num_experts,
                    m_total_max: moe.m_total,
                    force_mq4_grouped_fp16: false,
                    cpu_deferred_experts: false,
                    topk_indices: moe.topk_indices,
                    topk_weights: moe.topk_weights,
                    x_batch: b.h,
                    x_norm_batch: b.normed,
                    x_rot_batch: moe.x_f16,
                    expert_gate_up_ptrs: &m.expert_gate_up_ptrs,
                    expert_down_ptrs: &m.expert_down_ptrs,
                    expert_stage_ptrs: None,
                    routed_experts: m,
                    expert_down_awq_ptrs: None,
                    expert_dtype_tags: None,
                    // The fused unscatter writes only the gate rows.
                    gate_batch: moe.act,
                    up_batch: moe.act,
                    rot_batch: moe.act_f16,
                    // Read only by the expanded (non-grouped / EP) downs; the
                    // grouped route writes `y_down_grouped`, of ample capacity.
                    down_expanded: moe.y_down_grouped,
                    expert_token_counts: moe.expert_token_counts,
                    expert_offsets: moe.expert_offsets,
                    sorted_slot_index: moe.sorted_slot_index,
                    expert_tile_ids: moe.expert_tile_ids,
                    inverse_perm: moe.inverse_perm,
                    y_gate_up_grouped: moe.y_gate_up_grouped,
                    y_down_grouped: moe.y_down_grouped,
                    paro_gate_up: None,
                    paro_down: None,
                    down_awq_scale: None,
                    routed_out: None,
                    ep_row_gather: None,
                    shared_down_out: None,
                    ep_layer_idx: 0,
                    ep_tile_exchange: None,
                    expert_iu4: None,
                    mq2_lloyd: Some(lloyd),
                },
            )
            .map_err(|e| format!("seal prefill: {e:?}"))?,
        };
        Ok(vec![norm, Step::Moe(call)])
    }
}

/// Final norm of `x` (one row) + untied LM head into `logits`.
pub(crate) fn head_steps<'a>(
    cfg: &MapleConfig,
    weights: &'a MapleWeights,
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
pub(crate) fn lm_head_ref(weights: &MapleWeights) -> WeightRef<'_> {
    plain_ref(&weights.lm_head)
}
