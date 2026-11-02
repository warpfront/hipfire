// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Declarative LFM2 decoder-layer program.
//!
//! Each layer is a pre-norm mixer (LIV gated short convolution or GQA
//! attention with per-head QK-norm) followed by a pre-norm FFN (dense SwiGLU
//! or the sealed sigmoid/bias-routed MoE), then the final norm and the tied
//! LM head. This module only binds resident weights, KV/convolution state and
//! scratch to typed `hipfire_dispatch` steps; route selection, fusion and the
//! executor bodies live in dispatch.
//!
//! One program serves the single-lane decode and the continuous-batching
//! lanes ([`Lanes`]); the lanes differ only in the routes they bind: the
//! `Gemm` projection ladder with a separate residual add, per-row RoPE
//! positions and lane-isolated Q8 KV.

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{Ffn, Lfm2MoeLayerWeights, Lfm2MoeState, Lfm2MoeWeights, Mixer};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::families::moe::{
    MoeDtypes, MoeEpMode, MoeNormalization, MoeParams, MoeRecipe, MoeScores, MoeSelectionBias,
};
use hipfire_dispatch::pipeline::sandwich::{Rope, RopeKind};
use hipfire_dispatch::pipeline::{
    seal_decode, BoundMoeExperts, GatedShortConvOp, GemvInput, LaneAttentionOp, ProjectLadder,
    ProjectOp, Step,
};
use hipfire_dispatch::types::RotationPlan;
use hipfire_runtime::llama::{KvCache, KvCacheExt};
use rdna_compute::{DType, GpuTensor};

/// Which decode a program runs.
pub(crate) enum Lanes<'a> {
    /// One decode row at `position`: fused GEMV ladder, tiered KV attention,
    /// MoE scratch.
    Single {
        position: usize,
        state: &'a Lfm2MoeState,
    },
    /// `rows` independent lanes (row `r` is lane `r`): the `Gemm` projection
    /// ladder, lane-isolated Q8 KV.
    Batch {
        rows: usize,
        /// `rows` I32 positions.
        positions: &'a GpuTensor,
        kv: &'a KvCache,
        lane_capacity: usize,
        /// FWHT scratch, `[rows, max(hidden, intermediate)]`.
        rotation: &'a GpuTensor,
        /// FWHT scratch of the LM head, `[rows, hidden]`.
        lm_rotation: &'a GpuTensor,
    },
}

/// Activations, convolution state and route of one program, each `[rows, ·]`.
pub(crate) struct Bufs<'a> {
    pub h: &'a GpuTensor,
    /// Mixer-norm output; the lanes' residual-projection scratch once dead.
    pub tmp: &'a GpuTensor,
    pub ffn_in: &'a GpuTensor,
    pub head_in: &'a GpuTensor,
    pub conv_bcx: &'a GpuTensor,
    pub conv_y: &'a GpuTensor,
    pub fa_q: &'a GpuTensor,
    pub fa_k: &'a GpuTensor,
    pub fa_v: &'a GpuTensor,
    pub fa_attn_out: &'a GpuTensor,
    pub dense_gate: &'a GpuTensor,
    pub dense_up: &'a GpuTensor,
    pub dense_act: &'a GpuTensor,
    pub logits: &'a GpuTensor,
    /// Indexed by `ConvWeights::conv_state_idx`, `[rows, hidden, K - 1]` each.
    pub conv_states: &'a [GpuTensor],
    pub lanes: Lanes<'a>,
}

impl<'a> Bufs<'a> {
    /// The single-lane decode state at `position`.
    pub fn single(state: &'a Lfm2MoeState, position: usize) -> Self {
        Self {
            h: &state.h,
            tmp: &state.tmp,
            ffn_in: &state.tmp,
            head_in: &state.tmp,
            conv_bcx: &state.conv_bcx,
            conv_y: &state.conv_y,
            fa_q: &state.fa_q,
            fa_k: &state.fa_k,
            fa_v: &state.fa_v,
            fa_attn_out: &state.fa_attn_out,
            dense_gate: &state.dense_gate,
            dense_up: &state.dense_up,
            dense_act: &state.dense_act,
            logits: &state.logits,
            conv_states: &state.conv_states,
            lanes: Lanes::Single { position, state },
        }
    }

    fn rows(&self) -> usize {
        match self.lanes {
            Lanes::Single { .. } => 1,
            Lanes::Batch { rows, .. } => rows,
        }
    }

    /// `out = w · input`.
    fn project(&self, w: &'a WeightRef<'a>, input: &'a GpuTensor, out: &'a GpuTensor) -> Step<'a> {
        match self.lanes {
            Lanes::Single { .. } => Step::Gemv {
                w,
                input: GemvInput::Raw(input),
                out,
            },
            Lanes::Batch { rows, rotation, .. } => Step::Project(ProjectOp {
                weight: *w,
                input,
                output: out,
                rows,
                rotation: Some(rotation),
                ladder: ProjectLadder::Gemm,
                residual: None,
            }),
        }
    }

    /// `h += w · input`: the fused residual GEMV on one lane; on the lanes the
    /// projection lands in the dead `tmp` and is added.
    fn project_residual(&self, w: &'a WeightRef<'a>, input: &'a GpuTensor) -> Step<'a> {
        match self.lanes {
            Lanes::Single { .. } => Step::GemvResidual {
                w,
                input: GemvInput::Raw(input),
                residual: self.h,
                out: self.h,
            },
            Lanes::Batch { rows, rotation, .. } => Step::Project(ProjectOp {
                weight: *w,
                input,
                output: self.tmp,
                rows,
                rotation: Some(rotation),
                ladder: ProjectLadder::Gemm,
                residual: Some(self.h),
            }),
        }
    }

    /// RMS norm of every row of `h` into `out`, read raw by every projection.
    /// Each projection rotates its own input for its format, which keeps the
    /// per-projection kernels and reduction order of the pre-port route
    /// (prerotated inputs would admit the cross-projection fusion table).
    fn norm(&self, weight: &'a GpuTensor, out: &'a GpuTensor, cfg: &Lfm2MoeConfig) -> Step<'a> {
        Step::RmsnormAutomatic {
            x: self.h,
            norm_weight: weight,
            x_plain: out,
            out,
            awq_scale: None,
            k: cfg.hidden_size,
            eps: cfg.rms_norm_eps,
            rotation: RotationPlan::None,
            rows: self.rows(),
        }
    }
}

/// Resident weight refs of one layer, borrowed by its steps.
pub(crate) struct LayerRefs<'a> {
    layer: &'a Lfm2MoeLayerWeights,
    mixer: [Option<WeightRef<'a>>; 4],
    ffn: [Option<WeightRef<'a>>; 3],
}

impl<'a> LayerRefs<'a> {
    pub fn new(layer: &'a Lfm2MoeLayerWeights) -> Self {
        let mixer = match &layer.mixer {
            Mixer::Conv(c) => [
                Some(c.in_proj.dispatch_ref()),
                Some(c.out_proj.dispatch_ref()),
                None,
                None,
            ],
            Mixer::Attention(a) => [
                Some(a.wq.dispatch_ref()),
                Some(a.wk.dispatch_ref()),
                Some(a.wv.dispatch_ref()),
                Some(a.wo.dispatch_ref()),
            ],
        };
        let ffn = match &layer.ffn {
            Ffn::Dense(d) => [
                Some(d.w1.dispatch_ref()),
                Some(d.w3.dispatch_ref()),
                Some(d.w2.dispatch_ref()),
            ],
            Ffn::Moe(_) => [None, None, None],
        };
        Self { layer, mixer, ffn }
    }

    /// Mixer block: `h += mixer(operator_norm(h))`.
    pub fn mixer_steps(
        &'a self,
        cfg: &Lfm2MoeConfig,
        b: &'a Bufs<'a>,
    ) -> Result<Vec<Step<'a>>, String> {
        let r = |i: usize| self.mixer[i].as_ref().expect("bound mixer weight");
        let rows = b.rows();
        let norm = b.norm(&self.layer.operator_norm, b.tmp, cfg);
        Ok(match &self.layer.mixer {
            Mixer::Conv(c) => vec![
                norm,
                b.project(r(0), b.tmp, b.conv_bcx),
                Step::GatedShortConv(GatedShortConvOp {
                    bcx: b.conv_bcx,
                    state: &b.conv_states[c.conv_state_idx],
                    weight: &c.conv_weight,
                    output: b.conv_y,
                    rows,
                    channels: cfg.hidden_size,
                    kernel_size: cfg.conv_kernel_size,
                }),
                b.project_residual(r(1), b.conv_y),
            ],
            Mixer::Attention(a) => {
                let (n_heads, n_kv, head_dim) = (
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                );
                let rope = Rope {
                    kind: RopeKind::RotateHalf,
                    theta: cfg.rope_theta,
                };
                let kv = a.kv_idx;
                let mut steps = vec![
                    norm,
                    b.project(r(0), b.tmp, b.fa_q),
                    b.project(r(1), b.tmp, b.fa_k),
                    b.project(r(2), b.tmp, b.fa_v),
                    Step::QkNorm {
                        x: b.fa_q,
                        weight: &a.q_norm,
                        n_groups: rows * n_heads,
                        head_dim,
                        eps: cfg.rms_norm_eps,
                    },
                    Step::QkNorm {
                        x: b.fa_k,
                        weight: &a.k_norm,
                        n_groups: rows * n_kv,
                        head_dim,
                        eps: cfg.rms_norm_eps,
                    },
                ];
                match b.lanes {
                    Lanes::Single { position, state } => {
                        let plan = KvTierPlan::derive(KvTierInputs {
                            pos: position,
                            ..state.kv.tier_inputs()
                        })
                        .map_err(|e| format!("kv tier: {e}"))?;
                        steps.push(Step::Rope {
                            q: b.fa_q,
                            k: b.fa_k,
                            pos_buf: &state.pos_buf,
                            positions: None,
                            n_heads,
                            n_kv_heads: n_kv,
                            head_dim,
                            rope,
                        });
                        steps.push(Step::Attend {
                            plan,
                            io: AttnParams {
                                q: b.fa_q,
                                k: b.fa_k,
                                v: b.fa_v,
                                k_cache: &state.kv.k_gpu[kv],
                                v_cache: &state.kv.v_gpu[kv],
                                k_scales: None,
                                v_scales: None,
                                pos_buf: &state.pos_buf,
                                pos: position,
                                positions: None,
                                n_heads,
                                n_kv_heads: n_kv,
                                head_dim,
                                physical_cap: state.kv.physical_cap,
                                batch_size: 1,
                                max_ctx_len: 0,
                                flash_partials: None,
                                givens_cos: None,
                                givens_sin: None,
                                tree_bias: None,
                                block_start: 0,
                                block_cols: 0,
                                output_gate: None,
                                output_awq_scale: None,
                                output: b.fa_attn_out,
                            },
                        });
                    }
                    Lanes::Batch {
                        positions,
                        kv: cache,
                        lane_capacity,
                        ..
                    } => {
                        steps.push(Step::Rope {
                            q: b.fa_q,
                            k: b.fa_k,
                            pos_buf: &positions.buf,
                            positions: Some(positions),
                            n_heads,
                            n_kv_heads: n_kv,
                            head_dim,
                            rope,
                        });
                        steps.push(Step::LaneAttention(LaneAttentionOp {
                            q: b.fa_q,
                            k: b.fa_k,
                            v: b.fa_v,
                            k_cache: &cache.k_gpu[kv],
                            v_cache: &cache.v_gpu[kv],
                            positions,
                            output: b.fa_attn_out,
                            rows,
                            n_heads,
                            n_kv_heads: n_kv,
                            head_dim,
                            lane_capacity,
                            max_ctx_len: lane_capacity,
                        }));
                    }
                }
                steps.push(b.project_residual(r(3), b.fa_attn_out));
                steps
            }
        })
    }

    /// FFN block. Dense: `h += w2(silu(w1·x) ⊙ w3·x)`, `x = ffn_norm(h)`.
    /// MoE (single lane only): one sealed call (norm, sigmoid router with the
    /// expert selection bias, top-k routed experts, combine into `h`).
    pub fn ffn_steps(
        &'a self,
        cfg: &Lfm2MoeConfig,
        ctx: &'a DispatchCtx,
        b: &'a Bufs<'a>,
        layer_idx: usize,
    ) -> Result<Vec<Step<'a>>, String> {
        let m = match (&self.layer.ffn, &self.ffn) {
            (Ffn::Dense(_), [Some(w1), Some(w3), Some(w2)]) => {
                return Ok(vec![
                    b.norm(&self.layer.ffn_norm, b.ffn_in, cfg),
                    b.project(w1, b.ffn_in, b.dense_gate),
                    b.project(w3, b.ffn_in, b.dense_up),
                    Step::SiluMul {
                        gate: b.dense_gate,
                        up: b.dense_up,
                        out: b.dense_act,
                    },
                    b.project_residual(w2, b.dense_act),
                ]);
            }
            (Ffn::Moe(m), _) => m,
            (Ffn::Dense(_), _) => unreachable!("dense FFN refs are bound in LayerRefs::new"),
        };
        let Lanes::Single { state, .. } = b.lanes else {
            return Err("MoE FFN has no continuous-batching route".into());
        };
        let (gate_up, down) = (&m.experts[0].gate_up, &m.experts[0].down);
        let params = MoeParams {
            dtypes: MoeDtypes {
                router: m.router.gpu_dtype,
                shared: None,
                experts_all_gate_up_mq4: gate_up.gpu_dtype == DType::MQ4G256,
                routed_gate_up: gate_up.gpu_dtype,
                routed_down: down.gpu_dtype,
                // `MoeFfn::seal` admits only uniform-format layers.
                routed_has_mixed_experts: false,
                has_paro_shared: false,
                per_expert_gate_up: None,
                per_expert_down: None,
            },
            mq2_lloyd: None,
            recipe: MoeRecipe::SigmoidRoutedNoShared,
            route_policy: None,
            normalization: MoeNormalization::RmsNorm {
                weight: &self.layer.ffn_norm,
                plain_out: &state.ffn_tmp,
                eps: cfg.rms_norm_eps,
            },
            batch_size: 1,
            hidden: cfg.hidden_size,
            mi: cfg.moe_intermediate_size,
            k: cfg.num_experts_per_tok,
            n_exp: cfg.num_experts,
            norm_topk_prob: cfg.norm_topk_prob,
            selection_bias: Some(MoeSelectionBias {
                bias: &m.expert_bias,
                route_scale: cfg.routed_scaling_factor,
                scores: MoeScores::Sigmoid,
            }),
            x_rot_prerotated: false,
            defer_routed_combine: false,
            cpu_deferred_experts: false,
            ep_mode: MoeEpMode::None,
            layer_idx: layer_idx as u16,
            x_norm: &state.ffn_tmp,
            x_residual: &state.h,
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
            routed_gate_up_paro: None,
            routed_down_paro: None,
            router_logits: &state.router_logits,
            x_rot_local: &state.ffn_x_rot,
            // Host-routed fallback scratch: the biased router always routes on
            // the GPU, so these alias the GPU-route buffers and stay unread.
            gate_up_buf: &state.gate_batch,
            ffn_hidden: &state.up_batch,
            ffn_out: &state.down_expanded,
            gate_batch: &state.gate_batch,
            up_batch: &state.up_batch,
            rot_batch: &state.rot_batch,
            topk_indices: &state.topk_indices,
            topk_weights: &state.topk_weights,
            down_expanded: &state.down_expanded,
            ep_row_gather: None,
        };
        let bound = BoundMoeExperts::from_cache(&m.table, &m.cache)
            .map_err(|e| format!("bind experts: {e:?}"))?;
        let call = seal_decode(bound, ctx, params).map_err(|e| format!("seal MoE: {e:?}"))?;
        Ok(vec![Step::Moe(call)])
    }
}

/// Final norm + tied LM head into `logits`.
pub(crate) fn head_steps<'a>(
    cfg: &Lfm2MoeConfig,
    weights: &'a Lfm2MoeWeights,
    lm_head: &'a WeightRef<'a>,
    b: &'a Bufs<'a>,
) -> Vec<Step<'a>> {
    let head = match b.lanes {
        Lanes::Single { .. } => b.project(lm_head, b.head_in, b.logits),
        Lanes::Batch {
            rows, lm_rotation, ..
        } => Step::Project(ProjectOp {
            weight: *lm_head,
            input: b.head_in,
            output: b.logits,
            rows,
            rotation: Some(lm_rotation),
            ladder: ProjectLadder::GemmLmHead,
            residual: None,
        }),
    };
    vec![b.norm(&weights.embedding_norm, b.head_in, cfg), head]
}
