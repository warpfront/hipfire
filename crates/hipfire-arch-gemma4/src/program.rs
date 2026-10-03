// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative Gemma 4 decoder-layer program.
//!
//! Each layer is
//! `[SandwichAttention, SandwichMlp | ParallelMoeMlp, PerLayerInput?, Scale?]`:
//! MoE checkpoints run the parallel dense+expert block, E-series checkpoints
//! add the per-layer-input branch, and the scale step exists when the learned
//! layer scalar is not 1. This module only binds resident weights, KV storage
//! and scratch to the shared `hipfire_dispatch::pipeline::sandwich`
//! operations; route selection and executor bodies live in dispatch. The same
//! program serves single-token decode (`rows == 1`) and batched prefill /
//! verify (`rows > 1`), for both weight stacks (`gemma4` and `lowered`).

use crate::config::{Gemma4Config, LayerType, RopeType};
use crate::gemma4::{self, PerLayerBranchWeights};
use crate::lowered;
use hip_bridge::DeviceBuffer;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::sandwich::{
    Activation, KvProjection, ParallelMoeMlpOp, PerLayerInputOp, Rope, RopeKind, RoutedExperts,
    RoutedScratch, SandwichAttentionOp, SandwichKv, SandwichMlpOp, SandwichStream, ScaleOp,
    SoftcapOp,
};
use hipfire_dispatch::pipeline::{GemvInput, Step};
use hipfire_dispatch::types::RotationPlan;
use hipfire_runtime::llama::{KvCache, KvCacheExt, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Attention geometry of one layer type.
#[derive(Clone, Copy)]
pub(crate) struct AttnGeometry {
    pub head_dim: usize,
    pub n_kv_heads: usize,
    /// Sliding window; `0` = full causal.
    pub window: usize,
    pub rope: Rope,
}

/// Model geometry the program reads; built from either config type.
#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    pub dim: usize,
    pub eps: f32,
    pub n_heads: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub softcap: f32,
    /// Per-layer-input width (E-series); `0` without PLE.
    pub ple_width: usize,
    /// Routed experts per token and per-expert FFN width (MoE checkpoints).
    pub moe_top_k: usize,
    pub moe_hidden: usize,
    pub sliding: AttnGeometry,
    pub full: AttnGeometry,
}

fn full_rope(head_dim: usize, rope_type: RopeType, factor: f32, theta: f32) -> Rope {
    let rot_pairs = match rope_type {
        RopeType::Proportional => ((head_dim as f32) * factor * 0.5) as usize,
        RopeType::Default => head_dim / 2,
    };
    Rope {
        kind: RopeKind::PartialHalved { rot_pairs },
        theta,
    }
}

impl Geometry {
    pub fn eager(cfg: &Gemma4Config) -> Self {
        Self {
            dim: cfg.dim,
            eps: cfg.norm_eps,
            n_heads: cfg.n_heads,
            n_layers: cfg.n_layers,
            vocab: cfg.vocab_size,
            softcap: cfg.final_logit_softcapping,
            ple_width: cfg.hidden_size_per_layer_input,
            moe_top_k: 0,
            moe_hidden: 0,
            sliding: AttnGeometry {
                head_dim: cfg.sliding_head_dim,
                n_kv_heads: cfg.sliding_n_kv_heads,
                window: cfg.sliding_window,
                rope: Rope {
                    kind: RopeKind::RotateHalf,
                    theta: cfg.sliding_rope_theta,
                },
            },
            full: AttnGeometry {
                head_dim: cfg.full_head_dim,
                n_kv_heads: cfg.full_n_kv_heads,
                window: 0,
                rope: full_rope(
                    cfg.full_head_dim,
                    cfg.full_rope_type,
                    cfg.full_partial_rotary_factor,
                    cfg.full_rope_theta,
                ),
            },
        }
    }

    /// The arch-22 draft head's blocks (no softcap; the head is bound
    /// separately).
    pub fn drafter(cfg: &crate::drafter::Gemma4DrafterConfig) -> Self {
        Self {
            dim: cfg.hidden,
            eps: cfg.norm_eps,
            n_heads: cfg.n_heads,
            n_layers: cfg.n_layers,
            vocab: cfg.vocab_size,
            softcap: 0.0,
            ple_width: 0,
            moe_top_k: 0,
            moe_hidden: 0,
            sliding: AttnGeometry {
                head_dim: cfg.sliding_head_dim,
                n_kv_heads: cfg.sliding_n_kv_heads,
                window: cfg.sliding_window,
                rope: Rope {
                    kind: RopeKind::RotateHalf,
                    theta: cfg.sliding_rope_theta,
                },
            },
            full: AttnGeometry {
                head_dim: cfg.full_head_dim,
                n_kv_heads: cfg.full_n_kv_heads,
                window: 0,
                rope: full_rope(
                    cfg.full_head_dim,
                    cfg.full_rope_type,
                    cfg.full_partial_rotary_factor,
                    cfg.full_rope_theta,
                ),
            },
        }
    }

    pub fn lowered(cfg: &lowered::Gemma4Config) -> Self {
        let rope_type = match cfg.full_rope_type {
            lowered::RopeType::Default => RopeType::Default,
            lowered::RopeType::Proportional => RopeType::Proportional,
        };
        Self {
            dim: cfg.dim,
            eps: cfg.norm_eps,
            n_heads: cfg.n_heads,
            n_layers: cfg.n_layers,
            vocab: cfg.vocab_size,
            softcap: cfg.final_logit_softcapping,
            ple_width: 0,
            moe_top_k: cfg.top_k_experts,
            moe_hidden: cfg.moe_intermediate_size,
            sliding: AttnGeometry {
                head_dim: cfg.sliding_head_dim,
                n_kv_heads: cfg.sliding_n_kv_heads,
                window: cfg.sliding_window,
                rope: Rope {
                    kind: RopeKind::RotateHalf,
                    theta: cfg.sliding_rope_theta,
                },
            },
            full: AttnGeometry {
                head_dim: cfg.full_head_dim,
                n_kv_heads: cfg.full_n_kv_heads,
                window: 0,
                rope: full_rope(
                    cfg.full_head_dim,
                    rope_type,
                    cfg.full_partial_rotary_factor,
                    cfg.full_rope_theta,
                ),
            },
        }
    }
}

#[derive(Clone, Copy)]
struct LayerKvWeights<'a> {
    k: &'a WeightTensor,
    v: Option<&'a WeightTensor>,
    k_norm: &'a GpuTensor,
}

/// Per-layer weights the program binds, from any Gemma 4 weight stack.
pub(crate) struct LayerRefs<'a> {
    layer_type: LayerType,
    input_norm: &'a GpuTensor,
    post_attn_norm: &'a GpuTensor,
    pre_ffn_norm: &'a GpuTensor,
    post_ffn_norm: &'a GpuTensor,
    layer_scalar: f32,
    q: &'a WeightTensor,
    q_norm: &'a GpuTensor,
    /// Own K projection, V projection (`None` = K=V) and K norm; `None` on
    /// query-only layers that read another layer's cache.
    kv: Option<LayerKvWeights<'a>>,
    o: &'a WeightTensor,
    gate: &'a WeightTensor,
    up: &'a WeightTensor,
    down: &'a WeightTensor,
    ffn_hidden_dim: usize,
    per_layer: Option<&'a PerLayerBranchWeights>,
    moe: Option<&'a lowered::MoeLayerExtras>,
}

impl<'a> LayerRefs<'a> {
    pub fn eager(layer: &'a gemma4::LayerWeights) -> Self {
        match layer {
            gemma4::LayerWeights::Sliding(l) => Self {
                layer_type: LayerType::Sliding,
                input_norm: &l.input_layernorm,
                post_attn_norm: &l.post_attention_layernorm,
                pre_ffn_norm: &l.pre_feedforward_layernorm,
                post_ffn_norm: &l.post_feedforward_layernorm,
                layer_scalar: l.layer_scalar_host,
                q: &l.q_proj,
                q_norm: &l.q_norm,
                kv: Some(LayerKvWeights {
                    k: &l.k_proj,
                    v: Some(&l.v_proj),
                    k_norm: &l.k_norm,
                }),
                o: &l.o_proj,
                gate: &l.gate_proj,
                up: &l.up_proj,
                down: &l.down_proj,
                ffn_hidden_dim: l.ffn_hidden_dim,
                per_layer: l.per_layer.as_ref(),
                moe: None,
            },
            gemma4::LayerWeights::Full(l) => Self {
                layer_type: LayerType::Full,
                input_norm: &l.input_layernorm,
                post_attn_norm: &l.post_attention_layernorm,
                pre_ffn_norm: &l.pre_feedforward_layernorm,
                post_ffn_norm: &l.post_feedforward_layernorm,
                layer_scalar: l.layer_scalar_host,
                q: &l.q_proj,
                q_norm: &l.q_norm,
                kv: Some(LayerKvWeights {
                    k: &l.k_proj,
                    v: l.v_proj.as_ref(),
                    k_norm: &l.k_norm,
                }),
                o: &l.o_proj,
                gate: &l.gate_proj,
                up: &l.up_proj,
                down: &l.down_proj,
                ffn_hidden_dim: l.ffn_hidden_dim,
                per_layer: l.per_layer.as_ref(),
                moe: None,
            },
        }
    }

    /// `hidden_dim` is the lowered config's dense FFN width.
    pub fn lowered(layer: &'a lowered::LayerWeights, hidden_dim: usize) -> Self {
        match layer {
            lowered::LayerWeights::Sliding(l) => Self {
                layer_type: LayerType::Sliding,
                input_norm: &l.input_layernorm,
                post_attn_norm: &l.post_attention_layernorm,
                pre_ffn_norm: &l.pre_feedforward_layernorm,
                post_ffn_norm: &l.post_feedforward_layernorm,
                layer_scalar: l.layer_scalar_host,
                q: &l.q_proj,
                q_norm: &l.q_norm,
                kv: Some(LayerKvWeights {
                    k: &l.k_proj,
                    v: Some(&l.v_proj),
                    k_norm: &l.k_norm,
                }),
                o: &l.o_proj,
                gate: &l.gate_proj,
                up: &l.up_proj,
                down: &l.down_proj,
                ffn_hidden_dim: hidden_dim,
                per_layer: None,
                moe: l.moe.as_ref(),
            },
            lowered::LayerWeights::Full(l) => Self {
                layer_type: LayerType::Full,
                input_norm: &l.input_layernorm,
                post_attn_norm: &l.post_attention_layernorm,
                pre_ffn_norm: &l.pre_feedforward_layernorm,
                post_ffn_norm: &l.post_feedforward_layernorm,
                layer_scalar: l.layer_scalar_host,
                q: &l.q_proj,
                q_norm: &l.q_norm,
                kv: Some(LayerKvWeights {
                    k: &l.k_proj,
                    v: None,
                    k_norm: &l.k_norm,
                }),
                o: &l.o_proj,
                gate: &l.gate_proj,
                up: &l.up_proj,
                down: &l.down_proj,
                ffn_hidden_dim: hidden_dim,
                per_layer: None,
                moe: l.moe.as_ref(),
            },
        }
    }
}

impl<'a> LayerRefs<'a> {
    /// A draft-head block: query-only attention over the target's cache.
    pub fn drafter(
        layer: &'a crate::drafter::DrafterLayerWeights,
        layer_type: LayerType,
        hidden_dim: usize,
    ) -> Self {
        Self {
            layer_type,
            input_norm: &layer.input_layernorm,
            post_attn_norm: &layer.post_attention_layernorm,
            pre_ffn_norm: &layer.pre_feedforward_layernorm,
            post_ffn_norm: &layer.post_feedforward_layernorm,
            layer_scalar: layer.layer_scalar_host,
            q: &layer.q_proj,
            q_norm: &layer.q_norm,
            kv: None,
            o: &layer.o_proj,
            gate: &layer.gate_proj,
            up: &layer.up_proj,
            down: &layer.down_proj,
            ffn_hidden_dim: hidden_dim,
            per_layer: None,
            moe: None,
        }
    }
}

/// Resident buffers every layer of a forward shares.
pub(crate) struct Resident<'a> {
    pub kv_sliding: &'a KvCache,
    pub kv_full: &'a KvCache,
    /// Device position scalar (`rows == 1`).
    pub pos_buf: &'a DeviceBuffer,
    /// Ones vector of the larger head dim (weight-less V norm).
    pub v_norm_ones: &'a GpuTensor,
    pub flash_partials: &'a GpuTensor,
}

/// Which cache slot a layer attends, and whether it writes its own row.
#[derive(Clone, Copy)]
pub(crate) struct LayerKv {
    pub slot: usize,
    pub writes: bool,
}

/// Per-layer-input scratch; present on E-series checkpoints.
pub(crate) struct PleScratch<'a> {
    /// `[rows, n_layers * width]` per-layer inputs of every row.
    pub inputs: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub out: &'a GpuTensor,
}

/// Activation scratch of one forward, `rows` rows each.
pub(crate) struct LayerScratch<'a> {
    pub x: &'a GpuTensor,
    pub residual: &'a GpuTensor,
    pub normed: &'a GpuTensor,
    pub attn_rot: &'a GpuTensor,
    pub mlp_rot: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub mlp_out: &'a GpuTensor,
    pub ple: Option<PleScratch<'a>>,
    pub moe: Option<RoutedScratch<'a>>,
}

impl<'a> LayerScratch<'a> {
    /// The resident single-row scratch of the eager state.
    pub fn eager(state: &'a gemma4::Gemma4State) -> Self {
        let ple = match (
            &state.ple_projection_all,
            &state.ple_gate,
            &state.ple_hidden,
            &state.ple_out,
        ) {
            (Some(inputs), Some(gate), Some(act), Some(out)) => Some(PleScratch {
                inputs,
                gate,
                act,
                out,
            }),
            _ => None,
        };
        Self {
            x: &state.x,
            residual: &state.residual,
            normed: &state.tmp,
            attn_rot: &state.tmp_rot,
            mlp_rot: &state.tmp_rot,
            q: &state.q,
            k: &state.k,
            v: &state.v,
            attn_out: &state.attn_out,
            gate: &state.gate_ffn,
            up: &state.up_ffn,
            act: &state.ffn_hidden,
            mlp_out: &state.ffn_out,
            ple,
            moe: None,
        }
    }

    /// The resident single-row scratch of the lowered state.
    pub fn lowered(s: &'a lowered::Gemma4Scratch, moe: bool) -> Self {
        Self {
            x: &s.x,
            residual: &s.residual,
            normed: &s.tmp,
            attn_rot: &s.x_rot,
            mlp_rot: &s.x_rot,
            q: &s.q,
            k: &s.k,
            v: &s.v,
            attn_out: &s.attn_out,
            gate: &s.gate_ffn,
            up: &s.up_ffn,
            act: &s.ffn_hidden,
            mlp_out: &s.ffn_out,
            ple: None,
            moe: moe.then(|| routed_scratch(s)),
        }
    }
}

/// Flash-attention partials for `rows` query rows over a Q8 cache of
/// `max_seq` positions, sized for the larger attention geometry.
pub(crate) fn q8_flash_partials_len(
    gpu: &Gpu,
    geo: &Geometry,
    max_seq: usize,
    rows: usize,
) -> usize {
    [geo.sliding, geo.full]
        .into_iter()
        .map(|a| {
            let tile = rdna_compute::attention::q8_flash_tile_size(
                &gpu.arch,
                geo.n_heads,
                a.n_kv_heads,
                a.head_dim,
                max_seq,
            );
            rows * geo.n_heads * max_seq.div_ceil(tile) * (2 + a.head_dim)
        })
        .max()
        .unwrap_or(0)
}

/// The lowered state's single-row routed-expert scratch; batched forwards
/// route row by row through it.
pub(crate) fn routed_scratch(s: &lowered::Gemma4Scratch) -> RoutedScratch<'_> {
    RoutedScratch {
        input: &s.moe_pre2,
        input_rot: &s.moe_pre2_rot,
        router_in: &s.moe_router_in,
        router_logits: &s.moe_router_logits,
        topk_indices: &s.moe_topk_indices,
        topk_weights: &s.moe_topk_weights,
        gate: &s.moe_expert_gate_batch,
        up: &s.moe_expert_up_batch,
        act: &s.moe_expert_hidden_batch,
        out: &s.moe_cur_moe,
        dense_normed: &s.moe_cur_mlp,
    }
}

/// Owner of one batched forward's buffers; frees them on drop.
pub(crate) struct RowBuffers {
    gpu: *mut Gpu,
    tensors: Vec<GpuTensor>,
}

impl RowBuffers {
    pub fn new(gpu: &mut Gpu) -> Self {
        Self {
            gpu,
            tensors: Vec::with_capacity(24),
        }
    }

    /// An F32 `[n]` buffer owned until drop; the returned view aliases it.
    pub fn alloc(&mut self, n: usize, label: &str) -> Result<GpuTensor, String> {
        // SAFETY: a RowBuffers lives inside one forward call and the Gpu
        // reference it was built from outlives it.
        let gpu = unsafe { &mut *self.gpu };
        let tensor = gpu
            .alloc_tensor(&[n], DType::F32)
            .map_err(|e| format!("gemma4 batch alloc {label}: {e:?}"))?;
        let view = GpuTensor {
            // SAFETY: the owning tensor stays in this ledger until every view
            // has dropped and the ledger frees it exactly once.
            buf: unsafe { tensor.buf.alias() },
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
        };
        self.tensors.push(tensor);
        Ok(view)
    }
}

impl Drop for RowBuffers {
    fn drop(&mut self) {
        // SAFETY: see new().
        let gpu = unsafe { &mut *self.gpu };
        for tensor in self.tensors.drain(..) {
            gpu.free_tensor(tensor).ok();
        }
    }
}

/// Activations of one `rows`-row forward at positions
/// `[start_pos, start_pos + rows)`.
pub(crate) struct RowActivations {
    pub x: GpuTensor,
    residual: GpuTensor,
    normed: GpuTensor,
    /// FWHT scratch for attention projections (`o_proj` reads `max_q`).
    pub attn_rot: GpuTensor,
    mlp_rot: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    attn_out: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    act: GpuTensor,
    mlp_out: GpuTensor,
    /// `[rows]` i32 absolute positions.
    pub positions: GpuTensor,
}

/// Widest activation rows a layer program touches.
#[derive(Clone, Copy)]
pub(crate) struct RowWidths {
    pub dim: usize,
    pub max_q: usize,
    pub max_kv: usize,
    pub ffn: usize,
}

impl RowActivations {
    pub fn new(
        bufs: &mut RowBuffers,
        gpu: &mut Gpu,
        rows: usize,
        start_pos: usize,
        w: RowWidths,
    ) -> Result<Self, String> {
        let mut alloc = |n: usize, label: &str| bufs.alloc(rows * n, label);
        let acts = Self {
            x: alloc(w.dim, "x")?,
            residual: alloc(w.dim, "residual")?,
            normed: alloc(w.dim, "normed")?,
            attn_rot: alloc(w.max_q.max(w.dim), "attn_rot")?,
            mlp_rot: alloc(w.ffn.max(w.dim), "mlp_rot")?,
            q: alloc(w.max_q, "q")?,
            k: alloc(w.max_kv, "k")?,
            v: alloc(w.max_kv, "v")?,
            attn_out: alloc(w.max_q, "attn_out")?,
            gate: alloc(w.ffn, "gate")?,
            up: alloc(w.ffn, "up")?,
            act: alloc(w.ffn, "act")?,
            mlp_out: alloc(w.dim, "mlp_out")?,
            positions: alloc(1, "positions")?,
        };
        let bytes: Vec<u8> = (start_pos..start_pos + rows)
            .flat_map(|p| (p as i32).to_ne_bytes())
            .collect();
        gpu.hip
            .memcpy_htod(&acts.positions.buf, &bytes)
            .map_err(|e| format!("gemma4 batch htod positions: {e:?}"))?;
        Ok(acts)
    }

    pub fn scratch<'a>(
        &'a self,
        ple: Option<PleScratch<'a>>,
        moe: Option<RoutedScratch<'a>>,
    ) -> LayerScratch<'a> {
        LayerScratch {
            x: &self.x,
            residual: &self.residual,
            normed: &self.normed,
            attn_rot: &self.attn_rot,
            mlp_rot: &self.mlp_rot,
            q: &self.q,
            k: &self.k,
            v: &self.v,
            attn_out: &self.attn_out,
            gate: &self.gate,
            up: &self.up,
            act: &self.act,
            mlp_out: &self.mlp_out,
            ple,
            moe,
        }
    }
}

/// Binding of one forward over `rows` consecutive positions from `position`.
pub(crate) struct ProgramBinding<'a> {
    pub geo: Geometry,
    pub resident: Resident<'a>,
    pub rows: usize,
    pub position: usize,
    /// Per-row i32 positions (`rows > 1`).
    pub positions: Option<&'a GpuTensor>,
    pub scratch: LayerScratch<'a>,
}

impl<'a> ProgramBinding<'a> {
    fn stream(&self) -> SandwichStream<'a> {
        SandwichStream {
            rows: self.rows,
            hidden: self.geo.dim,
            eps: self.geo.eps,
            x: self.scratch.x,
            residual: self.scratch.residual,
            normed: self.scratch.normed,
        }
    }

    /// Append the steps of `layer_idx` to `steps`. A layer writes its own
    /// cache rows iff it has K/V weights and `kv_slot.writes`.
    pub fn layer(
        &self,
        layer_idx: usize,
        w: LayerRefs<'a>,
        kv_slot: LayerKv,
        steps: &mut Vec<Step<'a>>,
    ) -> Result<(), String> {
        let geo = &self.geo;
        let r = &self.resident;
        let sc = &self.scratch;
        let (kv, attn): (&KvCache, _) = match w.layer_type {
            LayerType::Sliding => (r.kv_sliding, geo.sliding),
            LayerType::Full => (r.kv_full, geo.full),
        };
        let kv_weights = w.kv.filter(|_| kv_slot.writes);
        steps.push(Step::SandwichAttention(SandwichAttentionOp {
            stream: self.stream(),
            position: self.position,
            n_heads: geo.n_heads,
            n_kv_heads: attn.n_kv_heads,
            head_dim: attn.head_dim,
            input_norm: w.input_norm,
            wq: w.q.dispatch_ref(),
            q_norm: w.q_norm,
            kv_proj: kv_weights.map(|kv| KvProjection {
                wk: kv.k.dispatch_ref(),
                wv: kv.v.map(WeightTensor::dispatch_ref),
                k_norm: kv.k_norm,
                v_norm: Some(r.v_norm_ones),
            }),
            q_scale: (attn.head_dim as f32).sqrt(),
            rope: attn.rope,
            kv: SandwichKv {
                tier: kv.tier_inputs(),
                k_cache: &kv.k_gpu[kv_slot.slot],
                v_cache: &kv.v_gpu[kv_slot.slot],
                physical_cap: kv.physical_cap,
                givens_cos: kv.givens_cos.as_ref(),
                givens_sin: kv.givens_sin.as_ref(),
                window: attn.window,
            },
            pos_buf: r.pos_buf,
            positions: self.positions,
            wo: w.o.dispatch_ref(),
            post_norm: w.post_attn_norm,
            x_rot: sc.attn_rot,
            q: sc.q,
            k: sc.k,
            v: sc.v,
            attn_out: sc.attn_out,
            flash_partials: r.flash_partials,
        }));
        let mlp = SandwichMlpOp {
            stream: self.stream(),
            pre_norm: w.pre_ffn_norm,
            w_gate: w.gate.dispatch_ref(),
            w_up: w.up.dispatch_ref(),
            w_down: w.down.dispatch_ref(),
            activation: Activation::GeluTanh,
            hidden_dim: w.ffn_hidden_dim,
            post_norm: w.post_ffn_norm,
            x_rot: sc.mlp_rot,
            gate: sc.gate,
            up: sc.up,
            act: sc.act,
            out: sc.mlp_out,
        };
        match w.moe {
            None => steps.push(Step::SandwichMlp(mlp)),
            Some(moe) => {
                let scratch = sc
                    .moe
                    .as_ref()
                    .ok_or_else(|| format!("gemma4 layer {layer_idx}: missing MoE scratch"))?;
                let expert = moe
                    .experts
                    .first()
                    .ok_or_else(|| format!("gemma4 layer {layer_idx}: MoE layer has no experts"))?;
                steps.push(Step::ParallelMoeMlp(ParallelMoeMlpOp {
                    mlp,
                    dense_post_norm: &moe.post_feedforward_layernorm_1,
                    experts: RoutedExperts {
                        pre_norm: &moe.pre_feedforward_layernorm_2,
                        router_norm: &moe.router_scale,
                        router_input_scale: 1.0 / (geo.dim as f32).sqrt(),
                        router: moe.router_proj.dispatch_ref(),
                        n_experts: moe.experts.len(),
                        top_k: geo.moe_top_k,
                        hidden_dim: geo.moe_hidden,
                        gate_up_dtype: expert.gate_up_proj.gpu_dtype,
                        down_dtype: expert.down_proj.gpu_dtype,
                        gate_up_ptrs: &moe.experts_gate_up_ptrs,
                        down_ptrs: &moe.experts_down_ptrs,
                        per_expert_scale: &moe.per_expert_scale,
                        post_norm: &moe.post_feedforward_layernorm_2,
                    },
                    scratch: RoutedScratch { ..*scratch },
                }));
            }
        }
        if let Some(ple) = w.per_layer.filter(|_| geo.ple_width != 0) {
            let s = sc
                .ple
                .as_ref()
                .ok_or_else(|| format!("gemma4 layer {layer_idx}: missing PLE scratch"))?;
            steps.push(Step::PerLayerInput(PerLayerInputOp {
                stream: self.stream(),
                layer: layer_idx,
                layer_width: geo.ple_width,
                n_layers: geo.n_layers,
                inputs: s.inputs,
                w_gate: ple.input_gate.dispatch_ref(),
                w_proj: ple.projection.dispatch_ref(),
                post_norm: &ple.post_input_norm,
                gate: s.gate,
                act: s.act,
                out: s.out,
            }));
        }
        if w.layer_scalar != 1.0 {
            steps.push(Step::Scale(ScaleOp {
                x: sc.x,
                factor: w.layer_scalar,
            }));
        }
        Ok(())
    }
}

/// KV slot and write ownership of every eager layer (E-series KV sharing).
pub(crate) fn eager_layer_kv(
    cfg: &Gemma4Config,
    state: &gemma4::Gemma4State,
    layer_idx: usize,
) -> Result<LayerKv, String> {
    match cfg.kv_shared_source_layer_idx(layer_idx) {
        Some(source) => Ok(LayerKv {
            slot: state.kv_slot_for_layer[source],
            writes: false,
        }),
        None if !cfg.is_kv_shared_layer(layer_idx) => Ok(LayerKv {
            slot: state.kv_slot_for_layer[layer_idx],
            writes: true,
        }),
        None => Err(format!(
            "gemma4 layer {layer_idx}: missing same-type KV sharing source"
        )),
    }
}

/// Final norm, tied LM head and logit soft cap of one row `x` into `logits`.
/// `lm_head` is the caller-held `weights.lm_head.dispatch_ref()`.
pub(crate) fn head<'a>(
    geo: &Geometry,
    final_norm: &'a GpuTensor,
    lm_head: &'a WeightRef<'a>,
    x: &'a GpuTensor,
    normed: &'a GpuTensor,
    logits: &'a GpuTensor,
    steps: &mut Vec<Step<'a>>,
) {
    steps.push(Step::RmsnormAutomatic {
        x,
        norm_weight: final_norm,
        x_plain: normed,
        out: normed,
        awq_scale: None,
        k: geo.dim,
        eps: geo.eps,
        rotation: RotationPlan::None,
    });
    steps.push(Step::Gemv {
        w: lm_head,
        input: GemvInput::Raw(normed),
        out: logits,
    });
    if geo.softcap > 0.0 {
        steps.push(Step::Softcap(SoftcapOp {
            logits,
            n: geo.vocab,
            cap: geo.softcap,
        }));
    }
}
