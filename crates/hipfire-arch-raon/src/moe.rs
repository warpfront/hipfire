// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon dense/shared FFN + top-6 MoE descriptors onto the generic
//! [`GroupedMoeOp`] / [`GatedFfnOp`] programs. Owner: slice F.
//!
//! This file only binds Raon's values (SiTU betas 4/25, sigmoid router with a
//! selection-only correction bias, `1e-20` renorm epsilon, scale 1, the ungated
//! shared expert added after the routed sum is cast, ascending-expert-id FP32
//! accumulation, logical 1088 -> physical 1280 down padding) to the shared
//! operators in `hipfire-dispatch`. No numerical kernel or admission rule
//! knows this model.
//!
//! Layer math (reference `M:72-80,475-493,640-812`):
//! ```text
//! g,u   = bf16(Wgate x), bf16(Wup x)
//! a     = bf16( [4 tanh(g/4) sigmoid(g)] * [25 tanh(u/25)] )
//! ffn   = bf16(Wdown pad(a))
//! top   = top6(sigmoid(Wr x_f32) + bias)        # tie -> lower expert id
//! w     = s[top] / (sum s[top] + 1e-20) * 1.0   # unbiased scores
//! out   = bf16( bf16( sum_{e ascending} bf16(expert_e) * w_e ) + shared )
//! ```

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::moe::{GroupedMoeGeometry, MOE_GROUPED_MAX_TOP_K};
use hipfire_dispatch::pipeline::{
    execute_gated_ffn, execute_grouped_moe, CastPolicy, CombineOrder, FfnActivation, GatedFfnOp,
    GroupedMoeActTier, GroupedMoeGroupedScratch, GroupedMoeIndexedScratch, GroupedMoeOp,
    GroupedMoePolicy, GroupedMoeProtocol, GroupedMoeRouted, GroupedMoeRouter, GroupedMoeScratch,
    GroupedMoeScratchPlan, GroupedMoeShared, GroupedMoeSharedScratch, RouterKind,
    SharedExpertPolicy,
};
use hipfire_runtime::llama::WeightTensor;
use rdna_compute::ffn_activation::single_expert_table_words;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Raon geometry and activation constants consumed by the generic programs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaonMoeDims {
    pub hidden: usize,
    pub n_experts: usize,
    pub top_k: usize,
    /// Routed and shared expert intermediate width (logical).
    pub logical_mi: usize,
    /// Physical K of the routed/shared down projections (`logical_mi` padded
    /// with exact zeros to a multiple of 256).
    pub down_k: usize,
    /// First-layer dense MLP intermediate width.
    pub dense_mi: usize,
    /// SiTU gate beta (`activation_situ_beta`).
    pub gate_beta: f32,
    /// SiTU linear/value beta (`activation_situ_linear_beta`).
    pub value_beta: f32,
    pub router_eps: f32,
    pub router_scale: f32,
    /// Activation tier of the grouped routed-expert GEMMs (`HIPFIRE_RAON_PREFILL_PRECISION`, set by the
    /// bundle at load; `F16` = the F16 WMMA core).
    pub routed_tier: GroupedMoeActTier,
}

impl RaonMoeDims {
    /// Raon-1-mini-preview: 128 routed experts, top-6, hidden 2560, expert
    /// width 1088 (down K padded to 1280), dense 12032, SiTU 4/25, router
    /// epsilon `1e-20`, scale `1.0`.
    pub const fn raon() -> Self {
        Self {
            hidden: 2560,
            n_experts: 128,
            top_k: 6,
            logical_mi: 1088,
            down_k: 1280,
            dense_mi: 12032,
            gate_beta: 4.0,
            value_beta: 25.0,
            router_eps: 1e-20,
            router_scale: 1.0,
            routed_tier: GroupedMoeActTier::F16,
        }
    }

    /// Routed + shared geometry of one MoE layer (the shared expert has the
    /// routed width, per `shared_size = moe_intermediate * num_shared_experts`
    /// with one shared expert).
    pub fn geometry(&self) -> GroupedMoeGeometry {
        GroupedMoeGeometry {
            n_experts: self.n_experts,
            top_k: self.top_k,
            hidden: self.hidden,
            logical_mi: self.logical_mi,
            down_k: self.down_k,
            shared_mi: self.logical_mi,
            shared_down_k: self.down_k,
        }
    }

    pub fn activation(&self) -> FfnActivation {
        FfnActivation::Situ {
            gate_beta: self.gate_beta,
            value_beta: self.value_beta,
        }
    }

    /// The sealed policy values of the Raon MoE layer.
    pub fn policy(&self) -> GroupedMoePolicy {
        GroupedMoePolicy {
            router: RouterKind::SigmoidBiasRenorm {
                eps: self.router_eps,
                scale: self.router_scale,
            },
            activation: self.activation(),
            shared: SharedExpertPolicy::UngatedAfterRoutedCast,
            cast: CastPolicy::Bf16Boundaries,
            combine: CombineOrder::ExpertIdAscending,
        }
    }

    /// Check the descriptor against the generic kernel limits.
    pub fn validate(&self) -> Result<(), String> {
        if self.top_k > MOE_GROUPED_MAX_TOP_K {
            return Err(format!(
                "raon moe: top_k {} exceeds the grouped kernels' limit {MOE_GROUPED_MAX_TOP_K}",
                self.top_k
            ));
        }
        self.geometry().validate().map_err(|e| e.to_string())?;
        if self.dense_mi == 0 || self.dense_mi % 256 != 0 {
            return Err(format!(
                "raon moe: dense intermediate {} is not a positive multiple of 256",
                self.dense_mi
            ));
        }
        Ok(())
    }
}

/// Borrowed tensors of one MoE layer (layers 1..24).
#[derive(Clone, Copy)]
pub struct RaonMoeLayerRefs<'a> {
    /// F32 `[n_experts, hidden]` router weight (BF16 values expanded).
    pub router_weight: &'a GpuTensor,
    /// F32 `[n_experts]` selection-only correction bias.
    pub correction_bias: &'a GpuTensor,
    /// `[n_experts]` u64 pointers to fused qt44 `[2*logical_mi, hidden]`
    /// gate-then-up tensors.
    pub gate_up_ptrs: &'a GpuTensor,
    /// `[n_experts]` u64 pointers to qt44 `[hidden, down_k]` down tensors.
    pub down_ptrs: &'a GpuTensor,
    /// Layer AWQ divide of the gate/up input basis (F32 `[hidden]`), shared by
    /// the routed and the shared gate/up.
    pub gate_up_awq: Option<&'a GpuTensor>,
    /// `[n_experts]` u64 pointers to per-expert F32 `[down_k]` AWQ divides.
    pub down_awq_ptrs: Option<&'a GpuTensor>,
    /// Shared expert projections (qt44): gate/up `[logical_mi, hidden]`, down
    /// `[hidden, down_k]`. Gate/up `awq_scale` equals `gate_up_awq`.
    pub shared: GroupedMoeShared<'a>,
    /// Every routed gate/up and down header is loader-verified symmetric
    /// (`zp == -8 * sc`); with a quantized routed tier selected, admits the grouped W4A8/W4A4 core.
    pub iu4_symmetric: bool,
}

impl<'a> RaonMoeLayerRefs<'a> {
    /// Bind a loaded layer. The loader verified that every routed gate/up and
    /// the shared gate/up carry the one layer AWQ divide `gate_up_awq`, so the
    /// shared gate/up references are bound to exactly that tensor (one rotated
    /// basis feeds all of them); the shared down keeps its own sidecar.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        router: &'a WeightTensor,
        correction_bias: &'a GpuTensor,
        shared_gate: &'a WeightTensor,
        shared_up: &'a WeightTensor,
        shared_down: &'a WeightTensor,
        gate_up_ptrs: &'a GpuTensor,
        down_ptrs: &'a GpuTensor,
        gate_up_awq: Option<&'a GpuTensor>,
        down_awq_ptrs: Option<&'a GpuTensor>,
    ) -> Self {
        let mut gate = shared_gate.dispatch_ref();
        let mut up = shared_up.dispatch_ref();
        // Only a qt44 shared gate/up consumes the rotated, AWQ-divided basis;
        // Q8_0/F16/F32 projections read the plain activation and take no scale.
        if gate.dtype == DType::MQ4G256V2 {
            gate.awq_scale = gate_up_awq;
        }
        if up.dtype == DType::MQ4G256V2 {
            up.awq_scale = gate_up_awq;
        }
        Self {
            router_weight: &router.buf,
            correction_bias,
            gate_up_ptrs,
            down_ptrs,
            gate_up_awq,
            down_awq_ptrs,
            shared: GroupedMoeShared {
                gate,
                up,
                down: shared_down.dispatch_ref(),
            },
            iu4_symmetric: false,
        }
    }

    /// Bind the loader's symmetric-header verdict ([`MoeFfnWeights::iu4_symmetric`](crate::weights::MoeFfnWeights)).
    pub fn with_iu4_symmetric(mut self, verified: bool) -> Self {
        self.iu4_symmetric = verified;
        self
    }
}

/// Borrowed projections of the first-layer dense MLP.
#[derive(Clone, Copy)]
pub struct RaonDenseRefs<'a> {
    pub weights: GroupedMoeShared<'a>,
    /// AWQ divide of the gate/up input basis (the gate's sidecar).
    pub input_awq: Option<&'a GpuTensor>,
}

impl<'a> RaonDenseRefs<'a> {
    /// Gate and up must both carry an AWQ sidecar or neither (they share one
    /// rotated input basis built with the gate's sidecar; the loader verifies
    /// the two sidecars value-identical). Shared/dense projections may be qt44,
    /// Q8_0 or F32; the generic program refuses any other format by name.
    pub fn from_parts(
        gate: &'a WeightTensor,
        up: &'a WeightTensor,
        down: &'a WeightTensor,
    ) -> Result<Self, String> {
        if gate.awq_scale.is_some() != up.awq_scale.is_some() {
            return Err(
                "raon dense ffn: gate and up must both carry an AWQ sidecar or neither".into(),
            );
        }
        Ok(Self {
            input_awq: gate.awq_scale.as_ref(),
            weights: GroupedMoeShared {
                gate: gate.dispatch_ref(),
                up: up.dispatch_ref(),
                down: down.dispatch_ref(),
            },
        })
    }
}

/// Persistent scratch of one MoE layer call, sized for up to `max_rows` rows
/// on both the indexed and the grouped core. Reused by every layer.
pub struct RaonMoeScratch {
    pub max_rows: usize,
    pub router_logits: GpuTensor,
    pub topk_indices: GpuTensor,
    pub topk_weights: GpuTensor,
    pub x_rot: GpuTensor,
    pub act: GpuTensor,
    pub rot: GpuTensor,
    pub down_rows: GpuTensor,
    pub indexed_gate: GpuTensor,
    pub indexed_up: GpuTensor,
    pub gate_up_rows: GpuTensor,
    pub expert_token_counts: GpuTensor,
    pub expert_offsets: GpuTensor,
    pub sorted_slot_index: GpuTensor,
    pub expert_tile_ids: GpuTensor,
    pub inverse_perm: GpuTensor,
    pub shared_gate: GpuTensor,
    pub shared_up: GpuTensor,
    pub shared_act: GpuTensor,
    pub shared_rot: GpuTensor,
    /// Cast shared expert output `[rows, hidden]` (the shared-only result).
    pub shared_out: GpuTensor,
    /// Single-expert grouping tables of the shared expert projections.
    pub shared_tables: GpuTensor,
}

fn alloc_words(gpu: &mut Gpu, words: usize) -> Result<GpuTensor, String> {
    // i32 tables are four-byte words like F32; the kernels only use the extent.
    gpu.alloc_tensor(&[words.max(1)], DType::F32)
        .map_err(|e| format!("raon moe scratch alloc: {e}"))
}

/// Bytes of one `alloc_words` tensor (exact size, 256 B minimum).
fn word_bytes(words: usize) -> usize {
    (words.max(1) * 4).max(256)
}

impl RaonMoeScratch {
    /// Device bytes [`Self::new`] allocates for `max_rows` rows.
    pub fn required_bytes(dims: &RaonMoeDims, max_rows: usize) -> Result<usize, String> {
        dims.validate()?;
        let p = GroupedMoeScratchPlan::for_rows(&dims.geometry(), max_rows.max(1))
            .map_err(|e| e.to_string())?;
        Ok([
            p.router_logits,
            p.topk_indices,
            p.topk_weights,
            p.x_rot,
            p.act,
            p.rot,
            p.down_rows,
            p.indexed_gate,
            p.indexed_up,
            p.gate_up_rows,
            p.expert_token_counts,
            p.expert_offsets,
            p.sorted_slot_index,
            p.expert_tile_ids,
            p.inverse_perm,
            p.shared_gate,
            p.shared_up,
            p.shared_act,
            p.shared_rot,
            p.shared_out,
            p.shared_tables,
        ]
        .into_iter()
        .map(word_bytes)
        .sum())
    }
}

impl RaonMoeScratch {
    pub fn new(gpu: &mut Gpu, dims: &RaonMoeDims, max_rows: usize) -> Result<Self, String> {
        dims.validate()?;
        let plan = GroupedMoeScratchPlan::for_rows(&dims.geometry(), max_rows.max(1))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            max_rows,
            router_logits: alloc_words(gpu, plan.router_logits)?,
            topk_indices: alloc_words(gpu, plan.topk_indices)?,
            topk_weights: alloc_words(gpu, plan.topk_weights)?,
            x_rot: alloc_words(gpu, plan.x_rot)?,
            act: alloc_words(gpu, plan.act)?,
            rot: alloc_words(gpu, plan.rot)?,
            down_rows: alloc_words(gpu, plan.down_rows)?,
            indexed_gate: alloc_words(gpu, plan.indexed_gate)?,
            indexed_up: alloc_words(gpu, plan.indexed_up)?,
            gate_up_rows: alloc_words(gpu, plan.gate_up_rows)?,
            expert_token_counts: alloc_words(gpu, plan.expert_token_counts)?,
            expert_offsets: alloc_words(gpu, plan.expert_offsets)?,
            sorted_slot_index: alloc_words(gpu, plan.sorted_slot_index)?,
            expert_tile_ids: alloc_words(gpu, plan.expert_tile_ids)?,
            inverse_perm: alloc_words(gpu, plan.inverse_perm)?,
            shared_gate: alloc_words(gpu, plan.shared_gate)?,
            shared_up: alloc_words(gpu, plan.shared_up)?,
            shared_act: alloc_words(gpu, plan.shared_act)?,
            shared_rot: alloc_words(gpu, plan.shared_rot)?,
            shared_out: alloc_words(gpu, plan.shared_out)?,
            shared_tables: alloc_words(gpu, plan.shared_tables)?,
        })
    }

    /// Borrowed operand view for [`GroupedMoeOp::scratch`].
    pub fn view(&self) -> GroupedMoeScratch<'_> {
        GroupedMoeScratch {
            router_logits: &self.router_logits,
            topk_indices: &self.topk_indices,
            topk_weights: &self.topk_weights,
            x_rot: &self.x_rot,
            act: &self.act,
            rot: &self.rot,
            down_rows: &self.down_rows,
            indexed: Some(GroupedMoeIndexedScratch {
                gate: &self.indexed_gate,
                up: &self.indexed_up,
            }),
            grouped: Some(GroupedMoeGroupedScratch {
                gate_up_rows: &self.gate_up_rows,
                expert_token_counts: &self.expert_token_counts,
                expert_offsets: &self.expert_offsets,
                sorted_slot_index: &self.sorted_slot_index,
                expert_tile_ids: &self.expert_tile_ids,
                inverse_perm: &self.inverse_perm,
            }),
            shared: Some(GroupedMoeSharedScratch {
                gate_out: &self.shared_gate,
                up_out: &self.shared_up,
                act: &self.shared_act,
                rot: &self.shared_rot,
                out: &self.shared_out,
                tables: &self.shared_tables,
            }),
        }
    }

    /// Release every tensor.
    pub fn free(self, gpu: &mut Gpu) {
        for t in [
            self.router_logits,
            self.topk_indices,
            self.topk_weights,
            self.x_rot,
            self.act,
            self.rot,
            self.down_rows,
            self.indexed_gate,
            self.indexed_up,
            self.gate_up_rows,
            self.expert_token_counts,
            self.expert_offsets,
            self.sorted_slot_index,
            self.expert_tile_ids,
            self.inverse_perm,
            self.shared_gate,
            self.shared_up,
            self.shared_act,
            self.shared_rot,
            self.shared_out,
            self.shared_tables,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Persistent scratch of the first-layer dense MLP (`dense_mi` wide).
pub struct RaonDenseScratch {
    pub max_rows: usize,
    pub x_rot: GpuTensor,
    pub gate_out: GpuTensor,
    pub up_out: GpuTensor,
    pub act: GpuTensor,
    pub rot: GpuTensor,
    /// Single-expert grouping tables of the dense projections.
    pub tables: GpuTensor,
}

impl RaonDenseScratch {
    /// Device bytes [`Self::new`] allocates for `max_rows` rows.
    pub fn required_bytes(dims: &RaonMoeDims, max_rows: usize) -> Result<usize, String> {
        dims.validate()?;
        let rows = max_rows.max(1);
        Ok([
            rows * dims.hidden,
            rows * dims.dense_mi,
            rows * dims.dense_mi,
            rows * dims.dense_mi,
            rows * dims.dense_mi,
            single_expert_table_words(rows),
        ]
        .into_iter()
        .map(word_bytes)
        .sum())
    }

    pub fn new(gpu: &mut Gpu, dims: &RaonMoeDims, max_rows: usize) -> Result<Self, String> {
        dims.validate()?;
        let rows = max_rows.max(1);
        Ok(Self {
            max_rows,
            x_rot: alloc_words(gpu, rows * dims.hidden)?,
            gate_out: alloc_words(gpu, rows * dims.dense_mi)?,
            up_out: alloc_words(gpu, rows * dims.dense_mi)?,
            act: alloc_words(gpu, rows * dims.dense_mi)?,
            rot: alloc_words(gpu, rows * dims.dense_mi)?,
            tables: alloc_words(gpu, single_expert_table_words(rows))?,
        })
    }

    pub fn free(self, gpu: &mut Gpu) {
        for t in [
            self.x_rot,
            self.gate_out,
            self.up_out,
            self.act,
            self.rot,
            self.tables,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Build the [`GroupedMoeOp`] of one MoE layer. The caller owns `x`, `out` and
/// `routed_out` (all F32 `[rows, hidden]`, BF16-valued) and the `ctx`.
#[allow(clippy::too_many_arguments)]
pub fn raon_moe_op<'a>(
    ctx: &'a DispatchCtx,
    dims: &RaonMoeDims,
    layer: &RaonMoeLayerRefs<'a>,
    protocol: GroupedMoeProtocol,
    rows: usize,
    x: &'a GpuTensor,
    scratch: &'a RaonMoeScratch,
    out: &'a GpuTensor,
    routed_out: Option<&'a GpuTensor>,
) -> GroupedMoeOp<'a> {
    GroupedMoeOp {
        ctx,
        policy: dims.policy(),
        geometry: dims.geometry(),
        protocol,
        rows,
        x,
        router: GroupedMoeRouter {
            weight: layer.router_weight,
            correction_bias: Some(layer.correction_bias),
        },
        routed: GroupedMoeRouted {
            dtype: DType::MQ4G256V2,
            gate_up_ptrs: layer.gate_up_ptrs,
            down_ptrs: layer.down_ptrs,
            input_awq: layer.gate_up_awq,
            down_awq_ptrs: layer.down_awq_ptrs,
            iu4_symmetric_verified: layer.iu4_symmetric,
            act_tier: dims.routed_tier,
        },
        shared: Some(layer.shared),
        scratch: scratch.view(),
        out,
        routed_out,
    }
}

/// Run one MoE layer: decode (`rows == 1`) on the indexed qt44 core, wider
/// batches on the grouped WMMA core (`GroupedMoeProtocol::Auto`).
#[allow(clippy::too_many_arguments)]
pub fn forward_moe(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    dims: &RaonMoeDims,
    layer: &RaonMoeLayerRefs<'_>,
    rows: usize,
    x: &GpuTensor,
    scratch: &RaonMoeScratch,
    out: &GpuTensor,
) -> Result<(), String> {
    if rows > scratch.max_rows {
        return Err(format!(
            "raon moe: {rows} rows exceed the scratch capacity {}",
            scratch.max_rows
        ));
    }
    let op = raon_moe_op(
        ctx,
        dims,
        layer,
        GroupedMoeProtocol::Auto,
        rows,
        x,
        scratch,
        out,
        None,
    );
    execute_grouped_moe(gpu, &op).map_err(|e| e.to_string())
}

/// Build the [`GatedFfnOp`] of the dense layer-0 MLP (SiTU, down K 12032).
#[allow(clippy::too_many_arguments)]
pub fn raon_dense_op<'a>(
    ctx: &'a DispatchCtx,
    dims: &RaonMoeDims,
    dense: &RaonDenseRefs<'a>,
    rows: usize,
    x: &'a GpuTensor,
    scratch: &'a RaonDenseScratch,
    shared_out: &'a GpuTensor,
    out: &'a GpuTensor,
) -> GatedFfnOp<'a> {
    GatedFfnOp {
        ctx,
        activation: dims.activation(),
        cast: CastPolicy::Bf16Boundaries,
        rows,
        hidden: dims.hidden,
        mi: dims.dense_mi,
        down_k: dims.dense_mi,
        x,
        input_awq: dense.input_awq,
        weights: dense.weights,
        x_rot: &scratch.x_rot,
        scratch: GroupedMoeSharedScratch {
            gate_out: &scratch.gate_out,
            up_out: &scratch.up_out,
            act: &scratch.act,
            rot: &scratch.rot,
            out: shared_out,
            tables: &scratch.tables,
        },
        out,
    }
}

/// Run the dense layer-0 MLP: `out = bf16(Wdown bf16(SiTU(bf16(Wg x), bf16(Wu x))))`.
#[allow(clippy::too_many_arguments)]
pub fn forward_dense(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    dims: &RaonMoeDims,
    dense: &RaonDenseRefs<'_>,
    rows: usize,
    x: &GpuTensor,
    scratch: &RaonDenseScratch,
    out: &GpuTensor,
) -> Result<(), String> {
    if rows > scratch.max_rows {
        return Err(format!(
            "raon dense ffn: {rows} rows exceed the scratch capacity {}",
            scratch.max_rows
        ));
    }
    let op = raon_dense_op(ctx, dims, dense, rows, x, scratch, out, out);
    execute_gated_ffn(gpu, &op).map_err(|e| e.to_string())
}

/// Reference-free structural checks of the descriptor values (CPU only).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raon_dims_admit_and_bind_the_documented_values() {
        let d = RaonMoeDims::raon();
        d.validate().unwrap();
        let g = d.geometry();
        assert_eq!(
            (g.n_experts, g.top_k, g.hidden, g.logical_mi, g.down_k),
            (128, 6, 2560, 1088, 1280)
        );
        assert_eq!((g.shared_mi, g.shared_down_k), (1088, 1280));
        let p = d.policy();
        assert_eq!(
            p.router,
            RouterKind::SigmoidBiasRenorm {
                eps: 1e-20,
                scale: 1.0
            }
        );
        assert_eq!(
            p.activation,
            FfnActivation::Situ {
                gate_beta: 4.0,
                value_beta: 25.0
            }
        );
        assert_eq!(p.shared, SharedExpertPolicy::UngatedAfterRoutedCast);
        assert_eq!(p.cast, CastPolicy::Bf16Boundaries);
        assert_eq!(p.combine, CombineOrder::ExpertIdAscending);
    }

    #[test]
    fn dims_refuse_out_of_limit_top_k() {
        let mut d = RaonMoeDims::raon();
        d.top_k = 9;
        assert!(d.validate().is_err());
        d.top_k = 6;
        d.dense_mi = 12000;
        assert!(d.validate().is_err());
    }

    #[test]
    fn scratch_plan_covers_prefill_chunk_and_decode() {
        let d = RaonMoeDims::raon();
        let plan = GroupedMoeScratchPlan::for_rows(&d.geometry(), 512).unwrap();
        assert_eq!(plan.m_total, 4992);
        assert_eq!(plan.topk_indices, 512 * 6);
        assert_eq!(plan.shared_out, 512 * 2560);
        let decode = GroupedMoeScratchPlan::for_rows(&d.geometry(), 1).unwrap();
        assert!(decode.indexed_gate > 0);
    }
}

/// Bytes of persistent device scratch [`RaonMoeScratch::new`] allocates for
/// `max_rows` rows (the F32/i32 word counts of the plan, four bytes each; every
/// tensor is at least one word).
pub fn moe_scratch_bytes(dims: &RaonMoeDims, max_rows: usize) -> Result<usize, String> {
    let p = GroupedMoeScratchPlan::for_rows(&dims.geometry(), max_rows.max(1))
        .map_err(|e| e.to_string())?;
    let words = [
        p.router_logits,
        p.topk_indices,
        p.topk_weights,
        p.x_rot,
        p.act,
        p.rot,
        p.down_rows,
        p.indexed_gate,
        p.indexed_up,
        p.gate_up_rows,
        p.expert_token_counts,
        p.expert_offsets,
        p.sorted_slot_index,
        p.expert_tile_ids,
        p.inverse_perm,
        p.shared_gate,
        p.shared_up,
        p.shared_act,
        p.shared_rot,
        p.shared_out,
        p.shared_tables,
    ];
    Ok(words.iter().map(|w| (*w).max(1) * 4).sum())
}

/// Bytes of [`RaonDenseScratch::new`] for `max_rows` rows.
pub fn dense_scratch_bytes(dims: &RaonMoeDims, max_rows: usize) -> usize {
    let rows = max_rows.max(1);
    (rows * dims.hidden + 4 * rows * dims.dense_mi + single_expert_table_words(rows)) * 4
}
