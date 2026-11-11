// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact QT44/QT53 top-10 grouped-prefill lowering, and the gfx942 qt=55
//! FP8E4M3B128 grouped lowering.
//!
//! This module is deliberately narrow.  It is reached only after the sealed
//! MoE predicate has admitted the immutable grouped-kernel geometry and replicated route.
//! Generic MoE prefill continues to own the k=8 path; no QT53 format is
//! admitted by this module through a representative-dtype fallback.
//!
//! # qt=55 FP8E4M3B128 route (`MoeRouteCapability::Fp8B128Grouped`)
//!
//! Every stage entrypoint below tests [`is_fp8`] before any legacy branch and
//! lowers the FP8 route through its own `fp8_*` function, so no QT44/QT53 arm
//! (rotation, F16 WMMA, symmetric IU4, indexed down, row fold, HC fuse, EP)
//! can run on qt=55 bytes, and no FP8 stage can run a QT44/QT53 kernel.  The
//! FP8 route is the grouped protocol at every row count (decode M=1 included):
//! `input_basis` (no rotation, no pass: the launchers stage BF16) -> router and
//! shared projections through `project_weight(s)` (gfx942 BF16 MFMA, all M) ->
//! `scatter` (stable pad16 grouping + one tile descriptor in the `gate_batch`
//! byte prefix) -> `gate_up` (F32 `[G, 2mi]`) -> `unscatter` (SwiGLU into
//! `rot_batch`, F32 holding BF16) -> `activation` (nothing owed) -> `down`
//! (F32 `[G, hidden]`) -> `combine` (canonical ascending expert-ID order).  The
//! stage's `use_path2` flag is ignored (there is no path 1) and `grouped_rows`
//! must equal `p.m_total_max`.  Every stage runs [`fp8_validate`], which checks
//! every buffer and weight the stage or any later FP8 stage touches before the
//! first launch; no stage allocates device memory.
//!
//! ## Activation math (why the QT44 activation kernels are not reused)
//!
//! The FP8 teacher computes `bf16(bf16(silu(bf16(gate))) * bf16(up))`: gate and
//! up are BF16, the SiLU result is rounded to BF16, and the product is rounded
//! to BF16.  The QT44 kernels (`moe_gate_up_unscatter_silu_top10` and
//! `shared_expert_activation_bf16_f32`) round gate and up and the final
//! product but keep `silu(gate)` in F32 (`(v / (1 + expf(-v))) * u`), so they
//! differ from the teacher.  The FP8 route therefore launches the dedicated
//! `Gfx942Device::fp8b128_moe_unscatter_silu` / `fp8b128_moe_shared_activation`
//! kernels, which round all of gate, up, SiLU and product.  The QT44 sources
//! are untouched.

use super::layer_ops::hip;
use crate::families::gemv::WeightRef;
use crate::families::moe::{
    MoeDecodeOwnerMask, MoePrefillParams, MoeRouteCapability, MoeSharedPrefill,
    MoeTileExchangeParams,
};
use crate::types::DispatchError;
use rdna_compute::moe::SharedExpertActivation;
use rdna_compute::shared_i8::SharedI8Weights;
use rdna_compute::tensor_ops::{bf16_scaled_add_batched, Bf16ScaledAddBatched};
use rdna_compute::{DType, Gpu, GpuTensor};

#[inline]
fn f32_view(source: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    source.sub_offset(offset, len)
}

#[inline]
fn grouped_geometry(p: &MoePrefillParams<'_>) -> bool {
    let Some(policy) = p.route_policy else {
        return false;
    };
    if policy.capability.is_slab() {
        return p.batch_size > 0
            && crate::families::moe::slab_route_geometry_supported(
                None,
                &policy,
                p.n_exp,
                p.k_top,
                p.gate_up_k,
                p.mi,
                p.gate_up_k,
                p.down_m,
                p.down_k,
                &p.dtypes,
                p.expert_dtype_tags.is_some(),
            );
    }
    p.batch_size > 0
        && crate::families::moe::grouped_route_geometry_supported(
            &policy,
            p.n_exp,
            p.k_top,
            p.gate_up_k,
            p.mi,
            p.gate_up_k,
            p.down_m,
            p.down_k,
            p.dtypes.routed_gate_up,
            p.dtypes.routed_down,
        )
}

fn require_geometry(p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    if grouped_geometry(p) {
        Ok(())
    } else {
        Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "grouped-prefill-shape",
            arch: "",
            quant: "declared route",
        })
    }
}

/// The declared route is the gfx942 qt=55 FP8E4M3B128 grouped route.  Tested
/// first by every stage entrypoint, before any legacy branch.
#[inline]
fn is_fp8(p: &MoePrefillParams<'_>) -> bool {
    p.route_policy
        .is_some_and(|policy| policy.capability.is_fp8_grouped())
}

/// The declared route is the sealed expert-slab route (gfx1100, gfx1151, gfx1201).
#[inline]
fn is_slab(p: &MoePrefillParams<'_>) -> bool {
    p.route_policy
        .is_some_and(|policy| policy.capability.is_slab())
}

/// Whether this slab call exceeds the actual eager exact-row admission and
/// runs K2 grouped IU4. K1 uses fixed eight-row physical tiles, not an
/// eight-row logical limit; recording and ordinary prefill retain their cap.
fn slab_k2(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    is_slab(p)
        && super::slab_route::arch_supported(&gpu.arch)
        && p.recipe.bf16_round_trip()
        && p.x_norm_batch.dtype == DType::F32
        && p.down_k == p.mi
        && gpu.route_rows(p.batch_size) > gpu.few_row_limit()
}

/// The sealed lowerer may have selected grouped stages before entering the
/// eager verify scope. Every stage must agree on slot-ordered decode layout.
fn grouped_path_for_rows(requested: bool, route: usize, limit: usize, wide_grouped: bool) -> bool {
    requested && (wide_grouped || !super::layer_ops::wide_exact_route(route, limit))
}

fn grouped_path(gpu: &Gpu, p: &MoePrefillParams<'_>, requested: bool) -> bool {
    grouped_path_for_rows(
        requested,
        gpu.route_rows(p.batch_size),
        gpu.few_row_limit(),
        gpu.wide_grouped_moe(),
    )
}

/// Geometry gate of whichever grouped route `p` declares.
fn require_route_geometry(p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    if is_fp8(p) {
        fp8_require_geometry(p)
    } else {
        require_geometry(p)
    }
}

const F32_BYTES: usize = std::mem::size_of::<f32>();
const BF16_BYTES: usize = 2;
/// Bytes the BF16 combine's rank-order scratch holds per route slot
/// (`moe_combine_order_top10`: `tokens * 10 * 8`).
const ORDER_BYTES_PER_SLOT: usize = 8;

fn scratch_overflow() -> DispatchError {
    DispatchError::Hip("qt44/qt53 grouped prefill: scratch extent overflows".into())
}

fn scratch_mul(a: usize, b: usize) -> Result<usize, DispatchError> {
    a.checked_mul(b).ok_or_else(scratch_overflow)
}

/// Bytes of the six scratch buffers whose size depends on the route the call
/// actually takes.  The same record is the requirement a route derives
/// ([`ScratchRoute::requirement`]) and the capacity the call's tensors hold
/// ([`ScratchBytes::held`]), so validation, lowering and the tests compare one
/// type.  A buffer the route never touches requires 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScratchBytes {
    pub(crate) gate_batch: usize,
    pub(crate) up_batch: usize,
    pub(crate) rot_batch: usize,
    pub(crate) down_expanded: usize,
    pub(crate) y_gate_up_grouped: usize,
    pub(crate) y_down_grouped: usize,
}

impl ScratchBytes {
    fn fields(&self) -> [(&'static str, usize); 6] {
        [
            ("gate_batch", self.gate_batch),
            ("up_batch", self.up_batch),
            ("rot_batch", self.rot_batch),
            ("down_expanded", self.down_expanded),
            ("y_gate_up_grouped", self.y_gate_up_grouped),
            ("y_down_grouped", self.y_down_grouped),
        ]
    }

    /// Physical bytes each scratch tensor of `p` can hold: the smaller of its
    /// logical extent and its device buffer, so neither forged shape metadata
    /// nor a short allocation passes.
    pub(crate) fn held(p: &MoePrefillParams<'_>) -> Self {
        Self {
            gate_batch: tensor_bytes(p.gate_batch),
            up_batch: tensor_bytes(p.up_batch),
            rot_batch: tensor_bytes(p.rot_batch),
            down_expanded: tensor_bytes(p.down_expanded),
            y_gate_up_grouped: tensor_bytes(p.y_gate_up_grouped),
            y_down_grouped: tensor_bytes(p.y_down_grouped),
        }
    }

    /// `Err` naming the first buffer of `held` that is below this requirement.
    pub(crate) fn require_within(&self, held: &Self) -> Result<(), DispatchError> {
        for ((name, need), (_, have)) in self.fields().into_iter().zip(held.fields()) {
            if have < need {
                return Err(DispatchError::Hip(format!(
                    "qt44/qt53/qt55 grouped prefill scratch: {name} holds {have} bytes, \
                     the selected route needs {need}; refusing before launch"
                )));
            }
        }
        Ok(())
    }
}

fn tensor_bytes(tensor: &GpuTensor) -> usize {
    let logical = tensor
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .and_then(|elements| elements.checked_mul(tensor.dtype.size()))
        .unwrap_or(0);
    logical.min(tensor.buf.size())
}

/// F32 `[rows, down_m]` shared-down rows, which the unfused shared down
/// writes into `down_expanded` whenever the BF16 epilogue does not take it.
fn shared_down_bytes(rows: usize, down_m: usize) -> Result<usize, DispatchError> {
    scratch_mul(scratch_mul(rows, down_m)?, F32_BYTES)
}

/// Rank-order scratch of the BF16-row combine (`tokens * k_top * 8` bytes).
fn combine_order_bytes(rows: usize, k_top: usize) -> Result<usize, DispatchError> {
    scratch_mul(scratch_mul(rows, k_top)?, ORDER_BYTES_PER_SLOT)
}

/// Invocation geometry the byte requirements depend on.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScratchDims {
    pub(crate) rows: usize,
    pub(crate) k_top: usize,
    pub(crate) mi: usize,
    pub(crate) down_m: usize,
    /// Padded grouped rows (path 2 only; 0 on path 1).
    pub(crate) grouped_rows: usize,
    /// The call carries a shared expert whose F32 down may land in
    /// `down_expanded` (no `shared_down_out`).
    pub(crate) shared_down: bool,
}

/// The kernel arms one invocation takes, from the same predicates the stage
/// lowering below branches on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScratchRoute {
    pub(crate) path2: bool,
    /// The qt=55 FP8E4M3B128 grouped route (every other flag is then false).
    pub(crate) fp8: bool,
    pub(crate) sym_iu4: bool,
    /// The wide-verify int8 chain (`HIPFIRE_QWEN4_MOE_GU_A8`, [`ep_a8_wide`]): BF16 grouped rows in
    /// both grouped buffers, the combine's rank order in `down_expanded`; exclusive of `sym_iu4`.
    pub(crate) ep_a8_wide: bool,
    pub(crate) gateup_bf16: bool,
    pub(crate) gateup_silu: bool,
    pub(crate) down_wmma: bool,
    pub(crate) indexed_down: bool,
}

impl ScratchRoute {
    pub(crate) fn of(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> Self {
        // The FP8 route consults none of the legacy arm predicates.
        if is_fp8(p) {
            return Self::fp8();
        }
        let use_path2 = grouped_path(gpu, p, use_path2);
        Self {
            path2: use_path2,
            fp8: false,
            sym_iu4: sym_iu4(gpu, p, use_path2),
            ep_a8_wide: ep_a8_wide(gpu, p, use_path2),
            gateup_bf16: use_path2 && gateup_bf16(gpu, p),
            gateup_silu: use_path2 && gateup_silu(gpu, p),
            down_wmma: use_path2 && down_wmma(gpu, p),
            indexed_down: indexed_down(gpu, p, use_path2),
        }
    }

    /// The qt=55 FP8E4M3B128 route: grouped at every row count, no legacy arm.
    pub(crate) fn fp8() -> Self {
        Self {
            path2: true,
            fp8: true,
            sym_iu4: false,
            ep_a8_wide: false,
            gateup_bf16: false,
            gateup_silu: false,
            down_wmma: false,
            indexed_down: false,
        }
    }

    /// Exact bytes each scratch buffer must hold for this route at `dims`:
    /// - symmetric IU4: BF16 `[G, mi]` SwiGLU rows, BF16 `[G, down_m]` down
    ///   rows, the combine's rank order;
    /// - path-2 F32 arms: F32 `[G, 2 mi]` / `[G, down_m]` (BF16 on the gfx11
    ///   WMMA arms), plus the unscattered activation unless gate/up stores
    ///   SwiGLU for the down's own unscatter;
    /// - indexed down and path 1: the `[rows * k_top, ...]`
    ///   batch buffers and the expanded down rows;
    /// - every route that has a shared expert: its F32 `[rows, down_m]` down
    ///   in `down_expanded`, because the BF16 epilogue may decline at run time.
    /// - qt=55 FP8: a 16-byte tile descriptor per padded 16-row tile in
    ///   `gate_batch`, F32 `[G, 2 mi]` / `[G, down_m]` grouped rows, the F32
    ///   `[rows * k_top, mi]` activation (BF16 values) in `rot_batch`, and the
    ///   unfused shared down's F32 rows; `up_batch` is untouched.
    pub(crate) fn requirement(&self, dims: &ScratchDims) -> Result<ScratchBytes, DispatchError> {
        let slots = scratch_mul(dims.rows, dims.k_top)?;
        let activation = scratch_mul(scratch_mul(slots, dims.mi)?, F32_BYTES)?;
        let expanded = scratch_mul(scratch_mul(slots, dims.down_m)?, F32_BYTES)?;
        let grouped = dims.grouped_rows;
        let mut need = ScratchBytes::default();
        if dims.shared_down {
            need.down_expanded = shared_down_bytes(dims.rows, dims.down_m)?;
        }
        if self.fp8 {
            need.gate_batch = scratch_mul(grouped_tiles(grouped), FP8_TILE_DESC_BYTES)?;
            need.rot_batch = activation;
            need.y_gate_up_grouped =
                scratch_mul(scratch_mul(grouped, scratch_mul(2, dims.mi)?)?, F32_BYTES)?;
            need.y_down_grouped = scratch_mul(scratch_mul(grouped, dims.down_m)?, F32_BYTES)?;
            return Ok(need);
        }
        if !self.path2 {
            need.gate_batch = activation;
            need.up_batch = activation;
            need.rot_batch = activation;
            need.down_expanded = need.down_expanded.max(expanded);
            return Ok(need);
        }
        let grouped_bytes = |width: usize, elem: usize| -> Result<usize, DispatchError> {
            scratch_mul(scratch_mul(grouped, width)?, elem)
        };
        let order = combine_order_bytes(dims.rows, dims.k_top)?;
        if self.sym_iu4 || self.ep_a8_wide {
            need.y_gate_up_grouped = grouped_bytes(dims.mi, BF16_BYTES)?;
            need.y_down_grouped = grouped_bytes(dims.down_m, BF16_BYTES)?;
            need.down_expanded = need.down_expanded.max(order);
        } else {
            let gate_up_width = scratch_mul(2, dims.mi)?;
            need.y_gate_up_grouped = match (self.gateup_bf16, self.gateup_silu) {
                (true, true) => grouped_bytes(dims.mi, BF16_BYTES)?,
                (true, false) => grouped_bytes(gate_up_width, BF16_BYTES)?,
                (false, _) => grouped_bytes(gate_up_width, F32_BYTES)?,
            };
            if !self.gateup_silu {
                need.rot_batch = activation;
            }
            if self.down_wmma {
                need.y_down_grouped = grouped_bytes(dims.down_m, BF16_BYTES)?;
                need.down_expanded = need.down_expanded.max(order);
            } else if !self.indexed_down {
                need.y_down_grouped = grouped_bytes(dims.down_m, F32_BYTES)?;
            }
        }
        if self.indexed_down {
            need.rot_batch = activation;
            need.down_expanded = need.down_expanded.max(expanded);
        }
        Ok(need)
    }
}

/// The bytes this invocation's route needs from the call's scratch, for the
/// route [`ScratchRoute::of`] derives at `p`'s actual rows.
pub(crate) fn scratch_requirement(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<ScratchBytes, DispatchError> {
    require_route_geometry(p)?;
    ScratchRoute::of(gpu, p, use_path2).requirement(&ScratchDims {
        rows: p.batch_size,
        k_top: p.k_top,
        mi: p.mi,
        down_m: p.down_m,
        grouped_rows: if use_path2 || is_fp8(p) {
            grouped_rows
        } else {
            0
        },
        shared_down: p.prelude.shared.is_some() && p.shared_down_out.is_none(),
    })
}

/// Preflight of one invocation: every scratch tensor holds the bytes the
/// selected route needs.  Run before the call's first launch by
/// `SealedMoeCall::validate_for_gpu`, and again by each stage that writes
/// scratch.
pub(crate) fn validate_scratch(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    // A fixed reduction partition (balanced trunk EP) is honoured by the expanded top-10
    // combine only: refuse every other arm before this call's first launch.
    require_reduction_partition_route(gpu, p, use_path2)?;
    // qt=55 FP8: the full-route preflight (scratch, weights, experts, router).
    if is_fp8(p) {
        return fp8_validate(gpu, p, grouped_rows).map(drop);
    }
    scratch_requirement(gpu, p, use_path2, grouped_rows)?.require_within(&ScratchBytes::held(p))?;
    if p.shared_down_out.is_some() && p.prelude.shared.is_some() {
        require_shared_rows(p, shared_down_bytes(p.batch_size, p.down_m)?)?;
    }
    Ok(())
}

/// The tensor the unfused shared down writes (`shared_down_out`, else
/// `down_expanded`) holds at least `need` bytes.
fn require_shared_rows(p: &MoePrefillParams<'_>, need: usize) -> Result<(), DispatchError> {
    let (name, tensor) = match p.shared_down_out {
        Some(out) => ("shared_down_out", out),
        None => ("down_expanded", p.down_expanded),
    };
    let have = tensor_bytes(tensor);
    if have < need {
        return Err(DispatchError::Hip(format!(
            "qt44/qt53 grouped prefill scratch: {name} holds {have} bytes, \
             the unfused shared down needs {need}; refusing before launch"
        )));
    }
    Ok(())
}

/// Whether the routed down of this call leaves its `rows * k_top` expanded
/// rows in slot order in `down_expanded` for the canonical top-10 combine to
/// read: the decode-kernel down (path 1, or path 2's few-row indexed down) with
/// no expert-stage table. This is the layout an EP row gather fills.
pub(crate) fn writes_slot_ordered_rows(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> bool {
    let use_path2 = grouped_path(gpu, p, use_path2);
    // The FP8 grouped down leaves grouped rows, never slot-ordered rows.
    !is_fp8(p)
        && p.expert_stage_ptrs.is_none()
        && !sym_iu4(gpu, p, use_path2)
        && !ep_a8_wide(gpu, p, use_path2)
        && (!use_path2
            || indexed_down(gpu, p, use_path2)
            || ep_grouped_down_unscatter(gpu, p, use_path2))
}

/// The call's fixed reduction partition `(rank, ranks)` (`MoeRowGather::reduction_partition` of
/// the layer the row gather itself receives, `ep_layer_idx`), `None` without a gather hook or
/// off a partitioned layer. Per call and layer, never process-global.
pub(crate) fn reduction_partition(p: &MoePrefillParams<'_>) -> Option<(usize, usize)> {
    p.ep_row_gather
        .and_then(|gather| gather.reduction_partition(p.ep_layer_idx))
}

/// The combine arm of one call's routed down, from the route's pure predicates (the arm order of
/// [`combine_stage`]): FP8 grouped rows, the expanded top-10 arms (path 1, the indexed down and
/// the EP grouped-down unscatter), the BF16 grouped rows of the F16 WMMA / symmetric IU4 downs, or
/// the plain F32 grouped rows.
fn combine_arm_of(
    fp8: bool,
    use_path2: bool,
    indexed_down: bool,
    ep_unscatter: bool,
    bf16_rows: bool,
) -> super::CombineArm {
    if fp8 {
        super::CombineArm::Fp8Grouped
    } else if !use_path2 || indexed_down || ep_unscatter {
        super::CombineArm::ExpandedTop10
    } else if bf16_rows {
        super::CombineArm::GroupedBf16InOwned
    } else {
        super::CombineArm::GroupedF32
    }
}

/// [`combine_arm_of`] at `p`'s actual rows on this device.
fn combine_arm(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> super::CombineArm {
    let use_path2 = grouped_path(gpu, p, use_path2);
    combine_arm_of(
        is_fp8(p),
        use_path2,
        indexed_down(gpu, p, use_path2),
        ep_grouped_down_unscatter(gpu, p, use_path2),
        down_wmma(gpu, p) || sym_iu4(gpu, p, use_path2) || ep_a8_wide(gpu, p, use_path2),
    )
}

/// Call-level preflight of the fixed reduction partition: a partitioned call's combine MUST be
/// the expanded top-10 arm (the partitioned leaf); any other arm would fold foreign experts
/// unmasked, so it fails here, before the call's first launch.
fn require_reduction_partition_route(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    match reduction_partition(p) {
        None => Ok(()),
        Some(partition) => {
            super::combine_leaf(combine_arm(gpu, p, use_path2), Some(partition)).map(drop)
        }
    }
}

/// An EP row-gathered call whose routed down is the plain grouped F32 down
/// (`HIPFIRE_QWEN4_WIDE_GROUPED_DOWN` at 9..=limit rows): its grouped rows are
/// restored to canonical flat slot order in `down_expanded`
/// (`moe_down_unscatter_k8`, copies every live slot) right after the down, so the
/// peer gather and the slot-order combine read the layout the indexed down writes.
/// Never on the symmetric IU4 or F16 WMMA downs (they keep BF16 grouped rows).
fn ep_grouped_down_unscatter(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    p.ep_row_gather.is_some()
        && use_path2
        && !is_fp8(p)
        && p.expert_stage_ptrs.is_none()
        && !sym_iu4(gpu, p, use_path2)
        && !ep_a8_wide(gpu, p, use_path2)
        && !down_wmma(gpu, p)
        && !indexed_down(gpu, p, use_path2)
}

/// Prepare the routed expert activation basis.  The route's router/shared
/// projections are BF16 and consume the natural input, while routed QT44
/// gate/up consumes the FWHT basis.  qt=55 FP8: no rotation and no pass; the
/// BF16 operands are produced by the launchers ([`fp8_input_basis`]).
pub(crate) fn input_basis(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    if is_fp8(p) {
        return fp8_input_basis(gpu, p);
    }
    require_geometry(p)?;
    if gateup_rotates_f16(gpu, p, use_path2) {
        return Ok(());
    }
    if ep_a8_wide(gpu, p, use_path2) && !shared_reads_x_rot(p) {
        // The wide-verify int8 chain quantizes `x_norm_batch` itself (`qwen4_moe_rotate256_i8`);
        // the router and the BF16 shared expert read the natural activation. `x_rot_batch` has no
        // reader on this route: see [`shared_reads_x_rot`] for the one conditional reader.
        return Ok(());
    }
    hip(gpu.rotate_x_mq_batched(p.x_norm_batch, p.x_rot_batch, p.gate_up_k, p.batch_size))
}

/// What the fused MoE front ([`front_applies`], [`input_basis_front`]) hands to the later
/// stages of one sealed grouped prefill call: the routed gate/up IU4 sidecar for
/// [`gate_up_front`] and the F16 conversion of `x_norm_batch` (in the shared F16 scratch)
/// for [`shared_gate_up_front`]'s router / shared-expert WMMA GEMMs.
pub(crate) struct MoeFront {
    xq: rdna_compute::scratch::Int4MmqPrepared,
    x16: GpuTensor,
    /// The shared gate/up INT8 GEMM's A8 blocks of the same rotated rows, when the producer
    /// also emitted them ([`shared_int8`]).
    x8: Option<rdna_compute::scratch::Int8MmqPrepared>,
}

/// Whether the symmetric IU4 route (exact gfx1201 or exact gfx1151,
/// `HIPFIRE_QWEN4_MOE_FRONT_FUSE`) reads
/// `x_norm_batch` once in one producer instead of the chain's `mq_rotate_x` (whose F32 output
/// nothing on this route reads: the symmetric gate/up consumes only the IU4 sidecar, and the
/// router and shared projections are BF16 on the natural activation),
/// `convert_f32_to_f16` (inside the router + shared F16 WMMA group launch, or on gfx1151 the
/// fused shared gate/up PeaceMaker launch's own router + selector groups, [`shared_gu_pm`])
/// and the gate/up stage's `mq_rotate_x_i4`. The producer feeds exactly those three
/// consumers, so it applies only where the router stage defers to the shared group's launch
/// ([`router_shares_f16`]: the stages between the producer and that launch convert nothing and
/// reserve no IU4 scratch), and the symmetric arm is the gate/up that runs ([`sym_iu4`], not
/// the slab or FP8 routes). The caller adds the per-invocation conditions (plain program, no
/// CPU split / deferral).
pub(crate) fn front_applies(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    let use_path2 = grouped_path(gpu, p, use_path2);
    gpu.qwen4_moe_front_fuse_applies()
        && !is_fp8(p)
        && !is_slab(p)
        && use_path2
        && grouped_geometry(p)
        && p.x_norm_batch.dtype == DType::F32
        && p.gate_up_k == p.prelude.router.k
        && sym_iu4(gpu, p, use_path2)
        && router_shares_f16(gpu, p)
}

/// The fused input-basis stage: one launch writes the IU4 sidecar and the F16 X. The F32
/// rotated basis `x_rot_batch` is not written: no reader exists on this route.
pub(crate) fn input_basis_front(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
) -> Result<MoeFront, DispatchError> {
    require_geometry(p)?;
    // The front only runs on path 2 (`front_applies`).
    if shared_int8(gpu, p, true) {
        let (xq, x16, x8) =
            hip(gpu.qwen4_moe_front_producer_i8(p.x_norm_batch, p.gate_up_k, p.batch_size))?;
        return Ok(MoeFront {
            xq,
            x16,
            x8: Some(x8),
        });
    }
    let (xq, x16) = hip(gpu.qwen4_moe_front_producer(p.x_norm_batch, p.gate_up_k, p.batch_size))?;
    Ok(MoeFront { xq, x16, x8: None })
}

/// Whether the path-2 F16 WMMA gate/up rotates the activation straight to
/// F16 itself (the bytes the F32 rotation + conversion produce): only when no
/// shared projection reads the F32 basis `x_rot_batch`.
fn gateup_rotates_f16(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    use_path2
        && gateup_bf16(gpu, p)
        && p.prelude.shared.as_ref().is_none_or(|shared| {
            [&shared.weights.selector, &shared.weights.gate]
                .iter()
                .all(|w| w.dtype != DType::MQ4G256V2)
        })
}

fn batch_projection(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
) -> Result<(), DispatchError> {
    match weight.dtype {
        // BF16 rows read the natural activation; this route materializes the
        // packed basis once per layer, so no per-projection rotation is owed.
        DType::BF16 => super::project_weight(gpu, weight, x, y, batch_size, None),
        DType::F32 => hip(gpu.gemm_f32_batched(weight.buf, x, y, weight.m, weight.k, batch_size)),
        DType::MQ4G256V2 if (2..=gpu.few_row_limit()).contains(&gpu.route_rows(batch_size)) => {
            let part = (weight.buf, y, weight.m);
            let none = (weight.buf, y, 0);
            hip(gpu.gemv_mq4g256v2_rows_x4([part, none, none, none], x, weight.k, batch_size))
        }
        DType::MQ4G256V2 => {
            hip(gpu.gemm_mq4g256v2(weight.buf, x, y, weight.m, weight.k, batch_size))
        }
        DType::MQ4G128V2 => {
            hip(gpu.gemm_mq4g128v2_batched(weight.buf, x, y, weight.m, weight.k, batch_size))
        }
        // Qwen4's few-row decode copies: rows within `Gpu::few_row_limit` run
        // the decode GEMV per row.
        DType::Q8_0 => {
            hip(gpu.gemm_q8_0_batched_f32_chunked(weight.buf, x, y, weight.m, weight.k, batch_size))
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "qt44-qt53-shared-projection-dtype",
            arch: "",
            quant: "unsupported",
        }),
    }
}

/// The grouped route's router is a row-batched projection even though the
/// generic MoE table intentionally has no BF16 router entry on RDNA3.
/// qt=55 FP8: `project_weight` of the BF16 router at every row count (gfx942
/// BF16 MFMA); no fused few-row router/shared launch.
pub(crate) fn router_projection(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
) -> Result<(), DispatchError> {
    if is_fp8(p) {
        return fp8_router_projection(gpu, p);
    }
    require_geometry(p)?;
    // Projected by the shared gate/up stage's launch instead.
    if router_with_shared(gpu, p)
        || router_with_shared_multi(gpu, p)
        || router_with_selector(gpu, p)
        || router_shares_f16(gpu, p)
    {
        return Ok(());
    }
    batch_projection(
        gpu,
        &p.prelude.router,
        p.x_norm_batch,
        p.prelude.router_logits,
        p.batch_size,
    )
}

fn router_shared_bf16_basis(router: &WeightRef<'_>, shared: [&WeightRef<'_>; 3]) -> bool {
    router.dtype == DType::BF16
        && shared
            .iter()
            .all(|w| w.dtype == DType::BF16 && w.k == router.k)
}

/// Whether a few-row forward (route rows within `Gpu::few_row_limit`: the
/// historic 8, or the verify scope's admitted exact cap, where the grouped
/// BF16 row kernel is bitwise each projection's own) projects the router
/// together with the all-BF16 shared selector/gate/up (every K the router's)
/// in one x4 launch (the shared gate/up stage, which the route always runs
/// right after the router stage and before route selection). Row counts keep
/// `route_rows`, never the actual `batch_size`. A missing shared branch or a
/// mismatched K keeps the independent router launch.
fn router_with_shared(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    gpu.route_rows(p.batch_size) <= gpu.few_row_limit()
        && p.prelude.shared.as_ref().is_some_and(|shared| {
            router_shared_bf16_basis(
                &p.prelude.router,
                [
                    &shared.weights.selector,
                    &shared.weights.gate,
                    &shared.weights.up,
                ],
            )
        })
}

/// Whether a few-row forward (route rows within `Gpu::few_row_limit`) on
/// gfx11+ projects the router together with the BF16 shared selector (one K,
/// both reading `x_norm_batch`) in one grouped BF16 row launch when the
/// shared gate/up are not BF16 (the gfx1151 Q8 decode copies, which keep
/// their own separate Q8 x2 launch; the BF16 router/selector weights are
/// never read through a Q8 interpretation). The grouped row kernel computes
/// each projection bitwise as its own launch; the shared gate/up stage runs
/// right after the router stage.
fn router_with_selector(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    (2..=gpu.few_row_limit()).contains(&gpu.route_rows(p.batch_size))
        && gpu.arch_caps.has_gfx11_plus_simt()
        && !router_with_shared(gpu, p)
        && p.prelude.router.dtype == DType::BF16
        && p.x_norm_batch.dtype == DType::F32
        && p.prelude.shared.as_ref().is_some_and(|shared| {
            // Only the shared gate/up stage's mixed-dtype branch projects the
            // fused pair; an all-BF16 shared set (whatever its K) takes the
            // grouped branch, which never issues the router.
            let weights = &shared.weights;
            ![&weights.selector, &weights.gate, &weights.up]
                .iter()
                .all(|w| w.dtype == DType::BF16)
                && weights.selector.dtype == DType::BF16
                && weights.selector.k == p.prelude.router.k
        })
}

/// Whether the unscoped wide route (`route_rows` past `Gpu::few_row_limit`, 16..=64 rows) projects
/// the router together with the all-BF16 shared selector/gate/up in the shared gate/up stage's
/// `project_weights` call (`HIPFIRE_QWEN4_LWMMA_MULTI_PART`): the four members are lane-WMMA
/// projections of the same `x_norm_batch` (none takes the F16 WMMA route, no EP wave split), which
/// `project_weights` runs as ONE multi-part launch, each output the router stage's / the
/// member's own launch's bytes. Like [`router_with_shared`], the stage runs right after the router
/// stage with no writer of `x_norm_batch` or reader of the logits in between.
fn router_with_shared_multi(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    let router = &p.prelude.router;
    gpu.flags.qwen4_lwmma_multi_part_enabled()
        && !is_fp8(p)
        && gpu.route_rows(p.batch_size) > gpu.few_row_limit()
        && p.x_norm_batch.dtype == DType::F32
        && p.prelude.router_logits.dtype == DType::F32
        && gpu.wave_lane_groups(p.batch_size).is_none()
        && p.prelude.shared.as_ref().is_some_and(|shared| {
            let w = &shared.weights;
            router_shared_bf16_basis(router, [&w.selector, &w.gate, &w.up])
                && [shared.scalar, shared.gate_out, shared.up_out]
                    .iter()
                    .all(|t| t.dtype == DType::F32)
                && [router, &w.selector, &w.gate, &w.up]
                    .iter()
                    .all(|m| !gpu.qwen4_f16_wmma_applies(m.buf, m.k, p.batch_size))
                && gpu.lane_wmma_multi_part_applies(
                    &[router.m, w.selector.m, w.gate.m, w.up.m],
                    router.k,
                    p.batch_size,
                )
        })
}

/// Whether the router and the all-BF16 shared selector/gate/up (one K) all
/// take the F16 WMMA route: the shared gate/up stage (which the route always
/// runs right after the router stage, with no writer of `x_norm_batch` in
/// between) then launches the router's own GEMM off the same F16 conversion
/// of `x_norm_batch` instead of converting it a second time.
fn router_shares_f16(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    let router = &p.prelude.router;
    gpu.route_rows(p.batch_size) > gpu.few_row_limit()
        && p.x_norm_batch.dtype == DType::F32
        && router.dtype == DType::BF16
        && gpu.qwen4_f16_wmma_applies(router.buf, router.k, p.batch_size)
        && p.prelude.shared.as_ref().is_some_and(|shared| {
            [
                &shared.weights.selector,
                &shared.weights.gate,
                &shared.weights.up,
            ]
            .iter()
            .all(|w| {
                w.dtype == DType::BF16
                    && w.k == router.k
                    && gpu.qwen4_f16_wmma_applies(w.buf, w.k, p.batch_size)
            })
        })
}

/// Whether this call's shared expert runs the fused gate/up PeaceMaker launch
/// (`HIPFIRE_QWEN4_SHARED_GU_PM`, [`Gpu::qwen4_shared_gu_pm_applies`]): the
/// router and selector on the shared F16 conversion ([`router_shares_f16`]),
/// the BF16 recipe, a BF16 shared down on the F16 WMMA route (which then
/// reads the F16 hidden the launch leaves in `rotated`'s bytes), and not path
/// 2's fused unscatter activation, which reads the F32 gate/up outputs this
/// route does not produce. The one decision of the gate/up, activation and
/// down stages: it depends only on the call's operands and the device.
fn shared_gu_pm(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    router_shares_f16(gpu, p)
        && p.recipe.bf16_round_trip()
        && !(use_path2 && shared_in_unscatter_base(gpu, p))
        && p.prelude.shared.as_ref().is_some_and(|shared| {
            let w = &shared.weights;
            w.gate.m == shared.intermediate
                && w.up.m == shared.intermediate
                && w.up.k == w.gate.k
                && w.down.dtype == DType::BF16
                && w.down.k == shared.intermediate
                && gpu.qwen4_f16_wmma_applies(w.down.buf, w.down.k, p.batch_size)
                && gpu.qwen4_shared_gu_pm_applies(
                    w.gate.buf,
                    w.up.buf,
                    w.gate.m,
                    w.gate.k,
                    p.batch_size,
                )
        })
}

/// The shared down GEMM's input tensor: the F16 hidden the fused gate/up left in
/// `rotated`'s bytes ([`shared_gu_pm`]), else the F32 activation.
fn shared_down_input(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    rotated: &GpuTensor,
    intermediate: usize,
) -> GpuTensor {
    if shared_gu_pm(gpu, p, use_path2) {
        rotated.prefix_as_f16(p.batch_size * intermediate)
    } else {
        rotated.shallow_clone()
    }
}

/// The shared expert's weights as the INT8 route reads them
/// ([`Gpu::qwen4_shared_i8_applies`]).
fn shared_i8_weights<'a>(shared: &MoeSharedPrefill<'a>) -> SharedI8Weights<'a> {
    let w = &shared.weights;
    SharedI8Weights {
        gate: w.gate.buf,
        up: w.up.buf,
        down: w.down.buf,
        intermediate: shared.intermediate,
        hidden: w.gate.k,
        down_rows: w.down.m,
    }
}

/// Whether this call's shared expert runs the INT8 route
/// (`HIPFIRE_QWEN4_SHARED_INT8`, exact gfx1201, [`Gpu::qwen4_shared_i8_applies`]): the
/// router and selector on the F16 WMMA group ([`router_shares_f16`], off the one F16
/// conversion), the BF16 recipe, all-BF16 shared gate/up/down with the router's K, and not
/// path 2's fused unscatter activation (which reads the F32 gate/up outputs this route does
/// not produce).  The gate/up GEMM leaves the BF16 SwiGLU hidden in `gate_out`'s bytes,
/// the activation stage turns it into rotated A8 blocks in `rotated`'s bytes (and applies the
/// selector sigmoid), and the shared down GEMM reads those blocks into its F32 rows.  The one
/// decision of the gate/up, activation and down stages: it depends only on the call's
/// operands and the device.  Never true with the fused gate/up PeaceMaker launch
/// ([`shared_gu_pm`], gfx1151 only).
fn shared_int8(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    router_shares_f16(gpu, p)
        && p.recipe.bf16_round_trip()
        && !(use_path2 && shared_in_unscatter_base(gpu, p))
        && p.prelude.shared.as_ref().is_some_and(|shared| {
            let w = &shared.weights;
            w.gate.m == shared.intermediate
                && w.up.m == shared.intermediate
                && w.up.k == w.gate.k
                && w.down.dtype == DType::BF16
                && w.down.k == shared.intermediate
                && w.down.m == p.down_m
                && w.down.rotation.is_none()
                && w.down.awq_scale.is_none()
                && gpu.qwen4_shared_i8_applies(&shared_i8_weights(shared), p.batch_size)
        })
}

/// Shared selector and gate/up are ordinary batched projections.  This is the
/// typed route's BF16 exception to the generic MoE prefill table: the common
/// family remains fail-closed for QT53 while this exact route uses the native
/// BF16 XF32 GEMM launcher.
/// qt=55 FP8: the shared selector/gate/up through `project_weights` (gfx942
/// BF16 MFMA, all row counts); the router is projected by its own stage.
///
/// `front`: the F16 X the fused front already left in the shared F16 scratch
/// ([`input_basis_front`]); the router + shared WMMA group launch (or the fused shared gate/up
/// PeaceMaker launch's router + selector groups, [`shared_gu_pm`]) then skips its own
/// `convert_f32_to_f16` of `x_norm_batch`.
pub(crate) fn shared_gate_up_front(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    front: Option<&MoeFront>,
) -> Result<(), DispatchError> {
    if front.is_some() && !router_shares_f16(gpu, p) {
        // The fused producer fed exactly the router + shared F16 group launch.
        return Err(DispatchError::Hip(
            "qt44/qt53 fused MoE front: the F16 X has no consumer on this route".into(),
        ));
    }
    if is_fp8(p) {
        return fp8_shared_gate_up(gpu, p);
    }
    require_geometry(p)?;
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let weights = &shared.weights;
    if router_with_shared(gpu, p) || router_with_shared_multi(gpu, p) {
        // All read the natural F32 activation; admitted rows share the exact x4 launch.
        super::layer_ops::project_weights(
            gpu,
            p.x_norm_batch,
            p.batch_size,
            None,
            &[
                (&p.prelude.router, p.prelude.router_logits),
                (&weights.selector, shared.scalar),
                (&weights.gate, shared.gate_out),
                (&weights.up, shared.up_out),
            ],
        )?;
    } else if router_shares_f16(gpu, p) {
        let router = &p.prelude.router;
        if shared_gu_pm(gpu, p, use_path2) {
            // The router and selector groups (their own launches, off one F16
            // conversion), then the gate/up GEMMs, the BF16-recipe activation
            // (selector sigmoid in place) and the hidden's F16 conversion as ONE
            // PeaceMaker launch, the F16 hidden left in `rotated`'s bytes.
            let hidden = shared
                .rotated
                .prefix_as_f16(p.batch_size * shared.intermediate);
            let launched = hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4_shared_gu_pm(
                (router.buf, p.prelude.router_logits, router.m),
                (weights.selector.buf, shared.scalar, weights.selector.m),
                weights.gate.buf,
                weights.up.buf,
                &hidden,
                p.x_norm_batch,
                front.map(|f| &f.x16),
                weights.gate.m,
                router.k,
                p.batch_size,
            ))?;
            if !launched {
                return Err(DispatchError::Hip(
                    "qt44/qt53 router + shared fused gate/up PM route did not apply".into(),
                ));
            }
            return Ok(());
        }
        if shared_int8(gpu, p, use_path2) {
            // The router and selector groups off the one F16 conversion (the shared
            // gate/up never read it), then the INT8 gate/up GEMM and its SwiGLU hidden.
            let launched = hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4_groups_x16(
                &[
                    &[(router.buf, p.prelude.router_logits, router.m)],
                    &[(weights.selector.buf, shared.scalar, weights.selector.m)],
                ],
                p.x_norm_batch,
                front.map(|f| &f.x16),
                router.k,
                p.batch_size,
            ))?;
            if !launched {
                return Err(DispatchError::Hip(
                    "qt44/qt53 router + selector F16 WMMA route did not apply".into(),
                ));
            }
            let w = shared_i8_weights(shared);
            return match front.and_then(|f| f.x8.as_ref()) {
                // The front's one pass already left the A8 blocks of this call's rows.
                Some(x8) => {
                    let xq = hip(gpu.int8_mmq_prepared_ptr(x8, w.hidden, p.batch_size))?;
                    hip(gpu.qwen4_shared_i8_gate_up_prequant(&w, xq, shared.gate_out, p.batch_size))
                }
                None => hip(gpu.qwen4_shared_i8_gate_up(
                    &w,
                    p.x_norm_batch,
                    shared.gate_out,
                    p.batch_size,
                )),
            };
        }
        // The router group first (its own launches, as `router_projection`
        // would issue them), then the shared group, off one F16 conversion.
        let launched = hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4_groups_x16(
            &[
                &[(router.buf, p.prelude.router_logits, router.m)],
                &[
                    (weights.selector.buf, shared.scalar, weights.selector.m),
                    (weights.gate.buf, shared.gate_out, weights.gate.m),
                    (weights.up.buf, shared.up_out, weights.up.m),
                ],
            ],
            p.x_norm_batch,
            front.map(|f| &f.x16),
            router.k,
            p.batch_size,
        ))?;
        if !launched {
            return Err(DispatchError::Hip(
                "qt44/qt53 router + shared F16 WMMA route did not apply".into(),
            ));
        }
    } else if [&weights.selector, &weights.gate, &weights.up]
        .iter()
        .all(|w| w.dtype == DType::BF16)
    {
        // All read the natural activation: one shared F16 conversion.
        super::layer_ops::project_weights(
            gpu,
            p.x_norm_batch,
            p.batch_size,
            None,
            &[
                (&weights.selector, shared.scalar),
                (&weights.gate, shared.gate_out),
                (&weights.up, shared.up_out),
            ],
        )?;
    } else {
        if router_with_selector(gpu, p) {
            super::layer_ops::project_weights(
                gpu,
                p.x_norm_batch,
                p.batch_size,
                None,
                &[
                    (&p.prelude.router, p.prelude.router_logits),
                    (&weights.selector, shared.scalar),
                ],
            )?;
        } else {
            let x_selector = match shared.weights.selector.dtype {
                DType::BF16 | DType::F32 => p.x_norm_batch,
                DType::MQ4G256V2 => p.x_rot_batch,
                _ => {
                    return Err(DispatchError::UnsupportedVariant {
                        family: "moe",
                        variant: "qt44-qt53-shared-selector-dtype",
                        arch: "",
                        quant: "unsupported",
                    })
                }
            };
            batch_projection(
                gpu,
                &shared.weights.selector,
                x_selector,
                shared.scalar,
                p.batch_size,
            )?;
        }
        let x_gate = match shared.weights.gate.dtype {
            DType::BF16 | DType::F32 | DType::Q8_0 => p.x_norm_batch,
            DType::MQ4G256V2 => p.x_rot_batch,
            _ => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "qt44-qt53-shared-gate-dtype",
                    arch: "",
                    quant: "unsupported",
                })
            }
        };
        let (gate, up) = (&shared.weights.gate, &shared.weights.up);
        if gate.dtype == DType::Q8_0
            && up.dtype == DType::Q8_0
            && gate.k == 2560
            && up.k == 2560
            && gpu.gemv_q8_0_staged_rows_supported(2560, gpu.route_rows(p.batch_size))
        {
            // Few-row verify: the Q8 decode copies of the shared gate and up
            // read the same rows; one launch, each row the single kernel's.
            hip(gpu.gemv_q8_0_k2560_staged_rows_x2(
                gate.buf,
                shared.gate_out,
                gate.m,
                up.buf,
                shared.up_out,
                up.m,
                x_gate,
                p.batch_size,
            ))?;
        } else {
            batch_projection(gpu, gate, x_gate, shared.gate_out, p.batch_size)?;
            batch_projection(gpu, up, x_gate, shared.up_out, p.batch_size)?;
        }
    }
    // BF16 recipe: every reader rounds what it reads (the top-10 router its
    // logits, the shared activation its selector, gate and up), so no
    // separate round trip is owed here.
    Ok(())
}

/// Apply the shared expert activation after the selector and gate/up projections.
/// qt=55 FP8: `fp8b128_moe_shared_activation` (BF16 gate/up/SiLU/product
/// rounds, the QT44 kernel's F32 SiLU does not match the teacher); never
/// fused into the unscatter.
pub(crate) fn shared_activation(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    if is_fp8(p) {
        return fp8_shared_activation(gpu, p);
    }
    require_geometry(p)?;
    // Path 2's fused unscatter launch runs it (see `shared_in_unscatter`).
    if use_path2 && shared_in_unscatter(gpu, p) {
        return Ok(());
    }
    // The fused shared gate/up already left the activation (and the sigmoid selector).
    if shared_gu_pm(gpu, p, use_path2) {
        return Ok(());
    }
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    if shared_int8(gpu, p, use_path2) {
        // The INT8 gate/up GEMM left the BF16 SwiGLU hidden in `gate_out`: FWHT-128 + A8
        // blocks into `rotated`, the selector sigmoid in place.
        return hip(gpu.qwen4_shared_i8_hidden(
            &shared_i8_weights(shared),
            shared.gate_out,
            shared.rotated,
            shared.scalar,
            p.batch_size,
        ));
    }
    // The live rows only: the shared buffers are sized for the chunk cap.
    let live = p.batch_size * shared.intermediate;
    let (gate, up, rotated) = (
        f32_view(shared.gate_out, 0, live),
        f32_view(shared.up_out, 0, live),
        f32_view(shared.rotated, 0, live),
    );
    if p.recipe.bf16_round_trip() {
        // The selector's round trip and sigmoid ride in the same launch; the
        // shared down's scaled add rounds the selector again on read.
        hip(gpu.shared_expert_activation_bf16_f32(
            &gate,
            &up,
            &rotated,
            shared.scalar,
            p.batch_size,
        ))?;
    } else {
        hip(gpu.silu_mul_f32(&gate, &up, &rotated))?;
    }
    if shared.weights.down.dtype == DType::MQ4G128V2 {
        hip(gpu.rotate_x_mq_128_v2(
            shared.rotated,
            shared.rotated,
            shared.intermediate,
            p.batch_size,
        ))?;
    }
    Ok(())
}

/// Fold the shared expert into the residual at the recipe's chosen placement,
/// applying the selector sigmoid exactly once.
/// qt=55 FP8: unfused `project_weight` of the BF16 shared down into F32 rows,
/// then `bf16_scaled_add_batched`; no GEMM-epilogue fusion.
pub(crate) fn shared_down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    if is_fp8(p) {
        return fp8_shared_down(gpu, p);
    }
    require_geometry(p)?;
    // The unfused shared down writes F32 rows into `shared_down_out` (an EP
    // row gather owns `down_expanded` until its peers finish reading it), else
    // into `down_expanded`; the BF16 epilogue below may decline at run time, so
    // the fallback is always owed.
    require_shared_rows(p, shared_down_bytes(p.batch_size, p.down_m)?)?;
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    // The fused shared gate/up leaves the activation as F16 in `rotated`'s bytes.
    let hidden = shared_down_input(gpu, p, use_path2, shared.rotated, shared.intermediate);
    if !super::layer_ops::wide_exact_rows(gpu, p.batch_size)
        && down.dtype == DType::BF16
        && p.recipe.bf16_round_trip()
        && !shared_int8(gpu, p, use_path2)
        && hip(gpu.qwen4_shared_down_bf16_epi(
            down.buf,
            &hidden,
            target,
            shared.scalar,
            down.m,
            down.k,
            p.batch_size,
        ))?
    {
        return Ok(());
    }
    // Opt-in fused epilogue (`HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI`): an MQ4G256V2
    // down over the F32 hidden, no BF16 round trip, and no other consumer of the
    // unfused down rows (`shared_down_out` is the EP row gather's / another
    // stage's input). The V3 IU4 ADD entry applies `target += sigmoid(scalar) *
    // down` itself, replacing the temp GEMM and the gated add below. A declined
    // call returns false with no tap and no launch and the incumbent runs.
    if down.dtype == DType::MQ4G256V2
        && hidden.dtype == DType::F32
        && !p.recipe.bf16_round_trip()
        && p.shared_down_out.is_none()
        && super::prefill_shared_down_sigmoid_epi_admitted(gpu, p)
        && hip(gpu.try_gemm_mq4g256v2_shared_down_sigmoid_epi(
            down.buf,
            &hidden,
            None,
            target,
            shared.scalar,
            down.m,
            down.k,
            p.batch_size,
        ))?
    {
        return Ok(());
    }
    let out = f32_view(
        p.shared_down_out.unwrap_or(p.down_expanded),
        0,
        p.batch_size * p.down_m,
    );
    shared_down_gemm(gpu, p, use_path2, down, &hidden, &out)?;
    if p.recipe.bf16_round_trip() {
        // The scaled add rounds `out`, the scalar and the residual to BF16
        // itself and stores a BF16 value, so no round trip brackets it.
        bf16_scaled_add_batched(
            gpu,
            &Bf16ScaledAddBatched {
                residual: target,
                value: &out,
                scalar: shared.scalar,
                rows: p.batch_size,
                elements: p.down_m,
            },
        )
        .map_err(|error| DispatchError::Hip(error.to_string()))?;
    } else {
        hip(gpu.sigmoid_scaled_residual_add_batched_f32(
            target,
            &out,
            shared.scalar,
            p.batch_size,
            p.down_m,
        ))?;
    }
    Ok(())
}

/// The shared-down projection of `hidden` (F32 or the fused F16 hidden) into the F32
/// rows `out` (`[batch_size, down_m]`): the launch [`shared_down`] runs before its
/// scaled add, and the gfx1201 row fold's (its rows land in the routed target instead).
fn shared_down_gemm(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    down: &WeightRef<'_>,
    hidden: &GpuTensor,
    out: &GpuTensor,
) -> Result<(), DispatchError> {
    if let Some(shared) = p.prelude.shared.as_ref() {
        if shared_int8(gpu, p, use_path2) {
            // `hidden` is `rotated`: the A8 blocks the INT8 activation stage left there.
            return hip(gpu.qwen4_shared_i8_down(
                &shared_i8_weights(shared),
                hidden,
                out,
                p.batch_size,
            ));
        }
    }
    match down.dtype {
        DType::BF16 | DType::F32 | DType::Q8_0 | DType::MQ4G256V2 | DType::MQ4G128V2 => {
            // The shared activation is already in the basis consumed by this
            // projection (G128 for QT53); the BF16 storage boundary, if any,
            // was applied by the recipe before this launch.
            if hidden.dtype == DType::F16 {
                // The fused F16 hidden (BF16 down on the F16 WMMA route, see
                // `shared_gu_pm`): the GEMM `batch_projection` would launch off
                // the F32 activation's conversion, minus the conversion.
                hip(gpu.gemm_bf16_xf16_f16_wmma(
                    down.buf,
                    hidden,
                    out,
                    down.m,
                    down.k,
                    p.batch_size,
                ))?;
            } else {
                batch_projection(gpu, down, hidden, out, p.batch_size)?;
            }
            Ok(())
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "qt44-qt53-shared-down-dtype",
            arch: "",
            quant: "unsupported",
        }),
    }
}

/// The HC write that follows a sealed MoE call and may ride the shared down's
/// GEMM epilogue (`HIPFIRE_QWEN4_HC_FUSE` >= 3): its streams, the gates the
/// paired HC read produced, and the tensor it reads as `mixed` (which must be
/// the routed target).
pub(crate) struct HcSharedDown {
    pub streams: GpuTensor,
    pub gates: GpuTensor,
    pub mixed: *mut std::ffi::c_void,
    pub rows: usize,
    pub hidden: usize,
    pub state_bf16: bool,
    /// `HIPFIRE_QWEN4_HC_ROW_FOLD`: the next HC read's operands, when the whole
    /// MoE tail runs as one row kernel (see [`row_fold_applies`]).
    pub row_fold: Option<HcRowFoldNext>,
}

/// The HC read that follows the HC write of a row-folded MoE call, and that
/// read's paired write: the row kernel computes the read's norm (its F16 row)
/// and that write's gate logits.
pub(crate) struct HcRowFoldNext {
    pub norm_weight: GpuTensor,
    pub gate_weight: GpuTensor,
    pub next_gates: GpuTensor,
}

impl HcRowFoldNext {
    /// What the row kernel needs of `read` and its paired `write`.
    pub(crate) fn offer(
        read: &super::layer_ops::HyperReadOp<'_>,
        write: &super::layer_ops::HyperWriteOp<'_>,
    ) -> Self {
        Self {
            norm_weight: f32_view(read.norm_weight, 0, read.norm_weight.numel()),
            gate_weight: f32_view(write.block_inject.buf, 0, write.block_inject.buf.numel()),
            next_gates: f32_view(write.gates, 0, read.rows * write.branches),
        }
    }
}

impl HcSharedDown {
    /// What the shared down needs of `write`, or `None` when its operands
    /// are not the plain four-branch F32-typed streams / gates it folds into.
    /// Always `None` for a write that carries the trunk-shard all-reduce
    /// ([`HyperWriteOp::ar`]): the folded epilogue would write the rank's UNREDUCED partial
    /// into the streams, so the shared down runs unfused and the fused all-reduce write follows.
    pub(crate) fn offer(write: &super::layer_ops::HyperWriteOp<'_>) -> Option<Self> {
        let wide = write.branches.checked_mul(write.hidden)?;
        let (streams_len, gates_len) = (write.rows.checked_mul(wide)?, write.rows.checked_mul(4)?);
        (write.ar.is_none()
            && write.branches == 4
            && write.input.buf.as_ptr() == write.output.buf.as_ptr()
            && write.input.dtype == DType::F32
            && write.input.numel() >= streams_len
            && write.gates.dtype == DType::F32
            && write.gates.numel() >= gates_len)
            .then(|| Self {
                streams: f32_view(write.input, 0, streams_len),
                gates: f32_view(write.gates, 0, gates_len),
                mixed: write.mixed.buf.as_ptr(),
                rows: write.rows,
                hidden: write.hidden,
                state_bf16: write.state_bf16,
                row_fold: None,
            })
    }
}

/// [`shared_down`] with the paired HC write folded into the GEMM epilogue
/// (`hc`).  Bitwise the shared down followed by `hyper_write`: the BF16 shared
/// down on the Qwen4 F16 WMMA route, the BF16 scaled add, then the write.  The
/// routed target is not rewritten (it is dead after the write).  Returns
/// `false`, having launched nothing, when `hc` does not fit this call; the
/// caller then runs [`shared_down`] and the write unfused.
/// Always declines (`Ok(false)`, nothing launched) on the qt=55 FP8 route,
/// whose shared down has no HC epilogue.
pub(crate) fn shared_down_hc(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    hc: &HcSharedDown,
    use_path2: bool,
) -> Result<bool, DispatchError> {
    if is_fp8(p) {
        if hc.row_fold.is_some() {
            return Err(fp8_refuse("the HC row fold is not available"));
        }
        return Ok(false);
    }
    require_geometry(p)?;
    let Some(shared) = p.prelude.shared.as_ref() else {
        return Ok(false);
    };
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    // gfx1201: the row fold replaces the generic shared down's scaled add + the HC write
    // + the next read's norm (its GEMM epilogue levels stay off there); no fold planned
    // means the unfused chain, never the hcsd epilogue.
    if let (true, Some(next)) = (gpu.arch_caps.is_gfx1201(), hc.row_fold.as_ref()) {
        if !row_fold_gfx1201_applies(gpu, p, use_path2, hc) {
            return Err(DispatchError::Hip(
                "qt44 shared down: the gfx1201 HC row fold was planned but its route no longer applies"
                    .into(),
            ));
        }
        shared_down_row_fold_gfx1201(gpu, p, hc, next, use_path2)?;
        return Ok(true);
    }
    if !hc_shared_down_admits(gpu, p, hc) {
        if hc.row_fold.is_some() {
            return Err(DispatchError::Hip(
                "qt44 shared down: the HC row fold was planned but its route no longer applies"
                    .into(),
            ));
        }
        return Ok(false);
    }
    if let Some(next) = &hc.row_fold {
        shared_down_row_fold(gpu, p, hc, next, use_path2)?;
        return Ok(true);
    }
    let hidden = shared_down_input(gpu, p, use_path2, shared.rotated, shared.intermediate);
    hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd(
        down.buf,
        &hidden,
        down.m,
        down.k,
        p.batch_size,
        target,
        shared.scalar,
        &hc.streams,
        &hc.gates,
        hc.state_bf16,
        false,
    ))?;
    Ok(true)
}

/// Whether the shared down of this call can carry `hc` (the conditions of
/// [`shared_down_hc`]).
fn hc_shared_down_admits(gpu: &Gpu, p: &MoePrefillParams<'_>, hc: &HcSharedDown) -> bool {
    let Some(shared) = p.prelude.shared.as_ref() else {
        return false;
    };
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    !super::layer_ops::wide_exact_rows(gpu, p.batch_size)
        && gpu.flags.qwen4_hc_fuse_level() >= 3
        && down.dtype == DType::BF16
        && down.rotation.is_none()
        && down.awq_scale.is_none()
        && p.recipe.bf16_round_trip()
        && p.recipe.shared_after_combine()
        && down.m == p.down_m
        && down.k == shared.intermediate
        && hc.rows == p.batch_size
        && hc.hidden == p.down_m
        && hc.mixed == target.buf.as_ptr()
        && target.numel() >= p.batch_size * p.down_m
        && shared.scalar.numel() >= p.batch_size
        && gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd_applies(down.buf, down.m, down.k, p.batch_size)
}

/// Whether this call's whole tail (combine, shared fold, `hc`'s HC write, the
/// next read's norm) can run as the HC row fold
/// (`HIPFIRE_QWEN4_HC_ROW_FOLD`): the zero-initialized BF16-row combine of
/// path 2 into the routed target (which, nothing reading it, then holds the
/// BF16 shared-down rows), the HC-write shared down, BF16 streams.
/// Never applies on the qt=55 FP8 route.
pub(crate) fn row_fold_applies(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    hc: &HcSharedDown,
) -> bool {
    if is_fp8(p) {
        return false;
    }
    if reduction_partition(p).is_some() {
        // The row kernel combines inside itself, unmasked.
        return false;
    }
    if gpu.arch_caps.is_gfx1201() {
        return row_fold_gfx1201_applies(gpu, p, use_path2, hc);
    }
    let target = p.routed_out.unwrap_or(p.x_batch);
    gpu.flags.qwen4_hc_row_fold_enabled()
        && hc.state_bf16
        && combine_initial_zero_rows_applies(gpu, p, use_path2)
        && hc_shared_down_admits(gpu, p, hc)
        && gpu.hc_row_fold_applies(hc.hidden)
        && p.k_top == 10
        && p.down_expanded.buf.size() >= p.batch_size * 10 * 8
        && target.buf.size() >= p.batch_size * p.down_m * 2
}

/// [`row_fold_applies`] on gfx1201 (`HIPFIRE_QWEN4_HC_ROW_FOLD`, independent of
/// `HIPFIRE_QWEN4_HC_FUSE` >= 2/3): the zero-initialized BF16-row combine of path 2,
/// then the GENERIC shared down (`HIPFIRE_QWEN4_SHARED_DOWN_EPI` off: BF16 weight,
/// F32 rows, no EP row gather reading them) + `bf16_scaled_add_batched`, then the HC
/// write `hc` on BF16 streams -- the launches whose bytes the fold reproduces, with
/// the shared-down rows landing in the (dead) routed target instead of
/// `down_expanded`, which holds the rank order.  Never on the qt=55 FP8 route.
fn row_fold_gfx1201_applies(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    hc: &HcSharedDown,
) -> bool {
    if is_fp8(p) {
        return false;
    }
    let Some(shared) = p.prelude.shared.as_ref() else {
        return false;
    };
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    gpu.flags.qwen4_hc_row_fold_moe_gfx1201_enabled()
        && hc.state_bf16
        && combine_initial_zero_rows_applies(gpu, p, use_path2)
        && !super::layer_ops::wide_exact_rows(gpu, p.batch_size)
        && !gpu.qwen4_shared_down_epi_enabled()
        && p.shared_down_out.is_none()
        && down.dtype == DType::BF16
        && down.rotation.is_none()
        && down.awq_scale.is_none()
        && p.recipe.bf16_round_trip()
        && p.recipe.shared_after_combine()
        && down.m == p.down_m
        && down.k == shared.intermediate
        && hc.rows == p.batch_size
        && hc.hidden == p.down_m
        && hc.mixed == target.buf.as_ptr()
        && target.dtype == DType::F32
        && target.numel() >= p.batch_size * p.down_m
        && shared.scalar.dtype == DType::F32
        && shared.scalar.numel() >= p.batch_size
        && p.k_top == 10
        && p.down_expanded.buf.size() >= p.batch_size * 10 * 8
        && (p.y_down_grouped.buf.as_ptr() as usize) % 16 == 0
        && gpu.hc_row_fold_gfx1201_applies(hc.rows, hc.hidden, &hc.streams, target)
}

/// The row-folded combine stage: only the rank order (the row kernel reads the
/// grouped rows itself).
pub(crate) fn combine_order_only(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    super::combine_leaf(super::CombineArm::RankOrderOnly, reduction_partition(p))?;
    require_geometry(p)?;
    validate_scratch(gpu, p, true, grouped_rows)?;
    // The wide-verify int8 chain's masked down never writes the foreign grouped rows, and this order
    // carries no owner mask on that route: a row fold over them would read unwritten rows. (Unreachable
    // today: a gathered call has a separate shared-down output, which the row fold excludes.)
    if ep_a8_wide(gpu, p, true) && ep_a8_owner_mask(gpu, p)?.is_some() {
        return Err(DispatchError::Hip(
            "qt44 combine order: the owner-masked wide-verify chain has no row-fold order".into(),
        ));
    }
    let owners = implicit_zero_owners(gpu, p);
    hip(gpu.moe_combine_order_top10_owned(
        p.inverse_perm,
        p.topk_indices,
        p.topk_weights,
        p.down_expanded,
        grouped_rows,
        p.batch_size,
        p.n_exp,
        owners,
    ))
}

/// Whether [`combine_stage`] takes the expanded-rows arm (`moe_down_combine_top10_batched`),
/// its final `else`: reached directly on path 1 and through the indexed-down re-entry on
/// path 2.
fn combine_expanded_arm(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    let use_path2 = grouped_path(gpu, p, use_path2);
    indexed_down(gpu, p, use_path2) || !use_path2
}

/// The shared-down F32 hidden and the combine's expanded rows of `p` (views).
fn epilogue_inputs(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Option<(GpuTensor, GpuTensor)> {
    let shared = p.prelude.shared.as_ref()?;
    let hidden = shared_down_input(gpu, p, use_path2, shared.rotated, shared.intermediate);
    let expanded = f32_view(
        p.down_expanded,
        0,
        p.batch_size.checked_mul(p.k_top)?.checked_mul(p.down_m)?,
    );
    Some((hidden, expanded))
}

/// The operands of `moe_shared_down_combine` for `p`: the shared-down weight, the combine's
/// route and its target (the routed target the shared fold lands in).
fn epilogue_operands<'a>(
    p: &'a MoePrefillParams<'_>,
    hidden: &'a GpuTensor,
    expanded: &'a GpuTensor,
    initial_zero: bool,
) -> Option<rdna_compute::tensor_ops::MoeSharedDownCombine<'a>> {
    let shared = p.prelude.shared.as_ref()?;
    let down = &shared.weights.down;
    Some(rdna_compute::tensor_ops::MoeSharedDownCombine {
        weight: down.buf,
        x: hidden,
        target: p.routed_out.unwrap_or(p.x_batch),
        initial_zero,
        expert_outputs: expanded,
        topk_indices: p.topk_indices,
        topk_weights: p.topk_weights,
        scalar: shared.scalar,
        m: down.m,
        k: down.k,
        rows: p.batch_size,
        n_experts: p.n_exp,
    })
}

/// Whether this call's combine + shared down + shared fold (`HIPFIRE_QWEN4_MOE_EPILOGUE_FUSE`)
/// run as the one launch `moe_shared_down_combine_rows_r<rows>` ([`shared_down_combined`]):
/// the expanded-rows combine arm, then the BF16 shared down on the few-row F32 rows GEMV
/// (`gemv_bf16_xf32_x4_rows`, or the one-row GEMV) and `bf16_scaled_add_batched`, bytewise
/// (NaN payload order aside) the chain the launch replaces; `target` is written exactly as
/// the chain leaves it, so nothing downstream changes. Pure: launches nothing. The caller
/// owns the stage-level conditions (no CPU split / deferral / tile exchange / row-fold write).
pub(crate) fn epilogue_fuse_applies(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    // The fused launch combines the ten expanded rows itself, unmasked: a partitioned call
    // declines it and runs the partitioned combine + the unfused shared down.
    if is_fp8(p) || !grouped_geometry(p) || reduction_partition(p).is_some() {
        return false;
    }
    let Some(shared) = p.prelude.shared.as_ref() else {
        return false;
    };
    let down = &shared.weights.down;
    let rows = p.batch_size;
    // The rows GEMV route of `project_weights_inner`: 2..=few_row_limit route rows on gfx11+
    // (eight at most here), else the one-row GEMV; never the F16 WMMA GEMM.
    let few_row_gemv = gpu.route_rows(rows) == rows
        && !super::layer_ops::wide_exact_rows(gpu, rows)
        && (rows == 1 || (rows <= gpu.few_row_limit() && gpu.arch_caps.has_gfx11_plus_simt()));
    few_row_gemv
        && down.dtype == DType::BF16
        && down.rotation.is_none()
        && down.awq_scale.is_none()
        && p.recipe.bf16_round_trip()
        && p.recipe.shared_after_combine()
        && down.m == p.down_m
        && down.k == shared.intermediate
        && p.k_top == 10
        && combine_expanded_arm(gpu, p, use_path2)
        && !gpu.qwen4_shared_down_epi_enabled()
        && !shared_int8(gpu, p, use_path2)
        && !gpu.qwen4_f16_wmma_applies(down.buf, down.k, rows)
        && epilogue_inputs(gpu, p, use_path2)
            .and_then(|(hidden, expanded)| {
                epilogue_operands(p, &hidden, &expanded, false).map(|o| o.applies(gpu))
            })
            .unwrap_or(false)
}

/// [`combine`] of a call whose combine rides the shared down ([`epilogue_fuse_applies`]):
/// nothing launches, the scratch preflight still runs.
pub(crate) fn combine_deferred(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    super::combine_leaf(
        super::CombineArm::DeferredToSharedDown,
        reduction_partition(p),
    )?;
    require_geometry(p)?;
    validate_scratch(gpu, p, use_path2, grouped_rows)
}

/// [`shared_down`] of a call whose combine was deferred ([`combine_deferred`]): the one launch
/// that combines the ten expanded rows into the target, computes the shared-down rows and
/// folds them in with the selector (`bf16_scaled_add_batched`'s expression).
/// `initial_zero` (`HIPFIRE_QWEN4_FFN_CLEAR_ABSORB_ROWS`): the target's zero fill was absorbed, so
/// the launch is the `_zinit` entry that seeds +0.0 and stores every `(token, column)` instead of
/// reading the filled target; admission ([`epilogue_fuse_applies`]) is independent of it.
pub(crate) fn shared_down_combined(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    initial_zero: bool,
) -> Result<(), DispatchError> {
    super::combine_leaf(
        super::CombineArm::DeferredToSharedDown,
        reduction_partition(p),
    )?;
    require_geometry(p)?;
    let (hidden, expanded) = epilogue_inputs(gpu, p, use_path2).ok_or_else(|| {
        DispatchError::Hip("qt44 epilogue fuse: shared weights vanished after admission".into())
    })?;
    let operands = epilogue_operands(p, &hidden, &expanded, initial_zero).ok_or_else(|| {
        DispatchError::Hip("qt44 epilogue fuse: shared weights vanished after admission".into())
    })?;
    hip(rdna_compute::tensor_ops::moe_shared_down_combine(
        gpu, &operands,
    ))
}

/// The row-folded shared down: the shared-down GEMM stores its BF16 rows into
/// the (otherwise unwritten) routed target, then one row kernel per token
/// combines the ten expert rows, folds the shared row, writes the HC streams
/// and runs the next read's norm + gate projection.  Bitwise
/// [`combine`] (`initial_zero`) + [`shared_down_hc`] + the next read's
/// `hyper_norm_gate_outputs`.
fn shared_down_row_fold(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    hc: &HcSharedDown,
    next: &HcRowFoldNext,
    use_path2: bool,
) -> Result<(), DispatchError> {
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    let hidden = shared_down_input(gpu, p, use_path2, shared.rotated, shared.intermediate);
    // The next read's F16 row lives in the shared F16 scratch, which the
    // shared-down GEMM's own F16 input also uses (it is dead by the time the
    // row kernel writes): size it first so nothing reallocates under the GEMM.
    let x16 = hip(gpu.qwen4_f16_x_scratch(hc.rows * 4 * hc.hidden))?;
    hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4_bf16st(
        down.buf,
        &hidden,
        down.m,
        down.k,
        p.batch_size,
        target,
    ))?;
    hip(
        gpu.hc_row_fold_norm_gate(&rdna_compute::hc_row_fold::HcRowFold {
            grouped_down: p.y_down_grouped,
            order: p.down_expanded,
            shared_bf16: target,
            selector: shared.scalar,
            streams: &hc.streams,
            gates: &hc.gates,
            norm_weight: &next.norm_weight,
            gate_weight: &next.gate_weight,
            next_gates: &next.next_gates,
            normalized_f16: &x16,
            rows: hc.rows,
            hidden: hc.hidden,
        }),
    )
}

/// The gfx1201 row-folded shared down: the generic shared-down projection, unchanged,
/// into the routed target as F32 rows, then one row kernel per token combines the ten
/// expert rows, applies the BF16 scaled add, writes the HC streams and runs the next
/// read's norm + gate projection.  Bytewise [`combine`] (`initial_zero`) +
/// [`shared_down`] + the HC write + the next read's `hyper_norm_gate_outputs`.
fn shared_down_row_fold_gfx1201(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    hc: &HcSharedDown,
    next: &HcRowFoldNext,
    use_path2: bool,
) -> Result<(), DispatchError> {
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    let hidden = shared_down_input(gpu, p, use_path2, shared.rotated, shared.intermediate);
    // The next read's F16 row lives in the shared F16 scratch, which the shared-down
    // GEMM's own F16 input also uses (dead once the GEMM ran): size it first so
    // nothing reallocates under the GEMM.
    let x16 = hip(gpu.qwen4_f16_x_scratch(hc.rows * 4 * hc.hidden))?;
    let rows = f32_view(target, 0, p.batch_size * p.down_m);
    shared_down_gemm(gpu, p, use_path2, down, &hidden, &rows)?;
    hip(
        gpu.hc_row_fold_moe_gfx1201(&rdna_compute::hc_row_fold::HcRowFoldMoeGfx1201 {
            grouped_down: p.y_down_grouped,
            order: p.down_expanded,
            shared_f32: &rows,
            selector: shared.scalar,
            streams: &hc.streams,
            gates: &hc.gates,
            norm_weight: &next.norm_weight,
            gate_weight: &next.gate_weight,
            next_gates: &next.next_gates,
            normalized_f16: &x16,
            rows: hc.rows,
            hidden: hc.hidden,
        }),
    )
}

/// Whether path 2 takes the symmetric IU4 arm (fn-moe-sym): the layer's
/// experts were verified symmetric at load (the policy says so), the recipe
/// keeps the BF16 boundaries the kernels implement, the activation is F32,
/// and the device admits it (`HIPFIRE_QWEN4_MOE_SYM_IU4` through the arch's symmetric IU4
/// descriptor: default on for gfx1100, gfx1151 and gfx1201, `=0` off; >= 512 rows, C2
/// producers). It replaces scatter,
/// gate/up, unscatter/rotation and down; the combine reads its BF16 rows like
/// the F16 WMMA arm's.
fn sym_iu4(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    if is_slab(p) {
        return use_path2 && slab_k2(gpu, p);
    }
    !super::layer_ops::wide_exact_moe_rows(gpu, p.batch_size)
        && use_path2
        && p.route_policy
            .is_some_and(|policy| policy.capability == MoeRouteCapability::Qt44Qt53GroupedSymmetric)
        && p.recipe.bf16_round_trip()
        && p.x_norm_batch.dtype == DType::F32
        && p.down_k == p.mi
        && gpu.qwen4_moe_sym_iu4_applies(p.batch_size)
}

/// Whether `p` has the Flash-Next geometry the wide-verify int8 chain's GEMMs serve: gate/up
/// `M = 2 * 640`, `K = 2560`, down `M = 2560`, `K = 640`, top-10. Pure.
fn ep_a8_wide_geometry(
    gate_up_k: usize,
    mi: usize,
    down_k: usize,
    down_m: usize,
    k_top: usize,
) -> bool {
    gate_up_k == 2560 && mi == 640 && down_k == 640 && down_m == 2560 && k_top == 10
}

/// Whether a layer declaring `capability` on a device whose experts are drop-zero planar
/// (`planar`, [`Gpu::mq4_experts_planar`]) has the symmetric experts the int8 chain's `_sym`
/// GEMMs need. Below the IU4 row floor a verified layer still declares the plain
/// `Qt44Qt53Grouped` (the loader couples the symmetric variant to that floor), so at 9..=64 rows
/// the verdict is the planar layout itself: the host repack verified `z == -8 s` on every header
/// and dropped the zero plane, and the incumbent planar kernels (g32 gate/up, PM A8 down) read no
/// zero point either. Pure.
fn ep_a8_wide_symmetric(capability: MoeRouteCapability, planar: bool) -> bool {
    planar
        && matches!(
            capability,
            MoeRouteCapability::Qt44Qt53Grouped | MoeRouteCapability::Qt44Qt53GroupedSymmetric
        )
}

/// Whether the shared expert's selector / gate / up read the FWHT-rotated basis `x_rot_batch`
/// (only the F32 `MQ4G256V2` projections do; BF16, F32 and Q8_0 read `x_norm_batch`). The one
/// conditional reader of `x_rot_batch` on the wide-verify int8 route
/// ([`shared_gate_up_front`]); the router reads `x_norm_batch`.
fn shared_reads_x_rot(p: &MoePrefillParams<'_>) -> bool {
    p.prelude.shared.as_ref().is_some_and(|shared| {
        [
            &shared.weights.selector,
            &shared.weights.gate,
            &shared.weights.up,
        ]
        .iter()
        .any(|w| w.dtype == DType::MQ4G256V2)
    })
}

/// Whether path 2 takes the wide-verify int8 chain of the Flash-Next EP2 decode / verify MoE
/// (`HIPFIRE_QWEN4_MOE_GU_A8`, default on, exact `0` off): `qwen4_moe_rotate256_i8` + the signed-IU8
/// gate/up with the SwiGLU epilogue, `qwen4_moe_rotate128_i8` + the signed-IU8 down, BF16 grouped
/// rows, and the BF16-row grouped combine, in place of the g32 gate/up, the unscatter + FWHT128,
/// the PM A8 down, the down unscatter and the expanded top-10 combine. All of:
/// * [`Gpu::qwen4_moe_gu_a8_applies`]: exact gfx1201, the switch not `0`, eager (no capture or
///   replay record), `batch_size` source rows in the g32 window 9..=64;
/// * the grouped path 2 at a route above the few-row limit (not the indexed per-slot down, not
///   the wide-exact singleton route), not FP8 / slab, not the symmetric IU4 route (prefill keeps
///   its own A4 / A8 chain), not the F16 WMMA arms (gfx11);
/// * drop-zero planar QT44/QT53 experts ([`Gpu::mq4_experts_planar`]: the EP decode tables,
///   symmetric by construction, [`ep_a8_wide_symmetric`]), the BF16 recipe, an F32
///   `x_norm_batch`, and the Flash-Next geometry ([`ep_a8_wide_geometry`]);
/// * an EP row-gathered call whose gather is a no-op ([`MoeRowGather::gather_skipped`]: the trunk
///   shard's partial FFN, owned-slot partial completed by the HC all-reduce, non-owned experts on
///   the zero dummy tables). Owner-masked ([`ep_a8_owner_mask`], default) both A8 GEMMs skip the
///   tiles of every non-owned expert and the combine folds only owned rows (a foreign rank adds
///   +0); unmasked (`HIPFIRE_QWEN4_MOE_GU_A8=2`, or a gather without a mask) every grouped row is
///   written, +0 for dummy tiles. Both are byte-identical. A call that gathers (the MTP head, a
///   non-shard EP, a balanced / partitioned layer) needs F32 slot-ordered rows and keeps the
///   incumbent chain;
/// * no tile exchange, no CPU split / deferral (their consumers need F32 slot-ordered rows).
pub(crate) fn ep_a8_wide(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    let use_path2 = grouped_path(gpu, p, use_path2);
    use_path2
        && !is_fp8(p)
        && !is_slab(p)
        && gpu.qwen4_moe_gu_a8_applies(p.batch_size)
        && !super::layer_ops::wide_exact_moe_rows(gpu, p.batch_size)
        && !indexed_down(gpu, p, use_path2)
        && p.route_policy
            .is_some_and(|policy| ep_a8_wide_symmetric(policy.capability, gpu.mq4_experts_planar()))
        && p.recipe.bf16_round_trip()
        && p.x_norm_batch.dtype == DType::F32
        && ep_a8_wide_geometry(p.gate_up_k, p.mi, p.down_k, p.down_m, p.k_top)
        && !sym_iu4(gpu, p, use_path2)
        && !down_wmma(gpu, p)
        && !gateup_bf16(gpu, p)
        && p.ep_tile_exchange.is_none()
        && !p.cpu_deferred_experts
        && p.routed_experts.cpu_split().is_none()
        && p.ep_row_gather
            .is_some_and(|gather| gather.gather_skipped(p.ep_layer_idx))
}

/// The owner mask of this call's wide-verify int8 chain ([`ep_a8_wide`] is the caller's gate): the
/// layer's gather hook ([`crate::families::moe::MoeRowGather::decode_owner_mask`]: the static `e % ranks`
/// ownership of the rank's compact decode table, real owned experts and zero dummies for the rest)
/// unless `HIPFIRE_QWEN4_MOE_GU_A8=2` ([`Gpu::qwen4_moe_gu_a8_masked`]) asks for the unmasked A/B.
/// `None`: the chain runs over unmasked tiles. A table over another expert count is an error.
fn ep_a8_owner_mask<'p>(
    gpu: &Gpu,
    p: &'p MoePrefillParams<'_>,
) -> Result<Option<MoeDecodeOwnerMask<'p>>, DispatchError> {
    if !gpu.qwen4_moe_gu_a8_masked() {
        return Ok(None);
    }
    let Some(mask) = p
        .ep_row_gather
        .and_then(|gather| gather.decode_owner_mask(p.ep_layer_idx))
    else {
        return Ok(None);
    };
    if mask.n_experts != p.n_exp {
        return Err(DispatchError::Hip(format!(
            "qt44 wide-verify owner mask: the owner table covers {} experts, the layer has {}",
            mask.n_experts, p.n_exp
        )));
    }
    Ok(Some(mask))
}

/// Whether [`scatter`] would launch the plain `moe_scatter_fused_top10` for this call, with
/// nothing else to do and no error to raise: the fused router + scatter launch
/// (`HIPFIRE_QWEN4_MOE_ROUTE_SCATTER`) replaces exactly that launch. Mirrors [`scatter`]'s own
/// branches (FP8, geometry, wide exact rows, symmetric IU4, owner-routed tile exchange); the
/// expert IU4 verdict is also left to the scatter stage, so a bound IU4 plan declines.
pub(crate) fn scatter_fusable(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    !is_fp8(p)
        && grouped_geometry(p)
        && p.expert_iu4.is_none()
        && !super::layer_ops::wide_exact_rows(gpu, p.batch_size)
        && !sym_iu4(gpu, p, true)
        && !(p.ep_tile_exchange.is_some() && wmma_tile_route(gpu, p, true))
}

/// Path-2 grouping (the stage runs on path 2 only).
/// qt=55 FP8: stable pad16 grouping, then the one tile descriptor both FP8
/// GEMMs read ([`fp8_scatter`]).
pub(crate) fn scatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    if is_fp8(p) {
        return fp8_scatter(gpu, p, grouped_rows);
    }
    require_geometry(p)?;
    if super::layer_ops::wide_exact_moe_rows(gpu, p.batch_size) {
        return Ok(());
    }
    let total_slots = p.batch_size * p.k_top;
    if sym_iu4(gpu, p, true) || (p.ep_tile_exchange.is_some() && wmma_tile_route(gpu, p, true)) {
        if is_slab(p) {
            // Sealed expert-slab: the arch-neutral entry (gfx1201 delegates to
            // the unchanged scatter, gfx1100 runs its own).
            return hip(gpu.moe_scatter_stable_top10_slab(
                p.topk_indices,
                p.expert_token_counts,
                p.expert_offsets,
                p.sorted_slot_index,
                p.expert_tile_ids,
                p.inverse_perm,
                total_slots,
                p.n_exp,
                grouped_rows,
            ));
        }
        return hip(gpu.moe_scatter_stable_top10(
            p.topk_indices,
            p.expert_token_counts,
            p.expert_offsets,
            p.sorted_slot_index,
            p.expert_tile_ids,
            p.inverse_perm,
            total_slots,
            p.n_exp,
            grouped_rows,
        ));
    }
    hip(gpu.moe_scatter_fused_top10(
        p.topk_indices,
        p.expert_token_counts,
        p.expert_offsets,
        p.sorted_slot_index,
        p.expert_tile_ids,
        p.inverse_perm,
        total_slots,
        p.n_exp,
        grouped_rows,
        crate::families::moe::MOE_GROUPED_BLOCK_M,
    ))
}

/// The owner mask a scatter-side glue-fold launch (`HIPFIRE_QWEN4_MOE_GLUE_FOLD`, X8a / X8b) writes
/// together with the scatter tables: the wide-verify int8 chain's owner mask ([`ep_a8_owner_mask`]) of
/// a call that takes that chain ([`ep_a8_wide`]); `None` for every other call.
pub(crate) fn glue_fold_owner_mask<'p>(
    gpu: &Gpu,
    p: &'p MoePrefillParams<'_>,
) -> Result<Option<MoeDecodeOwnerMask<'p>>, DispatchError> {
    if !ep_a8_wide(gpu, p, true) {
        return Ok(None);
    }
    ep_a8_owner_mask(gpu, p)
}

/// [`scatter`], with the owner tile mask of the wide-verify int8 chain written by the same launch when
/// `HIPFIRE_QWEN4_MOE_GLUE_FOLD` selects X8a and the call takes the plain grouped scatter
/// ([`scatter_fusable`]) on that chain. Returns whether the masked copy of the tile ids
/// (`tile_ids_owned`) has been written: the gate/up stage then does not launch `ep_tile_mask`.
pub(crate) fn scatter_masked(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<bool, DispatchError> {
    if gpu.moe_glue_fold_mask_admitted() && scatter_fusable(gpu, p) {
        require_geometry(p)?;
        if let Some(mask) = glue_fold_owner_mask(gpu, p)? {
            hip(gpu.moe_scatter_fused_top10_pmask(
                p.topk_indices,
                p.expert_token_counts,
                p.expert_offsets,
                p.sorted_slot_index,
                p.expert_tile_ids,
                p.inverse_perm,
                p.batch_size * p.k_top,
                p.n_exp,
                grouped_rows,
                crate::families::moe::MOE_GROUPED_BLOCK_M,
                rdna_compute::moe::TileMaskFold {
                    tile_ids_owned: mask.tile_ids_owned,
                    owner_table: mask.owner_table,
                    tiles: grouped_tiles(grouped_rows),
                    rank: mask.rank,
                },
            ))?;
            return Ok(true);
        }
    }
    scatter(gpu, p, grouped_rows)?;
    Ok(false)
}

/// Whether the grouped gate/up takes the F16 WMMA arm and hands the unscatter
/// BF16 rows: the recipe rounds gate/up to BF16 before SiLU anyway.  The
/// BF16-output entries exist in the gfx11 source only.
fn gateup_bf16(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    !super::layer_ops::wide_exact_moe_rows(gpu, p.batch_size)
        && !is_slab(p)
        && p.recipe.bf16_round_trip()
        && gpu.arch_caps.has_wmma_w32()
        && gpu.qwen4_moe_gateup_wmma_applies(2 * p.mi, p.gate_up_k, p.batch_size)
}

/// Whether the grouped down takes the F16 WMMA arm: rotated straight to F16 in
/// `down`, output as BF16 for the combine, which rounds every row to BF16.
fn down_wmma(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    !super::layer_ops::wide_exact_moe_rows(gpu, p.batch_size)
        && !is_slab(p)
        && p.down_k == p.mi
        && gpu.qwen4_moe_down_wmma_applies(p.down_m, p.down_k, p.batch_size * p.k_top)
}

/// The tile ids the grouped symmetric GEMMs read: the scatter's `expert_tile_ids`, or under the
/// owner-routed EP chunk exchange its masked copy (`-1` on every tile of an expert another rank
/// owns).
fn gemm_tile_ids<'p>(p: &'p MoePrefillParams<'_>) -> &'p GpuTensor {
    p.ep_tile_exchange
        .map_or(p.expert_tile_ids, |tx| tx.tile_ids_owned)
}

/// Whether path 2 takes the F16 WMMA grouped route end to end with BF16 grouped rows: SwiGLU
/// in the gate/up epilogue (`gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out` on gfx1100), the
/// down's own unscatter + rotation, the grouped WMMA down with BF16 rows
/// (`gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out`, 16-row tiles of `down_m * 2` bytes) and
/// the BF16-input combine. Not the symmetric IU4 arm, which replaces all of it.
fn wmma_tile_route(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    use_path2 && !sym_iu4(gpu, p, use_path2) && gateup_silu(gpu, p)
}

/// The routes that move peer-owned BF16 grouped down rows between ranks: the symmetric IU4
/// arm or the F16 WMMA grouped route. Both write the BF16 grouped rows in the same
/// `y_down_grouped` layout, from tile ids the masked copy replaces, over the stable grouping
/// every rank shares.
fn tile_exchange_route(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    sym_iu4(gpu, p, true) || wmma_tile_route(gpu, p, true)
}

fn tile_exchange_eager_only(gpu: &Gpu, what: &str) -> Result<(), DispatchError> {
    // The exchange waits on peer devices and a host barrier; neither can be
    // recorded into a graph or retained body. EP is eager-only.
    if gpu.graphs.capture_mode
        || gpu.graphs.verify.capturing.is_some()
        || gpu.graphs.replay.capturing.is_some()
        || gpu.replay.is_recording()
    {
        return Err(DispatchError::Hip(format!(
            "sealed moe: expert-parallel tile {what} refused while a graph capture or replay record is armed"
        )));
    }
    Ok(())
}

/// Padded tile capacity of a grouped layout (the symmetric GEMMs' grid height).
fn grouped_tiles(grouped_rows: usize) -> usize {
    grouped_rows.div_ceil(16)
}

/// Owner-routed EP chunk MoE, after the scatter: write the masked copy of the tile ids (the
/// owner of every expert is the exchange's table input). Valid only on the symmetric IU4 arm
/// or the F16 WMMA grouped route.
pub(crate) fn tile_mask(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    tx: MoeTileExchangeParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    if !tile_exchange_route(gpu, p) {
        return Err(DispatchError::Hip(
            "qt44 tile mask: the owner-routed chunk exchange needs the symmetric IU4 arm or the F16 WMMA grouped route".into(),
        ));
    }
    let (rank, table, n_exp) = tx.exchange.owners();
    if n_exp != p.n_exp {
        return Err(DispatchError::Hip(format!(
            "qt44 tile mask: the owner table covers {n_exp} experts, the layer has {}",
            p.n_exp
        )));
    }
    let tiles = grouped_tiles(grouped_rows);
    let mask_only = tx.exchange.mask_only();
    if !mask_only {
        tile_exchange_eager_only(gpu, "mask")?;
    }
    hip(gpu.ep_tile_mask(
        p.expert_tile_ids,
        tx.tile_ids_owned,
        table,
        tiles,
        n_exp,
        rank,
    ))?;
    if mask_only {
        // Partial mask (trunk shard): the masked tiles' grouped down rows are never written
        // (the GEMMs exit on the sentinel) and nothing is pulled into them, so they must read
        // as exact +0 BF16 rows, what the zero-dummy-table route wrote. Owned tiles overwrite
        // their rows in the down GEMM that follows. Under the implicit zeros the combine's
        // rank order marks every foreign row -1 (its consumers substitute the +0 row), so
        // nothing reads them and the clear is not owed.
        if implicit_zero_owners(gpu, p).is_none() {
            fill_grouped_down(gpu, p, grouped_rows, 0)?;
        } else if gpu.flags.qwen4_ep_pmask_implicit_zero_poison() {
            // Negative control: all-ones bytes (BF16 NaN) in every row nothing may read.
            fill_grouped_down(gpu, p, grouped_rows, 0xFF)?;
        }
    }
    Ok(())
}

/// `HIPFIRE_QWEN4_EP_PMASK_IMPLICIT_ZERO` (exact gfx1201, default on, `=0` kill switch): the
/// owner table of a partial tile mask (the trunk shard's mask-only chunk route, no exchange)
/// whose foreign grouped down rows only the rank order of `moe_combine_order_top10` would
/// read: the BF16-row combine arm or a HC row fold. The order then marks every rank of a
/// foreign expert `-1` with its weight and place kept, the consumers add a +0 BF16 row for
/// it (bitwise what the cleared row gave) and [`tile_mask`] drops the full-buffer clear.
/// `None` keeps the clear: no mask, a full exchange, the kill switch, or any other reader
/// (FP8, slab, indexed down, the EP grouped-down unscatter).
fn implicit_zero_owners<'p>(
    gpu: &Gpu,
    p: &'p MoePrefillParams<'_>,
) -> Option<rdna_compute::hc_row_fold::CombineOwners<'p>> {
    let tx = p.ep_tile_exchange.filter(|tx| tx.exchange.mask_only())?;
    if !gpu.flags.qwen4_ep_pmask_implicit_zero_enabled() || is_fp8(p) || is_slab(p) {
        return None;
    }
    let use_path2 = grouped_path(gpu, p, true);
    let bf16_rows_combine = use_path2
        && !indexed_down(gpu, p, use_path2)
        && !ep_grouped_down_unscatter(gpu, p, use_path2)
        && (down_wmma(gpu, p) || sym_iu4(gpu, p, use_path2));
    let (rank, table, n_exp) = tx.exchange.owners();
    (bf16_rows_combine && n_exp == p.n_exp)
        .then_some(rdna_compute::hc_row_fold::CombineOwners { table, rank })
}

/// Bytes of the BF16 grouped down rows `[grouped_rows, down_m]` the partial mask clears.
fn grouped_down_bytes(grouped_rows: usize, down_m: usize) -> Option<usize> {
    grouped_rows
        .checked_mul(down_m)?
        .checked_mul(std::mem::size_of::<u16>())
}

/// Stream-ordered byte fill (`value`: 0 is the clear) of `y_down_grouped[..grouped_rows]`
/// (BF16 rows).
fn fill_grouped_down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
    value: i32,
) -> Result<(), DispatchError> {
    let bytes = grouped_down_bytes(grouped_rows, p.down_m)
        .filter(|&bytes| bytes <= p.y_down_grouped.buf.size())
        .ok_or_else(|| {
            DispatchError::Hip(format!(
                "qt44 tile mask: {grouped_rows} grouped rows of {} BF16 exceed the grouped down buffer",
                p.down_m
            ))
        })?;
    match gpu.active_stream.as_ref() {
        Some(stream) => hip(gpu
            .hip
            .memset_async(&p.y_down_grouped.buf, value, bytes, stream)),
        None => hip(gpu.hip.memset(&p.y_down_grouped.buf, value, bytes)),
    }
}

/// Owner-routed EP chunk MoE, between the mutation fence and the combine: fill every
/// peer-owned live tile of `y_down_grouped` from its owner.
pub(crate) fn tile_exchange(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    tx: MoeTileExchangeParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    if !tile_exchange_route(gpu, p) {
        return Err(DispatchError::Hip(
            "qt44 tile exchange: the owner-routed chunk exchange needs the symmetric IU4 arm or the F16 WMMA grouped route".into(),
        ));
    }
    tile_exchange_eager_only(gpu, "exchange")?;
    tx.exchange
        .exchange(
            gpu,
            p.ep_layer_idx,
            p.expert_tile_ids,
            p.y_down_grouped,
            grouped_tiles(grouped_rows),
        )
        .map_err(|error| DispatchError::Hip(format!("qt44 tile exchange: {error}")))
}

/// Whether gate/up stores the SwiGLU activation (BF16) straight from its
/// epilogue and `down` unscatters it fused with the 128-wide rotation.
fn gateup_silu(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    gateup_bf16(gpu, p) && down_wmma(gpu, p) && p.mi % 128 == 0
}

/// Whether this layer's routed experts are read from host-mapped memory (spilled
/// past the VRAM budget); the symmetric IU4 GEMMs then take a wider expert-run
/// tile. This is the layer-level bit the caller derives from the expert set's
/// own residency (a per-expert tier layer: true iff ANY expert is host-tier;
/// never inferred from expert 0). A staged layer reads only VRAM (host-tier
/// experts copied to the stage, device-tier entries at their owner address),
/// so staging clears the bit.
fn experts_host_mapped(p: &MoePrefillParams<'_>) -> bool {
    p.expert_stage_ptrs.is_none() && p.routed_experts.host_mapped()
}

/// Routed gate/up GEMM.  qt=55 FP8: one `gemm_fp8b128_moe_grouped_bf16mfma`
/// over the tile descriptor, F32 `[G, 2 mi]` grouped rows ([`fp8_gate_up`]).
pub(crate) fn gate_up(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    gate_up_front(gpu, p, use_path2, grouped_rows, None, false)
}

/// [`gate_up`] consuming the IU4 sidecar the fused front already produced
/// ([`input_basis_front`]) instead of running `mq_rotate_x_i4`. A sidecar that is no longer
/// the live scratch (generation, shape or pointer mismatch) is never read: the producer
/// runs again, as the unfused chain would.
/// `tile_mask_written`: the scatter-side launch of this call already wrote the owner-masked tile ids
/// of the wide-verify int8 chain (`HIPFIRE_QWEN4_MOE_GLUE_FOLD`, [`scatter_masked`]); the chain's own
/// `ep_tile_mask` launch is skipped.
pub(crate) fn gate_up_front(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
    front: Option<MoeFront>,
    tile_mask_written: bool,
) -> Result<(), DispatchError> {
    if is_fp8(p) {
        return fp8_gate_up(gpu, p, grouped_rows);
    }
    require_geometry(p)?;
    let use_path2 = grouped_path(gpu, p, use_path2);
    validate_scratch(gpu, p, use_path2, grouped_rows)?;
    let gate_up_ptrs = p
        .expert_stage_ptrs
        .map_or(p.expert_gate_up_ptrs, |stage| stage.gate_up);
    if is_slab(p) {
        return slab_gate_up(gpu, p, grouped_rows);
    }
    if sym_iu4(gpu, p, use_path2) {
        if gpu.qwen4_moe_ep_a8_requested() {
            // HIPFIRE_QWEN4_MOE_EP_A8: the chain's int8 per-128 twin (FWHT256 + A8 producer, signed-IU8
            // gate/up with the same BF16 SwiGLU epilogue). The fused front's A4 sidecar is not read.
            let xq = hip(gpu.qwen4_moe_rotate256_i8(p.x_norm_batch, p.gate_up_k, p.batch_size))?;
            return hip(gpu.gemm_qwen4_moe_gate_up_silu_a8_sym(
                gate_up_ptrs,
                gemm_tile_ids(p),
                p.sorted_slot_index,
                &xq,
                p.y_gate_up_grouped,
                2 * p.mi,
                p.gate_up_k,
                p.k_top,
                grouped_rows,
                p.batch_size,
            ));
        }
        let xq = match front {
            Some(front)
                if gpu
                    .int4_mmq_prepared_ptr(&front.xq, p.gate_up_k, p.batch_size)
                    .is_ok() =>
            {
                front.xq
            }
            _ => hip(gpu.qwen4_moe_rotate256_i4(p.x_norm_batch, p.gate_up_k, p.batch_size))?,
        };
        return hip(gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
            gate_up_ptrs,
            gemm_tile_ids(p),
            p.sorted_slot_index,
            &xq,
            p.y_gate_up_grouped,
            2 * p.mi,
            p.gate_up_k,
            p.k_top,
            grouped_rows,
            p.batch_size,
            experts_host_mapped(p),
        ));
    }
    if ep_a8_wide(gpu, p, use_path2) {
        // HIPFIRE_QWEN4_MOE_GU_A8 (EP2 decode / verify, 9..=64 rows): int8 per-128 gate/up with the
        // SwiGLU epilogue (BF16 grouped rows). Owner-masked (default): `ep_tile_mask` copies the
        // scatter's tile ids with every non-owned expert's tile set to the `-1` sentinel; this GEMM
        // and the down both read the copy (the down runs later in the same call, stream-ordered), so
        // neither touches a tile whose weights are the zero dummy. Unmasked: every grouped row is
        // written, dummy tiles give +0.
        let mask = ep_a8_owner_mask(gpu, p)?;
        gpu.qwen4_moe_gu_a8_receipt(p.batch_size, grouped_rows, mask.is_some());
        let tile_ids = match mask {
            Some(mask) => {
                // `HIPFIRE_QWEN4_MOE_GLUE_FOLD` a / b: the scatter-side launch already wrote the
                // masked copy of the tile ids of this call.
                if !tile_mask_written {
                    hip(gpu.ep_tile_mask(
                        p.expert_tile_ids,
                        mask.tile_ids_owned,
                        mask.owner_table,
                        grouped_tiles(grouped_rows),
                        mask.n_experts,
                        mask.rank,
                    ))?;
                }
                mask.tile_ids_owned
            }
            None => gemm_tile_ids(p),
        };
        let xq = hip(gpu.qwen4_moe_rotate256_i8(p.x_norm_batch, p.gate_up_k, p.batch_size))?;
        if gpu.qwen4_moe_gu_rot_fold_applies(
            2 * p.mi,
            p.gate_up_k,
            p.down_m,
            p.down_k,
            grouped_rows,
            p.k_top,
        ) {
            // HIPFIRE_QWEN4_MOE_GLUE_FOLD=c: the down producer (`qwen4_moe_rotate128_i8`) runs in this
            // GEMM's epilogue; the sidecar waits in the dedicated down scratch for the down stage's
            // `gemm_qwen4_moe_down_a8_rot_sym` (same predicate, same arguments). No BF16 grouped rows.
            return hip(gpu.gemm_qwen4_moe_gate_up_silu_a8_rot_sym(
                gate_up_ptrs,
                tile_ids,
                p.sorted_slot_index,
                &xq,
                2 * p.mi,
                p.gate_up_k,
                p.k_top,
                grouped_rows,
                p.batch_size,
            ));
        }
        return hip(gpu.gemm_qwen4_moe_gate_up_silu_a8_sym(
            gate_up_ptrs,
            tile_ids,
            p.sorted_slot_index,
            &xq,
            p.y_gate_up_grouped,
            2 * p.mi,
            p.gate_up_k,
            p.k_top,
            grouped_rows,
            p.batch_size,
        ));
    }
    if use_path2 && gateup_bf16(gpu, p) {
        let x_f16 = if gateup_rotates_f16(gpu, p, use_path2) {
            Some(hip(gpu.rotate_x_mq_batched_f16(
                p.x_norm_batch,
                p.gate_up_k,
                p.batch_size,
            ))?)
        } else {
            None
        };
        let gemm = if gateup_silu(gpu, p) {
            Gpu::gemm_mq4g256v2_moe_grouped_top10_silu_bf16out
        } else {
            Gpu::gemm_mq4g256v2_moe_grouped_top10_bf16out
        };
        hip(gemm(
            gpu,
            gate_up_ptrs,
            gemm_tile_ids(p),
            p.sorted_slot_index,
            x_f16.as_ref().unwrap_or(p.x_rot_batch),
            p.y_gate_up_grouped,
            2 * p.mi,
            p.gate_up_k,
            p.k_top,
            grouped_rows,
            p.batch_size,
        ))
    } else if use_path2 {
        if gpu.arch_caps.is_gfx1151()
            && p.mi == 640
            && p.gate_up_k == 2560
            && p.k_top == 10
            && (1..=8).contains(&p.batch_size)
        {
            return hip(gpu.gemm_mq4g256v2_moe_grouped_top10_inverse(
                gate_up_ptrs,
                p.expert_tile_ids,
                p.sorted_slot_index,
                p.inverse_perm,
                p.x_rot_batch,
                p.y_gate_up_grouped,
                2 * p.mi,
                p.gate_up_k,
                p.k_top,
                grouped_rows,
                p.batch_size,
            ));
        }
        hip(gpu.gemm_mq4g256v2_moe_grouped_top10(
            gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.x_rot_batch,
            p.y_gate_up_grouped,
            2 * p.mi,
            p.gate_up_k,
            p.k_top,
            grouped_rows,
            p.batch_size,
        ))
    } else {
        #[cfg(feature = "test-utils")]
        super::bind_g1_window(
            gpu,
            p.routed_experts,
            p.n_exp,
            p.topk_weights,
            p.recipe.bf16_round_trip(),
        )?;
        hip(gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
            gate_up_ptrs,
            p.topk_indices,
            p.x_rot_batch,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            p.gate_up_k,
            p.batch_size,
        ))?;
        let active = p
            .batch_size
            .checked_mul(p.k_top)
            .and_then(|slots| slots.checked_mul(p.mi))
            .ok_or_else(|| DispatchError::Hip("grouped gate/up extent overflows".into()))?;
        // Wide verify uses the singleton fused reader below, which owns these
        // round trips. Keep the raw gate/up outputs just as decode does.
        if p.recipe.bf16_round_trip() && !super::layer_ops::wide_exact_moe_rows(gpu, p.batch_size) {
            hip(gpu.bf16_round_trip_f32(&f32_view(p.gate_batch, 0, active)))?;
            hip(gpu.bf16_round_trip_f32(&f32_view(p.up_batch, 0, active)))?;
        }
        Ok(())
    }
}

/// Path-2 unscatter fused with the SwiGLU activation: the grouped gate/up
/// rows go straight to the rotation input (`rot_batch`) with the recipe's BF16
/// round trips, bitwise the unfused unscatter -> round trip -> silu_mul ->
/// round trip sequence, rotated in the same launch when
/// [`unscatter_rotates`] (else [`activation`] rotates).
/// qt=55 FP8: `fp8b128_moe_unscatter_silu` (F32 grouped input, BF16 rounds of
/// gate, up, SiLU and product; not the QT44 unscatter, whose SiLU stays F32),
/// no rotation ([`fp8_unscatter`]).
pub(crate) fn unscatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    if is_fp8(p) {
        return fp8_unscatter(gpu, p, grouped_rows);
    }
    require_geometry(p)?;
    if super::layer_ops::wide_exact_moe_rows(gpu, p.batch_size) {
        return Ok(());
    }
    validate_scratch(gpu, p, true, grouped_rows)?;
    // The slab route has no unscatter: K2 keeps SwiGLU in the gate/up epilogue
    // and rotates in `down`; K1 writes slot-ordered F32 rows.
    if is_slab(p) {
        return Ok(());
    }
    // Symmetric IU4: SiLU in the gate/up epilogue, rotation + A4 in `down`.
    if sym_iu4(gpu, p, true) {
        return Ok(());
    }
    // Wide-verify int8 chain: SiLU in the gate/up epilogue, rotation + A8 in `down`.
    if ep_a8_wide(gpu, p, true) {
        return Ok(());
    }
    // SiLU in the gate/up epilogue, unscatter + rotation in the down stage.
    if gateup_silu(gpu, p) {
        return Ok(());
    }
    if gateup_bf16(gpu, p) {
        return hip(gpu.moe_gate_up_unscatter_silu_top10_bf16in(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.rot_batch,
            p.mi,
            grouped_rows,
            p.recipe.bf16_round_trip(),
        ));
    }
    if unscatter_rotates(gpu, p) {
        // The shared expert's activation rides in the same launch; its stage
        // skipped it (the shared down reads it after the combine).
        let shared = p
            .prelude
            .shared
            .as_ref()
            .filter(|_| shared_in_unscatter(gpu, p))
            .map(|shared| {
                let live = p.batch_size * shared.intermediate;
                (
                    f32_view(shared.gate_out, 0, live),
                    f32_view(shared.up_out, 0, live),
                    f32_view(shared.rotated, 0, live),
                    shared.scalar,
                )
            });
        let shared = shared
            .as_ref()
            .map(|(gate, up, out, selector)| SharedExpertActivation {
                gate,
                up,
                out,
                selector,
                selectors: p.batch_size,
            });
        return hip(gpu.moe_gate_up_unscatter_silu_rotate128_top10(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.rot_batch,
            p.mi,
            grouped_rows,
            p.recipe.bf16_round_trip(),
            shared.as_ref(),
        ));
    }
    hip(gpu.moe_gate_up_unscatter_silu_top10(
        p.y_gate_up_grouped,
        p.sorted_slot_index,
        p.rot_batch,
        p.mi,
        grouped_rows,
        p.recipe.bf16_round_trip(),
    ))
}

/// Whether path 2's F32 unscatter also applies the down's 128-wide rotation
/// (in the same launch), leaving [`activation`] nothing to do.
fn unscatter_rotates(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    unscatter_rotates_base(gpu, p) && !ep_a8_wide(gpu, p, true)
}

/// [`unscatter_rotates`] ignoring the wide-verify int8 chain, which replaces the unscatter: the
/// incumbent F32 chain's own decision, kept for the shared-expert routes that key on it
/// ([`shared_in_unscatter_base`]) so they stay what they are on the incumbent chain.
fn unscatter_rotates_base(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    !super::layer_ops::wide_exact_moe_rows(gpu, p.batch_size)
        && !is_slab(p)
        && !sym_iu4(gpu, p, true)
        && !gateup_bf16(gpu, p)
        && !down_wmma(gpu, p)
        && p.mi.is_multiple_of(128)
}

/// Whether path 2's fused unscatter launch also runs the BF16 shared expert
/// activation: its only reader, a natural-basis shared down, runs after the
/// combine (so after the unscatter). A CPU-deferred call lowers no unscatter,
/// so its shared activation always runs in its own stage. The wide-verify int8 chain
/// runs no unscatter: the shared activation then runs in its own stage
/// ([`shared_activation`], the same `shared_expert_activation_bf16_f32` launch the fused
/// unscatter folds in).
fn shared_in_unscatter(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    shared_in_unscatter_base(gpu, p) && !ep_a8_wide(gpu, p, true)
}

/// [`shared_in_unscatter`] as the incumbent F32 chain decides it. [`shared_gu_pm`] and
/// [`shared_int8`] key on this, not on the launch that actually runs, so the wide-verify int8
/// chain leaves the shared expert's projections and down exactly the incumbent chain's.
fn shared_in_unscatter_base(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    !p.cpu_deferred_experts
        && unscatter_rotates_base(gpu, p)
        && p.recipe.bf16_round_trip()
        && p.recipe.shared_after_combine()
        && p.prelude
            .shared
            .as_ref()
            .is_some_and(|shared| shared.weights.down.dtype != DType::MQ4G128V2)
}

/// Activation stage.  qt=55 FP8: nothing is owed (the unscatter produced the
/// BF16-valued activation and the FP8 down consumes it unrotated).
pub(crate) fn activation(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    if is_fp8(p) {
        return fp8_validate(gpu, p, p.m_total_max).map(drop);
    }
    require_geometry(p)?;
    let use_path2 = grouped_path(gpu, p, use_path2);
    let total_slots = p.batch_size * p.k_top;
    if is_slab(p) {
        return slab_activation(gpu, p);
    }
    let wide_exact =
        p.recipe.bf16_round_trip() && super::layer_ops::wide_exact_moe_rows(gpu, p.batch_size);
    // The F16 WMMA and symmetric IU4 downs rotate the grouped rows themselves
    // (see `down`); the F32 unscatter may already have rotated.
    let path2_done = !wide_exact
        && use_path2
        && (sym_iu4(gpu, p, use_path2)
            || ep_a8_wide(gpu, p, use_path2)
            || down_wmma(gpu, p)
            || unscatter_rotates(gpu, p));
    // The batch views are owed only when this stage runs a kernel over them.
    // The symmetric IU4 route's compact scratch holds no gate/up/rot rows
    // beyond the fallback cap, so viewing `batch_size * k_top * mi` eagerly
    // would index past that allocation on a prefill chunk.
    let live = activation_batch_extent(path2_done, total_slots, p.mi)
        .ok_or_else(|| DispatchError::Hip("routed activation extent overflows".into()))?;
    if live == 0 {
        return Ok(());
    }
    let gate = f32_view(p.gate_batch, 0, live);
    let up = f32_view(p.up_batch, 0, live);
    let rotated = f32_view(p.rot_batch, 0, live);
    if wide_exact {
        return hip(gpu.silu_mul_bf16_rt_rotate_x_mq_128_v2_shared(
            &gate,
            &up,
            &rotated,
            p.mi,
            total_slots,
            None,
        ));
    }
    // Path 2's unscatter already produced the activation.
    if !use_path2 {
        hip(gpu.silu_mul_f32(&gate, &up, &rotated))?;
        if p.recipe.bf16_round_trip() {
            hip(gpu.bf16_round_trip_f32(&rotated))?;
        }
    }
    hip(gpu.rotate_x_mq_128_v2(&rotated, &rotated, p.mi, total_slots))
}

/// F32 elements of the gate/up/rot batch buffers the non-slab activation stage
/// runs a kernel over: `0` when path 2's down or unscatter already consumed
/// the rotation (`path2_done`), else the live `total_slots * mi` rows. Pure;
/// the caller views the buffers only for a non-zero extent. `None` on overflow.
fn activation_batch_extent(path2_done: bool, total_slots: usize, mi: usize) -> Option<usize> {
    if path2_done {
        return Some(0);
    }
    total_slots.checked_mul(mi)
}

/// A few-row forward (speculative verify) on grouped path 2 without the WMMA
/// down: the route slots are nearly all distinct experts, so the grouped down
/// tile (16 slot rows per expert) runs almost empty; the decode kernel's
/// per-slot down over the unscattered, rotated rows streams the experts
/// faster.  Its per-slot dot is the grouped kernel's (bitwise), combined in
/// slot order.
fn indexed_down(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    indexed_down_for_rows(
        use_path2,
        gpu.route_rows(p.batch_size),
        gpu.few_row_limit(),
        gpu.wide_grouped_down(),
        down_wmma(gpu, p),
    )
}

/// `HIPFIRE_QWEN4_WIDE_GROUPED_DOWN` keeps the grouped, expert-unioned down (and the
/// grouped combine) for the 9..=`few_row_limit` rows that would otherwise take the
/// per-slot indexed down.
fn indexed_down_for_rows(
    use_path2: bool,
    route: usize,
    limit: usize,
    wide_grouped_down: bool,
    down_wmma: bool,
) -> bool {
    use_path2
        && route <= limit
        && !down_wmma
        && !(wide_grouped_down && route > rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS)
}

/// Release the stage immediately after the routed down, before combine.
fn release_stage(gpu: &Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    if let Some(stage) = p.expert_stage_ptrs {
        hip(gpu.hip.event_record(stage.free, gpu.active_stream.as_ref()))?;
    }
    Ok(())
}

/// Routed down GEMM.  qt=55 FP8: one `gemm_fp8b128_moe_grouped_bf16mfma` over
/// the same tile descriptor, F32 `[G, hidden]` grouped rows ([`fp8_down`]).
pub(crate) fn down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    if is_fp8(p) {
        return fp8_down(gpu, p, grouped_rows);
    }
    require_geometry(p)?;
    validate_scratch(gpu, p, use_path2, grouped_rows)?;
    down_stage(gpu, p, use_path2, grouped_rows)
}

/// [`down`] after its scratch preflight; the indexed arm re-enters on path 1
/// with the same (already validated) buffers.
fn down_stage(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let use_path2 = grouped_path(gpu, p, use_path2);
    let down_ptrs = p
        .expert_stage_ptrs
        .map_or(p.expert_down_ptrs, |stage| stage.down);
    let total_slots = p.batch_size * p.k_top;
    if is_slab(p) {
        slab_down(gpu, p, grouped_rows)?;
        return release_stage(gpu, p);
    }
    if sym_iu4(gpu, p, use_path2) {
        if gpu.qwen4_moe_ep_a8_requested() {
            // HIPFIRE_QWEN4_MOE_EP_A8: grouped BF16 SwiGLU rows -> FWHT128 -> flat-slot A8 -> signed-IU8 down,
            // the same BF16 grouped rows for the BF16-input combine.
            let xq = hip(gpu.qwen4_moe_rotate128_i8(
                p.y_gate_up_grouped,
                p.sorted_slot_index,
                p.down_k,
                grouped_rows,
                total_slots,
            ))?;
            if let Some(tx) = p.ep_tile_exchange.filter(|tx| !tx.exchange.mask_only()) {
                tx.exchange
                    .before_down(gpu)
                    .map_err(|error| DispatchError::Hip(format!("qt44 tile exchange: {error}")))?;
            }
            hip(gpu.gemm_qwen4_moe_down_a8_sym(
                down_ptrs,
                gemm_tile_ids(p),
                p.sorted_slot_index,
                &xq,
                p.y_down_grouped,
                p.down_m,
                p.down_k,
                1,
                grouped_rows,
                total_slots,
            ))?;
            return release_stage(gpu, p);
        }
        // Grouped BF16 SwiGLU rows -> FWHT128 -> compact flat-slot A4 ->
        // grouped IU4 down, BF16 per grouped row for the BF16-input combine.
        let xq = hip(gpu.qwen4_moe_rotate128_i4_gp(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.down_k,
            grouped_rows,
            total_slots,
        ))?;
        if let Some(tx) = p.ep_tile_exchange.filter(|tx| !tx.exchange.mask_only()) {
            // The peer reads this rank's `y_down_grouped` between its producer
            // and completion events of the previous layer's exchange: no write
            // before its pull completed.
            tx.exchange
                .before_down(gpu)
                .map_err(|error| DispatchError::Hip(format!("qt44 tile exchange: {error}")))?;
        }
        hip(gpu.gemm_qwen4_moe_down_iu4_sym(
            down_ptrs,
            gemm_tile_ids(p),
            p.sorted_slot_index,
            &xq,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            1,
            grouped_rows,
            total_slots,
            experts_host_mapped(p),
        ))?;
        return release_stage(gpu, p);
    }
    if ep_a8_wide(gpu, p, use_path2) {
        // HIPFIRE_QWEN4_MOE_GU_A8: grouped BF16 SwiGLU rows -> FWHT128 -> flat-slot A8 -> signed-IU8
        // down; BF16 rows for the BF16-input combine (no unscatter to slot order: the gather is a no-op).
        // The rotation reads the gate/up rows of every grouped slot, the rows of masked tiles
        // included (stale: gate/up never wrote them): each lands in its own flat-slot record
        // (`sorted_slot_index` is unique per grouped row, one CTA per row and K128 epoch), and the
        // down reads only the records of the tiles it runs, never a masked tile's.
        let tile_ids =
            ep_a8_owner_mask(gpu, p)?.map_or(gemm_tile_ids(p), |mask| mask.tile_ids_owned);
        if gpu.qwen4_moe_gu_rot_fold_applies(
            2 * p.mi,
            p.gate_up_k,
            p.down_m,
            p.down_k,
            grouped_rows,
            p.k_top,
        ) {
            // The gate/up stage already produced the flat-slot A8 sidecar (HIPFIRE_QWEN4_MOE_GLUE_FOLD=c).
            hip(gpu.gemm_qwen4_moe_down_a8_rot_sym(
                down_ptrs,
                tile_ids,
                p.sorted_slot_index,
                p.y_down_grouped,
                p.down_m,
                p.down_k,
                grouped_rows,
                total_slots,
            ))?;
            return release_stage(gpu, p);
        }
        let xq = hip(gpu.qwen4_moe_rotate128_i8(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.down_k,
            grouped_rows,
            total_slots,
        ))?;
        hip(gpu.gemm_qwen4_moe_down_a8_sym(
            down_ptrs,
            tile_ids,
            p.sorted_slot_index,
            &xq,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            1,
            grouped_rows,
            total_slots,
        ))?;
        return release_stage(gpu, p);
    }
    if indexed_down(gpu, p, use_path2) {
        return down_stage(gpu, p, false, grouped_rows);
    }
    if use_path2 && down_wmma(gpu, p) {
        // Rotation and GEMM in one stage: the rotated F16 rows live in the
        // shared FP16 scratch, which another stage's GEMM would overwrite.
        // Under the owner-routed exchange the rotation also reads the stale rows of masked
        // tiles: it writes them to their own flat slots (`sorted_slot_index` is unique per
        // grouped row), which no owned tile's GEMM reads.
        let x_f16 = if gateup_silu(gpu, p) {
            hip(gpu.moe_unscatter_rotate128_f16(
                p.y_gate_up_grouped,
                p.sorted_slot_index,
                p.mi,
                grouped_rows,
                total_slots,
            ))?
        } else {
            hip(gpu.rotate_x_mq_128_v2_f16(p.rot_batch, p.mi, total_slots))?
        };
        if let Some(tx) = p.ep_tile_exchange.filter(|tx| !tx.exchange.mask_only()) {
            // The peer reads this rank's `y_down_grouped` between its producer and
            // completion events of the previous layer's exchange: no write before its
            // pull completed.
            tx.exchange
                .before_down(gpu)
                .map_err(|error| DispatchError::Hip(format!("qt44 tile exchange: {error}")))?;
        }
        hip(gpu.gemm_mq4g128v2_moe_grouped_top10_xf16(
            down_ptrs,
            gemm_tile_ids(p),
            p.sorted_slot_index,
            &x_f16,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            grouped_rows,
            true,
        ))?;
    } else if use_path2 {
        hip(gpu.gemm_mq4g128v2_moe_grouped_top10(
            down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.rot_batch,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            1,
            grouped_rows,
            total_slots,
            p.n_exp,
        ))?;
        if p.ep_row_gather.is_some() && ep_grouped_down_unscatter(gpu, p, true) {
            hip(gpu.moe_down_unscatter_k8(
                p.y_down_grouped,
                p.inverse_perm,
                p.down_expanded,
                p.down_m,
                total_slots,
            ))?;
        }
        // A BF16-source recipe rounds each grouped expert output before
        // weighted combination, just as on the scalar route.  The grouped
        // combine kernel performs that same RNE rounding on every row it
        // reads, so a separate pass over all (padded) grouped rows would only
        // re-round values that are already BF16-exact.
    } else {
        #[cfg(feature = "test-utils")]
        super::bind_g1_window(
            gpu,
            p.routed_experts,
            p.n_exp,
            p.topk_weights,
            p.recipe.bf16_round_trip(),
        )?;
        hip(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            p.down_m,
            p.down_k,
            p.batch_size,
            p.n_exp,
            10,
        ))?;
        // The combine rounds every expert output it reads to BF16 itself.
    }
    release_stage(gpu, p)
}

/// Whether [`combine`] can start from +0.0 instead of the zero-filled target
/// (`HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`; the wide-verify int8 chain also needs
/// `HIPFIRE_QWEN4_FFN_CLEAR_ABSORB`, [`ffn_clear_absorb_a8`]): the BF16-row
/// combine of path 2, with nothing writing the target between its zero fill and
/// the combine (the shared down, the only other writer, runs after it).
/// Never applies on the qt=55 FP8 route.
pub(crate) fn combine_initial_zero_applies(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> bool {
    combine_initial_zero_rows_applies(gpu, p, use_path2) || ffn_clear_absorb_a8(gpu, p, use_path2)
}

/// [`combine_initial_zero_applies`] without the wide-verify int8 chain: the F16 WMMA /
/// symmetric IU4 BF16-row arms, the only ones the HC row folds were proven on.
fn combine_initial_zero_rows_applies(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    !is_fp8(p)
        && grouped_path(gpu, p, use_path2)
        && !indexed_down(gpu, p, use_path2)
        && (down_wmma(gpu, p) || sym_iu4(gpu, p, use_path2))
        && p.recipe.shared_after_combine()
}

/// `HIPFIRE_QWEN4_FFN_CLEAR_ABSORB` (exact gfx1201, default on): the wide-verify int8 chain
/// ([`ep_a8_wide`]: the 9..=64-row EP2 verify MoE, no gather, no tile exchange) combines through
/// `moe_down_combine_grouped_top10_bf16in_owned`, the same BF16-row combine entry the F16 WMMA /
/// symmetric IU4 arms use, so it can take its `_zinit` store arm too. That kernel visits every
/// `(token, column)` of the target whether or not the owner mask marks a rank's grouped rows
/// foreign (a foreign rank adds a +0 BF16 row), so every element the clear zeroes is stored.
/// Never with [`ep_grouped_down_unscatter`] (the slot-order combine accumulates).
fn ffn_clear_absorb_a8(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    gpu.flags.qwen4_ffn_clear_absorb_enabled()
        && !is_fp8(p)
        && grouped_path(gpu, p, use_path2)
        && !indexed_down(gpu, p, use_path2)
        && !ep_grouped_down_unscatter(gpu, p, use_path2)
        && ep_a8_wide(gpu, p, use_path2)
        && p.recipe.shared_after_combine()
}

/// `initial_zero`: the combine writes the target from +0.0 rather than
/// accumulating into zeros the caller filled (requires
/// [`combine_initial_zero_applies`]).
/// qt=55 FP8: `moe_down_combine_grouped_top10` (F32 grouped rows, canonical
/// ascending expert-ID order); `initial_zero` is refused ([`fp8_combine`]).
pub(crate) fn combine(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
    initial_zero: bool,
) -> Result<(), DispatchError> {
    if is_fp8(p) {
        return fp8_combine(gpu, p, grouped_rows, initial_zero);
    }
    require_geometry(p)?;
    validate_scratch(gpu, p, use_path2, grouped_rows)?;
    combine_stage(gpu, p, use_path2, grouped_rows, initial_zero)
}

/// [`combine`] after its scratch preflight; the indexed arm re-enters on
/// path 1 with the same (already validated) buffers.
fn combine_stage(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
    initial_zero: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let use_path2 = grouped_path(gpu, p, use_path2);
    let target = p.routed_out.unwrap_or(p.x_batch);
    if initial_zero && !combine_initial_zero_applies(gpu, p, use_path2) {
        return Err(DispatchError::Hip(
            "qt44 combine: zero-initialized target requested off the BF16-row path".into(),
        ));
    }
    if indexed_down(gpu, p, use_path2) {
        return combine_stage(gpu, p, false, grouped_rows, false);
    }
    // The call's fixed reduction partition (this layer's row gather hook, the same layer index
    // the gather itself receives): only the expanded top-10 arms can mask foreign experts.
    let partition = reduction_partition(p);
    if use_path2 && ep_grouped_down_unscatter(gpu, p, use_path2) {
        // The grouped down was unscattered to slot order for the EP gather: the
        // canonical top-10 slot-order combine reads it.
        let leaf = super::combine_leaf(super::CombineArm::ExpandedTop10, partition)?;
        let expanded = f32_view(p.down_expanded, 0, p.batch_size * p.k_top * p.down_m);
        leaf.launch(
            gpu,
            &expanded,
            p.topk_indices,
            p.topk_weights,
            target,
            p.down_m,
            p.batch_size,
            p.n_exp,
        )?;
    } else if use_path2
        && (down_wmma(gpu, p) || sym_iu4(gpu, p, use_path2) || ep_a8_wide(gpu, p, use_path2))
    {
        super::combine_leaf(super::CombineArm::GroupedBf16InOwned, partition)?;
        // Both downs store BF16 grouped rows. The rank order goes to
        // `down_expanded`: unused on this route until the shared down, which
        // runs after the combine. Under the partial mask's implicit zeros the
        // order marks the foreign rows -1 (their clear was skipped).
        // The wide-verify chain's owner mask (a masked down never wrote the foreign rows): the same
        // owner table the tile mask read, so the combine folds exactly the owned rows and a foreign
        // rank adds +0 with its own weight and rounding.
        let owners = if ep_a8_wide(gpu, p, use_path2) {
            ep_a8_owner_mask(gpu, p)?.map(|mask| rdna_compute::hc_row_fold::CombineOwners {
                table: mask.owner_table,
                rank: mask.rank,
            })
        } else {
            implicit_zero_owners(gpu, p)
        };
        // `HIPFIRE_QWEN4_MOE_COMBINE_INLINE_ORDER`: the wide-verify chain's owner-masked order
        // and combine as one launch (no order scratch, bytes identical to the pair); on by
        // default for exact gfx1201, `=0` keeps the pair.
        let inline_order = if ep_a8_wide(gpu, p, use_path2)
            && gpu.flags.qwen4_moe_combine_inline_order_enabled()
        {
            owners
        } else {
            None
        };
        if let Some(owners) = inline_order {
            hip(gpu.moe_down_combine_grouped_top10_bf16in_ordered(
                p.y_down_grouped,
                p.inverse_perm,
                p.topk_indices,
                p.topk_weights,
                target,
                p.down_m,
                grouped_rows,
                p.batch_size,
                initial_zero,
                p.n_exp,
                owners,
            ))?;
        } else {
            hip(gpu.moe_down_combine_grouped_top10_bf16in_owned(
                p.y_down_grouped,
                p.inverse_perm,
                p.topk_indices,
                p.topk_weights,
                target,
                p.down_expanded,
                p.down_m,
                grouped_rows,
                p.batch_size,
                initial_zero,
                p.n_exp,
                owners,
            ))?;
        }
    } else if use_path2 {
        super::combine_leaf(super::CombineArm::GroupedF32, partition)?;
        if implicit_zero_owners(gpu, p).is_some() {
            return Err(DispatchError::Hip(
                "qt44 combine: the F32 grouped combine would read foreign rows whose clear the implicit zeros skipped".into(),
            ));
        }
        hip(gpu.moe_down_combine_grouped_top10(
            p.y_down_grouped,
            p.inverse_perm,
            p.topk_indices,
            p.topk_weights,
            target,
            p.down_m,
            grouped_rows,
            p.batch_size,
            p.n_exp,
        ))?;
    } else {
        let leaf = super::combine_leaf(super::CombineArm::ExpandedTop10, partition)?;
        let expanded = f32_view(p.down_expanded, 0, p.batch_size * p.k_top * p.down_m);
        leaf.launch(
            gpu,
            &expanded,
            p.topk_indices,
            p.topk_weights,
            target,
            p.down_m,
            p.batch_size,
            p.n_exp,
        )?;
    }
    // A shared down placed after the combine rounds `target` to BF16 as its
    // first read, so the round trip is owed only when nothing follows.
    if p.recipe.bf16_round_trip() && !p.recipe.shared_after_combine() {
        hip(gpu.bf16_round_trip_f32(&f32_view(target, 0, p.batch_size * p.down_m)))?;
    }
    Ok(())
}

// ─────────────── sealed expert-slab (MQ4 + cold W3 / E8P) ───────────────

/// Slab gate/up. K2: rotate256 A4 once, then the layer's cold-codec (W3 or
/// E8P) and MQ4 NT1 kernels over
/// the same grouped Y rows, each on its own codec-masked tile ids (a uniform
/// layer launches its one kernel on the scatter's ids). K1: the lane-union GEMV
/// over the rotated F32 activation, BF16-rounded gate/up rows in path-1 shape.
fn slab_gate_up(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    use rdna_compute::slab_moe::SlabBuckets;
    if !super::slab_route::arch_supported(&gpu.arch) {
        return Err(DispatchError::Hip(
            "sealed expert-slab route is gfx1100/gfx1151/gfx1201-only".into(),
        ));
    }
    let layer = super::slab_route::layer(
        &gpu.arch,
        p.route_policy.as_ref(),
        &p.dtypes,
        p.expert_dtype_tags,
    )?;
    if !slab_k2(gpu, p) {
        // The activation basis the path-1 shape reads. Idempotent: the same
        // bytes the input-basis stage produced.
        hip(gpu.rotate_x_mq_batched(p.x_norm_batch, p.x_rot_batch, p.gate_up_k, p.batch_size))?;
        let active = p
            .batch_size
            .checked_mul(p.k_top)
            .and_then(|slots| slots.checked_mul(p.mi))
            .ok_or_else(|| DispatchError::Hip("slab gate/up extent overflows".into()))?;
        super::slab_route::k1_gate_up(
            gpu,
            layer,
            p.expert_gate_up_ptrs,
            p.topk_indices,
            p.x_rot_batch,
            p.gate_batch,
            p.up_batch,
            p.batch_size,
        )?;
        if p.recipe.bf16_round_trip() && !super::layer_ops::wide_exact_rows(gpu, p.batch_size) {
            hip(gpu.bf16_round_trip_f32(&f32_view(p.gate_batch, 0, active)))?;
            hip(gpu.bf16_round_trip_f32(&f32_view(p.up_batch, 0, active)))?;
        }
        return Ok(());
    }
    let xq = hip(gpu.qwen4_moe_rotate256_i4_slab(p.x_norm_batch, p.gate_up_k, p.batch_size))?;
    let host_mapped = experts_host_mapped(p);
    let gate_up_ptrs = p
        .expert_stage_ptrs
        .map_or(p.expert_gate_up_ptrs, |stage| stage.gate_up);
    let (m, k) = (2 * p.mi, p.gate_up_k);
    match layer.buckets {
        SlabBuckets::Uniform(0) => cold_gate_up(
            gpu,
            p,
            layer.cold_codec,
            gate_up_ptrs,
            p.expert_tile_ids,
            &xq,
            grouped_rows,
        ),
        SlabBuckets::Uniform(_) => hip(gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
            gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            &xq,
            p.y_gate_up_grouped,
            m,
            k,
            p.k_top,
            grouped_rows,
            p.batch_size,
            host_mapped,
        )),
        SlabBuckets::Table(tags) => {
            let owner = super::slab_route::codec_i32(tags)?;
            let [cold, mq4] = hip(gpu.qwen4_slab_tile_masks(
                p.expert_tile_ids,
                &owner,
                grouped_tiles(grouped_rows),
                layer.cold_codec,
            ))?;
            cold_gate_up(
                gpu,
                p,
                layer.cold_codec,
                gate_up_ptrs,
                &cold,
                &xq,
                grouped_rows,
            )?;
            hip(gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
                gate_up_ptrs,
                &mq4,
                p.sorted_slot_index,
                &xq,
                p.y_gate_up_grouped,
                m,
                k,
                p.k_top,
                grouped_rows,
                p.batch_size,
                host_mapped,
            ))
        }
    }
}

/// Slab activation: K2 owes nothing (SwiGLU is in the gate/up epilogue,
/// rotation + A4 in `down`); K1 runs the path-1 shape's SwiGLU with the
/// recipe's BF16 round trips and the down's 128-wide rotation.
fn slab_activation(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    if slab_k2(gpu, p) {
        return Ok(());
    }
    let total_slots = p.batch_size * p.k_top;
    let active = total_slots * p.mi;
    let gate = f32_view(p.gate_batch, 0, active);
    let up = f32_view(p.up_batch, 0, active);
    let rot = f32_view(p.rot_batch, 0, active);
    if p.recipe.bf16_round_trip() && super::layer_ops::wide_exact_rows(gpu, p.batch_size) {
        return hip(gpu.silu_mul_bf16_rt_rotate_x_mq_128_v2_shared(
            &gate,
            &up,
            &rot,
            p.mi,
            total_slots,
            None,
        ));
    }
    hip(gpu.silu_mul_f32(&gate, &up, &rot))?;
    if p.recipe.bf16_round_trip() {
        hip(gpu.bf16_round_trip_f32(&rot))?;
    }
    hip(gpu.rotate_x_mq_128_v2(p.rot_batch, p.rot_batch, p.mi, total_slots))
}

/// Slab down. K2: rotate128 A4 (the hipcc producer on gfx1201, K3 on gfx1100),
/// then the cold-codec and MQ4 down kernels over the grouped BF16 rows on codec-masked
/// tiles. K1: the lane-union GEMV over the rotated activation, unweighted F32
/// rows in slot order for the top-10 combine.
fn slab_down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    use rdna_compute::slab_moe::SlabBuckets;
    if !super::slab_route::arch_supported(&gpu.arch) {
        return Err(DispatchError::Hip(
            "sealed expert-slab route is gfx1100/gfx1151/gfx1201-only".into(),
        ));
    }
    let layer = super::slab_route::layer(
        &gpu.arch,
        p.route_policy.as_ref(),
        &p.dtypes,
        p.expert_dtype_tags,
    )?;
    let total_slots = p.batch_size * p.k_top;
    if !slab_k2(gpu, p) {
        let rot = f32_view(p.rot_batch, 0, total_slots * p.mi);
        let y0 = f32_view(p.down_expanded, 0, total_slots * p.down_m);
        return super::slab_route::k1_down(
            gpu,
            layer,
            p.topk_indices,
            p.expert_down_ptrs,
            &rot,
            &y0,
            p.batch_size,
        );
    }
    let xq = hip(gpu.qwen4_moe_rotate128_i4_slab(
        p.y_gate_up_grouped,
        p.sorted_slot_index,
        p.down_k,
        grouped_rows,
        total_slots,
    ))?;
    // A staged long prefill reads this layer's arena copy in VRAM (stage tables
    // rebased onto the parity stage buffer) and so takes the resident NT variant.
    let down_ptrs = p
        .expert_stage_ptrs
        .map_or(p.expert_down_ptrs, |stage| stage.down);
    let host_mapped = experts_host_mapped(p);
    match layer.buckets {
        SlabBuckets::Uniform(0) => cold_down(
            gpu,
            p,
            layer.cold_codec,
            down_ptrs,
            p.expert_tile_ids,
            &xq,
            grouped_rows,
        ),
        SlabBuckets::Uniform(_) => hip(gpu.gemm_qwen4_moe_down_iu4_sym(
            down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            &xq,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            1,
            grouped_rows,
            total_slots,
            host_mapped,
        )),
        SlabBuckets::Table(tags) => {
            let owner = super::slab_route::codec_i32(tags)?;
            let [cold, mq4] = hip(gpu.qwen4_slab_tile_masks(
                p.expert_tile_ids,
                &owner,
                grouped_tiles(grouped_rows),
                layer.cold_codec,
            ))?;
            cold_down(
                gpu,
                p,
                layer.cold_codec,
                down_ptrs,
                &cold,
                &xq,
                grouped_rows,
            )?;
            hip(gpu.gemm_qwen4_moe_down_iu4_sym(
                down_ptrs,
                &mq4,
                p.sorted_slot_index,
                &xq,
                p.y_down_grouped,
                p.down_m,
                p.down_k,
                1,
                grouped_rows,
                total_slots,
                host_mapped,
            ))
        }
    }
}

/// K2 gate/up of the layer's cold codec: E8P-I4 (codec 3) or W3 (codec 1).
#[allow(clippy::too_many_arguments)]
fn cold_gate_up(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    cold_codec: u8,
    ptrs: &GpuTensor,
    tiles: &GpuTensor,
    xq: &rdna_compute::scratch::Int4MmqPrepared,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    let (m, k) = (2 * p.mi, p.gate_up_k);
    hip(match cold_codec {
        3 => gpu.gemm_qwen4_moe_e8p_gate_up_silu_iu4_sym(
            ptrs,
            tiles,
            p.sorted_slot_index,
            xq,
            p.y_gate_up_grouped,
            m,
            k,
            p.k_top,
            grouped_rows,
            p.batch_size,
        ),
        1 => gpu.gemm_qwen4_moe_w3_gate_up_silu_iu4_sym(
            ptrs,
            tiles,
            p.sorted_slot_index,
            xq,
            p.y_gate_up_grouped,
            m,
            k,
            p.k_top,
            grouped_rows,
            p.batch_size,
        ),
        other => {
            return Err(DispatchError::Hip(format!(
                "sealed expert-slab route: cold codec {other} has no K2 gate/up kernel"
            )))
        }
    })
}

/// K2 down of the layer's cold codec: E8P-I4 (codec 3) or W3 (codec 1).
#[allow(clippy::too_many_arguments)]
fn cold_down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    cold_codec: u8,
    ptrs: &GpuTensor,
    tiles: &GpuTensor,
    xq: &rdna_compute::scratch::Int4MmqPrepared,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    let total_slots = p.batch_size * p.k_top;
    hip(match cold_codec {
        3 => gpu.gemm_qwen4_moe_e8p_down_iu4_sym(
            ptrs,
            tiles,
            p.sorted_slot_index,
            xq,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            1,
            grouped_rows,
            total_slots,
        ),
        1 => gpu.gemm_qwen4_moe_w3_down_iu4_sym(
            ptrs,
            tiles,
            p.sorted_slot_index,
            xq,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            1,
            grouped_rows,
            total_slots,
        ),
        other => {
            return Err(DispatchError::Hip(format!(
                "sealed expert-slab route: cold codec {other} has no K2 down kernel"
            )))
        }
    })
}

// ───────────────────────── qt=55 FP8E4M3B128 (gfx942) ─────────────────────────

/// Bytes of one tile descriptor (`{expert, row0, nrows, 0}` as four `i32`).
const FP8_TILE_DESC_BYTES: usize = 16;

/// Refusal of the qt=55 lowering, raised before the stage's first launch.
fn fp8_refuse(reason: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!(
        "qt55 fp8 grouped prefill: {reason}; refusing before launch"
    ))
}

fn fp8_not_gfx942() -> DispatchError {
    fp8_refuse("the grouped FP8 kernels need exact gfx942")
}

/// Whether the declared policy and live dims are exactly the FP8 kernels'
/// contract (fixed 512/10/2560/640 geometry, qt=55 gate_up + down, BF16
/// router/shared formats) and the recipe keeps the BF16 boundaries the kernels
/// implement.
fn fp8_geometry(p: &MoePrefillParams<'_>) -> bool {
    let Some(policy) = p.route_policy else {
        return false;
    };
    p.batch_size > 0
        && p.recipe.bf16_round_trip()
        && crate::families::moe::fp8_grouped_route_geometry_supported(
            &policy,
            p.n_exp,
            p.k_top,
            p.gate_up_k,
            p.mi,
            p.gate_up_k,
            p.down_m,
            p.down_k,
            p.dtypes.routed_gate_up,
            p.dtypes.routed_down,
        )
}

fn fp8_require_geometry(p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    if fp8_geometry(p) {
        Ok(())
    } else {
        Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "fp8b128-grouped-prefill-shape",
            arch: "",
            quant: "FP8E4M3B128",
        })
    }
}

/// Smallest padded grouped-row count that holds `slots` route slots over
/// `experts` experts in 16-row tiles (`sealed_moe::checked_grouped_m_total_bound`).
fn fp8_grouped_rows_bound(slots: usize, experts: usize) -> Result<usize, DispatchError> {
    let block = crate::families::moe::MOE_GROUPED_BLOCK_M;
    let padding = scratch_mul(slots.min(experts), block - 1)?;
    slots
        .checked_add(padding)
        .and_then(|padded| padded.checked_next_multiple_of(block))
        .ok_or_else(scratch_overflow)
}

fn fp8_block_stride(rows: usize, k: usize) -> Result<usize, DispatchError> {
    rdna_compute::fp8_e4m3::b128_block_stride(rows, k).ok_or_else(scratch_overflow)
}

/// The checked extents one FP8 stage launches with.
struct Fp8Plan {
    rows: usize,
    slots: usize,
    grouped_rows: usize,
    n_tiles: usize,
    gate_up_base: *mut std::ffi::c_void,
    down_base: *mut std::ffi::c_void,
    gate_up_stride: usize,
    down_stride: usize,
}

impl Fp8Plan {
    /// Borrowed whole-table view of one qt=55 table: `experts` blocks of
    /// `stride` bytes from expert 0's block.  No allocation, never freed.
    fn table(
        &self,
        p: &MoePrefillParams<'_>,
        base: *mut std::ffi::c_void,
        stride: usize,
        rows: usize,
        k: usize,
    ) -> GpuTensor {
        // SAFETY: `fp8_validate` proved, for this table, an immutable expert
        // set (`immutable_identity`) of `p.n_exp` resident experts whose first
        // and last block addresses are `base` and `base + (n_exp - 1) * stride`
        // with every block's buffer at least `stride` bytes (the parent binder
        // proves every in-between block the same way), so `n_exp * stride`
        // bytes from `base` are live device memory owned by those experts for
        // as long as `p` borrows them.  The wrapper is `Borrowed`: drop never
        // frees it.
        let buf = unsafe { hip_bridge::DeviceBuffer::from_raw(base, p.n_exp * stride) };
        GpuTensor {
            buf,
            shape: vec![p.n_exp, rows, k],
            dtype: DType::FP8E4M3B128,
        }
    }

    fn gate_up_table(&self, p: &MoePrefillParams<'_>) -> GpuTensor {
        self.table(
            p,
            self.gate_up_base,
            self.gate_up_stride,
            2 * p.mi,
            p.gate_up_k,
        )
    }

    fn down_table(&self, p: &MoePrefillParams<'_>) -> GpuTensor {
        self.table(p, self.down_base, self.down_stride, p.down_m, p.down_k)
    }
}

fn fp8_need_bytes(tensor: &GpuTensor, name: &str, need: usize) -> Result<(), DispatchError> {
    let have = tensor_bytes(tensor);
    if have < need {
        return Err(fp8_refuse(format!(
            "{name} holds {have} bytes, the route needs {need}"
        )));
    }
    Ok(())
}

fn fp8_need_f32(tensor: &GpuTensor, name: &str, need: usize) -> Result<(), DispatchError> {
    if tensor.dtype != DType::F32 {
        return Err(fp8_refuse(format!(
            "{name} is {:?}, the route needs F32",
            tensor.dtype
        )));
    }
    fp8_need_bytes(tensor, name, need)
}

/// A BF16 router/shared weight of exactly `[m, k]`, with no rotation or AWQ
/// sidecar (its `project_weight` reads the natural activation).
fn fp8_bf16_weight(w: &WeightRef<'_>, name: &str, m: usize, k: usize) -> Result<(), DispatchError> {
    if w.dtype != DType::BF16 || w.m != m || w.k != k {
        return Err(fp8_refuse(format!(
            "{name} is {:?} [{}, {}], the route needs BF16 [{m}, {k}]",
            w.dtype, w.m, w.k
        )));
    }
    if w.rotation.is_some() || w.awq_scale.is_some() {
        return Err(fp8_refuse(format!(
            "{name} carries a rotation or AWQ sidecar"
        )));
    }
    let need = scratch_mul(scratch_mul(m, k)?, BF16_BYTES)?;
    if w.buf.buf.size() < need {
        return Err(fp8_refuse(format!(
            "{name} buffer holds {} bytes, [{m}, {k}] BF16 needs {need}",
            w.buf.buf.size()
        )));
    }
    Ok(())
}

/// Base address of one qt=55 table from the first and last expert's block:
/// the layout, residency, alignment and contiguity the whole-table view relies
/// on.
fn fp8_expert_base(
    first: &WeightRef<'_>,
    last: &WeightRef<'_>,
    what: &str,
    rows: usize,
    k: usize,
    experts: usize,
    stride: usize,
) -> Result<*mut std::ffi::c_void, DispatchError> {
    for (label, w) in [("first", first), ("last", last)] {
        if w.dtype != DType::FP8E4M3B128 || w.m != rows || w.k != k {
            return Err(fp8_refuse(format!(
                "{what} {label} expert is {:?} [{}, {}], the route needs FP8E4M3B128 [{rows}, {k}]",
                w.dtype, w.m, w.k
            )));
        }
        if w.rotation.is_some() || w.awq_scale.is_some() {
            return Err(fp8_refuse(format!(
                "{what} {label} expert carries a rotation or AWQ sidecar"
            )));
        }
        if w.buf.buf.is_host_mapped() {
            return Err(fp8_refuse(format!("{what} {label} expert is host-mapped")));
        }
        if w.buf.buf.size() < stride {
            return Err(fp8_refuse(format!(
                "{what} {label} expert block holds {} bytes, the stride is {stride}",
                w.buf.buf.size()
            )));
        }
    }
    let base = first.buf.buf.as_ptr();
    if (base as usize) % rdna_compute::fp8_e4m3::B128_ALIGN != 0 {
        return Err(fp8_refuse(format!(
            "{what} base {:p} is not 256-byte aligned",
            base
        )));
    }
    let want_last = scratch_mul(experts - 1, stride)?
        .checked_add(base as usize)
        .ok_or_else(scratch_overflow)?;
    if last.buf.buf.as_ptr() as usize != want_last {
        return Err(fp8_refuse(format!(
            "{what} experts are not one contiguous table of {stride}-byte blocks"
        )));
    }
    Ok(base)
}

/// Preflight of the whole FP8 route at `grouped_rows`: the declaration and
/// device, the operands the legacy arms own (all refused), the grouped extent,
/// every scratch/router/shared/expert/route buffer any FP8 stage touches, and
/// the qt=55 tables.  Pure host checks: no launch, no allocation of device
/// memory.  Run by every FP8 stage before its first launch.
fn fp8_validate(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<Fp8Plan, DispatchError> {
    fp8_require_geometry(p)?;
    let Some(policy) = p.route_policy else {
        return Err(fp8_refuse("no route policy"));
    };
    if !gpu.arch_caps.is_gfx942() {
        return Err(fp8_not_gfx942());
    }
    for (name, present) in [
        ("expert_stage_ptrs", p.expert_stage_ptrs.is_some()),
        ("ep_row_gather", p.ep_row_gather.is_some()),
        ("ep_tile_exchange", p.ep_tile_exchange.is_some()),
        ("expert_dtype_tags", p.expert_dtype_tags.is_some()),
        ("expert_down_awq_ptrs", p.expert_down_awq_ptrs.is_some()),
        ("paro_gate_up", p.paro_gate_up.is_some()),
        ("paro_down", p.paro_down.is_some()),
        ("down_awq_scale", p.down_awq_scale.is_some()),
    ] {
        if present {
            return Err(fp8_refuse(format!("{name} belongs to a legacy route")));
        }
    }

    let rows = p.batch_size;
    let slots = scratch_mul(rows, p.k_top)?;
    let bound = fp8_grouped_rows_bound(slots, p.n_exp)?;
    if grouped_rows != p.m_total_max
        || grouped_rows < bound
        || grouped_rows % crate::families::moe::MOE_GROUPED_BLOCK_M != 0
        || i32::try_from(grouped_rows).is_err()
    {
        return Err(fp8_refuse(format!(
            "grouped rows {grouped_rows} must equal m_total_max {} and cover {bound}",
            p.m_total_max
        )));
    }
    let n_tiles = grouped_tiles(grouped_rows);

    // Scratch: the same record the tests compare, against what is physically held.
    ScratchRoute::fp8()
        .requirement(&ScratchDims {
            rows,
            k_top: p.k_top,
            mi: p.mi,
            down_m: p.down_m,
            grouped_rows,
            shared_down: p.prelude.shared.is_some() && p.shared_down_out.is_none(),
        })?
        .require_within(&ScratchBytes::held(p))?;
    if p.prelude.shared.is_some() && p.shared_down_out.is_some() {
        require_shared_rows(p, shared_down_bytes(rows, p.down_m)?)?;
    }
    if p.rot_batch.dtype != DType::F32 {
        return Err(fp8_refuse("rot_batch must be F32 (BF16 values)"));
    }

    // Route and activation operands.
    let index_bytes = scratch_mul(slots, F32_BYTES)?;
    fp8_need_bytes(p.topk_indices, "topk_indices", index_bytes)?;
    fp8_need_bytes(p.topk_weights, "topk_weights", index_bytes)?;
    fp8_need_bytes(p.inverse_perm, "inverse_perm", index_bytes)?;
    fp8_need_bytes(
        p.sorted_slot_index,
        "sorted_slot_index",
        scratch_mul(grouped_rows, F32_BYTES)?,
    )?;
    fp8_need_bytes(
        p.expert_tile_ids,
        "expert_tile_ids",
        scratch_mul(n_tiles, F32_BYTES)?,
    )?;
    fp8_need_bytes(
        p.expert_token_counts,
        "expert_token_counts",
        scratch_mul(p.n_exp, F32_BYTES)?,
    )?;
    fp8_need_bytes(
        p.expert_offsets,
        "expert_offsets",
        scratch_mul(p.n_exp + 1, F32_BYTES)?,
    )?;
    let x_elem = match p.x_norm_batch.dtype {
        DType::F32 => F32_BYTES,
        DType::BF16 => BF16_BYTES,
        other => {
            return Err(fp8_refuse(format!(
                "x_norm_batch is {other:?}, the route needs F32 or BF16"
            )))
        }
    };
    fp8_need_bytes(
        p.x_norm_batch,
        "x_norm_batch",
        scratch_mul(scratch_mul(rows, p.gate_up_k)?, x_elem)?,
    )?;
    let target = p.routed_out.unwrap_or(p.x_batch);
    fp8_need_f32(
        target,
        "combine target",
        scratch_mul(scratch_mul(rows, p.down_m)?, F32_BYTES)?,
    )?;

    // Router and shared expert: BF16 `project_weight`, F32 outputs.
    let router = &p.prelude.router;
    fp8_bf16_weight(router, "router", p.n_exp, p.gate_up_k)?;
    fp8_need_f32(
        p.prelude.router_logits,
        "router_logits",
        scratch_mul(scratch_mul(rows, p.n_exp)?, F32_BYTES)?,
    )?;
    if let Some(shared) = p.prelude.shared.as_ref() {
        let w = &shared.weights;
        let inter = shared.intermediate;
        if inter != policy.geometry.shared_intermediate {
            return Err(fp8_refuse(format!(
                "shared intermediate {inter} is not the declared {}",
                policy.geometry.shared_intermediate
            )));
        }
        fp8_bf16_weight(&w.selector, "shared selector", 1, p.gate_up_k)?;
        fp8_bf16_weight(&w.gate, "shared gate", inter, p.gate_up_k)?;
        fp8_bf16_weight(&w.up, "shared up", inter, p.gate_up_k)?;
        fp8_bf16_weight(&w.down, "shared down", p.down_m, inter)?;
        let live = scratch_mul(scratch_mul(rows, inter)?, F32_BYTES)?;
        fp8_need_f32(
            shared.scalar,
            "shared scalar",
            scratch_mul(rows, F32_BYTES)?,
        )?;
        fp8_need_f32(shared.gate_out, "shared gate_out", live)?;
        fp8_need_f32(shared.up_out, "shared up_out", live)?;
        fp8_need_f32(shared.rotated, "shared rotated", live)?;
    }

    // Routed experts: one immutable, resident, contiguous qt=55 table per projection.
    let experts = p.routed_experts;
    if experts.immutable_identity().is_none() {
        return Err(fp8_refuse(
            "the routed expert set has no immutable identity",
        ));
    }
    if experts.host_mapped() || experts.len() != p.n_exp {
        return Err(fp8_refuse(format!(
            "the routed expert set must be {} resident experts (len {}, host_mapped {})",
            p.n_exp,
            experts.len(),
            experts.host_mapped()
        )));
    }
    let (Some((first_gu, first_dn)), Some((last_gu, last_dn))) =
        (experts.get(0), experts.get(p.n_exp - 1))
    else {
        return Err(fp8_refuse(
            "the routed expert set does not resolve its first and last expert",
        ));
    };
    let gate_up_stride = fp8_block_stride(2 * p.mi, p.gate_up_k)?;
    let down_stride = fp8_block_stride(p.down_m, p.down_k)?;
    let gate_up_base = fp8_expert_base(
        &first_gu,
        &last_gu,
        "gate_up",
        2 * p.mi,
        p.gate_up_k,
        p.n_exp,
        gate_up_stride,
    )?;
    let down_base = fp8_expert_base(
        &first_dn,
        &last_dn,
        "down",
        p.down_m,
        p.down_k,
        p.n_exp,
        down_stride,
    )?;
    Ok(Fp8Plan {
        rows,
        slots,
        grouped_rows,
        n_tiles,
        gate_up_base,
        down_base,
        gate_up_stride,
        down_stride,
    })
}

fn fp8_shared<'p, 'a>(
    p: &'p MoePrefillParams<'a>,
) -> Result<&'p crate::families::moe::MoeSharedPrefill<'a>, DispatchError> {
    p.prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))
}

/// No rotation and no pass: validation only.  The expert GEMMs read
/// `x_norm_batch` directly (the F32 variant RNE-stages each element to BF16 in
/// its MFMA LDS tile, a BF16 input is the boundary already), and the router /
/// shared `project_weight`s stage the same BF16, so every BF16 operand is
/// produced by a launcher and `x_norm_batch` is never modified.  Launches nothing.
fn fp8_input_basis(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    fp8_validate(gpu, p, p.m_total_max).map(drop)
}

/// BF16 router through `project_weight` (gfx942 BF16 MFMA at every row count).
fn fp8_router_projection(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    fp8_validate(gpu, p, p.m_total_max)?;
    super::project_weight(
        gpu,
        &p.prelude.router,
        p.x_norm_batch,
        p.prelude.router_logits,
        p.batch_size,
        None,
    )
}

/// Shared selector/gate/up as one `project_weights` group (gfx942 BF16 MFMA
/// per member, one BF16 staging of the activation, every row count).
fn fp8_shared_gate_up(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    fp8_validate(gpu, p, p.m_total_max)?;
    let shared = fp8_shared(p)?;
    let weights = &shared.weights;
    super::layer_ops::project_weights(
        gpu,
        p.x_norm_batch,
        p.batch_size,
        None,
        &[
            (&weights.selector, shared.scalar),
            (&weights.gate, shared.gate_out),
            (&weights.up, shared.up_out),
        ],
    )
}

/// `fp8b128_moe_shared_activation` over the live rows: BF16 rounds of gate, up,
/// SiLU and product into `rotated`, the selector as the kernel defines it
/// (the shared down's scaled add rounds the selector again on read).
fn fp8_shared_activation(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    fp8_validate(gpu, p, p.m_total_max)?;
    let shared = fp8_shared(p)?;
    let live = p.batch_size * shared.intermediate;
    let (gate, up, rotated) = (
        f32_view(shared.gate_out, 0, live),
        f32_view(shared.up_out, 0, live),
        f32_view(shared.rotated, 0, live),
    );
    hip(gpu
        .try_gfx942()
        .ok_or_else(fp8_not_gfx942)?
        .fp8b128_moe_shared_activation(
            &gate,
            &up,
            &rotated,
            shared.scalar,
            p.batch_size,
            shared.intermediate,
        ))
}

/// Unfused shared down: `project_weight` of the BF16 down into the F32 rows of
/// `shared_down_out` (else `down_expanded`; the FP8 combine never reads it),
/// then `bf16_scaled_add_batched` into the combine target (selector rounded
/// again on read, residual/value/product/sum each BF16).
fn fp8_shared_down(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    fp8_validate(gpu, p, p.m_total_max)?;
    let shared = fp8_shared(p)?;
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    let out = f32_view(
        p.shared_down_out.unwrap_or(p.down_expanded),
        0,
        p.batch_size * p.down_m,
    );
    super::project_weight(gpu, down, shared.rotated, &out, p.batch_size, None)?;
    bf16_scaled_add_batched(
        gpu,
        &Bf16ScaledAddBatched {
            residual: target,
            value: &out,
            scalar: shared.scalar,
            rows: p.batch_size,
            elements: p.down_m,
        },
    )
    .map_err(|error| DispatchError::Hip(error.to_string()))
}

/// `moe_scatter_stable_top10` (three launches: ranks, prefix, scatter) then
/// `fp8b128_moe_tile_desc_pad16` once into the `gate_batch` byte prefix, which
/// both FP8 GEMMs read.  Nothing else writes `gate_batch` on this route.
fn fp8_scatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    let plan = fp8_validate(gpu, p, grouped_rows)?;
    hip(gpu.moe_scatter_stable_top10(
        p.topk_indices,
        p.expert_token_counts,
        p.expert_offsets,
        p.sorted_slot_index,
        p.expert_tile_ids,
        p.inverse_perm,
        plan.slots,
        p.n_exp,
        plan.grouped_rows,
    ))?;
    hip(gpu
        .try_gfx942()
        .ok_or_else(fp8_not_gfx942)?
        .fp8b128_moe_tile_desc_pad16(p.expert_tile_ids, p.gate_batch, plan.n_tiles))
}

/// gate_up over `x_norm_batch` (X row = slot / k_top): F32 `[G, 2 mi]`.
fn fp8_gate_up(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    let plan = fp8_validate(gpu, p, grouped_rows)?;
    let table = plan.gate_up_table(p);
    hip(gpu
        .try_gfx942()
        .ok_or_else(fp8_not_gfx942)?
        .gemm_fp8b128_moe_grouped_bf16mfma(
            &table,
            plan.gate_up_stride,
            p.n_exp,
            p.gate_batch,
            plan.n_tiles,
            p.sorted_slot_index,
            p.x_norm_batch,
            p.k_top,
            plan.rows,
            p.y_gate_up_grouped,
            plan.grouped_rows,
            2 * p.mi,
            p.gate_up_k,
        ))
}

/// `fp8b128_moe_unscatter_silu` into `rot_batch` (flat slot rows, F32 holding
/// BF16 values): BF16 rounds of gate, up, SiLU and product.
fn fp8_unscatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    let plan = fp8_validate(gpu, p, grouped_rows)?;
    hip(gpu
        .try_gfx942()
        .ok_or_else(fp8_not_gfx942)?
        .fp8b128_moe_unscatter_silu(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.rot_batch,
            p.mi,
            plan.grouped_rows,
            plan.slots,
        ))
}

/// down over the unscattered activation (X row = flat slot): F32 `[G, hidden]`.
fn fp8_down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    let plan = fp8_validate(gpu, p, grouped_rows)?;
    let table = plan.down_table(p);
    hip(gpu
        .try_gfx942()
        .ok_or_else(fp8_not_gfx942)?
        .gemm_fp8b128_moe_grouped_bf16mfma(
            &table,
            plan.down_stride,
            p.n_exp,
            p.gate_batch,
            plan.n_tiles,
            p.sorted_slot_index,
            p.rot_batch,
            1,
            plan.slots,
            p.y_down_grouped,
            plan.grouped_rows,
            p.down_m,
            p.down_k,
        ))
}

/// `moe_down_combine_grouped_top10` (F32 grouped rows) into the target, then
/// the BF16 round trip of the live target rows when no shared down follows.
/// Per token the ten ranks are folded in ascending expert-ID order through
/// `inverse_perm`, every weight, product and sum rounded to BF16, no atomics.
fn fp8_combine(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
    initial_zero: bool,
) -> Result<(), DispatchError> {
    // Foreign experts would fold unmasked: refuse before any launch.
    super::combine_leaf(super::CombineArm::Fp8Grouped, reduction_partition(p))?;
    if initial_zero {
        return Err(fp8_refuse(
            "a zero-initialized combine target is not available",
        ));
    }
    let plan = fp8_validate(gpu, p, grouped_rows)?;
    let target = p.routed_out.unwrap_or(p.x_batch);
    hip(gpu.moe_down_combine_grouped_top10(
        p.y_down_grouped,
        p.inverse_perm,
        p.topk_indices,
        p.topk_weights,
        target,
        p.down_m,
        plan.grouped_rows,
        plan.rows,
        p.n_exp,
    ))?;
    if !p.recipe.shared_after_combine() {
        hip(gpu.bf16_round_trip_f32(&f32_view(target, 0, plan.rows * p.down_m)))?;
    }
    Ok(())
}

// ───────────── S1 slow-link layer-major prefill hooks (`slowlink_prefill`) ─────────────

/// Which lowering of the incumbent a tile call takes inside the S1 window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SlowlinkTile {
    /// Sealed expert-slab K2: the grouped IU4 kernels (`route_rows > few_row_limit`).
    SlabK2,
    /// Sealed expert-slab K1 short tail: the lane-union GEMV, run per expert binding by
    /// [`slowlink_k1_panel`].
    SlabK1,
    /// Aggregate stacked-expert QT44 gate/up + QT53 down (no slab) on the F16 WMMA grouped
    /// route end to end: SwiGLU in the gate/up epilogue, BF16 grouped rows, the down's own
    /// unscatter + 128-wide rotation, the BF16-input combine ([`wmma_tile_route`]).
    AggregateWmma,
    /// The same aggregate route on the symmetric IU4 arm ([`sym_iu4`]: the layer's experts
    /// verified symmetric at load, the recipe, F32 activation and the device admitting it for
    /// `>= 512` route rows): the window's K2 tile of an `AggregateSymIu4` window, run by the
    /// grouped NT1 IU4 kernels over the window's A4 sidecars.
    AggregateIu4,
    /// The same aggregate route below the WMMA row threshold: the incumbent's F32 grouped
    /// (or, up to `few_row_limit` rows, F32-grouped gate/up + indexed down) arm.  Run per
    /// expert binding by [`slowlink_f32_tail_panel`], through the incumbent's own stage
    /// functions over the tile's route ids masked to the binding.
    AggregateF32Tail,
}

/// Pre-mutation admission of one tile call into the S1 layer-major window: the plain grouped
/// tail (no EP, CPU split/splice, stage tables, routed-out partial or FP8), BF16-row combine
/// arithmetic and a shared down after the combine, over either
///
/// * the sealed expert-slab route on a uniform MQ4 layer ([`SlowlinkTile::SlabK2`] /
///   [`SlowlinkTile::SlabK1`], the latter the incumbent's lane-union GEMV arithmetic for
///   `route_rows <= few_row_limit`), or
/// * the aggregate stacked-expert QT44/QT53 route (no slab declaration; the declared capability
///   is `Qt44Qt53Grouped`/`Qt44Qt53GroupedSymmetric`, uniform dtypes, no per-expert tags) whose
///   tile takes the F16 WMMA grouped arm ([`SlowlinkTile::AggregateWmma`]), the symmetric IU4
///   arm ([`SlowlinkTile::AggregateIu4`]: exactly the incumbent's own [`sym_iu4`] predicate) or,
///   below the WMMA row threshold, the F32 grouped arm ([`SlowlinkTile::AggregateF32Tail`]).
///   Path 1 has no panel lowering and is refused here.
///
/// `use_path2` is the call's own grouped-rows selection.  `Err` means the caller must stay on
/// the incumbent route; nothing has been launched or written.
pub(super) fn slowlink_admission(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<SlowlinkTile, DispatchError> {
    use rdna_compute::slab_moe::SlabBuckets;
    let refuse = |why: &str| {
        Err(DispatchError::Hip(format!(
            "slowlink layer-major prefill: {why}; staying on the incumbent route"
        )))
    };
    let slab = is_slab(p);
    if is_fp8(p) {
        return refuse("the FP8 grouped route has no panel lowering");
    }
    if slab && !super::slab_route::arch_supported(&gpu.arch) {
        return refuse("the sealed expert-slab route is gfx1100/gfx1151/gfx1201-only");
    }
    if !slab
        && !p.route_policy.is_some_and(|policy| {
            matches!(
                policy.capability,
                MoeRouteCapability::Qt44Qt53Grouped | MoeRouteCapability::Qt44Qt53GroupedSymmetric
            )
        })
    {
        return refuse(
            "the call declares neither the sealed expert-slab nor the aggregate QT44/QT53 route",
        );
    }
    if !grouped_geometry(p) {
        return refuse("the declared grouped geometry is not admitted");
    }
    if p.batch_size == 0
        || p.k_top != 10
        || p.cpu_deferred_experts
        || p.expert_stage_ptrs.is_some()
        || p.routed_out.is_some()
        || p.ep_row_gather.is_some()
        || p.ep_tile_exchange.is_some()
    {
        return refuse("the call is not a plain single-device grouped tail");
    }
    if !p.recipe.bf16_round_trip()
        || !p.recipe.shared_after_combine()
        || p.x_norm_batch.dtype != DType::F32
        || p.down_k != p.mi
    {
        return refuse("the recipe is not the BF16-row combine with the shared down after it");
    }
    if p.down_m % 64 != 0
        || p.down_m % 8 != 0
        || p.gate_up_k % 256 != 0
        || p.down_k % 128 != 0
        || (gpu.arch == "gfx1100" && p.down_k != 640)
    {
        return refuse("the projection geometry is not the sealed grouped geometry");
    }
    if !slab {
        if p.expert_dtype_tags.is_some() {
            return refuse("the aggregate layer carries per-expert dtype tags");
        }
        // The incumbent's own predicate (layer policy `Qt44Qt53GroupedSymmetric`, BF16 recipe,
        // F32 activation, `qwen4_moe_sym_iu4_applies(rows)`): such a tile is the K2 tile of
        // an `AggregateSymIu4` window, and `PreparedChunkCall::new` refuses it inside any
        // other window (an F16 window with an IU4-eligible layer bypasses to the incumbent).
        if sym_iu4(gpu, p, use_path2) {
            return Ok(SlowlinkTile::AggregateIu4);
        }
        if wmma_tile_route(gpu, p, use_path2) {
            return Ok(SlowlinkTile::AggregateWmma);
        }
        // The F32 grouped arms of path 2: neither WMMA predicate holds and the unscatter
        // rotates the F32 rows itself (`unscatter_rotates`).  Anything else (path 1, a
        // half-enabled WMMA arm) is not one of the arms this lowering reproduces.
        if use_path2 && !gateup_bf16(gpu, p) && !down_wmma(gpu, p) && unscatter_rotates(gpu, p) {
            // The tile's own scratch must hold its F32 grouped arm: checked here, before any
            // launch, by the incumbent's own validation.
            validate_scratch(gpu, p, use_path2, p.m_total_max)?;
            return Ok(SlowlinkTile::AggregateF32Tail);
        }
        return refuse("the aggregate tile takes an arm that has no panel lowering");
    }
    let layer = super::slab_route::layer(
        &gpu.arch,
        p.route_policy.as_ref(),
        &p.dtypes,
        p.expert_dtype_tags,
    )?;
    match layer.buckets {
        SlabBuckets::Uniform(bucket) if bucket != 0 => {}
        // A mixed MQ4 + cold-codec layer (mq3-xts): its S1 K2 panel runs per codec bucket on
        // codec-masked tiles (`slab_route::k2_panel_gate_up` / `k2_panel_down`).
        SlabBuckets::Table(_) => {}
        _ => return refuse("the layer is neither uniform MQ4 nor a mixed MQ4 + cold-codec slab"),
    }
    // The recipe/dtype/geometry predicates of `slab_k2` hold, so it differs from a K1 call
    // only in the route rows.
    Ok(if slab_k2(gpu, p) {
        SlowlinkTile::SlabK2
    } else {
        SlowlinkTile::SlabK1
    })
}

/// One expert binding of the K1 short-tail cohort of one tile: the incumbent's K1 path (rotate,
/// lane-union gate/up GEMV, BF16 round trips, SwiGLU + 128-wide rotation, lane-union down GEMV)
/// over `ids` (the tile's route ids masked to the binding, so slots whose expert the binding does
/// not hold take the kernels' no-load exit and leave their rows untouched), reading the
/// binding's GLOBAL-id pointer tables and writing the unweighted F32 down rows of the binding's
/// slots into `y0` (`[rows * 10, down_m]`).  `p` is the tile's call with `x_norm_batch` restored.
pub(super) fn slowlink_k1_panel(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    ids: &GpuTensor,
    gate_up_table: &GpuTensor,
    down_table: &GpuTensor,
    y0: &GpuTensor,
) -> Result<(), DispatchError> {
    if slab_k2(gpu, p) {
        return Err(DispatchError::Hip(
            "slowlink layer-major prefill: the K1 cohort tile is a K2 call".into(),
        ));
    }
    let layer = super::slab_route::layer(
        &gpu.arch,
        p.route_policy.as_ref(),
        &p.dtypes,
        p.expert_dtype_tags,
    )?;
    let total_slots = p.batch_size * p.k_top;
    hip(gpu.rotate_x_mq_batched(p.x_norm_batch, p.x_rot_batch, p.gate_up_k, p.batch_size))?;
    super::slab_route::k1_gate_up(
        gpu,
        layer,
        gate_up_table,
        ids,
        p.x_rot_batch,
        p.gate_batch,
        p.up_batch,
        p.batch_size,
    )?;
    let active = total_slots * p.mi;
    let (gate, up, rot) = (
        f32_view(p.gate_batch, 0, active),
        f32_view(p.up_batch, 0, active),
        f32_view(p.rot_batch, 0, active),
    );
    hip(gpu.bf16_round_trip_f32(&gate))?;
    hip(gpu.bf16_round_trip_f32(&up))?;
    hip(gpu.silu_mul_f32(&gate, &up, &rot))?;
    hip(gpu.bf16_round_trip_f32(&rot))?;
    hip(gpu.rotate_x_mq_128_v2(p.rot_batch, p.rot_batch, p.mi, total_slots))?;
    let rot = f32_view(p.rot_batch, 0, active);
    super::slab_route::k1_down(gpu, layer, ids, down_table, &rot, y0, p.batch_size)
}

/// The shared expert's activation of an aggregate F32-tail tile, run by the window's prepare
/// step.  On the incumbent's F32 grouped arm the activation rides in the routed unscatter launch
/// ([`shared_in_unscatter`]) and the prepare stage skips it; the window's routed tail runs the
/// unscatter per expert binding on the tile's route ids masked to the binding, with no shared
/// expert in it, so the activation is produced here, once, by the very launch the fused path
/// folds in (`shared_expert_activation_bf16_f32`), exactly as [`shared_activation`] runs it when
/// it is not fused.  Nothing to do when the prepare stage already produced it.
pub(super) fn slowlink_tail_shared_activation(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    if !(use_path2 && shared_in_unscatter(gpu, p)) || shared_gu_pm(gpu, p, use_path2) {
        return Ok(());
    }
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let live = p.batch_size * shared.intermediate;
    let (gate, up, rotated) = (
        f32_view(shared.gate_out, 0, live),
        f32_view(shared.up_out, 0, live),
        f32_view(shared.rotated, 0, live),
    );
    hip(gpu.shared_expert_activation_bf16_f32(&gate, &up, &rotated, shared.scalar, p.batch_size))
}

/// One expert binding of an aggregate F32-tail tile: the incumbent's own routed tail for the
/// tile (its stable-or-fused scatter, F32 grouped gate/up, unscatter + 128-wide rotation,
/// down arm and combine, each the incumbent's own stage function over the tile's own call, so
/// every kernel choice is the tile's) over the tile's route ids masked to the binding
/// (`ids`, `-1` for an expert the binding does not hold: the scatter leaves such slots
/// unrouted, the combine skips them) and the binding's GLOBAL-id pointer tables.  The combine
/// accumulates in place into `target` (the tile's rows of the window accumulator); bindings run
/// in ascending expert order, and both F32 combines pick ranks by ascending expert id and add no
/// trailing zero products, so the in-place chain over bindings is the one-launch combine.  `p`
/// is the tile's call with `x_norm_batch` and `topk_weights` restored by the caller; the shared
/// expert is not part of it (its activation ran in [`slowlink_tail_shared_activation`], its down
/// runs after, in the window's shared stage).
pub(super) fn slowlink_f32_tail_panel(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    ids: &GpuTensor,
    gate_up_table: &GpuTensor,
    down_table: &GpuTensor,
    target: &GpuTensor,
) -> Result<(), DispatchError> {
    if !use_path2 || wmma_tile_route(gpu, p, use_path2) || !unscatter_rotates(gpu, p) {
        return Err(DispatchError::Hip(
            "slowlink layer-major prefill: the F32 tail cohort tile is not an F32 grouped call"
                .into(),
        ));
    }
    let mut view: MoePrefillParams<'_> = *p;
    view.topk_indices = ids;
    view.expert_gate_up_ptrs = gate_up_table;
    view.expert_down_ptrs = down_table;
    view.x_batch = target;
    view.routed_out = None;
    view.prelude.shared = None;
    let grouped_rows = p.m_total_max;
    input_basis(gpu, &view, use_path2)?;
    scatter(gpu, &view, grouped_rows)?;
    gate_up(gpu, &view, use_path2, grouped_rows)?;
    unscatter(gpu, &view, grouped_rows)?;
    activation(gpu, &view, use_path2)?;
    down(gpu, &view, use_path2, grouped_rows)?;
    combine(gpu, &view, use_path2, grouped_rows, false)
}

/// The shared down of one tile, accumulating into `target` (the tile's rows of the
/// window combine accumulator) instead of the call's own target.  The selector and
/// activation tensors are the call's, restored by the caller; `use_path2` is the
/// call's own (it picked the shared gate/up form).
pub(super) fn slowlink_shared_down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    target: &GpuTensor,
) -> Result<(), DispatchError> {
    let mut view: MoePrefillParams<'_> = *p;
    view.x_batch = target;
    view.routed_out = None;
    shared_down(gpu, &view, use_path2)
}

#[cfg(test)]
mod scratch_tests {
    use super::*;

    const MI: usize = 640;
    const HIDDEN: usize = 2560;
    const K_TOP: usize = 10;
    const N_EXP: usize = 512;
    const WORD: usize = 4;

    #[test]
    fn wide_grouped_down_only_changes_the_rows_above_eight() {
        for limit in [8, 16, 32] {
            for rows in 1..=40 {
                let off = indexed_down_for_rows(true, rows, limit, false, false);
                assert_eq!(off, rows <= limit);
                let on = indexed_down_for_rows(true, rows, limit, true, false);
                assert_eq!(on, rows <= limit.min(8));
                assert!(!indexed_down_for_rows(false, rows, limit, true, false));
                assert!(!indexed_down_for_rows(true, rows, limit, false, true));
            }
        }
    }

    #[test]
    fn wide_grouped_moe_keeps_the_requested_grouped_route_at_every_row_count() {
        for limit in [1, 4, 8, 9, 16, 32, 64] {
            for rows in 1..=65 {
                assert!(grouped_path_for_rows(true, rows, limit, true));
                assert!(!grouped_path_for_rows(false, rows, limit, true));
            }
        }
    }

    #[test]
    fn eager_exact_rows_override_a_frozen_grouped_stage_layout() {
        for limit in [8, 9, 16, 32, 64] {
            for rows in 1..=65 {
                assert!(!grouped_path_for_rows(false, rows, limit, false));
                assert_eq!(
                    grouped_path_for_rows(true, rows, limit, false),
                    !(rows > 8 && rows <= limit),
                );
            }
        }
        for limit in [1, 4, 8] {
            for rows in 1..=64 {
                assert!(grouped_path_for_rows(true, rows, limit, false));
            }
        }
    }

    /// The combine arm of a ROW-GATHERED call (`ep_row_gather` installed, QT44/QT53 route, no
    /// expert-stage table, not slab/FP8) over `rows` route rows, from the same pure predicates
    /// [`combine_arm`] reads. `down_wmma` stands for the BF16-row downs (F16 WMMA / symmetric IU4).
    fn gathered_arm(
        rows: usize,
        limit: usize,
        wide_grouped_moe: bool,
        wide_grouped_down: bool,
        down_wmma: bool,
    ) -> (crate::pipeline::CombineArm, bool) {
        let use_path2 = grouped_path_for_rows(true, rows, limit, wide_grouped_moe);
        let indexed = indexed_down_for_rows(use_path2, rows, limit, wide_grouped_down, down_wmma);
        let unscatter = use_path2 && !down_wmma && !indexed;
        (
            combine_arm_of(false, use_path2, indexed, unscatter, down_wmma),
            unscatter,
        )
    }

    /// Every gathered width (one row, the 2..=8 verify rows and the wide-verify 9..=64 rows) on
    /// every wide-grouped flag combination ends in the expanded top-10 arm, so a fixed reduction
    /// partition selects the partitioned leaf for each of them and nothing runs unmasked.
    #[test]
    fn a_partitioned_gathered_call_selects_the_partitioned_leaf_at_every_width() {
        use crate::pipeline::{combine_leaf, CombineArm, CombineLeaf};
        for limit in [8usize, 16, 32, 64] {
            for rows in 1..=limit {
                for wide_grouped_moe in [false, true] {
                    for wide_grouped_down in [false, true] {
                        let (arm, _) =
                            gathered_arm(rows, limit, wide_grouped_moe, wide_grouped_down, false);
                        assert_eq!(
                            arm,
                            CombineArm::ExpandedTop10,
                            "rows {rows} limit {limit} moe {wide_grouped_moe} down {wide_grouped_down}"
                        );
                        for rank in 0..2usize {
                            assert_eq!(
                                combine_leaf(arm, Some((rank, 2))).unwrap(),
                                CombineLeaf::Top10Partitioned { rank, ranks: 2 }
                            );
                        }
                        // No partition (static route, MTP head): today's unmasked leaf.
                        assert_eq!(combine_leaf(arm, None).unwrap(), CombineLeaf::Top10);
                    }
                }
            }
        }
        // The explicit widths: a single row and the default 2..=8 verify rows.
        for rows in 1..=8usize {
            let (arm, unscatter) = gathered_arm(rows, 8, false, false, false);
            assert_eq!(arm, CombineArm::ExpandedTop10, "rows {rows}");
            assert!(!unscatter, "rows {rows} take the indexed/path-1 down");
        }
    }

    /// `HIPFIRE_QWEN4_WIDE_GROUPED_MOE` + `HIPFIRE_QWEN4_WIDE_GROUPED_DOWN` keep the grouped F32
    /// down for 9..=limit rows (A8 grouped / C8 PM live inside that down), unscattered to slot
    /// order for the gather: exactly that width band takes the unscatter arm, and it combines
    /// through the expanded leaf, partitioned under a partition.
    #[test]
    fn the_grouped_down_unscatter_widths_combine_through_the_partitioned_leaf() {
        use crate::pipeline::{combine_leaf, CombineArm, CombineLeaf};
        let limit = 64usize;
        for rows in 1..=limit {
            let (arm, unscatter) = gathered_arm(rows, limit, true, true, false);
            assert_eq!(unscatter, rows > 8, "rows {rows}");
            assert_eq!(arm, CombineArm::ExpandedTop10, "rows {rows}");
            assert_eq!(
                combine_leaf(arm, Some((1, 2))).unwrap(),
                CombineLeaf::Top10Partitioned { rank: 1, ranks: 2 }
            );
        }
        // Without the wide-down flag the 9..=64 band stays on the indexed down (no unscatter).
        for rows in 9..=limit {
            let (arm, unscatter) = gathered_arm(rows, limit, true, false, false);
            assert!(!unscatter, "rows {rows}");
            assert_eq!(arm, CombineArm::ExpandedTop10);
        }
        // A chunk wider than the few-row limit keeps the grouped F32 down + unscatter.
        let (arm, unscatter) = gathered_arm(65, 8, false, false, false);
        assert!(unscatter);
        assert_eq!(arm, CombineArm::ExpandedTop10);
    }

    /// The BF16-row grouped combines (F16 WMMA / symmetric IU4 downs, > few-row limit) and the
    /// plain F32 grouped combine cannot mask a partition: refused before launch.
    #[test]
    fn the_grouped_row_combines_and_fp8_refuse_a_partition() {
        use crate::pipeline::{combine_leaf, CombineArm};
        let (bf16, _) = gathered_arm(65, 8, false, false, true);
        assert_eq!(bf16, CombineArm::GroupedBf16InOwned);
        let error = combine_leaf(bf16, Some((0, 2))).unwrap_err();
        assert!(format!("{error:?}").contains("before launch"), "{error:?}");
        // Without a gather hook the grouped F32 rows are the plain combine.
        assert_eq!(
            combine_arm_of(false, true, false, false, false),
            CombineArm::GroupedF32
        );
        assert!(combine_leaf(CombineArm::GroupedF32, Some((0, 2))).is_err());
        assert_eq!(
            combine_arm_of(true, true, false, false, true),
            CombineArm::Fp8Grouped
        );
        assert!(combine_leaf(CombineArm::Fp8Grouped, Some((1, 2))).is_err());
        // Off a partition every arm keeps its own kernel (nothing refuses).
        for arm in [
            CombineArm::GroupedBf16InOwned,
            CombineArm::GroupedF32,
            CombineArm::Fp8Grouped,
        ] {
            assert!(combine_leaf(arm, None).is_ok());
        }
    }

    #[test]
    fn router_fusion_preserves_weight_descriptors_and_mixed_q8_copies() {
        let bf16 = GpuTensor::null_for_test();
        let q8 = GpuTensor::null_for_test();
        let weight = |buf, dtype, m, k| WeightRef {
            buf,
            dtype,
            m,
            k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        };
        let router = weight(&bf16, DType::BF16, N_EXP, HIDDEN);
        let selector = weight(&bf16, DType::BF16, 1, HIDDEN);
        let gate = weight(&bf16, DType::BF16, MI, HIDDEN);
        let up = weight(&bf16, DType::BF16, MI, HIDDEN);
        let gate_copy = weight(&q8, DType::Q8_0, MI, HIDDEN);
        let up_copy = weight(&q8, DType::Q8_0, MI, HIDDEN);
        let wrong_k = weight(&bf16, DType::BF16, MI, HIDDEN + 32);
        for rows in 1..=64 {
            assert!(!grouped_path_for_rows(false, rows, 64, false));
            assert!(router_shared_bf16_basis(&router, [&selector, &gate, &up]));
            assert!(!router_shared_bf16_basis(
                &router,
                [&selector, &gate_copy, &up_copy]
            ));
            assert!(!router_shared_bf16_basis(
                &router,
                [&selector, &gate, &wrong_k]
            ));
            // Route selection borrows, never recreates/requantizes, the immutable copy.
            assert!(std::ptr::eq(gate_copy.buf, &q8));
            assert_eq!(
                (gate_copy.dtype, gate_copy.m, gate_copy.k),
                (DType::Q8_0, MI, HIDDEN)
            );
            assert_eq!(
                (up_copy.dtype, up_copy.m, up_copy.k),
                (DType::Q8_0, MI, HIDDEN)
            );
        }
    }

    fn dims(rows: usize) -> ScratchDims {
        let slots = rows * K_TOP;
        ScratchDims {
            rows,
            k_top: K_TOP,
            mi: MI,
            down_m: HIDDEN,
            grouped_rows: grouped_rows(slots),
            shared_down: true,
        }
    }

    /// `checked_grouped_m_total_bound` for 16-row blocks: slots plus up to 15
    /// padding rows per active expert.
    fn grouped_rows(slots: usize) -> usize {
        (slots + 15 * slots.min(N_EXP)).div_ceil(16) * 16
    }

    fn route(path2: bool) -> ScratchRoute {
        ScratchRoute {
            path2,
            fp8: false,
            sym_iu4: false,
            ep_a8_wide: false,
            gateup_bf16: false,
            gateup_silu: false,
            down_wmma: false,
            indexed_down: false,
        }
    }

    fn sym() -> ScratchRoute {
        ScratchRoute {
            sym_iu4: true,
            ..route(true)
        }
    }

    fn wmma() -> ScratchRoute {
        ScratchRoute {
            gateup_bf16: true,
            gateup_silu: true,
            down_wmma: true,
            ..route(true)
        }
    }

    fn wmma_unfused() -> ScratchRoute {
        ScratchRoute {
            gateup_bf16: true,
            down_wmma: true,
            ..route(true)
        }
    }

    fn indexed() -> ScratchRoute {
        ScratchRoute {
            indexed_down: true,
            ..route(true)
        }
    }

    fn fp8() -> ScratchRoute {
        ScratchRoute::fp8()
    }

    /// What the generic validator demands of a Full-width F32 scratch.
    fn full_scratch(rows: usize) -> ScratchBytes {
        let slots = rows * K_TOP;
        let m_total = grouped_rows(slots);
        ScratchBytes {
            gate_batch: slots * MI * 4,
            up_batch: slots * MI * 4,
            rot_batch: slots * MI * 4,
            down_expanded: slots * HIDDEN * 4,
            y_gate_up_grouped: m_total * 2 * MI * 4,
            y_down_grouped: m_total * HIDDEN * 4,
        }
    }

    /// The frozen compact capacities (`c = min(n, 512)`) for chunk `n`.
    fn compact_scratch(n: usize) -> ScratchBytes {
        let c = n.min(512);
        let (gc, gn) = (grouped_rows(c * K_TOP), grouped_rows(n * K_TOP));
        ScratchBytes {
            gate_batch: c * 10 * 640 * 4,
            up_batch: c * 10 * 640 * 4,
            rot_batch: c * 10 * 640 * 4,
            down_expanded: (c * 10 * 2560 * 4).max(n * 2560 * 4).max(n * 10 * 8),
            y_gate_up_grouped: (gc * 1280 * 4).max(gn * 640 * 2),
            y_down_grouped: (gc * 2560 * 4).max(gn * 2560 * 2),
        }
    }

    #[test]
    fn a_separate_shared_down_output_leaves_down_expanded_to_the_routed_rows() {
        // With `shared_down_out` the F32 shared-down rows are not charged to
        // `down_expanded` (they are checked on that tensor by `validate_scratch`).
        let rows = 8192;
        let shared = ScratchDims {
            shared_down: true,
            ..dims(rows)
        };
        let separate = ScratchDims {
            shared_down: false,
            ..dims(rows)
        };
        let with = sym().requirement(&shared).unwrap();
        let without = sym().requirement(&separate).unwrap();
        assert_eq!(with.down_expanded, shared_down_bytes(rows, HIDDEN).unwrap());
        assert_eq!(without.down_expanded, rows * 80);
        // Everything else is the same.
        assert_eq!(
            ScratchBytes {
                down_expanded: 0,
                ..with
            },
            ScratchBytes {
                down_expanded: 0,
                ..without
            }
        );
        // The few-row indexed route's routed rows are the same either way.
        let few = ScratchDims {
            shared_down: false,
            ..dims(4)
        };
        assert_eq!(
            indexed().requirement(&few).unwrap().down_expanded,
            4 * K_TOP * HIDDEN * 4
        );
    }

    #[test]
    fn symmetric_route_needs_bf16_grouped_rows_and_the_shared_down_fallback() {
        let rows = 8192;
        let g = grouped_rows(rows * K_TOP);
        assert_eq!(g, 89_600);
        let need = sym().requirement(&dims(rows)).unwrap();
        assert_eq!(need.y_gate_up_grouped, g * MI * 2);
        assert_eq!(need.y_down_grouped, g * HIDDEN * 2);
        // The F32 shared-down fallback dominates the 80-byte-per-row order.
        assert_eq!(need.down_expanded, rows * HIDDEN * 4);
        assert!(need.down_expanded > rows * 80);
        assert_eq!((need.gate_batch, need.up_batch, need.rot_batch), (0, 0, 0));
    }

    /// The wide-verify int8 chain (`HIPFIRE_QWEN4_MOE_GU_A8`) writes the symmetric arm's BF16 grouped
    /// rows and rank order, over the verify window: its requirement is the symmetric one, and both
    /// BF16 grouped buffers fit inside the F32 buffers the incumbent grouped decode chain already holds.
    #[test]
    fn wide_verify_int8_route_needs_the_symmetric_sizes_and_fits_the_incumbent_buffers() {
        let a8 = ScratchRoute {
            ep_a8_wide: true,
            ..route(true)
        };
        for rows in [9usize, 16, 32, 48, 64] {
            let need = a8.requirement(&dims(rows)).unwrap();
            assert_eq!(need, sym().requirement(&dims(rows)).unwrap(), "rows {rows}");
            let g = grouped_rows(rows * K_TOP);
            assert_eq!(need.y_gate_up_grouped, g * MI * 2);
            assert_eq!(need.y_down_grouped, g * HIDDEN * 2);
            assert_eq!((need.gate_batch, need.up_batch, need.rot_batch), (0, 0, 0));
            let incumbent = route(true).requirement(&dims(rows)).unwrap();
            assert!(
                need.y_gate_up_grouped <= incumbent.y_gate_up_grouped,
                "rows {rows}"
            );
            assert!(
                need.y_down_grouped <= incumbent.y_down_grouped,
                "rows {rows}"
            );
        }
    }

    /// The geometry the wide-verify int8 chain serves: Flash-Next gate/up K 2560, `mi == down_k == 640`,
    /// down `M = 2560`, top-10; anything else keeps the incumbent chain.
    #[test]
    fn wide_verify_int8_route_geometry_is_the_flash_next_shape() {
        assert!(ep_a8_wide_geometry(2560, 640, 640, 2560, 10));
        assert!(!ep_a8_wide_geometry(2048, 640, 640, 2560, 10));
        assert!(!ep_a8_wide_geometry(2560, 512, 640, 2560, 10));
        assert!(!ep_a8_wide_geometry(2560, 640, 512, 2560, 10));
        assert!(!ep_a8_wide_geometry(2560, 640, 640, 2048, 10));
        assert!(!ep_a8_wide_geometry(2560, 640, 640, 2560, 8));
    }

    /// The symmetric verdict of the wide-verify int8 chain: the planar layout with either QT44/QT53
    /// capability (below the IU4 floor a verified layer declares the plain one); never a shipped
    /// (non-planar) layout, FP8 or slab layer.
    #[test]
    fn wide_verify_int8_route_symmetric_verdict_is_the_planar_layout() {
        use MoeRouteCapability::*;
        assert!(ep_a8_wide_symmetric(Qt44Qt53Grouped, true));
        assert!(ep_a8_wide_symmetric(Qt44Qt53GroupedSymmetric, true));
        assert!(!ep_a8_wide_symmetric(Qt44Qt53Grouped, false));
        assert!(!ep_a8_wide_symmetric(Qt44Qt53GroupedSymmetric, false));
        assert!(!ep_a8_wide_symmetric(Fp8B128Grouped, true));
        assert!(!ep_a8_wide_symmetric(SlabCodecGrouped, true));
    }

    /// A gather that offers no owner mask (every hook but the trunk shard's static layers, and a hook
    /// whose attach built none) leaves the wide-verify int8 chain over unmasked tiles: the trait
    /// default is `None`, for every layer.
    #[test]
    fn a_gather_without_an_owner_mask_leaves_the_wide_verify_chain_unmasked() {
        use crate::families::moe::MoeRowGather;
        struct PlainGather;
        impl MoeRowGather for PlainGather {
            fn gather_rows(
                &self,
                _gpu: &mut Gpu,
                _layer_idx: u16,
                _slots: usize,
                _down_expanded: &GpuTensor,
                _topk_indices: &GpuTensor,
            ) -> Result<(), String> {
                Ok(())
            }
        }
        for layer in [0u16, 3, 47, 48] {
            assert!(
                PlainGather.decode_owner_mask(layer).is_none(),
                "layer {layer}"
            );
        }
    }

    #[test]
    fn combine_order_alone_is_eighty_bytes_per_row() {
        let mut no_shared = dims(1024);
        no_shared.shared_down = false;
        assert_eq!(
            sym().requirement(&no_shared).unwrap().down_expanded,
            1024 * 80
        );
        assert_eq!(
            wmma().requirement(&no_shared).unwrap().down_expanded,
            1024 * 80
        );
    }

    #[test]
    fn f32_grouped_arms_keep_todays_full_sizes() {
        let rows = 700;
        let g = grouped_rows(rows * K_TOP);
        let need = route(true).requirement(&dims(rows)).unwrap();
        assert_eq!(need.y_gate_up_grouped, g * 2 * MI * 4);
        assert_eq!(need.y_down_grouped, g * HIDDEN * 4);
        assert_eq!(need.rot_batch, rows * K_TOP * MI * 4);
        assert_eq!(need.down_expanded, rows * HIDDEN * 4);
        assert_eq!((need.gate_batch, need.up_batch), (0, 0));
    }

    #[test]
    fn wmma_arms_store_bf16_grouped_rows() {
        let rows = 900;
        let g = grouped_rows(rows * K_TOP);
        let fused = wmma().requirement(&dims(rows)).unwrap();
        assert_eq!(fused.y_gate_up_grouped, g * MI * 2);
        assert_eq!(fused.y_down_grouped, g * HIDDEN * 2);
        assert_eq!(fused.rot_batch, 0);
        let unfused = wmma_unfused().requirement(&dims(rows)).unwrap();
        assert_eq!(unfused.y_gate_up_grouped, g * 2 * MI * 2);
        assert_eq!(unfused.rot_batch, rows * K_TOP * MI * 4);
    }

    #[test]
    fn indexed_down_and_path1_need_the_batch_buffers() {
        for rows in 1..=64 {
            let slots = rows * K_TOP;
            let few = indexed().requirement(&dims(rows)).unwrap();
            assert_eq!(few.down_expanded, slots * HIDDEN * 4);
            assert_eq!(few.rot_batch, slots * MI * 4);
            assert_eq!(few.y_down_grouped, 0);
            let path1 = route(grouped_path_for_rows(rows > 8, rows, 64, false))
                .requirement(&dims(rows))
                .unwrap();
            assert_eq!(path1.gate_batch, slots * MI * 4);
            assert_eq!(path1.up_batch, slots * MI * 4);
            assert_eq!(path1.rot_batch, slots * MI * 4);
            assert_eq!(path1.down_expanded, slots * HIDDEN * 4);
            assert_eq!((path1.y_gate_up_grouped, path1.y_down_grouped), (0, 0));
        }
    }

    #[test]
    fn full_width_buffers_always_pass() {
        for rows in [1, 8, 9, 64, 511, 512, 513, 4096, 8192] {
            for r in [
                route(false),
                route(true),
                sym(),
                wmma(),
                wmma_unfused(),
                indexed(),
                fp8(),
            ] {
                let need = r.requirement(&dims(rows)).unwrap();
                need.require_within(&full_scratch(rows))
                    .unwrap_or_else(|e| panic!("rows={rows} route={r:?}: {e:?}"));
            }
        }
    }

    #[test]
    fn buffers_short_by_one_word_are_rejected_naming_the_buffer() {
        for rows in [8, 513, 4096] {
            for r in [
                route(false),
                route(true),
                sym(),
                wmma(),
                wmma_unfused(),
                indexed(),
                fp8(),
            ] {
                let need = r.requirement(&dims(rows)).unwrap();
                // Exactly the requirement passes.
                need.require_within(&need).unwrap();
                for (name, bytes) in need.fields() {
                    if bytes == 0 {
                        continue;
                    }
                    let mut held = need;
                    match name {
                        "gate_batch" => held.gate_batch -= WORD,
                        "up_batch" => held.up_batch -= WORD,
                        "rot_batch" => held.rot_batch -= WORD,
                        "down_expanded" => held.down_expanded -= WORD,
                        "y_gate_up_grouped" => held.y_gate_up_grouped -= WORD,
                        "y_down_grouped" => held.y_down_grouped -= WORD,
                        other => panic!("unknown buffer {other}"),
                    }
                    let err = need.require_within(&held).expect_err(name);
                    let DispatchError::Hip(message) = err else {
                        panic!("unexpected error kind for {name}");
                    };
                    assert!(message.contains(name), "{message}");
                    assert!(message.contains(&bytes.to_string()), "{message}");
                }
            }
        }
    }

    #[test]
    fn shared_down_fallback_is_owed_beyond_the_combine_order() {
        let rows = 2048;
        let need = sym().requirement(&dims(rows)).unwrap();
        let mut held = need;
        // Room for the 80-byte-per-row order only.
        held.down_expanded = rows * 80;
        let DispatchError::Hip(message) = need.require_within(&held).unwrap_err() else {
            panic!("unexpected error kind");
        };
        assert!(message.contains("down_expanded"), "{message}");
    }

    #[test]
    fn compact_capacities_admit_the_proven_routes() {
        let n = 8192;
        let held = compact_scratch(n);
        // Ordinary large prefill on the symmetric IU4 route.
        sym()
            .requirement(&dims(n))
            .unwrap()
            .require_within(&held)
            .unwrap();
        // The same route at the 4096 chunk and every smaller row count.
        for rows in [1, 8, 9, 64, 511, 512, 513, 4096, 8192] {
            sym()
                .requirement(&dims(rows))
                .unwrap()
                .require_within(&held)
                .unwrap();
        }
        // Fallback arms at <= 512 rows keep their F32 sizes inside the cap.
        for rows in [1, 8, 9, 64, 511, 512] {
            for r in [route(false), route(true), wmma(), wmma_unfused(), indexed()] {
                r.requirement(&dims(rows))
                    .unwrap()
                    .require_within(&held)
                    .unwrap();
            }
        }
    }

    #[test]
    fn compact_scratch_refuses_fallback_arms_above_the_cap() {
        let held = compact_scratch(8192);
        for r in [route(false), route(true)] {
            assert!(
                r.requirement(&dims(513))
                    .unwrap()
                    .require_within(&held)
                    .is_err(),
                "route {r:?} must be refused above 512 rows on compact scratch"
            );
        }
        // The 4096 plan too: the F32 grouped writer needs more than the cap.
        assert!(route(true)
            .requirement(&dims(1024))
            .unwrap()
            .require_within(&compact_scratch(4096))
            .is_err());
    }

    #[test]
    fn full_prefill_chunk_activation_owes_no_view_into_the_compact_batch_buffers() {
        // pp16384 on the symmetric IU4 route: compact scratch holds gate/up/rot
        // rows for `min(n, 512)` chunk rows only (13_107_200 B each), while a
        // view over the chunk's `rows * k_top * mi` would be 419_430_400 B.
        let rows = 16_384;
        let held = compact_scratch(rows);
        assert_eq!(held.gate_batch, 13_107_200);
        let slots = rows * K_TOP;
        assert_eq!(slots * MI * 4, 419_430_400);
        assert!(slots * MI * 4 > held.gate_batch);
        let need = sym().requirement(&dims(rows)).unwrap();
        assert_eq!((need.gate_batch, need.up_batch, need.rot_batch), (0, 0, 0));
        need.require_within(&held).unwrap();
        // Path 2's down consumed the rotation: the stage views nothing.
        assert_eq!(activation_batch_extent(true, slots, MI), Some(0));
        // Every row count the compact fallback arms admit views what they hold.
        for rows in [1, 8, 9, 64, 511, 512] {
            let held = compact_scratch(rows.max(512));
            let slots = rows * K_TOP;
            let live = activation_batch_extent(false, slots, MI).unwrap();
            assert_eq!(live, slots * MI);
            for bytes in [held.gate_batch, held.up_batch, held.rot_batch] {
                assert!(live * 4 <= bytes, "rows={rows}");
            }
        }
        assert_eq!(activation_batch_extent(false, usize::MAX, 2), None);
    }

    #[test]
    fn tensor_bytes_takes_the_smaller_of_shape_and_allocation() {
        let tensor = |bytes: usize, shape: &[usize]| GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(0x1000 as *mut std::ffi::c_void, bytes)
            },
            shape: shape.to_vec(),
            dtype: DType::F32,
        };
        assert_eq!(tensor_bytes(&tensor(64, &[16])), 64);
        // Shape claims more than the allocation holds.
        assert_eq!(tensor_bytes(&tensor(60, &[16])), 60);
        // Allocation larger than the logical view.
        assert_eq!(tensor_bytes(&tensor(128, &[16])), 64);
        assert_eq!(tensor_bytes(&tensor(128, &[usize::MAX, 2])), 0);
    }

    #[test]
    fn extents_that_overflow_are_errors() {
        let mut huge = dims(1);
        huge.rows = usize::MAX / 4;
        assert!(sym().requirement(&huge).is_err());
    }

    #[test]
    fn fp8_route_needs_descriptor_f32_grouped_rows_and_the_bf16_activation_at_1536_rows() {
        let rows = 1536;
        let slots = rows * K_TOP;
        let g = grouped_rows(slots);
        assert_eq!(g, 23_040);
        assert_eq!(fp8_grouped_rows_bound(slots, N_EXP).unwrap(), g);
        let need = fp8().requirement(&dims(rows)).unwrap();
        assert_eq!(
            need,
            ScratchBytes {
                // 16 bytes per 16-row tile.
                gate_batch: g,
                up_batch: 0,
                rot_batch: slots * MI * 4,
                down_expanded: rows * HIDDEN * 4,
                y_gate_up_grouped: g * 2 * MI * 4,
                y_down_grouped: g * HIDDEN * 4,
            }
        );
        need.require_within(&full_scratch(rows)).unwrap();
    }

    #[test]
    fn fp8_route_needs_the_same_buffers_at_decode_m1() {
        let slots = K_TOP;
        let g = grouped_rows(slots);
        assert_eq!(g, 160);
        assert_eq!(fp8_grouped_rows_bound(slots, N_EXP).unwrap(), g);
        let need = fp8().requirement(&dims(1)).unwrap();
        assert_eq!(
            need,
            ScratchBytes {
                gate_batch: g,
                up_batch: 0,
                rot_batch: slots * MI * 4,
                down_expanded: HIDDEN * 4,
                y_gate_up_grouped: g * 2 * MI * 4,
                y_down_grouped: g * HIDDEN * 4,
            }
        );
        need.require_within(&full_scratch(1)).unwrap();
    }

    #[test]
    fn fp8_requirement_ignores_the_legacy_arm_flags_and_charges_shared_down_only_when_owed() {
        let rows = 700;
        let base = fp8().requirement(&dims(rows)).unwrap();
        let flagged = ScratchRoute {
            path2: false,
            indexed_down: true,
            gateup_bf16: true,
            ..fp8()
        };
        assert_eq!(flagged.requirement(&dims(rows)).unwrap(), base);
        let separate = ScratchDims {
            shared_down: false,
            ..dims(rows)
        };
        assert_eq!(fp8().requirement(&separate).unwrap().down_expanded, 0);
        assert_eq!(base.down_expanded, shared_down_bytes(rows, HIDDEN).unwrap());
    }

    #[test]
    fn fp8_route_needs_full_f32_grouped_scratch() {
        // Compact scratch (`c = min(n, 512)`) holds the F32 grouped rows only up to 512 rows.
        let held = compact_scratch(8192);
        fp8()
            .requirement(&dims(512))
            .unwrap()
            .require_within(&held)
            .unwrap();
        assert!(fp8()
            .requirement(&dims(513))
            .unwrap()
            .require_within(&held)
            .is_err());
        assert!(fp8()
            .requirement(&dims(8192))
            .unwrap()
            .require_within(&held)
            .is_err());
        // One word short of the F32 gate/up rows is refused naming the buffer.
        let rows = 1536;
        let need = fp8().requirement(&dims(rows)).unwrap();
        let mut short = need;
        short.y_gate_up_grouped -= WORD;
        let DispatchError::Hip(message) = need.require_within(&short).unwrap_err() else {
            panic!("unexpected error kind");
        };
        assert!(message.contains("y_gate_up_grouped"), "{message}");
    }

    #[test]
    fn fp8_expert_strides_are_the_qt55_block_strides() {
        assert_eq!(fp8_block_stride(2 * MI, HIDDEN).unwrap(), 3_277_312);
        assert_eq!(fp8_block_stride(HIDDEN, MI).unwrap(), 1_638_656);
        for stride in [3_277_312usize, 1_638_656] {
            assert_eq!(stride % rdna_compute::fp8_e4m3::B128_ALIGN, 0);
        }
    }

    #[test]
    fn fp8_extents_that_overflow_are_errors() {
        let mut huge = dims(1);
        huge.rows = usize::MAX / 4;
        assert!(fp8().requirement(&huge).is_err());
        assert!(fp8_grouped_rows_bound(usize::MAX, N_EXP).is_err());
    }
}

#[cfg(test)]
mod hc_ar_tests {
    use super::super::layer_ops::{HyperWriteAr, HyperWriteOp};
    use super::*;

    struct NoReduce;

    impl HyperWriteAr for NoReduce {
        fn ar_hyper_write(
            &self,
            _gpu: &mut Gpu,
            _partial: &GpuTensor,
            _streams: &GpuTensor,
            _gates: &GpuTensor,
            _rows: usize,
            _state_bf16: bool,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    fn fake(ptr: usize, elems: usize, dtype: DType) -> GpuTensor {
        GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(
                    ptr as *mut std::ffi::c_void,
                    elems * dtype.size(),
                )
            },
            shape: vec![elems],
            dtype,
        }
    }

    fn weight(buf: &GpuTensor, m: usize, k: usize) -> WeightRef<'_> {
        WeightRef {
            buf,
            dtype: buf.dtype,
            m,
            k,
            row_stride: k,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        }
    }

    /// The shared down's GEMM-epilogue HC fold (and, through it, the HC row fold) is offered
    /// only the plain local HC write: a write that carries the trunk-shard all-reduce is
    /// never offered, so the shared down runs unfused and the fused all-reduce write reduces
    /// the rank's partial (a folded epilogue would write the UNREDUCED partial into the
    /// streams). Without the all-reduce the offer is exactly the pre-shard one.
    #[test]
    fn shared_down_hc_fold_is_never_offered_a_fused_all_reduce_write() {
        let (rows, wide) = (2usize, 4 * 2560);
        let streams = fake(0x10_0000, rows * wide, DType::F32);
        let norm = fake(0x20_0000, wide, DType::BF16);
        let normalized = fake(0x30_0000, rows * wide, DType::F32);
        let mixed = fake(0x40_0000, rows * 2560, DType::F32);
        let gates = fake(0x50_0000, rows * 4, DType::F32);
        let inject = fake(0x60_0000, 4 * wide, DType::BF16);
        let rotation = fake(0x70_0000, rows * 2560, DType::F32);
        let write = |ar: Option<&'static dyn HyperWriteAr>| HyperWriteOp {
            state_bf16: true,
            input: &streams,
            norm_weight: &norm,
            block_inject: weight(&inject, 4, wide),
            normalized: &normalized,
            mixed: &mixed,
            gates: &gates,
            output: &streams,
            rows,
            branches: 4,
            hidden: 2560,
            rotation: &rotation,
            ar,
        };
        let local = HcSharedDown::offer(&write(None)).expect("the plain local write is offered");
        assert_eq!((local.rows, local.hidden), (rows, 2560));
        assert!(local.state_bf16 && local.row_fold.is_none());
        assert!(HcSharedDown::offer(&write(Some(&NoReduce))).is_none());
    }
}

#[cfg(test)]
mod partial_mask_tests {
    use super::*;

    /// The kernel law of `ep_tile_mask_pm_*` (the G1 runner's CPU reference):
    /// `owned[t] = 0 <= id < n && owner[id] == rank ? id : -1`.
    fn mask(ids: &[i32], owners: &[i32], rank: i32) -> Vec<i32> {
        ids.iter()
            .map(|&id| match usize::try_from(id) {
                Ok(e) if e < owners.len() && owners[e] == rank => id,
                _ => -1,
            })
            .collect()
    }

    /// The grouped layout's tile ids: one 16-row tile per started 16 slots of each expert in
    /// ascending expert order, then the `-1` padding tail to `capacity` tiles.
    fn tile_ids(counts: &[usize], capacity: usize) -> Vec<i32> {
        let mut ids: Vec<i32> = Vec::new();
        for (expert, &count) in counts.iter().enumerate() {
            ids.extend(std::iter::repeat(expert as i32).take(count.div_ceil(16)));
        }
        assert!(ids.len() <= capacity);
        ids.resize(capacity, -1);
        ids
    }

    fn live(ids: &[i32]) -> usize {
        ids.iter().filter(|&&id| id >= 0).count()
    }

    #[test]
    fn owned_only_tile_counts_partition_the_unmasked_tiles_static_owners() {
        let counts: Vec<usize> = (0..16).map(|e| (e * 7 + 3) % 40).collect();
        let ids = tile_ids(&counts, 96);
        let owners: Vec<i32> = (0..16).map(|e| e % 2).collect();
        let (r0, r1) = (mask(&ids, &owners, 0), mask(&ids, &owners, 1));
        // Each rank keeps exactly the tiles of its experts and nothing else.
        for (rank, owned) in [(0usize, &r0), (1, &r1)] {
            let expect: usize = counts
                .iter()
                .enumerate()
                .filter(|(e, _)| e % 2 == rank)
                .map(|(_, &c)| c.div_ceil(16))
                .sum();
            assert_eq!(live(owned), expect, "rank {rank}");
        }
        // The two ranks' tile sets partition the unmasked live tiles (nothing computed twice).
        assert_eq!(live(&r0) + live(&r1), live(&ids));
        for t in 0..ids.len() {
            assert_eq!(
                [r0[t], r1[t]].iter().filter(|&&id| id >= 0).count(),
                usize::from(ids[t] >= 0)
            );
            for owned in [&r0, &r1] {
                assert!(owned[t] == ids[t] || owned[t] == -1);
            }
        }
        // Routed work per rank drops to the owned share, strictly below the unmasked count.
        assert!(live(&r0) < live(&ids) && live(&r1) < live(&ids));
    }

    #[test]
    fn balanced_owner_tables_mask_by_the_table_not_by_parity() {
        let counts = vec![33, 1, 16, 17, 0, 48, 5, 5];
        let ids = tile_ids(&counts, 24);
        // Balanced assignment: rank 0 owns the heavy experts 0, 2, 5.
        let owners = [0, 1, 0, 1, 1, 0, 1, 1];
        let r0 = mask(&ids, &owners, 0);
        assert_eq!(
            r0.iter()
                .filter(|&&id| id >= 0)
                .copied()
                .collect::<Vec<_>>(),
            [0, 0, 0, 2, 5, 5, 5]
        );
        let r1 = mask(&ids, &owners, 1);
        assert_eq!(live(&r0) + live(&r1), live(&ids));
        // A layer with every expert on one rank: the other rank launches no tile at all.
        let all_zero = [0; 8];
        assert_eq!(live(&mask(&ids, &all_zero, 1)), 0);
        assert_eq!(live(&mask(&ids, &all_zero, 0)), live(&ids));
    }

    #[test]
    fn an_all_non_owned_routing_masks_every_tile_and_padding_stays_sentinel() {
        // Every token routed to odd experts: rank 0 owns none of them.
        let counts: Vec<usize> = (0..8).map(|e| if e % 2 == 1 { 20 } else { 0 }).collect();
        let ids = tile_ids(&counts, 12);
        let owners: Vec<i32> = (0..8).map(|e| e % 2).collect();
        let r0 = mask(&ids, &owners, 0);
        assert!(r0.iter().all(|&id| id == -1), "{r0:?}");
        // Padding tiles and out-of-table ids are never live.
        let odd = [-1, 99, i32::MAX, i32::MIN, 3];
        assert_eq!(mask(&odd, &owners, 1), [-1, -1, -1, -1, 3]);
    }

    #[test]
    fn host_mapped_and_unit_layers_mask_alike() {
        // The mask reads only the tile id and the owner table: a host-mapped layer (wider
        // expert-run tile in the GEMM) and a one-expert unit layer produce the same
        // ownership partition.
        for experts in [1usize, 2, 3, 8] {
            let counts: Vec<usize> = (0..experts).map(|e| 16 * (e + 1)).collect();
            let ids = tile_ids(&counts, 64);
            let owners: Vec<i32> = (0..experts as i32).map(|e| e % 2).collect();
            assert_eq!(
                live(&mask(&ids, &owners, 0)) + live(&mask(&ids, &owners, 1)),
                live(&ids)
            );
        }
    }

    #[test]
    fn zero_fill_covers_exactly_the_grouped_bf16_rows() {
        assert_eq!(grouped_down_bytes(16, 4096), Some(16 * 4096 * 2));
        assert_eq!(grouped_down_bytes(0, 4096), Some(0));
        assert_eq!(grouped_down_bytes(usize::MAX, 2), None);
        assert_eq!(grouped_tiles(33), 3);
    }
}
