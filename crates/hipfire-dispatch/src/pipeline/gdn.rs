// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx942 batched GDN decode Core for lane steps (`HIPFIRE_QWEN4_GDN_LANES`).
//!
//! On gfx942 a one-row lane's GDN Core is the persistent-batch branch of
//! [`super::layer_ops::execute_gated_delta_net`]: convolution + gate parameters
//! (`gated_delta_conv_params_batched`), the F32 pipe128 recurrence
//! (`gated_delta_step_batched`), the gated norm (`gated_delta_gate_batched`).
//! The recurrence is the one launch that costs a launch per lane per layer for
//! nothing (30.7 us per lane at 8 lanes, `beat/L/L-K2-GDN.md`): for a group of
//! consecutive one-row Core steps this module runs every lane's convolution,
//! then ONE `qwen4_gdn_lanes_step_f32_gfx942` launch over all lanes' own
//! tensors, then every lane's gated norm. The lanes share no written byte
//! (checked here), so each lane's state and outputs equal its per-lane chain
//! byte for byte; a group the route does not take (flag off, a shape or format
//! it does not cover, a lane the per-lane path would send elsewhere) is handed
//! back untouched and runs per lane.

use super::layer_ops::{
    gdn_persistent_batch, hip, rotation_basis, view, GatedDeltaNetOp, MixerPhase, RotationBasis,
};
use crate::types::DispatchError;
use rdna_compute::cdna::gfx942::{
    Qwen4GdnLanesF32Lane, Qwen4GdnLanesF32Shape, QWEN4_GDN_LANES_F32_MAX,
};
use rdna_compute::tensor_ops::{
    gated_delta_chunk_route, gated_delta_conv_params_batched, gated_delta_gate_batched,
    GatedDeltaConvBatched, GatedDeltaGateBatched, GatedDeltaParamsBatched, GatedDeltaStepBatched,
    GdnStateFormat,
};
use rdna_compute::{Gpu, GpuTensor};

/// What [`execute_core_lanes_f32_gfx942`] did with a group.
pub(crate) enum Gfx942GdnLanes<F> {
    /// Every lane's Core ran (`before_launch` was called once, first).
    Done,
    /// Not this route's group: nothing launched and `before_launch` was not called;
    /// the caller still owns it.
    Declined(F),
}

/// Whether this context runs the gfx942 batched GDN lanes core at all: exact
/// gfx942, `HIPFIRE_QWEN4_GDN_LANES` on (default off there), no replay tape
/// recording (the device lane table is not a declared kernel input) and no graph
/// capture (a lane set's table is staged on first use, which a capture cannot do).
pub(crate) fn lanes_f32_gfx942_admitted(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx942()
        && gpu.flags.qwen4_gdn_lanes_enabled()
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
}

/// Most consecutive GDN Core steps the lane-step grouper collects into one group: the
/// batched gfx942 launch's lane count when the route is on, else the 8 of the
/// gfx1151 / gfx1201 multi-lane kernels.
pub(crate) fn core_lanes_group_max(gpu: &Gpu) -> usize {
    if lanes_f32_gfx942_admitted(gpu) {
        QWEN4_GDN_LANES_F32_MAX
    } else {
        rdna_compute::tensor_ops::GDN_CORE_LANES_MAX
    }
}

/// Whether `op` can join an [`execute_core_lanes_f32_gfx942`] group: the one-row,
/// uncaptured F32 Core step of a lane (the persistent-batch branch on gfx942).
pub(crate) fn core_lane_f32_gfx942(gpu: &Gpu, op: &GatedDeltaNetOp<'_>) -> bool {
    lanes_f32_gfx942_admitted(gpu)
        && op.phase == MixerPhase::Core
        && op.rows == 1
        && op.row_capture.is_none()
        && op.format == GdnStateFormat::F32
        && op.residual.is_none()
        && gdn_persistent_batch(gpu, op, 1)
}

/// The tensors one lane's Core chain names, exactly as the persistent-batch branch of
/// `execute_gated_delta_net_hc_impl` takes them for a Core op of one row (`lo = 0`).
struct CoreViews {
    projection: GpuTensor,
    conv_output: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
    z: GpuTensor,
    recurrent_output: GpuTensor,
    gdn_output: GpuTensor,
}

fn core_views(op: &GatedDeltaNetOp<'_>) -> CoreViews {
    let value = op.value_heads * op.value_dim;
    let qkv = 2 * op.key_heads * op.key_dim + value;
    CoreViews {
        projection: view(op.projection, 0, qkv),
        conv_output: view(op.projection2, 0, qkv),
        a: view(op.a, 0, op.value_heads),
        b: view(op.b, 0, op.value_heads),
        gate: view(op.gate, 0, op.value_heads),
        beta: view(op.beta, 0, op.value_heads),
        z: view(op.z_output, 0, value),
        recurrent_output: view(op.recurrent_output, 0, value),
        gdn_output: view(op.output_scratch, 0, value),
    }
}

/// Whether the byte ranges `(start, bytes)` are pairwise disjoint.
fn disjoint(mut ranges: Vec<(usize, usize)>) -> bool {
    ranges.sort_unstable();
    ranges.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0)
}

fn extent(t: &GpuTensor) -> (usize, usize) {
    (t.buf.as_ptr() as usize, t.buf.size())
}

/// The Core phase of a group of lanes' one-row GDN mixers as one batched recurrence
/// launch (see the module docs). `ops` are consecutive Core steps of a lane step, each
/// passing [`core_lane_f32_gfx942`]; two or more, at most [`QWEN4_GDN_LANES_F32_MAX`].
/// `before_launch` runs exactly once, after every refusal check and before the first
/// launch (the grouped steps are announced there); its error is returned as is and
/// nothing launches. Host bookkeeping (positions) is the caller's, as on the per-lane
/// path.
pub(crate) fn execute_core_lanes_f32_gfx942<F>(
    gpu: &mut Gpu,
    ops: &[&GatedDeltaNetOp<'_>],
    before_launch: F,
) -> Result<Gfx942GdnLanes<F>, DispatchError>
where
    F: FnOnce(&mut Gpu) -> Result<(), DispatchError>,
{
    let Some(first) = ops.first() else {
        return Ok(Gfx942GdnLanes::Declined(before_launch));
    };
    if ops.len() < 2 || ops.len() > QWEN4_GDN_LANES_F32_MAX || !lanes_f32_gfx942_admitted(gpu) {
        return Ok(Gfx942GdnLanes::Declined(before_launch));
    }
    let shape = (
        first.key_heads,
        first.value_heads,
        first.key_dim,
        first.value_dim,
        first.conv_kernel,
    );
    let mut lane_views = Vec::with_capacity(ops.len());
    for &op in ops {
        op.check_phase()?;
        let views = core_views(op);
        let dims = GatedDeltaStepBatched {
            projection: &views.conv_output,
            gate: &views.gate,
            beta: &views.beta,
            state: op.recurrent,
            format: op.format,
            residual: op.residual,
            output: &views.recurrent_output,
            row_states: None,
            rows: 1,
            qkv_width: views.conv_output.numel(),
            key_heads: op.key_heads,
            value_heads: op.value_heads,
            key_dim: op.key_dim,
            value_dim: op.value_dim,
            position: op.start_position,
        };
        // Everything below is what the per-lane persistent-batch branch would do that this
        // route does not replicate: a lane that would is not this route's.
        #[cfg(feature = "deltanet")]
        let dense = rdna_compute::fn_gdn_dense::applies(gpu, &dims);
        #[cfg(not(feature = "deltanet"))]
        let dense = false;
        if !core_lane_f32_gfx942(gpu, op)
            || (op.key_heads, op.value_heads, op.key_dim, op.value_dim, op.conv_kernel) != shape
            || op.conv_kernel < 2
            // The chunked WMMA recurrence and the gfx1201 dense producer (whose qkv BF16
            // storage `gdn_dense_qkv_bf16` is a refinement of): neither exists on gfx942, but
            // a lane that would take them is not this route's.
            || gated_delta_chunk_route(gpu, &dims)
            || dense
            // The per-lane gate would rotate for the output projection; a Core phase of a
            // lane step does not own that rotation scratch (every lane would write row 0).
            || rotation_basis(op.output.dtype) == Some(RotationBasis::Aligned256)
        {
            return Ok(Gfx942GdnLanes::Declined(before_launch));
        }
        lane_views.push(views);
    }
    // The three phases reorder lane chains against each other: lanes must share no written
    // buffer (the conv ring, conv output, gate / beta, recurrent output, gated output and the
    // state), or a lane's chain would see another's writes.
    let sets: [Vec<(usize, usize)>; 7] = [
        ops.iter()
            .map(|op| extent(op.conv_state))
            .collect(),
        lane_views.iter().map(|v| extent(&v.conv_output)).collect(),
        lane_views.iter().map(|v| extent(&v.gate)).collect(),
        lane_views.iter().map(|v| extent(&v.beta)).collect(),
        lane_views
            .iter()
            .map(|v| extent(&v.recurrent_output))
            .collect(),
        lane_views.iter().map(|v| extent(&v.gdn_output)).collect(),
        ops.iter().map(|op| extent(op.recurrent)).collect(),
    ];
    if !sets.into_iter().all(disjoint) {
        return Ok(Gfx942GdnLanes::Declined(before_launch));
    }
    before_launch(gpu)?;

    let value_heads = first.value_heads;
    let qkv = lane_views[0].conv_output.numel();
    let history_rows = first.conv_kernel.saturating_sub(1);
    // Convolution + gate parameters of every lane.
    for (op, v) in ops.iter().zip(&lane_views) {
        let start_cursor = op.start_position % history_rows;
        let conv = GatedDeltaConvBatched {
            input: &v.projection,
            kernel: op.conv,
            history: op.conv_state,
            output: &v.conv_output,
            next_history: op.conv_state,
            rows: 1,
            channels: qkv,
            history_rows,
            kernel_size: op.conv_kernel,
            start_cursor,
        };
        let params = GatedDeltaParamsBatched {
            a: &v.a,
            b: &v.b,
            a_log: op.a_log,
            dt_bias: op.dt_bias,
            gate: &v.gate,
            beta: &v.beta,
            rows: 1,
            heads: value_heads,
        };
        hip(gated_delta_conv_params_batched(gpu, &conv, &params))?;
    }
    // The recurrence of every lane, one launch.
    let lanes: Vec<Qwen4GdnLanesF32Lane<'_>> = ops
        .iter()
        .zip(&lane_views)
        .map(|(op, v)| Qwen4GdnLanesF32Lane {
            projection: &v.conv_output,
            gate: &v.gate,
            beta: &v.beta,
            state: op.recurrent,
            output: &v.recurrent_output,
        })
        .collect();
    let mut device = gpu.try_gfx942().ok_or_else(|| {
        DispatchError::Hip("gfx942 GDN lanes: the context is not exact gfx942".into())
    })?;
    hip(device.qwen4_gdn_lanes_step_f32(
        &lanes,
        Qwen4GdnLanesF32Shape {
            rows: 1,
            key_heads: first.key_heads,
            value_heads,
            key_dim: first.key_dim,
            value_dim: first.value_dim,
        },
    ))?;
    // The gated norm of every lane.
    for (op, v) in ops.iter().zip(&lane_views) {
        hip(gated_delta_gate_batched(
            gpu,
            &GatedDeltaGateBatched {
                recurrent_output: &v.recurrent_output,
                z: &v.z,
                norm: op.norm,
                output: &v.gdn_output,
                rows: 1,
                value_heads,
                value_dim: op.value_dim,
            },
        ))?;
    }
    Ok(Gfx942GdnLanes::Done)
}
