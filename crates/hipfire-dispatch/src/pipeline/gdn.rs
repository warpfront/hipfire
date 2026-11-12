// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx942 batched GDN lane Core for lane steps.
//!
//! On gfx942 a lane's GDN Core is the persistent-batch branch of
//! [`super::layer_ops::execute_gated_delta_net`]: convolution + gate parameters
//! (`gated_delta_conv_params_batched`), the F32 recurrence (`gated_delta_step_batched`),
//! the gated norm (`gated_delta_gate_batched`). The recurrence is the one launch that costs
//! a launch per lane per layer for nothing (30.7 us per lane at 8 lanes,
//! `beat/L/L-K2-GDN.md`): for a group of consecutive Core steps this module runs every
//! lane's convolution, then ONE batched recurrence launch over all lanes' own tensors, then
//! every lane's gated norm. Two kinds of lane join a group:
//!
//! * the one-row AR lane (no capture ring): `qwen4_gdn_lanes_step_f32_gfx942`, the
//!   `gated_delta_step_pipe128_f32` body that lane's own launch runs;
//! * the 2..=8-row verify lane under its armed capture ring (native MTP lanes, F32 state): the
//!   capture branch's chain (qkv rows copied into the capture's `inputs`, convolution + gate
//!   parameters into the retained recurrence input, the recurrence with its final state stored
//!   to ring slot `rows - 1` with the live state left intact, gated norm), the recurrence as
//!   `qwen4_gdn_lanes_step_capture_f32_gfx942`, one launch per row count in the group.
//!
//! The lanes share no written byte (checked here), so each lane's state and outputs equal its
//! per-lane chain byte for byte; a group the route does not take (a shape or format it does
//! not cover, a lane the per-lane path would send elsewhere) is handed back untouched and
//! runs per lane.

use super::layer_ops::{
    gdn_persistent_batch, hip, rotation_basis, view, GatedDeltaNetOp, GdnRowCapture, MixerPhase,
    RotationBasis,
};
use crate::types::DispatchError;
use rdna_compute::cdna::gfx942::{
    qwen4_gdn_lanes_verify_admitted, Qwen4GdnLanesF32Lane, Qwen4GdnLanesF32Shape,
    Qwen4GdnLanesVerifyLane, Qwen4GdnLanesVerifyShape, QWEN4_G942_GDN_LANES_VERIFY_MAX,
    QWEN4_GDN_LANES_F32_MAX,
};
use rdna_compute::gdn_stream::GdnStreamPhase;
use rdna_compute::tensor_ops::{
    gated_delta_chunk_route, gated_delta_conv_params_batched, gated_delta_gate_batched,
    GatedDeltaConvBatched, GatedDeltaGateBatched, GatedDeltaParamsBatched, GatedDeltaStepBatched,
    GdnStateFormat, GDN_CORE_LANE_ROWS,
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
/// gfx942 (always on there, resolved by arch), no replay tape
/// recording (the device lane table is not a declared kernel input) and no graph
/// capture (a lane set's table is staged on first use, which a capture cannot do).
pub(crate) fn lanes_f32_gfx942_admitted(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx942()
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

/// Whether `op` can join an [`execute_core_lanes_f32_gfx942`] group: the F32 Core step of a
/// lane that is either one AR row with no capture ring, or a 2..=[`GDN_CORE_LANE_ROWS`]-row
/// verify window under its armed capture ring (both the persistent-batch branch on gfx942).
pub(crate) fn core_lane_f32_gfx942(gpu: &Gpu, op: &GatedDeltaNetOp<'_>) -> bool {
    lanes_f32_gfx942_admitted(gpu)
        && op.phase == MixerPhase::Core
        && op.format == GdnStateFormat::F32
        && op.residual.is_none()
        && match &op.row_capture {
            None => op.rows == 1,
            Some(_) => (2..=GDN_CORE_LANE_ROWS).contains(&op.rows),
        }
        // An open stream must run the per-op route, which seals it.
        && !op
            .carry
            .as_ref()
            .is_some_and(|view| view.carry.meta.get().phase == GdnStreamPhase::Open)
        && gdn_persistent_batch(gpu, op, op.rows)
}

/// The tensors one one-row AR lane's Core chain names, exactly as the persistent-batch branch of
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

/// The tensors one verify lane's Core chain names, exactly as the capture branch of
/// `execute_gated_delta_net_hc_impl` takes them for a Core op (`lo = 0`): the qkv rows and the
/// retained recurrence input (`conv_output`, `gate`, `beta`) live in the capture.
struct CaptureViews {
    projection: GpuTensor,
    inputs: GpuTensor,
    conv_output: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
    z: GpuTensor,
    recurrent_output: GpuTensor,
    gdn_output: GpuTensor,
}

fn capture_views(op: &GatedDeltaNetOp<'_>, capture: &GdnRowCapture<'_>) -> CaptureViews {
    let rows = op.rows;
    let value = op.value_heads * op.value_dim;
    let qkv = 2 * op.key_heads * op.key_dim + value;
    let heads = rows * op.value_heads;
    CaptureViews {
        projection: view(op.projection, 0, rows * qkv),
        inputs: view(capture.inputs, 0, rows * qkv),
        conv_output: view(capture.recurrence, 0, rows * qkv),
        a: view(op.a, 0, heads),
        b: view(op.b, 0, heads),
        gate: view(capture.recurrence, rows * qkv, heads),
        beta: view(capture.recurrence, rows * qkv + heads, heads),
        z: view(op.z_output, 0, rows * value),
        recurrent_output: view(op.recurrent_output, 0, rows * value),
        gdn_output: view(op.output_scratch, 0, rows * value),
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

/// Whether two views cover the same device bytes (a copy from one to the other is the identity).
fn is_same_extent(a: &GpuTensor, b: &GpuTensor) -> bool {
    a.buf.as_ptr() == b.buf.as_ptr() && a.byte_size() == b.byte_size()
}

/// The shape every lane of a group shares.
type LaneShape = (usize, usize, usize, usize, usize);

/// What the per-lane persistent-batch branch would do that this route does not replicate: a lane
/// that would is not this route's. `dims` is the lane's recurrence launch.
fn lane_declined(
    gpu: &Gpu,
    op: &GatedDeltaNetOp<'_>,
    shape: LaneShape,
    dims: &GatedDeltaStepBatched<'_>,
) -> bool {
    #[cfg(feature = "deltanet")]
    let dense = rdna_compute::fn_gdn_dense::applies(gpu, dims);
    #[cfg(not(feature = "deltanet"))]
    let dense = false;
    !core_lane_f32_gfx942(gpu, op)
        || (op.key_heads, op.value_heads, op.key_dim, op.value_dim, op.conv_kernel) != shape
        || op.conv_kernel < 2
        // The chunked WMMA recurrence and the gfx1201 dense producer (whose qkv BF16
        // storage `gdn_dense_qkv_bf16` is a refinement of): neither exists on gfx942, but
        // a lane that would take them is not this route's.
        || gated_delta_chunk_route(gpu, dims)
        || dense
        // The per-lane gate would rotate for the output projection; a Core phase of a
        // lane step does not own that rotation scratch (every lane would write row 0).
        || rotation_basis(op.output.dtype) == Some(RotationBasis::Aligned256)
}

/// The Core phase of a group of lanes' GDN mixers as batched recurrence launches (see the module
/// docs). `ops` are consecutive Core steps of a lane step, each passing
/// [`core_lane_f32_gfx942`]; two or more, at most [`QWEN4_GDN_LANES_F32_MAX`]. `before_launch`
/// runs exactly once, after every refusal check and before the first launch (the grouped steps
/// are announced there); its error is returned as is and nothing launches. Host bookkeeping
/// (positions, the ring commit) is the caller's, as on the per-lane path.
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
    let shape: LaneShape = (
        first.key_heads,
        first.value_heads,
        first.key_dim,
        first.value_dim,
        first.conv_kernel,
    );
    let mut ar_lanes: Vec<(&GatedDeltaNetOp<'_>, CoreViews)> = Vec::new();
    let mut capture_lanes: Vec<(&GatedDeltaNetOp<'_>, &GdnRowCapture<'_>, CaptureViews)> =
        Vec::new();
    for &op in ops {
        op.check_phase()?;
        match op.row_capture.as_ref() {
            None => {
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
                if lane_declined(gpu, op, shape, &dims) {
                    return Ok(Gfx942GdnLanes::Declined(before_launch));
                }
                ar_lanes.push((op, views));
            }
            Some(capture) => {
                let views = capture_views(op, capture);
                let dims = GatedDeltaStepBatched {
                    projection: &views.conv_output,
                    gate: &views.gate,
                    beta: &views.beta,
                    state: op.recurrent,
                    format: op.format,
                    residual: op.residual,
                    output: &views.recurrent_output,
                    row_states: Some(&capture.states),
                    rows: op.rows,
                    qkv_width: views.conv_output.numel() / op.rows,
                    key_heads: op.key_heads,
                    value_heads: op.value_heads,
                    key_dim: op.key_dim,
                    value_dim: op.value_dim,
                    position: op.start_position,
                };
                if lane_declined(gpu, op, shape, &dims) {
                    return Ok(Gfx942GdnLanes::Declined(before_launch));
                }
                capture_lanes.push((op, capture, views));
            }
        }
    }
    // The phases reorder lane chains against each other: lanes must share no written buffer
    // (the conv ring, the recurrence inputs and outputs, the gated output and the state / its
    // capture ring), or a lane's chain would see another's writes.
    let sets: [Vec<(usize, usize)>; 10] = [
        ar_lanes
            .iter()
            .map(|(op, _)| extent(op.conv_state))
            .chain(capture_lanes.iter().map(|(op, ..)| extent(op.conv_state)))
            .collect(),
        ar_lanes.iter().map(|(_, v)| extent(&v.conv_output)).collect(),
        ar_lanes.iter().map(|(_, v)| extent(&v.gate)).collect(),
        ar_lanes.iter().map(|(_, v)| extent(&v.beta)).collect(),
        ar_lanes
            .iter()
            .map(|(_, v)| extent(&v.recurrent_output))
            .chain(capture_lanes.iter().map(|(.., v)| extent(&v.recurrent_output)))
            .collect(),
        ar_lanes
            .iter()
            .map(|(_, v)| extent(&v.gdn_output))
            .chain(capture_lanes.iter().map(|(.., v)| extent(&v.gdn_output)))
            .collect(),
        ar_lanes.iter().map(|(op, _)| extent(op.recurrent)).collect(),
        // The capture's retained qkv rows, its recurrence input (convolution output, gate,
        // beta) and the ring (with every capture lane's read-only state beside it).
        capture_lanes.iter().map(|(_, c, _)| extent(c.inputs)).collect(),
        capture_lanes
            .iter()
            .map(|(_, c, _)| extent(c.recurrence))
            .collect(),
        capture_lanes
            .iter()
            .flat_map(|(op, c, _)| [extent(&c.states), extent(op.recurrent)])
            .collect(),
    ];
    if !sets.into_iter().all(disjoint) {
        return Ok(Gfx942GdnLanes::Declined(before_launch));
    }
    // Verify lanes: the whole group (the one-row AR lanes beside them included) runs the
    // Core chain as three launches per chunk of lanes (convolution + gate parameters + the
    // capture-input copy, the recurrence, the gated norm), every lane on its own rows. A group
    // the launch refuses is handed back untouched and runs per lane.
    if !capture_lanes.is_empty() {
        let history_rows = first.conv_kernel.saturating_sub(1);
        let mut lanes: Vec<Qwen4GdnLanesVerifyLane<'_>> = Vec::with_capacity(ops.len());
        for (op, v) in &ar_lanes {
            lanes.push(Qwen4GdnLanesVerifyLane {
                projection: &v.projection,
                inputs: None,
                kernel: op.conv,
                history: op.conv_state,
                conv_output: &v.conv_output,
                a: &v.a,
                b: &v.b,
                a_log: op.a_log,
                dt_bias: op.dt_bias,
                gate: &v.gate,
                beta: &v.beta,
                state: op.recurrent,
                ring: None,
                recurrent_output: &v.recurrent_output,
                z: &v.z,
                norm: op.norm,
                gdn_output: &v.gdn_output,
                rows: 1,
                start_cursor: op.start_position % history_rows,
            });
        }
        for (op, capture, v) in &capture_lanes {
            lanes.push(Qwen4GdnLanesVerifyLane {
                projection: &v.projection,
                // Already the capture's retained input (a direct in-projection): no copy.
                inputs: (!is_same_extent(&v.projection, &v.inputs)).then_some(&v.inputs),
                kernel: op.conv,
                history: op.conv_state,
                conv_output: &v.conv_output,
                a: &v.a,
                b: &v.b,
                a_log: op.a_log,
                dt_bias: op.dt_bias,
                gate: &v.gate,
                beta: &v.beta,
                state: op.recurrent,
                ring: Some(&capture.states),
                recurrent_output: &v.recurrent_output,
                z: &v.z,
                norm: op.norm,
                gdn_output: &v.gdn_output,
                rows: op.rows,
                start_cursor: op.start_position % history_rows,
            });
        }
        let shape = Qwen4GdnLanesVerifyShape {
            key_heads: first.key_heads,
            value_heads: first.value_heads,
            key_dim: first.key_dim,
            value_dim: first.value_dim,
            conv_kernel: first.conv_kernel,
        };
        if !lanes
            .chunks(QWEN4_G942_GDN_LANES_VERIFY_MAX)
            .all(|chunk| qwen4_gdn_lanes_verify_admitted(chunk, shape).is_ok())
        {
            return Ok(Gfx942GdnLanes::Declined(before_launch));
        }
        before_launch(gpu)?;
        let mut device = gpu.try_gfx942().ok_or_else(|| {
            DispatchError::Hip("gfx942 GDN lanes: the context is not exact gfx942".into())
        })?;
        for chunk in lanes.chunks(QWEN4_G942_GDN_LANES_VERIFY_MAX) {
            hip(device.qwen4_gdn_lanes_verify(chunk, shape))?;
        }
        return Ok(Gfx942GdnLanes::Done);
    }
    before_launch(gpu)?;

    let value_heads = first.value_heads;
    let history_rows = first.conv_kernel.saturating_sub(1);
    let qkv = 2 * first.key_heads * first.key_dim + first.value_heads * first.value_dim;

    // The one-row AR lanes: convolution + gate parameters, ONE recurrence launch, gated norms.
    if !ar_lanes.is_empty() {
        for (op, v) in &ar_lanes {
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
        let lanes: Vec<Qwen4GdnLanesF32Lane<'_>> = ar_lanes
            .iter()
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
        for (op, v) in &ar_lanes {
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
    }
    Ok(Gfx942GdnLanes::Done)
}
