// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A3B decode norm fold (`HIPFIRE_A3B_NORM_FOLD`, default on).
//!
//! On exact gfx1201, a one-row decode of an MQ4G256V2 hidden-2048 model runs a
//! single-workgroup `fused_rmsnorm_mq_rotate` before the MQ4G256V2 projection
//! GEMV that consumes its output (the mixer's QKV / QKVZA window, and the MoE
//! block's router + shared-expert quartet). [`Gpu::fused_rmsnorm_proj4_mq4g256v2_fold`]
//! runs both in one launch whose every workgroup recomputes the norm with the
//! incumbent's exact code, so the rotated activation (still stored to `x_rot`
//! for later readers) and every projection row are byte-identical to the two
//! launches it replaces.

use crate::families::gemv::WeightRef;
use crate::types::{DispatchError, RotationPlan};
use rdna_compute::norm_fold::{norm_fold_grid, NormFoldArgs, NORM_FOLD_HIDDEN};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Output rows per row wave of the folded launch for a mixer projection
/// (QKV 9216 / QKVZA 12352 rows).
const MIXER_ROWS_PER_WAVE: usize = 12;
/// Output rows per row wave of the folded router + shared gate/up launch
/// (1281 rows).
pub(super) const MOE_GATE_ROWS_PER_WAVE: usize = 3;

/// `HIPFIRE_A3B_NORM_FOLD` kill switch, read once per process: unset or any
/// value but `0` keeps the fold on, `0` restores the separate norm launch.
pub(super) fn env_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_A3B_NORM_FOLD").as_deref() != Ok("0")
    });
    *ENABLED
}

/// Per-call runtime admission: exact gfx1201 and no hook that needs to observe
/// the norm output or the projection input between the two launches.
pub(super) fn launchable(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx1201()
        && gpu.hessian_capture.is_none()
        && gpu.active_capture.is_none()
        && !super::moe_dump::enabled()
        && hipfire_config::developer_var("HIPFIRE_DUMP_HIDDEN").is_err()
}

/// A MQ4G256V2 projection weight the fold can stream: hidden 2048, no AWQ.
pub(super) fn weight_admitted(w: &WeightRef<'_>) -> bool {
    w.dtype == DType::MQ4G256V2 && w.k == NORM_FOLD_HIDDEN && w.awq_scale.is_none()
}

/// Mixer QKV (3 projections) / QKVZA (4 projections) window: step 0 is the
/// plain norm into `out`, steps 1.. are the projections. Returns `true` when
/// the fold launched (the caller then launches neither the norm nor the fused
/// projection).
pub(super) fn launch_mixer_window(
    gpu: &mut Gpu,
    steps: &[super::steps::Step<'_>],
) -> Result<bool, DispatchError> {
    use super::steps::Step;
    if !env_enabled() || !launchable(gpu) || !(steps.len() == 4 || steps.len() == 5) {
        return Ok(false);
    }
    let Step::RmsnormAutomatic {
        x,
        norm_weight,
        out,
        awq_scale: None,
        k,
        eps,
        rotation: RotationPlan::FwhtG256,
        rows: 1,
        ..
    } = &steps[0]
    else {
        return Ok(false);
    };
    if *k != NORM_FOLD_HIDDEN {
        return Ok(false);
    }
    let mut weights: Vec<&GpuTensor> = Vec::with_capacity(4);
    let mut outs: Vec<&GpuTensor> = Vec::with_capacity(4);
    let mut m: Vec<usize> = Vec::with_capacity(4);
    for step in &steps[1..] {
        match step {
            Step::Gemv { w, out: y, .. } if weight_admitted(w) => {
                weights.push(w.buf);
                outs.push(*y);
                m.push(w.m);
            }
            _ => return Ok(false),
        }
    }
    // A QKV window repeats its last source in slot 3 with no rows.
    if weights.len() == 3 {
        weights.push(weights[2]);
        outs.push(outs[2]);
        m.push(0);
    }
    let args = NormFoldArgs {
        x,
        norm_weight,
        x_rot: out,
        weights: [weights[0], weights[1], weights[2], weights[3]],
        outs: [outs[0], outs[1], outs[2], outs[3]],
        m: [m[0], m[1], m[2], m[3]],
        eps: *eps,
    };
    let grid = norm_fold_grid(args.total_rows(), MIXER_ROWS_PER_WAVE);
    gpu.fused_rmsnorm_proj4_mq4g256v2_fold(&args, grid)
        .map_err(|e| DispatchError::Hip(e.to_string()))?;
    Ok(true)
}
