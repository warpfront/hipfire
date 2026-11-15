// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx942 batched QSA decode Core for lane steps.
//!
//! On gfx942 a lane's QSA Core is five launches per layer (decode prologue, incremental
//! pool, row-8 scores, select-from-scores with the persistent mirror, per-head attention;
//! [`rdna_compute::tensor_ops::qsa_core_lanes_gfx942`] lists the exact kernels). For a group of
//! consecutive Core steps of one layer this module hands the lanes' per-lane launch arguments,
//! built by [`super::layer_ops::execute_indexed_attention_core_lanes`] exactly as the per-lane
//! interpreter builds them, to the admission, then runs each of the five launches ONCE over
//! every lane (`Gfx942Device::qwen4_qsa_lanes_core`, the production device templates with the
//! lane on `blockIdx.z`). The lanes share no written byte (checked by the admission), so each
//! lane's state and outputs equal its per-lane chain byte for byte; a group the route does
//! not take (flag off, a shape or format it does not cover, a lane the per-lane path would
//! send elsewhere) is declined untouched and runs per lane.

use super::layer_ops::hip;
use crate::types::DispatchError;
use rdna_compute::cdna::gfx942::QWEN4_QSA_LANES_MAX;
use rdna_compute::tensor_ops::{qsa_core_lanes_gfx942, QsaCoreLane, QSA_CORE_LANES_MAX};
use rdna_compute::Gpu;

/// Most consecutive QSA Core steps the lane-step grouper collects into one group: the lane
/// count of the batched gfx942 launches there, else the 8 of the gfx1201 multi-lane kernels.
pub(crate) fn core_lanes_group_max(gpu: &Gpu) -> usize {
    if gpu.arch_caps.is_gfx942() {
        QWEN4_QSA_LANES_MAX
    } else {
        QSA_CORE_LANES_MAX
    }
}

/// The QSA Core phase of a group of lanes as five launches over all of them (see the module
/// docs). `lanes` are the lanes' per-lane launch arguments, two or more, at most
/// [`QWEN4_QSA_LANES_MAX`]. `Ok(false)`: the set does not take the route, nothing launched
/// and `before_launch` was not called; run the lanes one by one. `before_launch` runs exactly
/// once, after every refusal check and before the first launch (the grouped steps are
/// announced there); its error is returned as is and nothing launches. Host bookkeeping
/// (positions, commits) is the caller's, as on the per-lane path.
pub(crate) fn execute_core_lanes_gfx942(
    gpu: &mut Gpu,
    lanes: &[QsaCoreLane<'_>],
    before_launch: impl FnOnce(&mut Gpu) -> Result<(), DispatchError>,
) -> Result<bool, DispatchError> {
    let Some(plan) = hip(qsa_core_lanes_gfx942(gpu, lanes))? else {
        return Ok(false);
    };
    before_launch(gpu)?;
    let mut device = gpu.try_gfx942().ok_or_else(|| {
        DispatchError::Hip("gfx942 QSA lanes: the context is not exact gfx942".into())
    })?;
    hip(device.qwen4_qsa_lanes_core(&plan))?;
    // The lanes' outputs are rewritten: no earlier rotation memo describes them.
    gpu.scratch.prerotated = None;
    Ok(true)
}
