// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! One-row projection with an architecture-prepared activation pair.

use std::sync::LazyLock;

use rdna_compute::{DType, Gpu, GpuTensor};

use crate::context::DispatchCtx;
use crate::families::gemv::{GemvFamily, WeightRef};
use crate::types::{dtype_needs_rotation, DispatchError};

/// Which kernel the E8 SoA (MFP4G32E8SOA) projection uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum E8SoaKernel {
    /// The prerotated SoA kernel.
    #[default]
    Prerotated,
    /// The unroll-4 SoA kernel.
    U4,
    /// The gfx1151 unroll-4 kernel with buffer cache-policy loads.
    U4Gfx1151,
}

/// `y = W · x` for one row. The caller prepares the input twice: FWHT-rotated
/// (`x_rotated`) for the rotated formats and plain (`x_plain`) for the rest;
/// the weight dtype picks which one is read. Neither input is rotated here,
/// so a prerotated activation is never rotated twice.
#[allow(clippy::too_many_arguments)]
pub fn project_prepared(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_rotated: &GpuTensor,
    x_plain: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    e8_soa: E8SoaKernel,
) -> Result<(), DispatchError> {
    static GEMV: LazyLock<GemvFamily> = LazyLock::new(GemvFamily::new);
    let hip = |r: Result<(), hip_bridge::HipError>, what: &str| {
        r.map_err(|e| DispatchError::Hip(format!("{what}: {e:?}")))
    };
    match weight.dtype {
        DType::MQ4G256 => hip(
            gpu.gemv_mq4g256_prerotated(weight, x_rotated, y, m, k),
            "gemv MQ4 prerotated",
        ),
        DType::MFP4G32E8 => hip(
            gpu.gemv_mfp4g32_e8_prerotated(weight, x_rotated, y, m, k),
            "gemv MFP4-E8",
        ),
        DType::MFP4G32E8SOA => match e8_soa {
            E8SoaKernel::U4Gfx1151 => hip(
                gpu.gemv_mfp4g32_e8_soa_u4_buffer_cpol_gfx1151(0, weight, x_rotated, y, m, k),
                "gemv MFP4-E8-SoA-U4",
            ),
            E8SoaKernel::U4 => hip(
                gpu.gemv_mfp4g32_e8_soa_u4(weight, x_rotated, y, m, k),
                "gemv MFP4-E8-SoA-U4",
            ),
            E8SoaKernel::Prerotated => hip(
                gpu.gemv_mfp4g32_e8_soa_prerotated(weight, x_rotated, y, m, k),
                "gemv MFP4-E8-SoA",
            ),
        },
        DType::MFP3G32E8 => hip(
            gpu.gemv_mfp3g32_e8_prerotated(weight, x_rotated, y, m, k),
            "gemv MFP3-E8",
        ),
        _ => {
            let ctx = DispatchCtx::new(gpu);
            let x = if dtype_needs_rotation(weight.dtype) {
                x_rotated
            } else {
                x_plain
            };
            let wr = WeightRef {
                buf: weight,
                dtype: weight.dtype,
                m,
                k,
                row_stride: 0,
                rotation: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            };
            GEMV.run_auto(&ctx, gpu, &wr, x, y)
                .map_err(|e| DispatchError::Hip(format!("gemv dispatch: {e}")))
        }
    }
}

/// [`project_prepared`] as a step operand.
pub struct PreparedProjectionOp<'a> {
    pub weight: &'a GpuTensor,
    pub x_rotated: &'a GpuTensor,
    pub x_plain: &'a GpuTensor,
    pub y: &'a GpuTensor,
    pub m: usize,
    pub k: usize,
    pub e8_soa: E8SoaKernel,
}

impl PreparedProjectionOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.m == 0 || self.k == 0 {
            return Err(DispatchError::Hip(
                "prepared projection has empty geometry".into(),
            ));
        }
        Ok(())
    }
}

pub fn execute_prepared_projection(
    gpu: &mut Gpu,
    op: &PreparedProjectionOp<'_>,
) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    project_prepared(
        gpu,
        op.weight,
        op.x_rotated,
        op.x_plain,
        op.y,
        op.m,
        op.k,
        op.e8_soa,
    )
}
