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

/// [`project_prepared`] as a step operand: one row (`Decode`), or a `rows`-row
/// window through [`project_prepared_rows`] (`Batched`).
pub struct PreparedProjectionOp<'a> {
    pub rows: usize,
    pub weight: &'a GpuTensor,
    pub x_rotated: &'a GpuTensor,
    pub x_plain: &'a GpuTensor,
    pub y: &'a GpuTensor,
    pub m: usize,
    pub k: usize,
    pub route: PreparedProjectionRoute<'a>,
}

/// Only the decode kernel and the batched projection knobs differ.
pub enum PreparedProjectionRoute<'a> {
    Decode {
        e8_soa: E8SoaKernel,
    },
    Batched {
        x_f16: Option<&'a GpuTensor>,
        projection: RowsProjection,
    },
}

impl PreparedProjectionOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, PreparedProjectionRoute::Decode { .. });
        super::attention::check_rows(self.rows, decode, || "prepared projection".to_string())?;
        if decode && (self.m == 0 || self.k == 0) {
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
    match &op.route {
        PreparedProjectionRoute::Decode { e8_soa } => project_prepared(
            gpu,
            op.weight,
            op.x_rotated,
            op.x_plain,
            op.y,
            op.m,
            op.k,
            *e8_soa,
        ),
        PreparedProjectionRoute::Batched { x_f16, projection } => project_prepared_rows(
            gpu,
            op.weight,
            op.x_rotated,
            op.x_plain,
            op.y,
            op.m,
            op.k,
            op.rows,
            *x_f16,
            projection,
        ),
    }
}

/// Route choices of [`project_prepared_rows`] that the model resolves from its
/// MQ2R backend and knobs once per program.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowsProjection {
    /// One-row E8 SoA kernel of the per-row fallback.
    pub e8_soa: E8SoaKernel,
    /// B2 / B4 E8 SoA WMMA token tiles admitted.
    pub e8_b2: bool,
    pub e8_b4: bool,
    /// Largest batch that takes the gfx1151 batched E8 decode GEMV instead of
    /// the WMMA token tile.
    pub e8_batched_gemv_max: usize,
    /// The weights are the exact gfx1201 MQ2R build (F16-activation E8 WMMA).
    pub gfx1201_e8: bool,
    /// gfx1201 wide token tiles at the 1,024-row chunk.
    pub gfx1201_wide: bool,
    /// gfx1151 paired B3 E8 GEMV for two projections of one input.
    pub e8_pair_b3: bool,
    /// HFQ4 F16 WMMA GEMM admitted (`gemm_hfq4g256_wmma`).
    pub hfq4_wmma: bool,
    /// Trace the first eight F32 GEMMs (developer).
    pub f32_trace: bool,
}

impl RowsProjection {
    fn e8_tiles(&self, rows: usize) -> usize {
        if rows > 32 && self.e8_b4 {
            4
        } else if rows > 16 && self.e8_b2 {
            2
        } else {
            1
        }
    }

    /// The gfx1151 batched E8 decode GEMV instead of the WMMA token tile:
    /// measured at M=K=4096 it runs 3.72x the tile at B=1 and 1.40x at B=6,
    /// but 0.60x at B=16, where the tile is full.
    pub(crate) fn e8_batched_gemv(&self, arch: &str, rows: usize, k: usize) -> bool {
        arch == "gfx1151"
            && rows <= self.e8_batched_gemv_max
            && k.is_multiple_of(256)
            && Gpu::E8_BATCHED_GEMV_BATCHES.contains(&rows)
    }
}

/// The cooperative E8 producer/consumer kernel only for the exact gfx1151
/// 1,024-row prefill shapes that cleared the bit-exact micro gate.
fn e8_coop4(arch: &str, m: usize, k: usize, rows: usize) -> bool {
    arch == "gfx1151"
        && rows == 1024
        && matches!(
            (m, k),
            (32768, 1024) | (4096, 8192) | (2048, 4096) | (4096, 2048)
        )
}

/// Measured gfx1201 MQ2R token-tile width: at the 1,024-row chunk more query
/// tiles per wave amortize the decoded E8 weight fragment (B8 for TP3 wq_b /
/// wo_b / shared down, B4 for wq_a and the 768-row shared-up shard, B2 for the
/// 512-row shard, B16 for full-width wq_b); tails keep the older selector.
fn gfx1201_e8_rows(m: usize, rows: usize, wide: bool) -> usize {
    if wide && rows == 1024 && m >= 16384 {
        16
    } else if wide && rows == 1024 && m >= 2048 {
        8
    } else if wide && rows == 1024 && m >= 768 {
        4
    } else if wide && rows == 1024 && m >= 512 {
        2
    } else if m >= 8192 {
        4
    } else if m == 4096 {
        2
    } else {
        1
    }
}

/// F16×F16→F32 GEMM: RDNA WMMA, or the MFMA port on gfx942 (the wave32 WMMA
/// builtin does not compile for CDNA3).
pub(crate) fn gemm_f16_x_f16(
    gpu: &mut Gpu,
    weight_f16: &GpuTensor,
    x_f16: &GpuTensor,
    y_f32: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
) -> hip_bridge::HipResult<()> {
    if gpu.arch_caps.is_gfx942() {
        gpu.gemm_f16_x_f16_mfma_gfx942(weight_f16, x_f16, y_f32, m, k, rows)
    } else {
        gpu.gemm_f16_x_f16_wmma(weight_f16, x_f16, y_f32, m, k, rows)
    }
}

/// `Y[r] = W · X[r]` for `rows` prepared rows (`[rows, k]` rotated and plain
/// inputs, `[rows, m]` output); the weight dtype, the row count and `route`
/// pick the GEMM. `x_f16` is the F16 activation scratch of the WMMA routes.
#[allow(clippy::too_many_arguments)]
pub fn project_prepared_rows(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_rotated: &GpuTensor,
    x_plain: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
    x_f16: Option<&GpuTensor>,
    route: &RowsProjection,
) -> Result<(), DispatchError> {
    let hip = |what: &'static str| {
        move |e: hip_bridge::HipError| DispatchError::Hip(format!("{what}: {e:?}"))
    };
    let e8_tiles = route.e8_tiles(rows);
    match weight.dtype {
        DType::MFP4G32E8SOA
            if route.gfx1201_e8
                && gpu.arch_caps.is_gfx1201()
                && k.is_multiple_of(256)
                && x_f16.is_some() =>
        {
            let scratch = x_f16.unwrap();
            gpu.deepseek4_convert_f32_to_f16(x_rotated, scratch, (rows * k) as i64)
                .map_err(hip("convert_f32_to_f16 (gfx1201 E8 WMMA)"))?;
            match gfx1201_e8_rows(m, rows, route.gfx1201_wide) {
                16 => gpu
                    .gemm_mfp4g32_e8_soa_wmma_b16_gfx1201_f16(weight, scratch, y, m, k, rows)
                    .map_err(hip("gemm gfx1201 MFP4-E8-SoA WMMA B16")),
                8 => gpu
                    .gemm_mfp4g32_e8_soa_wmma_b8_gfx1201_f16(weight, scratch, y, m, k, rows)
                    .map_err(hip("gemm gfx1201 MFP4-E8-SoA WMMA B8")),
                4 => gpu
                    .gemm_mfp4g32_e8_soa_wmma_b4_gfx1201_f16(weight, scratch, y, m, k, rows)
                    .map_err(hip("gemm gfx1201 MFP4-E8-SoA WMMA B4")),
                2 => gpu
                    .gemm_mfp4g32_e8_soa_wmma_b2_gfx1201_f16(weight, scratch, y, m, k, rows)
                    .map_err(hip("gemm gfx1201 MFP4-E8-SoA WMMA B2")),
                _ => gpu
                    .gemm_mfp4g32_e8_soa_wmma_gfx1201_f16(weight, scratch, y, m, k, rows)
                    .map_err(hip("gemm gfx1201 MFP4-E8-SoA WMMA B1")),
            }
        }
        DType::MFP4G32E8SOA if route.e8_batched_gemv(&gpu.arch, rows, k) => gpu
            .gemv_mfp4g32_e8_soa_batched_gfx1151(weight, x_rotated, y, rows, m, k)
            .map_err(hip("gemv MFP4-E8-SoA batched")),
        DType::MFP4G32E8SOA if e8_tiles == 4 && e8_coop4(&gpu.arch, m, k, rows) => gpu
            .gemm_mfp4g32_e8_soa_wmma_coop4(weight, x_rotated, y, m, k, rows)
            .map_err(hip("gemm MFP4-E8-SoA WMMA cooperative B4")),
        DType::MFP4G32E8SOA if e8_tiles == 4 => gpu
            .gemm_mfp4g32_e8_soa_wmma_b4(weight, x_rotated, y, m, k, rows)
            .map_err(hip("gemm MFP4-E8-SoA WMMA B4")),
        DType::MFP4G32E8SOA if e8_tiles == 2 => gpu
            .gemm_mfp4g32_e8_soa_wmma_b2(weight, x_rotated, y, m, k, rows)
            .map_err(hip("gemm MFP4-E8-SoA WMMA B2")),
        DType::MFP4G32E8SOA if gpu.arch == "gfx1151" => gpu
            .gemm_mfp4g32_e8_soa_wmma(weight, x_rotated, y, m, k, rows)
            .map_err(hip("gemm MFP4-E8-SoA WMMA B1")),
        DType::MFP4G32E8 | DType::MFP4G32E8SOA | DType::MFP3G32E8 => {
            if weight.dtype == DType::MFP4G32E8SOA
                && gpu
                    .rocblas_gemm_mfp4e8_soa_prefill_auto(weight, x_rotated, y, m, k, rows)
                    .map_err(hip("rocBLAS MFP4-E8-SoA prefill"))?
            {
                return Ok(());
            }
            // AoS and non-gfx1151 SoA keep the per-row correctness fallback
            // until an architecture-specific batched kernel is admitted.
            for r in 0..rows {
                let x_rot = x_rotated.sub_offset(r * k, k);
                let x_pl = x_plain.sub_offset(r * k, k);
                let y_row = y.sub_offset(r * m, m);
                project_prepared(gpu, weight, &x_rot, &x_pl, &y_row, m, k, route.e8_soa)?;
            }
            Ok(())
        }
        DType::F32 => {
            if route.f32_trace {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static N: AtomicUsize = AtomicUsize::new(0);
                let c = N.fetch_add(1, Ordering::Relaxed);
                if c < 8 {
                    eprintln!(
                        "[F32_TRACE #{c}] m={m} k={k} B={rows} weight.shape={:?}",
                        weight.shape
                    );
                }
            }
            gpu.gemm_f32_register_tiled(weight, x_plain, y, m, k, rows)
                .map_err(hip("gemm_f32_register_tiled"))
        }
        // Kernel selection (i8-MMQ / f16-WMMA / scalar) is rdna-compute's.
        DType::Q8_0 => gpu
            .gemm_q8_0_wmma_prefill_auto(weight, x_plain, x_f16, y, m, k, rows)
            .map_err(hip("gemm_q8_0_wmma_prefill_auto")),
        DType::F16 => {
            // gfx12 takes the validated F32-input f16 WMMA kernel; gfx11 /
            // gfx942 the f16×f16 route on a converted X.
            if gpu.arch_caps.has_wmma_w32_gfx12() {
                return gpu
                    .gemm_f16_wmma_mb8(weight, x_plain, y, m, k, rows)
                    .map_err(hip("gemm_f16_wmma_mb8 (gfx12 f16)"));
            }
            let Some(scratch) = x_f16 else {
                return Err(DispatchError::Hip(
                    "F16 weight requires an F16 GEMM path with x_f16_scratch".into(),
                ));
            };
            gpu.deepseek4_convert_f32_to_f16(x_plain, scratch, (rows * k) as i64)
                .map_err(hip("convert_f32_to_f16 (F16 weight)"))?;
            gemm_f16_x_f16(gpu, weight, scratch, y, m, k, rows)
                .map_err(hip("gemm_f16_x_f16 (F16 weight)"))
        }
        DType::MQ4G256 if gpu.arch == "gfx1151" && rows <= 8 => gpu
            .gemm_hfq4g256(weight, x_rotated, y, m, k, rows)
            .map_err(hip("gemm_hfq4g256 small-B")),
        _ => {
            // `gemm_hfq4g256_wmma` is wave32 WMMA: `has_wmma()` (false on
            // gfx942) gates it so CDNA3 takes `gemm_hfq4g256`.
            if gpu.arch_caps.has_wmma() && route.hfq4_wmma {
                if let Some(scratch) = x_f16 {
                    gpu.deepseek4_convert_f32_to_f16(x_rotated, scratch, (rows * k) as i64)
                        .map_err(hip("convert_f32_to_f16 (HFQ4 WMMA)"))?;
                    return gpu
                        .gemm_hfq4g256_wmma(weight, scratch, y, m, k, rows)
                        .map_err(hip("gemm_hfq4g256_wmma"));
                }
            }
            gpu.gemm_hfq4g256(weight, x_rotated, y, m, k, rows)
                .map_err(hip("gemm_hfq4g256"))
        }
    }
}

/// Two projections of one rotated input as the gfx1151 paired B3 E8 GEMV,
/// when admitted (three rows, both E8 SoA); `false` leaves both to the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_pair_b3(
    gpu: &mut Gpu,
    w0: &GpuTensor,
    w1: &GpuTensor,
    x_rotated: &GpuTensor,
    y0: &GpuTensor,
    y1: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
    route: &RowsProjection,
) -> Result<bool, DispatchError> {
    if !route.e8_pair_b3
        || gpu.arch != "gfx1151"
        || rows != 3
        || !k.is_multiple_of(256)
        || w0.dtype != DType::MFP4G32E8SOA
        || w1.dtype != DType::MFP4G32E8SOA
    {
        return Ok(false);
    }
    gpu.gemv_mfp4g32_e8_soa_batched_pair_b3_gfx1151(w0, w1, x_rotated, y0, y1, m, k)
        .map_err(|e| DispatchError::Hip(format!("paired MFP4-E8-SoA batched B3: {e:?}")))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx1201_e8_prefill_tiles_follow_measured_tp_shapes() {
        // TP3 wq_b: 24/24/16 local heads.
        assert_eq!(gfx1201_e8_rows(12288, 1024, true), 8);
        assert_eq!(gfx1201_e8_rows(8192, 1024, true), 8);
        // TP3 wo_b and shared-down retain 4,096 output rows.
        assert_eq!(gfx1201_e8_rows(4096, 1024, true), 8);
        // TP3 shared-up is 768/768/512 rows; replicated wq_a is 1,024.
        assert_eq!(gfx1201_e8_rows(1024, 1024, true), 4);
        assert_eq!(gfx1201_e8_rows(768, 1024, true), 4);
        assert_eq!(gfx1201_e8_rows(512, 1024, true), 2);
        // Unmeasured tails and the rollback path retain the promoted schedule.
        assert_eq!(gfx1201_e8_rows(12288, 512, true), 4);
        assert_eq!(gfx1201_e8_rows(4096, 1024, false), 2);
        assert_eq!(gfx1201_e8_rows(768, 1024, false), 1);
    }

    #[test]
    fn e8_prefill_tiles_do_not_compute_absent_rows() {
        let tiles = |rows, e8_b2, e8_b4| {
            RowsProjection {
                e8_b2,
                e8_b4,
                ..RowsProjection::default()
            }
            .e8_tiles(rows)
        };
        assert_eq!(tiles(1, true, true), 1);
        assert_eq!(tiles(16, true, true), 1);
        assert_eq!(tiles(17, true, true), 2);
        assert_eq!(tiles(32, true, true), 2);
        assert_eq!(tiles(33, true, true), 4);
        assert_eq!(tiles(64, true, true), 4);
        assert_eq!(tiles(64, true, false), 2);
        assert_eq!(tiles(64, false, false), 1);
    }
}
