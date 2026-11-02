// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
// PM multi-column ("xbatch") MQ4-G256 V2 GEMV entry points.
//
// The VMM exact batched route runs each projection over every decode row in
// one launch, each column byte-identical to the singleton kernel. These are
// the dispatch-layer owners of those launches so arch crates do not reach
// `Gpu::gemv_*` directly. Each forwards unchanged; `batch` is at most
// `rdna_compute::pm_xbatch::PM_XBATCH_MAX`.

use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

/// `y[b] = W · x[b]` for `batch` columns (`Gpu::gemv_mq4g256v2_xbatch_pm`).
pub fn plain(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch: usize,
) -> HipResult<()> {
    gpu.gemv_mq4g256v2_xbatch_pm(w, x, y, m, k, batch)
}

/// `y[b] += W · x[b]` for `batch` columns (`Gpu::gemv_mq4g256v2_residual_xbatch`).
pub fn residual(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch: usize,
) -> HipResult<()> {
    gpu.gemv_mq4g256v2_residual_xbatch(w, x, y, m, k, batch)
}

/// Multirow-R2 twin for the lm_head (`Gpu::gemv_mq4g256v2_multirow_r2_xbatch_pm`).
pub fn multirow_r2(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch: usize,
) -> HipResult<()> {
    gpu.gemv_mq4g256v2_multirow_r2_xbatch_pm(w, x, y, m, k, batch)
}
