// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
// Segment twins for cross-request batched speculative verify.
//
// Dispatch-layer owners of the `rdna_compute::verify_twins` launches so arch
// crates do not reach `Gpu::*_segs` directly. Each forwards unchanged.

use hip_bridge::HipResult;
pub use rdna_compute::verify_twins::AttnFp8Seg;
use rdna_compute::{Gpu, GpuTensor};

/// Stage fp8 attention segments at table entry `at`
/// (`Gpu::stage_attention_fp8_e4m3_kv_batched_segs`).
pub fn stage_attention_fp8_segs(
    gpu: &Gpu,
    table: &GpuTensor,
    at: usize,
    segs: &[AttnFp8Seg],
    n_heads: usize,
    head_dim: usize,
) -> HipResult<()> {
    gpu.stage_attention_fp8_e4m3_kv_batched_segs(table, at, segs, n_heads, head_dim)
}

/// Launch staged fp8 attention segments
/// (`Gpu::attention_fp8_e4m3_kv_batched_segs`).
#[allow(clippy::too_many_arguments)]
pub fn attention_fp8_segs(
    gpu: &mut Gpu,
    table: &GpuTensor,
    at: usize,
    max_rows: usize,
    max_ctx_lens: &[usize],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
) -> HipResult<()> {
    gpu.attention_fp8_e4m3_kv_batched_segs(
        table,
        at,
        max_rows,
        max_ctx_lens,
        n_heads,
        n_kv_heads,
        head_dim,
        max_seq,
    )
}
