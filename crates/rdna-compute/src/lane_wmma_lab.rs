// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane WMMA lab (developer screens only): the planar-W8 Q8_0 repack and the W8 twins of the tuned
//! prefill HC kernels. The product decode launchers (`lane_wmma_gemm`, `lane_wmma_read_up`, the weight and
//! scratch types) live in [`crate::lane_wmma`]; no product dispatch reaches this module.

use crate::dispatch::ReplayLaunchBindings;
use crate::lane_wmma::{err, i32_of, need_bytes, need_f32, u32_of, LaneWmmaLaunch};
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipResult, KernargBlob};

/// Host repack of Q8_0 bytes (34-byte blocks `{f16 d, i8 q[32]}`, row-major `[m, k / 32]`) into
/// the planar W8 layout: `q` bytes `[m, k]` and `d` f16 bits `[m, k / 32]`. A pure byte shuffle.
pub fn w8_planar_from_q8_0(q8_0: &[u8], m: usize, k: usize) -> (Vec<u8>, Vec<u16>) {
    assert!(k % 32 == 0, "w8_planar_from_q8_0: k % 32 != 0");
    let blocks = m * (k / 32);
    assert!(
        q8_0.len() >= blocks * 34,
        "w8_planar_from_q8_0: {} bytes < {} blocks * 34",
        q8_0.len(),
        blocks
    );
    let mut q = Vec::with_capacity(blocks * 32);
    let mut d = Vec::with_capacity(blocks);
    for block in q8_0[..blocks * 34].chunks_exact(34) {
        d.push(u16::from_le_bytes([block[0], block[1]]));
        q.extend_from_slice(&block[2..34]);
    }
    (q, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planar_repack_is_a_byte_shuffle() {
        let (m, k) = (3usize, 64usize);
        let mut q8 = Vec::new();
        for b in 0..m * (k / 32) {
            q8.extend_from_slice(&((0x3c00 + b as u16).to_le_bytes()));
            for e in 0..32 {
                q8.push((b * 32 + e) as u8);
            }
        }
        let (q, d) = w8_planar_from_q8_0(&q8, m, k);
        assert_eq!(q.len(), m * k);
        assert_eq!(d.len(), m * (k / 32));
        for b in 0..m * (k / 32) {
            assert_eq!(d[b], 0x3c00 + b as u16);
            for e in 0..32 {
                assert_eq!(q[b * 32 + e], (b * 32 + e) as u8);
            }
        }
    }
}

// ---- prefill W8 twins ----
//
// Deliverable B: W8 twins of the tuned prefill kernels (rows > 64; the A1/A2 decode family in
// `crate::lane_wmma` also runs there). The weight is the planar W8 (`q` i8 `[m, k]`, `d` f16 `[m, k / 32]`); the
// per-32 scale is applied in the f32 accumulation, never in an operand.

/// `kernels/src/qwen4_hc_down_fused_w8.gfx1201.hip`: twin of `qwen4_hc_down_fused.hip`.
pub const LANE_WMMA_HC_DOWN_W8_SRC: &str =
    include_str!("../../../kernels/src/qwen4_hc_down_fused_w8.gfx1201.hip");
/// `kernels/src/hyper_read_up_wmma_w8.gfx1201.hip`: twin of the `_v2` gfx1201 HC up+mix read.
pub const LANE_WMMA_HC_UP_W8_SRC: &str =
    include_str!("../../../kernels/src/hyper_read_up_wmma_w8.gfx1201.hip");

const LANE_WMMA_HC_DOWN_W8_MODULE: &str = "lane_wmma_hc_down_fused_w8";
const LANE_WMMA_HC_DOWN_W8_ENTRY: &str = "qwen4_hc_down_fused_w8_b64_k32";
const LANE_WMMA_HC_UP_W8_MODULE: &str = "lane_wmma_hc_up_w8";
const LANE_WMMA_HC_UP_W8_ENTRY: &str = "hyper_read_up_wmma_w8_gfx1201_v2";
/// Dynamic LDS of the W8 `_v2` read entry (`HRW_LDS_BYTES`): the 64 x (320 + 8) BF16 weight tile,
/// the 2 x 2048-entry gate table and the 10 x 64 F16 scale table.
const LANE_WMMA_HC_UP_W8_LDS: u32 = 64 * (320 + 8) * 2 + 16384 + 10 * 64 * 2;

fn need_dtype(t: &GpuTensor, want: DType, what: &str) -> HipResult<()> {
    if t.dtype != want {
        return Err(err(format!("{what}: dtype {:?} is not {want:?}", t.dtype)));
    }
    Ok(())
}

fn need_aligned(ptr: usize, align: usize, what: &str) -> HipResult<()> {
    if ptr % align != 0 {
        return Err(err(format!("{what}: pointer {ptr:#x} not {align}-byte aligned")));
    }
    Ok(())
}

/// HC down + `hc_act` on the planar W8 weight, gfx1201 prefill: `out_bf16[b, m] =
/// hc_act(sum_k W[m, k] x16[b, k], scale)` as packed BF16 bits, the operand of the HC up read.
/// `q` i8 `[320, 10240]`, `d` f16 `[320, 320]`, `x16` F16 `[rows, 10240]` (`hyper_norm_f16`),
/// `out_bf16` BF16 `[rows, 320]`. Only `m == 320`, `k == 10240` (the kernel's constants).
/// Rows above 64 are what it is tuned for, but any `rows >= 1` runs.
#[allow(clippy::too_many_arguments)]
pub fn lane_wmma_hc_down_fused_w8(
    gpu: &mut Gpu,
    q: &GpuTensor,
    d: &GpuTensor,
    x16: &GpuTensor,
    out_bf16: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
    scale: f32,
) -> HipResult<LaneWmmaLaunch> {
    if gpu.arch != "gfx1201" {
        return Err(err(format!(
            "lane_wmma_hc_down_fused_w8: gfx1201 only (arch {})",
            gpu.arch
        )));
    }
    if m != 320 || k != 10240 {
        return Err(err(format!(
            "lane_wmma_hc_down_fused_w8: m 320, k 10240 only (m {m}, k {k})"
        )));
    }
    if rows == 0 {
        return Err(err("lane_wmma_hc_down_fused_w8: rows must be non-zero"));
    }
    if rows * k * 2 >= (1usize << 32) {
        return Err(err(format!(
            "lane_wmma_hc_down_fused_w8: rows {rows} x k {k} x 2 bytes exceeds 2^32"
        )));
    }
    need_bytes(q, m * k, "hc_down w8 q")?;
    need_bytes(d, m * (k / 32) * 2, "hc_down w8 d")?;
    need_dtype(x16, DType::F16, "hc_down x16")?;
    need_bytes(x16, rows * k * 2, "hc_down x16")?;
    need_dtype(out_bf16, DType::BF16, "hc_down out_bf16")?;
    need_bytes(out_bf16, rows * m * 2, "hc_down out_bf16")?;
    let (qp, dp, xp, op) = (
        q.buf.as_ptr(),
        d.buf.as_ptr(),
        x16.buf.as_ptr(),
        out_bf16.buf.as_ptr(),
    );
    for (p, what) in [(qp, "q"), (dp, "d"), (xp, "x16"), (op, "out_bf16")] {
        need_aligned(p as usize, 16, what)?;
    }
    let grid = [u32_of(rows.div_ceil(64), "grid x")?, 1, 1];
    let block = [256u32, 1, 1];
    gpu.ensure_kernel_public(
        LANE_WMMA_HC_DOWN_W8_MODULE,
        LANE_WMMA_HC_DOWN_W8_SRC,
        LANE_WMMA_HC_DOWN_W8_ENTRY,
    )?;
    // Argument order of the kernel: Q, D, X, out, K, B, scale.
    let mut args = KernargBlob::new();
    args.push_ptr(qp);
    args.push_ptr(dp);
    args.push_ptr(xp);
    args.push_ptr(op);
    args.push_i32(i32_of(k, "k")?);
    args.push_i32(i32_of(rows, "rows")?);
    args.push_f32(scale);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        LANE_WMMA_HC_DOWN_W8_ENTRY,
        grid,
        block,
        0,
        args.as_mut_slice(),
        ReplayLaunchBindings::NONE,
    )?;
    Ok(LaneWmmaLaunch {
        kernel: LANE_WMMA_HC_DOWN_W8_ENTRY.to_string(),
        grid,
        block,
    })
}

/// HC up + branch mix on the planar W8 weight, gfx1201 prefill (the persistent W-stationary
/// `_v2` structure): `mixed [rows, hidden]` from `p.low` (packed BF16 `[rows, 320]`) and
/// `p.normalized` (F16 `[rows, 4 * hidden]`, `normalized_bf16` false), exactly what the BF16
/// route passes; `p.up_weight` is ignored (`q` i8 `[4 * hidden, 320]`, `d` f16
/// `[4 * hidden, 10]` are the weight). Only `low_rank == 320`, `hidden % 16 == 0`; any
/// `rows >= 1` runs (the BF16 route only takes `_v2` from 128 rows).
pub fn lane_wmma_hc_up_w8(
    gpu: &mut Gpu,
    q: &GpuTensor,
    d: &GpuTensor,
    p: &crate::tensor_ops::HyperReadUpFused<'_>,
) -> HipResult<LaneWmmaLaunch> {
    if gpu.arch != "gfx1201" {
        return Err(err(format!(
            "lane_wmma_hc_up_w8: gfx1201 only (arch {})",
            gpu.arch
        )));
    }
    if p.rows == 0 || p.hidden == 0 || p.hidden % 16 != 0 || p.low_rank != 320 {
        return Err(err(format!(
            "lane_wmma_hc_up_w8: rows {} non-zero, hidden {} a multiple of 16, low_rank {} == 320",
            p.rows, p.hidden, p.low_rank
        )));
    }
    if p.normalized_bf16 {
        return Err(err("lane_wmma_hc_up_w8: normalized must be the F16 copy"));
    }
    let wide = 4 * p.hidden;
    need_bytes(q, wide * p.low_rank, "read_up w8 q")?;
    need_bytes(d, wide * (p.low_rank / 32) * 2, "read_up w8 d")?;
    need_dtype(p.low, DType::BF16, "read_up low")?;
    need_bytes(p.low, p.rows * p.low_rank * 2, "read_up low")?;
    need_dtype(p.normalized, DType::F16, "read_up normalized")?;
    need_bytes(p.normalized, p.rows * wide * 2, "read_up normalized")?;
    need_f32(p.mixed, p.rows * p.hidden, "read_up mixed")?;
    let (qp, dp, lp, np, mp) = (
        q.buf.as_ptr(),
        d.buf.as_ptr(),
        p.low.buf.as_ptr(),
        p.normalized.buf.as_ptr(),
        p.mixed.buf.as_ptr(),
    );
    need_aligned(qp as usize, 16, "q")?;
    need_aligned(dp as usize, 4, "d")?;
    need_aligned(lp as usize, 16, "low")?;
    need_aligned(np as usize, 2, "normalized")?;
    need_aligned(mp as usize, 4, "mixed")?;
    let groups = crate::tensor_ops::hc_up_v2_groups(p.rows);
    let grid = [u32_of(p.hidden / 16, "grid x")?, u32_of(groups, "grid y")?, 1];
    let block = [512u32, 1, 1];
    gpu.ensure_kernel_public(
        LANE_WMMA_HC_UP_W8_MODULE,
        LANE_WMMA_HC_UP_W8_SRC,
        LANE_WMMA_HC_UP_W8_ENTRY,
    )?;
    // Argument order of the kernel: Wq, Wd, X, normalized, mixed, hidden, K, N.
    let mut args = KernargBlob::new();
    args.push_ptr(qp);
    args.push_ptr(dp);
    args.push_ptr(lp);
    args.push_ptr(np);
    args.push_ptr(mp);
    args.push_i32(i32_of(p.hidden, "hidden")?);
    args.push_i32(i32_of(p.low_rank, "low_rank")?);
    args.push_i32(i32_of(p.rows, "rows")?);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        LANE_WMMA_HC_UP_W8_ENTRY,
        grid,
        block,
        LANE_WMMA_HC_UP_W8_LDS,
        args.as_mut_slice(),
        ReplayLaunchBindings::NONE,
    )?;
    Ok(LaneWmmaLaunch {
        kernel: LANE_WMMA_HC_UP_W8_ENTRY.to_string(),
        grid,
        block,
    })
}
