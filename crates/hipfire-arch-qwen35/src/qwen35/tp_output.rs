// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Dense-TP2 output projection on measured exact gfx1100 pairs. Vocabulary rows
//! are independent: each rank reads a non-owning row view of its replicated
//! head, then rank zero gathers the second half. No new GPU allocation.

use super::{Qwen35Config, Qwen35Scratch, Qwen35Weights};
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::{
    context::DispatchCtx,
    pipeline::{execute_steps, GemvInput, Step},
};
use hipfire_runtime::{llama::WeightTensor, multi_gpu::Gpus};
use rdna_compute::{DType, Gpu, GpuTensor};

fn row_bytes(w: &WeightTensor) -> Option<usize> {
    if w.m == 0 || w.k == 0 || w.paro.is_some() || w.row_stride != 0 || w.k % 256 != 0 {
        return None;
    }
    let bytes = match w.gpu_dtype {
        DType::MQ4G256 | DType::MQ4G256V2 => (w.k / 256).checked_mul(136)?,
        DType::Q8_0 => (w.k / 32).checked_mul(34)?,
        _ => return None,
    };
    (w.m.checked_mul(bytes)? == w.buf.buf.size()).then_some(bytes)
}

fn supported_topology<'a>(mut arches: impl Iterator<Item = &'a str>) -> bool {
    arches.next() == Some("gfx1100") && arches.next() == Some("gfx1100") && arches.next().is_none()
}

fn measured_default(dtype: DType, rows: usize, cols: usize) -> bool {
    // The full TP2 end-to-end receipt covers the 27B MQ4V2 head. A row-local
    // oracle for another shape/codec does not establish an end-to-end win:
    // peer handoff and dispatch have a fixed cost even for tiny projections.
    dtype == DType::MQ4G256V2 && rows == 248_320 && cols == 5_120
}

/// Default-on only for the measured TP2 topology and head profile. `=0`
/// restores the root-only head; `=1` opts into other supported row layouts.
/// This does not admit another architecture or a new weight layout.
/// Whole 128-row tiles preserve all existing multirow kernel tails.
pub fn enabled(gpus: &Gpus, weights: &[Qwen35Weights], configs: &[Qwen35Config]) -> bool {
    supported_topology(gpus.devices.iter().map(|gpu| gpu.arch.as_str()))
        && weights.len() == 2
        && configs.len() == 2
        && configs[0].vocab_size > 0
        && configs[0].vocab_size % 256 == 0
        && weights.iter().zip(configs).all(|(w, cfg)| {
            cfg.dim == configs[0].dim
                && cfg.vocab_size == configs[0].vocab_size
                && w.output.m == cfg.vocab_size
                && w.output.k == cfg.dim
                && row_bytes(&w.output).is_some()
        })
        && hipfire_config::developer_bool(
            "HIPFIRE_TP_SHARD_LM_HEAD",
            weights
                .iter()
                .all(|w| measured_default(w.output.gpu_dtype, w.output.m, w.output.k)),
        )
}

/// Project an exact set of vocabulary rows without decoding/requantizing the
/// weights. Views borrow the original allocations for their entire GPU use;
/// graph kernargs retain their device addresses, not these stack descriptors.
pub fn project_rows(
    gpu: &mut Gpu,
    weight: &WeightTensor,
    input: &GpuTensor,
    logits: &GpuTensor,
    start: usize,
    rows: usize,
) -> HipResult<()> {
    let bytes =
        row_bytes(weight).ok_or_else(|| HipError::new(0, "unsupported TP head row layout"))?;
    if rows == 0 || start.checked_add(rows).is_none_or(|end| end > weight.m) {
        return Err(HipError::new(0, "TP head row range out of bounds"));
    }
    let view = GpuTensor {
        buf: weight.buf.buf.byte_view(start * bytes, rows * bytes),
        shape: vec![rows, weight.k],
        dtype: weight.buf.dtype,
    };
    let mut wr = weight.dispatch_ref();
    wr.buf = &view;
    wr.m = rows;
    let ctx = DispatchCtx::new(gpu);
    execute_steps(
        gpu,
        &ctx,
        &[Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(input),
            out: logits,
        }],
    )
    .map_err(|error| HipError::new(0, &error.to_string()))
}

pub(super) fn local(
    gpus: &mut Gpus,
    weights: &[Qwen35Weights],
    configs: &[Qwen35Config],
    scratches: &[Qwen35Scratch],
    inputs: &[GpuTensor],
) -> HipResult<()> {
    let rows = configs[0].vocab_size / 2;
    for rank in 0..2 {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread()?;
        gpu.rmsnorm_f32(
            &inputs[rank],
            &weights[rank].output_norm,
            &scratches[rank].tmp,
            configs[rank].norm_eps,
        )?;
        let out = scratches[rank].logits.sub_offset(rank * rows, rows);
        project_rows(
            gpu,
            &weights[rank].output,
            &scratches[rank].tmp,
            &out,
            rank * rows,
            rows,
        )?;
    }
    Ok(())
}

pub(super) fn gather(gpus: &mut Gpus, scratches: &[Qwen35Scratch], vocab: usize) -> HipResult<()> {
    if gpus.devices.iter().any(|gpu| gpu.graphs.capture_mode) {
        return Err(HipError::new(
            0,
            "TP head peer gather must remain outside graph capture",
        ));
    }
    // A persistent, system-release event handoff OUTSIDE graph capture. Only
    // the rank-local head compute may be captured by the segmented route.
    gpus.barrier_rank_streams_reuse()?;
    let rows = vocab / 2;
    let dst = scratches[0].logits.sub_offset(rows, rows);
    let src = scratches[1].logits.sub_offset(rows, rows);
    // A kernel reading peer VRAM can retain stale cache lines on gfx1100
    // whole-token replay even after the signal acquire (the residual route
    // documents the same trap). Use the runtime's peer transfer rather than
    // a plain load/store kernel; preserve stream ordering and allocate nothing.
    let root = &gpus.devices[0];
    root.bind_thread()?;
    let stream = root
        .active_stream
        .as_ref()
        .ok_or_else(|| HipError::new(0, "TP head gather requires a root stream"))?;
    root.hip.memcpy_peer_async(
        &dst.buf,
        root.device_id,
        &src.buf,
        gpus.devices[1].device_id,
        rows * 4,
        stream,
    )
    // Before rank 1 can write this storage on the next token it must pass
    // that token's first residual barrier, whose root producer is downstream
    // of this gather on the root FIFO. Thus no reverse-copy-complete event
    // or second signal generation is needed on the steady decode path.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_is_exactly_two_gfx1100_devices() {
        assert!(supported_topology(["gfx1100", "gfx1100"].into_iter()));
        for arches in [
            vec![],
            vec!["gfx1100"],
            vec!["gfx1100", "gfx1100", "gfx1100"],
            vec!["gfx1100", "gfx1201"],
            vec!["gfx1201", "gfx1100"],
            vec!["gfx1151", "gfx1151"],
            vec!["gfx1101", "gfx1101"],
        ] {
            assert!(!supported_topology(arches.into_iter()));
        }
    }

    #[test]
    fn default_is_limited_to_the_end_to_end_measured_head_profile() {
        assert!(measured_default(DType::MQ4G256V2, 248_320, 5_120));
        assert!(!measured_default(DType::MQ4G256, 248_320, 5_120));
        assert!(!measured_default(DType::Q8_0, 248_320, 2_560));
        assert!(!measured_default(DType::Q8_0, 248_320, 5_120));
        assert!(!measured_default(DType::MQ4G256V2, 256, 5_120));
        assert!(!measured_default(DType::MQ4G256V2, 248_320, 1_024));
    }

    fn head(dtype: DType, m: usize, k: usize, bytes: usize) -> WeightTensor {
        WeightTensor {
            buf: GpuTensor {
                // Metadata-only: this borrowed sentinel is never used by HIP.
                buf: unsafe { hip_bridge::DeviceBuffer::from_raw(std::ptr::null_mut(), bytes) },
                shape: vec![m, k],
                dtype,
            },
            gpu_dtype: dtype,
            m,
            k,
            row_stride: 0,
            paro: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        }
    }

    #[test]
    fn row_layout_checks_format_extent_and_alignment() {
        for dtype in [DType::MQ4G256, DType::MQ4G256V2] {
            assert_eq!(row_bytes(&head(dtype, 256, 512, 256 * 272)), Some(272));
            assert_eq!(row_bytes(&head(dtype, 256, 512, 256 * 272 - 1)), None);
            assert_eq!(row_bytes(&head(dtype, 256, 511, 256 * 272)), None);
        }
        assert_eq!(
            row_bytes(&head(DType::Q8_0, 256, 512, 256 * 544)),
            Some(544)
        );
        assert_eq!(row_bytes(&head(DType::MQ3G256, 256, 512, 256 * 208)), None);
        let mut padded = head(DType::Q8_0, 256, 512, 256 * 544);
        padded.row_stride = 544;
        assert_eq!(row_bytes(&padded), None);
    }
}
