// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scratch-growth contract for `Gpu::qwen4_f16_x_scratch`: growing the shared
//! FP16 activation scratch frees the old buffer, so a graph that captured the
//! old pointer must be invalidated before the free, and a re-capture over the
//! new buffer must replay to the eager result.
//!
//! Before the fix the grow skipped `invalidate_for_scratch_growth`: the AR
//! graph survived holding the freed pointer and the next replay read freed
//! memory. The test stops at that point instead of replaying into freed VRAM.
//!
//! `#[ignore]`d: needs a GPU with a working HIP toolchain. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --test scratch_growth_capture -- --ignored

use rdna_compute::{DType, Gpu, GpuTensor};

/// F32 view over the f16 scratch bytes (`copy_f32_buffer` is the probe kernel).
fn f32_view(t: &GpuTensor) -> GpuTensor {
    let n = t.buf.size() / 4;
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(t.buf.as_ptr(), n * 4) },
        shape: vec![n],
        dtype: DType::F32,
    }
}

fn fill(gpu: &Gpu, t: &GpuTensor, seed: f32) -> Vec<f32> {
    let vals: Vec<f32> = (0..t.numel()).map(|i| seed + i as f32).collect();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("fill scratch");
    vals
}

fn capture_copy(gpu: &mut Gpu, dst: &GpuTensor, src: &GpuTensor, n: usize) {
    let stream = gpu.active_stream.take().expect("capture stream");
    gpu.graphs
        .begin_graph_capture(&gpu.hip, gpu.device_id, &stream)
        .expect("begin capture");
    gpu.active_stream = Some(stream);
    gpu.copy_f32_buffer(dst, src, n).expect("captured copy");
    let stream = gpu.active_stream.take().unwrap();
    gpu.graphs
        .end_graph_capture(&gpu.hip, gpu.device_id, &stream)
        .expect("end capture");
    gpu.active_stream = Some(stream);
}

fn replay_and_read(gpu: &mut Gpu, out: &GpuTensor) -> Vec<f32> {
    let stream = gpu.active_stream.take().unwrap();
    gpu.graphs
        .graph_launch(&gpu.hip, gpu.device_id, &stream)
        .expect("replay");
    gpu.hip.stream_synchronize(&stream).expect("sync");
    gpu.active_stream = Some(stream);
    gpu.download_f32(out).expect("download")
}

#[test]
#[ignore]
fn qwen4_f16_x_scratch_growth_invalidates_captured_graph() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skip: no GPU ({e})");
            return;
        }
    };
    gpu.ensure_capture_stream().expect("stream");

    let small = 2048usize; // f16 elems
    let src = f32_view(&gpu.qwen4_f16_x_scratch(small).expect("scratch"));
    let n = src.numel();
    fill(&gpu, &src, 1.0);
    let out = gpu.zeros(&[n], DType::F32).expect("out");
    // Warm the kernel outside capture (no compile mid-capture).
    gpu.copy_f32_buffer(&out, &src, n).expect("warm copy");
    gpu.hip.device_synchronize().expect("sync");

    capture_copy(&mut gpu, &out, &src, n);
    assert!(gpu.graphs.graph_exec.is_some(), "graph not captured");
    let old_ptr = src.buf.as_ptr();

    // Force growth: the old buffer (captured by the graph) is freed.
    let big = f32_view(&gpu.qwen4_f16_x_scratch(small * 64).expect("grow"));
    assert_ne!(big.buf.as_ptr(), old_ptr, "scratch did not reallocate");
    assert!(
        gpu.graphs.graph_exec.is_none(),
        "stale graph survived scratch growth: it still reads freed {old_ptr:p}, \
         live scratch is {:p}",
        big.buf.as_ptr()
    );

    // Re-capture over the new buffer; replay must match eager.
    let want = fill(&gpu, &big, 7.0);
    let n_big = big.numel();
    let out_big = gpu.zeros(&[n_big], DType::F32).expect("out_big");
    let eager = gpu.zeros(&[n_big], DType::F32).expect("eager");
    gpu.copy_f32_buffer(&eager, &big, n_big).expect("eager copy");
    gpu.hip.device_synchronize().expect("sync");
    let eager = gpu.download_f32(&eager).expect("eager download");
    assert_eq!(eager, want, "eager copy wrong");

    capture_copy(&mut gpu, &out_big, &big, n_big);
    assert_eq!(replay_and_read(&mut gpu, &out_big), eager, "replay != eager");
}
