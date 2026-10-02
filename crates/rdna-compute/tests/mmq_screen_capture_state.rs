// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Gpu::mmq_screen_weight` runs eagerly and must leave graph-capture state
//! alone. It used to set `capture_mode = true` around its GEMMs, which
//! (1) left the flag stuck on when a GEMM returned an error, (2) suppressed the
//! graph invalidation of its own Q8_1 / FP16 scratch growth, and (3) pushed
//! stray kernarg blobs into `capture_blobs`.
//!
//! `#[ignore]`d: needs a GPU with a working HIP toolchain. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --test mmq_screen_capture_state -- --ignored

use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::Mutex;

/// The launch-fault seam is process-global: serialize the tests in this binary.
static LOCK: Mutex<()> = Mutex::new(());

const M: usize = 128;
const K: usize = 256;

fn gpu_or_skip() -> Option<Gpu> {
    match Gpu::init() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skip: no GPU ({e})");
            None
        }
    }
}

/// An all-zero HFQ4-G256 weight (136 B/group); its own device pointer keys
/// a fresh screen-cache entry.
fn weight(gpu: &mut Gpu) -> GpuTensor {
    let bytes = vec![0u8; M * (K / 256) * 136];
    gpu.upload_raw(&bytes, &[bytes.len()]).expect("weight")
}

#[test]
#[ignore]
fn screen_error_leaves_capture_mode_off() {
    let _l = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(mut gpu) = gpu_or_skip() else { return };
    let w = weight(&mut gpu);

    hip_bridge::arm_hip_fault("launch", 1);
    let safe = gpu.mmq_screen_weight(&w, M, K);
    hip_bridge::arm_hip_fault("launch", 0);

    assert!(!safe, "a failed screen must report MMQ unsafe");
    assert!(
        !gpu.graphs.capture_mode,
        "capture_mode stuck on after a screen error"
    );
    assert!(
        gpu.graphs.capture_blobs.is_empty(),
        "screen pushed {} capture blobs",
        gpu.graphs.capture_blobs.len()
    );
}

#[test]
#[ignore]
fn screen_scratch_growth_invalidates_captured_graph() {
    let _l = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let Some(mut gpu) = gpu_or_skip() else { return };
    // The screen's WMMA reference is the gfx11 kernel: elsewhere it fails to
    // compile before any scratch grows (gfx1201: "needs wmma-256b-insts"), so
    // there is no growth to observe. The error test above covers that arch.
    if !gpu.arch_caps.has_wmma_w32() {
        eprintln!("skip: {} has no gfx11 WMMA screen path", gpu.arch);
        return;
    }
    gpu.ensure_capture_stream().expect("stream");

    // Capture any graph so scratch growth has something to invalidate.
    let a = gpu.zeros(&[256], DType::F32).expect("a");
    let b = gpu.zeros(&[256], DType::F32).expect("b");
    gpu.copy_f32_buffer(&b, &a, 256).expect("warm copy");
    gpu.hip.device_synchronize().expect("sync");
    let stream = gpu.active_stream.take().expect("stream");
    gpu.graphs
        .begin_graph_capture(&gpu.hip, gpu.device_id, &stream)
        .expect("begin capture");
    gpu.active_stream = Some(stream);
    gpu.copy_f32_buffer(&b, &a, 256).expect("captured copy");
    let stream = gpu.active_stream.take().unwrap();
    gpu.graphs
        .end_graph_capture(&gpu.hip, gpu.device_id, &stream)
        .expect("end capture");
    gpu.active_stream = Some(stream);
    assert!(gpu.graphs.graph_exec.is_some(), "graph not captured");
    assert!(gpu.graphs.capture_blobs.is_empty());

    // First screen on a fresh Gpu grows the Q8_1 MMQ X (and FP16 X) scratch.
    let w = weight(&mut gpu);
    gpu.mmq_screen_weight(&w, M, K);

    assert!(!gpu.graphs.capture_mode, "capture_mode left on");
    assert!(
        gpu.graphs.graph_exec.is_none(),
        "screen scratch growth did not invalidate the captured graph"
    );
    assert!(
        gpu.graphs.capture_blobs.is_empty(),
        "screen pushed {} capture blobs",
        gpu.graphs.capture_blobs.len()
    );
}
