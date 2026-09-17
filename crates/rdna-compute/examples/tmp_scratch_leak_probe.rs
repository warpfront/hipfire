// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scratch-leak probe (gate 1 of the `grow_scratch_buffer` leak fix).
//!
//! Production pattern from the `grow_scratch_buffer` doc comment, reduced to
//! one slot: size a scratch buffer small (prefill A), capture a hipGraph that
//! embeds its pointer (decode capture), then grow the slot twice (prefills
//! B, C — larger shapes). Sample driver free VRAM after each phase.
//!
//! Pre-fix, every post-capture growth `std::mem::forget`s the old buffer, so
//! free VRAM steps down by the FULL new-buffer size at each growth (old
//! retained + new allocated). Post-fix, growth invalidates the captured
//! graphs first and frees the old buffer, so free VRAM steps down only by
//! the increment — flat to within one buffer's worth.
//!
//! The probe also proves the safety invariant post-fix: after growth the old
//! captured graph is gone (`graph_exec.is_none()`), a fresh capture at the
//! new size replays and synchronises cleanly (no `HipError(700)`).
//!
//! Slot choice: `q8_1_mmq_x_scratch` via the public `ensure_q8_1_mmq_x` — a
//! real scratch-consuming launch (the quantize prelude) whose kernargs embed
//! the scratch pointer, exactly like production. All slots share the
//! `grow_scratch_buffer` choke point, so one slot exercises the mechanism.
//!
//! Run: `cargo run -p rdna-compute --example tmp_scratch_leak_probe --features lab`
//! with `ROCR_VISIBLE_DEVICES=0 HOME=/tmp/home-ab0`.

use rdna_compute::{DType, Gpu};

const K: usize = 2048;
// (label, batch). needed = 16 blocks * batch * 144 B: ~18.9 / 37.7 / 75.5 / 151 MB.
const SHAPES: &[(&str, usize)] = &[
    ("A small ", 8192),
    ("B larger", 16384),
    ("C larger", 32768),
    ("D larger", 65535),
];

fn free_mb(gpu: &Gpu) -> f64 {
    let (free, _) = gpu.hip.get_vram_info().expect("hipMemGetInfo");
    free as f64 / 1e6
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("probe arch: {}", gpu.arch);
    gpu.ensure_capture_stream().expect("capture stream");

    // Staging X sized for the largest batch once, so X itself never moves and
    // every VRAM step below is scratch only.
    let max_batch = SHAPES.iter().map(|&(_, b)| b).max().unwrap();
    let x = gpu
        .alloc_tensor(&[max_batch * K], DType::F32)
        .expect("alloc x");
    // Zero it so the quantize prelude reads deterministic memory.
    gpu.hip
        .memset(&x.buf, 0, max_batch * K * 4)
        .expect("memset x");

    // Warmup: compile the quantize kernel AND size the scratch to shape A
    // outside any capture (mirrors the verify-warmup-then-capture flow).
    let (label_a, batch_a) = SHAPES[0];
    gpu.ensure_q8_1_mmq_x(&x, batch_a, K)
        .expect("warmup ensure");
    gpu.hip.device_synchronize().expect("warmup sync");
    let base = free_mb(&gpu);
    eprintln!("baseline after warmup ({label_a} b={batch_a}): {base:.1} MB free");

    // Decode capture: re-run the SAME shape under capture so the captured
    // graph embeds the shape-A scratch pointer. This latches the
    // any_graph_captured flag, arming the (previously leaking) branch.
    {
        let stream = gpu.active_stream.as_ref().unwrap() as *const _;
        // Split-borrow dance for the disjoint fields.
        let hip = &gpu.hip;
        let device_id = gpu.device_id;
        gpu.graphs
            .begin_graph_capture(hip, device_id, unsafe { &*stream })
            .expect("begin capture");
        gpu.ensure_q8_1_mmq_x(&x, batch_a, K)
            .expect("captured ensure");
        let hip = &gpu.hip;
        gpu.graphs
            .end_graph_capture(hip, device_id, unsafe { &*stream })
            .expect("end capture");
    }
    gpu.hip.device_synchronize().expect("capture sync");
    assert!(
        gpu.graphs.graph_exec.is_some(),
        "capture must produce a graph_exec"
    );
    eprintln!("captured graph embeds shape-A scratch pointer");

    // Growth phases: prefill B, C, D — each larger, each outside capture.
    let mut prev = base;
    for &(_label, batch) in &SHAPES[1..] {
        gpu.ensure_q8_1_mmq_x(&x, batch, K)
            .expect("growth ensure");
        gpu.hip.device_synchronize().expect("growth sync");
        let free = free_mb(&gpu);
        let step = prev - free;
        eprintln!("after growth to b={batch}: {free:.1} MB free (step {step:.1} MB)");
        prev = free;
    }

    // Post-fix invariant proof: the stale graph must be gone (it embedded a
    // freed pointer), and a fresh capture at the new size must replay cleanly.
    let stale_gone = gpu.graphs.graph_exec.is_none()
        && gpu.graphs.verify_graph_count() == 0
        && gpu.graphs.replay_graph_count() == 0;
    eprintln!("stale captured graph invalidated by growth: {stale_gone}");
    {
        let stream = gpu.active_stream.as_ref().unwrap() as *const _;
        let hip = &gpu.hip;
        let device_id = gpu.device_id;
        gpu.graphs
            .begin_graph_capture(hip, device_id, unsafe { &*stream })
            .expect("re-begin capture");
        gpu.ensure_q8_1_mmq_x(&x, max_batch, K)
            .expect("re-captured ensure");
        let hip = &gpu.hip;
        gpu.graphs
            .end_graph_capture(hip, device_id, unsafe { &*stream })
            .expect("re-end capture");
        let hip = &gpu.hip;
        let exec = gpu.graphs.graph_exec.as_ref().expect("re-captured exec");
        hip.graph_launch(exec, unsafe { &*stream })
            .expect("replay launch");
        gpu.hip.device_synchronize().expect("replay sync");
    }
    eprintln!("re-capture + replay at new size: clean sync, no HipError(700)");
}
