// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact replay screen for the gfx1100 TP2 graph-resident fused barrier and
//! rank-ordered residual reduction.

use hip_bridge::{Graph, GraphExec};
use hipfire_runtime::multi_gpu::Gpus;
use rdna_compute::{DType, GpuTensor};

const ELEMS: usize = 5_120;
const DELAY_BYTES: usize = 64 * 1024 * 1024;
const BARRIERS: usize = 128;
const CAPTURE_MODE_RELAXED: u32 = 2;

struct CapturedRank {
    graph: Graph,
    exec: GraphExec,
    _blobs: Vec<Vec<u8>>,
}

fn upload(gpu: &rdna_compute::Gpu, tensor: &GpuTensor, value: f32) {
    let host = vec![value; ELEMS];
    let bytes = unsafe {
        std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), host.len() * size_of::<f32>())
    };
    gpu.hip
        .memcpy_htod_async(
            &tensor.buf,
            bytes,
            gpu.active_stream.as_ref().expect("active stream"),
        )
        .expect("enqueue tensor upload");
}

fn main() {
    let replays = std::env::var("HIPFIRE_TP2_GRAPH_REPLAYS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100);
    assert!(replays > 1);

    let mut gpus = Gpus::init_uniform(2, 2).expect("init TP2 GPUs");
    assert!(gpus.devices.iter().all(|gpu| gpu.arch == "gfx1100"));
    assert!(
        gpus.enable_peer_all().expect("enable all peer links"),
        "screen requires complete peer access"
    );
    for gpu in &mut gpus.devices {
        gpu.bind_thread().expect("bind stream owner");
        gpu.active_stream = Some(gpu.hip.stream_create().expect("create stream"));
    }
    gpus.prepare_tp_graph_signals(BARRIERS)
        .expect("prepare graph signal tape");
    gpus.prepare_tp_graph_peer_shadows(ELEMS * size_of::<f32>())
        .expect("prepare peer payload shadows");

    let mut partials = Vec::with_capacity(2);
    let mut increments = Vec::with_capacity(2);
    let mut residuals = Vec::with_capacity(2);
    let mut delays = Vec::with_capacity(2);
    for rank in 0..2 {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread().expect("bind alloc owner");
        partials.push(
            gpu.alloc_tensor(&[ELEMS], DType::F32)
                .expect("partial tensor"),
        );
        increments.push(
            gpu.alloc_tensor(&[ELEMS], DType::F32)
                .expect("increment tensor"),
        );
        residuals.push(
            gpu.alloc_tensor(&[ELEMS], DType::F32)
                .expect("residual tensor"),
        );
        delays.push(gpu.hip.malloc(DELAY_BYTES).expect("delay buffer"));
        upload(gpu, &increments[rank], (rank + 1) as f32);
    }

    for gpu in &mut gpus.devices {
        gpu.bind_thread().expect("bind capture begin");
        gpu.graphs.capture_blobs.clear();
        gpu.graphs.capture_mode = true;
        gpu.hip
            .stream_begin_capture(
                gpu.active_stream.as_ref().expect("active stream"),
                CAPTURE_MODE_RELAXED,
            )
            .expect("begin rank capture");
    }
    gpus.begin_tp_graph_signal_capture()
        .expect("rewind signal cursor");
    let partial_refs: Vec<_> = partials.iter().map(|tensor| &tensor.buf).collect();
    let residual_refs: Vec<_> = residuals.iter().map(|tensor| &tensor.buf).collect();
    for barrier in 0..BARRIERS {
        // Reproduce the model's important coherency shape: every barrier is
        // preceded by a producer kernel that overwrites the same partial
        // allocation reused by the next layer.
        for rank in 0..2 {
            gpus.devices[rank]
                .add_f32(&partials[rank], &increments[rank], &partials[rank])
                .expect("capture partial producer");
        }
        gpus.all_reduce_sum_f32_peer_direct_add_alternating(
            &partial_refs,
            &residual_refs,
            ELEMS,
            barrier & 1,
        )
        .expect("capture fused TP2 reduction");
    }

    let mut captures = Vec::with_capacity(2);
    for gpu in &mut gpus.devices {
        gpu.bind_thread().expect("bind capture end");
        let graph = gpu
            .hip
            .stream_end_capture(gpu.active_stream.as_ref().expect("active stream"))
            .expect("end rank capture");
        gpu.graphs.capture_mode = false;
        let exec = gpu
            .hip
            .graph_instantiate(&graph)
            .expect("instantiate graph");
        captures.push(CapturedRank {
            graph,
            exec,
            _blobs: std::mem::take(&mut gpu.graphs.capture_blobs),
        });
    }

    for replay in 0..replays {
        for rank in 0..2 {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind upload owner");
            gpu.hip
                .memset_async(
                    &delays[rank],
                    replay as i32,
                    DELAY_BYTES,
                    gpu.active_stream.as_ref().expect("active stream"),
                )
                .expect("enqueue producer delay");
            upload(gpu, &partials[rank], (replay * 2 + rank + 1) as f32);
            upload(gpu, &residuals[rank], (100 + rank) as f32);
        }
        for rank in 0..2 {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind graph launch");
            gpu.hip
                .graph_launch(
                    &captures[rank].exec,
                    gpu.active_stream.as_ref().expect("active stream"),
                )
                .expect("launch rank graph");
        }
        for rank in 0..2 {
            let gpu = &gpus.devices[rank];
            gpu.bind_thread().expect("bind replay sync");
            gpu.hip
                .stream_synchronize(gpu.active_stream.as_ref().expect("active stream"))
                .expect("sync rank replay");
            let host = gpu
                .download_f32(&residuals[rank])
                .expect("download reduced residual");
            let b = BARRIERS as f32;
            let initial_pair = (replay * 2 + 1) as f32 + (replay * 2 + 2) as f32;
            let producer_triangle = 3.0 * b * (b + 1.0) * 0.5;
            let expected = (100 + rank) as f32 + b * initial_pair + producer_triangle;
            assert!(
                host.iter().all(|&value| value == expected),
                "replay {replay} rank {rank}: expected {expected}, head {:?}",
                &host[..16]
            );
        }
    }

    for rank in 0..2 {
        let gpu = &gpus.devices[rank];
        gpu.bind_thread().expect("bind graph destroy");
        let capture = captures.remove(0);
        gpu.hip
            .graph_exec_destroy(capture.exec)
            .expect("destroy graph exec");
        gpu.hip.graph_destroy(capture.graph).expect("destroy graph");
    }
    println!("PASS gfx1100 TP2 graph fused reduce: {replays} exact replays");
}
