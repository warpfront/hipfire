// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Can the CPU's host-DRAM weight stream and the GPU's PCIe read of the same
//! host RAM run at the same time, or is one host-memory bottleneck shared?
//!
//! This is the feasibility question behind handing a slice of an
//! `memory.offload_exec=cpu` spilled layer back to an idle GPU: the GPU's idle
//! fraction is large (the seam's `cpu exec: idle` line measures it), but the
//! gain exists only if the two readers do not saturate one shared resource.
//!
//! Three phases over the same 196 MB host-mapped buffer:
//!   A. CPU only — the production row-dot kernel (`hipfire_cpu::gemv`), rayon.
//!   B. GPU only — `memcpy_htod` of those bytes, which is the DMA read across
//!      the link the `pcie` arm's kernels perform per token.
//!   C. both, CPU on the main thread and the GPU loop on a second thread.
//!
//! If C's combined byte rate is close to A's, the CPU is already at the wall and
//! splitting rows across the two engines buys nothing. If C approaches A+B, the
//! idle GPU time is real headroom.
//!
//! Usage: cargo run --release -p hipfire-dispatch --example offload_split_probe

use std::time::Instant;

use hipfire_cpu::gemv::{gemv, row_bytes};
use hipfire_cpu::quant::CpuQuant;
use rdna_compute::Gpu;

/// ~196 MB, the buffer size the CPU-kernel ceiling was measured on.
const TARGET_BYTES: usize = 196 * 1024 * 1024;
const K: usize = 5120;
const Q: CpuQuant = CpuQuant::Mq4G256;

fn payload(len: usize) -> Vec<u8> {
    // Byte-diverse with plausible group headers; the probe measures bandwidth,
    // not values.
    let mut out = Vec::with_capacity(len);
    let (ge, gb) = (Q.group_elems(), Q.group_bytes());
    let groups = K / ge;
    while out.len() < len {
        for g in 0..groups {
            let mut bytes: Vec<u8> = (0..gb).map(|i| ((i + g) * 37 + 11) as u8).collect();
            bytes[..4].copy_from_slice(&0.0313f32.to_le_bytes());
            bytes[4..8].copy_from_slice(&(-0.4921f32).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
    }
    out.truncate(len);
    out
}

fn secs(f: impl FnOnce()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64()
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let per_row = row_bytes(Q, K);
    let m = TARGET_BYTES / per_row;
    let len = m * per_row;
    let reps = 60;
    println!(
        "arch={} {Q:?} m={m} k={K} buffer={:.1} MB reps={reps}",
        gpu.arch,
        len as f64 / 1e6
    );

    let host = gpu
        .upload_raw_host_mapped(&payload(len), &[len])
        .expect("host-mapped upload");
    assert!(gpu.host_located(&host));
    let dev = gpu.hip.malloc(len).expect("device dest");
    // Borrowed from the host-mapped tensor, so the reader thread can hold the
    // same pinned bytes the `pcie` kernels dereference (Gpu itself is not Sync).
    let bytes: &[u8] = gpu.host_bytes(&host).expect("host bytes");
    let x = vec![0.1f32; K];
    let mut y = vec![0.0f32; m];

    let cpu_alone = secs(|| {
        for _ in 0..reps {
            gemv(Q, bytes, m, K, &x, &mut y);
        }
    });
    let gpu_alone = secs(|| {
        for _ in 0..reps {
            gpu.memcpy_htod_auto(&dev, bytes).expect("h2d");
        }
    });

    // Warm both engines once so the concurrent phase is not paying for
    // page faults or pool wake-up of whichever ran second.
    gemv(Q, bytes, m, K, &x, &mut y);
    gpu.memcpy_htod_auto(&dev, bytes).expect("h2d");

    let both = std::thread::scope(|scope| {
        let cpu_thread = scope.spawn(|| {
            let mut y = vec![0.0f32; m];
            let t = Instant::now();
            for _ in 0..reps {
                gemv(Q, bytes, m, K, &x, &mut y);
            }
            t.elapsed().as_secs_f64()
        });
        let gpu_wall = secs(|| {
            for _ in 0..reps {
                gpu.memcpy_htod_auto(&dev, bytes).expect("h2d");
            }
        });
        (cpu_thread.join().expect("cpu thread"), gpu_wall)
    });

    let gb = |secs: f64| len as f64 * reps as f64 / 1e9 / secs;
    let (cpu_gbs, gpu_gbs) = (gb(cpu_alone), gb(gpu_alone));
    let (cpu_conc, gpu_conc) = (gb(both.0), gb(both.1));
    println!("A cpu alone : {cpu_gbs:6.2} GB/s ({:.2} ms/rep)", cpu_alone / reps as f64 * 1e3);
    println!("B gpu alone : {gpu_gbs:6.2} GB/s ({:.2} ms/rep)", gpu_alone / reps as f64 * 1e3);
    println!(
        "C concurrent: cpu {cpu_conc:6.2} GB/s, gpu {gpu_conc:6.2} GB/s, \
         combined {:.2} GB/s ({:.1}% of A+B, {:.1}% of A)",
        cpu_conc + gpu_conc,
        100.0 * (cpu_conc + gpu_conc) / (cpu_gbs + gpu_gbs),
        100.0 * (cpu_conc + gpu_conc) / cpu_gbs,
    );
    println!(
        "   cpu retention {:.0}%, gpu retention {:.0}% of their alone rates",
        100.0 * cpu_conc / cpu_gbs,
        100.0 * gpu_conc / gpu_gbs,
    );
}
