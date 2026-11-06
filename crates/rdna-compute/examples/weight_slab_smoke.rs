// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Hardware smoke for weight slabs (`HIPFIRE_WEIGHT_SLAB=off|2m|16m`).
//!
//! Uploads a Raon-shaped set of weight tensors (many 1-8 MiB tensors, a long
//! tail of small ones) inside a weight-load scope, then proves on hardware:
//! - every tensor reads back byte-identical and is 4 KiB aligned;
//! - the process's `drm-memory-vram` (fdinfo) at load, against the requested
//!   bytes, and the `own` + `ring` + `unledgered` admission split against it;
//! - individual frees (pool-parked or immediate) leave the slabs alive until
//!   the last view is gone, and `drain_pool` returns every slab;
//! - three load/unload cycles leave the process at its baseline VRAM.
//!
//!   HIPFIRE_WEIGHT_SLAB=2m cargo run --release -p rdna-compute --example weight_slab_smoke

use rdna_compute::{DType, Gpu, GpuTensor};

const MIB: usize = 1 << 20;

fn fdinfo_vram_mib(bdf: &str) -> f64 {
    let mut best = std::collections::HashMap::new();
    if let Ok(entries) = std::fs::read_dir("/proc/self/fdinfo") {
        for entry in entries.flatten() {
            let Ok(text) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            let (mut pdev, mut id, mut kib) = (None, None, None);
            for line in text.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    let v = v.trim();
                    match k {
                        "drm-pdev" => pdev = Some(v.to_string()),
                        "drm-client-id" => id = Some(v.to_string()),
                        "drm-memory-vram" => {
                            kib = v
                                .split_whitespace()
                                .next()
                                .and_then(|n| n.parse::<f64>().ok())
                        }
                        _ => {}
                    }
                }
            }
            if let (Some(p), Some(i), Some(k)) = (pdev, id, kib) {
                if p.eq_ignore_ascii_case(bdf) {
                    best.insert(i, k);
                }
            }
        }
    }
    best.values().sum::<f64>() / 1024.0
}

/// Tensor byte sizes of a Raon-like model: (size, count).
fn shape() -> Vec<(usize, usize)> {
    vec![
        (4096, 900),
        (64 * 1024, 700),
        (300 * 1024, 500),
        (MIB + 4096, 300),
        (MIB + 512 * 1024, 800),
        (2 * MIB - 4096, 200),
        (2 * MIB + 4096, 120),
        (3 * MIB, 150),
        (6 * MIB + MIB / 4, 200),
        (9 * MIB + 12345, 20),
        (40 * MIB, 6),
    ]
}

fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i.wrapping_mul(31) + seed * 7 + (i >> 9)) % 253) as u8)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    let bdf = gpu.hip.device_pci_bus_id(gpu.device_id)?;
    let mode = gpu.hip.slab_mode();
    println!(
        "device {} ({bdf}) HIPFIRE_WEIGHT_SLAB -> {}",
        gpu.device_id,
        mode.name()
    );
    // Force the HIP queue/ring BOs into existence before the baseline.
    let warm = gpu.upload_raw(&[1u8; 4096], &[4096])?;
    gpu.free_tensor(warm)?;
    gpu.drain_pool();
    let baseline = fdinfo_vram_mib(&bdf);
    println!("baseline fdinfo {baseline:.1} MiB");

    for cycle in 0..3 {
        let mut tensors: Vec<(GpuTensor, usize, usize)> = Vec::new();
        let mut requested = 0usize;
        gpu.weight_slab_scope_begin();
        let mut n = 0usize;
        for (size, count) in shape() {
            for _ in 0..count {
                let data = pattern(size, n);
                // Alternate the three weight entry points.
                let t = match n % 3 {
                    0 => gpu.upload_raw(&data, &[size])?,
                    1 => {
                        let mut t = gpu.alloc_weight_tensor(&[size], DType::Raw)?;
                        gpu.hip.memcpy_htod(&t.buf, &data)?;
                        t.shape = vec![size];
                        t
                    }
                    _ => {
                        let words = size / 4;
                        let f: Vec<f32> = data[..words * 4]
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        if f.iter().any(|v| !v.is_finite()) {
                            gpu.upload_raw(&data, &[size])?
                        } else {
                            gpu.upload_f32(&f, &[words])?
                        }
                    }
                };
                requested += t.buf.size();
                tensors.push((t, size, n));
                n += 1;
            }
        }
        gpu.weight_slab_scope_end();
        let loaded = fdinfo_vram_mib(&bdf) - baseline;
        let stats = gpu.hip.slab_stats();
        let cap = gpu.planned_capacity()?;
        println!(
            "cycle {cycle}: {} tensors, requested {:.1} MiB, fdinfo +{loaded:.1} MiB (rounding/slack {:+.1}), slabs {} ({:.1} MiB) live views {}",
            tensors.len(),
            requested as f64 / MIB as f64,
            loaded - requested as f64 / MIB as f64,
            stats.slabs,
            stats.slab_bytes as f64 / MIB as f64,
            stats.live_views
        );
        println!("cycle {cycle}: admission {cap}");

        // Readback + alignment of every tensor.
        let mut bad = 0usize;
        for (t, size, seed) in &tensors {
            if t.buf.as_ptr() as usize % 4096 != 0 {
                bad += 1;
                eprintln!("misaligned tensor seed {seed}");
            }
            let back = gpu.download_raw_bytes(t)?;
            let want = pattern(*size, *seed);
            let n_cmp = back.len().min(want.len());
            if back[..n_cmp] != want[..n_cmp] {
                bad += 1;
                eprintln!("readback mismatch tensor seed {seed} size {size}");
            }
        }
        assert_eq!(bad, 0, "readback/alignment failures");

        // Free half individually (alternating pooled and immediate), the rest
        // after a first drain: slabs must outlive their remaining views.
        let mut keep = Vec::new();
        for (i, (t, size, seed)) in tensors.into_iter().enumerate() {
            if i % 2 == 0 {
                if i % 4 == 0 {
                    gpu.free_tensor(t)?;
                } else {
                    gpu.release_tensor_immediate(t)?;
                }
            } else {
                keep.push((t, size, seed));
            }
        }
        gpu.drain_pool();
        let mid = gpu.hip.slab_stats();
        let kept_views = keep.iter().filter(|(t, _, _)| t.buf.is_slab_view()).count();
        assert_eq!(
            mid.live_views, kept_views,
            "live slab views after the peer frees"
        );
        for (t, size, seed) in &keep {
            let back = gpu.download_raw_bytes(t)?;
            let want = pattern(*size, *seed);
            let n_cmp = back.len().min(want.len());
            assert!(
                back[..n_cmp] == want[..n_cmp],
                "kept tensor {seed} corrupted after peer frees"
            );
        }
        for (t, _, _) in keep {
            gpu.free_tensor(t)?;
        }
        gpu.drain_pool();
        let end = gpu.hip.slab_stats();
        let after = fdinfo_vram_mib(&bdf) - baseline;
        println!(
            "cycle {cycle}: after unload fdinfo {after:+.1} MiB vs baseline, slabs {} live views {}",
            end.slabs, end.live_views
        );
        assert_eq!(end.slabs, 0, "slabs leaked after unload");
        assert_eq!(end.live_views, 0);
        assert!(
            after.abs() < 8.0,
            "VRAM did not return to baseline: {after:.1} MiB"
        );
    }
    println!("OK");
    Ok(())
}
