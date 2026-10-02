// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Offloaded weights must not consume the device allocation budget.
//!
//! This guards the single property that makes partial GPU offload worth
//! anything, and it is a property of the *mechanism*, not of any one model.
//!
//! The mechanism is load-bearing. A host-located VMM arena (`hipMemCreate`
//! PINNED/Host + `hipMemMap` into device VA) is charged against the device heap
//! 1:1 on gfx1201: measured, holding a 4 GiB host arena drops the allocatable
//! device headroom from 15360 MB to 11264 MB. Offload built on it is net-zero for
//! the VRAM it exists to free — it moves bytes to system RAM and charges the card
//! for them anyway. `hipHostMalloc(hipHostMallocMapped)` + `hipHostGetDevicePointer`
//! moves the same bytes for a measured cost of 0 MB, which is what
//! `Gpu::upload_raw_host` uses.
//!
//! A byte-identity or coherence test cannot catch a regression here: output stays
//! correct either way, only the capacity win disappears. Hence this measurement.
//!
//! Usage: cargo run --release -p rdna-compute --example host_offload_headroom

use hip_bridge::DeviceBuffer;
use rdna_compute::Gpu;

/// Host bytes to spill while measuring. Big enough that a 1:1 charge is
/// unmistakable against ladder granularity.
const SPILL_MB: usize = 4096;
/// A 1:1 charge would consume the whole spill. Anything above this fraction of it
/// means the mechanism started charging the device heap again.
const MAX_ACCEPTABLE_COST_FRACTION: usize = 4;

fn free_mb(gpu: &Gpu) -> i64 {
    let (free, _) = gpu.hip.get_vram_info().expect("hipMemGetInfo");
    (free / (1024 * 1024)) as i64
}

/// Device memory allocatable while the caller's allocations are live.
/// Allocated in 1 GiB steps until hipMalloc refuses, then all released.
fn device_headroom_mb(gpu: &Gpu) -> usize {
    let mut held: Vec<DeviceBuffer> = Vec::new();
    let mut total = 0usize;
    loop {
        match gpu.hip.malloc(1024 * 1024 * 1024) {
            Ok(b) => {
                held.push(b);
                total += 1024;
            }
            Err(_) => break,
        }
    }
    for b in held {
        let _ = gpu.hip.free(b);
    }
    total
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    println!("arch={} free={} MB", gpu.arch, free_mb(&gpu));
    assert_eq!(
        gpu.host_mapped_count(),
        0,
        "no host-mapped owners before the test"
    );

    let control = device_headroom_mb(&gpu);
    println!("control headroom            : {control} MB");

    let payload = vec![0u8; SPILL_MB * 1024 * 1024];
    let host = gpu.upload_raw_host(&payload, &[payload.len()]).unwrap();
    assert!(
        gpu.host_located(&host),
        "upload_raw_host did not host-locate"
    );
    assert_eq!(
        gpu.host_mapped_count(),
        1,
        "host-mapped owner not registered"
    );

    let with_spill = device_headroom_mb(&gpu);
    let cost = control.saturating_sub(with_spill);
    let limit = SPILL_MB / MAX_ACCEPTABLE_COST_FRACTION;
    println!(
        "headroom with {SPILL_MB} MB spilled: {with_spill} MB (cost {cost} MB, limit {limit} MB)"
    );

    gpu.free_tensor(host).unwrap();
    assert_eq!(
        gpu.host_mapped_count(),
        0,
        "host-mapped owner leaked after free"
    );
    let after = device_headroom_mb(&gpu);
    println!("headroom after free         : {after} MB");

    assert!(
        cost <= limit,
        "offloading {SPILL_MB} MB consumed {cost} MB of the device allocation budget \
         (limit {limit} MB): the host offload mechanism is charging device memory again, \
         which makes partial offload net-zero for VRAM. See Gpu::alloc_host_mapped_tensor."
    );
    assert!(
        after + limit >= control,
        "device headroom did not recover after freeing the spilled tensor \
         (control {control} MB, after {after} MB): the host allocation leaked device memory"
    );
    println!("host_offload_headroom: PASS");
}
