// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! T0: VCN JPEG ring bring-up (experiment/vcn-ring).
//!
//! Reproduces the plan's IP-8 discovery + NOP-fence result, then proves
//! SYNCOBJ_IN independently of JPEG two ways: (a) a chained producer/consumer
//! pair with no host wait between the submits — the consumer polls incomplete
//! at zero timeout and completes after the producer signals the syncobj on
//! the GPU; (b) a host-signaled syncobj gating a submit.
//!
//! UAPI corner (measured, kernel 7.0.0-31): SYNCOBJ_IN on a syncobj with no
//! attached fence yet — never signaled by a submit nor the host — is rejected
//! with EINVAL (-22) at submit time. The producer submit (or host signal)
//! attaches the fence; waiting on a fenced-but-unsignaled syncobj is the
//! production pattern and is what this example exercises.
//!
//! Run under the GPU lock:
//!   flock -w 300 /tmp/hipfire-gpu.lock ./target/debug/examples/vcn_jpeg_t0
//!   (build: cargo build -p redline --features vcn-jpeg --example vcn_jpeg_t0)

#[cfg(not(feature = "vcn-jpeg"))]
fn main() {
    eprintln!("vcn_jpeg_t0 requires --features vcn-jpeg");
}

#[cfg(feature = "vcn-jpeg")]
fn main() {
    use redline::device::Device;
    use redline::drm::{AMDGPU_HW_IP_VCN_JPEG, AMDGPU_IB_FLAG_EMIT_MEM_SYNC};
    use redline::queue::{ComputeQueue, Engine, SubmitSync, SyncObj};
    use std::time::Instant;

    let dev = Device::open(None).expect("device open");

    // 1. IP-8 (VCN JPEG) discovery — expect VCN 5.0.0, ring 0, aligns 256/64.
    let info = dev
        .query_hw_ip_info(AMDGPU_HW_IP_VCN_JPEG, 0)
        .expect("query_hw_ip_info(8,0)");
    let disco = info.ip_discovery_version;
    let (d_maj, d_min, d_rev) = ((disco >> 16) & 0xff, (disco >> 8) & 0xff, disco & 0xff);
    println!(
        "[t0] ip8 query: return=0 hw_ip_version={}.{} caps=0x{:x} \
         ib_start_align={} ib_size_align={} rings=0x{:x} userq_slots={} \
         ip_discovery={}.{}.{} (0x{disco:06x})",
        info.hw_ip_version_major,
        info.hw_ip_version_minor,
        info.capabilities_flags,
        info.ib_start_alignment,
        info.ib_size_alignment,
        info.available_rings,
        info.userq_num_slots,
        d_maj,
        d_min,
        d_rev,
    );
    assert_eq!(
        (info.hw_ip_version_major, info.hw_ip_version_minor),
        (5, 0),
        "T0 expects VCN 5.0"
    );
    assert_eq!(
        (d_maj, d_min, d_rev),
        (5, 0, 0),
        "T0 expects discovery 5.0.0"
    );
    assert_eq!(info.available_rings, 0x1, "T0 expects ring 0 only");
    assert_eq!(
        (info.ib_start_alignment, info.ib_size_alignment),
        (256, 64),
        "T0 expects aligns 256/64"
    );

    let queue = ComputeQueue::new(&dev).expect("queue");
    let jpeg = Engine {
        ip_type: AMDGPU_HW_IP_VCN_JPEG,
        ip_instance: 0,
        ring: 0,
    };

    // 2. JPEG NOP fence: eight Mesa NOP pairs {0x60000000,0} = 16 dwords.
    let ib = dev.alloc_vram(4096).expect("jpeg ib");
    assert_eq!(ib.gpu_addr % 256, 0, "IB VA must be 256-byte aligned");
    let mut words = Vec::with_capacity(16);
    for _ in 0..8 {
        words.push(0x6000_0000u32);
        words.push(0u32);
    }
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    dev.upload(&ib, &bytes).expect("upload nops");
    let t = Instant::now();
    let fence = queue
        .submit_async(&dev, jpeg, &ib, 16, &[&ib], &SubmitSync::default())
        .expect("jpeg nop submit");
    let submit_us = t.elapsed().as_secs_f64() * 1e6;
    let t = Instant::now();
    let done = queue
        .wait_fence(&dev, &fence, 10_000_000_000)
        .expect("jpeg fence wait");
    let wait_us = t.elapsed().as_secs_f64() * 1e6;
    println!(
        "[t0] jpeg nop: ib=64B/16dwords va=0x{:x} cs_submit return=0 seq={} \
         fence_query return=0 expired={} submit_us={:.3} fence_wait_us={:.3}",
        ib.gpu_addr, fence.seq_no, done as u8, submit_us, wait_us,
    );
    assert!(done, "JPEG NOP fence did not complete");

    // 3. Chained SYNCOBJ_IN proof on compute (production pattern): producer
    //    signals `gate` via SYNCOBJ_OUT; consumer waits via SYNCOBJ_IN with
    //    EMIT_MEM_SYNC. No host wait between the two submits.
    let gate = SyncObj::new(&dev).expect("syncobj create");
    println!("[t0] syncobj created: handle={} (unsignaled)", gate.handle);
    // Established 2-dword NOP (poc_submit.rs): the header claims 1 body dword,
    // so a 1-dword IB would overrun into unmapped memory and hang the fence.
    let nop: [u32; 2] = [(3 << 30) | (0x10 << 8) | 0, 0xDEADBEEF];
    let cnop = dev.alloc_vram(4096).expect("compute nop ib");
    let nop_bytes: Vec<u8> = nop.iter().flat_map(|d| d.to_le_bytes()).collect();
    dev.upload(&cnop, &nop_bytes).expect("upload compute nop");
    let prod = SubmitSync {
        wait: &[],
        signal: Some(gate),
        ib_flags: 0,
    };
    let fprod = queue
        .submit_async(&dev, Engine::COMPUTE, &cnop, 2, &[&cnop], &prod)
        .expect("producer submit");
    let cons = SubmitSync {
        wait: &[gate],
        signal: None,
        ib_flags: AMDGPU_IB_FLAG_EMIT_MEM_SYNC,
    };
    let fcons = queue
        .submit_async(&dev, Engine::COMPUTE, &cnop, 2, &[&cnop], &cons)
        .expect("consumer submit");
    let polled = queue
        .wait_fence(&dev, &fcons, 0)
        .expect("zero-timeout poll");
    println!(
        "[t0] chained compute: prod_seq={} cons_seq={} zero-timeout expired={} (expect 0)",
        fprod.seq_no, fcons.seq_no, polled as u8,
    );
    assert!(
        !polled,
        "consumer gated on pending syncobj must poll incomplete"
    );
    let done = queue
        .wait_fence(&dev, &fcons, 10_000_000_000)
        .expect("consumer wait");
    println!("[t0] consumer terminal expired={} (expect 1)", done as u8);
    assert!(done, "consumer did not complete after producer signal");

    // 4. Host-signal path: signal a fresh syncobj, then gate a submit on it.
    let s2 = SyncObj::new(&dev).expect("syncobj create");
    s2.signal(&dev).expect("host signal");
    let hs = SubmitSync {
        wait: &[s2],
        signal: None,
        ib_flags: 0,
    };
    let fhs = queue
        .submit_async(&dev, Engine::COMPUTE, &cnop, 2, &[&cnop], &hs)
        .expect("host-gated submit");
    let hdone = queue
        .wait_fence(&dev, &fhs, 10_000_000_000)
        .expect("host-gated wait");
    println!(
        "[t0] host-signal gated submit: seq={} expired={} (expect 1)",
        fhs.seq_no, hdone as u8
    );
    assert!(hdone, "host-gated submit did not complete");

    gate.destroy(&dev);
    s2.destroy(&dev);
    dev.free_buffer(ib).expect("free jpeg ib");
    dev.free_buffer(cnop).expect("free compute ib");
    queue.destroy(&dev);
    println!("[t0] PASS");
}
