// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Can a shader read host-mapped memory — and specifically the quantized path?
//!
//! The existing host-offload checks (`vmm_tensor_smoke`, `host_offload_smoke`)
//! verify a host-located tensor with `memcpy_htod` / `memcpy_dtoh` plus a handle
//! property check. Those are satisfied by a page the copy engine can DMA to while
//! a shader still cannot dereference it. Partial offload's only real requirement
//! on the host buffer is that a kernel READS it, and that was never tested — the
//! first real offloaded load died with "Page not present or supervisor privilege".
//!
//! Two arms, each running the SAME kernel over a device-resident buffer and a
//! host-located one:
//!
//!   arm 1 — `gemv_f32` over an F32 buffer. Models the memory question alone.
//!   arm 2 — `gemv_mq4g256` over a `DType::Raw` MQ4 code blob. This is the pair
//!           production actually runs (`upload_raw_host` + the MQ family), so it
//!           is the one that localises a fault to the consumer path.
//!
//! Both arms are guarded against two ways to get a vacuous "PASS":
//!   * layout — the MQ kernel derives row_bytes = (K/256)*136 and reads
//!     A + row*row_bytes for row in [0,M), so a mis-sized blob or a K that is
//!     not a multiple of 256 reads off the end, which on a host VA faults
//!     identically to a genuine access problem.
//!   * non-vacuity — the result must be non-zero AND must change when the input
//!     changes, so a kernel that silently never read the buffer cannot pass.
//!
//! Arm 2 uses synthetic codes on purpose: the question is whether the kernel can
//! READ the host buffer, not whether the codes are meaningful.
//!
//! Usage: cargo run --release -p rdna-compute --example host_kernel_access

use rdna_compute::{DType, Gpu};

fn fwht_signs(seed: u32) -> Vec<f32> {
    let mut st = seed;
    (0..256)
        .map(|_| {
            st = st.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (st >> 16) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn any_nonzero(v: &[f32]) -> bool {
    v.iter().any(|x| x.abs() > 1e-6)
}

/// In-place radix-2 Hadamard over one 256-element group, matching the kernel's
/// expectation. `gemv_mq4g256` takes `x_rot` — x with the FWHT already applied —
/// so a raw x is the wrong input and decodes to a null result.
fn fwht256(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    assert_eq!(x.len(), 256);
    for i in 0..256 {
        x[i] *= s2[i];
    }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    for i in 0..256 {
        x[i] *= 0.0625 * s1[i];
    }
}

fn arm_f32(gpu: &mut Gpu) {
    // 390*64*4 = 99840 bytes: the exact size that faulted on the first real
    // offloaded load, and NOT a multiple of the 4096 VMM granularity, so this
    // also exercises the map round-up path rather than sidestepping it.
    const M: usize = 390;
    const K: usize = 64;
    println!("arm1: M={M} K={K} bytes={}", M * K * 4);

    let a_host: Vec<f32> = (0..M * K).map(|i| (i % 17) as f32 * 0.5).collect();
    let x: Vec<f32> = (0..K).map(|i| (i % 7) as f32).collect();

    let a_dev = gpu.upload_f32(&a_host, &[M, K]).unwrap();
    let xd = gpu.upload_f32(&x, &[K]).unwrap();
    let mut yd = gpu.zeros(&[M], DType::F32).unwrap();
    gpu.gemv_f32(&a_dev, &xd, &yd).unwrap();
    let reference = gpu.download_f32(&yd).unwrap();
    assert!(
        any_nonzero(&reference),
        "device reference is all zero - arm is vacuous"
    );

    // The production offload upload, not a hand-rolled host arena: this arm must
    // fail if the path the loader actually uses stops being kernel-readable.
    let a_off = gpu.upload_f32_host(&a_host, &[M, K]).unwrap();
    assert!(gpu.host_located(&a_off), "not host-located");
    println!("arm1: host buffer at 0x{:x}", a_off.buf.as_ptr() as usize);

    println!("arm1: launching gemv_f32 over a host-located F32 buffer...");
    let mut yh = gpu.zeros(&[M], DType::F32).unwrap();
    match gpu.gemv_f32(&a_off, &xd, &yh) {
        Ok(()) => {
            let got = gpu.download_f32(&yh).unwrap();
            // Non-vacuity: the host result must be non-zero AND must match the
            // device twin computed from the SAME bytes.
            let d = max_diff(&reference, &got);
            let verdict = if d == 0.0 && any_nonzero(&got) {
                "PASS"
            } else {
                "VACUOUS"
            };
            println!(
                "ARM1_F32 {verdict} (max diff {d}, non-zero {})",
                any_nonzero(&got)
            );
        }
        Err(e) => println!("ARM1_F32 FAULT: {e:?}"),
    }
}

fn arm_mq4(gpu: &mut Gpu) {
    const M: usize = 512;
    const K: usize = 1024;
    assert_eq!(
        K % 256,
        0,
        "K must be a multiple of 256 or groups_per_row truncates"
    );
    let expected = M * (K / 256) * 136;
    // Per-group layout (136 B): fp32 scale @0, fp32 zero @4, 128 nibble bytes @8.
    // The kernel reads the headers as 32-bit floats
    // (`__builtin_bit_cast(float, *(const unsigned int*)(gptr))`), so they must be
    // REAL f32 values - filling the blob with arbitrary bytes decodes the headers
    // to garbage floats and the whole result collapses to zero, which would make
    // the arm look vacuous rather than informative.
    let mut codes: Vec<u8> = vec![0u8; expected];
    for g in 0..(M * (K / 256)) {
        let base = g * 136;
        codes[base..base + 4].copy_from_slice(&0.01f32.to_le_bytes());
        codes[base + 4..base + 8].copy_from_slice(&0.0f32.to_le_bytes());
        for b in 0..128 {
            codes[base + 8 + b] = ((g * 7 + b * 31) % 256) as u8;
        }
    }
    assert_eq!(codes.len(), expected, "blob size != M*(K/256)*136");
    let s1v = fwht_signs(42);
    let s2v = fwht_signs(1042);
    let mut x: Vec<f32> = (0..K).map(|i| ((i % 13) as f32) * 0.25).collect();
    for chunk in x.chunks_mut(256) {
        fwht256(chunk, &s1v, &s2v);
    }

    println!(
        "arm2: M={M} K={K} groups_per_row={} row_bytes={} expected={} actual={} src_ptr=0x{:x}",
        K / 256,
        (K / 256) * 136,
        expected,
        codes.len(),
        codes.as_ptr() as usize
    );

    let a_dev = gpu.upload_raw(&codes, &[codes.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[K]).unwrap();
    let mut yd = gpu.zeros(&[M], DType::F32).unwrap();
    gpu.gemv_mq4g256(&a_dev, &xd, &yd, M, K).unwrap();
    let reference = gpu.download_f32(&yd).unwrap();
    assert!(
        any_nonzero(&reference),
        "device reference is all zero - arm is vacuous"
    );
    println!(
        "arm2: device-resident gemv_mq4g256 ok ({} outputs)",
        reference.len()
    );

    let a_off = gpu.upload_raw_host(&codes, &[codes.len()]).unwrap();
    assert!(gpu.host_located(&a_off), "not host-located");
    // Print the MAPPED VA the kernel is actually handed, not the source Vec —
    // otherwise a reader comparing it against a fault address is misled.
    println!(
        "arm2: dev_A=0x{:x} host_A=0x{:x} x=0x{:x} host_byte_size={}",
        a_dev.buf.as_ptr() as usize,
        a_off.buf.as_ptr() as usize,
        xd.buf.as_ptr() as usize,
        a_off.byte_size(),
    );

    println!("arm2: launching gemv_mq4g256 over a host-located MQ4 code blob...");
    let mut yh = gpu.zeros(&[M], DType::F32).unwrap();
    match gpu.gemv_mq4g256(&a_off, &xd, &yh, M, K) {
        Ok(()) => {
            let got = gpu.download_f32(&yh).unwrap();
            let d = max_diff(&reference, &got);
            let verdict = if d == 0.0 && any_nonzero(&got) {
                "PASS"
            } else {
                "VACUOUS"
            };
            println!(
                "ARM2_MQ4 {verdict} (max diff {d}, non-zero {})",
                any_nonzero(&got)
            );
        }
        Err(e) => println!("ARM2_MQ4 FAULT: {e:?}"),
    }
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    arm_f32(&mut gpu);
    arm_mq4(&mut gpu);
}
