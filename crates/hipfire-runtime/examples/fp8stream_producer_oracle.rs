// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! THROWAWAY lab oracle (not for commit): fused RMSNorm/FWHT → FP8-stream
//! producer vs the standalone producer + `prepare_mq4v2_fp8_x_f32` chain,
//! compared bytewise (F32 `x_rot` bits, E4M3 codes, F32 half sums, F32 row
//! scales) on synthetic rows spanning zero/tiny/mid/large/saturating scales.
//!
//! Run: `HIPFIRE_GFX12_FP8_STREAM=1 <bin>` on exact gfx1201. Exit 1 on any
//! byte mismatch. Both the plain and AWQ twins are covered.

fn row_scale(r: usize) -> f32 {
    const SCALES: [f32; 7] = [0.0, 1e-3, 0.7, 3.0, 25.0, 260.0, 5000.0];
    SCALES[r % SCALES.len()]
}

fn hash01(i: usize) -> f32 {
    (((i.wrapping_mul(2654435761)) % 2001) as f32) / 1000.0 - 1.0
}

fn run_case(gpu: &mut rdna_compute::Gpu, n: usize, k: usize, awq: bool) -> bool {
    use rdna_compute::DType;
    let eps = 1e-6f32;

    // Inputs: row-varying scales + explicit zero / saturating rows.
    let mut x = vec![0.0f32; n * k];
    for r in 0..n {
        let s = if r == n - 1 { 1e5 } else { row_scale(r) };
        for c in 0..k {
            x[r * k + c] = s * hash01(r * k + c);
        }
    }
    let w: Vec<f32> = (0..k).map(|c| 1.0 + (hash01(c + 7) * 0.5 + 0.5) * 0.2).collect();
    let a: Vec<f32> = (0..k).map(|c| 0.5 + (hash01(c + 13) * 0.5 + 0.5)).collect();

    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();
    let d_w = gpu.upload_f32(&w, &[k]).unwrap();
    let d_a = gpu.upload_f32(&a, &[k]).unwrap();
    let d_rot_a = gpu.zeros(&[n, k], DType::F32).unwrap();
    let d_rot_b = gpu.zeros(&[n, k], DType::F32).unwrap();

    // Path A: standalone producer, then the standalone F32 pack.
    if awq {
        gpu.fused_rmsnorm_rotate_mq_awq_batched(&d_x, &d_w, &d_a, &d_rot_a, k, eps, n)
            .unwrap();
    } else {
        gpu.fused_rmsnorm_rotate_mq_batched(&d_x, &d_w, &d_rot_a, k, eps, n)
            .unwrap();
    }
    let prep_a = gpu.prepare_mq4v2_fp8_x(&d_rot_a, n, k, 1).unwrap();

    // Path B: fused producer → FP8 stream.
    let awq_opt = if awq { Some(&d_a) } else { None };
    let prep_b = gpu
        .fused_rmsnorm_rotate_mq_fp8_gfx12_batched(&d_x, &d_w, awq_opt, &d_rot_b, k, eps, n)
        .unwrap();

    gpu.hip.device_synchronize().unwrap();

    assert_eq!(prep_a.x_fp8_bytes, n * k);
    assert_eq!(prep_b.x_fp8_bytes, n * k);
    assert_eq!(prep_a.half_sums_bytes, prep_b.half_sums_bytes);
    assert_eq!(prep_a.row_scales_bytes, prep_b.row_scales_bytes);

    fn dtoh(gpu: &rdna_compute::Gpu, ptr: *mut std::ffi::c_void, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        let buf = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, len) };
        gpu.hip.memcpy_dtoh(&mut out, &buf).unwrap();
        out
    }

    let rot_a = gpu.download_f32(&d_rot_a).unwrap();
    let rot_b = gpu.download_f32(&d_rot_b).unwrap();
    let codes_a = dtoh(gpu, prep_a.x_fp8, prep_a.x_fp8_bytes);
    let codes_b = dtoh(gpu, prep_b.x_fp8, prep_b.x_fp8_bytes);
    let sums_a = dtoh(gpu, prep_a.half_sums, prep_a.half_sums_bytes);
    let sums_b = dtoh(gpu, prep_b.half_sums, prep_b.half_sums_bytes);
    let sc_a = dtoh(gpu, prep_a.row_scales, prep_a.row_scales_bytes);
    let sc_b = dtoh(gpu, prep_b.row_scales, prep_b.row_scales_bytes);

    let mut bad = 0usize;
    let mut report = |name: &str, a: &[u8], b: &[u8]| {
        assert_eq!(a.len(), b.len());
        let mut n = 0usize;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if x != y {
                if n < 4 {
                    eprintln!("    {name}[{i}]: A={x:02x} B={y:02x}");
                }
                n += 1;
            }
        }
        if n > 0 {
            eprintln!("    {name}: {n} byte mismatches / {}", a.len());
            bad += n;
        }
    };

    let ra: Vec<u8> = rot_a.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    let rb: Vec<u8> = rot_b.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    report("x_rot", &ra, &rb);
    report("codes", &codes_a, &codes_b);
    report("half_sums", &sums_a, &sums_b);
    report("row_scales", &sc_a, &sc_b);

    // Show the exercised scale range (from path A).
    let scales: Vec<f32> = sc_a
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (mn, mx) = scales.iter().fold((f32::INFINITY, 0f32), |(a, b), &v| {
        (a.min(v), b.max(v))
    });
    eprintln!("  N={n} K={k} awq={awq}: mismatched_bytes={bad} scales_min={mn} scales_max={mx}");

    for t in [d_x, d_w, d_a, d_rot_a, d_rot_b] {
        gpu.free_tensor(t).unwrap();
    }
    bad == 0
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);
    assert_eq!(gpu.arch, "gfx1201", "oracle requires exact gfx1201");
    gpu.ensure_mq_signs().unwrap();

    let mut pass = true;
    for &k in &[256usize, 2048, 4096, 5120] {
        for &n in &[64usize, 128, 512] {
            for &awq in &[false, true] {
                pass &= run_case(&mut gpu, n, k, awq);
            }
        }
    }
    if pass {
        eprintln!("ORACLE PASS: all producer-emitted fp8 planes byte-identical");
    } else {
        eprintln!("ORACLE FAIL");
        std::process::exit(1);
    }
}
