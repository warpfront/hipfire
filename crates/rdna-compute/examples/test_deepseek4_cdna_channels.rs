//! Focused DeepSeek4 AR channel gate for the gfx942 lowering.
//!
//! Covers the two frozen MQ2-Lloyd routed-expert projections and the dense
//! MFP4G32E8SOA projection against host references. This is a channel test,
//! not a model-level acceptance or performance claim.
//!
//! Run on MI300X (or a gfx942 functional emulator):
//!   cargo run --release -p rdna-compute --example test_deepseek4_cdna_channels

use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn bytes_of_i32(values: &[i32]) -> &[u8] {
    // SAFETY: i32 is plain data and the returned byte slice cannot outlive it.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn bytes_of_u64(values: &[u64]) -> &[u8] {
    // SAFETY: u64 is plain data and the returned byte slice cannot outlive it.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    let out = match exp {
        0 if mant == 0 => sign,
        0 => {
            let shift = mant.leading_zeros() - 22;
            let normalized = (mant << (shift + 1)) & 0x03ff;
            sign | ((113 - shift) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(out)
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.5)
        .collect()
}

fn build_mq2_expert(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    const CODEBOOK: [u16; 4] = [0xbc00, 0xb800, 0x3800, 0x3c00];
    let groups = k / 256;
    let row_bytes = groups * 72;
    let mut data = vec![0u8; m * row_bytes];
    let mut state = seed;
    for row in 0..m {
        for group in 0..groups {
            let off = row * row_bytes + group * 72;
            for (i, bits) in CODEBOOK.iter().enumerate() {
                data[off + i * 2..off + i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            }
            for byte in &mut data[off + 8..off + 72] {
                let mut packed = 0u8;
                for slot in 0..4 {
                    packed |= ((lcg(&mut state) & 3) as u8) << (slot * 2);
                }
                *byte = packed;
            }
        }
    }
    data
}

fn mq2_row_dot(weights: &[u8], row: usize, k: usize, x: &[f32]) -> f32 {
    let groups = k / 256;
    let row_bytes = groups * 72;
    let mut acc = 0.0f32;
    for group in 0..groups {
        let off = row * row_bytes + group * 72;
        let mut codebook = [0.0f32; 4];
        for (i, value) in codebook.iter_mut().enumerate() {
            *value = half_to_f32(u16::from_le_bytes([
                weights[off + i * 2],
                weights[off + i * 2 + 1],
            ]));
        }
        for i in 0..256 {
            let packed = weights[off + 8 + i / 4];
            let index = ((packed >> ((i & 3) * 2)) & 3) as usize;
            acc += codebook[index] * x[group * 256 + i];
        }
    }
    acc
}

fn upload_experts(gpu: &Gpu, weights: &[Vec<u8>]) -> (Vec<GpuTensor>, GpuTensor) {
    let tensors: Vec<_> = weights
        .iter()
        .map(|data| gpu.upload_raw(data, &[data.len()]).expect("upload expert"))
        .collect();
    let pointers: Vec<u64> = tensors
        .iter()
        .map(|tensor| tensor.buf.as_ptr() as u64)
        .collect();
    let table = gpu
        .upload_raw(bytes_of_u64(&pointers), &[pointers.len() * 8])
        .expect("upload expert pointer table");
    (tensors, table)
}

fn e4m3_scale(byte: u8) -> f32 {
    let exp = (byte >> 3) & 0xf;
    let mant = byte & 7;
    if exp == 0 {
        return 0.015625 * mant as f32 * 0.125;
    }
    if exp == 0xf && mant == 7 {
        return 448.0;
    }
    2.0f32.powi(exp as i32 - 7) * (1.0 + mant as f32 * 0.125)
}

fn decode_e8(codeword: u32) -> [f32; 8] {
    let coset = (codeword >> 31) & 1;
    let mut e = [0u32; 8];
    let mut sum = 0u32;
    for i in 0..7 {
        e[i] = (codeword >> (4 * i)) & 0xf;
        sum += e[i];
    }
    let p7 = ((codeword >> 28) & 7) << 1;
    e[7] = p7 | ((sum + p7) & 1);
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        let centered = e[i] as i32 - 7;
        out[i] = centered as f32 + if coset != 0 { 0.5 } else { 0.0 };
    }
    out
}

fn build_e8_soa(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let row_bytes = 16 + scale_padded + blocks * 16;
    let mut data = vec![0u8; m * row_bytes];
    let mut state = seed;
    for row in 0..m {
        let off = row * row_bytes;
        data[off..off + 2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5
        data[off + 4..off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        data[off + 6] = 0x06;
        for block in 0..blocks {
            data[off + 16 + block] = 0x38 | (lcg(&mut state) as u8 & 7);
            let cw_off = off + 16 + scale_padded + block * 16;
            for slot in 0..4 {
                data[cw_off + slot * 4..cw_off + slot * 4 + 4]
                    .copy_from_slice(&lcg(&mut state).to_le_bytes());
            }
        }
    }
    data
}

fn e8_row_dot(weights: &[u8], row: usize, k: usize, x: &[f32]) -> f32 {
    let blocks = k / 32;
    let scale_padded = blocks.div_ceil(16) * 16;
    let row_bytes = 16 + scale_padded + blocks * 16;
    let off = row * row_bytes;
    let row_scale = half_to_f32(u16::from_le_bytes([weights[off], weights[off + 1]]));
    let mut acc = 0.0f32;
    for block in 0..blocks {
        let scale = row_scale * e4m3_scale(weights[off + 16 + block]) * 0.88;
        let cw_off = off + 16 + scale_padded + block * 16;
        for slot in 0..4 {
            let base = cw_off + slot * 4;
            let codeword = u32::from_le_bytes([
                weights[base],
                weights[base + 1],
                weights[base + 2],
                weights[base + 3],
            ]);
            for (i, value) in decode_e8(codeword).iter().enumerate() {
                acc += scale * value * x[block * 32 + slot * 8 + i];
            }
        }
    }
    acc
}

fn assert_close(label: &str, got: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), expected.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (i, (&actual, &reference)) in got.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "{label}[{i}] is non-finite: {actual}");
        let abs = (actual - reference).abs();
        let rel = abs / reference.abs().max(1.0e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        assert!(
            abs <= atol + rtol * reference.abs(),
            "{label}[{i}] mismatch: got={actual:.8} ref={reference:.8} abs={abs:.3e} rel={rel:.3e}"
        );
    }
    println!("PASS {label}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
}

fn hash_f32_bits(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn bench_mq2_ar_shapes(gpu: &mut Gpu) {
    const TOP_K: usize = 6;
    const WARMUPS: usize = 20;
    const TRIALS: usize = 100;

    let topk_host = [0i32, 1, 2, 3, 4, 5];
    let topk = gpu
        .upload_raw(bytes_of_i32(&topk_host), &[TOP_K * 4])
        .expect("upload bench top-k");
    let topk_weights = gpu
        .upload_f32(&[1.0 / TOP_K as f32; TOP_K], &[TOP_K])
        .expect("upload bench weights");

    let run_gate = |gpu: &mut Gpu| {
        const M: usize = 4096;
        const K: usize = 4096;
        let weights: Vec<_> = (0..TOP_K)
            .map(|expert| build_mq2_expert(M, K, 0x8100 + expert as u64))
            .collect();
        let (_experts, ptrs) = upload_experts(gpu, &weights);
        let x = gpu
            .upload_f32(&make_x(K, 0x8200), &[K])
            .expect("upload bench gate x");
        let gate = gpu
            .upload_f32(&vec![0.0; TOP_K * M / 2], &[TOP_K * M / 2])
            .expect("upload bench gate y");
        let up = gpu
            .upload_f32(&vec![0.0; TOP_K * M / 2], &[TOP_K * M / 2])
            .expect("upload bench up y");
        for _ in 0..WARMUPS {
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
                &ptrs, &topk, &x, &gate, &up, M, K, TOP_K,
            )
            .expect("warm gate");
        }
        gpu.hip.device_synchronize().expect("warm gate sync");
        let start = Instant::now();
        for _ in 0..TRIALS {
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
                &ptrs, &topk, &x, &gate, &up, M, K, TOP_K,
            )
            .expect("bench gate");
        }
        gpu.hip.device_synchronize().expect("bench gate sync");
        let us = start.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64;
        let bytes = (TOP_K * M * (K / 256) * 72) as f64;
        println!(
            "BENCH mq2 gate_up M={M} K={K} topk={TOP_K}: {us:.3} us, {:.1} GB/s payload",
            bytes / us / 1.0e3
        );
    };

    let run_down = |gpu: &mut Gpu| {
        const M: usize = 4096;
        const K: usize = 2048;
        let weights: Vec<_> = (0..TOP_K)
            .map(|expert| build_mq2_expert(M, K, 0x8300 + expert as u64))
            .collect();
        let (_experts, ptrs) = upload_experts(gpu, &weights);
        let x = gpu
            .upload_f32(&make_x(TOP_K * K, 0x8400), &[TOP_K, K])
            .expect("upload bench down x");
        let residual = gpu
            .upload_f32(&vec![0.0; M], &[M])
            .expect("upload bench residual");
        for _ in 0..WARMUPS {
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
                &ptrs,
                &topk,
                &topk_weights,
                &x,
                &residual,
                M,
                K,
                TOP_K,
            )
            .expect("warm down");
        }
        gpu.hip.device_synchronize().expect("warm down sync");
        let start = Instant::now();
        for _ in 0..TRIALS {
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
                &ptrs,
                &topk,
                &topk_weights,
                &x,
                &residual,
                M,
                K,
                TOP_K,
            )
            .expect("bench down");
        }
        gpu.hip.device_synchronize().expect("bench down sync");
        let us = start.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64;
        let bytes = (TOP_K * M * (K / 256) * 72) as f64;
        println!(
            "BENCH mq2 down M={M} K={K} topk={TOP_K}: {us:.3} us, {:.1} GB/s payload",
            bytes / us / 1.0e3
        );

        let expanded = gpu
            .upload_f32(&vec![0.0; TOP_K * M], &[TOP_K, M])
            .expect("upload bench expanded");
        let deterministic_residual = gpu
            .upload_f32(&vec![0.0; M], &[M])
            .expect("upload bench deterministic residual");
        for _ in 0..WARMUPS {
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
                &ptrs, &topk, &x, &expanded, M, K, TOP_K, 1,
            )
            .expect("warm expanded down");
            gpu.moe_down_combine_k8_batched(
                &expanded,
                &topk_weights,
                &deterministic_residual,
                M,
                TOP_K,
                1,
            )
            .expect("warm deterministic combine");
        }
        gpu.hip
            .device_synchronize()
            .expect("warm deterministic down sync");
        let start = Instant::now();
        for _ in 0..TRIALS {
            gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
                &ptrs, &topk, &x, &expanded, M, K, TOP_K, 1,
            )
            .expect("bench expanded down");
            gpu.moe_down_combine_k8_batched(
                &expanded,
                &topk_weights,
                &deterministic_residual,
                M,
                TOP_K,
                1,
            )
            .expect("bench deterministic combine");
        }
        gpu.hip
            .device_synchronize()
            .expect("bench deterministic down sync");
        let deterministic_us = start.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64;
        println!(
            "BENCH mq2 down-expanded+combine M={M} K={K} topk={TOP_K}: {deterministic_us:.3} us, {:.1} GB/s payload",
            bytes / deterministic_us / 1.0e3
        );
    };

    run_gate(gpu);
    run_down(gpu);
}

fn bench_e8_grouped_olora(gpu: &mut Gpu) {
    const GROUPS: usize = 8;
    const M: usize = 1024;
    const K: usize = 4096;
    const WARMUPS: usize = 20;
    const TRIALS: usize = 100;

    let weights_host = build_e8_soa(GROUPS * M, K, 0x8e80);
    let row_bytes = weights_host.len() / (GROUPS * M);
    let weights = gpu
        .upload_raw(&weights_host, &[weights_host.len()])
        .expect("upload O-LoRA grouped bench weights");
    let x = gpu
        .upload_f32(&make_x(GROUPS * K, 0x8e81), &[GROUPS, K])
        .expect("upload O-LoRA grouped bench x");
    let y = gpu
        .upload_f32(&vec![0.0; GROUPS * M], &[GROUPS, M])
        .expect("upload O-LoRA grouped bench y");

    let direct = |gpu: &mut Gpu| {
        for group in 0..GROUPS {
            let w = weights.sub_offset(group * M * row_bytes, M * row_bytes);
            let xg = x.sub_offset(group * K, K);
            let yg = y.sub_offset(group * M, M);
            gpu.gemv_mfp4g32_e8_soa(&w, &xg, &yg, M, K)
                .expect("direct O-LoRA bench launch");
        }
    };
    for _ in 0..WARMUPS {
        direct(gpu);
    }
    gpu.hip.device_synchronize().expect("direct O-LoRA warmup sync");
    let start = Instant::now();
    for _ in 0..TRIALS {
        direct(gpu);
    }
    gpu.hip.device_synchronize().expect("direct O-LoRA bench sync");
    let direct_us = start.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64;

    for _ in 0..WARMUPS {
        gpu.gemv_mfp4g32_e8_soa_grouped_gfx942(&weights, &x, &y, GROUPS, M, K)
            .expect("grouped O-LoRA warmup launch");
    }
    gpu.hip.device_synchronize().expect("grouped O-LoRA warmup sync");
    let start = Instant::now();
    for _ in 0..TRIALS {
        gpu.gemv_mfp4g32_e8_soa_grouped_gfx942(&weights, &x, &y, GROUPS, M, K)
            .expect("grouped O-LoRA bench launch");
    }
    gpu.hip.device_synchronize().expect("grouped O-LoRA bench sync");
    let grouped_us = start.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64;
    let payload = weights_host.len() as f64;

    println!(
        "BENCH mfp4-e8 O-LoRA G={GROUPS} M={M} K={K}: direct8={direct_us:.3} us ({:.1} GB/s), grouped={grouped_us:.3} us ({:.1} GB/s), speedup={:.3}x",
        payload / direct_us / 1.0e3,
        payload / grouped_us / 1.0e3,
        direct_us / grouped_us,
    );
}

fn main() {
    const K: usize = 2048;
    const TOP_K: usize = 2;
    const EXPERTS: usize = 2;

    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.arch != "gfx942" {
        println!(
            "SKIP test_deepseek4_cdna_channels: detected {}, requires gfx942",
            gpu.arch
        );
        return;
    }
    println!("DeepSeek4 CDNA channel test on {}", gpu.arch);
    let f16_mfma = std::env::var("HIPFIRE_GFX942_MQ2_MFMA")
        .ok()
        .as_deref()
        == Some("2");

    let topk_indices_host = [1i32, 0i32];
    let topk_indices = gpu
        .upload_raw(bytes_of_i32(&topk_indices_host), &[TOP_K * 4])
        .expect("upload top-k indices");

    // Routed expert gate/up projection.
    const GATE_M: usize = 10;
    let gate_weights: Vec<_> = (0..EXPERTS)
        .map(|expert| build_mq2_expert(GATE_M, K, 0x1000 + expert as u64))
        .collect();
    let (_gate_experts, gate_ptrs) = upload_experts(&gpu, &gate_weights);
    let gate_x_host = make_x(K, 0x2000);
    let gate_x = gpu.upload_f32(&gate_x_host, &[K]).expect("upload gate x");
    let y_gate = gpu
        .upload_f32(&vec![0.0; TOP_K * GATE_M / 2], &[TOP_K * GATE_M / 2])
        .expect("upload y_gate");
    let y_up = gpu
        .upload_f32(&vec![0.0; TOP_K * GATE_M / 2], &[TOP_K * GATE_M / 2])
        .expect("upload y_up");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        &gate_ptrs,
        &topk_indices,
        &gate_x,
        &y_gate,
        &y_up,
        GATE_M,
        K,
        TOP_K,
    )
    .expect("MQ2 gate/up launch");
    gpu.hip.device_synchronize().expect("MQ2 gate/up sync");
    let mut gate_ref = vec![0.0f32; TOP_K * GATE_M / 2];
    let mut up_ref = gate_ref.clone();
    for rank in 0..TOP_K {
        let expert = topk_indices_host[rank] as usize;
        for row in 0..GATE_M {
            let value = mq2_row_dot(&gate_weights[expert], row, K, &gate_x_host);
            if row < GATE_M / 2 {
                gate_ref[rank * GATE_M / 2 + row] = value;
            } else {
                up_ref[rank * GATE_M / 2 + row - GATE_M / 2] = value;
            }
        }
    }
    assert_close(
        "mq2-lloyd gate",
        &gpu.download_f32(&y_gate).expect("download y_gate"),
        &gate_ref,
        if f16_mfma { 2.0e-3 } else { 2.0e-4 },
        if f16_mfma { 2.0e-4 } else { 2.0e-5 },
    );
    assert_close(
        "mq2-lloyd up",
        &gpu.download_f32(&y_up).expect("download y_up"),
        &up_ref,
        if f16_mfma { 2.0e-3 } else { 2.0e-4 },
        if f16_mfma { 2.0e-4 } else { 2.0e-5 },
    );

    // Routed expert down projection and scaled residual accumulation.
    const DOWN_M: usize = 9;
    let down_weights: Vec<_> = (0..EXPERTS)
        .map(|expert| build_mq2_expert(DOWN_M, K, 0x3000 + expert as u64))
        .collect();
    let (_down_experts, down_ptrs) = upload_experts(&gpu, &down_weights);
    let rot_host = make_x(TOP_K * K, 0x4000);
    let rot_batch = gpu
        .upload_f32(&rot_host, &[TOP_K, K])
        .expect("upload rot_batch");
    let topk_weights_host = [0.625f32, 0.375f32];
    let topk_weights = gpu
        .upload_f32(&topk_weights_host, &[TOP_K])
        .expect("upload top-k weights");
    let residual_host = make_x(DOWN_M, 0x5000);
    let residual = gpu
        .upload_f32(&residual_host, &[DOWN_M])
        .expect("upload residual");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
        &down_ptrs,
        &topk_indices,
        &topk_weights,
        &rot_batch,
        &residual,
        DOWN_M,
        K,
        TOP_K,
    )
    .expect("MQ2 down launch");
    gpu.hip.device_synchronize().expect("MQ2 down sync");
    let mut down_ref = residual_host.clone();
    for rank in 0..TOP_K {
        let expert = topk_indices_host[rank] as usize;
        let x = &rot_host[rank * K..(rank + 1) * K];
        for (row, value) in down_ref.iter_mut().enumerate() {
            *value += topk_weights_host[rank] * mq2_row_dot(&down_weights[expert], row, K, x);
        }
    }
    assert_close(
        "mq2-lloyd down",
        &gpu.download_f32(&residual).expect("download residual"),
        &down_ref,
        if f16_mfma { 3.0e-3 } else { 4.0e-4 },
        if f16_mfma { 3.0e-4 } else { 3.0e-5 },
    );

    // Deterministic shipping down path: expanded rank outputs followed by the
    // fixed-order weighted combine. The printed hashes are compared between
    // the portable and gfx942 wave64 processes for the byte-identity gate.
    let expanded = gpu
        .upload_f32(&vec![0.0; TOP_K * DOWN_M], &[TOP_K, DOWN_M])
        .expect("upload expanded outputs");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
        &down_ptrs,
        &topk_indices,
        &rot_batch,
        &expanded,
        DOWN_M,
        K,
        TOP_K,
        1,
    )
    .expect("MQ2 expanded down launch");
    gpu.hip.device_synchronize().expect("MQ2 expanded down sync");
    let expanded_got = gpu
        .download_f32(&expanded)
        .expect("download expanded outputs");
    let mut expanded_ref = vec![0.0f32; TOP_K * DOWN_M];
    for rank in 0..TOP_K {
        let expert = topk_indices_host[rank] as usize;
        let x = &rot_host[rank * K..(rank + 1) * K];
        for row in 0..DOWN_M {
            expanded_ref[rank * DOWN_M + row] =
                mq2_row_dot(&down_weights[expert], row, K, x);
        }
    }
    assert_close(
        "mq2-lloyd down-expanded",
        &expanded_got,
        &expanded_ref,
        3.0e-4,
        3.0e-5,
    );
    println!("HASH down-expanded {:016x}", hash_f32_bits(&expanded_got));

    let combined = gpu
        .upload_f32(&residual_host, &[DOWN_M])
        .expect("upload combined residual");
    gpu.moe_down_combine_k8_batched(
        &expanded,
        &topk_weights,
        &combined,
        DOWN_M,
        TOP_K,
        1,
    )
    .expect("MQ2 deterministic combine launch");
    gpu.hip
        .device_synchronize()
        .expect("MQ2 deterministic combine sync");
    let combined_got = gpu
        .download_f32(&combined)
        .expect("download combined residual");
    assert_close(
        "mq2-lloyd down-expanded-combine",
        &combined_got,
        &down_ref,
        4.0e-4,
        3.0e-5,
    );
    println!("HASH down-combined {:016x}", hash_f32_bits(&combined_got));

    // Dense MFP4G32E8SOA projection, including an odd row tail.
    const DENSE_M: usize = 7;
    let dense_weights_host = build_e8_soa(DENSE_M, K, 0x6000);
    let dense_weights = gpu
        .upload_raw(&dense_weights_host, &[dense_weights_host.len()])
        .expect("upload E8 SoA weights");
    let dense_x_host = make_x(K, 0x7000);
    let dense_x = gpu.upload_f32(&dense_x_host, &[K]).expect("upload dense x");
    let dense_y = gpu
        .upload_f32(&vec![0.0; DENSE_M], &[DENSE_M])
        .expect("upload dense y");
    gpu.gemv_mfp4g32_e8_soa(&dense_weights, &dense_x, &dense_y, DENSE_M, K)
        .expect("MFP4E8 SoA launch");
    gpu.hip.device_synchronize().expect("MFP4E8 SoA sync");
    let dense_ref: Vec<_> = (0..DENSE_M)
        .map(|row| e8_row_dot(&dense_weights_host, row, K, &dense_x_host))
        .collect();
    assert_close(
        "mfp4-e8-soa dense",
        &gpu.download_f32(&dense_y).expect("download dense y"),
        &dense_ref,
        3.0e-4,
        3.0e-5,
    );

    // Native gfx942 FP8 MFMA screen. This intentionally rounds activations to
    // FP8, so it is a tolerance/quality candidate rather than a byte gate.
    const FP8_M: usize = 17;
    let fp8_weights_host = build_e8_soa(FP8_M, K, 0x7050);
    let fp8_weights = gpu
        .upload_raw(&fp8_weights_host, &[fp8_weights_host.len()])
        .expect("upload FP8 MFMA E8 weights");
    let fp8_x_host = make_x(K, 0x7060);
    let fp8_x = gpu.upload_f32(&fp8_x_host, &[K]).expect("upload FP8 MFMA x");
    let fp8_y = gpu
        .upload_f32(&vec![0.0; FP8_M], &[FP8_M])
        .expect("upload FP8 MFMA y");
    gpu.gemv_mfp4g32_e8_soa_fp8_mfma_gfx942(&fp8_weights, &fp8_x, &fp8_y, FP8_M, K)
        .expect("MFP4E8 FP8 MFMA launch");
    gpu.hip.device_synchronize().expect("MFP4E8 FP8 MFMA sync");
    let fp8_ref: Vec<_> = (0..FP8_M)
        .map(|row| e8_row_dot(&fp8_weights_host, row, K, &fp8_x_host))
        .collect();
    assert_close(
        "mfp4-e8-soa fp8-mfma",
        &gpu.download_f32(&fp8_y).expect("download FP8 MFMA y"),
        &fp8_ref,
        0.30,
        0.035,
    );

    // Grouped O-LoRA route: one 2-D launch must preserve the direct gfx942
    // half-wave kernel's exact output for every group, including odd M.
    const DENSE_GROUPS: usize = 3;
    let grouped_weights_host = build_e8_soa(DENSE_GROUPS * DENSE_M, K, 0x7100);
    let grouped_weights = gpu
        .upload_raw(&grouped_weights_host, &[grouped_weights_host.len()])
        .expect("upload grouped E8 SoA weights");
    let grouped_x_host = make_x(DENSE_GROUPS * K, 0x7200);
    let grouped_x = gpu
        .upload_f32(&grouped_x_host, &[DENSE_GROUPS, K])
        .expect("upload grouped dense x");
    let grouped_direct = gpu
        .upload_f32(&vec![0.0; DENSE_GROUPS * DENSE_M], &[DENSE_GROUPS, DENSE_M])
        .expect("upload grouped direct y");
    let grouped_one_launch = gpu
        .upload_f32(&vec![0.0; DENSE_GROUPS * DENSE_M], &[DENSE_GROUPS, DENSE_M])
        .expect("upload grouped one-launch y");
    let row_bytes = grouped_weights_host.len() / (DENSE_GROUPS * DENSE_M);
    for group in 0..DENSE_GROUPS {
        let weight = grouped_weights.sub_offset(group * DENSE_M * row_bytes, DENSE_M * row_bytes);
        let x = grouped_x.sub_offset(group * K, K);
        let y = grouped_direct.sub_offset(group * DENSE_M, DENSE_M);
        gpu.gemv_mfp4g32_e8_soa(&weight, &x, &y, DENSE_M, K)
            .expect("direct grouped-control launch");
    }
    gpu.gemv_mfp4g32_e8_soa_grouped_gfx942(
        &grouped_weights,
        &grouped_x,
        &grouped_one_launch,
        DENSE_GROUPS,
        DENSE_M,
        K,
    )
    .expect("gfx942 grouped E8 launch");
    gpu.hip.device_synchronize().expect("grouped E8 sync");
    let grouped_direct_host = gpu
        .download_f32(&grouped_direct)
        .expect("download grouped direct y");
    let grouped_one_launch_host = gpu
        .download_f32(&grouped_one_launch)
        .expect("download grouped one-launch y");
    assert_eq!(
        hash_f32_bits(&grouped_one_launch_host),
        hash_f32_bits(&grouped_direct_host),
        "grouped gfx942 E8 output differs from direct launches"
    );
    assert!(
        grouped_one_launch_host
            .iter()
            .zip(&grouped_direct_host)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "grouped gfx942 E8 output is not bit-identical"
    );
    println!(
        "PASS mfp4-e8-soa grouped gfx942: bit-identical hash={:016x}",
        hash_f32_bits(&grouped_one_launch_host)
    );

    println!("PASS all DeepSeek4 gfx942 AR channel checks");
    if std::env::var_os("HIPFIRE_CHANNEL_BENCH").is_some() {
        bench_mq2_ar_shapes(&mut gpu);
    }
    if std::env::var_os("HIPFIRE_E8_GROUPED_BENCH").is_some() {
        bench_e8_grouped_olora(&mut gpu);
    }
}
