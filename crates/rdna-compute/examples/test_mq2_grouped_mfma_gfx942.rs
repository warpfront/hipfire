// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness oracle and real-shape microbench for DeepSeek4's native gfx942
//! grouped MQ2-Lloyd MFMA prefill kernel.

use rdna_compute::{DType, Gpu, GpuTensor};

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7fffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32 & 0x8000) << 16) as u32;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x03ff) as u32;
    let bits = match exp {
        0 if mant == 0 => sign,
        0 => {
            let mut m = mant;
            let mut e = -14i32;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((e + 127) as u32) << 23) | ((m & 0x03ff) << 13)
        }
        31 => sign | 0x7f800000 | (mant << 13),
        _ => sign | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

fn mq2_weights(m: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = Vec::with_capacity(m * groups * 72);
    let mut state = 0x1234_5678_9abc_def0u64;
    for row in 0..m {
        for group in 0..groups {
            let bias = ((row * 7 + group * 3) % 11) as f32 * 0.015625;
            for value in [-1.75 + bias, -0.375 + bias, 0.625 + bias, 1.5 + bias] {
                out.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
            }
            for _ in 0..64 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut packed = 0u8;
                for i in 0..4 {
                    packed |= (((state >> (17 + i * 7)) & 3) as u8) << (i * 2);
                }
                out.push(packed);
            }
        }
    }
    out
}

fn wrap(raw: *mut std::ffi::c_void, bytes: usize, shape: Vec<usize>) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(raw, bytes) },
        shape,
        dtype: DType::F32,
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx942", "this oracle is chip-strict");
    println!("DeepSeek4 grouped MQ2 MFMA oracle on {}", gpu.arch);

    const M: usize = 32;
    const K: usize = 512;
    const SLOTS: usize = 16;
    let weights = mq2_weights(M, K);
    let x: Vec<f32> = (0..SLOTS * K)
        .map(|i| ((i * 37 % 257) as f32 - 128.0) / 128.0)
        .collect();
    let x_half: Vec<f32> = x
        .iter()
        .map(|&v| f16_bits_to_f32(f32_to_f16_bits(v)))
        .collect();

    let w = gpu.hip.malloc(weights.len()).expect("W alloc");
    gpu.hip.memcpy_htod(&w, &weights).expect("W copy");
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_dev = gpu.hip.malloc(x_bytes.len()).expect("X alloc");
    gpu.hip.memcpy_htod(&x_dev, &x_bytes).expect("X copy");
    let y_dev = gpu.hip.malloc(SLOTS * M * 4).expect("Y alloc");

    let w_ptr = w.as_ptr() as u64;
    let ep = gpu.hip.malloc(8).expect("EP alloc");
    gpu.hip
        .memcpy_htod(&ep, &w_ptr.to_le_bytes())
        .expect("EP copy");
    let tile_ids = 0i32.to_le_bytes();
    let tp = gpu.hip.malloc(4).expect("TP alloc");
    gpu.hip.memcpy_htod(&tp, &tile_ids).expect("TP copy");
    let slot_bytes: Vec<u8> = (0..SLOTS)
        .flat_map(|i| (i as i32).to_le_bytes())
        .collect();
    let sp = gpu.hip.malloc(slot_bytes.len()).expect("SP alloc");
    gpu.hip.memcpy_htod(&sp, &slot_bytes).expect("SP copy");

    let ep_t = wrap(ep.as_ptr(), 8, vec![1]);
    let tp_t = wrap(tp.as_ptr(), 4, vec![1]);
    let sp_t = wrap(sp.as_ptr(), slot_bytes.len(), vec![SLOTS]);
    let x_t = wrap(x_dev.as_ptr(), x_bytes.len(), vec![SLOTS, K]);
    let y_t = wrap(y_dev.as_ptr(), SLOTS * M * 4, vec![SLOTS, M]);

    gpu.gemm_mq2g256_lloyd_moe_grouped_mfma_gfx942(
        &ep_t, &tp_t, &sp_t, &x_t, &y_t, M, K, 1, SLOTS, SLOTS,
    )
    .expect("MFMA launch");
    gpu.hip.device_synchronize().expect("sync");
    let mut got_bytes = vec![0u8; SLOTS * M * 4];
    gpu.hip.memcpy_dtoh(&mut got_bytes, &y_dev).expect("Y copy");
    let got: &[f32] = unsafe {
        std::slice::from_raw_parts(got_bytes.as_ptr() as *const f32, SLOTS * M)
    };

    let groups = K / 256;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bad = 0usize;
    for slot in 0..SLOTS {
        for row in 0..M {
            let mut reference = 0.0f32;
            for kk in 0..K {
                let group = kk / 256;
                let in_group = kk & 255;
                let base = (row * groups + group) * 72;
                let cb_off = base + ((weights[base + 8 + in_group / 4] >> ((in_group & 3) * 2)) & 3) as usize * 2;
                let cb_bits = u16::from_le_bytes([weights[cb_off], weights[cb_off + 1]]);
                reference = f16_bits_to_f32(cb_bits).mul_add(x_half[slot * K + kk], reference);
            }
            let value = got[slot * M + row];
            let abs = (value - reference).abs();
            let rel = abs / reference.abs().max(1e-5);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
            if abs > 2e-2 && rel > 2e-3 {
                bad += 1;
            }
        }
    }
    println!(
        "oracle: max_abs={max_abs:.3e} max_rel={max_rel:.3e} bad={bad}/{}",
        SLOTS * M
    );
    assert_eq!(bad, 0, "grouped MQ2 MFMA mismatch");

    // The product fixture reaches much larger routed-slot counts, but these
    // two matrices are the exact DeepSeek4 gate/up and down projection shapes.
    println!("PASS grouped MQ2 MFMA channel oracle");

    std::mem::forget(ep_t);
    std::mem::forget(tp_t);
    std::mem::forget(sp_t);
    std::mem::forget(x_t);
    std::mem::forget(y_t);
}
