// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Comprehensive kernel test harness. Tests every dispatch path with synthetic data.
//! No model loading required — validates kernels independently.
//! Usage: cargo run --release --features deltanet --example test_kernels

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {} ({:.1} GB VRAM)", gpu.arch, {
        let (_, total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
        total as f64 / 1e9
    });

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    macro_rules! test {
        ($name:expr, $body:expr) => {{
            eprint!("  {:50} ", $name);
            let t = Instant::now();
            let mut closure = || -> Result<(), String> { $body };
            match closure() {
                Ok(()) => {
                    passed += 1;
                    eprintln!("OK ({:.1}ms)", t.elapsed().as_secs_f64() * 1000.0);
                }
                Err(e) => {
                    failed += 1;
                    eprintln!("FAIL: {e}");
                }
            }
        }};
    }

    macro_rules! skip {
        ($name:expr, $reason:expr) => {
            eprint!("  {:50} ", $name);
            skipped += 1;
            eprintln!("SKIP ({})", $reason);
        };
    }

    eprintln!("\n--- Basic ops ---");
    test!("alloc + free", {
        let t = gpu
            .alloc_tensor(&[1024], DType::F32)
            .map_err(|e| format!("{e}"))?;
        gpu.free_tensor(t).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("upload + download f32", {
        let data = vec![1.0f32; 256];
        let t = gpu.upload_f32(&data, &[256]).map_err(|e| format!("{e}"))?;
        let back = gpu.download_f32(&t).map_err(|e| format!("{e}"))?;
        assert_eq!(back.len(), 256);
        assert!((back[0] - 1.0).abs() < 1e-6, "got {}", back[0]);
        gpu.free_tensor(t).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("add_inplace_f32", {
        let a = gpu
            .upload_f32(&vec![1.0f32; 64], &[64])
            .map_err(|e| format!("{e}"))?;
        let b = gpu
            .upload_f32(&vec![2.0f32; 64], &[64])
            .map_err(|e| format!("{e}"))?;
        gpu.add_inplace_f32(&a, &b).map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&a).map_err(|e| format!("{e}"))?;
        assert!((r[0] - 3.0).abs() < 1e-6);
        gpu.free_tensor(a).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(b).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("rmsnorm_f32", {
        let x = gpu
            .upload_f32(&vec![1.0f32; 128], &[128])
            .map_err(|e| format!("{e}"))?;
        let w = gpu
            .upload_f32(&vec![1.0f32; 128], &[128])
            .map_err(|e| format!("{e}"))?;
        let o = gpu
            .alloc_tensor(&[128], DType::F32)
            .map_err(|e| format!("{e}"))?;
        gpu.rmsnorm_f32(&x, &w, &o, 1e-6)
            .map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&o).map_err(|e| format!("{e}"))?;
        assert!(r[0].is_finite(), "rmsnorm produced NaN");
        gpu.free_tensor(x).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(w).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(o).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("softmax_f32", {
        let x = gpu
            .upload_f32(&vec![1.0f32; 32], &[1, 32])
            .map_err(|e| format!("{e}"))?;
        gpu.softmax_f32(&x).map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&x).map_err(|e| format!("{e}"))?;
        let sum: f32 = r.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "softmax sum={sum}");
        gpu.free_tensor(x).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });

    test!("MoE down unscatter canonical slots", {
        let width = 19;
        let mut grouped_values: Vec<f32> = (0..8 * width)
            .map(|i| (i as f32 + 1.0) * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        grouped_values[0] = -0.0;
        let grouped = gpu
            .upload_f32(&grouped_values, &[8 * width])
            .map_err(|e| format!("{e}"))?;
        let output = gpu
            .upload_f32(&vec![12345.0; 6 * width], &[6 * width])
            .map_err(|e| format!("{e}"))?;
        let mut expected = vec![12345.0f32; 6 * width];
        // Different rank-local permutations must recover the canonical rows.
        // Reusing a dirty, oversized destination must touch only live slots.
        for permutation in [&[5u32, 1, 7, 0][..], &[2u32, 4][..]] {
            let indices: Vec<f32> = permutation.iter().copied().map(f32::from_bits).collect();
            let inverse = gpu
                .upload_f32(&indices, &[indices.len()])
                .map_err(|e| format!("{e}"))?;
            gpu.moe_down_unscatter_k8(&grouped, &inverse, &output, width, permutation.len())
                .map_err(|e| format!("{e}"))?;
            for (slot, &row) in permutation.iter().enumerate() {
                let row = row as usize;
                expected[slot * width..(slot + 1) * width]
                    .copy_from_slice(&grouped_values[row * width..(row + 1) * width]);
            }
            let actual = gpu.download_f32(&output).map_err(|e| format!("{e}"))?;
            for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                if actual.to_bits() != expected.to_bits() {
                    return Err(format!(
                        "canonical output element {i}: {actual:?} != {expected:?}"
                    ));
                }
            }
            gpu.free_tensor(inverse).map_err(|e| format!("{e}"))?;
        }
        gpu.free_tensor(grouped).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(output).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });

    eprintln!("\n--- Attention kernels ---");
    for (label, n_heads, n_kv, hd, seq) in [
        ("attention_f32 hd=128 h=8 kv=2", 8, 2, 128, 16),
        ("attention_f32 hd=256 h=16 kv=4", 16, 4, 256, 16),
        ("attention_f32 hd=256 h=10 kv=2", 10, 2, 256, 16),
    ] {
        test!(label, {
            let q = gpu
                .upload_f32(&vec![0.1f32; n_heads * hd], &[n_heads * hd])
                .map_err(|e| format!("{e}"))?;
            let kv_dim = n_kv * hd;
            let k = gpu
                .upload_f32(&vec![0.1f32; seq * kv_dim], &[seq * kv_dim])
                .map_err(|e| format!("{e}"))?;
            let v = gpu
                .upload_f32(&vec![0.1f32; seq * kv_dim], &[seq * kv_dim])
                .map_err(|e| format!("{e}"))?;
            let o = gpu
                .alloc_tensor(&[n_heads * hd], DType::F32)
                .map_err(|e| format!("{e}"))?;
            let pos_buf = gpu.hip.malloc(4).map_err(|e| format!("{e}"))?;
            let pos_val = (seq - 1) as i32;
            gpu.hip
                .memcpy_htod(&pos_buf, &pos_val.to_ne_bytes())
                .map_err(|e| format!("{e}"))?;
            gpu.attention_f32(&q, &k, &v, &o, &pos_buf, seq, n_heads, n_kv, hd, seq)
                .map_err(|e| format!("{e}"))?;
            let r = gpu.download_f32(&o).map_err(|e| format!("{e}"))?;
            assert!(r[0].is_finite(), "attention produced NaN");
            gpu.free_tensor(q).map_err(|e| format!("{e}"))?;
            gpu.free_tensor(k).map_err(|e| format!("{e}"))?;
            gpu.free_tensor(v).map_err(|e| format!("{e}"))?;
            gpu.free_tensor(o).map_err(|e| format!("{e}"))?;
            gpu.hip.free(pos_buf).map_err(|e| format!("{e}"))?;
            Ok::<(), String>(())
        });
    }

    eprintln!("\n--- Q8 KV kernels ---");
    for (label, n_kv, hd) in [
        ("q8 write+attn hd=128 kv=8", 8, 128),
        ("q8 write+attn hd=256 kv=4", 4, 256),
        ("q8 write+attn hd=256 kv=2", 2, 256),
    ] {
        test!(label, {
            let n_heads = n_kv * 4; // GQA ratio 4
            let seq = 8;
            let q8_blocks = hd / 32;
            let q8_bytes_per_pos = n_kv * q8_blocks * 34;
            let cache_bytes = seq * q8_bytes_per_pos;
            let cache_elems = (cache_bytes + 3) / 4;
            let k_cache = gpu
                .zeros(&[cache_elems], DType::F32)
                .map_err(|e| format!("{e}"))?;
            let v_cache = gpu
                .zeros(&[cache_elems], DType::F32)
                .map_err(|e| format!("{e}"))?;
            let pos_buf = gpu.hip.malloc(4).map_err(|e| format!("{e}"))?;

            // Write a few positions
            for p in 0..4 {
                let kv_data = gpu
                    .upload_f32(&vec![0.1f32; n_kv * hd], &[n_kv * hd])
                    .map_err(|e| format!("{e}"))?;
                let pv = p as i32;
                gpu.hip
                    .memcpy_htod(&pos_buf, &pv.to_ne_bytes())
                    .map_err(|e| format!("{e}"))?;
                gpu.kv_cache_write_q8_0(&k_cache, &kv_data, &pos_buf, n_kv, hd)
                    .map_err(|e| format!("{e}"))?;
                gpu.kv_cache_write_q8_0(&v_cache, &kv_data, &pos_buf, n_kv, hd)
                    .map_err(|e| format!("{e}"))?;
                gpu.free_tensor(kv_data).map_err(|e| format!("{e}"))?;
            }

            // Attention at pos 3
            let q = gpu
                .upload_f32(&vec![0.1f32; n_heads * hd], &[n_heads * hd])
                .map_err(|e| format!("{e}"))?;
            let o = gpu
                .alloc_tensor(&[n_heads * hd], DType::F32)
                .map_err(|e| format!("{e}"))?;
            let pv = 3i32;
            gpu.hip
                .memcpy_htod(&pos_buf, &pv.to_ne_bytes())
                .map_err(|e| format!("{e}"))?;
            gpu.attention_q8_0_kv(
                &q, &k_cache, &v_cache, &o, &pos_buf, 4, n_heads, n_kv, hd, seq,
            )
            .map_err(|e| format!("{e}"))?;
            let r = gpu.download_f32(&o).map_err(|e| format!("{e}"))?;
            assert!(
                r[0].is_finite(),
                "q8 attention produced NaN at r[0]={}",
                r[0]
            );
            gpu.free_tensor(q).map_err(|e| format!("{e}"))?;
            gpu.free_tensor(o).map_err(|e| format!("{e}"))?;
            gpu.free_tensor(k_cache).map_err(|e| format!("{e}"))?;
            gpu.free_tensor(v_cache).map_err(|e| format!("{e}"))?;
            gpu.hip.free(pos_buf).map_err(|e| format!("{e}"))?;
            Ok::<(), String>(())
        });
    }

    eprintln!("\n--- GDN (tiled LDS) ---");
    for (label, n_heads, hd) in [
        ("gdn_q8 h=32 hd=128 (9B DeltaNet)", 32, 128),
        ("gdn_q8 h=16 hd=128 (4B DeltaNet)", 16, 128),
    ] {
        test!(label, {
            let s_size = n_heads * hd * hd;
            let scale_size = n_heads * hd;
            let s_q8 = gpu
                .zeros(&[s_size], DType::F32)
                .map_err(|e| format!("{e}"))?; // int8 but alloc as bytes
            let s_scales = gpu
                .upload_f32(&vec![1.0f32; scale_size], &[scale_size])
                .map_err(|e| format!("{e}"))?;
            let q = gpu
                .upload_f32(&vec![0.01f32; n_heads * hd], &[n_heads * hd])
                .map_err(|e| format!("{e}"))?;
            let k = gpu
                .upload_f32(&vec![0.01f32; n_heads * hd], &[n_heads * hd])
                .map_err(|e| format!("{e}"))?;
            let v = gpu
                .upload_f32(&vec![0.01f32; n_heads * hd], &[n_heads * hd])
                .map_err(|e| format!("{e}"))?;
            let alpha = gpu
                .upload_f32(&vec![0.5f32; n_heads], &[n_heads])
                .map_err(|e| format!("{e}"))?;
            let beta = gpu
                .upload_f32(&vec![0.5f32; n_heads], &[n_heads])
                .map_err(|e| format!("{e}"))?;
            let o = gpu
                .alloc_tensor(&[n_heads * hd], DType::F32)
                .map_err(|e| format!("{e}"))?;
            gpu.gated_delta_net_q8(
                &q, &k, &v, &alpha, &beta, &s_q8, &s_scales, &o, 1, n_heads, hd, None,
            )
            .map_err(|e| format!("{e}"))?;
            let r = gpu.download_f32(&o).map_err(|e| format!("{e}"))?;
            assert!(r[0].is_finite(), "gdn produced NaN");
            for t in [q, k, v, alpha, beta, s_q8, s_scales, o] {
                gpu.free_tensor(t).map_err(|e| format!("{e}"))?;
            }
            Ok::<(), String>(())
        });
    }

    eprintln!("\n--- Vision encoder kernels ---");
    test!("gemm_f16 (vision GEMM)", {
        let m = 32;
        let k = 64;
        let n = 8;
        let w_data: Vec<u8> = vec![0; m * k * 2]; // F16 zeros
        let w = gpu
            .upload_raw(&w_data, &[w_data.len()])
            .map_err(|e| format!("{e}"))?;
        let x = gpu
            .upload_f32(&vec![1.0f32; n * k], &[n * k])
            .map_err(|e| format!("{e}"))?;
        let y = gpu
            .alloc_tensor(&[m * n], DType::F32)
            .map_err(|e| format!("{e}"))?;
        gpu.gemm_f16(&w, &x, &y, m, k, n)
            .map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&y).map_err(|e| format!("{e}"))?;
        assert_eq!(r.len(), m * n);
        assert!(r[0].is_finite());
        gpu.free_tensor(w).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(x).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(y).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("layernorm_batched", {
        let batch = 4;
        let dim = 64;
        let x = gpu
            .upload_f32(&vec![1.0f32; batch * dim], &[batch * dim])
            .map_err(|e| format!("{e}"))?;
        let w = gpu
            .upload_f32(&vec![1.0f32; dim], &[dim])
            .map_err(|e| format!("{e}"))?;
        let b = gpu
            .upload_f32(&vec![0.0f32; dim], &[dim])
            .map_err(|e| format!("{e}"))?;
        let o = gpu
            .alloc_tensor(&[batch * dim], DType::F32)
            .map_err(|e| format!("{e}"))?;
        gpu.layernorm_batched(&x, &w, &b, &o, batch, dim, 1e-6)
            .map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&o).map_err(|e| format!("{e}"))?;
        assert!(r[0].is_finite());
        gpu.free_tensor(x).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(w).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(b).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(o).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("transpose_f32", {
        let rows = 4;
        let cols = 8;
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let src = gpu
            .upload_f32(&data, &[rows * cols])
            .map_err(|e| format!("{e}"))?;
        let dst = gpu
            .alloc_tensor(&[rows * cols], DType::F32)
            .map_err(|e| format!("{e}"))?;
        gpu.transpose_f32(&src, &dst, rows, cols)
            .map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&dst).map_err(|e| format!("{e}"))?;
        // r[0] = data[0*cols+0]=0, r[1] = data[1*cols+0]=8, r[2] = data[2*cols+0]=16
        assert!(
            (r[1] - 8.0).abs() < 0.01,
            "transpose: r[1]={} expected 8",
            r[1]
        );
        gpu.free_tensor(src).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(dst).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });

    eprintln!("\n--- Qwen4 symmetric IU4 MoE (fn-moe-sym) ---");
    if gpu.arch == "gfx1151" {
        test!(
            "qwen4 sym IU4 QT53 K640 exact fold",
            qwen4_moe_sym_k640_reference(&mut gpu)
        );
    } else {
        skip!("qwen4 sym IU4 QT53 K640 exact fold", "gfx1151-only route");
    }

    eprintln!("\n--- Summary ---");
    eprintln!("  Passed:  {passed}");
    eprintln!("  Failed:  {failed}");
    eprintln!("  Skipped: {skipped}");
    if failed > 0 {
        std::process::exit(1);
    }
}

fn f16_bits_to_f32(b: u16) -> f32 {
    let sign = if b & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 10) & 0x1f) as i32;
    let m = (b & 0x3ff) as f32;
    sign * match e {
        0 => m * 2f32.powi(-24),
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The symmetric IU4 down route on one QT53 K640 expert: BF16 grouped rows
/// through the production FWHT128/A4 producer and grouped IU4 down, checked
/// byte-exact against the host fold `sum = fma(RN(sc*d), float(C_h), sum)`
/// over the produced sidecar (five epochs, nibble extremes 0/15, distinct
/// scales per epoch, padding slots +0, sentinel tile untouched), plus the
/// header checker refusing a one-ulp asymmetric QT53 header.
fn qwen4_moe_sym_k640_reference(gpu: &mut Gpu) -> Result<(), String> {
    const M: usize = 2560;
    const K: usize = 640;
    const LIVE: usize = 12;
    const ROWS: usize = 32; // tile 0: 12 live + 4 padding; tile 1: sentinel
    let err = |e: hip_bridge::HipError| e.to_string();
    // Weights: code 0 or 15 (w = -8 sc or 7 sc) by row/position parity,
    // header (sc, -8 sc) varied per row and epoch.
    let mut w = vec![0u8; M * 5 * 68];
    let mut scales = vec![0u16; M * 5];
    for row in 0..M {
        for h in 0..5 {
            let g = (row * 5 + h) * 68;
            let exp = 5 + ((row + 3 * h) % 6) as u16;
            let man = ((row * 37 + h * 101) % 1024) as u16;
            let sign = if (row + h) % 3 == 0 { 0x8000 } else { 0 };
            let sc = sign | (exp << 10) | man;
            let zp = (sign ^ 0x8000) | ((exp + 3) << 10) | man;
            scales[row * 5 + h] = sc;
            w[g..g + 2].copy_from_slice(&sc.to_le_bytes());
            w[g + 2..g + 4].copy_from_slice(&zp.to_le_bytes());
            for i in 0..64 {
                let lo = if (row + i) % 2 == 0 { 0x0 } else { 0xF };
                let hi = if (row + i + h) % 3 == 0 { 0x0 } else { 0xF };
                w[g + 4 + i] = lo | (hi << 4);
            }
        }
    }
    let wt = gpu.upload_raw(&w, &[w.len()]).map_err(err)?;
    let ptr = (wt.buf.as_ptr() as u64).to_le_bytes().repeat(512);
    let ptrs = gpu.upload_raw(&ptr, &[512 * 8]).map_err(err)?;
    if !gpu.qwen4_moe_sym_check(&ptrs, M, K, 512, 68).map_err(err)? {
        return Err("symmetric QT53 weights refused by the header check".into());
    }
    // One asymmetric header (zp off by one ulp) must be refused.
    let off = (1234 * 5 + 3) * 68 + 2;
    gpu.hip
        .memcpy_htod_offset(&wt.buf, off, &[w[off] ^ 1])
        .map_err(err)?;
    let refused = !gpu.qwen4_moe_sym_check(&ptrs, M, K, 512, 68).map_err(err)?;
    gpu.hip.memcpy_htod_offset(&wt.buf, off, &[w[off]]).map_err(err)?;
    if !refused {
        return Err("asymmetric QT53 header accepted by the header check".into());
    }
    // BF16 SwiGLU rows (exact small multiples of 1/4), grouped row p = slot p.
    let h: Vec<u8> = (0..ROWS * K)
        .flat_map(|i| {
            let v = (((i * 7 + (i / K) * 13) % 29) as f32 - 14.0) * 0.25;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();
    let ht = gpu.upload_raw(&h, &[h.len()]).map_err(err)?;
    let sorted: Vec<i32> = (0..ROWS as i32).map(|p| if p < LIVE as i32 { p } else { -1 }).collect();
    let sorted_b: Vec<u8> = sorted.iter().flat_map(|v| v.to_le_bytes()).collect();
    let st = gpu.upload_raw(&sorted_b, &[ROWS]).map_err(err)?;
    let tiles: Vec<u8> = [0i32, -1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let tt = gpu.upload_raw(&tiles, &[2]).map_err(err)?;
    let y = gpu.upload_raw(&vec![0x5Au8; ROWS * M * 2], &[ROWS * M * 2]).map_err(err)?;
    let xq = gpu.qwen4_moe_rotate128_i4(&ht, &st, K, ROWS, LIVE).map_err(err)?;
    gpu.gemm_qwen4_moe_down_iu4_sym(&ptrs, &tt, &st, &xq, &y, M, K, 1, ROWS, LIVE, false)
        .map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    let side = gpu.scratch.qwen4_moe_down_i4_scratch.as_ref().ok_or("no sidecar")?;
    let mut xb = vec![0u8; 5 * LIVE * 72];
    gpu.hip.memcpy_dtoh(&mut xb, side).map_err(err)?;
    let got = gpu.download_raw_bytes(&y).map_err(err)?;
    for p in 0..ROWS {
        for row in 0..M {
            let bits = u16::from_le_bytes([got[(p * M + row) * 2], got[(p * M + row) * 2 + 1]]);
            let want = if p >= 16 {
                0x5A5A // sentinel tile: untouched
            } else if p >= LIVE {
                0 // padding slot: +0
            } else {
                let mut sum = 0f32;
                for e in 0..5 {
                    let b = &xb[(e * LIVE + p) * 72..(e * LIVE + p + 1) * 72];
                    let d = f32::from_le_bytes(b[0..4].try_into().unwrap());
                    let g = (row * 5 + e) * 68;
                    let mut c = 0i32;
                    for i in 0..128 {
                        let wn = (w[g + 4 + i / 2] >> (4 * (i % 2))) & 15;
                        let xn = ((b[8 + i / 2] >> (4 * (i % 2))) << 4) as i8 >> 4;
                        c += (wn as i32 - 8) * xn as i32;
                    }
                    sum = (f16_bits_to_f32(scales[row * 5 + e]) * d).mul_add(c as f32, sum);
                }
                let u = sum.to_bits();
                ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
            };
            if bits != want {
                return Err(format!("slot {p} row {row}: {bits:#06x} != {want:#06x}"));
            }
        }
    }
    for t in [wt, ptrs, ht, st, tt, y] {
        gpu.free_tensor(t).map_err(err)?;
    }
    Ok(())
}
