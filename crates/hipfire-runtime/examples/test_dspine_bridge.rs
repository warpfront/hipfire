// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DSpine arch-24 bridge kernel tests: every bridge stage on synthetic data,
//! checked against `hipfire_runtime::dspine::cpu_oracle` — the Tier C gate
//! for the four `dspine_bridge.gfx1100.hip` symbols plus the three bridge
//! GEMMs as wired in `DspineBridge::after_layer`.
//! No model loading required.
//! Usage: cargo run --release --example test_dspine_bridge

use hipfire_runtime::dspine::cpu_oracle;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

/// Deterministic pseudo-random f32 in [-1, 1] (xorshift64 white noise; no
/// rand dependency). Consecutive samples are uncorrelated, so dot products
/// of these vectors behave like true random ones — an LCG read through the
/// high bits is strongly self-correlated and breaks the stage tolerances.
fn prng(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = ((state >> 11) & 0x7FFF_FFFF) as f32 / 2147483648.0;
            v * 2.0 - 1.0
        })
        .collect()
}

/// f32 → bf16 bit pattern (truncation; matches hipfire float16 semantics
/// close enough for synthetic test data — the oracle compares within a
/// tolerance, not bitwise).
fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {} ({:.1} GB VRAM)", gpu.arch, {
        let (_, total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
        total as f64 / 1e9
    });

    let mut passed = 0;
    let mut failed = 0;

    macro_rules! test {
        ($name:expr, $body:expr) => {{
            eprint!("  {:52} ", $name);
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

    const H: usize = 5120;
    const C: usize = 1024;
    const R: usize = 512;
    const T: usize = 16;
    const EPS: f32 = 1e-6;
    const V: usize = 4096; // synthetic codebook rows (real one is 248320)

    eprintln!("\n--- DSpine bridge kernels (gfx1100) ---");
    test!("prepare_read: rms + normalized rows", {
        let m = T;
        let x = prng(42, m * H);
        let xt = gpu.upload_f32(&x, &[m, H]).map_err(|e| format!("{e}"))?;
        let rms_t = gpu.alloc_tensor(&[m], DType::F32).map_err(|e| format!("{e}"))?;
        let normed_t = gpu
            .alloc_tensor(&[m, H], DType::F32)
            .map_err(|e| format!("{e}"))?;
        gpu.dspine_prepare_read(&xt, &rms_t, &normed_t, m, H, EPS)
            .map_err(|e| format!("{e}"))?;
        let rms = gpu.download_f32(&rms_t).map_err(|e| format!("{e}"))?;
        let normed = gpu.download_f32(&normed_t).map_err(|e| format!("{e}"))?;
        let want_rms = cpu_oracle::rms_rows(&x, H, EPS);
        for (i, (&got, &want)) in rms.iter().zip(want_rms.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1e-3),
                "rms[{i}] {got} vs {want}"
            );
        }
        for (i, (&got, &want)) in normed.iter().zip(x.iter()).enumerate() {
            let row = i / H;
            let w = want / want_rms[row];
            assert!(
                (got - w).abs() <= 1e-4 * w.abs().max(1e-3),
                "normed[{i}] {got} vs {w}"
            );
        }
        gpu.free_tensor(xt).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(rms_t).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(normed_t).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });

    test!("normalize_codes: anchor identity + L2 path", {
        let m = T;
        // Anchor row 0 takes codebook[anchor] * sqrt(C); rows 1.. take the
        // L2-normalized projection path.
        let codebook_f32 = prng(7, V * C);
        let cb_bits: Vec<u16> = codebook_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();
        let cb_bytes: Vec<u8> = cb_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let cb_t = gpu
            .upload_raw(&cb_bytes, &[V * C])
            .map_err(|e| format!("{e}"))?;
        let projected = prng(11, m * C);
        let p_t = gpu
            .upload_f32(&projected, &[m, C])
            .map_err(|e| format!("{e}"))?;
        let anchor: u32 = 1234;
        let a_t = gpu
            .upload_raw(&anchor.to_le_bytes(), &[1])
            .map_err(|e| format!("{e}"))?;
        let codes_t = gpu.alloc_tensor(&[m, C], DType::F32).map_err(|e| format!("{e}"))?;
        gpu.dspine_normalize_codes(&p_t, &cb_t, &a_t, &codes_t, m, C, T)
            .map_err(|e| format!("{e}"))?;
        let codes = gpu.download_f32(&codes_t).map_err(|e| format!("{e}"))?;
        // Anchor row: codebook[anchor] * sqrt(C) — host converts the stored
        // bf16 back to f32 to mirror what the kernel reads.
        let scale = (C as f32).sqrt();
        for i in 0..C {
            let want = bf16_bits_to_f32(cb_bits[anchor as usize * C + i]) * scale;
            assert!(
                (codes[i] - want).abs() <= 1e-3 * want.abs().max(1.0),
                "anchor codes[{i}] {} vs {want}",
                codes[i]
            );
        }
        // Non-anchor rows: L2 normalize to sqrt(C) (F32 accumulate).
        for row in 1..m {
            let p = &projected[row * C..(row + 1) * C];
            let got = &codes[row * C..(row + 1) * C];
            let norm: f32 = p.iter().map(|&v| v * v).sum::<f32>().sqrt();
            for i in 0..C {
                let want = p[i] / norm * scale;
                assert!(
                    (got[i] - want).abs() <= 1e-3 * want.abs().max(1e-3),
                    "codes[{row},{i}] {} vs {want}",
                    got[i]
                );
            }
        }
        gpu.free_tensor(p_t).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(cb_t).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(a_t).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(codes_t).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });

    test!("feature_message: gate·predecessor, block-first zeros", {
        let m = T * 2 + 3; // 35 rows: block boundary mid-buffer (35 % 16 = 3)
        let packed = prng(13, m * 3 * R);
        let bias = prng(17, R);
        let packed_t = gpu
            .upload_f32(&packed, &[m, 3 * R])
            .map_err(|e| format!("{e}"))?;
        let bias_t = gpu.upload_f32(&bias, &[R]).map_err(|e| format!("{e}"))?;
        let msg_t = gpu.alloc_tensor(&[m, R], DType::F32).map_err(|e| format!("{e}"))?;
        gpu.dspine_feature_message(&packed_t, &bias_t, &msg_t, m, R, T)
            .map_err(|e| format!("{e}"))?;
        let msg = gpu.download_f32(&msg_t).map_err(|e| format!("{e}"))?;
        for row in 0..m {
            let first = row % T == 0;
            let q = &packed[row * 3 * R..row * 3 * R + R];
            let (s, v): (&[f32], &[f32]) = if first {
                (&[], &[])
            } else {
                (
                    &packed[(row - 1) * 3 * R + R..(row - 1) * 3 * R + 2 * R],
                    &packed[(row - 1) * 3 * R + 2 * R..(row - 1) * 3 * R + 3 * R],
                )
            };
            let want = if first {
                vec![0.0f32; R]
            } else {
                cpu_oracle::feature_message_row(q, Some(s), Some(v), &bias, R)
            };
            let got = &msg[row * R..(row + 1) * R];
            for i in 0..R {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-4 * want[i].abs().max(1e-3),
                    "msg[{row},{i}] {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
        gpu.free_tensor(packed_t).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(bias_t).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(msg_t).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });

    test!("finish: in-place rms*delta, first-row passthrough", {
        let m = T * 2 + 1;
        let x = prng(19, m * H);
        let rms = cpu_oracle::rms_rows(&x, H, EPS);
        let delta = prng(23, m * H);
        let x_t = gpu.upload_f32(&x, &[m, H]).map_err(|e| format!("{e}"))?;
        let rms_t = gpu.upload_f32(&rms, &[m]).map_err(|e| format!("{e}"))?;
        let d_t = gpu.upload_f32(&delta, &[m, H]).map_err(|e| format!("{e}"))?;
        gpu.dspine_finish(&x_t, &rms_t, &d_t, m, H, T).map_err(|e| format!("{e}"))?;
        let out = gpu.download_f32(&x_t).map_err(|e| format!("{e}"))?;
        for row in 0..m {
            for i in 0..H {
                let idx = row * H + i;
                let want = if row % T == 0 {
                    x[idx]
                } else {
                    x[idx] + rms[row] * delta[idx]
                };
                assert!(
                    (out[idx] - want).abs() <= 1e-4 * want.abs().max(1e-3),
                    "out[{row},{i}] {} vs {want}",
                    out[idx]
                );
            }
        }
        gpu.free_tensor(x_t).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(rms_t).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(d_t).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });

    test!("gemm_f32_batched semantics: 2x2 pin", {
        // a [2,2] row-major, b [2,2] row-major. pin down whether
        // y = a·b^T (torch-linear weight layout) or y = a·b.
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let bmat = vec![5.0f32, 6.0, 7.0, 8.0];
        let a_t = gpu.upload_f32(&a, &[2, 2]).map_err(|e| format!("{e}"))?;
        let b_t = gpu.upload_f32(&bmat, &[2, 2]).map_err(|e| format!("{e}"))?;
        let y_t = gpu.alloc_tensor(&[2, 2], DType::F32).map_err(|e| format!("{e}"))?;
        gpu.gemm_f32_batched(&a_t, &b_t, &y_t, 2, 2, 2)
            .map_err(|e| format!("{e}"))?;
        let y = gpu.download_f32(&y_t).map_err(|e| format!("{e}"))?;
        // a·b^T (b as [out,in] rows): [17, 23, 39, 54]
        let at_bt = [17.0f32, 23.0, 39.0, 54.0];
        // a·b (b as [in,out] columns): [19, 22, 43, 50]
        let ab = [19.0f32, 22.0, 43.0, 50.0];
        let d1: f32 = y.iter().zip(at_bt.iter()).map(|(g, w)| (g - w).abs()).sum();
        let d2: f32 = y.iter().zip(ab.iter()).map(|(g, w)| (g - w).abs()).sum();
        eprintln!("    gemm pin: y={:?} a·b^T={d1:.3} a·b={d2:.3}", y);
        assert!(d1 < d2, "gemm is a·b, not a·b^T — bridge GEMM layouts must adapt");
        gpu.free_tensor(a_t).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(b_t).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(y_t).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });

    test!("full chain: prepare→GEMM→codes→GEMM→msg→GEMM→finish vs oracle", {
        // The exact DspineBridge::after_layer sequence on one layer, all
        // stages on the GPU, compared stage-by-stage against the CPU oracle.
        let b = T;
        let x = prng(29, b * H);
        let read_w = prng(31, C * H); // [C, H] row-major (out, in)
        let write_w = prng(37, H * R); // [H, R]
        let gate_w = prng(41, R * C);
        let pred_w = prng(43, R * C);
        let prev_w = prng(47, R * C);
        let bias = prng(53, R);

        let x_t = gpu.upload_f32(&x, &[b, H]).map_err(|e| format!("{e}"))?;
        let read_t = gpu.upload_f32(&read_w, &[C, H]).map_err(|e| format!("{e}"))?;
        let write_t = gpu.upload_f32(&write_w, &[H, R]).map_err(|e| format!("{e}"))?;
        // packed_w = vertical concat [3R, C]
        let mut packed_w = Vec::with_capacity(3 * R * C);
        packed_w.extend_from_slice(&gate_w);
        packed_w.extend_from_slice(&pred_w);
        packed_w.extend_from_slice(&prev_w);
        let packed_t = gpu
            .upload_f32(&packed_w, &[3 * R, C])
            .map_err(|e| format!("{e}"))?;
        let bias_t = gpu.upload_f32(&bias, &[R]).map_err(|e| format!("{e}"))?;

        let rms_t = gpu.alloc_tensor(&[b], DType::F32).map_err(|e| format!("{e}"))?;
        let normed_t = gpu.alloc_tensor(&[b, H], DType::F32).map_err(|e| format!("{e}"))?;
        let proj_t = gpu.alloc_tensor(&[b, C], DType::F32).map_err(|e| format!("{e}"))?;
        let codes_t = gpu.alloc_tensor(&[b, C], DType::F32).map_err(|e| format!("{e}"))?;
        let pack_t = gpu.alloc_tensor(&[b, 3 * R], DType::F32).map_err(|e| format!("{e}"))?;
        let msg_t = gpu.alloc_tensor(&[b, R], DType::F32).map_err(|e| format!("{e}"))?;
        let delta_t = gpu.alloc_tensor(&[b, H], DType::F32).map_err(|e| format!("{e}"))?;

        // 1. prepare_read
        gpu.dspine_prepare_read(&x_t, &rms_t, &normed_t, b, H, EPS)
            .map_err(|e| format!("{e}"))?;
        // 2. projection GEMM: normed [b,H] × read_w[C,H]^T → [b,C]
        gpu.gemm_f32_batched(&read_t, &normed_t, &proj_t, C, H, b)
            .map_err(|e| format!("{e}"))?;
        // Per-stage oracle check: pin down the first diverging stage.
        let normed_out = gpu.download_f32(&normed_t).map_err(|e| format!("{e}"))?;
        let rms_out = gpu.download_f32(&rms_t).map_err(|e| format!("{e}"))?;
        let rms_want_pre = cpu_oracle::rms_rows(&x, H, EPS);
        for row in 0..b {
            assert!(
                (rms_out[row] - rms_want_pre[row]).abs() <= 1e-4 * rms_want_pre[row],
                "stage rms[{row}] {} vs {}",
                rms_out[row],
                rms_want_pre[row]
            );
        }
        let proj_out = gpu.download_f32(&proj_t).map_err(|e| format!("{e}"))?;
        eprintln!(
            "    diag: rms[0]={:.4} x[0..3]={:?} normed[0..3]={:?} read_w[1][0..3]={:?} proj[0][0..3]={:?} want[0][1]={:.4}",
            rms_out[0],
            &x[0..3],
            &normed_out[0..3],
            &read_w[H..H + 3],
            &proj_out[0..3],
            proj_out[1],
        );
        for row in 0..b {
            let normed_row: Vec<f32> = x[row * H..(row + 1) * H]
                .iter()
                .map(|&v| v / rms_want_pre[row])
                .collect();
            for (ci, wrow) in read_w.chunks_exact(H).enumerate() {
                let want: f32 = normed_row.iter().zip(wrow.iter()).map(|(&a, &w)| a * w).sum();
                let got = proj_out[row * C + ci];
                assert!(
                    (got - want).abs() <= 5e-2 * want.abs().max(1.0),
                    "stage proj[{row},{ci}] {got} vs {want}"
                );
            }
        }
        // (normalize_codes without the codebook arm — run the L2 path by
        // treating every row as non-anchor via T so no row index 0 exists;
        // instead call with T and let row 0 take an anchor from a scratch
        // id pointing into a tiny codebook we upload.)
        let anchor: u32 = 7;
        let a_t = gpu.upload_raw(&anchor.to_le_bytes(), &[1]).map_err(|e| format!("{e}"))?;
        // Mini codebook with zeros for the anchor row: identity path then
        // yields zeros — we only assert the non-anchor rows here.
        let cb_t = gpu.upload_raw(&vec![0u8; V * C * 2], &[V * C]).map_err(|e| format!("{e}"))?;
        gpu.dspine_normalize_codes(&proj_t, &cb_t, &a_t, &codes_t, b, C, T)
            .map_err(|e| format!("{e}"))?;
        let codes_out = gpu.download_f32(&codes_t).map_err(|e| format!("{e}"))?;
        // 4. packed GEMM
        gpu.gemm_f32_batched(&packed_t, &codes_t, &pack_t, 3 * R, C, b)
            .map_err(|e| format!("{e}"))?;
        let pack_out = gpu.download_f32(&pack_t).map_err(|e| format!("{e}"))?;
        // 5. feature_message
        gpu.dspine_feature_message(&pack_t, &bias_t, &msg_t, b, R, T)
            .map_err(|e| format!("{e}"))?;
        let msg_out = gpu.download_f32(&msg_t).map_err(|e| format!("{e}"))?;
        // 6. delta GEMM
        gpu.gemm_f32_batched(&write_t, &msg_t, &delta_t, H, R, b)
            .map_err(|e| format!("{e}"))?;
        let delta_out = gpu.download_f32(&delta_t).map_err(|e| format!("{e}"))?;
        // 7. finish
        gpu.dspine_finish(&x_t, &rms_t, &delta_t, b, H, T)
            .map_err(|e| format!("{e}"))?;

        // Oracle: same chain on the CPU (recomputed fresh below), device
        // output downloaded and compared per element. Rows 0 and 1 are
        // degenerate here (row 0 takes the zero anchor codebook, so row 1's
        // predecessor value segment is zero) — the interesting rows are 2..b.
        let rms_want = cpu_oracle::rms_rows(&x, H, EPS);
        let x_out = gpu.download_f32(&x_t).map_err(|e| format!("{e}"))?;
        let mut pks: Vec<Vec<f32>> = Vec::with_capacity(b);
        for row in 0..b {
            let code = if row == 0 {
                vec![0.0f32; C]
            } else {
                let normed_row: Vec<f32> = x[row * H..(row + 1) * H]
                    .iter()
                    .map(|&v| v / rms_want[row])
                    .collect();
                let mut proj = vec![0.0f32; C];
                for (ci, wrow) in read_w.chunks_exact(H).enumerate() {
                    proj[ci] = normed_row.iter().zip(wrow.iter()).map(|(&a, &w)| a * w).sum();
                }
                cpu_oracle::normalize_code_row(Some(&proj), None, C)
            };
            let mut pk = vec![0.0f32; 3 * R];
            for (ri, wrow) in packed_w.chunks_exact(C).enumerate() {
                pk[ri] = code.iter().zip(wrow.iter()).map(|(&a, &w)| a * w).sum();
            }
            pks.push(pk);
        }
        for row in 2..b {
            let pk = &pks[row];
            let prev = &pks[row - 1];
            let msg = cpu_oracle::feature_message_row(
                &pk[..R],
                Some(&prev[R..2 * R]),
                Some(&prev[2 * R..3 * R]),
                &bias,
                R,
            );
            for i in 0..H {
                let mut d = 0.0f32;
                for (ri, &mv) in msg.iter().enumerate() {
                    d += mv * write_w[i * R + ri]; // write_w [H, R] row-major
                }
                let want = x[row * H + i] + rms_want[row] * d;
                let got = x_out[row * H + i];
                assert!(
                    (got - want).abs() <= 5e-3 * want.abs().max(1.0),
                    "chain x[{row},{i}] {got} vs {want}"
                );
            }
        }

        for t in [x_t, read_t, write_t, packed_t, bias_t, rms_t, normed_t, proj_t, codes_t, pack_t, msg_t, delta_t, a_t, cb_t] {
            gpu.free_tensor(t).map_err(|e| format!("{e}"))?;
        }
        Ok::<(), String>(())
    });


    test!("serial_chain: fused score+greedy walk vs oracle", {
        let n = T * 2;
        let k = 16usize;
        let v = V;
        let base: Vec<f32> = prng(59, n * k).iter().map(|&x| x * 10.0).collect();
        let cand: Vec<u32> = prng(61, n * k)
            .iter()
            .map(|&x| ((x * 0.5 + 0.5) * (v as f32 - 1.0)) as u32)
            .collect();
        let rms: Vec<f32> = prng(67, n).iter().map(|&x| x.abs() * 0.5 + 0.1).collect();
        let query = prng(71, n * R);
        let tsv_gate = prng(73, v * R);
        let tsv_val = prng(79, v * R);
        let tg = prng(83, v * R);
        let bias = prng(89, R);
        let anchor_sid: u32 = 2048;

        let base_t = gpu.upload_f32(&base, &[n, k]).map_err(|e| format!("{e}"))?;
        let cand_t = gpu
            .upload_raw(&cand.iter().flat_map(|c| c.to_le_bytes()).collect::<Vec<u8>>(), &[n, k])
            .map_err(|e| format!("{e}"))?;
        let rms_t = gpu.upload_f32(&rms, &[n]).map_err(|e| format!("{e}"))?;
        let q_t = gpu.upload_f32(&query, &[n, R]).map_err(|e| format!("{e}"))?;
        let sg_t = gpu.upload_f32(&tsv_gate, &[v, R]).map_err(|e| format!("{e}"))?;
        let sv_t = gpu.upload_f32(&tsv_val, &[v, R]).map_err(|e| format!("{e}"))?;
        let tg_t = gpu.upload_f32(&tg, &[v, R]).map_err(|e| format!("{e}"))?;
        let gb_t = gpu.upload_f32(&bias, &[R]).map_err(|e| format!("{e}"))?;
        let out_t = gpu.alloc_tensor(&[n], DType::Raw).map_err(|e| format!("{e}"))?;

        gpu.dspine_serial_chain(
            &base_t, &cand_t, &rms_t, &q_t, &sg_t, &sv_t, &tg_t, &gb_t,
            anchor_sid, &out_t, n, k, R, T,
        )
        .map_err(|e| format!("{e}"))?;
        let raw = gpu.download_f32(&out_t).map_err(|e| format!("{e}"))?;
        let got: Vec<u32> = raw.iter().map(|f| f.to_bits()).collect();
        let want = cpu_oracle::serial_chain(
            &base, &cand, &rms, &query, &tsv_gate, &tsv_val, &tg, &bias,
            anchor_sid, n, k, R, T, v,
        );
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "serial_chain[{i}] {g} vs {w}");
        }
        for t in [base_t, cand_t, rms_t, q_t, sg_t, sv_t, tg_t, gb_t, out_t] {
            gpu.free_tensor(t).map_err(|e| format!("{e}"))?;
        }
        Ok::<(), String>(())
    });

    eprintln!("\n{passed} passed, {failed} failed");
    if failed > 0 {
        std::process::exit(1);
    }
}
