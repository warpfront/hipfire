// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Attention parity: the eager (GEMM-based) decomposition vs a host reference,
//! per head. Sibling of `vit_attention_qtiled_f32`'s parity harness in spirit —
//! a GPU kernel cannot be debugged by reading it.
//!
//! Geometry is small but exercises every axis: GQA (heads > kv_heads) so the
//! two KV-head groups are distinguishable, and a multiple of 64 in both n and
//! head_dim for the tiled GEMM's contract.
//!
//! Usage: cargo run --release -p hipfire-arch-qwen35-embed \
//!          --example attn_eager_parity --features lab

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let (n, heads, kvh, hd) = (128usize, 4usize, 2usize, 64usize);
    // n_valid < n exercises the masked tail exactly as production's padded
    // segments run it: rows [n_valid, n) are zeros in K/V and must not take
    // a softmax share.
    let n_valid = 96usize;
    let (qd, kvd) = (heads * hd, kvh * hd);
    let rep = heads / kvh;
    println!(
        "gpu {} | n={n} n_valid={n_valid} heads={heads} kv_heads={kvh} hd={hd}",
        gpu.arch
    );

    let mut s: u32 = 987654321;
    let mut frnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
    };
    let qf: Vec<f32> = (0..n * qd).map(|_| frnd()).collect();
    // Padding rows are zeros in K and V, as the batch layout zeroes them.
    let kf: Vec<f32> = (0..n * kvd)
        .map(|i| if i / kvd < n_valid { frnd() } else { 0.0 })
        .collect();
    let vf: Vec<f32> = (0..n * kvd)
        .map(|i| if i / kvd < n_valid { frnd() } else { 0.0 })
        .collect();
    // forward.rs folds the 1/sqrt(head_dim) into Q in f32 *before* the f16
    // cast (the GEMM has no scale operand). Mirror that exactly — the
    // reference below then takes the scores raw.
    let qscale = 1.0f32 / (hd as f32).sqrt();
    let qf: Vec<f32> = qf.iter().map(|x| x * qscale).collect();
    let q16: Vec<u16> = qf
        .iter()
        .map(|x| half::f16::from_f32(*x).to_bits())
        .collect();
    let k16: Vec<u16> = kf
        .iter()
        .map(|x| half::f16::from_f32(*x).to_bits())
        .collect();

    // ---- host reference (f32, from the same f16 Q/K the GPU sees) ----
    let qh: Vec<f32> = q16
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();
    let kh: Vec<f32> = k16
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();
    let mut ref_ctx = vec![0f32; n * qd];
    for h in 0..heads {
        let khd = h / rep;
        let mut scores = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n_valid {
                let mut acc = 0f32;
                for d in 0..hd {
                    acc += qh[i * qd + h * hd + d] * kh[j * kvd + khd * hd + d];
                }
                scores[i * n + j] = acc;
            }
        }
        // Masked softmax per row: keys [n_valid, n) get no share.
        for i in 0..n {
            let row = &mut scores[i * n..(i + 1) * n];
            let m = row[..n_valid].iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0f32;
            for x in row[..n_valid].iter_mut() {
                *x = (*x - m).exp();
                sum += *x;
            }
            for x in row[..n_valid].iter_mut() {
                *x /= sum;
            }
            for x in row[n_valid..].iter_mut() {
                *x = 0.0;
            }
        }
        for i in 0..n {
            for d in 0..hd {
                let mut acc = 0f32;
                for j in 0..n_valid {
                    acc += scores[i * n + j] * vf[j * kvd + khd * hd + d];
                }
                ref_ctx[i * qd + h * hd + d] = acc;
            }
        }
    }

    // ---- the production helpers: same tile dispatch as Enc::gemm_ld ----
    let gemm_production = |a: &rdna_compute::GpuTensor,
                           x: &rdna_compute::GpuTensor,
                           y: &rdna_compute::GpuTensor,
                           m: usize,
                           k: usize,
                           batch: usize,
                           lda: usize,
                           ldx: usize,
                           qk: bool,
                           gpu: &mut Gpu| {
        hipfire_arch_qwen35_embed::forward::attention_gemm_ld(
            gpu, a, x, y, m, k, batch, lda, ldx, qk,
        )
    };

    // ---- sanity: prove the GEMM's operand roles with an identity A ----
    // Y[b, m] = sum_k X[b, k] * A[m, k]. With A = I this must reproduce X, so
    // any mismatch here is an operand-mapping error, not attention math.
    {
        let (tn, tk) = (64usize, 64usize);
        let idn: Vec<u16> = (0..tn)
            .flat_map(|i| {
                (0..tk).map(move |j| {
                    if i == j {
                        half::f16::from_f32(1.0)
                    } else {
                        half::f16::from_f32(0.0)
                    }
                })
            })
            .map(|v| v.to_bits())
            .collect();
        let xn: Vec<u16> = (0..tn * tk)
            .map(|i| half::f16::from_f32((i % 7) as f32 * 0.125 - 0.5).to_bits())
            .collect();
        let xh: Vec<f32> = xn
            .iter()
            .map(|b| half::f16::from_bits(*b).to_f32())
            .collect();
        let a_dev = gpu
            .upload_f16_bits(&idn, &[tn, tk])
            .map_err(|e| format!("{e:?}"))?;
        let x_dev = gpu
            .upload_f16_bits(&xn, &[tn, tk])
            .map_err(|e| format!("{e:?}"))?;
        let y_dev = gpu
            .alloc_tensor(&[tn, tn], DType::F32)
            .map_err(|e| format!("{e:?}"))?;
        gemm_production(&a_dev, &x_dev, &y_dev, tn, tk, tn, tk, tk, true, &mut gpu)
            .map_err(|e| format!("identity gemm: {e:?}"))?;
        let y = gpu.download_f32(&y_dev).map_err(|e| format!("{e:?}"))?;
        let mut worst = 0f32;
        for i in 0..tn {
            for j in 0..tn {
                worst = worst.max((y[i * tn + j] - xh[i * tk + j]).abs());
            }
        }
        println!("identity check: Y[b,m]=sum_k X[b,k]A[m,k] with A=I -> max|Y-X| = {worst:.3e}");
        if y.iter().any(|v| !v.is_finite()) || worst > 1e-3 {
            return Err("identity GEMM disagrees with X * A^T".into());
        }
    }

    // ---- does a wider row pitch than k give the same product as packed? ----
    // The layout trick this design rests on: head `h` is addressed as a view
    // with `ldx`/`lda` wider than `k`, so no gather is needed. If that disagrees
    // with the packed operands, the trick is invalid and the heads must be
    // gathered into contiguous buffers instead.
    {
        let (tn, tk, pitch_a, pitch_x) = (64usize, 64usize, 128usize, 256usize);
        let a_full: Vec<u16> = (0..tn * pitch_a)
            .map(|i| half::f16::from_f32((i % 11) as f32 * 0.0625 - 0.3).to_bits())
            .collect();
        let x_full: Vec<u16> = (0..tn * pitch_x)
            .map(|i| half::f16::from_f32((i % 13) as f32 * 0.0625 - 0.4).to_bits())
            .collect();
        // packed: first tk columns of each row.
        let a_packed: Vec<u16> = (0..tn)
            .flat_map(|r| a_full[r * pitch_a..r * pitch_a + tk].iter().copied())
            .collect();
        let x_packed: Vec<u16> = (0..tn)
            .flat_map(|r| x_full[r * pitch_x..r * pitch_x + tk].iter().copied())
            .collect();
        let ap = gpu
            .upload_f16_bits(&a_packed, &[tn, tk])
            .map_err(|e| format!("{e:?}"))?;
        let xp = gpu
            .upload_f16_bits(&x_packed, &[tn, tk])
            .map_err(|e| format!("{e:?}"))?;
        let yp = gpu
            .alloc_tensor(&[tn, tn], DType::F32)
            .map_err(|e| format!("{e:?}"))?;
        gemm_production(&ap, &xp, &yp, tn, tk, tn, tk, tk, true, &mut gpu)
            .map_err(|e| format!("packed gemm: {e:?}"))?;
        let ref_y = gpu.download_f32(&yp).map_err(|e| format!("{e:?}"))?;
        // same data, but addressed through wider-pitch views.
        let af = gpu
            .upload_f16_bits(&a_full, &[tn, pitch_a])
            .map_err(|e| format!("{e:?}"))?;
        let xf = gpu
            .upload_f16_bits(&x_full, &[tn, pitch_x])
            .map_err(|e| format!("{e:?}"))?;
        let av = af.sub_offset(0, (tn - 1) * pitch_a + tk);
        let xv = xf.sub_offset(0, (tn - 1) * pitch_x + tk);
        let yv = gpu
            .alloc_tensor(&[tn, tn], DType::F32)
            .map_err(|e| format!("{e:?}"))?;
        gemm_production(&av, &xv, &yv, tn, tk, tn, pitch_a, pitch_x, true, &mut gpu)
            .map_err(|e| format!("pitched gemm: {e:?}"))?;
        let got_y = gpu.download_f32(&yv).map_err(|e| format!("{e:?}"))?;
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..tn * tn {
            num += ((got_y[i] - ref_y[i]) as f64).powi(2);
            den += (ref_y[i] as f64).powi(2);
        }
        println!(
            "pitched vs packed GEMM: rel_l2 = {:.3e}",
            (num / den).sqrt()
        );
        if let Some(i) = got_y
            .iter()
            .zip(&ref_y)
            .position(|(got, expected)| got.to_bits() != expected.to_bits())
        {
            return Err(format!(
                "pitched vs packed GEMM differs at {i}: {:08x} != {:08x}",
                got_y[i].to_bits(),
                ref_y[i].to_bits(),
            ));
        }
    }

    // ---- GPU: the eager sequence, exactly as forward.rs runs it ----
    let q_dev = gpu
        .upload_f16_bits(&q16, &[n, qd])
        .map_err(|e| format!("{e:?}"))?;
    let k_dev = gpu
        .upload_f16_bits(&k16, &[n, kvd])
        .map_err(|e| format!("{e:?}"))?;
    let v_dev = gpu
        .upload_f32(&vf, &[n * kvd])
        .map_err(|e| format!("{e:?}"))?;
    let sc = gpu
        .alloc_tensor(&[n, n], DType::F32)
        .map_err(|e| format!("{e:?}"))?;
    let p16 = gpu
        .alloc_tensor(&[n * n], DType::F16)
        .map_err(|e| format!("{e:?}"))?;
    let vh = gpu
        .alloc_tensor(&[n * hd], DType::F32)
        .map_err(|e| format!("{e:?}"))?;
    let vht = gpu
        .alloc_tensor(&[hd * n], DType::F32)
        .map_err(|e| format!("{e:?}"))?;
    let vht16 = gpu
        .alloc_tensor(&[hd * n], DType::F16)
        .map_err(|e| format!("{e:?}"))?;
    let o = gpu
        .alloc_tensor(&[n * hd], DType::F32)
        .map_err(|e| format!("{e:?}"))?;
    let ctx = gpu
        .alloc_tensor(&[n * qd], DType::F32)
        .map_err(|e| format!("{e:?}"))?;

    for h in 0..heads {
        let khd = h / rep;
        let q_h = q_dev.sub_offset(h * hd, (n - 1) * qd + hd);
        let k_h = k_dev.sub_offset(khd * hd, (n - 1) * kvd + hd);
        gemm_production(&k_h, &q_h, &sc, n, hd, n, kvd, qd, true, &mut gpu)
            .map_err(|e| format!("QK^T h{h}: {e:?}"))?;
        {
            // Localize: compare the raw QK^T scores before softmax.
            let gs = gpu.download_f32(&sc).map_err(|e| format!("{e:?}"))?;
            let (mut num, mut den) = (0f64, 0f64);
            for i in 0..n {
                for j in 0..n_valid {
                    let mut acc = 0f32;
                    for d in 0..hd {
                        acc += qh[i * qd + h * hd + d] * kh[j * kvd + (h / rep) * hd + d];
                    }
                    let want = acc;
                    let got = gs[i * n + j];
                    num += ((got - want) as f64).powi(2);
                    den += (want as f64).powi(2);
                }
            }
            println!(
                "  head {h} scores (pre-softmax): rel_l2 = {:.3e}",
                (num / den).sqrt()
            );
        }
        // Masked row softmax, fused f32->f16 — the production op, not a
        // softmax+cast pair (it is what actually runs over padded segments).
        gpu.attention_rows_softmax_f16(&sc, &p16, n, n, 0, n_valid)
            .map_err(|e| format!("softmax h{h}: {e:?}"))?;
        let v_h = v_dev.sub_offset(khd * hd, (n - 1) * kvd + hd);
        gpu.copy_rows_strided_f32(&v_h, &vh, n, hd, kvd, hd, 0, None)
            .map_err(|e| format!("V gather h{h}: {e:?}"))?;
        gpu.transpose_f32(&vh, &vht, n, hd)
            .map_err(|e| format!("V transpose h{h}: {e:?}"))?;
        gpu.cast_f32_to_f16(&vht, &vht16)
            .map_err(|e| format!("cast V h{h}: {e:?}"))?;
        gemm_production(&vht16, &p16, &o, hd, n, n, n, n, false, &mut gpu)
            .map_err(|e| format!("PV h{h}: {e:?}"))?;
        gpu.copy_rows_strided_f32(&o, &ctx, n, hd, hd, qd, h * hd, None)
            .map_err(|e| format!("O scatter h{h}: {e:?}"))?;
    }
    let got = gpu.download_f32(&ctx).map_err(|e| format!("{e:?}"))?;

    // ---- compare, per head so a localised fault shows up ----
    let mut worst = 0f32;
    for h in 0..heads {
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..n {
            for d in 0..hd {
                let a = got[i * qd + h * hd + d] as f64;
                let b = ref_ctx[i * qd + h * hd + d] as f64;
                num += (a - b) * (a - b);
                den += b * b;
            }
        }
        let r = (num / den).sqrt();
        if !r.is_finite() {
            return Err(format!("non-finite eager attention error for head {h}"));
        }
        worst = worst.max(r as f32);
        println!("  head {h} (kv head {}): rel_l2 = {r:.3e}", h / rep);
    }
    println!("\nworst head rel_l2 = {worst:.3e}  (f16 operands => ~1e-3 expected)");
    if worst > 5e-2 {
        return Err("eager attention disagrees with the host reference".into());
    }
    println!("PARITY OK");
    Ok(())
}
