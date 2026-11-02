// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Focused correctness probe for `vit_attention_qtiled_f32`: runs the new
//! kernel ALONE on a freshly-zeroed output buffer (no prior kernel writing
//! the same tensor, so a not-actually-running kernel shows up as all-zeros)
//! and diffs against the naive kernel on an identical input, plus a CPU
//! reference at small sizes.
//!
//! Run inside a ROCm container: /dev/kfd + /opt/rocm/lib.

use rdna_compute::{DType, Gpu};

const HIDDEN: usize = 1024;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;

fn cpu_reference(qkv: &[f32], n: usize) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let stride = 3 * HIDDEN;
    let mut out = vec![0.0f32; n * HIDDEN];
    for qi in 0..n {
        for h in 0..HEADS {
            let q_off = h * HEAD_DIM;
            let k_off = HIDDEN + h * HEAD_DIM;
            let v_off = 2 * HIDDEN + h * HEAD_DIM;
            let mut scores = vec![0.0f32; n];
            let mut m = f32::MIN;
            for j in 0..n {
                let mut dot = 0.0f32;
                for d in 0..HEAD_DIM {
                    dot += qkv[qi * stride + q_off + d] * qkv[j * stride + k_off + d];
                }
                scores[j] = dot * scale;
                m = m.max(scores[j]);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            for d in 0..HEAD_DIM {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += scores[j] * qkv[j * stride + v_off + d];
                }
                out[qi * HIDDEN + h * HEAD_DIM + d] = acc / sum;
            }
        }
    }
    out
}

fn summarize(name: &str, out: &[f32], reference: &[f32]) {
    let all_zero = out.iter().all(|&v| v == 0.0);
    let mut maxe = 0.0f32;
    let mut n_diff = 0usize;
    let mut first_bad: Option<usize> = None;
    for (i, (x, y)) in out.iter().zip(reference.iter()).enumerate() {
        let e = (x - y).abs();
        if e > 0.0 {
            n_diff += 1;
            if first_bad.is_none() {
                first_bad = Some(i);
            }
        }
        if e > maxe {
            maxe = e;
        }
    }
    println!(
        "{name}: all_zero={all_zero} n_diff={n_diff}/{} max_err={maxe:.3e} first_diff={first_bad:?}",
        reference.len()
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut seed: u64 = 0xdeadbeefcafe;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f32 / (1u64 << 53) as f32 - 1.0
    };

    for &n in &[97usize, 676, 2704] {
        println!("== N={n}");
        let stride = 3 * HIDDEN;
        let qkv_h: Vec<f32> = (0..n * stride).map(|_| rng()).collect();
        let qkv = gpu.upload_f32(&qkv_h, &[n * stride]).expect("upload");

        let out_n = gpu.alloc_tensor(&[n * HIDDEN], DType::F32).expect("alloc");
        gpu.vit_attention_f32(&qkv, &out_n, n, HIDDEN, HEADS, HEAD_DIM)
            .expect("naive launch");
        let naive_out = gpu.download_f32(&out_n).expect("download");

        // Fresh zeroed buffer, qtiled runs alone into it.
        let out_q_h = vec![0.0f32; n * HIDDEN];
        let out_q = gpu.upload_f32(&out_q_h, &[n * HIDDEN]).expect("alloc");
        gpu.vit_attention_qtiled_f32(&qkv, &out_q, n, HIDDEN, HEADS, HEAD_DIM)
            .expect("qtiled launch");
        let q_out = gpu.download_f32(&out_q).expect("download");

        summarize("qtiled vs naive", &q_out, &naive_out);
        if n <= 676 {
            let host_ref = cpu_reference(&qkv_h, n);
            summarize("qtiled vs cpu", &q_out, &host_ref);
            summarize("naive  vs cpu", &naive_out, &host_ref);
            // Forensics: show the raw bits at the first few positions where
            // naive differs from the CPU reference, alongside q and cpu.
            let mut shown = 0;
            for i in 0..naive_out.len().min(host_ref.len()) {
                if naive_out[i] != host_ref[i] {
                    println!(
                        "  [{i}] naive={:08x}({}) q={:08x}({}) cpu={:08x}({})",
                        naive_out[i].to_bits(),
                        naive_out[i],
                        q_out[i].to_bits(),
                        q_out[i],
                        host_ref[i].to_bits(),
                        host_ref[i]
                    );
                    shown += 1;
                    if shown >= 4 {
                        break;
                    }
                }
            }
        }
    }
}
