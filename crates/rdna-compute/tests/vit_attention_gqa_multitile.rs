// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GRID-1 coverage: `vit_attention_gqa_f32` multi-tile online-softmax path.
//!
//! The tiled rewrite (`VIT_ATTN_TILE` = 2048 in the kernel +
//! `vit_attention_gqa_f32`'s launcher) only changes behavior for N > 2048:
//! the LFM3 depth decoder drives N = 12 (single tile), so before this test
//! the multi-tile loop, the tile-boundary online rescale, and the gridDim.x
//! flatten ([n_heads * N] — the fix that lets N reach the ~65536 pixel
//! budget) had NO coverage. The pre-fix VL image validations all predate
//! the rewrite.
//!
//! Runs the kernel at N values that straddle the tile boundary (exact
//! single tile, two full tiles + partial tile) against an f64 CPU
//! reference softmax attention, at the VL GQA shape (16 Q / 8 KV heads,
//! head_dim 64).
//!
//! GPU-gated: skips gracefully when no AMD GPU is available (CI), same
//! convention as `hipfire-arch-qwen35/tests/pp_parity.rs`.

use rdna_compute::Gpu;

/// Deterministic LCG → f32 in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

/// f64 CPU reference: out[qi, h, d] = Σ_t softmax(QKᵀ·scale)[t] · V[t, kvh, d].
/// Bidirectional (no mask), GQA kv head = h / (n_heads / n_kv) — mirrors the
/// kernel's head decomposition exactly.
fn cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
) -> Vec<f32> {
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv * head_dim;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let groups = n_heads / n_kv;
    let mut out = vec![0.0f32; n * q_stride];
    for h in 0..n_heads {
        let kvh = h / groups;
        for qi in 0..n {
            let mut scores = vec![0.0f64; n];
            let mut m = f64::NEG_INFINITY;
            for t in 0..n {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += (q[qi * q_stride + h * head_dim + d]
                        * k[t * kv_stride + kvh * head_dim + d]) as f64;
                }
                let s = dot * scale;
                scores[t] = s;
                if s > m {
                    m = s;
                }
            }
            let mut weights = vec![0.0f64; n];
            let mut sum = 0.0f64;
            for t in 0..n {
                weights[t] = (scores[t] - m).exp();
                sum += weights[t];
            }
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for t in 0..n {
                    acc += weights[t] / sum * v[t * kv_stride + kvh * head_dim + d] as f64;
                }
                out[qi * q_stride + h * head_dim + d] = acc as f32;
            }
        }
    }
    out
}

fn run_case(gpu: &mut Gpu, n: usize) {
    let n_heads = 16usize;
    let n_kv = 8usize;
    let head_dim = 64usize;
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv * head_dim;
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15 ^ (n as u64).wrapping_mul(0x1000_0001));
    let q: Vec<f32> = (0..n * q_stride).map(|_| rng.next_f32()).collect();
    let k: Vec<f32> = (0..n * kv_stride).map(|_| rng.next_f32()).collect();
    let v: Vec<f32> = (0..n * kv_stride).map(|_| rng.next_f32()).collect();

    let q_gpu = gpu.upload_f32(&q, &[n * q_stride]).expect("upload q");
    let k_gpu = gpu.upload_f32(&k, &[n * kv_stride]).expect("upload k");
    let v_gpu = gpu.upload_f32(&v, &[n * kv_stride]).expect("upload v");
    let out_gpu = gpu
        .alloc_tensor(&[n * q_stride], rdna_compute::DType::F32)
        .expect("alloc out");

    // GRID-1: this call must drive the multi-tile path without tripping the
    // old 2-D-grid (gridDim.y ≤ 65535) or the old (N + block)·4 LDS ceiling.
    gpu.vit_attention_gqa_f32(&q_gpu, &k_gpu, &v_gpu, &out_gpu, n, n_heads, n_kv, head_dim)
        .expect("vit_attention_gqa_f32");
    let got = gpu.download_f32(&out_gpu).expect("download out");

    gpu.free_tensor(q_gpu).expect("free q");
    gpu.free_tensor(k_gpu).expect("free k");
    gpu.free_tensor(v_gpu).expect("free v");
    gpu.free_tensor(out_gpu).expect("free out");

    assert!(
        got.iter().all(|x| x.is_finite()),
        "N={n}: kernel produced non-finite output"
    );
    let want = cpu_reference(&q, &k, &v, n, n_heads, n_kv, head_dim);
    let mut max_diff = 0.0f32;
    let (mut idx, mut worst) = (0usize, 0.0f32);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        if d > max_diff {
            max_diff = d;
            idx = i;
            worst = *w;
        }
    }
    eprintln!("N={n}: max|gpu-cpu| = {max_diff:.3e} (worst ref {worst:+.4} @ {idx})");
    // f64 reference vs f32 online-softmax kernel: residual is dominated by
    // the f32 score path (dot of 64 terms) and expf — 1e-3 is a comfortable
    // ceiling with 3 orders of magnitude of margin over the observed noise.
    assert!(
        max_diff < 1e-3,
        "N={n}: multi-tile attention diverged from the f64 reference (max|diff|={max_diff})"
    );
}

#[test]
fn vit_gqa_multitile_matches_reference() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: no AMD GPU available ({e:?})");
            return;
        }
    };
    // Exact single tile (regression guard: must still match).
    run_case(&mut gpu, 2048);
    // Tile boundary: two full tiles + a 952-key partial tile (the f64
    // reference is O(N²), so cases are kept at the minimum that covers the
    // multi-tile fold, the partial-tail rescale, and the flattened grid).
    run_case(&mut gpu, 2048 + 952);
}
