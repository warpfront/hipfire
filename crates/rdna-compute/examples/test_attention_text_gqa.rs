// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-reference parity test for `attention_t5_bias_f32`'s GQA + key-padding
//! mask extension (plan Task 14, prep for the Task 15 Qwen3 GPU encoder).
//!
//! Cases:
//!  1. GQA + causal + key-padding mask: n=37, heads=8, n_kv_heads=2, hd=16,
//!     the last 5 keys masked. This is the Qwen3 shape the kernel change
//!     targets — `kv_off` picks a K/V head narrower than `q`'s, and the
//!     masked tail must contribute exactly zero.
//!  2. n_kv_heads == heads (MHA), no mask, causal off, with an additive
//!     `[heads, n, n]` bias — the exact T5 call shape, to prove that path is
//!     numerically unchanged by this commit.
//!  3. GQA + causal + a key-padding mask over the FIRST 4 keys (indices
//!     0..3). Case 1's suffix mask never produces a fully-masked row: under
//!     causal attention every row's window `[0, qpos]` includes index 0,
//!     which case 1 never masks, so the kernel's row_max == -inf zero-guard
//!     is unexercised there — a broken guard (wrong `off`/`d` in the zero
//!     write) would pass case 1 undetected. A prefix mask forces rows
//!     0..3 fully masked (their causal window is a subset of the masked
//!     prefix), which both proves the CPU reference's zero-row convention
//!     and lets this case assert the GPU output for those rows is exactly
//!     0.0, not just close to the CPU reference's zero.
//!
//! CPU reference is a plain triple loop (the Qwen3 CPU attention loop from
//! `hipfire_arch_diffusion::qwen3::encode_taps`, generalized with an optional
//! additive bias). A fully-masked row (no visible key in the causal window)
//! leaves `out` untouched at its zero initializer — the same convention the
//! kernel's zero-guard implements. Inputs are a small deterministic LCG (no
//! crates). Tolerance 1e-4 on max abs diff; any case over tolerance, or any
//! non-exact-zero element in an asserted zero row, → exit 1.
//!
//! Build: `cargo run --release --example test_attention_text_gqa -p rdna-compute --features lab`

use rdna_compute::Gpu;

const TOL: f32 = 1e-4;

/// Deterministic LCG in [-1, 1), scaled by `scale`. No external crates.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32; // top 24 bits
        (((bits as f32) / (1u32 << 24) as f32) * 2.0 - 1.0) * scale
    }
}

fn fill(lcg: &mut Lcg, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| lcg.next_f32(scale)).collect()
}

/// Which keys a case masks (1.0 visible / 0.0 masked), or none.
#[derive(Clone, Copy)]
enum MaskKind {
    None,
    /// Mask the last `n` keys (case 1: causal windows never fully overlap
    /// this, since index 0 is always visible).
    LastN(usize),
    /// Mask the first `n` keys (case 3: forces every causal row whose
    /// window `[0, qpos]` sits entirely inside `0..n` to be fully masked).
    FirstN(usize),
}

impl MaskKind {
    fn build(self, n: usize) -> Option<Vec<f32>> {
        match self {
            MaskKind::None => None,
            MaskKind::LastN(k) => {
                let mut m = vec![1.0f32; n];
                for slot in m.iter_mut().skip(n - k) {
                    *slot = 0.0;
                }
                Some(m)
            }
            MaskKind::FirstN(k) => {
                let mut m = vec![1.0f32; n];
                for slot in m.iter_mut().take(k) {
                    *slot = 0.0;
                }
                Some(m)
            }
        }
    }
}

/// CPU reference: GQA causal/non-causal attention with optional additive
/// bias and optional key-padding mask. Structural mirror of
/// `qwen3::encode_taps`'s inner attention loop, generalized to accept a bias
/// term (T5's call shape) instead of assuming none.
#[allow(clippy::too_many_arguments)]
fn cpu_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias: Option<&[f32]>,
    mask: Option<&[f32]>,
    n: usize,
    heads: usize,
    n_kv_heads: usize,
    hd: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let d = heads * hd;
    let d_kv = n_kv_heads * hd;
    let group = heads / n_kv_heads;
    let mut out = vec![0.0f32; n * d];
    for h in 0..heads {
        let kh = h / group;
        for qp in 0..n {
            let kmax = if causal { qp + 1 } else { n };
            let mut scores = vec![f32::NEG_INFINITY; kmax];
            for (kp, score) in scores.iter_mut().enumerate() {
                if let Some(m) = mask {
                    if m[kp] == 0.0 {
                        continue;
                    }
                }
                let mut acc = 0.0f32;
                for t in 0..hd {
                    acc += q[qp * d + h * hd + t] * k[kp * d_kv + kh * hd + t];
                }
                acc *= scale;
                if let Some(b) = bias {
                    acc += b[(h * n + qp) * n + kp];
                }
                *score = acc;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if !m.is_finite() {
                continue; // fully masked row: output stays zero
            }
            let mut sum = 0.0f32;
            let probs: Vec<f32> = scores
                .iter()
                .map(|s| {
                    let e = (s - m).exp();
                    sum += e;
                    e
                })
                .collect();
            for (kp, &p_unnorm) in probs.iter().enumerate() {
                let p = p_unnorm / sum;
                if p == 0.0 {
                    continue;
                }
                for t in 0..hd {
                    out[qp * d + h * hd + t] += p * v[kp * d_kv + kh * hd + t];
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &mut Gpu,
    n: usize,
    heads: usize,
    n_kv_heads: usize,
    hd: usize,
    scale: f32,
    causal: bool,
    with_bias: bool,
    mask_kind: MaskKind,
    assert_zero_rows: Option<(usize, usize)>,
    seed: u64,
    label: &str,
) -> usize {
    let mut lcg = Lcg(seed);
    let d = heads * hd;
    let d_kv = n_kv_heads * hd;

    let q = fill(&mut lcg, n * d, 0.3);
    let k = fill(&mut lcg, n * d_kv, 0.3);
    let v = fill(&mut lcg, n * d_kv, 0.3);
    let bias = if with_bias {
        Some(fill(&mut lcg, heads * n * n, 0.1))
    } else {
        None
    };
    let mask = mask_kind.build(n);

    let want = cpu_attn(
        &q,
        &k,
        &v,
        bias.as_deref(),
        mask.as_deref(),
        n,
        heads,
        n_kv_heads,
        hd,
        scale,
        causal,
    );

    let g_q = gpu.upload_f32(&q, &[n, d]).unwrap();
    let g_k = gpu.upload_f32(&k, &[n, d_kv]).unwrap();
    let g_v = gpu.upload_f32(&v, &[n, d_kv]).unwrap();
    let g_bias = bias
        .as_ref()
        .map(|b| gpu.upload_f32(b, &[heads, n, n]).unwrap());
    let g_mask = mask.as_ref().map(|m| gpu.upload_f32(m, &[n]).unwrap());
    let g_out = gpu.upload_f32(&vec![0.0f32; n * d], &[n, d]).unwrap();

    gpu.attention_text_f32(
        &g_q,
        &g_k,
        &g_v,
        g_bias.as_ref(),
        g_mask.as_ref(),
        &g_out,
        n,
        heads,
        n_kv_heads,
        hd,
        scale,
        causal,
    )
    .unwrap();
    let got = gpu.download_f32(&g_out).unwrap();

    gpu.free_tensor(g_q).unwrap();
    gpu.free_tensor(g_k).unwrap();
    gpu.free_tensor(g_v).unwrap();
    if let Some(t) = g_bias {
        gpu.free_tensor(t).unwrap();
    }
    if let Some(t) = g_mask {
        gpu.free_tensor(t).unwrap();
    }
    gpu.free_tensor(g_out).unwrap();

    let mut max_abs: f32 = 0.0;
    for (a, b) in got.iter().zip(want.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }

    let mut zero_row_fails = 0usize;
    if let Some((row_lo, row_hi)) = assert_zero_rows {
        for qp in row_lo..row_hi {
            for h in 0..heads {
                for t in 0..hd {
                    let got_val = got[qp * d + h * hd + t];
                    if got_val != 0.0 {
                        zero_row_fails += 1;
                        if zero_row_fails <= 5 {
                            eprintln!(
                                "{label}: FAIL row {qp} head {h} lane {t} = {got_val} (want exactly 0.0)"
                            );
                        }
                    }
                }
            }
        }
    }

    if max_abs > TOL || zero_row_fails > 0 {
        eprintln!(
            "{label}: FAIL max_abs={max_abs:.3e} tol={TOL} zero_row_fails={zero_row_fails} n={n} heads={heads} n_kv_heads={n_kv_heads} hd={hd} causal={causal} bias={with_bias}"
        );
        1
    } else {
        println!("{label}: PASS max_abs={max_abs:.3e}");
        0
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let mut fails = 0;
    const N_CASES: usize = 3;

    // 1. GQA + causal + key-padding mask (the Qwen3 shape).
    fails += run_case(
        &mut gpu,
        37,
        8,
        2,
        16,
        1.0 / (16.0f32).sqrt(),
        true,
        false,
        MaskKind::LastN(5),
        None,
        0x51A7_1CE0_0000_0001,
        "gqa-causal-keymask",
    );

    // 2. MHA (n_kv_heads == heads), no mask, non-causal, with bias — the T5
    // call shape, to prove it is numerically unchanged.
    fails += run_case(
        &mut gpu,
        37,
        8,
        8,
        16,
        1.0,
        false,
        true,
        MaskKind::None,
        None,
        0x51A7_1CE0_0000_0002,
        "mha-bias-t5shape",
    );

    // 3. GQA + causal + a key-padding mask over the FIRST 4 keys: forces
    // rows 0..3 fully masked (their causal window is a subset of the masked
    // prefix), exercising the kernel's row_max == -inf zero-guard, which
    // case 1's suffix mask can never reach under causal attention.
    fails += run_case(
        &mut gpu,
        37,
        8,
        2,
        16,
        1.0 / (16.0f32).sqrt(),
        true,
        false,
        MaskKind::FirstN(4),
        Some((0, 4)),
        0x51A7_1CE0_0000_0003,
        "gqa-causal-fullymasked-prefix",
    );

    if fails > 0 {
        eprintln!("FAIL: {fails}/{N_CASES} subtests failed");
        std::process::exit(1);
    }
    println!("PASS: attention_text_f32 GQA + key-padding-mask parity vs CPU reference");
}
