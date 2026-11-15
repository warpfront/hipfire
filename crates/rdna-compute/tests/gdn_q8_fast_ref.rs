// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU regression for GitHub #1136 (Qwen3.5 on gfx942 / MI300X): the Q8-state
//! GDN recurrence `Gpu::gated_delta_net_q8` (default route: the lean
//! `kernels/src/gated_delta_net_q8_fast.hip`) against an f64 CPU reference.
//!
//! This test is also the compile regression for #1136 on gfx9 (gfx906 /
//! gfx94x): the kernel used `__builtin_amdgcn_permlanex16` under
//! `HIPFIRE_GDN_DPP_REDUCE`, a gfx10+ instruction, and `hipcc` rejected it on
//! gfx9 with `needs target feature gfx10-insts`, so the whole recurrence (and
//! every Qwen3.5 linear-attention layer) failed to build. A preprocessor guard
//! now keeps the `shfl_down` tree on gfx9. Merely reaching the launch below
//! proves the kernel compiles on the present arch; the numeric checks prove it
//! computes the right recurrence there.
//!
//! Layout (host passes `qk_ratio = 1`, `ef_residual = None`):
//!   q, k   [n_tokens, n_heads, 128]   (Q/K already expanded to n_heads)
//!   v      [n_tokens, n_heads, 128]
//!   gate, beta [n_tokens, n_heads]
//!   s_q8   int8 [n_heads, 128 (row r), 128 (col c)], s_scales f32 [n_heads, 128]
//!   output [n_tokens, n_heads, 128]
//! Per token / head, state `S[r][c]` (row r is the value/output index):
//!   alpha = exp(gate); kv_r = sum_c S[r][c] k[c];
//!   delta_r = (v[r] - alpha kv_r) * beta;
//!   S[r][c] = alpha S[r][c] + k[c] delta_r;  out[r] = sum_c S[r][c] q[c].
//! The state stays FP32 in LDS across the call; it is requantized once at the
//! END (the fast kernel; `dn_requant_per_token()` is false unless
//! `HIPFIRE_DN_REQUANT_PER_TOKEN` is set, which this test refuses). Without an
//! EF residual the requant is stochastic-rounded:
//!   scale_r = max|S[r]| / 127 (1.0 for an all-zero row), q = floor(S/scale + u),
//!   u in [0, 1), so |q*scale - S| < scale.
//! A zero initial state is s_q8 = 0, s_scales = 0 (scale 0 decodes to 0).
//!
//! Gates: outputs of all tokens finite, rel L2 <= 1e-4 and max abs err
//! <= 1e-4 * max(1, max|ref|); the final quantized state dequantizes to the
//! reference state within one quantization step per row, and each row scale
//! matches max|S_ref|/127.
//!
//! `#[ignore]`d: needs a GPU, no model files.
//!
//!   cargo test -p rdna-compute --release --features deltanet --test gdn_q8_fast_ref \
//!       -- --ignored --nocapture
#![cfg(feature = "deltanet")]

use rdna_compute::{DType, Gpu, GpuTensor};

const HD: usize = 128;
const N_HEADS: usize = 4;
const N_TOKENS: usize = 6;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [0, 1).
    fn f32_unit(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF;
        f32::from_bits(0x3F80_0000 | a) - 1.0
    }
    /// Uniform in [-1, 1).
    fn f32_signed(&mut self) -> f32 {
        self.f32_unit() * 2.0 - 1.0
    }
}

fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod f32");
    t
}

/// [n_tokens, n_heads, HD] vectors, each (token, head) row L2-normalized, then
/// multiplied by `scale`.
fn unit_vectors(rng: &mut Lcg, scale: f32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..N_TOKENS * N_HEADS * HD)
        .map(|_| rng.f32_signed())
        .collect();
    for row in v.chunks_mut(HD) {
        let norm = row
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();
        for x in row.iter_mut() {
            *x = (*x as f64 / norm * scale as f64) as f32;
        }
    }
    v
}

fn rel_l2(got: &[f64], want: &[f64]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(g, r)| (g - r) * (g - r)).sum();
    let den: f64 = want.iter().map(|r| r * r).sum();
    (num / den).sqrt()
}

#[test]
#[ignore]
fn gdn_q8_fast_matches_cpu_reference() {
    if let Some(v) = std::env::var_os("HIPFIRE_DN_REQUANT_PER_TOKEN") {
        let v = v.to_string_lossy();
        let v = v.trim();
        assert!(
            v.is_empty() || v == "0",
            "HIPFIRE_DN_REQUANT_PER_TOKEN={v} selects the per-token-requant kernel; \
             this test covers the fast kernel (unset it)"
        );
    }

    let mut gpu = Gpu::init().expect("Gpu::init");
    println!("arch={}", gpu.arch);

    let mut rng = Lcg(0x1136_0942);
    let q = unit_vectors(&mut rng, 1.0 / (HD as f32).sqrt());
    let k = unit_vectors(&mut rng, 1.0);
    let v: Vec<f32> = (0..N_TOKENS * N_HEADS * HD)
        .map(|_| rng.f32_signed())
        .collect();
    let gate: Vec<f32> = (0..N_TOKENS * N_HEADS)
        .map(|_| -1.0 + 0.95 * rng.f32_unit())
        .collect();
    let beta: Vec<f32> = (0..N_TOKENS * N_HEADS)
        .map(|_| 0.1 + 0.8 * rng.f32_unit())
        .collect();

    // f64 reference from a zero state; S[h][r][c].
    let mut state = vec![0.0f64; N_HEADS * HD * HD];
    let mut want = vec![0.0f64; N_TOKENS * N_HEADS * HD];
    for t in 0..N_TOKENS {
        for h in 0..N_HEADS {
            let off = (t * N_HEADS + h) * HD;
            let (qt, kt, vt) = (&q[off..off + HD], &k[off..off + HD], &v[off..off + HD]);
            let alpha = (gate[t * N_HEADS + h] as f64).exp();
            let b = beta[t * N_HEADS + h] as f64;
            for r in 0..HD {
                let row = &mut state[(h * HD + r) * HD..(h * HD + r + 1) * HD];
                let kv: f64 = row.iter().zip(kt).map(|(s, kc)| s * *kc as f64).sum();
                let delta = (vt[r] as f64 - alpha * kv) * b;
                let mut out = 0.0f64;
                for c in 0..HD {
                    row[c] = alpha * row[c] + kt[c] as f64 * delta;
                    out += row[c] * qt[c] as f64;
                }
                want[off + r] = out;
            }
        }
    }

    let q_t = upload_f32(&mut gpu, &q);
    let k_t = upload_f32(&mut gpu, &k);
    let v_t = upload_f32(&mut gpu, &v);
    let g_t = upload_f32(&mut gpu, &gate);
    let b_t = upload_f32(&mut gpu, &beta);
    let s_q8 = gpu
        .zeros(&[N_HEADS * HD * HD], DType::Raw)
        .expect("zeros s_q8");
    let s_scales = gpu
        .zeros(&[N_HEADS * HD], DType::F32)
        .expect("zeros s_scales");
    let out_t = upload_f32(&mut gpu, &vec![f32::NAN; N_TOKENS * N_HEADS * HD]);

    gpu.gated_delta_net_q8(
        &q_t, &k_t, &v_t, &g_t, &b_t, &s_q8, &s_scales, &out_t, N_TOKENS, N_HEADS, HD, None,
    )
    .expect("gated_delta_net_q8");

    let got: Vec<f64> = gpu
        .download_f32(&out_t)
        .expect("download output")
        .into_iter()
        .map(|x| x as f64)
        .collect();
    assert_eq!(got.len(), want.len(), "output length");

    let ref_max = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let tol = 1e-4 * ref_max.max(1.0);
    let mut max_err = 0.0f64;
    let mut first_bad: Option<usize> = None;
    for (i, (g, r)) in got.iter().zip(&want).enumerate() {
        let err = (g - r).abs();
        if (!g.is_finite() || err > tol) && first_bad.is_none() {
            first_bad = Some(i);
        }
        if g.is_finite() {
            max_err = max_err.max(err);
        }
    }
    let l2 = rel_l2(&got, &want);
    println!(
        "output: max_abs_err={max_err:.3e} (tol {tol:.3e}) rel_l2={l2:.3e} ref_max={ref_max:.3e}"
    );
    if let Some(i) = first_bad {
        panic!(
            "output: first bad index {i} (token {}, head {}, row {}): gpu={} ref={} tol={tol:.3e}",
            i / (N_HEADS * HD),
            (i / HD) % N_HEADS,
            i % HD,
            got[i],
            want[i]
        );
    }
    assert!(l2 <= 1e-4, "output rel_l2 {l2:.3e} > 1e-4");

    // Final state: dequantize and compare to the reference state.
    let mut codes = vec![0u8; N_HEADS * HD * HD];
    gpu.hip
        .memcpy_dtoh(&mut codes, &s_q8.buf)
        .expect("dtoh s_q8");
    let scales = gpu.download_f32(&s_scales).expect("download s_scales");
    assert_eq!(scales.len(), N_HEADS * HD, "scales length");

    let mut worst_ratio = 0.0f64; // max |deq - S_ref| / scale_row
    let mut worst_scale_rel = 0.0f64;
    for h in 0..N_HEADS {
        for r in 0..HD {
            let row = h * HD + r;
            let scale = scales[row] as f64;
            let s_ref = &state[row * HD..(row + 1) * HD];
            let row_max = s_ref.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(
                scale.is_finite() && scale > 0.0,
                "state head {h} row {r}: scale {scale} is not positive finite"
            );
            // scale = max|S| / 127 (the kernel's formula), up to f32 noise.
            let scale_rel = (scale - row_max / 127.0).abs() / (row_max / 127.0);
            worst_scale_rel = worst_scale_rel.max(scale_rel);
            assert!(
                scale_rel <= 1e-3,
                "state head {h} row {r}: scale {scale:.6e} vs max|S_ref|/127 {:.6e}",
                row_max / 127.0
            );
            for c in 0..HD {
                let deq = (codes[row * HD + c] as i8) as f64 * scale;
                let err = (deq - s_ref[c]).abs();
                // Stochastic rounding: error < 1 step; 1% slack for f32 state noise.
                let bound = 1.01 * scale + 1e-6;
                worst_ratio = worst_ratio.max(err / scale);
                assert!(
                    err <= bound,
                    "state head {h} row {r} col {c}: deq={deq:.6e} ref={:.6e} err={err:.3e} \
                     > bound {bound:.3e} (scale {scale:.3e})",
                    s_ref[c]
                );
            }
        }
    }
    println!(
        "state: worst |deq-ref|/scale={worst_ratio:.3} (bound 1.01), \
         worst scale rel err={worst_scale_rel:.3e}"
    );

    for t in [q_t, k_t, v_t, g_t, b_t, s_q8, s_scales, out_t] {
        gpu.free_tensor(t).expect("free_tensor");
    }
}
