// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Chunked FP32 gated_delta_net parity.
//!
//! Default (no env): F32 sequential-vs-chunked parity over several (T, CS)
//! configs. The retired C1.0 F32-chunked speed gate is no longer fatal.
//!
//! Build / run:
//!   cargo build --release -p rdna-compute --example gdn_chunk_parity \
//!     --features lab,deltanet
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-gdn \
//!     ./target/release/examples/gdn_chunk_parity

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("gdn_chunk_parity requires --features deltanet");
    std::process::exit(2);
}


#[cfg(feature = "deltanet")]
fn main() {
    use rdna_compute::DType;
    use std::time::Instant;

    const HD: usize = 128;
    const N_HEADS: usize = 48;
    const TOL: f32 = 1e-4;
    const BENCH_ITERS: usize = 21;

    let mut gpu = init_gpu();

    // (T, CS) configs — tails + production T=512 shapes.
    let configs: &[(usize, usize)] = &[
        (16, 16),
        (32, 16),
        (32, 32),
        (37, 16),
        (33, 8),
        (512, 16),
        (512, 32),
    ];

    let mut overall_ok = true;
    println!(
        "{:>4} {:>4} {:>6} {:>12} {:>12}  pass",
        "T", "CS", "heads", "out_max", "state_max"
    );

    for &(t_tokens, cs) in configs {
        let (q, k, v, gate, beta, s0) = gen_inputs(t_tokens, N_HEADS, HD, t_tokens, cs);

        let q_gpu = gpu.upload_f32(&q, &[t_tokens, N_HEADS * HD]).unwrap();
        let k_gpu = gpu.upload_f32(&k, &[t_tokens, N_HEADS * HD]).unwrap();
        let v_gpu = gpu.upload_f32(&v, &[t_tokens, N_HEADS * HD]).unwrap();
        let g_gpu = gpu.upload_f32(&gate, &[t_tokens, N_HEADS]).unwrap();
        let b_gpu = gpu.upload_f32(&beta, &[t_tokens, N_HEADS]).unwrap();

        let s_seq = gpu.upload_f32(&s0, &[N_HEADS * HD * HD]).unwrap();
        let s_chunk = gpu.upload_f32(&s0, &[N_HEADS * HD * HD]).unwrap();
        let out_seq = gpu.zeros(&[t_tokens, N_HEADS * HD], DType::F32).unwrap();
        let out_chunk = gpu.zeros(&[t_tokens, N_HEADS * HD], DType::F32).unwrap();

        gpu.gated_delta_net_f32_batch_seq(
            &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &s_seq, &out_seq, t_tokens, N_HEADS, HD,
        )
        .unwrap();

        gpu.gated_delta_net_f32_chunked(
            &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &s_chunk, &out_chunk, t_tokens, N_HEADS, HD, cs,
        )
        .unwrap();

        let out_seq_h = gpu.download_f32(&out_seq).unwrap();
        let out_chunk_h = gpu.download_f32(&out_chunk).unwrap();
        let s_seq_h = gpu.download_f32(&s_seq).unwrap();
        let s_chunk_h = gpu.download_f32(&s_chunk).unwrap();

        let out_max = max_abs_diff(&out_seq_h, &out_chunk_h);
        let s_max = max_abs_diff(&s_seq_h, &s_chunk_h);
        let ok = out_max < TOL && s_max < TOL;
        overall_ok &= ok;
        println!(
            "{:>4} {:>4} {:>6} {:>12.3e} {:>12.3e}  {}",
            t_tokens,
            cs,
            N_HEADS,
            out_max,
            s_max,
            if ok { "OK" } else { "FAIL" }
        );

        for tns in [
            q_gpu, k_gpu, v_gpu, g_gpu, b_gpu, s_seq, s_chunk, out_seq, out_chunk,
        ] {
            gpu.free_tensor(tns).unwrap();
        }
    }

    if !overall_ok {
        eprintln!("F32 sequential-vs-chunked parity failed (tol {TOL:.0e})");
        std::process::exit(1);
    }
    println!("\nF32 parity PASS");

    // Info-only shipping Q8 timing (retired C1 speed gate is non-fatal).
    let (q, k, v, gate, beta, s0) = gen_inputs(512, N_HEADS, HD, 512, 0);
    let (s_q8, s_scales, s_ef) = quantize_s_q8_ef(&s0, N_HEADS, HD);
    let q_gpu = gpu.upload_f32(&q, &[512, N_HEADS * HD]).unwrap();
    let k_gpu = gpu.upload_f32(&k, &[512, N_HEADS * HD]).unwrap();
    let v_gpu = gpu.upload_f32(&v, &[512, N_HEADS * HD]).unwrap();
    let g_gpu = gpu.upload_f32(&gate, &[512, N_HEADS]).unwrap();
    let b_gpu = gpu.upload_f32(&beta, &[512, N_HEADS]).unwrap();
    let out = gpu.zeros(&[512, N_HEADS * HD], DType::F32).unwrap();
    {
        let sq = upload_i8(&mut gpu, &s_q8, &[N_HEADS * HD * HD]);
        let sc = gpu.upload_f32(&s_scales, &[N_HEADS * HD]).unwrap();
        let ef = gpu.upload_f16_bits(&s_ef, &[N_HEADS * HD * HD]).unwrap();
        gpu.gated_delta_net_q8_batch_seq(
            &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &sq, &sc, &out, 512, N_HEADS, HD, Some(&ef),
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.free_tensor(sq).unwrap();
        gpu.free_tensor(sc).unwrap();
        gpu.free_tensor(ef).unwrap();
    }
    let mut q8_us = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let sq = upload_i8(&mut gpu, &s_q8, &[N_HEADS * HD * HD]);
        let sc = gpu.upload_f32(&s_scales, &[N_HEADS * HD]).unwrap();
        let ef = gpu.upload_f16_bits(&s_ef, &[N_HEADS * HD * HD]).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        gpu.gated_delta_net_q8_batch_seq(
            &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &sq, &sc, &out, 512, N_HEADS, HD, Some(&ef),
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();
        q8_us.push(t0.elapsed().as_secs_f64() * 1e6);
        gpu.free_tensor(sq).unwrap();
        gpu.free_tensor(sc).unwrap();
        gpu.free_tensor(ef).unwrap();
    }
    q8_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "\nQ8-fast shipping warm median us (T=512): {:>10.1}",
        q8_us[BENCH_ITERS / 2]
    );
    for tns in [q_gpu, k_gpu, v_gpu, g_gpu, b_gpu, out] {
        gpu.free_tensor(tns).unwrap();
    }
}


#[cfg(feature = "deltanet")]
fn gen_inputs(
    t_tokens: usize,
    n_heads: usize,
    hd: usize,
    seed_t: usize,
    seed_cs: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Lcg::new(0xC0FFEE ^ ((seed_t as u64) << 8) ^ seed_cs as u64);
    let mut q = vec![0f32; t_tokens * n_heads * hd];
    let mut k = vec![0f32; t_tokens * n_heads * hd];
    let mut v = vec![0f32; t_tokens * n_heads * hd];
    for x in q.iter_mut() {
        *x = rng.normal() * 0.5;
    }
    for x in k.iter_mut() {
        *x = rng.normal() * 0.5;
    }
    for x in v.iter_mut() {
        *x = rng.normal() * 0.5;
    }
    l2_normalize_rows(&mut q, t_tokens * n_heads, hd);
    l2_normalize_rows(&mut k, t_tokens * n_heads, hd);

    let mut gate = vec![0f32; t_tokens * n_heads];
    let mut beta = vec![0f32; t_tokens * n_heads];
    for g in gate.iter_mut() {
        *g = -rng.normal().abs() * 0.1 - 0.01; // log-decay < 0
    }
    for b in beta.iter_mut() {
        *b = rng.uniform(); // (0,1)
    }
    let s0: Vec<f32> = (0..n_heads * hd * hd).map(|_| rng.normal() * 0.1).collect();
    (q, k, v, gate, beta, s0)
}

/// Host-side match of `gated_delta_net_q8_fast` EF requant
/// (`kernels/src/gated_delta_net_q8_fast.hip` ~258-297): per-row absmax,
/// scale = max/127 (or 1), codes = rintf(clamp(s*inv, -128, 127)),
/// EF = s - qf*scale as f16 bits.
#[cfg(feature = "deltanet")]
fn quantize_s_q8_ef(s_f32: &[f32], n_heads: usize, hd: usize) -> (Vec<i8>, Vec<f32>, Vec<u16>) {
    let n_rows = n_heads * hd;
    assert_eq!(s_f32.len(), n_rows * hd);
    let mut codes = vec![0i8; n_rows * hd];
    let mut scales = vec![0f32; n_rows];
    let mut ef = vec![0u16; n_rows * hd];
    for row in 0..n_rows {
        let base = row * hd;
        let row_s = &s_f32[base..base + hd];
        let mut my_max = 0.0f32;
        for &x in row_s {
            my_max = my_max.max(x.abs());
        }
        let inv_s = if my_max > 0.0 { 127.0 / my_max } else { 0.0 };
        let scale = if my_max > 0.0 { my_max / 127.0 } else { 1.0 };
        scales[row] = scale;
        for c in 0..hd {
            let s = row_s[c];
            let qf = (s * inv_s).round().clamp(-128.0, 127.0);
            codes[base + c] = qf as i8;
            let resid = s - qf * scale;
            ef[base + c] = f32_to_f16_bits(resid);
        }
    }
    (codes, scales, ef)
}

#[cfg(feature = "deltanet")]
fn upload_i8(gpu: &mut rdna_compute::Gpu, data: &[i8], shape: &[usize]) -> rdna_compute::GpuTensor {
    let t = gpu.alloc_tensor(shape, rdna_compute::DType::Raw).unwrap();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}

/// Round-to-nearest f32→f16 bits (finite residuals only; matches common harnesses).
#[cfg(feature = "deltanet")]
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp_f32 == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let exp = exp_f32 - 127 + 15;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant_full = mant | 0x800000;
        let shift = 1 - exp;
        let rounded = (mant_full + (1 << (shift + 12))) >> (shift + 13);
        return sign | (rounded as u16);
    }
    let half_mant = mant >> 13;
    let round_bit = (mant >> 12) & 1;
    let sticky = mant & 0xfff;
    let mut hm = half_mant;
    if round_bit != 0 && (sticky != 0 || (half_mant & 1) != 0) {
        hm += 1;
    }
    if hm == 0x400 {
        // mantissa overflow into exp
        return sign | (((exp as u16) + 1) << 10);
    }
    sign | ((exp as u16) << 10) | (hm as u16)
}

#[cfg(feature = "deltanet")]
fn init_gpu() -> rdna_compute::Gpu {
    rdna_compute::Gpu::init().expect("GPU init")
}

#[cfg(feature = "deltanet")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(feature = "deltanet")]
fn l2_normalize_rows(data: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
        for x in row.iter_mut() {
            *x *= inv;
        }
    }
}

/// Tiny deterministic LCG + Box-Muller, so the harness needs no rand crate.
#[cfg(feature = "deltanet")]
struct Lcg {
    state: u64,
}

#[cfg(feature = "deltanet")]
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg {
            state: seed.wrapping_mul(0x2545F4914F6CDD1D) | 1,
        }
    }
    fn next_u32(&mut self) -> u32 {
        // LCG (Numerical Recipes) then take the high bits.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }
    fn uniform(&mut self) -> f32 {
        // (0,1) open-ish; +1 avoids exact 0.
        ((self.next_u32() as f64 + 1.0) / 4294967297.0) as f32
    }
    fn normal(&mut self) -> f32 {
        // Box-Muller using two uniforms.
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        let r = (-2.0f32 * u1.ln()).sqrt();
        r * (2.0f32 * std::f32::consts::PI * u2).cos()
    }
}
