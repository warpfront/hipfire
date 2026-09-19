// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Q8 scan oracle: serial `gated_delta_net_q8_fast` (EF arm) vs the
//! chunked-scan `gated_delta_net_q8_scan_gfx1201` (WMMA + VALU legs) vs an
//! f64 host serial reference.
//!
//! Reports, per (T, CS) config: out max-abs + tail-p99 of scan-vs-serial,
//! Q8 state agreement (codes/scales/EF), f64 deviations of serial and scan
//! (gate: scan <= serial, both metrics), host-f64 conditioning guard
//! max rowsum |T| (must be <= 10), bit-determinism across repeat launches,
//! commit-format invariants, and info-only host-timed medians at H=48.
//!
//! Build / run (needs GPU):
//!   cargo build --release -p rdna-compute --example gdn_q8_scan_oracle \
//!     --features deltanet
//!   HOME=... ROCR_VISIBLE_DEVICES=1 HIPFIRE_KERNEL_CACHE=... \
//!     ./target/release/examples/gdn_q8_scan_oracle

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("requires --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
fn main() {
    use rdna_compute::DType;
    use std::time::Instant;

    const HD: usize = 128;
    const H: usize = 4;
    const TIMING_H: usize = 48;

    // NOTE: HIPFIRE_GFX12_GDN_SCAN_CS / _WMMA are read from the
    // process-start snapshot (ACTIVE_PROCESS_CONFIG): mid-process set_var
    // is invisible. Sweep variants via separate processes with the env
    // exported at launch. T is data-only and loops freely in-process.
    let cs = rdna_compute::norm::gdn_q8_scan_chunk_size();
    let wmma = rdna_compute::norm::gdn_q8_scan_wmma();
    let tol: f32 = std::env::var("HIPFIRE_GDN_SCAN_ORACLE_TOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-2);

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");

    let t_list: &[usize] = &[512, 100, 64];
    let mut overall_ok = true;
    println!(
        "scan leg: {} CS={} tol={:.1e}",
        if wmma { "wmma" } else { "valu" },
        cs,
        tol
    );
    println!(
        "{:>4} {:>3} {:>12} {:>12} {:>10} {:>10} {:>9}  notes",
        "T", "CS", "out_max", "out_p99", "dev_ser", "dev_scan", "cond"
    );

    for &t_tokens in t_list {
        let (q, k, v, gate, beta, s0) = gen_inputs(t_tokens, H, HD, t_tokens, cs);
        let (s_q8, s_scales, s_ef) = quantize_s_q8_ef(&s0, H, HD);

        let q_gpu = gpu.upload_f32(&q, &[t_tokens, H * HD]).unwrap();
        let k_gpu = gpu.upload_f32(&k, &[t_tokens, H * HD]).unwrap();
        let v_gpu = gpu.upload_f32(&v, &[t_tokens, H * HD]).unwrap();
        let g_gpu = gpu.upload_f32(&gate, &[t_tokens, H]).unwrap();
        let b_gpu = gpu.upload_f32(&beta, &[t_tokens, H]).unwrap();

        // Serial reference arm (EF).
        let out_s = gpu.zeros(&[t_tokens, H * HD], DType::F32).unwrap();
        let (sq_s, sc_s, ef_s) = dev_state(&mut gpu, &s_q8, &s_scales, &s_ef, H, HD);
        gpu.gated_delta_net_q8_batch_seq(
            &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &sq_s, &sc_s, &out_s,
            t_tokens, H, HD, Some(&ef_s),
        )
        .unwrap();

        // Scan arm (snapshot leg) + repeat for bit-determinism.
        let out_w = gpu.zeros(&[t_tokens, H * HD], DType::F32).unwrap();
        let (sq_w, sc_w, ef_w) = dev_state(&mut gpu, &s_q8, &s_scales, &s_ef, H, HD);
        gpu.gated_delta_net_q8_scan(
            &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &sq_w, &sc_w, &out_w,
            t_tokens, H, HD, &ef_w,
        )
        .unwrap();
        let out_w2 = gpu.zeros(&[t_tokens, H * HD], DType::F32).unwrap();
        let (sq_w2, sc_w2, ef_w2) = dev_state(&mut gpu, &s_q8, &s_scales, &s_ef, H, HD);
        gpu.gated_delta_net_q8_scan(
            &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &sq_w2, &sc_w2, &out_w2,
            t_tokens, H, HD, &ef_w2,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();

        let out_s_h = gpu.download_f32(&out_s).unwrap();
        let out_w_h = gpu.download_f32(&out_w).unwrap();
        let out_w2_h = gpu.download_f32(&out_w2).unwrap();
        let codes_s = download_i8(&gpu, &sq_s);
        let codes_w = download_i8(&gpu, &sq_w);
        let sc_s_h = gpu.download_f32(&sc_s).unwrap();
        let sc_w_h = gpu.download_f32(&sc_w).unwrap();
        let ef_s_h = download_u16(&gpu, &ef_s);
        let ef_w_h = download_u16(&gpu, &ef_w);

        // f64 host serial reference from the same dequantized S0.
        let s0_dq = dequant(&s_q8, &s_scales);
        let out_f64 = serial_f64(&q, &k, &v, &gate, &beta, &s0_dq, t_tokens, H, HD);
        let dev_s = max_abs_diff_f(&out_s_h, &out_f64);
        let dev_w = max_abs_diff_f(&out_w_h, &out_f64);
        let cond = max_t_rowsum(&k, &gate, &beta, t_tokens, H, HD, cs);

        let out_max = max_abs_diff(&out_s_h, &out_w_h);
        let out_p99 = tail_p99(&out_s_h, &out_w_h);
        let det = max_abs_diff(&out_w_h, &out_w2_h);
        let code_diff = frac_diff_i8(&codes_s, &codes_w);
        let sc_max = max_abs_diff(&sc_s_h, &sc_w_h);
        let ef_diff = frac_diff_u16(&ef_s_h, &ef_w_h);
        let commit_ok = commit_invariants(&codes_w, &sc_w_h, &ef_w_h, &out_w_h);

        let ok = out_max <= tol
            && det == 0.0
            && cond <= 10.0
            && commit_ok;
        overall_ok &= ok;
        println!(
            "{:>4} {:>3} {:>12.3e} {:>12.3e} {:>10.3e} {:>10.3e} {:>9.3}  {}",
            t_tokens,
            cs,
            out_max,
            out_p99,
            dev_s,
            dev_w,
            cond,
            if ok { "OK" } else { "FAIL" }
        );
        println!(
            "      det_max={:.3e} codes_diff={:.4} sc_max={:.3e} ef_diff={:.4} commit={}",
            det, code_diff, sc_max, ef_diff,
            if commit_ok { "ok" } else { "BAD" }
        );

        for tns in [q_gpu, k_gpu, v_gpu, g_gpu, b_gpu] {
            gpu.free_tensor(tns).unwrap();
        }
        for tns in [out_s, out_w, out_w2] {
            gpu.free_tensor(tns).unwrap();
        }
        for (a, b, c) in [(sq_s, sc_s, ef_s), (sq_w, sc_w, ef_w), (sq_w2, sc_w2, ef_w2)] {
            gpu.free_tensor(a).unwrap();
            gpu.free_tensor(b).unwrap();
            gpu.free_tensor(c).unwrap();
        }
    }

    // One-hot fixtures: q=k=e0, v=e1, gate=0, beta=1, S0=0.
    // Exact expectation: out dim1 all-ones; scan must be BIT-EXACT vs
    // serial (all values 0/1, exact in f16 and f32).
    for &t_tokens in &[8usize, 64] {
        let mut q = vec![0f32; t_tokens * H * HD];
        let mut k = vec![0f32; t_tokens * H * HD];
        let mut v = vec![0f32; t_tokens * H * HD];
        for t in 0..t_tokens {
            for h in 0..H {
                q[(t * H + h) * HD] = 1.0;
                k[(t * H + h) * HD] = 1.0;
                v[(t * H + h) * HD + 1] = 1.0;
            }
        }
        let gate = vec![0f32; t_tokens * H];
        let beta = vec![1f32; t_tokens * H];
        let s_q8 = vec![0i8; H * HD * HD];
        let s_scales = vec![1f32; H * HD];
        let s_ef = vec![0u16; H * HD * HD];

        let q_gpu = gpu.upload_f32(&q, &[t_tokens, H * HD]).unwrap();
        let k_gpu = gpu.upload_f32(&k, &[t_tokens, H * HD]).unwrap();
        let v_gpu = gpu.upload_f32(&v, &[t_tokens, H * HD]).unwrap();
        let g_gpu = gpu.upload_f32(&gate, &[t_tokens, H]).unwrap();
        let b_gpu = gpu.upload_f32(&beta, &[t_tokens, H]).unwrap();
        let out_s = gpu.zeros(&[t_tokens, H * HD], DType::F32).unwrap();
        let (sq_s, sc_s, ef_s) = dev_state(&mut gpu, &s_q8, &s_scales, &s_ef, H, HD);
        gpu.gated_delta_net_q8_batch_seq(
            &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &sq_s, &sc_s, &out_s,
            t_tokens, H, HD, Some(&ef_s),
        )
        .unwrap();
        let out_w = gpu.zeros(&[t_tokens, H * HD], DType::F32).unwrap();
        let (sq_w, sc_w, ef_w) = dev_state(&mut gpu, &s_q8, &s_scales, &s_ef, H, HD);
        gpu.gated_delta_net_q8_scan(
            &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &sq_w, &sc_w, &out_w,
            t_tokens, H, HD, &ef_w,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();
        let out_s_h = gpu.download_f32(&out_s).unwrap();
        let out_w_h = gpu.download_f32(&out_w).unwrap();
        let d = max_abs_diff(&out_s_h, &out_w_h);
        let traj_w: Vec<f32> = (0..t_tokens).map(|t| out_w_h[(t * H) * HD + 1]).collect();
        let traj_s: Vec<f32> = (0..t_tokens).map(|t| out_s_h[(t * H) * HD + 1]).collect();
        let ok = d == 0.0;
        overall_ok &= ok;
        println!(
            "onehot T={} CS={} max_abs={:.3e} {}",
            t_tokens, cs, d, if ok { "OK" } else { "FAIL" }
        );
        println!("   serial traj[1]: {:?}", traj_s);
        println!("   scan   traj[1]: {:?}", traj_w);
        for tns in [q_gpu, k_gpu, v_gpu, g_gpu, b_gpu, out_s, out_w] {
            gpu.free_tensor(tns).unwrap();
        }
        for (a, b, c) in [(sq_s, sc_s, ef_s), (sq_w, sc_w, ef_w)] {
            gpu.free_tensor(a).unwrap();
            gpu.free_tensor(b).unwrap();
            gpu.free_tensor(c).unwrap();
        }
    }

    // Info-only timing at production width (serial + snapshot scan leg).
    let t = 512;
    let (q, k, v, gate, beta, s0) = gen_inputs(t, TIMING_H, HD, t, 0);
    let (s_q8, s_scales, s_ef) = quantize_s_q8_ef(&s0, TIMING_H, HD);
    let q_gpu = gpu.upload_f32(&q, &[t, TIMING_H * HD]).unwrap();
    let k_gpu = gpu.upload_f32(&k, &[t, TIMING_H * HD]).unwrap();
    let v_gpu = gpu.upload_f32(&v, &[t, TIMING_H * HD]).unwrap();
    let g_gpu = gpu.upload_f32(&gate, &[t, TIMING_H]).unwrap();
    let b_gpu = gpu.upload_f32(&beta, &[t, TIMING_H]).unwrap();
    let out = gpu.zeros(&[t, TIMING_H * HD], DType::F32).unwrap();
    let time_arm = |gpu: &mut rdna_compute::Gpu, serial: bool| -> f64 {
        let mut us = Vec::with_capacity(21);
        for _ in 0..21 {
            let (sq, sc, ef) = dev_state(gpu, &s_q8, &s_scales, &s_ef, TIMING_H, HD);
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            if serial {
                gpu.gated_delta_net_q8_batch_seq(
                    &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &sq, &sc, &out,
                    t, TIMING_H, HD, Some(&ef),
                )
                .unwrap();
            } else {
                gpu.gated_delta_net_q8_scan(
                    &q_gpu, &k_gpu, &v_gpu, &g_gpu, &b_gpu, &sq, &sc, &out,
                    t, TIMING_H, HD, &ef,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            us.push(t0.elapsed().as_secs_f64() * 1e6);
            gpu.free_tensor(sq).unwrap();
            gpu.free_tensor(sc).unwrap();
            gpu.free_tensor(ef).unwrap();
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        us[10]
    };
    // Warmup compiles (untimed).
    time_arm(&mut gpu, true);
    let m_ser = time_arm(&mut gpu, true);
    let m_scan = time_arm(&mut gpu, false);
    println!(
        "\ninfo-only warm medians us/launch H=48 T=512: serial={:.1} scan_{}={:.1}",
        m_ser, if wmma { "wmma" } else { "valu" }, m_scan
    );

    if !overall_ok {
        eprintln!("Q8 SCAN ORACLE FAILED");
        std::process::exit(1);
    }
    println!("Q8 SCAN ORACLE PASS");
}

#[cfg(feature = "deltanet")]
type Dev3 = (rdna_compute::GpuTensor, rdna_compute::GpuTensor, rdna_compute::GpuTensor);

#[cfg(feature = "deltanet")]
fn dev_state(
    gpu: &mut rdna_compute::Gpu,
    s_q8: &[i8],
    s_scales: &[f32],
    s_ef: &[u16],
    n_heads: usize,
    hd: usize,
) -> Dev3 {
    let sq = upload_i8(gpu, s_q8, &[n_heads * hd * hd]);
    let sc = gpu.upload_f32(s_scales, &[n_heads * hd]).unwrap();
    let ef = gpu.upload_f16_bits(s_ef, &[n_heads * hd * hd]).unwrap();
    (sq, sc, ef)
}

#[cfg(feature = "deltanet")]
fn download_i8(gpu: &rdna_compute::Gpu, t: &rdna_compute::GpuTensor) -> Vec<i8> {
    let n: usize = t.shape.iter().product();
    let mut bytes = vec![0u8; n];
    gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).unwrap();
    bytes.into_iter().map(|b| b as i8).collect()
}

#[cfg(feature = "deltanet")]
fn download_u16(gpu: &rdna_compute::Gpu, t: &rdna_compute::GpuTensor) -> Vec<u16> {
    let n: usize = t.shape.iter().product();
    let mut bytes = vec![0u8; n * 2];
    gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).unwrap();
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect()
}

#[cfg(feature = "deltanet")]
fn dequant(codes: &[i8], scales: &[f32]) -> Vec<f64> {
    let hd = 128usize;
    let rows = scales.len();
    let mut out = vec![0f64; rows * hd];
    for r in 0..rows {
        for c in 0..hd {
            out[r * hd + c] = codes[r * hd + c] as f64 * scales[r] as f64;
        }
    }
    out
}

/// f64 serial GDN recurrence (alpha-inside-delta), one segment, no requant.
#[cfg(feature = "deltanet")]
fn serial_f64(
    q: &[f32], k: &[f32], v: &[f32], gate: &[f32], beta: &[f32], s0: &[f64],
    t: usize, h: usize, hd: usize,
) -> Vec<f32> {
    let mut s = s0.to_vec();
    let mut out = vec![0f32; t * h * hd];
    for head in 0..h {
        for tt in 0..t {
            let alpha = (gate[tt * h + head] as f64).exp();
            let bv = beta[tt * h + head] as f64;
            let qb = (tt * h + head) * hd;
            // kv[r], delta[r] over value rows.
            for r in 0..hd {
                let base = (head * hd + r) * hd;
                let mut kv = 0f64;
                for c in 0..hd {
                    kv += s[base + c] * k[qb + c] as f64;
                }
                let delta = (v[qb + r] as f64 - alpha * kv) * bv;
                for c in 0..hd {
                    s[base + c] = alpha * s[base + c] + k[qb + c] as f64 * delta;
                }
                let mut o = 0f64;
                for c in 0..hd {
                    o += s[base + c] * q[qb + c] as f64;
                }
                out[(tt * h + head) * hd + r] = o as f32;
            }
        }
    }
    out
}

/// Host-f64 conditioning guard: max over chunks/heads of rowsum |T| where
/// T = (I + diag(beta) A)^{-1}, A strict-lower KK Gram with decay mask.
#[cfg(feature = "deltanet")]
fn max_t_rowsum(
    k: &[f32], gate: &[f32], beta: &[f32], t: usize, h: usize, hd: usize, cs: usize,
) -> f64 {
    let mut worst = 0f64;
    for head in 0..h.min(2) {
        let mut c0 = 0;
        while c0 < t {
            let c = (t - c0).min(cs);
            let mut g = vec![0f64; c];
            let mut acc = 0f64;
            for i in 0..c {
                acc += gate[(c0 + i) * h + head] as f64;
                g[i] = acc;
            }
            // L strict-lower.
            let mut l = vec![0f64; c * c];
            for j in 0..c {
                for li in 0..j {
                    let mut kk = 0f64;
                    for d in 0..hd {
                        kk += k[((c0 + j) * h + head) * hd + d] as f64
                            * k[((c0 + li) * h + head) * hd + d] as f64;
                    }
                    l[j * c + li] =
                        beta[(c0 + j) * h + head] as f64 * (g[j] - g[li]).exp() * kk;
                }
            }
            // T via fwd-subst against identity.
            let mut tmat = vec![0f64; c * c];
            for i in 0..c {
                for j in 0..=i.min(c - 1) {
                    if j == i {
                        tmat[i * c + j] = 1.0;
                    } else if j < i {
                        let mut a = 0f64;
                        for li in j..i {
                            a += l[i * c + li] * tmat[li * c + j];
                        }
                        tmat[i * c + j] = -a;
                    }
                }
            }
            for i in 0..c {
                let rs: f64 = (0..c).map(|j| tmat[i * c + j].abs()).sum();
                worst = worst.max(rs);
            }
            c0 += c;
        }
    }
    worst
}

#[cfg(feature = "deltanet")]
fn commit_invariants(codes: &[i8], scales: &[f32], ef: &[u16], out: &[f32]) -> bool {
    let _ = codes; // i8 range is structural.
    scales.iter().all(|s| s.is_finite() && *s > 0.0)
        && ef.iter().all(|e| (e & 0x7c00) != 0x7c00)
        && out.iter().all(|o| o.is_finite())
}

#[cfg(feature = "deltanet")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[cfg(feature = "deltanet")]
fn max_abs_diff_f(a: &[f32], b: &[f32]) -> f32 {
    max_abs_diff(a, b)
}

#[cfg(feature = "deltanet")]
fn tail_p99(a: &[f32], b: &[f32]) -> f32 {
    let mut d: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).collect();
    d.sort_by(|x, y| x.partial_cmp(y).unwrap());
    d[(d.len() * 99 / 100).min(d.len() - 1)]
}

#[cfg(feature = "deltanet")]
fn frac_diff_i8(a: &[i8], b: &[i8]) -> f64 {
    let n = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    n as f64 / a.len() as f64
}

#[cfg(feature = "deltanet")]
fn frac_diff_u16(a: &[u16], b: &[u16]) -> f64 {
    let n = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    n as f64 / a.len() as f64
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
        *g = -rng.normal().abs() * 0.1 - 0.01;
    }
    for b in beta.iter_mut() {
        *b = rng.uniform();
    }
    let s0: Vec<f32> = (0..n_heads * hd * hd).map(|_| rng.normal() * 0.1).collect();
    (q, k, v, gate, beta, s0)
}

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
        return sign | (((exp as u16) + 1) << 10);
    }
    sign | ((exp as u16) << 10) | (hm as u16)
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
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }
    fn uniform(&mut self) -> f32 {
        ((self.next_u32() as f64 + 1.0) / 4294967297.0) as f32
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        let r = (-2.0f32 * u1.ln()).sqrt();
        r * (2.0f32 * std::f32::consts::PI * u2).cos()
    }
}
