// SPDX-License-Identifier: Apache-2.0
// Slice-B gate-2 device oracle (throwaway): FA2-KMODE8 Q0 output vs the
// native-fp8 scalar-batched reference on the SAME native K/V.
// Both kernels read one shared fp8 cache written by
// kv_cache_write_fp8_e4m3_batched; the only difference is fill-time f16
// rounding + f16 WMMA (FA2) vs f32 arithmetic (reference). Reports
// max-abs / mean-abs / tail-1% mean-abs — informational; KLD admits.
use rdna_compute::{DType, Gpu};

fn bytes_of_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn bytes_of_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn next_normal(&mut self) -> f32 {
        // Box-Muller from two uniforms
        let u1 = (self.next_u64() as f64 / u64::MAX as f64).max(1e-12);
        let u2 = self.next_u64() as f64 / u64::MAX as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("=== SKIP === exact gfx1201 only (arch={})", gpu.arch);
        return;
    }
    const NH: usize = 24;
    const NKV: usize = 4;
    const HD: usize = 256;
    const BATCH: usize = 64;
    const CAP: usize = 128;
    const ROW: usize = NKV * (HD + 2); // 1032
    let scale_attn = 1.0f32 / (HD as f32).sqrt();

    let mut rng = Lcg(0x12345678);
    // Q/K/V f32 with model-like magnitudes (N(0,1)); K/V amax over 256
    // lands ~3 so scales/codes span the normal e4m3 range.
    let q_host: Vec<f32> = (0..BATCH * NH * HD).map(|_| rng.next_normal()).collect();
    let k_host: Vec<f32> = (0..BATCH * NKV * HD)
        .map(|_| rng.next_normal() * 1.5)
        .collect();
    let v_host: Vec<f32> = (0..BATCH * NKV * HD)
        .map(|_| rng.next_normal() * 0.7)
        .collect();
    let pos_host: Vec<i32> = (0..BATCH as i32).collect();

    let q = gpu.alloc_tensor(&[BATCH * NH * HD], DType::F32).unwrap();
    let k = gpu.alloc_tensor(&[BATCH * NKV * HD], DType::F32).unwrap();
    let v = gpu.alloc_tensor(&[BATCH * NKV * HD], DType::F32).unwrap();
    let positions = gpu.alloc_tensor(&[BATCH], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&q.buf, bytes_of_f32(&q_host)).unwrap();
    gpu.hip.memcpy_htod(&k.buf, bytes_of_f32(&k_host)).unwrap();
    gpu.hip.memcpy_htod(&v.buf, bytes_of_f32(&v_host)).unwrap();
    gpu.hip.memcpy_htod(&positions.buf, bytes_of_i32(&pos_host)).unwrap();

    let k_cache = gpu.alloc_tensor(&[CAP * ROW], DType::Raw).unwrap();
    let v_cache = gpu.alloc_tensor(&[CAP * ROW], DType::Raw).unwrap();
    gpu.kv_cache_write_fp8_e4m3_batched(&k_cache, &k, &positions, NKV, HD, BATCH)
        .unwrap();
    gpu.kv_cache_write_fp8_e4m3_batched(&v_cache, &v, &positions, NKV, HD, BATCH)
        .unwrap();

    let out_ref = gpu.alloc_tensor(&[BATCH * NH * HD], DType::F32).unwrap();
    let out_fa2 = gpu.alloc_tensor(&[BATCH * NH * HD], DType::F32).unwrap();
    gpu.attention_fp8_e4m3_kv_batched(
        &q, &k_cache, &v_cache, &out_ref, &positions, NH, NKV, HD, CAP, BATCH, BATCH, None, 0, 0,
    )
    .unwrap();
    gpu.attention_fp8_e4m3_fa2_gqa_f16_gfx1201(
        &q, &k_cache, &v_cache, &out_fa2, &positions, NH, NKV, HD, BATCH, BATCH,
    )
    .unwrap();

    fn as_bytes_mut(v: &mut [f32]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
    }
    let mut hr = vec![0f32; BATCH * NH * HD];
    let mut hf = vec![0f32; BATCH * NH * HD];
    gpu.hip.memcpy_dtoh(as_bytes_mut(&mut hr), &out_ref.buf).unwrap();
    gpu.hip.memcpy_dtoh(as_bytes_mut(&mut hf), &out_fa2.buf).unwrap();

    let n = hr.len();
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut pairs: Vec<(f32, f32)> = hr.iter().zip(hf.iter()).map(|(&a, &b)| (a, (a - b).abs())).collect();
    for &(_, d) in &pairs {
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    pairs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap());
    let tail_n = (n / 100).max(1);
    let tail_sum: f64 = pairs[..tail_n].iter().map(|&(_, d)| d as f64).sum();
    // max relative where |ref| > 1e-3
    let mut max_rel = 0f32;
    for (&a, &b) in hr.iter().zip(hf.iter()) {
        if a.abs() > 1e-3 {
            max_rel = max_rel.max((a - b).abs() / a.abs());
        }
    }
    let _ = scale_attn;
    println!("fa2q0-vs-tiledref n={n} batch={BATCH}");
    println!("max_abs={max_abs:.6e} mean_abs={:.6e}", sum_abs / n as f64);
    println!("tail1pct_mean_abs={:.6e} max_rel(|ref|>1e-3)={max_rel:.6e}", tail_sum / tail_n as f64);
    let bad = hr.iter().zip(hf.iter()).filter(|(&a, &b)| !a.is_finite() || !b.is_finite()).count();
    println!("nonfinite={bad}");
}
