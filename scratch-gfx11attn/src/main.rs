use half::f16;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const BLOCKS: usize = HD / 32;
const ROW_BYTES: usize = NKV * BLOCKS * 34;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn rng(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 16) as u32
}

fn fixture(batch: usize, ctx: usize, seed: u64) -> (Vec<f32>, Vec<u8>, Vec<u8>, Vec<i32>) {
    assert!(batch <= ctx);
    let mut state = seed;
    let mut q = Vec::with_capacity(batch * NH * HD);
    for _ in 0..q.capacity() {
        let x = (rng(&mut state) as i32 % 2001 - 1000) as f32 * 0.0007;
        q.push(x);
    }
    let mut make_cache = |salt: u32| {
        let mut cache = vec![0u8; ctx * ROW_BYTES];
        for (bi, block) in cache.chunks_exact_mut(34).enumerate() {
            let scale = 0.0015 + ((bi as u32 * 17 + salt) % 31) as f32 * 0.00017;
            let bits = f16::from_f32(scale).to_bits().to_le_bytes();
            block[0] = bits[0];
            block[1] = bits[1];
            for (j, code) in block[2..].iter_mut().enumerate() {
                let raw = ((rng(&mut state).wrapping_add(salt).wrapping_add(j as u32)) % 31) as i8 - 15;
                *code = raw as u8;
            }
        }
        cache
    };
    let k = make_cache(11);
    let v = make_cache(97);
    let start = ctx - batch;
    let positions = (0..batch).map(|i| (start + i) as i32).collect();
    (q, k, v, positions)
}

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> rdna_compute::GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
    };
    gpu.upload_raw(bytes, &[values.len()]).expect("positions upload")
}

fn launch_incumbent(
    gpu: &mut Gpu,
    q: &rdna_compute::GpuTensor,
    k: &rdna_compute::GpuTensor,
    v: &rdna_compute::GpuTensor,
    out: &rdna_compute::GpuTensor,
    pos: &rdna_compute::GpuTensor,
    batch: usize,
    ctx: usize,
) {
    gpu.attention_q8_0_fa2_gqa_gfx11(q, k, v, out, pos, NH, NKV, HD, ctx, batch)
        .expect("incumbent launch");
}

fn launch_wide(
    gpu: &mut Gpu,
    q: &rdna_compute::GpuTensor,
    k: &rdna_compute::GpuTensor,
    v: &rdna_compute::GpuTensor,
    out: &rdna_compute::GpuTensor,
    pos: &rdna_compute::GpuTensor,
    batch: usize,
    ctx: usize,
) {
    gpu.attention_q8_0_fa2_gqa_wide_gfx11(q, k, v, out, pos, NH, NKV, HD, ctx, batch)
        .expect("wide launch");
}

fn time_arm(
    gpu: &mut Gpu,
    arm: char,
    iters: usize,
    q: &rdna_compute::GpuTensor,
    k: &rdna_compute::GpuTensor,
    v: &rdna_compute::GpuTensor,
    out: &rdna_compute::GpuTensor,
    pos: &rdna_compute::GpuTensor,
    batch: usize,
    ctx: usize,
) -> f64 {
    gpu.hip.device_synchronize().expect("pre timing sync");
    let started = Instant::now();
    for _ in 0..iters {
        match arm {
            'A' => launch_incumbent(gpu, q, k, v, out, pos, batch, ctx),
            'B' => launch_wide(gpu, q, k, v, out, pos, batch, ctx),
            _ => panic!("unknown arm {arm}"),
        }
    }
    gpu.hip.device_synchronize().expect("post timing sync");
    started.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn scale(cache: &[u8], token: usize, kvh: usize, block: usize) -> f32 {
    let base = token * ROW_BYTES + (kvh * BLOCKS + block) * 34;
    f16::from_bits(u16::from_le_bytes([cache[base], cache[base + 1]])).to_f32()
}

fn value(cache: &[u8], token: usize, kvh: usize, dim: usize) -> f32 {
    let block = dim / 32;
    let base = token * ROW_BYTES + (kvh * BLOCKS + block) * 34;
    scale(cache, token, kvh, block) * (cache[base + 2 + dim % 32] as i8) as f32
}

fn scalar_f32_oracle(
    q: &[f32],
    k: &[u8],
    v: &[u8],
    positions: &[i32],
    batch: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * NH * HD];
    let attn_scale = 1.0f32 / (HD as f32).sqrt();
    let mut scores = Vec::<f32>::new();
    for qr in 0..batch {
        let upto = positions[qr] as usize + 1;
        scores.resize(upto, 0.0);
        for h in 0..NH {
            let kvh = h / (NH / NKV);
            let qb = (qr * NH + h) * HD;
            let ob = qb;
            let mut m = f32::NEG_INFINITY;
            for token in 0..upto {
                let mut dot = 0.0f32;
                for d in 0..HD {
                    dot += q[qb + d] * value(k, token, kvh, d);
                }
                let s = dot * attn_scale;
                scores[token] = s;
                m = m.max(s);
            }
            let mut denom = 0.0f32;
            for s in &mut scores[..upto] {
                *s = (*s - m).exp();
                denom += *s;
            }
            let inv = 1.0f32 / denom;
            for token in 0..upto {
                let p = scores[token] * inv;
                for d in 0..HD {
                    out[ob + d] += p * value(v, token, kvh, d);
                }
            }
        }
    }
    out
}

#[derive(Debug)]
struct ErrorStats {
    nrmse: f64,
    max_abs: f32,
    bit_mismatches: usize,
}

fn error_stats(got: &[f32], reference: &[f32]) -> ErrorStats {
    let mut err2 = 0.0f64;
    let mut ref2 = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut bit_mismatches = 0usize;
    for (&g, &r) in got.iter().zip(reference) {
        let e = g - r;
        err2 += (e as f64) * (e as f64);
        ref2 += (r as f64) * (r as f64);
        max_abs = max_abs.max(e.abs());
        bit_mismatches += usize::from(g.to_bits() != r.to_bits());
    }
    ErrorStats {
        nrmse: (err2 / ref2.max(f64::MIN_POSITIVE)).sqrt(),
        max_abs,
        bit_mismatches,
    }
}

fn numeric_check(gpu: &mut Gpu) {
    let batch = env_usize("CHECK_B", 17);
    let ctx = env_usize("CHECK_CTX", 83);
    let (q_h, k_h, v_h, pos_h) = fixture(batch, ctx, 0x8f1d_57c3_2249_a61b);
    let q = gpu.upload_f32(&q_h, &[q_h.len()]).expect("check q");
    let k = gpu.upload_raw(&k_h, &[k_h.len()]).expect("check k");
    let v = gpu.upload_raw(&v_h, &[v_h.len()]).expect("check v");
    let pos = upload_i32(gpu, &pos_h);
    let a = gpu.zeros(&[q_h.len()], DType::F32).expect("check a");
    let b = gpu.zeros(&[q_h.len()], DType::F32).expect("check b");
    launch_incumbent(gpu, &q, &k, &v, &a, &pos, batch, ctx);
    launch_wide(gpu, &q, &k, &v, &b, &pos, batch, ctx);
    gpu.hip.device_synchronize().expect("check sync");
    let a_h = gpu.download_f32(&a).expect("download incumbent");
    let b_h = gpu.download_f32(&b).expect("download wide");
    let oracle = scalar_f32_oracle(&q_h, &k_h, &v_h, &pos_h, batch);
    let a_oracle = error_stats(&a_h, &oracle);
    let b_oracle = error_stats(&b_h, &oracle);
    let b_a = error_stats(&b_h, &a_h);
    println!(
        "NUMERIC arch={} B={} S={} incumbent_vs_f32={{nrmse:{:.9e},max_abs:{:.9e}}} wide_vs_f32={{nrmse:{:.9e},max_abs:{:.9e}}} wide_vs_incumbent={{nrmse:{:.9e},max_abs:{:.9e},bit_mismatches:{}}}",
        gpu.arch,
        batch,
        ctx,
        a_oracle.nrmse,
        a_oracle.max_abs,
        b_oracle.nrmse,
        b_oracle.max_abs,
        b_a.nrmse,
        b_a.max_abs,
        b_a.bit_mismatches,
    );
    assert!(a_h.iter().all(|x| x.is_finite()));
    assert!(b_h.iter().all(|x| x.is_finite()));
}

fn main() {
    let batch = env_usize("B", 512);
    let ctx = env_usize("S", 8283);
    let warmups = env_usize("WARMUPS", 3);
    let iters = env_usize("ITERS", 8);
    let order = std::env::var("ORDER").unwrap_or_else(|_| "ABBA".to_string());
    let expected_arch = std::env::var("EXPECT_ARCH").expect("EXPECT_ARCH is required");
    assert!(batch <= 512 && batch <= ctx);
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, expected_arch, "wrong leased GPU");
    println!("ARCH_ASSERT expected={} actual={}", expected_arch, gpu.arch);

    numeric_check(&mut gpu);

    let (q_h, k_h, v_h, pos_h) = fixture(batch, ctx, 0xc841_54ef_93d2_7719);
    let q = gpu.upload_f32(&q_h, &[q_h.len()]).expect("q upload");
    let k = gpu.upload_raw(&k_h, &[k_h.len()]).expect("k upload");
    let v = gpu.upload_raw(&v_h, &[v_h.len()]).expect("v upload");
    let pos = upload_i32(&mut gpu, &pos_h);
    let out = gpu.zeros(&[q_h.len()], DType::F32).expect("out");

    for _ in 0..warmups {
        launch_incumbent(&mut gpu, &q, &k, &v, &out, &pos, batch, ctx);
        launch_wide(&mut gpu, &q, &k, &v, &out, &pos, batch, ctx);
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    for (slot, arm) in order.chars().enumerate() {
        let ms = time_arm(&mut gpu, arm, iters, &q, &k, &v, &out, &pos, batch, ctx);
        let tokens_s = batch as f64 * 1000.0 / ms;
        println!(
            "TIMING arch={} slot={} arm={} B={} S={} iters={} ms={:.6} tokens_s={:.3}",
            gpu.arch, slot, arm, batch, ctx, iters, ms, tokens_s
        );
    }
}
