//! Throwaway B1/B2 seven-shape FA2 gfx11 oracle (§7.4).
//!
//! For each fixture (batch, start_pos, seq_len) and K mode (Q8 / fwht3):
//! runs incumbent FA2 (`HIPFIRE_FA2_GFX11_VARIANT=base`) and the candidate
//! (`b1` or `b1b2`, default `b1`; override via CLI arg / env) and prints
//! max-abs-diff + bitwise-equal. TIME=1 adds host-timed us/launch.
//!
//! Usage:
//!   cargo run --release -p hipfire-runtime --example tmp_fa2_gfx11_oracle -- [b1|b1b2]
//!   TIME=1 ROCR_VISIBLE_DEVICES=1 cargo run --release -p hipfire-runtime \
//!     --example tmp_fa2_gfx11_oracle -- b1b2
//!
//! Halo (hipx device 1):
//!   ROCR_VISIBLE_DEVICES=1 HIPFIRE_FA2_GFX11_VARIANT=b1 \
//!     cargo run --release -p hipfire-runtime --example tmp_fa2_gfx11_oracle -- b1

use rdna_compute::{gen_fwht_signs, DType, Gpu};
use std::time::Instant;

const N_HEADS: usize = 24;
const N_KV: usize = 4;
const HD: usize = 256;
const BPH: usize = HD / 32; // 8
const Q8_BLOCK: usize = 34;
const ROW_STRIDE: usize = N_KV * BPH * Q8_BLOCK; // 1088
const FWHT3_HEAD: usize = 100;
const FWHT3_POS: usize = N_KV * FWHT3_HEAD; // 400

/// (batch, start_position, max_position+1) fixtures from plan §7.4.
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 0, 1),
    (7, 26, 33),
    (64, 0, 64),
    (128, 384, 512),
    (512, 0, 512),
    (512, 7680, 8192),
    (512, 32256, 32768),
];

fn prng_u32(i: u64, salt: u32) -> u32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(salt)
        .wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0xc2b2ae35);
    x ^ (x >> 16)
}

fn prng_f32(i: u64, salt: u32) -> f32 {
    (prng_u32(i, salt) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign << 15;
    }
    if exp == 0xff {
        let m = if mant != 0 { 0x200 } else { 0 };
        return (sign << 15) | 0x7c00 | m;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return (sign << 15) | 0x7c00;
    }
    if new_exp <= 0 {
        return sign << 15;
    }
    let new_mant = (mant >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | new_mant
}

fn pack_q8_block(scale: f32, codes: &[i8; 32]) -> [u8; 34] {
    let mut out = [0u8; 34];
    let s = f32_to_f16_bits(scale);
    out[0] = (s & 0xff) as u8;
    out[1] = (s >> 8) as u8;
    for w in 0..8 {
        let mut u = 0u32;
        for c in 0..4 {
            let v = codes[w * 4 + c] as u8;
            u |= (v as u32) << (c * 8);
        }
        out[2 + w * 4..2 + w * 4 + 4].copy_from_slice(&u.to_le_bytes());
    }
    out
}

fn fill_q8_cache(seq_len: usize, salt: u32) -> Vec<u8> {
    let mut buf = vec![0u8; seq_len * ROW_STRIDE];
    for g in 0..seq_len {
        for kv in 0..N_KV {
            for b in 0..BPH {
                let mut codes = [0i8; 32];
                for c in 0..32 {
                    let r = prng_u32((g * 64 + kv * 8 + b) as u64 * 32 + c as u64, salt);
                    codes[c] = ((r % 61) as i8).wrapping_sub(30);
                }
                let scale = 0.01 + (prng_u32((g * 32 + kv * 8 + b) as u64, salt ^ 0xabc) % 100) as f32 * 0.001;
                let blk = pack_q8_block(scale, &codes);
                let off = g * ROW_STRIDE + (kv * BPH + b) * Q8_BLOCK;
                buf[off..off + 34].copy_from_slice(&blk);
            }
        }
    }
    buf
}

fn fill_fwht3_k(seq_len: usize, salt: u32) -> Vec<u8> {
    let mut buf = vec![0u8; seq_len * FWHT3_POS];
    for g in 0..seq_len {
        for kv in 0..N_KV {
            let base = g * FWHT3_POS + kv * FWHT3_HEAD;
            let cnorm = 0.05
                + (prng_u32((g * 4 + kv) as u64, salt) % 200) as f32 * 0.001;
            buf[base..base + 4].copy_from_slice(&cnorm.to_le_bytes());
            // 96 B of 3-bit codes: 32 groups of 3 B = 32 * 8 codes = 256 dims.
            for g3 in 0..32 {
                let mut packed = 0u32;
                for i in 0..8 {
                    let code = (prng_u32((g * 256 + kv * 64 + g3 * 8 + i) as u64, salt) % 8) as u32;
                    packed |= code << (3 * i);
                }
                let bytes = packed.to_le_bytes();
                buf[base + 4 + g3 * 3..base + 4 + g3 * 3 + 3].copy_from_slice(&bytes[..3]);
            }
        }
    }
    buf
}

fn fill_q(batch: usize, salt: u32) -> Vec<f32> {
    let n = batch * N_HEADS * HD;
    (0..n)
        .map(|i| prng_f32(i as u64, salt) * 0.5)
        .collect()
}

fn fill_positions(batch: usize, start: usize) -> Vec<i32> {
    (0..batch).map(|b| (start + b) as i32).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn bitwise_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn upload_i32(gpu: &mut Gpu, data: &[i32], shape: &[usize]) -> rdna_compute::GpuTensor {
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.upload_raw(bytes, shape).unwrap()
}

fn run_q8(
    gpu: &mut Gpu,
    q: &rdna_compute::GpuTensor,
    k: &rdna_compute::GpuTensor,
    v: &rdna_compute::GpuTensor,
    out: &rdna_compute::GpuTensor,
    pos: &rdna_compute::GpuTensor,
    batch: usize,
    seq_len: usize,
) {
    gpu.attention_q8_0_fa2_gqa_gfx11(q, k, v, out, pos, N_HEADS, N_KV, HD, seq_len, batch)
        .unwrap_or_else(|e| panic!("q8 fa2 launch: {e:?}"));
    gpu.hip.device_synchronize().unwrap();
}

fn run_fwht3(
    gpu: &mut Gpu,
    q: &rdna_compute::GpuTensor,
    k: &rdna_compute::GpuTensor,
    v: &rdna_compute::GpuTensor,
    out: &rdna_compute::GpuTensor,
    pos: &rdna_compute::GpuTensor,
    s1: &rdna_compute::GpuTensor,
    s2: &rdna_compute::GpuTensor,
    batch: usize,
    seq_len: usize,
) {
    gpu.attention_q8_0_fa2_gqa_fwht3k_gfx11(
        q, k, v, out, pos, s1, s2, N_HEADS, N_KV, HD, seq_len, batch,
    )
    .unwrap_or_else(|e| panic!("fwht3 fa2 launch: {e:?}"));
    gpu.hip.device_synchronize().unwrap();
}

fn with_variant<T>(variant: &str, f: impl FnOnce() -> T) -> T {
    // Live env read in the launcher (not process snapshot), so set_var works.
    std::env::set_var("HIPFIRE_FA2_GFX11_VARIANT", variant);
    let out = f();
    out
}

fn main() {
    let candidate = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("HIPFIRE_FA2_GFX11_VARIANT").ok())
        .unwrap_or_else(|| "b1".into());
    let candidate = match candidate.as_str() {
        "b1" | "b1b2" => candidate,
        "base" => {
            eprintln!("candidate must be b1 or b1b2 (got base)");
            std::process::exit(2);
        }
        other => other.to_string(),
    };
    if candidate != "b1" && candidate != "b1b2" {
        eprintln!("usage: tmp_fa2_gfx11_oracle [b1|b1b2]");
        std::process::exit(2);
    }
    let time = std::env::var("TIME").ok().as_deref() == Some("1");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "tmp_fa2_gfx11_oracle arch={} candidate={} time={}",
        gpu.arch, candidate, time
    );
    if !matches!(
        gpu.arch.as_str(),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151"
    ) {
        eprintln!("skip: need gfx11, got {}", gpu.arch);
        return;
    }

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let d_s1 = gpu.upload_f32(&signs1, &[256]).unwrap();
    let d_s2 = gpu.upload_f32(&signs2, &[256]).unwrap();

    let mut n_fail = 0usize;

    for &(batch, start, seq_len) in SHAPES {
        let tag = format!("b{batch}_s{start}_L{seq_len}");
        let pos_h = fill_positions(batch, start);
        let d_pos = upload_i32(&mut gpu, &pos_h, &[batch]);

        // ── Q8-K ──────────────────────────────────────────────────────
        {
            let q_h = fill_q(batch, 0x1111);
            let k_h = fill_q8_cache(seq_len, 0x2222);
            let v_h = fill_q8_cache(seq_len, 0x3333);
            let d_k = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
            let d_v = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
            let d_q_base = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
            let d_q_cand = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
            let d_out_base = gpu.zeros(&[batch * N_HEADS * HD], DType::F32).unwrap();
            let d_out_cand = gpu.zeros(&[batch * N_HEADS * HD], DType::F32).unwrap();

            with_variant("base", || {
                run_q8(
                    &mut gpu, &d_q_base, &d_k, &d_v, &d_out_base, &d_pos, batch, seq_len,
                )
            });
            with_variant(&candidate, || {
                run_q8(
                    &mut gpu, &d_q_cand, &d_k, &d_v, &d_out_cand, &d_pos, batch, seq_len,
                )
            });

            let o_base = gpu.download_f32(&d_out_base).unwrap();
            let o_cand = gpu.download_f32(&d_out_cand).unwrap();
            let mad = max_abs_diff(&o_base, &o_cand);
            let beq = bitwise_equal(&o_base, &o_cand);
            if !beq {
                n_fail += 1;
            }
            eprintln!(
                "Q8  {tag:>24}  max_abs={mad:.6e}  bitwise_eq={beq}"
            );

            if time {
                // Warm once each, then measure candidate.
                with_variant("base", || {
                    run_q8(
                        &mut gpu, &d_q_base, &d_k, &d_v, &d_out_base, &d_pos, batch, seq_len,
                    )
                });
                with_variant(&candidate, || {
                    run_q8(
                        &mut gpu, &d_q_cand, &d_k, &d_v, &d_out_cand, &d_pos, batch, seq_len,
                    )
                });
                const N: usize = 20;
                let mut us_base = Vec::with_capacity(N);
                let mut us_cand = Vec::with_capacity(N);
                for _ in 0..N {
                    with_variant("base", || {
                        gpu.hip.device_synchronize().unwrap();
                        let t0 = Instant::now();
                        run_q8(
                            &mut gpu, &d_q_base, &d_k, &d_v, &d_out_base, &d_pos, batch, seq_len,
                        );
                        us_base.push(t0.elapsed().as_secs_f64() * 1e6);
                    });
                    with_variant(&candidate, || {
                        gpu.hip.device_synchronize().unwrap();
                        let t0 = Instant::now();
                        run_q8(
                            &mut gpu, &d_q_cand, &d_k, &d_v, &d_out_cand, &d_pos, batch, seq_len,
                        );
                        us_cand.push(t0.elapsed().as_secs_f64() * 1e6);
                    });
                }
                us_base.sort_by(|a, b| a.partial_cmp(b).unwrap());
                us_cand.sort_by(|a, b| a.partial_cmp(b).unwrap());
                eprintln!(
                    "TIME Q8  {tag:>24}  base_med={:.1}us  {candidate}_med={:.1}us",
                    us_base[N / 2],
                    us_cand[N / 2]
                );
            }
        }

        // ── fwht3-K (clone Q per arm: in-place rotate) ────────────────
        {
            let q_h = fill_q(batch, 0x4444);
            let k_h = fill_fwht3_k(seq_len, 0x5555);
            let v_h = fill_q8_cache(seq_len, 0x6666);
            let d_k = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
            let d_v = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
            let d_q_base = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
            let d_q_cand = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
            let d_out_base = gpu.zeros(&[batch * N_HEADS * HD], DType::F32).unwrap();
            let d_out_cand = gpu.zeros(&[batch * N_HEADS * HD], DType::F32).unwrap();

            with_variant("base", || {
                run_fwht3(
                    &mut gpu,
                    &d_q_base,
                    &d_k,
                    &d_v,
                    &d_out_base,
                    &d_pos,
                    &d_s1,
                    &d_s2,
                    batch,
                    seq_len,
                )
            });
            with_variant(&candidate, || {
                run_fwht3(
                    &mut gpu,
                    &d_q_cand,
                    &d_k,
                    &d_v,
                    &d_out_cand,
                    &d_pos,
                    &d_s1,
                    &d_s2,
                    batch,
                    seq_len,
                )
            });

            let o_base = gpu.download_f32(&d_out_base).unwrap();
            let o_cand = gpu.download_f32(&d_out_cand).unwrap();
            let q_base = gpu.download_f32(&d_q_base).unwrap();
            let q_cand = gpu.download_f32(&d_q_cand).unwrap();
            let mad = max_abs_diff(&o_base, &o_cand);
            let beq = bitwise_equal(&o_base, &o_cand);
            let q_beq = bitwise_equal(&q_base, &q_cand);
            if !beq || !q_beq {
                n_fail += 1;
            }
            eprintln!(
                "F3  {tag:>24}  max_abs={mad:.6e}  bitwise_eq={beq}  q_post_eq={q_beq}"
            );

            if time {
                // Need fresh Q for each timed launch (rotation is in-place).
                const N: usize = 10;
                let mut us_base = Vec::with_capacity(N);
                let mut us_cand = Vec::with_capacity(N);
                for _ in 0..N {
                    let qb = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
                    with_variant("base", || {
                        gpu.hip.device_synchronize().unwrap();
                        let t0 = Instant::now();
                        run_fwht3(
                            &mut gpu, &qb, &d_k, &d_v, &d_out_base, &d_pos, &d_s1, &d_s2, batch,
                            seq_len,
                        );
                        us_base.push(t0.elapsed().as_secs_f64() * 1e6);
                    });
                    let qc = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
                    with_variant(&candidate, || {
                        gpu.hip.device_synchronize().unwrap();
                        let t0 = Instant::now();
                        run_fwht3(
                            &mut gpu, &qc, &d_k, &d_v, &d_out_cand, &d_pos, &d_s1, &d_s2, batch,
                            seq_len,
                        );
                        us_cand.push(t0.elapsed().as_secs_f64() * 1e6);
                    });
                }
                us_base.sort_by(|a, b| a.partial_cmp(b).unwrap());
                us_cand.sort_by(|a, b| a.partial_cmp(b).unwrap());
                eprintln!(
                    "TIME F3  {tag:>24}  base_med={:.1}us  {candidate}_med={:.1}us",
                    us_base[N / 2],
                    us_cand[N / 2]
                );
            }
        }
    }

    if n_fail == 0 {
        eprintln!("ORACLE PASS  candidate={candidate}  shapes={}", SHAPES.len());
    } else {
        eprintln!("ORACLE FAIL  n_fail={n_fail}  candidate={candidate}");
        std::process::exit(1);
    }
}
