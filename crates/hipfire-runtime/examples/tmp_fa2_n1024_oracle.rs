//! Throwaway Gate F prereq (1) FA2 N=1024 bit-identity oracle.
//!
//! For each fixture (batch=1024, start, L) and K mode (Q8 / fwht3):
//!   (a) one N=1024 direct launch via `*_direct_unchecked`
//!   (b) two N=512 launches on rows [0,512) and [512,1024) with identical KV
//! Compare outputs bitwise (and post-launch Q for fwht3). Print max_abs /
//! bitwise_eq per shape/mode and a final `F1 PASS/FAIL` line.
//! TIME=1 prints host us for (a) vs (b) summed.
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-f \
//!     cargo run --release -p hipfire-runtime --example tmp_fa2_n1024_oracle
//!
//! ORACLE ONLY — no host envelope change (prereq 2).

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
const N: usize = 1024;
const HALF: usize = 512;
const QO_ROW: usize = N_HEADS * HD; // 6144

/// (batch, start_position, max_position+1) fixtures from plan §11 prereq (1).
const SHAPES: &[(usize, usize, usize)] = &[
    (1024, 0, 1024),
    (1024, 7168, 8192),
    (1024, 31744, 32768),
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
                let scale =
                    0.01 + (prng_u32((g * 32 + kv * 8 + b) as u64, salt ^ 0xabc) % 100) as f32 * 0.001;
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
            let cnorm = 0.05 + (prng_u32((g * 4 + kv) as u64, salt) % 200) as f32 * 0.001;
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
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
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
    // N=1024 needs the oracle-only unchecked twin; N=512 also uses it so both
    // arms share one code path (production wrapper still caps at 512).
    gpu.attention_q8_0_fa2_gqa_gfx11_direct_unchecked(
        q, k, v, out, pos, N_HEADS, N_KV, HD, seq_len, batch,
    )
    .unwrap_or_else(|e| panic!("q8 fa2 launch batch={batch}: {e:?}"));
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
    gpu.attention_q8_0_fa2_gqa_fwht3k_gfx11_direct_unchecked(
        q, k, v, out, pos, s1, s2, N_HEADS, N_KV, HD, seq_len, batch,
    )
    .unwrap_or_else(|e| panic!("fwht3 fa2 launch batch={batch}: {e:?}"));
    gpu.hip.device_synchronize().unwrap();
}

fn main() {
    let time = std::env::var("TIME").ok().as_deref() == Some("1");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "tmp_fa2_n1024_oracle arch={} time={}  (Gate F prereq 1)",
        gpu.arch, time
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
        assert_eq!(batch, N);
        let tag = format!("b{batch}_s{start}_L{seq_len}");
        let pos_h = fill_positions(batch, start);

        // ── Q8-K ──────────────────────────────────────────────────────
        {
            let q_h = fill_q(batch, 0x1111);
            let k_h = fill_q8_cache(seq_len, 0x2222);
            let v_h = fill_q8_cache(seq_len, 0x3333);

            // (a) one N=1024 — clone KV (+Q) before launch
            let d_k_a = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
            let d_v_a = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
            let d_q_a = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
            let d_pos_a = upload_i32(&mut gpu, &pos_h, &[batch]);
            let d_out_a = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();
            run_q8(
                &mut gpu, &d_q_a, &d_k_a, &d_v_a, &d_out_a, &d_pos_a, batch, seq_len,
            );
            let o_a = gpu.download_f32(&d_out_a).unwrap();

            // (b) two N=512 on [0,512) and [512,1024) — fresh KV clone
            let d_k_b = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
            let d_v_b = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
            let d_q_b = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
            let d_pos_b = upload_i32(&mut gpu, &pos_h, &[batch]);
            let d_out_b = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();

            let q0 = d_q_b.sub_offset(0, HALF * QO_ROW);
            let q1 = d_q_b.sub_offset(HALF * QO_ROW, HALF * QO_ROW);
            let o0 = d_out_b.sub_offset(0, HALF * QO_ROW);
            let o1 = d_out_b.sub_offset(HALF * QO_ROW, HALF * QO_ROW);
            let p0 = d_pos_b.sub_offset(0, HALF);
            let p1 = d_pos_b.sub_offset(HALF, HALF);

            run_q8(&mut gpu, &q0, &d_k_b, &d_v_b, &o0, &p0, HALF, seq_len);
            run_q8(&mut gpu, &q1, &d_k_b, &d_v_b, &o1, &p1, HALF, seq_len);
            let o_b = gpu.download_f32(&d_out_b).unwrap();

            let mad = max_abs_diff(&o_a, &o_b);
            let beq = bitwise_equal(&o_a, &o_b);
            if !beq {
                n_fail += 1;
            }
            eprintln!("Q8  {tag:>24}  max_abs={mad:.6e}  bitwise_eq={beq}");

            if time {
                // Warm
                run_q8(
                    &mut gpu, &d_q_a, &d_k_a, &d_v_a, &d_out_a, &d_pos_a, batch, seq_len,
                );
                run_q8(&mut gpu, &q0, &d_k_b, &d_v_b, &o0, &p0, HALF, seq_len);
                run_q8(&mut gpu, &q1, &d_k_b, &d_v_b, &o1, &p1, HALF, seq_len);
                const IT: usize = 10;
                let mut us_a = Vec::with_capacity(IT);
                let mut us_b = Vec::with_capacity(IT);
                for _ in 0..IT {
                    gpu.hip.device_synchronize().unwrap();
                    let t0 = Instant::now();
                    run_q8(
                        &mut gpu, &d_q_a, &d_k_a, &d_v_a, &d_out_a, &d_pos_a, batch, seq_len,
                    );
                    us_a.push(t0.elapsed().as_secs_f64() * 1e6);

                    gpu.hip.device_synchronize().unwrap();
                    let t0 = Instant::now();
                    run_q8(&mut gpu, &q0, &d_k_b, &d_v_b, &o0, &p0, HALF, seq_len);
                    run_q8(&mut gpu, &q1, &d_k_b, &d_v_b, &o1, &p1, HALF, seq_len);
                    us_b.push(t0.elapsed().as_secs_f64() * 1e6);
                }
                us_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
                us_b.sort_by(|a, b| a.partial_cmp(b).unwrap());
                eprintln!(
                    "TIME Q8  {tag:>24}  n1024_med={:.1}us  two512_sum_med={:.1}us",
                    us_a[IT / 2],
                    us_b[IT / 2]
                );
            }
        }

        // ── fwht3-K (clone Q per arm: in-place rotate) ────────────────
        {
            let q_h = fill_q(batch, 0x4444);
            let k_h = fill_fwht3_k(seq_len, 0x5555);
            let v_h = fill_q8_cache(seq_len, 0x6666);

            // (a) N=1024 with dedicated Q clone
            let d_k_a = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
            let d_v_a = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
            let d_q_a = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
            let d_pos_a = upload_i32(&mut gpu, &pos_h, &[batch]);
            let d_out_a = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();
            run_fwht3(
                &mut gpu, &d_q_a, &d_k_a, &d_v_a, &d_out_a, &d_pos_a, &d_s1, &d_s2, batch,
                seq_len,
            );
            let o_a = gpu.download_f32(&d_out_a).unwrap();
            let q_a = gpu.download_f32(&d_q_a).unwrap();

            // (b) two N=512 — fresh KV + fresh Q (clone), write into full out/q
            let d_k_b = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
            let d_v_b = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
            let d_q_b = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
            let d_pos_b = upload_i32(&mut gpu, &pos_h, &[batch]);
            let d_out_b = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();

            let q0 = d_q_b.sub_offset(0, HALF * QO_ROW);
            let q1 = d_q_b.sub_offset(HALF * QO_ROW, HALF * QO_ROW);
            let o0 = d_out_b.sub_offset(0, HALF * QO_ROW);
            let o1 = d_out_b.sub_offset(HALF * QO_ROW, HALF * QO_ROW);
            let p0 = d_pos_b.sub_offset(0, HALF);
            let p1 = d_pos_b.sub_offset(HALF, HALF);

            run_fwht3(
                &mut gpu, &q0, &d_k_b, &d_v_b, &o0, &p0, &d_s1, &d_s2, HALF, seq_len,
            );
            run_fwht3(
                &mut gpu, &q1, &d_k_b, &d_v_b, &o1, &p1, &d_s1, &d_s2, HALF, seq_len,
            );
            let o_b = gpu.download_f32(&d_out_b).unwrap();
            let q_b = gpu.download_f32(&d_q_b).unwrap();

            let mad = max_abs_diff(&o_a, &o_b);
            let beq = bitwise_equal(&o_a, &o_b);
            let q_beq = bitwise_equal(&q_a, &q_b);
            if !beq || !q_beq {
                n_fail += 1;
            }
            eprintln!(
                "F3  {tag:>24}  max_abs={mad:.6e}  bitwise_eq={beq}  q_post_eq={q_beq}"
            );

            if time {
                const IT: usize = 5;
                let mut us_a = Vec::with_capacity(IT);
                let mut us_b = Vec::with_capacity(IT);
                for _ in 0..IT {
                    // Fresh Q each timed launch (rotation is in-place).
                    let qa = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
                    gpu.hip.device_synchronize().unwrap();
                    let t0 = Instant::now();
                    run_fwht3(
                        &mut gpu, &qa, &d_k_a, &d_v_a, &d_out_a, &d_pos_a, &d_s1, &d_s2, batch,
                        seq_len,
                    );
                    us_a.push(t0.elapsed().as_secs_f64() * 1e6);

                    let qb = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
                    let qb0 = qb.sub_offset(0, HALF * QO_ROW);
                    let qb1 = qb.sub_offset(HALF * QO_ROW, HALF * QO_ROW);
                    gpu.hip.device_synchronize().unwrap();
                    let t0 = Instant::now();
                    run_fwht3(
                        &mut gpu, &qb0, &d_k_b, &d_v_b, &o0, &p0, &d_s1, &d_s2, HALF, seq_len,
                    );
                    run_fwht3(
                        &mut gpu, &qb1, &d_k_b, &d_v_b, &o1, &p1, &d_s1, &d_s2, HALF, seq_len,
                    );
                    us_b.push(t0.elapsed().as_secs_f64() * 1e6);
                }
                us_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
                us_b.sort_by(|a, b| a.partial_cmp(b).unwrap());
                eprintln!(
                    "TIME F3  {tag:>24}  n1024_med={:.1}us  two512_sum_med={:.1}us",
                    us_a[IT / 2],
                    us_b[IT / 2]
                );
            }
        }
    }

    if n_fail == 0 {
        eprintln!("F1 PASS  shapes={} modes=Q8,fwht3", SHAPES.len());
    } else {
        eprintln!("F1 FAIL  n_fail={n_fail}");
        std::process::exit(1);
    }
}
