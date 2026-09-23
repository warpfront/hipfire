// THROWAWAY Slice-A G2 harness. DELETE BEFORE COMMIT. Not for production.
//
// Deterministic direct + fwht3 + split/merge raw-output dumps plus HIP-event
// TIME. Run the OFF (HEAD) and ON (slice A) builds with the same --out dirs
// and compare offline. Usage:
//   tmp_fa2a_g2 --out <dir> [--reps N]

use rdna_compute::{gen_fwht_signs, DType, Gpu};

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const BPH: usize = HD / 32;
const Q8_BLOCK: usize = 34;
const ROW_STRIDE: usize = NKV * BPH * Q8_BLOCK;
const FWHT3_HEAD: usize = 100;
const FWHT3_POS: usize = NKV * FWHT3_HEAD;

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

fn f32_to_f16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        if mant != 0 {
            return sign | 0x7e00 | ((mant >> 13) as u16 & 0x1ff);
        }
        return sign | 0x7c00;
    }
    let e16 = exp - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7bff;
    }
    if e16 <= 0 {
        return sign;
    }
    let m = mant >> 13;
    // round to nearest even on the dropped 13 bits
    let dropped = mant & 0x1fff;
    let m = if dropped > 0x1000 || (dropped == 0x1000 && (m & 1) == 1) {
        m + 1
    } else {
        m
    };
    if m >= 0x400 {
        return sign | (((e16 + 1) as u16) << 10);
    }
    sign | ((e16 as u16) << 10) | (m as u16)
}

fn pack_q8_block(scale: f32, codes: &[i8; 32]) -> [u8; 34] {
    let mut out = [0u8; 34];
    let s = f32_to_f16_rne(scale);
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
        for kv in 0..NKV {
            for b in 0..BPH {
                let mut codes = [0i8; 32];
                for c in 0..32 {
                    let r = prng_u32((g * 64 + kv * 8 + b) as u64 * 32 + c as u64, salt);
                    codes[c] = ((r % 61) as i8).wrapping_sub(30);
                }
                let scale =
                    0.01 + (prng_u32((g * 32 + kv * 8 + b) as u64, salt ^ 0xabc) % 100) as f32 * 0.001;
                let blk = pack_q8_block(scale, &codes);
                let base = g * ROW_STRIDE + (kv * BPH + b) * Q8_BLOCK;
                buf[base..base + 34].copy_from_slice(&blk);
            }
        }
    }
    buf
}

fn fill_fwht3_k(seq_len: usize, salt: u32) -> Vec<u8> {
    let mut buf = vec![0u8; seq_len * FWHT3_POS];
    for g in 0..seq_len {
        for kv in 0..NKV {
            let base = g * FWHT3_POS + kv * FWHT3_HEAD;
            let cnorm = 0.05 + (prng_u32((g * 4 + kv) as u64, salt) % 200) as f32 * 0.001;
            buf[base..base + 4].copy_from_slice(&cnorm.to_le_bytes());
            for g3 in 0..32 {
                let mut packed = 0u32;
                for i in 0..8 {
                    let code =
                        (prng_u32((g * 256 + kv * 64 + g3 * 8 + i) as u64, salt) % 8) as u32;
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
    let n = batch * NH * HD;
    (0..n).map(|i| prng_f32(i as u64, salt) * 0.5).collect()
}

fn pos_causal(batch: usize, ctx: usize) -> Vec<i32> {
    assert!(ctx >= batch);
    (0..batch).map(|b| (ctx - batch + b) as i32).collect()
}

fn pos_trunc(batch: usize, ctx: usize) -> Vec<i32> {
    (0..batch).map(|q| std::cmp::min(q, ctx - 1) as i32).collect()
}

fn pos_bytes(pos: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pos.len() * 4);
    for &p in pos {
        bytes.extend_from_slice(&p.to_le_bytes());
    }
    bytes
}

fn write_f32(path: &str, v: &[f32]) {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    std::fs::write(path, bytes).unwrap();
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_q8(
    gpu: &mut Gpu,
    d_q: &rdna_compute::GpuTensor,
    d_k: &rdna_compute::GpuTensor,
    d_v: &rdna_compute::GpuTensor,
    d_out: &rdna_compute::GpuTensor,
    d_pos: &rdna_compute::GpuTensor,
    batch: usize,
    ctx: usize,
    reps: usize,
) -> f64 {
    let mut us = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&start, gpu.active_stream.as_ref()).unwrap();
        gpu.attention_q8_0_fa2_gqa_gfx1201(d_q, d_k, d_v, d_out, d_pos, NH, NKV, HD, ctx, batch)
            .unwrap();
        gpu.hip.event_record(&stop, gpu.active_stream.as_ref()).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        us.push(gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1000.0);
        let _ = gpu.hip.event_destroy(start);
        let _ = gpu.hip.event_destroy(stop);
    }
    median(us)
}

fn run_direct(
    gpu: &mut Gpu,
    outdir: &str,
    tag: &str,
    batch: usize,
    ctx: usize,
    pos: Vec<i32>,
    kq8: &[u8],
    vq8: &[u8],
    q: &[f32],
    reps: usize,
    tlog: &mut String,
) {
    let d_k = gpu.upload_raw(kq8, &[kq8.len()]).unwrap();
    let d_v = gpu.upload_raw(vq8, &[vq8.len()]).unwrap();
    let d_q = gpu.upload_f32(q, &[batch, NH, HD]).unwrap();
    let d_pos = gpu.upload_raw(&pos_bytes(&pos), &[batch * 4]).unwrap();
    let d_out = gpu.zeros(&[batch * NH * HD], DType::F32).unwrap();
    // warmup
    for _ in 0..2 {
        gpu.attention_q8_0_fa2_gqa_gfx1201(&d_q, &d_k, &d_v, &d_out, &d_pos, NH, NKV, HD, ctx, batch)
            .unwrap();
    }
    let us = time_q8(gpu, &d_q, &d_k, &d_v, &d_out, &d_pos, batch, ctx, reps);
    let o: Vec<f32> = gpu.download_f32(&d_out).unwrap();
    assert_eq!(o.len(), batch * NH * HD);
    assert!(o.iter().all(|x| x.is_finite()), "non-finite {tag}");
    let path = format!("{outdir}/direct_{tag}_b{batch}_c{ctx}.f32");
    write_f32(&path, &o);
    let cksum: u64 = o.iter().fold(0, |a, &x| a.wrapping_add(x.to_bits() as u64));
    eprintln!("direct {tag} batch={batch} ctx={ctx} median_us={us:.2} cksum={cksum:016x}");
    tlog.push_str(&format!("{tag} batch={batch} ctx={ctx} median_us={us:.2}\n"));
}

fn run_fwht3(
    gpu: &mut Gpu,
    outdir: &str,
    tag: &str,
    batch: usize,
    ctx: usize,
    kfw: &[u8],
    vq8: &[u8],
    q: &[f32],
) {
    let pos = if batch > ctx { pos_trunc(batch, ctx) } else { pos_causal(batch, ctx) };
    let d_k = gpu.upload_raw(kfw, &[kfw.len()]).unwrap();
    let d_v = gpu.upload_raw(vq8, &[vq8.len()]).unwrap();
    let d_q = gpu.upload_f32(q, &[batch, NH, HD]).unwrap();
    let d_pos = gpu.upload_raw(&pos_bytes(&pos), &[batch * 4]).unwrap();
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let d_s1 = gpu.upload_f32(&s1, &[256]).unwrap();
    let d_s2 = gpu.upload_f32(&s2, &[256]).unwrap();
    let d_out = gpu.zeros(&[batch * NH * HD], DType::F32).unwrap();
    gpu.attention_q8_0_fa2_gqa_fwht3k_gfx1201(
        &d_q, &d_k, &d_v, &d_out, &d_pos, &d_s1, &d_s2, NH, NKV, HD, ctx, batch,
    )
    .unwrap();
    let o: Vec<f32> = gpu.download_f32(&d_out).unwrap();
    assert!(o.iter().all(|x| x.is_finite()), "non-finite fwht3 {tag}");
    let path = format!("{outdir}/fwht3_{tag}_b{batch}_c{ctx}.f32");
    write_f32(&path, &o);
    let cksum: u64 = o.iter().fold(0, |a, &x| a.wrapping_add(x.to_bits() as u64));
    eprintln!("fwht3 {tag} batch={batch} ctx={ctx} cksum={cksum:016x}");
}

fn run_split(
    gpu: &mut Gpu,
    outdir: &str,
    batch: usize,
    ctx: usize,
    splits: usize,
    kq8: &[u8],
    vq8: &[u8],
    q: &[f32],
) {
    let d_k = gpu.upload_raw(kq8, &[kq8.len()]).unwrap();
    let d_v = gpu.upload_raw(vq8, &[vq8.len()]).unwrap();
    let d_q = gpu.upload_f32(q, &[batch, NH, HD]).unwrap();
    let pos = pos_causal(batch, ctx);
    let d_pos = gpu.upload_raw(&pos_bytes(&pos), &[batch * 4]).unwrap();
    let d_out = gpu.zeros(&[batch * NH * HD], DType::F32).unwrap();
    let d_part = gpu
        .zeros(&[splits * batch * NH * (HD + 2)], DType::F32)
        .unwrap();
    gpu.attention_q8_0_fa2_gqa_split_gfx1201_bench(
        &d_q, &d_k, &d_v, &d_out, &d_pos, &d_part, NH, NKV, HD, batch, splits,
    )
    .unwrap();
    let o: Vec<f32> = gpu.download_f32(&d_out).unwrap();
    let p: Vec<f32> = gpu.download_f32(&d_part).unwrap();
    assert!(o.iter().all(|x| x.is_finite()), "non-finite split out");
    write_f32(&format!("{outdir}/split_b{batch}_c{ctx}_s{splits}.f32"), &o);
    write_f32(&format!("{outdir}/partials_b{batch}_c{ctx}_s{splits}.f32"), &p);
    let ck = |v: &[f32]| v.iter().fold(0u64, |a, &x| a.wrapping_add(x.to_bits() as u64));
    eprintln!(
        "split batch={batch} ctx={ctx} s={splits} o_ck={:016x} p_ck={:016x}",
        ck(&o),
        ck(&p)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut outdir = String::from(".");
    let mut reps = 10usize;
    let mut only = String::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                outdir = args[i + 1].clone();
                i += 2;
            }
            "--reps" => {
                reps = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--only" => {
                only = args[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }
    std::fs::create_dir_all(&outdir).unwrap();
    let mut gpu = Gpu::init().expect("gpu init");
    let mut tlog = String::new();

    if !only.starts_with("split") {
    // 7 shapes (TIME + dumps).
    for &(b, c) in &[(4, 256), (8, 512), (32, 2048), (64, 4096), (128, 8192), (256, 16384), (512, 32768)] {
        let k = fill_q8_cache(c, 0xA11 + b as u32);
        let v = fill_q8_cache(c, 0xB22 + b as u32);
        let q = fill_q(b, 0xC33 + b as u32);
        run_direct(&mut gpu, &outdir, "shape", b, c, pos_causal(b, c), &k, &v, &q, reps, &mut tlog);
    }
    // Truncated tiles.
    for &(b, c) in &[(16, 15), (16, 16), (16, 17), (32, 31), (32, 32), (32, 33), (64, 63), (64, 64), (64, 65)] {
        let k = fill_q8_cache(c, 0xA11);
        let v = fill_q8_cache(c, 0xB22);
        let q = fill_q(b, 0xC33);
        run_direct(&mut gpu, &outdir, "trunc", b, c, pos_trunc(b, c), &k, &v, &q, 3, &mut tlog);
    }
    // Retained tails.
    for &(b, c) in &[(3, 100), (6, 2000), (8, 5000)] {
        let k = fill_q8_cache(c, 0xA11);
        let v = fill_q8_cache(c, 0xB22);
        let q = fill_q(b, 0xC33);
        run_direct(&mut gpu, &outdir, "tail", b, c, pos_causal(b, c), &k, &v, &q, reps, &mut tlog);
    }
    // Odd batches.
    for &(b, c) in &[(1, 256), (7, 512), (9, 512)] {
        let k = fill_q8_cache(c, 0xA11);
        let v = fill_q8_cache(c, 0xB22);
        let q = fill_q(b, 0xC33);
        run_direct(&mut gpu, &outdir, "batch", b, c, pos_causal(b, c), &k, &v, &q, 3, &mut tlog);
    }
    // Position variants on (8,512).
    {
        let (b, c) = (8, 512);
        let k = fill_q8_cache(c, 0xA11);
        let v = fill_q8_cache(c, 0xB22);
        let q = fill_q(b, 0xC33);
        run_direct(&mut gpu, &outdir, "posdup", b, c,
            vec![504, 504, 506, 506, 508, 508, 511, 511], &k, &v, &q, 3, &mut tlog);
        run_direct(&mut gpu, &outdir, "posnonmono", b, c,
            vec![511, 500, 505, 495, 502, 480, 511, 509], &k, &v, &q, 3, &mut tlog);
        run_direct(&mut gpu, &outdir, "posallneg", b, c,
            vec![-1; 8], &k, &v, &q, 3, &mut tlog);
    }
    // fwht3 vs own baseline.
    for &(b, c) in &[(64, 4096), (16, 17)] {
        let kfw = fill_fwht3_k(c, 0xF30D);
        let v = fill_q8_cache(c, 0xB22);
        let q = fill_q(b, 0xC33);
        run_fwht3(&mut gpu, &outdir, "k3", b, c, &kfw, &v, &q);
    }
    }
    // Partial/merge at splits 1 and 8.
    if only.is_empty() || only.starts_with("split") {
    for &(b, c) in &[(64, 4096), (16, 17)] {
        // --only split1: s=1 only; --only split8: s=8 only; else both.
        let splits: &[usize] = if only == "split1" {
            &[1]
        } else if only == "split8" {
            &[8]
        } else {
            &[1, 8]
        };
        for &s in splits {
            let k = fill_q8_cache(c, 0xA11);
            let v = fill_q8_cache(c, 0xB22);
            let q = fill_q(b, 0xC33);
            run_split(&mut gpu, &outdir, b, c, s, &k, &v, &q);
        }
    }
    }
    std::fs::write(format!("{outdir}/time.log"), &tlog).unwrap();
    eprintln!("wrote {outdir}/time.log");
}
