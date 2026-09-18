// SPDX-License-Identifier: Apache-2.0
// Temporary G2 differential: v1 vs candidate bit identity on direct launches
// (truncated causal tiles, batch tails) and partial-split [m,l,O]/merged
// outputs. Deterministic RNG inputs identical across arms; run once per arm
// with HIPFIRE_GFX12_FA2_FP8=1 and cmp the files offline.

use rdna_compute::{DType, Gpu};

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const BPH: usize = HD / 32;
const ROW_STRIDE: usize = NKV * BPH * 34;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn half_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let s = (x >> 16) & 0x8000;
    let e = ((x >> 23) & 0xFF) as i32;
    let m = x & 0x7FFFFF;
    if e == 0xFF {
        return (s | 0x7BFF) as u16;
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return (s | 0x7BFF) as u16;
    }
    if e16 <= 0 {
        return s as u16;
    }
    (s | ((e16 as u32) << 10) | (m >> 13)) as u16
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut outdir = String::from("/tmp/fa2_g2split");
    let mut batch: usize = 4;
    let mut ctx: usize = 64;
    let mut splits: Vec<usize> = vec![1, 8];
    let mut direct = true;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                outdir = args[i + 1].clone();
                i += 2;
            }
            "--batch" => {
                batch = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ctx" => {
                ctx = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--splits" => {
                splits = args[i + 1]
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect();
                i += 2;
            }
            "--direct" => {
                direct = args[i + 1].parse::<u32>().unwrap() != 0;
                i += 2;
            }
            _ => i += 1,
        }
    }
    std::fs::create_dir_all(&outdir).unwrap();
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("fa2_fp8 flag on: {}", gpu.flags.gfx12_fa2_fp8_enabled());
    // Deterministic inputs, same construction as tmp_fa2_fp8_oracle shape 0.
    let mut rng = Rng(0x9E3779B97F4A7C15u64 ^ 0xBF58476D1CE4E5B9);
    let nblk = ctx * NKV * BPH;
    let mut kv = vec![0u8; nblk * 34];
    for (bi, blk) in kv.chunks_mut(34).enumerate() {
        let scale: f32 = 0.02 + ((bi * 7) % 13) as f32 * 0.005;
        let h = half_bits(scale);
        blk[0] = (h & 0xFF) as u8;
        blk[1] = (h >> 8) as u8;
        for b in blk[2..].iter_mut() {
            *b = (rng.below(253) as i32 - 126) as i8 as u8;
        }
    }
    let mut kv2 = kv.clone();
    for (idx, b) in kv2.iter_mut().enumerate() {
        if idx % 34 >= 2 {
            *b = (*b).wrapping_add(11);
        }
    }
    let qn = batch * NH * HD;
    let mut q = vec![0f32; qn];
    for v in q.iter_mut() {
        *v = (rng.below(20001) as f32 / 10000.0 - 1.0) * 4.0;
    }
    let pos: Vec<i32> = (0..batch).map(|qq| (ctx - batch + qq) as i32).collect();
    let mut pos_bytes = vec![0u8; batch * 4];
    for (qq, p) in pos.iter().enumerate() {
        pos_bytes[qq * 4..qq * 4 + 4].copy_from_slice(&p.to_le_bytes());
    }
    let d_k = gpu.upload_raw(&kv, &[kv.len()]).unwrap();
    let d_v = gpu.upload_raw(&kv2, &[kv2.len()]).unwrap();
    let d_q = gpu.upload_f32(&q, &[batch, NH, HD]).unwrap();
    let d_pos = gpu.upload_raw(&pos_bytes, &[pos_bytes.len()]).unwrap();
    if direct {
        let d_out = gpu.zeros(&[qn], DType::F32).unwrap();
        gpu.attention_q8_0_fa2_gqa_gfx1201(&d_q, &d_k, &d_v, &d_out, &d_pos, NH, NKV, HD, ctx, batch)
            .unwrap_or_else(|e| panic!("fa2 direct launch: {e:?}"));
        let o: Vec<f32> = gpu.download_f32(&d_out).unwrap();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(o.as_ptr() as *const u8, qn * 4) };
        let path = format!("{outdir}/direct_b{batch}_c{ctx}.f32");
        std::fs::write(&path, bytes).unwrap();
        eprintln!("wrote {path}");
    }
    for ns in splits {
        let nrec = ns * batch * NH * (HD + 2);
        let d_partials = gpu.zeros(&[nrec], DType::F32).unwrap();
        let d_out = gpu.zeros(&[qn], DType::F32).unwrap();
        gpu.attention_q8_0_fa2_gqa_split_gfx1201_bench(
            &d_q, &d_k, &d_v, &d_out, &d_pos, &d_partials, NH, NKV, HD, batch, ns,
        )
        .unwrap_or_else(|e| panic!("fa2 split launch ns={ns}: {e:?}"));
        let p: Vec<f32> = gpu.download_f32(&d_partials).unwrap();
        let o: Vec<f32> = gpu.download_f32(&d_out).unwrap();
        let pb: &[u8] =
            unsafe { std::slice::from_raw_parts(p.as_ptr() as *const u8, p.len() * 4) };
        let ob: &[u8] =
            unsafe { std::slice::from_raw_parts(o.as_ptr() as *const u8, o.len() * 4) };
        let pp = format!("{outdir}/split{ns}_b{batch}_c{ctx}.partials.f32");
        let mp = format!("{outdir}/split{ns}_b{batch}_c{ctx}.merged.f32");
        std::fs::write(&pp, pb).unwrap();
        std::fs::write(&mp, ob).unwrap();
        eprintln!("wrote {pp} + {mp}");
    }
}
