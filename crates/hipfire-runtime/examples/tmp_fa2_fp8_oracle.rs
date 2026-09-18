// SPDX-License-Identifier: Apache-2.0
// Temporary stage-a tolerance oracle: fp8-plane FA2 vs f16 FA2, gated O.
// Two processes (HIPFIRE_GFX12_FA2_FP8=0/1), deterministic inputs per shape,
// gated O written to files; compared offline in f64.
//
//   cargo run --release -p hipfire-runtime --features lab --example tmp_fa2_fp8_oracle -- --all --out /tmp/fa2_ora/f16
//   HIPFIRE_GFX12_FA2_FP8=1 ... --out /tmp/fa2_ora/fp8
//
// Shapes (batch,ctx): (4,256),(8,512),(32,2048),(64,4096),(128,8192),(256,16384),(512,32768).

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
    let mut outdir = String::from("/tmp/fa2_ora");
    let mut only: Option<usize> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                outdir = args[i + 1].clone();
                i += 2;
            }
            "--shape" => {
                only = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            _ => i += 1,
        }
    }
    let shapes: [(usize, usize); 7] = [
        (4, 256),
        (8, 512),
        (32, 2048),
        (64, 4096),
        (128, 8192),
        (256, 16384),
        (512, 32768),
    ];
    std::fs::create_dir_all(&outdir).unwrap();
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "fa2_fp8 flag on: {}",
        gpu.flags.gfx12_fa2_fp8_enabled()
    );
    for (si, &(batch, ctx)) in shapes.iter().enumerate() {
        if let Some(o) = only {
            if o != si {
                continue;
            }
        }
        let mut rng = Rng(0x9E3779B97F4A7C15u64 ^ (si as u64 * 0xBF58476D1CE4E5B9));
        let scale1 = std::env::var("ORA_SCALE1").as_deref() == Ok("1");
        let smallcodes = std::env::var("ORA_SMALLCODES").as_deref() == Ok("1");
        // SCALE1: all block scales = 1.0 (exact); SMALLCODES: codes in
        // +-10 (exact in e4m3 AND f16). Both on => fp8 must match f16
        // bit-near-exactly (~1e-3); any large error is structural.
        let nblk = ctx * NKV * BPH;
        let mut kv = vec![0u8; nblk * 34];
        for (bi, blk) in kv.chunks_mut(34).enumerate() {
            let scale: f32 = if scale1 {
                1.0
            } else {
                0.02 + ((bi * 7 + si * 13) % 13) as f32 * 0.005
            };
            let h = half_bits(scale);
            blk[0] = (h & 0xFF) as u8;
            blk[1] = (h >> 8) as u8;
            for b in blk[2..].iter_mut() {
                let c = if smallcodes {
                    (rng.below(21) as i32 - 10) as i8 as u8
                } else {
                    (rng.below(253) as i32 - 126) as i8 as u8
                };
                *b = c;
            }
        }
        let mut kv2 = kv.clone();
        // V codes differ from K (except in the exactness probe, where the
        // +11 would push codes off the e4m3-exact lattice).
        let voff: u8 = if smallcodes { 0 } else { 11 };
        for (idx, b) in kv2.iter_mut().enumerate() {
            if idx % 34 >= 2 {
                *b = (*b).wrapping_add(voff);
            }
        }
        // Q magnitude knob below (QSCALE); base count here.
        let qn = batch * NH * HD;
        // QSCALE: shrink Q to tame score magnitudes (IID synthetic Q at
        // full scale makes winner-take-all scores where any rounding flips
        // rows; real activations are softer).
        let qscale: f32 = std::env::var("ORA_QSCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        // Q: ~(u-0.5)*8*qscale, deterministic.
        let qn = batch * NH * HD;
        let mut q = vec![0f32; qn];
        for v in q.iter_mut() {
            *v = (rng.below(20001) as f32 / 10000.0 - 1.0) * 4.0 * qscale;
        }
        // positions: causal prefill, C-B+q.
        let pos: Vec<i32> = (0..batch).map(|qq| (ctx - batch + qq) as i32).collect();
        let mut pos_bytes = vec![0u8; batch * 4];
        for (qq, p) in pos.iter().enumerate() {
            pos_bytes[qq * 4..qq * 4 + 4].copy_from_slice(&p.to_le_bytes());
        }
        // gate: uniform(-2,2).
        let mut gate = vec![0f32; qn];
        for v in gate.iter_mut() {
            *v = (rng.below(20001) as f32 / 10000.0 - 1.0) * 2.0;
        }
        let d_k = gpu.upload_raw(&kv, &[kv.len()]).unwrap();
        let d_v = gpu.upload_raw(&kv2, &[kv2.len()]).unwrap();
        let d_q = gpu.upload_f32(&q, &[batch, NH, HD]).unwrap();
        let d_pos = gpu.upload_raw(&pos_bytes, &[pos_bytes.len()]).unwrap();
        let d_out = gpu.zeros(&[batch * NH * HD], DType::F32).unwrap();
        gpu.attention_q8_0_fa2_gqa_gfx1201(&d_q, &d_k, &d_v, &d_out, &d_pos, NH, NKV, HD, ctx, batch)
            .unwrap_or_else(|e| panic!("fa2 launch shape {si}: {e:?}"));
        let o: Vec<f32> = gpu.download_f32(&d_out).unwrap();
        assert_eq!(o.len(), qn);
        // gate on host in f64, write gated f32.
        let mut gated = vec![0f32; qn];
        let mut cksum = 0u64;
        for idx in 0..qn {
            let g = 1.0 / (1.0 + (-(gate[idx] as f64)).exp());
            let v = o[idx] as f64 * g;
            gated[idx] = v as f32;
            cksum = cksum.wrapping_add(gated[idx].to_bits() as u64);
        }
        let path = format!("{outdir}/shape{si}_b{batch}_c{ctx}.f32");
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(gated.as_ptr() as *const u8, qn * 4)
        };
        std::fs::write(&path, bytes).unwrap();
        eprintln!("shape{si} batch={batch} ctx={ctx} wrote {path} cksum={cksum:016x}");
    }
}
