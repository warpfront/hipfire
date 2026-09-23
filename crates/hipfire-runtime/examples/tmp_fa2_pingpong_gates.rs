// Temporary FA2 ping-pong gate harness (throwaway; UNTRACKED, do not commit).
// 7 synthetic (batch,ctx) shapes with the SAME deterministic data as
// tmp_fa2_fp8_oracle (same Rng seeds/code → gated-O checksums must match
// ora_base), plus 3 truncated-tile shapes (seq_len % 64 != 0 exercising
// null K/V blocks, do=false subtiles, qok=false rows).
// Modes: --dump OUTDIR (gated O + cksum per shape) and --time (median
// per-launch us over timed iters, N launches + 1 download per iter).
// f16 path only (FP8 flag must be OFF; asserts it).

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

fn median_f64(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn build_inputs(si: usize, batch: usize, ctx: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>) {
    let mut rng = Rng(0x9E3779B97F4A7C15u64 ^ (si as u64).wrapping_mul(0xBF58476D1CE4E5B9));
    let nblk = ctx * NKV * BPH;
    let mut kv = vec![0u8; nblk * 34];
    for (bi, blk) in kv.chunks_mut(34).enumerate() {
        let scale: f32 = 0.02 + ((bi * 7 + si * 13) % 13) as f32 * 0.005;
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
    let qscale: f32 = std::env::var("ORA_QSCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let qn = batch * NH * HD;
    let mut q = vec![0f32; qn];
    for v in q.iter_mut() {
        *v = (rng.below(20001) as f32 / 10000.0 - 1.0) * 4.0 * qscale;
    }
    let pos: Vec<i32> = (0..batch).map(|qq| (ctx - batch + qq) as i32).collect();
    let mut pos_bytes = vec![0u8; batch * 4];
    for (qq, p) in pos.iter().enumerate() {
        pos_bytes[qq * 4..qq * 4 + 4].copy_from_slice(&p.to_le_bytes());
    }
    let mut gate = vec![0f32; qn];
    for v in gate.iter_mut() {
        *v = (rng.below(20001) as f32 / 10000.0 - 1.0) * 2.0;
    }
    (kv, kv2, q, pos_bytes, gate)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mode = "dump";
    let mut outdir = String::from("/tmp/fa2_ping");
    let mut only: Option<usize> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dump" => {
                mode = "dump";
                outdir = args[i + 1].clone();
                i += 2;
            }
            "--time" => {
                mode = "time";
                i += 1;
            }
            "--shape" => {
                only = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            _ => i += 1,
        }
    }
    // 0..6 = oracle-identical; 7..9 = truncated tiles.
    let shapes: [(usize, usize); 10] = [
        (4, 256),
        (8, 512),
        (32, 2048),
        (64, 4096),
        (128, 8192),
        (256, 16384),
        (512, 32768),
        (3, 100),
        (6, 2000),
        (8, 5000),
    ];
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(
        !gpu.flags.gfx12_fa2_fp8_enabled(),
        "f16 gate harness requires FP8 flag OFF"
    );
    if mode == "dump" {
        std::fs::create_dir_all(&outdir).unwrap();
    }
    for (si, &(batch, ctx)) in shapes.iter().enumerate() {
        if let Some(o) = only {
            if o != si {
                continue;
            }
        }
        let (kv, kv2, q, pos_bytes, gate) = build_inputs(si, batch, ctx);
        let qn = batch * NH * HD;
        let d_k = gpu.upload_raw(&kv, &[kv.len()]).unwrap();
        let d_v = gpu.upload_raw(&kv2, &[kv2.len()]).unwrap();
        let d_q = gpu.upload_f32(&q, &[batch, NH, HD]).unwrap();
        let d_pos = gpu.upload_raw(&pos_bytes, &[pos_bytes.len()]).unwrap();
        let d_out = gpu.zeros(&[batch * NH * HD], DType::F32).unwrap();
        if mode == "time" {
            // Warmup, then iters of (N launches + 1 download); per-launch median.
            let n = match si {
                0 | 1 | 7 => 200,
                2 | 3 | 8 | 9 => 50,
                4 => 20,
                _ => 8,
            };
            for _ in 0..3 {
                gpu.attention_q8_0_fa2_gqa_gfx1201(
                    &d_q, &d_k, &d_v, &d_out, &d_pos, NH, NKV, HD, ctx, batch,
                )
                .unwrap();
            }
            let _ = gpu.download_f32(&d_out).unwrap();
            let mut samples = Vec::with_capacity(11);
            for _ in 0..11 {
                let t = std::time::Instant::now();
                for _ in 0..n {
                    gpu.attention_q8_0_fa2_gqa_gfx1201(
                        &d_q, &d_k, &d_v, &d_out, &d_pos, NH, NKV, HD, ctx, batch,
                    )
                    .unwrap();
                }
                let _ = gpu.download_f32(&d_out).unwrap();
                samples.push(t.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
            let med = median_f64(&mut samples);
            let mn = samples.iter().cloned().fold(f64::INFINITY, f64::min);
            println!("shape{si} batch={batch} ctx={ctx} median_us={med:.2} min_us={mn:.2}");
        } else {
            gpu.attention_q8_0_fa2_gqa_gfx1201(
                &d_q, &d_k, &d_v, &d_out, &d_pos, NH, NKV, HD, ctx, batch,
            )
            .unwrap_or_else(|e| panic!("fa2 launch shape {si}: {e:?}"));
            let o: Vec<f32> = gpu.download_f32(&d_out).unwrap();
            assert_eq!(o.len(), qn);
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
}
