//! Qwen4 symmetric MQ4 trunk on the dense IU4 route (fn-trunk-iu4, gfx1151;
//! developer example).
//!
//! Drives the production `Gpu` calls the Qwen4 trunk makes under
//! `HIPFIRE_QWEN4_TRUNK_IU4=1` (`layer_ops::project_trunk`):
//!     reserve_int4_mmq + rotate_x_mq_i4_batched (FWHT-256 + `block_i4_128`)
//!     + gemm_mq4g256v2_mmq_set_prequant_iu4 (V2B / `pm_v2b` / X5)
//!     + gemm_qwen4_trunk_zba_iu4 (GDN Z|beta|alpha V2B SET + split)
//! on every Flash-Next trunk shape and checks every byte against a host
//! emulator of the dense IU4 math:
//!   producer  x * s1, FWHT butterflies (strides 1..128), * 1/16 * s2; per
//!             128-block amax, two candidates (amax/7)*0.5*{12/7, 2} chosen by
//!             strict-< MSE (lane fma chains, xor-butterfly sum), rint/clamp
//!             [-8, 7], exact code sum;
//!   GEMM      per output and ascending K128 half h: C_h = exact int sum of
//!             (w - 8) * x codes, sum = fma(RN(float(sc_h) * d_h), float(C_h), sum).
//! The A4 activations, every output and (zba) the beta / alpha scatter must
//! match bitwise. Output rows are poisoned first; the bytes past each output's
//! extent must stay poison.
//!
//! CLI: [--x DIR] [--w DIR] [--layer L] [--tokens N[,N..]]
//!   --x DIR: fn-trunk-cap capture (L{L}.{gin,gout,ain,aout}.bin, F32 rows);
//!            default: deterministic synthetic activations with outliers.
//!   --w DIR: sym QT44 payloads (`<tensor name>.bin`, e.g. the GPTQ solve);
//!            default: synthetic symmetric weights.
//! One JSON line per case on stdout; exit 1 on any mismatch.

use rdna_compute::{DType, Gpu, GpuTensor};

const GB: usize = 136;
const POISON: u32 = 0x7fc0_dead;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let (a, b) = (self.unit().max(1e-7), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let v = m as f32 * (1.0 / 16_777_216.0);
            return if s != 0 { -v } else { v };
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | (((e - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Truncating f32 -> f16 (hipfire float16 encoder), normal range only.
fn f32_to_f16_trunc(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let a = x.abs();
    if a == 0.0 {
        return s;
    }
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&e), "synthetic scale out of f16 normal range");
    s | ((e as u16) << 10) | (((b >> 13) & 0x3ff) as u16)
}

/// Synthetic symmetric QT44 rows (zp = -8*sc, f16-exact).
fn synth_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed | 1);
    let mut out = vec![0u8; m * k / 256 * GB];
    for g in out.chunks_exact_mut(GB) {
        for h in 0..2 {
            let sc = f32_to_f16_trunc(0.001 + 0.03 * rng.unit());
            let zp = f32_to_f16_trunc(-8.0 * f16_bits_to_f32(sc));
            g[4 * h..4 * h + 2].copy_from_slice(&sc.to_le_bytes());
            g[4 * h + 2..4 * h + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for b in &mut g[8..] {
            *b = (rng.next() & 0xff) as u8;
        }
    }
    out
}

fn synth_x(n: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    let col: Vec<f32> = (0..k).map(|_| (0.5 * rng.normal()).exp()).collect();
    (0..n * k)
        .map(|i| {
            let v = rng.normal() * col[i % k];
            // One large outlier per ~300 values: the A4 worst case.
            if rng.next() % 307 == 0 { v * 40.0 } else { v }
        })
        .collect()
}

/// Host FWHT-256 of one group, exactly the kernel's operation order.
fn fwht256(x: &[f32], s1: &[f32], s2: &[f32]) -> [f32; 256] {
    let mut v = [0f32; 256];
    for i in 0..256 {
        v[i] = x[i] * s1[i];
    }
    let mut h = 1;
    while h < 256 {
        for base in (0..256).step_by(2 * h) {
            for i in base..base + h {
                let (a, b) = (v[i], v[i + h]);
                v[i] = a + b;
                v[i + h] = a - b;
            }
        }
        h *= 2;
    }
    for i in 0..256 {
        v[i] = v[i] * 0.0625 * s2[i];
    }
    v
}

/// Host `block_i4_128` (two-candidate producer set) of one 128-block:
/// returns the 72 bytes.
fn quant_block(x: &[f32]) -> [u8; 72] {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut best_d = 1.0f32;
    if amax != 0.0 {
        let mut best_mse = 1.0e30f32;
        let base = (amax / 7.0) * 0.5;
        for mult in [f32::from_bits(0x3fdb_6db7), 2.0f32] {
            let dj = base * mult;
            let mut lanes = [0f32; 32];
            for (l, lane) in lanes.iter_mut().enumerate() {
                let mut mse = 0f32;
                for e in 0..4 {
                    let xv = x[4 * l + e];
                    let q = (xv / dj).round_ties_even().clamp(-8.0, 7.0);
                    let err = (-q).mul_add(dj, xv);
                    mse = err.mul_add(err, mse);
                }
                *lane = mse;
            }
            for off in [16, 8, 4, 2, 1] {
                let prev = lanes;
                for i in 0..32 {
                    lanes[i] = prev[i] + prev[i ^ off];
                }
            }
            if lanes[0] < best_mse {
                best_mse = lanes[0];
                best_d = dj;
            }
        }
    }
    let mut out = [0u8; 72];
    let mut s = 0i32;
    let mut q = [0i32; 128];
    for e in 0..128 {
        q[e] = if amax == 0.0 { 0 } else { (x[e] / best_d).round_ties_even().clamp(-8.0, 7.0) as i32 };
        s += q[e];
    }
    out[0..4].copy_from_slice(&best_d.to_le_bytes());
    out[4..8].copy_from_slice(&s.to_le_bytes());
    for b in 0..64 {
        out[8 + b] = ((q[2 * b] & 15) | ((q[2 * b + 1] & 15) << 4)) as u8;
    }
    out
}

/// Host A4 activations `[K/128][N]` of 72-byte blocks.
fn host_xq(x: &[f32], n: usize, k: usize, s1: &[f32], s2: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; k / 128 * n * 72];
    for t in 0..n {
        for g in 0..k / 256 {
            let r = fwht256(&x[t * k + g * 256..t * k + g * 256 + 256], s1, s2);
            for h in 0..2 {
                let blk = quant_block(&r[h * 128..h * 128 + 128]);
                let at = ((2 * g + h) * n + t) * 72;
                out[at..at + 72].copy_from_slice(&blk);
            }
        }
    }
    out
}

/// Host dense IU4 SET of `m x k` sym QT44 `w` on A4 `xq` (`[K/128][n]`):
/// Y token-major `[n][m]`.
fn host_gemm(w: &[u8], xq: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let halves = k / 128;
    // Decode activation codes / scales once: [n][k] i8, [n][halves] f32.
    let mut xc = vec![0i8; n * k];
    let mut xd = vec![0f32; n * halves];
    for h in 0..halves {
        for t in 0..n {
            let b = &xq[(h * n + t) * 72..(h * n + t) * 72 + 72];
            xd[t * halves + h] = f32::from_le_bytes(b[0..4].try_into().unwrap());
            for e in 0..64 {
                let byte = b[8 + e];
                xc[t * k + h * 128 + 2 * e] = (((byte & 15) as i8) << 4) >> 4;
                xc[t * k + h * 128 + 2 * e + 1] = ((byte as i8) >> 4) as i8;
            }
        }
    }
    let row_bytes = k / 256 * GB;
    let mut y = vec![0f32; n * m];
    let threads = std::thread::available_parallelism().map_or(8, |p| p.get()).min(32);
    let rows_per = m.div_ceil(threads);
    let cols: Vec<Vec<f32>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|ti| {
                let (xc, xd) = (&xc, &xd);
                scope.spawn(move || {
                    let r0 = (ti * rows_per).min(m);
                    let r1 = ((ti + 1) * rows_per).min(m);
                    let mut part = vec![0f32; (r1 - r0) * n];
                    let mut wc = vec![0i8; k];
                    let mut sc = vec![0f32; halves];
                    for r in r0..r1 {
                        let row = &w[r * row_bytes..(r + 1) * row_bytes];
                        for g in 0..k / 256 {
                            let grp = &row[g * GB..(g + 1) * GB];
                            for h in 0..2 {
                                sc[2 * g + h] = f16_bits_to_f32(u16::from_le_bytes([grp[4 * h], grp[4 * h + 1]]));
                                for e in 0..64 {
                                    let byte = grp[8 + 64 * h + e];
                                    wc[g * 256 + h * 128 + 2 * e] = (byte & 15) as i8 - 8;
                                    wc[g * 256 + h * 128 + 2 * e + 1] = (byte >> 4) as i8 - 8;
                                }
                            }
                        }
                        for t in 0..n {
                            let xt = &xc[t * k..(t + 1) * k];
                            let mut sum = 0f32;
                            for h in 0..halves {
                                let mut c = 0i32;
                                for e in h * 128..h * 128 + 128 {
                                    c += wc[e] as i32 * xt[e] as i32;
                                }
                                sum = (sc[h] * xd[t * halves + h]).mul_add(c as f32, sum);
                            }
                            part[(r - r0) * n + t] = sum;
                        }
                    }
                    part
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (ti, part) in cols.iter().enumerate() {
        let r0 = (ti * rows_per).min(m);
        let rows = part.len() / n.max(1);
        for rr in 0..rows {
            for t in 0..n {
                y[t * m + r0 + rr] = part[rr * n + t];
            }
        }
    }
    y
}

fn poisoned(gpu: &mut Gpu, elems: usize) -> GpuTensor {
    let bytes: Vec<u8> = std::iter::repeat(POISON.to_le_bytes()).take(elems + 1024).flatten().collect();
    let mut t = gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload poison");
    t.dtype = DType::F32;
    t.shape = vec![elems + 1024];
    t
}

/// Compare a device output `[n][m]` (plus 1024 poison words) with the host.
fn compare(gpu: &Gpu, dev: &GpuTensor, host: &[f32]) -> (usize, usize, f32) {
    let bytes = gpu.download_raw_bytes(dev).expect("download");
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut bad = 0;
    let mut max_abs = 0f32;
    for (i, h) in host.iter().enumerate() {
        if words[i] != h.to_bits() {
            bad += 1;
            max_abs = max_abs.max((f32::from_bits(words[i]) - h).abs());
        }
    }
    let tail = words[host.len()..host.len() + 1024].iter().filter(|w| **w != POISON).count();
    (bad, tail, max_abs)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let xdir = arg("--x");
    let wdir = arg("--w");
    let layer: usize = arg("--layer").map_or(0, |v| v.parse().unwrap());
    let tokens: Vec<usize> = arg("--tokens")
        .map_or(vec![1536, 512, 1280], |v| v.split(',').map(|s| s.parse().unwrap()).collect());
    let mut gpu = Gpu::init().expect("gpu init");
    gpu.mq4v2_symmetric = true;
    assert_eq!(gpu.arch, "gfx1151", "fn-trunk-iu4 oracle targets gfx1151");
    gpu.ensure_mq_signs().expect("mq signs");
    let s1 = gpu.download_f32(gpu.scratch.mq_signs1.as_ref().unwrap()).expect("signs1");
    let s2 = gpu.download_f32(gpu.scratch.mq_signs2.as_ref().unwrap()).expect("signs2");
    let attn = layer % 4 == 3;
    let pre = |t: &str| {
        format!(
            "model.language_model.layers.{layer}.{}.{t}.weight",
            if attn { "self_attn" } else { "linear_attn" }
        )
    };
    // (case, tag, tensors (name, m), k); a GDN "zba" case folds z|b|a.
    let cases: Vec<(&str, &str, Vec<(String, usize)>, usize)> = if attn {
        vec![
            ("q", "ain", vec![(pre("q_proj"), 12288)], 2560),
            ("k", "ain", vec![(pre("k_proj"), 512)], 2560),
            ("v", "ain", vec![(pre("v_proj"), 512)], 2560),
            ("indexer", "ain", vec![(pre("indexer.index_qk_proj"), 640)], 2560),
            ("o", "aout", vec![(pre("o_proj"), 2560)], 6144),
        ]
    } else {
        vec![
            ("qkv", "gin", vec![(pre("in_proj_qkv"), 10240)], 2560),
            (
                "zba",
                "gin",
                vec![(pre("in_proj_z"), 6144), (pre("in_proj_b"), 48), (pre("in_proj_a"), 48)],
                2560,
            ),
            ("out", "gout", vec![(pre("out_proj"), 2560)], 6144),
        ]
    };
    let mut failures = 0;
    for &n in &tokens {
        for (ci, (case, tag, tensors, k)) in cases.iter().enumerate() {
            let k = *k;
            let x: Vec<f32> = match &xdir {
                Some(dir) => {
                    let raw = std::fs::read(format!("{dir}/L{layer}.{tag}.bin")).expect("capture file");
                    raw[..n * k * 4].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
                }
                None => synth_x(n, k, 0x9e37 + k as u64),
            };
            let weights: Vec<Vec<u8>> = tensors
                .iter()
                .enumerate()
                .map(|(i, (name, m))| match &wdir {
                    Some(dir) => std::fs::read(format!("{dir}/{name}.bin")).expect("payload"),
                    None => synth_weights(*m, k, 0x1234 + (ci * 7 + i) as u64),
                })
                .collect();
            for (w, (name, m)) in weights.iter().zip(tensors) {
                assert_eq!(w.len(), m * k / 256 * GB, "{name} payload size");
            }
            let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload x");
            let res = gpu.reserve_int4_mmq(k, n).expect("reserve");
            let prep = gpu.rotate_x_mq_i4_batched(&d_x, None, None, res, k, n).expect("producer");
            let xq_ptr = gpu.int4_mmq_prepared_ptr(&prep, k, n).expect("prepared");
            let xq_len = k / 128 * n * 72;
            let xq_view = unsafe { hip_bridge::DeviceBuffer::from_raw(xq_ptr, xq_len) };
            let mut xq_dev = vec![0u8; xq_len];
            gpu.hip.memcpy_dtoh(&mut xq_dev, &xq_view).expect("download xq");
            let xq_host = host_xq(&x, n, k, &s1, &s2);
            let xq_bad = xq_dev.chunks_exact(72).zip(xq_host.chunks_exact(72)).filter(|(a, b)| a != b).count();
            let d_w: Vec<GpuTensor> = if *case == "zba" {
                let rb = k / 256 * GB;
                let mut fold = vec![0u8; (6144 + 256) * rb];
                fold[..6144 * rb].copy_from_slice(&weights[0]);
                fold[6144 * rb..6192 * rb].copy_from_slice(&weights[1]);
                fold[6192 * rb..6240 * rb].copy_from_slice(&weights[2]);
                vec![gpu.upload_raw(&fold, &[fold.len()]).expect("upload fold")]
            } else {
                vec![gpu.upload_raw(&weights[0], &[weights[0].len()]).expect("upload w")]
            };
            let outs: Vec<GpuTensor> = tensors.iter().map(|(_, m)| poisoned(&mut gpu, n * m)).collect();
            let t0 = std::time::Instant::now();
            let route = if *case == "zba" {
                let ok = gpu
                    .gemm_qwen4_trunk_zba_iu4(&d_w[0], &prep, &outs[0], &outs[1], &outs[2], 6144, k, n)
                    .expect("zba");
                if !ok {
                    // The trunk's fallback: the three SETs on their own rows.
                    let rb = k / 256 * GB;
                    for (i, (_, m)) in tensors.iter().enumerate() {
                        let row0 = [0, 6144, 6192][i];
                        let w = d_w[0].sub_offset(row0 * rb, m * rb);
                        let xq = gpu.int4_mmq_prepared_ptr(&prep, k, n).unwrap();
                        gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&w, xq, &outs[i], *m, k, n).expect("set");
                    }
                }
                if ok { "zba_v2b_scatter" } else { "set_x3" }
            } else {
                gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&d_w[0], xq_ptr, &outs[0], tensors[0].1, k, n)
                    .expect("set");
                "set_prequant_iu4"
            };
            gpu.hip.device_synchronize().expect("sync");
            let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;
            let mut report = Vec::new();
            for (i, (name, m)) in tensors.iter().enumerate() {
                let y = host_gemm(&weights[i], &xq_dev, *m, k, n);
                let (bad, tail, max_abs) = compare(&gpu, &outs[i], &y);
                failures += usize::from(bad != 0 || tail != 0);
                report.push(format!(
                    "{{\"tensor\":\"{name}\",\"m\":{m},\"mismatch\":{bad},\"of\":{},\"poison_overwrites\":{tail},\"max_abs\":{max_abs}}}",
                    n * m
                ));
            }
            failures += usize::from(xq_bad != 0);
            println!(
                "{{\"case\":\"{case}\",\"layer\":{layer},\"n\":{n},\"k\":{k},\"route\":\"{route}\",\"xq_block_mismatch\":{xq_bad},\"xq_blocks\":{},\"first_call_ms\":{gpu_ms:.2},\"x\":\"{}\",\"w\":\"{}\",\"outputs\":[{}]}}",
                xq_len / 72,
                xdir.as_deref().unwrap_or("synthetic"),
                wdir.as_deref().unwrap_or("synthetic"),
                report.join(",")
            );
            for t in outs.into_iter().chain(d_w).chain([d_x]) {
                gpu.free_tensor(t).expect("free");
            }
        }
    }
    eprintln!("qwen4_trunk_iu4: {failures} failing checks");
    std::process::exit(i32::from(failures != 0));
}
