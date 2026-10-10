// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//
// The Gemma 4 26B-A4B gfx1151 kernels against an f64 CPU reference, at the
// model's shapes (head_dim 256 / 16q:8kv / window 1024 sliding, head_dim 512 /
// 16q:2kv full; routed experts M 2816, K 704, top-8):
//   * decode Q8_0 flash attention (`attention_flash_q8_0_windowed`); run with
//     HIPFIRE_GFX1151_Q8_DECODE_ATTN_GQA_GEMMA=0 to compare the reference tile;
//   * batched prefill attention (`attention_q8_0_prefill_gqa_wmma`);
//   * the HFQ4-G128 routed down projection, per-row and grouped;
// plus eager timings of the decode attention at 290 and 32768 keys. Exit 1
// when a relative error exceeds its bound (1e-4 F32 routes, 2e-3 F16 WMMA).
// Run: cargo run --release -p rdna-compute --example gemma4_kernel_f64_check --features lab

use rdna_compute::{DType, Gpu};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn f16_bits(x: f32) -> u16 {
    half_from_f32(x)
}

// Minimal round-to-nearest f32 -> f16 for normal positive ranges used here.
fn half_from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    let m = (man + 0x1000) >> 13;
    sign | ((exp as u16) << 10) + m as u16
}

fn f16_to_f64(h: u16) -> f64 {
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f64;
    let v = (1.0 + man / 1024.0) * 2f64.powi(exp - 15);
    if h & 0x8000 != 0 { -v } else { v }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut ok = true;
    for &(nh, nkv, hd, window, seq) in &[
        (16usize, 8usize, 256usize, 1024i32, 3000usize),
        (16, 8, 256, 1024, 700),
        (16, 2, 512, 0, 3000),
        (16, 2, 512, 0, 129),
    ] {
        let max_seq = 4096usize;
        let mut rng = Rng(seq as u64 * 31 + hd as u64);
        let nb = hd / 32;
        let row = nkv * nb * 34;
        let mut kv = [vec![0u8; max_seq * row], vec![0u8; max_seq * row]];
        for cache in kv.iter_mut() {
            for blk in cache.chunks_mut(34) {
                let s = f16_bits((0.01 + 0.02 * rng.next()) as f32);
                blk[0] = s as u8;
                blk[1] = (s >> 8) as u8;
                for c in &mut blk[2..] {
                    *c = ((rng.next() * 254.0) as i32 - 127) as i8 as u8;
                }
            }
        }
        let qh: Vec<f32> = (0..nh * hd).map(|_| (rng.next() * 2.0 - 1.0) as f32).collect();
        let deq = |cache: &[u8], t: usize, kh: usize, d: usize| -> f64 {
            let off = t * row + (kh * nb + d / 32) * 34;
            f16_to_f64(u16::from_le_bytes([cache[off], cache[off + 1]]))
                * (cache[off + 2 + d % 32] as i8) as f64
        };
        let lo = if window > 0 { seq.saturating_sub(window as usize) } else { 0 };
        let scale = 1.0 / (hd as f64).sqrt();
        let mut want = vec![0f64; nh * hd];
        for h in 0..nh {
            let kh = h / (nh / nkv);
            let sc: Vec<f64> = (lo..seq)
                .map(|t| (0..hd).map(|d| qh[h * hd + d] as f64 * deq(&kv[0], t, kh, d)).sum::<f64>() * scale)
                .collect();
            let m = sc.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for d in 0..hd {
                want[h * hd + d] =
                    (lo..seq).zip(&e).map(|(t, w)| w * deq(&kv[1], t, kh, d)).sum::<f64>() / z;
            }
        }
        let q = gpu.upload_f32(&qh, &[nh * hd]).unwrap();
        let k = gpu.upload_raw(&kv[0], &[kv[0].len()]).unwrap();
        let v = gpu.upload_raw(&kv[1], &[kv[1].len()]).unwrap();
        let out = gpu.zeros(&[nh * hd], DType::F32).unwrap();
        let pos = ((seq - 1) as i32).to_le_bytes();
        let pos_t = gpu.upload_raw(&pos, &[1]).unwrap();
        let partials = gpu.zeros(&[nh * max_seq.div_ceil(16) * (2 + hd)], DType::F32).unwrap();
        gpu.attention_flash_q8_0_windowed(
            &q, &k, &v, &out, &pos_t.buf, seq, nh, nkv, hd, max_seq, &partials, window,
        )
        .unwrap();
        let got = gpu.download_f32(&out).unwrap();
        let peak = want.iter().fold(0f64, |a, b| a.max(b.abs()));
        let err = got.iter().zip(&want).fold(0f64, |a, (g, w)| a.max((*g as f64 - w).abs()));
        let rel = err / peak;
        println!("hd={hd} q:kv={nh}:{nkv} window={window} seq={seq}: max|err|/peak = {rel:.3e}");
        ok &= rel < 1e-4;
    }
    // Batched prefill attention (GQA WMMA route on gfx11) against f64.
    for &(nh, nkv, hd, window, seq, nb) in &[
        (16usize, 8usize, 256usize, 1024i32, 3000usize, 256usize),
        (16, 2, 512, 0, 3000, 200),
        (16, 2, 512, 0, 40, 40),
        (16, 8, 256, 1024, 70, 64),
    ] {
        let max_seq = 4096usize;
        let mut rng = Rng(seq as u64 * 7 + hd as u64 + nb as u64);
        let nbk = hd / 32;
        let row = nkv * nbk * 34;
        let mut kv = [vec![0u8; max_seq * row], vec![0u8; max_seq * row]];
        for cache in kv.iter_mut() {
            for blk in cache.chunks_mut(34) {
                let s = f16_bits((0.01 + 0.02 * rng.next()) as f32);
                blk[0] = s as u8;
                blk[1] = (s >> 8) as u8;
                for c in &mut blk[2..] {
                    *c = ((rng.next() * 254.0) as i32 - 127) as i8 as u8;
                }
            }
        }
        let qh: Vec<f32> = (0..nb * nh * hd).map(|_| (rng.next() * 2.0 - 1.0) as f32 * 4.0).collect();
        let deq = |cache: &[u8], t: usize, kh: usize, d: usize| -> f64 {
            let off = t * row + (kh * nbk + d / 32) * 34;
            f16_to_f64(u16::from_le_bytes([cache[off], cache[off + 1]]))
                * (cache[off + 2 + d % 32] as i8) as f64
        };
        let scale = 1.0 / (hd as f64).sqrt();
        let pos0 = seq - nb;
        let mut want = vec![0f64; nb * nh * hd];
        for b in 0..nb {
            let p = pos0 + b;
            let lo = if window > 0 { (p + 1).saturating_sub(window as usize) } else { 0 };
            for h in 0..nh {
                let kh = h / (nh / nkv);
                let qo = (b * nh + h) * hd;
                let sc: Vec<f64> = (lo..=p)
                    .map(|t| (0..hd).map(|d| qh[qo + d] as f64 * deq(&kv[0], t, kh, d)).sum::<f64>() * scale)
                    .collect();
                let m = sc.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for d in 0..hd {
                    want[qo + d] = (lo..=p).zip(&e).map(|(t, w)| w * deq(&kv[1], t, kh, d)).sum::<f64>() / z;
                }
            }
        }
        let q = gpu.upload_f32(&qh, &[nb * nh * hd]).unwrap();
        let k = gpu.upload_raw(&kv[0], &[kv[0].len()]).unwrap();
        let v = gpu.upload_raw(&kv[1], &[kv[1].len()]).unwrap();
        let out = gpu.zeros(&[nb * nh * hd], DType::F32).unwrap();
        let pos: Vec<u8> = (pos0..seq).flat_map(|p| (p as i32).to_le_bytes()).collect();
        let pos_t = gpu.upload_raw(&pos, &[nb]).unwrap();
        gpu.attention_q8_0_prefill_gqa_wmma(&q, &k, &v, &out, &pos_t, nh, nkv, hd, nb, window).unwrap();
        let got = gpu.download_f32(&out).unwrap();
        let peak = want.iter().fold(0f64, |a, b| a.max(b.abs()));
        let err = got.iter().zip(&want).fold(0f64, |a, (g, w)| a.max((*g as f64 - w).abs()));
        println!("prefill attn hd={hd} q:kv={nh}:{nkv} window={window} seq={seq} rows={nb}: max|err|/peak = {:.3e}", err / peak);
        ok &= err / peak < 2e-3;
    }
    // HFQ4-G128 routed down (M 2816, K 704, k8) against f64.
    {
        let (m, k, n_exp) = (2816usize, 704usize, 16usize);
        let gpr = k.div_ceil(128);
        let mut rng = Rng(7);
        let mut experts = Vec::new();
        let mut host = Vec::new();
        for _ in 0..n_exp {
            let mut bytes = vec![0u8; m * gpr * 72];
            for grp in bytes.chunks_mut(72) {
                grp[0..4].copy_from_slice(&((0.01 + 0.02 * rng.next()) as f32).to_le_bytes());
                grp[4..8].copy_from_slice(&((rng.next() * 0.2 - 0.1) as f32).to_le_bytes());
                for b in &mut grp[8..] {
                    *b = (rng.next() * 256.0) as u8;
                }
            }
            experts.push(gpu.upload_raw(&bytes, &[bytes.len()]).unwrap());
            host.push(bytes);
        }
        let ptrs: Vec<u8> = experts.iter().flat_map(|t| (t.buf.as_ptr() as u64).to_le_bytes()).collect();
        let ptrs_t = gpu.upload_raw(&ptrs, &[ptrs.len()]).unwrap();
        let idx: Vec<i32> = vec![3, 0, 15, 7, 9, 1, 12, 5];
        let idx_t = gpu.upload_raw(&idx.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[8]).unwrap();
        let wts: Vec<f32> = (0..8).map(|_| rng.next() as f32).collect();
        let pes: Vec<f32> = (0..n_exp).map(|_| (0.5 + rng.next()) as f32).collect();
        let hid: Vec<f32> = (0..8 * k).map(|_| (rng.next() * 2.0 - 1.0) as f32).collect();
        let wts_t = gpu.upload_f32(&wts, &[8]).unwrap();
        let pes_t = gpu.upload_f32(&pes, &[n_exp]).unwrap();
        let hid_t = gpu.upload_f32(&hid, &[8 * k]).unwrap();
        let out = gpu.zeros(&[m], DType::F32).unwrap();
        gpu.gemv_hfq4g128_moe_down_residual_scaled_k8_indexed(
            &ptrs_t, &idx_t, &wts_t, &pes_t, &hid_t, &out, m, k,
        )
        .unwrap();
        let got = gpu.download_f32(&out).unwrap();
        let mut want = vec![0f64; m];
        for (r, w) in want.iter_mut().enumerate() {
            for (kr, &e) in idx.iter().enumerate() {
                let b = &host[e as usize];
                let mut acc = 0f64;
                for d in 0..k {
                    let off = (r * gpr + d / 128) * 72;
                    let sc = f32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as f64;
                    let z = f32::from_le_bytes(b[off + 4..off + 8].try_into().unwrap()) as f64;
                    let byte = b[off + 8 + (d % 128) / 2];
                    let q = if d % 2 == 0 { byte & 0xf } else { byte >> 4 } as f64;
                    acc += (sc * q + z) * hid[kr * k + d] as f64;
                }
                *w += wts[kr] as f64 * pes[e as usize] as f64 * acc;
            }
        }
        let peak = want.iter().fold(0f64, |a, b| a.max(b.abs()));
        let err = got.iter().zip(&want).fold(0f64, |a, (g, w)| a.max((*g as f64 - w).abs()));
        println!("moe down hfq4g128 m={m} k={k}: max|err|/peak = {:.3e}", err / peak);
        ok &= err / peak < 1e-4;
        gpu.hip.device_synchronize().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..50 {
            gpu.gemv_hfq4g128_moe_down_residual_scaled_k8_indexed(
                &ptrs_t, &idx_t, &wts_t, &pes_t, &hid_t, &out, m, k,
            )
            .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        println!("time moe down: {:.1} us/call", t.elapsed().as_secs_f64() * 1e6 / 50.0);
        // Grouped: 4 tiles of 16 slots; tile 1 unused, tile 3 half padding.
        let tiles: Vec<i32> = vec![3, -1, 9, 15];
        let mut slots = vec![-1i32; 64];
        let n_act = 40usize;
        let mut r2 = Rng(11);
        for (t, &e) in tiles.iter().enumerate() {
            if e < 0 { continue; }
            let n = if t == 3 { 8 } else { 16 };
            for j in 0..n { slots[t * 16 + j] = ((r2.next() * n_act as f64) as i32).min(n_act as i32 - 1); }
        }
        let act: Vec<f32> = (0..n_act * k).map(|_| (r2.next() * 2.0 - 1.0) as f32).collect();
        let i32b = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let tiles_t = gpu.upload_raw(&i32b(&tiles), &[4]).unwrap();
        let slots_t = gpu.upload_raw(&i32b(&slots), &[64]).unwrap();
        let act_t = gpu.upload_f32(&act, &[n_act * k]).unwrap();
        let y_t = gpu.zeros(&[64 * m], DType::F32).unwrap();
        gpu.gemv_hfq4g128_moe_down_grouped(&ptrs_t, &tiles_t, &slots_t, &pes_t, &act_t, &y_t, m, k, 64).unwrap();
        let y = gpu.download_f32(&y_t).unwrap();
        let (mut err, mut peak) = (0f64, 0f64);
        for (gr, &sl) in slots.iter().enumerate() {
            let e = tiles[gr / 16];
            if sl < 0 || e < 0 { continue; }
            let bw = &host[e as usize];
            for r in 0..m {
                let mut acc = 0f64;
                for d in 0..k {
                    let off = (r * gpr + d / 128) * 72;
                    let sc = f32::from_le_bytes(bw[off..off + 4].try_into().unwrap()) as f64;
                    let z = f32::from_le_bytes(bw[off + 4..off + 8].try_into().unwrap()) as f64;
                    let byte = bw[off + 8 + (d % 128) / 2];
                    let q = if d % 2 == 0 { byte & 0xf } else { byte >> 4 } as f64;
                    acc += (sc * q + z) * act[sl as usize * k + d] as f64;
                }
                let want = pes[e as usize] as f64 * acc;
                peak = peak.max(want.abs());
                err = err.max((y[gr * m + r] as f64 - want).abs());
            }
        }
        println!("moe down grouped: max|err|/peak = {:.3e}", err / peak);
        // gfx11 takes the F16-WMMA route (F16 weights and activations).
        ok &= err / peak < 1e-3;
    }
    // Timing at the 26B decode shapes (max_seq 34816, as the harness).
    for &(nh, nkv, hd, window) in &[(16usize, 8usize, 256usize, 1024i32), (16, 2, 512, 0)] {
        let max_seq = 34816usize;
        let row = nkv * hd / 32 * 34;
        let k = gpu.zeros(&[max_seq * row / 4], DType::F32).unwrap();
        let v = gpu.zeros(&[max_seq * row / 4], DType::F32).unwrap();
        let q = gpu.zeros(&[nh * hd], DType::F32).unwrap();
        let out = gpu.zeros(&[nh * hd], DType::F32).unwrap();
        let partials = gpu.zeros(&[nh * max_seq.div_ceil(16) * (2 + hd)], DType::F32).unwrap();
        for seq in [290usize, 32768] {
            let pos_t = gpu.upload_raw(&((seq - 1) as i32).to_le_bytes(), &[1]).unwrap();
            let mut run = |gpu: &mut Gpu| {
                gpu.attention_flash_q8_0_windowed(
                    &q, &k, &v, &out, &pos_t.buf, seq, nh, nkv, hd, max_seq, &partials, window,
                )
                .unwrap()
            };
            run(&mut gpu);
            gpu.hip.device_synchronize().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..50 {
                run(&mut gpu);
            }
            gpu.hip.device_synchronize().unwrap();
            println!(
                "time hd={hd} window={window} seq={seq}: {:.1} us/call (tile+reduce)",
                t.elapsed().as_secs_f64() * 1e6 / 50.0
            );
        }
    }
    std::process::exit(if ok { 0 } else { 1 });
}
