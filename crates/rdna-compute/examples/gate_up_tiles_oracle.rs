// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU byte oracle for `HIPFIRE_GFX12_GATE_UP_TILE_LOOP` (gfx1201).
//!
//! Runs the public `gemm_gate_up_hfq4g256_mq4v2` launcher with the tile-loop
//! flag ON and OFF over random MQ4G256V2 weights and adversarial activations
//! and requires every output word to be bitwise equal. Outputs are NaN-
//! sentinel prefilled with guard words after each; inputs are re-downloaded
//! and must be unchanged. Then times both arms cold on the Qwen3.8-27B MLP
//! shape (17408 + 17408 rows, K 5120).
//!
//! Usage: `gate_up_tiles_oracle [--no-time]`. Exits nonzero on any mismatch.

use rdna_compute::{Gpu, GpuTensor};
use std::sync::Arc;

type R<T> = Result<T, String>;
fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

const SENTINEL: u32 = 0x7FC0_DEAD;
const GUARD: usize = 64;
const NS: [usize; 7] = [1, 2, 3, 4, 5, 8, 16];

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
}

fn f16_bits(rng: &mut Rng) -> u16 {
    let sign = if rng.next() & 1 == 1 { 0x8000u16 } else { 0 };
    match rng.next() % 4 {
        0 => {
            // normal, magnitude ~ [2^-10, 2^-3]
            let v = (0.5 + rng.unit()) * (2.0f32).powi(-(3 + (rng.next() % 8) as i32));
            half::f16::from_f32(v).to_bits() | sign
        }
        1 => ((rng.next() % 0x3ff) as u16 + 1) | sign, // subnormal
        2 => sign,                                     // +-0
        _ => {
            // negative normal
            let v = (0.5 + rng.unit()) * (2.0f32).powi(-(2 + (rng.next() % 6) as i32));
            half::f16::from_f32(v).to_bits() | 0x8000
        }
    }
}

fn weights(rng: &mut Rng, rows: usize, k: usize) -> Vec<u8> {
    let gpr = k / 256;
    let mut out = vec![0u8; rows * gpr * 136];
    for g in 0..rows * gpr {
        let b = &mut out[g * 136..(g + 1) * 136];
        for i in 0..4 {
            b[i * 2..i * 2 + 2].copy_from_slice(&f16_bits(rng).to_le_bytes());
        }
        for x in b[8..].iter_mut() {
            *x = rng.next() as u8;
        }
    }
    out
}

fn activations(rng: &mut Rng, n: usize, k: usize) -> Vec<f32> {
    let mut x = vec![0f32; n * k];
    let mut i = 0;
    while i < x.len() {
        let c = rng.next() % 100;
        let sign = if rng.next() & 1 == 1 { -1.0f32 } else { 1.0 };
        let v = match c {
            0..=5 => 0.0,
            6..=11 => -0.0,
            12..=21 => sign * (1 + rng.next() % 1023) as f32 * (2.0f32).powi(-24),
            22..=31 if i + 1 < x.len() => {
                let m = (0.5 + rng.unit()) * (2.0f32).powi((rng.next() % 9) as i32 - 4);
                x[i] = m;
                x[i + 1] = -m;
                i += 2;
                continue;
            }
            _ => sign * (0.5 + rng.unit()) * (2.0f32).powi((rng.next() % 21) as i32 - 14),
        };
        x[i] = v;
        i += 1;
    }
    x
}

fn set_flag(gpu: &mut Gpu, on: bool) {
    Arc::make_mut(&mut gpu.flags).gfx12_gate_up_tile_loop = on;
}

fn out_tensor(gpu: &mut Gpu, len: usize) -> R<GpuTensor> {
    let host = vec![f32::from_bits(SENTINEL); len + GUARD];
    gpu.upload_f32(&host, &[len + GUARD]).map_err(err)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

struct Case {
    gate_m: usize,
    up_m: usize,
    k: usize,
}

fn run_arm(
    gpu: &mut Gpu,
    on: bool,
    c: &Case,
    n: usize,
    ag: &GpuTensor,
    au: &GpuTensor,
    x: &GpuTensor,
    x16: Option<&GpuTensor>,
) -> R<(Vec<u32>, Vec<u32>)> {
    set_flag(gpu, on);
    let yg = out_tensor(gpu, n * c.gate_m)?;
    let yu = out_tensor(gpu, n * c.up_m)?;
    match x16 {
        // The DFlash/MTP F16-producer entry (pre-converted F16 x, same kernel).
        Some(x16) => gpu
            .gemm_gate_up_mq4g256v2_wmma_f16(ag, au, x16, &yg, &yu, c.gate_m, c.up_m, c.k, n, false)
            .map_err(err)?,
        None => gpu
            .gemm_gate_up_hfq4g256_mq4v2(ag, au, x, &yg, &yu, c.gate_m, c.up_m, c.k, n)
            .map_err(err)?,
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let g = bits(&gpu.download_f32(&yg).map_err(err)?);
    let u = bits(&gpu.download_f32(&yu).map_err(err)?);
    gpu.free_tensor(yg).map_err(err)?;
    gpu.free_tensor(yu).map_err(err)?;
    Ok((g, u))
}

fn guard_ok(v: &[u32], body: usize) -> bool {
    v.len() == body + GUARD && v[body..].iter().all(|&w| w == SENTINEL)
}

fn check_case(gpu: &mut Gpu, c: &Case, seed: u64, cells: &mut u64, bad: &mut u64) -> R<()> {
    let mut rng = Rng(seed ^ ((c.gate_m as u64) << 20) ^ (c.k as u64) ^ 0x9E37_79B9_7F4A_7C15);
    let wg = weights(&mut rng, c.gate_m, c.k);
    let wu = weights(&mut rng, c.up_m, c.k);
    let ag = gpu.upload_raw(&wg, &[wg.len()]).map_err(err)?;
    let au = gpu.upload_raw(&wu, &[wu.len()]).map_err(err)?;
    for &n in &NS {
        let xh = activations(&mut rng, n, c.k);
        let x = gpu.upload_f32(&xh, &[xh.len()]).map_err(err)?;
        let x16_bits: Vec<u16> = xh
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let x16 = gpu.upload_f16_bits(&x16_bits, &[n, c.k]).map_err(err)?;
        let (g_off, u_off) = run_arm(gpu, false, c, n, &ag, &au, &x, None)?;
        let (g_on, u_on) = run_arm(gpu, true, c, n, &ag, &au, &x, None)?;
        let (gf_off, uf_off) = run_arm(gpu, false, c, n, &ag, &au, &x, Some(&x16))?;
        let (gf_on, uf_on) = run_arm(gpu, true, c, n, &ag, &au, &x, Some(&x16))?;
        let mut mism = 0u64;
        let gm = n * c.gate_m;
        let um = n * c.up_m;
        for (a, b, body) in [
            (&g_off, &g_on, gm),
            (&u_off, &u_on, um),
            (&gf_off, &gf_on, gm),
            (&uf_off, &uf_on, um),
        ] {
            if a.len() != b.len() {
                mism += 1;
                continue;
            }
            mism += a.iter().zip(b.iter()).filter(|(p, q)| p != q).count() as u64;
            // Every body word must have been written (sentinel never survives).
            mism += a[..body].iter().filter(|&&w| w == SENTINEL).count() as u64;
            mism += b[..body].iter().filter(|&&w| w == SENTINEL).count() as u64;
        }
        let guards = [(&g_off, gm), (&u_off, um), (&g_on, gm), (&u_on, um)]
            .into_iter()
            .chain([(&gf_off, gm), (&uf_off, um), (&gf_on, gm), (&uf_on, um)])
            .all(|(v, body)| guard_ok(v, body));
        // Inputs unchanged.
        let xd = gpu.download_f32(&x).map_err(err)?;
        let x_ok = bits(&xd) == bits(&xh);
        let ag_ok = {
            let mut back = vec![0u8; wg.len()];
            gpu.hip.memcpy_dtoh(&mut back, &ag.buf).map_err(err)?;
            back == wg
        };
        let au_ok = {
            let mut back = vec![0u8; wu.len()];
            gpu.hip.memcpy_dtoh(&mut back, &au.buf).map_err(err)?;
            back == wu
        };
        let n_cells = 2 * (gm + um) as u64;
        *cells += n_cells;
        let failed = mism > 0 || !guards || !x_ok || !ag_ok || !au_ok;
        if failed {
            *bad += mism.max(1);
        }
        println!(
            "  seed {seed:#x} {}+{} K{} N{n}: cells {n_cells} mismatches {mism} guards {} inputs {} {}",
            c.gate_m,
            c.up_m,
            c.k,
            if guards { "ok" } else { "CLOBBERED" },
            if x_ok && ag_ok && au_ok { "ok" } else { "CHANGED" },
            if failed { "FAIL" } else { "ok" }
        );
        gpu.free_tensor(x).map_err(err)?;
        gpu.free_tensor(x16).map_err(err)?;
    }
    gpu.free_tensor(ag).map_err(err)?;
    gpu.free_tensor(au).map_err(err)?;
    Ok(())
}

fn timing(gpu: &mut Gpu) -> R<()> {
    const COPIES: usize = 9; // 9 x ~94.7 MB >= 768 MB
    let (m, k) = (17408usize, 5120usize);
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let wg = weights(&mut rng, m, k);
    let wu = weights(&mut rng, m, k);
    let mut ags = Vec::new();
    let mut aus = Vec::new();
    for _ in 0..COPIES {
        ags.push(gpu.upload_raw(&wg, &[wg.len()]).map_err(err)?);
        aus.push(gpu.upload_raw(&wu, &[wu.len()]).map_err(err)?);
    }
    let set_mb = (wg.len() + wu.len()) as f64 / 1e6;
    println!(
        "timing: {} + {} rows, K {}, cold rotation {} sets x {:.1} MB = {:.0} MB; median of 5 x 100 launches",
        m, m, k, COPIES, set_mb, set_mb * COPIES as f64
    );
    let wbytes = (wg.len() + wu.len()) as f64;
    let ev: Vec<_> = (0..2)
        .map(|_| gpu.hip.event_create().map_err(err))
        .collect::<R<_>>()?;
    println!("   N | OFF us | ON us | OFF GB/s | ON GB/s | speedup");
    for &n in &NS {
        let xh = activations(&mut rng, n, k);
        let x = gpu.upload_f32(&xh, &[xh.len()]).map_err(err)?;
        let yg = out_tensor(gpu, n * m)?;
        let yu = out_tensor(gpu, n * m)?;
        let mut med = [0f64; 2];
        for (slot, on) in [false, true].into_iter().enumerate() {
            set_flag(gpu, on);
            for s in 0..COPIES {
                gpu.gemm_gate_up_hfq4g256_mq4v2(&ags[s], &aus[s], &x, &yg, &yu, m, m, k, n)
                    .map_err(err)?;
            }
            gpu.hip.device_synchronize().map_err(err)?;
            let mut samples = Vec::new();
            for rep in 0..5 {
                gpu.hip.event_record(&ev[0], None).map_err(err)?;
                for it in 0..100 {
                    let s = (rep * 100 + it) % COPIES;
                    gpu.gemm_gate_up_hfq4g256_mq4v2(&ags[s], &aus[s], &x, &yg, &yu, m, m, k, n)
                        .map_err(err)?;
                }
                gpu.hip.event_record(&ev[1], None).map_err(err)?;
                gpu.hip.event_synchronize(&ev[1]).map_err(err)?;
                let ms = gpu.hip.event_elapsed_ms(&ev[0], &ev[1]).map_err(err)?;
                samples.push(f64::from(ms) * 1000.0 / 100.0);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            med[slot] = samples[2];
        }
        println!(
            " {:>3} | {:>6.1} | {:>5.1} | {:>8.1} | {:>7.1} | {:.3}x",
            n,
            med[0],
            med[1],
            wbytes / (med[0] * 1e3),
            wbytes / (med[1] * 1e3),
            med[0] / med[1]
        );
        gpu.free_tensor(x).map_err(err)?;
        gpu.free_tensor(yg).map_err(err)?;
        gpu.free_tensor(yu).map_err(err)?;
    }
    Ok(())
}

fn main() -> R<()> {
    let no_time = std::env::args().any(|a| a == "--no-time");
    let mut gpu = Gpu::init().map_err(err)?;
    println!("arch {}", gpu.arch);
    if gpu.arch != "gfx1201" {
        return Err(format!("needs exact gfx1201, got {}", gpu.arch));
    }
    // Tile counts 2048 == slots and 2049 == slots + 1 assume 64 CUs (R9700);
    // 4097 tiles exercises t = 3.
    let cases = [
        Case {
            gate_m: 17408,
            up_m: 17408,
            k: 5120,
        },
        Case {
            gate_m: 17400,
            up_m: 17401,
            k: 5120,
        },
        Case {
            gate_m: 16384,
            up_m: 16384,
            k: 512,
        },
        Case {
            gate_m: 16384,
            up_m: 16385,
            k: 512,
        },
        Case {
            gate_m: 17400,
            up_m: 17401,
            k: 256,
        },
        Case {
            gate_m: 16384,
            up_m: 16384,
            k: 256,
        },
        Case {
            gate_m: 16384,
            up_m: 16385,
            k: 256,
        },
        Case {
            gate_m: 32768,
            up_m: 32784,
            k: 512,
        },
        Case {
            gate_m: 19,
            up_m: 7,
            k: 256,
        },
    ];
    let (mut cells, mut bad) = (0u64, 0u64);
    for seed in [0xA11CE_u64, 0xB0B5_u64] {
        for c in &cases {
            check_case(&mut gpu, c, seed, &mut cells, &mut bad)?;
        }
    }
    println!("SUMMARY: cells checked {cells}, mismatches {bad}");
    if !no_time {
        timing(&mut gpu)?;
    }
    if bad > 0 {
        std::process::exit(1);
    }
    Ok(())
}
