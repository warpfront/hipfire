// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle and timing of the batched multi-slot lm_head (`HIPFIRE_BATCHED_HEAD`):
//! `Gpu::gemv_mq4g256v2_multirow_r2_xbatch_pm` over B = 1..=8 rotated rows at the
//! A3B head shape (M = 248,320 rows, K = 2,048) against the singleton head GEMV
//! (`Gpu::gemv_mq4g256v2` once per row on `x[b]` / `y[b]` sub-views, which is
//! `gemv_mq4g256v2_multirow_r2` on exact gfx1201), bit for bit over the whole
//! output allocation including poisoned guard words. Rows are written at a
//! non-zero offset of a larger buffer (the product writes slot rows straight into
//! `logits_out`). x modes: random, all zero, F16-hostile mantissas, +-0, wide
//! dynamic range, per row. Repeat determinism and a negative control (one
//! perturbed x element must change the output) are checked.
//!
//! `time`: medians over 15 samples of B singleton launches vs one batched launch
//! (HIP-synchronised wall clock) for B = 1, 2, 4, 8.
//!
//! Prints PASS/FAIL per case; exits nonzero on any FAIL. Requires exact gfx1201.
//! `HIPFIRE_PM_DECODE=0` covers the HIP incumbent fallback.
//!
//! `cargo run --release -p rdna-compute --features lab --example head_batch_oracle [-- time]`

use rdna_compute::{Gpu, GpuTensor};

const M: usize = 248_320;
const K: usize = 2_048;
const POISON: u32 = 0x7fc0_dead;
const GUARD: usize = 64;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}

fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut m10 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m10 & 1 == 1) {
        m10 += 1;
    }
    if m10 == 0x400 {
        m10 = 0;
        exp += 1;
    }
    assert!((1..=30).contains(&exp), "{v} is not a normal f16");
    sign | ((exp as u16) << 10) | m10 as u16
}

fn random_scale(rng: &mut Rng) -> u16 {
    let mag = 2f32.powi(-9) + rng.unit() * 0.2;
    f16_bits(if rng.next() & 1 == 0 { mag } else { -mag })
}

fn random_zero(rng: &mut Rng) -> u16 {
    let mut v = (rng.unit() - 0.5) * 4.0;
    if v.abs() < 2f32.powi(-12) {
        v = 0.125;
    }
    f16_bits(v)
}

/// `[m][k / 256]` groups of 136 bytes: hA (scale s0, zero z0), hB (s1, z1), then
/// 128 payload bytes.
fn pack_mq4g256v2(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let groups = k / 256;
    let mut out = Vec::with_capacity(m * groups * 136);
    for _ in 0..m * groups {
        for _half in 0..2 {
            out.extend_from_slice(&random_scale(rng).to_le_bytes());
            out.extend_from_slice(&random_zero(rng).to_le_bytes());
        }
        for _ in 0..128 {
            out.push((rng.next() & 0xff) as u8);
        }
    }
    out
}

fn gen_x(rng: &mut Rng, k: usize, mode: u64) -> Vec<f32> {
    (0..k)
        .map(|i| match mode % 5 {
            0 => {
                if i % 5 == 0 {
                    0.0
                } else {
                    (rng.unit() + rng.unit() - 1.0) * 1.5
                }
            }
            1 => 0.0,
            2 => {
                let m = if rng.next() & 1 == 0 {
                    1.0 + 2f32.powi(-12)
                } else {
                    1.0 + 2f32.powi(-11)
                };
                let e = (rng.next() % 13) as i32 - 6;
                let s = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
                s * m * 2f32.powi(e)
            }
            3 => match rng.next() % 4 {
                0 => 0.0,
                1 => -0.0,
                2 => (rng.unit() - 0.5) * 1e-3,
                _ => (rng.unit() - 0.5) * 3.0,
            },
            _ => (rng.unit() - 0.5) * 2f32.powi((rng.next() % 20) as i32 - 10),
        })
        .collect()
}

fn poisoned(gpu: &mut Gpu, n: usize) -> GpuTensor {
    gpu.upload_f32(&vec![f32::from_bits(POISON); n], &[n])
        .expect("poison upload")
}

fn fetch(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync before download");
    gpu.download_f32(t)
        .expect("download")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn diff(a: &[u32], b: &[u32]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (n, first)
}

/// One case: `batch` rows, y written at row offset `row0` of a `(row0 + batch)`
/// row buffer plus guard words.
fn run_case(
    gpu: &mut Gpu,
    rng: &mut Rng,
    w: &GpuTensor,
    batch: usize,
    row0: usize,
    seed_mode: u64,
) -> Result<(), String> {
    let mut x_h: Vec<f32> = Vec::with_capacity(batch * K);
    for b in 0..batch {
        let mode = rng.next() % 5 + seed_mode + b as u64;
        x_h.extend(gen_x(rng, K, mode));
    }
    let x = gpu.upload_f32(&x_h, &[batch * K]).expect("x");
    let total = (row0 + batch) * M + GUARD;
    let y_old = poisoned(gpu, total);
    let y_new = poisoned(gpu, total);
    let y_new2 = poisoned(gpu, total);
    for b in 0..batch {
        let xs = x.sub_offset(b * K, K);
        let ys = y_old.sub_offset((row0 + b) * M, M);
        gpu.gemv_mq4g256v2(w, &xs, &ys, M, K)
            .expect("singleton head");
    }
    for y in [&y_new, &y_new2] {
        let ys = y.sub_offset(row0 * M, batch * M);
        gpu.gemv_mq4g256v2_multirow_r2_xbatch_pm(w, &x, &ys, M, K, batch)
            .expect("batched head");
    }
    let old = fetch(gpu, &y_old);
    let new = fetch(gpu, &y_new);
    let new2 = fetch(gpu, &y_new2);
    let mut result = Ok(());
    let (n, first) = diff(&old, &new);
    if n != 0 {
        let i = first.unwrap();
        let rel = i.saturating_sub(row0 * M);
        result = Err(format!(
            "{n} of {} output bits differ (first at column {} row {}: singleton {:#010x} batched {:#010x})",
            old.len(),
            rel / M,
            rel % M,
            old[i],
            new[i]
        ));
    } else if new != new2 {
        result = Err("batched head is not repeat-deterministic".into());
    } else if new[(row0 + batch) * M..].iter().any(|&b| b != POISON) {
        result = Err("guard words past y were written".into());
    } else if new[..row0 * M].iter().any(|&b| b != POISON) {
        result = Err("words before the first row were written".into());
    } else if new[row0 * M..(row0 + batch) * M]
        .iter()
        .any(|&b| b == POISON)
    {
        result = Err("an output word was not written".into());
    }
    for t in [x, y_old, y_new, y_new2] {
        gpu.free_tensor(t).expect("free");
    }
    result
}

/// A perturbed x element must change that column's output (the comparison can
/// fail).
fn negative_control(gpu: &mut Gpu, rng: &mut Rng, w: &GpuTensor) -> Result<(), String> {
    let batch = 3;
    let mut x_h: Vec<f32> = Vec::new();
    for _ in 0..batch {
        x_h.extend(gen_x(rng, K, 0));
    }
    let x = gpu.upload_f32(&x_h, &[batch * K]).expect("x");
    x_h[K + 17] += 0.5;
    let x2 = gpu.upload_f32(&x_h, &[batch * K]).expect("x2");
    let ya = poisoned(gpu, batch * M);
    let yb = poisoned(gpu, batch * M);
    gpu.gemv_mq4g256v2_multirow_r2_xbatch_pm(w, &x, &ya, M, K, batch)
        .expect("a");
    gpu.gemv_mq4g256v2_multirow_r2_xbatch_pm(w, &x2, &yb, M, K, batch)
        .expect("b");
    let (a, b) = (fetch(gpu, &ya), fetch(gpu, &yb));
    let col = |v: &[u32], c: usize| v[c * M..(c + 1) * M].to_vec();
    let ok = col(&a, 0) == col(&b, 0) && col(&a, 2) == col(&b, 2) && col(&a, 1) != col(&b, 1);
    for t in [x, x2, ya, yb] {
        gpu.free_tensor(t).expect("free");
    }
    if ok {
        Ok(())
    } else {
        Err("perturbed column did not (only) change".into())
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time(gpu: &mut Gpu, rng: &mut Rng, w: &GpuTensor) {
    let mut x_h = Vec::new();
    for _ in 0..8 {
        x_h.extend(gen_x(rng, K, 0));
    }
    let x = gpu.upload_f32(&x_h, &[8 * K]).expect("x");
    let y = poisoned(gpu, 8 * M);
    for batch in [1usize, 2, 4, 8] {
        let mut single = Vec::new();
        let mut batched = Vec::new();
        for it in 0..20 {
            gpu.hip.device_synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for b in 0..batch {
                let xs = x.sub_offset(b * K, K);
                let ys = y.sub_offset(b * M, M);
                gpu.gemv_mq4g256v2(w, &xs, &ys, M, K).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let ts = t0.elapsed().as_secs_f64() * 1e6;
            let t0 = std::time::Instant::now();
            let ys = y.sub_offset(0, batch * M);
            gpu.gemv_mq4g256v2_multirow_r2_xbatch_pm(w, &x, &ys, M, K, batch)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let tb = t0.elapsed().as_secs_f64() * 1e6;
            if it >= 5 {
                single.push(ts);
                batched.push(tb);
            }
        }
        let (s, b) = (median(&mut single), median(&mut batched));
        println!("TIME B={batch}: {batch} singleton launches {s:.1} us, one batched launch {b:.1} us ({:.2}x)", s / b);
    }
    gpu.free_tensor(x).expect("free");
    gpu.free_tensor(y).expect("free");
}

fn main() {
    let timing = std::env::args().any(|a| a == "time");
    let mut gpu = Gpu::init().expect("Gpu::init");
    println!("arch {}", gpu.arch);
    let mut rng = Rng(0x4845_4144_4241_5443);
    let bytes = pack_mq4g256v2(M, K, &mut rng);
    let w = gpu
        .upload_raw(&bytes, &[bytes.len()])
        .expect("weight upload");
    println!("weight {} bytes, M {M} K {K}", bytes.len());
    let mut cases = 0usize;
    let mut failures = 0usize;
    let mut report = |label: String, r: Result<(), String>| {
        cases += 1;
        match r {
            Ok(()) => println!("PASS {label}"),
            Err(why) => {
                failures += 1;
                println!("FAIL {label}: {why}");
            }
        }
    };
    for batch in 1..=8usize {
        for (row0, mode) in [(0usize, 0u64), (3, 1), (1, 2)] {
            let label = format!("batch {batch} row0 {row0} mode {mode}");
            report(label, run_case(&mut gpu, &mut rng, &w, batch, row0, mode));
        }
    }
    report(
        "negative control".into(),
        negative_control(&mut gpu, &mut rng, &w),
    );
    if timing {
        time(&mut gpu, &mut rng, &w);
    }
    println!("{cases} cases, {failures} failures");
    if failures != 0 {
        std::process::exit(1);
    }
}
