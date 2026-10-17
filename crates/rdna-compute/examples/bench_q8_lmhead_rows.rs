// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Q8_0 lm_head at DFlash draft/verify widths (HIPF-100, #754).
//!
//! Uploads a real Q8_0 lm_head from an HFQ file and times every Q8 batched
//! entry point at N = 1..16 rows, against an F64 CPU reference on a row sample;
//! the gfx1151 XLDS arm must be bitwise the base `gemm_q8_0_wmma` arm.
//!
//! Run: bench_q8_lmhead_rows <file.hfq> <byte offset> <M> <K> [arm,arm,...]

use rdna_compute::{DType, Gpu, GpuTensor};
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

const WARMUP: usize = 3;
const TRIALS: usize = 20;
const ARMS: [&str; 4] = [
    "batched_chunked(prod)",
    "gemm_q8_0_wmma base",
    "gemm_q8_0_wmma xlds",
    "gemv_q8_0 x n",
];

fn wrap(ptr: *mut std::ffi::c_void, bytes: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) },
        shape,
        dtype,
    }
}

fn f16(bits: u16) -> f32 {
    let s = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((bits >> 10) & 0x1f) as i32;
    let m = (bits & 0x3ff) as f32;
    s * if e == 0 {
        m * 2f32.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

fn run(
    g: &mut Gpu,
    arm: usize,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> hip_bridge::HipResult<()> {
    match arm {
        0 => g.gemm_q8_0_batched_chunked(w, x, y, m, k, n),
        1 | 2 => {
            std::sync::Arc::make_mut(&mut g.flags).gfx1151_xlds = arm == 2;
            g.gemm_q8_0_wmma(w, x, y, m, k, n)
        }
        3 => {
            for r in 0..n {
                g.gemv_q8_0(w, &x.sub_offset(r * k, k), &y.sub_offset(r * m, m), m, k)?;
            }
            Ok(())
        }
        _ => unreachable!(),
    }
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Base vs XLDS `gemm_q8_0_wmma` raw f32 bits on random Q8_0 weights at odd
/// shapes (M not a multiple of 64, partial batch tiles). Panics on any diff.
fn odd_shapes(gpu: &mut Gpu) {
    let mut s = 0x2545F4914F6CDD1Du64;
    for &(m, k) in &[(1000usize, 256usize), (4100, 768), (77, 512)] {
        let mut w = vec![0u8; m * (k / 32) * 34];
        for (i, b) in w.iter_mut().enumerate() {
            *b = if i % 34 == 1 {
                0x20 + (xorshift(&mut s) % 8) as u8
            } else {
                xorshift(&mut s) as u8
            };
        }
        let x: Vec<f32> = (0..16 * k)
            .map(|_| (xorshift(&mut s) >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            .collect();
        let wd = gpu.upload_raw(&w, &[w.len()]).unwrap();
        let wt = wrap(wd.buf.as_ptr(), w.len(), vec![m, k], DType::Q8_0);
        let xd = gpu.upload_f32(&x, &[16 * k]).unwrap();
        let yd = gpu.zeros(&[16 * m], DType::F32).unwrap();
        for n in [1usize, 5, 16] {
            let xt = xd.sub_offset(0, n * k);
            let yt = yd.sub_offset(0, n * m);
            let mut out = Vec::new();
            for arm in [1, 2] {
                run(gpu, arm, &wt, &xt, &yt, m, k, n).unwrap();
                out.push(gpu.download_f32(&yt).unwrap());
            }
            let same = out[0]
                .iter()
                .zip(&out[1])
                .all(|(p, q)| p.to_bits() == q.to_bits());
            println!(
                "odd M={m} K={k} N={n}: {}",
                if same { "bitwise==base" } else { "DIFFERS" }
            );
            assert!(same, "XLDS differs from base at M={m} K={k} N={n}");
        }
        std::mem::forget(wt);
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, off, m, k): (&str, u64, usize, usize) = (
        &a[1],
        a[2].parse().unwrap(),
        a[3].parse().unwrap(),
        a[4].parse().unwrap(),
    );
    let arms: Vec<usize> = a.get(5).map_or((0..ARMS.len()).collect(), |s| {
        s.split(',').map(|v| v.parse().unwrap()).collect()
    });
    let wbytes = m * (k / 32) * 34;
    let mut w = vec![0u8; wbytes];
    let mut f = std::fs::File::open(path).unwrap();
    f.seek(SeekFrom::Start(off)).unwrap();
    f.read_exact(&mut w).unwrap();

    let mut gpu = Gpu::init().expect("gpu init");
    odd_shapes(&mut gpu);
    println!(
        "arch {} M={m} K={k} weight {:.1} MB",
        gpu.arch,
        wbytes as f64 / 1e6
    );
    let wd = gpu.hip.malloc(wbytes).unwrap();
    gpu.hip.memcpy_htod(&wd, &w).unwrap();
    let wt = wrap(wd.as_ptr(), wbytes, vec![m, k], DType::Q8_0);

    let nmax = 16;
    let mut s = 0x9E3779B97F4A7C15u64;
    let x: Vec<f32> = (0..nmax * k)
        .map(|_| ((xorshift(&mut s) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
        .collect();
    let xd = gpu.hip.malloc(x.len() * 4).unwrap();
    gpu.hip
        .memcpy_htod(&xd, unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
        })
        .unwrap();
    let yd = gpu.hip.malloc(nmax * m * 4).unwrap();

    // CPU reference on a row sample (every 97th row).
    let sample: Vec<usize> = (0..m).step_by(97).collect();
    let refv = |n: usize, r: usize| -> f64 {
        let row = &w[r * (k / 32) * 34..(r + 1) * (k / 32) * 34];
        let mut acc = 0f64;
        for b in 0..k / 32 {
            let sc = f16(u16::from_le_bytes([row[b * 34], row[b * 34 + 1]])) as f64;
            for i in 0..32 {
                acc += sc * (row[b * 34 + 2 + i] as i8) as f64 * x[n * k + b * 32 + i] as f64;
            }
        }
        acc
    };

    for n in [1usize, 4, 8, 15, 16] {
        let mut base: Option<Vec<f32>> = None;
        let xt = wrap(xd.as_ptr(), n * k * 4, vec![n, k], DType::F32);
        let yt = wrap(yd.as_ptr(), n * m * 4, vec![n, m], DType::F32);
        for &arm in &arms {
            let name = ARMS[arm];
            if let Err(e) = run(&mut gpu, arm, &wt, &xt, &yt, m, k, n) {
                println!("N={n:2} {name:24} ERR {e:?}");
                continue;
            }
            gpu.hip.device_synchronize().unwrap();
            let y = gpu.download_f32(&yt).unwrap();
            let bitwise = match (&base, arm) {
                (Some(b), 2) if b.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()) => {
                    "bitwise==base"
                }
                (Some(_), 2) => "DIFFERS from base",
                _ => "",
            };
            if arm == 1 {
                base = Some(y.clone());
            }
            let (mut maxrel, mut nan) = (0f64, 0usize);
            for r in 0..n {
                for &row in &sample {
                    let v = y[r * m + row] as f64;
                    if !v.is_finite() {
                        nan += 1;
                        continue;
                    }
                    let rv = refv(r, row);
                    maxrel = maxrel.max((v - rv).abs() / rv.abs().max(1.0));
                }
            }
            for _ in 0..WARMUP {
                run(&mut gpu, arm, &wt, &xt, &yt, m, k, n).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..TRIALS {
                run(&mut gpu, arm, &wt, &xt, &yt, m, k, n).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;
            println!(
                "N={n:2} {name:24} {us:9.1} us  {:6.1} GB/s  max_err/max(|ref|,1)={maxrel:.2e} nan={nan} {bitwise}",
                wbytes as f64 / us / 1e3
            );
        }
        std::mem::forget(xt);
        std::mem::forget(yt);
    }
    std::mem::forget(wt);
}
