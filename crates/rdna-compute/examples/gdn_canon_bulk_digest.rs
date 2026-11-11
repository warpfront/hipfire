// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LAB ONLY: byte digest of the INCUMBENT aligned whole-call GDN chunk-scan route using only the
//! legacy bindings (`gdn_chunk_prep`, batched KKT, `gdn_chunk_scan_layer_mseg` /
//! `gdn_chunk_scan_segment`), so the same file builds unchanged against the base commit and the
//! S2 branch. Run it on both trees and diff the dumps: equal files prove the aligned bulk bytes
//! (out rows, final Q8 codes/scales/EF, conv ring) are unchanged.
//!
//! Build: cargo build --release -p rdna-compute --features lab,deltanet --example gdn_canon_bulk_digest
//! Run:   gdn_canon_bulk_digest <out_dir> [--hv 48|32]

use rdna_compute::norm::GdnScanOut;
use rdna_compute::{DType, Gpu, GpuTensor};

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;
const HD: usize = 128;
const HK: usize = 16;

fn lcg(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * (((s >> 40) as u32) as f32 / (1u32 << 24) as f32)
        })
        .collect()
}
fn fb(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}
fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign << 15;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return (sign << 15) | 0x7bff;
    }
    if e <= 0 {
        return sign << 15;
    }
    (sign << 15) | ((e as u16) << 10) | (mant >> 13) as u16
}
fn raw(gpu: &mut Gpu, n: usize) -> Result<GpuTensor> {
    Ok(gpu.zeros(&[n.max(16)], DType::Raw)?)
}
fn up(gpu: &mut Gpu, d: &[u8]) -> Result<GpuTensor> {
    let t = raw(gpu, d.len())?;
    gpu.memcpy_htod_auto(&t.buf, d)?;
    Ok(t)
}
fn dl(gpu: &Gpu, t: &GpuTensor, n: usize) -> Result<Vec<u8>> {
    gpu.hip.device_synchronize()?;
    let mut d = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut d, &t.buf)?;
    d.truncate(n);
    Ok(d)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().ok_or("usage: gdn_canon_bulk_digest <out_dir> [--hv 48|32]")?;
    let mut hv = 48usize;
    while let Some(a) = args.next() {
        if a == "--hv" {
            hv = args.next().ok_or("--hv value")?.parse()?;
        }
    }
    std::fs::create_dir_all(&dir)?;
    let mut gpu = Gpu::init()?;
    let fmt = if gpu.arch == "gfx1201" { GdnScanOut::Bf16 } else { GdnScanOut::F32 };
    let out_elem = if fmt == GdnScanOut::Bf16 { 2 } else { 4 };
    let xdim = (2 * HK + hv) * HD;
    let total = 1100usize;
    let qkv = lcg(1, total * xdim, -1.732_050_8, 1.732_050_8);
    let beta_raw = lcg(3, total * hv, -2.0, 2.0);
    let alpha_raw = lcg(4, total * hv, -1.732_050_8, 1.732_050_8);
    let cw = gpu.upload_f32(&lcg(5, 4 * xdim, -0.5, 0.5), &[4 * xdim])?;
    let dtb = gpu.upload_f32(&lcg(6, hv, -4.0, -2.0), &[hv])?;
    let alog = gpu.upload_f32(&lcg(7, hv, 0.0, 2.0), &[hv])?;
    let conv0 = fb(&lcg(9, 3 * xdim, -0.5, 0.5));
    let q80: Vec<u8> = lcg(10, hv * HD * HD, -64.0, 64.0).iter().map(|x| *x as i8 as u8).collect();
    let sc0 = fb(&lcg(11, hv * HD, 0.005, 0.02));
    let ef0: Vec<u8> = lcg(12, hv * HD * HD, -0.004, 0.004).iter().flat_map(|x| f16_bits(*x).to_ne_bytes()).collect();
    for n in [1usize, 25, 63, 64, 65, 300, 512, 513, 1100] {
        let conv = up(&mut gpu, &conv0)?;
        let codes = up(&mut gpu, &q80)?;
        let scales = up(&mut gpu, &sc0)?;
        let ef = up(&mut gpu, &ef0)?;
        let alloc = n.div_ceil(64) * 64;
        let x = up(&mut gpu, &fb(&qkv[..n * xdim]))?;
        let a = up(&mut gpu, &fb(&alpha_raw[..n * hv]))?;
        let b = up(&mut gpu, &fb(&beta_raw[..n * hv]))?;
        let q = raw(&mut gpu, alloc * HK * HD * 2)?;
        let k = raw(&mut gpu, alloc * HK * HD * 2)?;
        let v = raw(&mut gpu, alloc * hv * HD * 2)?;
        let aa = raw(&mut gpu, alloc * hv * 64 * 2)?;
        let o = raw(&mut gpu, n * hv * HD * out_elem)?;
        let seg = 512usize;
        gpu.gdn_chunk_prepare(fmt)?;
        let qs = 1.0 / (HD as f32).sqrt();
        let kkt_batched = gpu.gdn_kkt_batched_enabled() && n > seg;
        gpu.gdn_chunk_prep(&x, &cw, &conv, &a, &b, &dtb, &alog, &q, &k, &v, n, qs, 1e-6, HK, hv)?;
        if kkt_batched {
            gpu.gdn_chunk_kkt_solve_batched(&k, &a, &b, &aa, n, HK, hv)?;
        }
        if kkt_batched && gpu.gdn_scan_mseg_enabled(fmt, seg) {
            gpu.gdn_chunk_scan_layer_mseg(&q, &k, &v, &aa, &a, &b, &codes, &scales, &ef, &o, n, HK, hv)?;
        } else {
            for row0 in (0..n).step_by(seg) {
                let rows = (n - row0).min(seg);
                let a_seg = if kkt_batched {
                    aa.sub_offset(row0 * hv * 64 * 2, rows.div_ceil(64) * 64 * hv * 64 * 2)
                } else {
                    aa.shallow_clone()
                };
                gpu.gdn_chunk_scan_segment(&q, &k, &v, &a_seg, &a, &b, &codes, &scales, &ef, &o, fmt, row0, rows, !kkt_batched, HK, hv)?;
            }
        }
        std::fs::write(format!("{dir}/n{n}.out"), dl(&gpu, &o, n * hv * HD * out_elem)?)?;
        std::fs::write(format!("{dir}/n{n}.codes"), dl(&gpu, &codes, hv * HD * HD)?)?;
        std::fs::write(format!("{dir}/n{n}.scales"), dl(&gpu, &scales, hv * HD * 4)?)?;
        std::fs::write(format!("{dir}/n{n}.ef"), dl(&gpu, &ef, hv * HD * HD * 2)?)?;
        std::fs::write(format!("{dir}/n{n}.conv"), dl(&gpu, &conv, 3 * xdim * 4)?)?;
        for t in [conv, codes, scales, ef, x, a, b, q, k, v, aa, o] {
            gpu.free_tensor(t)?;
        }
    }
    println!("{{\"oracle\":\"gdn_canon_bulk_digest\",\"arch\":\"{}\",\"hv\":{hv},\"dir\":\"{dir}\"}}", gpu.arch);
    Ok(())
}
