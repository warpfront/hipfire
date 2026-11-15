// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle of the served call site of the multi-part lane-WMMA launch
//! (`HIPFIRE_QWEN4_LWMMA_MULTI_PART=1`, exact gfx1201): `project_weights` over the BF16 members
//! of the wide MoE block's dense front (router | selector | shared gate | shared up, K = 2560,
//! rows 16..=64; plus a five-member group, which must chunk four + one) against today's
//! per-member `lane_wmma_gemm_fixed` (two terms, auto split-K). Outputs start NaN-dirty with
//! sentinel tails; every word is compared. With the flag on a `_multi` entry must have been
//! loaded, with it off none (the oracle then proves the default path is the per-member loop).
//!
//! ```text
//! [HIPFIRE_QWEN4_LWMMA_MULTI_PART=1] cargo run --release -p hipfire-dispatch \
//!   --features test-utils --example lwmma_multi_front_oracle
//! ```

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::project_weights;
use rdna_compute::lane_wmma::LaneWmmaWeight;
use rdna_compute::{DType, Gpu, GpuTensor};

type R<T> = Result<T, Box<dyn std::error::Error>>;

const K: usize = 2560;
const TAIL: usize = 512;
const CASES: &[(&str, &[usize])] = &[
    ("front_r0", &[512, 1, 384, 384]),
    ("front_r1", &[512, 1, 256, 256]),
    ("sel_gate_up", &[1, 384, 384]),
    ("five_members", &[512, 1, 384, 384, 384]),
    ("pair", &[384, 384]),
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn upload_words(gpu: &mut Gpu, d: &[u32]) -> R<GpuTensor> {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) };
    let t = gpu.alloc_tensor(&[d.len()], DType::F32)?;
    gpu.hip.memcpy_htod(&t.buf, bytes)?;
    Ok(t)
}

fn download(gpu: &Gpu, t: &GpuTensor, words: usize) -> R<Vec<u32>> {
    let mut b = vec![0u8; words * 4];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf)?;
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn dirty(words: usize, pat: u32) -> Vec<u32> {
    let mut v = vec![pat; words];
    v.extend((0..TAIL).map(|i| 0xDEAD_0000 | i as u32));
    v
}

fn wref(buf: &GpuTensor, m: usize) -> WeightRef<'_> {
    WeightRef {
        buf,
        dtype: DType::BF16,
        m,
        k: K,
        row_stride: K,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

fn main() -> R<()> {
    let mut gpu = Gpu::init()?;
    if !gpu.arch_caps.is_gfx1201() {
        return Err("requires exact gfx1201".into());
    }
    let flag_on = gpu.flags.qwen4_lwmma_multi_part_enabled();
    println!("HIPFIRE_QWEN4_LWMMA_MULTI_PART enabled: {flag_on}");
    let (mut cells, mut words, mut mismatches) = (0usize, 0u64, 0u64);
    for &(label, ms) in CASES {
        let mut ws = Vec::new();
        for (p, &m) in ms.iter().enumerate() {
            let mut r = Rng(0x4D50_0000 + p as u64);
            let mut b = Vec::with_capacity(m * K * 2);
            for _ in 0..m * K {
                let v = (r.unit() - 0.5) * 0.1;
                b.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
            let t = gpu.alloc_tensor(&[m * K], DType::BF16)?;
            gpu.hip.memcpy_htod(&t.buf, &b)?;
            ws.push(t);
        }
        for rows in [16usize, 24, 32, 48, 64] {
            let mut r = Rng(0x1000 + rows as u64);
            let xh: Vec<u32> = (0..rows * K)
                .map(|_| ((r.unit() - 0.5) * 4.0).to_bits())
                .collect();
            let x = upload_words(&mut gpu, &xh)?;
            let mut ys_ref = Vec::new();
            let mut ys_new = Vec::new();
            for &m in ms {
                ys_ref.push(upload_words(&mut gpu, &dirty(rows * m, 0x7FC0_A5A5))?);
                ys_new.push(upload_words(&mut gpu, &dirty(rows * m, 0x7FC0_5A5A))?);
            }
            for (p, &m) in ms.iter().enumerate() {
                gpu.lane_wmma_gemm_fixed(
                    &LaneWmmaWeight::Bf16(&ws[p]),
                    &x,
                    &ys_ref[p],
                    m,
                    K,
                    rows,
                    2,
                    None,
                )?;
            }
            let refs: Vec<WeightRef<'_>> = ms
                .iter()
                .enumerate()
                .map(|(p, &m)| wref(&ws[p], m))
                .collect();
            let projections: Vec<(&WeightRef<'_>, &GpuTensor)> =
                refs.iter().zip(ys_new.iter()).collect();
            project_weights(&mut gpu, &x, rows, None, &projections)?;
            gpu.hip.device_synchronize()?;
            for (p, &m) in ms.iter().enumerate() {
                let n = rows * m + TAIL;
                let a = download(&gpu, &ys_ref[p], n)?;
                let b = download(&gpu, &ys_new[p], n)?;
                let bad = a.iter().zip(&b).filter(|(u, v)| u != v).count() as u64;
                words += n as u64;
                mismatches += bad;
                if bad != 0 {
                    println!("MISMATCH {label} rows {rows} part {p}: {bad}");
                }
            }
            cells += 1;
        }
    }
    let multi: Vec<String> = gpu
        .loaded_kernel_names()
        .into_iter()
        .filter(|n| n.ends_with("_n1_multi"))
        .collect();
    println!("multi entries loaded: {multi:?}");
    let route_ok = multi.is_empty() == !flag_on;
    println!(
        "cells {cells} | words compared {words} | mismatches {mismatches} | multi loaded iff flag: {route_ok}"
    );
    let pass = mismatches == 0 && route_ok;
    println!("ORACLE {}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
