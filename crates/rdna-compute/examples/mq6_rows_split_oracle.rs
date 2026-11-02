// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bitwise oracle of the stream-split MQ6G256V2 few-row verify kernels
//! (`gemm_mq6g256v2_f32_rows[_x4]_split_r{R}_w{W}_s{S}`, `HIPFIRE_MQ6_ROWS_SPLIT`).
//!
//! ```text
//! mq6_rows_split_oracle
//! ```
//!
//! For every case (shape, seed) and every `R` in `1..=16` three arms must agree
//! on every output word:
//!
//! * `on`: the public launcher (`gemm_mq6g256v2_f32_rows[_x4]`, `R == 1`: the
//!   decode GEMV) with the split flag on;
//! * `off`: the same launcher with the flag off (incumbent rows kernels);
//! * the reference: the decode singleton (`gemv_mq6g256v2[_x4]`) on each
//!   activation row alone.
//!
//! Additionally every one of the 36 split exports is launched directly by symbol
//! at its `R` (3..=8) and compared to the reference, so every `(R, W, S)` is
//! covered whatever the routing table picks, and a CPU implementation of the
//! singleton DAG (`f32::mul_add`, exact f16 -> f32) is compared with the
//! reference at `R = 1..=8` on the edge shapes and on the GDN-out trunk shape.
//!
//! Each output lives in `64 guard floats | 16 * M floats | 64 guard floats`
//! memset to the `0xFF` NaN sentinel before every launch; an `R`-row launch may
//! write only its `R` rows. Activations and weights are downloaded after every
//! case and must equal what was uploaded. Exit status is nonzero on any mismatch.

use half::f16;
use rdna_compute::{kernels, Gpu, GpuTensor};
use std::sync::Arc;

const GUARD: usize = 64;
const MAX_R: usize = 16;
const SEEDS: [u64; 2] = [0x6d71_3672_6f77_7331, 0x0fed_cba9_8765_4321];
const CONFIGS: [(usize, usize); 3] = [(8, 4), (4, 4), (4, 1)];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 11) % n
    }
}

#[derive(Default)]
struct Stats {
    cells: u64,
    words: u64,
    mismatches: u64,
}

struct Case {
    name: String,
    ms: [usize; 4],
    k: usize,
    x4: bool,
    cpu: bool,
}

/// Finite f16 header word: subnormal, zero, negative and normal values mixed.
fn f16_word(rng: &mut Rng) -> u16 {
    let sign = if rng.below(2) == 0 { 0 } else { 0x8000 };
    match rng.below(8) {
        0 => sign | (1 + rng.below(0x3ff) as u16),
        1 => sign,
        2 => 0x8000 | (1 + rng.below(0x3ff) as u16),
        _ => sign | ((5 + rng.below(20) as u16) << 10) | rng.below(0x400) as u16,
    }
}

fn gen_weights(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 256) * 200];
    for g in bytes.chunks_exact_mut(200) {
        for h in 0..4 {
            g[h * 2..h * 2 + 2].copy_from_slice(&f16_word(rng).to_le_bytes());
        }
        for b in &mut g[8..] {
            *b = rng.next() as u8;
        }
    }
    bytes
}

fn wide(rng: &mut Rng, lo: i32, hi: i32) -> f32 {
    let e = lo + rng.below((hi - lo + 1) as u64) as i32;
    let sign = (rng.below(2) as u32) << 31;
    f32::from_bits(sign | (((e + 127) as u32) << 23) | (rng.next() as u32 & 0x7f_ffff))
}

fn gen_x_row(r: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v = Vec::with_capacity(k);
    match r % 8 {
        0 => (0..k).for_each(|_| v.push(wide(rng, -20, 20))),
        1 => (0..k).for_each(|_| v.push(wide(rng, -60, -30))),
        2 => (0..k).for_each(|_| v.push(wide(rng, 10, 25))),
        3 => (0..k).for_each(|_| {
            v.push(match rng.below(4) {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(rng.next() as u32 & 0x7f_ffff),
                _ => f32::from_bits(0x8000_0000 | (rng.next() as u32 & 0x7f_ffff)),
            })
        }),
        4 => {
            while v.len() < k {
                let a = wide(rng, -20, 20);
                v.push(a);
                v.push(-a);
            }
            v.truncate(k);
        }
        5 => v.resize(k, 0.0),
        6 => (0..k).for_each(|_| {
            v.push(match rng.below(3) {
                0 => wide(rng, -20, 20),
                1 => 0.0,
                _ => f32::from_bits(rng.next() as u32 & 0x7f_ffff),
            })
        }),
        _ => {
            let c = rng.below(61) as i32 - 30;
            (0..k).for_each(|_| v.push(wide(rng, c - 3, c + 3)));
        }
    }
    v
}

/// One output row of the singleton DAG for each activation row in `xs`.
fn cpu_row(w: &[u8], k: usize, row: usize, xs: &[&[f32]]) -> Vec<f32> {
    let gpr = k / 256;
    let mut acc = vec![[[0f32; 4]; 32]; xs.len()];
    for g in 0..gpr {
        let gp = &w[(row * gpr + g) * 200..][..200];
        let hdr = |o: usize| {
            (
                f16::from_bits(u16::from_le_bytes([gp[o], gp[o + 1]])).to_f32(),
                f16::from_bits(u16::from_le_bytes([gp[o + 2], gp[o + 3]])).to_f32(),
            )
        };
        for lane in 0..32 {
            let (sc, zp) = if lane < 16 { hdr(0) } else { hdr(4) };
            let d = &gp[8 + lane * 6..][..6];
            let (b0, b1, b2, b3, b4, b5) = (
                d[0] as u32,
                d[1] as u32,
                d[2] as u32,
                d[3] as u32,
                d[4] as u32,
                d[5] as u32,
            );
            let q = [
                b0 & 63,
                (b0 >> 6) | ((b1 & 0xF) << 2),
                (b1 >> 4) | ((b2 & 3) << 4),
                b2 >> 2,
                b3 & 63,
                (b3 >> 6) | ((b4 & 0xF) << 2),
                (b4 >> 4) | ((b5 & 3) << 4),
                b5 >> 2,
            ];
            let wv: [f32; 8] = std::array::from_fn(|i| sc.mul_add(q[i] as f32, zp));
            for (r, x) in xs.iter().enumerate() {
                let xb = &x[g * 256 + lane * 8..][..8];
                let mut dd = xb[1] * wv[1];
                dd = xb[0].mul_add(wv[0], dd);
                for i in 2..8 {
                    dd = xb[i].mul_add(wv[i], dd);
                }
                acc[r][lane][g % 4] += dd;
            }
        }
    }
    acc.iter()
        .map(|a| {
            let mut v: [f32; 32] =
                std::array::from_fn(|l| (a[l][0] + a[l][1]) + (a[l][2] + a[l][3]));
            for off in [16usize, 8, 4, 2, 1] {
                let prev = v;
                for l in 0..32 {
                    v[l] = prev[l] + if l + off < 32 { prev[l + off] } else { prev[l] };
                }
            }
            v[0]
        })
        .collect()
}

fn err<E: std::fmt::Debug>(e: E) -> String {
    format!("{e:?}")
}

struct Fixture {
    ws: Vec<GpuTensor>,
    dummy_w: GpuTensor,
    x: GpuTensor,
    w_host: Vec<Vec<u8>>,
    x_host: Vec<f32>,
}

fn alloc_ys(gpu: &mut Gpu, ms: [usize; 4]) -> Result<Vec<GpuTensor>, String> {
    let mut ys = Vec::new();
    for m in ms {
        let t = gpu
            .alloc_tensor(&[2 * GUARD + MAX_R * m.max(1)], rdna_compute::DType::F32)
            .map_err(err)?;
        gpu.hip.memset(&t.buf, 0xff, t.buf.size()).map_err(err)?;
        ys.push(t);
    }
    Ok(ys)
}

/// Compare every word of each output allocation with the reference rows.
fn check(
    gpu: &Gpu,
    ys: &[GpuTensor],
    ms: [usize; 4],
    rows: usize,
    refs: &[Vec<Vec<f32>>],
    tag: &str,
    st: &mut Stats,
) -> Result<(), String> {
    st.cells += 1;
    for (p, y) in ys.iter().enumerate() {
        let m = ms[p];
        let got = gpu.download_f32(y).map_err(err)?;
        let mut bad = 0u64;
        for (i, g) in got.iter().enumerate() {
            let want = if i < GUARD || i >= GUARD + rows * m {
                0xffff_ffffu32
            } else {
                let j = i - GUARD;
                refs[j / m][p][j % m].to_bits()
            };
            if g.to_bits() != want {
                bad += 1;
                if bad <= 3 {
                    eprintln!(
                        "MISMATCH {tag} matrix {p} word {i}: got {:#010x} want {want:#010x}",
                        g.to_bits()
                    );
                }
            }
        }
        st.words += got.len() as u64;
        st.mismatches += bad;
    }
    Ok(())
}

fn parts_of<'a>(
    fx: &'a Fixture,
    ms: [usize; 4],
    ys: &'a [GpuTensor],
) -> [(&'a GpuTensor, &'a GpuTensor, usize); 4] {
    let mut next = 0;
    std::array::from_fn(|p| {
        let w = if ms[p] == 0 {
            &fx.dummy_w
        } else {
            next += 1;
            &fx.ws[next - 1]
        };
        (w, &ys[p], ms[p])
    })
}

fn run_case(gpu: &mut Gpu, case: &Case, seed: u64, st: &mut Stats) -> Result<(), String> {
    let mut rng = Rng(seed ^ (case.k as u64) ^ ((case.ms[0] as u64) << 20) ^ 0x9e37_79b9_7f4a_7c15);
    let k = case.k;
    let ms = case.ms;
    let mut fx = Fixture {
        ws: Vec::new(),
        dummy_w: gpu.upload_raw(&[0u8; 200], &[1, 256]).map_err(err)?,
        x: gpu.upload_f32(&[0.0], &[1]).map_err(err)?,
        w_host: Vec::new(),
        x_host: Vec::new(),
    };
    for &m in ms.iter().filter(|&&m| m > 0) {
        let h = gen_weights(m, k, &mut rng);
        fx.ws.push(gpu.upload_raw(&h, &[m, k]).map_err(err)?);
        fx.w_host.push(h);
    }
    for r in 0..MAX_R {
        fx.x_host.extend(gen_x_row(r, k, &mut rng));
    }
    fx.x = gpu.upload_f32(&fx.x_host, &[MAX_R, k]).map_err(err)?;

    // Reference: the decode singleton on each activation row alone.
    let mut refs: Vec<Vec<Vec<f32>>> = Vec::new();
    for r in 0..MAX_R {
        let xr = fx.x.sub_offset(r * k, k);
        let ys: Vec<GpuTensor> = ms
            .iter()
            .map(|&m| gpu.zeros(&[m.max(1)], rdna_compute::DType::F32))
            .collect::<Result<_, _>>()
            .map_err(err)?;
        if case.x4 {
            gpu.gemv_mq6g256v2_x4(parts_of(&fx, ms, &ys), &xr, k)
                .map_err(err)?;
        } else {
            gpu.gemv_mq6g256v2(&fx.ws[0], &xr, &ys[0], ms[0], k)
                .map_err(err)?;
        }
        let mut row = Vec::new();
        for (p, y) in ys.iter().enumerate() {
            let mut v = gpu.download_f32(y).map_err(err)?;
            v.truncate(ms[p]);
            row.push(v);
        }
        refs.push(row);
    }

    if case.cpu {
        let xs: Vec<&[f32]> = (0..8).map(|r| &fx.x_host[r * k..(r + 1) * k]).collect();
        let mut wi = 0;
        for (p, &m) in ms.iter().enumerate() {
            if m == 0 {
                continue;
            }
            let mut bad = 0u64;
            for row in 0..m {
                for (r, v) in cpu_row(&fx.w_host[wi], k, row, &xs).iter().enumerate() {
                    if v.to_bits() != refs[r][p][row].to_bits() {
                        bad += 1;
                        if bad <= 3 {
                            eprintln!("MISMATCH cpu {} p{p} row {row} r{r}", case.name);
                        }
                    }
                }
            }
            wi += 1;
            st.cells += 1;
            st.words += (m * 8) as u64;
            st.mismatches += bad;
        }
    }

    let launch_name = if case.x4 {
        "gemm_mq6g256v2_f32_rows_x4"
    } else {
        "gemm_mq6g256v2_f32_rows"
    };
    for rows in 1..=MAX_R {
        for on in [true, false] {
            Arc::make_mut(&mut gpu.flags).mq6_rows_split = on;
            let ys = alloc_ys(gpu, ms)?;
            let views: Vec<GpuTensor> = ys
                .iter()
                .zip(ms)
                .map(|(y, m)| y.sub_offset(GUARD, rows * m))
                .collect();
            let xv = fx.x.sub_offset(0, rows * k);
            if case.x4 {
                let parts = parts_of(&fx, ms, &views);
                if rows == 1 {
                    gpu.gemv_mq6g256v2_x4(parts, &xv, k).map_err(err)?;
                } else {
                    gpu.gemm_mq6g256v2_f32_rows_x4(parts, &xv, k, rows)
                        .map_err(err)?;
                }
            } else if rows == 1 {
                gpu.gemv_mq6g256v2(&fx.ws[0], &xv, &views[0], ms[0], k)
                    .map_err(err)?;
            } else {
                gpu.gemm_mq6g256v2_f32_rows(&fx.ws[0], &xv, &views[0], ms[0], k, rows)
                    .map_err(err)?;
            }
            let tag = format!(
                "{} seed {seed:#x} {launch_name} R{rows} flag={on}",
                case.name
            );
            check(gpu, &ys, ms, rows, &refs, &tag, st)?;
        }
    }

    // Every split export by symbol.
    gpu.ensure_kernel_public(
        "gemm_mq6g256v2_f32_rows",
        kernels::GEMM_MQ6G256V2_F32_ROWS_SRC,
        "gemm_mq6g256v2_f32_rows_r2",
    )
    .map_err(err)?;
    for rows in 3..=8usize {
        for (w, s) in CONFIGS {
            let sym = format!(
                "gemm_mq6g256v2_f32_rows_{}split_r{rows}_w{w}_s{s}",
                if case.x4 { "x4_" } else { "" }
            );
            gpu.ensure_kernel_public(
                "gemm_mq6g256v2_f32_rows",
                kernels::GEMM_MQ6G256V2_F32_ROWS_SRC,
                &sym,
            )
            .map_err(err)?;
            let ys = alloc_ys(gpu, ms)?;
            let views: Vec<GpuTensor> = ys
                .iter()
                .zip(ms)
                .map(|(y, m)| y.sub_offset(GUARD, rows * m))
                .collect();
            let parts = parts_of(&fx, ms, &views);
            let mut b = hip_bridge::KernargBlob::new();
            let grid: usize;
            if case.x4 {
                for p in &parts {
                    b.push_ptr(p.0.buf.as_ptr());
                }
                b.push_ptr(fx.x.buf.as_ptr());
                for p in &parts {
                    b.push_ptr(p.1.buf.as_ptr());
                }
                for p in &parts {
                    b.push_i32(p.2 as i32);
                }
                b.push_i32(k as i32);
                grid = ms.iter().map(|m| m.div_ceil(w)).sum();
            } else {
                b.push_ptr(parts[0].0.buf.as_ptr());
                b.push_ptr(fx.x.buf.as_ptr());
                b.push_ptr(parts[0].1.buf.as_ptr());
                b.push_i32(ms[0] as i32);
                b.push_i32(k as i32);
                grid = ms[0].div_ceil(w);
            }
            gpu.launch_kernel_blob(
                &sym,
                [grid as u32, 1, 1],
                [(32 * s) as u32, 1, 1],
                0,
                b.as_mut_slice(),
            )
            .map_err(err)?;
            let tag = format!("{} seed {seed:#x} raw {sym}", case.name);
            check(gpu, &ys, ms, rows, &refs, &tag, st)?;
        }
    }

    // Inputs untouched.
    let xd = gpu.download_f32(&fx.x).map_err(err)?;
    st.cells += 1;
    st.words += xd.len() as u64;
    if xd
        .iter()
        .zip(&fx.x_host)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        eprintln!("MISMATCH x modified in {}", case.name);
        st.mismatches += 1;
    }
    for (t, h) in fx.ws.iter().zip(&fx.w_host) {
        let mut d = vec![0u8; t.buf.size()];
        gpu.hip.memcpy_dtoh(&mut d, &t.buf).map_err(err)?;
        st.cells += 1;
        st.words += (h.len() / 4) as u64;
        if d[..h.len()] != h[..] {
            eprintln!("MISMATCH weights modified in {}", case.name);
            st.mismatches += 1;
        }
    }
    Ok(())
}

fn cases() -> Vec<Case> {
    let single = |name: &str, m: usize, k: usize, cpu: bool| Case {
        name: name.to_string(),
        ms: [m, 0, 0, 0],
        k,
        x4: false,
        cpu,
    };
    let quad = |name: &str, ms: [usize; 4], k: usize, cpu: bool| Case {
        name: name.to_string(),
        ms,
        k,
        x4: true,
        cpu,
    };
    let mut v = vec![
        quad("gdn_in_x4", [10240, 6144, 48, 48], 2560, false),
        single("gdn_out", 2560, 6144, true),
        quad("qsa_in_x4", [640, 12288, 512, 512], 2560, false),
        single("qsa_o", 2560, 6144, false),
        single("lm_head", 248320, 2560, false),
    ];
    for k in [256, 512, 768, 1280, 2304] {
        for m in [1, 7, 48, 1001] {
            v.push(single(&format!("edge_m{m}_k{k}"), m, k, true));
        }
        v.push(quad(
            &format!("edge_x4_one_empty_k{k}"),
            [1001, 7, 0, 48],
            k,
            true,
        ));
        v.push(quad(
            &format!("edge_x4_three_empty_k{k}"),
            [48, 0, 0, 0],
            k,
            true,
        ));
        v.push(quad(&format!("edge_x4_gap_k{k}"), [0, 0, 1001, 1], k, true));
    }
    v
}

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().map_err(err)?;
    let arch = gpu.arch.clone();
    let mut st = Stats::default();
    for case in cases() {
        for seed in SEEDS {
            let before = st.mismatches;
            run_case(&mut gpu, &case, seed, &mut st)?;
            eprintln!(
                "{} seed {seed:#x}: mismatches so far {} (+{})",
                case.name,
                st.mismatches,
                st.mismatches - before
            );
        }
    }
    println!(
        "mq6_rows_split_oracle arch={arch} cells={} words={} mismatches={}",
        st.cells, st.words, st.mismatches
    );
    if st.mismatches != 0 {
        std::process::exit(1);
    }
    Ok(())
}
