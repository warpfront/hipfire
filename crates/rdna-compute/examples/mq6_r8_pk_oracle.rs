// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle + timing of the packed-epilogue MQ6G256V2 few-row kernels
//! (`gemm_mq6g256v2_f32_rows[_x4]_pk_r{R}_w{W}_s{S}`, `HIPFIRE_QWEN4_MQ6_R8_PK`,
//! default off) against the incumbent stream-split kernels.
//!
//! ```text
//! mq6_r8_pk_oracle            # correctness (all shapes) + timing (trunk shapes)
//! mq6_r8_pk_oracle --no-time  # correctness only
//! mq6_r8_pk_oracle --time-only
//! ```
//!
//! Correctness, per shape, seed and `R` in `1..=8` (x rows include +-0, subnormal,
//! all-zero and cancelling rows, so a `-0.0` vs `+0.0` difference shows):
//! * reference: the decode singleton (`gemv_mq6g256v2[_x4]`) on each row alone;
//! * `off`: the public launcher, `HIPFIRE_QWEN4_MQ6_R8_PK` off (incumbent);
//! * `pk`: the public launcher with the flag on;
//! * every pk export (`R` 3..=8 x (W, S) in (4,1), (4,4), (8,4)) by symbol, and
//!   every incumbent split export by symbol.
//! All must equal the reference word for word; output allocations carry
//! 64 NaN-sentinel guard floats on both sides. Exit status is nonzero on any
//! mismatch.
//!
//! Timing (trunk shard shapes of the C8 AR step, K = 2560 / 3072): the weight
//! set is replicated to > 512 MB and cycled so every launch reads cold HBM, like
//! 36 layers back to back. Prints one `TIME` line per (shape, R, variant):
//! microseconds per launch (host-timed loop of back-to-back launches, median of
//! 5 passes) and weight GB/s. Host launch cost is included; for kernel-exact
//! numbers run the same binary under `rocprofv3 --kernel-trace`.

use rdna_compute::{kernels, Gpu, GpuTensor};
use std::sync::Arc;
use std::time::Instant;

const GUARD: usize = 64;
const MAX_R: usize = 8;
const SEEDS: [u64; 2] = [0x6d71_3672_6f77_7331, 0x0fed_cba9_8765_4321];
const CONFIGS: [(usize, usize); 3] = [(8, 4), (4, 4), (4, 1)];
const PK_MODULE: &str = "gemm_mq6g256v2_f32_rows_pk";
const INC_MODULE: &str = "gemm_mq6g256v2_f32_rows";

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
    timed: bool,
}

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

fn err<E: std::fmt::Debug>(e: E) -> String {
    format!("{e:?}")
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
    ws: &'a [GpuTensor],
    dummy: &'a GpuTensor,
    ms: [usize; 4],
    ys: &'a [GpuTensor],
) -> [(&'a GpuTensor, &'a GpuTensor, usize); 4] {
    let mut next = 0;
    std::array::from_fn(|p| {
        let w = if ms[p] == 0 {
            dummy
        } else {
            next += 1;
            &ws[next - 1]
        };
        (w, &ys[p], ms[p])
    })
}

fn set_flags(gpu: &mut Gpu, pk: bool) {
    let f = Arc::make_mut(&mut gpu.flags);
    f.mq6_rows_split = true;
    f.qwen4_mq6_r8_pk = pk;
}

fn sym_name(x4: bool, pk: bool, rows: usize, w: usize, s: usize) -> String {
    format!(
        "gemm_mq6g256v2_f32_rows_{}{}_r{rows}_w{w}_s{s}",
        if x4 { "x4_" } else { "" },
        if pk { "pk" } else { "split" }
    )
}

/// Launch one export by symbol at `rows` (`x` holds at least `rows` rows).
#[allow(clippy::too_many_arguments)]
fn launch_sym(
    gpu: &mut Gpu,
    sym: &str,
    pk: bool,
    x4: bool,
    parts: &[(&GpuTensor, &GpuTensor, usize); 4],
    x: &GpuTensor,
    k: usize,
    w: usize,
    s: usize,
) -> Result<(), String> {
    let (module, src) = if pk {
        (PK_MODULE, kernels::GEMM_MQ6G256V2_F32_ROWS_PK_SRC)
    } else {
        (INC_MODULE, kernels::GEMM_MQ6G256V2_F32_ROWS_SRC)
    };
    gpu.ensure_kernel_public(module, src, sym).map_err(err)?;
    let mut b = hip_bridge::KernargBlob::new();
    let grid: usize;
    if x4 {
        for p in parts {
            b.push_ptr(p.0.buf.as_ptr());
        }
        b.push_ptr(x.buf.as_ptr());
        for p in parts {
            b.push_ptr(p.1.buf.as_ptr());
        }
        for p in parts {
            b.push_i32(p.2 as i32);
        }
        b.push_i32(k as i32);
        grid = parts.iter().map(|p| p.2.div_ceil(w)).sum();
    } else {
        b.push_ptr(parts[0].0.buf.as_ptr());
        b.push_ptr(x.buf.as_ptr());
        b.push_ptr(parts[0].1.buf.as_ptr());
        b.push_i32(parts[0].2 as i32);
        b.push_i32(k as i32);
        grid = parts[0].2.div_ceil(w);
    }
    gpu.launch_kernel_blob(
        sym,
        [grid as u32, 1, 1],
        [(32 * s) as u32, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .map_err(err)
}

/// Public launcher at `rows` (rows == 1: the decode singleton).
fn launch_public(
    gpu: &mut Gpu,
    x4: bool,
    parts: [(&GpuTensor, &GpuTensor, usize); 4],
    x: &GpuTensor,
    k: usize,
    rows: usize,
) -> Result<(), String> {
    if x4 {
        if rows == 1 {
            gpu.gemv_mq6g256v2_x4(parts, x, k).map_err(err)
        } else {
            gpu.gemm_mq6g256v2_f32_rows_x4(parts, x, k, rows)
                .map_err(err)
        }
    } else if rows == 1 {
        gpu.gemv_mq6g256v2(parts[0].0, x, parts[0].1, parts[0].2, k)
            .map_err(err)
    } else {
        gpu.gemm_mq6g256v2_f32_rows(parts[0].0, x, parts[0].1, parts[0].2, k, rows)
            .map_err(err)
    }
}

fn upload_case(
    gpu: &mut Gpu,
    case: &Case,
    seed: u64,
) -> Result<(Vec<Vec<u8>>, Vec<f32>, Rng), String> {
    let mut rng = Rng(seed ^ (case.k as u64) ^ ((case.ms[0] as u64) << 20) ^ 0x9e37_79b9_7f4a_7c15);
    let _ = gpu;
    let mut hosts = Vec::new();
    for &m in case.ms.iter().filter(|&&m| m > 0) {
        hosts.push(gen_weights(m, case.k, &mut rng));
    }
    let mut xh = Vec::new();
    for r in 0..MAX_R {
        xh.extend(gen_x_row(r, case.k, &mut rng));
    }
    Ok((hosts, xh, rng))
}

fn run_case(gpu: &mut Gpu, case: &Case, seed: u64, st: &mut Stats) -> Result<(), String> {
    let k = case.k;
    let ms = case.ms;
    let (hosts, xh, _) = upload_case(gpu, case, seed)?;
    let dummy = gpu.upload_raw(&[0u8; 200], &[1, 256]).map_err(err)?;
    let mut ws = Vec::new();
    for (h, &m) in hosts.iter().zip(ms.iter().filter(|&&m| m > 0)) {
        ws.push(gpu.upload_raw(h, &[m, k]).map_err(err)?);
    }
    let x = gpu.upload_f32(&xh, &[MAX_R, k]).map_err(err)?;

    // Reference: the decode singleton on each activation row alone.
    let mut refs: Vec<Vec<Vec<f32>>> = Vec::new();
    for r in 0..MAX_R {
        let xr = x.sub_offset(r * k, k);
        let ys: Vec<GpuTensor> = ms
            .iter()
            .map(|&m| gpu.zeros(&[m.max(1)], rdna_compute::DType::F32))
            .collect::<Result<_, _>>()
            .map_err(err)?;
        launch_public(gpu, case.x4, parts_of(&ws, &dummy, ms, &ys), &xr, k, 1)?;
        let mut row = Vec::new();
        for (p, y) in ys.iter().enumerate() {
            let mut v = gpu.download_f32(y).map_err(err)?;
            v.truncate(ms[p]);
            row.push(v);
        }
        refs.push(row);
    }

    // Public launcher, flag off (incumbent) and on, rows 1..=8.
    for rows in 1..=MAX_R {
        for pk in [false, true] {
            set_flags(gpu, pk);
            let ys = alloc_ys(gpu, ms)?;
            let views: Vec<GpuTensor> = ys
                .iter()
                .zip(ms)
                .map(|(y, m)| y.sub_offset(GUARD, rows * m))
                .collect();
            let xv = x.sub_offset(0, rows * k);
            launch_public(
                gpu,
                case.x4,
                parts_of(&ws, &dummy, ms, &views),
                &xv,
                k,
                rows,
            )?;
            let tag = format!("{} seed {seed:#x} public R{rows} pk={pk}", case.name);
            check(gpu, &ys, ms, rows, &refs, &tag, st)?;
        }
    }
    set_flags(gpu, false);

    // Every export by symbol: incumbent split and pk.
    for rows in 3..=MAX_R {
        for (w, s) in CONFIGS {
            for pk in [false, true] {
                let sym = sym_name(case.x4, pk, rows, w, s);
                let ys = alloc_ys(gpu, ms)?;
                let views: Vec<GpuTensor> = ys
                    .iter()
                    .zip(ms)
                    .map(|(y, m)| y.sub_offset(GUARD, rows * m))
                    .collect();
                let parts = parts_of(&ws, &dummy, ms, &views);
                launch_sym(gpu, &sym, pk, case.x4, &parts, &x, k, w, s)?;
                let tag = format!("{} seed {seed:#x} raw {sym}", case.name);
                check(gpu, &ys, ms, rows, &refs, &tag, st)?;
            }
        }
    }

    // Inputs untouched.
    let xd = gpu.download_f32(&x).map_err(err)?;
    st.cells += 1;
    st.words += xd.len() as u64;
    if xd.iter().zip(&xh).any(|(a, b)| a.to_bits() != b.to_bits()) {
        eprintln!("MISMATCH x modified in {}", case.name);
        st.mismatches += 1;
    }
    for (t, h) in ws.iter().zip(&hosts) {
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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median-of-5 host-timed passes of `iters` back-to-back launches, in microseconds.
fn time_us(
    gpu: &mut Gpu,
    iters: usize,
    f: &mut dyn FnMut(&mut Gpu, usize) -> Result<(), String>,
) -> Result<f64, String> {
    for i in 0..iters / 4 + 4 {
        f(gpu, i)?;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let mut passes = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        for i in 0..iters {
            f(gpu, i)?;
        }
        gpu.hip.device_synchronize().map_err(err)?;
        passes.push(t.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }
    Ok(median(&mut passes))
}

fn time_case(gpu: &mut Gpu, case: &Case) -> Result<(), String> {
    let k = case.k;
    let ms = case.ms;
    let (hosts, xh, _) = upload_case(gpu, case, SEEDS[0])?;
    let bytes: usize = ms.iter().map(|m| m * (k / 256) * 200).sum();
    // > 512 MB of weights, cycled: every launch reads cold HBM.
    let copies = (512usize << 20).div_ceil(bytes).clamp(2, 96);
    let dummy = gpu.upload_raw(&[0u8; 200], &[1, 256]).map_err(err)?;
    let mut sets: Vec<Vec<GpuTensor>> = Vec::new();
    for _ in 0..copies {
        let mut ws = Vec::new();
        for (h, &m) in hosts.iter().zip(ms.iter().filter(|&&m| m > 0)) {
            ws.push(gpu.upload_raw(h, &[m, k]).map_err(err)?);
        }
        sets.push(ws);
    }
    let x = gpu.upload_f32(&xh, &[MAX_R, k]).map_err(err)?;
    let ys = alloc_ys(gpu, ms)?;
    let views: Vec<GpuTensor> = ys
        .iter()
        .zip(ms)
        .map(|(y, m)| y.sub_offset(GUARD, MAX_R * m))
        .collect();
    let iters = (copies * 2).max(96);
    let gbps = |us: f64| bytes as f64 / (us * 1e-6) / 1e9;
    let shape = format!(
        "{} M={} K={} MB={:.2}",
        case.name,
        ms.iter().sum::<usize>(),
        k,
        bytes as f64 / 1e6
    );
    for rows in 1..=MAX_R {
        let xv = x.sub_offset(0, rows * k);
        let ysv: Vec<GpuTensor> = views
            .iter()
            .zip(ms)
            .map(|(y, m)| y.sub_offset(0, rows * m))
            .collect();
        set_flags(gpu, false);
        let us = time_us(gpu, iters, &mut |g, i| {
            launch_public(
                g,
                case.x4,
                parts_of(&sets[i % copies], &dummy, ms, &ysv),
                &xv,
                k,
                rows,
            )
        })?;
        println!(
            "TIME {shape} R={rows} incumbent_public us={us:.2} GBps={:.0}",
            gbps(us)
        );
        if rows < 3 {
            continue;
        }
        for (w, s) in CONFIGS {
            for pk in [false, true] {
                let sym = sym_name(case.x4, pk, rows, w, s);
                let us = time_us(gpu, iters, &mut |g, i| {
                    let parts = parts_of(&sets[i % copies], &dummy, ms, &ysv);
                    launch_sym(g, &sym, pk, case.x4, &parts, &xv, k, w, s)
                })?;
                println!(
                    "TIME {shape} R={rows} {}_w{w}s{s} us={us:.2} GBps={:.0}",
                    if pk { "pk" } else { "split" },
                    gbps(us)
                );
            }
        }
        set_flags(gpu, true);
        let us = time_us(gpu, iters, &mut |g, i| {
            launch_public(
                g,
                case.x4,
                parts_of(&sets[i % copies], &dummy, ms, &ysv),
                &xv,
                k,
                rows,
            )
        })?;
        println!(
            "TIME {shape} R={rows} pk_public us={us:.2} GBps={:.0}",
            gbps(us)
        );
        set_flags(gpu, false);
    }
    Ok(())
}

fn cases() -> Vec<Case> {
    let single = |name: &str, m: usize, k: usize, timed: bool| Case {
        name: name.to_string(),
        ms: [m, 0, 0, 0],
        k,
        x4: false,
        timed,
    };
    let quad = |name: &str, ms: [usize; 4], k: usize, timed: bool| Case {
        name: name.to_string(),
        ms,
        k,
        x4: true,
        timed,
    };
    // Trunk shard shapes of the EP2 C8 AR step (per rank): GDN in-projection
    // x4 (grid 65920 threads / 32 = 2060 one-wave blocks of 4 rows), out
    // projection M2560 K3072 (grid 81920), QSA x4 (grid 249856 / 128 = 1952).
    let mut v = vec![
        quad("gdn_in_x4_shard", [5120, 3072, 24, 24], 2560, true),
        single("out_proj_shard", 2560, 3072, true),
        quad("qsa_in_x4_shard", [1152, 6144, 256, 256], 2560, true),
        quad("gdn_in_x4_full", [10240, 6144, 48, 48], 2560, false),
        single("lm_head_like", 20000, 2560, false),
    ];
    for k in [256, 512, 768, 1280, 2304] {
        for m in [1, 7, 48, 1001] {
            v.push(single(&format!("edge_m{m}_k{k}"), m, k, false));
        }
        v.push(quad(
            &format!("edge_x4_one_empty_k{k}"),
            [1001, 7, 0, 48],
            k,
            false,
        ));
        v.push(quad(
            &format!("edge_x4_three_empty_k{k}"),
            [48, 0, 0, 0],
            k,
            false,
        ));
        v.push(quad(
            &format!("edge_x4_gap_k{k}"),
            [0, 0, 1001, 1],
            k,
            false,
        ));
    }
    v
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let do_time = !args.iter().any(|a| a == "--no-time");
    let do_check = !args.iter().any(|a| a == "--time-only");
    let mut gpu = Gpu::init().map_err(err)?;
    let arch = gpu.arch.clone();
    let mut st = Stats::default();
    if do_check {
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
            "mq6_r8_pk_oracle arch={arch} cells={} words={} mismatches={}",
            st.cells, st.words, st.mismatches
        );
        if st.mismatches != 0 {
            std::process::exit(1);
        }
    }
    if do_time {
        for case in cases().into_iter().filter(|c| c.timed) {
            time_case(&mut gpu, &case)?;
        }
    }
    Ok(())
}
