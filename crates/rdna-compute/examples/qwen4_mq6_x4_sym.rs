// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Symmetric-MQ6 x A8 trunk GEMM multi-part (x4) launcher: byte ORACLE + timing bench
//! (exact gfx1201; developer example).
//!
//! `oracle`: for the GDN group [10240, 6144, 48, 48] and the QSA group [12288, 512, 512, 640]
//! (K = 2560) at N in {9,16,17,24,32,33,40,47,48,64}: one `gemm_mq6sym_i8_prequant_x4` against
//! the per-member `gemm_mq6sym_i8_prequant` reference, word-compared (u32 bits), tails past
//! M_i x N sentinel-checked, a reversed-order permutation check per group, 2- and 3-part GDN
//! cases, and one negative control proving the comparer is live.
//! Each launch shape additionally runs the committed PM twin
//! (`kernels/gemm_mq6sym_i8_pm_gfx1201.hxaco`, sha-checked, loaded directly; live dispatch never
//! loads it) and word-compares per-member incumbent vs hipcc x4 vs PM x4, with PM tails
//! sentinel-checked, raw `block_i8_128` edge cells (+-0 / +-Inf / NaN-payload / subnormal / max
//! scales, extreme qs) on all four x4 tiles, 1-3 part launches (absent members null, M = 0), and
//! negative controls for all three comparers.
//! `bench`: per-member / summed / back-to-back / x4 launch times over 8 rotating weight sets.
//!
//! Usage: qwen4_mq6_x4_sym oracle | bench

use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const K: usize = 2560;
const GROUP_BYTES: usize = 200;
const TAIL_WORDS: usize = 4096;
const GROUPS: [(&str, [usize; 4]); 2] = [
    ("GDN", [10240, 6144, 48, 48]),
    ("QSA", [12288, 512, 512, 640]),
];
const SENTINELS: [u32; 4] = [0xDEAD_BEEF, 0xCAFE_F00D, 0x1234_5678, 0xA5A5_5A5A];

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

/// `m` rows x `k/256` symmetric MQ6G256V2 groups: F16 scales 2^-9..2^-5 with random mantissa,
/// z = -32 s exactly (exponent + 5), 192 payload bytes of random 6-bit codes.
fn sym_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed ^ 0x5359_4d36_0000_0001);
    let mut w = vec![0u8; m * (k / 256) * GROUP_BYTES];
    for g in w.chunks_exact_mut(GROUP_BYTES) {
        for h in 0..2 {
            let e = 6 + (r.next() % 5) as u16; // biased exponent 6..=10
            let s = (e << 10) | (r.next() % 1024) as u16;
            let z = 0x8000 | (s + (5 << 10));
            g[4 * h..4 * h + 2].copy_from_slice(&s.to_le_bytes());
            g[4 * h + 2..4 * h + 4].copy_from_slice(&z.to_le_bytes());
        }
        for c in g[8..].chunks_mut(8) {
            let v = r.next().to_le_bytes();
            c.copy_from_slice(&v[..c.len()]);
        }
    }
    w
}

fn activations(seed: u64, n: usize, k: usize) -> Vec<f32> {
    let mut r = Rng(seed ^ 0x5841_0000_0000_0002);
    (0..n * k)
        .enumerate()
        .map(|(i, _)| {
            let x = (r.unit() - 0.5) * 4.0;
            if i % 97 == 13 {
                x * 8.0
            } else {
                x
            }
        })
        .collect()
}

fn upload_f32(gpu: &mut Gpu, d: &[f32]) -> Res<GpuTensor> {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) };
    let t = gpu.alloc_tensor(&[d.len()], DType::F32)?;
    gpu.hip.memcpy_htod(&t.buf, bytes)?;
    Ok(t)
}

fn upload_words(gpu: &mut Gpu, d: &[u32]) -> Res<GpuTensor> {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) };
    let t = gpu.alloc_tensor(&[d.len()], DType::F32)?;
    gpu.hip.memcpy_htod(&t.buf, bytes)?;
    Ok(t)
}

fn download_words(gpu: &Gpu, t: &GpuTensor, words: usize) -> Res<Vec<u32>> {
    let mut b = vec![0u8; words * 4];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf)?;
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Mismatching words between two equally long word slices.
fn count_mismatch(a: &[u32], b: &[u32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn check_arch(gpu: &Gpu) -> Res<()> {
    if !gpu.arch_caps.is_gfx1201() {
        return Err("requires exact gfx1201".into());
    }
    Ok(())
}

fn upload_weights(gpu: &mut Gpu, m: usize, seed: u64) -> Res<GpuTensor> {
    let b = sym_weights(m, K, seed);
    Ok(gpu.upload_raw(&b, &[b.len()])?)
}

#[derive(Default)]
struct Totals {
    cells: usize,
    words: u64,
    mismatches: u64,
    tail_words: u64,
    tail_violations: u64,
}

struct X4Out {
    /// per member, in MEMBER order (not launch order)
    words: Vec<Vec<u32>>,
    tails: Vec<Vec<u32>>,
    symbol: &'static str,
}

/// One x4 launch of `members` (indices into `ws`/`ms`) in the given order, each into a fresh
/// sentinel-filled y of `m*n + TAIL_WORDS` words.
fn run_x4(
    gpu: &mut Gpu,
    ws: &[GpuTensor],
    ms: &[usize],
    order: &[usize],
    xq: *mut c_void,
    n: usize,
) -> Res<X4Out> {
    let mut ys: Vec<GpuTensor> = Vec::new();
    for (i, &m) in ms.iter().enumerate() {
        ys.push(upload_words(
            gpu,
            &vec![SENTINELS[i % 4]; m * n + TAIL_WORDS],
        )?);
    }
    let lm: Vec<usize> = order.iter().map(|&i| ms[i]).collect();
    let symbol = gpu
        .gemm_mq6sym_i8_x4_kernel_name(&lm, n)
        .ok_or("x4 route not taken (kernel_name = None)")?;
    let parts: Vec<(&GpuTensor, &GpuTensor, usize)> =
        order.iter().map(|&i| (&ws[i], &ys[i], ms[i])).collect();
    gpu.gemm_mq6sym_i8_prequant_x4(&parts, xq, K, n)?;
    gpu.hip.device_synchronize()?;
    let mut words = Vec::new();
    let mut tails = Vec::new();
    for (i, &m) in ms.iter().enumerate() {
        let all = download_words(gpu, &ys[i], m * n + TAIL_WORDS)?;
        words.push(all[..m * n].to_vec());
        tails.push(all[m * n..].to_vec());
    }
    Ok(X4Out {
        words,
        tails,
        symbol,
    })
}

/// The committed PM twin bundle, loaded directly (oracle only; live dispatch never loads it).
struct PmTwin {
    _module: hip_bridge::Module,
    funcs: Vec<(&'static str, hip_bridge::Function)>,
    sha: String,
}

const PM_SYMBOLS: [&str; 4] = [
    "gemm_mq6sym_i8_set_m64n16_x4_gfx1201",
    "gemm_mq6sym_i8_set_m32n32_x4_gfx1201",
    "gemm_mq6sym_i8_set_m64n48_x4_gfx1201",
    "gemm_mq6sym_i8_set_m64n16y_x4_gfx1201",
];

/// Load the committed `kernels/gemm_mq6sym_i8_pm_gfx1201.hxaco`, check it against its
/// `.sha256` sidecar, and resolve the four x4 entries.
fn load_pm_twin(gpu: &Gpu) -> Res<PmTwin> {
    use sha2::{Digest, Sha256};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gemm_mq6sym_i8_pm_gfx1201.hxaco"
    );
    let bytes = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let want = std::fs::read_to_string(format!("{path}.sha256"))
        .map_err(|e| format!("{path}.sha256: {e}"))?;
    if want.split_whitespace().next() != Some(sha.as_str()) {
        return Err(format!("{path}: sha256 {sha} != sidecar {}", want.trim()).into());
    }
    let module = gpu.hip.module_load_data(&bytes)?;
    let mut funcs = Vec::new();
    for s in PM_SYMBOLS {
        funcs.push((s, gpu.hip.module_get_function(&module, s)?));
    }
    Ok(PmTwin {
        _module: module,
        funcs,
        sha,
    })
}

/// Same launch as [`run_x4`] through the PM twin's entry `symbol` (96-byte kernarg: W0..W3,
/// Y0..Y3, Xq, M0..M3, K, N; absent parts null / M = 0). Geometry mirrors the shipped launcher.
fn run_pm_x4(
    gpu: &mut Gpu,
    pm: &PmTwin,
    symbol: &'static str,
    ws: &[GpuTensor],
    ms: &[usize],
    order: &[usize],
    xq: *mut c_void,
    n: usize,
) -> Res<X4Out> {
    let mut ys: Vec<GpuTensor> = Vec::new();
    for (i, &m) in ms.iter().enumerate() {
        ys.push(upload_words(
            gpu,
            &vec![SENTINELS[i % 4]; m * n + TAIL_WORDS],
        )?);
    }
    let func = &pm
        .funcs
        .iter()
        .find(|(s, _)| *s == symbol)
        .ok_or_else(|| format!("PM twin has no {symbol}"))?
        .1;
    let m32 = symbol.contains("m32n32");
    let rows = if m32 { 32 } else { 64 };
    let blocks: usize = order.iter().map(|&i| ms[i].div_ceil(rows)).sum();
    let (grid, block_x) = if m32 {
        ([blocks as u32, 1, 1], 64)
    } else if symbol.contains("m64n16y") {
        ([blocks as u32, n.div_ceil(16) as u32, 1], 128)
    } else {
        ([blocks as u32, 1, 1], 128)
    };
    let mut ka = [0u8; 96];
    for i in 0..4 {
        let w = order.get(i).map_or(0u64, |&j| ws[j].buf.as_ptr() as u64);
        let y = order.get(i).map_or(0u64, |&j| ys[j].buf.as_ptr() as u64);
        let m = order.get(i).map_or(0usize, |&j| ms[j]);
        ka[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
        ka[32 + 8 * i..40 + 8 * i].copy_from_slice(&y.to_le_bytes());
        ka[72 + 4 * i..76 + 4 * i].copy_from_slice(&(m as i32).to_le_bytes());
    }
    ka[64..72].copy_from_slice(&(xq as u64).to_le_bytes());
    ka[88..92].copy_from_slice(&(K as i32).to_le_bytes());
    ka[92..96].copy_from_slice(&(n as i32).to_le_bytes());
    // SAFETY: kernarg follows the 96-byte x4 ABI and addresses live allocations.
    unsafe {
        gpu.hip.launch_kernel_blob(
            func,
            grid,
            [block_x, 1, 1],
            0,
            gpu.active_stream.as_ref(),
            &mut ka,
        )?;
    }
    gpu.hip.device_synchronize()?;
    let mut words = Vec::new();
    let mut tails = Vec::new();
    for (i, &m) in ms.iter().enumerate() {
        let all = download_words(gpu, &ys[i], m * n + TAIL_WORDS)?;
        words.push(all[..m * n].to_vec());
        tails.push(all[m * n..].to_vec());
    }
    Ok(X4Out {
        words,
        tails,
        symbol,
    })
}

/// Raw `block_i8_128` activations (`{f32 d; i32 s; i8 qs[128]}`, layout `[K/128][n]`) with
/// hostile scales: +0, -0, +Inf, -Inf, quiet NaN with payload, signaling-pattern negative NaN,
/// smallest subnormal, f32::MAX; qs random or extreme (-128/127/0 runs); `s` = sum(qs).
/// The kernels' words are compared bitwise, never numerically.
fn edge_xq(n: usize, seed: u64) -> Vec<u8> {
    const SCALES: [u32; 8] = [
        0x0000_0000,
        0x8000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_1234,
        0xFFA0_0001,
        0x0000_0001,
        0x7F7F_FFFF,
    ];
    let mut r = Rng(seed);
    let mut out = Vec::with_capacity(K / 128 * n * 136);
    for kb in 0..K / 128 {
        for t in 0..n {
            let sel = (t + kb) % 8;
            let mut qs = [0i8; 128];
            for (j, q) in qs.iter_mut().enumerate() {
                *q = match (t + 2 * kb) % 4 {
                    0 => (r.next() & 0xFF) as u8 as i8,
                    1 => {
                        if j % 2 == 0 {
                            -128
                        } else {
                            127
                        }
                    }
                    2 => {
                        if (j / 16) % 2 == 0 {
                            127
                        } else {
                            0
                        }
                    }
                    _ => -128,
                };
            }
            let s: i32 = qs.iter().map(|&q| q as i32).sum();
            out.extend_from_slice(&SCALES[sel].to_le_bytes());
            out.extend_from_slice(&s.to_le_bytes());
            out.extend(qs.iter().map(|&q| q as u8));
        }
    }
    out
}

#[derive(Default)]
struct Stats {
    /// hipcc x4 vs per-member incumbent
    hip: Totals,
    /// PM x4 vs per-member incumbent
    pm: Totals,
    /// PM x4 vs hipcc x4, same launch shape
    hp_words: u64,
    hp_mism: u64,
    perm_cells: usize,
    edge_cells: usize,
    neg: Vec<(String, bool)>,
    symbols: std::collections::BTreeSet<&'static str>,
    fails: Vec<String>,
}

/// All cases of one (group, N) cell: per-member incumbent refs, then for each launch shape the
/// hipcc x4 and the PM x4 outputs, compared three ways.
#[allow(clippy::too_many_arguments)]
fn cell(
    gpu: &mut Gpu,
    pm: &PmTwin,
    st: &mut Stats,
    gname: &str,
    ms4: &[usize; 4],
    ws: &[GpuTensor],
    xq: *mut c_void,
    n: usize,
    edge: bool,
) -> Res<()> {
    let mut refs: Vec<Vec<u32>> = Vec::new();
    for (i, &m) in ms4.iter().enumerate() {
        let y = upload_words(gpu, &vec![0xFFC0_0001u32; m * n])?;
        gpu.gemm_mq6sym_i8_prequant(&ws[i], xq, &y, m, K, n)?;
        gpu.hip.device_synchronize()?;
        refs.push(download_words(gpu, &y, m * n)?);
    }
    let mut cases: Vec<(usize, Vec<usize>, String)> = vec![
        (4, vec![0, 1, 2, 3], "fwd".into()),
        (4, vec![3, 2, 1, 0], "rev".into()),
    ];
    if edge || (gname == "GDN" && (n == 24 || n == 48)) {
        cases.push((2, vec![0, 1], "p2".into()));
        cases.push((3, vec![0, 1, 2], "p3".into()));
    }
    for (np, order, tag) in cases {
        let sub_ms: Vec<usize> = ms4[..np].to_vec();
        let hip = run_x4(gpu, &ws[..np], &sub_ms, &order, xq, n)?;
        let pmo = run_pm_x4(gpu, pm, hip.symbol, &ws[..np], &sub_ms, &order, xq, n)?;
        st.symbols.insert(hip.symbol);
        let label = format!("{gname}{}/N{n}/{tag}", if edge { "/edge" } else { "" });
        let b0 = (
            st.hip.words,
            st.hip.mismatches,
            st.pm.mismatches,
            st.pm.tail_violations,
            st.hp_mism,
        );
        compare(&mut st.hip, &label, &hip, &refs[..np], &mut st.fails);
        compare(
            &mut st.pm,
            &format!("{label}/pm"),
            &pmo,
            &refs[..np],
            &mut st.fails,
        );
        let mut hp = 0usize;
        for i in 0..np {
            hp += count_mismatch(&hip.words[i], &pmo.words[i]);
            st.hp_words += hip.words[i].len() as u64;
        }
        st.hp_mism += hp as u64;
        if hp != 0 {
            st.fails
                .push(format!("{label}: hipcc x4 vs PM x4 {hp} word mismatches"));
        }
        if tag == "rev" {
            st.perm_cells += 1;
        }
        if edge && tag == "fwd" {
            st.edge_cells += 1;
        }
        println!(
            "{label:<22} {np}{tag:>4} | words {:>9} hip-mism {:>4} pm-mism {:>4} pm-tail {:>3} hip-vs-pm {:>4} | {}",
            st.hip.words - b0.0,
            st.hip.mismatches - b0.1,
            st.pm.mismatches - b0.2,
            st.pm.tail_violations - b0.3,
            st.hp_mism - b0.4,
            hip.symbol
        );
        // negative controls, once each: a flipped bit must be seen by every comparer
        if st.neg.is_empty() {
            let mut bad = refs[0].clone();
            let mid = bad.len() / 2;
            bad[mid] ^= 1 << 7;
            let a = count_mismatch(&hip.words[0], &bad) >= 1;
            let b = count_mismatch(&pmo.words[0], &bad) >= 1;
            let mut bad_hip = hip.words[0].clone();
            bad_hip[mid] ^= 1 << 7;
            let c = count_mismatch(&bad_hip, &pmo.words[0]) == 1;
            println!("negative control ({label} member 0, 1 flipped bit): incumbent-vs-hipcc {a}, incumbent-vs-PM {b}, hipcc-vs-PM {c}");
            st.neg.push(("hip-vs-incumbent".into(), a));
            st.neg.push(("pm-vs-incumbent".into(), b));
            st.neg.push(("hip-vs-pm".into(), c));
        }
    }
    Ok(())
}

/// One group's random cells (`ns`) and raw-edge cells (`edge_ns`).
#[allow(clippy::too_many_arguments)]
fn group(
    gpu: &mut Gpu,
    pm: &PmTwin,
    st: &mut Stats,
    gname: &str,
    ms4: &[usize; 4],
    seed: u64,
    ns: &[usize],
    edge_ns: &[usize],
) -> Res<()> {
    let mut ws: Vec<GpuTensor> = Vec::new();
    for (i, &m) in ms4.iter().enumerate() {
        ws.push(upload_weights(gpu, m, seed + i as u64)?);
    }
    for &n in ns {
        let x = upload_f32(gpu, &activations(0x77 + n as u64 + seed, n, K))?;
        let xq = gpu.mq6_i8_prepare_x(&x, K, n)?;
        cell(gpu, pm, st, gname, ms4, &ws, xq, n, false)?;
    }
    for &n in edge_ns {
        let raw = edge_xq(n, 0xED6E + n as u64 + seed);
        let t = gpu.alloc_tensor(&[raw.len() / 4], DType::F32)?;
        gpu.hip.memcpy_htod(&t.buf, &raw)?;
        let xq = t.buf.as_ptr() as *mut c_void;
        cell(gpu, pm, st, gname, ms4, &ws, xq, n, true)?;
    }
    Ok(())
}

fn compare(t: &mut Totals, label: &str, out: &X4Out, refs: &[Vec<u32>], fails: &mut Vec<String>) {
    t.cells += 1;
    for (i, r) in refs.iter().enumerate() {
        let mm = count_mismatch(&out.words[i], r);
        t.words += r.len() as u64;
        t.mismatches += mm as u64;
        let sent = SENTINELS[i % 4];
        let tv = out.tails[i].iter().filter(|&&w| w != sent).count();
        t.tail_words += out.tails[i].len() as u64;
        t.tail_violations += tv as u64;
        if mm != 0 || tv != 0 {
            fails.push(format!(
                "{label} member {i}: {mm} mismatches, {tv} tail violations"
            ));
        }
    }
}

fn oracle() -> Res<()> {
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    gpu.set_qwen4_trunk_a8_verified(true);
    let pm = load_pm_twin(&gpu)?;
    println!("PM twin sha256 {}", pm.sha);
    let ns = [9usize, 16, 17, 24, 32, 33, 40, 47, 48, 64];
    let mut st = Stats::default();
    println!("cell | per-launch word counts / mismatches | symbol");
    // Random finite cells; edge cells hit all four tiles: GDN n16 m64n16, n32 m32n32, n48
    // m64n48, n64 m64n16y (M = 10240 largest).
    group(
        &mut gpu,
        &pm,
        &mut st,
        GROUPS[0].0,
        &GROUPS[0].1,
        0x1000,
        &ns,
        &[16, 32, 48, 64],
    )?;
    group(
        &mut gpu,
        &pm,
        &mut st,
        GROUPS[1].0,
        &GROUPS[1].1,
        0x2000,
        &ns,
        &[16, 32, 48, 64],
    )?;
    // Small-M members: n >= 24 with M <= 2560 is the token-split m64n16y tile below n = 49.
    group(
        &mut gpu,
        &pm,
        &mut st,
        "SMALL",
        &[2048, 512, 128, 64],
        0x3000,
        &[24, 40],
        &[24, 40],
    )?;
    for want in PM_SYMBOLS {
        if !st.symbols.contains(want) {
            st.fails.push(format!("tile {want} never exercised"));
        }
    }
    let neg_ok = st.neg.len() == 3 && st.neg.iter().all(|(_, ok)| *ok);
    println!(
        "TOTAL cells={} edge_cells={} words={} hip_mismatches={} hip_tail_violations={} pm_mismatches={} pm_tail_violations={} hipcc_vs_pm_words={} hipcc_vs_pm_mismatches={} perm_cells={} tiles={}/4 negative_control={}",
        st.hip.cells,
        st.edge_cells,
        st.hip.words,
        st.hip.mismatches,
        st.hip.tail_violations,
        st.pm.mismatches,
        st.pm.tail_violations,
        st.hp_words,
        st.hp_mism,
        st.perm_cells,
        st.symbols.len(),
        if neg_ok { "PASS" } else { "FAIL" }
    );
    for f in &st.fails {
        eprintln!("FAIL {f}");
    }
    if !st.fails.is_empty()
        || st.hip.mismatches != 0
        || st.hip.tail_violations != 0
        || st.pm.mismatches != 0
        || st.pm.tail_violations != 0
        || st.hp_mism != 0
        || !neg_ok
    {
        std::process::exit(1);
    }
    println!("oracle PASS");
    Ok(())
}

fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn bench() -> Res<()> {
    const SETS: usize = 8;
    const WARM: usize = 30;
    const ITERS: usize = 200;
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    gpu.set_qwen4_trunk_a8_verified(true);
    let e0 = gpu.hip.event_create()?;
    let e1 = gpu.hip.event_create()?;
    let mut json: Vec<String> = Vec::new();
    println!("group N | per-member us [m0 m1 m2 m3] sum | seq4 wall | x4 | saved(sum-x4) | saved(seq-x4) | symbol  (median; p10/p90 in JSON)");
    for (gi, (gname, ms)) in GROUPS.iter().enumerate() {
        // weights[set][member]
        let mut sets: Vec<Vec<GpuTensor>> = Vec::new();
        {
            let host: Vec<Vec<u8>> = ms
                .iter()
                .enumerate()
                .map(|(i, &m)| sym_weights(m, K, 0x2000 * (gi as u64 + 1) + i as u64))
                .collect();
            for _ in 0..SETS {
                let mut s = Vec::new();
                for b in &host {
                    s.push(gpu.upload_raw(b, &[b.len()])?);
                }
                sets.push(s);
            }
        }
        for &n in &[24usize, 32, 48, 64] {
            let x = upload_f32(&mut gpu, &activations(0x99 + n as u64, n, K))?;
            let xq = gpu.mq6_i8_prepare_x(&x, K, n)?;
            let ys: Vec<GpuTensor> = ms
                .iter()
                .map(|&m| gpu.alloc_tensor(&[m * n], DType::F32))
                .collect::<Result<_, _>>()?;
            let symbol = gpu
                .gemm_mq6sym_i8_x4_kernel_name(ms, n)
                .ok_or("x4 route not taken")?;
            let mut per: Vec<Vec<f64>> = vec![Vec::new(); 4];
            let mut sums = Vec::new();
            let mut seqs = Vec::new();
            let mut x4s = Vec::new();
            for it in 0..WARM + ITERS {
                let s_m = it % SETS;
                let s_q = (it + 3) % SETS;
                let s_x = (it + 5) % SETS;
                let mut sum = 0.0;
                let mut pm = [0.0f64; 4];
                for i in 0..4 {
                    gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                    gpu.gemm_mq6sym_i8_prequant(&sets[s_m][i], xq, &ys[i], ms[i], K, n)?;
                    gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                    gpu.hip.event_synchronize(&e1)?;
                    pm[i] = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0;
                    sum += pm[i];
                }
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                for i in 0..4 {
                    gpu.gemm_mq6sym_i8_prequant(&sets[s_q][i], xq, &ys[i], ms[i], K, n)?;
                }
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                gpu.hip.event_synchronize(&e1)?;
                let seq = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0;
                let parts: Vec<(&GpuTensor, &GpuTensor, usize)> =
                    (0..4).map(|i| (&sets[s_x][i], &ys[i], ms[i])).collect();
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                gpu.gemm_mq6sym_i8_prequant_x4(&parts, xq, K, n)?;
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                gpu.hip.event_synchronize(&e1)?;
                let x4 = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0;
                if it >= WARM {
                    for i in 0..4 {
                        per[i].push(pm[i]);
                    }
                    sums.push(sum);
                    seqs.push(seq);
                    x4s.push(x4);
                }
            }
            let med: Vec<f64> = per.iter_mut().map(|v| pct(v, 0.5)).collect();
            let (ms_, mq, mx) = (pct(&mut sums, 0.5), pct(&mut seqs, 0.5), pct(&mut x4s, 0.5));
            println!(
                "{gname} {n} | [{:.1} {:.1} {:.1} {:.1}] {ms_:.1} | {mq:.1} | {mx:.1} | {:.1} | {:.1} | {symbol}",
                med[0], med[1], med[2], med[3], ms_ - mx, mq - mx
            );
            let trip = |v: &mut Vec<f64>| {
                format!("[{:.2},{:.2},{:.2}]", pct(v, 0.1), pct(v, 0.5), pct(v, 0.9))
            };
            json.push(format!(
                "{{\"g\":\"{gname}\",\"n\":{n},\"sym\":\"{symbol}\",\"m0\":{},\"m1\":{},\"m2\":{},\"m3\":{},\"sum\":{},\"seq4\":{},\"x4\":{}}}",
                trip(&mut per[0]),
                trip(&mut per[1]),
                trip(&mut per[2]),
                trip(&mut per[3]),
                trip(&mut sums),
                trip(&mut seqs),
                trip(&mut x4s)
            ));
        }
    }
    println!(
        "{{\"unit\":\"us\",\"stats\":\"[p10,p50,p90]\",\"iters\":{ITERS},\"warmups\":{WARM},\"sets\":{SETS},\"cells\":[{}]}}",
        json.join(",")
    );
    let _ = gpu.hip.event_destroy(e0);
    let _ = gpu.hip.event_destroy(e1);
    Ok(())
}

fn main() -> Res<()> {
    match std::env::args().nth(1).as_deref() {
        Some("oracle") => oracle(),
        Some("bench") => bench(),
        _ => {
            eprintln!("usage: qwen4_mq6_x4_sym oracle | bench");
            std::process::exit(2);
        }
    }
}
