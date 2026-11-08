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
    let ns = [9usize, 16, 17, 24, 32, 33, 40, 47, 48, 64];
    let mut t = Totals::default();
    let mut fails: Vec<String> = Vec::new();
    let mut neg: Option<bool> = None;
    let mut perm_cells = 0usize;
    println!("group  N parts | words      mism  tail_chk  tail_viol | symbol");
    for (gi, (gname, ms4)) in GROUPS.iter().enumerate() {
        let mut ws: Vec<GpuTensor> = Vec::new();
        for (i, &m) in ms4.iter().enumerate() {
            ws.push(upload_weights(
                &mut gpu,
                m,
                0x1000 * (gi as u64 + 1) + i as u64,
            )?);
        }
        for &n in &ns {
            let x = upload_f32(&mut gpu, &activations(0x77 + n as u64 + gi as u64, n, K))?;
            let xq = gpu.mq6_i8_prepare_x(&x, K, n)?;
            // reference: today's per-member launches
            let mut refs: Vec<Vec<u32>> = Vec::new();
            for (i, &m) in ms4.iter().enumerate() {
                let y = upload_words(&mut gpu, &vec![0xFFC0_0001u32; m * n])?;
                gpu.gemm_mq6sym_i8_prequant(&ws[i], xq, &y, m, K, n)?;
                gpu.hip.device_synchronize()?;
                refs.push(download_words(&gpu, &y, m * n)?);
            }
            // (parts, order, label)
            let mut cases: Vec<(usize, Vec<usize>, String)> =
                vec![(4, vec![0, 1, 2, 3], "fwd".into())];
            // permutation check: once per group (first N), plus every cell is cheap so do all N
            cases.push((4, vec![3, 2, 1, 0], "rev".into()));
            if gi == 0 && (n == 24 || n == 48) {
                cases.push((2, vec![0, 1], "p2".into()));
                cases.push((3, vec![0, 1, 2], "p3".into()));
            }
            for (np, order, tag) in cases {
                let sub_ms: Vec<usize> = ms4[..np].to_vec();
                let out = run_x4(&mut gpu, &ws[..np], &sub_ms, &order, xq, n)?;
                let before = (t.words, t.mismatches, t.tail_words, t.tail_violations);
                let label = format!("{gname}/N{n}/{tag}");
                compare(&mut t, &label, &out, &refs[..np], &mut fails);
                if tag == "rev" {
                    perm_cells += 1;
                }
                println!(
                    "{gname} {n:>3} {np}{tag:>4} | {:>9} {:>5} {:>9} {:>10} | {}",
                    t.words - before.0,
                    t.mismatches - before.1,
                    t.tail_words - before.2,
                    t.tail_violations - before.3,
                    out.symbol
                );
                // negative control, once: corrupt one reference word, comparer must see it
                if neg.is_none() {
                    let mut bad = refs[0].clone();
                    let mid = bad.len() / 2;
                    bad[mid] ^= 1 << 7;
                    let mm = count_mismatch(&out.words[0], &bad);
                    neg = Some(mm >= 1);
                    println!(
                        "negative control ({label} member 0, 1 flipped bit): {mm} mismatch(es)"
                    );
                }
            }
        }
    }
    let neg_ok = neg == Some(true);
    println!(
        "TOTAL cells={} words={} mismatches={} tail_words={} tail_violations={} perm_cells={} negative_control={}",
        t.cells,
        t.words,
        t.mismatches,
        t.tail_words,
        t.tail_violations,
        perm_cells,
        if neg_ok { "PASS" } else { "FAIL" }
    );
    for f in &fails {
        eprintln!("FAIL {f}");
    }
    if t.mismatches != 0 || t.tail_violations != 0 || !neg_ok {
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
