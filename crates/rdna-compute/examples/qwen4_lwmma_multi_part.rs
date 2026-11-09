// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-part lane-WMMA launch (`HIPFIRE_QWEN4_LWMMA_MULTI_PART`): byte ORACLE + timing bench
//! (exact gfx1201; developer example).
//!
//! `oracle`: up to four BF16 projections of one activation (the wide MoE block's dense front:
//! router | selector | shared gate | shared up) in ONE `lwmma_gemm_bf16_t2_m<MT>_n1_multi` launch
//! against today's `lane_wmma_gemm` (two terms, product config, auto split-K) once per part.
//! Real shapes (K = 2560; router M = 512, selector M = 1, shared gate/up M = 384 | 256), part
//! subsets and orders, ragged M, a part whose auto split-K is 1, other K, rows 9..=64 (every
//! token-tile count MT = 1..=4). Every cell word-compares (u32 bits) all outputs, the inputs
//! (must be untouched) and sentinel tails past each output's extent, outputs start NaN-dirty
//! (a different pattern per arm, so an unwritten word cannot match), the multi scratch counters
//! must be left zero, the multi launch is repeated over the dirty partial scratch, and `x`
//! carries inf / NaN rows in some cells. Negative controls (flipped output bit, perturbed
//! weight) must be DETECTED. The `Gpu::gemv_bf16_xf32_x4_rows` route cells (flag
//! `HIPFIRE_QWEN4_LWMMA_MULTI_PART=1` in the environment) check that the product route loads a
//! `_multi` entry exactly when the flag is on and that it is byte-identical to the per-part
//! reference either way.
//! `bench`: per-layer launch-sequence times, one launch per part vs one multi launch, over 8
//! rotating weight sets.
//!
//! Usage: [HIPFIRE_QWEN4_LWMMA_MULTI_PART=1] qwen4_lwmma_multi_part oracle | bench

use rdna_compute::lane_wmma::{
    lane_wmma_gemm, lane_wmma_gemm_multi, lane_wmma_multi_plan, LaneWmmaCfg, LaneWmmaMultiPart,
    LaneWmmaScratch, LaneWmmaWeight, LANE_WMMA_MULTI_SCRATCH_COUNTERS,
    LANE_WMMA_MULTI_SCRATCH_PARTIAL_FLOATS, LANE_WMMA_SCRATCH_COUNTERS,
    LANE_WMMA_SCRATCH_PARTIAL_FLOATS,
};
use rdna_compute::{DType, Gpu, GpuTensor};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const ROWS: [usize; 13] = [9, 12, 16, 17, 24, 31, 32, 33, 47, 48, 49, 63, 64];
const TAIL_WORDS: usize = 1024;
const SENTINELS: [u32; 4] = [0xDEAD_BEEF, 0xCAFE_F00D, 0x1234_5678, 0xA5A5_5A5A];
const DIRTY_REF: u32 = 0x7FC0_A5A5;
const DIRTY_MULTI: u32 = 0x7FC0_5A5A;

/// (label, K, part output widths)
const CASES: &[(&str, usize, &[usize])] = &[
    ("front_r0", 2560, &[512, 1, 384, 384]),
    ("front_r1", 2560, &[512, 1, 256, 256]),
    ("sel_gate_up_r0", 2560, &[1, 384, 384]),
    ("sel_gate_up_r1", 2560, &[1, 256, 256]),
    ("router_sel", 2560, &[512, 1]),
    ("gate_up", 2560, &[384, 384]),
    ("permuted", 2560, &[384, 512, 1, 384]),
    ("ragged", 2560, &[37, 1, 100, 17]),
    ("split1_part", 2560, &[6144, 1, 384]),
    ("k2048", 2048, &[256, 1, 512, 512]),
    ("k4096", 4096, &[512, 1, 512, 512]),
];

const EDGES: [u32; 8] = [
    0x7FC0_0000,
    0x7F80_0001,
    0x7F80_0000,
    0xFF80_0000,
    0x8000_0000,
    0x0000_0400,
    0x7149_F2CA,
    0x3F80_8000,
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

/// `rows * k` BF16 weight bits (u16 LE), magnitude ~0.05.
fn bf16_weights(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed ^ 0x4C57_4D4D_4D50_0001);
    let mut out = Vec::with_capacity(rows * k * 2);
    for _ in 0..rows * k {
        let v = (r.unit() - 0.5) * 0.1;
        out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    out
}

fn activations(seed: u64, n: usize, k: usize) -> Vec<f32> {
    let mut r = Rng(seed ^ 0x5841_0000_0000_0002);
    (0..n * k)
        .map(|i| {
            let mut x = (r.unit() - 0.5) * 4.0;
            if i % 97 == 13 {
                x *= 8.0;
            }
            x
        })
        .collect()
}

fn upload_words(gpu: &mut Gpu, d: &[u32]) -> Res<GpuTensor> {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) };
    let t = gpu.alloc_tensor(&[d.len()], DType::F32)?;
    gpu.hip.memcpy_htod(&t.buf, bytes)?;
    Ok(t)
}

fn upload_f32(gpu: &mut Gpu, d: &[f32]) -> Res<GpuTensor> {
    let w: Vec<u32> = d.iter().map(|v| v.to_bits()).collect();
    upload_words(gpu, &w)
}

fn download_words(gpu: &Gpu, t: &GpuTensor, words: usize) -> Res<Vec<u32>> {
    let mut b = vec![0u8; words * 4];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf)?;
    Ok(b
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn count_mismatch(a: &[u32], b: &[u32]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count() as u64
}

fn check_arch(gpu: &Gpu) -> Res<()> {
    if !gpu.arch_caps.is_gfx1201() {
        return Err("requires exact gfx1201".into());
    }
    Ok(())
}

/// `words` live words of `dirty`, then `TAIL_WORDS` sentinels.
fn dirty_with_tail(words: usize, dirty: u32) -> Vec<u32> {
    let mut v = vec![dirty; words];
    for i in 0..TAIL_WORDS {
        v.push(SENTINELS[i % 4]);
    }
    v
}

fn tail_violations(w: &[u32], from: usize) -> u64 {
    w[from..]
        .iter()
        .enumerate()
        .filter(|(i, v)| **v != SENTINELS[i % 4])
        .count() as u64
}

#[derive(Default)]
struct Totals {
    cells: usize,
    words: u64,
    mismatches: u64,
    tail_words: u64,
    tail_violations: u64,
    edge_cells: usize,
    counter_dirty: u64,
}

struct Cell {
    ms: Vec<usize>,
    k: usize,
    rows: usize,
    seed: u64,
    x_edge: bool,
}

struct Inputs {
    w: Vec<GpuTensor>,
    x: GpuTensor,
    x_words: Vec<u32>,
}

fn make_inputs(gpu: &mut Gpu, c: &Cell, perturb_weight: Option<usize>) -> Res<Inputs> {
    let mut w = Vec::new();
    for (p, &m) in c.ms.iter().enumerate() {
        let mut b = bf16_weights(m, c.k, c.seed ^ (0x100 * (p as u64 + 1)));
        if perturb_weight == Some(p) {
            // One BF16 element of the part's last weight row.
            let at = (m * c.k - 1) * 2 + 1;
            b[at] ^= 0x01;
        }
        w.push(gpu.upload_raw(&b, &[b.len()])?);
    }
    let mut xh = activations(c.seed ^ 0x33, c.rows, c.k);
    if c.x_edge {
        // One token row carries +inf and NaN activations: its outputs are NaN.
        let t = c.rows / 2;
        xh[t * c.k + 5] = f32::INFINITY;
        xh[t * c.k + 777 % c.k] = f32::NAN;
        let mut r = Rng(c.seed ^ 0x77);
        for e in EDGES {
            let i = (r.next() as usize) % (c.rows * c.k);
            xh[i] = f32::from_bits(e);
        }
    }
    let x_words: Vec<u32> = xh.iter().map(|v| v.to_bits()).collect();
    let x = upload_f32(gpu, &xh)?;
    Ok(Inputs { w, x, x_words })
}

struct Arm {
    outs: Vec<Vec<u32>>,
    x: Vec<u32>,
}

/// One part launch each, the product `lane_wmma_gemm` (two terms, auto split-K).
fn run_reference(gpu: &mut Gpu, scratch: &LaneWmmaScratch, c: &Cell, inp: &Inputs) -> Res<Arm> {
    let mut ys = Vec::new();
    for &m in &c.ms {
        ys.push(upload_words(gpu, &dirty_with_tail(c.rows * m, DIRTY_REF))?);
    }
    for (p, &m) in c.ms.iter().enumerate() {
        lane_wmma_gemm(
            gpu,
            &LaneWmmaWeight::Bf16(&inp.w[p]),
            &inp.x,
            &ys[p],
            m,
            c.k,
            c.rows,
            2,
            None,
            LaneWmmaCfg::default(),
            scratch,
        )?;
    }
    gpu.hip.device_synchronize()?;
    let mut outs = Vec::new();
    for (p, &m) in c.ms.iter().enumerate() {
        outs.push(download_words(gpu, &ys[p], c.rows * m + TAIL_WORDS)?);
    }
    Ok(Arm {
        outs,
        x: download_words(gpu, &inp.x, c.rows * c.k)?,
    })
}

/// One multi launch (twice, the second over the dirtied partial scratch and re-dirtied outputs).
fn run_multi(
    gpu: &mut Gpu,
    mscratch: &LaneWmmaScratch,
    c: &Cell,
    inp: &Inputs,
    tot: &mut Totals,
) -> Res<(Arm, Arm, String, [u32; 3])> {
    let mut arms = Vec::new();
    let mut kernel = String::new();
    let mut grid = [0u32; 3];
    for _ in 0..2 {
        let mut ys = Vec::new();
        for &m in &c.ms {
            ys.push(upload_words(gpu, &dirty_with_tail(c.rows * m, DIRTY_MULTI))?);
        }
        let parts: Vec<LaneWmmaMultiPart<'_>> = c
            .ms
            .iter()
            .enumerate()
            .map(|(p, &m)| LaneWmmaMultiPart {
                w: &inp.w[p],
                y: &ys[p],
                m,
            })
            .collect();
        let l = lane_wmma_gemm_multi(gpu, &parts, &inp.x, c.k, c.rows, mscratch)?;
        kernel = l.kernel;
        grid = l.grid;
        gpu.hip.device_synchronize()?;
        let counters = download_words(gpu, &mscratch.counters, LANE_WMMA_MULTI_SCRATCH_COUNTERS)?;
        tot.counter_dirty += counters.iter().filter(|&&w| w != 0).count() as u64;
        let mut outs = Vec::new();
        for (p, &m) in c.ms.iter().enumerate() {
            outs.push(download_words(gpu, &ys[p], c.rows * m + TAIL_WORDS)?);
        }
        arms.push(Arm {
            outs,
            x: download_words(gpu, &inp.x, c.rows * c.k)?,
        });
    }
    let second = arms.pop().unwrap();
    let first = arms.pop().unwrap();
    Ok((first, second, kernel, grid))
}

/// Compare an arm against the reference; adds to the totals, returns the mismatch count.
fn compare(
    reference: &Arm,
    other: &Arm,
    c: &Cell,
    x_words: &[u32],
    tot: &mut Totals,
    count: bool,
) -> u64 {
    let mut mism = 0u64;
    for (p, &m) in c.ms.iter().enumerate() {
        let live = c.rows * m;
        mism += count_mismatch(&reference.outs[p], &other.outs[p]);
        if count {
            tot.words += (live + TAIL_WORDS) as u64;
            tot.tail_words += TAIL_WORDS as u64;
            tot.tail_violations += tail_violations(&other.outs[p], live);
        }
    }
    // Inputs untouched.
    mism += count_mismatch(x_words, &other.x);
    if count {
        tot.words += x_words.len() as u64;
        tot.mismatches += mism;
    }
    mism
}

/// Words one `compare` covers for the cell (outputs with tails, then `x`).
fn first_words(c: &Cell, x_words: &[u32]) -> u64 {
    c.ms.iter().map(|&m| (c.rows * m + TAIL_WORDS) as u64).sum::<u64>() + x_words.len() as u64
}

fn oracle() -> Res<()> {
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    let flag_on = gpu.flags.qwen4_lwmma_multi_part_enabled();
    println!(
        "flag HIPFIRE_QWEN4_LWMMA_MULTI_PART enabled: {flag_on} (route cells {})",
        if flag_on {
            "must load a _multi entry"
        } else {
            "must NOT load a _multi entry"
        }
    );
    let single = LaneWmmaScratch::new(
        &mut gpu,
        LANE_WMMA_SCRATCH_PARTIAL_FLOATS,
        LANE_WMMA_SCRATCH_COUNTERS,
    )?;
    let multi = LaneWmmaScratch::new(
        &mut gpu,
        LANE_WMMA_MULTI_SCRATCH_PARTIAL_FLOATS,
        LANE_WMMA_MULTI_SCRATCH_COUNTERS,
    )?;
    let mut tot = Totals::default();
    let mut route_tot = Totals::default();
    let mut failures = 0usize;

    // ---- Gpu::gemv_bf16_xf32_x4_rows route cells (before any direct multi launch loads one) ----
    for &(label, k, ms) in CASES.iter().filter(|c| c.2.len() >= 2 && c.2.len() <= 4) {
        // The x4 entry takes 4 parts, 16..=64 rows (the lane-WMMA arm), K % 128 == 0.
        for &rows in &[16usize, 24, 32, 48, 64] {
            let c = Cell {
                ms: ms.to_vec(),
                k,
                rows,
                seed: 0xA000 + rows as u64,
                x_edge: false,
            };
            let inp = make_inputs(&mut gpu, &c, None)?;
            let reference = run_reference(&mut gpu, &single, &c, &inp)?;
            let mut ys = Vec::new();
            for &m in &c.ms {
                ys.push(upload_words(&mut gpu, &dirty_with_tail(c.rows * m, DIRTY_MULTI))?);
            }
            let slot = |i: usize| {
                let j = i.min(c.ms.len() - 1);
                (&inp.w[j], &ys[j], if i < c.ms.len() { c.ms[j] } else { 0 })
            };
            gpu.gemv_bf16_xf32_x4_rows([slot(0), slot(1), slot(2), slot(3)], &inp.x, k, rows)?;
            gpu.hip.device_synchronize()?;
            let mut outs = Vec::new();
            for (p, &m) in c.ms.iter().enumerate() {
                outs.push(download_words(&gpu, &ys[p], c.rows * m + TAIL_WORDS)?);
            }
            let arm = Arm {
                outs,
                x: download_words(&gpu, &inp.x, c.rows * c.k)?,
            };
            let before = route_tot.mismatches;
            compare(&reference, &arm, &c, &inp.x_words, &mut route_tot, true);
            route_tot.cells += 1;
            if route_tot.mismatches != before {
                failures += 1;
                println!("ROUTE MISMATCH {label} rows {rows}");
            }
        }
    }
    let loaded = gpu.loaded_kernel_names();
    let multi_loaded = loaded
        .iter()
        .filter(|n| n.starts_with("lwmma_gemm_bf16_t2_m") && n.ends_with("_n1_multi"))
        .cloned()
        .collect::<Vec<_>>();
    println!("route cells loaded multi entries: {multi_loaded:?}");
    if multi_loaded.is_empty() == flag_on {
        failures += 1;
        println!("ROUTE FAIL: multi entry loaded != flag");
    }

    // ---- direct kernel cells ----
    let mut grids = std::collections::BTreeMap::<String, usize>::new();
    for &(label, k, ms) in CASES {
        for &rows in &ROWS {
            for x_edge in [false, true] {
                let c = Cell {
                    ms: ms.to_vec(),
                    k,
                    rows,
                    seed: 0x5000 + (rows as u64) * 7 + x_edge as u64,
                    x_edge,
                };
                if lane_wmma_multi_plan(ms, k, rows).is_none() {
                    println!("skip (no plan): {label} rows {rows}");
                    continue;
                }
                let inp = make_inputs(&mut gpu, &c, None)?;
                let reference = run_reference(&mut gpu, &single, &c, &inp)?;
                let (first, second, kernel, grid) = run_multi(&mut gpu, &multi, &c, &inp, &mut tot)?;
                let before = tot.mismatches;
                compare(&reference, &first, &c, &inp.x_words, &mut tot, true);
                // The repeat over the dirtied partial scratch is word-compared too.
                let repeat = compare(&reference, &second, &c, &inp.x_words, &mut tot, false);
                tot.words += first_words(&c, &inp.x_words);
                tot.mismatches += repeat;
                tot.cells += 1;
                tot.edge_cells += x_edge as usize;
                *grids.entry(format!("{kernel} grid {grid:?}")).or_default() += 1;
                if tot.mismatches != before {
                    failures += 1;
                    println!("MISMATCH {label} rows {rows} edge {x_edge}: {}", tot.mismatches - before);
                }
            }
        }
    }

    // ---- negative controls ----
    let controls_ok;
    {
        let c = Cell {
            ms: vec![512, 1, 384, 384],
            k: 2560,
            rows: 32,
            seed: 0xC0DE,
            x_edge: false,
        };
        let inp = make_inputs(&mut gpu, &c, None)?;
        let reference = run_reference(&mut gpu, &single, &c, &inp)?;
        let mut scratch_tot = Totals::default();
        let (first, _, _, _) = run_multi(&mut gpu, &multi, &c, &inp, &mut scratch_tot)?;
        let clean = compare(&reference, &first, &c, &inp.x_words, &mut Totals::default(), false);
        // (a) one flipped output bit of the last part.
        let mut flipped = Arm {
            outs: first.outs.clone(),
            x: first.x.clone(),
        };
        flipped.outs[3][17] ^= 1;
        let a = compare(&reference, &flipped, &c, &inp.x_words, &mut Totals::default(), false);
        // (b) the multi launch over a weight with one perturbed BF16 element in part 2.
        let inp_p = make_inputs(&mut gpu, &c, Some(2))?;
        let (pert, _, _, _) = run_multi(&mut gpu, &multi, &c, &inp_p, &mut scratch_tot)?;
        let b = compare(&reference, &pert, &c, &inp.x_words, &mut Totals::default(), false);
        println!(
            "controls: clean {clean} | flipped-bit {a} | perturbed-weight {b}"
        );
        controls_ok = clean == 0 && a > 0 && b > 0;
    }

    println!("--- direct kernel cells ---");
    for (g, n) in &grids {
        println!("  {g}: {n} cells");
    }
    println!(
        "direct: cells {} (x-edge cells {}) | words compared {} | mismatches {} | tail words {} | tail violations {} | counter words left dirty {}",
        tot.cells, tot.edge_cells, tot.words, tot.mismatches, tot.tail_words, tot.tail_violations, tot.counter_dirty
    );
    println!(
        "route (Gpu::gemv_bf16_xf32_x4_rows): cells {} | words compared {} | mismatches {} | tail violations {}",
        route_tot.cells, route_tot.words, route_tot.mismatches, route_tot.tail_violations
    );
    println!("negative controls detected: {}", if controls_ok { "PASS" } else { "FAIL" });
    let pass = failures == 0
        && tot.mismatches == 0
        && tot.tail_violations == 0
        && tot.counter_dirty == 0
        && route_tot.mismatches == 0
        && route_tot.tail_violations == 0
        && controls_ok;
    println!("ORACLE {}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn bench() -> Res<()> {
    const SETS: usize = 8;
    const WARM: usize = 30;
    const ITERS: usize = 200;
    const K: usize = 2560;
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    let multi = LaneWmmaScratch::new(
        &mut gpu,
        LANE_WMMA_MULTI_SCRATCH_PARTIAL_FLOATS,
        LANE_WMMA_MULTI_SCRATCH_COUNTERS,
    )?;
    let e0 = gpu.hip.event_create()?;
    let e1 = gpu.hip.event_create()?;
    let mut json: Vec<String> = Vec::new();
    println!("parts rank rows | loop us | multi us | saved us | (median; p10/p90 in JSON)");
    // (label, part widths incl. router)
    let cases: [(&str, &str, [usize; 4]); 4] = [
        ("front4", "rank0", [512, 1, 384, 384]),
        ("front4", "rank1", [512, 1, 256, 256]),
        ("sgu3", "rank0", [1, 384, 384, 0]),
        ("sgu3", "rank1", [1, 256, 256, 0]),
    ];
    for (label, rname, ms4) in cases {
        let ms: Vec<usize> = ms4.iter().copied().filter(|&m| m > 0).collect();
        let mut sets: Vec<Vec<GpuTensor>> = Vec::new();
        for s in 0..SETS {
            let mut v = Vec::new();
            for (p, &m) in ms.iter().enumerate() {
                let b = bf16_weights(m, K, 0x7000 + (s * 8 + p) as u64);
                v.push(gpu.upload_raw(&b, &[b.len()])?);
            }
            sets.push(v);
        }
        for &rows in &[16usize, 24, 32, 48, 64] {
            let x = upload_f32(&mut gpu, &activations(0x99 + rows as u64, rows, K))?;
            let ys: Vec<GpuTensor> = ms
                .iter()
                .map(|&m| gpu.alloc_tensor(&[rows * m], DType::F32))
                .collect::<Result<_, _>>()?;
            let mut t_loop = Vec::new();
            let mut t_multi = Vec::new();
            for it in 0..WARM + ITERS {
                let a = &sets[it % SETS];
                let b = &sets[(it + 3) % SETS];
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                for (p, &m) in ms.iter().enumerate() {
                    gpu.lane_wmma_gemm_fixed(&LaneWmmaWeight::Bf16(&a[p]), &x, &ys[p], m, K, rows, 2, None)?;
                }
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                gpu.hip.event_synchronize(&e1)?;
                let tl = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0;
                let parts: Vec<LaneWmmaMultiPart<'_>> = ms
                    .iter()
                    .enumerate()
                    .map(|(p, &m)| LaneWmmaMultiPart {
                        w: &b[p],
                        y: &ys[p],
                        m,
                    })
                    .collect();
                gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
                lane_wmma_gemm_multi(&mut gpu, &parts, &x, K, rows, &multi)?;
                gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
                gpu.hip.event_synchronize(&e1)?;
                let tm = gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1000.0;
                if it >= WARM {
                    t_loop.push(tl);
                    t_multi.push(tm);
                }
            }
            let (ml, mm) = (pct(&mut t_loop, 0.5), pct(&mut t_multi, 0.5));
            println!("{label} {rname} {rows} | {ml:.2} | {mm:.2} | {:.2}", ml - mm);
            let trip = |v: &mut Vec<f64>| {
                format!("[{:.2},{:.2},{:.2}]", pct(v, 0.1), pct(v, 0.5), pct(v, 0.9))
            };
            json.push(format!(
                "{{\"parts\":\"{label}\",\"rank\":\"{rname}\",\"rows\":{rows},\"loop\":{},\"multi\":{}}}",
                trip(&mut t_loop),
                trip(&mut t_multi)
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
            eprintln!("usage: qwen4_lwmma_multi_part oracle | bench");
            std::process::exit(2);
        }
    }
}
