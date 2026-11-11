// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-batched MQ6G256V2 draft-head re-score (`HIPFIRE_QWEN4_DRAFT_TOPK8_ROWS`): byte ORACLE +
//! timing bench (exact gfx1201; developer example).
//!
//! `oracle`: `Gpu::topk8_rescore_k2560_rows` (`topk8_partial_f32_rows` +
//! `topk8_rescore_mq6g256v2_k2560_rows`) against `rows` incumbent `Gpu::topk8_rescore_k2560`
//! calls, each on its own logits row / x row / partial slot / out slot. Every cell compares every
//! 72-byte out slot, the gaps between padded slots and a guard after the last one, and every
//! row's 8192-byte partial slot plus a guard slot, byte for byte (both arms start from sentinel
//! fills). Sweep: rows 1..=8 (packed strides, and padded strides), the 2 + 2 + ... runs of the
//! served split, n in {248320, 151936, 100276, 1000, 5}, orders identity and a
//! (front, special, tail) reorder, logit families finite random / BF16 ties / small-integer ties /
//! constant / signed zeros / NaN rows and NaN chunks / +-inf / -inf padding / ties on group edges
//! (each row of a cell takes a different family; the family offset rotates), x rows finite / with
//! NaN+inf / zero, against a random-bytes MQ6G256V2 head of [n_max x 2560] (200 B per 256-group;
//! most group headers finite, one in sixteen fully random). Controls (must DETECT): one logit of
//! the winning row flipped, and two x rows swapped, both applied to the rows arm only.
//! Prints `PASS` / `FAIL`, the counts, and writes JSON to the output path.
//!
//! `bench`: interleaved (arm order alternates every rep), device-synchronized, 200 reps after
//! 30 warmups, per-row loop (2 launches x rows) vs the rows pair (2 launches) at rows 2, 3, 4, 8
//! and n = 248320 and 100276: median / min GPU us (hipEvent) and host wall us (launch included)
//! per round, and the saving.
//!
//! Usage: qwen4_topk8_rows oracle [out.json] | bench [out.json]
//! Device: the standard `Gpu::init()` selection (`HIPFIRE_DEVICES`).

use rdna_compute::gemv::TOPK8_ROWS_MAX;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const K: usize = 2560;
const OUT: usize = Gpu::TOPK8_OUT_BYTES;
const PART: usize = Gpu::TOPK8_PARTIAL_BYTES;
const N_MAX: usize = 248_320;
const GUARD: usize = 256;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

fn bf16(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xffff_0000)
}

fn check_arch(gpu: &Gpu) -> Res<()> {
    if !gpu.arch_caps.is_gfx1201() {
        return Err("requires exact gfx1201".into());
    }
    Ok(())
}

const FAMILIES: usize = 11;
const FAMILY_NAMES: [&str; FAMILIES] = [
    "random",
    "bf16-ties",
    "int-ties",
    "constant",
    "signed-zeros",
    "nan-row",
    "nan-chunk",
    "+-inf",
    "-inf-padding",
    "edge-ties",
    "all-nan",
];

/// One logit row of family `f` over `n` entries.
fn make_row(g: &mut Rng, n: usize, f: usize) -> Vec<f32> {
    let chunk = n.div_ceil(128);
    let edge = (64 * chunk).min(n.saturating_sub(1));
    let mut r: Vec<f32> = (0..n).map(|_| g.unit() * 8.0).collect();
    match f {
        0 => {}
        1 => r = (0..n).map(|_| bf16(g.unit() * 14.0)).collect(),
        2 => r = (0..n).map(|_| (g.next() % 5) as f32).collect(),
        3 => r = vec![1.5; n],
        4 => {
            r = (0..n)
                .map(|_| if g.next() & 1 == 0 { 0.0 } else { -0.0 })
                .collect()
        }
        5 => {
            for _ in 0..n.min(64) {
                let i = g.next() as usize % n;
                r[i] = f32::NAN;
            }
        }
        6 => {
            // A whole group of NaN.
            let lo = (g.next() as usize % 128) * chunk;
            for v in r.iter_mut().skip(lo).take(chunk) {
                *v = f32::NAN;
            }
        }
        7 => {
            for _ in 0..n.min(32) {
                let i = g.next() as usize % n;
                r[i] = if g.next() & 1 == 0 {
                    f32::INFINITY
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
        8 => {
            r = vec![f32::NEG_INFINITY; n];
            for _ in 0..40.min(n) {
                let i = g.next() as usize % n;
                r[i] = bf16(g.unit());
            }
        }
        9 => {
            for v in &mut r {
                *v = -1.0 - v.abs();
            }
            for k in 0..8usize {
                if edge >= 1 + k {
                    r[edge - 1 - k] = 20.0;
                }
                if edge + k < n {
                    r[edge + k] = 20.0;
                }
            }
        }
        _ => r = vec![f32::NAN; n],
    }
    r
}

fn make_x(g: &mut Rng, kind: usize) -> Vec<f32> {
    let mut x: Vec<f32> = (0..K).map(|_| g.unit()).collect();
    match kind {
        0 => {}
        1 => {
            x[g.next() as usize % K] = f32::NAN;
            x[g.next() as usize % K] = f32::INFINITY;
            x[g.next() as usize % K] = f32::NEG_INFINITY;
        }
        _ => x = vec![0.0; K],
    }
    x
}

/// A random-bytes MQ6G256V2 head of `rows` rows at K = 2560 (10 groups of 200 bytes); one group
/// header in sixteen keeps its random bytes, the rest get finite f16 scales / zero-points.
fn mq6_head(gpu: &mut Gpu, g: &mut Rng, rows: usize) -> Res<GpuTensor> {
    let mut b = vec![0u8; rows * 10 * 200];
    for group in b.chunks_mut(200) {
        for byte in group.iter_mut() {
            *byte = g.next() as u8;
        }
        if g.next() % 16 != 0 {
            for h in 0..4 {
                let half = 0x1800u16 + (g.next() & 0x07ff) as u16;
                let half = if h % 2 == 1 && g.next() & 1 == 0 {
                    half | 0x8000
                } else {
                    half
                };
                group[2 * h..2 * h + 2].copy_from_slice(&half.to_le_bytes());
            }
        }
    }
    let mut t = gpu.upload_raw(&b, &[b.len()])?;
    t.dtype = DType::MQ6G256V2;
    Ok(t)
}

fn raw_filled(gpu: &mut Gpu, bytes: usize, fill: u8) -> Res<GpuTensor> {
    let t = gpu.zeros(&[bytes], DType::Raw)?;
    gpu.hip.memset(&t.buf, fill as i32, bytes)?;
    Ok(t)
}

/// Strides of a cell: logits (f32), x (f32), out (bytes).
#[derive(Clone, Copy)]
struct Layout {
    name: &'static str,
    v_pad: usize,
    x_pad: usize,
    out_pad: usize,
}
const PACKED: Layout = Layout {
    name: "packed",
    v_pad: 0,
    x_pad: 0,
    out_pad: 0,
};
const PADDED: Layout = Layout {
    name: "padded",
    v_pad: 5,
    x_pad: 3,
    out_pad: 8,
};

struct Cell {
    n: usize,
    order: (usize, usize, usize),
    rows: usize,
    layout: Layout,
    fam0: usize,
    xkind: usize,
}

/// Host inputs of a cell: the strided logits / x buffers and their uploads.
struct Inputs {
    logits_h: Vec<f32>,
    x_h: Vec<f32>,
    v_stride: usize,
    x_stride: usize,
    out_stride: usize,
}

fn inputs(g: &mut Rng, c: &Cell) -> Inputs {
    let v_stride = c.n + c.layout.v_pad;
    let x_stride = K + c.layout.x_pad;
    let out_stride = OUT + c.layout.out_pad;
    let mut logits_h = vec![0.0f32; (c.rows - 1) * v_stride + c.n];
    let mut x_h = vec![0.0f32; (c.rows - 1) * x_stride + K];
    for r in 0..c.rows {
        let row = make_row(g, c.n, (c.fam0 + r) % FAMILIES);
        logits_h[r * v_stride..r * v_stride + c.n].copy_from_slice(&row);
        let xr = make_x(g, (c.xkind + r) % 3);
        x_h[r * x_stride..r * x_stride + K].copy_from_slice(&xr);
    }
    Inputs {
        logits_h,
        x_h,
        v_stride,
        x_stride,
        out_stride,
    }
}

/// Whether every row of the cell has at least 8 non-NaN logits. A row with fewer leaves sentinel
/// token ids (`INT_MAX - tail`) in its partial slot, which no re-score may consume (it would read
/// head row ~2^31), so such a cell compares the partial slots only (no re-score launch at all).
fn rescore_safe(c: &Cell, i: &Inputs) -> bool {
    (0..c.rows).all(|r| {
        i.logits_h[r * i.v_stride..r * i.v_stride + c.n]
            .iter()
            .filter(|v| !v.is_nan())
            .count()
            >= 8
    })
}

struct Run {
    out: Vec<u8>,
    partial: Vec<u8>,
}

fn out_len(c: &Cell, i: &Inputs) -> usize {
    (c.rows - 1) * i.out_stride + OUT + GUARD
}
fn part_len(c: &Cell) -> usize {
    (c.rows + 1) * PART
}

/// The incumbent: one `topk8_rescore_k2560` per row into its own slots.
fn run_incumbent(
    gpu: &mut Gpu,
    head: &GpuTensor,
    c: &Cell,
    i: &Inputs,
    partial_only: bool,
) -> Res<Run> {
    let logits = gpu.upload_f32(&i.logits_h, &[i.logits_h.len()])?;
    let x = gpu.upload_f32(&i.x_h, &[i.x_h.len()])?;
    let out = raw_filled(gpu, out_len(c, i), 0xCD)?;
    let partial = raw_filled(gpu, part_len(c), 0xAB)?;
    for r in 0..c.rows {
        let lr = logits.sub_offset(r * i.v_stride, c.n);
        let xr = x.sub_offset(r * i.x_stride, K);
        let pr = partial.sub_offset(r * PART, PART);
        let or = out.sub_offset(r * i.out_stride, OUT);
        if partial_only {
            // `topk8_partial_f32_blocks` over all 128 groups: byte-identical to the incumbent
            // `topk8_partial_f32` grid (tests/x1_topk8_blocks_gpu.rs).
            gpu.topk8_partial_blocks(&lr, 0, c.n, c.order, &pr, 0, 128)?;
        } else {
            gpu.topk8_rescore_k2560(&lr, c.n, c.order, head, &xr, &pr, &or)?;
        }
    }
    gpu.hip.device_synchronize()?;
    let run = Run {
        out: gpu.download_raw_bytes(&out)?,
        partial: gpu.download_raw_bytes(&partial)?,
    };
    for t in [logits, x, out, partial] {
        gpu.free_tensor(t)?;
    }
    Ok(run)
}

/// The rows pair over the cell, with optional perturbations (controls).
fn run_rows(
    gpu: &mut Gpu,
    head: &GpuTensor,
    c: &Cell,
    i: &Inputs,
    flip_logit_of_row: Option<usize>,
    swap_x_rows: Option<(usize, usize)>,
    partial_only: bool,
) -> Res<Run> {
    let mut lh = i.logits_h.clone();
    if let Some(r) = flip_logit_of_row {
        let row = &mut lh[r * i.v_stride..r * i.v_stride + c.n];
        let top = (0..c.n)
            .filter(|&j| row[j].is_finite())
            .max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap())
            .unwrap_or(0);
        row[top] = -1.0e30;
    }
    let mut xh = i.x_h.clone();
    if let Some((a, b)) = swap_x_rows {
        for k in 0..K {
            xh.swap(a * i.x_stride + k, b * i.x_stride + k);
        }
    }
    let logits = gpu.upload_f32(&lh, &[lh.len()])?;
    let x = gpu.upload_f32(&xh, &[xh.len()])?;
    let out = raw_filled(gpu, out_len(c, i), 0xCD)?;
    let partial = raw_filled(gpu, part_len(c), 0xAB)?;
    if partial_only {
        gpu.topk8_partial_rows(&logits, i.v_stride, c.rows, c.n, c.order, &partial)?;
    } else {
        gpu.topk8_rescore_k2560_rows(
            &logits,
            i.v_stride,
            c.rows,
            c.n,
            c.order,
            head,
            &x,
            i.x_stride,
            &partial,
            &out,
            i.out_stride,
        )?;
    }
    gpu.hip.device_synchronize()?;
    let run = Run {
        out: gpu.download_raw_bytes(&out)?,
        partial: gpu.download_raw_bytes(&partial)?,
    };
    for t in [logits, x, out, partial] {
        gpu.free_tensor(t)?;
    }
    Ok(run)
}

/// Differing bytes of two runs: `(out, partial)`.
fn diff(a: &Run, b: &Run) -> (usize, usize) {
    let d = |x: &[u8], y: &[u8]| {
        assert_eq!(x.len(), y.len());
        x.iter().zip(y).filter(|(p, q)| p != q).count()
    };
    (d(&a.out, &b.out), d(&a.partial, &b.partial))
}

#[derive(Default)]
struct Totals {
    cells: usize,
    out_bytes: usize,
    partial_bytes: usize,
    mismatch_cells: usize,
    controls: usize,
    controls_detected: usize,
    controls_skipped: usize,
    partial_only_cells: usize,
}

fn oracle(json_path: &str) -> Res<()> {
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    let mut g = Rng(0x5EED_0000_7089_0001);
    let head = mq6_head(&mut gpu, &mut g, N_MAX)?;
    let mut t = Totals::default();
    let mut fails: Vec<String> = Vec::new();
    let ns = [N_MAX, 151_936, 100_276, 1000, 5];
    for &n in &ns {
        // Identity, and a (front, special, tail) reorder with token ids inside the head.
        let reorder = {
            let front = (n / 3).max(1).min(n);
            let tail = n / 5;
            let special = n.saturating_sub(tail + 1);
            (front, special, tail)
        };
        for order in [(n, n, 0), reorder] {
            for layout in [PACKED, PADDED] {
                for rows in 1..=TOPK8_ROWS_MAX {
                    // Padded strides only at a few row counts, packed everywhere.
                    if layout.v_pad != 0 && !matches!(rows, 2 | 3 | 5 | 8) {
                        continue;
                    }
                    // Every family at every row index: rotate the family offset.
                    let offsets: Vec<usize> = if n == N_MAX && rows > 1 {
                        vec![0, 3, 6]
                    } else {
                        (0..FAMILIES).collect()
                    };
                    for fam0 in offsets {
                        let c = Cell {
                            n,
                            order,
                            rows,
                            layout,
                            fam0,
                            xkind: fam0 % 3,
                        };
                        let inp = inputs(&mut g, &c);
                        let safe = rescore_safe(&c, &inp);
                        let want = run_incumbent(&mut gpu, &head, &c, &inp, !safe)?;
                        let got = run_rows(&mut gpu, &head, &c, &inp, None, None, !safe)?;
                        let (o, p) = diff(&want, &got);
                        t.cells += 1;
                        if !safe {
                            t.partial_only_cells += 1;
                        }
                        t.out_bytes += want.out.len();
                        t.partial_bytes += want.partial.len();
                        if o + p != 0 {
                            t.mismatch_cells += 1;
                            if fails.len() < 40 {
                                fails.push(format!(
                                    "n {n} order {order:?} {} rows {rows} fam0 {fam0} ({}): {o} out bytes, {p} partial bytes differ",
                                    layout.name, FAMILY_NAMES[fam0 % FAMILIES]
                                ));
                            }
                        }
                    }
                }
            }
            // Served 2-runs: rows 2 launched as 2+2+2+2 over one batch of 8 (slots in order).
            {
                let c = Cell {
                    n,
                    order,
                    rows: 8,
                    layout: PACKED,
                    fam0: 1,
                    xkind: 0,
                };
                let inp = inputs(&mut g, &c);
                let safe = rescore_safe(&c, &inp);
                let want = run_incumbent(&mut gpu, &head, &c, &inp, !safe)?;
                let logits = gpu.upload_f32(&inp.logits_h, &[inp.logits_h.len()])?;
                let x = gpu.upload_f32(&inp.x_h, &[inp.x_h.len()])?;
                let out = raw_filled(&mut gpu, out_len(&c, &inp), 0xCD)?;
                let partial = raw_filled(&mut gpu, part_len(&c), 0xAB)?;
                for pair in 0..4 {
                    let lv = logits.sub_offset(pair * 2 * inp.v_stride, inp.v_stride + c.n);
                    let xv = x.sub_offset(pair * 2 * inp.x_stride, inp.x_stride + K);
                    let pv = partial.sub_offset(pair * 2 * PART, 2 * PART);
                    let ov = out.sub_offset(pair * 2 * inp.out_stride, inp.out_stride + OUT);
                    if safe {
                        gpu.topk8_rescore_k2560_rows(
                            &lv,
                            inp.v_stride,
                            2,
                            c.n,
                            c.order,
                            &head,
                            &xv,
                            inp.x_stride,
                            &pv,
                            &ov,
                            inp.out_stride,
                        )?;
                    } else {
                        gpu.topk8_partial_rows(&lv, inp.v_stride, 2, c.n, c.order, &pv)?;
                    }
                }
                gpu.hip.device_synchronize()?;
                let got = Run {
                    out: gpu.download_raw_bytes(&out)?,
                    partial: gpu.download_raw_bytes(&partial)?,
                };
                for tt in [logits, x, out, partial] {
                    gpu.free_tensor(tt)?;
                }
                let (o, p) = diff(&want, &got);
                t.cells += 1;
                t.out_bytes += want.out.len();
                t.partial_bytes += want.partial.len();
                if o + p != 0 {
                    t.mismatch_cells += 1;
                    fails.push(format!(
                        "n {n} order {order:?} 2-runs: {o} out, {p} partial"
                    ));
                }
            }
            // Controls: random finite rows; the perturbed rows arm must differ from the incumbent.
            for (name, flip, swap) in [
                ("flipped logit of row 1", Some(1usize), None),
                ("swapped x rows 0,2", None, Some((0usize, 2usize))),
            ] {
                let c = Cell {
                    n,
                    order,
                    rows: 3,
                    layout: PACKED,
                    fam0: 0,
                    xkind: 0,
                };
                let mut inp = inputs(&mut g, &c);
                // Rows of the control are finite random (family 0 on every row).
                for r in 0..3 {
                    let row = make_row(&mut g, n, 0);
                    inp.logits_h[r * inp.v_stride..r * inp.v_stride + n].copy_from_slice(&row);
                }
                let safe = rescore_safe(&c, &inp);
                if !safe && swap.is_some() {
                    // The x swap is only visible through a re-score, which this cell cannot run.
                    t.controls_skipped += 1;
                    continue;
                }
                let want = run_incumbent(&mut gpu, &head, &c, &inp, !safe)?;
                let got = run_rows(&mut gpu, &head, &c, &inp, flip, swap, !safe)?;
                let (o, p) = diff(&want, &got);
                t.controls += 1;
                if o + p != 0 {
                    t.controls_detected += 1;
                } else {
                    fails.push(format!("control NOT detected: n {n} {name}"));
                }
            }
        }
    }
    let pass = t.mismatch_cells == 0 && t.controls == t.controls_detected && fails.is_empty();
    println!(
        "{}: {} cells ({} partial-only: a row with < 8 non-NaN logits), {} out bytes + {} partial bytes compared, {} mismatching cells; controls {}/{} detected ({} x-swap controls skipped: partial-only cell)",
        if pass { "PASS" } else { "FAIL" },
        t.cells,
        t.partial_only_cells,
        t.out_bytes,
        t.partial_bytes,
        t.mismatch_cells,
        t.controls_detected,
        t.controls,
        t.controls_skipped
    );
    for f in &fails {
        println!("  {f}");
    }
    let json = format!(
        "{{\"result\":\"{}\",\"cells\":{},\"partial_only_cells\":{},\"out_bytes\":{},\"partial_bytes\":{},\"mismatch_cells\":{},\"controls\":{},\"controls_detected\":{},\"failures\":[{}]}}\n",
        if pass { "PASS" } else { "FAIL" },
        t.cells,
        t.partial_only_cells,
        t.out_bytes,
        t.partial_bytes,
        t.mismatch_cells,
        t.controls,
        t.controls_detected,
        fails
            .iter()
            .map(|f| format!("\"{}\"", f.replace('"', "'")))
            .collect::<Vec<_>>()
            .join(",")
    );
    std::fs::write(json_path, json)?;
    if pass {
        Ok(())
    } else {
        std::process::exit(1)
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// `(gpu us, host wall us)` of one synchronized run of `f`.
fn measure(
    gpu: &mut Gpu,
    e0: &hip_bridge::Event,
    e1: &hip_bridge::Event,
    f: &mut dyn FnMut(&mut Gpu) -> Res<()>,
) -> Res<(f64, f64)> {
    gpu.hip.device_synchronize()?;
    let t0 = Instant::now();
    gpu.hip.event_record(e0, gpu.active_stream.as_ref())?;
    f(gpu)?;
    gpu.hip.event_record(e1, gpu.active_stream.as_ref())?;
    gpu.hip.event_synchronize(e1)?;
    let wall = t0.elapsed().as_secs_f64() * 1e6;
    Ok((gpu.hip.event_elapsed_ms(e0, e1)? as f64 * 1000.0, wall))
}

fn bench(json_path: &str) -> Res<()> {
    const WARM: usize = 30;
    const REPS: usize = 200;
    let mut gpu = Gpu::init()?;
    check_arch(&gpu)?;
    let e0 = gpu.hip.event_create()?;
    let e1 = gpu.hip.event_create()?;
    let mut g = Rng(0xBE4C_0000_0000_0007);
    let head = mq6_head(&mut gpu, &mut g, N_MAX)?;
    println!("n rows | per-row gpu med/min | rows gpu med/min | saved gpu | per-row wall med/min | rows wall med/min | saved wall   (us per round, 200 reps interleaved)");
    let mut json: Vec<String> = Vec::new();
    for &n in &[N_MAX, 100_276usize] {
        let order = if n == N_MAX {
            (n, n, 0)
        } else {
            (n / 3, n - n / 5 - 1, n / 5)
        };
        for &rows in &[2usize, 3, 4, 8] {
            let data: Vec<f32> = (0..rows)
                .flat_map(|r| make_row(&mut g, n, if r % 2 == 0 { 0 } else { 1 }))
                .collect();
            let logits = gpu.upload_f32(&data, &[data.len()])?;
            let xs: Vec<f32> = (0..rows).flat_map(|_| make_x(&mut g, 0)).collect();
            let x = gpu.upload_f32(&xs, &[xs.len()])?;
            let partial = raw_filled(&mut gpu, rows * PART, 0)?;
            let out = raw_filled(&mut gpu, rows * OUT, 0)?;
            let mut per_row = |gpu: &mut Gpu| -> Res<()> {
                for r in 0..rows {
                    let lr = logits.sub_offset(r * n, n);
                    let xr = x.sub_offset(r * K, K);
                    let pr = partial.sub_offset(0, PART);
                    let or = out.sub_offset(r * OUT, OUT);
                    gpu.topk8_rescore_k2560(&lr, n, order, &head, &xr, &pr, &or)?;
                }
                Ok(())
            };
            let mut batched = |gpu: &mut Gpu| -> Res<()> {
                Ok(gpu.topk8_rescore_k2560_rows(
                    &logits, n, rows, n, order, &head, &x, K, &partial, &out, OUT,
                )?)
            };
            let (mut ag, mut aw, mut bg, mut bw) = (vec![], vec![], vec![], vec![]);
            for rep in 0..WARM + REPS {
                let (a, b);
                if rep % 2 == 0 {
                    a = measure(&mut gpu, &e0, &e1, &mut per_row)?;
                    b = measure(&mut gpu, &e0, &e1, &mut batched)?;
                } else {
                    b = measure(&mut gpu, &e0, &e1, &mut batched)?;
                    a = measure(&mut gpu, &e0, &e1, &mut per_row)?;
                }
                if rep >= WARM {
                    ag.push(a.0);
                    aw.push(a.1);
                    bg.push(b.0);
                    bw.push(b.1);
                }
            }
            let (agm, agn) = (pct(&mut ag.clone(), 0.5), pct(&mut ag, 0.0));
            let (bgm, bgn) = (pct(&mut bg.clone(), 0.5), pct(&mut bg, 0.0));
            let (awm, awn) = (pct(&mut aw.clone(), 0.5), pct(&mut aw, 0.0));
            let (bwm, bwn) = (pct(&mut bw.clone(), 0.5), pct(&mut bw, 0.0));
            println!(
                "{n} {rows} | {agm:.2}/{agn:.2} | {bgm:.2}/{bgn:.2} | {:.2} | {awm:.2}/{awn:.2} | {bwm:.2}/{bwn:.2} | {:.2}",
                agm - bgm,
                awm - bwm
            );
            json.push(format!(
                "{{\"n\":{n},\"rows\":{rows},\"per_row_gpu_us\":{{\"median\":{agm:.2},\"min\":{agn:.2}}},\"rows_gpu_us\":{{\"median\":{bgm:.2},\"min\":{bgn:.2}}},\"saved_gpu_us\":{:.2},\"per_row_wall_us\":{{\"median\":{awm:.2},\"min\":{awn:.2}}},\"rows_wall_us\":{{\"median\":{bwm:.2},\"min\":{bwn:.2}}},\"saved_wall_us\":{:.2}}}",
                agm - bgm,
                awm - bwm
            ));
            for tt in [logits, x, partial, out] {
                gpu.free_tensor(tt)?;
            }
        }
    }
    std::fs::write(json_path, format!("[{}]\n", json.join(",")))?;
    Ok(())
}

fn main() -> Res<()> {
    let path = std::env::args().nth(2);
    match std::env::args().nth(1).as_deref() {
        Some("oracle") => oracle(path.as_deref().unwrap_or("topk8_rows_oracle.json")),
        Some("bench") => bench(path.as_deref().unwrap_or("topk8_rows_bench.json")),
        _ => {
            eprintln!("usage: qwen4_topk8_rows oracle [out.json] | bench [out.json]");
            std::process::exit(2);
        }
    }
}
