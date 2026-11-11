// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle (and in-process timer) of the gfx1201 persistent PeaceMaker g32 gate/up module
//! (`kernels/qwen4_moe_gu_c8_pm_gfx1201.hxaco`, entries
//! `gemm_mq4g256v2p_moe_grouped_top10_g32_{r8,r4}_nodz_pm_persist`) against the hipcc planar
//! `gemm_mq4g256v2p_moe_grouped_top10_g32_{r8,r4}_nodz`, on the REAL planar gate/up experts of a
//! `.mq4-xts` artifact, for both ranks' expert sets (`e % ranks == rank`, repacked drop-zero planar).
//!
//! The expert pointer table spans all 512 experts like the EP product's: owned experts point at their real
//! weights, the NON-owned half at ONE all-zero expert buffer (the EP zero dummy); the grouping covers every
//! expert, so zero-expert tiles are launched too.
//!
//! Per layer, source-row count (9, 12, 16, 17, 24, 32, 41, 48 by default) and routing, the launches go
//! through `gemm_mq4g256v2_moe_grouped_top10_g32_with` over the same grouped routing and F32 X, each into a
//! 0xa5-sentinel-filled grouped output:
//!   * `hipcc` — `pm = false`: the hipcc `g32_{r8,r4}_nodz` kernel;
//!   * `pm`    — `pm = true` with `n_blocks` in {1, 7, 64, 128, 512} (`--blocks`).
//! EVERY output word of every `pm` arm must equal `hipcc` (live rows bitwise; dead rows must stay sentinel
//! in both). Routings: `uniform` (top-10 uniform), `corr` (8 lanes x rows/8 positions sharing a base route
//! with probability `--corr`), `same` (every row the same 10 experts: the fewest, fullest tiles), `c8` (a
//! C8-like route: a hot set of experts with more than 16 slots each, so they span 2-3 tiles, over a wide
//! pool of cold experts, ~85 distinct at 48 rows). X is finite random (exact +-0 and large magnitudes
//! sprinkled in); the last variant of every routing carries finite X of magnitude ~1e15.
//! After the byte checks, a cold-ish timing per row count: every routing of every layer is launched once
//! per sweep in a fixed shuffled order (layers rotate, so launches do not find their experts in the 64 MB
//! cache), each launch event-timed; sweeps alternate the arm ABBA (`--ab-iters`); the median per-sweep mean
//! of the module (`--time-blocks`) over hipcc is the ratio (informational).
//! Output: one `PASS|FAIL ...` line per case, `AB ...` lines, a final `PASS n/n` or `FAIL ...` line and a
//! JSON line; exit 1 on any mismatch or launch error, 2 when the device is not gfx1201.
//!
//! Build: cargo build --release -p hipfire-runtime --features lab --example moe_gu_c8_pm_oracle
//! Run:   moe_gu_c8_pm_oracle --model PATH [--layers 0,17,47] [--rows 9,12,16,17,24,32,41,48] [--seeds 2]
//!        [--ranks 2] [--corr 0.83] [--blocks 1,7,64,128,512] [--time-blocks 128] [--ab-iters 20]
//!        [--device N]

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::mq4_planar::{
    planar_bytes, repack_tensor, shipped_bytes, Mq4PlanarKind, FLASH_NEXT_GATE_UP,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;
use std::path::Path;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const K_TOP: usize = 10;
const TILE: usize = 16;
const GATE_M: usize = FLASH_NEXT_GATE_UP.0;
const GATE_K: usize = FLASH_NEXT_GATE_UP.1;
const SENTINEL_BYTE: i32 = 0xa5;
const SENTINEL_WORD: u32 = 0xa5a5_a5a5;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn act(&mut self) -> f32 {
        // roughly normal, with exact +0.0 / -0.0 and large magnitudes sprinkled in
        match self.below(64) {
            0 => 0.0,
            1 => -0.0,
            2 => 40.0 * (self.unit() - 0.5),
            _ => (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7,
        }
    }
}

struct Cfg {
    model: String,
    layers: Vec<usize>,
    rows: Vec<usize>,
    seeds: usize,
    ranks: usize,
    corr: f64,
    blocks: Vec<u32>,
    time_blocks: u32,
    ab_iters: usize,
    device: i32,
    seed: u64,
    total_experts: usize,
}

const USAGE: &str = "usage: moe_gu_c8_pm_oracle --model PATH [--layers 0,17,47] \
[--rows 9,12,16,17,24,32,41,48] [--seeds 2] [--ranks 2] [--corr 0.83] [--blocks 1,7,64,128,512] \
[--time-blocks 128] [--ab-iters 20] [--device N] [--seed S]";

fn parse_list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|p| p.trim().parse::<usize>().map_err(err))
        .collect()
}

fn parse_args() -> Result<Cfg> {
    let mut c = Cfg {
        model: String::new(),
        layers: vec![0, 17, 47],
        rows: vec![9, 12, 16, 17, 24, 32, 41, 48],
        seeds: 2,
        ranks: 2,
        corr: 0.83,
        blocks: vec![1, 7, 64, 128, 512],
        time_blocks: 128,
        ab_iters: 20,
        device: 0,
        seed: 0xc8,
        total_experts: 512,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut it = argv.iter();
    while let Some(f) = it.next() {
        let mut v = |n: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{n} needs a value\n{USAGE}"))
        };
        match f.as_str() {
            "--model" => c.model = v("--model")?,
            "--layers" => c.layers = parse_list(&v("--layers")?)?,
            "--rows" => c.rows = parse_list(&v("--rows")?)?,
            "--seeds" => c.seeds = v("--seeds")?.parse().map_err(err)?,
            "--ranks" => c.ranks = v("--ranks")?.parse().map_err(err)?,
            "--corr" => c.corr = v("--corr")?.parse().map_err(err)?,
            "--blocks" => {
                c.blocks = parse_list(&v("--blocks")?)?
                    .into_iter()
                    .map(|b| b as u32)
                    .collect()
            }
            "--time-blocks" => c.time_blocks = v("--time-blocks")?.parse().map_err(err)?,
            "--ab-iters" => c.ab_iters = v("--ab-iters")?.parse().map_err(err)?,
            "--device" => c.device = v("--device")?.parse().map_err(err)?,
            "--seed" => c.seed = v("--seed")?.parse().map_err(err)?,
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if let Some(r) = c.rows.iter().find(|&&r| r <= 8) {
        return Err(format!(
            "--rows {r}: the g32 nodz entries serve more than 8 source rows\n{USAGE}"
        ));
    }
    if c.model.is_empty() {
        return Err(format!("--model is required\n{USAGE}"));
    }
    Ok(c)
}

struct Stack {
    /// owned planar experts, then the all-zero dummy expert
    data: GpuTensor,
    /// 512 expert pointers: owned -> real weights, non-owned -> the zero dummy
    table: GpuTensor,
}

fn load_gate_up(
    gpu: &mut Gpu,
    hfq: &HfqFile,
    layer: usize,
    rank: usize,
    cfg: &Cfg,
) -> Result<Stack> {
    let name = format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj");
    let (_, bytes) = hfq
        .tensor_data(&name)
        .ok_or_else(|| format!("tensor {name} not in artifact"))?;
    let per = shipped_bytes(Mq4PlanarKind::GateUp, GATE_M, GATE_K).map_err(err)?;
    let total = bytes.len() / per;
    if total != cfg.total_experts {
        return Err(format!("{name}: {total} experts"));
    }
    let owned: Vec<usize> = (0..total).filter(|e| e % cfg.ranks == rank).collect();
    let out_per = planar_bytes(Mq4PlanarKind::GateUp, GATE_M, GATE_K).map_err(err)?;
    // owned experts back to back, one all-zero expert after them
    let mut buf = vec![0u8; (owned.len() + 1) * out_per];
    for (i, &e) in owned.iter().enumerate() {
        let src = &bytes[e * per..(e + 1) * per];
        repack_tensor(
            Mq4PlanarKind::GateUp,
            GATE_M,
            GATE_K,
            src,
            &mut buf[i * out_per..(i + 1) * out_per],
        )
        .map_err(|x| format!("{name}[{e}]: {x}"))?;
    }
    let mut data = gpu
        .upload_raw(&buf, &[owned.len() + 1, GATE_M, GATE_K])
        .map_err(err)?;
    data.dtype = DType::MQ4G256V2P;
    let base = data.buf.as_ptr() as usize as u64;
    let zero = base + (owned.len() * out_per) as u64;
    let tb: Vec<u8> = (0..total)
        .flat_map(|e| {
            let p = if e % cfg.ranks == rank {
                base + ((e / cfg.ranks) * out_per) as u64
            } else {
                zero
            };
            p.to_le_bytes()
        })
        .collect();
    let table = gpu.upload_raw(&tb, &[total]).map_err(err)?;
    Ok(Stack { data, table })
}

#[derive(Clone, Copy)]
enum Mode {
    Uniform,
    Corr,
    Same,
    C8,
}
impl Mode {
    fn name(&self) -> &'static str {
        match self {
            Mode::Uniform => "uniform",
            Mode::Corr => "corr",
            Mode::Same => "same",
            Mode::C8 => "c8",
        }
    }
}

fn distinct_from(rng: &mut Rng, total: usize, have: &[usize]) -> usize {
    loop {
        let e = rng.below(total);
        if !have.contains(&e) {
            return e;
        }
    }
}

fn pick_set(rng: &mut Rng, total: usize) -> Vec<usize> {
    let mut p: Vec<usize> = Vec::new();
    while p.len() < K_TOP {
        let e = distinct_from(rng, total, &p);
        p.push(e);
    }
    p
}

/// Global top-10 ids `[rows * 10]` for one case.
fn picks(mode: Mode, rows: usize, rng: &mut Rng, cfg: &Cfg) -> Vec<usize> {
    let total = cfg.total_experts;
    match mode {
        Mode::Uniform => (0..rows).flat_map(|_| pick_set(rng, total)).collect(),
        Mode::Corr => {
            let lanes = 8;
            let bases: Vec<Vec<usize>> = (0..rows.div_ceil(lanes))
                .map(|_| pick_set(rng, total))
                .collect();
            (0..rows)
                .flat_map(|r| {
                    let mut p: Vec<usize> = Vec::new();
                    for j in 0..K_TOP {
                        let mut e = bases[r / lanes][j];
                        if rng.unit() as f64 >= cfg.corr || p.contains(&e) {
                            e = distinct_from(rng, total, &p);
                        }
                        p.push(e);
                    }
                    p
                })
                .collect()
        }
        Mode::Same => pick_set(rng, total).repeat(rows),
        Mode::C8 => {
            // 8 hot experts, 4 of them in every row (rows * 4 / 8 slots each: more than 16 slots from 33
            // rows, 2-3 tiles at 48), the other 6 picks from a cold pool of 77: ~85 distinct at 48 rows.
            let mut hot: Vec<usize> = Vec::new();
            while hot.len() < 8 {
                let e = distinct_from(rng, total, &hot);
                hot.push(e);
            }
            let mut cold: Vec<usize> = Vec::new();
            while cold.len() < 77 {
                let e = distinct_from(rng, total, &[hot.as_slice(), cold.as_slice()].concat());
                cold.push(e);
            }
            (0..rows)
                .flat_map(|r| {
                    let mut p: Vec<usize> = Vec::new();
                    // rotating window over the hot set keeps every hot expert at ~rows/2 slots
                    for j in 0..4 {
                        p.push(hot[(r * 4 + j) % 8]);
                    }
                    while p.len() < K_TOP {
                        let e = cold[rng.below(cold.len())];
                        if !p.contains(&e) {
                            p.push(e);
                        }
                    }
                    p
                })
                .collect()
        }
    }
}

struct Step {
    tiles: Vec<i32>,
    sorted: Vec<i32>,
    distinct: usize,
    max_slots: usize,
    rows: usize,
}

/// Group every expert's slots (owned or not: non-owned tiles read the zero dummy) into 16-slot tiles; the
/// tile id is the GLOBAL expert index into the 512-entry table.
fn group(p: &[usize], cfg: &Cfg) -> Step {
    let mut per: Vec<Vec<i32>> = vec![Vec::new(); cfg.total_experts];
    for (flat, &e) in p.iter().enumerate() {
        per[e].push(flat as i32);
    }
    let (mut tiles, mut sorted, mut distinct, mut max_slots) = (Vec::new(), Vec::new(), 0, 0);
    for (e, slots) in per.iter().enumerate() {
        if !slots.is_empty() {
            distinct += 1;
        }
        max_slots = max_slots.max(slots.len());
        for chunk in slots.chunks(TILE) {
            tiles.push(e as i32);
            sorted.extend_from_slice(chunk);
            sorted.extend(std::iter::repeat(-1).take(TILE - chunk.len()));
        }
    }
    Step {
        tiles,
        sorted,
        distinct,
        max_slots,
        rows: p.len() / K_TOP,
    }
}

fn i32b(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Dev {
    tiles: GpuTensor,
    sorted: GpuTensor,
    x: GpuTensor,
    y: GpuTensor,
    grouped_rows: usize,
}

/// `None` = the hipcc `g32_nodz` entry; `Some(n_blocks)` = the PeaceMaker persistent module.
fn launch(gpu: &mut Gpu, st: &Stack, d: &Dev, rows: usize, arm: Option<u32>) -> Result<()> {
    gpu.gemm_mq4g256v2_moe_grouped_top10_g32_with(
        &st.table,
        &d.tiles,
        &d.sorted,
        &d.x,
        &d.y,
        GATE_M,
        GATE_K,
        K_TOP,
        d.grouped_rows,
        rows,
        arm.is_some(),
        arm.unwrap_or(0),
    )
    .map_err(err)
}

fn readback(gpu: &Gpu, t: &GpuTensor, words: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; words * 4];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).map_err(err)?;
    Ok(b)
}

/// `(mismatching words, first mismatch)` of `got` against `want`.
fn compare(want: &[u8], got: &[u8]) -> (usize, Option<(usize, u32, u32)>) {
    let mut mism = 0;
    let mut first = None;
    for (i, (wa, wb)) in want.chunks_exact(4).zip(got.chunks_exact(4)).enumerate() {
        let a = u32::from_le_bytes(wa.try_into().unwrap());
        let b = u32::from_le_bytes(wb.try_into().unwrap());
        if a != b {
            mism += 1;
            first.get_or_insert((i, a, b));
        }
    }
    (mism, first)
}

/// Words of the dead rows (`sorted == -1`) that are not the sentinel.
fn dead_dirty(out: &[u8], sorted: &[i32]) -> usize {
    let mut n = 0;
    for (row, &s) in sorted.iter().enumerate() {
        if s >= 0 {
            continue;
        }
        n += out[row * GATE_M * 4..(row + 1) * GATE_M * 4]
            .chunks_exact(4)
            .filter(|w| u32::from_le_bytes((*w).try_into().unwrap()) != SENTINEL_WORD)
            .count();
    }
    n
}

fn run() -> Result<i32> {
    let cfg = parse_args()?;
    let mut gpu = Gpu::init_with_device(cfg.device).map_err(err)?;
    if gpu.arch != "gfx1201" {
        println!(
            "SKIP device arch is {}, the module is gfx1201 only",
            gpu.arch
        );
        return Ok(2);
    }
    gpu.set_mq4_experts_planar(true);
    if !gpu.moe_gu_c8_pm_covers(GATE_M, GATE_K, K_TOP, 48 * K_TOP) {
        return Err("the module does not cover the Qwen4 gate/up shape on this GPU".into());
    }
    let hfq = HfqFile::open(Path::new(&cfg.model)).map_err(|e| format!("open: {e}"))?;
    let modes = [Mode::Uniform, Mode::Corr, Mode::Same, Mode::C8];
    let ev0 = gpu.hip.event_create().map_err(err)?;
    let ev1 = gpu.hip.event_create().map_err(err)?;
    let (mut pass, mut fail) = (0usize, 0usize);
    let mut ab: Vec<serde_json::Value> = Vec::new();

    for rank in 0..cfg.ranks {
        let mut stacks: Vec<(usize, Stack)> = Vec::new();
        for &layer in &cfg.layers {
            stacks.push((layer, load_gate_up(&mut gpu, &hfq, layer, rank, &cfg)?));
        }
        for &rows in &cfg.rows {
            let mut ab_steps: Vec<(usize, Dev, usize)> = Vec::new();
            for (si, (layer, st)) in stacks.iter().enumerate() {
                let layer = *layer;
                for mode in modes {
                    // `seeds` normal X variants, then one large-magnitude finite X
                    for v in 0..=cfg.seeds {
                        let large = v == cfg.seeds;
                        let mut rng = Rng(cfg.seed
                            ^ (layer as u64) << 40
                            ^ (rows as u64) << 20
                            ^ (v as u64) << 8
                            ^ rank as u64);
                        let p = picks(mode, rows, &mut rng, &cfg);
                        let step = group(&p, &cfg);
                        let grouped_rows = step.sorted.len();
                        let nrows = step.rows;
                        let xv: Vec<f32> = (0..nrows * GATE_K)
                            .map(|_| {
                                let a = rng.act();
                                if large {
                                    a * 1.0e15
                                } else {
                                    a
                                }
                            })
                            .collect();
                        let tiles = gpu
                            .upload_raw(&i32b(&step.tiles), &[step.tiles.len()])
                            .map_err(err)?;
                        let sorted = gpu
                            .upload_raw(&i32b(&step.sorted), &[step.sorted.len()])
                            .map_err(err)?;
                        let x = gpu.upload_f32(&xv, &[xv.len()]).map_err(err)?;
                        let y = gpu
                            .alloc_tensor(&[grouped_rows * GATE_M], DType::F32)
                            .map_err(err)?;
                        let d = Dev {
                            tiles,
                            sorted,
                            x,
                            y,
                            grouped_rows,
                        };
                        let words = grouped_rows * GATE_M;
                        gpu.hip
                            .memset(&d.y.buf, SENTINEL_BYTE, words * 4)
                            .map_err(err)?;
                        launch(&mut gpu, st, &d, nrows, None)?;
                        gpu.hip.device_synchronize().map_err(err)?;
                        let want = readback(&gpu, &d.y, words)?;
                        let nonzero = want
                            .chunks_exact(4)
                            .filter(|w| {
                                let v = u32::from_le_bytes((*w).try_into().unwrap());
                                v != 0 && v != SENTINEL_WORD
                            })
                            .count();
                        let dirty_hip = dead_dirty(&want, &step.sorted);
                        let (mut mism_total, mut dirty_pm) = (0usize, 0usize);
                        let mut detail = String::new();
                        for &nb in &cfg.blocks {
                            gpu.hip
                                .memset(&d.y.buf, SENTINEL_BYTE, words * 4)
                                .map_err(err)?;
                            launch(&mut gpu, st, &d, nrows, Some(nb))?;
                            gpu.hip.device_synchronize().map_err(err)?;
                            let got = readback(&gpu, &d.y, words)?;
                            let (m, first) = compare(&want, &got);
                            mism_total += m;
                            dirty_pm += dead_dirty(&got, &step.sorted);
                            if let Some((i, a, b)) = first {
                                detail.push_str(&format!(
                                    " blocks{nb}_first=word{i} hipcc=0x{a:08x} pm=0x{b:08x}"
                                ));
                            }
                        }
                        let ok = mism_total == 0 && dirty_hip == 0 && dirty_pm == 0 && nonzero > 0;
                        if ok {
                            pass += 1;
                        } else {
                            fail += 1;
                        }
                        println!(
                            "{} rank {rank} layer {layer} rows {nrows} mode {} x {} distinct {} max_slots {} grouped_rows {grouped_rows} words {words} nonzero {nonzero} blocks {:?} pm_mism {mism_total} dead_dirty hipcc {dirty_hip} pm {dirty_pm}{detail}",
                            if ok { "PASS" } else { "FAIL" },
                            mode.name(),
                            if large { "large" } else { "normal" },
                            step.distinct,
                            step.max_slots,
                            cfg.blocks,
                        );
                        if v == 0 {
                            ab_steps.push((si, d, nrows));
                        } else {
                            let _ = gpu.free_tensor(d.tiles);
                            let _ = gpu.free_tensor(d.sorted);
                            let _ = gpu.free_tensor(d.x);
                            let _ = gpu.free_tensor(d.y);
                        }
                    }
                }
            }
            // Cold-ish timing: one arm launches every routing of every layer in a fixed shuffled order (so
            // a launch does not find its experts in the 64 MB cache from the previous launch), timed per
            // launch; sweeps alternate ABBA; the per-sweep means are compared.
            if cfg.ab_iters > 0 && !ab_steps.is_empty() {
                let mut order: Vec<usize> = (0..ab_steps.len()).collect();
                let mut rng = Rng(cfg.seed ^ 0xab ^ (rows as u64) << 20);
                for i in (1..order.len()).rev() {
                    order.swap(i, rng.below(i + 1));
                }
                let (mut t_hip, mut t_pm) = (Vec::new(), Vec::new());
                for sweep in 0..cfg.ab_iters {
                    let pm = matches!(sweep % 4, 1 | 2);
                    let arm = if pm { Some(cfg.time_blocks) } else { None };
                    let mut sum = 0.0f64;
                    for &i in &order {
                        let (si, d, nrows) = &ab_steps[i];
                        gpu.hip.event_record(&ev0, None).map_err(err)?;
                        launch(&mut gpu, &stacks[*si].1, d, *nrows, arm)?;
                        gpu.hip.event_record(&ev1, None).map_err(err)?;
                        gpu.hip.event_synchronize(&ev1).map_err(err)?;
                        sum += gpu.hip.event_elapsed_ms(&ev0, &ev1).map_err(err)? as f64 * 1e3;
                    }
                    let per = sum / order.len() as f64;
                    if pm {
                        t_pm.push(per)
                    } else {
                        t_hip.push(per)
                    }
                }
                let med = |v: &mut Vec<f64>| {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    v[v.len() / 2]
                };
                let (h, p) = (med(&mut t_hip), med(&mut t_pm));
                println!(
                    "AB rank {rank} rows {rows}: {} launches/sweep x {} sweeps, mean us/launch hipcc {h:.2} pm(blocks {}) {p:.2}  pm/hipcc {:.4}",
                    order.len(),
                    cfg.ab_iters,
                    cfg.time_blocks,
                    p / h,
                );
                ab.push(
                    json!({"rank": rank, "rows": rows, "launches_per_sweep": order.len(),
                    "sweeps": cfg.ab_iters, "blocks": cfg.time_blocks,
                    "hipcc_us": h, "pm_us": p, "ratio": p / h}),
                );
            }
            for (_, d, _) in ab_steps {
                let _ = gpu.free_tensor(d.tiles);
                let _ = gpu.free_tensor(d.sorted);
                let _ = gpu.free_tensor(d.x);
                let _ = gpu.free_tensor(d.y);
            }
        }
        for (_, st) in stacks {
            let _ = gpu.free_tensor(st.data);
            let _ = gpu.free_tensor(st.table);
        }
    }
    let total = pass + fail;
    let verdict = if fail == 0 && pass > 0 {
        "PASS"
    } else {
        "FAIL"
    };
    if verdict == "PASS" {
        println!("PASS {pass}/{total}");
    } else {
        println!("FAIL {pass}/{total}");
    }
    println!(
        "{}",
        json!({"schema": "moe-gu-c8-pm-oracle-v1", "model": cfg.model, "layers": cfg.layers, "rows": cfg.rows,
            "blocks": cfg.blocks,
            "pm_image_sha256": rdna_compute::kernels::QWEN4_MOE_GU_C8_PM_GFX1201_SHA256,
            "cases": total, "pass": pass, "fail": fail, "ab": ab, "verdict": verdict})
    );
    Ok(if verdict == "PASS" { 0 } else { 1 })
}

fn main() {
    match run() {
        Ok(c) => std::process::exit(c),
        Err(e) => {
            eprintln!("moe_gu_c8_pm_oracle: {e}");
            println!(
                "{}",
                json!({"schema": "moe-gu-c8-pm-oracle-v1", "verdict": "FAIL", "error": e})
            );
            std::process::exit(1);
        }
    }
}
