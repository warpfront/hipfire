// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle (and in-process A/B timer) of the gfx1201 PeaceMaker few-row MoE gate/up module
//! (`kernels/qwen4_moe_gu_few_pm_gfx1201.hxaco`, WideNet F07) against the hipcc planar
//! `gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4`, on the REAL planar gate/up experts of a
//! `.mq4-xts` artifact (the rank's owned experts `e % ranks == rank`, repacked drop-zero planar).
//!
//! Per layer, source-row count (1..=8 by default: every few-row count, including the C1 MTP verify
//! counts 2..=6 that the WideNet captures lack) and routing mode, three launches over the same
//! grouped routing and F32 X, each into a 0xa5-sentinel-filled grouped output:
//!   * `hipcc`   — `gemm_mq4g256v2_moe_grouped_top10_few_with(.., pm = false)`;
//!   * `pm`      — the same with `pm = true` (the module, at every row count);
//!   * `product` — `gemm_mq4g256v2_moe_grouped_top10` (the product route: the module at the row counts
//!                 of `MOE_GU_FEW_PM_ROWS` unless `HIPFIRE_QWEN4_MOE_GU_FEW_PM=0`, else hipcc).
//! EVERY output word (NaN payloads, zeros, dead-slot words and unwritten sentinel words) of `pm` and
//! `product` must equal `hipcc`. Routing modes: `uniform` (top-10 uniform), `corr` (8 lanes x rows/8
//! positions sharing a base route with probability `--corr`), `same` (every row the same 10 experts:
//! the fewest, fullest tiles), `file` (real verify routes from `--routefile`, records of <= rows rows).
//! Activations are random with exact +-0 and large magnitudes; `--nan` adds NaN/Inf payload rows.
//! After the byte checks, a cold-weight A/B per row count: every routing of every layer is launched once
//! per sweep in a fixed shuffled order (consecutive launches touch different experts, so as in serving the
//! weights are not in the 64 MB cache), each launch event-timed; sweeps alternate the arm ABBA
//! (`--ab-iters` sweeps); the median per-sweep mean of the module over the hipcc kernel is the ratio.
//! Output: one `PASS|FAIL ...` line per case, `AB ...` lines, a final JSON line; exit 1 on any
//! mismatch or launch error, 2 when the device is not gfx1201.
//!
//! Build: cargo build --release -p hipfire-runtime --features lab --example moe_gu_few_pm_oracle
//! Run:   moe_gu_few_pm_oracle --model PATH [--layers 0,17,47] [--rows 1,2,3,4,5,6,7,8] [--seeds 3]
//!        [--rank 0] [--ranks 2] [--corr 0.83] [--routefile F] [--ab-iters 20] [--device N] [--nan]

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
    rank: usize,
    ranks: usize,
    corr: f64,
    routefile: Option<String>,
    ab_iters: usize,
    device: i32,
    seed: u64,
    total_experts: usize,
    nan: bool,
}

const USAGE: &str = "usage: moe_gu_few_pm_oracle --model PATH [--layers 0,17,47] [--rows 1,2,3,4,5,6,7,8] \
[--seeds 3] [--rank 0] [--ranks 2] [--corr 0.83] [--routefile F] [--ab-iters 20] [--device N] [--seed S] [--nan]";

fn parse_list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|p| p.trim().parse::<usize>().map_err(err))
        .collect()
}

fn parse_args() -> Result<Cfg> {
    let mut c = Cfg {
        model: String::new(),
        layers: vec![0, 17, 47],
        rows: vec![1, 2, 3, 4, 5, 6, 7, 8],
        seeds: 3,
        rank: 0,
        ranks: 2,
        corr: 0.83,
        routefile: None,
        ab_iters: 20,
        device: 0,
        seed: 0x32,
        total_experts: 512,
        nan: false,
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
            "--rank" => c.rank = v("--rank")?.parse().map_err(err)?,
            "--ranks" => c.ranks = v("--ranks")?.parse().map_err(err)?,
            "--corr" => c.corr = v("--corr")?.parse().map_err(err)?,
            "--routefile" => c.routefile = Some(v("--routefile")?),
            "--ab-iters" => c.ab_iters = v("--ab-iters")?.parse().map_err(err)?,
            "--device" => c.device = v("--device")?.parse().map_err(err)?,
            "--seed" => c.seed = v("--seed")?.parse().map_err(err)?,
            "--nan" => c.nan = true,
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if c.model.is_empty() {
        return Err(format!("--model is required\n{USAGE}"));
    }
    Ok(c)
}

struct Stack {
    data: GpuTensor,
    table: GpuTensor,
    n: usize,
}

fn load_gate_up(gpu: &mut Gpu, hfq: &HfqFile, layer: usize, cfg: &Cfg) -> Result<Stack> {
    let name = format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj");
    let (_, bytes) = hfq
        .tensor_data(&name)
        .ok_or_else(|| format!("tensor {name} not in artifact"))?;
    let per = shipped_bytes(Mq4PlanarKind::GateUp, GATE_M, GATE_K).map_err(err)?;
    let total = bytes.len() / per;
    if total != cfg.total_experts {
        return Err(format!("{name}: {total} experts"));
    }
    let owned: Vec<usize> = (0..total).filter(|e| e % cfg.ranks == cfg.rank).collect();
    let out_per = planar_bytes(Mq4PlanarKind::GateUp, GATE_M, GATE_K).map_err(err)?;
    let mut buf = vec![0u8; owned.len() * out_per];
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
        .upload_raw(&buf, &[owned.len(), GATE_M, GATE_K])
        .map_err(err)?;
    data.dtype = DType::MQ4G256V2P;
    let base = data.buf.as_ptr() as usize as u64;
    let tb: Vec<u8> = (0..owned.len())
        .flat_map(|i| (base + (i * out_per) as u64).to_le_bytes())
        .collect();
    let table = gpu.upload_raw(&tb, &[owned.len()]).map_err(err)?;
    Ok(Stack {
        data,
        table,
        n: owned.len(),
    })
}

enum Mode {
    Uniform,
    Corr,
    Same,
    File,
}
impl Mode {
    fn name(&self) -> &'static str {
        match self {
            Mode::Uniform => "uniform",
            Mode::Corr => "corr",
            Mode::Same => "same",
            Mode::File => "file",
        }
    }
}

/// Global top-10 ids `[rows * 10]` for one case.
fn picks(
    mode: &Mode,
    rows: usize,
    rng: &mut Rng,
    cfg: &Cfg,
    file: &[(usize, Vec<i32>)],
) -> Vec<usize> {
    let total = cfg.total_experts;
    let distinct = |rng: &mut Rng, have: &[usize]| loop {
        let e = rng.below(total);
        if !have.contains(&e) {
            return e;
        }
    };
    match mode {
        Mode::Uniform => (0..rows)
            .flat_map(|_| {
                let mut p: Vec<usize> = Vec::new();
                while p.len() < K_TOP {
                    let e = distinct(rng, &p);
                    p.push(e);
                }
                p
            })
            .collect(),
        Mode::Corr => {
            let lanes = 8;
            let bases: Vec<Vec<usize>> = (0..rows.div_ceil(lanes))
                .map(|_| {
                    let mut b: Vec<usize> = Vec::new();
                    while b.len() < K_TOP {
                        let e = distinct(rng, &b);
                        b.push(e);
                    }
                    b
                })
                .collect();
            (0..rows)
                .flat_map(|r| {
                    let mut p: Vec<usize> = Vec::new();
                    for j in 0..K_TOP {
                        let mut e = bases[r / lanes][j];
                        if rng.unit() as f64 >= cfg.corr || p.contains(&e) {
                            e = distinct(rng, &p);
                        }
                        p.push(e);
                    }
                    p
                })
                .collect()
        }
        Mode::Same => {
            let mut p: Vec<usize> = Vec::new();
            while p.len() < K_TOP {
                let e = distinct(rng, &p);
                p.push(e);
            }
            p.repeat(rows)
        }
        Mode::File => {
            let cand: Vec<&(usize, Vec<i32>)> = file.iter().filter(|(r, _)| *r == rows).collect();
            if cand.is_empty() {
                return Vec::new();
            }
            let rec = cand[rng.below(cand.len())];
            rec.1.iter().map(|&e| e as usize).collect()
        }
    }
}

struct Step {
    tiles: Vec<i32>,
    sorted: Vec<i32>,
    distinct: usize,
    rows: usize,
}

fn group(p: &[usize], cfg: &Cfg, n_local: usize) -> Step {
    let mut per: Vec<Vec<i32>> = vec![Vec::new(); n_local];
    for (flat, &e) in p.iter().enumerate() {
        if e % cfg.ranks == cfg.rank {
            per[e / cfg.ranks].push(flat as i32);
        }
    }
    let (mut tiles, mut sorted, mut distinct) = (Vec::new(), Vec::new(), 0);
    for (l, slots) in per.iter().enumerate() {
        if !slots.is_empty() {
            distinct += 1;
        }
        for chunk in slots.chunks(TILE) {
            tiles.push(l as i32);
            sorted.extend_from_slice(chunk);
            sorted.extend(std::iter::repeat(-1).take(TILE - chunk.len()));
        }
    }
    Step {
        tiles,
        sorted,
        distinct,
        rows: p.len() / K_TOP,
    }
}

fn i32b(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_routefile(path: &str) -> Result<Vec<(usize, Vec<i32>)>> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let n = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let mut pos = 4;
    let mut out = Vec::new();
    for _ in 0..n {
        if pos + 4 > b.len() {
            break;
        }
        let layer = u16::from_le_bytes(b[pos..pos + 2].try_into().unwrap()) as usize;
        let rows = u16::from_le_bytes(b[pos + 2..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let cnt = rows * K_TOP;
        if pos + cnt * 4 > b.len() {
            break;
        }
        let ids: Vec<i32> = (0..cnt)
            .map(|i| i32::from_le_bytes(b[pos + i * 4..pos + i * 4 + 4].try_into().unwrap()))
            .collect();
        pos += cnt * 4;
        out.push((rows, ids));
        let _ = layer;
    }
    Ok(out)
}

struct Dev {
    tiles: GpuTensor,
    sorted: GpuTensor,
    x: GpuTensor,
    y: GpuTensor,
    grouped_rows: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Hipcc,
    Pm,
    Product,
}

fn launch(gpu: &mut Gpu, st: &Stack, d: &Dev, rows: usize, arm: Arm) -> Result<()> {
    match arm {
        Arm::Product => gpu.gemm_mq4g256v2_moe_grouped_top10(
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
        ),
        _ => gpu.gemm_mq4g256v2_moe_grouped_top10_few_with(
            &st.table,
            &d.tiles,
            &d.sorted,
            &d.x,
            &d.y,
            GATE_M,
            GATE_K,
            K_TOP,
            d.grouped_rows,
            arm == Arm::Pm,
        ),
    }
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
    if !gpu.moe_gu_few_pm_covers(GATE_M, GATE_K, K_TOP) {
        return Err("the module does not cover the Qwen4 gate/up shape on this GPU".into());
    }
    let hfq = HfqFile::open(Path::new(&cfg.model)).map_err(|e| format!("open: {e}"))?;
    let file = match &cfg.routefile {
        Some(p) => read_routefile(p)?,
        None => Vec::new(),
    };
    let mut modes = vec![Mode::Uniform, Mode::Corr, Mode::Same];
    if !file.is_empty() {
        modes.push(Mode::File);
    }
    let ev0 = gpu.hip.event_create().map_err(err)?;
    let ev1 = gpu.hip.event_create().map_err(err)?;
    let (mut pass, mut fail) = (0usize, 0usize);
    let mut ab: Vec<serde_json::Value> = Vec::new();

    let mut stacks: Vec<(usize, Stack)> = Vec::new();
    for &layer in &cfg.layers {
        stacks.push((layer, load_gate_up(&mut gpu, &hfq, layer, &cfg)?));
    }
    for &rows in &cfg.rows {
        let routed = rdna_compute::gemm::MOE_GU_FEW_PM_ROWS.contains(&rows);
        let mut ab_steps: Vec<(usize, Dev, usize)> = Vec::new();
        for (si, (layer, st)) in stacks.iter().enumerate() {
            let layer = *layer;
            for mode in &modes {
                for s in 0..cfg.seeds {
                    let mut rng =
                        Rng(cfg.seed ^ (layer as u64) << 40 ^ (rows as u64) << 20 ^ s as u64);
                    let p = picks(mode, rows, &mut rng, &cfg, &file);
                    if p.is_empty() {
                        continue;
                    }
                    let step = group(&p, &cfg, st.n);
                    let grouped_rows = step.sorted.len();
                    let nrows = step.rows;
                    if grouped_rows == 0 {
                        continue;
                    }
                    let mut xv: Vec<f32> = (0..nrows * GATE_K).map(|_| rng.act()).collect();
                    if cfg.nan && s == 0 {
                        // one row of NaN payloads / infinities: payload propagation is part of the bytes
                        let r = rng.below(nrows);
                        for (j, v) in xv[r * GATE_K..(r + 1) * GATE_K].iter_mut().enumerate() {
                            *v = match j % 5 {
                                0 => f32::from_bits(0x7fc0_0000 | (j as u32 & 0x3f_ffff)),
                                1 => f32::from_bits(0xffc0_1234),
                                2 => f32::INFINITY,
                                3 => f32::NEG_INFINITY,
                                _ => *v,
                            };
                        }
                    }
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
                    let mut outs: Vec<Vec<u8>> = Vec::new();
                    for arm in [Arm::Hipcc, Arm::Pm, Arm::Product] {
                        gpu.hip
                            .memset(&d.y.buf, SENTINEL_BYTE, words * 4)
                            .map_err(err)?;
                        launch(&mut gpu, st, &d, nrows, arm)?;
                        gpu.hip.device_synchronize().map_err(err)?;
                        outs.push(readback(&gpu, &d.y, words)?);
                    }
                    let nonzero = outs[0]
                        .chunks_exact(4)
                        .filter(|w| {
                            let v = u32::from_le_bytes((*w).try_into().unwrap());
                            v != 0 && v != SENTINEL_WORD
                        })
                        .count();
                    let (m_pm, f_pm) = compare(&outs[0], &outs[1]);
                    let (m_prod, f_prod) = compare(&outs[0], &outs[2]);
                    let ok = m_pm == 0 && m_prod == 0 && nonzero > 0;
                    if ok {
                        pass += 1;
                    } else {
                        fail += 1;
                    }
                    println!(
                        "{} layer {layer} rows {nrows} mode {} seed {s} distinct {} grouped_rows {grouped_rows} words {words} nonzero {nonzero} pm_mism {m_pm} product_mism {m_prod} product_routed {routed}{}{}",
                        if ok { "PASS" } else { "FAIL" },
                        mode.name(),
                        step.distinct,
                        f_pm.map(|(i, a, b)| format!(" pm_first=word{i} hipcc=0x{a:08x} pm=0x{b:08x}")).unwrap_or_default(),
                        f_prod.map(|(i, a, b)| format!(" product_first=word{i} hipcc=0x{a:08x} product=0x{b:08x}")).unwrap_or_default(),
                    );
                    ab_steps.push((si, d, nrows));
                }
            }
        }
        // Cold-weight A/B: one arm launches every routing of every layer in a fixed shuffled order (so a
        // launch does not find its experts in the 64 MB cache from the previous launch, as in serving),
        // timed per launch; sweeps alternate ABBA; the per-sweep sums are compared.
        if cfg.ab_iters > 0 && !ab_steps.is_empty() {
            let mut order: Vec<usize> = (0..ab_steps.len()).collect();
            let mut rng = Rng(cfg.seed ^ 0xab ^ (rows as u64) << 20);
            for i in (1..order.len()).rev() {
                order.swap(i, rng.below(i + 1));
            }
            let (mut t_hip, mut t_pm) = (Vec::new(), Vec::new());
            for sweep in 0..cfg.ab_iters {
                let pm = matches!(sweep % 4, 1 | 2);
                let arm = if pm { Arm::Pm } else { Arm::Hipcc };
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
            let (lo_hi, lo_pm) = (
                t_hip.iter().cloned().fold(f64::MAX, f64::min),
                t_pm.iter().cloned().fold(f64::MAX, f64::min),
            );
            let (h, p) = (med(&mut t_hip), med(&mut t_pm));
            println!(
                "AB rows {rows}: {} launches/sweep x {} sweeps, mean us/launch hipcc {h:.2} pm {p:.2}  pm/hipcc median {:.4} min {:.4}",
                order.len(),
                cfg.ab_iters,
                p / h,
                lo_pm / lo_hi
            );
            ab.push(
                json!({"rows": rows, "launches_per_sweep": order.len(), "sweeps": cfg.ab_iters,
                "hipcc_us": h, "pm_us": p, "ratio": p / h, "ratio_min": lo_pm / lo_hi}),
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
    let verdict = if fail == 0 && pass > 0 {
        "PASS"
    } else {
        "FAIL"
    };
    println!(
        "{}",
        json!({"schema": "moe-gu-few-pm-oracle-v1", "model": cfg.model, "layers": cfg.layers, "rows": cfg.rows,
            "pm_image_sha256": rdna_compute::kernels::QWEN4_MOE_GU_FEW_PM_GFX1201_SHA256,
            "routed_rows": rdna_compute::gemm::MOE_GU_FEW_PM_ROWS,
            "cases": pass + fail, "pass": pass, "fail": fail, "ab": ab, "verdict": verdict})
    );
    Ok(if verdict == "PASS" { 0 } else { 1 })
}

fn main() {
    match run() {
        Ok(c) => std::process::exit(c),
        Err(e) => {
            eprintln!("moe_gu_few_pm_oracle: {e}");
            println!(
                "{}",
                json!({"schema": "moe-gu-few-pm-oracle-v1", "verdict": "FAIL", "error": e})
            );
            std::process::exit(1);
        }
    }
}
