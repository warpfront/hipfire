// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S3 wide-pass routed-expert byte oracle (and in-process A/B timer) on the REAL expert tensors of
//! a `.mq4-xts` artifact: the production grouped gate/up (`gemm_mq4g256v2_moe_grouped_top10`, the
//! o4_r8_x4 / o4_r2_x4 kernels) versus the `g32` family selected by `HIPFIRE_QWEN4_MOE32_GU`
//! (`Gpu::set_moe32_gu`), bit for bit.
//!
//! Per layer (the rank's owned experts `e % ranks == rank` uploaded in the shipped layout) and per
//! case (rows x routing mode x seed): the same grouped routing and F32 rotated X run twice, flag off
//! and on, outputs pre-filled with the 0xa5 sentinel; EVERY output word (NaN payloads, dead-slot
//! zeros and unwritten sentinel words included) must be identical. Routing modes:
//!   * `uniform`  — top-10 uniform over the artifact's experts;
//!   * `corr`     — W1-style correlated routing (8 lanes x rows/8 positions, each lane keeps a
//!                  position's base expert with probability `--corr`, default 0.83 = the 87-union);
//!   * `file`     — real routes from `--routefile` (the compact K0 verify-route dump written by
//!                  `moekern-out/hb/routes_diverse32.bin`: u32 n, then per record u16 layer, u16 rows,
//!                  rows*10 i32 global expert ids).
//! After the byte checks an interleaved A/B timing of the two kernels (events around one launch,
//! alternating arms, `--ab-iters` pairs per case) prints median us and GB/s of distinct bytes.
//!
//! Output: one `PASS|FAIL layer L rows R mode M ...` line per case, A/B lines, a final JSON line;
//! exit 1 on any mismatch/launch error. A device that is not gfx1201 prints SKIP and exits 2.
//!
//! Build: cargo build --release -p hipfire-runtime --features lab --example moe_wide_oracle
//! Run:   moe_wide_oracle --model PATH [--layers 0,17,47] [--rows 9,16,24,32,48] [--seeds 3]
//!        [--rank 0] [--ranks 2] [--corr 0.83] [--routefile F] [--ab-iters 200] [--device N]

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
    mode: u8,
    planar: bool,
}

const USAGE: &str = "usage: moe_wide_oracle --model PATH [--layers 0,17,47] [--rows 9,16,24,32,48] \
[--seeds 3] [--rank 0] [--ranks 2] [--corr 0.83] [--routefile F] [--ab-iters 200] [--device N] [--seed S] [--mode 1|2] [--planar]";

fn parse_list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|p| p.trim().parse::<usize>().map_err(err))
        .collect()
}

fn parse_args() -> Result<Cfg> {
    let mut c = Cfg {
        model: String::new(),
        layers: vec![0, 17, 47],
        rows: vec![9, 16, 24, 32, 48],
        seeds: 3,
        rank: 0,
        ranks: 2,
        corr: 0.83,
        routefile: None,
        ab_iters: 200,
        device: 0,
        seed: 0x32,
        total_experts: 512,
        mode: 1,
        planar: false,
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
            "--mode" => c.mode = v("--mode")?.parse().map_err(err)?,
            "--planar" => c.planar = true,
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
    per: usize,
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
    let per_plan = planar_bytes(Mq4PlanarKind::GateUp, GATE_M, GATE_K).map_err(err)?;
    let out_per = if cfg.planar { per_plan } else { per };
    let mut buf = vec![0u8; owned.len() * out_per];
    for (i, &e) in owned.iter().enumerate() {
        let src = &bytes[e * per..(e + 1) * per];
        if cfg.planar {
            repack_tensor(
                Mq4PlanarKind::GateUp,
                GATE_M,
                GATE_K,
                src,
                &mut buf[i * out_per..(i + 1) * out_per],
            )
            .map_err(|x| format!("{name}[{e}]: {x}"))?;
        } else {
            buf[i * out_per..(i + 1) * out_per].copy_from_slice(src);
        }
    }
    let per = out_per;
    let mut data = gpu
        .upload_raw(&buf, &[owned.len(), GATE_M, GATE_K])
        .map_err(err)?;
    data.dtype = if cfg.planar {
        DType::MQ4G256V2P
    } else {
        DType::MQ4G256V2
    };
    let base = data.buf.as_ptr() as usize as u64;
    let tb: Vec<u8> = (0..owned.len())
        .flat_map(|i| (base + (i * per) as u64).to_le_bytes())
        .collect();
    let table = gpu.upload_raw(&tb, &[owned.len()]).map_err(err)?;
    Ok(Stack {
        data,
        table,
        n: owned.len(),
        per,
    })
}

enum Mode {
    Uniform,
    Corr,
    File,
}
impl Mode {
    fn name(&self) -> &'static str {
        match self {
            Mode::Uniform => "uniform",
            Mode::Corr => "corr",
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
        Mode::File => {
            let cand: Vec<&(usize, Vec<i32>)> = file.iter().filter(|(r, _)| *r <= rows).collect();
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

fn launch(gpu: &mut Gpu, st: &Stack, d: &Dev, rows: usize) -> Result<()> {
    gpu.gemm_mq4g256v2_moe_grouped_top10(
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
    )
    .map_err(err)
}

fn readback(gpu: &Gpu, t: &GpuTensor, words: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; words * 4];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).map_err(err)?;
    Ok(b)
}

fn run() -> Result<i32> {
    let cfg = parse_args()?;
    let mut gpu = Gpu::init_with_device(cfg.device).map_err(err)?;
    if gpu.arch != "gfx1201" {
        println!(
            "SKIP device arch is {}, the g32 family is gfx1201 only",
            gpu.arch
        );
        return Ok(2);
    }
    gpu.set_mq4_experts_planar(cfg.planar);
    let hfq = HfqFile::open(Path::new(&cfg.model)).map_err(|e| format!("open: {e}"))?;
    let file = match &cfg.routefile {
        Some(p) => read_routefile(p)?,
        None => Vec::new(),
    };
    let mut modes = vec![Mode::Uniform, Mode::Corr];
    if !file.is_empty() {
        modes.push(Mode::File);
    }
    let ev0 = gpu.hip.event_create().map_err(err)?;
    let ev1 = gpu.hip.event_create().map_err(err)?;
    let (mut pass, mut fail) = (0usize, 0usize);
    let mut results: Vec<serde_json::Value> = Vec::new();

    for &layer in &cfg.layers {
        let st = load_gate_up(&mut gpu, &hfq, layer, &cfg)?;
        for &rows in &cfg.rows {
            for mode in &modes {
                let mut ab_steps: Vec<(Dev, usize, usize)> = Vec::new();
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
                    let xv: Vec<f32> = (0..nrows * GATE_K).map(|_| rng.act()).collect();
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
                    for arm_mode in [0u8, cfg.mode] {
                        gpu.set_moe32_gu(arm_mode);
                        gpu.hip
                            .memset(&d.y.buf, SENTINEL_BYTE, words * 4)
                            .map_err(err)?;
                        launch(&mut gpu, &st, &d, nrows)?;
                        gpu.hip.device_synchronize().map_err(err)?;
                        outs.push(readback(&gpu, &d.y, words)?);
                    }
                    gpu.set_moe32_gu(0);
                    let (a, b) = (&outs[0], &outs[1]);
                    let (mut mism, mut nonzero) = (0usize, 0usize);
                    let mut first: Option<(usize, u32, u32)> = None;
                    for (i, (wa, wb)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
                        // mode 2 (_nodz) never writes dead slots: compare live grouped rows only
                        if cfg.mode == 2 && step.sorted[i / GATE_M] < 0 {
                            continue;
                        }
                        let x0 = u32::from_le_bytes(wa.try_into().unwrap());
                        let y0 = u32::from_le_bytes(wb.try_into().unwrap());
                        if x0 != 0 && x0 != SENTINEL_WORD {
                            nonzero += 1;
                        }
                        if x0 != y0 {
                            mism += 1;
                            first.get_or_insert((i, x0, y0));
                        }
                    }
                    let ok = mism == 0 && nonzero > 0;
                    if ok {
                        pass += 1;
                    } else {
                        fail += 1;
                    }
                    println!(
                        "{} layer {layer} rows {nrows} mode {} seed {s} distinct {} grouped_rows {grouped_rows} words {words} nonzero {nonzero}{}",
                        if ok { "PASS" } else { "FAIL" },
                        mode.name(),
                        step.distinct,
                        match first {
                            Some((i, x0, y0)) => format!(
                                " mismatches={mism} first=word{i} prod=0x{x0:08x} g32=0x{y0:08x}"
                            ),
                            None if nonzero == 0 => " VACUOUS".to_string(),
                            None => String::new(),
                        }
                    );
                    results.push(json!({"layer": layer, "rows": nrows, "mode": mode.name(), "seed": s,
                        "distinct": step.distinct, "grouped_rows": grouped_rows, "mismatches": mism, "nonzero": nonzero, "pass": ok}));
                    ab_steps.push((d, nrows, step.distinct));
                }
                // interleaved A/B timing over the seeds' routings (alternating arms)
                if cfg.ab_iters > 0 && !ab_steps.is_empty() {
                    let mut t_off: Vec<f64> = Vec::new();
                    let mut t_on: Vec<f64> = Vec::new();
                    let mut dsum = 0usize;
                    for i in 0..cfg.ab_iters {
                        let (d, nrows, dist) = &ab_steps[i % ab_steps.len()];
                        dsum += *dist;
                        for arm in 0..2 {
                            let on = (i + arm) % 2 == 1; // alternate which arm goes first
                            gpu.set_moe32_gu(if on { cfg.mode } else { 0 });
                            gpu.hip.event_record(&ev0, None).map_err(err)?;
                            launch(&mut gpu, &st, d, *nrows)?;
                            gpu.hip.event_record(&ev1, None).map_err(err)?;
                            gpu.hip.event_synchronize(&ev1).map_err(err)?;
                            let us =
                                gpu.hip.event_elapsed_ms(&ev0, &ev1).map_err(err)? as f64 * 1e3;
                            if on {
                                t_on.push(us)
                            } else {
                                t_off.push(us)
                            }
                        }
                    }
                    gpu.set_moe32_gu(0);
                    let med = |v: &mut Vec<f64>| {
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        v[v.len() / 2]
                    };
                    let (mo, mn) = (med(&mut t_off), med(&mut t_on));
                    let mean_d = dsum as f64 / cfg.ab_iters as f64;
                    let gb = |us: f64| mean_d * st.per as f64 / (us * 1e-6) / 1e9;
                    println!(
                        "AB layer {layer} rows {rows} mode {} : prod {:.1} us ({:.0} GB/s)  g32 {:.1} us ({:.0} GB/s)  speedup {:.3}x (mean distinct {:.1})",
                        mode.name(), mo, gb(mo), mn, gb(mn), mo / mn, mean_d
                    );
                }
                for (d, _, _) in ab_steps {
                    let _ = gpu.free_tensor(d.tiles);
                    let _ = gpu.free_tensor(d.sorted);
                    let _ = gpu.free_tensor(d.x);
                    let _ = gpu.free_tensor(d.y);
                }
            }
        }
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
        json!({"schema": "moe-wide-oracle-v1", "model": cfg.model, "layers": cfg.layers, "rows": cfg.rows,
            "cases": pass + fail, "pass": pass, "fail": fail, "verdict": verdict})
    );
    Ok(if verdict == "PASS" { 0 } else { 1 })
}

fn main() {
    match run() {
        Ok(c) => std::process::exit(c),
        Err(e) => {
            eprintln!("moe_wide_oracle: {e}");
            println!(
                "{}",
                json!({"schema": "moe-wide-oracle-v1", "verdict": "FAIL", "error": e})
            );
            std::process::exit(1);
        }
    }
}
