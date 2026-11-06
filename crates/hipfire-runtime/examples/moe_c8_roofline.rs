// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Roofline microbench of the routed-expert decode kernels at the EP2 C8 AR shape, on the REAL
//! expert tensors of a `.mq4-xts` artifact, all VRAM-resident.
//!
//! Production C8 AR (8 lanes x 1 row, top-10, EP2) per rank and layer:
//!   * gate_up  `gemm_mq4g256v2_moe_grouped_top10` (-> `..._o4_r2_x4`, x_src_rows <= 8), grouped
//!              by owned expert, 16-slot tiles;
//!   * down     `gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded` (-> `..._k640_r4`), flat
//!              top-10 ids, non-owned picks = -1.
//! The routing here is uniform over the artifact's expert count; the picks the rank owns
//! (`e % ranks == rank`) are kept, so ~35-37 distinct experts/layer/rank like the measured 35.67.
//! Every iteration uses a fresh routing and cycles through `--layers` so the 64 MB MALL / 8 MB L2 do
//! not serve the weights (working set per layer ~62 MB gate_up + ~31 MB down).
//!
//! Timing: hip events around `--batch` back-to-back launches (launch gaps ~1-3 us vs 100+ us
//! kernels). Reported: us per launch, GB/s of DISTINCT-expert weight bytes (the roofline numerator,
//! what the production floor counts), % of the measured device copy rate and of 640 GB/s.
//!
//! Build: cargo build --release -p hipfire-runtime --features lab --example moe_c8_roofline
//! Run:   moe_c8_roofline --model PATH [--layers 0,1,2,3] [--iters 200] [--rank 0] [--ranks 2]
//!        [--planar] [--device N] [--rows 8]

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::mq4_planar::{
    planar_bytes, repack_tensor, shipped_bytes, Mq4PlanarKind, FLASH_NEXT_DOWN, FLASH_NEXT_GATE_UP,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const K_TOP: usize = 10;
const TILE: usize = 16;
const GATE_M: usize = FLASH_NEXT_GATE_UP.0;
const GATE_K: usize = FLASH_NEXT_GATE_UP.1;
const DOWN_M: usize = FLASH_NEXT_DOWN.0;
const DOWN_K: usize = FLASH_NEXT_DOWN.1;

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
    fn f32(&mut self) -> f32 {
        // roughly normal: sum of 4 uniforms
        let mut s = 0.0f32;
        for _ in 0..4 {
            s += (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        }
        s * 1.7
    }
}

struct Cfg {
    model: String,
    layers: Vec<usize>,
    iters: usize,
    batch: usize,
    rank: usize,
    ranks: usize,
    rows: usize,
    planar: bool,
    device: i32,
    seed: u64,
    total_experts: usize,
}

fn parse_args() -> Result<Cfg> {
    let mut c = Cfg {
        model: String::new(),
        layers: vec![0, 1, 2, 3],
        iters: 200,
        batch: 1,
        rank: 0,
        ranks: 2,
        rows: 8,
        planar: false,
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
                .ok_or_else(|| format!("{n} needs a value"))
        };
        match f.as_str() {
            "--model" => c.model = v("--model")?,
            "--layers" => {
                c.layers = v("--layers")?
                    .split(',')
                    .map(|p| p.trim().parse::<usize>().map_err(err))
                    .collect::<Result<_>>()?
            }
            "--iters" => c.iters = v("--iters")?.parse().map_err(err)?,
            "--batch" => c.batch = v("--batch")?.parse().map_err(err)?,
            "--rank" => c.rank = v("--rank")?.parse().map_err(err)?,
            "--ranks" => c.ranks = v("--ranks")?.parse().map_err(err)?,
            "--rows" => c.rows = v("--rows")?.parse().map_err(err)?,
            "--device" => c.device = v("--device")?.parse().map_err(err)?,
            "--seed" => c.seed = v("--seed")?.parse().map_err(err)?,
            "--planar" => c.planar = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if c.model.is_empty() {
        return Err("--model is required".into());
    }
    Ok(c)
}

struct Stack {
    data: GpuTensor,
    table: GpuTensor,
    per: usize,
}

fn upload_stack(
    gpu: &mut Gpu,
    bytes: &[u8],
    n: usize,
    per: usize,
    rows: usize,
    k: usize,
    dtype: DType,
) -> Result<Stack> {
    let mut data = gpu.upload_raw(bytes, &[n, rows, k]).map_err(err)?;
    data.dtype = dtype;
    let base = data.buf.as_ptr() as usize as u64;
    let tb: Vec<u8> = (0..n)
        .flat_map(|i| (base + (i * per) as u64).to_le_bytes())
        .collect();
    let table = gpu.upload_raw(&tb, &[n]).map_err(err)?;
    Ok(Stack { data, table, per })
}

/// Owned experts (`e % ranks == rank`) of one projection, shipped or planar layout.
fn load_projection(
    gpu: &mut Gpu,
    hfq: &HfqFile,
    layer: usize,
    suffix: &str,
    kind: Mq4PlanarKind,
    (rows, k): (usize, usize),
    cfg: &Cfg,
) -> Result<Stack> {
    let name = format!("model.language_model.layers.{layer}.mlp.experts.{suffix}");
    let (_, bytes) = hfq
        .tensor_data(&name)
        .ok_or_else(|| format!("tensor {name} not in artifact"))?;
    let per_ship = shipped_bytes(kind, rows, k).map_err(err)?;
    let per_plan = planar_bytes(kind, rows, k).map_err(err)?;
    let total = bytes.len() / per_ship;
    if total != cfg.total_experts {
        return Err(format!(
            "{name}: {total} experts != --total {}",
            cfg.total_experts
        ));
    }
    let owned: Vec<usize> = (0..total).filter(|e| e % cfg.ranks == cfg.rank).collect();
    let per = if cfg.planar { per_plan } else { per_ship };
    let mut buf = vec![0u8; owned.len() * per];
    for (i, &e) in owned.iter().enumerate() {
        let src = &bytes[e * per_ship..(e + 1) * per_ship];
        if cfg.planar {
            repack_tensor(kind, rows, k, src, &mut buf[i * per..(i + 1) * per])
                .map_err(|x| format!("{name}[{e}]: {x}"))?;
        } else {
            buf[i * per..(i + 1) * per].copy_from_slice(src);
        }
    }
    let dtype = match (kind, cfg.planar) {
        (Mq4PlanarKind::GateUp, false) => DType::MQ4G256V2,
        (Mq4PlanarKind::GateUp, true) => DType::MQ4G256V2P,
        (Mq4PlanarKind::Down, false) => DType::MQ4G128V2,
        (Mq4PlanarKind::Down, true) => DType::MQ4G128V2P,
    };
    upload_stack(gpu, &buf, owned.len(), per, rows, k, dtype)
}

struct Step {
    topk_local: Vec<i32>, // [rows*10], -1 = not owned (down kernel)
    tiles: Vec<i32>,
    sorted: Vec<i32>,
    distinct: usize,
    owned_slots: usize,
    max_slots_per_expert: usize,
}

fn make_step(rng: &mut Rng, cfg: &Cfg, n_local: usize) -> Step {
    let mut topk_local = vec![-1i32; cfg.rows * K_TOP];
    let mut per: Vec<Vec<i32>> = vec![Vec::new(); n_local];
    let mut owned_slots = 0;
    for r in 0..cfg.rows {
        let mut picks = [usize::MAX; K_TOP];
        let mut c = 0;
        while c < K_TOP {
            let e = rng.below(cfg.total_experts);
            if !picks[..c].contains(&e) {
                picks[c] = e;
                c += 1;
            }
        }
        for (j, &e) in picks.iter().enumerate() {
            if e % cfg.ranks == cfg.rank {
                let l = e / cfg.ranks;
                let flat = (r * K_TOP + j) as i32;
                topk_local[flat as usize] = l as i32;
                per[l].push(flat);
                owned_slots += 1;
            }
        }
    }
    let mut tiles = Vec::new();
    let mut sorted = Vec::new();
    let mut distinct = 0;
    let mut maxs = 0;
    for (l, slots) in per.iter().enumerate() {
        if !slots.is_empty() {
            distinct += 1;
            maxs = maxs.max(slots.len());
        }
        for chunk in slots.chunks(TILE) {
            tiles.push(l as i32);
            sorted.extend_from_slice(chunk);
            sorted.extend(std::iter::repeat(-1).take(TILE - chunk.len()));
        }
    }
    Step {
        topk_local,
        tiles,
        sorted,
        distinct,
        owned_slots,
        max_slots_per_expert: maxs,
    }
}

fn i32b(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct LayerBuf {
    gate: Stack,
    down: Stack,
}

struct StepDev {
    topk: GpuTensor,
    tiles: GpuTensor,
    sorted: GpuTensor,
    grouped_rows: usize,
    distinct: usize,
}

fn run() -> Result<()> {
    let cfg = parse_args()?;
    let mut gpu = Gpu::init_with_device(cfg.device).map_err(err)?;
    if gpu.arch != "gfx1201" {
        return Err(format!("arch {} != gfx1201", gpu.arch));
    }
    gpu.set_mq4_experts_planar(cfg.planar);
    let hfq = HfqFile::open(Path::new(&cfg.model)).map_err(|e| format!("open: {e}"))?;
    let mut layers = Vec::new();
    for &l in &cfg.layers {
        let gate = load_projection(
            &mut gpu,
            &hfq,
            l,
            "gate_up_proj",
            Mq4PlanarKind::GateUp,
            (GATE_M, GATE_K),
            &cfg,
        )?;
        let down = load_projection(
            &mut gpu,
            &hfq,
            l,
            "down_proj",
            Mq4PlanarKind::Down,
            (DOWN_M, DOWN_K),
            &cfg,
        )?;
        layers.push(LayerBuf { gate, down });
    }
    let n_local = layers[0].gate.table.shape[0];
    let gate_bytes = layers[0].gate.per;
    let down_bytes = layers[0].down.per;

    // Device copy rate reference (read+write), 1 GiB
    let ev0 = gpu.hip.event_create().map_err(err)?;
    let ev1 = gpu.hip.event_create().map_err(err)?;
    let big = 1usize << 30;
    let a = gpu.alloc_tensor(&[big / 4], DType::F32).map_err(err)?;
    let b = gpu.alloc_tensor(&[big / 4], DType::F32).map_err(err)?;
    gpu.hip.memset(&a.buf, 1, big).map_err(err)?;
    gpu.hip.memcpy_dtod(&b.buf, &a.buf, big).map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    gpu.hip.event_record(&ev0, None).map_err(err)?;
    for _ in 0..5 {
        gpu.hip.memcpy_dtod(&b.buf, &a.buf, big).map_err(err)?;
    }
    gpu.hip.event_record(&ev1, None).map_err(err)?;
    gpu.hip.event_synchronize(&ev1).map_err(err)?;
    let ms = gpu.hip.event_elapsed_ms(&ev0, &ev1).map_err(err)? as f64;
    let copy_gbs = (5.0 * 2.0 * big as f64) / (ms * 1e-3) / 1e9;
    println!("device dtod copy (read+write bytes): {copy_gbs:.0} GB/s  (spec 640)");
    let _ = gpu.free_tensor(a);
    let _ = gpu.free_tensor(b);

    let mut rng = Rng(cfg.seed);
    // activations: rotated gate_up input [rows, K]; down input [rows*10, 640]
    let x: Vec<f32> = (0..cfg.rows * GATE_K).map(|_| rng.f32()).collect();
    let h: Vec<f32> = (0..cfg.rows * K_TOP * DOWN_K).map(|_| rng.f32()).collect();
    let x_t = gpu.upload_f32(&x, &[x.len()]).map_err(err)?;
    let h_t = gpu.upload_f32(&h, &[h.len()]).map_err(err)?;
    let max_grouped = cfg.rows * K_TOP * TILE;
    let yg = gpu
        .alloc_tensor(&[max_grouped * GATE_M], DType::F32)
        .map_err(err)?;
    let yd = gpu
        .alloc_tensor(&[cfg.rows * K_TOP * DOWN_M], DType::F32)
        .map_err(err)?;

    let mk_step = |gpu: &mut Gpu, rng: &mut Rng| -> Result<StepDev> {
        let s = make_step(rng, &cfg, n_local);
        let topk = gpu
            .upload_raw(&i32b(&s.topk_local), &[s.topk_local.len()])
            .map_err(err)?;
        let tiles = gpu
            .upload_raw(&i32b(&s.tiles), &[s.tiles.len()])
            .map_err(err)?;
        let sorted = gpu
            .upload_raw(&i32b(&s.sorted), &[s.sorted.len()])
            .map_err(err)?;
        let _ = (s.owned_slots, s.max_slots_per_expert);
        Ok(StepDev {
            topk,
            tiles,
            sorted,
            grouped_rows: s.sorted.len(),
            distinct: s.distinct,
        })
    };

    // Pre-generate every step's routing on device (no uploads inside the timed region).
    let nsteps = cfg.iters;
    let mut steps = Vec::new();
    for _ in 0..nsteps {
        steps.push(mk_step(&mut gpu, &mut rng)?);
    }
    let mean_distinct = steps.iter().map(|s| s.distinct).sum::<usize>() as f64 / nsteps as f64;
    let mean_grouped = steps.iter().map(|s| s.grouped_rows).sum::<usize>() as f64 / nsteps as f64;
    println!(
        "layout={} experts/layer(local)={} gate_up B/expert={} down B/expert={} rows={} mean distinct/step={:.2} mean grouped_rows={:.1}",
        if cfg.planar { "planar" } else { "shipped" },
        n_local,
        gate_bytes,
        down_bytes,
        cfg.rows,
        mean_distinct,
        mean_grouped
    );

    let launch_gu = |gpu: &mut Gpu, l: &LayerBuf, s: &StepDev| {
        gpu.gemm_mq4g256v2_moe_grouped_top10(
            &l.gate.table,
            &s.tiles,
            &s.sorted,
            &x_t,
            &yg,
            GATE_M,
            GATE_K,
            K_TOP,
            s.grouped_rows,
            cfg.rows,
        )
    };
    let launch_dn = |gpu: &mut Gpu, l: &LayerBuf, s: &StepDev| {
        gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            &l.down.table,
            &s.topk,
            &h_t,
            &yd,
            DOWN_M,
            DOWN_K,
            cfg.rows,
            n_local,
            K_TOP,
        )
    };

    // warm: compile + one of each
    launch_gu(&mut gpu, &layers[0], &steps[0]).map_err(err)?;
    launch_dn(&mut gpu, &layers[0], &steps[0]).map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;

    for (name, is_gu) in [("gate_up", true), ("down", false)] {
        let mut per_us: Vec<f64> = Vec::new();
        let mut gbs: Vec<f64> = Vec::new();
        let per_expert = if is_gu { gate_bytes } else { down_bytes } as f64;
        let mut i = 0;
        while i + cfg.batch <= nsteps {
            gpu.hip.event_record(&ev0, None).map_err(err)?;
            let mut dist = 0usize;
            for j in 0..cfg.batch {
                let l = &layers[(i + j) % layers.len()];
                let s = &steps[i + j];
                dist += s.distinct;
                if is_gu {
                    launch_gu(&mut gpu, l, s).map_err(err)?;
                } else {
                    launch_dn(&mut gpu, l, s).map_err(err)?;
                }
            }
            gpu.hip.event_record(&ev1, None).map_err(err)?;
            gpu.hip.event_synchronize(&ev1).map_err(err)?;
            let us = gpu.hip.event_elapsed_ms(&ev0, &ev1).map_err(err)? as f64 * 1e3;
            per_us.push(us / cfg.batch as f64);
            gbs.push(dist as f64 * per_expert / (us * 1e-6) / 1e9);
            i += cfg.batch;
        }
        per_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        gbs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = |v: &Vec<f64>| v[v.len() / 2];
        let mean = per_us.iter().sum::<f64>() / per_us.len() as f64;
        println!(
            "{name:8} us/launch median {:.1} mean {:.1} p10 {:.1} p90 {:.1} | GB/s of distinct-expert bytes median {:.0} (p10 {:.0} p90 {:.0}) = {:.0}% of 640, {:.0}% of copy-rate/2 | us/expert {:.2}",
            med(&per_us),
            mean,
            per_us[per_us.len() / 10],
            per_us[per_us.len() * 9 / 10],
            med(&gbs),
            gbs[gbs.len() / 10],
            gbs[gbs.len() * 9 / 10],
            med(&gbs) / 640.0 * 100.0,
            med(&gbs) / (copy_gbs / 2.0) * 100.0,
            med(&per_us) / mean_distinct,
        );
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("moe_c8_roofline: {e}");
        std::process::exit(1);
    }
}
