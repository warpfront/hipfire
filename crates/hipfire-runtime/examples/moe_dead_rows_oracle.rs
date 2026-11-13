// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dead grouped-row store skip oracle (`HIPFIRE_QWEN4_MOE_DEAD_STORES`) + bench on the REAL `down_proj` expert tensors of
//! a `.mq4-xts` artifact (rank-owned experts, shipped layout or the F1 planar twin with `--planar`).
//!
//! Subject: the production grouped routed down (`Gpu::gemm_mq4g128v2_moe_grouped_top10`, `o8_r16`, gfx1201) with the
//! dead-row zero stores skipped (default) versus restored (`Gpu::set_moe_dead_stores(true)`, the reference).
//!
//! Check mode, per (layer, rows, route mode, seed, grouped_rows bound):
//!   * production grouping metadata built exactly like the production fused scatter (padded 16-row expert tiles in
//!     ascending local-expert order, `sorted_slots` -1 over the whole bound, `tile_ids` -1 beyond the live tiles,
//!     `inverse_perm` flat -> grouped row, -1 unrouted); seed 0 = ascending flat order inside an expert, seeds > 0
//!     shuffle inside an expert;
//!   * the skip run's Y pre-poisoned with two different patterns A (NaN payloads) / B (finite large) by an oracle kernel,
//!     and again through the Gpu's own `HIPFIRE_QWEN4_MOE_DEAD_POISON` seeds 1/2: every LIVE Y word and the guard
//!     words equal the zero-store reference, and the production combine `moe_down_combine_grouped_top10` output
//!     (reads through inverse_perm) is byte-identical for reference, poison A and poison B;
//!   * negative control: one routed inverse_perm entry is redirected to a dead slot; the combine comparison MUST then
//!     differ (a control that does not fail is reported FAIL);
//!   * `--reps N`: N repeated skip launches (poison A re-applied) must all reproduce the live words.
//!   Route modes: `uniform`, `corr` (`--corr`), `file` (`--routefile`, records with EXACTLY `rows` rows), `invalid`
//!   (out-of-range / -1 ids plus tiles masked to the -1 sentinel).
//! Bench mode: cold (6 real layers rotated, every launch a different layer), hip events, arms interleaved in a rotating
//!   order: `prod_stores` vs `prod_skip`, median us, weights-only GB/s of distinct expert bytes, total traffic GB/s.
//!
//! Build: cargo build --release -p hipfire-runtime --features lab --example moe_dead_rows_oracle
//! Run:   moe_dead_rows_oracle --mode check|bench --model ABS --rows 9,16,24,32,48 [--routefile ABS] [--corr 0.83]
//!        [--uniform] [--layers 0,17,47] [--bench-layers 0,8,16,24,32,47] [--iters 200] [--out DIR] [--planar]
//!        [--grouped bound,tight] [--seeds 3] [--reps 1] [--rank 0] [--ranks 2]

use hip_bridge::KernargBlob;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::mq4_planar::{
    planar_bytes, repack_tensor, shipped_bytes, Mq4PlanarKind, FLASH_NEXT_DOWN,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::path::Path;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const K_TOP: usize = 10;
const TILE: usize = 16;
const DOWN_M: usize = FLASH_NEXT_DOWN.0;
const DOWN_K: usize = FLASH_NEXT_DOWN.1;
const SENTINEL_BYTE: i32 = 0xa5;
/// Oracle-owned helper kernels (poison fill, word compare). They are plain and shared by every arm.
const HELPERS: &str = r#"
#include <hip/hip_runtime.h>
static __device__ __forceinline__ unsigned k2w_poison_word(int pat, unsigned long long i) {
    if (pat == 0) return 0x7fc00000u | (unsigned)(i & 0x3fffffull);
    if (pat == 1) return 0x4a000000u + (unsigned)((i * 2654435761ull) & 0x7fffffull);
    return 0xa5a5a5a5u;
}
extern "C" __global__ void k2w_poison(unsigned* y, unsigned long long n, int pat) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = k2w_poison_word(pat, i);
}
// out[0] mismatching words in [0, live_words) (a vs b); out[1] mismatches inside LIVE slots; out[2] first mismatch;
// out[3] first live mismatch; out[4] DEAD-slot words of `a` that differ from poison pat_a (= dead words written);
// out[5] guard words (>= live_words) of `a` differing from poison pat_a; out[6] same for `b` / pat_b;
// out[7] nonzero non-sentinel words of `b` in live slots (anti-vacuity).
extern "C" __global__ void k2w_cmp(const unsigned* a, const unsigned* b, const unsigned char* live, unsigned long long n,
                                   unsigned long long live_words, int M, int pat_a, int pat_b, unsigned* out) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i >= live_words) {
        if (a[i] != k2w_poison_word(pat_a, i)) atomicAdd(&out[5], 1u);
        if (b[i] != k2w_poison_word(pat_b, i)) atomicAdd(&out[6], 1u);
        return;
    }
    const bool lv = live[i / (unsigned long long)M] != 0;
    const bool mm = a[i] != b[i];
    if (mm) { atomicAdd(&out[0], 1u); atomicMin(&out[2], (unsigned)i); }
    if (mm && lv) { atomicAdd(&out[1], 1u); atomicMin(&out[3], (unsigned)i); }
    if (!lv && pat_a >= 0 && a[i] != k2w_poison_word(pat_a, i)) atomicAdd(&out[4], 1u);
    if (lv && b[i] != 0u && b[i] != 0xa5a5a5a5u) atomicAdd(&out[7], 1u);
}
"#;

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
        match self.below(64) {
            0 => 0.0,
            1 => -0.0,
            2 => 40.0 * (self.unit() - 0.5),
            _ => (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum RunMode {
    Check,
    Bench,
}
#[derive(Clone, Copy, PartialEq)]
enum Grouped {
    /// grouped_rows = production `grouped_m_total_bound` (what the launcher is really called with)
    Bound,
    /// grouped_rows = live padded rows (what the lab harness / moe_wide_oracle use)
    Tight,
}
impl Grouped {
    fn name(self) -> &'static str {
        match self {
            Grouped::Bound => "bound",
            Grouped::Tight => "tight",
        }
    }
}

struct Cfg {
    mode: RunMode,
    model: String,
    layers: Vec<usize>,
    bench_layers: Vec<usize>,
    rows: Vec<usize>,
    seeds: usize,
    reps: usize,
    rank: usize,
    ranks: usize,
    corr: Option<f64>,
    uniform: bool,
    routefile: Option<String>,
    planar: bool,
    iters: usize,
    out: String,
    device: i32,
    seed: u64,
    total_experts: usize,
    grouped: Vec<Grouped>,
}

const USAGE: &str = "usage: moe_dead_rows_oracle --mode check|bench --model ABS --rows 9,16,24,32,48 [--routefile ABS] [--corr 0.83] \
[--uniform] [--layers 0,17,47] [--bench-layers 0,8,16,24,32,47] [--iters 200] [--out DIR] [--planar] [--grouped bound,tight] \
[--seeds 3] [--reps 1] [--rank 0] [--ranks 2] [--device N] [--seed S] [--variants name:K2W_X=v,K2W_Y=w;name2:...] [--kernel-src PATH] [--live-only] [--gu] [--no-cand] [--no-prod] [--no-lab]";

fn parse_list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|p| p.trim().parse::<usize>().map_err(err))
        .collect()
}

fn parse_args() -> Result<Cfg> {
    let mut c = Cfg {
        mode: RunMode::Check,
        model: String::new(),
        layers: vec![0, 17, 47],
        bench_layers: vec![0, 8, 16, 24, 32, 47],
        rows: vec![9, 16, 24, 32, 48],
        seeds: 3,
        reps: 1,
        rank: 0,
        ranks: 2,
        corr: None,
        uniform: false,
        routefile: None,
        planar: false,
        iters: 200,
        out: String::new(),
        device: 0,
        seed: 0x2a,
        total_experts: 512,
        grouped: vec![Grouped::Bound, Grouped::Tight],
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
            "--mode" => {
                c.mode = match v("--mode")?.as_str() {
                    "check" => RunMode::Check,
                    "bench" => RunMode::Bench,
                    o => return Err(format!("--mode {o}\n{USAGE}")),
                }
            }
            "--model" => c.model = v("--model")?,
            "--layers" => c.layers = parse_list(&v("--layers")?)?,
            "--bench-layers" => c.bench_layers = parse_list(&v("--bench-layers")?)?,
            "--rows" => c.rows = parse_list(&v("--rows")?)?,
            "--seeds" => c.seeds = v("--seeds")?.parse().map_err(err)?,
            "--reps" => c.reps = v("--reps")?.parse().map_err(err)?,
            "--rank" => c.rank = v("--rank")?.parse().map_err(err)?,
            "--ranks" => c.ranks = v("--ranks")?.parse().map_err(err)?,
            "--corr" => c.corr = Some(v("--corr")?.parse().map_err(err)?),
            "--uniform" => c.uniform = true,
            "--routefile" => c.routefile = Some(v("--routefile")?),
            "--planar" => c.planar = true,
            "--iters" => c.iters = v("--iters")?.parse().map_err(err)?,
            "--out" => c.out = v("--out")?,
            "--device" => c.device = v("--device")?.parse().map_err(err)?,
            "--seed" => c.seed = v("--seed")?.parse().map_err(err)?,
            "--grouped" => {
                c.grouped = v("--grouped")?
                    .split(',')
                    .map(|g| match g.trim() {
                        "bound" => Ok(Grouped::Bound),
                        "tight" => Ok(Grouped::Tight),
                        o => Err(format!("--grouped {o}")),
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if c.model.is_empty() {
        return Err(format!("--model is required\n{USAGE}"));
    }
    if c.corr.is_none() && !c.uniform && c.routefile.is_none() {
        c.corr = Some(0.83);
        c.uniform = true;
    }
    Ok(c)
}

// ---------------------------------------------------------------- expert stacks
struct Stack {
    data: GpuTensor,
    table: GpuTensor,
    n: usize,
    per: usize,
}

fn load_down(gpu: &mut Gpu, hfq: &HfqFile, layer: usize, cfg: &Cfg) -> Result<Stack> {
    let name = format!("model.language_model.layers.{layer}.mlp.experts.down_proj");
    let (_, bytes) = hfq
        .tensor_data(&name)
        .ok_or_else(|| format!("tensor {name} not in artifact"))?;
    let per_ship = shipped_bytes(Mq4PlanarKind::Down, DOWN_M, DOWN_K).map_err(err)?;
    let per_plan = planar_bytes(Mq4PlanarKind::Down, DOWN_M, DOWN_K).map_err(err)?;
    let total = bytes.len() / per_ship;
    if total != cfg.total_experts {
        return Err(format!("{name}: {total} experts"));
    }
    let owned: Vec<usize> = (0..total).filter(|e| e % cfg.ranks == cfg.rank).collect();
    let per = if cfg.planar { per_plan } else { per_ship };
    let mut buf = vec![0u8; owned.len() * per];
    for (i, &e) in owned.iter().enumerate() {
        let src = &bytes[e * per_ship..(e + 1) * per_ship];
        if cfg.planar {
            repack_tensor(
                Mq4PlanarKind::Down,
                DOWN_M,
                DOWN_K,
                src,
                &mut buf[i * per..(i + 1) * per],
            )
            .map_err(|x| format!("{name}[{e}]: {x}"))?;
        } else {
            buf[i * per..(i + 1) * per].copy_from_slice(src);
        }
    }
    let mut data = gpu
        .upload_raw(&buf, &[owned.len(), DOWN_M, DOWN_K])
        .map_err(err)?;
    data.dtype = if cfg.planar {
        DType::MQ4G128V2P
    } else {
        DType::MQ4G128V2
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

fn free_stack(gpu: &mut Gpu, st: Stack) {
    let _ = gpu.free_tensor(st.data);
    let _ = gpu.free_tensor(st.table);
}

// ---------------------------------------------------------------- routes
#[derive(Clone, Copy, PartialEq)]
enum RouteMode {
    Uniform,
    Corr,
    File,
    Invalid,
}
impl RouteMode {
    fn name(self) -> &'static str {
        match self {
            RouteMode::Uniform => "uniform",
            RouteMode::Corr => "corr",
            RouteMode::File => "file",
            RouteMode::Invalid => "invalid",
        }
    }
}

fn distinct_pick(rng: &mut Rng, total: usize, have: &[usize]) -> usize {
    loop {
        let e = rng.below(total);
        if !have.contains(&e) {
            return e;
        }
    }
}

/// Global top-10 ids `[rows * 10]` for one case. Empty = not available (file mode without an exact-rows record).
fn picks(
    mode: RouteMode,
    rows: usize,
    rng: &mut Rng,
    cfg: &Cfg,
    file: &[(usize, Vec<i32>)],
) -> Vec<usize> {
    let total = cfg.total_experts;
    match mode {
        RouteMode::Uniform => (0..rows)
            .flat_map(|_| {
                let mut p: Vec<usize> = Vec::new();
                while p.len() < K_TOP {
                    let e = distinct_pick(rng, total, &p);
                    p.push(e);
                }
                p
            })
            .collect(),
        RouteMode::Corr | RouteMode::Invalid => {
            let corr = cfg.corr.unwrap_or(0.83);
            let lanes = 8;
            let bases: Vec<Vec<usize>> = (0..rows.div_ceil(lanes))
                .map(|_| {
                    let mut b: Vec<usize> = Vec::new();
                    while b.len() < K_TOP {
                        let e = distinct_pick(rng, total, &b);
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
                        if rng.unit() as f64 >= corr || p.contains(&e) {
                            e = distinct_pick(rng, total, &p);
                        }
                        p.push(e);
                    }
                    p
                })
                .collect()
        }
        RouteMode::File => {
            let cand: Vec<&(usize, Vec<i32>)> = file.iter().filter(|(r, _)| *r == rows).collect();
            if cand.is_empty() {
                return Vec::new();
            }
            let rec = cand[rng.below(cand.len())];
            rec.1.iter().map(|&e| e as usize).collect()
        }
    }
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
    }
    Ok(out)
}

// ---------------------------------------------------------------- production grouping metadata
struct Meta {
    rows: usize,
    n_local: usize,
    topk_local: Vec<i32>,
    tiles: Vec<i32>,
    sorted: Vec<i32>,
    inverse: Vec<i32>,
    grouped_rows: usize,
    live_slot: Vec<u8>,
    distinct: usize,
    live_slots: usize,
    live_picks: usize,
}

/// `grouped_m_total_bound(total_slots, n_exp)` of hipfire-arch-qwen4 (`program.rs:390`), the grouped_rows the launcher gets.
fn m_total_bound(slots: usize, n_exp: usize) -> usize {
    (slots + TILE.saturating_sub(1) * slots.min(n_exp)).div_ceil(TILE) * TILE
}

fn build_meta(
    p: &[usize],
    cfg: &Cfg,
    grouped: Grouped,
    mode: RouteMode,
    shuffle: Option<u64>,
    n_local: usize,
) -> Meta {
    let rows = p.len() / K_TOP;
    let slots = rows * K_TOP;
    let mut topk: Vec<i32> = p
        .iter()
        .map(|&e| {
            if e % cfg.ranks == cfg.rank {
                (e / cfg.ranks) as i32
            } else {
                -1
            }
        })
        .collect();
    if mode == RouteMode::Invalid {
        for (i, t) in topk.iter_mut().enumerate() {
            if i % 7 == 3 {
                *t = -1;
            } else if i % 13 == 5 {
                *t = (n_local + 3) as i32;
            } else if i % 31 == 11 {
                *t = i32::MAX;
            }
        }
    }
    let mut per: Vec<Vec<i32>> = vec![Vec::new(); n_local];
    for (flat, &e) in topk.iter().enumerate() {
        if e >= 0 && (e as usize) < n_local {
            per[e as usize].push(flat as i32);
        }
    }
    if let Some(s) = shuffle {
        let mut rng = Rng(s);
        for l in per.iter_mut() {
            for i in (1..l.len()).rev() {
                l.swap(i, rng.below(i + 1));
            }
        }
    }
    let m_live: usize = per.iter().map(|l| l.len().div_ceil(TILE) * TILE).sum();
    // the launcher is handed the bound of the GLOBAL expert count (program.rs: moe_dims.num_experts = 512), not the rank-local one
    let bound = m_total_bound(slots, cfg.total_experts);
    assert!(m_live <= bound);
    let grouped_rows = if grouped == Grouped::Bound {
        bound
    } else {
        m_live
    };
    let mut tiles = vec![-1i32; grouped_rows / TILE];
    let mut sorted = vec![-1i32; grouped_rows];
    let mut inverse = vec![-1i32; slots];
    let mut off = 0usize;
    for (e, l) in per.iter().enumerate() {
        for (j, &flat) in l.iter().enumerate() {
            sorted[off + j] = flat;
            inverse[flat as usize] = (off + j) as i32;
        }
        for t in 0..l.len().div_ceil(TILE) {
            tiles[off / TILE + t] = e as i32;
        }
        off += l.len().div_ceil(TILE) * TILE;
    }
    if mode == RouteMode::Invalid {
        // the owner-routed exchange's masked tile-id copy: first live tile of the two lowest experts becomes the sentinel;
        // their picks are no longer routed rows (inverse -1), their sorted entries stay as the scatter wrote them
        let mut masked = 0;
        for t in 0..tiles.len() {
            if tiles[t] >= 0 && masked < 2 {
                for s in t * TILE..(t + 1) * TILE {
                    if sorted[s] >= 0 {
                        inverse[sorted[s] as usize] = -1;
                    }
                }
                tiles[t] = -1;
                masked += 1;
            }
        }
    }
    let live_slot: Vec<u8> = (0..grouped_rows)
        .map(|s| {
            (tiles[s / TILE] >= 0 && (tiles[s / TILE] as usize) < n_local && sorted[s] >= 0) as u8
        })
        .collect();
    let mut dset = std::collections::BTreeSet::new();
    for &t in &tiles {
        if t >= 0 {
            dset.insert(t);
        }
    }
    let live_slots = live_slot.iter().filter(|&&b| b != 0).count();
    let live_picks = inverse.iter().filter(|&&i| i >= 0).count();
    Meta {
        rows,
        n_local,
        topk_local: topk,
        tiles,
        sorted,
        inverse,
        grouped_rows,
        live_slot,
        distinct: dset.len(),
        live_slots,
        live_picks,
    }
}

fn i32b(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct DevMeta {
    tiles: GpuTensor,
    sorted: GpuTensor,
    inverse: GpuTensor,
    topk: GpuTensor,
    topk_w: GpuTensor,
    x: GpuTensor,
    live: GpuTensor,
}

fn upload_meta(gpu: &mut Gpu, m: &Meta, rng: &mut Rng) -> Result<DevMeta> {
    let xv: Vec<f32> = (0..m.rows * K_TOP * DOWN_K).map(|_| rng.act()).collect();
    let wv: Vec<f32> = (0..m.rows * K_TOP).map(|_| 0.05 + rng.unit()).collect();
    Ok(DevMeta {
        tiles: gpu
            .upload_raw(&i32b(&m.tiles), &[m.tiles.len()])
            .map_err(err)?,
        sorted: gpu
            .upload_raw(&i32b(&m.sorted), &[m.sorted.len()])
            .map_err(err)?,
        inverse: gpu
            .upload_raw(&i32b(&m.inverse), &[m.inverse.len()])
            .map_err(err)?,
        topk: gpu
            .upload_raw(&i32b(&m.topk_local), &[m.topk_local.len()])
            .map_err(err)?,
        topk_w: gpu.upload_f32(&wv, &[wv.len()]).map_err(err)?,
        x: gpu.upload_f32(&xv, &[xv.len()]).map_err(err)?,
        live: gpu
            .upload_raw(&m.live_slot, &[m.live_slot.len()])
            .map_err(err)?,
    })
}

fn free_meta(gpu: &mut Gpu, d: DevMeta) {
    for t in [d.tiles, d.sorted, d.inverse, d.topk, d.topk_w, d.x, d.live] {
        let _ = gpu.free_tensor(t);
    }
}

// ---------------------------------------------------------------- kernels
fn load_helpers(gpu: &mut Gpu) -> Result<()> {
    for f in ["k2w_poison", "k2w_cmp"] {
        gpu.ensure_kernel_public("k2w_oracle_helpers", HELPERS, f)
            .map_err(err)?;
    }
    Ok(())
}

fn launch_prod(gpu: &mut Gpu, st: &Stack, d: &DevMeta, y: &GpuTensor, m: &Meta) -> Result<()> {
    gpu.gemm_mq4g128v2_moe_grouped_top10(
        &st.table,
        &d.tiles,
        &d.sorted,
        &d.x,
        y,
        DOWN_M,
        DOWN_K,
        1,
        m.grouped_rows,
        m.rows * K_TOP,
        st.n,
    )
    .map_err(err)
}

fn launch_poison(gpu: &Gpu, y: &GpuTensor, words: usize, pat: i32) -> Result<()> {
    let mut b = KernargBlob::new();
    b.push_ptr(y.buf.as_ptr());
    b.push_u64(words as u64);
    b.push_i32(pat);
    b.pad_to(16);
    gpu.launch_kernel_blob(
        "k2w_poison",
        [words.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .map_err(err)
}

struct CmpOut {
    mism: u32,
    mism_live: u32,
    first: u32,
    first_live: u32,
    dead_written: u32,
    guard_a: u32,
    guard_b: u32,
    nonzero: u32,
}

fn compare(
    gpu: &mut Gpu,
    a: &GpuTensor,
    b: &GpuTensor,
    d: &DevMeta,
    m: &Meta,
    guard_words: usize,
    pat_a: i32,
    pat_b: i32,
) -> Result<CmpOut> {
    let init: [u32; 8] = [0, 0, u32::MAX, u32::MAX, 0, 0, 0, 0];
    let outt = gpu
        .upload_raw(
            &init
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
            &[8],
        )
        .map_err(err)?;
    let live_words = m.grouped_rows * DOWN_M;
    let total = live_words + guard_words;
    let mut bl = KernargBlob::new();
    bl.push_ptr(a.buf.as_ptr());
    bl.push_ptr(b.buf.as_ptr());
    bl.push_ptr(d.live.buf.as_ptr());
    bl.push_u64(total as u64);
    bl.push_u64(live_words as u64);
    bl.push_i32(DOWN_M as i32);
    bl.push_i32(pat_a);
    bl.push_i32(pat_b);
    bl.push_ptr(outt.buf.as_ptr());
    bl.pad_to(16);
    gpu.launch_kernel_blob(
        "k2w_cmp",
        [total.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        bl.as_mut_slice(),
    )
    .map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    let mut raw = [0u8; 32];
    gpu.hip.memcpy_dtoh(&mut raw, &outt.buf).map_err(err)?;
    let _ = gpu.free_tensor(outt);
    let w = |i: usize| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
    Ok(CmpOut {
        mism: w(0),
        mism_live: w(1),
        first: w(2),
        first_live: w(3),
        dead_written: w(4),
        guard_a: w(5),
        guard_b: w(6),
        nonzero: w(7),
    })
}

fn readback(gpu: &Gpu, t: &GpuTensor, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).map_err(err)?;
    Ok(b)
}

/// Run the production combine on `y` with `inverse` into a fresh copy of `resid0`; returns the target bytes.
fn run_combine(
    gpu: &mut Gpu,
    y: &GpuTensor,
    inverse: &GpuTensor,
    d: &DevMeta,
    m: &Meta,
    resid: &GpuTensor,
    resid0: &[u8],
) -> Result<Vec<u8>> {
    gpu.hip.memcpy_htod(&resid.buf, resid0).map_err(err)?;
    gpu.moe_down_combine_grouped_top10(
        y,
        inverse,
        &d.topk,
        &d.topk_w,
        resid,
        DOWN_M,
        m.grouped_rows,
        m.rows,
        m.n_local,
    )
    .map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    readback(gpu, resid, resid0.len())
}

// ---------------------------------------------------------------- check
struct Tally {
    pass: usize,
    fail: usize,
    skipped: usize,
    negctl_pass: usize,
    negctl_skip: usize,
    lines: Vec<Value>,
}

const GUARD_WORDS: usize = 4096;

#[allow(clippy::too_many_arguments)]
fn check_case(
    gpu: &mut Gpu,
    st: &Stack,
    layer: usize,
    mode: RouteMode,
    grouped: Grouped,
    seed: usize,
    p: &[usize],
    cfg: &Cfg,
    rng: &mut Rng,
    t: &mut Tally,
) -> Result<()> {
    let m = build_meta(
        p,
        cfg,
        grouped,
        mode,
        if seed == 0 {
            None
        } else {
            Some(cfg.seed ^ (seed as u64) << 8)
        },
        st.n,
    );
    let d = upload_meta(gpu, &m, rng)?;
    let words = m.grouped_rows * DOWN_M;
    let tot_words = words + GUARD_WORDS;
    let y_prod = gpu.alloc_tensor(&[tot_words], DType::F32).map_err(err)?;
    let y_ex = gpu.alloc_tensor(&[tot_words], DType::F32).map_err(err)?;
    let y_a = gpu.alloc_tensor(&[tot_words], DType::F32).map_err(err)?;
    let y_b = gpu.alloc_tensor(&[tot_words], DType::F32).map_err(err)?;
    gpu.hip
        .memset(&y_prod.buf, SENTINEL_BYTE, tot_words * 4)
        .map_err(err)?;
    // reference = production with the dead-row zero stores (HIPFIRE_QWEN4_MOE_DEAD_STORES=1)
    gpu.set_moe_dead_stores(true);
    launch_prod(gpu, st, &d, &y_prod, &m)?;
    gpu.set_moe_dead_stores(false);
    gpu.hip.device_synchronize().map_err(err)?;
    let resid0: Vec<u8> = (0..m.rows * DOWN_M)
        .flat_map(|_| (rng.act() * 0.5).to_le_bytes())
        .collect();
    let resid = gpu.upload_raw(&resid0, &[m.rows * DOWN_M]).map_err(err)?;
    let res_p = run_combine(gpu, &y_prod, &d.inverse, &d, &m, &resid, &resid0)?;
    // a negative-control redirect: a routed pick -> a dead slot
    let dead_slot = m.live_slot.iter().position(|&b| b == 0);
    let routed = m.inverse.iter().position(|&i| i >= 0);
    let inv2 = match (dead_slot, routed) {
        (Some(ds), Some(r)) => {
            let mut v = m.inverse.clone();
            v[r] = ds as i32;
            Some(gpu.upload_raw(&i32b(&v), &[v.len()]).map_err(err)?)
        }
        _ => None,
    };
    let res_p2 = match &inv2 {
        Some(i2) => Some(run_combine(gpu, &y_prod, i2, &d, &m, &resid, &resid0)?),
        None => None,
    };
    let mut arms: Vec<(String, bool, Value)> = Vec::new(); // (label, ok, detail)
                                                           // production determinism control: the same production launch repeated must reproduce itself
    if cfg.reps > 1 {
        let mut mism = 0u32;
        for _ in 0..cfg.reps {
            gpu.hip
                .memset(&y_ex.buf, SENTINEL_BYTE, tot_words * 4)
                .map_err(err)?;
            gpu.set_moe_dead_stores(true);
            launch_prod(gpu, st, &d, &y_ex, &m)?;
            gpu.set_moe_dead_stores(false);
            gpu.hip.device_synchronize().map_err(err)?;
            mism += compare(gpu, &y_ex, &y_prod, &d, &m, GUARD_WORDS, 2, 2)?.mism;
        }
        arms.push((
            "prod_repeat".into(),
            mism == 0,
            json!({"mism": mism, "reps": cfg.reps}),
        ));
    }
    // production dead-store skip (HIPFIRE_QWEN4_MOE_DEAD_STORES=0, the default): poison pattern A/B written by the helper
    // kernel, then the Gpu's own HIPFIRE_QWEN4_MOE_DEAD_POISON seeds 1/2 (per-call counter); live words, guards, combine bytes
    // must equal the zero-store production; the redirect negative control must fail.
    for (label, gpu_seed) in [
        ("prod_skip.poisonAB", false),
        ("prod_skip.gpupoison12", true),
    ] {
        launch_poison(gpu, &y_a, tot_words, 0)?;
        launch_poison(gpu, &y_b, tot_words, 1)?;
        gpu.set_moe_dead_stores(false);
        gpu.set_moe_dead_poison(if gpu_seed { Some(1) } else { None });
        launch_prod(gpu, st, &d, &y_a, &m)?;
        gpu.set_moe_dead_poison(if gpu_seed { Some(2) } else { None });
        launch_prod(gpu, st, &d, &y_b, &m)?;
        gpu.set_moe_dead_poison(None);
        gpu.hip.device_synchronize().map_err(err)?;
        let pa = if gpu_seed { -1 } else { 0 };
        let pb = if gpu_seed { -1 } else { 1 };
        let (mut ca, cb) = (
            compare(gpu, &y_a, &y_prod, &d, &m, GUARD_WORDS, 0, 2)?,
            compare(gpu, &y_b, &y_prod, &d, &m, GUARD_WORDS, 1, 2)?,
        );
        let _ = (pa, pb);
        // same-case determinism: `--reps` repeated skip launches (poison A re-applied each time) must all reproduce the live words
        let mut rep_mism = 0u32;
        if !gpu_seed {
            for _ in 1..cfg.reps {
                launch_poison(gpu, &y_a, tot_words, 0)?;
                gpu.set_moe_dead_stores(false);
                launch_prod(gpu, st, &d, &y_a, &m)?;
                gpu.hip.device_synchronize().map_err(err)?;
                rep_mism += compare(gpu, &y_a, &y_prod, &d, &m, GUARD_WORDS, 0, 2)?.mism_live;
            }
        }
        ca.mism_live += rep_mism;
        let ra = run_combine(gpu, &y_a, &d.inverse, &d, &m, &resid, &resid0)?;
        let rb = run_combine(gpu, &y_b, &d.inverse, &d, &m, &resid, &resid0)?;
        let live_ok = ca.mism_live == 0
            && cb.mism_live == 0
            && ca.guard_a == 0
            && cb.guard_a == 0
            && ca.nonzero > 0;
        let comb_ok = ra == res_p && rb == res_p;
        arms.push((
            label.to_string(),
            live_ok && comb_ok,
            json!({"live_mism_A": ca.mism_live, "live_mism_B": cb.mism_live, "first_live_A": ca.first_live, "guard_A": ca.guard_a, "guard_B": cb.guard_a,
                   "dead_written_A": ca.dead_written, "combine_A_eq_ref": ra == res_p, "combine_B_eq_ref": rb == res_p, "nonzero": ca.nonzero}),
        ));
        if let (Some(i2), Some(rp2)) = (&inv2, &res_p2) {
            let ra2 = run_combine(gpu, &y_a, i2, &d, &m, &resid, &resid0)?;
            let rb2 = run_combine(gpu, &y_b, i2, &d, &m, &resid, &resid0)?;
            let fails_as_designed = &ra2 != rp2 && ra2 != rb2;
            if fails_as_designed {
                t.negctl_pass += 1;
            }
            arms.push((format!("{label}.negctl"), fails_as_designed, json!({"redirected_combine_differs_from_ref": &ra2 != rp2, "A_differs_from_B": ra2 != rb2})));
        } else {
            t.negctl_skip += 1;
        }
    }
    let ok = arms.iter().all(|(_, ok, _)| *ok);
    if ok {
        t.pass += 1;
    } else {
        t.fail += 1;
    }
    let detail: Vec<Value> = arms
        .iter()
        .map(|(n, ok, dv)| json!({"arm": n, "ok": ok, "d": dv}))
        .collect();
    println!(
        "{} layer {layer} rows {} mode {} grouped {} (rows {}) seed {seed} planar {} distinct {} live_slots {} live_picks {} {}",
        if ok { "PASS" } else { "FAIL" },
        m.rows,
        mode.name(),
        grouped.name(),
        m.grouped_rows,
        cfg.planar,
        m.distinct,
        m.live_slots,
        m.live_picks,
        arms.iter().map(|(n, ok, _)| format!("{n}={}", if *ok { "ok" } else { "BAD" })).collect::<Vec<_>>().join(" ")
    );
    if !ok {
        for (n, ok, dv) in &arms {
            if !ok {
                println!("   FAILDETAIL {n} {dv}");
            }
        }
    }
    t.lines.push(json!({"layer": layer, "rows": m.rows, "mode": mode.name(), "grouped": grouped.name(), "grouped_rows": m.grouped_rows, "seed": seed, "ok": ok,
        "distinct": m.distinct, "live_slots": m.live_slots, "live_picks": m.live_picks, "arms": detail}));
    for tt in [y_prod, y_ex, y_a, y_b, resid] {
        let _ = gpu.free_tensor(tt);
    }
    if let Some(i2) = inv2 {
        let _ = gpu.free_tensor(i2);
    }
    free_meta(gpu, d);
    Ok(())
}

fn route_modes(cfg: &Cfg, have_file: bool) -> Vec<RouteMode> {
    let mut v = Vec::new();
    if cfg.uniform {
        v.push(RouteMode::Uniform);
    }
    if cfg.corr.is_some() {
        v.push(RouteMode::Corr);
        v.push(RouteMode::Invalid);
    }
    if have_file {
        v.push(RouteMode::File);
    }
    v
}

fn run_check(
    gpu: &mut Gpu,
    hfq: &HfqFile,
    cfg: &Cfg,
    file: &[(usize, Vec<i32>)],
) -> Result<(i32, Value)> {
    load_helpers(gpu)?;
    let mut t = Tally {
        pass: 0,
        fail: 0,
        skipped: 0,
        negctl_pass: 0,
        negctl_skip: 0,
        lines: Vec::new(),
    };
    let modes = route_modes(cfg, !file.is_empty());
    for &layer in &cfg.layers {
        let st = load_down(gpu, hfq, layer, cfg)?;
        for &rows in &cfg.rows {
            for &mode in &modes {
                for &g in &cfg.grouped {
                    for seed in 0..cfg.seeds {
                        let mut rng = Rng(cfg.seed
                            ^ (layer as u64) << 40
                            ^ (rows as u64) << 20
                            ^ (seed as u64) << 4
                            ^ (mode as u64));
                        let p = picks(mode, rows, &mut rng, cfg, file);
                        if p.is_empty() {
                            if seed == 0 && g == cfg.grouped[0] {
                                println!("SKIP layer {layer} rows {rows} mode {}: no routefile record with exactly {rows} rows", mode.name());
                                t.skipped += 1;
                            }
                            continue;
                        }
                        check_case(gpu, &st, layer, mode, g, seed, &p, cfg, &mut rng, &mut t)?;
                    }
                }
            }
        }
        free_stack(gpu, st);
    }
    let negctl_ok = t.negctl_pass > 0;
    let verdict = if t.fail == 0 && t.pass > 0 && negctl_ok {
        "PASS"
    } else {
        "FAIL"
    };
    println!(
        "SUMMARY check planar={} cases={} pass={} fail={} skipped_cells={} negctl_pass_arms={} negctl_skipped_arms={} verdict={verdict}",
        cfg.planar, t.pass + t.fail, t.pass, t.fail, t.skipped, t.negctl_pass, t.negctl_skip
    );
    Ok((
        if verdict == "PASS" { 0 } else { 1 },
        json!({"schema": "moe-dead-rows-oracle-v1", "mode": "check", "planar": cfg.planar, "model": cfg.model, "layers": cfg.layers, "rows": cfg.rows,
               "tensor": "model.language_model.layers.<L>.mlp.experts.down_proj", "kernel": "gemm_mq4g128v2[p]_moe_grouped_top10_o8_r16 (dead-store skip)",
               "cases": t.pass + t.fail, "pass": t.pass, "fail": t.fail, "skipped_cells": t.skipped, "negctl_pass_arms": t.negctl_pass,
               "negctl_skipped_arms": t.negctl_skip, "verdict": verdict, "results": t.lines}),
    ))
}

// ---------------------------------------------------------------- bench
#[derive(Clone)]
enum ArmKind {
    Prod(bool), // true = zero stores (HIPFIRE_QWEN4_MOE_DEAD_STORES=1), false = skip (default)
}

struct Arm {
    name: String,
    kind: ArmKind,
    times: Vec<f64>,
}

fn run_bench(
    gpu: &mut Gpu,
    hfq: &HfqFile,
    cfg: &Cfg,
    file: &[(usize, Vec<i32>)],
) -> Result<(i32, Value)> {
    load_helpers(gpu)?;
    let mut stacks: Vec<Stack> = Vec::new();
    for &l in &cfg.bench_layers {
        stacks.push(load_down(gpu, hfq, l, cfg)?);
    }
    let per = stacks[0].per;
    let ev0 = gpu.hip.event_create().map_err(err)?;
    let ev1 = gpu.hip.event_create().map_err(err)?;
    let modes = route_modes(cfg, !file.is_empty());
    const STEPS: usize = 12;
    let mut cells: Vec<Value> = Vec::new();
    for &rows in &cfg.rows {
        for &mode in &modes {
            if mode == RouteMode::Invalid {
                continue; // timing cells are real-route shapes only
            }
            for &g in &cfg.grouped {
                let mut metas: Vec<Meta> = Vec::new();
                let mut devs: Vec<DevMeta> = Vec::new();
                let mut ys: Vec<GpuTensor> = Vec::new();
                for s in 0..STEPS {
                    let mut rng = Rng(cfg.seed
                        ^ 0xbe
                        ^ (rows as u64) << 20
                        ^ (s as u64) << 4
                        ^ (mode as u64));
                    let p = picks(mode, rows, &mut rng, cfg, file);
                    if p.is_empty() {
                        break;
                    }
                    let m = build_meta(
                        &p,
                        cfg,
                        g,
                        mode,
                        if s % 3 == 0 {
                            None
                        } else {
                            Some(cfg.seed ^ s as u64)
                        },
                        stacks[0].n,
                    );
                    devs.push(upload_meta(gpu, &m, &mut rng)?);
                    ys.push(
                        gpu.alloc_tensor(&[m.grouped_rows * DOWN_M + GUARD_WORDS], DType::F32)
                            .map_err(err)?,
                    );
                    metas.push(m);
                }
                if metas.is_empty() {
                    println!("SKIP bench rows {rows} mode {}: no routefile record with exactly {rows} rows", mode.name());
                    continue;
                }
                let mut arms: Vec<Arm> = vec![
                    Arm {
                        name: "prod_stores".into(),
                        kind: ArmKind::Prod(true),
                        times: Vec::new(),
                    },
                    Arm {
                        name: "prod_skip".into(),
                        kind: ArmKind::Prod(false),
                        times: Vec::new(),
                    },
                ];
                let na = arms.len();
                let mut launch_no = 0usize;
                let mut dsum = 0f64;
                let mut xs = 0f64;
                let mut wr_exact = 0f64;
                let mut wr_live = 0f64;
                for it in 0..cfg.iters + 3 * STEPS {
                    let si = it % metas.len();
                    let (m, d, y) = (&metas[si], &devs[si], &ys[si]);
                    let warm = it < 3 * STEPS;
                    if !warm {
                        dsum += m.distinct as f64;
                        xs += (m.live_picks * DOWN_K * 4) as f64;
                        wr_exact += (m.grouped_rows * DOWN_M * 4) as f64;
                        wr_live += (m.live_slots * DOWN_M * 4) as f64;
                    }
                    for k in 0..na {
                        let ai = (k + it) % na;
                        // COLD: every launch of the interleaved sequence reads a different real layer than the previous one
                        // (global launch counter mod layers), so no arm ever finds the previous arm's weights in the MALL;
                        // the route / X / metadata is the same for all arms of one iteration.
                        launch_no += 1;
                        let st = &stacks[launch_no % stacks.len()];
                        gpu.hip.event_record(&ev0, None).map_err(err)?;
                        match arms[ai].kind.clone() {
                            ArmKind::Prod(stores) => {
                                gpu.set_moe_dead_stores(stores);
                                launch_prod(gpu, st, d, y, m)?
                            }
                        }
                        gpu.hip.event_record(&ev1, None).map_err(err)?;
                        gpu.hip.event_synchronize(&ev1).map_err(err)?;
                        let us = gpu.hip.event_elapsed_ms(&ev0, &ev1).map_err(err)? as f64 * 1e3;
                        if !warm {
                            arms[ai].times.push(us);
                        }
                    }
                }
                let n = cfg.iters as f64;
                let (md, mx, mwe, mwl) = (dsum / n, xs / n, wr_exact / n, wr_live / n);
                let mut res: Vec<Value> = Vec::new();
                let mut line = format!("BENCH planar={} rows {rows} mode {} grouped {} (mean rows {:.0}, mean distinct {:.1})", cfg.planar, mode.name(), g.name(),
                    metas.iter().map(|m| m.grouped_rows as f64).sum::<f64>() / metas.len() as f64, md);
                for a in arms.iter_mut() {
                    a.times.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    let len = a.times.len();
                    let (med, p10, p90) =
                        (a.times[len / 2], a.times[len / 10], a.times[len * 9 / 10]);
                    let wb = md * per as f64;
                    let writes = match &a.kind {
                        ArmKind::Prod(false) => mwl,
                        ArmKind::Prod(true) => mwe,
                    };
                    let gbs_w = wb / (med * 1e-6) / 1e9;
                    let gbs_t = (wb + mx + writes) / (med * 1e-6) / 1e9;
                    line.push_str(&format!("\n   {:<16} med {:8.1} us (p10 {:.1} p90 {:.1})  weights {:6.1} GB/s  total {:6.1} GB/s", a.name, med, p10, p90, gbs_w, gbs_t));
                    res.push(json!({"arm": a.name, "us_median": med, "us_p10": p10, "us_p90": p90, "gbs_weights": gbs_w, "gbs_total": gbs_t,
                        "weights_bytes_mean": wb, "x_bytes_mean": mx, "write_bytes_mean": writes}));
                }
                println!("{line}");
                cells.push(json!({"rows": rows, "mode": mode.name(), "grouped": g.name(), "planar": cfg.planar, "mean_distinct": md,
                    "bytes_per_expert": per, "iters": cfg.iters, "bench_layers": cfg.bench_layers, "arms": res}));
                for dv in devs {
                    free_meta(gpu, dv);
                }
                for y in ys {
                    let _ = gpu.free_tensor(y);
                }
            }
        }
    }
    for s in stacks {
        free_stack(gpu, s);
    }
    Ok((
        0,
        json!({"schema": "moe-dead-rows-oracle-v1", "mode": "bench", "planar": cfg.planar, "model": cfg.model, "cells": cells}),
    ))
}

fn run() -> Result<i32> {
    let cfg = parse_args()?;
    let mut gpu = Gpu::init_with_device(cfg.device).map_err(err)?;
    if gpu.arch != "gfx1201" {
        println!(
            "SKIP device arch is {}, the K2W down is gfx1201 only",
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
    let (code, mut js) = match cfg.mode {
        RunMode::Check => run_check(&mut gpu, &hfq, &cfg, &file)?,
        RunMode::Bench => run_bench(&mut gpu, &hfq, &cfg, &file)?,
    };
    if let Value::Object(o) = &mut js {
        o.insert("routefile".into(), json!(cfg.routefile));
    }
    if !cfg.out.is_empty() {
        std::fs::create_dir_all(&cfg.out).map_err(err)?;
        let name = format!(
            "{}{}.json",
            if cfg.mode == RunMode::Check {
                "check"
            } else {
                "bench"
            },
            if cfg.planar { "-planar" } else { "-shipped" }
        );
        std::fs::write(
            Path::new(&cfg.out).join(name),
            serde_json::to_vec_pretty(&js).map_err(err)?,
        )
        .map_err(err)?;
    }
    println!(
        "{}",
        json!({"schema": "moe-dead-rows-oracle-v1", "mode": js["mode"], "verdict": js.get("verdict")})
    );
    Ok(code)
}

fn main() {
    match run() {
        Ok(c) => std::process::exit(c),
        Err(e) => {
            eprintln!("moe_dead_rows_oracle: {e}");
            println!(
                "{}",
                json!({"schema": "moe-dead-rows-oracle-v1", "verdict": "FAIL", "error": e})
            );
            std::process::exit(1);
        }
    }
}
