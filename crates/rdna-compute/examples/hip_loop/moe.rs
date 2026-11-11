// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `moe_k2`: `gemm_mq4g256v2_moe_grouped_wmma_k2` (`kernels/src/gemm_mq4g256v2_moe_grouped_wmma_k2.hip`,
//! `kernels::GEMM_MQ4G256V2_MOE_GROUPED_WMMA_K2_SRC`, runtime module = the kernel name), the gfx1100 grouped
//! MoE WMMA GEMM over MQ4G256V2 (qt44) experts: 39.7% of Raon's pp512 kernel time on the 7900 XTX.
//!
//! # Production calls (Raon-1-mini-preview, pp512), traced
//!
//! One launcher, `Gpu::gemm_mq4g256v2_moe_grouped_wmma_k2` (`gemm.rs:15830`, body `..._impl` `:15861`):
//! gfx1100 takes `mq4g256v2_moe_grouped_wmma_source(false)` (`kernels.rs:4078`) = this node's incumbent
//! (`self.arch_caps.is_gfx1151()` would swap in the Qwen4 copy; gfx1100 does not), converts an F32 `x_src` to
//! F16 UNCACHED (`convert_fp16_x_uncached`, a separate kernel; an F16 `x_src` is used as is), and launches
//! `grid [ceil(M/16), ceil(m_total/16), 1]`, `block [32,1,1]`, 0 dynamic LDS (`:15948-15956`), kernargs
//! `(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_f16, Y_f32, M, K, x_row_div, m_total,
//! n_experts)` = 5 pointers (0..40) + 5 i32 (40..60); `n_experts = moe_expert_table_len(table) = table bytes / 8`.
//!
//! Raon reaches it from `hipfire-arch-raon/src/forward.rs:529` (`ffn_block`) -> `moe.rs:436` `forward_moe` ->
//! `hipfire-dispatch/src/pipeline/moe_program.rs:7287` `execute_grouped_moe` (rows > 1 resolves
//! `GroupedMoeProtocol::Auto` to the Grouped core, `:6598-6608`) and, for the dense layer and the shared
//! expert, `moe.rs:504` `forward_dense` / `moe_program.rs:7086` `gated_ffn_from_basis` (qt44 weights + rows >= 2
//! + BF16 policy + WMMA arch -> single-expert grouping tables, `:7111-7139`). Raon dims (`moe.rs:61`):
//! 128 experts, top-6, hidden 2560, logical expert width 1088 (down K padded to 1280), dense width 12032,
//! 25 layers of which layer 0 is dense (`config.rs:460`, `first_k_dense_replace = 1`) = 24 MoE layers.
//!
//! | launch | where | M | K | x_row_div | m_total | x_src_rows | tables |
//! |---|---|---:|---:|---:|---:|---:|---|
//! | routed gate_up (`gu`) | `moe_program.rs:7390` | 2*1088 = 2176 | 2560 | top_k = 6 | bound(slots, 128) | rows (tokens) | 128 experts, scatter |
//! | routed down (`dn`) | `moe_program.rs:7418` | 2560 | 1280 | 1 | bound(slots, 128) | slots = 6*rows | same scatter |
//! | shared gate (`sg`) | `moe_program.rs:7153-7163` via `:7136` | 1088 | 2560 | 1 | rows | rows | 1 expert |
//! | shared up (`sg`, other weights) | same | 1088 | 2560 | 1 | rows | rows | 1 expert |
//! | shared down (`sd`) | `:7184` via `:7136` | 2560 | 1280 | 1 | rows | rows | 1 expert |
//! | dense gate / up (`dg`, layer 0) | `moe.rs:504` -> `:7153` | 12032 | 2560 | 1 | rows | rows | 1 expert |
//! | dense down (`dd`, layer 0) | same | 2560 | 12032 | 1 | rows | rows | 1 expert |
//!
//! Calls per pp512: a MoE layer issues routed gate_up + routed down + shared gate + shared up + shared down = 5
//! launches, the dense layer 3 (gate, up, down): 24 x 5 + 3 = 123 (the profile's 123 calls). The routed pair
//! dominates the kernel time: gate_up does 2 * (3072 live + padding) * 2176 * 2560 FLOP, the dense/shared calls
//! are one expert each (tens of ms of weights, warm).
//!
//! `bound(slots, E) = ceil((slots + min(slots, E) * 15) / 16) * 16` (`families/moe.rs:321`; pp512: 3072 slots ->
//! 4992, pp2048: 12288 -> 14208). The scatter (`Gpu::moe_scatter_fused_k8`, `moe.rs:574`, kernel
//! `kernels/src/moe_scatter_fused_k8.hip`) pads every touched expert's slot count to 16, writes
//! `sorted_slot_index[m_total_bound]` (flat slot `token * top_k + rank`, or -1 padding; the whole bound is
//! pre-filled -1) and `expert_tile_ids[bound / 16]` (the expert of each 16-slot tile, -1 past the live rows:
//! such tiles `return` at the top of the kernel). The grid is the BOUND, not the live row count: the dead tail
//! (bound - live) is a handful of empty workgroups. Within an expert the production single-workgroup scatter
//! orders slots by an LDS `atomicAdd` race; this node places them in ascending flat-slot order (what the
//! stable gfx1201 variant `moe_scatter_stable_k8_launch` produces). Each output row is computed
//! independently, so the order only moves where a slot's row lands, never its value. Padding slots of a
//! live tile gather the zero vector and are written as 0.
//!
//! Expert pointer table (`weights.rs:1055` `ptr_table`, `:1144` `load_experts`): `[E]` u64 device addresses
//! (stored as `[2E]` F32 words), one separately allocated qt44 buffer per expert: gate_up `[2176, 2560]`
//! = 2176 * 10 groups * 136 B = 2,959,360 B per expert (361 MiB for 128), down `[2560, 1280]` = 2560 * 5 *
//! 136 = 1,740,800 B (212.5 MiB). Group (136 B, `GEMM_MQ4G256V2_...K2_SRC` header): `[sc0:f16][zp0:f16]
//! [sc1:f16][zp1:f16]` then 128 nibble bytes; the Raon artifact is symmetric (`hipfire quantize
//! --mq4v2-symmetric`: `zp = f16(-8 * scale)`, both halves, `hipfire-quantize/src/calibration.rs:1342`).
//! The kernel's output is `Y[slot][row]` f32 (`[m_total x M]`).
//!
//! # Shapes
//!
//! Presets `gu512` (default), `dn512`, `gu2048`, `dn2048`, `sg512`, `sd512`, `dg512`, `dd512` (+ `sg2048`,
//! `sd2048`, `dg2048`, `dd2048`), `emu`. `emu`: 8 experts, top-2, 37 tokens (74 slots, bound 208 = 13 tiles),
//! M = 96 (not a multiple of the 64-row candidate tile), K = 512 (2 groups), ragged per-expert slot counts
//! with a hot expert (>= 2 tiles), partially filled tiles and a dead tail. Keys: `tokens`, `experts`, `top_k`
//! (grouped presets), `M`, `K` (multiple of 256), `route=uniform|skew` (default `skew`), `skew=<sigma>` (lognormal
//! sigma of the expert popularity, default 1.0), `route_seed=<n>`.
//!
//! ROUTING is a deterministic function of the SHAPE (not of `--seed`): each token takes the top-`top_k` of
//! `experts` by a Gumbel-top-k draw over lognormal popularities (`skew`) or equal popularities (`uniform`).
//! The node's live tile count, owner map and `work()` therefore need no device and no seed; `--seed` seeds the
//! weights and the activations only. The routing is synthetic (Raon's sigmoid router on real text is not
//! reproduced); the skew is what makes hot experts own dozens of tiles and cold ones one partial tile, as
//! the Qwen4 source comments describe for prefill.
//!
//! # Candidate header (`// hip_loop: ...`)
//!
//! `rows_per_block` (16; grid X = ceil(M / rows_per_block), multiple of 16), `slots_per_tile` (16; grid Y =
//! ceil(m_total / slots_per_tile), multiple of 16; `expert_tile_ids` stays per 16 slots), `block`
//! (`2 * rows_per_block`, the product's 32 at 16 and the gfx1151 `_x4` entry's 128 at 64; multiple of 32,
//! <= 1024), `lds_dynamic` (0), `tile_run` (1: every 16-slot tile has its own workgroup; r > 1: the grid is
//! unchanged, but a workgroup at an expert-local tile index that is a multiple of r computes the next r - 1
//! tiles of its expert too and their workgroups return, as the gfx1151 `_x4` entry does with r = 4; it needs
//! `slots_per_tile=16`; it only changes the output-owner map the `--emu` extent gate uses). The candidate
//! exports the incumbent's symbol with the incumbent's signature.
//!
//! # Timing pool
//!
//! One weight set is the whole 128-expert table: 361 MiB (gate_up) / 212.5 MiB (down), already above the
//! 192 MiB cold-pool threshold (`support/cold.rs` `POOL_BYTES`), and a pp512 prefill touches every expert, so a
//! dispatch streams more than the 96 MB last-level cache before it revisits an expert: ONE set is used
//! (`timing_pool.sets = 1`) instead of the cold-pool minimum of 3 sets (1.1 GB). Smaller sets (the dense /
//! shared single-expert calls, `emu`) use `pool_sets` as the lane nodes do (dispatch `i` uses set `i % sets`
//! through `select_set`, which re-points kernarg 0 at that set's pointer table). X, the tile tables and Y stay
//! warm across dispatches, as the activations do in a layer.
//!
//! # `--emu`
//!
//! The expert pointer table holds absolute addresses, so the emulator case predicts the emulator's region
//! layout (`Memory::map_guarded`: a bump allocator from an empty address space, 4 KiB aligned, 1 MiB guard
//! below every region; asserted against `peacemaker_ir::emu::Memory` by a test): up to 16 experts get one
//! guarded region each (a read past an expert's end faults), larger tables one concatenated region. Elements
//! of dead tiles are `PAD_OWNER` in the owner map (no workgroup may write them), so the sampled extent gate
//! never asks an arm to write them.
//!
//! Included with `#[path = "hip_loop/moe.rs"] mod moe;`.
#![allow(dead_code)]

use crate::node::*;
use serde_json::{json, Value};

pub const SYMBOL: &str = "gemm_mq4g256v2_moe_grouped_wmma_k2";
const MODULE: &str = SYMBOL;
/// qt44 group: 8 B header (two f16 scale/zero pairs) + 128 nibble bytes.
pub const GROUP_BYTES: usize = 136;
/// `MOE_GROUPED_BLOCK_M` (`hipfire-dispatch/src/families/moe.rs:50`): slots per tile.
pub const BLOCK_M: usize = 16;
/// Output pre-fill: a finite value no kernel produces, so an unwritten element is visible.
const SENTINEL: f32 = -1234.5;
/// Routing seed of every shape unless `route_seed=` overrides it.
pub const DEFAULT_ROUTE_SEED: u64 = 0x6D6F_655F_726F_7574;
pub const DEFAULT_SKEW: f32 = 1.0;
/// At most this many experts get one emulator region each.
const EMU_SPLIT: usize = 16;
const EXPERT_NAMES: [&str; EMU_SPLIT] = [
    "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13", "w14",
    "w15",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Routed gate||up: `x_row_div = top_k`, `x_src_rows = tokens`.
    GateUp,
    /// Routed down: `x_row_div = 1`, `x_src_rows = tokens * top_k`.
    Down,
    /// One-expert tables (shared / dense FFN): `m_total = tokens`, identity slots.
    Single,
}

#[derive(Clone, Copy, Debug)]
pub struct Preset {
    pub name: &'static str,
    pub kind: Kind,
    pub tokens: usize,
    pub experts: usize,
    pub top_k: usize,
    pub m: usize,
    pub k: usize,
}

const fn p(
    name: &'static str,
    kind: Kind,
    tokens: usize,
    experts: usize,
    top_k: usize,
    m: usize,
    k: usize,
) -> Preset {
    Preset {
        name,
        kind,
        tokens,
        experts,
        top_k,
        m,
        k,
    }
}

/// Raon-1-mini-preview: 128 experts, top-6, hidden 2560, expert width 1088 (down K 1280), dense 12032.
pub const PRESETS: [Preset; 13] = [
    p("gu512", Kind::GateUp, 512, 128, 6, 2176, 2560),
    p("dn512", Kind::Down, 512, 128, 6, 2560, 1280),
    p("gu2048", Kind::GateUp, 2048, 128, 6, 2176, 2560),
    p("dn2048", Kind::Down, 2048, 128, 6, 2560, 1280),
    p("sg512", Kind::Single, 512, 1, 1, 1088, 2560),
    p("sd512", Kind::Single, 512, 1, 1, 2560, 1280),
    p("dg512", Kind::Single, 512, 1, 1, 12032, 2560),
    p("dd512", Kind::Single, 512, 1, 1, 2560, 12032),
    p("sg2048", Kind::Single, 2048, 1, 1, 1088, 2560),
    p("sd2048", Kind::Single, 2048, 1, 1, 2560, 1280),
    p("dg2048", Kind::Single, 2048, 1, 1, 12032, 2560),
    p("dd2048", Kind::Single, 2048, 1, 1, 2560, 12032),
    p("emu", Kind::GateUp, 37, 8, 2, 96, 512),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Uniform,
    Skew,
}

impl Route {
    pub fn name(self) -> &'static str {
        match self {
            Route::Uniform => "uniform",
            Route::Skew => "skew",
        }
    }
}

/// `grouped_moe_m_total_bound` (`hipfire-dispatch/src/families/moe.rs:321`): the padded grouped-row bound
/// every grouped launch is sized by.
pub fn m_total_bound(total_slots: usize, n_experts: usize) -> usize {
    let live_expert_bound = total_slots.min(n_experts);
    (total_slots + live_expert_bound * (BLOCK_M - 1)).div_ceil(BLOCK_M) * BLOCK_M
}

/// The scatter's outputs for a launch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Routing {
    /// `[tokens * top_k]` expert ids (grouped kinds), empty for single-expert tables.
    pub topk: Vec<i32>,
    /// `[bound]` flat slot or -1 padding.
    pub sorted: Vec<i32>,
    /// `[bound / 16]` expert of each tile, -1 past the live rows.
    pub tiles: Vec<i32>,
    /// `[E + 1]` exclusive prefix sum of the padded per-expert counts; the last cell is the live row count.
    pub offsets: Vec<i32>,
    /// Padded live rows (`offsets[E]`).
    pub live: usize,
}

impl Routing {
    pub fn live_tiles(&self) -> usize {
        self.live / BLOCK_M
    }
    /// Experts owning at least one live tile.
    pub fn touched_experts(&self) -> usize {
        self.offsets.windows(2).filter(|w| w[1] > w[0]).count()
    }
    /// Most live tiles of one expert.
    pub fn max_tiles_per_expert(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize / BLOCK_M)
            .max()
            .unwrap_or(0)
    }
    /// Tiles whose last slot is padding (partially filled).
    pub fn partial_tiles(&self) -> usize {
        (0..self.live_tiles())
            .filter(|&t| self.sorted[t * BLOCK_M + BLOCK_M - 1] < 0)
            .count()
    }
}

/// Deterministic top-`top_k` of `experts` per token: Gumbel-top-k over per-expert popularities (`Skew`:
/// lognormal with sigma `skew`; `Uniform`: equal). Tokens' picks are in descending key order.
pub fn route_topk(
    tokens: usize,
    top_k: usize,
    experts: usize,
    route: Route,
    skew: f32,
    seed: u64,
) -> Vec<i32> {
    let mut r = Xs64::new(seed);
    let lnw: Vec<f64> = (0..experts)
        .map(|_| match route {
            Route::Skew => f64::from(skew) * f64::from(r.normal()),
            Route::Uniform => 0.0,
        })
        .collect();
    let mut out = Vec::with_capacity(tokens * top_k);
    let mut keys: Vec<(f64, usize)> = Vec::with_capacity(experts);
    for _ in 0..tokens {
        keys.clear();
        for (e, &lw) in lnw.iter().enumerate() {
            let u = ((r.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            keys.push((lw - (-u.ln()).ln(), e));
        }
        keys.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        out.extend(keys[..top_k].iter().map(|k| k.1 as i32));
    }
    out
}

/// `moe_scatter_fused_k8` on the host (`kernels/src/moe_scatter_fused_k8.hip`): per-expert histogram, counts
/// padded to [`BLOCK_M`], exclusive prefix sum, slots placed in ascending flat-slot order within an expert,
/// `sorted` pre-filled -1 over `m_total_max`, `tiles` pre-filled -1 over `m_total_max / 16` then set to the
/// owning expert of each live tile.
pub fn scatter(topk: &[i32], experts: usize, m_total_max: usize) -> Result<Routing> {
    if m_total_max % BLOCK_M != 0 {
        return Err(format!(
            "scatter: m_total_max {m_total_max} is not a multiple of {BLOCK_M}"
        ));
    }
    let mut counts = vec![0usize; experts];
    for &e in topk {
        if e >= 0 && (e as usize) < experts {
            counts[e as usize] += 1;
        }
    }
    let mut offsets = Vec::with_capacity(experts + 1);
    let mut acc = 0usize;
    for &c in &counts {
        offsets.push(acc as i32);
        acc += c.div_ceil(BLOCK_M) * BLOCK_M;
    }
    offsets.push(acc as i32);
    if acc > m_total_max {
        return Err(format!(
            "scatter: {acc} padded live rows exceed the bound {m_total_max}"
        ));
    }
    let mut sorted = vec![-1i32; m_total_max];
    let mut fill = vec![0usize; experts];
    for (i, &e) in topk.iter().enumerate() {
        if e >= 0 && (e as usize) < experts {
            let e = e as usize;
            sorted[offsets[e] as usize + fill[e]] = i as i32;
            fill[e] += 1;
        }
    }
    let mut tiles = vec![-1i32; m_total_max / BLOCK_M];
    for e in 0..experts {
        for t in offsets[e] as usize / BLOCK_M..offsets[e + 1] as usize / BLOCK_M {
            tiles[t] = e as i32;
        }
    }
    Ok(Routing {
        topk: topk.to_vec(),
        sorted,
        tiles,
        offsets,
        live: acc,
    })
}

/// `ffn_single_expert_tables` (`kernels/src/ffn_grouped_activation_pad.hip:122`): identity slots (-1 past
/// `rows`), every tile expert 0, over `rows_pad = ceil(rows / 16) * 16`.
pub fn single_tables(rows: usize) -> Routing {
    let pad = rows.div_ceil(BLOCK_M) * BLOCK_M;
    Routing {
        topk: Vec::new(),
        sorted: (0..pad)
            .map(|i| if i < rows { i as i32 } else { -1 })
            .collect(),
        tiles: vec![0; pad / BLOCK_M],
        offsets: vec![0, pad as i32],
        live: pad,
    }
}

/// `rows` qt44 rows of `groups` groups, symmetric like the Raon artifact: per 128-weight half an f16 scale
/// `d` in `[0.002, 0.006)` and `zp = f16(-8 d)`, uniformly random nibbles. Deterministic in `seed`.
pub fn synth_rows(rows: usize, groups: usize, seed: u64) -> Vec<u8> {
    let mut r = Xs64::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut out = vec![0u8; rows * groups * GROUP_BYTES];
    for g in out.chunks_exact_mut(GROUP_BYTES) {
        for half in 0..2 {
            let u = (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
            let d = half::f16::from_f32(0.002 + 0.004 * u);
            let z = half::f16::from_f32(-8.0 * d.to_f32());
            g[4 * half..4 * half + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            g[4 * half + 2..4 * half + 4].copy_from_slice(&z.to_bits().to_le_bytes());
        }
        for c in g[8..].chunks_exact_mut(8) {
            c.copy_from_slice(&r.next_u64().to_le_bytes());
        }
    }
    out
}

/// Base addresses `Memory::map_guarded(.., guard)` gives regions of `sizes` bytes mapped in order into an
/// empty emulator address space (`peacemaker-ir/src/emu/mod.rs:248`): each above every earlier region with
/// `guard` unmapped bytes between, 4 KiB aligned.
pub fn bump_bases(sizes: &[usize], guard: u64) -> Vec<u64> {
    let mut top = 0u64;
    sizes
        .iter()
        .map(|&s| {
            let base = (top + guard + 0xfff) & !0xfff;
            top = base + s as u64;
            base
        })
        .collect()
}

struct Set {
    experts: Vec<Buf>,
    table: Buf,
    table_ptr: u64,
}

struct State {
    sets: Vec<Set>,
    x: Buf,
    tiles: Buf,
    sorted: Buf,
    y: Buf,
    y0: Buf,
}

pub struct MoeK2 {
    label: String,
    kind: Kind,
    tokens: usize,
    experts: usize,
    top_k: usize,
    m: usize,
    k: usize,
    route: Route,
    skew: f32,
    route_seed: u64,
    routing: Routing,
    st: Option<State>,
}

/// A parsed candidate header (tile sizes and launch shape).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tiles {
    rows_per_block: usize,
    slots_per_tile: usize,
    block: usize,
    lds: usize,
    /// Same-expert tiles one leader workgroup computes (owner map only; the grid is unchanged).
    tile_run: usize,
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl MoeK2 {
    pub fn new(shape: &ShapeSpec) -> Result<Self> {
        let mut s = shape.clone();
        let preset = s.preset.clone().unwrap_or_else(|| "gu512".to_owned());
        let Some(&pr) = PRESETS.iter().find(|q| q.name == preset) else {
            let names: Vec<&str> = PRESETS.iter().map(|q| q.name).collect();
            return Err(format!(
                "moe_k2: unknown shape preset `{preset}` (have {})",
                names.join(", ")
            ));
        };
        let single = pr.kind == Kind::Single;
        let tokens = s.take_usize("tokens", Some(pr.tokens))?;
        let experts = s.take_usize("experts", Some(pr.experts))?;
        let top_k = s.take_usize("top_k", Some(pr.top_k))?;
        let m = s.take_usize("M", Some(pr.m))?;
        let k = s.take_usize("K", Some(pr.k))?;
        let route_seed = match s.kv.remove("route_seed") {
            Some(v) => v
                .parse::<u64>()
                .map_err(|e| format!("--shape route_seed={v}: {e}"))?,
            None => DEFAULT_ROUTE_SEED,
        };
        let route_kv = s.kv.remove("route");
        let skew_kv = s.kv.remove("skew");
        let route = match route_kv.as_deref() {
            None | Some("skew") => Route::Skew,
            Some("uniform") => Route::Uniform,
            Some(o) => return Err(format!("moe_k2: route={o}: uniform or skew")),
        };
        let skew = match &skew_kv {
            Some(v) => v
                .parse::<f32>()
                .ok()
                .filter(|x| x.is_finite() && *x >= 0.0)
                .ok_or_else(|| format!("moe_k2: skew={v}: a finite sigma >= 0"))?,
            None => DEFAULT_SKEW,
        };
        s.no_leftover()?;
        if single && (experts != 1 || top_k != 1) {
            return Err(format!(
                "moe_k2: `{preset}` is a one-expert launch (experts=1, top_k=1), got experts={experts} top_k={top_k}"
            ));
        }
        if single && (route_kv.is_some() || skew_kv.is_some()) {
            return Err(format!(
                "moe_k2: route/skew have no effect on the one-expert preset `{preset}`"
            ));
        }
        if tokens == 0 || m == 0 || k == 0 || k % 256 != 0 || experts == 0 || experts > 1024 {
            return Err(format!(
                "moe_k2: tokens={tokens} experts={experts} M={m} K={k}: need positive dims, K % 256 == 0, experts <= 1024"
            ));
        }
        if top_k == 0 || top_k > experts {
            return Err(format!(
                "moe_k2: top_k={top_k} must be in 1..=experts ({experts})"
            ));
        }
        let slots = tokens
            .checked_mul(top_k)
            .filter(|v| v.saturating_add(experts * BLOCK_M) < i32::MAX as usize)
            .ok_or_else(|| {
                format!("moe_k2: tokens*top_k = {tokens}*{top_k} overflows the i32 slot indices")
            })?;
        let routing = match pr.kind {
            Kind::Single => single_tables(tokens),
            _ => {
                let topk = route_topk(tokens, top_k, experts, route, skew, route_seed);
                scatter(&topk, experts, m_total_bound(slots, experts))?
            }
        };
        let label = if shape.kv.is_empty() {
            preset
        } else {
            let mut l = preset;
            for (k, v) in &shape.kv {
                l.push_str(&format!("-{k}{v}"));
            }
            l
        };
        Ok(Self {
            label,
            kind: pr.kind,
            tokens,
            experts,
            top_k,
            m,
            k,
            route,
            skew,
            route_seed,
            routing,
            st: None,
        })
    }

    fn st(&self) -> Result<&State> {
        self.st
            .as_ref()
            .ok_or_else(|| "moe_k2: setup has not run".to_owned())
    }

    /// Routed slots (`tokens * top_k`; one per row for the single-expert tables).
    pub fn slots(&self) -> usize {
        self.tokens * self.top_k
    }
    /// The kernel's `x_row_div` argument.
    pub fn x_row_div(&self) -> usize {
        match self.kind {
            Kind::GateUp => self.top_k,
            _ => 1,
        }
    }
    /// Rows of the F16 `X_src` (`x_src_rows`).
    pub fn x_src_rows(&self) -> usize {
        match self.kind {
            Kind::Down => self.slots(),
            _ => self.tokens,
        }
    }
    /// The kernel's `m_total` argument: the scatter bound (grouped) or the row count (single-expert).
    pub fn m_total(&self) -> usize {
        match self.kind {
            Kind::Single => self.tokens,
            _ => m_total_bound(self.slots(), self.experts),
        }
    }
    /// Bytes of one expert's qt44 matrix.
    pub fn expert_bytes(&self) -> usize {
        self.m * (self.k / 256) * GROUP_BYTES
    }
    /// Bytes of one weight set (all experts).
    pub fn set_bytes(&self) -> usize {
        self.experts * self.expert_bytes()
    }
    /// Weight sets of the timing pool: one set above the cold threshold, else the lane nodes' policy.
    pub fn pool_sets(&self) -> usize {
        if self.set_bytes() > crate::cold::POOL_BYTES {
            1
        } else {
            crate::cold::pool_sets(self.set_bytes())
        }
    }
    fn x_scale(&self) -> f32 {
        match self.kind {
            Kind::Down => 0.5,
            _ => 1.0,
        }
    }
    pub fn routing(&self) -> &Routing {
        &self.routing
    }

    /// Seeded F16 `X_src` bytes `[x_src_rows][K]`, N(0, `x_scale`).
    pub fn host_x(&self, seed: u64) -> Vec<u8> {
        let mut r = Xs64::new(seed ^ 0xC2B2_AE3D_27D4_EB4F);
        let sc = self.x_scale();
        let mut x = Vec::with_capacity(self.x_src_rows() * self.k * 2);
        for _ in 0..self.x_src_rows() * self.k {
            x.extend_from_slice(&f16_bits(sc * r.normal()).to_le_bytes());
        }
        x
    }

    /// The seed of expert `e` of weight set `set`.
    pub fn expert_seed(seed: u64, set: usize, e: usize) -> u64 {
        seed ^ ((set as u64) << 40) ^ ((e as u64 + 1) << 16)
    }

    /// One expert's qt44 matrix `[M][K / 256]` groups.
    pub fn host_expert(&self, seed: u64, set: usize, e: usize) -> Vec<u8> {
        synth_rows(self.m, self.k / 256, Self::expert_seed(seed, set, e))
    }

    fn y_words(&self) -> usize {
        self.m_total() * self.m
    }

    /// Resolve the header into tile sizes (consumes the keys it understands; refuses the rest).
    fn parse_tiles(&self, hdr: &mut Header) -> Result<Tiles> {
        let rpb = hdr.take_usize("rows_per_block", BLOCK_M)?;
        let spt = hdr.take_usize("slots_per_tile", BLOCK_M)?;
        let block = hdr.take_usize("block", 2 * rpb)?;
        let lds = hdr.take_usize("lds_dynamic", 0)?;
        let tile_run = hdr.take_usize("tile_run", 1)?;
        hdr.no_leftover()?;
        if rpb == 0 || rpb % 16 != 0 || rpb > 1024 {
            return Err(format!(
                "hip_loop header: rows_per_block={rpb} must be a multiple of 16 in 16..=1024"
            ));
        }
        if spt == 0 || spt % BLOCK_M != 0 || spt > 1024 {
            return Err(format!(
                "hip_loop header: slots_per_tile={spt} must be a multiple of 16 in 16..=1024"
            ));
        }
        if block == 0 || block % 32 != 0 || block > 1024 {
            return Err(format!(
                "hip_loop header: block={block} must be a multiple of 32 in 32..=1024"
            ));
        }
        if lds > 64 * 1024 {
            return Err(format!(
                "hip_loop header: lds_dynamic={lds} exceeds the 64 KiB workgroup LDS"
            ));
        }
        if tile_run == 0 || tile_run > 16 || (tile_run > 1 && spt != BLOCK_M) {
            return Err(format!(
                "hip_loop header: tile_run={tile_run} must be 1..=16 and needs slots_per_tile=16 above 1"
            ));
        }
        Ok(Tiles {
            rows_per_block: rpb,
            slots_per_tile: spt,
            block,
            lds,
            tile_run,
        })
    }

    /// The five i32 kernargs after the five pointers: `M, K, x_row_div, m_total, n_experts`.
    fn karg_ints(&self) -> [i32; 5] {
        [
            self.m as i32,
            self.k as i32,
            self.x_row_div() as i32,
            self.m_total() as i32,
            self.experts as i32,
        ]
    }
}

impl Node for MoeK2 {
    fn name(&self) -> &'static str {
        "moe_k2"
    }
    fn shape_label(&self) -> String {
        self.label.clone()
    }
    fn shape_json(&self) -> Value {
        let r = &self.routing;
        let sets = self.pool_sets();
        json!({
            "kind": match self.kind { Kind::GateUp => "routed gate_up", Kind::Down => "routed down", Kind::Single => "one-expert (shared / dense)" },
            "tokens": self.tokens, "experts": self.experts, "top_k": self.top_k,
            "M": self.m, "K": self.k, "x_row_div": self.x_row_div(), "x_src_rows": self.x_src_rows(),
            "m_total": self.m_total(), "tiles_launched": self.m_total().div_ceil(BLOCK_M),
            "live_rows": r.live, "live_tiles": r.live_tiles(),
            "dead_tiles": self.m_total().div_ceil(BLOCK_M).saturating_sub(r.live_tiles()),
            "touched_experts": if self.kind == Kind::Single { 1 } else { r.touched_experts() },
            "max_tiles_per_expert": r.max_tiles_per_expert(), "partial_tiles": r.partial_tiles(),
            "route": self.route.name(), "skew_sigma": self.skew, "route_seed": self.route_seed,
            "weights": "MQ4G256V2 qt44 (136 B groups), symmetric (zp = f16(-8 scale)), separate buffer per expert",
            "expert_bytes": self.expert_bytes(),
            "x": "f16 [x_src_rows][K]", "y": "f32 [m_total][M] (live tiles overwritten, dead tail untouched)",
            "timing_pool": crate::cold::pool_json(sets, self.set_bytes()),
        })
    }
    fn incumbent(&self) -> Result<Incumbent> {
        let (name, source) = rdna_compute::kernels::mq4g256v2_moe_grouped_wmma_source(false);
        debug_assert_eq!(name, SYMBOL);
        Ok(Incumbent {
            module: MODULE.to_owned(),
            source: source.to_owned(),
            symbol: SYMBOL.to_owned(),
            extra_flags: String::new(),
            selection: "crates/rdna-compute/src/gemm.rs Gpu::gemm_mq4g256v2_moe_grouped_wmma_k2_impl (:15861; \
                        kernels::mq4g256v2_moe_grouped_wmma_source(false), gfx1100); callers \
                        hipfire-dispatch/src/pipeline/moe_program.rs execute_grouped_moe (:7390, :7418) and \
                        gated_ffn_from_basis (:7136)"
                .to_owned(),
        })
    }
    fn geometry_with(&self, hdr: &mut Header) -> Result<Geometry> {
        let t = self.parse_tiles(hdr)?;
        let g = |v: usize| u32::try_from(v).map_err(|_| format!("grid dimension {v} exceeds u32"));
        Ok(Geometry {
            grid: [
                g(self.m.div_ceil(t.rows_per_block))?,
                g(self.m_total().div_ceil(t.slots_per_tile))?,
                1,
            ],
            block: [t.block as u32, 1, 1],
            lds: t.lds as u32,
        })
    }
    fn work(&self) -> Work {
        let (m, k) = (self.m as f64, self.k as f64);
        let touched = if self.kind == Kind::Single {
            1
        } else {
            self.routing.touched_experts()
        };
        Work {
            flops: 2.0 * self.m_total() as f64 * m * k,
            bytes: (touched * self.expert_bytes()) as f64
                + (self.x_src_rows() * self.k * 2) as f64
                + (self.routing.live * self.m * 4) as f64,
        }
    }
    fn static_hazards(&self) -> bool {
        true
    }
    fn extent_shapes(&self) -> Vec<String> {
        ["gu512", "dn512", "gu2048", "dn2048"]
            .iter()
            .chain(["sg512", "sd512", "dg512", "dd512"].iter())
            .map(|s| (*s).to_owned())
            .collect()
    }
    fn out_owner<'a>(&'a self, hdr: &Header) -> Result<Box<dyn Fn(usize) -> [u32; 3] + 'a>> {
        // Y is f32 [m_total][M]; the workgroup is (row block, slot tile). Dead tiles are never written.
        // `tile_run = r > 1`: a workgroup at an expert-local tile index that is a multiple of r also
        // computes the next r - 1 tiles of the same expert (their own workgroups return): the owner of a
        // tile is the run leader.
        let t = self.parse_tiles(&mut hdr.clone())?;
        let (m, tiles, offsets) = (self.m, &self.routing.tiles, &self.routing.offsets);
        Ok(Box::new(move |e| {
            let slot = e / m;
            let tile = slot / BLOCK_M;
            let Some(&expert) = tiles.get(tile).filter(|&&x| x >= 0) else {
                return PAD_OWNER;
            };
            let y = if t.tile_run > 1 {
                let first = offsets[expert as usize] as usize / BLOCK_M;
                tile - (tile - first) % t.tile_run
            } else {
                slot / t.slots_per_tile
            };
            [(e % m / t.rows_per_block) as u32, y as u32, 0]
        }))
    }
    fn emu_case(&self, seed: u64) -> Result<EmuCase> {
        let split = self.experts <= EMU_SPLIT;
        let x = self.host_x(seed);
        let tiles = i32s(&self.routing.tiles);
        let sorted = i32s(&self.routing.sorted);
        let y = f32s(&vec![SENTINEL; self.y_words()]);
        // Regions in mapping order; the pointer table needs the weights' predicted addresses.
        let wsizes: Vec<usize> = if split {
            vec![self.expert_bytes(); self.experts]
        } else {
            vec![self.set_bytes()]
        };
        let mut sizes = wsizes.clone();
        sizes.extend([
            x.len(),
            tiles.len(),
            sorted.len(),
            8 * self.experts,
            y.len(),
        ]);
        let bases = bump_bases(&sizes, crate::emu::GUARD_BYTES);
        let table: Vec<u8> = (0..self.experts)
            .flat_map(|e| {
                let a = if split {
                    bases[e]
                } else {
                    bases[0] + (e * self.expert_bytes()) as u64
                };
                a.to_le_bytes()
            })
            .collect();
        let mut bufs = Vec::new();
        if split {
            for e in 0..self.experts {
                bufs.push(EmuBuf {
                    name: EXPERT_NAMES[e],
                    bytes: self.host_expert(seed, 0, e),
                });
            }
        } else {
            let mut w = Vec::with_capacity(self.set_bytes());
            for e in 0..self.experts {
                w.extend_from_slice(&self.host_expert(seed, 0, e));
            }
            bufs.push(EmuBuf {
                name: "w",
                bytes: w,
            });
        }
        let nw = bufs.len();
        bufs.extend([
            EmuBuf {
                name: "x",
                bytes: x,
            },
            EmuBuf {
                name: "tiles",
                bytes: tiles,
            },
            EmuBuf {
                name: "sorted",
                bytes: sorted,
            },
            EmuBuf {
                name: "ptrs",
                bytes: table,
            },
            EmuBuf {
                name: "y",
                bytes: y,
            },
        ]);
        let mut args = vec![
            EmuArg::Buf(nw + 3),
            EmuArg::Buf(nw + 1),
            EmuArg::Buf(nw + 2),
            EmuArg::Buf(nw),
            EmuArg::Buf(nw + 4),
        ];
        args.extend(self.karg_ints().map(EmuArg::I32));
        Ok(EmuCase {
            bufs,
            args,
            out: nw + 4,
        })
    }

    fn setup(&mut self, gpu: &Gpu, seed: u64) -> Result<()> {
        let mut sets = Vec::new();
        for s in 0..self.pool_sets() {
            let mut experts = Vec::with_capacity(self.experts);
            for e in 0..self.experts {
                experts.push(Buf::from_bytes(gpu, &self.host_expert(seed, s, e))?);
            }
            let table: Vec<u8> = experts
                .iter()
                .flat_map(|b| (b.ptr() as u64).to_le_bytes())
                .collect();
            let table = Buf::from_bytes(gpu, &table)?;
            let table_ptr = table.ptr() as u64;
            sets.push(Set {
                experts,
                table,
                table_ptr,
            });
        }
        let y = f32s(&vec![SENTINEL; self.y_words()]);
        self.st = Some(State {
            sets,
            x: Buf::from_bytes(gpu, &self.host_x(seed))?,
            tiles: Buf::from_bytes(gpu, &i32s(&self.routing.tiles))?,
            sorted: Buf::from_bytes(gpu, &i32s(&self.routing.sorted))?,
            y: Buf::from_bytes(gpu, &y)?,
            y0: Buf::from_bytes(gpu, &y)?,
        });
        Ok(())
    }
    fn reset(&mut self, gpu: &Gpu) -> Result<()> {
        let s = self.st()?;
        s.y.copy_from(gpu, &s.y0)
    }
    fn kernarg(&self) -> Result<Vec<u8>> {
        let s = self.st()?;
        let mut a = Args::new();
        a.raw_ptr(s.sets[0].table_ptr)
            .ptr(&s.tiles)
            .ptr(&s.sorted)
            .ptr(&s.x)
            .ptr(&s.y);
        for v in self.karg_ints() {
            a.i32(v);
        }
        Ok(a.finish())
    }
    fn select_set(&self, karg: &mut [u8], i: usize) {
        if let Ok(s) = self.st() {
            karg[..8].copy_from_slice(&s.sets[i % s.sets.len()].table_ptr.to_le_bytes());
        }
    }
    fn read_output(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        self.st()?.y.download(gpu)
    }
    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(s) = self.st.take() {
            for set in s.sets {
                for b in set.experts {
                    b.free(gpu);
                }
                set.table.free(gpu);
            }
            for b in [s.x, s.tiles, s.sorted, s.y, s.y0] {
                b.free(gpu);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(shape: &str) -> MoeK2 {
        MoeK2::new(&ShapeSpec::parse(Some(shape)).unwrap()).unwrap()
    }

    #[test]
    fn production_geometry_matches_the_launcher() {
        // gemm.rs:15948-15956: grid [ceil(M/16), ceil(m_total/16), 1], block 32, no dynamic LDS.
        for (name, m, mt) in [
            ("gu512", 2176usize, 4992usize),
            ("dn512", 2560, 4992),
            ("gu2048", 2176, 14208),
            ("dn2048", 2560, 14208),
        ] {
            let n = node(name);
            assert_eq!(n.m_total(), mt, "{name}");
            let g = n.geometry();
            assert_eq!(g.grid, [(m / 16) as u32, (mt / 16) as u32, 1], "{name}");
            assert_eq!((g.block, g.lds), ([32, 1, 1], 0), "{name}");
            assert_eq!(n.incumbent().unwrap().symbol, SYMBOL);
        }
        // moe_program.rs:7519 `scratch_plan_matches_grouped_row_bound`: 512 rows -> 4992 rows, 312 tile ids.
        assert_eq!(m_total_bound(512 * 6, 128), 4992);
        assert_eq!(m_total_bound(2048 * 6, 128), 14208);
        // one-expert tables: m_total = rows, y = ceil(rows / 16)
        let s = node("sg512");
        assert_eq!(s.geometry().grid, [68, 32, 1]);
        assert_eq!(node("dd512").geometry().grid, [160, 32, 1]);
        assert_eq!(node("dg2048").geometry().grid, [752, 128, 1]);
    }

    #[test]
    fn presets_carry_the_production_launch_arguments() {
        let gu = node("gu512");
        assert_eq!(
            (gu.m, gu.k, gu.x_row_div(), gu.x_src_rows(), gu.experts),
            (2176, 2560, 6, 512, 128)
        );
        let dn = node("dn2048");
        assert_eq!(
            (dn.m, dn.k, dn.x_row_div(), dn.x_src_rows()),
            (2560, 1280, 1, 12288)
        );
        let sd = node("sd512");
        assert_eq!(
            (
                sd.m,
                sd.k,
                sd.x_row_div(),
                sd.x_src_rows(),
                sd.m_total(),
                sd.experts
            ),
            (2560, 1280, 1, 512, 512, 1)
        );
        assert_eq!(gu.expert_bytes(), 2_959_360);
        assert_eq!(node("dn512").expert_bytes(), 1_740_800);
        assert_eq!(gu.set_bytes(), 128 * 2_959_360);
        // the call count the profile shows: 24 MoE layers x 5 + the dense layer's 3
        assert_eq!(24 * 5 + 3, 123);
        assert_eq!(node("gu512").label, "gu512");
        assert_eq!(node("emu").label, "emu");
        assert_eq!(
            MoeK2::new(&ShapeSpec::parse(None).unwrap()).unwrap().label,
            "gu512"
        );
    }

    #[test]
    fn shape_grammar_and_refusals() {
        let n = node(
            "gu512,tokens=64,experts=16,top_k=3,M=128,K=512,route=uniform,skew=0.5,route_seed=9",
        );
        assert_eq!(
            (
                n.tokens,
                n.experts,
                n.top_k,
                n.m,
                n.k,
                n.route,
                n.route_seed
            ),
            (64, 16, 3, 128, 512, Route::Uniform, 9)
        );
        assert_eq!(n.skew, 0.5);
        assert_eq!(n.m_total(), m_total_bound(192, 16));
        assert!(n.label.starts_with("gu512-"));
        for bad in [
            "nope",
            "gu512,q=1",
            "gu512,K=100",
            "gu512,K=0",
            "gu512,tokens=0",
            "gu512,top_k=0",
            "gu512,top_k=129",
            "gu512,experts=0",
            "gu512,experts=2000",
            "gu512,route=bad",
            "gu512,skew=-1",
            "gu512,skew=x",
            "gu512,route_seed=x",
            "gu512,dn512",
            "sg512,experts=4",
            "sg512,top_k=2",
            "sg512,route=uniform",
            "gu512,tokens=2147483647",
        ] {
            let r = ShapeSpec::parse(Some(bad)).and_then(|sp| MoeK2::new(&sp));
            assert!(r.is_err(), "{bad}");
        }
        // a one-expert preset may still resize tokens and dims
        let s = node("sg512,tokens=100,M=48");
        assert_eq!(
            (s.tokens, s.m_total(), s.geometry().grid),
            (100, 100, [3, 7, 1])
        );
        assert_eq!(s.routing().tiles.len(), 7);
        assert_eq!(s.routing().sorted[99], 99);
        assert_eq!(s.routing().sorted[100], -1);
    }

    #[test]
    fn header_geometry_and_refusals() {
        let n = node("gu512");
        let mut h = Header::parse("// hip_loop: rows_per_block=64").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!((g.grid, g.block, g.lds), ([34, 312, 1], [128, 1, 1], 0));
        let mut h = Header::parse(
            "// hip_loop: rows_per_block=32 slots_per_tile=32 block=64 lds_dynamic=4096",
        )
        .unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!((g.grid, g.block, g.lds), ([68, 156, 1], [64, 1, 1], 4096));
        // M = 96 under a 64-row block: ragged, still covered
        let e = node("emu");
        let mut h = Header::parse("// hip_loop: rows_per_block=64").unwrap();
        assert_eq!(e.geometry_with(&mut h).unwrap().grid, [2, 13, 1]);
        let mut h = Header::parse("// hip_loop: rows_per_block=16 block=32").unwrap();
        assert_eq!(e.geometry_with(&mut h).unwrap(), e.geometry());
        for bad in [
            "rows_per_block=0",
            "rows_per_block=24",
            "rows_per_block=2048",
            "slots_per_tile=8",
            "slots_per_tile=0",
            "slots_per_tile=20",
            "block=100",
            "block=2048",
            "block=0",
            "lds_dynamic=70000",
            "bogus=1",
            "tile_m=16",
            "rows_per_block=abc",
        ] {
            let mut h = Header::parse(&format!("// hip_loop: {bad}")).unwrap();
            assert!(n.geometry_with(&mut h).is_err(), "{bad}");
        }
    }

    #[test]
    fn scatter_follows_the_production_kernel() {
        // 6 slots over 3 experts with one partial tile each, bound 6 + 3*15 = 51 -> 64.
        let topk = [2, 0, 2, 1, 0, 2];
        let r = scatter(&topk, 3, m_total_bound(6, 3)).unwrap();
        assert_eq!(m_total_bound(6, 3), 64);
        assert_eq!(r.offsets, vec![0, 16, 32, 48]);
        assert_eq!(r.live, 48);
        assert_eq!(r.tiles, vec![0, 1, 2, -1]);
        assert_eq!(&r.sorted[0..3], &[1, 4, -1]);
        assert_eq!(&r.sorted[16..18], &[3, -1]);
        assert_eq!(&r.sorted[32..36], &[0, 2, 5, -1]);
        assert!(r.sorted[48..].iter().all(|&s| s == -1));
        assert_eq!(
            (r.live_tiles(), r.touched_experts(), r.partial_tiles()),
            (3, 3, 3)
        );
        assert!(scatter(&topk, 3, 32).is_err());
        assert!(scatter(&topk, 3, 20).is_err());
        // every slot appears exactly once, in its own expert's range
        let n = node("gu512");
        let r = n.routing();
        let mut seen = vec![0u8; n.slots()];
        for (row, &s) in r.sorted.iter().enumerate() {
            if s >= 0 {
                seen[s as usize] += 1;
                assert_eq!(r.tiles[row / 16], r.topk[s as usize], "row {row}");
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
        assert!(r.live <= n.m_total());
        assert_eq!(r.live % 16, 0);
    }

    #[test]
    fn routing_is_deterministic_distinct_and_skewed() {
        let a = route_topk(512, 6, 128, Route::Skew, 1.0, 7);
        assert_eq!(a, route_topk(512, 6, 128, Route::Skew, 1.0, 7));
        assert_ne!(a, route_topk(512, 6, 128, Route::Skew, 1.0, 8));
        for t in a.chunks_exact(6) {
            let mut s = t.to_vec();
            s.sort_unstable();
            s.dedup();
            assert_eq!(s.len(), 6, "{t:?}");
            assert!(t.iter().all(|&e| (0..128).contains(&e)));
        }
        let hist = |v: &[i32]| {
            let mut h = vec![0usize; 128];
            for &e in v {
                h[e as usize] += 1;
            }
            h.sort_unstable();
            h
        };
        let (sk, un) = (
            hist(&a),
            hist(&route_topk(512, 6, 128, Route::Uniform, 1.0, 7)),
        );
        // skew: a hot expert far above the 24-slot mean, many cold ones; uniform stays near the mean
        assert!(sk[127] > 3 * 24 && sk[0] < 24 / 2, "{sk:?}");
        assert!(un[127] < 2 * 24 + 16 && un[0] > 4, "{un:?}");
        // the default routes of the production shapes touch (nearly) every expert and leave a dead tail
        for name in ["gu512", "dn2048"] {
            let n = node(name);
            let r = n.routing();
            assert!(r.touched_experts() > 100, "{name}");
            assert!(r.max_tiles_per_expert() >= 4, "{name}");
            assert!(r.live <= n.m_total() && r.live > n.slots(), "{name}");
        }
        // gu and dn of one token count share the scatter
        assert_eq!(node("gu512").routing(), node("dn512").routing());
    }

    #[test]
    fn emu_routing_is_ragged_with_a_hot_expert_and_a_dead_tail() {
        let n = node("emu");
        let r = n.routing();
        assert_eq!((n.slots(), n.m_total(), r.tiles.len()), (74, 208, 13));
        assert!(r.max_tiles_per_expert() >= 2, "{r:?}");
        assert!(r.partial_tiles() >= 3, "{r:?}");
        assert!(r.live_tiles() < 13, "{r:?}");
        assert!(r.touched_experts() >= 6, "{r:?}");
    }

    #[test]
    fn synthetic_weights_are_valid_symmetric_qt44() {
        let w = synth_rows(48, 2, 7);
        assert_eq!(w.len(), 48 * 2 * GROUP_BYTES);
        assert_eq!(w, synth_rows(48, 2, 7));
        assert_ne!(w, synth_rows(48, 2, 8));
        for g in w.chunks_exact(GROUP_BYTES) {
            for h in 0..2 {
                let d = half::f16::from_le_bytes([g[4 * h], g[4 * h + 1]]).to_f32();
                let z = half::f16::from_le_bytes([g[4 * h + 2], g[4 * h + 3]]).to_f32();
                assert!((0.002..0.0061).contains(&d), "{d}");
                assert_eq!(z, half::f16::from_f32(-8.0 * d).to_f32());
            }
            // the nibbles are not degenerate
            assert!(g[8..].iter().any(|&b| b != g[8]));
        }
        // seeds differ per expert and per set, the first set's expert 0 is reproducible
        let n = node("emu");
        assert_eq!(n.host_expert(3, 0, 1), n.host_expert(3, 0, 1));
        assert_ne!(n.host_expert(3, 0, 1), n.host_expert(3, 0, 2));
        assert_ne!(n.host_expert(3, 0, 1), n.host_expert(3, 1, 1));
    }

    #[test]
    fn work_and_pool_policy() {
        let n = node("gu512");
        let w = n.work();
        assert_eq!(w.flops, 2.0 * 4992.0 * 2176.0 * 2560.0);
        let r = n.routing();
        assert_eq!(
            w.bytes,
            (r.touched_experts() * 2_959_360 + 512 * 2560 * 2 + r.live * 2176 * 4) as f64
        );
        // 361 / 212.5 MiB sets are above the 192 MiB cold threshold: one set; small sets cycle
        assert_eq!(n.pool_sets(), 1);
        assert_eq!(node("dn512").pool_sets(), 1);
        assert!(node("sg512").pool_sets() > 20);
        assert_eq!(node("sg512").pool_sets() % 2, 1);
        let dg = node("dg512");
        assert!(dg.pool_sets() * dg.set_bytes() > crate::cold::POOL_BYTES);
        assert_eq!(n.shape_json()["timing_pool"]["sets"], 1);
        assert_eq!(n.shape_json()["m_total"], 4992);
    }

    #[test]
    fn extents_are_the_production_launches() {
        let ex = node("emu").extent_shapes();
        assert_eq!(
            ex,
            ["gu512", "dn512", "gu2048", "dn2048", "sg512", "sd512", "dg512", "dd512"]
        );
        for s in ex {
            assert!(PRESETS.iter().any(|p| p.name == s), "{s}");
        }
        assert!(node("emu").static_hazards());
        assert_eq!(node("emu").arch(), ARCH);
    }

    #[test]
    fn owner_map_follows_the_output_layout_and_skips_dead_tiles() {
        let n = node("emu");
        let r = n.routing();
        let live = r.live_tiles();
        let own = n.out_owner(&Header::default()).unwrap();
        // Y[slot][row], 96 rows: (slot 0, row 0) -> (0, 0); (slot 17, row 95) -> (5, 1)
        assert_eq!(own(0), [0, 0, 0]);
        assert_eq!(own(17 * 96 + 95), [5, 1, 0]);
        let h = Header::parse("// hip_loop: rows_per_block=64 slots_per_tile=32").unwrap();
        let own64 = n.out_owner(&h).unwrap();
        assert_eq!(own64(17 * 96 + 95), [1, 0, 0]);
        assert_eq!(own64(40 * 96 + 70), [1, 1, 0]);
        // dead tiles belong to no workgroup (candidate header or not)
        let first_dead = live * 16 * 96;
        assert_eq!(own(first_dead), PAD_OWNER);
        assert_eq!(own64(first_dead + 5), PAD_OWNER);
        assert_eq!(own(first_dead - 1)[1] as usize, live - 1);
        let mut bad = Header::default();
        bad.kv.insert("bogus".into(), "1".into());
        assert!(n.out_owner(&bad).is_err());
        // every element of a live tile is owned by a workgroup inside the launch grid
        let g = n.geometry();
        for e in (0..live * 16 * 96).step_by(7) {
            let o = own(e);
            assert!(o[0] < g.grid[0] && o[1] < g.grid[1], "{e}: {o:?}");
        }
        // one-expert tables own every element
        let s = node("sg512,tokens=100,M=48");
        let o = s.out_owner(&Header::default()).unwrap();
        assert!((0..100 * 48).all(|e| o(e) != PAD_OWNER));
    }

    #[test]
    fn emu_case_uses_the_production_kernarg_layout_and_a_valid_pointer_table() {
        let n = node("emu");
        let c = n.emu_case(1).unwrap();
        assert_eq!(c.args.len(), 10);
        let nw = c.bufs.len() - 5;
        assert_eq!(nw, 8);
        let names: Vec<&str> = c.bufs.iter().map(|b| b.name).collect();
        assert_eq!(&names[nw..], ["x", "tiles", "sorted", "ptrs", "y"]);
        assert_eq!(c.out, c.bufs.len() - 1);
        assert_eq!(
            c.args[..5],
            [
                EmuArg::Buf(nw + 3),
                EmuArg::Buf(nw + 1),
                EmuArg::Buf(nw + 2),
                EmuArg::Buf(nw),
                EmuArg::Buf(nw + 4)
            ]
        );
        let ints: Vec<i32> = c.args[5..]
            .iter()
            .map(|a| match a {
                EmuArg::I32(v) => *v,
                o => panic!("{o:?}"),
            })
            .collect();
        assert_eq!(ints, vec![96, 512, 2, 208, 8]);
        // sizes: x f16 [37][512], tile ids [13], sorted [208], table 8 x u64, y f32 [208][96]
        assert_eq!(c.bufs[nw].bytes.len(), 37 * 512 * 2);
        assert_eq!(c.bufs[nw + 1].bytes.len(), 13 * 4);
        assert_eq!(c.bufs[nw + 2].bytes.len(), 208 * 4);
        assert_eq!(c.bufs[nw + 3].bytes.len(), 8 * 8);
        assert_eq!(c.bufs[nw + 4].bytes.len(), 208 * 96 * 4);
        for b in &c.bufs[..nw] {
            assert_eq!(b.bytes.len(), 96 * 2 * 136);
        }
        // y is pre-filled with the sentinel
        assert!(c.bufs[c.out]
            .bytes
            .chunks_exact(4)
            .all(|w| f32::from_le_bytes(w.try_into().unwrap()) == SENTINEL));
        // the pointer table holds the addresses the emulator will give the weight regions
        let mut mem = peacemaker_ir::emu::Memory::default();
        let mut bases = Vec::new();
        for b in &c.bufs {
            bases.push(
                mem.map_guarded(b.name.to_owned(), b.bytes.clone(), crate::emu::GUARD_BYTES)
                    .unwrap(),
            );
        }
        let table = &c.bufs[nw + 3].bytes;
        for e in 0..8 {
            let a = u64::from_le_bytes(table[8 * e..8 * e + 8].try_into().unwrap());
            assert_eq!(a, bases[e], "expert {e}");
        }
        // more than 16 experts: one concatenated region, entries stride by the expert size
        let big = node("emu,experts=20,top_k=2,tokens=20");
        let cb = big.emu_case(1).unwrap();
        assert_eq!(cb.bufs.len(), 6);
        assert_eq!(cb.bufs[0].name, "w");
        assert_eq!(cb.bufs[0].bytes.len(), 20 * 96 * 2 * 136);
        let mut mem = peacemaker_ir::emu::Memory::default();
        let b0 = mem
            .map_guarded(
                "w".into(),
                cb.bufs[0].bytes.clone(),
                crate::emu::GUARD_BYTES,
            )
            .unwrap();
        let t = &cb.bufs[4].bytes;
        for e in 0..20usize {
            let a = u64::from_le_bytes(t[8 * e..8 * e + 8].try_into().unwrap());
            assert_eq!(a, b0 + (e * 96 * 2 * 136) as u64);
        }
        // one-expert case: a single weight region and a one-entry table
        let sc = node("sg512,tokens=40,M=48,K=256").emu_case(1).unwrap();
        assert_eq!(sc.bufs.len(), 6);
        assert_eq!(sc.bufs[sc.out - 1].bytes.len(), 8);
    }

    #[test]
    fn bump_bases_follow_map_guarded() {
        let sizes = [1usize, 4096, 4097, 100, 0x10_0000];
        let mut mem = peacemaker_ir::emu::Memory::default();
        let want: Vec<u64> = sizes
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                mem.map_guarded(format!("r{i}"), vec![0u8; s], crate::emu::GUARD_BYTES)
                    .unwrap()
            })
            .collect();
        assert_eq!(bump_bases(&sizes, crate::emu::GUARD_BYTES), want);
    }

    #[test]
    fn kernarg_offsets_follow_the_incumbent_signature() {
        // 5 pointers (0..40) then 5 i32 (40..60): the order of the incumbent's declaration.
        let n = node("emu");
        let src = n.incumbent().unwrap().source;
        let at = src
            .find(&format!("void {SYMBOL}("))
            .expect("the kernel definition");
        let sig = &src[at + format!("void {SYMBOL}(").len()..];
        let sig = &sig[..sig.find(") {").unwrap()];
        let sig: String = sig
            .lines()
            .map(|l| l.split("//").next().unwrap())
            .collect::<Vec<_>>()
            .join(" ");
        let names: Vec<&str> = sig
            .split(',')
            .map(|p| p.trim().rsplit([' ', '*']).next().unwrap().trim())
            .filter(|s| !s.is_empty())
            .collect();
        // the parameters on a shared line (`int M, int K, ...`) split on commas as well
        let want = [
            "expert_weight_ptrs",
            "expert_tile_ids",
            "sorted_slot_index",
            "X_src",
            "Y_grouped",
            "M",
            "K",
            "x_row_div",
            "m_total",
            "n_experts",
        ];
        assert_eq!(names, want, "{sig}");
        let c = n.emu_case(1).unwrap();
        assert_eq!(c.args.len(), want.len());
        for (i, a) in c.args.iter().enumerate() {
            assert_eq!(matches!(a, EmuArg::I32(_)), i >= 5, "{i}");
        }
        // the gemm.rs launcher (:15959-15971) pushes ptr x5 then i32 x5, natural alignment
        assert_eq!(n.kernarg_layout(), expected_layout());
    }

    fn expected_layout() -> Vec<(u64, u64, String)> {
        let mut v: Vec<(u64, u64, String)> =
            (0..5).map(|i| (8 * i, 8, "pointer".to_owned())).collect();
        v.extend((0..5).map(|i| (40 + 4 * i, 4, "value".to_owned())));
        v
    }

    impl MoeK2 {
        /// Explicit kernarg layout `(offset, size, kind)` of the launch, natural alignment.
        fn kernarg_layout(&self) -> Vec<(u64, u64, String)> {
            let c = self.emu_case(1).unwrap();
            let mut off = 0u64;
            let mut v = Vec::new();
            for a in &c.args {
                let (size, kind) = match a {
                    EmuArg::I32(_) | EmuArg::U32(_) | EmuArg::F32(_) => (4, "value"),
                    EmuArg::U64(_) => (8, "value"),
                    _ => (8, "pointer"),
                };
                off = off.next_multiple_of(size);
                v.push((off, size, kind.to_owned()));
                off += size;
            }
            v
        }
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("hip_loop-moe-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn offline_compile_matches_the_metadata_and_a_copy_reproduces_the_text_sha() {
        if !crate::jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("offline");
        crate::jit::isolate_caches(&dir).unwrap();
        let n = node("emu");
        let inc = n.incumbent().unwrap();
        let a = crate::jit::compile(&dir, "a", &inc, &inc.source).unwrap();
        // an unmodified copy, with a candidate header line in front (comments do not change the object)
        let copy = format!("// hip_loop: rows_per_block=16\n{}", inc.source);
        let b = crate::jit::compile(&dir, "b", &inc, &copy).unwrap();
        let (ka, kb) = (a.kernel(SYMBOL).unwrap(), b.kernel(SYMBOL).unwrap());
        assert_eq!(ka.code_sha256, kb.code_sha256);
        assert_eq!(ka.code_sha256.len(), 64);
        // launch contract: block 32, the launcher's kernarg layout, no scratch
        assert_eq!(ka.max_flat_workgroup_size, 32);
        assert_eq!(ka.wavefront_size, 32);
        assert_eq!(ka.explicit_args, n.kernarg_layout());
        assert_eq!(ka.explicit_args, expected_layout());
        assert_eq!((ka.scratch_bytes, ka.vgpr_spill), (0, 0));
        assert!(kb.kernarg_mismatch(ka).is_none());
    }

    #[test]
    fn emu_gate_passes_the_incumbent_and_fails_closed_on_a_short_x() {
        if !crate::jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("emu");
        crate::jit::isolate_caches(&dir).unwrap();
        let n = node("emu");
        let inc = n.incumbent().unwrap();
        let built = crate::jit::compile(&dir, "baseline", &inc, &inc.source).unwrap();
        let geo = n.geometry();
        assert_eq!(geo.grid, [6, 13, 1]);
        let case = n.emu_case(1).unwrap();
        let (pass, rec) = crate::emu::gate(
            &inc.symbol,
            &mut case.clone(),
            (&built.image, &geo),
            (&built.image, &geo),
            n.shape_json(),
            1,
        );
        assert!(pass, "{rec}");
        assert_eq!(rec["identical_to_baseline"], true);
        assert_eq!(rec["workgroups"]["baseline"], 78u64);
        // a short x buffer: the last tile's gather lands in the guard
        let mut short = case.clone();
        let x = short.bufs.iter_mut().find(|b| b.name == "x").unwrap();
        x.bytes.truncate(x.bytes.len() - 4);
        let (pass, rec) = crate::emu::gate(
            &inc.symbol,
            &mut short,
            (&built.image, &geo),
            (&built.image, &geo),
            n.shape_json(),
            1,
        );
        assert!(!pass);
        assert!(
            rec["reason"]
                .as_str()
                .unwrap()
                .contains("unmapped memory access"),
            "{rec}"
        );
    }

    #[test]
    fn tile_run_owner_is_the_expert_local_run_leader() {
        let n = node("gu512");
        let r = n.routing();
        let h = Header::parse("// hip_loop: rows_per_block=64 tile_run=4").unwrap();
        let own = n.out_owner(&h).unwrap();
        // the grid is the product's y extent (leaders and the followers that return)
        assert_eq!(n.geometry_with(&mut h.clone()).unwrap().grid, [34, 312, 1]);
        let m = n.m;
        let mut leaders = std::collections::BTreeSet::new();
        for t in 0..r.live_tiles() {
            let e = r.tiles[t] as usize;
            let first = r.offsets[e] as usize / 16;
            let o = own(t * 16 * m + 3 * 64 + 5);
            assert_eq!(o[0], 3);
            let y = o[1] as usize;
            assert!(
                y <= t && t - y < 4 && r.tiles[y] as usize == e && (y - first) % 4 == 0,
                "tile {t}"
            );
            leaders.insert(y);
        }
        // a run of at most four tiles per leader: fewer leaders than tiles on a skewed table
        assert!(leaders.len() < r.live_tiles() && leaders.len() * 4 >= r.live_tiles());
        // dead tiles are still nobody's
        assert_eq!(own(r.live * m), PAD_OWNER);
        // one-expert tables: runs of four from tile 0
        let s = node("sg512");
        let o = s.out_owner(&h).unwrap();
        assert_eq!(o(5 * 16 * 1088)[1], 4);
        assert_eq!(o(11 * 16 * 1088)[1], 8);
        for bad in ["tile_run=0", "tile_run=17", "tile_run=4 slots_per_tile=32"] {
            let mut hh = Header::parse(&format!("// hip_loop: {bad}")).unwrap();
            assert!(n.geometry_with(&mut hh).is_err(), "{bad}");
        }
    }
}
