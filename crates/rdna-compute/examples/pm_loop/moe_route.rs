// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `--kernel qwen4_moe_route --arch gfx1100`: the ROUTE judge of the Flash-Next (FN) S1 prefill
//! routed-expert sweep on the XTX. Baseline = the shipped gfx1100 default, the F16 WMMA grouped
//! route; candidate = the symmetric-IU4 PM chain that `HIPFIRE_QWEN4_MOE_SYM_IU4=1` selects on
//! the same inputs. Both arms are enqueued through `rdna_compute::Gpu`'s public launchers with the
//! literal arguments of `hipfire-dispatch/src/pipeline/slowlink_prefill.rs::execute_panel`
//! (`WindowRoute::Aggregate` vs `WindowRoute::AggregateSymIu4`). There is NO identity gate (A4
//! activations differ from F16 by design): the record carries the numeric distance of the two
//! routes' outputs instead, plus each route against an independent CPU reference.
//!
//! Included by the driver with `#[path = "pm_loop/moe_route.rs"] mod moe_route;`. Only
//! `--arch gfx1100` is admitted (the F16 grouped route and the IU4 NT1 PM entries it compares are
//! the XTX's; the other arches default to IU4 already).
//!
//! # Shape grammar (`--shape`)
//!
//! `[fn,] key=value, ...` (`fn` = Flash-Next `H=2560, I=640, TOP=10`, the only geometry the F16
//! predicates `qwen4_moe_f16_wmma_grouped_applies` (m==1280, k==2560, mi%128==0) and the K3 down
//! producer (`I == 640`) admit):
//!
//! * `T` window rows (default 16896: one S1 window). `T >= 512` (`QWEN4_F16_WMMA_MIN_TOKENS`,
//!   `QWEN4_MOE_SYM_IU4_MIN_ROWS`, `gemm.rs`), `T * 10 <= 2^22` (`QWEN4_MOE_SYM_PM_MAX_SLOTS`).
//! * `E` experts (default 512, `<= 1024`), `D` distinct synthetic experts (default `E`; expert `e`
//!   aliases synthetic expert `e % D`, a memory cap), `route=uniform|skew` (as `moe.rs`/
//!   `moe_verify.rs`: TOP distinct experts per token; `skew` puts experts `0..5` on every token),
//!   `seed`.
//! * `panel` grouped rows per call (default 2048 = `PanelLimits::S1.max_grouped_rows`), `bind`
//!   experts per transport binding (default 32 = `PanelLimits::S1.experts_per_slot`; panels never
//!   cross a binding), `panels=all|N` (default `all`: every panel of the layer, the real sweep;
//!   `N`: the first N panels in schedule order, e.g. `panels=1` is the PF-1 panel form).
//!
//! # Knobs (`--key value`)
//!
//! `refslots` live grouped rows the CPU references compute (default 48, evenly spaced through the
//! selected panels), `stage-reps` event-bracketed chains per arm of the stage profile (default
//! 20), `f16-bound` / `a4-recipe-bound` / `a4-bound` correctness sanity bounds (defaults 0.02 /
//! 0.05 / 0.5, see "Correctness").
//!
//! # The two chains (production panel schedule, traced from source)
//!
//! One chain = the window's activation producer once, then every selected panel in order
//! (`execute_panel`, `slowlink_prefill.rs:1910-2082`; panel boundaries: `LayerRoutePlan::schedule`,
//! `:1437-1538`, S1 bounds `:128-133`: 2048 grouped rows, 32-expert bindings; at FN `T=16896`
//! that is ~96 panels per layer, PF-1's 4009 calls / 42 host layers). The window grouping
//! (`moe_scatter_stable_top10`) and the panel combine (`slowlink_combine_panel`) are common to both
//! routes and not part of the chain; the synthetic grouping is built on the host from the same
//! definition and checked against the device kernel in `setup`.
//!
//! | # | scope | F16 route (baseline, `WindowRoute::Aggregate`) | IU4 route (candidate, `AggregateSymIu4`) |
//! |---|---|---|---|
//! | 1 | window | `rotate_x_mq_batched_f16(x_norm, H, T)` (`mq_rotate_x_f16`) + D2D append into the window `x_f16` (`:1250-1263`) | `qwen4_moe_rotate256_i4(x_norm, H, T)` (`mq_rotate_x_i4`, hipcc; `:1575-1582`) |
//! | 2 | panel | `gemm_mq4g256v2_moe_grouped_top10_silu_bf16out` (`gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out`, `:2013-2024`) | `gemm_qwen4_moe_gate_up_silu_iu4_sym` (`qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1100`, `:1966-1978`) |
//! | 3 | panel | `slowlink_remap_sorted` + `moe_unscatter_rotate128_f16` (`:2025-2033`) | `qwen4_moe_rotate128_i4` (K3 `mq_i4_a4_r128_k640_gfx1100` of `mq_i4_a4_pm_gfx1100`, `:1979-1985`) |
//! | 4 | panel | `gemm_mq4g128v2_moe_grouped_top10_xf16` (`gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out`, `:2034-2044`) | `gemm_qwen4_moe_down_iu4_sym` (`qwen4_moe_down_iu4_sym_pm_gfx1100`, `:1986-1998`) |
//!
//! Grids (rocprof reports threads: workgroups x block on X): F16 gate_up `(2I/16, rows/16)` x 32 =
//! `(2560, 128, 1)` for a 2048-row panel, F16 down `(H/16, rows/16)` x 32 = `(5120, 128, 1)`
//! (`moe.rs::gemm_mq4g128v2_moe_grouped_wmma_gfx1151`, `gemm.rs::gemm_mq4g256v2_moe_grouped_wmma_k2_impl`:
//! PF-1's two top kernels); IU4 NT1 gate_up `(2I/64, rows/16)` x 64 = `(1280, 128, 1)`, down
//! `(H/64, rows/16)` x 128 = `(5120, 128, 1)` (`qwen4_moe_sym_nt1_gemm`, NT1 blocks 64/128).
//! `--offline` prints them per arm from these definitions.
//!
//! The IU4 route is forced through the public launchers (production gfx1100 default is OFF,
//! `default_requested: false`; `HIPFIRE_QWEN4_MOE_SYM_IU4=1` turns it on): `setup` refuses unless
//! the live selector picks the documented symbols and the F16 route applies on this device/env
//! (`qwen4_moe_f16_wmma_grouped_applies`, `HIPFIRE_QWEN4_F16_WMMA` not 0).
//!
//! # Record
//!
//! Schema 1 like the other chains, plus: `route` (grouping, panel schedule), `chain` (per arm
//! window/panel stages with launcher, symbol, origin hip|pm and the launch grid in workgroups and
//! threads, for a full panel), `baseline.pm_objects` / `candidate.pm_objects` (the shipped PM
//! bundles the candidate launches, M7-certified; offline too), `correctness` (the streaming
//! distance of the two routes over every live grouped row: `rel_l2` = ||cand - base|| / ||base||,
//! `max_abs`, `mean_abs`, non-finite counts; each arm against the CPU f64 chain and the IU4 arm
//! against the exact A4 recipe on `refslots` sampled slots), `stages` (event-bracketed per-stage
//! medians per arm: `x_producer`, `gate_up`, `down_producer`, `down`, with `calls`, `total_us`,
//! `mean_call_us`, `full_panel_median_us` and the cand/base ratios) and `timing` (ABBA over the
//! WHOLE chain like `qwen4_moe_verify`: ratio cand/base, sign test, crossover buffer sets; plus
//! `series.x_producer` / `series.panels`, the same ABBA samples split at the window producer's
//! event, so the window and per-panel ratios come from the same rounds).
//!
//! # Correctness (no identity gate)
//!
//! Streaming pass, both arms interleaved panel by panel (buffer set 0 baseline, set 1 candidate),
//! then repeated on the swapped sets and once more after the timing: each arm's outputs must be
//! bit-identical across passes (hazard / stale-buffer guard). Gates: every live output finite;
//! baseline vs the CPU f64 chain `rel_l2 <= f16-bound`; candidate vs the CPU A4 recipe
//! (`cpu_iu4_chain`'s arithmetic: FWHT256 + C2 `block_i4_128` quantiser, exact int32 fold,
//! portable SwiGLU, FWHT128 + quantiser, exact fold) `rel_l2 <= a4-recipe-bound`, and vs the f64
//! chain `rel_l2 <= a4-bound`. These are sanity bounds (a layout or table bug reads ~1.0), not
//! quality claims: F16 route ~1e-3 (F16 operands + BF16 stores), A4 vs f64 ~0.1-0.2 (two 4-bit
//! activation quantisations; the recipe is exact except SwiGLU roundings the portable twin cannot
//! decide). The distance between the routes is recorded, never gated. `--null-control aa` runs the
//! baseline chain as the candidate arm and then DOES require bit identity.
//!
//! # Pre-card emulator gate (always runs before a device is opened)
//!
//! X1-ROUTE faulted the XTX (`HipError(700)` in the prime chain): the shipped K3 producer stored 4 GiB
//! past its sidecar once `g * L + slot >= 828,505` (`mad64` aliasing, fixed in `mq_i4_a4_gfx1100.rs`),
//! which no small-shape gate can reach. GPU mode therefore ALWAYS runs `Bench::emu_gate` first (fail
//! closed: stage `emu_gate`, exit 3, before `Gpu::init`); `--emu` additionally requests it with
//! `--offline`. It runs the three PM launches of the IU4 chain (gate_up, K3, down) on the peacemaker
//! emulator with the chain's literal 64/48-byte kernargs and every buffer an exact-size guarded region
//! (any access past an end is an error): a small shape in full (every workgroup of every panel) and the
//! REAL shape's panels at quartiles plus the panels holding the smallest and largest slot ids, at the grid
//! extremes (first/last/padded tiles and the tiles with the extreme slot ids, all x blocks, all rows of
//! those tiles for K3). Each live row's down output must equal the exact A4 recipe. Not emulated: the
//! hipcc window producer `mq_rotate_x_i4` (the sidecar is the host recipe) and the whole F16 arm (hipcc).
#![allow(dead_code)]

use crate::common::{bf16_bits, bf16_val, dev_err, f16_bits, f16_val, sha256_hex};
use crate::moe::{fold_slots, portable_silu_mul, rt_f32, XRow};
use crate::moe_verify_iu4::Iu4Arm;
use crate::rotate::{quantize, rotate, GROUP, REC};
use crate::workload::{Arm, Knobs, Result};
use crate::{Opts, Outcome, Sampled};
use peacemaker_isa::kernels::qwen4_moe_sym::Kind as SymKind;
use rayon::prelude::*;
use rdna_compute::scratch::Int4MmqPrepared;
use rdna_compute::{gen_fwht_signs, DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// `--kernel` name.
pub const KERNEL: &str = "qwen4_moe_route";
/// The only arch of this judge.
pub const ARCH: &str = "gfx1100";

const H: usize = 2560;
const I: usize = 640;
const TOP: usize = 10;
const DEFAULT_E: usize = 512;
const MAX_E: usize = 1024;
const DEFAULT_T: usize = 16896;
/// `QWEN4_F16_WMMA_MIN_TOKENS` == `QWEN4_MOE_SYM_IU4_MIN_ROWS` (`gemm.rs`).
const MIN_T: usize = 512;
/// `QWEN4_MOE_SYM_PM_MAX_SLOTS`.
const MAX_SLOTS: usize = 1 << 22;
/// `MOE_GROUPED_BLOCK_M`.
const BLOCK_M: usize = 16;
/// `PanelLimits::S1`.
const S1_PANEL_ROWS: usize = 2048;
const S1_BINDING_EXPERTS: usize = 32;
const DEFAULT_SEED: u64 = 0x5EED_0044;
const DEFAULT_REFSLOTS: usize = 48;
const DEFAULT_STAGE_REPS: usize = 20;
const DEFAULT_F16_BOUND: f64 = 0.02;
const DEFAULT_A4_RECIPE_BOUND: f64 = 0.05;
const DEFAULT_A4_BOUND: f64 = 0.5;
/// `QWEN4_MOE_SYM_NT1_BLOCK` (gate/up, down): the gfx1100 NT1 entries' workgroup sizes.
const IU4_NT1_BLOCK: [u32; 2] = [64, 128];
/// Rows of slack past every scratch / route tensor (multiple of `BLOCK_M`); see `setup`.
const SLACK_ROWS: usize = 64;
const _: () = assert!(SLACK_ROWS % BLOCK_M == 0);

const GATE_F16: &str = "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out";
const DOWN_F16: &str = "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out";
const GATE_IU4: &str = "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1100";
const DOWN_IU4: &str = "qwen4_moe_down_iu4_sym_pm_gfx1100";

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn hip<T, E2: std::fmt::Display>(what: &str, r: std::result::Result<T, E2>) -> Result<T> {
    r.map_err(|e| format!("{what}: {}", dev_err(e)))
}

// ---------------------------------------------------------------------------
// Deterministic generators (splitmix64, as moe.rs / moe_verify.rs)

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Uniform in `[0, 1)`.
    fn unit01(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f32 {
        let u = |r: &mut Self| ((r.next() >> 11) + 1) as f64 / (1u64 << 53) as f64;
        let (a, b) = (u(self), u(self));
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    }
}

/// QT44 gate/up bytes per output row (`H/256` groups of 136 B).
const GU_ROW_BYTES: usize = H / 256 * 136;
const GU_BYTES: usize = 2 * I * GU_ROW_BYTES;
/// MQ4G128 down bytes per output row (`I/128` groups of 68 B).
const DN_ROW_BYTES: usize = I / 128 * 68;
const DN_BYTES: usize = H * DN_ROW_BYTES;

fn put_header(group: &mut [u8], off: usize, scale: f32) {
    let sc = f16_bits(scale);
    // Exact: scaling by 8 only moves the exponent of a normal f16.
    let zp = f16_bits(-8.0 * f16_val(sc));
    group[off..off + 2].copy_from_slice(&sc.to_le_bytes());
    group[off + 2..off + 4].copy_from_slice(&zp.to_le_bytes());
}

fn fill_payload(r: &mut Mix, payload: &mut [u8]) {
    for chunk in payload.chunks_mut(8) {
        let w = r.next().to_le_bytes();
        chunk.copy_from_slice(&w[..chunk.len()]);
    }
}

/// Synthetic expert `e`'s symmetric weights: QT44 gate/up groups (136 B: `[f16 sc0, zp0, sc1, zp1]`
/// + 128 nibble bytes, weights 0..127 grid 0, 128..255 grid 1) and MQ4G128 down groups (68 B:
/// `[f16 sc, zp]` + 64 nibble bytes). Every header has `zp == -8*sc` and a finite positive `sc`
/// (what both routes' load gates and `qwen4_moe_sym_check` accept).
fn gen_expert(seed: u64, e: usize, gu: &mut [u8], dn: &mut [u8]) {
    debug_assert_eq!(gu.len(), GU_BYTES);
    debug_assert_eq!(dn.len(), DN_BYTES);
    let key = seed ^ (e as u64 + 1).wrapping_mul(GOLDEN);
    let mut r = Mix(key ^ 0x6A09_E667_F3BC_C908);
    for group in gu.chunks_exact_mut(136) {
        let s0 = 0.002 + 0.004 * r.unit01();
        let s1 = 0.002 + 0.004 * r.unit01();
        put_header(group, 0, s0);
        put_header(group, 4, s1);
        fill_payload(&mut r, &mut group[8..]);
    }
    let mut r = Mix(key ^ 0xBB67_AE85_84CA_A73B);
    for group in dn.chunks_exact_mut(68) {
        let s = 0.006 + 0.006 * r.unit01();
        put_header(group, 0, s);
        fill_payload(&mut r, &mut group[4..]);
    }
}

/// Host-side symmetric-header check (the device gate is `qwen4_moe_sym_check` in `setup`).
fn check_symmetric(gu: &[u8], dn: &[u8]) -> std::result::Result<(), String> {
    let header = |g: &[u8], off: usize| -> std::result::Result<(), String> {
        let sc = f16_val(u16::from_le_bytes([g[off], g[off + 1]]));
        let zp = f16_val(u16::from_le_bytes([g[off + 2], g[off + 3]]));
        if !sc.is_finite() || !zp.is_finite() || zp != -8.0 * sc {
            return Err(format!("asymmetric or non-finite header sc={sc} zp={zp}"));
        }
        Ok(())
    };
    for g in gu.chunks_exact(136) {
        header(g, 0)?;
        header(g, 4)?;
    }
    for g in dn.chunks_exact(68) {
        header(g, 0)?;
    }
    Ok(())
}

/// Row `token` of the synthetic F32 activations `[T, H]` (the routed block's normed input).
fn x_row(seed: u64, token: usize, out: &mut [f32]) {
    let mut r = Mix(seed ^ 0x1111_2222_3333_4444 ^ (token as u64 + 1).wrapping_mul(GOLDEN));
    for v in out.iter_mut() {
        *v = r.normal();
    }
}

// ---------------------------------------------------------------------------
// Shape

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteKind {
    Uniform,
    Skew,
}

impl RouteKind {
    fn name(self) -> &'static str {
        match self {
            RouteKind::Uniform => "uniform",
            RouteKind::Skew => "skew",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelSel {
    All,
    First(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub t: usize,
    pub e: usize,
    pub distinct: usize,
    pub panel_rows: usize,
    pub bind: usize,
    pub panels: PanelSel,
    pub route: RouteKind,
    pub seed: u64,
}

impl Shape {
    pub fn slots(&self) -> usize {
        self.t * TOP
    }
}

fn parse_usize(key: &str, v: &str) -> Result<usize> {
    v.parse::<usize>()
        .map_err(|_| format!("--shape {key}={v}: not a non-negative integer"))
}

pub fn parse_shape(s: &str) -> Result<Shape> {
    let mut sh = Shape {
        t: DEFAULT_T,
        e: DEFAULT_E,
        distinct: 0,
        panel_rows: S1_PANEL_ROWS,
        bind: S1_BINDING_EXPERTS,
        panels: PanelSel::All,
        route: RouteKind::Uniform,
        seed: DEFAULT_SEED,
    };
    let mut distinct = None;
    let mut seen: BTreeSet<&'static str> = BTreeSet::new();
    let mut tokens = s.split(',').map(str::trim).filter(|t| !t.is_empty());
    let mut first = true;
    while let Some(token) = tokens.next() {
        if first {
            first = false;
            if token == "fn" {
                continue;
            }
            if !token.contains('=') {
                return Err(format!(
                    "--shape preset `{token}`: {KERNEL} admits only the `fn` (Flash-Next \
                     H=2560/I=640/TOP=10) preset"
                ));
            }
        }
        let (k, v) = token
            .split_once('=')
            .ok_or_else(|| format!("--shape item `{token}`: expected key=value"))?;
        let (k, v) = (k.trim(), v.trim());
        let canonical: &'static str = match k {
            "T" | "t" => "T",
            "E" => "E",
            "D" => "D",
            "panel" => "panel",
            "bind" => "bind",
            "panels" => "panels",
            "route" => "route",
            "seed" => "seed",
            other => {
                return Err(format!(
                    "--shape key `{other}`: expected T, E, D, panel, bind, panels, route, seed"
                ))
            }
        };
        if !seen.insert(canonical) {
            return Err(format!("--shape key `{canonical}` given twice"));
        }
        match canonical {
            "T" => sh.t = parse_usize("T", v)?,
            "E" => sh.e = parse_usize("E", v)?,
            "D" => distinct = Some(parse_usize("D", v)?),
            "panel" => sh.panel_rows = parse_usize("panel", v)?,
            "bind" => sh.bind = parse_usize("bind", v)?,
            "panels" => {
                sh.panels = if v == "all" {
                    PanelSel::All
                } else {
                    PanelSel::First(parse_usize("panels", v)?)
                }
            }
            "route" => {
                sh.route = match v {
                    "uniform" => RouteKind::Uniform,
                    "skew" => RouteKind::Skew,
                    other => return Err(format!("--shape route={other}: expected uniform|skew")),
                }
            }
            _ => {
                sh.seed = v
                    .parse()
                    .map_err(|_| format!("--shape seed={v}: not a u64"))?
            }
        }
    }
    sh.distinct = distinct.unwrap_or(sh.e);
    validate_shape(&sh)?;
    Ok(sh)
}

fn validate_shape(sh: &Shape) -> Result<()> {
    if sh.t < MIN_T {
        return Err(format!(
            "--shape T={}: both routes need T >= {MIN_T} rows (QWEN4_F16_WMMA_MIN_TOKENS for the F16 \
             grouped route, QWEN4_MOE_SYM_IU4_MIN_ROWS for the IU4 route; gemm.rs)",
            sh.t
        ));
    }
    if sh.slots() > MAX_SLOTS {
        return Err(format!(
            "--shape T={}: {} routed slots exceed {MAX_SLOTS} (QWEN4_MOE_SYM_PM_MAX_SLOTS)",
            sh.t,
            sh.slots()
        ));
    }
    if sh.e < TOP || sh.e > MAX_E {
        return Err(format!(
            "--shape E={}: need {TOP} <= E <= {MAX_E} (top-{TOP} routing, scatter bound)",
            sh.e
        ));
    }
    if sh.distinct == 0 || sh.distinct > sh.e {
        return Err(format!(
            "--shape D={}: must be in 1..=E ({})",
            sh.distinct, sh.e
        ));
    }
    if sh.panel_rows < BLOCK_M || sh.panel_rows % BLOCK_M != 0 || sh.panel_rows > 1 << 16 {
        return Err(format!(
            "--shape panel={}: a multiple of {BLOCK_M} in {BLOCK_M}..=65536",
            sh.panel_rows
        ));
    }
    if sh.bind == 0 {
        return Err("--shape bind=0: a binding holds at least one expert".into());
    }
    if sh.panels == PanelSel::First(0) {
        return Err("--shape panels=0: run `all` or at least one panel".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The synthetic route and its grouping (pure host, the production definitions)

/// Flat slot (`token * TOP + j`) -> expert id: TOP distinct experts per token.
pub fn route_table(sh: &Shape) -> Vec<i32> {
    let mut rt = Mix(sh.seed ^ 0x0ABC_DEF0_1234_5678);
    let hot = if sh.route == RouteKind::Skew {
        TOP.min(5)
    } else {
        0
    };
    let mut topk = Vec::with_capacity(sh.slots());
    for _ in 0..sh.t {
        let mut row: Vec<i32> = (0..hot as i32).collect();
        while row.len() < TOP {
            let e = (hot as u64 + rt.below((sh.e - hot) as u64)) as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        topk.extend(row);
    }
    topk
}

/// Production's grouped row bound (`program.rs::grouped_m_total_bound`, block 16).
pub fn grouped_bound(slots: usize, experts: usize) -> usize {
    (slots + (BLOCK_M - 1) * slots.min(experts)).div_ceil(BLOCK_M) * BLOCK_M
}

/// The stable `(expert, flat slot)` pad-16 grouping (`moe_scatter_stable_top10`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grouping {
    pub experts: usize,
    pub slots: usize,
    /// `grouped_bound(slots, experts)`: length of `sorted`.
    pub bound: usize,
    pub padded: Vec<usize>,
    /// `[experts + 1]` grouped-row start of every expert.
    pub offsets: Vec<usize>,
    /// Live + pad rows (`offsets[experts]`).
    pub total_rows: usize,
    /// `[bound]` flat slot of every grouped row, `-1` padding and tail.
    pub sorted: Vec<i32>,
    /// `[bound / 16]` expert of every tile, `-1` tail.
    pub tiles: Vec<i32>,
}

impl Grouping {
    pub fn new(topk: &[i32], experts: usize) -> Result<Grouping> {
        let slots = topk.len();
        let mut counts = vec![0usize; experts];
        for &e in topk {
            *counts
                .get_mut(usize::try_from(e).map_err(|_| format!("negative expert id {e}"))?)
                .ok_or_else(|| format!("expert id {e} outside [0, {experts})"))? += 1;
        }
        let padded: Vec<usize> = counts
            .iter()
            .map(|c| c.div_ceil(BLOCK_M) * BLOCK_M)
            .collect();
        let mut offsets = Vec::with_capacity(experts + 1);
        offsets.push(0usize);
        for p in &padded {
            offsets.push(offsets.last().unwrap() + p);
        }
        let total_rows = *offsets.last().unwrap();
        let bound = grouped_bound(slots, experts);
        if total_rows > bound {
            return Err(format!(
                "grouping spans {total_rows} rows, above the production bound {bound}"
            ));
        }
        let mut sorted = vec![-1i32; bound];
        let mut cursor = vec![0usize; experts];
        for (f, &e) in topk.iter().enumerate() {
            let e = e as usize;
            sorted[offsets[e] + cursor[e]] = f as i32;
            cursor[e] += 1;
        }
        let mut tiles = vec![-1i32; bound / BLOCK_M];
        for e in 0..experts {
            for t in offsets[e] / BLOCK_M..offsets[e + 1] / BLOCK_M {
                tiles[t] = e as i32;
            }
        }
        Ok(Grouping {
            experts,
            slots,
            bound,
            padded,
            offsets,
            total_rows,
            sorted,
            tiles,
        })
    }

    /// Expert holding grouped row `row` (`LayerRoutePlan::expert_of_row`).
    pub fn expert_of_row(&self, row: usize) -> usize {
        (self.offsets.partition_point(|&s| s <= row) - 1).min(self.experts - 1)
    }
}

/// One bounded panel of grouped rows of one binding (`PanelRange`, rows > 0 only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Panel {
    pub index: usize,
    pub binding: usize,
    pub row_base: usize,
    pub rows: usize,
    pub expert_lo: usize,
    pub expert_hi: usize,
}

/// `LayerRoutePlan::schedule` over consecutive `bind`-expert bindings (the K1/ownership parts of
/// the production plan do not touch the K2 GEMMs and are not modelled): rows never cross a
/// binding, a binding emits `ceil(rows / panel_rows)` panels, a binding without rows none.
pub fn schedule(g: &Grouping, bind: usize, panel_rows: usize) -> Vec<Panel> {
    let mut panels: Vec<Panel> = Vec::new();
    let (mut lo, mut binding) = (0usize, 0usize);
    while lo < g.experts {
        let hi = (lo + bind).min(g.experts);
        let (start, end) = (g.offsets[lo], g.offsets[hi]);
        let mut base = start;
        while base < end {
            let rows = (end - base).min(panel_rows);
            panels.push(Panel {
                index: panels.len(),
                binding,
                row_base: base,
                rows,
                expert_lo: g.expert_of_row(base),
                expert_hi: g.expert_of_row(base + rows - 1) + 1,
            });
            base += rows;
        }
        lo = hi;
        binding += 1;
    }
    panels
}

// ---------------------------------------------------------------------------
// Chain description (pure)

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    F16,
    Iu4,
}

impl Route {
    fn name(self) -> &'static str {
        match self {
            Route::F16 => "f16_wmma_grouped",
            Route::Iu4 => "iu4_sym_pm",
        }
    }
}

/// A stage boundary the chain reports to a hook after the stage's launch(es).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    XProduce,
    XAppend,
    /// Marker after the last window stage (no launch).
    WindowEnd,
    GateUp,
    Remap,
    Unscatter,
    DownProduce,
    Down,
}

impl Kind {
    fn group(self) -> &'static str {
        match self {
            Kind::XProduce | Kind::XAppend | Kind::WindowEnd => "x_producer",
            Kind::GateUp => "gate_up",
            Kind::Remap | Kind::Unscatter | Kind::DownProduce => "down_producer",
            Kind::Down => "down",
        }
    }
    fn name(self) -> &'static str {
        match self {
            Kind::XProduce => "x_produce",
            Kind::XAppend => "x_append_copy",
            Kind::WindowEnd => "window_end",
            Kind::GateUp => "gate_up",
            Kind::Remap => "remap_sorted",
            Kind::Unscatter => "unscatter_rotate128_f16",
            Kind::DownProduce => "rotate128_i4",
            Kind::Down => "down",
        }
    }
}

const GROUPS: [&str; 4] = ["x_producer", "gate_up", "down_producer", "down"];

/// Ordered launches of one route: `(kind, window?)`.
pub fn stage_kinds(route: Route) -> Vec<(Kind, bool)> {
    match route {
        Route::F16 => vec![
            (Kind::XProduce, true),
            (Kind::XAppend, true),
            (Kind::GateUp, false),
            (Kind::Remap, false),
            (Kind::Unscatter, false),
            (Kind::Down, false),
        ],
        Route::Iu4 => vec![
            (Kind::XProduce, true),
            (Kind::GateUp, false),
            (Kind::DownProduce, false),
            (Kind::Down, false),
        ],
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// Launch grid in workgroups.
    pub blocks: [u32; 3],
    /// Workgroup threads (X).
    pub block: u32,
}

impl Geometry {
    /// The grid as rocprof reports it: X in threads.
    pub fn threads(&self) -> [u64; 3] {
        [
            u64::from(self.blocks[0]) * u64::from(self.block),
            u64::from(self.blocks[1]),
            u64::from(self.blocks[2]),
        ]
    }
}

/// Launch geometry of one stage: `rows` is the window rows `T` for window stages, the panel's
/// grouped rows otherwise. `None` for the D2D copy (no kernel).
pub fn geometry(route: Route, kind: Kind, rows: usize) -> Option<Geometry> {
    let g = |blocks: [usize; 3], block: u32| {
        Some(Geometry {
            blocks: [blocks[0] as u32, blocks[1] as u32, blocks[2] as u32],
            block,
        })
    };
    match (route, kind) {
        (_, Kind::XAppend) | (_, Kind::WindowEnd) => None,
        // rotate_x_mq_batched_f16 / rotate_x_mq_i4_batched: one wave per 256-group.
        (_, Kind::XProduce) => g([(H / 256) * rows, 1, 1], 32),
        (Route::F16, Kind::GateUp) => g([2 * I / 16, rows / BLOCK_M, 1], 32),
        (Route::F16, Kind::Remap) => g([rows.div_ceil(256), 1, 1], 256),
        (Route::F16, Kind::Unscatter) => g([rows, I / 128, 1], 32),
        (Route::F16, Kind::Down) => g([H / 16, rows / BLOCK_M, 1], 32),
        (Route::Iu4, Kind::GateUp) => g([2 * I / 64, rows / BLOCK_M, 1], IU4_NT1_BLOCK[0]),
        (Route::Iu4, Kind::DownProduce) => g([(I / 128) * rows, 1, 1], 32),
        (Route::Iu4, Kind::Down) => g([H / 64, rows / BLOCK_M, 1], IU4_NT1_BLOCK[1]),
        _ => None,
    }
}

/// `(launcher, kernel symbol, origin)` of one stage.
fn stage_symbol(route: Route, kind: Kind) -> (&'static str, &'static str, &'static str) {
    match (route, kind) {
        (Route::F16, Kind::XProduce) => ("Gpu::rotate_x_mq_batched_f16", "mq_rotate_x_f16", "hip"),
        (Route::F16, Kind::XAppend) => (
            "Gpu::memcpy_dtod_at_ordered_async",
            "hipMemcpyDtoDAsync (__amd_rocclr_copyBuffer)",
            "runtime",
        ),
        (Route::F16, Kind::GateUp) => (
            "Gpu::gemm_mq4g256v2_moe_grouped_top10_silu_bf16out",
            GATE_F16,
            "hip",
        ),
        (Route::F16, Kind::Remap) => ("Gpu::slowlink_remap_sorted", "slowlink_remap_sorted", "hip"),
        (Route::F16, Kind::Unscatter) => (
            "Gpu::moe_unscatter_rotate128_f16",
            "moe_unscatter_rotate128_f16",
            "hip",
        ),
        (Route::F16, Kind::Down) => (
            "Gpu::gemm_mq4g128v2_moe_grouped_top10_xf16",
            DOWN_F16,
            "hip",
        ),
        (Route::Iu4, Kind::XProduce) => ("Gpu::qwen4_moe_rotate256_i4", "mq_rotate_x_i4", "hip"),
        (Route::Iu4, Kind::GateUp) => ("Gpu::gemm_qwen4_moe_gate_up_silu_iu4_sym", GATE_IU4, "pm"),
        (Route::Iu4, Kind::DownProduce) => (
            "Gpu::qwen4_moe_rotate128_i4",
            rdna_compute::slab_moe::MQ_I4_A4_GFX1100_SYMBOL,
            "pm",
        ),
        (Route::Iu4, Kind::Down) => ("Gpu::gemm_qwen4_moe_down_iu4_sym", DOWN_IU4, "pm"),
        _ => ("", "", ""),
    }
}

fn stage_json(route: Route, kind: Kind, rows: usize, window: bool, sh: &Shape) -> Value {
    let (launcher, symbol, origin) = stage_symbol(route, kind);
    let mut v = json!({
        "stage": kind.name(), "group": kind.group(),
        "scope": if window { "window" } else { "panel" },
        "launcher": launcher, "symbols": [symbol], "origin": origin, "rows": rows,
    });
    match geometry(route, kind, rows) {
        Some(g) => {
            v["grid_workgroups"] = json!(g.blocks);
            v["block"] = json!(g.block);
            v["grid_threads"] = json!(g.threads());
        }
        None => {
            v["bytes"] = json!(sh.t * H * 2);
        }
    }
    v
}

/// The planned launches of one route: the window stages once (`rows = T`) and the panel stages for
/// a panel of `panel_rows` grouped rows (the grids of the full-size panel).
pub fn chain_plan(route: Route, sh: &Shape, panel_rows: usize, n_panels: usize) -> Value {
    let mut window = Vec::new();
    let mut panel = Vec::new();
    for (kind, w) in stage_kinds(route) {
        if w {
            window.push(stage_json(route, kind, sh.t, true, sh));
        } else {
            panel.push(stage_json(route, kind, panel_rows, false, sh));
        }
    }
    json!({
        "route": route.name(),
        "window_stages": window, "panel_stages": panel,
        "calls": {"window": 1, "panel": n_panels},
        "launches_per_chain": window.len() + panel.len() * n_panels,
    })
}

// ---------------------------------------------------------------------------
// The workload

#[derive(Clone, Copy, Debug)]
struct Tune {
    refslots: usize,
    stage_reps: usize,
    f16_bound: f64,
    a4_recipe_bound: f64,
    a4_bound: f64,
}

fn take_f64(knobs: &mut Knobs, key: &str, default: f64) -> Result<f64> {
    match knobs.take(key) {
        None => Ok(default),
        Some(v) => v
            .parse::<f64>()
            .ok()
            .filter(|x| x.is_finite() && *x > 0.0)
            .ok_or_else(|| format!("--{key} {v}: expected a bound > 0")),
    }
}

fn take_count(knobs: &mut Knobs, key: &str, default: usize) -> Result<usize> {
    match knobs.take(key) {
        None => Ok(default),
        Some(v) => v
            .parse::<usize>()
            .ok()
            .filter(|x| *x > 0)
            .ok_or_else(|| format!("--{key} {v}: expected a positive integer")),
    }
}

struct PanelDev {
    index: usize,
    rows: usize,
    row_base: usize,
    tiles: GpuTensor,
    sorted: GpuTensor,
}

/// One arm's own scratch (two sets; the crossover binding swaps them every ABBA round).
struct ArmSet {
    x_f16: GpuTensor,
    local: GpuTensor,
    y_gu: GpuTensor,
    y_down: GpuTensor,
}

struct Dev {
    x_norm: GpuTensor,
    gate_up_ptrs: GpuTensor,
    down_ptrs: GpuTensor,
    panels: Vec<PanelDev>,
    sets: [ArmSet; 2],
    /// Expert arenas (never freed; `GpuTensor` has no freeing `Drop`).
    _weights: Vec<GpuTensor>,
}

enum Window {
    F16,
    Iu4(Int4MmqPrepared),
}

type Hook<'a> = dyn FnMut(&mut Gpu, Option<usize>, Kind) -> Result<()> + 'a;

pub struct Bench {
    shape: Shape,
    aa: bool,
    tune: Tune,
    topk: Vec<i32>,
    grouping: Grouping,
    panels: Vec<Panel>,
    /// How many of `panels` the chain runs.
    run: usize,
    dev: Option<Dev>,
    swapped: bool,
}

/// GPU-free constructor: shape grammar, knob parsing, the synthetic route, its grouping and the
/// panel schedule.
pub fn new(shape: &str, knobs: &mut Knobs, aa: bool) -> Result<Bench> {
    let shape = parse_shape(shape)?;
    let tune = Tune {
        refslots: take_count(knobs, "refslots", DEFAULT_REFSLOTS)?,
        stage_reps: take_count(knobs, "stage-reps", DEFAULT_STAGE_REPS)?,
        f16_bound: take_f64(knobs, "f16-bound", DEFAULT_F16_BOUND)?,
        a4_recipe_bound: take_f64(knobs, "a4-recipe-bound", DEFAULT_A4_RECIPE_BOUND)?,
        a4_bound: take_f64(knobs, "a4-bound", DEFAULT_A4_BOUND)?,
    };
    Bench::from_parts(shape, aa, tune)
}

impl Bench {
    /// The route, grouping and panel schedule of an already validated (or, for the emulator
    /// gate's small shapes, deliberately unadmitted) `shape`.
    fn from_parts(shape: Shape, aa: bool, tune: Tune) -> Result<Bench> {
        if shape.e < TOP
            || shape.bind == 0
            || shape.panel_rows == 0
            || shape.panel_rows % BLOCK_M != 0
        {
            // route_table draws TOP distinct experts per token: E < TOP would never terminate.
            return Err(format!("shape E={} bind={} panel={}: need E >= {TOP}, bind > 0, panel a multiple of {BLOCK_M}",
                shape.e, shape.bind, shape.panel_rows));
        }
        let topk = route_table(&shape);
        let grouping = Grouping::new(&topk, shape.e)?;
        let panels = schedule(&grouping, shape.bind, shape.panel_rows);
        let run = match shape.panels {
            PanelSel::All => panels.len(),
            PanelSel::First(n) => n.min(panels.len()),
        };
        if run == 0 {
            return Err("the synthetic route produced no panel".into());
        }
        Ok(Bench {
            shape,
            aa,
            tune,
            topk,
            grouping,
            panels,
            run,
            dev: None,
            swapped: false,
        })
    }
}

impl Bench {
    fn selected(&self) -> &[Panel] {
        &self.panels[..self.run]
    }

    fn arm_route(&self, arm: Arm) -> Route {
        if arm == Arm::Baseline || self.aa {
            Route::F16
        } else {
            Route::Iu4
        }
    }

    /// Buffer set an arm runs on under the current crossover binding.
    fn set_index(&self, arm: Arm) -> usize {
        (arm == Arm::Candidate) as usize ^ self.swapped as usize
    }

    fn max_rows(&self) -> usize {
        self.selected().iter().map(|p| p.rows).max().unwrap_or(0)
    }

    pub fn shape_json(&self) -> Value {
        let sh = &self.shape;
        let sel = self.selected();
        let rows_run: usize = sel.iter().map(|p| p.rows).sum();
        let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
        for p in &self.panels {
            *hist.entry(p.rows).or_default() += 1;
        }
        let live = self.grouping.sorted.iter().filter(|s| **s >= 0).count();
        json!({
            "preset": "fn", "T": sh.t, "E": sh.e, "D": sh.distinct, "TOP": TOP, "H": H, "I": I,
            "route": sh.route.name(), "seed": sh.seed,
            "slots": sh.slots(), "live_rows": live,
            "grouped_rows": self.grouping.total_rows, "grouped_rows_bound": self.grouping.bound,
            "panel_rows": sh.panel_rows, "bind": sh.bind,
            "panels": match sh.panels { PanelSel::All => json!("all"), PanelSel::First(n) => json!(n) },
            "schedule": {
                "panels_total": self.panels.len(), "panels_run": sel.len(),
                "rows_run": rows_run,
                "rows_histogram": hist.iter().map(|(r, n)| json!({"rows": r, "panels": n})).collect::<Vec<_>>(),
                "first": sel.first().map(|p| json!({"row_base": p.row_base, "rows": p.rows,
                    "expert_lo": p.expert_lo, "expert_hi": p.expert_hi, "binding": p.binding})),
                "source": "hipfire-dispatch slowlink_prefill.rs LayerRoutePlan::schedule (:1437-1538), PanelLimits::S1 (:128-133)",
            },
        })
    }

    pub fn numeric_contract(&self) -> Value {
        json!({
            "relation": "numeric",
            "baseline": "F16 grouped WMMA: X rotated (FWHT256) and rounded to F16, F16 dequantised weights, \
                         f32 accumulate, BF16 gate/up/SwiGLU rows (silu fused in the epilogue), down X = BF16 \
                         rows FWHT128 rotated to F16, BF16 down rows",
            "candidate": "sym IU4 (A4): X = FWHT256 then per-128 `block_i4_128` (C2 candidate set), exact int32 \
                          fold per K128 epoch, BF16 SwiGLU rows, down X = FWHT128 + the same A4 quantiser, \
                          BF16 down rows",
            "output": "BF16 down rows of every live grouped row (the panel combine, common to both routes, is \
                       not part of the chain)",
            "identity_gate": "none: A4 vs F16 differ by design; the distance is recorded in correctness.distance. \
                              --null-control aa does require bit identity",
        })
    }

    /// Offline plan: the pure description of both chains and the schedule (no device).
    pub fn plan_json(&self) -> Value {
        let sel = self.selected();
        let full = self.shape.panel_rows.min(self.max_rows());
        let cand_route = self.arm_route(Arm::Candidate);
        json!({
            "mode": "eager",
            "note": "one chain = the window X producer once + every selected panel in order; the window grouping \
                     and the panel combine are common to both routes and not part of the chain",
            "baseline": chain_plan(Route::F16, &self.shape, full, sel.len()),
            "candidate": chain_plan(cand_route, &self.shape, full, sel.len()),
            "forcing": {
                "baseline": "shipped gfx1100 default (HIPFIRE_QWEN4_MOE_SYM_IU4 unset = off; \
                             WindowRoute::Aggregate); requires qwen4_moe_f16_wmma_grouped_applies",
                "candidate": if self.aa { "A/A: the baseline chain on the candidate's own buffer set" } else {
                    "HIPFIRE_QWEN4_MOE_SYM_IU4=1 route (WindowRoute::AggregateSymIu4) forced through the public \
                     launchers; setup refuses unless the live selector picks the documented symbols" },
            },
        })
    }

    /// The embedded PM bundles the candidate launches (GPU-free).
    pub fn pm_objects(&self, arm: Arm) -> Result<Vec<crate::workload::Object>> {
        if arm == Arm::Baseline || self.aa {
            return Ok(Vec::new());
        }
        Iu4Arm::new(ARCH)?.pm_objects()
    }

    // -- device -------------------------------------------------------------

    fn dev(&self) -> Result<&Dev> {
        self.dev
            .as_ref()
            .ok_or_else(|| format!("{KERNEL}: setup has not run"))
    }

    /// Live-selector checks, weights, route tensors, scratch and a primed chain; everything that can
    /// fail does so here, before any timing.
    pub fn setup(&mut self, gpu: &mut Gpu) -> Result<()> {
        let sh = self.shape.clone();
        let (t, slots) = (sh.t, sh.slots());
        if gpu.arch != ARCH {
            return Err(format!("device is {}, {KERNEL} needs {ARCH}", gpu.arch));
        }
        if gpu.route_rows(t) != t {
            return Err(
                "the prefill-invariant scope is active (route_rows != rows): the shipped route differs".into(),
            );
        }
        if !gpu.qwen4_moe_f16_wmma_grouped_applies(t, I, H) {
            return Err(format!(
                "the F16 WMMA grouped route does not apply at T={t} on this device/env \
                 (qwen4_moe_f16_wmma_grouped_applies; HIPFIRE_QWEN4_F16_WMMA=0?): there is no baseline"
            ));
        }
        if !self.aa {
            if gpu.qwen4_moe_sym_kernel().is_none() {
                return Err(
                    "no symmetric-IU4 MoE descriptor on this device (MoeSymIu4Kernel needs IU4 WMMA and \
                     the K2 inventory row, gemm.rs)"
                        .into(),
                );
            }
            if !gpu.iu4_a4_c2_candidates_exact() {
                return Err(
                    "the A4 producers and K2 kernels need exactly -DIU4_A4_CANDIDATES=2 \
                     (sym_iu4_request refuses otherwise; gfx1100 appends it unless \
                     HIPFIRE_GFX11_A4_CANDIDATES overrides)"
                        .into(),
                );
            }
            let gate = gpu
                .qwen4_moe_sym_gate_up_symbol(false, [2 * I, H, TOP, t, sh.e])
                .ok_or("qwen4_moe_sym_gate_up_symbol: no descriptor")?;
            let [_, down] = gpu
                .qwen4_moe_sym_gemm_symbols_wide(false, slots, sh.e)
                .ok_or("qwen4_moe_sym_gemm_symbols_wide: no descriptor")?;
            if (gate, down) != (GATE_IU4, DOWN_IU4) {
                return Err(format!(
                    "the live IU4 selector picks ({gate}, {down}), the documented plan is \
                     ({GATE_IU4}, {DOWN_IU4}) (an environment override such as \
                     HIPFIRE_QWEN4_MOE_SYM_PM changed what is launched)"
                ));
            }
        }

        // Weights: only the experts of the selected panels (every table entry stays a valid
        // pointer: unused ones alias the first arena slot and are never addressed by a tile).
        let sel: Vec<Panel> = self.selected().to_vec();
        let mut needed: BTreeSet<usize> = BTreeSet::new();
        for p in &sel {
            for &e in &self.grouping.tiles[p.row_base / BLOCK_M..(p.row_base + p.rows) / BLOCK_M] {
                if e >= 0 {
                    needed.insert(e as usize % sh.distinct);
                }
            }
        }
        let arena: Vec<usize> = needed.iter().copied().collect();
        let slot_of: BTreeMap<usize, usize> =
            arena.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let mut gu = vec![0u8; arena.len() * GU_BYTES];
        let mut dn = vec![0u8; arena.len() * DN_BYTES];
        gu.par_chunks_mut(GU_BYTES)
            .zip(dn.par_chunks_mut(DN_BYTES))
            .enumerate()
            .for_each(|(i, (g, d))| gen_expert(sh.seed, arena[i], g, d));
        check_symmetric(&gu[..GU_BYTES], &dn[..DN_BYTES])?;
        let gu_t = hip("upload gate/up experts", gpu.upload_raw(&gu, &[gu.len()]))?;
        let dn_t = hip("upload down experts", gpu.upload_raw(&dn, &[dn.len()]))?;
        drop((gu, dn));
        let table = |base: u64, stride: usize| -> Vec<u8> {
            (0..sh.e)
                .flat_map(|e| {
                    let i = slot_of.get(&(e % sh.distinct)).copied().unwrap_or(0);
                    (base + (i * stride) as u64).to_le_bytes()
                })
                .collect()
        };
        let gate_up_ptrs = hip(
            "upload gate/up table",
            gpu.upload_raw(&table(gu_t.buf.as_ptr() as u64, GU_BYTES), &[sh.e * 8]),
        )?;
        let down_ptrs = hip(
            "upload down table",
            gpu.upload_raw(&table(dn_t.buf.as_ptr() as u64, DN_BYTES), &[sh.e * 8]),
        )?;
        if !hip(
            "qwen4_moe_sym_check(gate_up)",
            gpu.qwen4_moe_sym_check(&gate_up_ptrs, 2 * I, H, sh.e, 136),
        )? {
            return Err("qwen4_moe_sym_check rejected the generated QT44 gate/up headers".into());
        }
        if !hip(
            "qwen4_moe_sym_check(down)",
            gpu.qwen4_moe_sym_check(&down_ptrs, H, I, sh.e, 68),
        )? {
            return Err("qwen4_moe_sym_check rejected the generated MQ4G128 down headers".into());
        }

        // Activations.
        let mut x = vec![0f32; t * H];
        x.par_chunks_mut(H)
            .enumerate()
            .for_each(|(r, row)| x_row(sh.seed, r, row));
        let x_norm = hip("upload x_norm", gpu.upload_f32(&x, &[t, H]))?;
        drop(x);

        // The device grouping must be the host grouping the panels are cut from.
        self.check_device_grouping(gpu)?;

        // Panel route tensors (global slot ids, as the production panel views).
        let i32_bytes = |v: &[i32], pad: usize| -> Vec<u8> {
            v.iter()
                .copied()
                .chain(std::iter::repeat(-1).take(pad))
                .flat_map(|x| x.to_le_bytes())
                .collect()
        };
        let mut panels = Vec::with_capacity(sel.len());
        for p in &sel {
            let tiles = &self.grouping.tiles[p.row_base / BLOCK_M..(p.row_base + p.rows) / BLOCK_M];
            let sorted = &self.grouping.sorted[p.row_base..p.row_base + p.rows];
            // Slack of sentinel (-1) entries past the panel: no kernel reads beyond `rows`
            // (pm-krsi-1100/gpu/route/AUDIT.md), this is margin against an addressing slip.
            let (tile_pad, sorted_pad) = (SLACK_ROWS / BLOCK_M, SLACK_ROWS);
            panels.push(PanelDev {
                index: p.index,
                rows: p.rows,
                row_base: p.row_base,
                tiles: hip(
                    "upload panel tiles",
                    gpu.upload_raw(&i32_bytes(tiles, tile_pad), &[tiles.len() + tile_pad, 4]),
                )?,
                sorted: hip(
                    "upload panel sorted",
                    gpu.upload_raw(
                        &i32_bytes(sorted, sorted_pad),
                        &[sorted.len() + sorted_pad, 4],
                    ),
                )?,
            });
        }

        // Two buffer sets, every scratch padded by `SLACK_ROWS` rows past its largest panel
        // (and the window F16 X by as many rows): the audited max touch is `rows`, the slack
        // keeps an addressing slip from reaching a neighbouring allocation (the PM Y descriptors
        // are unbounded).
        let rows = self.max_rows() + SLACK_ROWS;
        let set = |gpu: &mut Gpu| -> Result<ArmSet> {
            Ok(ArmSet {
                x_f16: hip("alloc x_f16", gpu.zeros(&[t + SLACK_ROWS, H], DType::F16))?,
                local: hip("alloc local", gpu.zeros(&[rows, 4], DType::Raw))?,
                y_gu: hip("alloc y_gate_up", gpu.zeros(&[rows, I], DType::BF16))?,
                y_down: hip("alloc y_down", gpu.zeros(&[rows, H], DType::BF16))?,
            })
        };
        let sets = [set(gpu)?, set(gpu)?];

        // Launcher scratch sized for exactly this window: nothing grows inside a chain.
        hip(
            "reserve_qwen4_moe_sym",
            gpu.reserve_qwen4_moe_sym(t, TOP, sh.e, H, I),
        )?;
        self.dev = Some(Dev {
            x_norm,
            gate_up_ptrs,
            down_ptrs,
            panels,
            sets,
            _weights: vec![gu_t, dn_t],
        });
        // Prime: module loads, sign tables, scratch binding, outside any timed interval.
        for arm in [Arm::Baseline, Arm::Candidate] {
            self.encode(gpu, arm, &mut |_, _, _| Ok(()))?;
        }
        hip("prime synchronize", gpu.hip.device_synchronize())?;
        Ok(())
    }

    /// Run the device stable grouping on the synthetic route and compare it with the host grouping.
    fn check_device_grouping(&self, gpu: &mut Gpu) -> Result<()> {
        let sh = &self.shape;
        let g = &self.grouping;
        let i32_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let topk = hip(
            "upload topk",
            gpu.upload_raw(&i32_bytes(&self.topk), &[sh.t, TOP, 4]),
        )?;
        // +4 KiB slack on every output (the kernels write exactly the audited extents).
        let raw =
            |gpu: &mut Gpu, bytes: usize| hip("alloc", gpu.zeros(&[bytes + 4096], DType::Raw));
        let counts = raw(gpu, sh.e * 4)?;
        let offsets = raw(gpu, (sh.e + 1) * 4)?;
        let sorted = raw(gpu, g.bound * 4)?;
        let tiles = raw(gpu, g.bound / BLOCK_M * 4)?;
        let inverse = raw(gpu, g.slots * 4)?;
        hip(
            "moe_scatter_stable_top10",
            gpu.moe_scatter_stable_top10(
                &topk, &counts, &offsets, &sorted, &tiles, &inverse, g.slots, sh.e, g.bound,
            ),
        )?;
        hip("scatter synchronize", gpu.hip.device_synchronize())?;
        let dl = |gpu: &Gpu, t: &GpuTensor| -> Result<Vec<i32>> {
            let b = hip("download", gpu.download_raw_bytes(t))?;
            Ok(b.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let (dsorted, dtiles) = (dl(gpu, &sorted)?, dl(gpu, &tiles)?);
        let first = |a: &[i32], b: &[i32]| a.iter().zip(b).position(|(x, y)| x != y);
        if let Some(p) = first(&dsorted, &g.sorted) {
            return Err(format!(
                "host grouping differs from the device stable grouping at sorted[{p}]: device {} host {}",
                dsorted[p], g.sorted[p]
            ));
        }
        if let Some(p) = first(&dtiles, &g.tiles) {
            return Err(format!(
                "host grouping differs from the device stable grouping at tiles[{p}]: device {} host {}",
                dtiles[p], g.tiles[p]
            ));
        }
        for tn in [topk, counts, offsets, sorted, tiles, inverse] {
            hip("free", gpu.free_tensor(tn))?;
        }
        Ok(())
    }

    // -- the chain ----------------------------------------------------------

    /// The window stages of `arm` (producer; F16 also appends into the arm's `x_f16`).
    fn encode_window(&self, gpu: &mut Gpu, arm: Arm, hook: &mut Hook<'_>) -> Result<Window> {
        let d = self.dev()?;
        let set = &d.sets[self.set_index(arm)];
        let t = self.shape.t;
        match self.arm_route(arm) {
            Route::F16 => {
                let x16 = hip(
                    "rotate_x_mq_batched_f16",
                    gpu.rotate_x_mq_batched_f16(&d.x_norm, H, t),
                )?;
                hook(gpu, None, Kind::XProduce)?;
                hip(
                    "x_f16 append",
                    gpu.memcpy_dtod_at_ordered_async(&set.x_f16.buf, 0, &x16.buf, 0, t * H * 2),
                )?;
                hook(gpu, None, Kind::XAppend)?;
                hook(gpu, None, Kind::WindowEnd)?;
                Ok(Window::F16)
            }
            Route::Iu4 => {
                let xq = hip(
                    "qwen4_moe_rotate256_i4",
                    gpu.qwen4_moe_rotate256_i4(&d.x_norm, H, t),
                )?;
                hook(gpu, None, Kind::XProduce)?;
                hook(gpu, None, Kind::WindowEnd)?;
                Ok(Window::Iu4(xq))
            }
        }
    }

    /// One panel of `arm` (stages 2-4 of the module table).
    fn encode_panel(
        &self,
        gpu: &mut Gpu,
        arm: Arm,
        pi: usize,
        win: &Window,
        hook: &mut Hook<'_>,
    ) -> Result<()> {
        let d = self.dev()?;
        let set = &d.sets[self.set_index(arm)];
        let p = &d.panels[pi];
        let at = |what: &str| format!("{what} (panel {} rows {})", p.index, p.rows);
        let (t, slots) = (self.shape.t, self.shape.slots());
        match win {
            Window::F16 => {
                hip(
                    &at("gemm_mq4g256v2_moe_grouped_top10_silu_bf16out"),
                    gpu.gemm_mq4g256v2_moe_grouped_top10_silu_bf16out(
                        &d.gate_up_ptrs,
                        &p.tiles,
                        &p.sorted,
                        &set.x_f16,
                        &set.y_gu,
                        2 * I,
                        H,
                        TOP,
                        p.rows,
                        t,
                    ),
                )?;
                hook(gpu, Some(pi), Kind::GateUp)?;
                hip(
                    &at("slowlink_remap_sorted"),
                    gpu.slowlink_remap_sorted(&p.sorted, &set.local, p.rows),
                )?;
                hook(gpu, Some(pi), Kind::Remap)?;
                let xd = hip(
                    &at("moe_unscatter_rotate128_f16"),
                    gpu.moe_unscatter_rotate128_f16(&set.y_gu, &set.local, I, p.rows, p.rows),
                )?;
                hook(gpu, Some(pi), Kind::Unscatter)?;
                hip(
                    &at("gemm_mq4g128v2_moe_grouped_top10_xf16"),
                    gpu.gemm_mq4g128v2_moe_grouped_top10_xf16(
                        &d.down_ptrs,
                        &p.tiles,
                        &set.local,
                        &xd,
                        &set.y_down,
                        H,
                        I,
                        p.rows,
                        true,
                    ),
                )?;
                hook(gpu, Some(pi), Kind::Down)?;
            }
            Window::Iu4(xq) => {
                hip(
                    &at("gemm_qwen4_moe_gate_up_silu_iu4_sym"),
                    gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
                        &d.gate_up_ptrs,
                        &p.tiles,
                        &p.sorted,
                        xq,
                        &set.y_gu,
                        2 * I,
                        H,
                        TOP,
                        p.rows,
                        t,
                        false,
                    ),
                )?;
                hook(gpu, Some(pi), Kind::GateUp)?;
                let xd = hip(
                    &at("qwen4_moe_rotate128_i4"),
                    gpu.qwen4_moe_rotate128_i4(&set.y_gu, &p.sorted, I, p.rows, slots),
                )?;
                hook(gpu, Some(pi), Kind::DownProduce)?;
                hip(
                    &at("gemm_qwen4_moe_down_iu4_sym"),
                    gpu.gemm_qwen4_moe_down_iu4_sym(
                        &d.down_ptrs,
                        &p.tiles,
                        &p.sorted,
                        &xd,
                        &set.y_down,
                        H,
                        I,
                        1,
                        p.rows,
                        slots,
                        false,
                    ),
                )?;
                hook(gpu, Some(pi), Kind::Down)?;
            }
        }
        Ok(())
    }

    /// Enqueue `arm`'s whole chain on the default stream; `hook` runs after every stage.
    fn encode(&self, gpu: &mut Gpu, arm: Arm, hook: &mut Hook<'_>) -> Result<()> {
        let win = self.encode_window(gpu, arm, hook)?;
        for pi in 0..self.dev()?.panels.len() {
            self.encode_panel(gpu, arm, pi, &win, hook)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CPU references (independent of every device kernel)

fn bf16r(x: f32) -> f32 {
    bf16_val(bf16_bits(x))
}

/// FWHT over `x` (power-of-two length): `x *= s1`, Sylvester butterfly, `x *= (1/sqrt n) * s2`.
fn fwht_f64(x: &mut [f64], s1: &[f32], s2: &[f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two() && s1.len() >= n && s2.len() >= n);
    for i in 0..n {
        x[i] *= s1[i] as f64;
    }
    let mut stride = 1;
    while stride < n {
        let mut i = 0;
        while i < n {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 1.0 / (n as f64).sqrt();
    for i in 0..n {
        x[i] *= scale * s2[i] as f64;
    }
}

fn dequant_gu_row(gu: &[u8], row: usize, out: &mut [f64]) {
    let base = row * (H / 256) * 136;
    for g in 0..H / 256 {
        let gp = &gu[base + g * 136..base + (g + 1) * 136];
        let sc0 = f16_val(u16::from_le_bytes([gp[0], gp[1]])) as f64;
        let sc1 = f16_val(u16::from_le_bytes([gp[4], gp[5]])) as f64;
        let payload = &gp[8..];
        for w in 0..256 {
            let byte = payload[w / 2];
            let u = if w % 2 == 0 { byte & 0xF } else { byte >> 4 } as f64;
            let sc = if w < 128 { sc0 } else { sc1 };
            out[g * 256 + w] = sc * (u - 8.0);
        }
    }
}

fn dequant_dn_row(dn: &[u8], row: usize, out: &mut [f64]) {
    let base = row * (I / 128) * 68;
    for g in 0..I / 128 {
        let gp = &dn[base + g * 68..base + (g + 1) * 68];
        let sc = f16_val(u16::from_le_bytes([gp[0], gp[1]])) as f64;
        let payload = &gp[4..];
        for w in 0..128 {
            let byte = payload[w / 2];
            let u = if w % 2 == 0 { byte & 0xF } else { byte >> 4 } as f64;
            out[g * 128 + w] = sc * (u - 8.0);
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Signs {
    s1_256: Vec<f32>,
    s2_256: Vec<f32>,
    s1_128: Vec<f32>,
    s2_128: Vec<f32>,
}

impl Signs {
    fn new() -> Self {
        Signs {
            s1_256: gen_fwht_signs(42, 256),
            s2_256: gen_fwht_signs(1042, 256),
            s1_128: gen_fwht_signs(43, 128),
            s2_128: gen_fwht_signs(1043, 128),
        }
    }
}

/// One slot through the unquantised chain in f64: FWHT256 of the F32 row, gate/up dot, the
/// recipe's BF16 round trips of gate/up/SwiGLU, FWHT128, down dot. Unrounded `[H]` output.
fn f64_slot(x: &[f32], gu: &[u8], dn: &[u8], signs: &Signs) -> Vec<f64> {
    let mut xr: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    for g in xr.chunks_mut(256) {
        fwht_f64(g, &signs.s1_256, &signs.s2_256);
    }
    let mut row = vec![0.0f64; H];
    let mut gate = vec![0.0f32; I];
    let mut up = vec![0.0f32; I];
    for r in 0..2 * I {
        dequant_gu_row(gu, r, &mut row);
        let d = dot(&row, &xr) as f32;
        if r < I {
            gate[r] = d;
        } else {
            up[r - I] = d;
        }
    }
    let mut rot: Vec<f64> = (0..I)
        .map(|k| {
            let v = bf16r(gate[k]);
            let u = bf16r(up[k]);
            let silu = v / (1.0 + (-v).exp());
            bf16r(silu * u) as f64
        })
        .collect();
    for g in rot.chunks_mut(128) {
        fwht_f64(g, &signs.s1_128, &signs.s2_128);
    }
    let mut drow = vec![0.0f64; I];
    (0..H)
        .map(|o| {
            dequant_dn_row(dn, o, &mut drow);
            dot(&drow, &rot)
        })
        .collect()
}

/// The sym-IU4 chain's gate/up rotation (`mq_rotate_x_i4`) of one 256-group in f32, op for op.
fn rotate256(x: &[f32], s1: &[f32], s2: &[f32]) -> [f32; 256] {
    let mut v = [0f32; 256];
    for i in 0..256 {
        v[i] = x[i] * s1[i];
    }
    let mut stride = 1;
    while stride < 256 {
        for base in (0..256).step_by(2 * stride) {
            for j in base..base + stride {
                let (lo, hi) = (v[j], v[j + stride]);
                v[j] = lo + hi;
                v[j + stride] = lo - hi;
            }
        }
        stride *= 2;
    }
    for i in 0..256 {
        v[i] = (v[i] * 0.0625f32) * s2[i];
    }
    v
}

/// Scale and signed codes of one `block_i4_128` record.
fn decode_record(rec: &[u8; REC]) -> (f32, [i8; GROUP]) {
    let d = f32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
    let mut q = [0i8; GROUP];
    for (i, c) in q.iter_mut().enumerate() {
        let byte = rec[8 + i / 2];
        let nib = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
        *c = ((nib << 4) as i8) >> 4;
    }
    (d, q)
}

/// The gate/up A4 sidecar row of one token (`qwen4_moe_rotate256_i4`).
fn gate_xrow(x: &[f32], signs: &Signs) -> XRow {
    let mut d = Vec::with_capacity(x.len() / GROUP);
    let mut v = Vec::with_capacity(x.len());
    for g in x.chunks_exact(256) {
        let r = rotate256(g, &signs.s1_256, &signs.s2_256);
        for half in 0..2 {
            let mut blk = [0f32; GROUP];
            blk.copy_from_slice(&r[half * GROUP..(half + 1) * GROUP]);
            let (dd, q) = decode_record(&quantize(&blk));
            d.push(dd);
            v.extend_from_slice(&q);
        }
    }
    XRow { d, v }
}

/// One K128 block of the down A4 sidecar from 128 BF16 activations (`qwen4_moe_rotate128_i4`).
fn down_block(act: &[u16], signs: &Signs) -> (f32, [i8; GROUP]) {
    let mut h = [0u16; GROUP];
    h.copy_from_slice(&act[..GROUP]);
    let mut s1 = [0f32; GROUP];
    let mut s2 = [0f32; GROUP];
    s1.copy_from_slice(&signs.s1_128[..GROUP]);
    s2.copy_from_slice(&signs.s2_128[..GROUP]);
    decode_record(&quantize(&rotate(&h, &s1, &s2)))
}

/// One slot through the exact A4 recipe: BF16-valued `[H]` down row.
fn a4_slot(x: &[f32], gu: &[u8], dn: &[u8], signs: &Signs) -> Vec<f32> {
    let xrow = gate_xrow(x, signs);
    let sums = fold_slots(
        SymKind::GateUp,
        2 * I,
        H / GROUP,
        GU_ROW_BYTES,
        gu,
        std::slice::from_ref(&xrow),
    );
    let bits: Vec<u16> = (0..I)
        .map(|k| bf16_bits(portable_silu_mul(rt_f32(sums[k]), rt_f32(sums[I + k]))))
        .collect();
    let mut d = Vec::with_capacity(I / GROUP);
    let mut v = Vec::with_capacity(I);
    for b in 0..I / GROUP {
        let (dd, q) = down_block(&bits[b * GROUP..], signs);
        d.push(dd);
        v.extend_from_slice(&q);
    }
    let nominal = XRow { d, v };
    fold_slots(
        SymKind::Down,
        H,
        I / GROUP,
        DN_ROW_BYTES,
        dn,
        std::slice::from_ref(&nominal),
    )
    .into_iter()
    .map(bf16r)
    .collect()
}

// ---------------------------------------------------------------------------
// Streaming comparison of the two routes' outputs

#[derive(Clone, Debug, Default)]
struct Dist {
    elems: u64,
    sum_d2: f64,
    sum_base2: f64,
    sum_cand2: f64,
    sum_abs: f64,
    max_abs: f64,
    nonfinite_base: u64,
    nonfinite_cand: u64,
    identical: u64,
    worst_panel: Option<(f64, usize)>,
}

impl Dist {
    fn merge(mut self, o: Dist) -> Dist {
        self.elems += o.elems;
        self.sum_d2 += o.sum_d2;
        self.sum_base2 += o.sum_base2;
        self.sum_cand2 += o.sum_cand2;
        self.sum_abs += o.sum_abs;
        self.max_abs = self.max_abs.max(o.max_abs);
        self.nonfinite_base += o.nonfinite_base;
        self.nonfinite_cand += o.nonfinite_cand;
        self.identical += o.identical;
        self
    }

    fn rel_l2(&self) -> f64 {
        (self.sum_d2 / self.sum_base2).sqrt()
    }

    fn json(&self) -> Value {
        let n = self.elems.max(1) as f64;
        json!({
            "elems": self.elems,
            "rel_l2": self.rel_l2(),
            "max_abs": self.max_abs,
            "mean_abs": self.sum_abs / n,
            "rms_baseline": (self.sum_base2 / n).sqrt(),
            "rms_candidate": (self.sum_cand2 / n).sqrt(),
            "identical_elems": self.identical,
            "nonfinite_baseline": self.nonfinite_base,
            "nonfinite_candidate": self.nonfinite_cand,
            "worst_panel": self.worst_panel.map(|(r, p)| json!({"panel": p, "rel_l2": r})),
            "definition": "rel_l2 = ||candidate - baseline||_2 / ||baseline||_2 over every live grouped row \
                           of the BF16 down output; max_abs/mean_abs over the same elements",
        })
    }
}

fn bf16_at(b: &[u8], i: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])) << 16)
}

/// Distance statistics of one panel's `[rows, H]` BF16 outputs over its live rows.
fn panel_dist(base: &[u8], cand: &[u8], live: &[bool], panel: usize) -> Dist {
    let mut d = (0..live.len())
        .into_par_iter()
        .filter(|r| live[*r])
        .map(|r| {
            let mut s = Dist::default();
            for c in 0..H {
                let i = r * H + c;
                let (a, b) = (bf16_at(base, i), bf16_at(cand, i));
                s.elems += 1;
                if !a.is_finite() {
                    s.nonfinite_base += 1;
                }
                if !b.is_finite() {
                    s.nonfinite_cand += 1;
                }
                if a.to_bits() == b.to_bits() {
                    s.identical += 1;
                }
                if a.is_finite() && b.is_finite() {
                    let (a, b) = (f64::from(a), f64::from(b));
                    let diff = (b - a).abs();
                    s.sum_d2 += diff * diff;
                    s.sum_base2 += a * a;
                    s.sum_cand2 += b * b;
                    s.sum_abs += diff;
                    s.max_abs = s.max_abs.max(diff);
                }
            }
            s
        })
        .reduce(Dist::default, Dist::merge);
    if d.sum_base2 > 0.0 {
        d.worst_panel = Some((d.rel_l2(), panel));
    }
    d
}

fn merge_dist(mut a: Dist, b: Dist) -> Dist {
    let worst = match (a.worst_panel, b.worst_panel) {
        (Some(x), Some(y)) => Some(if y.0 > x.0 { y } else { x }),
        (x, y) => x.or(y),
    };
    a = a.merge(b);
    a.worst_panel = worst;
    a
}

/// Reference error accumulator: `||got - ref||`, `||ref||`, max abs, worst slot.
#[derive(Clone, Debug, Default)]
struct RefStat {
    n: u64,
    sum_d2: f64,
    sum_r2: f64,
    max_abs: f64,
    worst_slot_rel: f64,
    nonfinite: u64,
}

impl RefStat {
    fn add_slot(&mut self, got: &[f32], reference: &[f64]) {
        let (mut d2, mut r2) = (0.0f64, 0.0f64);
        for (g, r) in got.iter().zip(reference) {
            if !g.is_finite() {
                self.nonfinite += 1;
                continue;
            }
            let d = (f64::from(*g) - r).abs();
            d2 += d * d;
            r2 += r * r;
            self.max_abs = self.max_abs.max(d);
        }
        self.n += got.len() as u64;
        self.sum_d2 += d2;
        self.sum_r2 += r2;
        if r2 > 0.0 {
            self.worst_slot_rel = self.worst_slot_rel.max((d2 / r2).sqrt());
        }
    }
    fn rel_l2(&self) -> f64 {
        (self.sum_d2 / self.sum_r2).sqrt()
    }
    fn json(&self, bound: f64) -> Value {
        json!({
            "rel_l2": self.rel_l2(), "max_abs": self.max_abs,
            "worst_slot_rel_l2": self.worst_slot_rel, "elems": self.n,
            "nonfinite": self.nonfinite, "bound_rel_l2": bound,
            "pass": self.nonfinite == 0 && self.rel_l2() <= bound,
        })
    }
}

/// `n` items spaced evenly through `live` (first and last included).
fn pick_evenly(live: &[usize], n: usize) -> Vec<usize> {
    if n >= live.len() {
        return live.to_vec();
    }
    if n == 1 {
        return vec![live[0]];
    }
    (0..n)
        .map(|i| live[i * (live.len() - 1) / (n - 1)])
        .collect()
}

/// One streaming pass over every selected panel, arms interleaved panel by panel.
struct PassResult {
    dist: Dist,
    base_hash: String,
    cand_hash: String,
    /// `(grouped row, baseline row, candidate row)` of the sampled rows.
    samples: Vec<(usize, Vec<u16>, Vec<u16>)>,
}

impl Bench {
    fn live_rows_of(&self, p: &Panel) -> Vec<bool> {
        self.grouping.sorted[p.row_base..p.row_base + p.rows]
            .iter()
            .map(|s| *s >= 0)
            .collect()
    }

    /// Sampled grouped rows: `refslots` live rows evenly spaced through the selected panels.
    fn sample_rows(&self) -> Vec<usize> {
        let mut live = Vec::new();
        for p in self.selected() {
            for r in p.row_base..p.row_base + p.rows {
                if self.grouping.sorted[r] >= 0 {
                    live.push(r);
                }
            }
        }
        pick_evenly(&live, self.tune.refslots)
    }

    fn pass(
        &mut self,
        gpu: &mut Gpu,
        swapped: bool,
        sample: &BTreeSet<usize>,
    ) -> Result<PassResult> {
        self.swapped = swapped;
        let mut noop = |_: &mut Gpu, _: Option<usize>, _: Kind| -> Result<()> { Ok(()) };
        let wb = self.encode_window(gpu, Arm::Baseline, &mut noop)?;
        let wc = self.encode_window(gpu, Arm::Candidate, &mut noop)?;
        let (mut hb, mut hc) = (Sha256::new(), Sha256::new());
        let mut dist = Dist::default();
        let mut samples = Vec::new();
        let n_panels = self.dev()?.panels.len();
        for pi in 0..n_panels {
            let panel = self.selected()[pi];
            let download = |gpu: &mut Gpu, this: &Bench, arm: Arm| -> Result<Vec<u8>> {
                hip("panel synchronize", gpu.hip.device_synchronize())?;
                let d = this.dev()?;
                let t = &d.sets[this.set_index(arm)].y_down;
                hip("download y_down", gpu.download_raw_bytes(t))
            };
            self.encode_panel(gpu, Arm::Baseline, pi, &wb, &mut noop)?;
            let base = download(gpu, self, Arm::Baseline)?;
            self.encode_panel(gpu, Arm::Candidate, pi, &wc, &mut noop)?;
            let cand = download(gpu, self, Arm::Candidate)?;
            let live = self.live_rows_of(&panel);
            for (r, l) in live.iter().enumerate() {
                if *l {
                    hb.update(&base[r * H * 2..(r + 1) * H * 2]);
                    hc.update(&cand[r * H * 2..(r + 1) * H * 2]);
                }
            }
            dist = merge_dist(dist, panel_dist(&base, &cand, &live, panel.index));
            for r in panel.row_base..panel.row_base + panel.rows {
                if sample.contains(&r) {
                    let l = r - panel.row_base;
                    let row = |b: &[u8]| -> Vec<u16> {
                        b[l * H * 2..(l + 1) * H * 2]
                            .chunks_exact(2)
                            .map(|c| u16::from_le_bytes([c[0], c[1]]))
                            .collect()
                    };
                    samples.push((r, row(&base), row(&cand)));
                }
            }
        }
        self.swapped = false;
        let hex = |h: Sha256| {
            h.finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        Ok(PassResult {
            dist,
            base_hash: hex(hb),
            cand_hash: hex(hc),
            samples,
        })
    }

    /// The reference statistics of the sampled rows.
    fn references(&self, samples: &[(usize, Vec<u16>, Vec<u16>)]) -> Result<Value> {
        let (seed, distinct, aa) = (self.shape.seed, self.shape.distinct, self.aa);
        let (sorted, tiles) = (&self.grouping.sorted, &self.grouping.tiles);
        let signs = Signs::new();
        let rows: Vec<(Vec<f64>, Vec<f32>, &Vec<u16>, &Vec<u16>)> = samples
            .par_iter()
            .map(|(r, b, c)| {
                let slot = sorted[*r] as usize;
                let expert = tiles[r / BLOCK_M] as usize;
                let mut gu = vec![0u8; GU_BYTES];
                let mut dn = vec![0u8; DN_BYTES];
                gen_expert(seed, expert % distinct, &mut gu, &mut dn);
                let mut x = vec![0f32; H];
                x_row(seed, slot / TOP, &mut x);
                let f = f64_slot(&x, &gu, &dn, &signs);
                let a = if aa {
                    Vec::new()
                } else {
                    a4_slot(&x, &gu, &dn, &signs)
                };
                (f, a, b, c)
            })
            .collect();
        let widen = |v: &[u16]| -> Vec<f32> {
            v.iter()
                .map(|b| f32::from_bits(u32::from(*b) << 16))
                .collect()
        };
        let (mut base_f64, mut cand_f64, mut cand_a4) =
            (RefStat::default(), RefStat::default(), RefStat::default());
        for (f, a, b, c) in &rows {
            base_f64.add_slot(&widen(b), f);
            cand_f64.add_slot(&widen(c), f);
            if !a.is_empty() {
                let af: Vec<f64> = a.iter().map(|v| f64::from(*v)).collect();
                cand_a4.add_slot(&widen(c), &af);
            }
        }
        let t = self.tune;
        let mut v = json!({
            "slots": samples.len(),
            "f64_chain": {
                "baseline": base_f64.json(t.f16_bound),
                "candidate": cand_f64.json(if self.aa { t.f16_bound } else { t.a4_bound }),
            },
            "definition": "f64_chain: FWHT256 of the F32 row, gate/up dot, BF16 round trips, FWHT128, down dot \
                           (unquantised); a4_recipe: the exact sym-IU4 arithmetic (cpu_iu4_chain's, portable SiLU)",
        });
        if !self.aa {
            v["a4_recipe"] = json!({"candidate": cand_a4.json(t.a4_recipe_bound)});
        }
        Ok(v)
    }

    /// The checked correctness stage (module doc, "Correctness").
    pub fn correctness(&mut self, gpu: &mut Gpu) -> Result<(Value, bool, String)> {
        let rows = self.sample_rows();
        let set: BTreeSet<usize> = rows.iter().copied().collect();
        let p1 = self.pass(gpu, false, &set)?;
        let p2 = self.pass(gpu, true, &BTreeSet::new())?;
        let refs = self.references(&p1.samples)?;
        let base_repeat = p1.base_hash == p2.base_hash;
        let cand_repeat = p1.cand_hash == p2.cand_hash;
        let finite = p1.dist.nonfinite_base == 0 && p1.dist.nonfinite_cand == 0;
        let refs_pass = refs["f64_chain"]["baseline"]["pass"] == true
            && refs["f64_chain"]["candidate"]["pass"] == true
            && (self.aa || refs["a4_recipe"]["candidate"]["pass"] == true);
        let aa_identical = !self.aa || p1.base_hash == p1.cand_hash;
        let pass = finite && refs_pass && base_repeat && cand_repeat && aa_identical;
        let out = json!({
            "pass": pass,
            "pass_basis": "every live output finite; baseline vs f64 chain and candidate vs A4 recipe / f64 chain \
                           within their sanity bounds; both arms reproduce their outputs bit-for-bit on a repeat \
                           pass over the swapped buffer sets (and after timing); --null-control aa: candidate == \
                           baseline bit-for-bit. The cross-route distance is recorded, not gated.",
            "distance": p1.dist.json(),
            "identical_to_baseline": p1.base_hash == p1.cand_hash,
            "references": refs,
            "baseline_repeat_identical": base_repeat,
            "candidate_repeat_identical": cand_repeat,
            "repeat_on_swapped_buffers": true,
            "baseline_sha256": p1.base_hash,
            "candidate_sha256": p1.cand_hash,
            "null_control_identical": if self.aa { json!(aa_identical) } else { Value::Null },
            "emu": "not applicable (chain)",
        });
        Ok((out, pass, p1.base_hash + &p1.cand_hash))
    }

    /// Output hashes only (the post-timing guard).
    pub fn hashes(&mut self, gpu: &mut Gpu) -> Result<String> {
        let p = self.pass(gpu, false, &BTreeSet::new())?;
        Ok(p.base_hash + &p.cand_hash)
    }
}

// ---------------------------------------------------------------------------
// Timing: ABBA over the whole chain, split at the window producer; per-stage profile

#[derive(Default)]
struct Extra {
    base_prod: Vec<f64>,
    cand_prod: Vec<f64>,
    base_pan: Vec<f64>,
    cand_pan: Vec<f64>,
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

impl Bench {
    fn sample_rounds(
        &mut self,
        gpu: &mut Gpu,
        n: usize,
        rounds: usize,
        s: &mut Sampled,
        x: &mut Extra,
    ) -> Result<()> {
        let done = s.base_ns.len() / (2 * n);
        let (calls, _) = crate::abba::timing_counts(n, rounds)?;
        for v in [
            &mut s.base_ns,
            &mut s.cand_ns,
            &mut x.base_prod,
            &mut x.cand_prod,
            &mut x.base_pan,
            &mut x.cand_pan,
        ] {
            v.reserve(calls);
        }
        let mut evs = Vec::new();
        for _ in 0..3 {
            evs.push(hip("event_create", gpu.hip.event_create())?);
        }
        let measured = (|| -> Result<()> {
            for (i, which) in crate::abba::timing_schedule(n, rounds)?.enumerate() {
                if i % (4 * n) == 0 {
                    // Crossover: even rounds bind each arm to its own set, odd rounds swap.
                    self.swapped = (done + i / (4 * n)) % 2 == 1;
                }
                let arm = match which {
                    crate::abba::TimingArm::Incumbent => Arm::Baseline,
                    crate::abba::TimingArm::Candidate => Arm::Candidate,
                };
                hip("event_record", gpu.hip.event_record(&evs[0], None))?;
                {
                    let mid = &evs[1];
                    let mut hook = |g: &mut Gpu, panel: Option<usize>, k: Kind| -> Result<()> {
                        if panel.is_none() && k == Kind::WindowEnd {
                            hip("event_record", g.hip.event_record(mid, None))
                        } else {
                            Ok(())
                        }
                    };
                    self.encode(gpu, arm, &mut hook)?;
                }
                hip("event_record", gpu.hip.event_record(&evs[2], None))?;
                hip("event_synchronize", gpu.hip.event_synchronize(&evs[2]))?;
                let ms = |a: usize, b: usize| -> Result<f64> {
                    Ok(f64::from(hip(
                        "event_elapsed_ms",
                        gpu.hip.event_elapsed_ms(&evs[a], &evs[b]),
                    )?) * 1e6)
                };
                let (chain, prod, pan) = (ms(0, 2)?, ms(0, 1)?, ms(1, 2)?);
                match which {
                    crate::abba::TimingArm::Incumbent => {
                        s.base_ns.push(chain);
                        x.base_prod.push(prod);
                        x.base_pan.push(pan);
                    }
                    crate::abba::TimingArm::Candidate => {
                        s.cand_ns.push(chain);
                        x.cand_prod.push(prod);
                        x.cand_pan.push(pan);
                    }
                }
            }
            Ok(())
        })();
        self.swapped = false;
        let mut destroyed = Ok(());
        for e in evs {
            if let Err(e) = gpu.hip.event_destroy(e) {
                destroyed = Err(err(e));
            }
        }
        measured?;
        destroyed
    }

    /// ABBA over the whole chain (+ the window / panel series split at the producer's event).
    pub fn time(&mut self, gpu: &mut Gpu, o: &Opts, baseline_sha: &str) -> Result<Value> {
        let start = std::time::Instant::now();
        let mut extra = Extra::default();
        let (s, stop, secs) = crate::drive_sampling(
            o.time,
            o.rounds,
            o.max_rounds,
            o.time_budget_s,
            &mut || start.elapsed().as_secs_f64(),
            &mut |more, s| self.sample_rounds(gpu, o.time, more, s, &mut extra),
        )?;
        let shape = self.shape_json();
        let mut t = crate::timed_record(o, &shape, baseline_sha, &s, stop, secs)?;
        t["unit"] = json!(
            "us per chain (hipEvent around the window producer + every selected panel, default stream)"
        );
        t["crossover"] = crate::crossover_json(&t)?;
        let brief = |base: Vec<f64>, cand: Vec<f64>| -> Result<Value> {
            let sp = Sampled {
                base_ns: base,
                cand_ns: cand,
            };
            let r = crate::timing_record(&sp, o.time, o.rounds, stop, secs, o.min_effect)?;
            let mut out = serde_json::Map::new();
            for k in [
                "baseline_us",
                "candidate_us",
                "ratio",
                "block_ratio",
                "separable",
                "direction",
                "material",
                "sign_p",
                "rounds_faster",
                "rounds_slower",
                "rounds_tied",
                "round_ratio_min",
                "round_ratio_max",
            ] {
                out.insert(k.into(), r[k].clone());
            }
            Ok(Value::Object(out))
        };
        t["series"] = json!({
            "x_producer": brief(extra.base_prod, extra.cand_prod)?,
            "panels": brief(extra.base_pan, extra.cand_pan)?,
            "note": "the same ABBA slots as the chain series, cut at the event after the window producer \
                     (event_record adds one stream marker per launch slot)",
        });
        t["post_outputs_unchanged"] = json!(true);
        Ok(t)
    }

    /// Event-bracketed per-stage medians of both arms (alternating `stage-reps` chains each).
    pub fn stage_profile(&mut self, gpu: &mut Gpu) -> Result<Value> {
        let reps = self.tune.stage_reps;
        let n_panels = self.dev()?.panels.len();
        let max_events = 1 + 3 + n_panels * 4 + 2;
        let mut evs = Vec::with_capacity(max_events);
        for _ in 0..max_events {
            evs.push(hip("event_create", gpu.hip.event_create())?);
        }
        // per arm: (group -> per-rep totals), (kind name -> per-rep totals), per-call samples of
        // full-size panels per group.
        type Acc = (
            BTreeMap<&'static str, Vec<f64>>,
            BTreeMap<&'static str, Vec<f64>>,
            BTreeMap<&'static str, Vec<f64>>,
            Vec<f64>,
        );
        let mut acc: [Acc; 2] = Default::default();
        let full_rows = self.shape.panel_rows;
        let panel_rows: Vec<usize> = self.dev()?.panels.iter().map(|p| p.rows).collect();
        let result = (|| -> Result<()> {
            for _ in 0..reps {
                for (ai, arm) in [Arm::Baseline, Arm::Candidate].into_iter().enumerate() {
                    let mut labels: Vec<(Option<usize>, Kind)> = Vec::new();
                    let mut idx = 1usize;
                    hip("event_record", gpu.hip.event_record(&evs[0], None))?;
                    {
                        let evs_ref = &evs;
                        let labels_ref = &mut labels;
                        let idx_ref = &mut idx;
                        let mut hook = |g: &mut Gpu, panel: Option<usize>, k: Kind| -> Result<()> {
                            if k == Kind::WindowEnd {
                                return Ok(());
                            }
                            hip("event_record", g.hip.event_record(&evs_ref[*idx_ref], None))?;
                            *idx_ref += 1;
                            labels_ref.push((panel, k));
                            Ok(())
                        };
                        self.encode(gpu, arm, &mut hook)?;
                    }
                    hip(
                        "event_synchronize",
                        gpu.hip.event_synchronize(&evs[idx - 1]),
                    )?;
                    let mut by_group: BTreeMap<&'static str, f64> = BTreeMap::new();
                    let mut by_kind: BTreeMap<&'static str, f64> = BTreeMap::new();
                    let mut total = 0.0;
                    for (j, (panel, kind)) in labels.iter().enumerate() {
                        let us = f64::from(hip(
                            "event_elapsed_ms",
                            gpu.hip.event_elapsed_ms(&evs[j], &evs[j + 1]),
                        )?) * 1e3;
                        *by_group.entry(kind.group()).or_default() += us;
                        *by_kind.entry(kind.name()).or_default() += us;
                        total += us;
                        if let Some(pi) = panel {
                            if panel_rows[*pi] == full_rows {
                                acc[ai].2.entry(kind.group()).or_default().push(us);
                            }
                        }
                    }
                    for (g, v) in by_group {
                        acc[ai].0.entry(g).or_default().push(v);
                    }
                    for (k, v) in by_kind {
                        acc[ai].1.entry(k).or_default().push(v);
                    }
                    acc[ai].3.push(total);
                }
            }
            Ok(())
        })();
        for e in evs {
            let _ = gpu.hip.event_destroy(e);
        }
        result?;
        let calls = |group: &str, route: Route| -> usize {
            stage_kinds(route)
                .iter()
                .filter(|(k, _)| k.group() == group)
                .map(|(_, w)| if *w { 1 } else { n_panels })
                .sum()
        };
        let mut arms = serde_json::Map::new();
        let mut totals = [BTreeMap::new(), BTreeMap::new()];
        for (ai, arm) in [Arm::Baseline, Arm::Candidate].into_iter().enumerate() {
            let route = self.arm_route(arm);
            let a = &mut acc[ai];
            let mut groups = serde_json::Map::new();
            for g in GROUPS {
                let Some(v) = a.0.get_mut(g) else { continue };
                let total = median(v);
                let c = calls(g, route);
                let mut e =
                    json!({"total_us": total, "calls": c, "mean_call_us": total / c as f64});
                if let Some(f) = a.2.get_mut(g) {
                    e["full_panel_median_us"] = json!(median(f));
                    e["full_panel_calls"] = json!(f.len());
                }
                totals[ai].insert(g, total);
                groups.insert(g.into(), e);
            }
            let mut kinds = serde_json::Map::new();
            for (k, v) in a.1.iter_mut() {
                kinds.insert((*k).into(), json!({"total_us": median(v)}));
            }
            let sum = median(&mut a.3);
            totals[ai].insert("stage_sum", sum);
            arms.insert(
                if ai == 0 { "baseline" } else { "candidate" }.into(),
                json!({"route": route.name(), "groups": groups, "substages": kinds, "stage_sum_us": sum}),
            );
        }
        let mut ratios = serde_json::Map::new();
        for g in GROUPS.iter().chain(["stage_sum"].iter()) {
            if let (Some(b), Some(c)) = (totals[0].get(g), totals[1].get(g)) {
                ratios.insert((*g).into(), json!(c / b));
            }
        }
        Ok(json!({
            "reps": reps,
            "method": "hipEvents between consecutive stages of one chain per rep, arms alternating; median over \
                       reps of each stage's total over all panels; full_panel_median_us = median per-call time \
                       of the full-size panels. The extra markers serialise the stream slightly: compare with \
                       timing for the unbracketed chain.",
            "baseline": arms["baseline"], "candidate": arms["candidate"],
            "ratio_candidate_over_baseline": ratios,
        }))
    }
}

// ---------------------------------------------------------------------------
// Pre-card emulator gate (`--emu`; always run before a device is opened)
//
// The three PM launches of the IU4 chain (`qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1100`, the K3 down
// producer `mq_i4_a4_r128_k640_gfx1100`, `qwen4_moe_down_iu4_sym_pm_gfx1100`) run on the peacemaker
// emulator (`gfx1100_as_gfx1151`, strict LDS) with the chain's literal kernarg values, every buffer an
// exact-size region `EMU_GUARD` unmapped bytes from its neighbours (an access past either end is an
// error), at (1) a small shape, every workgroup of every panel, with the outputs checked against the
// exact A4 recipe, and (2) the REAL shape's first / middle / last panel at the grid extremes (first,
// last, padded and middle tiles, all x blocks). The hipcc producers (`mq_rotate_x_i4`) and the whole F16
// arm are not emulated: the A4 window sidecar is built by the host recipe (`gate_records`) and the F16
// arm is hipcc code that the A/A run of the same harness exercises on the card.

const EMU_GUARD: u64 = 0x2_0000;

struct EmuProgs {
    gemm: peacemaker_ir::Program,
    k3: peacemaker_ir::Program,
}

fn emu_program(obj: &crate::workload::Object) -> Result<peacemaker_ir::Program> {
    peacemaker_lift::lift_object(
        &obj.bytes,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .map(|l| l.program)
    .map_err(|e| format!("lift {}: {e:?}", obj.launch_symbol))
}

fn emu_progs() -> Result<EmuProgs> {
    let objs = Iu4Arm::new(ARCH)?.pm_objects()?;
    if objs.len() != 2 {
        return Err(format!(
            "expected the GEMM bundle and the K3 producer, got {} objects",
            objs.len()
        ));
    }
    Ok(EmuProgs {
        gemm: emu_program(&objs[0])?,
        k3: emu_program(&objs[1])?,
    })
}

fn emu_make<'a>(
    p: &'a peacemaker_ir::Program,
    symbol: &str,
) -> Result<peacemaker_ir::emu::Emulator<'a>> {
    Ok(peacemaker_ir::emu::Emulator::new(p, symbol)?
        .hw_model()
        .gfx1100_as_gfx1151())
}

/// The 64-byte kernarg of the grouped IU4 GEMMs (`qwen4_moe_sym_gfx1100/mod.rs`:
/// `ptrs@0 tiles@8 sorted@16 Xq@24 Y@32 M@40 K@44 x_row_div@48 m_total@52 x_src_rows@56 n_experts@60`).
#[allow(clippy::too_many_arguments)]
fn gemm_kernarg(ptrs: u64, tiles: u64, sorted: u64, xq: u64, y: u64, dims: [u32; 6]) -> Vec<u8> {
    let mut b = Vec::with_capacity(64);
    for p in [ptrs, tiles, sorted, xq, y] {
        b.extend_from_slice(&p.to_le_bytes());
    }
    for d in dims {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b
}

/// The 48-byte kernarg of the K3 producer (`h@0 sorted@8 xq@16 signs1@24 signs2@32 P@40 L@44`).
fn k3_kernarg(h: u64, sorted: u64, xq: u64, s1: u64, s2: u64, p: u32, l: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(48);
    for v in [h, sorted, xq, s1, s2] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&p.to_le_bytes());
    b.extend_from_slice(&l.to_le_bytes());
    b
}

fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The A4 window sidecar records (`[H/128]` per token) of one token row: `mq_rotate_x_i4`'s output.
fn gate_records(x: &[f32], signs: &Signs) -> Vec<[u8; REC]> {
    let mut out = Vec::with_capacity(H / GROUP);
    for g in x.chunks_exact(256) {
        let r = rotate256(g, &signs.s1_256, &signs.s2_256);
        for half in 0..2 {
            let mut blk = [0f32; GROUP];
            blk.copy_from_slice(&r[half * GROUP..(half + 1) * GROUP]);
            out.push(quantize(&blk));
        }
    }
    out
}

/// Tiles to run when not running the whole grid: first, second, middle, last two, and the first /
/// middle / last tile that carries pad rows (`sorted == -1`).
fn tile_sample(sorted: &[i32], tiles: usize) -> Vec<usize> {
    let mut s = BTreeSet::new();
    for t in [
        0,
        1,
        tiles / 2,
        tiles.saturating_sub(2),
        tiles.saturating_sub(1),
    ] {
        if t < tiles {
            s.insert(t);
        }
    }
    let padded: Vec<usize> = (0..tiles)
        .filter(|t| sorted[t * BLOCK_M..(t + 1) * BLOCK_M].contains(&-1))
        .collect();
    for i in [0, padded.len() / 2, padded.len().saturating_sub(1)] {
        if let Some(t) = padded.get(i) {
            s.insert(*t);
        }
    }
    // The tiles holding the panel's smallest and largest slot ids: the largest slot at the highest
    // epoch group is where `g * L + slot` (the K3 record index) peaks.
    let live = |t: usize| {
        sorted[t * BLOCK_M..(t + 1) * BLOCK_M]
            .iter()
            .copied()
            .filter(|v| *v >= 0)
    };
    if let Some(t) = (0..tiles).max_by_key(|&t| live(t).max().unwrap_or(-1)) {
        s.insert(t);
    }
    if let Some(t) = (0..tiles).min_by_key(|&t| live(t).min().unwrap_or(i32::MAX)) {
        s.insert(t);
    }
    s.into_iter().collect()
}

struct EmuRun<'a> {
    emu: peacemaker_ir::emu::Emulator<'a>,
    block: u32,
}

impl EmuRun<'_> {
    /// Run the listed workgroups over `mem` with the kernarg already mapped at `kernarg`.
    fn exec(
        &self,
        mem: &mut peacemaker_ir::emu::Memory,
        kernarg: u64,
        wgs: &[[u32; 3]],
    ) -> Result<u64> {
        let mut n = 0;
        for wg in wgs {
            n += self
                .emu
                .run_workgroup_counted(
                    &peacemaker_ir::emu::Launch {
                        kernarg_ptr: kernarg,
                        workgroup: *wg,
                        block: [self.block, 1, 1],
                        dynamic_lds: 0,
                    },
                    mem,
                )
                .map_err(|e| format!("workgroup {wg:?}: {e}"))?;
        }
        Ok(n)
    }
}

fn region<'m>(
    mem: &'m peacemaker_ir::emu::Memory,
    name: &str,
) -> Result<&'m peacemaker_ir::emu::Region> {
    mem.regions
        .iter()
        .find(|r| r.name == name)
        .ok_or_else(|| format!("emulator memory lost the {name} region"))
}

/// Expert tables (`gate_up` or `down` bytes) of the experts the sampled tiles address; every other
/// table entry is the unmapped address 0, so a stray dereference is an error.
fn map_experts(
    mem: &mut peacemaker_ir::emu::Memory,
    e_total: usize,
    needed: &BTreeMap<usize, Vec<u8>>,
) -> Result<u64> {
    let mut bases: BTreeMap<usize, u64> = BTreeMap::new();
    for (e, bytes) in needed {
        bases.insert(
            *e,
            mem.map_guarded(format!("expert{e}"), bytes.clone(), EMU_GUARD)?,
        );
    }
    let mut table = Vec::with_capacity(e_total * 8);
    for e in 0..e_total {
        table.extend_from_slice(&bases.get(&e).copied().unwrap_or(0).to_le_bytes());
    }
    mem.map_guarded("table".into(), table, EMU_GUARD)
}

struct PanelEmu {
    json: Value,
    stat: RefStat,
    failures: Vec<String>,
}

impl Bench {
    /// One panel through the three PM launches on guarded exact-size buffers. `full`: every
    /// workgroup; else the sampled tiles (all x blocks of those tiles, all rows of those tiles for the
    /// producer).
    fn emu_panel(&self, progs: &EmuProgs, panel: &Panel, full: bool) -> Result<PanelEmu> {
        let sh = &self.shape;
        let g = &self.grouping;
        let (rows, tiles_n) = (panel.rows, panel.rows / BLOCK_M);
        let tiles = &g.tiles[panel.row_base / BLOCK_M..panel.row_base / BLOCK_M + tiles_n];
        let sorted = &g.sorted[panel.row_base..panel.row_base + rows];
        let gys: Vec<usize> = if full {
            (0..tiles_n).collect()
        } else {
            tile_sample(sorted, tiles_n)
        };
        let (t, slots) = (sh.t, sh.slots());
        let signs = Signs::new();
        let t_panel = std::time::Instant::now();
        let i32_le = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

        // The rows (and their tokens / experts) of the run tiles.
        let mut live: Vec<(usize, usize, usize)> = Vec::new(); // (row, slot, expert)
        let mut experts: BTreeSet<usize> = BTreeSet::new();
        for &gy in &gys {
            let e = tiles[gy];
            if e < 0 || e as usize >= sh.e {
                return Err(format!(
                    "panel {} tile {gy}: expert id {e} outside [0,{})",
                    panel.index, sh.e
                ));
            }
            experts.insert(e as usize);
            for r in gy * BLOCK_M..(gy + 1) * BLOCK_M {
                if sorted[r] >= 0 {
                    live.push((r, sorted[r] as usize, e as usize));
                }
            }
        }
        let mut gu_bytes: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
        let mut dn_bytes: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
        let mut weights: BTreeMap<usize, (Vec<u8>, Vec<u8>)> = BTreeMap::new();
        for &e in &experts {
            let (mut gu, mut dn) = (vec![0u8; GU_BYTES], vec![0u8; DN_BYTES]);
            gen_expert(sh.seed, e % sh.distinct, &mut gu, &mut dn);
            gu_bytes.insert(e, gu.clone());
            dn_bytes.insert(e, dn.clone());
            weights.insert(e, (gu, dn));
        }

        // The A4 window sidecar `[H/128][T]` of 72-byte records; only the run rows' tokens are filled.
        let mut xq_gate = vec![0u8; (H / GROUP) * t * REC];
        let mut x_rows: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
        for &(_, slot, _) in &live {
            let tok = slot / TOP;
            if x_rows.contains_key(&tok) {
                continue;
            }
            let mut x = vec![0f32; H];
            x_row(sh.seed, tok, &mut x);
            for (e, rec) in gate_records(&x, &signs).iter().enumerate() {
                let off = (e * t + tok) * REC;
                xq_gate[off..off + REC].copy_from_slice(rec);
            }
            x_rows.insert(tok, x);
        }
        let wg2 = |xs: usize| -> Vec<[u32; 3]> {
            gys.iter()
                .flat_map(|&gy| (0..xs as u32).map(move |gx| [gx, gy as u32, 0]))
                .collect()
        };
        let mut failures: Vec<String> = Vec::new();
        let mut json = json!({"panel": panel.index, "rows": rows, "row_base": panel.row_base,
            "experts": [panel.expert_lo, panel.expert_hi], "tiles_run": gys.len(), "tiles": tiles_n,
            "live_rows_run": live.len(), "full_grid": full});

        eprintln!(
            "moe_route emu: panel {} start rows={rows} tiles_run={} live_run={} experts={}",
            panel.index,
            gys.len(),
            live.len(),
            experts.len()
        );
        // Launch A: gate_up + SwiGLU.
        let gemm_gate = emu_make(&progs.gemm, GATE_IU4)?;
        let run_gate = EmuRun {
            emu: gemm_gate,
            block: IU4_NT1_BLOCK[0],
        };
        let mut mem = peacemaker_ir::emu::Memory::default();
        let table = map_experts(&mut mem, sh.e, &gu_bytes)?;
        let tiles_va = mem.map_guarded("tiles".into(), i32_le(tiles), EMU_GUARD)?;
        let sorted_va = mem.map_guarded("sorted".into(), i32_le(sorted), EMU_GUARD)?;
        let xq_len = xq_gate.len();
        let xq_va = mem.map_guarded("xq".into(), xq_gate, EMU_GUARD)?;
        let y_gu_len = rows * I * 2;
        let y_va = mem.map_guarded("y".into(), vec![0u8; y_gu_len], EMU_GUARD)?;
        let karg = gemm_kernarg(
            table,
            tiles_va,
            sorted_va,
            xq_va,
            y_va,
            [
                (2 * I) as u32,
                H as u32,
                TOP as u32,
                rows as u32,
                t as u32,
                sh.e as u32,
            ],
        );
        let kva = mem.map_guarded("kernarg".into(), karg, EMU_GUARD)?;
        let wgs_gate = wg2(2 * I / 64);
        let instr_gate = run_gate.exec(&mut mem, kva, &wgs_gate).map_err(|e| {
            format!(
                "panel {} gate_up (xq {xq_len} B, y {y_gu_len} B): {e}",
                panel.index
            )
        })?;
        let y_gu = region(&mem, "y")?;
        let gate_written = y_gu.written.count() as u64;
        let y_gu = y_gu.bytes.clone();
        drop(mem);

        eprintln!(
            "moe_route emu: panel {} gate_up done wgs={} instr={instr_gate} t={:.1}s",
            panel.index,
            wgs_gate.len(),
            t_panel.elapsed().as_secs_f64()
        );
        // Launch B: K3 down producer over the rows of the run tiles.
        let run_k3 = EmuRun {
            emu: emu_make(&progs.k3, rdna_compute::slab_moe::MQ_I4_A4_GFX1100_SYMBOL)?,
            block: 32,
        };
        let mut mem = peacemaker_ir::emu::Memory::default();
        let h_va = mem.map_guarded("h".into(), y_gu, EMU_GUARD)?;
        let sorted_va = mem.map_guarded("sorted".into(), i32_le(sorted), EMU_GUARD)?;
        let xd_len = (I / GROUP) * slots * REC;
        let xd_va = mem.map_guarded("xq".into(), vec![0u8; xd_len], EMU_GUARD)?;
        let s1 = mem.map_guarded("signs1".into(), f32_le(&signs.s1_128), EMU_GUARD)?;
        let s2 = mem.map_guarded("signs2".into(), f32_le(&signs.s2_128), EMU_GUARD)?;
        let kva = mem.map_guarded(
            "kernarg".into(),
            k3_kernarg(h_va, sorted_va, xd_va, s1, s2, rows as u32, slots as u32),
            EMU_GUARD,
        )?;
        let wgs_k3: Vec<[u32; 3]> = gys
            .iter()
            .flat_map(|&gy| {
                (gy * BLOCK_M..(gy + 1) * BLOCK_M).flat_map(|p| {
                    (0..(I / GROUP) as u32)
                        .map(move |gg| [p as u32 * (I / GROUP) as u32 + gg, 0, 0])
                })
            })
            .collect();
        let instr_k3 = run_k3.exec(&mut mem, kva, &wgs_k3).map_err(|e| {
            format!(
                "panel {} K3 producer (P {rows}, L {slots}, xq {xd_len} B): {e}",
                panel.index
            )
        })?;
        let xd = region(&mem, "xq")?;
        let k3_written = xd.written.count() as u64;
        let xd = xd.bytes.clone();
        drop(mem);

        eprintln!(
            "moe_route emu: panel {} K3 done wgs={} instr={instr_k3} t={:.1}s",
            panel.index,
            wgs_k3.len(),
            t_panel.elapsed().as_secs_f64()
        );
        // Launch C: down.
        let run_down = EmuRun {
            emu: emu_make(&progs.gemm, DOWN_IU4)?,
            block: IU4_NT1_BLOCK[1],
        };
        let mut mem = peacemaker_ir::emu::Memory::default();
        let table = map_experts(&mut mem, sh.e, &dn_bytes)?;
        let tiles_va = mem.map_guarded("tiles".into(), i32_le(tiles), EMU_GUARD)?;
        let sorted_va = mem.map_guarded("sorted".into(), i32_le(sorted), EMU_GUARD)?;
        let xq_va = mem.map_guarded("xq".into(), xd, EMU_GUARD)?;
        let y_dn_len = rows * H * 2;
        let y_va = mem.map_guarded("y".into(), vec![0u8; y_dn_len], EMU_GUARD)?;
        let karg = gemm_kernarg(
            table,
            tiles_va,
            sorted_va,
            xq_va,
            y_va,
            [
                H as u32,
                I as u32,
                1,
                rows as u32,
                slots as u32,
                sh.e as u32,
            ],
        );
        let kva = mem.map_guarded("kernarg".into(), karg, EMU_GUARD)?;
        let wgs_dn = wg2(H / 64);
        let instr_dn = run_down.exec(&mut mem, kva, &wgs_dn).map_err(|e| {
            format!(
                "panel {} down (xq {xd_len} B, y {y_dn_len} B): {e}",
                panel.index
            )
        })?;
        let yd = region(&mem, "y")?;
        let down_written = yd.written.count() as u64;
        let y_down = yd.bytes.clone();
        drop(mem);

        eprintln!(
            "moe_route emu: panel {} down done wgs={} instr={instr_dn} t={:.1}s",
            panel.index,
            wgs_dn.len(),
            t_panel.elapsed().as_secs_f64()
        );
        // Coverage and the exact-recipe check of every live row of the run tiles.
        let nl = live.len() as u64;
        for (what, got, need) in [
            ("gate_up y", gate_written, nl * (I as u64) * 2),
            ("K3 xq", k3_written, nl * ((I / GROUP) * REC) as u64),
            ("down y", down_written, nl * (H as u64) * 2),
        ] {
            if got < need {
                failures.push(format!(
                    "panel {}: {what} wrote {got} B, a run of {nl} live rows needs >= {need}",
                    panel.index
                ));
            }
        }
        let refs: Vec<(usize, Vec<f32>)> = live
            .par_iter()
            .map(|&(r, slot, e)| {
                let (gu, dn) = &weights[&e];
                (r, a4_slot(&x_rows[&(slot / TOP)], gu, dn, &signs))
            })
            .collect();
        let mut stat = RefStat::default();
        for (r, a) in &refs {
            let got: Vec<f32> = (0..H).map(|c| bf16_at(&y_down, r * H + c)).collect();
            let af: Vec<f64> = a.iter().map(|v| f64::from(*v)).collect();
            stat.add_slot(&got, &af);
        }
        let bound = self.tune.a4_recipe_bound;
        if stat.nonfinite > 0 || !(stat.rel_l2() <= bound) {
            failures.push(format!(
                "panel {}: down output vs the exact A4 recipe rel_l2 {} (bound {bound}), {} non-finite",
                panel.index,
                stat.rel_l2(),
                stat.nonfinite
            ));
        }
        json["workgroups"] =
            json!({"gate_up": wgs_gate.len(), "k3": wgs_k3.len(), "down": wgs_dn.len()});
        json["instructions"] = json!({"gate_up": instr_gate, "k3": instr_k3, "down": instr_dn});
        json["bytes_written"] =
            json!({"gate_up_y": gate_written, "k3_xq": k3_written, "down_y": down_written});
        json["vs_a4_recipe"] = stat.json(bound);
        Ok(PanelEmu {
            json,
            stat,
            failures,
        })
    }

    /// The pre-card emulator gate: the small shape in full, then the real shape's first / middle /
    /// last panel at the grid extremes. Fails closed: any refusal, access outside a guarded buffer,
    /// short coverage or recipe mismatch is a `fail`.
    pub fn emu_gate(&self) -> Value {
        let t0 = std::time::Instant::now();
        if self.aa {
            return json!({"status": "pass", "reason": "--null-control aa launches no PM kernel; the F16 \
                chain is hipcc code (not emulated)"});
        }
        let progs = match emu_progs() {
            Ok(p) => p,
            Err(e) => return json!({"status": "fail", "reason": format!("emulator setup: {e}")}),
        };
        eprintln!(
            "moe_route emu: objects lifted in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        let small = Shape {
            t: 24,
            e: 12,
            distinct: 12,
            panel_rows: 64,
            bind: 2,
            panels: PanelSel::All,
            route: RouteKind::Uniform,
            seed: self.shape.seed,
        };
        let mut cases = Vec::new();
        let mut failures: Vec<String> = Vec::new();
        let small_bench = Bench::from_parts(small, false, self.tune);
        let real: Vec<usize> = {
            let n = self.panels.len();
            // Quartiles, and the panels that hold the smallest and largest slot ids of the window.
            let slot_panel = |slot: usize| -> Option<usize> {
                let row = self
                    .grouping
                    .sorted
                    .iter()
                    .position(|s| *s >= 0 && *s as usize == slot)?;
                self.panels
                    .iter()
                    .position(|p| row >= p.row_base && row < p.row_base + p.rows)
            };
            let mut v = vec![0, n / 4, n / 2, 3 * n / 4, n.saturating_sub(1)];
            v.extend(slot_panel(0));
            v.extend(slot_panel(self.shape.slots() - 1));
            v.sort_unstable();
            v.dedup();
            v
        };
        let plan: Vec<(&str, &Bench, Vec<usize>, bool)> = match &small_bench {
            Ok(b) => vec![
                ("small_full", b, (0..b.panels.len()).collect(), true),
                ("real_sampled", self, real, false),
            ],
            Err(_) => vec![("real_sampled", self, real, false)],
        };
        if let Err(e) = &small_bench {
            failures.push(format!("small shape: {e}"));
        }
        for (name, b, idx, full) in plan {
            let mut panels = Vec::new();
            let mut agg = RefStat::default();
            for i in idx {
                match b.emu_panel(&progs, &b.panels[i], full) {
                    Ok(p) => {
                        failures.extend(p.failures.iter().map(|f| format!("{name}: {f}")));
                        agg.n += p.stat.n;
                        agg.sum_d2 += p.stat.sum_d2;
                        agg.sum_r2 += p.stat.sum_r2;
                        agg.max_abs = agg.max_abs.max(p.stat.max_abs);
                        panels.push(p.json);
                    }
                    Err(e) => {
                        let e = e.split(" | ").next().unwrap_or(&e).to_owned();
                        failures.push(format!("{name}: {e}"));
                        panels.push(json!({"panel": b.panels[i].index, "error": e}));
                    }
                }
            }
            cases.push(
                json!({"name": name, "shape": b.shape_json()["schedule"].clone(),
                "T": b.shape.t, "E": b.shape.e, "full_grid": full, "panels": panels,
                "vs_a4_recipe": agg.json(self.tune.a4_recipe_bound)}),
            );
        }
        json!({
            "status": if failures.is_empty() { "pass" } else { "fail" },
            "reason": if failures.is_empty() { "ok".to_owned() } else { failures.join(" | ") },
            "emulated": ["qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1100", "mq_i4_a4_r128_k640_gfx1100",
                         "qwen4_moe_down_iu4_sym_pm_gfx1100"],
            "not_emulated": ["mq_rotate_x_i4 (hipcc window producer; sidecar from the host recipe)",
                             "the F16 arm (hipcc kernels)"],
            "model": "peacemaker emulator, gfx1100 program on the gfx1151 model (gfx1100_as_gfx1151), strict LDS; \
                      exact-size guarded buffers (every access past an end is an error)",
            "guard_bytes": EMU_GUARD,
            "cases": cases,
            "seconds": t0.elapsed().as_secs_f64(),
        })
    }
}

// ---------------------------------------------------------------------------
// Driver

/// CLI admission (called by `parse_args`): this kernel is gfx1100 only and has no `.rip` or
/// candidate-object medium. `--emu` is accepted: the emulator gate always runs before a device is
/// opened (and on request with `--offline`).
pub fn check_cli(arch: &str, _emu: bool) -> Result<()> {
    if arch != ARCH {
        return Err(format!(
            "--kernel {KERNEL} is gfx1100 only: pass --arch {ARCH} (got --arch {arch})"
        ));
    }
    Ok(())
}

fn objects_record(
    bench: &Bench,
    o: &Opts,
    rec: &mut Value,
) -> std::result::Result<(), (Value, Outcome)> {
    let out_dir = crate::out_dir_of(o);
    let base = bench
        .pm_objects(Arm::Baseline)
        .map_err(|e| crate::fail(rec.clone(), "baseline", e))?;
    let (base_rows, base_fail) = crate::certify_chain_objects(&out_dir, "baseline", &base, ARCH);
    rec["baseline"] = json!({"pm_objects": base_rows});
    let cand = bench
        .pm_objects(Arm::Candidate)
        .map_err(|e| crate::fail(rec.clone(), "emit", e))?;
    let (cand_rows, cand_fail) = crate::certify_chain_objects(&out_dir, "candidate", &cand, ARCH);
    rec["candidate"] = json!({"pm_objects": cand_rows});
    // The baseline has no PM object; an unsupported candidate certification gates.
    let _ = base_fail;
    if let Some((stage, msg)) = cand_fail {
        return Err(crate::fail(rec.clone(), stage, msg));
    }
    Ok(())
}

/// The whole run (offline plan or the GPU judge).
pub fn run(o: &mut Opts) -> (Value, Outcome) {
    let mut rec = json!({"schema": crate::SCHEMA, "kernel": KERNEL, "arch": ARCH, "mode": "eager"});
    if let Err(e) = check_cli(o.arch, o.emu) {
        return crate::fail(rec, "args", e);
    }
    let shape = o.shape.clone().unwrap_or_default();
    let mut knobs = std::mem::take(&mut o.knobs);
    let mut bench = match new(&shape, &mut knobs, o.null_control).and_then(|b| {
        knobs.finish()?;
        Ok(b)
    }) {
        Ok(b) => b,
        Err(e) => return crate::fail(rec, "args", e),
    };
    rec["shape"] = bench.shape_json();
    rec["numeric_contract"] = bench.numeric_contract();
    rec["chain"] = bench.plan_json();
    if let Err(r) = objects_record(&bench, o, &mut rec) {
        return r;
    }
    // Pre-card gate: GPU mode ALWAYS runs it (fail closed, before `Gpu::init`); `--offline` only
    // with `--emu`.
    if o.emu || !o.offline {
        let gate = bench.emu_gate();
        let pass = gate["status"] == "pass";
        let reason = gate["reason"].as_str().unwrap_or("").to_owned();
        rec["emu_gate"] = gate;
        if !pass {
            return crate::fail(rec, "emu_gate", reason);
        }
    }
    if o.offline {
        rec["offline"] = json!(true);
        return (rec, Outcome::Ok);
    }

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => return crate::fail(rec, "gpu_init", err(e)),
    };
    rec["device"] = crate::device_json(&gpu);
    if gpu.arch != ARCH {
        let msg = format!("device is {}, {KERNEL} needs {ARCH}", gpu.arch);
        return crate::fail(rec, "device", msg);
    }
    if let Err(e) = bench.setup(&mut gpu) {
        return crate::fail(rec, "setup", e);
    }
    let base_sha = sha256_hex(&serde_json::to_vec(&rec["chain"]["baseline"]).unwrap_or_default());
    let cand_sha = sha256_hex(&serde_json::to_vec(&rec["chain"]["candidate"]).unwrap_or_default());
    rec["baseline"]["sha256"] = json!(base_sha);
    rec["candidate"]["sha256"] = json!(cand_sha);
    if o.null_control {
        rec["null_control"] = json!({
            "kind": "aa", "identical_chains": base_sha == cand_sha, "separate_modules": false,
            "baseline_sha256": base_sha, "candidate_sha256": cand_sha,
            "note": "candidate arm runs the baseline chain through the same launchers and loaded modules on its own buffer set",
        });
    }

    let (corr, pass, hashes) = match bench.correctness(&mut gpu) {
        Ok(v) => v,
        Err(e) => return crate::fail(rec, "launch", e),
    };
    rec["correctness"] = corr;
    if !pass {
        rec["timing"] = json!({"status": "refused: failed correctness"});
        return (rec, Outcome::CorrectnessFailed);
    }
    match bench.stage_profile(&mut gpu) {
        Ok(v) => rec["stages"] = v,
        Err(e) => return crate::fail(rec, "stages", e),
    }
    match bench.time(&mut gpu, o, &base_sha) {
        Ok(t) => rec["timing"] = t,
        Err(e) => return crate::fail(rec, "timing", e),
    }
    match bench.hashes(&mut gpu) {
        Ok(h) if h == hashes => {}
        Ok(_) => {
            rec["timing"]["post_outputs_unchanged"] = json!(false);
            rec["timing"]["status"] = json!("post-sampling outputs changed; no performance claim");
            return (rec, Outcome::CorrectnessFailed);
        }
        Err(e) => return crate::fail(rec, "timing", e),
    }
    rec["device_after"] = crate::device_json(&gpu);
    (rec, Outcome::Ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(s: &str) -> Shape {
        parse_shape(s).unwrap()
    }

    fn bench(s: &str) -> Bench {
        new(s, &mut Knobs::default(), false).unwrap()
    }

    #[test]
    fn shape_defaults_and_keys() {
        let d = shape("fn");
        assert_eq!(
            (
                d.t,
                d.e,
                d.distinct,
                d.panel_rows,
                d.bind,
                d.panels,
                d.route
            ),
            (16896, 512, 512, 2048, 32, PanelSel::All, RouteKind::Uniform)
        );
        assert_eq!(parse_shape("").unwrap(), d);
        let s = shape("fn,T=1024,E=64,D=8,panel=256,bind=8,panels=3,route=skew,seed=7");
        assert_eq!(
            (s.t, s.e, s.distinct, s.panel_rows, s.bind),
            (1024, 64, 8, 256, 8)
        );
        assert_eq!(
            (s.panels, s.route, s.seed),
            (PanelSel::First(3), RouteKind::Skew, 7)
        );
        assert_eq!(shape("T=2048").t, 2048);
    }

    #[test]
    fn shape_refusals() {
        for (bad, why) in [
            ("fn,T=511", "T >= 512"),
            ("a3b", "only the `fn`"),
            ("fn,T=1000000", "QWEN4_MOE_SYM_PM_MAX_SLOTS"),
            ("fn,E=9", "top-10"),
            ("fn,E=2000", "<= E"),
            ("fn,E=64,D=65", "1..=E"),
            ("fn,panel=100", "multiple of 16"),
            ("fn,panel=0", "multiple of 16"),
            ("fn,bind=0", "at least one expert"),
            ("fn,panels=0", "panels=0"),
            ("fn,T=1024,T=2048", "given twice"),
            ("fn,bogus=1", "expected T, E"),
            ("fn,route=zipf", "uniform|skew"),
            ("fn,T=abc", "not a non-negative"),
        ] {
            let e = parse_shape(bad).unwrap_err();
            assert!(e.contains(why), "{bad}: {e}");
        }
    }

    #[test]
    fn grouping_is_stable_padded_and_tiled() {
        // T=3 tokens, E=4 experts (grouping is shape-agnostic).
        let topk = [2, 0, 1, 2, 3, 1, 2, 0, 3];
        let g = Grouping::new(&topk, 4).unwrap();
        assert_eq!(g.padded, vec![16, 16, 16, 16]);
        assert_eq!(g.offsets, vec![0, 16, 32, 48, 64]);
        assert_eq!(g.total_rows, 64);
        assert_eq!(g.bound, grouped_bound(9, 4));
        assert_eq!(g.bound, (9 + 15 * 4usize).div_ceil(16) * 16);
        // expert 0 holds slots 1 and 7, ascending, then padding.
        assert_eq!(&g.sorted[0..3], &[1, 7, -1]);
        assert_eq!(&g.sorted[16..19], &[2, 5, -1]);
        assert_eq!(&g.sorted[32..36], &[0, 3, 6, -1]);
        assert_eq!(&g.sorted[48..51], &[4, 8, -1]);
        assert!(g.sorted[64..].iter().all(|s| *s == -1));
        assert_eq!(&g.tiles[..4], &[0, 1, 2, 3]);
        assert!(g.tiles[4..].iter().all(|t| *t == -1));
        assert_eq!(g.expert_of_row(0), 0);
        assert_eq!(g.expert_of_row(31), 1);
        assert_eq!(g.expert_of_row(63), 3);
        assert!(Grouping::new(&[0, 4], 4).is_err());
        assert!(Grouping::new(&[-1], 4).is_err());
    }

    #[test]
    fn route_table_is_valid_distinct_and_skewed() {
        let s = shape("fn,T=600");
        let t = route_table(&s);
        assert_eq!(t.len(), 6000);
        for row in t.chunks(TOP) {
            let mut r = row.to_vec();
            r.sort();
            r.dedup();
            assert_eq!(r.len(), TOP);
            assert!(row.iter().all(|e| (0..512).contains(e)));
        }
        let k = route_table(&shape("fn,T=600,route=skew"));
        for row in k.chunks(TOP) {
            assert_eq!(&row[..5], &[0, 1, 2, 3, 4]);
        }
        assert_eq!(route_table(&s), t, "deterministic");
    }

    #[test]
    fn schedule_follows_the_production_panel_rules() {
        // Hand counts: 6 experts, bindings of 4: rows never cross a binding; 48-row panel bound.
        let g = Grouping {
            experts: 6,
            slots: 0,
            bound: 0,
            padded: vec![32, 0, 64, 16, 16, 0],
            offsets: vec![0, 32, 32, 96, 112, 128, 128],
            total_rows: 128,
            sorted: Vec::new(),
            tiles: Vec::new(),
        };
        let p = schedule(&g, 4, 48);
        // binding 0 = experts 0..4, rows [0, 112): 48 + 48 + 16; binding 1 = experts 4..6, rows
        // [112, 128): 16.
        let rows: Vec<(usize, usize, usize)> =
            p.iter().map(|p| (p.binding, p.row_base, p.rows)).collect();
        assert_eq!(
            rows,
            vec![(0, 0, 48), (0, 48, 48), (0, 96, 16), (1, 112, 16)]
        );
        assert_eq!((p[0].expert_lo, p[0].expert_hi), (0, 3));
        assert_eq!((p[2].expert_lo, p[2].expert_hi), (3, 4));
        assert_eq!((p[3].expert_lo, p[3].expert_hi), (4, 5));
        assert!(p.iter().enumerate().all(|(i, p)| p.index == i));
        // A single expert larger than the bound splits across panels.
        let big = Grouping {
            experts: 1,
            slots: 0,
            bound: 0,
            padded: vec![130 / 16 * 16 + 16],
            offsets: vec![0, 144],
            total_rows: 144,
            sorted: Vec::new(),
            tiles: Vec::new(),
        };
        let rows: Vec<usize> = schedule(&big, 32, 64).iter().map(|p| p.rows).collect();
        assert_eq!(rows, vec![64, 64, 16]);
    }

    #[test]
    fn fn_window_schedule_matches_pf1() {
        let b = bench("fn");
        let g = &b.grouping;
        assert_eq!(g.slots, 168_960);
        assert_eq!(g.bound, grouped_bound(168_960, 512));
        let total: usize = b.panels.iter().map(|p| p.rows).sum();
        assert_eq!(
            total, g.total_rows,
            "panels cover every grouped row exactly once"
        );
        for w in b.panels.windows(2) {
            assert_eq!(w[1].row_base, w[0].row_base + w[0].rows, "contiguous");
            assert!(w[1].binding >= w[0].binding);
        }
        // PF-1: 4009 calls over 42 host layers = ~95.5 panels per layer.
        assert!(
            (90..=102).contains(&b.panels.len()),
            "{} panels",
            b.panels.len()
        );
        assert_eq!(
            b.panels[0].rows, 2048,
            "the first panel is a full 2048-row panel"
        );
        assert!(b.panels.iter().all(|p| p.rows <= 2048 && p.rows % 16 == 0));
        assert_eq!(b.run, b.panels.len());
        assert_eq!(bench("fn,panels=1").run, 1);
        assert_eq!(bench("fn,panels=100000").run, b.panels.len());
    }

    #[test]
    fn pf1_grids_for_a_2048_row_panel() {
        // PF-1 (rocprof threads): gate_up (2560, 128, 1), down (5120, 128, 1).
        let gu = geometry(Route::F16, Kind::GateUp, 2048).unwrap();
        assert_eq!(
            (gu.blocks, gu.block, gu.threads()),
            ([80, 128, 1], 32, [2560, 128, 1])
        );
        let dn = geometry(Route::F16, Kind::Down, 2048).unwrap();
        assert_eq!(
            (dn.blocks, dn.block, dn.threads()),
            ([160, 128, 1], 32, [5120, 128, 1])
        );
        // The IU4 NT1 entries (block 64 / 128).
        let gu = geometry(Route::Iu4, Kind::GateUp, 2048).unwrap();
        assert_eq!(
            (gu.blocks, gu.block, gu.threads()),
            ([20, 128, 1], 64, [1280, 128, 1])
        );
        let dn = geometry(Route::Iu4, Kind::Down, 2048).unwrap();
        assert_eq!(
            (dn.blocks, dn.block, dn.threads()),
            ([40, 128, 1], 128, [5120, 128, 1])
        );
        // Window producers: one wave per 256-group, T rows.
        assert_eq!(
            geometry(Route::F16, Kind::XProduce, 16896).unwrap().blocks,
            [(2560 / 256 * 16896) as u32, 1, 1]
        );
        assert_eq!(
            geometry(Route::Iu4, Kind::DownProduce, 2048)
                .unwrap()
                .blocks,
            [5 * 2048, 1, 1]
        );
        assert_eq!(
            geometry(Route::F16, Kind::Unscatter, 2048).unwrap().blocks,
            [2048, 5, 1]
        );
        assert_eq!(
            geometry(Route::F16, Kind::Remap, 2048).unwrap().blocks,
            [8, 1, 1]
        );
        assert!(geometry(Route::F16, Kind::XAppend, 16896).is_none());
        // A short panel keeps the same X and shrinks Y.
        let short = geometry(Route::F16, Kind::GateUp, 512).unwrap();
        assert_eq!(short.threads(), [2560, 32, 1]);
    }

    #[test]
    fn offline_plan_names_every_stage_of_both_chains() {
        let b = bench("fn,panels=1");
        let plan = b.plan_json();
        let base = &plan["baseline"];
        let cand = &plan["candidate"];
        let names = |c: &Value, scope: &str| -> Vec<String> {
            c[scope]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s["symbols"][0].as_str().unwrap().to_owned())
                .collect()
        };
        assert_eq!(
            names(base, "window_stages"),
            vec![
                "mq_rotate_x_f16",
                "hipMemcpyDtoDAsync (__amd_rocclr_copyBuffer)"
            ]
        );
        assert_eq!(
            names(base, "panel_stages"),
            vec![
                GATE_F16,
                "slowlink_remap_sorted",
                "moe_unscatter_rotate128_f16",
                DOWN_F16
            ]
        );
        assert_eq!(names(cand, "window_stages"), vec!["mq_rotate_x_i4"]);
        assert_eq!(
            names(cand, "panel_stages"),
            vec![GATE_IU4, "mq_i4_a4_r128_k640_gfx1100", DOWN_IU4]
        );
        // The grids printed for the full-size first panel are PF-1's.
        let g = |c: &Value, i: usize| c["panel_stages"][i]["grid_threads"].clone();
        assert_eq!(g(base, 0), json!([2560, 128, 1]));
        assert_eq!(g(base, 3), json!([5120, 128, 1]));
        assert_eq!(g(cand, 0), json!([1280, 128, 1]));
        assert_eq!(g(cand, 2), json!([5120, 128, 1]));
        assert_eq!(base["calls"]["panel"], 1);
        assert_eq!(base["launches_per_chain"], 2 + 4);
        assert_eq!(cand["launches_per_chain"], 1 + 3);
        let origin = |c: &Value| -> Vec<String> {
            c["panel_stages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s["origin"].as_str().unwrap().to_owned())
                .collect()
        };
        assert_eq!(origin(cand), vec!["pm", "pm", "pm"]);
        assert_eq!(origin(base), vec!["hip"; 4]);
        // The shape record carries the schedule.
        let s = b.shape_json();
        assert_eq!(s["schedule"]["panels_run"], 1);
        assert_eq!(s["schedule"]["first"]["rows"], 2048);
        assert_eq!(s["T"], 16896);
    }

    #[test]
    fn null_control_runs_the_baseline_chain_as_candidate() {
        let mut k = Knobs::default();
        let b = new("fn,panels=1", &mut k, true).unwrap();
        assert_eq!(b.arm_route(Arm::Candidate), Route::F16);
        assert!(b.pm_objects(Arm::Candidate).unwrap().is_empty());
        let plan = b.plan_json();
        assert_eq!(plan["baseline"]["route"], plan["candidate"]["route"]);
    }

    #[test]
    fn candidate_pm_objects_are_the_shipped_gfx1100_bundles() {
        let b = bench("fn,panels=1");
        assert!(b.pm_objects(Arm::Baseline).unwrap().is_empty());
        let objs = b.pm_objects(Arm::Candidate).unwrap();
        assert_eq!(objs.len(), 2);
        assert_eq!(objs[0].launch_symbol, GATE_IU4);
        assert!(objs[0].exports.iter().any(|e| e == DOWN_IU4));
        assert_eq!(objs[1].launch_symbol, "mq_i4_a4_r128_k640_gfx1100");
    }

    #[test]
    fn buffer_sets_alternate_with_the_crossover_binding() {
        let mut b = bench("fn,panels=1");
        assert_eq!(
            (b.set_index(Arm::Baseline), b.set_index(Arm::Candidate)),
            (0, 1)
        );
        b.swapped = true;
        assert_eq!(
            (b.set_index(Arm::Baseline), b.set_index(Arm::Candidate)),
            (1, 0)
        );
    }

    #[test]
    fn knobs_parse_and_leftovers_are_visible() {
        let mut k = Knobs::default();
        k.0.insert("refslots".into(), "7".into());
        k.0.insert("stage-reps".into(), "3".into());
        k.0.insert("f16-bound".into(), "0.1".into());
        k.0.insert("bogus".into(), "1".into());
        let b = new("fn,panels=1", &mut k, false).unwrap();
        assert_eq!(
            (b.tune.refslots, b.tune.stage_reps, b.tune.f16_bound),
            (7, 3, 0.1)
        );
        assert!(k.finish().unwrap_err().contains("--bogus"));
        let mut k = Knobs::default();
        k.0.insert("refslots".into(), "0".into());
        assert!(new("fn,panels=1", &mut k, false).is_err());
        let mut k = Knobs::default();
        k.0.insert("a4-bound".into(), "nan".into());
        assert!(new("fn,panels=1", &mut k, false).is_err());
    }

    #[test]
    fn cli_admission() {
        let args = |s: &str| crate::parse_args(s.split_whitespace().map(str::to_owned));
        let o =
            args("--kernel qwen4_moe_route --arch gfx1100 --offline --shape fn,T=16896").unwrap();
        assert!(o.offline && o.kernel == KERNEL && o.arch == ARCH);
        assert!(args("--kernel qwen4_moe_route --arch gfx1100 --null-control aa").is_ok());
        // Default arch (gfx1201) and the other arches are refused.
        for bad in [
            "--kernel qwen4_moe_route",
            "--kernel qwen4_moe_route --arch gfx1151",
            "--kernel qwen4_moe_route --arch gfx1100 --rip x.rip",
        ] {
            assert!(args(bad).is_err(), "{bad}");
        }
        let e = args("--kernel qwen4_moe_route").unwrap_err();
        assert!(e.contains("gfx1100 only"), "{e}");
    }

    #[test]
    fn generator_is_symmetric_and_distinct() {
        let (mut gu, mut dn) = (vec![0u8; GU_BYTES], vec![0u8; DN_BYTES]);
        gen_expert(1, 3, &mut gu, &mut dn);
        check_symmetric(&gu, &dn).unwrap();
        let (mut gu2, mut dn2) = (vec![0u8; GU_BYTES], vec![0u8; DN_BYTES]);
        gen_expert(1, 4, &mut gu2, &mut dn2);
        assert_ne!(gu, gu2);
        gen_expert(1, 3, &mut gu2, &mut dn2);
        assert_eq!((gu, dn), (gu2, dn2), "deterministic");
        let mut bad = vec![0u8; 136];
        bad[0] = 0x00;
        bad[1] = 0x3c; // sc = 1.0
        bad[2] = 0x00;
        bad[3] = 0x3c; // zp = 1.0 != -8
        assert!(check_symmetric(&bad, &[]).is_err());
    }

    #[test]
    fn pick_evenly_includes_the_ends() {
        let live: Vec<usize> = (0..100).collect();
        assert_eq!(pick_evenly(&live, 1), vec![0]);
        assert_eq!(pick_evenly(&live, 2), vec![0, 99]);
        let p = pick_evenly(&live, 5);
        assert_eq!((p[0], *p.last().unwrap(), p.len()), (0, 99, 5));
        assert_eq!(pick_evenly(&live, 500), live);
    }

    #[test]
    fn dist_streams_rel_l2_and_flags_nonfinite() {
        let bf = |v: f32| bf16_bits(v).to_le_bytes();
        let bytes = |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| bf(*v)).collect() };
        // One live row of width H: base = 1.0 everywhere, cand = 1.0 except one element 2.0.
        let base = bytes(&vec![1.0; H]);
        let mut c = vec![1.0f32; H];
        c[5] = 2.0;
        let cand = bytes(&c);
        let d = panel_dist(&base, &cand, &[true], 3);
        assert_eq!(d.elems, H as u64);
        assert_eq!(d.identical, H as u64 - 1);
        assert!((d.rel_l2() - (1.0 / H as f64).sqrt()).abs() < 1e-12);
        assert_eq!(d.max_abs, 1.0);
        assert_eq!(d.worst_panel.map(|w| w.1), Some(3));
        // A dead row is ignored; a NaN is counted and excluded from the norms.
        let mut c2 = vec![1.0f32; 2 * H];
        c2[H + 1] = f32::NAN;
        let d = panel_dist(&bytes(&vec![1.0; 2 * H]), &bytes(&c2), &[false, true], 0);
        assert_eq!(
            (d.elems, d.nonfinite_cand, d.nonfinite_base),
            (H as u64, 1, 0)
        );
        let m = merge_dist(
            panel_dist(&base, &cand, &[true], 3),
            panel_dist(&base, &bytes(&vec![3.0; H]), &[true], 9),
        );
        assert_eq!(m.worst_panel.map(|w| w.1), Some(9));
    }

    #[test]
    fn median_and_ref_stat() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median(&mut []).is_nan());
        let mut s = RefStat::default();
        s.add_slot(&[1.0, 2.0], &[1.0, 2.0]);
        assert_eq!((s.rel_l2(), s.max_abs), (0.0, 0.0));
        s.add_slot(&[f32::INFINITY, 1.0], &[0.0, 1.0]);
        assert_eq!(s.nonfinite, 1);
        assert_eq!(s.json(0.5)["pass"], false);
    }

    #[test]
    fn a4_recipe_tracks_the_f64_chain_within_quantisation_noise() {
        let signs = Signs::new();
        let (mut gu, mut dn) = (vec![0u8; GU_BYTES], vec![0u8; DN_BYTES]);
        gen_expert(DEFAULT_SEED, 17, &mut gu, &mut dn);
        let mut x = vec![0f32; H];
        x_row(DEFAULT_SEED, 5, &mut x);
        let f = f64_slot(&x, &gu, &dn, &signs);
        let a = a4_slot(&x, &gu, &dn, &signs);
        assert_eq!((f.len(), a.len()), (H, H));
        let mut s = RefStat::default();
        s.add_slot(&a, &f);
        assert_eq!(s.nonfinite, 0);
        eprintln!("a4 recipe vs f64 chain on one slot: rel_l2 {}", s.rel_l2());
        // Two 4-bit activation quantisations: a real but bounded distance (module doc ~0.1-0.2).
        assert!(
            s.rel_l2() > 1e-3,
            "A4 cannot equal the unquantised chain: {}",
            s.rel_l2()
        );
        assert!(s.rel_l2() < DEFAULT_A4_BOUND, "{}", s.rel_l2());
        // BF16-valued output.
        assert!(a.iter().all(|v| bf16_bits(*v) == bf16_bits(bf16r(*v))));
    }

    fn small_shape() -> Shape {
        Shape {
            t: 24,
            e: 12,
            distinct: 12,
            panel_rows: 64,
            bind: 2,
            panels: PanelSel::All,
            route: RouteKind::Uniform,
            seed: DEFAULT_SEED,
        }
    }

    #[test]
    fn from_parts_refuses_shapes_the_route_generator_cannot_serve() {
        let tune = bench("fn,panels=1").tune;
        let mut s = small_shape();
        s.e = TOP - 1; // TOP distinct experts per token cannot be drawn: would loop forever
        assert!(Bench::from_parts(s, false, tune).is_err());
        let mut s = small_shape();
        s.bind = 0;
        assert!(Bench::from_parts(s, false, tune).is_err());
        assert!(Bench::from_parts(small_shape(), false, tune).is_ok());
    }

    #[test]
    fn tile_sample_includes_the_slot_extremes_and_pad_tiles() {
        // 8 tiles; tile 5 holds the largest slot, tile 6 the smallest, tile 3 has a pad row.
        let mut sorted = vec![100i32; 8 * BLOCK_M];
        for (i, s) in sorted.iter_mut().enumerate() {
            *s = 100 + i as i32;
        }
        sorted[5 * BLOCK_M] = 9_999;
        sorted[6 * BLOCK_M] = 1;
        sorted[3 * BLOCK_M + 4] = -1;
        let t = tile_sample(&sorted, 8);
        for want in [0, 1, 3, 4, 5, 6, 7] {
            assert!(t.contains(&want), "{want} in {t:?}");
        }
        assert!(t.windows(2).all(|w| w[0] < w[1]), "sorted unique {t:?}");
    }

    /// The K3 producer's record index peaks where the largest slot meets the highest epoch group:
    /// the panel holding slot S-1 at the real window must run clean (it faulted before the
    /// generator fix; `peacemaker-isa/tests/mq_i4_a4_gfx1100_xq_extent.rs` pins the instruction order).
    #[test]
    fn emulator_gate_runs_the_real_windows_largest_slot_panel_clean() {
        let b = bench("fn");
        let progs = emu_progs().unwrap();
        let row = b
            .grouping
            .sorted
            .iter()
            .position(|s| *s as usize == b.shape.slots() - 1)
            .unwrap();
        let panel = *b
            .panels
            .iter()
            .find(|p| row >= p.row_base && row < p.row_base + p.rows)
            .unwrap();
        let r = b
            .emu_panel(&progs, &panel, false)
            .expect("guarded exact-size buffers: no stray access");
        assert!(r.failures.is_empty(), "{:?}", r.failures);
        // The device recipe is bit-exact against the CPU A4 recipe here (the fold is exact int32).
        assert_eq!(r.stat.rel_l2(), 0.0);
        assert!(r.stat.n > 0);
    }

    #[test]
    fn emulator_gate_small_shape_full_grid_is_bit_exact() {
        let b = Bench::from_parts(small_shape(), false, bench("fn,panels=1").tune).unwrap();
        let progs = emu_progs().unwrap();
        for p in b.panels.iter().take(2) {
            let r = b.emu_panel(&progs, p, true).unwrap();
            assert!(r.failures.is_empty(), "{:?}", r.failures);
            assert_eq!(r.stat.rel_l2(), 0.0);
        }
    }
}
