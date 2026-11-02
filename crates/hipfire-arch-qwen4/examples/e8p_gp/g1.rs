// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real-layer E8P G1 (`g1.e8p.real`): every one of a layer's 512 real E8P experts resident in one slab, the shipped PM
//! gate/up and down kernels and their `_acc` twins launched through raw PM module launches (the real-G1 method), and
//! every output compared bit for bit with the K-E0 CPU twin:
//!  * F32 `_acc` accumulators == `Oracle::dense_rows` (unpack_word_ref nibbles, MAGIC seed, fold128, epoch_sum);
//!  * BF16 output == the frozen epilogue DAG (`grouped_prefill_epilogue_with`) over the twin F32 sums with the device
//!    `v_exp_f32` / `v_ldexp_f32` / IEEE-divide probes (`Hook::dag_epilogue`; never `PortableEpilogue`).
//!
//! Scenarios (real weights, synthetic bounded/stress activations; the receipt states the real scope):
//!  * `coverage_*`: ONE tile per expert for ALL 512 experts (non-monotone hashed expert order, mixed live/pad slots,
//!    sentinel tiles), bounded x with x_row_div 1 and stress x with x_row_div 10 -- the all-512-resident qualification.
//!  * `parity_*`: the real-G1 per-expert scenario set (rows 1/2/7/8/15/16/17 x x_row_div 1/10, non-monotone slots, pad
//!    slots, sentinel tiles, stress activations) plus a mixed-expert non-monotone tile order, on `--g1-experts` hashed experts.
//!  * `route_head`: (only with `--trace-dir`) the first `--route-tokens` tokens of the recorded real route, real slot ids
//!    `token * topk + j`, padded tiles per expert, x_row_div topk (gate/up) / 1 (down), bounded x.

use std::collections::{BTreeMap, BTreeSet};

use hip_bridge::{DeviceBuffer, HipRuntime};
use hipfire_runtime::mq_i4_reference::A4Block;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::common::*;
use super::hw::{E8pModule, Hook};
use super::records::{E8pLayer, EXPERTS};
use super::route::Route;
use super::twin::*;

pub const SCHEMA: &str = "g1.e8p.real";
pub const ROWS_SET: [usize; 7] = [1, 2, 7, 8, 15, 16, 17];
const FIRST_LIMIT: usize = 8;

pub type Tile = (i32, [i32; 16]);
pub fn tile(expert: i32, slots: &[i32]) -> Tile {
    let mut t = [-1i32; 16];
    t[..slots.len()].copy_from_slice(slots);
    (expert, t)
}
pub fn tiles_for(expert: i32, slots: Vec<i32>) -> Vec<Tile> {
    slots.chunks(16).map(|c| tile(expert, c)).collect()
}
pub fn sentinel_tile() -> Tile {
    (-1, [0; 16])
}

/// `n` distinct non-monotone slots in `[0, x_rows * div)`.
fn spread_slots(n: usize, x_rows: usize, div: usize) -> Vec<i32> {
    let total = x_rows * div;
    assert!(n <= total);
    let mut a = ((total * 618 / 1000).max(2)) | 1;
    while gcd(a, total) != 1 {
        a += 2;
    }
    let b = total / 3;
    (0..n).map(|i| ((i * a + b) % total) as i32).collect()
}

#[derive(Clone, Copy)]
pub enum XSrc {
    Stress(u64),
    Bounded(u64),
}

pub struct Scenario {
    pub name: String,
    pub kind: &'static str,
    /// Per projection `[gate_up, down]`.
    pub div: [usize; 2],
    pub x_rows: [usize; 2],
    pub tiles: Vec<Tile>,
    pub x: XSrc,
    pub per_expert: bool,
}

pub struct RunOut {
    pub report: Value,
    pub acc_bad: u64,
    pub h_bad: u64,
    pub bad_slots: u64,
}

// ---------------------------------------------------------------------------------------------------------------
// device set: all 512 experts in one slab (gate_up ++ down at 256-aligned offsets, 0xFF guards)
// ---------------------------------------------------------------------------------------------------------------

pub struct DevSet {
    pub slab: DeviceBuffer,
    pub ptrs: [DeviceBuffer; 2],
    pub offsets: Vec<usize>,
    pub slab_bytes: usize,
}

pub fn make_set(hip: &HipRuntime, layer: &E8pLayer, neg: u8) -> Result<DevSet> {
    let mut buf = vec![0xffu8; GUARD];
    let mut offsets = Vec::with_capacity(layer.blobs.len());
    for b in &layer.blobs {
        offsets.push(buf.len());
        buf.extend_from_slice(&b[0]);
        buf.extend_from_slice(&b[1]);
        buf.resize(align(buf.len()) + GUARD, 0xff);
    }
    let clean_len = buf.len();
    if neg == 2 {
        // device-side corruption only; the CPU twins keep the clean records
        buf[offsets[0] + 1000] ^= 0x10;
    }
    let slab = up(hip, &buf)?;
    let base = slab.as_ptr() as u64;
    let table = |p: Proj| -> Vec<u8> {
        layer
            .blobs
            .iter()
            .zip(&offsets)
            .flat_map(|(b, o)| {
                (base + (*o + if p == Proj::Down { b[0].len() } else { 0 }) as u64).to_le_bytes()
            })
            .collect()
    };
    let ptrs = [up(hip, &table(Proj::GateUp))?, up(hip, &table(Proj::Down))?];
    Ok(DevSet {
        slab,
        ptrs,
        offsets,
        slab_bytes: clean_len,
    })
}

pub fn free_set(hip: &HipRuntime, s: DevSet) -> Result<()> {
    let [a, b] = s.ptrs;
    free_all(hip, vec![s.slab, a, b])
}

// ---------------------------------------------------------------------------------------------------------------
// one scenario x one projection
// ---------------------------------------------------------------------------------------------------------------

pub struct G1Ctx<'a> {
    pub hip: &'a HipRuntime,
    pub hook: &'a Hook<'a>,
    pub oracle: &'a Oracle,
    pub layer: &'a E8pLayer,
    pub module: &'a E8pModule,
    pub set: &'a DevSet,
    pub neg: u8,
}

fn make_x(src: XSrc, groups: usize, rows: usize) -> (Vec<u8>, Vec<A4Block>) {
    match src {
        XSrc::Stress(seed) => make_x_stress(groups, rows, seed),
        XSrc::Bounded(key) => {
            let tok = bounded_tok(groups, rows, key);
            (epoch_major(&tok, rows, groups), tok)
        }
    }
}

pub fn run_one(ctx: &G1Ctx, sc: &Scenario, proj: Proj) -> Result<RunOut> {
    let hip = ctx.hip;
    let (m, k, h, groups, pi) = (proj.m(), proj.k(), proj.h(), proj.groups(), proj.idx());
    let (div, x_rows) = (sc.div[pi], sc.x_rows[pi]);
    let (xq, x_tok) = make_x(sc.x, groups, x_rows);
    let ntiles = sc.tiles.len();
    let m_total = ntiles * 16;
    let tile_ids: Vec<i32> = sc.tiles.iter().map(|t| t.0).collect();
    let sorted: Vec<i32> = sc.tiles.iter().flat_map(|t| t.1).collect();
    let tiles_dev = up(hip, &i32s_le(&tile_ids))?;
    let sorted_dev = up(hip, &i32s_le(&sorted))?;
    let x_dev = up(hip, &xq)?;
    let (acc_bytes, h_bytes) = (m_total * m * 4, m_total * h * 2);
    let (y_acc, y_h) = (
        hip.malloc(acc_bytes).map_err(err)?,
        hip.malloc(h_bytes).map_err(err)?,
    );
    hip.memset(&y_acc, 0xa5, acc_bytes).map_err(err)?;
    hip.memset(&y_h, 0xa5, h_bytes).map_err(err)?;
    let launch = |f: &hip_bridge::Function, y: &DeviceBuffer| -> Result<()> {
        let mut a = kernarg(
            &ctx.set.ptrs[pi],
            &tiles_dev,
            &sorted_dev,
            &x_dev,
            y,
            m,
            k,
            div,
            m_total,
            x_rows,
            ctx.module.ka[pi],
        );
        // SAFETY: the shared PM 64-byte kernarg ABI (5 pointers, 6 u32 incl. n_experts@60 = the table's actual entry count), zero padded to the code object's
        // kernarg segment size; every buffer outlives the synchronize below.
        unsafe {
            hip.launch_kernel_blob(
                f,
                [proj.grid_x(), ntiles as u32, 1],
                [proj.threads(), 1, 1],
                0,
                None,
                &mut a,
            )
        }
        .map_err(err)?;
        hip.device_synchronize().map_err(err)
    };
    launch(
        &ctx.module
            .acc
            .as_ref()
            .ok_or("E8P module has no _acc entries")?[pi],
        &y_acc,
    )?;
    launch(&ctx.module.ship[pi], &y_h)?;
    let got_acc = read_back(hip, &y_acc, acc_bytes)?;
    let got_h = read_back(hip, &y_h, h_bytes)?;
    free_all(hip, vec![tiles_dev, sorted_dev, x_dev, y_acc, y_h])?;

    // CPU twins: per used expert, only the x rows its live slots read.
    let mut need: BTreeMap<i32, BTreeSet<usize>> = BTreeMap::new();
    for t in &sc.tiles {
        if t.0 >= 0 {
            if t.0 as usize >= EXPERTS {
                return Err(format!(
                    "{}: tile expert {} outside the resident layer",
                    sc.name, t.0
                ));
            }
            let s = need.entry(t.0).or_default();
            for &slot in t.1.iter().filter(|&&s| s >= 0) {
                let xr = slot as usize / div;
                if xr >= x_rows {
                    return Err(format!(
                        "{}: slot {slot} reads x row {xr} >= {x_rows}",
                        sc.name
                    ));
                }
                s.insert(xr);
            }
        }
    }
    let mut dense: BTreeMap<i32, (Vec<usize>, Vec<Vec<f32>>)> = BTreeMap::new();
    for (&e, rows) in &need {
        let xr: Vec<usize> = rows.iter().copied().collect();
        let mut d = ctx
            .oracle
            .dense_rows(proj, &ctx.layer.blobs[e as usize][pi], &x_tok, &xr);
        if ctx.neg == 1 {
            for r in &mut d {
                r[0] = f32::from_bits(r[0].to_bits() ^ 1);
            }
        }
        dense.insert(e, (xr, d));
    }

    let mut sums = vec![0f32; m_total * m];
    let mut slot_ids = vec![-1i32; m_total];
    let (mut acc_bad, mut acc_checked, mut live_slots) = (0u64, 0u64, 0usize);
    let mut acc_first: Vec<Value> = Vec::new();
    let mut bad_pos = vec![false; m_total];
    let mut pe: BTreeMap<i32, [u64; 3]> = BTreeMap::new();
    for (ti, t) in sc.tiles.iter().enumerate() {
        for lane in 0..16 {
            let p = ti * 16 + lane;
            let live = t.0 >= 0 && t.1[lane] >= 0;
            let twin_row: Option<&[f32]> = if live {
                let xr = t.1[lane] as usize / div;
                let (xrows, rows) = &dense[&t.0];
                let at = xrows
                    .binary_search(&xr)
                    .map_err(|_| format!("{}: internal: x row {xr} not twinned", sc.name))?;
                Some(rows[at].as_slice())
            } else {
                None
            };
            if let Some(row) = twin_row {
                sums[p * m..(p + 1) * m].copy_from_slice(row);
                slot_ids[p] = t.1[lane];
                live_slots += 1;
                pe.entry(t.0).or_default()[2] += 1;
            }
            for r in 0..m {
                let o = (p * m + r) * 4;
                let bits = u32::from_le_bytes(got_acc[o..o + 4].try_into().unwrap());
                let want = if t.0 < 0 {
                    0xa5a5_a5a5
                } else if let Some(row) = twin_row {
                    row[r].to_bits()
                } else {
                    0
                };
                acc_checked += 1;
                if bits != want {
                    acc_bad += 1;
                    bad_pos[p] = true;
                    if t.0 >= 0 {
                        pe.entry(t.0).or_default()[0] += 1;
                    }
                    if acc_first.len() < FIRST_LIMIT {
                        acc_first.push(json!({"position": p, "tile_expert": t.0, "row": r, "device": format!("{bits:#010x}"), "twin": format!("{want:#010x}")}));
                    }
                }
            }
        }
    }
    // BF16 output vs the frozen DAG over the twin sums; sentinel tiles must stay 0xa5a5, pad slots are +0.
    let mut want_h = ctx.hook.dag_epilogue(proj.kind(), m, &sums, &slot_ids)?;
    for (ti, t) in sc.tiles.iter().enumerate() {
        if t.0 < 0 {
            for v in &mut want_h[ti * 16 * h..(ti + 1) * 16 * h] {
                *v = 0xa5a5;
            }
        }
    }
    let (mut h_bad, mut h_first) = (0u64, Vec::<Value>::new());
    for (i, want) in want_h.iter().enumerate() {
        let got = u16::from_le_bytes([got_h[2 * i], got_h[2 * i + 1]]);
        if got != *want {
            h_bad += 1;
            let p = i / h;
            bad_pos[p] = true;
            let te = sc.tiles[p / 16].0;
            if te >= 0 {
                pe.entry(te).or_default()[1] += 1;
            }
            if h_first.len() < FIRST_LIMIT {
                h_first.push(json!({"position": p, "tile_expert": te, "col": i % h, "device": format!("{got:#06x}"), "twin": format!("{want:#06x}")}));
            }
        }
    }
    let bad_slots = bad_pos.iter().filter(|&&b| b).count() as u64;
    let mut report = json!({
        "scenario": sc.name, "kind": sc.kind, "proj": proj.name(), "x_row_div": div, "x_rows": x_rows,
        "tiles": ntiles, "live_slots": live_slots, "pad_slots": m_total - live_slots,
        "experts_used": need.len(), "expert_range": [need.keys().next(), need.keys().next_back()],
        "x_activations": match sc.x { XSrc::Stress(_) => "stress (extreme activation scales, synthetic parity inputs)", XSrc::Bounded(_) => "deterministic bounded synthetic A4 (scale 1/1024..1/64) through REAL E8P weights; not corpus/route activations" },
        "x_sha256": tok_sha(&x_tok),
        "acc_outputs_checked": acc_checked, "acc_mismatches": acc_bad, "acc_first": acc_first,
        "h_outputs_checked": want_h.len(), "h_mismatches": h_bad, "h_first": h_first,
        "mismatching_slots": bad_slots,
        "acc_sha256": sha(&got_acc), "h_sha256": sha(&got_h),
    });
    if sc.per_expert {
        report["per_expert"] = Value::Array(pe.iter().map(|(e, c)| json!({"expert": e, "live_slots": c[2], "acc_mismatches": c[0], "h_mismatches": c[1]})).collect());
    }
    Ok(RunOut {
        report,
        acc_bad,
        h_bad,
        bad_slots,
    })
}

// ---------------------------------------------------------------------------------------------------------------
// scenario builders
// ---------------------------------------------------------------------------------------------------------------

/// One tile per expert for ALL 512 experts: hashed non-monotone expert order, mixed full/partial tiles, a sentinel
/// tile every 64 tiles and at the end. `x_rows * div` is the slot space, `a` is coprime to it.
pub fn coverage_scenario(name: &str, x: XSrc, x_rows: usize, div: usize, seed: u64) -> Scenario {
    let total = x_rows * div;
    let mut a = 29usize;
    while gcd(a, total) != 1 {
        a += 2;
    }
    let mut order: Vec<(u64, usize)> = (0..EXPERTS)
        .map(|e| (fnv64(&format!("hipfire.r8.e8p.cover.v1|{seed}|{e}")), e))
        .collect();
    order.sort();
    let mut tiles = Vec::with_capacity(EXPERTS + EXPERTS / 64 + 1);
    for (i, &(hsh, e)) in order.iter().enumerate() {
        let live = if hsh % 3 == 0 {
            1 + ((hsh >> 8) % 16) as usize
        } else {
            16
        };
        let b = (e * 7) % total;
        let slots: Vec<i32> = (0..live).map(|j| ((j * a + b) % total) as i32).collect();
        debug_assert!(live <= total);
        tiles.push(tile(e as i32, &slots));
        if i % 64 == 63 {
            tiles.push(sentinel_tile());
        }
    }
    tiles.push(sentinel_tile());
    Scenario {
        name: name.to_owned(),
        kind: "coverage",
        div: [div; 2],
        x_rows: [x_rows; 2],
        tiles,
        x,
        per_expert: true,
    }
}

/// The real-G1 parity set of one expert: rows 1/2/7/8/15/16/17 x x_row_div 1/10 (stress activations, non-monotone slots,
/// pad slots, sentinel tile), then optional mixed-expert scenario over three experts.
pub fn parity_scenarios(tag: &str, pos: &[usize], seed: u64) -> Vec<Scenario> {
    let mut out = Vec::new();
    for &p in pos {
        for div in [1usize, 10] {
            for n in ROWS_SET {
                let x_rows = n.div_ceil(div) + 2;
                let mut tiles = tiles_for(p as i32, spread_slots(n, x_rows, div));
                tiles.push(sentinel_tile());
                let name = format!("{tag}_E{p}_rows{n}_div{div}");
                out.push(Scenario {
                    x: XSrc::Stress(seed ^ fnv64(&name)),
                    name,
                    kind: "parity",
                    div: [div; 2],
                    x_rows: [x_rows; 2],
                    tiles,
                    per_expert: false,
                });
            }
        }
    }
    for trio in pos.chunks_exact(3) {
        let (p0, p1, p2) = (trio[0] as i32, trio[1] as i32, trio[2] as i32);
        // mixed experts, non-monotone expert order and slot order, pad holes, mid-sequence and trailing sentinel tiles
        let mut tiles = tiles_for(p2, vec![5, 3, 19, 0, 1, 2, 11, 12]);
        tiles.push(sentinel_tile());
        tiles.extend(tiles_for(
            p0,
            (0..17).map(|i| (i * 3 % 20) as i32).collect(),
        ));
        tiles.extend(tiles_for(p1, vec![7]));
        tiles.push(tile(p0, &[-1, 4, -1, 9]));
        tiles.push(sentinel_tile());
        let name = format!("{tag}_mixed_E{p0}_E{p1}_E{p2}_unsorted_pad");
        out.push(Scenario {
            x: XSrc::Stress(seed ^ fnv64(&name)),
            name,
            kind: "parity",
            div: [1; 2],
            x_rows: [20; 2],
            tiles,
            per_expert: false,
        });
    }
    out
}

/// First `tokens` tokens of the recorded route's first chunk: real expert ids, real slot ids `token * topk + j`.
pub fn route_head_scenario(route: &Route, tokens: usize, seed: u64) -> Result<Scenario> {
    let c = route.chunks.first().ok_or("route has no chunks")?;
    if tokens == 0 || tokens > c.rows {
        return Err(format!("--route-tokens {tokens} outside [1, {}]", c.rows));
    }
    let topk = route.topk;
    let mut per: Vec<Vec<i32>> = vec![Vec::new(); EXPERTS];
    for t in 0..tokens {
        for j in 0..topk {
            per[c.ids[t * topk + j] as usize].push((t * topk + j) as i32);
        }
    }
    let mut tiles = Vec::new();
    for (e, slots) in per.iter().enumerate() {
        tiles.extend(tiles_for(e as i32, slots.clone()));
    }
    tiles.push(sentinel_tile());
    let name = format!("route_head_L{}_chunk0_tokens{tokens}", route.layer);
    Ok(Scenario {
        x: XSrc::Bounded(seed ^ fnv64(&name)),
        name,
        kind: "route_head",
        div: [topk, 1],
        x_rows: [tokens, tokens * topk],
        tiles,
        per_expert: true,
    })
}

/// Representative experts of the parity set: SHA-256 ranking (same scheme as the real-G1 expert selection), rotated if monotone.
pub fn select_experts(layer: usize, per: usize) -> Vec<usize> {
    let mut v: Vec<([u8; 32], usize)> = (0..EXPERTS)
        .map(|e| {
            (
                Sha256::digest(format!("hipfire.r8.e8p.select.v1|L{layer}|E{e}").as_bytes()).into(),
                e,
            )
        })
        .collect();
    v.sort();
    let mut ids: Vec<usize> = v.into_iter().take(per).map(|x| x.1).collect();
    if ids.windows(2).all(|w| w[0] < w[1]) || ids.windows(2).all(|w| w[0] > w[1]) {
        ids.rotate_left(1);
    }
    ids
}

// ---------------------------------------------------------------------------------------------------------------
// mode g1
// ---------------------------------------------------------------------------------------------------------------

/// Everything the mode needs from the shared CPU preparation.
pub struct Prep<'a> {
    pub args: &'a super::Args,
    pub argv: &'a [String],
    pub layer: &'a E8pLayer,
    pub oracle: &'a Oracle,
    pub oracle_check: &'a Value,
    pub harness: &'a Value,
    pub out_dir: &'a std::path::Path,
    pub records_file: &'a Value,
}

fn scenario_summary(sc: &Scenario) -> Value {
    let live: usize = sc
        .tiles
        .iter()
        .map(|t| {
            if t.0 >= 0 {
                t.1.iter().filter(|&&s| s >= 0).count()
            } else {
                0
            }
        })
        .sum();
    json!({"name": sc.name, "kind": sc.kind, "tiles": sc.tiles.len(), "live_slots": live, "x_row_div": sc.div, "x_rows": sc.x_rows,
        "experts": sc.tiles.iter().filter(|t| t.0 >= 0).map(|t| t.0).collect::<BTreeSet<_>>().len()})
}

/// Returns `Ok(true)` iff the whole real-layer qualification is exact and complete.
pub fn run_g1(p: &Prep) -> Result<bool> {
    let a = p.args;
    let variant = a.single_variant()?;
    let cfg = a.variant_cfg(&variant)?;
    let seed = a.seed;
    let route: Option<Route> = match &a.trace_dir {
        Some(d) => Some(super::route::load_route(
            &d.join(format!("layer{}.trace.json", a.layer)),
            a.route_bin.as_deref(),
            a.layer,
        )?),
        None => None,
    };

    // ---- scenarios (CPU only) ----------------------------------------------------------------------------------
    let picks = select_experts(a.layer, a.g1_experts);
    if picks.len() < 3 || picks.len() % 3 != 0 {
        return Err(format!(
            "--g1-experts must be a positive multiple of 3 (mixed scenarios take trios), got {}",
            a.g1_experts
        ));
    }
    let mut scenarios: Vec<Scenario> = vec![
        coverage_scenario(
            &format!("coverage_L{}_bounded_div1", a.layer),
            XSrc::Bounded(seed ^ 0xC0DE_0001),
            48,
            1,
            seed,
        ),
        coverage_scenario(
            &format!("coverage_L{}_stress_div10", a.layer),
            XSrc::Stress(seed ^ 0xC0DE_0002),
            5,
            10,
            seed.wrapping_add(1),
        ),
    ];
    scenarios.extend(parity_scenarios(
        &format!("parity_L{}", a.layer),
        &picks,
        seed,
    ));
    if let Some(r) = &route {
        scenarios.push(route_head_scenario(r, a.route_tokens, seed)?);
    }
    let summaries: Vec<Value> = scenarios.iter().map(scenario_summary).collect();
    let scope_base = json!({"layer": a.layer, "experts_resident": EXPERTS, "all_512_real_e8p_experts_resident": true, "parity_experts": picks, "scenarios": summaries,
        "route_head": route.as_ref().map(|r| json!({"trace": r.trace_path.display().to_string(), "route_bin_sha256": r.route_sha256, "tokens": a.route_tokens, "topk": r.topk})),
        "claim": "every one of the layer's 512 real solved E8P experts resident at once and exercised (one tile each, gate_up and down) with zero F32/BF16 mismatches against the K-E0 twin; weights are REAL, activations are synthetic bounded/stress A4 (plus the real route's expert/slot structure in route_head); NOT a full-model or corpus-activation claim"});
    if a.plan_only {
        let r = json!({"schema": SCHEMA, "status": "plan_only_no_gpu", "variant": variant, "scope": scope_base, "inputs": p.layer.inputs, "e8p_records_file": p.records_file,
            "oracle_self_check": p.oracle_check, "code": {"harness": p.harness}, "command": {"argv": p.argv}});
        let path = p.out_dir.join("g1.e8p.real.plan.json");
        let b = write_json(&path, &r)?;
        println!(
            "{}",
            json!({"schema": SCHEMA, "status": "plan_only_no_gpu", "receipt": path.display().to_string(), "receipt_sha256": sha(&b), "scenarios": scenarios.len()})
        );
        return Ok(true);
    }

    // ---- device --------------------------------------------------------------------------------------------------
    let card = a
        .card
        .clone()
        .ok_or("--card is required for a device run")?;
    let fabric = fabric_provenance(
        &card,
        "correct",
        a.fabric_owner.as_deref(),
        a.fabric_wrapper.as_deref(),
    )?;
    let probe_co = a.probe_co.clone().ok_or("--probe-codeobject is required")?;
    let probe_img = read_file(&probe_co)?;
    if let Some(pin) = &a.probe_sha {
        if sha(&probe_img) != *pin {
            return Err(format!(
                "probe code object sha256 {} != pinned {pin}",
                sha(&probe_img)
            ));
        }
    }
    let e8p_img = read_file(&cfg.co)?;
    let bind_check = bind_module(
        "e8p",
        &cfg.co,
        &e8p_img,
        cfg.sha.as_deref(),
        cfg.bundle.as_deref(),
    )?;
    let _ = bind_check;
    let hip = HipRuntime::load().map_err(err)?;
    let id = hip.current_device().map_err(err)?;
    let (uuid, bdf, arch) = (
        hip.device_uuid(id).map_err(err)?,
        hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase(),
        hip.get_arch(id).map_err(err)?,
    );
    let identity_match = arch == "gfx1201" && a.expect_bdf.as_deref().is_none_or(|b| b == bdf);
    let device = json!({"card": card, "device_id": id, "uuid": uuid, "bdf": bdf, "arch": arch, "expected_arch": "gfx1201", "expected_bdf": a.expect_bdf, "identity_match": identity_match,
        "env": {"HIP_VISIBLE_DEVICES": std::env::var("HIP_VISIBLE_DEVICES").ok(), "ROCR_VISIBLE_DEVICES": std::env::var("ROCR_VISIBLE_DEVICES").ok()}});
    if !identity_match {
        return Err(format!(
            "device identity mismatch: arch={arch} bdf={bdf} expected gfx1201 bdf={:?}",
            a.expect_bdf
        ));
    }
    let module = super::hw::load_e8p(
        &hip,
        &cfg.co,
        &e8p_img,
        cfg.sha.as_deref(),
        cfg.bundle.as_deref(),
        a.e8p_symbols(),
        true,
    )?;
    let hook = Hook::load(&hip, &probe_img, &probe_co, a.probe_ka)?;
    let (div_kat, div_kat_ok) = hook.divide_known_answer()?;
    let set = make_set(&hip, p.layer, a.neg)?;
    let ctx = G1Ctx {
        hip: &hip,
        hook: &hook,
        oracle: p.oracle,
        layer: p.layer,
        module: &module,
        set: &set,
        neg: a.neg,
    };

    let mut reports: Vec<Value> = Vec::new();
    let (mut acc_total, mut h_total, mut slots_total, mut runs) = (0u64, 0u64, 0u64, 0u64);
    // per projection: expert -> (live slots exercised, mismatches) over the coverage scenarios
    let mut covered: [BTreeMap<u64, (u64, u64)>; 2] = [BTreeMap::new(), BTreeMap::new()];
    for sc in &scenarios {
        for proj in Proj::ALL {
            let r = run_one(&ctx, sc, proj)?;
            acc_total += r.acc_bad;
            h_total += r.h_bad;
            slots_total += r.bad_slots;
            runs += 1;
            if sc.kind == "coverage" {
                for e in r.report["per_expert"].as_array().into_iter().flatten() {
                    let c = covered[proj.idx()]
                        .entry(e["expert"].as_u64().unwrap_or(u64::MAX))
                        .or_default();
                    c.0 += e["live_slots"].as_u64().unwrap_or(0);
                    c.1 += e["acc_mismatches"].as_u64().unwrap_or(0)
                        + e["h_mismatches"].as_u64().unwrap_or(0);
                }
            }
            reports.push(r.report);
        }
    }
    let cov_ok = |pi: usize| {
        covered[pi].len() == EXPERTS && covered[pi].values().all(|c| c.0 > 0 && c.1 == 0)
    };
    let coverage = json!({
        "gate_up": {"experts_exercised": covered[0].len(), "experts_exact_with_live_slots": covered[0].values().filter(|c| c.0 > 0 && c.1 == 0).count(), "all_512_exact": cov_ok(0)},
        "down": {"experts_exercised": covered[1].len(), "experts_exact_with_live_slots": covered[1].values().filter(|c| c.0 > 0 && c.1 == 0).count(), "all_512_exact": cov_ok(1)},
    });
    let div_mismatch = hook.div_mismatch.get();
    let div_ieee = hook.div_ieee_mismatch.get();
    let mismatch_total = acc_total + h_total;
    let guard_clean = p.layer.inputs["writer_guard_rejects_total"].as_u64() == Some(0);
    let pass = mismatch_total == 0
        && div_mismatch == 0
        && div_ieee == 0
        && div_kat_ok
        && cov_ok(0)
        && cov_ok(1)
        && guard_clean
        && identity_match
        && a.neg == 0;
    let mut scope = scope_base;
    scope["coverage"] = coverage;
    let receipt = json!({
        "schema": SCHEMA, "pass": pass, "variant": variant,
        "exact_mismatch_total": mismatch_total, "acc_mismatches": acc_total, "h_mismatches": h_total, "mismatching_slots": slots_total,
        "writer_guard_clean": guard_clean,
        "divide_transcription_self_check": {"requests_checked": hook.div_checked.get(), "probe_sequence_vs_fdiv_rn_mismatches": div_mismatch, "probe_sequence_vs_cpu_ieee_mismatches": div_ieee,
            "mismatch_list": hook.div_list.borrow().clone(), "note": "explicit v_div_* f32 sequence (probe) vs hipcc __fdiv_rn vs CPU IEEE over the divides the twin consumed"},
        "divide_known_answer": div_kat,
        "layers": [a.layer],
        "scope": scope,
        "inputs": p.layer.inputs, "e8p_records_file": p.records_file,
        "oracle_self_check": p.oracle_check,
        "device": device, "fabric": fabric,
        "code": {
            "harness": p.harness,
            "probe_codeobject": {"path": probe_co.display().to_string(), "bytes": probe_img.len(), "sha256": sha(&probe_img), "pinned_sha256": a.probe_sha,
                "kernarg_bytes": {"exp2": hook.probe.ka[0], "ldexp": hook.probe.ka[1], "div": hook.probe.ka[2]},
                "source": a.probe_source.as_ref().and_then(|s| read_file(s).ok().map(|b| json!({"path": s.display().to_string(), "sha256": sha(&b)})))},
            "e8p": {"variant": variant, "module": module.bind.clone(), "symbols": {"gate_up": module.syms[0], "down": module.syms[1], "acc_suffix": "_acc"}, "kernarg_bytes": {"gate_up": module.ka[0], "down": module.ka[1]}},
            "epilogue_twin": {"function": "mq_i4_reference::grouped_prefill_epilogue_with", "intrinsics": "device probes (v_exp_f32, v_ldexp_f32, IEEE div expansion), NOT PortableEpilogue; no portable-exp fallback",
                "dag_replay_passes": hook.dag_passes.get(), "dag_final_pass_intrinsic_calls": hook.dag_calls.get()},
            "f32_twin": "e8p_i4_reference: unpack_word_ref nibbles o, a = MAGIC + sum(o*q) per K128, fold128(a, s), epoch_sum(d_act, d, a', sum) from +0 over ascending epochs",
        },
        "slab": {"bytes": set.slab_bytes, "experts": EXPERTS, "alignment": ALIGN, "guard_bytes": GUARD, "first_offsets": &set.offsets[..4usize.min(set.offsets.len())]},
        "scenarios_run": runs, "scenarios": reports,
        "command": {"argv": p.argv, "line": p.argv.join(" "), "cwd": std::env::current_dir().ok().map(|d| d.display().to_string()), "seed": seed, "negative_control": a.neg,
            "rc": "exit code recorded by the fabric wrapper: 0 pass, 3 ran but not exact/complete, 1 error",
            "env": {"HIP_VISIBLE_DEVICES": std::env::var("HIP_VISIBLE_DEVICES").ok(), "ROCR_VISIBLE_DEVICES": std::env::var("ROCR_VISIBLE_DEVICES").ok()}},
    });
    free_set(&hip, set)?;
    let path = p.out_dir.join("g1.e8p.real.json");
    let b = write_json(&path, &receipt)?;
    println!(
        "{}",
        json!({"schema": SCHEMA, "pass": pass, "variant": variant, "exact_mismatch_total": mismatch_total, "acc_mismatches": acc_total, "h_mismatches": h_total, "mismatching_slots": slots_total,
        "divide_self_check_mismatches": div_mismatch, "scenarios_run": runs, "coverage": receipt["scope"]["coverage"], "receipt": path.display().to_string(), "receipt_sha256": sha(&b), "device": receipt["device"]})
    );
    Ok(pass)
}
