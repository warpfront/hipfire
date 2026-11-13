// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-image byte oracle of the exact-gfx942 wide-lane MoE glue module
//! (`kernels/src/qwen4_g942_moe_wide.gfx942.hip`, module `qwen4_g942_moe_wide`, entries
//! `qwen4_g942_moe_wide_router`, `_group_scatter_desc`, `_unscatter_act`, `_combine_shared_add`).
//!
//! Three arms run on the same inputs; every output buffer (poison-filled first, 0x7f guard words
//! on both sides that must survive) must be byte-equal:
//!
//! - REFERENCE: the route each entry replaces, through its public launcher:
//!   `Gpu::moe_router_softmax_top10_f32` (BF16 recipe), `Gfx942Device::g942_fold_moe_group_scatter_desc`,
//!   `g942_fold_moe_unscatter_act`, `g942_fold_moe_combine_shared_add`;
//! - MODULE: the four module entries launched directly through `Gpu::launch_kernel_blob`. The
//!   module is loaded the way the product loads it (`lab_ensure_moe_wide_entry`): from the hipcc
//!   source, or, with `HIPFIRE_QWEN4_G942_MOE_WIDE_IMAGE=/absolute/device.elf`, from that code
//!   object (the twin proof: same module, same entry names, same kernarg ABI);
//! - PRODUCT: the public launchers the dispatch calls: `Gfx942Device::g942_moe_wide_router`,
//!   `g942_moe_wide_group_scatter_desc`, `g942_moe_wide_unscatter_act`,
//!   `g942_moe_wide_combine_shared_add` (they route to the module and honour the same hook).
//!
//! A second REFERENCE launch must equal the first (the reference is repeatable). The public
//! launchers need the C1 folds on (`HIPFIRE_QWEN4_G942_FOLD_MOE`, default on); with the folds
//! killed the oracle reports the launcher refusal and fails.
//!
//! Coverage: router at 8..64 tokens (80..640 slots) in {0, 1} `normalize_topk_prob`, eleven
//! logit families (gaussian, coarse ties, all equal, +-0 / tiny, huge, NaN / Inf rows, large
//! negative, BF16 ties, an 11-way tie at the top-10 boundary, ascending / descending, per-row
//! mix) and nonzero layer / row base (the fault record is compared too); scatter at 80..640 flat
//! slots incl. 80, 81, 319, 320, 321, 639, 640 and 512 experts (the Flash-Next stock-FP8 model:
//! top-10 of 512) plus other expert counts, random / per-token-distinct / all-slots-one-expert /
//! two-expert / hot-set / sequential / 15-16-17-per-expert / top-of-range / marker-carrying /
//! all-unrouted routings and `grouped_rows` from the tight bound to the worst-case bound; the
//! unscatter + shared activation and the combine + shared add on those groupings (mi 640 /
//! shared_mi 640 / hidden 2560 and odd widths), the combine also with corrupted inverse rows and
//! out-of-range expert ids, and slot counts beyond 640 (the unscatter and the combine take no
//! upper cap) on a host-built grouping.
//!
//! Planted controls the oracle MUST detect (module output differs from the clean reference):
//! one flipped byte in every operand of every entry, a wrong slot / token / expert / row count,
//! a wrong `grouped_rows` (guard overrun), a wrong normalize flag and swapped operand bindings.
//! Exit 0 only when every case matches and every control is detected. Prints PASS / FAIL counts.
//!
//! Build: `cargo build --release -p rdna-compute --features lab --example qwen4_g942_moe_wide_oracle`
//! Run:   `qwen4_g942_moe_wide_oracle [--out FILE.json] [--seed U64] [--quick]`
//! Twin image: `HIPFIRE_QWEN4_G942_MOE_WIDE_IMAGE=/absolute/device.elf qwen4_g942_moe_wide_oracle`

mod g942_oracle_common;

use g942_oracle_common::*;
use hip_bridge::KernargBlob;
use rdna_compute::tensor_ops::lab_ensure_moe_wide_entry;
use rdna_compute::{Gpu, GpuTensor};
use serde_json::json;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

const E_MODEL: usize = 512;
const TOPK: usize = 10;
const ROUTER: &str = "qwen4_g942_moe_wide_router";
const SCATTER: &str = "qwen4_g942_moe_wide_group_scatter_desc";
const UNSCATTER: &str = "qwen4_g942_moe_wide_unscatter_act";
const COMBINE: &str = "qwen4_g942_moe_wide_combine_shared_add";
const LAYER: usize = 7;
const ROW_BASE: usize = 1000;

/// Host-mirror vs reference scatter disagreements (informational: the mirror only builds the
/// beyond-640-slot groupings and sizes `grouped_rows`).
static MIRROR_MISMATCH: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Reference,
    Module,
    Product,
}

fn gfx(gpu: &mut Gpu) -> Res<rdna_compute::cdna::gfx942::Gfx942Device<'_>> {
    gpu.try_gfx942()
        .ok_or_else(|| "not an exact gfx942 device".to_owned())
}

fn round16(n: usize) -> usize {
    n.div_ceil(16) * 16
}

// ---------------------------------------------------------------------------------------------
// Value generators
// ---------------------------------------------------------------------------------------------

/// Floats of one family: 0 gaussian, 1 wide dynamic range, 2 special values and BF16 ties,
/// 3 random finite bit patterns (exponent 60..160, subnormals and ties by construction).
fn fvals(rng: &mut Rng, n: usize, kind: usize) -> Vec<f32> {
    const SPECIAL: [u32; 16] = [
        0x0000_0000,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f80_8000,
        0xbf80_8000,
        0x3f80_7fff,
        0x3f80_8001,
        0x3f00_0000,
        0x4040_0000,
        0x0000_0400,
        0x8000_0400,
        0x7149_f2ca,
        0xf149_f2ca,
        0x2e80_0000,
        0x4780_8000,
    ];
    (0..n)
        .map(|_| match kind {
            0 => (rng.gauss() * 2.0) as f32,
            1 => {
                let sign = if rng.below(2) == 0 { 1.0 } else { -1.0 };
                let exp = rng.below(61) as i32 - 30;
                (sign * (1.0 + rng.unit()) * (exp as f64).exp2()) as f32
            }
            2 => f32::from_bits(rng.pick(&SPECIAL)),
            _ => {
                let exp = 60 + rng.below(100) as u32;
                let sign = (rng.next() as u32 & 1) << 31;
                f32::from_bits(sign | (exp << 23) | (rng.next() as u32 & 0x007f_ffff))
            }
        })
        .collect()
}

/// Router logits of one row of family `kind` (0..=7 mirror the glue harness; 8 = an 11-way tie
/// at the top-10 boundary; 9 = ascending; 10 = descending).
fn router_row(rng: &mut Rng, kind: usize) -> Vec<f32> {
    let mut row: Vec<f32> = (0..E_MODEL)
        .map(|e| match kind {
            0 | 5 => (rng.gauss() * 3.0) as f32,
            1 => ((rng.gauss() * 4.0).round() * 0.25) as f32,
            2 => 0.75,
            3 => rng.pick(&[0.0, -0.0, 1e-30, -1e-30, 5.0, 5.00001, 5.01, -5.0]),
            4 => ((rng.unit() - 0.5) * 2e30) as f32,
            6 => (-100.0 + rng.gauss() * 0.3) as f32,
            7 => ((rng.gauss() * 40.0).round() / 128.0) as f32,
            8 => (rng.gauss() * 0.5) as f32,
            9 => e as f32 * 0.01,
            _ => -(e as f32) * 0.01,
        })
        .collect();
    match kind {
        1 if rng.below(3) == 0 => {
            // F32-different, BF16-tied values.
            for v in &mut row {
                *v = f32::from_bits(v.to_bits() ^ (rng.next() as u32 & 0x3fff));
            }
        }
        5 if rng.below(2) == 0 => {
            for _ in 0..1 + rng.below(3) {
                let e = rng.below(E_MODEL);
                row[e] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][rng.below(3)];
            }
        }
        8 => {
            // Eleven experts tied exactly at the top: rank ten decides on the index.
            let top = 4.0 + rng.below(4) as f32;
            let mut chosen = Vec::new();
            while chosen.len() < 11 {
                let e = rng.below(E_MODEL);
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }
            for e in chosen {
                row[e] = top;
            }
        }
        _ => {}
    }
    row
}

const ROUTER_KINDS: usize = 12;

fn router_logits(rng: &mut Rng, tokens: usize, kind: usize) -> Vec<f32> {
    (0..tokens)
        .flat_map(|_| {
            let row_kind = if kind == 11 { rng.below(11) } else { kind };
            router_row(rng, row_kind)
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------------------------

struct RouterOut {
    ids: Vec<u8>,
    weights: Vec<u8>,
    fault: String,
    guards: bool,
}

#[derive(Clone, Copy)]
struct RouterTweak {
    tokens: usize,
    norm: bool,
    swap_outputs: bool,
}

fn router_arm(
    gpu: &mut Gpu,
    arm: Arm,
    logits: &GpuTensor,
    tokens: usize,
    norm: bool,
    tweak: RouterTweak,
) -> Res<RouterOut> {
    let n = tokens * TOPK;
    let ids = Out::poisoned(gpu, n)?;
    let weights = Out::poisoned(gpu, n)?;
    gpu.hip.device_synchronize().map_err(err)?;
    gpu.clear_router_fault();
    match arm {
        Arm::Reference => gpu
            .moe_router_softmax_top10_f32(
                logits,
                &ids.body,
                &weights.body,
                tokens,
                norm,
                true,
                LAYER,
                ROW_BASE,
            )
            .map_err(err)?,
        Arm::Product => gfx(gpu)?
            .g942_moe_wide_router(
                logits,
                &ids.body,
                &weights.body,
                tokens,
                norm,
                LAYER,
                ROW_BASE,
            )
            .map_err(err)?,
        Arm::Module => {
            let symbol = lab_ensure_moe_wide_entry(gpu, ROUTER).map_err(err)?;
            let (first, second) = if tweak.swap_outputs {
                (&weights.body, &ids.body)
            } else {
                (&ids.body, &weights.body)
            };
            let mut args = KernargBlob::new();
            args.push_ptr(logits.buf.as_ptr());
            args.push_ptr(first.buf.as_ptr());
            args.push_ptr(second.buf.as_ptr());
            for value in [E_MODEL, TOPK, usize::from(tweak.norm), 1, tweak.tokens] {
                args.push_i32(value as i32);
            }
            args.push_ptr(gpu.router_fault_ptr() as *const c_void);
            args.push_i32(LAYER as i32);
            args.push_i32(ROW_BASE as i32);
            args.pad_to(16);
            gpu.launch_kernel_blob(
                symbol,
                [tweak.tokens.div_ceil(4) as u32, 1, 1],
                [256, 1, 1],
                0,
                args.as_mut_slice(),
            )
            .map_err(err)?;
        }
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let fault = format!("{:?}", gpu.take_router_fault());
    let (ids_bytes, g1) = ids.read(gpu)?;
    let (weight_bytes, g2) = weights.read(gpu)?;
    ids.release(gpu);
    weights.release(gpu);
    Ok(RouterOut {
        ids: ids_bytes,
        weights: weight_bytes,
        fault,
        guards: g1 && g2,
    })
}

fn router_diff(a: &RouterOut, b: &RouterOut) -> usize {
    diff(&a.ids, &b.ids) + diff(&a.weights, &b.weights) + usize::from(a.fault != b.fault)
}

fn router_case(
    gpu: &mut Gpu,
    tally: &mut Tally,
    label: &str,
    tokens: usize,
    norm: bool,
    logits: &[f32],
) -> Res<()> {
    let lg = upload_f32s(gpu, logits)?;
    let clean = RouterTweak {
        tokens,
        norm,
        swap_outputs: false,
    };
    let r = router_arm(gpu, Arm::Reference, &lg, tokens, norm, clean)?;
    let r2 = router_arm(gpu, Arm::Reference, &lg, tokens, norm, clean)?;
    let m = router_arm(gpu, Arm::Module, &lg, tokens, norm, clean)?;
    let p = router_arm(gpu, Arm::Product, &lg, tokens, norm, clean)?;
    release(gpu, lg);
    let diffs = vec![
        ("module".to_owned(), router_diff(&m, &r)),
        ("product".to_owned(), router_diff(&p, &r)),
        ("reference_repeat".to_owned(), router_diff(&r2, &r)),
    ];
    let guards = r.guards && r2.guards && m.guards && p.guards;
    tally.case("router", label, &diffs, guards);
    Ok(())
}

fn router_controls(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally) -> Res<()> {
    let tokens = 32usize;
    let logits = router_logits(rng, tokens, 0);
    let lg = upload_f32s(gpu, &logits)?;
    let clean = RouterTweak {
        tokens,
        norm: true,
        swap_outputs: false,
    };
    let r = router_arm(gpu, Arm::Reference, &lg, tokens, true, clean)?;
    let m = router_arm(gpu, Arm::Module, &lg, tokens, true, clean)?;
    if router_diff(&m, &r) != 0 {
        return Err("router controls: the clean module run differs from the reference".to_owned());
    }
    // One flipped byte of the largest logit of token 0 (its exponent byte).
    let top = (0..E_MODEL)
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap_or(0);
    let mut flipped = logits.clone();
    flipped[top] = f32::from_bits(flipped[top].to_bits() ^ 0x4000_0000);
    let lg_flipped = upload_f32s(gpu, &flipped)?;
    let got = router_arm(gpu, Arm::Module, &lg_flipped, tokens, true, clean)?;
    tally.control("router flip_logit_byte", router_diff(&got, &r), got.guards);
    release(gpu, lg_flipped);
    let got = router_arm(
        gpu,
        Arm::Module,
        &lg,
        tokens,
        true,
        RouterTweak {
            tokens: tokens - 1,
            ..clean
        },
    )?;
    tally.control("router tokens_minus_1", router_diff(&got, &r), got.guards);
    let got = router_arm(
        gpu,
        Arm::Module,
        &lg,
        tokens,
        true,
        RouterTweak {
            norm: false,
            ..clean
        },
    )?;
    tally.control(
        "router normalize_flipped",
        router_diff(&got, &r),
        got.guards,
    );
    let got = router_arm(
        gpu,
        Arm::Module,
        &lg,
        tokens,
        true,
        RouterTweak {
            swap_outputs: true,
            ..clean
        },
    )?;
    tally.control(
        "router swapped_output_bindings",
        router_diff(&got, &r),
        got.guards,
    );
    release(gpu, lg);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Routings and the host grouping mirror
// ---------------------------------------------------------------------------------------------

const ROUTING_KINDS: usize = 13;

fn routing_name(kind: usize) -> &'static str {
    [
        "uniform",
        "per_token_distinct",
        "one_expert",
        "one_expert_last",
        "two_experts",
        "hot_set",
        "sequential",
        "groups_of_16",
        "groups_of_15",
        "top_of_range",
        "uniform_markers",
        "all_unrouted",
        "distinct_markers",
    ][kind]
}

fn markers(e: usize) -> [i32; 7] {
    [-1, e as i32, e as i32 + 1, 99_999, i32::MIN, i32::MAX, -7]
}

fn gen_routing(rng: &mut Rng, kind: usize, l: usize, e: usize) -> Vec<i32> {
    let mut topk: Vec<i32> = match kind {
        0 | 10 => (0..l).map(|_| rng.below(e) as i32).collect(),
        1 | 12 => {
            let k = TOPK.min(e);
            let mut out = Vec::with_capacity(l);
            while out.len() < l {
                let mut row = Vec::new();
                while row.len() < k {
                    let x = rng.below(e) as i32;
                    if !row.contains(&x) {
                        row.push(x);
                    }
                }
                out.extend(row);
            }
            out.truncate(l);
            out
        }
        2 => {
            let x = rng.below(e) as i32;
            vec![x; l]
        }
        3 => vec![e as i32 - 1; l],
        4 => {
            let (a, b) = (rng.below(e) as i32, rng.below(e) as i32);
            (0..l).map(|s| if s % 2 == 0 { a } else { b }).collect()
        }
        5 => {
            let hot: Vec<i32> = (0..8).map(|_| rng.below(e) as i32).collect();
            (0..l)
                .map(|_| {
                    if rng.below(10) < 9 {
                        hot[rng.below(8)]
                    } else {
                        rng.below(e) as i32
                    }
                })
                .collect()
        }
        6 => (0..l).map(|s| (s % e) as i32).collect(),
        7 => (0..l).map(|s| (s / 16 % e) as i32).collect(),
        8 => (0..l).map(|s| (s / 15 % e) as i32).collect(),
        9 => (0..l)
            .map(|_| (e - 1 - rng.below(TOPK.min(e))) as i32)
            .collect(),
        _ => vec![-1; l],
    };
    let marker_rate = match kind {
        10 => 20,
        12 => 33,
        _ => 0,
    };
    if marker_rate > 0 {
        for slot in topk.iter_mut() {
            if rng.below(marker_rate) == 0 {
                *slot = rng.pick(&markers(e));
            }
        }
    }
    if kind == 11 {
        for slot in topk.iter_mut() {
            *slot = rng.pick(&markers(e));
        }
    }
    topk
}

#[derive(Clone)]
struct Group {
    counts: Vec<i32>,
    offsets: Vec<i32>,
    sorted: Vec<i32>,
    tiles: Vec<i32>,
    inverse: Vec<i32>,
    desc: Vec<i32>,
    used: usize,
}

/// Stable pad16 grouping + tile descriptors, the specification the scatter entries implement.
fn host_group(topk: &[i32], e: usize, pmax: usize) -> Group {
    let l = topk.len();
    let routed = |t: i32| (t >= 0 && (t as usize) < e).then_some(t as usize);
    let mut count = vec![0usize; e];
    for &t in topk {
        if let Some(x) = routed(t) {
            count[x] += 1;
        }
    }
    let pad: Vec<usize> = count.iter().map(|&c| round16(c)).collect();
    let mut offsets = vec![0usize; e + 1];
    for i in 0..e {
        offsets[i + 1] = offsets[i] + pad[i];
    }
    let used = offsets[e];
    let mut sorted = vec![-1i32; pmax];
    let mut inverse = vec![-1i32; l];
    let mut fill = vec![0usize; e];
    for (s, &t) in topk.iter().enumerate() {
        if let Some(x) = routed(t) {
            let pos = offsets[x] + fill[x];
            fill[x] += 1;
            sorted[pos] = s as i32;
            inverse[s] = pos as i32;
        }
    }
    let mut tiles = vec![-1i32; pmax / 16];
    let mut desc = vec![0i32; pmax / 16 * 4];
    for x in 0..e {
        for p in (offsets[x]..offsets[x + 1]).step_by(16) {
            tiles[p / 16] = x as i32;
            desc[p / 16 * 4..p / 16 * 4 + 4].copy_from_slice(&[x as i32, p as i32, 16, 0]);
        }
    }
    for t in used / 16..pmax / 16 {
        desc[t * 4..t * 4 + 4].copy_from_slice(&[-1, (t * 16) as i32, 16, 0]);
    }
    Group {
        counts: pad.iter().map(|&c| c as i32).collect(),
        offsets: offsets.iter().map(|&c| c as i32).collect(),
        sorted,
        tiles,
        inverse,
        desc,
        used,
    }
}

/// `grouped_rows` of mode 0 (the worst-case bound), 1 (the tightest legal), 2 (tight + slack).
fn pick_pmax(rng: &mut Rng, mode: usize, l: usize, e: usize, used: usize) -> usize {
    let bound = round16(l + 15 * e.min(l));
    let tight = used.max(round16(l));
    match mode {
        0 => bound,
        1 => tight,
        _ => (tight + 16 * rng.below(5)).min(bound.max(tight)),
    }
}

// ---------------------------------------------------------------------------------------------
// Scatter
// ---------------------------------------------------------------------------------------------

const SC_PARTS: [&str; 6] = [
    "counts",
    "offsets",
    "sorted",
    "tiles",
    "inverse",
    "tile_desc",
];

struct ScOut {
    parts: Vec<Vec<u8>>,
    guards: bool,
}

#[derive(Clone, Copy)]
struct ScShape {
    l: usize,
    e: usize,
    pmax: usize,
}

fn sc_words(s: ScShape) -> [usize; 6] {
    [s.e, s.e + 1, s.pmax, s.pmax / 16, s.l, s.pmax / 16 * 4]
}

/// `launch`: the (l, e, pmax) the arm passes; the buffers are always sized by `shape`.
fn scatter_arm(
    gpu: &mut Gpu,
    arm: Arm,
    topk: &GpuTensor,
    shape: ScShape,
    launch: ScShape,
) -> Res<ScOut> {
    let outs: Vec<Out> = sc_words(shape)
        .iter()
        .map(|&w| Out::poisoned(gpu, w))
        .collect::<Res<_>>()?;
    let b = |i: usize| &outs[i].body;
    match arm {
        Arm::Reference => gfx(gpu)?
            .g942_fold_moe_group_scatter_desc(
                topk,
                b(0),
                b(1),
                b(2),
                b(3),
                b(4),
                b(5),
                launch.l,
                launch.e,
                launch.pmax,
            )
            .map_err(err)?,
        Arm::Product => gfx(gpu)?
            .g942_moe_wide_group_scatter_desc(
                topk,
                b(0),
                b(1),
                b(2),
                b(3),
                b(4),
                b(5),
                launch.l,
                launch.e,
                launch.pmax,
            )
            .map_err(err)?,
        Arm::Module => {
            let symbol = lab_ensure_moe_wide_entry(gpu, SCATTER).map_err(err)?;
            let mut args = KernargBlob::new();
            args.push_ptr(topk.buf.as_ptr());
            for out in &outs {
                args.push_ptr(out.body.buf.as_ptr());
            }
            for value in [launch.l, launch.e, launch.pmax] {
                args.push_i32(value as i32);
            }
            args.pad_to(16);
            gpu.launch_kernel_blob(symbol, [1, 1, 1], [1024, 1, 1], 0, args.as_mut_slice())
                .map_err(err)?;
        }
    }
    let mut parts = Vec::new();
    let mut guards = true;
    for out in outs {
        let (bytes, ok) = out.read(gpu)?;
        parts.push(bytes);
        guards &= ok;
        out.release(gpu);
    }
    Ok(ScOut { parts, guards })
}

fn sc_diffs(prefix: &str, a: &ScOut, b: &ScOut) -> Vec<(String, usize)> {
    SC_PARTS
        .iter()
        .enumerate()
        .map(|(i, name)| (format!("{prefix}.{name}"), diff(&a.parts[i], &b.parts[i])))
        .collect()
}

/// A grouping handed to the unscatter / combine stages.
struct Grouped {
    l: usize,
    e: usize,
    pmax: usize,
    sorted: Vec<i32>,
    inverse: Vec<i32>,
}

fn scatter_case(
    gpu: &mut Gpu,
    rng: &mut Rng,
    tally: &mut Tally,
    kind: usize,
    l: usize,
    e: usize,
    pmax_mode: usize,
) -> Res<(Vec<i32>, Grouped)> {
    let topk = gen_routing(rng, kind, l, e);
    let mirror = host_group(&topk, e, round16(l + 15 * e.min(l)));
    let pmax = pick_pmax(rng, pmax_mode, l, e, mirror.used);
    let mirror = host_group(&topk, e, pmax);
    let label = format!("{} L={l} E={e} pmax={pmax}", routing_name(kind));
    let topk_dev = upload_i32s(gpu, &topk)?;
    let shape = ScShape { l, e, pmax };
    let r = scatter_arm(gpu, Arm::Reference, &topk_dev, shape, shape)?;
    let r2 = scatter_arm(gpu, Arm::Reference, &topk_dev, shape, shape)?;
    let m = scatter_arm(gpu, Arm::Module, &topk_dev, shape, shape)?;
    let p = scatter_arm(gpu, Arm::Product, &topk_dev, shape, shape)?;
    release(gpu, topk_dev);
    let mut diffs = sc_diffs("module", &m, &r);
    diffs.extend(sc_diffs("product", &p, &r));
    diffs.push((
        "reference_repeat".to_owned(),
        sc_diffs("", &r2, &r).iter().map(|(_, d)| d).sum(),
    ));
    let guards = r.guards && r2.guards && m.guards && p.guards;
    tally.case("scatter", &label, &diffs, guards);
    let mirror_parts = [
        i32s_to_bytes(&mirror.counts),
        i32s_to_bytes(&mirror.offsets),
        i32s_to_bytes(&mirror.sorted),
        i32s_to_bytes(&mirror.tiles),
        i32s_to_bytes(&mirror.inverse),
        i32s_to_bytes(&mirror.desc),
    ];
    if mirror_parts
        .iter()
        .zip(&r.parts)
        .any(|(host, device)| diff(host, device) != 0)
    {
        MIRROR_MISMATCH.fetch_add(1, Ordering::Relaxed);
        eprintln!("note: host grouping mirror differs from the reference at {label}");
    }
    let grouped = Grouped {
        l,
        e,
        pmax,
        sorted: bytes_to_i32s(&r.parts[2]),
        inverse: bytes_to_i32s(&r.parts[4]),
    };
    Ok((topk, grouped))
}

fn scatter_controls(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally) -> Res<()> {
    let (l, e) = (320usize, E_MODEL);
    let mut topk = gen_routing(rng, 0, l, e);
    topk[l - 1] = e as i32 - 1;
    let pmax = round16(l + 15 * e.min(l));
    let shape = ScShape { l, e, pmax };
    let topk_dev = upload_i32s(gpu, &topk)?;
    let r = scatter_arm(gpu, Arm::Reference, &topk_dev, shape, shape)?;
    let m = scatter_arm(gpu, Arm::Module, &topk_dev, shape, shape)?;
    if sc_diffs("", &m, &r).iter().any(|(_, d)| *d != 0) || !m.guards {
        return Err("scatter controls: the clean module run differs from the reference".to_owned());
    }
    let total = |a: &ScOut| -> usize { sc_diffs("", a, &r).iter().map(|(_, d)| d).sum() };
    let tweaks = [
        ("slots_minus_1", ScShape { l: l - 1, ..shape }),
        ("experts_minus_1", ScShape { e: e - 1, ..shape }),
        (
            "grouped_rows_plus_16",
            ScShape {
                pmax: pmax + 16,
                ..shape
            },
        ),
    ];
    for (name, launch) in tweaks {
        let got = scatter_arm(gpu, Arm::Module, &topk_dev, shape, launch)?;
        tally.control(&format!("scatter {name}"), total(&got), got.guards);
    }
    release(gpu, topk_dev);
    for (name, mutate) in [
        ("flip_topk_byte", 0usize),
        ("inject_unrouted_marker", 1usize),
    ] {
        let mut t = topk.clone();
        if mutate == 0 {
            t[0] ^= 0x04;
        } else {
            t[1] = -1;
        }
        let dev = upload_i32s(gpu, &t)?;
        let got = scatter_arm(gpu, Arm::Module, &dev, shape, shape)?;
        tally.control(&format!("scatter {name}"), total(&got), got.guards);
        release(gpu, dev);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Unscatter + shared activation
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
struct UnHost {
    y: Vec<f32>,
    sorted: Vec<i32>,
    inverse: Vec<i32>,
    sg: Vec<f32>,
    su: Vec<f32>,
    sel: Vec<f32>,
}

#[derive(Clone, Copy)]
struct UnGeo {
    l: usize,
    pmax: usize,
    mi: usize,
    shmi: usize,
    rows: usize,
}

struct UnDev {
    y: GpuTensor,
    sorted: GpuTensor,
    inverse: GpuTensor,
    sg: GpuTensor,
    su: GpuTensor,
}

impl UnHost {
    fn upload(&self, gpu: &Gpu) -> Res<UnDev> {
        Ok(UnDev {
            y: upload_f32s(gpu, &self.y)?,
            sorted: upload_i32s(gpu, &self.sorted)?,
            inverse: upload_i32s(gpu, &self.inverse)?,
            sg: upload_f32s(gpu, &self.sg)?,
            su: upload_f32s(gpu, &self.su)?,
        })
    }
}

impl UnDev {
    fn release(self, gpu: &Gpu) {
        for t in [self.y, self.sorted, self.inverse, self.sg, self.su] {
            release(gpu, t);
        }
    }
}

struct UnOut {
    act: Vec<u8>,
    shared_act: Vec<u8>,
    selector: Vec<u8>,
    guards: bool,
}

#[derive(Clone, Copy)]
struct UnTweak {
    slots: usize,
    rows: usize,
    swap_shared: bool,
}

fn un_diff(a: &UnOut, b: &UnOut) -> usize {
    diff(&a.act, &b.act) + diff(&a.shared_act, &b.shared_act) + diff(&a.selector, &b.selector)
}

fn unscatter_arm(
    gpu: &mut Gpu,
    arm: Arm,
    dev: &UnDev,
    sel_host: &[f32],
    geo: UnGeo,
    tweak: UnTweak,
) -> Res<UnOut> {
    let act = Out::poisoned(gpu, geo.l * geo.mi)?;
    let shared_act = Out::poisoned(gpu, geo.rows * geo.shmi)?;
    let sel_words: Vec<u32> = sel_host.iter().map(|v| v.to_bits()).collect();
    let selector = Out::with_body(gpu, &sel_words)?;
    match arm {
        Arm::Reference => gfx(gpu)?
            .g942_fold_moe_unscatter_act(
                &dev.y,
                &dev.sorted,
                &act.body,
                geo.mi,
                geo.pmax,
                geo.l,
                &dev.sg,
                &dev.su,
                &shared_act.body,
                &selector.body,
                geo.rows,
                geo.shmi,
            )
            .map_err(err)?,
        Arm::Product => gfx(gpu)?
            .g942_moe_wide_unscatter_act(
                &dev.y,
                &dev.inverse,
                &act.body,
                geo.mi,
                geo.pmax,
                geo.l,
                &dev.sg,
                &dev.su,
                &shared_act.body,
                &selector.body,
                geo.rows,
                geo.shmi,
            )
            .map_err(err)?,
        Arm::Module => {
            let symbol = lab_ensure_moe_wide_entry(gpu, UNSCATTER).map_err(err)?;
            let grid_y = geo.mi.div_ceil(256);
            let act_blocks = (geo.rows * geo.shmi).div_ceil(256);
            let (gate, up) = if tweak.swap_shared {
                (&dev.su, &dev.sg)
            } else {
                (&dev.sg, &dev.su)
            };
            let mut args = KernargBlob::new();
            args.push_ptr(dev.y.buf.as_ptr());
            args.push_ptr(dev.inverse.buf.as_ptr());
            args.push_ptr(act.body.buf.as_ptr());
            for value in [geo.mi, geo.pmax, tweak.slots] {
                args.push_i32(value as i32);
            }
            args.push_ptr(gate.buf.as_ptr());
            args.push_ptr(up.buf.as_ptr());
            args.push_ptr(shared_act.body.buf.as_ptr());
            args.push_ptr(selector.body.buf.as_ptr());
            for value in [tweak.rows, geo.shmi, act_blocks] {
                args.push_i32(value as i32);
            }
            args.pad_to(16);
            let grid_x = tweak.slots + act_blocks.div_ceil(grid_y);
            gpu.launch_kernel_blob(
                symbol,
                [grid_x as u32, grid_y as u32, 1],
                [256, 1, 1],
                0,
                args.as_mut_slice(),
            )
            .map_err(err)?;
        }
    }
    let (a, g1) = act.read(gpu)?;
    let (s, g2) = shared_act.read(gpu)?;
    let (l, g3) = selector.read(gpu)?;
    act.release(gpu);
    shared_act.release(gpu);
    selector.release(gpu);
    Ok(UnOut {
        act: a,
        shared_act: s,
        selector: l,
        guards: g1 && g2 && g3,
    })
}

fn unscatter_host(
    rng: &mut Rng,
    g: &Grouped,
    mi: usize,
    shmi: usize,
    kind: usize,
) -> (UnHost, UnGeo) {
    let rows = g.l.div_ceil(TOPK);
    let host = UnHost {
        y: fvals(rng, g.pmax * 2 * mi, kind),
        sorted: g.sorted.clone(),
        inverse: g.inverse.clone(),
        sg: fvals(rng, rows * shmi, kind),
        su: fvals(rng, rows * shmi, kind),
        sel: fvals(rng, rows, kind),
    };
    let geo = UnGeo {
        l: g.l,
        pmax: g.pmax,
        mi,
        shmi,
        rows,
    };
    (host, geo)
}

#[allow(clippy::too_many_arguments)]
fn unscatter_case(
    gpu: &mut Gpu,
    rng: &mut Rng,
    tally: &mut Tally,
    label: &str,
    g: &Grouped,
    mi: usize,
    shmi: usize,
    kind: usize,
) -> Res<()> {
    let (host, geo) = unscatter_host(rng, g, mi, shmi, kind);
    let dev = host.upload(gpu)?;
    let clean = UnTweak {
        slots: geo.l,
        rows: geo.rows,
        swap_shared: false,
    };
    let r = unscatter_arm(gpu, Arm::Reference, &dev, &host.sel, geo, clean)?;
    let r2 = unscatter_arm(gpu, Arm::Reference, &dev, &host.sel, geo, clean)?;
    let m = unscatter_arm(gpu, Arm::Module, &dev, &host.sel, geo, clean)?;
    let p = unscatter_arm(gpu, Arm::Product, &dev, &host.sel, geo, clean)?;
    dev.release(gpu);
    let diffs = vec![
        ("module".to_owned(), un_diff(&m, &r)),
        ("product".to_owned(), un_diff(&p, &r)),
        ("reference_repeat".to_owned(), un_diff(&r2, &r)),
    ];
    let guards = r.guards && r2.guards && m.guards && p.guards;
    tally.case(
        "unscatter",
        &format!("{label} mi={mi} shared_mi={shmi} values={kind}"),
        &diffs,
        guards,
    );
    Ok(())
}

fn flip(value: f32) -> f32 {
    f32::from_bits(value.to_bits() ^ 0x4000_0000)
}

fn unscatter_controls(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally, g: &Grouped) -> Res<()> {
    let (mi, shmi) = (640usize, 640usize);
    let (mut host, geo) = unscatter_host(rng, g, mi, shmi, 0);
    let s0 = g
        .inverse
        .iter()
        .position(|&r| r >= 0)
        .ok_or("unscatter controls: no routed slot")?;
    let row0 = g.inverse[s0] as usize;
    // Known anchor values so every flip changes the BF16 result.
    host.y[row0 * 2 * mi + s0 % mi] = 1.5;
    host.y[row0 * 2 * mi + mi + s0 % mi] = -1.25;
    host.sg[0] = 1.5;
    host.su[0] = -1.25;
    host.sel[0] = 1.5;
    let clean = UnTweak {
        slots: geo.l,
        rows: geo.rows,
        swap_shared: false,
    };
    let dev = host.upload(gpu)?;
    let r = unscatter_arm(gpu, Arm::Reference, &dev, &host.sel, geo, clean)?;
    let m = unscatter_arm(gpu, Arm::Module, &dev, &host.sel, geo, clean)?;
    if un_diff(&m, &r) != 0 || !m.guards {
        return Err(
            "unscatter controls: the clean module run differs from the reference".to_owned(),
        );
    }
    let launch_controls = [
        (
            "slots_minus_1",
            UnTweak {
                slots: geo.l - 1,
                ..clean
            },
        ),
        (
            "rows_minus_1",
            UnTweak {
                rows: geo.rows - 1,
                ..clean
            },
        ),
        (
            "swapped_shared_bindings",
            UnTweak {
                swap_shared: true,
                ..clean
            },
        ),
    ];
    for (name, tweak) in launch_controls {
        let got = unscatter_arm(gpu, Arm::Module, &dev, &host.sel, geo, tweak)?;
        tally.control(&format!("unscatter {name}"), un_diff(&got, &r), got.guards);
    }
    dev.release(gpu);
    type Mutate = fn(&mut UnHost, usize, usize, usize);
    let mutations: [(&str, Mutate); 6] = [
        ("flip_gate_up_gate_byte", |h, row0, s0, mi| {
            h.y[row0 * 2 * mi + s0 % mi] = flip(h.y[row0 * 2 * mi + s0 % mi])
        }),
        ("flip_gate_up_up_byte", |h, row0, s0, mi| {
            h.y[row0 * 2 * mi + mi + s0 % mi] = flip(h.y[row0 * 2 * mi + mi + s0 % mi])
        }),
        ("flip_shared_gate_byte", |h, _, _, _| {
            h.sg[0] = flip(h.sg[0])
        }),
        ("flip_shared_up_byte", |h, _, _, _| h.su[0] = flip(h.su[0])),
        ("flip_selector_byte", |h, _, _, _| h.sel[0] = flip(h.sel[0])),
        ("flip_inverse_row", |h, _, s0, _| {
            let other = h
                .inverse
                .iter()
                .copied()
                .find(|&r| r >= 0 && r != h.inverse[s0])
                .unwrap_or(0);
            h.inverse[s0] = other;
        }),
    ];
    for (name, mutate) in mutations {
        let mut h = host.clone();
        mutate(&mut h, row0, s0, mi);
        let dev = h.upload(gpu)?;
        let got = unscatter_arm(gpu, Arm::Module, &dev, &h.sel, geo, clean)?;
        tally.control(&format!("unscatter {name}"), un_diff(&got, &r), got.guards);
        dev.release(gpu);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Combine + shared add
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
struct CbHost {
    gd: Vec<f32>,
    inverse: Vec<i32>,
    idx: Vec<i32>,
    w: Vec<f32>,
    res: Vec<f32>,
    sh: Vec<f32>,
    sel: Vec<f32>,
}

#[derive(Clone, Copy)]
struct CbGeo {
    tokens: usize,
    hidden: usize,
    pmax: usize,
    n_exp: usize,
}

struct CbDev {
    gd: GpuTensor,
    inverse: GpuTensor,
    idx: GpuTensor,
    w: GpuTensor,
    sh: GpuTensor,
    sel: GpuTensor,
}

impl CbHost {
    fn upload(&self, gpu: &Gpu) -> Res<CbDev> {
        Ok(CbDev {
            gd: upload_f32s(gpu, &self.gd)?,
            inverse: upload_i32s(gpu, &self.inverse)?,
            idx: upload_i32s(gpu, &self.idx)?,
            w: upload_f32s(gpu, &self.w)?,
            sh: upload_f32s(gpu, &self.sh)?,
            sel: upload_f32s(gpu, &self.sel)?,
        })
    }
}

impl CbDev {
    fn release(self, gpu: &Gpu) {
        for t in [self.gd, self.inverse, self.idx, self.w, self.sh, self.sel] {
            release(gpu, t);
        }
    }
}

#[derive(Clone, Copy)]
struct CbTweak {
    tokens: usize,
    n_exp: usize,
    swap_target_shared: bool,
}

struct CbOut {
    residual: Vec<u8>,
    guards: bool,
}

fn combine_arm(
    gpu: &mut Gpu,
    arm: Arm,
    dev: &CbDev,
    res_host: &[f32],
    geo: CbGeo,
    tweak: CbTweak,
) -> Res<CbOut> {
    let res_words: Vec<u32> = res_host.iter().map(|v| v.to_bits()).collect();
    let res = Out::with_body(gpu, &res_words)?;
    match arm {
        Arm::Reference => gfx(gpu)?
            .g942_fold_moe_combine_shared_add(
                &dev.gd,
                &dev.inverse,
                &dev.idx,
                &dev.w,
                &res.body,
                &dev.sh,
                &dev.sel,
                geo.hidden,
                geo.pmax,
                geo.tokens,
                geo.n_exp,
            )
            .map_err(err)?,
        Arm::Product => gfx(gpu)?
            .g942_moe_wide_combine_shared_add(
                &dev.gd,
                &dev.inverse,
                &dev.idx,
                &dev.w,
                &res.body,
                &dev.sh,
                &dev.sel,
                geo.hidden,
                geo.pmax,
                geo.tokens,
                geo.n_exp,
            )
            .map_err(err)?,
        Arm::Module => {
            let symbol = lab_ensure_moe_wide_entry(gpu, COMBINE).map_err(err)?;
            let (target, shared) = if tweak.swap_target_shared {
                (&dev.sh, &res.body)
            } else {
                (&res.body, &dev.sh)
            };
            let mut args = KernargBlob::new();
            args.push_ptr(dev.gd.buf.as_ptr());
            args.push_ptr(dev.inverse.buf.as_ptr());
            args.push_ptr(dev.idx.buf.as_ptr());
            args.push_ptr(dev.w.buf.as_ptr());
            args.push_ptr(target.buf.as_ptr());
            for value in [geo.hidden, geo.pmax, tweak.tokens, tweak.n_exp] {
                args.push_i32(value as i32);
            }
            args.push_ptr(shared.buf.as_ptr());
            args.push_ptr(dev.sel.buf.as_ptr());
            args.pad_to(16);
            gpu.launch_kernel_blob(
                symbol,
                [geo.hidden.div_ceil(256) as u32, tweak.tokens as u32, 1],
                [256, 1, 1],
                0,
                args.as_mut_slice(),
            )
            .map_err(err)?;
        }
    }
    let (residual, guards) = res.read(gpu)?;
    res.release(gpu);
    Ok(CbOut { residual, guards })
}

/// Combine inputs on the grouping of `topk` (`tokens * 10` slots): `corrupt` 0 clean, 1 random
/// out-of-range inverse rows, 2 out-of-range expert ids.
fn combine_host(
    rng: &mut Rng,
    g: &Grouped,
    topk: &[i32],
    hidden: usize,
    corrupt: usize,
    kind: usize,
) -> (CbHost, CbGeo) {
    let tokens = g.l / TOPK;
    let mut inverse = g.inverse.clone();
    let mut idx = topk.to_vec();
    match corrupt {
        1 => {
            for r in inverse.iter_mut() {
                if rng.below(20) == 0 {
                    *r = rng.pick(&[-1, g.pmax as i32, g.pmax as i32 + 5, i32::MAX, i32::MIN]);
                }
            }
        }
        2 => {
            for x in idx.iter_mut() {
                if rng.below(20) == 0 {
                    *x = rng.pick(&markers(g.e));
                }
            }
        }
        _ => {}
    }
    let host = CbHost {
        gd: fvals(rng, g.pmax * hidden, kind),
        inverse,
        idx,
        w: (0..tokens * TOPK)
            .map(|_| {
                if rng.below(8) == 0 {
                    0.0
                } else {
                    rng.unit() as f32
                }
            })
            .collect(),
        res: fvals(rng, tokens * hidden, kind),
        sh: fvals(rng, tokens * hidden, kind),
        sel: (0..tokens).map(|_| rng.unit() as f32).collect(),
    };
    let geo = CbGeo {
        tokens,
        hidden,
        pmax: g.pmax,
        n_exp: g.e,
    };
    (host, geo)
}

fn cb_diff(a: &CbOut, b: &CbOut) -> usize {
    diff(&a.residual, &b.residual)
}

fn combine_case(
    gpu: &mut Gpu,
    tally: &mut Tally,
    label: &str,
    host: &CbHost,
    geo: CbGeo,
    kind: usize,
    corrupt: usize,
) -> Res<()> {
    let dev = host.upload(gpu)?;
    let clean = CbTweak {
        tokens: geo.tokens,
        n_exp: geo.n_exp,
        swap_target_shared: false,
    };
    let r = combine_arm(gpu, Arm::Reference, &dev, &host.res, geo, clean)?;
    let r2 = combine_arm(gpu, Arm::Reference, &dev, &host.res, geo, clean)?;
    let m = combine_arm(gpu, Arm::Module, &dev, &host.res, geo, clean)?;
    let p = combine_arm(gpu, Arm::Product, &dev, &host.res, geo, clean)?;
    dev.release(gpu);
    let diffs = vec![
        ("module".to_owned(), cb_diff(&m, &r)),
        ("product".to_owned(), cb_diff(&p, &r)),
        ("reference_repeat".to_owned(), cb_diff(&r2, &r)),
    ];
    let guards = r.guards && r2.guards && m.guards && p.guards;
    tally.case(
        "combine",
        &format!(
            "{label} hidden={} values={kind} corrupt={corrupt}",
            geo.hidden
        ),
        &diffs,
        guards,
    );
    Ok(())
}

fn combine_controls(gpu: &mut Gpu, rng: &mut Rng, tally: &mut Tally) -> Res<()> {
    let (tokens, e, hidden) = (32usize, E_MODEL, 2560usize);
    let topk = gen_routing(rng, 0, tokens * TOPK, e);
    let pmax = round16(tokens * TOPK + 15 * e.min(tokens * TOPK));
    let group = host_group(&topk, e, pmax);
    let g = Grouped {
        l: tokens * TOPK,
        e,
        pmax,
        sorted: group.sorted,
        inverse: group.inverse,
    };
    let (mut host, geo) = combine_host(rng, &g, &topk, hidden, 0, 0);
    let row0 = g.inverse[0] as usize;
    host.gd[row0 * hidden] = 1.5;
    host.w[0] = 0.5;
    host.res[0] = 0.25;
    host.sh[0] = 0.5;
    host.sel[0] = 1.0;
    let clean = CbTweak {
        tokens,
        n_exp: e,
        swap_target_shared: false,
    };
    let dev = host.upload(gpu)?;
    let r = combine_arm(gpu, Arm::Reference, &dev, &host.res, geo, clean)?;
    let m = combine_arm(gpu, Arm::Module, &dev, &host.res, geo, clean)?;
    if cb_diff(&m, &r) != 0 || !m.guards {
        return Err("combine controls: the clean module run differs from the reference".to_owned());
    }
    for (name, tweak) in [
        (
            "tokens_minus_1",
            CbTweak {
                tokens: tokens - 1,
                ..clean
            },
        ),
        ("n_experts_1", CbTweak { n_exp: 1, ..clean }),
        (
            "swapped_target_shared_bindings",
            CbTweak {
                swap_target_shared: true,
                ..clean
            },
        ),
    ] {
        let got = combine_arm(gpu, Arm::Module, &dev, &host.res, geo, tweak)?;
        tally.control(&format!("combine {name}"), cb_diff(&got, &r), got.guards);
    }
    dev.release(gpu);
    type Mutate = fn(&mut CbHost, usize, usize);
    let mutations: [(&str, Mutate); 7] = [
        ("flip_grouped_down_byte", |h, row0, hidden| {
            h.gd[row0 * hidden] = flip(h.gd[row0 * hidden])
        }),
        ("flip_inverse_row", |h, _, _| h.inverse[0] ^= 0x4000_0000),
        ("flip_topk_index", |h, _, _| h.idx[0] ^= 0x4000_0000),
        ("flip_topk_weight_byte", |h, _, _| h.w[0] = flip(h.w[0])),
        ("flip_residual_byte", |h, _, _| h.res[0] = flip(h.res[0])),
        ("flip_shared_out_byte", |h, _, _| h.sh[0] = flip(h.sh[0])),
        ("flip_selector_byte", |h, _, _| h.sel[0] = flip(h.sel[0])),
    ];
    for (name, mutate) in mutations {
        let mut h = host.clone();
        mutate(&mut h, row0, hidden);
        let dev = h.upload(gpu)?;
        let got = combine_arm(gpu, Arm::Module, &dev, &h.res, geo, clean)?;
        tally.control(&format!("combine {name}"), cb_diff(&got, &r), got.guards);
        dev.release(gpu);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Chains
// ---------------------------------------------------------------------------------------------

/// The unscatter (and, when `slots` is a whole number of rows, the combine) on one grouping.
fn chain(
    gpu: &mut Gpu,
    rng: &mut Rng,
    tally: &mut Tally,
    label: &str,
    topk: &[i32],
    g: &Grouped,
) -> Res<()> {
    let (mi, shmi) = rng.pick(&[(640, 640), (640, 640), (257, 100), (256, 513), (1, 1)]);
    let kind = rng.below(4);
    unscatter_case(gpu, rng, tally, label, g, mi, shmi, kind)?;
    if g.l.is_multiple_of(TOPK) {
        let hidden = rng.pick(&[2560, 2560, 257, 256, 1]);
        let corrupt = rng.below(3);
        let kind = rng.below(4);
        let (host, geo) = combine_host(rng, g, topk, hidden, corrupt, kind);
        combine_case(gpu, tally, label, &host, geo, kind, corrupt)?;
    }
    Ok(())
}

fn main() {
    let args = Args::parse(0x4d57_4944_4531_0001);
    let mut gpu = init_gfx942();
    println!(
        "{}",
        module_banner("qwen4_g942_moe_wide", "HIPFIRE_QWEN4_G942_MOE_WIDE_IMAGE")
    );
    let mut rng = Rng(args.seed);
    let mut tally = Tally::default();
    let quick = args.quick;
    let run = |r: Res<()>| {
        if let Err(e) = r {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    // Router.
    let tokens_list: &[usize] = if quick {
        &[8, 9, 32, 63, 64]
    } else {
        &[8, 9, 10, 16, 17, 31, 32, 33, 48, 63, 64]
    };
    for &tokens in tokens_list {
        for kind in 0..ROUTER_KINDS {
            if quick && kind % 3 == 1 {
                continue;
            }
            for norm in [false, true] {
                let logits = router_logits(&mut rng, tokens, kind);
                let label = format!("T={tokens} norm={} logits={kind}", u8::from(norm));
                run(router_case(
                    &mut gpu, &mut tally, &label, tokens, norm, &logits,
                ));
            }
        }
    }
    println!("router: cumulative PASS {} FAIL {}", tally.pass, tally.fail);

    // Scatter, then the unscatter and combine on the same groupings.
    let slot_list: &[usize] = if quick {
        &[80, 81, 320, 639, 640]
    } else {
        &[80, 81, 89, 90, 160, 319, 320, 321, 479, 512, 513, 639, 640]
    };
    let expert_list = [
        E_MODEL, E_MODEL, E_MODEL, 1, 2, 7, 10, 33, 64, 100, 256, 500, 511,
    ];
    let mut index = 0usize;
    for &l in slot_list {
        for kind in 0..ROUTING_KINDS {
            if quick && kind % 2 == 1 && kind != 11 {
                continue;
            }
            let e = if kind % 3 == 0 || kind == 1 {
                E_MODEL
            } else {
                rng.pick(&expert_list)
            };
            let mode = rng.below(3);
            match scatter_case(&mut gpu, &mut rng, &mut tally, kind, l, e, mode) {
                Ok((topk, grouped)) => {
                    let every = if quick { 4 } else { 2 };
                    if index.is_multiple_of(every) && grouped.pmax > 0 {
                        let label = format!("{} L={l} E={e}", routing_name(kind));
                        run(chain(
                            &mut gpu, &mut rng, &mut tally, &label, &topk, &grouped,
                        ));
                    }
                }
                Err(e) => run(Err(e)),
            }
            index += 1;
        }
    }
    println!(
        "scatter + chain: cumulative PASS {} FAIL {}",
        tally.pass, tally.fail
    );

    // Slot counts beyond the scatter's 640: the unscatter and the combine take no upper cap
    // (host-built grouping; the scatter entry itself is not involved).
    let big_tokens: &[usize] = if quick {
        &[70]
    } else {
        &[65, 70, 100, 128, 256]
    };
    for &tokens in big_tokens {
        for kind in [0usize, 2, 12] {
            let (l, e) = (tokens * TOPK, E_MODEL);
            let topk = gen_routing(&mut rng, kind, l, e);
            let pmax = round16(l + 15 * e.min(l));
            let group = host_group(&topk, e, pmax);
            let grouped = Grouped {
                l,
                e,
                pmax,
                sorted: group.sorted,
                inverse: group.inverse,
            };
            let label = format!("big {} L={l} E={e}", routing_name(kind));
            run(chain(
                &mut gpu, &mut rng, &mut tally, &label, &topk, &grouped,
            ));
        }
    }
    println!(
        "beyond-640 chain: cumulative PASS {} FAIL {}",
        tally.pass, tally.fail
    );

    // Planted controls.
    run(router_controls(&mut gpu, &mut rng, &mut tally));
    run(scatter_controls(&mut gpu, &mut rng, &mut tally));
    {
        let (l, e) = (320usize, E_MODEL);
        let topk = gen_routing(&mut rng, 0, l, e);
        let pmax = round16(l + 15 * e.min(l));
        // The grouping the controls feed the unscatter is the reference scatter's.
        let topk_dev = match upload_i32s(&gpu, &topk) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        let shape = ScShape { l, e, pmax };
        let r = match scatter_arm(&mut gpu, Arm::Reference, &topk_dev, shape, shape) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        release(&gpu, topk_dev);
        let grouped = Grouped {
            l,
            e,
            pmax,
            sorted: bytes_to_i32s(&r.parts[2]),
            inverse: bytes_to_i32s(&r.parts[4]),
        };
        run(unscatter_controls(&mut gpu, &mut rng, &mut tally, &grouped));
    }
    run(combine_controls(&mut gpu, &mut rng, &mut tally));

    let mirror = MIRROR_MISMATCH.load(Ordering::Relaxed);
    if mirror > 0 {
        eprintln!(
            "note: {mirror} scatter cases differ from the host grouping mirror (informational)"
        );
    }
    let code = finish(
        "qwen4_g942_moe_wide_oracle",
        &args,
        &tally,
        "HIPFIRE_QWEN4_G942_MOE_WIDE_IMAGE",
        json!({"host_mirror_mismatches": mirror}),
    );
    std::process::exit(code);
}
