// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 MoE-stage Tier C cases (router, scatter, unscatter+SiLU, combines, argmax).
//!
//! Census of record: `subq4/review/G942-ROUTE-CENSUS.md` (family `moe`).
//!
//! gfx942 route facts these cases rest on (verified by reading the code, see census):
//! * Today the sealed qt44/qt53 prefill route reaches path 1 on gfx942 because
//!   `MoePrefillResolution::use_path2 = moe_grouped_gemm && arch.has_wmma()`
//!   (`hipfire-dispatch/src/families/moe.rs:1685`) and `has_wmma()` is RDNA3|RDNA4
//!   only (`rdna-compute/src/arch_caps.rs:144`). The scatter / unscatter / grouped
//!   combine kernels below are therefore launched on gfx942 only by a future
//!   path-2 arm (FP8 slice b); they are portable HIP and are validated here so
//!   that arm starts from a proven wave64 base. The router, `moe_down_combine_top10_batched`
//!   and `argmax_f32` are launched on gfx942 today.
//! * Every CPU twin mirrors the kernel's operation order. The one thing a CPU
//!   cannot reproduce bit-for-bit is the device `expf`, so wherever `expf` feeds
//!   an output the case compares at the frozen `F32_ACCUM`, and the BF16-rounded
//!   variants are tied to their F32 variant *on the device* with an exact check
//!   (round(GPU F32 result on BF16-exact inputs) == GPU BF16 result).

use super::common::*;
use super::Q4Case;
use rdna_compute::tensor_ops::{argmax_f32, ArgmaxF32};
use rdna_compute::{DType, Gpu, GpuTensor};

pub const CASES: &[Q4Case] = &[
    Q4Case {
        name: "qwen4_moe_router_softmax_top10_f32",
        timeout_s: 60,
        run: router_case,
    },
    Q4Case {
        name: "qwen4_moe_scatter_fused_top10",
        timeout_s: 90,
        run: scatter_case,
    },
    Q4Case {
        name: "qwen4_moe_scatter_stable_top10",
        timeout_s: 120,
        run: scatter_stable_case,
    },
    Q4Case {
        name: "qwen4_moe_gate_up_unscatter_silu_top10",
        timeout_s: 90,
        run: unscatter_silu_case,
    },
    Q4Case {
        name: "qwen4_moe_gate_up_unscatter_silu_rotate128_top10",
        timeout_s: 90,
        run: unscatter_silu_rotate_case,
    },
    Q4Case {
        name: "qwen4_moe_down_combine_top10_batched",
        timeout_s: 60,
        run: combine_batched_case,
    },
    Q4Case {
        name: "qwen4_moe_down_combine_grouped_top10",
        timeout_s: 90,
        run: combine_grouped_case,
    },
    Q4Case {
        name: "qwen4_moe_down_combine_grouped_top10_bf16in",
        timeout_s: 90,
        run: combine_grouped_bf16in_case,
    },
    Q4Case {
        name: "qwen4_argmax_f32_vocab248320",
        timeout_s: 60,
        run: argmax_case,
    },
    Q4Case {
        name: "qwen4_gemv_bf16_xf32_x4",
        timeout_s: 60,
        run: gemv_x4_case,
    },
    Q4Case {
        name: "qwen4_gemv_bf16_xf32_bf16_scaled_add",
        timeout_s: 60,
        run: gemv_scaled_add_case,
    },
    Q4Case {
        name: "qwen4_silu_mul_f32",
        timeout_s: 60,
        run: silu_mul_case,
    },
    Q4Case {
        name: "qwen4_shared_expert_activation_bf16_f32",
        timeout_s: 60,
        run: shared_activation_case,
    },
    Q4Case {
        name: "qwen4_mq_rotate_x_128_v2",
        timeout_s: 60,
        run: rotate128_case,
    },
    Q4Case {
        name: "qwen4_mq_rotate_x_128_v2_silu_bf16",
        timeout_s: 60,
        run: rotate128_silu_case,
    },
    Q4Case {
        name: "qwen4_mq_rotate_x_gfx942",
        timeout_s: 90,
        run: rotate256_case,
    },
    Q4Case {
        name: "qwen4_fp8b128_moe_unscatter_silu",
        timeout_s: 120,
        run: fp8_unscatter_silu_case,
    },
    Q4Case {
        name: "qwen4_fp8b128_moe_shared_activation",
        timeout_s: 90,
        run: fp8_shared_activation_case,
    },
];

/// `MOE_GROUPED_BLOCK_M` (`hipfire-dispatch/src/families/moe.rs:31`).
const BLOCK_M: usize = 16;
const K: usize = MOE_TOP_K;
const E: usize = MOE_EXPERTS;
const MI: usize = MOE_INTERMEDIATE;

fn bf16r(v: f32) -> f32 {
    bf16_round(v)
}

fn ints(gpu: &mut Gpu, data: &[i32]) -> Result<GpuTensor, String> {
    up_i32(gpu, data)
}
fn int_sentinel(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    up_i32(gpu, &vec![-2i32; n])
}

// ── Routing-table generation / CPU scatter reference ──────────────────────────
const HOT: [i32; 12] = [7, 100, 101, 255, 256, 257, 300, 333, 400, 500, 510, 511];

#[derive(Clone, Copy, PartialEq)]
enum Spread {
    Uniform,
    Skewed,
    Identical,
    Few,
}

/// `rows` tokens × ten distinct experts each (the router never repeats one).
fn make_topk(rng: &mut Rng, rows: usize, spread: Spread) -> Vec<i32> {
    let mut out = Vec::with_capacity(rows * K);
    for _ in 0..rows {
        let mut chosen: Vec<i32> = Vec::with_capacity(K);
        if spread == Spread::Identical {
            chosen.extend([3, 17, 64, 65, 200, 255, 256, 300, 480, 511]);
        } else {
            while chosen.len() < K {
                let c = if spread == Spread::Few {
                    // Every token routes inside 12 experts: runs of up to `rows` slots.
                    HOT[rng.below(HOT.len())]
                } else if spread == Spread::Skewed && rng.below(4) != 0 {
                    HOT[rng.below(HOT.len())]
                } else {
                    rng.below(E) as i32
                };
                if !chosen.contains(&c) {
                    chosen.push(c);
                }
            }
        }
        out.extend(chosen);
    }
    out
}

/// `checked_grouped_m_total_bound` (`hipfire-dispatch/src/pipeline/sealed_moe.rs:164`).
fn grouped_rows_bound(slots: usize) -> usize {
    (slots + (BLOCK_M - 1) * slots.min(E)).div_ceil(BLOCK_M) * BLOCK_M
}

struct Layout {
    raw: Vec<usize>,
    counts: Vec<i32>,
    offsets: Vec<i32>,
    /// Ascending flat slot inside every expert run (the kernel's order inside a
    /// run is an LDS-atomic arrival order and is NOT deterministic).
    sorted: Vec<i32>,
    tiles: Vec<i32>,
    inverse: Vec<i32>,
    m_total_max: usize,
}

/// Transcription of `kernels/src/qwen4_moe_scatter_fused_k8.hip:66-153` with the
/// intra-run order fixed to ascending flat slot.
fn scatter_ref(topk: &[i32]) -> Layout {
    let slots = topk.len();
    let m_total_max = grouped_rows_bound(slots);
    let mut raw = vec![0usize; E];
    for &ex in topk {
        raw[ex as usize] += 1;
    }
    let mut counts = vec![0i32; E];
    let mut offsets = vec![0i32; E + 1];
    let mut acc = 0usize;
    for ex in 0..E {
        let padded = raw[ex].div_ceil(BLOCK_M) * BLOCK_M;
        counts[ex] = padded as i32;
        offsets[ex] = acc as i32;
        acc += padded;
    }
    offsets[E] = acc as i32;
    let mut sorted = vec![-1i32; m_total_max];
    let mut inverse = vec![-2i32; slots];
    let mut cursor = vec![0usize; E];
    for (i, &ex) in topk.iter().enumerate() {
        let ex = ex as usize;
        let pos = offsets[ex] as usize + cursor[ex];
        cursor[ex] += 1;
        sorted[pos] = i as i32;
        inverse[i] = pos as i32;
    }
    let mut tiles = vec![-1i32; m_total_max / BLOCK_M];
    for ex in 0..E {
        for t in (offsets[ex] as usize / BLOCK_M)..(offsets[ex + 1] as usize / BLOCK_M) {
            tiles[t] = ex as i32;
        }
    }
    Layout {
        raw,
        counts,
        offsets,
        sorted,
        tiles,
        inverse,
        m_total_max,
    }
}

// ── Router ────────────────────────────────────────────────────────────────────
/// Transcription of `kernels/src/moe_router_softmax_top10_f32.hip:77-176`
/// (and `hipfire-arch-qwen4/src/ops.rs:655 route_top10` for the selection rule:
/// value descending, lowest index first on ties): full F32 softmax in expert
/// order, selected-probability renormalisation, UNROUNDED weights.
fn router_ref(row: &[f32], normalize: bool, round: bool) -> ([i32; K], [f32; K]) {
    let x: Vec<f32> = if round {
        row.iter().map(|&v| bf16r(v)).collect()
    } else {
        row.to_vec()
    };
    let mut order: Vec<usize> = (0..E).collect();
    order.sort_by(|&a, &b| x[b].partial_cmp(&x[a]).unwrap().then(a.cmp(&b)));
    let max_logit = x[order[0]];
    let mut denom = 0.0f32;
    for &v in &x {
        denom += (v - max_logit).exp();
    }
    let inv_denom = 1.0f32 / denom;
    let mut ids = [0i32; K];
    let mut p = [0.0f32; K];
    for r in 0..K {
        ids[r] = order[r] as i32;
        p[r] = (x[order[r]] - max_logit).exp() * inv_denom;
    }
    if normalize {
        let mut sum = 0.0f32;
        for r in 0..K {
            sum += p[r];
        }
        let inv = 1.0f32 / sum;
        for r in 0..K {
            p[r] *= inv;
        }
    }
    (ids, p)
}

/// Rows engineered for the stable-order contract: ties across all 32-lane
/// halves of a 256-thread block, boundary ties at rank 10, all-equal rows,
/// coarse-grid rows (many ties), plain random rows.
fn router_logits(rng: &mut Rng, rows: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * E);
    for r in 0..rows {
        let mut row = rng.vec(E, -4.0, 4.0);
        match (r + rows) % 5 {
            0 => row.iter_mut().for_each(|v| *v = 0.5),
            1 => {
                row = rng.vec(E, -2.0, 2.0);
                for &i in &[3usize, 64, 130, 255, 256, 300, 511] {
                    row[i] = 3.5;
                }
                for &i in &[9usize, 128, 200, 400] {
                    row[i] = 3.0;
                }
            }
            2 => row
                .iter_mut()
                .for_each(|v| *v = (rng.below(4) as f32) - 1.0),
            _ => {}
        }
        out.extend(row);
    }
    out
}

fn run_router(
    gpu: &mut Gpu,
    logits: &[f32],
    rows: usize,
    normalize: bool,
    round: bool,
) -> Result<(Vec<i32>, Vec<f32>), String> {
    let t_logits = up_f32(gpu, logits)?;
    let t_ids = int_sentinel(gpu, rows * K)?;
    let t_w = sentinel_f32(gpu, rows * K)?;
    // gfx942: `qwen4_router_fast_enabled` is false (gfx1151/gfx1201 only), so the
    // incumbent `moe_router_softmax_top10_f32` runs for every row count.
    gpu.moe_router_softmax_top10_f32(&t_logits, &t_ids, &t_w, rows, normalize, round, 0, 0)
        .map_err(e)?;
    sync(gpu)?;
    let ids = down_i32(gpu, &t_ids)?;
    let w = down_f32(gpu, &t_w)?;
    free_all(gpu, vec![t_logits, t_ids, t_w])?;
    Ok((ids, w))
}

fn router_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_0001);
    let mut results = Vec::new();
    for &rows in &ROW_COUNTS {
        let logits = router_logits(&mut rng, rows);
        let rounded: Vec<f32> = logits.iter().map(|&v| bf16r(v)).collect();
        for normalize in [true, false] {
            let (ids, w) = run_router(gpu, &logits, rows, normalize, false)?;
            let (mut want_ids, mut want_w) = (Vec::new(), Vec::new());
            for r in 0..rows {
                let (i, p) = router_ref(&logits[r * E..(r + 1) * E], normalize, false);
                want_ids.extend(i);
                want_w.extend(p);
            }
            let tag = format!("rows={rows} norm={normalize} round=0");
            results.push(exact_ints(&format!("{tag} ids"), &ids, &want_ids));
            results.push(close_f32(&format!("{tag} w"), &w, &want_w, F32_ACCUM));
        }
        // BF16 recipe (what the Qwen4 route launches): logits are read through a
        // BF16 round trip and the weights stored through one.
        let (ids_r, w_r) = run_router(gpu, &rounded, rows, true, false)?;
        let (ids1, w1) = run_router(gpu, &logits, rows, true, true)?;
        let mut want_ids = Vec::new();
        for r in 0..rows {
            want_ids.extend(router_ref(&logits[r * E..(r + 1) * E], true, true).0);
        }
        let tag = format!("rows={rows} norm=true round=1");
        results.push(exact_ints(&format!("{tag} ids"), &ids1, &want_ids));
        results.push(exact_ints(
            &format!("{tag} ids vs rounded-input run"),
            &ids1,
            &ids_r,
        ));
        let want_w1: Vec<f32> = w_r.iter().map(|&v| bf16r(v)).collect();
        results.push(exact_f32(
            &format!("{tag} w == bf16(round=0 w of rounded logits)"),
            &w1,
            &want_w1,
        ));
    }
    all(results)
}

// ── Scatter ───────────────────────────────────────────────────────────────────
fn check_scatter(gpu: &mut Gpu, label: &str, topk: &[i32]) -> Result<String, String> {
    let slots = topk.len();
    let lay = scatter_ref(topk);
    let t_topk = ints(gpu, topk)?;
    let t_counts = int_sentinel(gpu, E)?;
    let t_offsets = int_sentinel(gpu, E + 1)?;
    let t_sorted = int_sentinel(gpu, lay.m_total_max)?;
    let t_tiles = int_sentinel(gpu, lay.m_total_max / BLOCK_M)?;
    let t_inv = int_sentinel(gpu, slots)?;
    // gfx942: the `__gfx1151__` block-scan arm is compiled out; the serial
    // thread-0 offset walk runs. Histogram and bucket fill use LDS atomicAdd.
    gpu.moe_scatter_fused_top10(
        &t_topk,
        &t_counts,
        &t_offsets,
        &t_sorted,
        &t_tiles,
        &t_inv,
        slots,
        E,
        lay.m_total_max,
        BLOCK_M,
    )
    .map_err(e)?;
    sync(gpu)?;
    let counts = down_i32(gpu, &t_counts)?;
    let offsets = down_i32(gpu, &t_offsets)?;
    let sorted = down_i32(gpu, &t_sorted)?;
    let tiles = down_i32(gpu, &t_tiles)?;
    let inverse = down_i32(gpu, &t_inv)?;
    free_all(
        gpu,
        vec![t_topk, t_counts, t_offsets, t_sorted, t_tiles, t_inv],
    )?;

    // Intra-run order is atomic-arrival order: canonicalise each expert's live
    // run (ascending) before comparing the permutation; report how many runs the
    // device left unsorted.
    let mut canon = sorted.clone();
    let mut unsorted_runs = 0usize;
    for ex in 0..E {
        let (lo, n) = (lay.offsets[ex] as usize, lay.raw[ex]);
        if lo + n > canon.len() {
            return Err(format!(
                "{label}: reference run of expert {ex} outside sorted buffer"
            ));
        }
        if !canon[lo..lo + n].windows(2).all(|w| w[0] <= w[1]) {
            unsorted_runs += 1;
        }
        canon[lo..lo + n].sort_unstable();
    }
    // Inverse permutation consistency: sorted[inverse[i]] == i and the position
    // lies in the run of the slot's own expert.
    let mut inv_bad = None;
    for (i, &pos) in inverse.iter().enumerate() {
        let ex = topk[i] as usize;
        let (lo, n) = (lay.offsets[ex], lay.raw[ex] as i32);
        let ok = pos >= lo
            && pos < lo + n
            && (pos as usize) < sorted.len()
            && sorted[pos as usize] == i as i32;
        if !ok {
            inv_bad = Some((i, pos));
            break;
        }
    }
    let inv_res = match inv_bad {
        None => Ok(format!("{label} inverse_perm consistent n={slots}")),
        Some((i, pos)) => Err(format!(
            "{label}: inverse_perm[{i}]={pos} is not a position of slot {i} inside expert {}'s run",
            topk[i]
        )),
    };
    all(vec![
        exact_ints(&format!("{label} counts"), &counts, &lay.counts),
        exact_ints(&format!("{label} offsets"), &offsets, &lay.offsets),
        exact_ints(&format!("{label} tile_ids"), &tiles, &lay.tiles),
        exact_ints(
            &format!("{label} sorted (per-run canonical, pads -1)"),
            &canon,
            &lay.sorted,
        ),
        inv_res,
        Ok(format!(
            "{label} runs not ascending on device: {unsorted_runs} (informational)"
        )),
    ])
}

fn scatter_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_0002);
    let mut results = Vec::new();
    // 1536 = the default prefill chunk of non-gfx1151/1201 archs (gpu_forward.rs:83-89).
    for &rows in &[1usize, 7, 33, 512, 1536] {
        let spreads: &[Spread] = if rows >= 33 {
            &[Spread::Uniform, Spread::Skewed, Spread::Identical]
        } else {
            &[Spread::Uniform, Spread::Identical]
        };
        for &spread in spreads {
            let topk = make_topk(&mut rng, rows, spread);
            let label = format!(
                "rows={rows} {}",
                match spread {
                    Spread::Uniform => "uniform",
                    Spread::Skewed => "skewed",
                    Spread::Identical => "identical",
                    Spread::Few => "few-experts",
                }
            );
            results.push(check_scatter(gpu, &label, &topk));
        }
    }
    all(results)
}

/// `Gpu::moe_scatter_stable_top10` (`gemm.rs`, three launches `qwen4_moe_group_ranks` /
/// `qwen4_moe_group_prefix` / `qwen4_moe_group_scatter`, `kernels/src/qwen4_moe_scatter_stable_top10.hip`):
/// slots ordered by (expert, flat_slot), pad16 runs, no atomics. ALL six outputs are
/// deterministic, so every one is compared exactly against the CPU stable counting sort
/// (`scatter_ref`: ascending flat slot inside each expert run). Wave64 notes: block 256 /
/// block = experts (512) with LDS `s_pad[1024]` Hillis-Steele scan and `__syncthreads`, no
/// lane intrinsics, no atomics.
fn check_stable(gpu: &mut Gpu, label: &str, topk: &[i32]) -> Result<String, String> {
    let slots = topk.len();
    let lay = scatter_ref(topk);
    let t_topk = ints(gpu, topk)?;
    let t_counts = int_sentinel(gpu, E)?;
    let t_offsets = int_sentinel(gpu, E + 1)?;
    let t_sorted = int_sentinel(gpu, lay.m_total_max)?;
    let t_tiles = int_sentinel(gpu, lay.m_total_max / BLOCK_M)?;
    let t_inv = int_sentinel(gpu, slots)?;
    gpu.moe_scatter_stable_top10(
        &t_topk,
        &t_counts,
        &t_offsets,
        &t_sorted,
        &t_tiles,
        &t_inv,
        slots,
        E,
        lay.m_total_max,
    )
    .map_err(e)?;
    sync(gpu)?;
    let counts = down_i32(gpu, &t_counts)?;
    let offsets = down_i32(gpu, &t_offsets)?;
    let sorted = down_i32(gpu, &t_sorted)?;
    let tiles = down_i32(gpu, &t_tiles)?;
    let inverse = down_i32(gpu, &t_inv)?;
    free_all(
        gpu,
        vec![t_topk, t_counts, t_offsets, t_sorted, t_tiles, t_inv],
    )?;
    all(vec![
        exact_ints(&format!("{label} counts"), &counts, &lay.counts),
        exact_ints(&format!("{label} offsets"), &offsets, &lay.offsets),
        exact_ints(
            &format!("{label} sorted ((expert, flat_slot) order, pads/tail -1)"),
            &sorted,
            &lay.sorted,
        ),
        exact_ints(&format!("{label} tile_ids"), &tiles, &lay.tiles),
        exact_ints(&format!("{label} inverse_perm"), &inverse, &lay.inverse),
    ])
}

fn scatter_stable_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_0009);
    let mut results = Vec::new();
    for &rows in &[1usize, 2, 7, 8, 16, 17, 128, 1536] {
        for (spread, name) in [
            (Spread::Uniform, "uniform"),
            (Spread::Skewed, "skewed"),
            (Spread::Few, "few-experts"),
            (Spread::Identical, "identical"),
        ] {
            let topk = make_topk(&mut rng, rows, spread);
            results.push(check_stable(gpu, &format!("rows={rows} {name}"), &topk));
        }
    }
    all(results)
}

// ── Unscatter + SiLU (+ rotate128) ───────────────────────────────────────────
/// `kernels/src/moe_gate_up_unscatter_silu_top10.hip:54-55` (expression of
/// `silu_mul.hip`; `hipfire-arch-qwen4/src/ops.rs:86 silu` is the same function
/// up to the v/(1+e^-v) vs v*sigmoid(v) rounding, so the kernel's form is used).
fn silu_mul_ref(v: f32, u: f32, rt: bool) -> f32 {
    let (v, u) = if rt { (bf16r(v), bf16r(u)) } else { (v, u) };
    let out = (v / (1.0f32 + (-v).exp())) * u;
    if rt {
        bf16r(out)
    } else {
        out
    }
}

fn unscatter_ref(y: &[f32], sorted: &[i32], slots: usize, rt: bool) -> Vec<f32> {
    let mut out = vec![f32::NAN; slots * MI];
    for (p, &flat) in sorted.iter().enumerate() {
        if flat < 0 {
            continue;
        }
        for col in 0..MI {
            out[flat as usize * MI + col] =
                silu_mul_ref(y[p * 2 * MI + col], y[p * 2 * MI + MI + col], rt);
        }
    }
    out
}

fn run_unscatter(
    gpu: &mut Gpu,
    t_y: &GpuTensor,
    t_sorted: &GpuTensor,
    grouped_rows: usize,
    slots: usize,
    rt: bool,
) -> Result<Vec<f32>, String> {
    let t_act = sentinel_f32(gpu, slots * MI)?;
    gpu.moe_gate_up_unscatter_silu_top10(t_y, t_sorted, &t_act, MI, grouped_rows, rt)
        .map_err(e)?;
    sync(gpu)?;
    let out = down_f32(gpu, &t_act)?;
    free_all(gpu, vec![t_act])?;
    Ok(out)
}

fn unscatter_silu_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_0003);
    let mut results = Vec::new();
    for &rows in &[1usize, 7, 33, 512] {
        let topk = make_topk(&mut rng, rows, Spread::Skewed);
        let lay = scatter_ref(&topk);
        let slots = rows * K;
        let y = rng.vec(lay.m_total_max * 2 * MI, -3.0, 3.0);
        let y_rt: Vec<f32> = y.iter().map(|&v| bf16r(v)).collect();
        let t_sorted = ints(gpu, &lay.sorted)?;
        let t_y = up_f32(gpu, &y)?;
        let t_y_rt = up_f32(gpu, &y_rt)?;
        // gfx942 takes the F32 `moe_gate_up_unscatter_silu_top10` (gate/up grouped rows are
        // F32 on the non-WMMA arm); `_bf16in` is only fed by the gfx11 WMMA BF16-out GEMMs.
        let a = run_unscatter(gpu, &t_y, &t_sorted, lay.m_total_max, slots, false)?;
        let b = run_unscatter(gpu, &t_y_rt, &t_sorted, lay.m_total_max, slots, false)?;
        let c = run_unscatter(gpu, &t_y, &t_sorted, lay.m_total_max, slots, true)?;
        free_all(gpu, vec![t_sorted, t_y, t_y_rt])?;
        let tag = format!("rows={rows}");
        results.push(close_f32(
            &format!("{tag} rt=0"),
            &a,
            &unscatter_ref(&y, &lay.sorted, slots, false),
            F32_ACCUM,
        ));
        results.push(close_f32(
            &format!("{tag} rt=0 on bf16-exact input"),
            &b,
            &unscatter_ref(&y_rt, &lay.sorted, slots, false),
            F32_ACCUM,
        ));
        // rt=1 on the raw input == BF16 round of the rt=0 result on the BF16-exact input
        // (same device expf, so no cross-implementation exp difference).
        let want_c: Vec<f32> = b.iter().map(|&v| bf16r(v)).collect();
        results.push(exact_f32(
            &format!("{tag} rt=1 == bf16(rt=0 on bf16-exact input)"),
            &c,
            &want_c,
        ));
    }
    all(results)
}

/// `kernels/src/moe_gate_up_unscatter_silu_top10.hip:129-157` for one 128-wide
/// block: in-thread 4-point butterfly, five `ds_swizzle` XOR stages
/// (xor 1,2,4,8,16 within the 32 active lanes; `tid & stride` lanes subtract),
/// then `* scale * signs2`.
fn rotate128_ref(a: &[f32], s1: &[f32], s2: &[f32]) -> Vec<f32> {
    let mut v = [[0.0f32; 4]; 32];
    for tid in 0..32 {
        let d0 = tid * 4;
        let (mut v0, mut v1, mut v2, mut v3) = (
            a[d0] * s1[d0],
            a[d0 + 1] * s1[d0 + 1],
            a[d0 + 2] * s1[d0 + 2],
            a[d0 + 3] * s1[d0 + 3],
        );
        let t = v0;
        v0 = v0 + v1;
        v1 = t - v1;
        let t = v2;
        v2 = v2 + v3;
        v3 = t - v3;
        let t = v0;
        v0 = v0 + v2;
        v2 = t - v2;
        let t = v1;
        v1 = v1 + v3;
        v3 = t - v3;
        v[tid] = [v0, v1, v2, v3];
    }
    for stride in [1usize, 2, 4, 8, 16] {
        let old = v;
        for tid in 0..32 {
            for el in 0..4 {
                let partner = old[tid ^ stride][el];
                let mine = old[tid][el];
                v[tid][el] = if tid & stride != 0 {
                    partner - mine
                } else {
                    mine + partner
                };
            }
        }
    }
    let scale = 0.0883883476f32;
    let mut out = vec![0.0f32; 128];
    for tid in 0..32 {
        for el in 0..4 {
            let d = tid * 4 + el;
            out[d] = v[tid][el] * scale * s2[d];
        }
    }
    out
}

fn unscatter_silu_rotate_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_0004);
    gpu.ensure_mq_signs_128().map_err(e)?;
    let (s1, s2) = {
        let g: &Gpu = &*gpu;
        let t1 = g
            .scratch
            .mq_signs1_128
            .as_ref()
            .ok_or("mq_signs1_128 missing")?;
        let t2 = g
            .scratch
            .mq_signs2_128
            .as_ref()
            .ok_or("mq_signs2_128 missing")?;
        (down_f32(g, t1)?, down_f32(g, t2)?)
    };
    let mut results = Vec::new();
    for &rows in &[1usize, 7, 33] {
        let topk = make_topk(&mut rng, rows, Spread::Uniform);
        let lay = scatter_ref(&topk);
        let slots = rows * K;
        let y = rng.vec(lay.m_total_max * 2 * MI, -3.0, 3.0);
        let t_sorted = ints(gpu, &lay.sorted)?;
        let t_y = up_f32(gpu, &y)?;
        for rt in [false, true] {
            // This is the launch the gfx942 non-WMMA path-2 arm would make
            // (`unscatter_rotates`: mi % 128 == 0), block 32 on a wave64 device.
            let act = run_unscatter(gpu, &t_y, &t_sorted, lay.m_total_max, slots, rt)?;
            let t_rot = sentinel_f32(gpu, slots * MI)?;
            gpu.moe_gate_up_unscatter_silu_rotate128_top10(
                &t_y,
                &t_sorted,
                &t_rot,
                MI,
                lay.m_total_max,
                rt,
                None,
            )
            .map_err(e)?;
            sync(gpu)?;
            let got = down_f32(gpu, &t_rot)?;
            free_all(gpu, vec![t_rot])?;
            // The kernel forms the activation with the unscatter's own expression, so the
            // device activation is the twin's input (no cross-implementation expf).
            let mut want = vec![0.0f32; slots * MI];
            for flat in 0..slots {
                for c in 0..MI / 128 {
                    let base = flat * MI + c * 128;
                    let r = rotate128_ref(&act[base..base + 128], &s1, &s2);
                    want[base..base + 128].copy_from_slice(&r);
                }
            }
            results.push(close_f32(
                &format!("rows={rows} rt={}", rt as u8),
                &got,
                &want,
                F32_ACCUM,
            ));
        }
        free_all(gpu, vec![t_sorted, t_y])?;
    }
    all(results)
}

// ── Combines ──────────────────────────────────────────────────────────────────
/// `kernels/src/moe_down_combine_top10_batched.hip:44-70` /
/// `moe_down_combine_grouped_top10.hip:44-75`: ranks visited in ascending expert
/// id, weight / output / product / accumulator each rounded to BF16.
fn combine_ref(residual: f32, ids: &[i32], weights: &[f32], outputs: &[f32; K]) -> f32 {
    let mut order: Vec<usize> = (0..K).collect();
    order.sort_by_key(|&r| ids[r]);
    let mut acc = bf16r(residual);
    for r in order {
        let w = bf16r(weights[r]);
        let o = bf16r(outputs[r]);
        let weighted = bf16r(o * w);
        acc = bf16r(acc + weighted);
    }
    acc
}

fn route_weights(rng: &mut Rng, rows: usize) -> Vec<f32> {
    rng.vec(rows * K, 0.0, 0.5)
}

fn combine_batched_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_0005);
    let mut results = Vec::new();
    for &rows in &[1usize, 7, 33, 128] {
        let topk = make_topk(&mut rng, rows, Spread::Uniform);
        let w = route_weights(&mut rng, rows);
        let expanded = rng.vec(rows * K * HIDDEN, -2.0, 2.0);
        let residual = rng.vec(rows * HIDDEN, -2.0, 2.0);
        let t_exp = up_f32(gpu, &expanded)?;
        let t_ids = ints(gpu, &topk)?;
        let t_w = up_f32(gpu, &w)?;
        let t_res = up_f32(gpu, &residual)?;
        gpu.moe_down_combine_top10_batched(&t_exp, &t_ids, &t_w, &t_res, HIDDEN, rows, E)
            .map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &t_res)?;
        free_all(gpu, vec![t_exp, t_ids, t_w, t_res])?;
        let mut want = vec![0.0f32; rows * HIDDEN];
        for t in 0..rows {
            for col in 0..HIDDEN {
                let mut outs = [0.0f32; K];
                for r in 0..K {
                    outs[r] = expanded[(t * K + r) * HIDDEN + col];
                }
                want[t * HIDDEN + col] = combine_ref(
                    residual[t * HIDDEN + col],
                    &topk[t * K..(t + 1) * K],
                    &w[t * K..(t + 1) * K],
                    &outs,
                );
            }
        }
        results.push(exact_f32(&format!("rows={rows}"), &got, &want));
    }
    all(results)
}

fn combine_grouped_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_0006);
    let mut results = Vec::new();
    for &rows in &ROW_COUNTS {
        let topk = make_topk(&mut rng, rows, Spread::Skewed);
        let lay = scatter_ref(&topk);
        let w = route_weights(&mut rng, rows);
        let grouped = rng.vec(lay.m_total_max * HIDDEN, -2.0, 2.0);
        let residual = rng.vec(rows * HIDDEN, -2.0, 2.0);
        let t_grouped = up_f32(gpu, &grouped)?;
        let t_inv = ints(gpu, &lay.inverse)?;
        let t_ids = ints(gpu, &topk)?;
        let t_w = up_f32(gpu, &w)?;
        let t_res = up_f32(gpu, &residual)?;
        gpu.moe_down_combine_grouped_top10(
            &t_grouped,
            &t_inv,
            &t_ids,
            &t_w,
            &t_res,
            HIDDEN,
            lay.m_total_max,
            rows,
            E,
        )
        .map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &t_res)?;
        free_all(gpu, vec![t_grouped, t_inv, t_ids, t_w, t_res])?;
        let mut want = vec![0.0f32; rows * HIDDEN];
        for t in 0..rows {
            for col in 0..HIDDEN {
                let mut outs = [0.0f32; K];
                for r in 0..K {
                    let row = lay.inverse[t * K + r] as usize;
                    outs[r] = grouped[row * HIDDEN + col];
                }
                want[t * HIDDEN + col] = combine_ref(
                    residual[t * HIDDEN + col],
                    &topk[t * K..(t + 1) * K],
                    &w[t * K..(t + 1) * K],
                    &outs,
                );
            }
        }
        results.push(exact_f32(&format!("rows={rows}"), &got, &want));
    }
    all(results)
}

fn combine_grouped_bf16in_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_0007);
    let mut results = Vec::new();
    for &rows in &ROW_COUNTS {
        let topk = make_topk(&mut rng, rows, Spread::Skewed);
        let lay = scatter_ref(&topk);
        let w = route_weights(&mut rng, rows);
        let grouped = rng.vec_bf16_exact(lay.m_total_max * HIDDEN, -2.0, 2.0);
        let residual = rng.vec(rows * HIDDEN, -2.0, 2.0);
        for initial_zero in [false, true] {
            let t_grouped = up_bf16(gpu, &grouped)?;
            let t_inv = ints(gpu, &lay.inverse)?;
            let t_ids = ints(gpu, &topk)?;
            let t_w = up_f32(gpu, &w)?;
            let t_res = if initial_zero {
                sentinel_f32(gpu, rows * HIDDEN)?
            } else {
                up_f32(gpu, &residual)?
            };
            let t_order = zeros_f32(gpu, rows * K * 2)?;
            // The WMMA/sym arms feed this kernel on gfx11; the kernel itself is portable
            // HIP (no WMMA, no lane ops), so it is exercised on gfx942 directly.
            gpu.moe_down_combine_grouped_top10_bf16in(
                &t_grouped,
                &t_inv,
                &t_ids,
                &t_w,
                &t_res,
                &t_order,
                HIDDEN,
                lay.m_total_max,
                rows,
                initial_zero,
                E,
            )
            .map_err(e)?;
            sync(gpu)?;
            let got = down_f32(gpu, &t_res)?;
            free_all(gpu, vec![t_grouped, t_inv, t_ids, t_w, t_res, t_order])?;
            let mut want = vec![0.0f32; rows * HIDDEN];
            for t in 0..rows {
                for col in 0..HIDDEN {
                    let mut outs = [0.0f32; K];
                    for r in 0..K {
                        let row = lay.inverse[t * K + r] as usize;
                        outs[r] = grouped[row * HIDDEN + col];
                    }
                    let start = if initial_zero {
                        0.0
                    } else {
                        residual[t * HIDDEN + col]
                    };
                    want[t * HIDDEN + col] = combine_ref(
                        start,
                        &topk[t * K..(t + 1) * K],
                        &w[t * K..(t + 1) * K],
                        &outs,
                    );
                }
            }
            results.push(exact_f32(
                &format!("rows={rows} zinit={}", initial_zero as u8),
                &got,
                &want,
            ));
        }
    }
    all(results)
}

// ── argmax_f32 (the Qwen4 greedy head: tensor_ops.rs:5374 → tensor_ops.hip:5132) ──
/// `llama::argmax` semantics of `kernels/src/tensor_ops.hip:5126-5168`: first
/// index of the largest FINITE value; no finite value → 0.
fn argmax_ref(row: &[f32]) -> i32 {
    let (mut best, mut id) = (f32::NEG_INFINITY, 0usize);
    for (i, &v) in row.iter().enumerate() {
        if v.is_finite() && v > best {
            best = v;
            id = i;
        }
    }
    id as i32
}

fn argmax_rows(rng: &mut Rng, rows: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * VOCAB);
    for r in 0..rows {
        let mut row = rng.vec(VOCAB, -10.0, 10.0);
        match (r + rows) % 6 {
            // Ties on the maximum, across threads and across a thread's strides
            // (block 1024: ids 1 and 1025 and 1024 live on threads 1, 1 and 0).
            1 => {
                for &i in &[VOCAB - 1, 123_457, 1_025, 1_024, 1] {
                    row[i] = 11.0;
                }
            }
            // Non-finite values never win; the finite maximum does.
            2 => {
                row[7] = f32::INFINITY;
                row[8] = f32::NAN;
                row[200_000] = f32::INFINITY;
                row[1_023] = 9.99;
                row[2_047] = 9.99;
                row[4] = f32::NEG_INFINITY;
            }
            // No finite value → id 0.
            3 => {
                row.iter_mut().for_each(|v| *v = f32::NEG_INFINITY);
                row[500] = f32::INFINITY;
                row[600] = f32::NAN;
            }
            4 => row.iter_mut().for_each(|v| *v = f32::NAN),
            // All equal → lowest index.
            5 => row.iter_mut().for_each(|v| *v = 1.0),
            _ => {}
        }
        out.extend(row);
    }
    out
}

fn argmax_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_0008);
    let mut results = Vec::new();
    for &rows in &ROW_COUNTS {
        let logits = argmax_rows(&mut rng, rows);
        let t_logits = up_f32(gpu, &logits)?;
        // `indices` is DType::Raw with numel >= rows * 4 (tensor_ops.rs:5387).
        let t_idx = gpu.zeros(&[rows * 4], DType::Raw).map_err(e)?;
        argmax_f32(
            gpu,
            &ArgmaxF32 {
                logits: &t_logits,
                indices: &t_idx,
                rows,
                vocab: VOCAB,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let got = down_i32(gpu, &t_idx)?;
        free_all(gpu, vec![t_logits, t_idx])?;
        let want: Vec<i32> = (0..rows)
            .map(|r| argmax_ref(&logits[r * VOCAB..(r + 1) * VOCAB]))
            .collect();
        results.push(exact_ints(
            &format!("rows={rows}"),
            &got[..rows.min(got.len())],
            &want,
        ));
    }
    all(results)
}

// ── Decode-stage BF16 GEMVs, activations and rotations (rocprofv3 census of the MQ4 route) ──
/// Mirror of `gemv_bf16_xf32_row` (`kernels/src/qwen4_gemv_bf16_xf32.hip:44-95`): 32 lanes, lane
/// `t` owns elements `[t*8, t*8+8)` of every 256-element step, four chains `a[e % 4]` in element
/// order, `(a0+a1)+(a2+a3)`, then the `__shfl_down` tree (offsets 16..1; only lane 0 is consumed,
/// so the width-64 default on wave64 is equivalent to width 32). Compared at `F32_ACCUM` because
/// hipcc may contract the multiply-adds.
fn gemv_row_ref(w: &[f32], x: &[f32]) -> f32 {
    let k = w.len();
    let mut s = [0.0f32; 32];
    for lane in 0..32 {
        let mut a = [0.0f32; 4];
        if k % 8 == 0 {
            let mut i = lane * 8;
            while i + 8 <= k {
                for el in 0..8 {
                    a[el % 4] += w[i + el] * x[i + el];
                }
                i += 256;
            }
        }
        let mut j = (if k % 8 == 0 { k & !7 } else { 0 }) + lane;
        while j < k {
            a[0] += w[j] * x[j];
            j += 32;
        }
        s[lane] = (a[0] + a[1]) + (a[2] + a[3]);
    }
    for off in [16usize, 8, 4, 2, 1] {
        let old = s;
        for l in 0..32 {
            s[l] = old[l] + if l + off < 32 { old[l + off] } else { old[l] };
        }
    }
    s[0]
}

/// `kernels/src/qwen4_gemv_bf16_xf32.hip` `gemv_bf16_xf32_x4`: router + shared selector + shared
/// gate + shared up in one launch (`hipfire-dispatch/src/pipeline/mod.rs` decode gate-side stage,
/// the `qt44_qt53_grouped` BF16 arm). Block 32 on wave64.
fn gemv_x4_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_000A);
    let k = HIDDEN;
    let mut results = Vec::new();
    // (router 512, selector 1, shared gate 640, shared up 640); the Q8 shared recipe passes M = 0
    // for gate/up (`rows(m)` in the call site).
    for (cfg, ms) in [
        ("bf16-shared", [512usize, 1, 640, 640]),
        ("q8-shared", [512, 1, 0, 0]),
    ] {
        let x = rng.vec(k, -1.0, 1.0);
        let ws: Vec<Vec<f32>> = ms
            .iter()
            .map(|&m| rng.vec_bf16_exact(m.max(1) * k, -0.05, 0.05))
            .collect();
        let t_x = up_f32(gpu, &x)?;
        let t_w: Vec<GpuTensor> = ws
            .iter()
            .map(|w| up_bf16(gpu, w))
            .collect::<Result<_, _>>()?;
        let t_y: Vec<GpuTensor> = ms
            .iter()
            .map(|&m| sentinel_f32(gpu, m.max(1)))
            .collect::<Result<_, _>>()?;
        let t_s: Vec<GpuTensor> = ms
            .iter()
            .map(|&m| sentinel_f32(gpu, m.max(1)))
            .collect::<Result<_, _>>()?;
        gpu.gemv_bf16_xf32_x4(
            [
                (&t_w[0], &t_y[0], ms[0]),
                (&t_w[1], &t_y[1], ms[1]),
                (&t_w[2], &t_y[2], ms[2]),
                (&t_w[3], &t_y[3], ms[3]),
            ],
            &t_x,
            k,
        )
        .map_err(e)?;
        for i in 0..4 {
            if ms[i] > 0 {
                gpu.gemv_bf16_xf32(&t_w[i], &t_x, &t_s[i], ms[i], k)
                    .map_err(e)?;
            }
        }
        sync(gpu)?;
        for i in 0..4 {
            if ms[i] == 0 {
                continue;
            }
            let got = down_f32(gpu, &t_y[i])?;
            let single = down_f32(gpu, &t_s[i])?;
            let want: Vec<f32> = (0..ms[i])
                .map(|r| gemv_row_ref(&ws[i][r * k..(r + 1) * k], &x))
                .collect();
            results.push(exact_f32(
                &format!("{cfg} part{i} m={} == single-matrix launch", ms[i]),
                &got,
                &single,
            ));
            results.push(close_f32(
                &format!("{cfg} part{i} m={} vs CPU mirror", ms[i]),
                &got,
                &want,
                F32_ACCUM,
            ));
        }
        let mut all_t = vec![t_x];
        all_t.extend(t_w);
        all_t.extend(t_y);
        all_t.extend(t_s);
        free_all(gpu, all_t)?;
    }
    all(results)
}

/// `gemv_bf16_xf32_bf16_scaled_add` (decode shared-expert down + selector fold,
/// `hipfire-dispatch/src/pipeline/mod.rs` `decode_shared_down_stage`): bitwise `gemv_bf16_xf32`
/// then `bf16_scaled_add_f32`.
fn gemv_scaled_add_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_000B);
    let mut results = Vec::new();
    for &(m, k) in &[(HIDDEN, MOE_INTERMEDIATE), (512usize, HIDDEN)] {
        for trial in 0..3 {
            let w = rng.vec_bf16_exact(m * k, -0.05, 0.05);
            let x = rng.vec(k, -2.0, 2.0);
            let res = rng.vec(m, -2.0, 2.0);
            let scalar = [rng.range(0.0, 1.0)];
            let t_w = up_bf16(gpu, &w)?;
            let t_x = up_f32(gpu, &x)?;
            let t_fused = up_f32(gpu, &res)?;
            let t_pair = up_f32(gpu, &res)?;
            let t_y = sentinel_f32(gpu, m)?;
            let t_sc = up_f32(gpu, &scalar)?;
            gpu.gemv_bf16_xf32_bf16_scaled_add(&t_w, &t_x, &t_fused, &t_sc, m, k)
                .map_err(e)?;
            gpu.gemv_bf16_xf32(&t_w, &t_x, &t_y, m, k).map_err(e)?;
            rdna_compute::tensor_ops::bf16_scaled_add(
                gpu,
                &rdna_compute::tensor_ops::Bf16ScaledAdd {
                    residual: &t_pair,
                    value: &t_y,
                    scalar: &t_sc,
                    elements: m,
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            let fused = down_f32(gpu, &t_fused)?;
            let pair = down_f32(gpu, &t_pair)?;
            let y = down_f32(gpu, &t_y)?;
            // The epilogue (`gemv_bf16_xf32_bf16_scaled_add` tail): every step an exact BF16 round.
            let gate = bf16r(scalar[0]);
            let want: Vec<f32> = (0..m)
                .map(|r| {
                    let product = bf16r(gate * bf16r(y[r]));
                    bf16r(bf16r(res[r]) + product)
                })
                .collect();
            let tag = format!("m={m} k={k} trial={trial}");
            results.push(exact_f32(
                &format!("{tag} fused == gemv + bf16_scaled_add"),
                &fused,
                &pair,
            ));
            results.push(exact_f32(
                &format!("{tag} fused == CPU epilogue of the device GEMV"),
                &fused,
                &want,
            ));
            free_all(gpu, vec![t_w, t_x, t_fused, t_pair, t_y, t_sc])?;
        }
    }
    all(results)
}

fn silu_ref(v: f32, u: f32) -> f32 {
    (v / (1.0f32 + (-v).exp())) * u
}

/// `silu_mul_f32` (`kernels/src/silu_mul.hip:7`): prefill path-1 activation and the non-BF16 decode arm.
fn silu_mul_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_000C);
    let mut results = Vec::new();
    for &rows in &ROW_COUNTS {
        let n = rows * K * MI;
        let g = rng.vec(n, -4.0, 4.0);
        let u = rng.vec(n, -4.0, 4.0);
        let (t_g, t_u, t_o) = (up_f32(gpu, &g)?, up_f32(gpu, &u)?, sentinel_f32(gpu, n)?);
        gpu.silu_mul_f32(&t_g, &t_u, &t_o).map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &t_o)?;
        free_all(gpu, vec![t_g, t_u, t_o])?;
        let want: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| silu_ref(a, b)).collect();
        results.push(close_f32(
            &format!("rows={rows} n={n}"),
            &got,
            &want,
            F32_ACCUM,
        ));
    }
    all(results)
}

/// `shared_expert_activation_bf16_f32` + `silu_mul_bf16_rt_f32` (`kernels/src/qwen4_silu_mul.hip`).
fn shared_activation_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_000D);
    let mut results = Vec::new();
    for &rows in &ROW_COUNTS {
        let n = rows * MI;
        let g = rng.vec(n, -4.0, 4.0);
        let u = rng.vec(n, -4.0, 4.0);
        let sel = rng.vec(rows, -4.0, 4.0);
        let g_rt: Vec<f32> = g.iter().map(|&v| bf16r(v)).collect();
        let u_rt: Vec<f32> = u.iter().map(|&v| bf16r(v)).collect();
        let (t_g, t_u) = (up_f32(gpu, &g)?, up_f32(gpu, &u)?);
        let (t_grt, t_urt) = (up_f32(gpu, &g_rt)?, up_f32(gpu, &u_rt)?);
        let (t_o_rt, t_o_shared, t_o_f32) = (
            sentinel_f32(gpu, n)?,
            sentinel_f32(gpu, n)?,
            sentinel_f32(gpu, n)?,
        );
        let t_sel = up_f32(gpu, &sel)?;
        gpu.silu_mul_bf16_rt_f32(&t_g, &t_u, &t_o_rt).map_err(e)?;
        gpu.shared_expert_activation_bf16_f32(&t_g, &t_u, &t_o_shared, &t_sel, rows)
            .map_err(e)?;
        gpu.silu_mul_f32(&t_grt, &t_urt, &t_o_f32).map_err(e)?;
        sync(gpu)?;
        let o_rt = down_f32(gpu, &t_o_rt)?;
        let o_shared = down_f32(gpu, &t_o_shared)?;
        let o_f32 = down_f32(gpu, &t_o_f32)?;
        let sel_got = down_f32(gpu, &t_sel)?;
        free_all(
            gpu,
            vec![t_g, t_u, t_grt, t_urt, t_o_rt, t_o_shared, t_o_f32, t_sel],
        )?;
        let tag = format!("rows={rows}");
        // Same device expf in all three launches: BF16 round of the F32 launch on BF16-exact
        // inputs is the BF16 recipe's result, bit for bit.
        let want_rt: Vec<f32> = o_f32.iter().map(|&v| bf16r(v)).collect();
        results.push(exact_f32(
            &format!("{tag} silu_mul_bf16_rt == bf16(silu_mul_f32 of rounded inputs)"),
            &o_rt,
            &want_rt,
        ));
        results.push(exact_f32(
            &format!("{tag} shared activation out == silu_mul_bf16_rt"),
            &o_shared,
            &o_rt,
        ));
        let want_f32: Vec<f32> = g_rt
            .iter()
            .zip(&u_rt)
            .map(|(&a, &b)| silu_ref(a, b))
            .collect();
        results.push(close_f32(
            &format!("{tag} silu_mul_f32 vs CPU"),
            &o_f32,
            &want_f32,
            F32_ACCUM,
        ));
        let want_sel: Vec<f32> = sel
            .iter()
            .map(|&s| 1.0f32 / (1.0f32 + (-bf16r(s)).exp()))
            .collect();
        results.push(close_f32(
            &format!("{tag} selector sigmoid(bf16(sel))"),
            &sel_got,
            &want_sel,
            F32_ACCUM,
        ));
    }
    all(results)
}

/// Generic mirror of the FWHT kernels' butterfly: `lanes` lanes × four adjacent values
/// (`mq_rotate_x_128_v2.hip:41-76` with 32 lanes / scale 1/sqrt(128); `mq_rotate_x.gfx942.hip:30-79`
/// with 64 lanes / scale 1/16): sign1, in-register stages (strides 1, 2), cross-lane XOR stages
/// (`tid & stride` lanes compute `partner - mine`), `* scale * sign2`. Signs are +-1, so FMA
/// contraction cannot change a bit: the mirror is bit-exact.
fn fwht_ref(a: &[f32], s1: &[f32], s2: &[f32], lanes: usize, scale: f32) -> Vec<f32> {
    let mut v = vec![[0.0f32; 4]; lanes];
    for tid in 0..lanes {
        let d0 = tid * 4;
        let (mut v0, mut v1, mut v2, mut v3) = (
            a[d0] * s1[d0],
            a[d0 + 1] * s1[d0 + 1],
            a[d0 + 2] * s1[d0 + 2],
            a[d0 + 3] * s1[d0 + 3],
        );
        let t = v0;
        v0 = v0 + v1;
        v1 = t - v1;
        let t = v2;
        v2 = v2 + v3;
        v3 = t - v3;
        let t = v0;
        v0 = v0 + v2;
        v2 = t - v2;
        let t = v1;
        v1 = v1 + v3;
        v3 = t - v3;
        v[tid] = [v0, v1, v2, v3];
    }
    let mut stride = 1usize;
    while stride < lanes {
        let old = v.clone();
        for tid in 0..lanes {
            for el in 0..4 {
                let partner = old[tid ^ stride][el];
                let mine = old[tid][el];
                v[tid][el] = if tid & stride != 0 {
                    partner - mine
                } else {
                    mine + partner
                };
            }
        }
        stride <<= 1;
    }
    let mut out = vec![0.0f32; lanes * 4];
    for tid in 0..lanes {
        for el in 0..4 {
            let d = tid * 4 + el;
            out[d] = v[tid][el] * scale * s2[d];
        }
    }
    out
}

/// Row-batched rotation of `rows` rows of width `k` in groups of `group` (zero-padded last
/// group, only `k` values kept: `mq_rotate_x_128_v2`'s tail contract).
fn rotate_rows_ref(
    x: &[f32],
    k: usize,
    rows: usize,
    group: usize,
    s1: &[f32],
    s2: &[f32],
) -> Vec<f32> {
    let (lanes, scale) = if group == 128 {
        (32, 0.0883883476f32)
    } else {
        (64, 0.0625f32)
    };
    let mut out = vec![0.0f32; rows * k];
    for r in 0..rows {
        for g in 0..k.div_ceil(group) {
            let mut block = vec![0.0f32; group];
            for i in 0..group {
                let idx = g * group + i;
                if idx < k {
                    block[i] = x[r * k + idx];
                }
            }
            let rot = fwht_ref(&block, s1, s2, lanes, scale);
            for i in 0..group {
                let idx = g * group + i;
                if idx < k {
                    out[r * k + idx] = rot[i];
                }
            }
        }
    }
    out
}

fn mq_signs(gpu: &mut Gpu, g128: bool) -> Result<(Vec<f32>, Vec<f32>), String> {
    if g128 {
        gpu.ensure_mq_signs_128().map_err(e)?;
    } else {
        gpu.ensure_mq_signs().map_err(e)?;
    }
    let g: &Gpu = &*gpu;
    let (t1, t2) = if g128 {
        (
            g.scratch.mq_signs1_128.as_ref(),
            g.scratch.mq_signs2_128.as_ref(),
        )
    } else {
        (g.scratch.mq_signs1.as_ref(), g.scratch.mq_signs2.as_ref())
    };
    let (t1, t2) = (
        t1.ok_or("mq signs1 missing")?,
        t2.ok_or("mq signs2 missing")?,
    );
    Ok((down_f32(g, t1)?, down_f32(g, t2)?))
}

/// `mq_rotate_x_128_v2` (qt53 down activation basis, K = 640, plus a non-multiple-of-128 K that
/// exercises the zero-filled tail). `ds_swizzle` stages, block 32 on wave64.
fn rotate128_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_000E);
    let (s1, s2) = mq_signs(gpu, true)?;
    let mut results = Vec::new();
    for &(k, rows) in &[(MI, 1usize), (MI, 7), (MI, 10), (MI, 33), (200usize, 3)] {
        let x = rng.vec(rows * k, -3.0, 3.0);
        let t_x = up_f32(gpu, &x)?;
        let t_o = sentinel_f32(gpu, rows * k)?;
        gpu.rotate_x_mq_128_v2(&t_x, &t_o, k, rows).map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &t_o)?;
        free_all(gpu, vec![t_x, t_o])?;
        let want = rotate_rows_ref(&x, k, rows, 128, &s1, &s2);
        results.push(exact_f32(&format!("k={k} rows={rows}"), &got, &want));
    }
    all(results)
}

/// `mq_rotate_x_128_v2_silu_bf16` (decode activation + down FWHT128 in one launch, optional shared
/// expert tail in the extra grid row).
fn rotate128_silu_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_000F);
    let (s1, s2) = mq_signs(gpu, true)?;
    let k = MI;
    let mut results = Vec::new();
    for &rows in &[1usize, 7, 10, 33] {
        let n = rows * k;
        let g = rng.vec(n, -4.0, 4.0);
        let u = rng.vec(n, -4.0, 4.0);
        let (t_g, t_u) = (up_f32(gpu, &g)?, up_f32(gpu, &u)?);
        let t_act = sentinel_f32(gpu, n)?;
        let t_pair = sentinel_f32(gpu, n)?;
        gpu.silu_mul_bf16_rt_f32(&t_g, &t_u, &t_act).map_err(e)?;
        gpu.rotate_x_mq_128_v2(&t_act, &t_pair, k, rows)
            .map_err(e)?;
        for with_shared in [false, true] {
            let t_fused = sentinel_f32(gpu, n)?;
            let sg = rng.vec(MI, -4.0, 4.0);
            let su = rng.vec(MI, -4.0, 4.0);
            let sel = [rng.range(-4.0, 4.0)];
            let (t_sg, t_su) = (up_f32(gpu, &sg)?, up_f32(gpu, &su)?);
            let (t_so, t_sel) = (sentinel_f32(gpu, MI)?, up_f32(gpu, &sel)?);
            let (t_ro, t_rsel) = (sentinel_f32(gpu, MI)?, up_f32(gpu, &sel)?);
            if with_shared {
                gpu.shared_expert_activation_bf16_f32(&t_sg, &t_su, &t_ro, &t_rsel, 1)
                    .map_err(e)?;
            }
            gpu.silu_mul_bf16_rt_rotate_x_mq_128_v2_shared(
                &t_g,
                &t_u,
                &t_fused,
                k,
                rows,
                if with_shared {
                    Some((&t_sg, &t_su, &t_so, &t_sel))
                } else {
                    None
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            let fused = down_f32(gpu, &t_fused)?;
            let pair = down_f32(gpu, &t_pair)?;
            let act = down_f32(gpu, &t_act)?;
            let tag = format!("rows={rows} shared={}", with_shared as u8);
            results.push(exact_f32(
                &format!("{tag} fused == silu_mul_bf16_rt + rotate_x_mq_128_v2"),
                &fused,
                &pair,
            ));
            results.push(exact_f32(
                &format!("{tag} fused == CPU FWHT128 of the device activation"),
                &fused,
                &rotate_rows_ref(&act, k, rows, 128, &s1, &s2),
            ));
            if with_shared {
                results.push(exact_f32(
                    &format!("{tag} shared out == shared_expert_activation_bf16_f32"),
                    &down_f32(gpu, &t_so)?,
                    &down_f32(gpu, &t_ro)?,
                ));
                results.push(exact_f32(
                    &format!("{tag} selector[0] == shared_expert_activation_bf16_f32"),
                    &down_f32(gpu, &t_sel)?,
                    &down_f32(gpu, &t_rsel)?,
                ));
            }
            free_all(gpu, vec![t_fused, t_sg, t_su, t_so, t_sel, t_ro, t_rsel])?;
        }
        free_all(gpu, vec![t_g, t_u, t_act, t_pair])?;
    }
    all(results)
}

/// `mq_rotate_x_gfx942` (gfx942-specific source, 64-lane `__shfl_xor(v, stride, 64)` stages,
/// block 64, 256-element groups) through `rotate_x_mq` and `rotate_x_mq_batched`.
fn rotate256_case(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x4D0E_0010);
    let (s1, s2) = mq_signs(gpu, false)?;
    let mut results = Vec::new();
    for &k in &[HIDDEN, HC_WIDE] {
        for &rows in &ROW_COUNTS {
            let x = rng.vec(rows * k, -3.0, 3.0);
            let want = rotate_rows_ref(&x, k, rows, 256, &s1, &s2);
            let t_x = up_f32(gpu, &x)?;
            let t_b = sentinel_f32(gpu, rows * k)?;
            gpu.rotate_x_mq_batched(&t_x, &t_b, k, rows).map_err(e)?;
            sync(gpu)?;
            let got = down_f32(gpu, &t_b)?;
            results.push(exact_f32(
                &format!("rotate_x_mq_batched k={k} rows={rows}"),
                &got,
                &want,
            ));
            free_all(gpu, vec![t_x, t_b])?;
        }
        // Single-row entry (`rotate_x_mq`).
        let x = rng.vec(k, -3.0, 3.0);
        let t_x = up_f32(gpu, &x)?;
        let t_o = sentinel_f32(gpu, k)?;
        gpu.rotate_x_mq(&t_x, &t_o, k).map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &t_o)?;
        results.push(exact_f32(
            &format!("rotate_x_mq k={k}"),
            &got,
            &rotate_rows_ref(&x, k, 1, 256, &s1, &s2),
        ));
        free_all(gpu, vec![t_x, t_o])?;
    }
    all(results)
}

// ── FP8B128 grouped-MoE activation kernels (gfx942-only, `Gfx942Device`) ─────────────────────────
const SENT_BITS: u32 = 0x7FC0_DEAD;

/// F32 values that stress `expf` and the BF16 boundaries: mostly [-4, 4], 5% up to |20|, exact
/// +-0, +-tiny (1e-20 .. 1e-38) and F32 denormals, and +-17.5 / +-88 / +-100 (`expf(-g)` large /
/// overflowing to +inf).
fn stress_vals(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| match rng.below(20) {
            0 => rng.range(-20.0, 20.0),
            1 => {
                if rng.below(2) == 0 {
                    0.0
                } else {
                    -0.0
                }
            }
            2 => {
                let m = [1e-20f32, 1e-30, 1e-38][rng.below(3)];
                if rng.below(2) == 0 {
                    m
                } else {
                    -m
                }
            }
            3 => [-100.0f32, 100.0, -88.0, 88.0, -17.5, 17.5][rng.below(6)],
            4 => [1e-40f32, -1e-40, 5e-45, -5e-45][rng.below(4)],
            _ => rng.range(-4.0, 4.0),
        })
        .collect()
}

fn bf16_one_ulp_apart(a: f32, b: f32) -> bool {
    a.is_finite()
        && b.is_finite()
        && (a.to_bits() >> 31) == (b.to_bits() >> 31)
        && ((a.to_bits() >> 16) as i32 - (b.to_bits() >> 16) as i32).abs() == 1
}

/// Bitwise comparison with a diagnosis: `want` elements holding the NaN sentinel bits are
/// "must stay untouched". No tolerance. A mismatch reports the counts, the first index, got/want
/// and how many live mismatches are exactly one BF16 ulp apart (device `expf` vs CPU `expf`).
fn exact_bits_report(label: &str, got: &[f32], want: &[f32]) -> Result<String, String> {
    if got.len() != want.len() {
        return Err(format!("{label}: len {} != {}", got.len(), want.len()));
    }
    let (mut live_bad, mut one_ulp, mut untouched_bad, mut live, mut first) =
        (0usize, 0usize, 0usize, 0usize, None);
    for i in 0..got.len() {
        let untouched = want[i].to_bits() == SENT_BITS;
        if !untouched {
            live += 1;
        }
        if got[i].to_bits() == want[i].to_bits() {
            continue;
        }
        first.get_or_insert(i);
        if untouched {
            untouched_bad += 1;
        } else {
            live_bad += 1;
            if bf16_one_ulp_apart(got[i], want[i]) {
                one_ulp += 1;
            }
        }
    }
    match first {
        None => Ok(format!("{label}: bit-exact, {live} live elements, untouched slots intact")),
        Some(i) => Err(format!(
            "{label}: {} mismatches ({live_bad} live of {live}, of which {one_ulp} are exactly 1 BF16 ulp apart; \
             {untouched_bad} untouched slots overwritten); first @{i}: got {:?} ({:#010x}) want {:?} ({:#010x})",
            live_bad + untouched_bad,
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        )),
    }
}

/// Det-teacher BF16 SwiGLU (`kernels/src/fp8b128_moe_unscatter_silu.gfx942.hip:50-53`).
fn fp8_swiglu_ref(gate: f32, up: f32) -> f32 {
    let g = bf16r(gate);
    let u = bf16r(up);
    let s = bf16r(g / (1.0f32 + (-g).exp()));
    bf16r(s * u)
}

const FP8_ACT_ROWS: [usize; 8] = [1, 2, 7, 8, 16, 17, 128, 1536];

/// `Gfx942Device::fp8b128_moe_unscatter_silu` (`rdna-compute/src/cdna/gfx942.rs`,
/// `kernels/src/fp8b128_moe_unscatter_silu.gfx942.hip`): grouped gate/up rows of a pad-16 layout,
/// `sorted` carries -1 for padding; activation slots never named by `sorted` stay NaN-sentinel.
fn fp8_unscatter_silu_case(gpu: &mut Gpu) -> Result<String, String> {
    if gpu.try_gfx942().is_none() {
        return Ok("skipped: not gfx942".to_string());
    }
    let mut rng = Rng::new(0x4D0E_0011);
    let mut results = Vec::new();
    for &rows in &FP8_ACT_ROWS {
        let slots = rows * K;
        let topk = make_topk(
            &mut rng,
            rows,
            if rows < 16 {
                Spread::Uniform
            } else {
                Spread::Skewed
            },
        );
        let lay = scatter_ref(&topk);
        // Grouped rows actually used (a multiple of 16); the rest of the layout is -1 tail.
        let n_sorted = lay.offsets[E] as usize;
        // ~10% of the live entries are dropped (-1) and ~10% pushed out of the slot range
        // (`flat >= slots`): the kernel must skip both, leaving those activation rows at the
        // NaN sentinel.
        let mut sorted_v: Vec<i32> = lay.sorted[..n_sorted].to_vec();
        for entry in sorted_v.iter_mut() {
            if *entry >= 0 {
                match rng.below(10) {
                    0 => *entry = -1,
                    1 => *entry = (slots + 3) as i32,
                    _ => {}
                }
            }
        }
        let sorted = &sorted_v[..];
        let y = {
            let mut v = stress_vals(&mut rng, n_sorted * 2 * MI);
            // Padding rows carry arbitrary finite garbage: they must not reach the output.
            for (p, &flat) in sorted.iter().enumerate() {
                if flat < 0 {
                    v[p * 2 * MI..(p + 1) * 2 * MI]
                        .iter_mut()
                        .for_each(|x| *x = 7.0);
                }
            }
            v
        };
        let t_y = up_f32(gpu, &y)?;
        let t_sorted = ints(gpu, sorted)?;
        let t_act = sentinel_f32(gpu, slots * MI)?;
        {
            let mut dev = gpu.try_gfx942().ok_or("not gfx942")?;
            dev.fp8b128_moe_unscatter_silu(&t_y, &t_sorted, &t_act, MI, n_sorted, slots)
                .map_err(e)?;
        }
        sync(gpu)?;
        let got = down_f32(gpu, &t_act)?;
        free_all(gpu, vec![t_y, t_sorted, t_act])?;
        let mut want = vec![f32::from_bits(SENT_BITS); slots * MI];
        for (p, &flat) in sorted.iter().enumerate() {
            if flat < 0 || flat as usize >= slots {
                continue;
            }
            for col in 0..MI {
                want[flat as usize * MI + col] =
                    fp8_swiglu_ref(y[p * 2 * MI + col], y[p * 2 * MI + MI + col]);
            }
        }
        results.push(exact_bits_report(
            &format!("rows={rows} G={n_sorted}"),
            &got,
            &want,
        ));
    }
    all(results)
}

/// `Gfx942Device::fp8b128_moe_shared_activation`
/// (`kernels/src/fp8b128_moe_shared_activation.gfx942.hip`): BF16 SwiGLU rows plus the in-place
/// selector `bf16(sigmoid(bf16(logit)))`.
fn fp8_shared_activation_case(gpu: &mut Gpu) -> Result<String, String> {
    if gpu.try_gfx942().is_none() {
        return Ok("skipped: not gfx942".to_string());
    }
    let mut rng = Rng::new(0x4D0E_0012);
    let mut results = Vec::new();
    for &rows in &FP8_ACT_ROWS {
        let n = rows * MI;
        let gate = stress_vals(&mut rng, n);
        let up = stress_vals(&mut rng, n);
        let sel = stress_vals(&mut rng, rows);
        let (t_g, t_u) = (up_f32(gpu, &gate)?, up_f32(gpu, &up)?);
        let t_act = sentinel_f32(gpu, n)?;
        let t_sel = up_f32(gpu, &sel)?;
        {
            let mut dev = gpu.try_gfx942().ok_or("not gfx942")?;
            dev.fp8b128_moe_shared_activation(&t_g, &t_u, &t_act, &t_sel, rows, MI)
                .map_err(e)?;
        }
        sync(gpu)?;
        let got_act = down_f32(gpu, &t_act)?;
        let got_sel = down_f32(gpu, &t_sel)?;
        free_all(gpu, vec![t_g, t_u, t_act, t_sel])?;
        let want_act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| fp8_swiglu_ref(g, u))
            .collect();
        let want_sel: Vec<f32> = sel
            .iter()
            .map(|&s| {
                let l = bf16r(s);
                bf16r(1.0f32 / (1.0f32 + (-l).exp()))
            })
            .collect();
        results.push(exact_bits_report(
            &format!("rows={rows} activation"),
            &got_act,
            &want_act,
        ));
        results.push(exact_bits_report(
            &format!("rows={rows} selector (in place, once)"),
            &got_sel,
            &want_sel,
        ));
    }
    all(results)
}
