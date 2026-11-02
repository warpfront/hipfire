// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle, CPU structural proof and timing harness for the wave-per-row
//! fused prefill MoE router (`Gpu::moe_router_softmax_topk_k8_multirow_prefill`,
//! symbol `moe_router_softmax_topk_k8_wave`: one wave32 per row, 8 rows per
//! 256-thread block, A3B shape: 256 experts, top-8).
//!
//! Three routes are compared on identical inputs (n_exp 256, k 8):
//! * incumbent pair: `gpu.softmax_f32(&logits)` + `gpu.moe_topk_renorm_k8_batched(..)`.
//! * current fused: `gpu.moe_router_softmax_topk_k8_prefill(.., lanes = 128)`.
//! * new: `gpu.moe_router_softmax_topk_k8_multirow_prefill(..)` (wave-per-row).
//!
//! Modes (one per invocation):
//! * `--plan` (CPU, no GPU): prints the launch geometry, the admission table, and
//!   the exact GPU case grid with an FNV-1a/64 hash of every deterministic input,
//!   so the owner run can be matched to the plan.
//! * `--self-check` (CPU, no GPU): proves the wave structure without libm/GPU exp
//!   assumptions. An independent serial reference of the original 256-thread
//!   softmax + mode-0 top-8 (strict `>` scan, tree s = 128..1,
//!   replace-on-strictly-greater, sequential renorm, clamp) is compared bit for
//!   bit with a lane-level emulation of the new kernel (32 lanes, two float4
//!   per lane, xor/down shuffles, bit-reversed tie priority). Both sides use the
//!   same CPU `exp`/division, so only the structure (trees, ties, lane layout,
//!   row mapping) is under test. Also checks: the row/block/wave layout, the
//!   tie rule (winner among equal maxima = smallest bit-reversed index, proved on
//!   the serial reference), hand-derived tie orders (all-zero picks 0,128,64,192;
//!   {5,133} -> 5,133; {1,2} -> 2,1: tree position, not lowest index), non-finite
//!   rows and the pure admission rule.
//! * default / `--gpu` (wave32 GPU, one process): rows {8, 513, 4096, 7808, 8192}
//!   (7808 = the A3B second chunk, 513 = odd tail block) x n_exp 256
//!   x norm {false, true} x inputs {allzero, random, tie_heavy, tie_tree_cross,
//!   mixed_rows, nonfinite, flat_tail}. Ids (raw i32 in F32-typed tensors) and
//!   weights are compared by raw bit pattern: new vs current fused AND new vs
//!   incumbent (and, for the record, current fused vs incumbent). Then per rows x
//!   norm the event-timed median over 50 reps (10 warmups, >= 20 required) of each
//!   route on the random input. Every rep first restores the raw logits (softmax is
//!   in place) from a GPU-resident pristine copy outside the event interval.
//!
//! Output (stdout, one JSON object per line): per-comparison verdict lines, timing
//! lines `{"timing":true,"rows","norm","median_us":{"incumbent_pair","current_fused","new"},..}`
//! and a final summary line containing `all_byte_identical` (bool) and
//! `first_mismatch` (null or an object with comparison, rows, variant, norm, row,
//! pick, ids and weight hex bit patterns). Exit code: 0 all pass; 1 mismatch;
//! 2 unusable device/arguments.
//!
//! Commands:
//! * CPU: `cargo run --release -p rdna-compute --features lab --example moe_router_topk_multirow_oracle -- --plan`
//!   `cargo run --release -p rdna-compute --features lab --example moe_router_topk_multirow_oracle -- --self-check`
//! * GPU (owner, wave32 device): `cargo run --release -p rdna-compute --features lab --example moe_router_topk_multirow_oracle`
//!
//! The runtime flag `HIPFIRE_MOE_ROUTER_TOPK_MULTIROW` is intentionally not
//! consulted: the raw launcher is exercised directly.

use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

const TOP_K: usize = 8;
const N_EXP: usize = 256;
/// Physical lanes per row of the current fused (one-row, 128-lane) route.
const LANES: usize = 128;
/// Lanes (one wave32) per row and rows per block of the new wave-per-row route.
const WAVE: usize = 32;
const ROWS_PER_BLOCK: usize = 8;

const GPU_ROWS: [usize; 5] = [8, 513, 4096, 7808, 8192];
const VARIANTS: [&str; 7] = [
    "allzero",
    "random",
    "tie_heavy",
    "tie_tree_cross",
    "mixed_rows",
    "nonfinite",
    "flat_tail",
];
const MIXED_KINDS: [&str; 4] = ["allzero", "random", "tie_heavy", "tie_tree_cross"];
/// Equal-maximum position sets that cross virtual tree positions (n_exp 256: one
/// element per virtual thread, so the position is the virtual thread).
const TREE_SETS: [&[usize]; 10] = [
    &[0, 128],
    &[1, 2],
    &[5, 133],
    &[127, 255],
    &[64, 192],
    &[31, 32, 159, 160],
    &[0, 64, 128, 192],
    &[3, 67, 131, 195, 7, 71, 135, 199],
    &[255, 254, 127, 126],
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 130, 131, 132],
];
/// Poison pattern for unwritten output slots (not a valid raw i32 expert id).
const SENTINEL_BITS: u32 = 0xdead_beef;

const WARMUPS: usize = 10;
const REPS: usize = 50;
const _: () = assert!(REPS >= 20);
const BUDGET_S: f64 = 120.0;
/// Stop starting new timing groups past this elapsed time (seconds).
const TIMING_CUTOFF_S: f64 = 100.0;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [a] if a == "--plan" => plan(),
        [a] if a == "--self-check" => self_check(),
        [] => gpu_mode(),
        [a] if a == "--gpu" => gpu_mode(),
        _ => {
            eprintln!("usage: moe_router_topk_multirow_oracle [--plan | --self-check | --gpu]");
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// Admission (pure) shared by --plan, --self-check and the GPU mode.
// ---------------------------------------------------------------------------

fn expected_admission(
    enabled: bool,
    k: usize,
    n_exp: usize,
    batch: usize,
    has_wave32: bool,
) -> bool {
    enabled && k == 8 && n_exp == 256 && has_wave32 && batch >= 1
}

fn check_admission() -> Result<usize, String> {
    let mut checked = 0usize;
    for enabled in [false, true] {
        for k in [1usize, 4, 8, 10] {
            for n_exp in [64usize, 128, 255, 256, 257, 512, 1024] {
                for batch in [0usize, 1, 2, 3, 4, 7, 8, 9, 512, 4095, 4096, 8192, 8193] {
                    for wave32 in [false, true] {
                        let got = Gpu::moe_router_topk_multirow_admitted(
                            enabled, k, n_exp, batch, wave32,
                        );
                        let want = expected_admission(enabled, k, n_exp, batch, wave32);
                        checked += 1;
                        if got != want {
                            return Err(format!(
                                "admission mismatch enabled={enabled} k={k} n_exp={n_exp} \
                                 batch={batch} wave32={wave32}: got {got}, want {want}"
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(checked)
}

// ---------------------------------------------------------------------------
// --plan
// ---------------------------------------------------------------------------

fn plan() {
    println!(
        "{{\"plan\":true,\"geometry\":{{\"lanes_per_row\":{WAVE},\"rows_per_block\":{ROWS_PER_BLOCK},\
         \"block\":[{},1,1],\"n_exp\":{N_EXP},\"k\":{TOP_K},\"dynamic_lds_bytes\":0,\
         \"floats_per_lane\":{}}}}}",
        WAVE * ROWS_PER_BLOCK,
        N_EXP / WAVE,
    );
    for &rows in &GPU_ROWS {
        println!(
            "{{\"plan_grid\":true,\"rows\":{rows},\"grid\":[{}, 1, 1],\"admitted_wave32\":{}}}",
            rows.div_ceil(ROWS_PER_BLOCK),
            expected_admission(true, TOP_K, N_EXP, rows, true)
        );
    }
    match check_admission() {
        Ok(n) => println!("{{\"plan_admission\":true,\"combinations_checked\":{n},\"pass\":true}}"),
        Err(e) => {
            println!("{{\"plan_admission\":true,\"pass\":false,\"error\":\"{e}\"}}");
            std::process::exit(1);
        }
    }
    let mut cases = 0usize;
    for &rows in &GPU_ROWS {
        for variant in VARIANTS {
            let logits = make_logits(variant, rows);
            let hash = fnv1a(&logits);
            for norm in [false, true] {
                cases += 1;
                println!(
                    "{{\"plan_case\":true,\"rows\":{rows},\"n_exp\":{N_EXP},\"variant\":\"{variant}\",\
                     \"norm\":{norm},\"input_fnv1a64\":\"{hash:016x}\",\
                     \"comparisons\":[\"new_vs_current_fused\",\"new_vs_incumbent\",\"current_fused_vs_incumbent\"]}}"
                );
            }
        }
    }
    println!(
        "{{\"plan_summary\":true,\"cases\":{cases},\"gated_comparisons\":{},\"timing_groups\":{},\
         \"timing_warmups\":{WARMUPS},\"timing_reps\":{REPS},\"budget_s\":{BUDGET_S}}}",
        cases * 2,
        GPU_ROWS.len() * 2
    );
}

// ---------------------------------------------------------------------------
// --self-check: CPU structural proof
// ---------------------------------------------------------------------------

fn self_check() {
    let mut failures: Vec<String> = Vec::new();
    let mut checks = 0usize;
    fn record(name: &str, r: Result<(), String>, failures: &mut Vec<String>) {
        let pass = r.is_ok();
        println!("{{\"self_check\":\"{name}\",\"pass\":{pass}}}");
        if let Err(e) = r {
            eprintln!("FAIL {name}: {e}");
            failures.push(format!("{name}: {e}"));
        }
    }

    record("row_block_wave_layout", check_layout(), &mut failures);
    checks += 1;
    record(
        "admission_rule",
        check_admission().map(|_| ()),
        &mut failures,
    );
    checks += 1;
    record("tie_rule_bit_reversed", check_tie_rule(), &mut failures);
    checks += 1;
    record("hand_derived_ties", check_hand_ties(), &mut failures);
    checks += 1;

    for variant in VARIANTS {
        for norm in [false, true] {
            // 133 rows: 16 full blocks + a 5-row tail block.
            let logits = make_logits(variant, 133);
            let name = format!("emulation_vs_reference/{variant}/norm={norm}");
            record(&name, check_emulation(&logits, 133, norm), &mut failures);
            checks += 1;
        }
    }

    println!(
        "{{\"self_check_summary\":true,\"checks\":{checks},\"failures\":{},\"pass\":{}}}",
        failures.len(),
        failures.is_empty()
    );
    if !failures.is_empty() {
        std::process::exit(1);
    }
}

/// `row = blockIdx.x * 8 + wave` over grid `ceil(batch / 8)` with the
/// `row >= batch` early exit covers every row exactly once and every output slot
/// `row * 8 + [0, 8)` exactly once (lane < 8 of the owning wave), for every
/// batch size including partial tail blocks.
fn check_layout() -> Result<(), String> {
    for batch in [1usize, 2, 7, 8, 9, 15, 16, 17, 133, 513, 4095, 7808, 8192] {
        let grid = batch.div_ceil(ROWS_PER_BLOCK);
        let mut seen = vec![0u8; batch];
        let mut out = vec![0u8; batch * TOP_K];
        for block in 0..grid {
            for wave in 0..ROWS_PER_BLOCK {
                let row = block * ROWS_PER_BLOCK + wave;
                if row >= batch {
                    continue;
                }
                seen[row] += 1;
                for lane in 0..TOP_K {
                    out[row * TOP_K + lane] += 1;
                }
            }
        }
        if seen.iter().any(|&c| c != 1) || out.iter().any(|&c| c != 1) {
            return Err(format!(
                "batch {batch}: rows/outputs not covered exactly once"
            ));
        }
        if (grid - 1) * ROWS_PER_BLOCK >= batch {
            return Err(format!("batch {batch}: empty trailing block"));
        }
    }
    // Two float4 loads per lane cover the 256-expert row exactly once.
    let mut covered = [0u8; N_EXP];
    for lane in 0..WAVE {
        for c in 0..4 {
            covered[4 * lane + c] += 1;
            covered[128 + 4 * lane + c] += 1;
        }
    }
    if covered.iter().any(|&c| c != 1) {
        return Err("lane layout does not cover every expert exactly once".into());
    }
    Ok(())
}

/// The serial reference resolves equal maxima by the smallest bit-reversed
/// (8-bit) expert index; the wave kernel's priority key relies on exactly this.
fn check_tie_rule() -> Result<(), String> {
    let mut rng = Rng::new(7, 9001);
    for round in 0..4000usize {
        let count = 1 + rng.below(24);
        let set = rng.positions(count);
        let probs = {
            let mut p = vec![0.01f32; N_EXP];
            for &i in &set {
                p[i] = 0.5;
            }
            p
        };
        let (ids, _) = reference_row(&probs, false);
        let mut want: Vec<usize> = set.clone();
        want.sort_by_key(|&i| (i as u8).reverse_bits());
        for k in 0..TOP_K.min(count) {
            if ids[k] as usize != want[k] {
                return Err(format!(
                    "round {round}: pick {k} is {}, bit-reversed order says {} (set {set:?})",
                    ids[k], want[k]
                ));
            }
        }
    }
    Ok(())
}

fn check_hand_ties() -> Result<(), String> {
    let cases: [(&str, Vec<f32>, &[i32]); 3] = [
        ("allzero", vec![0.0; N_EXP], &[0, 128, 64, 192][..]),
        ("pair_5_133", pair_row(&[5, 133], 11), &[5, 133][..]),
        ("pair_1_2", pair_row(&[1, 2], 12), &[2, 1][..]),
    ];
    for (name, row, want) in cases {
        for norm in [false, true] {
            let (ids, _) = reference_row(&reference_softmax(&row), norm);
            if ids[..want.len()] != *want {
                return Err(format!(
                    "{name} norm={norm}: reference picks {:?}, expected prefix {want:?}",
                    &ids[..want.len()]
                ));
            }
            check_emulation(&row, 1, norm).map_err(|e| format!("{name} norm={norm}: {e}"))?;
        }
    }
    Ok(())
}

/// Compares the lane-level emulation of the new kernel to the serial reference,
/// bit for bit, for every row of `logits`.
fn check_emulation(logits: &[f32], rows: usize, norm: bool) -> Result<(), String> {
    if logits.len() != rows * N_EXP {
        return Err("logits must be sized rows * n_exp".into());
    }
    for row in 0..rows {
        let row_logits = &logits[row * N_EXP..(row + 1) * N_EXP];
        let (got_ids, got_w) = emulate_wave_row(row_logits, norm);
        let (ref_ids, ref_w) = reference_row(&reference_softmax(row_logits), norm);
        for pick in 0..TOP_K {
            if got_ids[pick] != ref_ids[pick] || got_w[pick].to_bits() != ref_w[pick].to_bits() {
                return Err(format!(
                    "row {row} pick {pick}: emulated id {} weight {:#010x}, reference id {} \
                     weight {:#010x}",
                    got_ids[pick],
                    got_w[pick].to_bits(),
                    ref_ids[pick],
                    ref_w[pick].to_bits()
                ));
            }
        }
    }
    Ok(())
}

fn fmax(a: f32, b: f32) -> f32 {
    a.max(b)
}

fn fadd(a: f32, b: f32) -> f32 {
    a + b
}

/// Serial emulation of the incumbent `softmax_f32` (block `min(256, n)`):
/// per-thread strided max from -1e30, tree max, per-thread strided sum of exp,
/// tree sum, direct divide.
fn reference_softmax(row: &[f32]) -> Vec<f32> {
    let n = row.len();
    let vb = n.min(256);
    let mut red = vec![0.0f32; vb];
    for (t, slot) in red.iter_mut().enumerate() {
        let mut m = -1e30f32;
        let mut i = t;
        while i < n {
            m = fmax(m, row[i]);
            i += vb;
        }
        *slot = m;
    }
    let mut s = vb >> 1;
    while s > 0 {
        for t in 0..s {
            red[t] = fmax(red[t], red[t + s]);
        }
        s >>= 1;
    }
    let max_val = red[0];
    let mut ex = vec![0.0f32; n];
    for (t, slot) in red.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        let mut i = t;
        while i < n {
            let e = (row[i] - max_val).exp();
            ex[i] = e;
            sum += e;
            i += vb;
        }
        *slot = sum;
    }
    let mut s = vb >> 1;
    while s > 0 {
        for t in 0..s {
            red[t] = fadd(red[t], red[t + s]);
        }
        s >>= 1;
    }
    let sum_all = red[0];
    ex.iter().map(|e| e / sum_all).collect()
}

/// Serial emulation of the original mode-0 `moe_topk_renorm_k8_batched`
/// (256 threads, padded smem, strict `>` scan, replace-on-strictly-greater tree,
/// sequential renorm sum, direct division, NaN/OOB clamp).
fn reference_row(probs: &[f32], norm: bool) -> ([i32; TOP_K], [f32; TOP_K]) {
    let n = probs.len();
    let mut smem_v = vec![f32::NEG_INFINITY; 1024];
    let mut smem_i = vec![0i32; 1024];
    for i in 0..1024 {
        smem_v[i] = if i < n { probs[i] } else { f32::NEG_INFINITY };
        smem_i[i] = i as i32;
    }
    let mut picked_idx = [0i32; TOP_K];
    let mut picked_w = [0.0f32; TOP_K];
    for k in 0..TOP_K {
        let mut rv = vec![0.0f32; 256];
        let mut ri = vec![0i32; 256];
        for tid in 0..256 {
            let mut best_v = f32::NEG_INFINITY;
            let mut best_i = -1i32;
            let mut i = tid;
            while i < n {
                let v = smem_v[i];
                if v > best_v {
                    best_v = v;
                    best_i = smem_i[i];
                }
                i += 256;
            }
            rv[tid] = best_v;
            ri[tid] = best_i;
        }
        let mut s = 128usize;
        while s > 0 {
            for t in 0..s {
                let (va, vb) = (rv[t], rv[t + s]);
                if vb > va {
                    rv[t] = vb;
                    ri[t] = ri[t + s];
                }
            }
            s >>= 1;
        }
        let winner = ri[0];
        picked_idx[k] = winner;
        picked_w[k] = rv[0];
        if winner >= 0 && (winner as usize) < n {
            smem_v[winner as usize] = f32::NEG_INFINITY;
        }
    }
    let mut s = 0.0f32;
    for w in picked_w.iter() {
        s += *w;
    }
    if norm && s > 0.0 {
        for w in picked_w.iter_mut() {
            *w /= s;
        }
    }
    for k in 0..TOP_K {
        if picked_idx[k] < 0 || picked_idx[k] as usize >= n {
            picked_idx[k] = 0;
            picked_w[k] = 0.0;
        }
    }
    (picked_idx, picked_w)
}

/// `__shfl_xor(v, off)` across the wave.
fn shfl_xor<T: Copy>(v: &[T; WAVE], lane: usize, off: usize) -> T {
    v[lane ^ off]
}

/// `__shfl_down(v, off)`: lane + off, or the lane's own value past the wave end.
fn shfl_down<T: Copy>(v: &[T; WAVE], lane: usize, off: usize) -> T {
    if lane + off < WAVE {
        v[lane + off]
    } else {
        v[lane]
    }
}

/// `a` beats `b`: larger value, equal value with the smaller priority.
fn better(va: f32, pa: i32, vb: f32, pb: i32) -> bool {
    va > vb || (va == vb && pa < pb)
}

/// Lane-level emulation of one wave of `moe_router_softmax_topk_k8_wave`
/// (32 lanes run in lockstep; every shuffle is a whole-wave step, so the lane loops
/// below are exact). Lane `l` holds experts `4l + c` (`x[l][c]`) and
/// `128 + 4l + c` (`x[l][4 + c]`).
fn emulate_wave_row(row: &[f32], norm: bool) -> ([i32; TOP_K], [f32; TOP_K]) {
    let mut x = [[0.0f32; 8]; WAVE];
    for l in 0..WAVE {
        for c in 0..4 {
            x[l][c] = row[4 * l + c];
            x[l][4 + c] = row[128 + 4 * l + c];
        }
    }

    // Row max: in-lane fold from -1e30, xor butterfly.
    let mut m = [-1e30f32; WAVE];
    for l in 0..WAVE {
        for j in 0..8 {
            m[l] = fmax(m[l], x[l][j]);
        }
    }
    for off in [16usize, 8, 4, 2, 1] {
        let prev = m;
        for l in 0..WAVE {
            m[l] = fmax(prev[l], shfl_xor(&prev, l, off));
        }
    }
    let max_val = m[0];

    // exp(x - max), tree sum.
    let mut e = [[0.0f32; 8]; WAVE];
    for l in 0..WAVE {
        for j in 0..8 {
            e[l][j] = (x[l][j] - max_val).exp();
        }
    }
    let mut u = [[0.0f32; 4]; WAVE];
    for l in 0..WAVE {
        for c in 0..4 {
            u[l][c] = e[l][c] + e[l][4 + c]; // s = 128
        }
    }
    for off in [16usize, 8, 4, 2, 1] {
        // s = 64, 32, 16, 8, 4
        for c in 0..4 {
            let mut col = [0.0f32; WAVE];
            for l in 0..WAVE {
                col[l] = u[l][c];
            }
            for l in 0..WAVE {
                u[l][c] = col[l] + shfl_down(&col, l, off);
            }
        }
    }
    let q0 = u[0][0] + u[0][2]; // s = 2
    let q1 = u[0][1] + u[0][3];
    let sum_all = q0 + q1; // s = 1

    // Probabilities and bit-reversed priorities.
    let mut v = [[0.0f32; 8]; WAVE];
    let mut pr = [[0i32; 8]; WAVE];
    for l in 0..WAVE {
        for j in 0..8 {
            let p = e[l][j] / sum_all;
            v[l][j] = if p > f32::NEG_INFINITY {
                p
            } else {
                f32::NEG_INFINITY
            };
            let pos = (if j < 4 { 0 } else { 128 }) + 4 * l + (j & 3);
            pr[l][j] = i32::from((pos as u8).reverse_bits());
        }
    }

    let mut pick_i = [0i32; TOP_K];
    let mut pick_w = [0.0f32; TOP_K];
    for k in 0..TOP_K {
        let mut bv = [0.0f32; WAVE];
        let mut bp = [0i32; WAVE];
        for l in 0..WAVE {
            bv[l] = v[l][0];
            bp[l] = pr[l][0];
            for j in 1..8 {
                if better(v[l][j], pr[l][j], bv[l], bp[l]) {
                    bv[l] = v[l][j];
                    bp[l] = pr[l][j];
                }
            }
        }
        for off in [16usize, 8, 4, 2, 1] {
            let (pv, pp) = (bv, bp);
            for l in 0..WAVE {
                let (ov, op) = (shfl_xor(&pv, l, off), shfl_xor(&pp, l, off));
                if better(ov, op, pv[l], pp[l]) {
                    bv[l] = ov;
                    bp[l] = op;
                }
            }
        }
        // Every lane holds the same winner.
        if (1..WAVE).any(|l| bv[l].to_bits() != bv[0].to_bits() || bp[l] != bp[0]) {
            panic!("xor butterfly lanes disagree");
        }
        pick_i[k] = if bv[0] > f32::NEG_INFINITY {
            i32::from((bp[0] as u8).reverse_bits())
        } else {
            -1
        };
        pick_w[k] = bv[0];
        for l in 0..WAVE {
            for j in 0..8 {
                if pr[l][j] == bp[0] {
                    v[l][j] = f32::NEG_INFINITY;
                }
            }
        }
    }

    let mut s = 0.0f32;
    for w in pick_w.iter() {
        s += *w;
    }
    if norm && s > 0.0 {
        for w in pick_w.iter_mut() {
            *w /= s;
        }
    }
    for k in 0..TOP_K {
        if pick_i[k] < 0 || pick_i[k] as usize >= N_EXP {
            pick_i[k] = 0;
            pick_w[k] = 0.0;
        }
    }
    (pick_i, pick_w)
}

// ---------------------------------------------------------------------------
// GPU mode
// ---------------------------------------------------------------------------

struct Report {
    comparisons: usize,
    failures: usize,
    first_mismatch: Option<String>,
}

fn gpu_mode() {
    let start = Instant::now();
    let mut gpu = Gpu::init().expect("GPU init");
    if !gpu.arch_caps.is_wave32() {
        eprintln!(
            "REJECT: moe_router_topk_multirow_oracle requires a wave32 device; arch {} is not wave32",
            gpu.arch
        );
        std::process::exit(2);
    }
    for &rows in &GPU_ROWS {
        if !Gpu::moe_router_topk_multirow_admitted(true, TOP_K, N_EXP, rows, true) {
            eprintln!(
                "REJECT: multirow router not admitted for k={TOP_K} n_exp={N_EXP} rows={rows}"
            );
            std::process::exit(2);
        }
    }

    let mut report = Report {
        comparisons: 0,
        failures: 0,
        first_mismatch: None,
    };
    let mut cases = 0usize;
    for &rows in &GPU_ROWS {
        for variant in VARIANTS {
            let logits = make_logits(variant, rows);
            let hash = fnv1a(&logits);
            for norm in [false, true] {
                cases += 1;
                run_case(&mut gpu, &logits, rows, variant, norm, hash, &mut report);
            }
        }
    }

    let mut timings: Vec<String> = Vec::new();
    let mut timing_truncated = false;
    for &rows in &GPU_ROWS {
        for norm in [false, true] {
            if start.elapsed().as_secs_f64() > TIMING_CUTOFF_S {
                timing_truncated = true;
                continue;
            }
            let line = run_timing(&mut gpu, rows, norm);
            println!("{line}");
            timings.push(line);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let all_byte_identical = report.failures == 0;
    let first = report.first_mismatch.as_deref().unwrap_or("null");
    println!(
        "{{\"summary\":true,\"all_byte_identical\":{all_byte_identical},\"cases\":{cases},\
         \"comparisons\":{},\"failures\":{},\"first_mismatch\":{first},\
         \"timing_groups\":{},\"timing_truncated\":{timing_truncated},\"timing_reps\":{REPS},\
         \"elapsed_s\":{elapsed:.1},\"within_{}s\":{}}}",
        report.comparisons,
        report.failures,
        timings.len(),
        BUDGET_S as u32,
        elapsed <= BUDGET_S,
    );
    if !all_byte_identical || timing_truncated || elapsed > BUDGET_S {
        std::process::exit(1);
    }
}

fn upload_sentinel(gpu: &mut Gpu, rows: usize) -> (GpuTensor, GpuTensor) {
    let sentinel = vec![f32::from_bits(SENTINEL_BITS); rows * TOP_K];
    let ids = gpu.upload_f32(&sentinel, &[rows, TOP_K]).expect("ids");
    let w = gpu.upload_f32(&sentinel, &[rows, TOP_K]).expect("weights");
    (ids, w)
}

fn run_case(
    gpu: &mut Gpu,
    logits: &[f32],
    rows: usize,
    variant: &str,
    norm: bool,
    hash: u64,
    report: &mut Report,
) {
    // Incumbent pair (softmax is in place).
    let l_inc = gpu
        .upload_f32(logits, &[rows, N_EXP])
        .expect("upload incumbent logits");
    let (i_inc, w_inc) = upload_sentinel(gpu, rows);
    gpu.softmax_f32(&l_inc).expect("incumbent softmax");
    gpu.moe_topk_renorm_k8_batched(&l_inc, &i_inc, &w_inc, N_EXP, norm, rows)
        .expect("incumbent batched top-k");
    let inc_idx = gpu.download_f32(&i_inc).expect("download incumbent ids");
    let inc_w = gpu
        .download_f32(&w_inc)
        .expect("download incumbent weights");
    for t in [l_inc, i_inc, w_inc] {
        gpu.free_tensor(t).expect("free");
    }

    // Current fused, 128 lanes.
    let l_fus = gpu
        .upload_f32(logits, &[rows, N_EXP])
        .expect("upload fused logits");
    let (i_fus, w_fus) = upload_sentinel(gpu, rows);
    gpu.moe_router_softmax_topk_k8_prefill(&l_fus, &i_fus, &w_fus, N_EXP, norm, rows, LANES)
        .expect("current fused router");
    let fus_idx = gpu.download_f32(&i_fus).expect("download fused ids");
    let fus_w = gpu.download_f32(&w_fus).expect("download fused weights");
    for t in [l_fus, i_fus, w_fus] {
        gpu.free_tensor(t).expect("free");
    }

    // New multirow route.
    let l_new = gpu
        .upload_f32(logits, &[rows, N_EXP])
        .expect("upload new logits");
    let (i_new, w_new) = upload_sentinel(gpu, rows);
    gpu.moe_router_softmax_topk_k8_multirow_prefill(&l_new, &i_new, &w_new, N_EXP, norm, rows)
        .expect("multirow router");
    let new_idx = gpu.download_f32(&i_new).expect("download new ids");
    let new_w = gpu.download_f32(&w_new).expect("download new weights");
    for t in [l_new, i_new, w_new] {
        gpu.free_tensor(t).expect("free");
    }

    let ctx = CaseCtx {
        rows,
        variant,
        norm,
    };
    // (comparison name, reference ids, reference weights, got ids, got weights)
    type Comparison<'a> = (&'a str, &'a [f32], &'a [f32], &'a [f32], &'a [f32]);
    let pairs: [Comparison<'_>; 3] = [
        (
            "new_vs_current_fused",
            fus_idx.as_slice(),
            fus_w.as_slice(),
            new_idx.as_slice(),
            new_w.as_slice(),
        ),
        (
            "new_vs_incumbent",
            inc_idx.as_slice(),
            inc_w.as_slice(),
            new_idx.as_slice(),
            new_w.as_slice(),
        ),
        (
            "current_fused_vs_incumbent",
            inc_idx.as_slice(),
            inc_w.as_slice(),
            fus_idx.as_slice(),
            fus_w.as_slice(),
        ),
    ];
    for (name, ref_idx, ref_w, got_idx, got_w) in pairs {
        let mismatch = first_mismatch(ref_idx, ref_w, got_idx, got_w);
        let pass = mismatch.is_none();
        println!(
            "{{\"rows\":{rows},\"n_exp\":{N_EXP},\"variant\":\"{variant}\",\"norm\":{norm},\
             \"input_fnv1a64\":\"{hash:016x}\",\"comparison\":\"{name}\",\"pass\":{pass}}}"
        );
        // The third pair is informational (existing fused route vs incumbent);
        // a failure there is still a failure of the byte oracle.
        report.comparisons += 1;
        if let Some(m) = mismatch {
            report.failures += 1;
            let json = ctx.mismatch_json(name, &m);
            eprintln!("MISMATCH {json}");
            if report.first_mismatch.is_none() {
                report.first_mismatch = Some(json);
            }
        }
    }
}

struct CaseCtx<'a> {
    rows: usize,
    variant: &'a str,
    norm: bool,
}

struct Mismatch {
    row: usize,
    pick: usize,
    ref_id: u32,
    got_id: u32,
    ref_w: u32,
    got_w: u32,
}

impl CaseCtx<'_> {
    fn mismatch_json(&self, comparison: &str, m: &Mismatch) -> String {
        format!(
            "{{\"comparison\":\"{comparison}\",\"rows\":{},\"n_exp\":{N_EXP},\"variant\":\"{}\",\
             \"norm\":{},\"row\":{},\"pick\":{},\"ref_id\":{},\"got_id\":{},\
             \"ref_id_hex\":\"{:#010x}\",\"got_id_hex\":\"{:#010x}\",\
             \"ref_weight_bits\":\"{:#010x}\",\"got_weight_bits\":\"{:#010x}\"}}",
            self.rows,
            self.variant,
            self.norm,
            m.row,
            m.pick,
            m.ref_id as i32,
            m.got_id as i32,
            m.ref_id,
            m.got_id,
            m.ref_w,
            m.got_w,
        )
    }
}

/// First raw-bit difference (ids or weights), `None` when byte-identical.
fn first_mismatch(
    ref_idx: &[f32],
    ref_w: &[f32],
    got_idx: &[f32],
    got_w: &[f32],
) -> Option<Mismatch> {
    if ref_idx.len() != got_idx.len() || ref_w.len() != got_w.len() || ref_idx.len() != ref_w.len()
    {
        return Some(Mismatch {
            row: usize::MAX,
            pick: usize::MAX,
            ref_id: ref_idx.len() as u32,
            got_id: got_idx.len() as u32,
            ref_w: ref_w.len() as u32,
            got_w: got_w.len() as u32,
        });
    }
    (0..ref_idx.len()).find_map(|i| {
        let differs = ref_idx[i].to_bits() != got_idx[i].to_bits()
            || ref_w[i].to_bits() != got_w[i].to_bits();
        differs.then(|| Mismatch {
            row: i / TOP_K,
            pick: i % TOP_K,
            ref_id: ref_idx[i].to_bits(),
            got_id: got_idx[i].to_bits(),
            ref_w: ref_w[i].to_bits(),
            got_w: got_w[i].to_bits(),
        })
    })
}

/// Times the three routes on the random input. Returns one JSON line.
fn run_timing(gpu: &mut Gpu, rows: usize, norm: bool) -> String {
    let logits = make_logits("random", rows);
    let numel = rows * N_EXP;
    let src = gpu.upload_f32(&logits, &[rows, N_EXP]).expect("upload src");
    let work = gpu
        .upload_f32(&logits, &[rows, N_EXP])
        .expect("upload work");
    let (idx, w) = upload_sentinel(gpu, rows);
    gpu.hip.device_synchronize().expect("sync uploads");

    let inc = measure(gpu, &src, &work, numel, &mut |gpu: &mut Gpu| {
        gpu.softmax_f32(&work).expect("timing incumbent softmax");
        gpu.moe_topk_renorm_k8_batched(&work, &idx, &w, N_EXP, norm, rows)
            .expect("timing incumbent top-k");
    });
    let fus = measure(gpu, &src, &work, numel, &mut |gpu: &mut Gpu| {
        gpu.moe_router_softmax_topk_k8_prefill(&work, &idx, &w, N_EXP, norm, rows, LANES)
            .expect("timing current fused router");
    });
    let new = measure(gpu, &src, &work, numel, &mut |gpu: &mut Gpu| {
        gpu.moe_router_softmax_topk_k8_multirow_prefill(&work, &idx, &w, N_EXP, norm, rows)
            .expect("timing multirow router");
    });
    for t in [src, work, idx, w] {
        gpu.free_tensor(t).expect("free timing tensor");
    }
    let (inc_m, inc_lo, inc_hi) = stats(inc);
    let (fus_m, fus_lo, fus_hi) = stats(fus);
    let (new_m, new_lo, new_hi) = stats(new);
    format!(
        "{{\"timing\":true,\"rows\":{rows},\"n_exp\":{N_EXP},\"norm\":{norm},\
         \"median_us\":{{\"incumbent_pair\":{inc_m:.3},\"current_fused\":{fus_m:.3},\"new\":{new_m:.3}}},\
         \"p10_us\":{{\"incumbent_pair\":{inc_lo:.3},\"current_fused\":{fus_lo:.3},\"new\":{new_lo:.3}}},\
         \"p90_us\":{{\"incumbent_pair\":{inc_hi:.3},\"current_fused\":{fus_hi:.3},\"new\":{new_hi:.3}}},\
         \"warmups\":{WARMUPS},\"reps\":{REPS}}}"
    )
}

/// WARMUPS + REPS reps of `run`; returns the REPS timed durations in
/// microseconds. Each rep restores the raw logits from `src` into `work` and
/// synchronizes outside the event-bracketed interval.
fn measure(
    gpu: &mut Gpu,
    src: &GpuTensor,
    work: &GpuTensor,
    numel: usize,
    run: &mut dyn FnMut(&mut Gpu),
) -> Vec<f64> {
    let start = gpu.hip.event_create().expect("create start event");
    let stop = gpu.hip.event_create().expect("create stop event");
    let mut samples = Vec::with_capacity(REPS);
    for rep in 0..WARMUPS + REPS {
        gpu.copy_f32_buffer(work, src, numel)
            .expect("restore raw logits");
        gpu.hip.device_synchronize().expect("sync restore");

        gpu.hip
            .event_record(&start, gpu.active_stream.as_ref())
            .expect("record start");
        run(gpu);
        gpu.hip
            .event_record(&stop, gpu.active_stream.as_ref())
            .expect("record stop");
        gpu.hip.event_synchronize(&stop).expect("sync stop");
        let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
        if rep >= WARMUPS {
            samples.push(f64::from(ms) * 1_000.0);
        }
    }
    gpu.hip.event_destroy(start).expect("destroy start event");
    gpu.hip.event_destroy(stop).expect("destroy stop event");
    samples
}

/// (median = mean of the two middle samples, p10, p90) of REPS samples.
fn stats(mut samples: Vec<f64>) -> (f64, f64, f64) {
    assert_eq!(samples.len(), REPS);
    samples.sort_by(f64::total_cmp);
    let median = (samples[REPS / 2 - 1] + samples[REPS / 2]) / 2.0;
    let p10 = samples[(0.1 * (REPS - 1) as f64).round() as usize];
    let p90 = samples[(0.9 * (REPS - 1) as f64).round() as usize];
    (median, p10, p90)
}

// ---------------------------------------------------------------------------
// Deterministic inputs
// ---------------------------------------------------------------------------

fn fnv1a(values: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

struct Rng(u64);

impl Rng {
    fn new(rows: usize, tag: usize) -> Self {
        Rng(0x9e37_79b9_7f4a_7c15 ^ ((rows as u64) << 40) ^ ((N_EXP as u64) << 20) ^ tag as u64)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    /// `count` distinct positions below `N_EXP`.
    fn positions(&mut self, count: usize) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::with_capacity(count);
        while out.len() < count.min(N_EXP) {
            let p = self.below(N_EXP);
            if !out.contains(&p) {
                out.push(p);
            }
        }
        out
    }
}

/// Row of random lows in [-6, -1) with the given positions set to the shared
/// maximum 7.0.
fn pair_row(positions: &[usize], seed: usize) -> Vec<f32> {
    let mut rng = Rng::new(2, 1000 + seed);
    let mut v: Vec<f32> = (0..N_EXP).map(|_| rng.range(-6.0, -1.0)).collect();
    for &p in positions {
        v[p] = 7.0;
    }
    v
}

fn make_logits(variant: &str, rows: usize) -> Vec<f32> {
    let tag = VARIANTS
        .iter()
        .position(|v| *v == variant)
        .unwrap_or(VARIANTS.len());
    let mut rng = Rng::new(rows, tag);
    let mut out = Vec::with_capacity(rows * N_EXP);
    for row in 0..rows {
        let kind = if variant == "mixed_rows" {
            MIXED_KINDS[row % MIXED_KINDS.len()]
        } else {
            variant
        };
        out.extend(make_row(kind, row, &mut rng));
    }
    out
}

fn make_row(kind: &str, row: usize, rng: &mut Rng) -> Vec<f32> {
    match kind {
        "allzero" => vec![0.0; N_EXP],
        "random" => (0..N_EXP).map(|_| rng.range(-12.0, 12.0)).collect(),
        "tie_heavy" => match row % 3 {
            // Seven coarse levels: ~2/7 of the experts tie at the maximum.
            0 => {
                const LEVELS: [f32; 7] = [3.0, 3.0, 2.0, 2.0, 1.0, 0.0, -1.0];
                (0..N_EXP)
                    .map(|_| LEVELS[rng.below(LEVELS.len())])
                    .collect()
            }
            // Two levels, ~25% at the top.
            1 => (0..N_EXP)
                .map(|_| if rng.below(4) == 0 { 1.0 } else { 0.0 })
                .collect(),
            // 9..=24 maxima over a flat floor: the k = 8 cut falls inside the tie.
            _ => {
                let mut v = vec![-2.0f32; N_EXP];
                let count = 9 + rng.below(16);
                for p in rng.positions(count) {
                    v[p] = 5.0;
                }
                v
            }
        },
        "tie_tree_cross" => {
            // Equal maxima at fixed virtual-tree positions (same lane across the
            // two virtual halves, adjacent, mirrored, wave-boundary and sibling
            // subtrees); later passes over the set list shift the positions.
            let set = TREE_SETS[row % TREE_SETS.len()];
            const SHIFTS: [usize; 5] = [0, 0, 1, 32, 127];
            let shift = SHIFTS[(row / TREE_SETS.len()) % SHIFTS.len()];
            let mut v: Vec<f32> = (0..N_EXP).map(|_| rng.range(-6.0, -1.0)).collect();
            for &p in set {
                v[(p + shift) % N_EXP] = 7.0;
            }
            v
        }
        "nonfinite" => nonfinite_row(row, rng),
        "flat_tail" => match row % 6 {
            // One spike over a flat floor: 255 equal probabilities, the k = 8
            // cut falls deep inside the tie (bit-reversed order).
            0 => {
                let mut v = vec![-3.0f32; N_EXP];
                v[rng.below(N_EXP)] = 5.0;
                v
            }
            // Exp underflow: every non-spike probability is exactly 0.0.
            1 => {
                let mut v = vec![-200.0f32; N_EXP];
                v[rng.below(N_EXP)] = 0.0;
                v
            }
            // Logits one or two ulps apart: distinct exps whose divided
            // probabilities may round equal.
            2 => (0..N_EXP)
                .map(|_| 1.0f32 + (rng.below(3) as f32) * f32::EPSILON)
                .collect(),
            // Every logit below the incumbent's -1e30 max seed, with duplicates.
            3 => (0..N_EXP)
                .map(|_| -1.0e30f32 * (2.0 + rng.below(3) as f32))
                .collect(),
            // All equal at a large magnitude.
            4 => vec![-65504.0f32; N_EXP],
            // Tiny magnitudes around zero (exp ~ 1, rounding-sensitive sums).
            _ => (0..N_EXP).map(|_| rng.range(-1.0e-6, 1.0e-6)).collect(),
        },
        other => panic!("unknown variant {other}"),
    }
}

/// Non-finite and extreme-range rows (the incumbent ignores NaN in the max,
/// never picks NaN, and clamps a round without a valid candidate to (0, 0.0f)).
fn nonfinite_row(row: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..N_EXP).map(|_| rng.range(-12.0, 12.0)).collect();
    match row % 11 {
        // One NaN logit: the row sum is NaN, every probability is NaN.
        0 => v[rng.below(N_EXP)] = f32::NAN,
        1 => v = vec![f32::NAN; N_EXP],
        // One +inf: x - max is inf - inf = NaN for it, -inf for the rest.
        2 => v[rng.below(N_EXP)] = f32::INFINITY,
        3 => {
            for p in rng.positions(128) {
                v[p] = f32::NEG_INFINITY;
            }
        }
        4 => v = vec![f32::NEG_INFINITY; N_EXP],
        // Below the -1e30 seed, with some -inf.
        5 => {
            v = (0..N_EXP)
                .map(|_| rng.range(-3.0e38, -1.5e30))
                .collect::<Vec<_>>();
            for p in rng.positions(8) {
                v[p] = f32::NEG_INFINITY;
            }
        }
        6 => {
            for p in rng.positions(3) {
                v[p] = f32::NAN;
            }
            for p in rng.positions(3) {
                v[p] = f32::INFINITY;
            }
            for p in rng.positions(3) {
                v[p] = f32::NEG_INFINITY;
            }
        }
        // Wide range: most exps underflow to exactly 0, picks tie at 0.0.
        7 => v = (0..N_EXP).map(|_| rng.range(-1.0e4, 1.0e4)).collect(),
        // Two +inf.
        8 => {
            for p in rng.positions(2) {
                v[p] = f32::INFINITY;
            }
        }
        // Subnormal / signed-zero logits.
        9 => {
            v = (0..N_EXP)
                .map(|i| match i % 4 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => 1.0e-40,
                    _ => -1.0e-40,
                })
                .collect()
        }
        // Huge positive logits (max subtraction keeps exp finite).
        _ => v = (0..N_EXP).map(|_| rng.range(3.0e38, 3.4e38)).collect(),
    }
    v
}
