// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! SP (short-prompt prefill) per-layer stream/CPU split planner — **pure, no wiring**.
//!
//! For one host layer of an S1 prefill window, after the router has produced
//! `counts[e]` = routed tokens of expert `e` (512 experts), decide which experts'
//! weights are streamed H2D to the GPU (top-`k` by token count) and which are
//! computed on the CPU by the existing D2 F32 engine ("E1",
//! `cpu_exec::moe_cpu_host_slot_outputs`-class code). Idle experts (0 tokens) are
//! neither streamed nor computed.
//!
//! The model is the one in
//! `xtx-decode/screens/SP/MODEL.md` §2 / `sp_model.py:layer_time` (the MoE phase
//! of `T_layer`; the trunk term `A(N)` is a constant that does not move the argmin):
//!
//! ```text
//! S(k)   = k * stream_ms_per_expert
//! gpu(k) = slots_streamed * gpu_moe_ms_per_slot + gpu_moe_ms_per_expert * k
//! C(rest)= cpu_fixed_ms + (16 / cpu_workers) * sum_{e in rest} t_E1(c_e) / 1000
//! k == U (nothing on CPU):  moe = max(S, gpu) + 0.3 * sync_ms
//! else:                     moe = max(S + dy, dx + C + dy, gpu) + sync_ms
//! ```
//!
//! The S1 SP hook uses [`SplitCost::sp_hook_xtx`]: the same shape with `dx = 0`,
//! `dy = dy_ms + cpu_slots(k) * dy_ms_per_cpu_slot` (BF16 rows of the CPU slots only, plus the
//! inverse map) and the constraint `cpu_slots(k) <= cpu_slot_cap` (`k` leaving more CPU slots is
//! infeasible; `k = U` always is). Goldens: `screens/SP/sp_hook_gold.py`.
//!
//! `k*` is the exact argmin over `0..=U` (`U` = number of experts with tokens>0),
//! smallest `k` on ties. (The Python `best_k` scans a coarse k grid; this planner
//! scans every k with prefix/suffix sums, `O(512 log 512)`.)
//!
//! # Constants and sources
//!
//! | constant | value | source |
//! |---|---|---|
//! | expert bytes | 2,611,200 B (gate_up 1,740,800 + down 870,400) | MODEL.md §1.2; `sp_model.py:EXP_BYTES` |
//! | host link | 7.1 GB/s pinned H2D → **0.3678 ms/expert** | MODEL.md §1.2 [MEAS]; `S_EXP_MS` |
//! | E1 per-expert µs at c routed rows | c=1 28.81, 2 33.13, 4 46.61, 8 93.50, 16 191.63, 32 438.51, 64 1056.79; beyond 64: 16.51 µs/row | `SHALLOW/data/cpu_batch_bench_16t_rows8204.txt` [MEAS, hiptrx TR 9970X, 16 workers, U=256, real gather]; tail [INFERENCE]; `sp_model.py:E1_R/E1_US` |
//! | E1 hipx scale | ×1.10 for c ≥ 4, ×1.04 for c ∈ 1..3 | MODEL.md §1.3 / SHALLOW §6 (clock; D1 decode 30 vs 28.8 µs) |
//! | E1 interpolation | piecewise-linear in c between the table points, `t(0)=0` | `sp_model.py:e1_us` (`np.interp`) |
//! | CPU worker reference | table is the **16-worker** per-expert time; `cpu_workers` rescales the CPU sum by `16/cpu_workers` (experts run back-to-back on the whole pool, so the makespan is the sum — the model's formula, not LPT over single-worker tasks) | MODEL.md §2 `C(rest)` |
//! | CPU fixed region | 0.1 ms/layer | `FIXED_REGION_MS` (D1 RESULT ~27 µs/region + quantize, rounded up) |
//! | sync/round trip | 1.0 ms/layer (0.3 × that when nothing is on the CPU) | `SYNC_MS` (SHALLOW §3 conservative) |
//! | D2H `dx` / H2D `dy` | `rows * (10,240 + 80) B` / `rows * 10,240 B` at 7.1 GB/s (D2H rate assumed = H2D [INFERENCE]) | `sp_model.py:comm_ms` |
//! | GPU MoE | 30 ms/host layer at 8,204 rows → 3.656 µs/row/layer (10 slots) → 3.6564e-4 ms/slot, +0.003 ms/expert | MODEL.md §1.2; `layer_time` |
//!
//! # Determinism
//!
//! Experts are totally ordered by `(count desc, expert id asc)`; the order, `k*`
//! and the f64 sums depend only on `counts` and `SplitCost` (all sums run in that
//! fixed order), never on thread timing. Ties in `k` resolve to the smallest `k`.

/// Number of routed experts per layer (Qwen3.5-35B-A3B-class mq4-xts).
pub const NUM_EXPERTS: usize = 512;

/// Top-k routing width (slots per row); used only by the GPU MoE term's definition.
pub const ROUTE_TOPK: usize = 10;

/// Worker count the E1 table was measured with.
pub const E1_REF_WORKERS: u32 = 16;

/// E1 table: routed rows per expert.
const E1_R: [f64; 8] = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
/// E1 table: µs per expert (hiptrx, unscaled).
const E1_US: [f64; 8] = [0.0, 28.81, 33.13, 46.61, 93.50, 191.63, 438.51, 1056.79];
/// µs/row beyond c = 64 (1056.79 / 64).
const E1_TAIL_US_PER_ROW: f64 = 1056.79 / 64.0;
/// hiptrx → hipx clock factor for c ≥ 4.
const E1_SCALE: f64 = 1.10;
/// c ∈ 1..3 factor (D1 decode arm 30 µs vs 28.8 µs bench).
const E1_SCALE_C1: f64 = 1.04;

/// Bytes of one expert (gate_up + down).
pub const EXPERT_BYTES: f64 = 2_611_200.0;
/// Pinned H2D link rate.
pub const LINK_BPS: f64 = 7.1e9;
/// One F32 activation / partial-y row.
pub const ROW_BYTES: f64 = 2560.0 * 4.0;
/// Per-row route bytes (ids + weights) riding with the D2H.
pub const ROUTE_BYTES: f64 = ROUTE_TOPK as f64 * 8.0;

/// Hook transport: link time of one expert streamed as per-expert copies (ms) =
/// 0.3678 ms × 1.10 (j2853 k*-cell stream / S(k) median) [MEAS].
pub const HOOK_STREAM_MS_PER_EXPERT: f64 = 0.405;
/// Hook transport: effective link rate (B/s) the stream cost above corresponds to; the CPU
/// result upload shares the same link.
pub const HOOK_LINK_BPS: f64 = 6.45e9;
/// Largest window (rows) the SP hook engages on (HOOK-DESIGN §5.1).
pub const SP_MAX_ROWS: usize = 5_376;
/// Most CPU slots of one layer the SP hook can stage (`5 * SP_MAX_ROWS`, HOOK-DESIGN §6).
pub const HOOK_CPU_SLOT_CAP: u64 = 5 * SP_MAX_ROWS as u64;
/// BF16 bytes per element of a returned CPU row.
const BF16_BYTES: f64 = 2.0;
/// Inverse-map bytes per window row (10 slots × i32).
const INVERSE_BYTES_PER_ROW: f64 = ROUTE_TOPK as f64 * 4.0;

/// E1 (D2 F32 engine) CPU cost per expert in µs for `c` routed rows on hipx,
/// 16 workers. `t(0) = 0`; piecewise-linear; see module docs for sources.
pub fn e1_expert_us(c: u32) -> f64 {
    if c == 0 {
        return 0.0;
    }
    let x = c as f64;
    let t = if x > E1_R[7] {
        E1_US[7] + (x - E1_R[7]) * E1_TAIL_US_PER_ROW
    } else {
        // first point with E1_R[i] >= x (i >= 1 since x >= 1)
        let i = E1_R.partition_point(|&r| r < x);
        let (r0, r1, u0, u1) = (E1_R[i - 1], E1_R[i], E1_US[i - 1], E1_US[i]);
        u0 + (u1 - u0) * (x - r0) / (r1 - r0)
    };
    t * if c >= 4 { E1_SCALE } else { E1_SCALE_C1 }
}

/// Cost model inputs for [`plan_layer_split`] / [`plan_threshold`].
#[derive(Clone, Debug, PartialEq)]
pub struct SplitCost {
    /// Link time to stream one expert H2D (ms). Default 0.3678.
    pub stream_ms_per_expert: f64,
    /// Workers in the CPU pool; the E1 table is the 16-worker time and is scaled
    /// by `16 / cpu_workers`.
    pub cpu_workers: u32,
    /// Extra multiplier on the CPU time (sensitivity; 1.0 = model).
    pub cpu_scale: f64,
    /// Fixed per-layer CPU region cost (ms) when any expert runs on the CPU.
    pub cpu_fixed_ms: f64,
    /// D2H of the window's x rows + route (ms); 0 when `rows` unknown.
    pub dx_ms: f64,
    /// Fixed part of the H2D of the CPU result (ms), charged whenever any expert is on the CPU.
    pub dy_ms: f64,
    /// H2D of one CPU slot's result row (ms per slot), added to `dy_ms` for the slots the CPU
    /// computes. 0 for the per-row F32 partial-y model.
    pub dy_ms_per_cpu_slot: f64,
    /// Hard cap on CPU slots; a `k` leaving more is infeasible. `None` = unconstrained.
    pub cpu_slot_cap: Option<u64>,
    /// Per-layer round trip + event sync (ms).
    pub sync_ms: f64,
    /// Fraction of `sync_ms` charged when no expert runs on the CPU.
    pub sync_stream_only_frac: f64,
    /// GPU MoE kernel time per streamed slot (ms).
    pub gpu_moe_ms_per_slot: f64,
    /// GPU MoE per-streamed-expert launch overhead (ms).
    pub gpu_moe_ms_per_expert: f64,
}

impl SplitCost {
    /// Model defaults for a window of `rows` rows (sets `dx_ms`/`dy_ms`).
    pub fn e1_for_rows(rows: u32) -> Self {
        let n = rows as f64;
        Self {
            stream_ms_per_expert: EXPERT_BYTES / LINK_BPS * 1e3,
            cpu_workers: E1_REF_WORKERS,
            cpu_scale: 1.0,
            cpu_fixed_ms: 0.1,
            dx_ms: n * (ROW_BYTES + ROUTE_BYTES) / LINK_BPS * 1e3,
            dy_ms: n * ROW_BYTES / LINK_BPS * 1e3,
            dy_ms_per_cpu_slot: 0.0,
            cpu_slot_cap: None,
            sync_ms: 1.0,
            sync_stream_only_frac: 0.3,
            gpu_moe_ms_per_slot: 30.0 / 8204.0 / ROUTE_TOPK as f64,
            gpu_moe_ms_per_expert: 0.003,
        }
    }

    /// The S1 SP hook's transport (HOOK-DESIGN §5.2) for a window of `rows` rows, activations
    /// of width `hidden` and a `cpu_workers`-thread CPU pool:
    ///
    /// * `stream_ms_per_expert` = [`HOOK_STREAM_MS_PER_EXPERT`] (0.405 ms: per-expert copies);
    /// * `dx_ms` = 0 (the x D2H runs per tile under the trunk);
    /// * `dy` = `cpu_slots * hidden * 2 B` (BF16 rows of the CPU slots only) + `rows * 40 B`
    ///   (the inverse map), over [`HOOK_LINK_BPS`];
    /// * `cpu_slots(k) <= HOOK_CPU_SLOT_CAP`.
    ///
    /// The E1 table, sync and GPU terms are those of [`SplitCost::e1_for_rows`].
    pub fn sp_hook_xtx(rows: u32, hidden: usize, cpu_workers: u32) -> Self {
        Self {
            stream_ms_per_expert: HOOK_STREAM_MS_PER_EXPERT,
            cpu_workers,
            dx_ms: 0.0,
            dy_ms: rows as f64 * INVERSE_BYTES_PER_ROW / HOOK_LINK_BPS * 1e3,
            dy_ms_per_cpu_slot: hidden as f64 * BF16_BYTES / HOOK_LINK_BPS * 1e3,
            cpu_slot_cap: Some(HOOK_CPU_SLOT_CAP),
            ..Self::e1_for_rows(rows)
        }
    }

    fn cpu_factor(&self) -> f64 {
        self.cpu_scale * E1_REF_WORKERS as f64 / self.cpu_workers.max(1) as f64
    }
}

impl Default for SplitCost {
    /// Model defaults with `rows = 0` (no D2H/H2D row terms).
    fn default() -> Self {
        Self::e1_for_rows(0)
    }
}

/// Result of a split decision for one layer.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitPlan {
    /// Number of experts streamed (the `k` largest by `(count desc, id asc)`).
    pub k: usize,
    /// Streamed expert ids, sorted ascending.
    pub stream_ids: Vec<u16>,
    /// Placement mask over all experts: 1 = computed on the CPU, 0 = streamed
    /// (GPU) or idle (0 tokens).
    pub cpu_mask: [u8; NUM_EXPERTS],
    /// Experts with tokens > 0 (`U`).
    pub n_active: usize,
    /// Experts computed on the CPU (`U - k`).
    pub n_cpu: usize,
    /// Routed (expert, token) slots computed on the CPU: the sum of the CPU experts' counts.
    pub cpu_slots: u64,
    /// Predicted MoE-phase time of the layer (ms): `max(...) + sync`.
    pub predicted_ms: f64,
    /// Link term `k * stream_ms (+ dy)` (ms).
    pub stream_ms: f64,
    /// CPU term `dx + C + dy` (ms); 0 when nothing is on the CPU.
    pub cpu_ms: f64,
    /// GPU MoE term (ms).
    pub gpu_ms: f64,
}

/// Experts with tokens > 0, ordered `(count desc, id asc)`.
fn order_active(counts: &[u32]) -> Vec<u16> {
    assert_eq!(
        counts.len(),
        NUM_EXPERTS,
        "counts must have {NUM_EXPERTS} entries"
    );
    let mut ids: Vec<u16> = (0..NUM_EXPERTS as u16)
        .filter(|&e| counts[e as usize] > 0)
        .collect();
    ids.sort_unstable_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]).then(a.cmp(&b)));
    ids
}

struct Eval {
    predicted: f64,
    stream: f64,
    cpu: f64,
    gpu: f64,
}

/// Model time with the first `k` of `order` streamed. `slot_prefix[i]` = tokens of
/// `order[..i]`; `cpu_suffix_us[i]` = sum of `t_E1` over `order[i..]` (µs).
fn eval_k(
    k: usize,
    n_active: usize,
    slot_prefix: &[u64],
    cpu_suffix_us: &[f64],
    cost: &SplitCost,
) -> Eval {
    let s_ms = k as f64 * cost.stream_ms_per_expert;
    let gpu =
        slot_prefix[k] as f64 * cost.gpu_moe_ms_per_slot + cost.gpu_moe_ms_per_expert * k as f64;
    if k == n_active {
        let predicted = s_ms.max(gpu) + cost.sync_ms * cost.sync_stream_only_frac;
        return Eval {
            predicted,
            stream: s_ms,
            cpu: 0.0,
            gpu,
        };
    }
    let cpu_slots = (slot_prefix[n_active] - slot_prefix[k]) as f64;
    let dy_ms = cost.dy_ms + cpu_slots * cost.dy_ms_per_cpu_slot;
    let c_ms = cpu_suffix_us[k] / 1e3 * cost.cpu_factor() + cost.cpu_fixed_ms;
    let link = s_ms + dy_ms;
    let cpu = cost.dx_ms + c_ms + dy_ms;
    let predicted = link.max(cpu).max(gpu) + cost.sync_ms;
    Eval {
        predicted,
        stream: link,
        cpu,
        gpu,
    }
}

struct Prepared {
    order: Vec<u16>,
    slot_prefix: Vec<u64>,
    cpu_suffix_us: Vec<f64>,
}

impl Prepared {
    /// Routed slots left on the CPU when the first `k` of `order` are streamed.
    fn cpu_slots(&self, k: usize) -> u64 {
        self.slot_prefix[self.order.len()] - self.slot_prefix[k]
    }

    /// `k` satisfies the cost's CPU-slot cap (`k = U` always does: it leaves no CPU slot).
    fn feasible(&self, k: usize, cost: &SplitCost) -> bool {
        cost.cpu_slot_cap
            .map_or(true, |cap| self.cpu_slots(k) <= cap)
    }
}

fn prepare(counts: &[u32]) -> Prepared {
    let order = order_active(counts);
    let n = order.len();
    let mut slot_prefix = vec![0u64; n + 1];
    for i in 0..n {
        slot_prefix[i + 1] = slot_prefix[i] + counts[order[i] as usize] as u64;
    }
    let mut cpu_suffix_us = vec![0.0f64; n + 1];
    for i in (0..n).rev() {
        cpu_suffix_us[i] = cpu_suffix_us[i + 1] + e1_expert_us(counts[order[i] as usize]);
    }
    Prepared {
        order,
        slot_prefix,
        cpu_suffix_us,
    }
}

fn finish(p: &Prepared, k: usize, cost: &SplitCost) -> SplitPlan {
    let n = p.order.len();
    let e = eval_k(k, n, &p.slot_prefix, &p.cpu_suffix_us, cost);
    let mut stream_ids: Vec<u16> = p.order[..k].to_vec();
    stream_ids.sort_unstable();
    let mut cpu_mask = [0u8; NUM_EXPERTS];
    for &id in &p.order[k..] {
        cpu_mask[id as usize] = 1;
    }
    SplitPlan {
        k,
        stream_ids,
        cpu_mask,
        n_active: n,
        n_cpu: n - k,
        cpu_slots: p.cpu_slots(k),
        predicted_ms: e.predicted,
        stream_ms: e.stream,
        cpu_ms: e.cpu,
        gpu_ms: e.gpu,
    }
}

/// Optimal split: stream the `k*` largest experts, CPU the remaining active ones.
///
/// `k*` = exact argmin of the model MoE-phase time over `k in 0..=U`; ties → smallest `k`.
///
/// # Panics
/// If `counts.len() != 512`.
pub fn plan_layer_split(counts: &[u32], cost: &SplitCost) -> SplitPlan {
    let p = prepare(counts);
    let n = p.order.len();
    let mut best_k = n;
    let mut best_t = f64::INFINITY;
    for k in 0..=n {
        if !p.feasible(k, cost) {
            continue;
        }
        let t = eval_k(k, n, &p.slot_prefix, &p.cpu_suffix_us, cost).predicted;
        if t < best_t {
            best_t = t;
            best_k = k;
        }
    }
    finish(&p, best_k, cost)
}

/// Threshold rule: stream iff `count >= theta` (and `count > 0`), CPU the other
/// active experts; `predicted_ms` uses the same cost model as [`plan_layer_split`].
///
/// # Panics
/// If `counts.len() != 512`.
pub fn plan_threshold(counts: &[u32], theta: u32, cost: &SplitCost) -> SplitPlan {
    let p = prepare(counts);
    let theta = theta.max(1);
    let k = p.order.partition_point(|&e| counts[e as usize] >= theta);
    finish(&p, k, cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// d32k window rows [0,1024), host layer 24, per-expert routed tokens
    /// (`sp_model.load_prefill_last` + `counts_window`). Σ = 10,240.
    const D32K_N1024_L24: [u32; 512] = [
        0, 0, 2, 3, 94, 0, 13, 0, 6, 11, 0, 36, 2, 9, 14, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 18, 25,
        11, 1, 33, 16, 19, 72, 8, 92, 207, 26, 1, 0, 66, 102, 5, 0, 5, 0, 5, 0, 0, 10, 103, 12, 9,
        23, 22, 155, 0, 7, 0, 7, 34, 1, 0, 5, 7, 124, 31, 14, 24, 1, 2, 0, 0, 14, 0, 0, 2, 1, 2,
        36, 3, 9, 158, 2, 0, 0, 11, 3, 88, 65, 7, 0, 9, 8, 0, 45, 19, 0, 15, 146, 0, 0, 0, 4, 54,
        26, 0, 0, 52, 54, 30, 39, 38, 5, 11, 47, 4, 1, 0, 0, 0, 0, 0, 0, 0, 2, 52, 19, 0, 0, 1, 11,
        0, 0, 19, 20, 0, 0, 28, 0, 6, 0, 3, 0, 0, 0, 3, 1, 0, 3, 9, 4, 1, 1, 112, 66, 1, 0, 6, 0,
        6, 71, 15, 426, 3, 8, 0, 0, 0, 18, 1, 30, 81, 17, 7, 26, 13, 9, 1, 51, 0, 20, 10, 12, 24,
        0, 165, 13, 25, 0, 27, 7, 0, 0, 0, 0, 0, 15, 31, 57, 0, 38, 61, 6, 7, 32, 138, 18, 1, 0,
        21, 1, 0, 7, 6, 44, 5, 114, 85, 2, 0, 3, 0, 5, 17, 5, 0, 0, 0, 1, 5, 6, 9, 27, 0, 0, 33,
        65, 0, 22, 0, 5, 12, 2, 5, 2, 1, 0, 0, 24, 1, 5, 1, 6, 21, 0, 0, 0, 20, 0, 6, 0, 3, 1, 0,
        0, 0, 2, 98, 9, 8, 0, 38, 0, 4, 12, 0, 80, 0, 3, 31, 0, 1, 14, 61, 1, 0, 38, 6, 0, 19, 0,
        3, 0, 40, 10, 2, 0, 3, 0, 0, 19, 0, 0, 41, 240, 6, 15, 25, 3, 0, 7, 1, 6, 3, 61, 3, 1, 2,
        5, 21, 1, 0, 0, 0, 9, 0, 38, 6, 1, 68, 89, 1, 1, 9, 40, 0, 5, 11, 3, 0, 19, 240, 0, 0, 9,
        0, 11, 41, 3, 11, 3, 0, 35, 141, 41, 116, 11, 1, 0, 88, 0, 0, 0, 0, 22, 1, 90, 0, 45, 11,
        2, 0, 182, 0, 3, 0, 73, 112, 0, 0, 143, 0, 54, 0, 0, 2, 4, 0, 0, 4, 0, 1, 0, 2, 59, 0, 0,
        0, 106, 0, 0, 15, 0, 0, 7, 0, 2, 0, 19, 0, 0, 2, 0, 0, 0, 0, 1, 0, 14, 6, 2, 2, 0, 0, 11,
        8, 26, 0, 0, 0, 2, 17, 51, 0, 0, 23, 2, 1, 28, 86, 0, 59, 0, 0, 0, 151, 0, 17, 0, 0, 0, 1,
        0, 9, 18, 5, 106, 1, 4, 0, 0, 4, 24, 0, 11, 57, 2, 3, 0, 15, 41, 0, 0, 0, 4, 0, 20, 0, 1,
        0, 0, 6, 0, 1, 0, 17, 2, 510, 40, 14, 0, 188, 0, 0, 0, 22, 1, 61, 1, 72, 0, 0, 46, 1, 13,
        0, 172, 281, 42, 0, 0,
    ];

    /// d32k window rows [0,4096), host layer 24. Σ = 40,960.
    const D32K_N4096_L24: [u32; 512] = [
        11, 0, 7, 25, 305, 0, 62, 0, 28, 38, 7, 65, 31, 29, 63, 0, 0, 0, 0, 37, 0, 11, 26, 0, 10,
        2, 90, 62, 69, 46, 198, 59, 104, 204, 27, 305, 577, 55, 4, 0, 184, 387, 23, 0, 98, 0, 6, 0,
        0, 189, 514, 31, 35, 182, 53, 622, 0, 106, 0, 50, 280, 33, 23, 20, 22, 316, 58, 118, 140,
        10, 26, 0, 1, 215, 0, 0, 18, 9, 41, 185, 6, 42, 603, 5, 0, 0, 71, 21, 455, 255, 49, 0, 61,
        98, 1, 98, 93, 0, 43, 344, 0, 0, 0, 9, 181, 103, 0, 0, 185, 198, 157, 164, 277, 17, 24,
        140, 61, 4, 2, 5, 1, 1, 0, 9, 0, 7, 193, 205, 3, 0, 8, 60, 0, 10, 75, 51, 4, 0, 214, 0,
        136, 0, 7, 1, 17, 1, 73, 19, 15, 18, 70, 4, 2, 6, 440, 119, 22, 0, 12, 9, 42, 278, 40,
        1614, 9, 17, 15, 3, 0, 107, 8, 147, 284, 55, 25, 32, 49, 52, 7, 154, 0, 43, 33, 47, 189, 2,
        633, 59, 189, 0, 94, 14, 0, 0, 5, 0, 1, 39, 121, 151, 14, 85, 201, 11, 30, 184, 366, 52, 4,
        1, 38, 34, 0, 63, 9, 185, 35, 497, 259, 3, 54, 28, 1, 37, 92, 13, 0, 0, 3, 6, 32, 24, 224,
        215, 0, 0, 176, 246, 0, 91, 0, 66, 125, 10, 5, 3, 1, 0, 0, 108, 1, 94, 65, 18, 114, 0, 0,
        5, 72, 4, 22, 2, 3, 2, 1, 0, 11, 8, 421, 75, 89, 11, 95, 2, 7, 141, 0, 234, 1, 3, 82, 10,
        18, 42, 166, 15, 0, 107, 14, 0, 141, 0, 46, 2, 108, 74, 13, 0, 24, 0, 0, 130, 3, 0, 221,
        796, 22, 74, 131, 37, 1, 17, 25, 19, 22, 133, 3, 49, 5, 10, 54, 1, 14, 2, 3, 10, 1, 90, 96,
        27, 280, 516, 12, 1, 45, 130, 2, 68, 32, 11, 0, 49, 614, 0, 1, 60, 15, 134, 88, 5, 30, 11,
        0, 160, 502, 107, 364, 88, 31, 0, 273, 0, 0, 24, 0, 96, 23, 439, 0, 189, 19, 2, 0, 517, 1,
        16, 22, 218, 387, 0, 1, 530, 0, 241, 1, 0, 12, 26, 0, 0, 28, 0, 5, 10, 9, 118, 5, 0, 11,
        286, 28, 3, 96, 0, 1, 87, 0, 6, 0, 41, 0, 5, 8, 0, 0, 1, 0, 6, 4, 115, 73, 2, 2, 0, 32, 29,
        17, 181, 0, 3, 0, 27, 147, 131, 15, 1, 158, 14, 19, 37, 280, 5, 229, 4, 0, 0, 603, 0, 139,
        0, 0, 23, 7, 0, 51, 126, 13, 476, 18, 9, 0, 9, 11, 85, 3, 157, 236, 2, 30, 13, 99, 118, 0,
        3, 1, 14, 0, 23, 0, 12, 0, 0, 8, 0, 17, 10, 88, 32, 1487, 466, 21, 0, 643, 8, 1, 0, 44, 19,
        134, 60, 251, 0, 1, 201, 5, 85, 0, 737, 930, 73, 2, 0,
    ];

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn e1_table_points_and_scale() {
        close(e1_expert_us(0), 0.0);
        close(e1_expert_us(1), 28.81 * 1.04);
        close(e1_expert_us(2), 33.13 * 1.04);
        close(e1_expert_us(3), (46.61 + 33.13) / 2.0 * 1.04);
        close(e1_expert_us(4), 46.61 * 1.10);
        close(e1_expert_us(64), 1056.79 * 1.10);
        close(e1_expert_us(128), (1056.79 + 64.0 * 1056.79 / 64.0) * 1.10);
        close(e1_expert_us(12), (93.50 + 191.63) / 2.0 * 1.10);
    }

    #[test]
    fn default_stream_cost_is_0_3678() {
        let c = SplitCost::default();
        assert!((c.stream_ms_per_expert - 0.3678).abs() < 5e-5);
    }

    #[test]
    fn golden_matches_model_full_scan() {
        // Reference: sp_model.layer_time(e1_us) scanned over EVERY k (numpy, this
        // fixture); MoE phase = layer_time - trunk_layer_ms(n).
        // N=1024: k*=87, moe 34.4733 ms, cpu term (C) 30.3800, link S 31.9964.
        let cost = SplitCost::e1_for_rows(1024);
        let p = plan_layer_split(&D32K_N1024_L24, &cost);
        assert_eq!(p.k, 87);
        assert_eq!(p.n_active, 325);
        assert_eq!(p.n_cpu, 325 - 87);
        close(p.predicted_ms, 34.473261971830986);
        // plan.cpu_ms includes dx+dy; model's C(rest) is the printed 30.38.
        close(p.cpu_ms - cost.dx_ms - cost.dy_ms, 30.380033450000006);
        close(p.stream_ms - cost.dy_ms, 31.996394366197183);
        // N=4096: k*=170, moe 69.6788, C 56.8178, S 62.5217.
        let cost = SplitCost::e1_for_rows(4096);
        let p = plan_layer_split(&D32K_N4096_L24, &cost);
        assert_eq!(p.k, 170);
        close(p.predicted_ms, 69.67884820774648);
        close(p.cpu_ms - cost.dx_ms - cost.dy_ms, 56.81775525);
        close(p.stream_ms - cost.dy_ms, 62.521690140845074);
    }

    #[test]
    fn golden_beats_all_cpu_and_all_stream() {
        // Model refs (moe phase): N=1024 k=0 → 178.9798 ms, k=U → 119.8268 ms.
        let cost = SplitCost::e1_for_rows(1024);
        let p = plan_layer_split(&D32K_N1024_L24, &cost);
        let all_cpu = plan_threshold(&D32K_N1024_L24, u32::MAX, &cost);
        let all_str = plan_threshold(&D32K_N1024_L24, 1, &cost);
        assert_eq!(all_cpu.k, 0);
        assert_eq!(all_str.k, 325);
        close(all_cpu.predicted_ms, 178.97982504881162);
        close(all_str.predicted_ms, 119.82676056338028);
        assert!(p.predicted_ms < all_cpu.predicted_ms && p.predicted_ms < all_str.predicted_ms);
    }

    #[test]
    fn placement_is_consistent() {
        let cost = SplitCost::e1_for_rows(1024);
        let c = &D32K_N1024_L24;
        let p = plan_layer_split(c, &cost);
        assert_eq!(p.stream_ids.len(), p.k);
        assert!(p.stream_ids.windows(2).all(|w| w[0] < w[1]));
        let mut cpu = 0;
        for e in 0..NUM_EXPERTS {
            let streamed = p.stream_ids.binary_search(&(e as u16)).is_ok();
            if c[e] == 0 {
                assert!(!streamed && p.cpu_mask[e] == 0, "idle expert {e} placed");
            } else {
                assert_ne!(
                    streamed,
                    p.cpu_mask[e] == 1,
                    "expert {e} must be on exactly one side"
                );
            }
            cpu += p.cpu_mask[e] as usize;
        }
        assert_eq!(cpu, p.n_cpu);
        // every streamed expert has count >= every CPU expert's count
        let min_stream = p.stream_ids.iter().map(|&e| c[e as usize]).min().unwrap();
        let max_cpu = (0..NUM_EXPERTS)
            .filter(|&e| p.cpu_mask[e] == 1)
            .map(|e| c[e])
            .max()
            .unwrap();
        assert!(min_stream >= max_cpu);
    }

    #[test]
    fn determinism_under_ties() {
        // All active experts have the same count: the stream set must be the
        // lowest ids (id asc tiebreak), and repeated calls are identical.
        let mut counts = [0u32; NUM_EXPERTS];
        for e in (0..NUM_EXPERTS).step_by(3) {
            counts[e] = 40;
        }
        let cost = SplitCost::e1_for_rows(1024);
        let a = plan_layer_split(&counts, &cost);
        let b = plan_layer_split(&counts, &cost);
        assert_eq!(a, b);
        let expect: Vec<u16> = (0..NUM_EXPERTS as u16).step_by(3).take(a.k).collect();
        assert_eq!(a.stream_ids, expect);
        // threshold variant: tie group is all-in or all-out
        assert_eq!(
            plan_threshold(&counts, 40, &cost).k,
            counts.iter().filter(|&&c| c == 40).count()
        );
        assert_eq!(plan_threshold(&counts, 41, &cost).k, 0);
        // permuting equal-count ids must permute the ids, never the k
        let mut shifted = [0u32; NUM_EXPERTS];
        for e in (1..NUM_EXPERTS).step_by(3) {
            shifted[e] = 40;
        }
        let s = plan_layer_split(&shifted, &cost);
        assert_eq!(s.k, a.k);
        assert_eq!(
            s.stream_ids,
            (1..NUM_EXPERTS as u16)
                .step_by(3)
                .take(a.k)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn edge_all_idle() {
        let p = plan_layer_split(&[0u32; NUM_EXPERTS], &SplitCost::default());
        assert_eq!((p.k, p.n_active, p.n_cpu), (0, 0, 0));
        assert!(p.stream_ids.is_empty() && p.cpu_mask.iter().all(|&m| m == 0));
    }

    #[test]
    fn edge_k_zero_when_link_is_slow() {
        // Few small experts + very slow link: CPU-all (k = 0).
        let mut counts = [0u32; NUM_EXPERTS];
        for e in 0..20 {
            counts[e] = 3 + e as u32;
        }
        let mut cost = SplitCost::e1_for_rows(64);
        cost.stream_ms_per_expert = 50.0;
        let p = plan_layer_split(&counts, &cost);
        assert_eq!(p.k, 0);
        assert!(p.stream_ids.is_empty());
        assert_eq!(p.n_cpu, 20);
        assert_eq!(p.cpu_mask.iter().map(|&m| m as usize).sum::<usize>(), 20);
    }

    #[test]
    fn edge_k_equals_u_when_link_is_free() {
        // Free link, slow CPU: stream every active expert, CPU nothing.
        let cost = {
            let mut c = SplitCost::e1_for_rows(1024);
            c.stream_ms_per_expert = 0.0;
            c
        };
        let p = plan_layer_split(&D32K_N1024_L24, &cost);
        assert_eq!(p.k, p.n_active);
        assert_eq!(p.n_cpu, 0);
        assert!(p.cpu_mask.iter().all(|&m| m == 0));
        assert_eq!(p.stream_ids.len(), 325);
        assert_eq!(p.cpu_ms, 0.0);
    }

    #[test]
    fn k_is_monotone_in_stream_cost() {
        // Cheaper link never streams fewer experts; predicted time never falls
        // as the link gets slower.
        let mut prev_k = usize::MAX;
        let mut prev_ms = 0.0;
        for i in 0..40 {
            let mut cost = SplitCost::e1_for_rows(1024);
            cost.stream_ms_per_expert = 0.02 * (i as f64 + 1.0); // 0.02 .. 0.8 ms
            let p = plan_layer_split(&D32K_N1024_L24, &cost);
            assert!(
                p.k <= prev_k,
                "k rose {prev_k} -> {} at stream_ms {}",
                p.k,
                cost.stream_ms_per_expert
            );
            assert!(p.predicted_ms >= prev_ms - 1e-9);
            prev_k = p.k;
            prev_ms = p.predicted_ms;
        }
        // and extremes of the sweep differ
        let mut lo = SplitCost::e1_for_rows(1024);
        lo.stream_ms_per_expert = 0.02;
        let mut hi = SplitCost::e1_for_rows(1024);
        hi.stream_ms_per_expert = 0.8;
        assert!(
            plan_layer_split(&D32K_N1024_L24, &lo).k > plan_layer_split(&D32K_N1024_L24, &hi).k
        );
    }

    #[test]
    fn more_cpu_workers_streams_fewer() {
        let mut a = SplitCost::e1_for_rows(1024);
        a.cpu_workers = 8;
        let mut b = SplitCost::e1_for_rows(1024);
        b.cpu_workers = 32;
        assert!(plan_layer_split(&D32K_N1024_L24, &a).k >= plan_layer_split(&D32K_N1024_L24, &b).k);
    }

    #[test]
    fn optimal_is_never_worse_than_any_threshold() {
        let cost = SplitCost::e1_for_rows(1024);
        let opt = plan_layer_split(&D32K_N1024_L24, &cost);
        for theta in [1, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 1 << 20] {
            let t = plan_threshold(&D32K_N1024_L24, theta, &cost);
            assert!(opt.predicted_ms <= t.predicted_ms + 1e-12, "theta {theta}");
        }
        // MODEL.md §5: E1 best global θ at N=1K is ~32 (θ grid 8..512); within a few % of opt.
        let t32 = plan_threshold(&D32K_N1024_L24, 32, &cost);
        assert!(t32.predicted_ms / opt.predicted_ms < 1.15);
    }

    // ---- SplitCost::sp_hook_xtx (HOOK-DESIGN §5.2). Goldens: screens/SP/sp_hook_gold.py
    // (full scan over every k, numpy `sp_model.e1_us`, same fixtures).

    #[test]
    fn hook_cost_terms() {
        let c = SplitCost::sp_hook_xtx(1024, 2560, 16);
        assert_eq!(c.stream_ms_per_expert, 0.405);
        assert_eq!(c.dx_ms, 0.0);
        assert_eq!(c.cpu_slot_cap, Some(26_880));
        assert_eq!(c.cpu_workers, 16);
        // inverse map: 1024 * 40 B; per CPU slot: 2560 * 2 B; both at 6.45 GB/s.
        close(c.dy_ms, 1024.0 * 40.0 / 6.45e9 * 1e3);
        close(c.dy_ms_per_cpu_slot, 5120.0 / 6.45e9 * 1e3);
        // the other constructors are unchanged: no per-slot term, no cap.
        let e = SplitCost::e1_for_rows(1024);
        assert_eq!((e.dy_ms_per_cpu_slot, e.cpu_slot_cap), (0.0, None));
        assert_eq!(SplitCost::default().cpu_slot_cap, None);
    }

    #[test]
    fn hook_golden_matches_python_full_scan() {
        let cost = SplitCost::sp_hook_xtx(1024, 2560, 16);
        let p = plan_layer_split(&D32K_N1024_L24, &cost);
        assert_eq!((p.k, p.cpu_slots), (82, 2315));
        close(p.predicted_ms, 36.05399379844961);
        close(p.stream_ms, 35.05399379844961);
        close(p.cpu_ms, 34.78460587344963);
        close(p.gpu_ms, 3.1439765967820574);
        let cost = SplitCost::sp_hook_xtx(4096, 2560, 16);
        let p = plan_layer_split(&D32K_N4096_L24, &cost);
        assert_eq!((p.k, p.cpu_slots), (162, 4202));
        close(p.predicted_ms, 69.97094263565891);
        close(p.stream_ms, 68.97094263565891);
        close(p.cpu_ms, 68.09698051065894);
        close(p.gpu_ms, 13.927491955143832);
        // 8 workers: the CPU is slower, so more experts stream.
        let p = plan_layer_split(&D32K_N1024_L24, &SplitCost::sp_hook_xtx(1024, 2560, 8));
        assert_eq!((p.k, p.cpu_slots), (109, 1564));
        close(p.predicted_ms, 46.3928511627907);
    }

    #[test]
    fn hook_slot_cap_is_a_constraint() {
        // Cap 1500 / 2000 bind (the unconstrained optima leave 2315 / 4202 CPU slots).
        let mut cost = SplitCost::sp_hook_xtx(1024, 2560, 16);
        cost.cpu_slot_cap = Some(1500);
        let p = plan_layer_split(&D32K_N1024_L24, &cost);
        assert_eq!((p.k, p.cpu_slots), (112, 1498));
        close(p.predicted_ms, 47.555460465116276);
        let mut cost = SplitCost::sp_hook_xtx(4096, 2560, 16);
        cost.cpu_slot_cap = Some(2000);
        let p = plan_layer_split(&D32K_N4096_L24, &cost);
        assert_eq!((p.k, p.cpu_slots), (214, 1995));
        close(p.predicted_ms, 89.27902945736435);
        // A cap of 0 forces k = U (no CPU slot is feasible).
        cost.cpu_slot_cap = Some(0);
        let p = plan_layer_split(&D32K_N4096_L24, &cost);
        assert_eq!((p.k, p.n_cpu, p.cpu_slots), (p.n_active, 0, 0));
    }

    #[test]
    fn hook_cpu_slots_field_matches_the_mask() {
        for cap in [None, Some(1000), Some(26_880)] {
            let mut cost = SplitCost::sp_hook_xtx(1024, 2560, 16);
            cost.cpu_slot_cap = cap;
            let p = plan_layer_split(&D32K_N1024_L24, &cost);
            let from_mask: u64 = (0..NUM_EXPERTS)
                .filter(|&e| p.cpu_mask[e] == 1)
                .map(|e| D32K_N1024_L24[e] as u64)
                .sum();
            assert_eq!(p.cpu_slots, from_mask);
            if let Some(cap) = cap {
                assert!(p.cpu_slots <= cap);
            }
        }
    }
}
