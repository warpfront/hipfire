// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab gate: the expert-deduplicating routed kernels of the A3B lane-batched
//! decode MoE block (`gemv_mq4g256v2_moe_gate_up_k8_dedup_rows`,
//! `gemv_mq4g128v2_moe_down_dedup_rows` + `gemv_mq4g128v2_moe_down_combine_rows`,
//! `kernels/src/moe_dedup_mq4rs.gfx1201.hip`) against
//!
//!   * the REAL singleton launches (`gemv_mq4g256v2_moe_gate_up_k8_indexed`,
//!     `gemv_mq4g128v2_moe_ninepath_d4`) on each lane alone, and
//!   * the existing non-dedup lane twins (`gemv_..._rows`),
//!
//! at the production geometry (hidden 2048, 256 distinct experts, top-8,
//! routed intermediate 512), random MQ4G256V2 / MQ4G128V2 expert weights.
//! Routing is controlled directly (the top-k id table is an input): all lanes
//! on the same experts, the same set in rotated slot order, partial overlap,
//! disjoint, random over all 256, random over a small pool (heavy collisions in
//! every slot position), and out-of-range ids; N in {1, 2, 3, 5, 8} lanes.
//! Activations are random and adversarial (wide dynamic range, exact and
//! negative zeros, saturating silu inputs). Per lane `gate_batch`, `up_batch`
//! (and every guard row past the launched lanes) and the residual rows of
//! `x_batch` must be bit-identical to both references; the expanded scratch
//! must be fully written for every launched pair and untouched beyond. Also: a
//! negative control (one perturbed input bit changes that lane only; a swapped
//! reference mismatches), 200 back-to-back launches, and the chained
//! gate/up -> down path.
//!
//! `moe_dedup_timing_hw` times old vs new (gate/up; down + combine) at N = 8
//! for 46 / 64 / 8 distinct experts, warm and cold-cache (2 x 96 MB flush), with
//! hipEvents; run it with `--class timing`.
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --features lab --test moe_dedup \
//!       -- --ignored --nocapture --test-threads=1
#![cfg(feature = "lab")]

use rdna_compute::moe_decode_lanes::MoeDecodeLanesArgs;
use rdna_compute::{DType, Gpu, GpuTensor};

const HIDDEN: usize = 2048;
const N_EXP: usize = 256;
const K_TOP: usize = 8;
const MI: usize = 512;
const MAXL: usize = 8;
const POISON: u32 = 0x7FC0_DEAD;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32_unit(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF;
        f32::from_bits(0x3F80_0000 | a) - 1.0
    }
    fn f32_signed(&mut self) -> f32 {
        let v = self.f32_unit();
        if self.next_u32() & 1 == 1 {
            -v
        } else {
            v
        }
    }
    fn f16_normal(&mut self, lo: u32, hi: u32, signed: bool) -> u16 {
        let exp = lo + self.next_u32() % (hi - lo + 1);
        let mant = self.next_u32() & 0x3FF;
        let sign = if signed { self.next_u32() & 1 } else { 0 };
        ((sign << 15) | (exp << 10) | mant) as u16
    }
    fn below(&mut self, n: usize) -> usize {
        self.next_u32() as usize % n
    }
}

fn f32_bytes(d: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) }
}
fn i32_bytes(d: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) }
}
fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc u8");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod u8");
    t
}
fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    upload_u8(gpu, bytes)
}
fn poisoned(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc poison");
    poison(gpu, &t, n);
    t
}
fn poison(gpu: &mut Gpu, t: &GpuTensor, n: usize) {
    let v = vec![POISON; n];
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("poison");
}
fn put_f32(gpu: &mut Gpu, t: &GpuTensor, off: usize, data: &[f32]) {
    let view = t.sub_offset(off, data.len());
    gpu.hip
        .memcpy_htod(&view.buf, f32_bytes(data))
        .expect("put");
}
fn put_i32(gpu: &mut Gpu, t: &GpuTensor, off: usize, data: &[i32]) {
    let view = t.sub_offset(off, data.len());
    gpu.hip
        .memcpy_htod(&view.buf, i32_bytes(data))
        .expect("put");
}
fn dl(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

/// MQ4G256V2 (qt=44) rows: `groups` x 136 B.
fn synth_qt44(rows: usize, groups: usize, seed: u64, sc_lo: u32, sc_hi: u32) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = vec![0u8; rows * groups * 136];
    for r in 0..rows * groups {
        let off = r * 136;
        for half in 0..2 {
            let sc = rng.f16_normal(sc_lo, sc_hi, false);
            let zp = rng.f16_normal(sc_lo - 1, sc_hi, true);
            out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&sc.to_le_bytes());
            out[off + 4 * half + 2..off + 4 * half + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for w in 0..32 {
            let pk = rng.next_u32() ^ (rng.next_u32() << 16);
            out[off + 8 + 4 * w..off + 12 + 4 * w].copy_from_slice(&pk.to_le_bytes());
        }
    }
    out
}

/// MQ4G128V2 (qt=53) expert: `rows` x 4 groups of 68 B.
fn synth_qt53(rows: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = vec![0u8; rows * 4 * 68];
    for r in 0..rows * 4 {
        let off = r * 68;
        let sc = rng.f16_normal(9, 12, false);
        let zp = rng.f16_normal(8, 12, true);
        out[off..off + 2].copy_from_slice(&sc.to_le_bytes());
        out[off + 2..off + 4].copy_from_slice(&zp.to_le_bytes());
        for w in 0..16 {
            let pk = rng.next_u32() ^ (rng.next_u32() << 16);
            out[off + 4 + 4 * w..off + 8 + 4 * w].copy_from_slice(&pk.to_le_bytes());
        }
    }
    out
}

struct Weights {
    gu_ptrs: GpuTensor,
    down_ptrs: GpuTensor,
    sdown: GpuTensor,
    _keep: Vec<GpuTensor>,
}

/// 256 DISTINCT experts (no blob sharing), so a column read from the wrong
/// expert cannot hide behind equal weights.
fn build_weights(gpu: &mut Gpu, seed: u64) -> Weights {
    let mut keep = Vec::new();
    let mut gu_tab = Vec::new();
    let mut dn_tab = Vec::new();
    for e in 0..N_EXP {
        let gu = upload_u8(
            gpu,
            &synth_qt44(2 * MI, 8, seed.wrapping_mul(31) ^ (0x100 + e as u64), 7, 9),
        );
        gu_tab.push(gu.buf.as_ptr() as u64);
        keep.push(gu);
        let dn = upload_u8(
            gpu,
            &synth_qt53(HIDDEN, seed.wrapping_mul(37) ^ (0x2000 + e as u64)),
        );
        dn_tab.push(dn.buf.as_ptr() as u64);
        keep.push(dn);
    }
    let gu_ptrs = upload_u64(gpu, &gu_tab);
    let down_ptrs = upload_u64(gpu, &dn_tab);
    let sdown = upload_u8(gpu, &vec![0u8; HIDDEN * 2 * 136]);
    Weights {
        gu_ptrs,
        down_ptrs,
        sdown,
        _keep: keep,
    }
}

// ---------------------------------------------------------------------------
// Controlled routing
// ---------------------------------------------------------------------------

type Ids = Vec<[i32; K_TOP]>; // per lane

fn distinct_ids(rng: &mut Lcg, pool: &[i32], k: usize) -> [i32; K_TOP] {
    let mut out = [0i32; K_TOP];
    let mut taken: Vec<i32> = Vec::new();
    while taken.len() < k {
        let c = pool[rng.below(pool.len())];
        if !taken.contains(&c) {
            taken.push(c);
        }
    }
    out[..k].copy_from_slice(&taken);
    out
}

fn scenario_ids(kind: &str, n: usize, seed: u64) -> Ids {
    let mut rng = Lcg(0xDED0_0000 ^ seed.wrapping_mul(0x9E37_79B9));
    let all: Vec<i32> = (0..N_EXP as i32).collect();
    let base = distinct_ids(&mut rng, &all, K_TOP);
    (0..n)
        .map(|l| match kind {
            "same" => base,
            "rotated" => {
                let mut r = base;
                r.rotate_left(l % K_TOP);
                r
            }
            "partial" => {
                let mut r = base;
                let fresh = distinct_ids(&mut rng, &all, K_TOP);
                for i in 4..K_TOP {
                    if !r[..i].contains(&fresh[i]) && !base[..4].contains(&fresh[i]) {
                        r[i] = fresh[i];
                    }
                }
                // keep the lane's ids distinct
                let mut seen = Vec::new();
                for v in r.iter_mut() {
                    while seen.contains(v) {
                        *v = (*v + 1) % N_EXP as i32;
                    }
                    seen.push(*v);
                }
                r
            }
            "disjoint" => {
                let mut r = [0i32; K_TOP];
                for (i, v) in r.iter_mut().enumerate() {
                    *v = ((l * K_TOP + i) * 3 + 1) as i32 % N_EXP as i32;
                }
                r
            }
            "random256" => distinct_ids(&mut rng, &all, K_TOP),
            "small12" => {
                let pool: Vec<i32> = (0..12).map(|i| (i * 21 + 5) as i32).collect();
                distinct_ids(&mut rng, &pool, K_TOP)
            }
            "small20" => {
                let pool: Vec<i32> = (0..20).map(|i| (i * 13 + 2) as i32).collect();
                distinct_ids(&mut rng, &pool, K_TOP)
            }
            "slotshift" => {
                // lane l reads the same set but expert i sits in slot (i + l) % 8:
                // every expert collides across lanes in a DIFFERENT slot.
                let mut r = [0i32; K_TOP];
                for i in 0..K_TOP {
                    r[(i + l) % K_TOP] = base[i];
                }
                r
            }
            "invalid" => {
                // shared out-of-range ids (zero outputs in the singleton) mixed
                // with valid ones.
                let mut r = distinct_ids(&mut rng, &[3, 77, 200, 9, 140, 31], 6);
                r[6] = 300 + (l as i32 % 2);
                r[7] = -1;
                if l % 3 == 0 {
                    r[6] = 300;
                }
                r
            }
            other => panic!("unknown scenario {other}"),
        })
        .collect()
}

const KINDS: [&str; 10] = [
    "same",
    "rotated",
    "partial",
    "disjoint",
    "random256",
    "small12",
    "small20",
    "slotshift",
    "random256",
    "invalid",
];

fn distinct_count(ids: &Ids) -> usize {
    let mut s = std::collections::BTreeSet::new();
    for l in ids {
        s.extend(l.iter().copied());
    }
    s.len()
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct LaneIn {
    x_rot: Vec<f32>,
    x: Vec<f32>,
    gate: Vec<f32>, // [8][512] independent activations for the down path
    up: Vec<f32>,
    w: [f32; K_TOP],
}

fn make_lane(seed: u64, lane: usize) -> LaneIn {
    let mut rng = Lcg(0xA3B0_D0D0 ^ seed.wrapping_mul(0x9E37_79B9) ^ ((lane as u64) << 40));
    let mode = (seed as usize + lane) % 5;
    let mag = 0.25 + ((seed as usize + lane) % 5) as f32 * 0.75;
    let x_rot: Vec<f32> = (0..HIDDEN)
        .map(|i| match mode {
            // wide dynamic range
            1 => rng.f32_signed() * 10f32.powi((rng.below(9) as i32) - 4),
            // exact / negative zeros and sparse
            2 => match rng.below(6) {
                0 => 0.0,
                1 => -0.0,
                _ => rng.f32_signed() * mag,
            },
            // alternating large / tiny
            3 => {
                if i % 2 == 0 {
                    rng.f32_signed() * 1.0e4
                } else {
                    rng.f32_signed() * 1.0e-4
                }
            }
            _ => rng.f32_signed() * mag,
        })
        .collect();
    let gu_mag = [1.0f32, 6.0, 30.0, 0.05, 120.0][(seed as usize + lane * 2) % 5];
    let act = |rng: &mut Lcg, n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| match rng.below(12) {
                0 => 0.0,
                1 => -0.0,
                2 => 100.0 * if rng.below(2) == 0 { 1.0 } else { -1.0 }, // silu saturates
                3 => 1.0e3 * rng.f32_signed(),
                _ => rng.f32_signed() * gu_mag,
            })
            .collect()
    };
    let gate = act(&mut rng, K_TOP * MI);
    let up = act(&mut rng, K_TOP * MI);
    let mut w = [0f32; K_TOP];
    for (k, v) in w.iter_mut().enumerate() {
        *v = match (seed as usize + lane + k) % 7 {
            0 => 0.0,
            1 => 1.0,
            2 => 1.0e-6,
            3 => 40.0,
            _ => rng.f32_unit(),
        };
    }
    LaneIn {
        x_rot,
        x: (0..HIDDEN).map(|_| rng.f32_signed() * 3.0).collect(),
        gate,
        up,
        w,
    }
}

// ---------------------------------------------------------------------------
// Singleton arm
// ---------------------------------------------------------------------------

struct Single {
    idx: GpuTensor,
    w: GpuTensor,
    x_rot: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    x: GpuTensor,
}

fn alloc_single(gpu: &mut Gpu) -> Single {
    Single {
        idx: poisoned(gpu, K_TOP),
        w: poisoned(gpu, K_TOP),
        x_rot: poisoned(gpu, HIDDEN),
        gate: poisoned(gpu, K_TOP * MI),
        up: poisoned(gpu, K_TOP * MI),
        x: poisoned(gpu, HIDDEN),
    }
}

/// One lane's real singleton results.
#[derive(Clone, PartialEq)]
struct Ref {
    /// gate/up GEMV outputs for the lane's x_rot.
    gu_gate: Vec<u32>,
    gu_up: Vec<u32>,
    /// ninepath residual for the lane's independent gate/up input.
    down_x: Vec<u32>,
    /// ninepath residual chained after the singleton gate/up.
    chain_x: Vec<u32>,
}

fn run_singleton(gpu: &mut Gpu, w: &Weights, s: &Single, ids: &[i32; K_TOP], l: &LaneIn) -> Ref {
    put_i32(gpu, &s.idx, 0, ids);
    put_f32(gpu, &s.w, 0, &l.w);
    put_f32(gpu, &s.x_rot, 0, &l.x_rot);
    // gate/up
    poison(gpu, &s.gate, K_TOP * MI);
    poison(gpu, &s.up, K_TOP * MI);
    gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(
        &w.gu_ptrs,
        &s.idx,
        &s.x_rot,
        &s.gate,
        &s.up,
        2 * MI,
        HIDDEN,
    )
    .expect("singleton gate_up");
    gpu.hip.device_synchronize().expect("sync");
    let gu_gate = dl(gpu, &s.gate, K_TOP * MI);
    let gu_up = dl(gpu, &s.up, K_TOP * MI);
    // chained ninepath on the singleton gate/up just produced
    put_f32(gpu, &s.x, 0, &l.x);
    gpu.gemv_mq4g128v2_moe_ninepath_d4(
        &w.down_ptrs,
        &s.idx,
        &s.w,
        &s.gate,
        &s.up,
        &s.x,
        HIDDEN,
        512,
        N_EXP,
    )
    .expect("singleton ninepath chain");
    gpu.hip.device_synchronize().expect("sync");
    let chain_x = dl(gpu, &s.x, HIDDEN);
    // ninepath on the independent activations
    put_f32(gpu, &s.gate, 0, &l.gate);
    put_f32(gpu, &s.up, 0, &l.up);
    put_f32(gpu, &s.x, 0, &l.x);
    gpu.gemv_mq4g128v2_moe_ninepath_d4(
        &w.down_ptrs,
        &s.idx,
        &s.w,
        &s.gate,
        &s.up,
        &s.x,
        HIDDEN,
        512,
        N_EXP,
    )
    .expect("singleton ninepath");
    gpu.hip.device_synchronize().expect("sync");
    let down_x = dl(gpu, &s.x, HIDDEN);
    Ref {
        gu_gate,
        gu_up,
        down_x,
        chain_x,
    }
}

// ---------------------------------------------------------------------------
// Batched arm
// ---------------------------------------------------------------------------

struct Batch {
    router_logits: GpuTensor,
    scalar: GpuTensor,
    sgate: GpuTensor,
    sup: GpuTensor,
    srot: GpuTensor,
    idx: GpuTensor,
    w: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    x_rot: GpuTensor,
    x: GpuTensor,
    down_exp: GpuTensor,
}

fn alloc_batch(gpu: &mut Gpu) -> Batch {
    Batch {
        router_logits: poisoned(gpu, MAXL * N_EXP),
        scalar: poisoned(gpu, MAXL),
        sgate: poisoned(gpu, MAXL * 512),
        sup: poisoned(gpu, MAXL * 512),
        srot: poisoned(gpu, MAXL * 512),
        idx: poisoned(gpu, MAXL * K_TOP),
        w: poisoned(gpu, MAXL * K_TOP),
        gate: poisoned(gpu, MAXL * K_TOP * MI),
        up: poisoned(gpu, MAXL * K_TOP * MI),
        x_rot: poisoned(gpu, MAXL * HIDDEN),
        x: poisoned(gpu, MAXL * HIDDEN),
        down_exp: poisoned(gpu, MAXL * K_TOP * HIDDEN),
    }
}

fn args<'a>(w: &'a Weights, b: &'a Batch) -> MoeDecodeLanesArgs<'a> {
    MoeDecodeLanesArgs {
        router_logits: &b.router_logits,
        shared_scalar: &b.scalar,
        shared_gate: &b.sgate,
        shared_up: &b.sup,
        shared_rot: &b.srot,
        topk_idx: &b.idx,
        topk_w: &b.w,
        gate_batch: &b.gate,
        up_batch: &b.up,
        x_rot_batch: &b.x_rot,
        x_batch: &b.x,
        shared_down_w: &w.sdown,
        gate_up_ptrs: &w.gu_ptrs,
        down_ptrs: &w.down_ptrs,
        down_expanded: &b.down_exp,
        n_exp: N_EXP,
        norm_topk: true,
    }
}

fn load_routing(gpu: &mut Gpu, b: &Batch, ids: &Ids, lanes: &[LaneIn]) {
    poison(gpu, &b.idx, MAXL * K_TOP);
    poison(gpu, &b.w, MAXL * K_TOP);
    poison(gpu, &b.x_rot, MAXL * HIDDEN);
    for (r, (id, l)) in ids.iter().zip(lanes).enumerate() {
        put_i32(gpu, &b.idx, r * K_TOP, id);
        put_f32(gpu, &b.w, r * K_TOP, &l.w);
        put_f32(gpu, &b.x_rot, r * HIDDEN, &l.x_rot);
    }
}

fn load_down_inputs(gpu: &mut Gpu, b: &Batch, lanes: &[LaneIn]) {
    poison(gpu, &b.gate, MAXL * K_TOP * MI);
    poison(gpu, &b.up, MAXL * K_TOP * MI);
    poison(gpu, &b.x, MAXL * HIDDEN);
    poison(gpu, &b.down_exp, MAXL * K_TOP * HIDDEN);
    for (r, l) in lanes.iter().enumerate() {
        put_f32(gpu, &b.gate, r * K_TOP * MI, &l.gate);
        put_f32(gpu, &b.up, r * K_TOP * MI, &l.up);
        put_f32(gpu, &b.x, r * HIDDEN, &l.x);
    }
}

fn load_x_only(gpu: &mut Gpu, b: &Batch, lanes: &[LaneIn]) {
    poison(gpu, &b.x, MAXL * HIDDEN);
    for (r, l) in lanes.iter().enumerate() {
        put_f32(gpu, &b.x, r * HIDDEN, &l.x);
    }
}

fn cmp(tag: &str, what: &str, got: &[u32], want: &[u32]) -> usize {
    assert_eq!(got.len(), want.len());
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if let Some(&i) = bad.first() {
        eprintln!(
            "  [{tag}] MISMATCH {what}: {}/{} words differ, first @{i}: got {:#010x} want {:#010x}",
            bad.len(),
            got.len(),
            got[i],
            want[i]
        );
    }
    bad.len()
}

fn sync(gpu: &Gpu) {
    gpu.hip.device_synchronize().expect("sync");
}

/// Words of `d` past `lanes` rows of `len` that are not poison.
fn guard(tag: &str, what: &str, d: &[u32], lanes: usize, len: usize) -> usize {
    let n = d[lanes * len..].iter().filter(|&&w| w != POISON).count();
    if n != 0 {
        eprintln!("  [{tag}] GUARD {what}: {n} words past lane {lanes} were written");
    }
    n
}

/// Every dedup-vs-singleton-vs-twin comparison for one routing scenario.
fn check_scenario(
    gpu: &mut Gpu,
    w: &Weights,
    single: &Single,
    bat: &Batch,
    tag: &str,
    ids: &Ids,
    lanes: &[LaneIn],
    refs: &[Ref],
) -> usize {
    let n = ids.len();
    let mut bad = 0usize;
    load_routing(gpu, bat, ids, lanes);

    // ---- gate/up: twin vs dedup vs singleton ------------------------------
    poison(gpu, &bat.gate, MAXL * K_TOP * MI);
    poison(gpu, &bat.up, MAXL * K_TOP * MI);
    gpu.moe_routed_gate_up_lanes(&args(w, bat), n, false)
        .expect("twin gate_up");
    sync(gpu);
    let twin_g = dl(gpu, &bat.gate, MAXL * K_TOP * MI);
    let twin_u = dl(gpu, &bat.up, MAXL * K_TOP * MI);
    poison(gpu, &bat.gate, MAXL * K_TOP * MI);
    poison(gpu, &bat.up, MAXL * K_TOP * MI);
    gpu.moe_routed_gate_up_lanes(&args(w, bat), n, true)
        .expect("dedup gate_up");
    sync(gpu);
    let ded_g = dl(gpu, &bat.gate, MAXL * K_TOP * MI);
    let ded_u = dl(gpu, &bat.up, MAXL * K_TOP * MI);
    bad += cmp(tag, "gate_up dedup vs twin: gate_batch", &ded_g, &twin_g);
    bad += cmp(tag, "gate_up dedup vs twin: up_batch", &ded_u, &twin_u);
    for r in 0..n {
        let s = r * K_TOP * MI..(r + 1) * K_TOP * MI;
        bad += cmp(
            tag,
            &format!("gate_up lane {r} gate vs singleton"),
            &ded_g[s.clone()],
            &refs[r].gu_gate,
        );
        bad += cmp(
            tag,
            &format!("gate_up lane {r} up vs singleton"),
            &ded_u[s],
            &refs[r].gu_up,
        );
    }
    bad += guard(tag, "gate_batch", &ded_g, n, K_TOP * MI);
    bad += guard(tag, "up_batch", &ded_u, n, K_TOP * MI);
    assert!(
        refs.iter().all(|r| r.gu_gate.iter().all(|&v| v != POISON)),
        "singleton left poison in gate_up"
    );

    // ---- down: independent activations ------------------------------------
    let mut twin_x = Vec::new();
    for dedup in [false, true] {
        load_down_inputs(gpu, bat, lanes);
        gpu.moe_routed_down_lanes(&args(w, bat), n, dedup)
            .expect("down");
        sync(gpu);
        let x = dl(gpu, &bat.x, MAXL * HIDDEN);
        if dedup {
            bad += cmp(tag, "down dedup vs twin: x_batch", &x, &twin_x);
            for r in 0..n {
                bad += cmp(
                    tag,
                    &format!("down lane {r} x vs singleton"),
                    &x[r * HIDDEN..(r + 1) * HIDDEN],
                    &refs[r].down_x,
                );
            }
            // expanded scratch: every launched pair written, nothing beyond.
            let ex = dl(gpu, &bat.down_exp, MAXL * K_TOP * HIDDEN);
            let unwritten = ex[..n * K_TOP * HIDDEN]
                .iter()
                .filter(|&&v| v == POISON)
                .count();
            if unwritten != 0 {
                eprintln!(
                    "  [{tag}] expanded scratch: {unwritten} words of launched pairs unwritten"
                );
            }
            bad += unwritten;
            bad += guard(tag, "down_expanded", &ex, n * K_TOP, HIDDEN);
        } else {
            twin_x = x;
        }
        bad += guard(tag, "x_batch", &dl(gpu, &bat.x, MAXL * HIDDEN), n, HIDDEN);
    }

    // ---- chained gate/up -> down (dedup both) ------------------------------
    load_routing(gpu, bat, ids, lanes);
    poison(gpu, &bat.gate, MAXL * K_TOP * MI);
    poison(gpu, &bat.up, MAXL * K_TOP * MI);
    load_x_only(gpu, bat, lanes);
    gpu.moe_routed_gate_up_lanes(&args(w, bat), n, true)
        .expect("chain gate_up");
    gpu.moe_routed_down_lanes(&args(w, bat), n, true)
        .expect("chain down");
    sync(gpu);
    let x = dl(gpu, &bat.x, MAXL * HIDDEN);
    for r in 0..n {
        bad += cmp(
            tag,
            &format!("chain lane {r} x vs singleton"),
            &x[r * HIDDEN..(r + 1) * HIDDEN],
            &refs[r].chain_x,
        );
    }
    let _ = single;
    bad
}

fn require_gfx1201() -> Gpu {
    let gpu = Gpu::init().expect("GPU required");
    assert!(
        gpu.arch_caps.is_gfx1201(),
        "this gate requires an exact gfx1201 device (got {})",
        gpu.arch
    );
    gpu
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn moe_dedup_matches_singleton_and_twins_hw() {
    let mut gpu = require_gfx1201();
    gpu.ensure_mq_signs_128().expect("signs");
    let w = build_weights(&mut gpu, 5);
    let single = alloc_single(&mut gpu);
    let bat = alloc_batch(&mut gpu);
    let mut total_bad = 0usize;
    let mut cases = 0usize;
    let mut lanes_compared = 0usize;
    let mut sum_distinct = [0usize; 9];
    let mut sum_picks = [0usize; 9];

    for &n in &[1usize, 2, 3, 5, 8] {
        for (ki, kind) in KINDS.iter().enumerate() {
            for rep in 0..3u64 {
                let seed = (ki as u64) * 17 + rep * 5 + n as u64;
                let ids = scenario_ids(kind, n, seed);
                let lanes: Vec<LaneIn> = (0..n).map(|l| make_lane(seed, l)).collect();
                let refs: Vec<Ref> = (0..n)
                    .map(|l| run_singleton(&mut gpu, &w, &single, &ids[l], &lanes[l]))
                    .collect();
                // the comparison must move the residual (not vacuous)
                for (r, l) in refs.iter().zip(&lanes) {
                    let moved = r
                        .down_x
                        .iter()
                        .zip(&l.x)
                        .filter(|(a, b)| **a != b.to_bits())
                        .count();
                    assert!(moved > HIDDEN / 2, "singleton did not move the residual");
                }
                let tag = format!("n{n} {kind} rep{rep}");
                total_bad += check_scenario(&mut gpu, &w, &single, &bat, &tag, &ids, &lanes, &refs);
                cases += 1;
                lanes_compared += n;
                sum_distinct[n] += distinct_count(&ids);
                sum_picks[n] += n * K_TOP;
                if rep == 0 && n == 8 {
                    eprintln!("n8 {kind}: distinct experts = {}/64", distinct_count(&ids));
                }
            }
        }
    }
    for n in [1usize, 2, 3, 5, 8] {
        eprintln!(
            "n{n}: mean distinct experts per step {:.1} of {} picks",
            sum_distinct[n] as f64 / (sum_picks[n] / (n * K_TOP)) as f64,
            n * K_TOP
        );
    }
    eprintln!(
        "{cases} routing cases, {lanes_compared} lane compares; total mismatched words = {total_bad}"
    );

    // ---- negative controls --------------------------------------------------
    {
        let n = 3usize;
        let ids = scenario_ids("small12", n, 99);
        let lanes: Vec<LaneIn> = (0..n).map(|l| make_lane(99, l)).collect();
        let base_refs: Vec<Ref> = (0..n)
            .map(|l| run_singleton(&mut gpu, &w, &single, &ids[l], &lanes[l]))
            .collect();
        load_routing(&mut gpu, &bat, &ids, &lanes);
        poison(&mut gpu, &bat.gate, MAXL * K_TOP * MI);
        poison(&mut gpu, &bat.up, MAXL * K_TOP * MI);
        gpu.moe_routed_gate_up_lanes(&args(&w, &bat), n, true)
            .unwrap();
        sync(&gpu);
        let base_g = dl(&gpu, &bat.gate, MAXL * K_TOP * MI);

        // (a) one input bit of lane 1's activation: lane 1 changes and equals
        // the singleton on the perturbed input; lanes 0 and 2 do not change.
        let mut pert = lanes.clone();
        pert[1].x_rot[5] = f32::from_bits(pert[1].x_rot[5].to_bits() ^ 1);
        load_routing(&mut gpu, &bat, &ids, &pert);
        poison(&mut gpu, &bat.gate, MAXL * K_TOP * MI);
        poison(&mut gpu, &bat.up, MAXL * K_TOP * MI);
        gpu.moe_routed_gate_up_lanes(&args(&w, &bat), n, true)
            .unwrap();
        sync(&gpu);
        let got_g = dl(&gpu, &bat.gate, MAXL * K_TOP * MI);
        let want1 = run_singleton(&mut gpu, &w, &single, &ids[1], &pert[1]);
        let s1 = K_TOP * MI..2 * K_TOP * MI;
        assert_eq!(
            cmp(
                "neg-control",
                "perturbed lane 1",
                &got_g[s1.clone()],
                &want1.gu_gate
            ),
            0
        );
        let n_diff = (0..K_TOP * MI)
            .filter(|&i| got_g[K_TOP * MI + i] != base_g[K_TOP * MI + i])
            .count();
        eprintln!(
            "negative control (x_rot bit): lane 1 gate differs from baseline in {n_diff} words"
        );
        assert!(n_diff > 0, "a one-bit x_rot change left lane 1 unchanged");
        for r in [0usize, 2] {
            let s = r * K_TOP * MI..(r + 1) * K_TOP * MI;
            assert_eq!(
                cmp(
                    "neg-control bleed",
                    &format!("lane {r}"),
                    &got_g[s.clone()],
                    &base_g[s]
                ),
                0,
                "lane {r} changed when only lane 1 was perturbed"
            );
        }

        // (b) a deliberately wrong reference (lane 0's singleton result
        // compared with lane 1's dedup result) must mismatch: the comparison
        // has teeth.
        let wrong = cmp(
            "neg-control",
            "swapped reference (expected mismatch)",
            &base_g[K_TOP * MI..2 * K_TOP * MI],
            &base_refs[0].gu_gate,
        );
        assert!(
            wrong > 0,
            "swapped reference matched: comparison is vacuous"
        );

        // (c) a corrupted routing table (one id changed) changes the output
        // of exactly that pair's slice.
        let mut ids2 = ids.clone();
        ids2[2][3] = (ids2[2][3] + 100) % N_EXP as i32;
        load_routing(&mut gpu, &bat, &ids2, &lanes);
        poison(&mut gpu, &bat.gate, MAXL * K_TOP * MI);
        poison(&mut gpu, &bat.up, MAXL * K_TOP * MI);
        gpu.moe_routed_gate_up_lanes(&args(&w, &bat), n, true)
            .unwrap();
        sync(&gpu);
        let g2 = dl(&gpu, &bat.gate, MAXL * K_TOP * MI);
        let moved = (0..g2.len()).filter(|&i| g2[i] != base_g[i]).count();
        assert!(
            moved > 0 && moved <= MI + 8,
            "id change moved {moved} gate words (expected only that slot)"
        );
    }

    // ---- 200 back-to-back launches, every one compared ----------------------
    {
        let n = 8usize;
        let ids = scenario_ids("small20", n, 321);
        let lanes: Vec<LaneIn> = (0..n).map(|l| make_lane(321, l)).collect();
        let refs: Vec<Ref> = (0..n)
            .map(|l| run_singleton(&mut gpu, &w, &single, &ids[l], &lanes[l]))
            .collect();
        let mut bad = 0;
        for it in 0..200 {
            load_routing(&mut gpu, &bat, &ids, &lanes);
            poison(&mut gpu, &bat.gate, MAXL * K_TOP * MI);
            poison(&mut gpu, &bat.up, MAXL * K_TOP * MI);
            poison(&mut gpu, &bat.down_exp, MAXL * K_TOP * HIDDEN);
            load_x_only(&mut gpu, &bat, &lanes);
            // gate/up then down, no host sync between the launches
            gpu.moe_routed_gate_up_lanes(&args(&w, &bat), n, true)
                .unwrap();
            gpu.moe_routed_down_lanes(&args(&w, &bat), n, true).unwrap();
            sync(&gpu);
            let g = dl(&gpu, &bat.gate, MAXL * K_TOP * MI);
            let x = dl(&gpu, &bat.x, MAXL * HIDDEN);
            for r in 0..n {
                bad += cmp(
                    &format!("repeat {it}"),
                    &format!("lane {r} gate"),
                    &g[r * K_TOP * MI..(r + 1) * K_TOP * MI],
                    &refs[r].gu_gate,
                );
                bad += cmp(
                    &format!("repeat {it}"),
                    &format!("lane {r} chain x"),
                    &x[r * HIDDEN..(r + 1) * HIDDEN],
                    &refs[r].chain_x,
                );
            }
        }
        // 200 un-synchronised gate/up launches in a row, one readback
        load_routing(&mut gpu, &bat, &ids, &lanes);
        poison(&mut gpu, &bat.gate, MAXL * K_TOP * MI);
        poison(&mut gpu, &bat.up, MAXL * K_TOP * MI);
        for _ in 0..200 {
            gpu.moe_routed_gate_up_lanes(&args(&w, &bat), n, true)
                .unwrap();
        }
        sync(&gpu);
        let g = dl(&gpu, &bat.gate, MAXL * K_TOP * MI);
        for r in 0..n {
            bad += cmp(
                "repeat-nosync",
                &format!("lane {r} gate"),
                &g[r * K_TOP * MI..(r + 1) * K_TOP * MI],
                &refs[r].gu_gate,
            );
        }
        eprintln!("200 back-to-back 8-lane dedup launches: mismatched words = {bad}");
        total_bad += bad;
    }

    assert_eq!(
        total_bad, 0,
        "dedup kernels are not byte-identical to the singleton / the lane twins"
    );
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// 8 lanes whose 64 picks hit exactly `distinct` experts (every lane's 8 ids
/// distinct). `distinct` in 8..=64.
fn ids_with_distinct(distinct: usize, seed: u64) -> Ids {
    let all: Vec<i32> = (0..N_EXP as i32).collect();
    let mut rng = Lcg(0x71AE_0000 ^ seed);
    if distinct == 64 {
        let pool = all[..64].to_vec();
        return (0..8)
            .map(|l| {
                let mut r = [0i32; K_TOP];
                r.copy_from_slice(&pool[l * 8..l * 8 + 8]);
                r
            })
            .collect();
    }
    if distinct == 8 {
        let base = distinct_ids(&mut rng, &all, K_TOP);
        return (0..8)
            .map(|l| {
                let mut r = base;
                r.rotate_left(l);
                r
            })
            .collect();
    }
    // random picks over a pool sized so the expected distinct count is close,
    // retried until the count is exact.
    let pool_size = match distinct {
        46 => 80,
        d => d * 2,
    };
    for _ in 0..200_000 {
        let pool: Vec<i32> = (0..pool_size).map(|i| (i * 3 + 1) as i32 % 256).collect();
        let ids: Ids = (0..8)
            .map(|_| distinct_ids(&mut rng, &pool, K_TOP))
            .collect();
        if distinct_count(&ids) == distinct {
            return ids;
        }
    }
    panic!("no routing with {distinct} distinct experts found");
}

fn flush(gpu: &mut Gpu, bufs: &[GpuTensor; 2]) {
    for (i, b) in bufs.iter().enumerate() {
        gpu.hip
            .memset(&b.buf, 0x11 * (i as i32 + 1), b.buf.size())
            .expect("flush");
    }
}

fn stats(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[0])
}

#[test]
#[ignore = "needs an exact gfx1201 GPU; timing (run with --class timing)"]
fn moe_dedup_timing_hw() {
    let mut gpu = require_gfx1201();
    gpu.ensure_mq_signs_128().expect("signs");
    let w = build_weights(&mut gpu, 5);
    let bat = alloc_batch(&mut gpu);
    let flush_bufs = [
        gpu.alloc_tensor(&[96 << 20], DType::Raw).expect("flush a"),
        gpu.alloc_tensor(&[96 << 20], DType::Raw).expect("flush b"),
    ];
    let ev0 = gpu.hip.event_create().unwrap();
    let ev1 = gpu.hip.event_create().unwrap();
    let iters = 300usize;
    let n = 8usize;
    let lanes: Vec<LaneIn> = (0..n).map(|l| make_lane(7, l)).collect();
    eprintln!(
        "N=8 lanes, hipEvent timing per launch (us), warm = back-to-back, cold = 2x96 MB flush before each; {iters} iters"
    );
    eprintln!(
        "{:<26} {:<9} {:<5} {:>9} {:>9}",
        "case", "kernel", "cache", "median", "min"
    );
    for distinct in [46usize, 64, 8] {
        let ids = ids_with_distinct(distinct, 3);
        assert_eq!(distinct_count(&ids), distinct);
        load_routing(&mut gpu, &bat, &ids, &lanes);
        load_down_inputs(&mut gpu, &bat, &lanes);
        for (what, is_down) in [("gate_up", false), ("down", true)] {
            for dedup in [false, true] {
                let mut medians = Vec::new();
                for cold in [false, true] {
                    // warm-up
                    for _ in 0..10 {
                        if is_down {
                            gpu.moe_routed_down_lanes(&args(&w, &bat), n, dedup)
                                .unwrap();
                        } else {
                            gpu.moe_routed_gate_up_lanes(&args(&w, &bat), n, dedup)
                                .unwrap();
                        }
                    }
                    sync(&gpu);
                    let mut s = Vec::with_capacity(iters);
                    for _ in 0..iters {
                        if cold {
                            flush(&mut gpu, &flush_bufs);
                            sync(&gpu);
                        }
                        gpu.hip.event_record(&ev0, None).unwrap();
                        if is_down {
                            gpu.moe_routed_down_lanes(&args(&w, &bat), n, dedup)
                                .unwrap();
                        } else {
                            gpu.moe_routed_gate_up_lanes(&args(&w, &bat), n, dedup)
                                .unwrap();
                        }
                        gpu.hip.event_record(&ev1, None).unwrap();
                        gpu.hip.event_synchronize(&ev1).unwrap();
                        s.push(gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1000.0);
                    }
                    let (med, min) = stats(s);
                    medians.push(med);
                    eprintln!(
                        "{:<26} {:<9} {:<5} {:>9.1} {:>9.1}",
                        format!(
                            "{distinct}/64 distinct {}",
                            if dedup { "NEW" } else { "old" }
                        ),
                        what,
                        if cold { "cold" } else { "warm" },
                        med,
                        min
                    );
                }
                let _ = medians;
            }
        }
    }
    gpu.hip.event_destroy(ev0).unwrap();
    gpu.hip.event_destroy(ev1).unwrap();
}
