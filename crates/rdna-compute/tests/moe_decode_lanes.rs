// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab gate: the lane-batched A3B decode MoE block (`Gpu::moe_decode_lanes_mq4rs`
//! and the batched projections feeding it) against the REAL singleton launches
//! of the one-row decode, at the production geometry (hidden 2048, 256 experts,
//! top-8, moe/shared intermediate 512), random MQ4G256V2 / MQ4G128V2 weights in
//! the real layouts.
//!
//! Test 1 `moe_decode_lanes_twins_match_singleton_launches_hw` (twins only):
//! the batched projection outputs are synthesised directly (random and
//! adversarial: duplicated logits, all-equal, two-level, sparse logits whose
//! probabilities underflow to exactly 0, exp-overflow magnitudes, ramps, lanes
//! sharing / partly sharing experts), N in {1,2,3,8} lanes with different data.
//! Per lane every buffer one MoE block writes (router top-k ids/weights, shared
//! silu+rotate, routed gate/up, the residual-stream row) must equal the
//! singleton `moe_router_softmax_topk_k8_wave64`, `fused_silu_mul_rotate_mq`,
//! `gemv_mq4g256v2_residual_sigmoid_scaled_k512`, `gemv_mq4g256v2_moe_gate_up_k8_indexed`
//! and `gemv_mq4g128v2_moe_ninepath_d4` launches on that lane alone. Also:
//! rows beyond `lanes` stay untouched, a negative control (one perturbed input
//! bit must change that lane and only that lane), 200 back-to-back launches.
//!
//! Test 2 `moe_chain_pm_xbatch_and_batched_norm_match_singleton_hw` (projections):
//! real chain from pre-norm lane activations: `fused_rmsnorm_rotate_mq_batched`
//! (the launch `hipfire_runtime::llama::fused_rmsnorm_rotate_mq_batched_for`
//! makes for a non-AWQ weight) per row equals the singleton
//! `fused_rmsnorm_rotate_mq`; the four `pm_xbatch::plain` projections (router
//! m=256, selector m=1, shared gate m=512, shared up m=512, K=2048) per lane
//! equal the singleton `fused_qkvza_mq4g256v2` (1281-row) launch and the norm-fold
//! launch; and the whole chain through the lane entry equals the per-lane
//! singleton chain, with the router rows duplicated / equal so ties sit inside
//! the top-8.
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --features lab --test moe_decode_lanes \
//!       -- --ignored --nocapture --test-threads=1
#![cfg(feature = "lab")]

use rdna_compute::moe_decode_lanes::MoeDecodeLanesArgs;
use rdna_compute::norm_fold::{norm_fold_grid, NormFoldArgs};
use rdna_compute::{DType, Gpu, GpuTensor};

const HIDDEN: usize = 2048;
const N_EXP: usize = 256;
const K_TOP: usize = 8;
const MI: usize = 512;
const SMI: usize = 512;
const MAXL: usize = 8;
const POOL: usize = 64; // distinct expert blobs, shared round-robin by the 256 table slots
const EPS: f32 = 1e-6;
const POISON: u32 = 0x7FC0_DEAD;
const ROW256: usize = 8 * 136;

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
}

fn f32_bytes(d: &[f32]) -> &[u8] {
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
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip
        .memcpy_htod(&t.buf, f32_bytes(data))
        .expect("htod f32");
    t
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
/// Write `data` into `t` at element offset `off`.
fn put_f32(gpu: &mut Gpu, t: &GpuTensor, off: usize, data: &[f32]) {
    let view = t.sub_offset(off, data.len());
    gpu.hip
        .memcpy_htod(&view.buf, f32_bytes(data))
        .expect("put");
}
fn dl(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

/// MQ4G256V2 (qt=44) rows: `groups` x 136 B
/// `[sc0:f16][zp0:f16][sc1:f16][zp1:f16][128 nibble bytes]`.
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

/// MQ4G128V2 (qt=53) expert: `rows` x 4 groups of 68 B
/// `[scale:f16][zero:f16][64 payload bytes]`.
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Router {
    Random,
    /// Router rows 2i+1 == 2i: every logit appears twice, so exact ties sit
    /// inside the top-8 of every lane.
    DupPairs,
    /// Every router row identical: all 256 logits tie.
    AllEqual,
    /// Four distinct row values, 64 copies each.
    FourLevel,
}

fn router_bytes(kind: Router, seed: u64) -> Vec<u8> {
    let base = synth_qt44(N_EXP, 8, seed ^ 0x0A, 6, 8);
    let mut out = Vec::with_capacity(base.len());
    match kind {
        Router::Random => return base,
        Router::DupPairs => {
            for r in 0..N_EXP {
                let src = (r / 2) * 2;
                out.extend_from_slice(&base[src * ROW256..(src + 1) * ROW256]);
            }
        }
        Router::AllEqual => {
            for _ in 0..N_EXP {
                out.extend_from_slice(&base[..ROW256]);
            }
        }
        Router::FourLevel => {
            for r in 0..N_EXP {
                let src = r % 4;
                out.extend_from_slice(&base[src * ROW256..(src + 1) * ROW256]);
            }
        }
    }
    out
}

struct Weights {
    router: GpuTensor,
    selector: GpuTensor,
    sgate: GpuTensor,
    sup: GpuTensor,
    sdown: GpuTensor,
    gu_ptrs: GpuTensor,
    down_ptrs: GpuTensor,
    norm_w: GpuTensor,
    _keep: Vec<GpuTensor>,
}

fn build_weights(gpu: &mut Gpu, seed: u64, kind: Router) -> Weights {
    let router = upload_u8(gpu, &router_bytes(kind, seed));
    let selector = upload_u8(gpu, &synth_qt44(1, 8, seed ^ 0x0B, 8, 10));
    let sgate = upload_u8(gpu, &synth_qt44(SMI, 8, seed ^ 0x0C, 7, 9));
    let sup = upload_u8(gpu, &synth_qt44(SMI, 8, seed ^ 0x0D, 7, 9));
    let sdown = upload_u8(gpu, &synth_qt44(HIDDEN, 2, seed ^ 0x0E, 8, 11));
    let mut keep = Vec::new();
    let mut gu_pool = Vec::new();
    let mut dn_pool = Vec::new();
    for p in 0..POOL {
        let gu = upload_u8(
            gpu,
            &synth_qt44(2 * MI, 8, seed.wrapping_mul(31) ^ (0x100 + p as u64), 7, 9),
        );
        gu_pool.push(gu.buf.as_ptr() as u64);
        keep.push(gu);
        let dn = upload_u8(
            gpu,
            &synth_qt53(HIDDEN, seed.wrapping_mul(37) ^ (0x200 + p as u64)),
        );
        dn_pool.push(dn.buf.as_ptr() as u64);
        keep.push(dn);
    }
    let gu_tab: Vec<u64> = (0..N_EXP).map(|e| gu_pool[e % POOL]).collect();
    let dn_tab: Vec<u64> = (0..N_EXP).map(|e| dn_pool[(e * 7 + 3) % POOL]).collect();
    let gu_ptrs = upload_u64(gpu, &gu_tab);
    let down_ptrs = upload_u64(gpu, &dn_tab);
    let mut rng = Lcg(seed ^ 0x77);
    let nw: Vec<f32> = (0..HIDDEN).map(|_| 0.5 + rng.f32_unit()).collect();
    let norm_w = upload_f32(gpu, &nw);
    Weights {
        router,
        selector,
        sgate,
        sup,
        sdown,
        gu_ptrs,
        down_ptrs,
        norm_w,
        _keep: keep,
    }
}

/// Inputs of one lane's MoE block after the projections.
#[derive(Clone)]
struct LaneIn {
    logits: Vec<f32>,
    scalar: f32,
    sgate: Vec<f32>,
    sup: Vec<f32>,
    x_rot: Vec<f32>,
    x: Vec<f32>,
}

/// Every buffer one lane's MoE block reads or writes, as raw words.
#[derive(Clone, PartialEq)]
struct LaneOut {
    x_rot: Vec<u32>,
    router_logits: Vec<u32>,
    scalar: Vec<u32>,
    sgate: Vec<u32>,
    sup: Vec<u32>,
    shared_rot: Vec<u32>,
    topk_idx: Vec<u32>,
    topk_w: Vec<u32>,
    gate: Vec<u32>,
    up: Vec<u32>,
    x: Vec<u32>,
}

impl LaneOut {
    fn fields(&self) -> [(&'static str, &Vec<u32>); 11] {
        [
            ("x_rot", &self.x_rot),
            ("router_logits", &self.router_logits),
            ("scalar", &self.scalar),
            ("shared_gate", &self.sgate),
            ("shared_up", &self.sup),
            ("shared_rot", &self.shared_rot),
            ("topk_idx", &self.topk_idx),
            ("topk_w", &self.topk_w),
            ("gate_batch", &self.gate),
            ("up_batch", &self.up),
            ("x", &self.x),
        ]
    }
}

/// Differing words between two lane results, printing each differing field.
fn diff(tag: &str, got: &LaneOut, want: &LaneOut) -> usize {
    let mut total = 0;
    for ((name, g), (_, w)) in got.fields().into_iter().zip(want.fields()) {
        let bad: Vec<usize> = (0..g.len()).filter(|&i| g[i] != w[i]).collect();
        if !bad.is_empty() {
            total += bad.len();
            let i = bad[0];
            eprintln!(
                "  [{tag}] MISMATCH {name}: {}/{} words differ, first @{i}: got {:#010x} want {:#010x}",
                bad.len(),
                g.len(),
                g[i],
                w[i]
            );
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Singleton arm: the real one-row launches
// ---------------------------------------------------------------------------

struct Single {
    x_rot: GpuTensor,
    router_logits: GpuTensor,
    scalar: GpuTensor,
    sgate: GpuTensor,
    sup: GpuTensor,
    shared_rot: GpuTensor,
    topk_idx: GpuTensor,
    topk_w: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    x: GpuTensor,
}

fn alloc_single(gpu: &mut Gpu) -> Single {
    Single {
        x_rot: poisoned(gpu, HIDDEN),
        router_logits: poisoned(gpu, N_EXP),
        scalar: poisoned(gpu, 4),
        sgate: poisoned(gpu, SMI),
        sup: poisoned(gpu, SMI),
        shared_rot: poisoned(gpu, SMI),
        topk_idx: poisoned(gpu, K_TOP),
        topk_w: poisoned(gpu, K_TOP),
        gate: poisoned(gpu, K_TOP * MI),
        up: poisoned(gpu, K_TOP * MI),
        x: poisoned(gpu, HIDDEN),
    }
}

fn poison_single(gpu: &mut Gpu, s: &Single) {
    poison(gpu, &s.x_rot, HIDDEN);
    poison(gpu, &s.router_logits, N_EXP);
    poison(gpu, &s.scalar, 4);
    poison(gpu, &s.sgate, SMI);
    poison(gpu, &s.sup, SMI);
    poison(gpu, &s.shared_rot, SMI);
    poison(gpu, &s.topk_idx, K_TOP);
    poison(gpu, &s.topk_w, K_TOP);
    poison(gpu, &s.gate, K_TOP * MI);
    poison(gpu, &s.up, K_TOP * MI);
    poison(gpu, &s.x, HIDDEN);
}

fn download_single(gpu: &mut Gpu, s: &Single) -> LaneOut {
    gpu.hip.device_synchronize().expect("sync");
    LaneOut {
        x_rot: dl(gpu, &s.x_rot, HIDDEN),
        router_logits: dl(gpu, &s.router_logits, N_EXP),
        scalar: dl(gpu, &s.scalar, 1),
        sgate: dl(gpu, &s.sgate, SMI),
        sup: dl(gpu, &s.sup, SMI),
        shared_rot: dl(gpu, &s.shared_rot, SMI),
        topk_idx: dl(gpu, &s.topk_idx, K_TOP),
        topk_w: dl(gpu, &s.topk_w, K_TOP),
        gate: dl(gpu, &s.gate, K_TOP * MI),
        up: dl(gpu, &s.up, K_TOP * MI),
        x: dl(gpu, &s.x, HIDDEN),
    }
}

/// The five singleton launches after the projections (router, silu, shared
/// down, gate/up, nine-path), exactly as the one-row decode issues them.
fn run_five(gpu: &mut Gpu, w: &Weights, s: &Single) {
    gpu.moe_router_softmax_topk_k8_wave64(&s.router_logits, &s.topk_idx, &s.topk_w, N_EXP, true)
        .expect("router");
    gpu.fused_silu_mul_rotate_mq(&s.sgate, &s.sup, &s.shared_rot, SMI)
        .expect("silu");
    gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512(
        &w.sdown,
        &s.shared_rot,
        &s.x,
        &s.scalar,
        HIDDEN,
        512,
    )
    .expect("shared down");
    gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(
        &w.gu_ptrs,
        &s.topk_idx,
        &s.x_rot,
        &s.gate,
        &s.up,
        2 * MI,
        HIDDEN,
    )
    .expect("gate_up");
    gpu.gemv_mq4g128v2_moe_ninepath_d4(
        &w.down_ptrs,
        &s.topk_idx,
        &s.topk_w,
        &s.gate,
        &s.up,
        &s.x,
        HIDDEN,
        512,
        N_EXP,
    )
    .expect("ninepath");
}

/// Singleton arm from synthesised projection outputs.
fn ref_from_projections(gpu: &mut Gpu, w: &Weights, s: &Single, inp: &LaneIn) -> LaneOut {
    poison_single(gpu, s);
    put_f32(gpu, &s.x_rot, 0, &inp.x_rot);
    put_f32(gpu, &s.router_logits, 0, &inp.logits);
    put_f32(gpu, &s.scalar, 0, &[inp.scalar]);
    put_f32(gpu, &s.sgate, 0, &inp.sgate);
    put_f32(gpu, &s.sup, 0, &inp.sup);
    put_f32(gpu, &s.x, 0, &inp.x);
    run_five(gpu, w, s);
    download_single(gpu, s)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Proj {
    /// `fused_rmsnorm_rotate_mq` then the 1281-row `fused_qkvza_mq4g256v2`.
    Unfolded,
    /// `fused_rmsnorm_proj4_mq4g256v2_fold` (the production norm-fold launch).
    Fold,
}

/// Singleton arm from the pre-norm residual `x`: norm + projections + five.
fn ref_from_chain(gpu: &mut Gpu, w: &Weights, s: &Single, x: &[f32], proj: Proj) -> LaneOut {
    poison_single(gpu, s);
    put_f32(gpu, &s.x, 0, x);
    match proj {
        Proj::Unfolded => {
            gpu.fused_rmsnorm_rotate_mq(&s.x, &w.norm_w, &s.x_rot, HIDDEN, EPS)
                .expect("norm");
            gpu.fused_qkvza_hfq4g256_mq4v2(
                &w.router,
                &w.selector,
                &w.sgate,
                &w.sup,
                &s.x_rot,
                &s.router_logits,
                &s.scalar,
                &s.sgate,
                &s.sup,
                N_EXP,
                1,
                SMI,
                SMI,
                HIDDEN,
            )
            .expect("qkvza");
        }
        Proj::Fold => {
            let args = NormFoldArgs {
                x: &s.x,
                norm_weight: &w.norm_w,
                x_rot: &s.x_rot,
                weights: [&w.router, &w.selector, &w.sgate, &w.sup],
                outs: [&s.router_logits, &s.scalar, &s.sgate, &s.sup],
                m: [N_EXP, 1, SMI, SMI],
                eps: EPS,
            };
            gpu.fused_rmsnorm_proj4_mq4g256v2_fold(&args, norm_fold_grid(1281, 3))
                .expect("fold");
        }
    }
    run_five(gpu, w, s);
    download_single(gpu, s)
}

// ---------------------------------------------------------------------------
// Batched arm
// ---------------------------------------------------------------------------

struct Batch {
    x_rot: GpuTensor,
    router_logits: GpuTensor,
    scalar: GpuTensor,
    sgate: GpuTensor,
    sup: GpuTensor,
    shared_rot: GpuTensor,
    topk_idx: GpuTensor,
    topk_w: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    x: GpuTensor,
    down_exp: GpuTensor,
}

fn alloc_batch(gpu: &mut Gpu) -> Batch {
    Batch {
        x_rot: poisoned(gpu, MAXL * HIDDEN),
        router_logits: poisoned(gpu, MAXL * N_EXP),
        scalar: poisoned(gpu, MAXL),
        sgate: poisoned(gpu, MAXL * SMI),
        sup: poisoned(gpu, MAXL * SMI),
        shared_rot: poisoned(gpu, MAXL * SMI),
        topk_idx: poisoned(gpu, MAXL * K_TOP),
        topk_w: poisoned(gpu, MAXL * K_TOP),
        gate: poisoned(gpu, MAXL * K_TOP * MI),
        up: poisoned(gpu, MAXL * K_TOP * MI),
        x: poisoned(gpu, MAXL * HIDDEN),
        down_exp: poisoned(gpu, MAXL * K_TOP * HIDDEN),
    }
}

fn poison_batch(gpu: &mut Gpu, b: &Batch) {
    poison(gpu, &b.x_rot, MAXL * HIDDEN);
    poison(gpu, &b.router_logits, MAXL * N_EXP);
    poison(gpu, &b.scalar, MAXL);
    poison(gpu, &b.sgate, MAXL * SMI);
    poison(gpu, &b.sup, MAXL * SMI);
    poison(gpu, &b.shared_rot, MAXL * SMI);
    poison(gpu, &b.topk_idx, MAXL * K_TOP);
    poison(gpu, &b.topk_w, MAXL * K_TOP);
    poison(gpu, &b.gate, MAXL * K_TOP * MI);
    poison(gpu, &b.up, MAXL * K_TOP * MI);
    poison(gpu, &b.x, MAXL * HIDDEN);
}

fn load_projections(gpu: &mut Gpu, b: &Batch, lanes: &[LaneIn]) {
    for (r, l) in lanes.iter().enumerate() {
        put_f32(gpu, &b.x_rot, r * HIDDEN, &l.x_rot);
        put_f32(gpu, &b.router_logits, r * N_EXP, &l.logits);
        put_f32(gpu, &b.scalar, r, &[l.scalar]);
        put_f32(gpu, &b.sgate, r * SMI, &l.sgate);
        put_f32(gpu, &b.sup, r * SMI, &l.sup);
        put_f32(gpu, &b.x, r * HIDDEN, &l.x);
    }
}

fn run_entry(gpu: &mut Gpu, w: &Weights, b: &Batch, lanes: usize) {
    let args = MoeDecodeLanesArgs {
        router_logits: &b.router_logits,
        shared_scalar: &b.scalar,
        shared_gate: &b.sgate,
        shared_up: &b.sup,
        shared_rot: &b.shared_rot,
        topk_idx: &b.topk_idx,
        topk_w: &b.topk_w,
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
    };
    gpu.moe_decode_lanes_mq4rs(&args, lanes)
        .expect("lane entry");
}

/// Whole 8-lane buffers as raw words.
struct Dump {
    x_rot: Vec<u32>,
    router_logits: Vec<u32>,
    scalar: Vec<u32>,
    sgate: Vec<u32>,
    sup: Vec<u32>,
    shared_rot: Vec<u32>,
    topk_idx: Vec<u32>,
    topk_w: Vec<u32>,
    gate: Vec<u32>,
    up: Vec<u32>,
    x: Vec<u32>,
}

fn download_batch(gpu: &mut Gpu, b: &Batch) -> Dump {
    gpu.hip.device_synchronize().expect("sync");
    Dump {
        x_rot: dl(gpu, &b.x_rot, MAXL * HIDDEN),
        router_logits: dl(gpu, &b.router_logits, MAXL * N_EXP),
        scalar: dl(gpu, &b.scalar, MAXL),
        sgate: dl(gpu, &b.sgate, MAXL * SMI),
        sup: dl(gpu, &b.sup, MAXL * SMI),
        shared_rot: dl(gpu, &b.shared_rot, MAXL * SMI),
        topk_idx: dl(gpu, &b.topk_idx, MAXL * K_TOP),
        topk_w: dl(gpu, &b.topk_w, MAXL * K_TOP),
        gate: dl(gpu, &b.gate, MAXL * K_TOP * MI),
        up: dl(gpu, &b.up, MAXL * K_TOP * MI),
        x: dl(gpu, &b.x, MAXL * HIDDEN),
    }
}

fn lane_out(d: &Dump, r: usize) -> LaneOut {
    let s = |v: &Vec<u32>, len: usize| v[r * len..(r + 1) * len].to_vec();
    LaneOut {
        x_rot: s(&d.x_rot, HIDDEN),
        router_logits: s(&d.router_logits, N_EXP),
        scalar: s(&d.scalar, 1),
        sgate: s(&d.sgate, SMI),
        sup: s(&d.sup, SMI),
        shared_rot: s(&d.shared_rot, SMI),
        topk_idx: s(&d.topk_idx, K_TOP),
        topk_w: s(&d.topk_w, K_TOP),
        gate: s(&d.gate, K_TOP * MI),
        up: s(&d.up, K_TOP * MI),
        x: s(&d.x, HIDDEN),
    }
}

/// Rows `>= lanes` of every buffer must still hold their pre-launch poison.
fn guard_rows_untouched(tag: &str, d: &Dump, lanes: usize) -> usize {
    let mut bad = 0;
    for (name, v, len) in [
        ("shared_rot", &d.shared_rot, SMI),
        ("topk_idx", &d.topk_idx, K_TOP),
        ("topk_w", &d.topk_w, K_TOP),
        ("gate_batch", &d.gate, K_TOP * MI),
        ("up_batch", &d.up, K_TOP * MI),
        ("x", &d.x, HIDDEN),
    ] {
        let n = v[lanes * len..].iter().filter(|&&w| w != POISON).count();
        if n != 0 {
            eprintln!("  [{tag}] GUARD {name}: {n} words past lane {lanes} were written");
            bad += n;
        }
    }
    bad
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Logits {
    Random,
    AllEqual,
    TwoLevel,
    DupMax,
    Sparse,
    BigMagnitude,
    Descending,
    Ascending,
    /// Lane 0's logits with the 3 best experts replaced (partial overlap).
    PartialOverlap,
    /// Exactly lane 0's logits (full overlap, different activations).
    FullOverlap,
}
const KINDS: [Logits; 10] = [
    Logits::Random,
    Logits::AllEqual,
    Logits::TwoLevel,
    Logits::DupMax,
    Logits::Sparse,
    Logits::BigMagnitude,
    Logits::Descending,
    Logits::Ascending,
    Logits::PartialOverlap,
    Logits::FullOverlap,
];

fn make_logits(kind: Logits, rng: &mut Lcg, lane0: &[f32]) -> Vec<f32> {
    let mut l: Vec<f32> = (0..N_EXP).map(|_| rng.f32_signed() * 3.0).collect();
    match kind {
        Logits::Random => {}
        Logits::AllEqual => l.iter_mut().for_each(|v| *v = 0.5),
        Logits::TwoLevel => {
            for (i, v) in l.iter_mut().enumerate() {
                *v = if i < 100 { 2.0 } else { 1.0 };
            }
        }
        Logits::DupMax => {
            let m = l.iter().cloned().fold(f32::MIN, f32::max) + 1.0;
            for i in [17usize, 200, 255, 0] {
                l[i] = m;
            }
            // two further exactly-tied pairs inside the top-8
            l[3] = m - 0.5;
            l[130] = m - 0.5;
            l[64] = m - 0.75;
            l[65] = m - 0.75;
        }
        Logits::Sparse => {
            l.iter_mut().for_each(|v| *v = -1000.0);
            for i in [200usize, 5, 77, 31, 142] {
                l[i] = 10.0;
            }
        }
        Logits::BigMagnitude => l.iter_mut().for_each(|v| *v *= 30.0),
        Logits::Descending => l
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = -(i as f32) * 0.25),
        Logits::Ascending => l
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = (i as f32) * 0.25),
        Logits::PartialOverlap => {
            l = lane0.to_vec();
            let mut order: Vec<usize> = (0..N_EXP).collect();
            order.sort_by(|a, b| l[*b].partial_cmp(&l[*a]).unwrap());
            for (k, &e) in order.iter().take(3).enumerate() {
                l[e] = -5.0 - k as f32;
                l[order[100 + k]] = 9.0 + k as f32; // new experts take their place
            }
        }
        Logits::FullOverlap => l = lane0.to_vec(),
    }
    l
}

fn make_lane(kind: Logits, seed: u64, lane: usize, lane0: &[f32]) -> LaneIn {
    let mut rng = Lcg(0xA3B0_2000 ^ seed.wrapping_mul(0x9E37_79B9) ^ ((lane as u64) << 40));
    let mag = 0.5 + ((seed as usize + lane) % 5) as f32 * 0.5;
    let logits = make_logits(kind, &mut rng, lane0);
    let scalar = match (seed as usize + lane) % 6 {
        0 => 30.0,
        1 => -100.0,
        _ => rng.f32_signed() * 4.0,
    };
    let big = 1.0 + 4.0 * ((seed as usize + lane) % 3) as f32;
    LaneIn {
        logits,
        scalar,
        sgate: (0..SMI).map(|_| rng.f32_signed() * 3.0 * big).collect(),
        sup: (0..SMI).map(|_| rng.f32_signed() * 2.0).collect(),
        x_rot: (0..HIDDEN).map(|_| rng.f32_signed() * mag).collect(),
        x: (0..HIDDEN).map(|_| rng.f32_signed() * 3.0).collect(),
    }
}

fn make_lanes(seed: u64, n: usize) -> Vec<LaneIn> {
    let mut out: Vec<LaneIn> = Vec::new();
    for r in 0..n {
        let kind = KINDS[(seed as usize + r * 3) % KINDS.len()];
        let lane0 = if r == 0 {
            vec![0.0; N_EXP]
        } else {
            out[0].logits.clone()
        };
        // overlap kinds need a lane 0 of their own to copy; fall back to random.
        let kind = if r == 0 && matches!(kind, Logits::PartialOverlap | Logits::FullOverlap) {
            Logits::Random
        } else {
            kind
        };
        out.push(make_lane(kind, seed, r, &lane0));
    }
    out
}

fn distinct_experts(outs: &[LaneOut]) -> usize {
    let mut s = std::collections::BTreeSet::new();
    for o in outs {
        s.extend(o.topk_idx.iter().copied());
    }
    s.len()
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn moe_decode_lanes_twins_match_singleton_launches_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    assert!(
        gpu.arch_caps.is_gfx1201(),
        "this gate requires an exact gfx1201 device (got {})",
        gpu.arch
    );
    let single = alloc_single(&mut gpu);
    let bat = alloc_batch(&mut gpu);
    let mut total_bad = 0usize;
    let mut compared_lanes = 0usize;
    let mut overlap_cases = 0usize;
    let mut tie_cases = 0usize;

    // ---- random + adversarial inputs, N in {1,2,3,8}, 4 weight sets -------
    for (wseed, kind) in [
        (1u64, Router::Random),
        (2, Router::DupPairs),
        (3, Router::AllEqual),
        (4, Router::FourLevel),
    ] {
        let w = build_weights(&mut gpu, wseed, kind);
        for &n in &[1usize, 2, 3, 8] {
            for seed in 0..10u64 {
                let lanes = make_lanes(seed * 13 + wseed, n);
                poison_batch(&mut gpu, &bat);
                load_projections(&mut gpu, &bat, &lanes);
                run_entry(&mut gpu, &w, &bat, n);
                let dump = download_batch(&mut gpu, &bat);
                let mut wants = Vec::new();
                for (r, l) in lanes.iter().enumerate() {
                    let want = ref_from_projections(&mut gpu, &w, &single, l);
                    let got = lane_out(&dump, r);
                    let bad = diff(&format!("w{wseed} n{n} seed{seed} lane{r}"), &got, &want);
                    total_bad += bad;
                    compared_lanes += 1;
                    // The comparison must not be vacuous: the reference
                    // wrote every output and moved the residual.
                    assert!(
                        want.topk_idx.iter().all(|&i| (i as usize) < N_EXP)
                            && want.gate.iter().all(|&v| v != POISON)
                            && want.shared_rot.iter().all(|&v| v != POISON),
                        "reference left poison in an output"
                    );
                    let moved = want
                        .x
                        .iter()
                        .zip(&l.x)
                        .filter(|(a, b)| **a != b.to_bits())
                        .count();
                    assert!(moved > HIDDEN / 2, "reference did not move the residual");
                    wants.push(want);
                }
                total_bad += guard_rows_untouched(&format!("w{wseed} n{n} seed{seed}"), &dump, n);
                // coverage accounting
                for r in 0..n {
                    let ids: std::collections::BTreeSet<u32> =
                        wants[r].topk_idx.iter().copied().collect();
                    let ties = {
                        let lg: Vec<u32> = lanes[r].logits.iter().map(|v| v.to_bits()).collect();
                        let mut t = 0;
                        for &i in &wants[r].topk_idx {
                            t += lg.iter().filter(|&&v| v == lg[i as usize]).count() - 1;
                        }
                        t
                    };
                    if ties > 0 {
                        tie_cases += 1;
                    }
                    for r2 in r + 1..n {
                        if wants[r2].topk_idx.iter().any(|i| ids.contains(i)) {
                            overlap_cases += 1;
                        }
                    }
                }
                if seed == 0 {
                    eprintln!(
                        "w{wseed} {kind:?} n{n}: distinct experts across lanes = {}",
                        distinct_experts(&wants)
                    );
                }
            }
        }
    }
    eprintln!(
        "compared {compared_lanes} lanes; lanes with a tied logit among their top-8 = {tie_cases}; lane pairs sharing an expert = {overlap_cases}; total mismatched words = {total_bad}"
    );
    assert!(tie_cases > 20, "tie coverage too thin: {tie_cases}");
    assert!(
        overlap_cases > 20,
        "overlap coverage too thin: {overlap_cases}"
    );

    // ---- negative control -------------------------------------------------
    {
        let w = build_weights(&mut gpu, 7, Router::Random);
        let lanes = make_lanes(5, 3);
        poison_batch(&mut gpu, &bat);
        load_projections(&mut gpu, &bat, &lanes);
        run_entry(&mut gpu, &w, &bat, 3);
        let base = download_batch(&mut gpu, &bat);
        // (a) one input bit of lane 1's rotated activation
        let mut pert = lanes.clone();
        pert[1].x_rot[5] = f32::from_bits(pert[1].x_rot[5].to_bits() ^ 1);
        poison_batch(&mut gpu, &bat);
        load_projections(&mut gpu, &bat, &pert);
        run_entry(&mut gpu, &w, &bat, 3);
        let got = download_batch(&mut gpu, &bat);
        let want1 = ref_from_projections(&mut gpu, &w, &single, &pert[1]);
        assert_eq!(
            diff(
                "neg-control lane1 vs perturbed singleton",
                &lane_out(&got, 1),
                &want1
            ),
            0
        );
        let n_diff = diff(
            "neg-control (expected)",
            &lane_out(&got, 1),
            &lane_out(&base, 1),
        );
        eprintln!("negative control (x_rot bit): lane 1 differs from baseline in {n_diff} words");
        assert!(n_diff > 0, "a one-bit x_rot change left lane 1 unchanged");
        for r in [0usize, 2] {
            assert_eq!(
                diff("neg-control bleed", &lane_out(&got, r), &lane_out(&base, r)),
                0,
                "lane {r} changed when only lane 1 was perturbed"
            );
        }
        // (b) a router-logit change that flips the top-k of lane 2
        let mut pert = lanes.clone();
        let base_top: Vec<u32> = lane_out(&base, 2).topk_idx;
        let promoted = (0..N_EXP as u32)
            .rev()
            .find(|e| !base_top.contains(e))
            .expect("an expert outside the top-8") as usize;
        pert[2].logits[promoted] = 50.0;
        poison_batch(&mut gpu, &bat);
        load_projections(&mut gpu, &bat, &pert);
        run_entry(&mut gpu, &w, &bat, 3);
        let got = download_batch(&mut gpu, &bat);
        assert_ne!(
            lane_out(&got, 2).topk_idx,
            lane_out(&base, 2).topk_idx,
            "perturbed logit did not change the routing"
        );
        let want2 = ref_from_projections(&mut gpu, &w, &single, &pert[2]);
        assert_eq!(
            diff("neg-control lane2 routing", &lane_out(&got, 2), &want2),
            0
        );
        assert_eq!(
            diff("neg-control bleed", &lane_out(&got, 0), &lane_out(&base, 0)),
            0
        );
    }

    // ---- 200 back-to-back launches, every one compared ---------------------
    {
        let w = build_weights(&mut gpu, 9, Router::Random);
        let lanes = make_lanes(11, 8);
        let wants: Vec<LaneOut> = lanes
            .iter()
            .map(|l| ref_from_projections(&mut gpu, &w, &single, l))
            .collect();
        let mut bad = 0;
        for it in 0..200 {
            poison_batch(&mut gpu, &bat);
            load_projections(&mut gpu, &bat, &lanes);
            run_entry(&mut gpu, &w, &bat, 8);
            let dump = download_batch(&mut gpu, &bat);
            for r in 0..8 {
                bad += diff(
                    &format!("repeat {it} lane {r}"),
                    &lane_out(&dump, r),
                    &wants[r],
                );
            }
        }
        eprintln!("200 back-to-back 8-lane launches: mismatched words = {bad}");
        total_bad += bad;
    }

    assert_eq!(
        total_bad, 0,
        "lane twins are not byte-identical to the singleton"
    );
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn moe_chain_pm_xbatch_and_batched_norm_match_singleton_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    assert!(
        gpu.arch_caps.is_gfx1201(),
        "this gate requires an exact gfx1201 device (got {})",
        gpu.arch
    );
    assert!(
        gpu.flags.pm_decode,
        "kernel.pm_decode is off: the x-batch projections would run the hipcc fallback, not the PM objects"
    );
    gpu.ensure_mq_signs().expect("signs");
    let single = alloc_single(&mut gpu);
    let bat = alloc_batch(&mut gpu);
    let mut total_bad = 0usize;
    let mut compared = 0usize;

    for (wseed, kind) in [
        (21u64, Router::Random),
        (22, Router::DupPairs),
        (23, Router::AllEqual),
        (24, Router::FourLevel),
    ] {
        let w = build_weights(&mut gpu, wseed, kind);
        for &n in &[1usize, 2, 3, 8] {
            for seed in 0..6u64 {
                let mut rng = Lcg(0xC0DE_0000 ^ (seed << 20) ^ (wseed << 8) ^ n as u64);
                let mag = 0.5 + (seed % 4) as f32 * 2.0;
                let xs: Vec<Vec<f32>> = (0..n)
                    .map(|_| (0..HIDDEN).map(|_| rng.f32_signed() * mag).collect())
                    .collect();

                // ---- batched chain: norm (batched), four pm_xbatch plain, entry
                poison_batch(&mut gpu, &bat);
                for (r, x) in xs.iter().enumerate() {
                    put_f32(&mut gpu, &bat.x, r * HIDDEN, x);
                }
                gpu.fused_rmsnorm_rotate_mq_batched(&bat.x, &w.norm_w, &bat.x_rot, HIDDEN, EPS, n)
                    .expect("batched norm");
                gpu.gemv_mq4g256v2_xbatch_pm(
                    &w.router,
                    &bat.x_rot,
                    &bat.router_logits,
                    N_EXP,
                    HIDDEN,
                    n,
                )
                .expect("router xbatch");
                gpu.gemv_mq4g256v2_xbatch_pm(&w.selector, &bat.x_rot, &bat.scalar, 1, HIDDEN, n)
                    .expect("selector xbatch");
                gpu.gemv_mq4g256v2_xbatch_pm(&w.sgate, &bat.x_rot, &bat.sgate, SMI, HIDDEN, n)
                    .expect("shared gate xbatch");
                gpu.gemv_mq4g256v2_xbatch_pm(&w.sup, &bat.x_rot, &bat.sup, SMI, HIDDEN, n)
                    .expect("shared up xbatch");
                // snapshot the stage outputs before the entry reads them
                let stage = download_batch(&mut gpu, &bat);
                run_entry(&mut gpu, &w, &bat, n);
                let dump = download_batch(&mut gpu, &bat);

                for (r, x) in xs.iter().enumerate() {
                    let tag = format!("w{wseed} n{n} seed{seed} lane{r}");
                    let want_u = ref_from_chain(&mut gpu, &w, &single, x, Proj::Unfolded);
                    let want_f = ref_from_chain(&mut gpu, &w, &single, x, Proj::Fold);
                    // fold == unfolded singleton chain (the product's folded launch)
                    total_bad += diff(&format!("{tag} fold-vs-unfolded"), &want_f, &want_u);

                    // stage attribution: norm row, then the four projections
                    let st = lane_out(&stage, r);
                    total_bad += diff(
                        &format!("{tag} stage(norm+xbatch)"),
                        &LaneOut {
                            shared_rot: want_u.shared_rot.clone(),
                            topk_idx: want_u.topk_idx.clone(),
                            topk_w: want_u.topk_w.clone(),
                            gate: want_u.gate.clone(),
                            up: want_u.up.clone(),
                            x: want_u.x.clone(),
                            ..st
                        },
                        &want_u,
                    );
                    // whole chain through the lane entry
                    let got = lane_out(&dump, r);
                    total_bad += diff(&format!("{tag} chain"), &got, &want_u);
                    compared += 1;
                    assert!(
                        want_u.router_logits.iter().all(|&v| v != POISON)
                            && want_u.scalar[0] != POISON
                            && want_u.x_rot.iter().all(|&v| v != POISON),
                        "reference left poison in a projection output"
                    );
                }
                total_bad += guard_rows_untouched(&format!("w{wseed} n{n} seed{seed}"), &dump, n);
            }
        }
    }

    // ---- negative control: a one-bit change of one lane's pre-norm x must
    // change that lane's chain and leave the other lanes alone ---------------
    {
        let w = build_weights(&mut gpu, 31, Router::Random);
        let n = 3;
        let mut rng = Lcg(0x5EED);
        let xs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..HIDDEN).map(|_| rng.f32_signed() * 2.0).collect())
            .collect();
        let run = |gpu: &mut Gpu, xs: &[Vec<f32>]| {
            poison_batch(gpu, &bat);
            for (r, x) in xs.iter().enumerate() {
                put_f32(gpu, &bat.x, r * HIDDEN, x);
            }
            gpu.fused_rmsnorm_rotate_mq_batched(&bat.x, &w.norm_w, &bat.x_rot, HIDDEN, EPS, n)
                .unwrap();
            gpu.gemv_mq4g256v2_xbatch_pm(
                &w.router,
                &bat.x_rot,
                &bat.router_logits,
                N_EXP,
                HIDDEN,
                n,
            )
            .unwrap();
            gpu.gemv_mq4g256v2_xbatch_pm(&w.selector, &bat.x_rot, &bat.scalar, 1, HIDDEN, n)
                .unwrap();
            gpu.gemv_mq4g256v2_xbatch_pm(&w.sgate, &bat.x_rot, &bat.sgate, SMI, HIDDEN, n)
                .unwrap();
            gpu.gemv_mq4g256v2_xbatch_pm(&w.sup, &bat.x_rot, &bat.sup, SMI, HIDDEN, n)
                .unwrap();
            run_entry(gpu, &w, &bat, n);
            download_batch(gpu, &bat)
        };
        let base = run(&mut gpu, &xs);
        let mut pert = xs.clone();
        // A single ulp can vanish in the norm's sums, so flip a mid mantissa bit.
        pert[1][700] = f32::from_bits(pert[1][700].to_bits() ^ (1 << 16));
        let got = run(&mut gpu, &pert);
        let want1 = ref_from_chain(&mut gpu, &w, &single, &pert[1], Proj::Unfolded);
        assert_eq!(
            diff("neg-control lane1 vs singleton", &lane_out(&got, 1), &want1),
            0
        );
        let n_diff = diff(
            "neg-control (expected)",
            &lane_out(&got, 1),
            &lane_out(&base, 1),
        );
        eprintln!(
            "negative control (pre-norm x bit): lane 1 differs from baseline in {n_diff} words"
        );
        assert!(
            n_diff > 0,
            "a one-bit input change left the chain unchanged"
        );
        for r in [0usize, 2] {
            assert_eq!(diff("bleed", &lane_out(&got, r), &lane_out(&base, r)), 0);
        }
    }

    eprintln!("compared {compared} lane chains; total mismatched words = {total_bad}");
    assert_eq!(
        total_bad, 0,
        "batched chain is not byte-identical to the singleton chain"
    );
}
