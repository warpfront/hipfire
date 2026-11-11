// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness oracle of the multi-lane Flash-Next GDN VERIFY core
//! (`gated_delta_core_lanes_rows`: `gated_delta_conv_params_bf16_f32_rows_lanes`
//! + `gated_delta_step_halves_state128_persistent256_capture_q8_lanes` + the
//! incumbent batched gate over all rows, three launches for up to 8 lanes of
//! 2..=8 rows each) against the per-lane chain `layer_ops.rs`
//! `execute_gated_delta_net_hc_impl` issues for one lane under the armed capture
//! ring: `copy_d2d(projection -> capture.inputs)`,
//! `gated_delta_conv_params_batched`, `gated_delta_step_batched` (with the
//! ring), `gated_delta_gate_batched[_rotate]`. The reference lanes are built
//! from the very same `GdnCoreRowsLane` arguments the candidate takes.
//!
//! The groups are the interpreter's: lanes with one row are AR lanes (the
//! per-row branch: `gated_delta_conv_params` + `gated_delta_step_gated`; two or
//! more of them go through `gated_delta_core_lanes`, launched as
//! `launch_rotated`: ONE fused launch under HIPFIRE_QWEN4_GDN_LANES_FUSE, beta's
//! default), lanes with 2..=8 rows are verify lanes (the capture chain; two or more
//! of them go through `gated_delta_core_lanes_rows`, gated on
//! HIPFIRE_QWEN4_GDN_LANES_ROWS); a set with a single member runs per lane in
//! both arms, exactly as `execute_gated_delta_net_core_lanes` does. A MIXED
//! group is both sets of one layer: every kernel of each set launches once for
//! the whole group (one-row kernels over the AR lanes, rows kernels over the
//! verify lanes). A step in which no class holds two lanes (N = 1, or one AR and
//! one verify lane) is per lane by design (`execute_window` groups two or more
//! steps) and is reported as `per_lane_steps`.
//!
//! One synthetic Flash-Next GDN layer: key heads 16, value heads 48, head dim
//! 128 (q|k|v conv channels 10240, history 3 x 10240 F32, Q8 slot 48 x 128 x
//! 132 B). Lanes share the conv weight, `a_log`, `dt_bias` and gated-norm weight,
//! live in ONE packed row scratch per tensor (lane segments in `seg_order`),
//! and own a history, a capture ring of two halves of `RING_ROWS` Q8 slots
//! (live slot in one half, a verify writes slot `rows - 1` of the OTHER half and
//! then commits it live, as `Qwen4State` does), capture `inputs` and capture
//! `recurrence` (`[rows*qkv | rows*48 | rows*48]`).
//!
//! Per case: N lanes, rows per lane and step from a pattern (`all1`, `all2`,
//! `all8`, `ragged` 1..=8, `mixed` AR lanes among verify lanes, `verify_ragged`
//! 2..=8), `--steps` carried steps (state, ring halves and positions advance
//! exactly as the arch crate commits them, every step has fresh random inputs),
//! ragged positions (`position % 3` cursors, Q8 frames up to near `u32::MAX`), and
//! three orderings: the lanes TABLE order, the packed memory order of the lane
//! segments, and the order of the per-lane reference launches.
//!
//! Compared as BYTES, per step: every shared scratch (projection, convolution
//! output of AR lanes, a, b, gate / beta scratch, recurrent output, z, gdn
//! output), and per lane the history, the WHOLE ring (the slot written and, by
//! comparison with the reference, every slot not to be touched: the live slot
//! included), the capture inputs and the capture recurrence. Sentinel-filled
//! outputs and guard bands around every buffer must stay intact; the candidate
//! arm's ring is also checked for writes outside the one slot a lane may write.
//!
//! Also: launch counts (per-lane chain = 3 kernels + 1 device copy per verify
//! lane, 2 per AR lane; grouped = 3 per verify set, 1 per AR set under FUSE, else
//! 2), negative controls the oracle MUST detect (two lanes' capture rings
//! swapped, two lanes' live slots swapped, one lane's Q8 frame off by one, one
//! lane's convolution cursor off by one, one lane's rows claimed one short, an AR
//! lane's frame off by one), admission probes (flags included), and a timing arm
//! (N = 8 lanes, rows 2..4 by default, `--reps` replays with event timing).
//!
//! `cargo run --release -p rdna-compute --features lab --example
//!    gdn_lanes_rows_oracle -- [--out DIR] [--steps S] [--lanes 1,2,3,8]
//!    [--seed U64] [--skip-controls] [--skip-timing] [--time-rows 2,3,4]
//!    [--reps 50] [--bench ITERS [--bench-runs 7]]`
//!
//! `--bench ITERS`: timing only (a fabric `--class timing` job): per layer step,
//! the per-lane chain against the grouped launches for 8 lanes x rows 1, 2, 4 and
//! the mixed window [1,4,1,4,2,1,3,8], the median of `--bench-runs` (>= 5) runs
//! of ITERS replays, with the launches one layer step issues counted.
//!
//! Prints one line per case / control / probe, a final `gdn_lanes_rows_oracle:
//! PASS|FAIL` line and a JSON summary (stdout, and `DIR/summary.json` with
//! `--out`). Exit 0 = PASS, 1 = a mismatch / undetected control / wrong launch
//! count / wrong admission, 2 = the GPU cannot run the route (needs exact
//! gfx1201).

use hip_bridge::memory_effects;
use rdna_compute::tensor_ops::{
    gated_delta_conv_params, gated_delta_conv_params_batched, gated_delta_core_lanes,
    gated_delta_core_lanes_rows, gated_delta_gate_batched, gated_delta_gate_batched_rotate,
    gated_delta_step_batched, gated_delta_step_gated, GatedDeltaConv, GatedDeltaConvBatched,
    GatedDeltaGate, GatedDeltaGateBatched, GatedDeltaParams, GatedDeltaParamsBatched,
    GatedDeltaStep, GatedDeltaStepBatched, GdnCoreLane, GdnCoreRowsLane, GdnStateFormat,
    GDN_CORE_LANES_MAX, GDN_CORE_LANE_ROWS,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::sync::Arc;

const NK: usize = 16;
const NV: usize = 48;
const HD: usize = 128;
const QK: usize = NK * HD;
const VALUE: usize = NV * HD;
const QKV: usize = 2 * QK + VALUE;
const RING: usize = 3;
const SLOT: usize = NV * HD * (HD + 4);
/// Rows one capture ring half holds (one more than a verify may use, so the
/// probes can form a 9-row lane).
const RING_ROWS: usize = GDN_CORE_LANE_ROWS + 1;
/// Guard elements around every F32 buffer (1 KiB), bytes around raw ones.
const GUARD_F32: usize = 256;
const GUARD_RAW: usize = 1024;
const F32_SENTINEL: u32 = 0x7fc0_dead;
const RAW_SENTINEL: u8 = 0xA5;

/// The repeating byte pattern of a sentinel-filled buffer.
fn sentinel_pattern(raw: bool) -> Vec<u8> {
    if raw {
        vec![RAW_SENTINEL]
    } else {
        F32_SENTINEL.to_le_bytes().to_vec()
    }
}

/// Guard bytes of a `guard | body | guard` image `all` (`guard` and `len` in
/// elements of `raw ? 1 : 4` bytes) that no longer hold the sentinel pattern.
fn guard_violations(raw: bool, guard: usize, len: usize, all: &[u8]) -> usize {
    let elem = if raw { 1 } else { 4 };
    let pattern = sentinel_pattern(raw);
    let head = guard * elem;
    let tail_at = head + len * elem;
    assert_eq!(all.len(), tail_at + head, "image length");
    (0..head)
        .filter(|&i| all[i] != pattern[i % pattern.len()])
        .count()
        + (tail_at..all.len())
            .filter(|&i| all[i] != pattern[(i - tail_at) % pattern.len()])
            .count()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Uniform in [-1, 1).
    fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn f32s(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|_| rng.sym() * amp).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// BF16 bits of `v` (truncation; the values only have to be representable).
fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| bf16_bits(*x).to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------
// Device buffers with guard bands
// ---------------------------------------------------------------------------

/// One device buffer: `guard | body | guard`; `view` addresses the body.
struct Guarded {
    t: GpuTensor,
    /// Guard length in elements (F32 buffers) or bytes (raw buffers).
    guard: usize,
    /// Body length in elements / bytes.
    len: usize,
    raw: bool,
}

impl Guarded {
    fn elem(&self) -> usize {
        if self.raw {
            1
        } else {
            4
        }
    }

    /// Full host image: sentinel guards around `body` (sentinel when `None`).
    fn image_of(raw: bool, guard: usize, len: usize, body: Option<&[u8]>) -> Vec<u8> {
        let elem = if raw { 1 } else { 4 };
        let pattern = sentinel_pattern(raw);
        let fill = |n: usize| -> Vec<u8> { (0..n).map(|i| pattern[i % pattern.len()]).collect() };
        let mut out = fill(guard * elem);
        match body {
            Some(b) => {
                assert_eq!(b.len(), len * elem, "body length");
                out.extend_from_slice(b);
            }
            None => out.extend(fill(len * elem)),
        }
        out.extend(fill(guard * elem));
        out
    }

    fn image(&self, body: Option<&[u8]>) -> Vec<u8> {
        Self::image_of(self.raw, self.guard, self.len, body)
    }

    fn new_f32(gpu: &mut Gpu, len: usize, body: Option<&[f32]>) -> Self {
        let bytes = body.map(f32_bytes);
        let image = Self::image_of(false, GUARD_F32, len, bytes.as_deref());
        let floats: Vec<f32> = image
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Self {
            t: gpu
                .upload_f32(&floats, &[floats.len()])
                .expect("upload f32"),
            guard: GUARD_F32,
            len,
            raw: false,
        }
    }

    fn new_raw(gpu: &mut Gpu, body: &[u8]) -> Self {
        let image = Self::image_of(true, GUARD_RAW, body.len(), Some(body));
        Self {
            t: gpu.upload_raw(&image, &[image.len()]).expect("upload raw"),
            guard: GUARD_RAW,
            len: body.len(),
            raw: true,
        }
    }

    /// Rewrite the whole buffer (guards sentinel, body = `body` or sentinel).
    fn write(&self, gpu: &Gpu, body: Option<&[u8]>) {
        let image = self.image(body);
        gpu.hip
            .memcpy_htod(&self.t.buf, &image)
            .expect("rewrite buffer");
    }

    fn view(&self, offset: usize, len: usize) -> GpuTensor {
        assert!(offset + len <= self.len, "view outside the body");
        self.t.sub_offset(self.guard + offset, len)
    }

    fn download(&self, gpu: &Gpu) -> Vec<u8> {
        gpu.download_raw_bytes(&self.t).expect("download")
    }

    /// Number of guard bytes that no longer hold the sentinel.
    fn guard_bad(&self, all: &[u8]) -> usize {
        guard_violations(self.raw, self.guard, self.len, all)
    }

    fn body<'b>(&self, all: &'b [u8]) -> &'b [u8] {
        let g = self.guard * self.elem();
        &all[g..g + self.len * self.elem()]
    }
}

/// Shared device weights (identical pointers across lanes).
struct Weights {
    conv: GpuTensor,
    a_log: GpuTensor,
    dt_bias: GpuTensor,
    norm: GpuTensor,
}

struct WeightsHost {
    conv: Vec<u8>,
    a_log: Vec<u8>,
    dt_bias: Vec<u8>,
    norm: Vec<u8>,
}

impl WeightsHost {
    fn new(rng: &mut Rng) -> Self {
        Self {
            conv: bf16_bytes(&f32s(rng, QKV * 4, 0.5)),
            a_log: bf16_bytes(&f32s(rng, NV, 1.0)),
            dt_bias: bf16_bytes(&f32s(rng, NV, 0.5)),
            norm: bf16_bytes(&(0..HD).map(|_| 1.0 + rng.sym() * 0.5).collect::<Vec<f32>>()),
        }
    }
}

impl Weights {
    fn upload(gpu: &mut Gpu, h: &WeightsHost) -> Self {
        let bf16 = |bytes: &[u8]| {
            let mut t = gpu.upload_raw(bytes, &[bytes.len()]).expect("weight");
            t.dtype = DType::BF16;
            t.shape = vec![bytes.len() / 2];
            t
        };
        Self {
            conv: bf16(&h.conv),
            a_log: bf16(&h.a_log),
            dt_bias: bf16(&h.dt_bias),
            norm: bf16(&h.norm),
        }
    }

    fn free(self, gpu: &mut Gpu) {
        for t in [self.conv, self.a_log, self.dt_bias, self.norm] {
            gpu.free_tensor(t).expect("free weight");
        }
    }
}

// ---------------------------------------------------------------------------
// Host data
// ---------------------------------------------------------------------------

struct StepIn {
    rows: usize,
    qkv: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    z: Vec<f32>,
}

/// One logical lane: initial live slot, history, position and per-step inputs.
struct LaneHost {
    hist: Vec<f32>,
    state: Vec<u8>,
    base_pos: usize,
    steps: Vec<StepIn>,
}

fn random_q8_slot(rng: &mut Rng) -> Vec<u8> {
    let mut state = Vec::with_capacity(SLOT);
    for _ in 0..NV * HD * HD {
        state.push(((rng.next() % 255) as i32 - 127) as i8 as u8);
    }
    for _ in 0..NV * HD {
        state.extend_from_slice(&(0.001 + rng.unit() * 0.02).to_le_bytes());
    }
    assert_eq!(state.len(), SLOT);
    state
}

fn make_lane_host(rng: &mut Rng, base_pos: usize, rows: &[usize]) -> LaneHost {
    LaneHost {
        hist: f32s(rng, RING * QKV, 1.0),
        state: random_q8_slot(rng),
        base_pos,
        steps: rows
            .iter()
            .map(|&r| StepIn {
                rows: r,
                qkv: f32s(rng, r * QKV, 1.5),
                a: f32s(rng, r * NV, 2.0),
                b: f32s(rng, r * NV, 2.0),
                z: f32s(rng, r * VALUE, 1.5),
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// One arm: packed row scratch + per-lane state
// ---------------------------------------------------------------------------

struct LaneBufs {
    history: Guarded,
    /// Two halves of `RING_ROWS` Q8 slots; the live slot is one of them.
    ring: Guarded,
    inputs: Guarded,
    recurrence: Guarded,
}

struct Arm {
    /// Row capacity of the packed scratches.
    cap: usize,
    projection: Guarded,
    projection2: Guarded,
    a: Guarded,
    b: Guarded,
    gate: Guarded,
    beta: Guarded,
    recurrent: Guarded,
    z: Guarded,
    out: Guarded,
    rot: Guarded,
    lanes: Vec<LaneBufs>,
    /// Per logical lane: a verify lane's qkv projection rows sit IN its capture `inputs`
    /// (`HIPFIRE_QWEN4_GDN_CAPTURE_DIRECT`: the in-projection wrote the capture in place), so
    /// its lane's `projection` IS its `conv.input` and the per-lane chain issues no copy.
    alias: Vec<bool>,
}

impl Arm {
    fn new(gpu: &mut Gpu, cap: usize, host: &[LaneHost], alias: &[bool]) -> Self {
        let sent = |gpu: &mut Gpu, w: usize| Guarded::new_f32(gpu, cap * w, None);
        let lanes = host
            .iter()
            .map(|h| {
                // Live slot 0 holds the lane's Q8 state; every other slot is
                // sentinel (what a fresh ring holds that nothing ever wrote).
                let mut ring = vec![RAW_SENTINEL; 2 * RING_ROWS * SLOT];
                ring[..SLOT].copy_from_slice(&h.state);
                LaneBufs {
                    history: Guarded::new_f32(gpu, RING * QKV, Some(&h.hist)),
                    ring: Guarded::new_raw(gpu, &ring),
                    inputs: Guarded::new_f32(gpu, RING_ROWS * QKV, None),
                    recurrence: Guarded::new_f32(gpu, RING_ROWS * (QKV + 2 * NV), None),
                }
            })
            .collect();
        Self {
            cap,
            projection: sent(gpu, QKV),
            projection2: sent(gpu, QKV),
            a: sent(gpu, NV),
            b: sent(gpu, NV),
            gate: sent(gpu, NV),
            beta: sent(gpu, NV),
            recurrent: sent(gpu, VALUE),
            z: sent(gpu, VALUE),
            out: sent(gpu, VALUE),
            rot: sent(gpu, VALUE),
            lanes,
            alias: alias.to_vec(),
        }
    }

    fn free(self, gpu: &mut Gpu) {
        let Self {
            projection,
            projection2,
            a,
            b,
            gate,
            beta,
            recurrent,
            z,
            out,
            rot,
            lanes,
            ..
        } = self;
        let mut all = vec![
            projection,
            projection2,
            a,
            b,
            gate,
            beta,
            recurrent,
            z,
            out,
            rot,
        ];
        for l in lanes {
            all.extend([l.history, l.ring, l.inputs, l.recurrence]);
        }
        for g in all {
            gpu.free_tensor(g.t).expect("free arm");
        }
    }

    /// Fill the input scratches (the lanes' segments, sentinel elsewhere) and
    /// reset every output scratch and capture buffer to the sentinel. `seg` and
    /// `rows` are per logical lane; the lane staged last wins an overlap.
    fn stage(&self, gpu: &Gpu, host: &[LaneHost], step: usize, seg: &[usize], rows: &[usize]) {
        let mut by_seg: Vec<usize> = (0..host.len()).collect();
        by_seg.sort_by_key(|&l| seg[l]);
        let fill = |width: usize, pick: fn(&StepIn) -> &Vec<f32>| -> Vec<u8> {
            let mut body = vec![f32::from_bits(F32_SENTINEL); self.cap * width];
            for &l in &by_seg {
                let data = pick(&host[l].steps[step]);
                assert_eq!(host[l].steps[step].rows, rows[l], "staged lane rows");
                assert_eq!(data.len(), rows[l] * width, "staged rows");
                body[seg[l] * width..seg[l] * width + data.len()].copy_from_slice(data);
            }
            f32_bytes(&body)
        };
        self.projection.write(gpu, Some(&fill(QKV, |s| &s.qkv)));
        self.a.write(gpu, Some(&fill(NV, |s| &s.a)));
        self.b.write(gpu, Some(&fill(NV, |s| &s.b)));
        self.z.write(gpu, Some(&fill(VALUE, |s| &s.z)));
        for g in [
            &self.projection2,
            &self.gate,
            &self.beta,
            &self.recurrent,
            &self.out,
            &self.rot,
        ] {
            g.write(gpu, None);
        }
        for l in &self.lanes {
            l.inputs.write(gpu, None);
            l.recurrence.write(gpu, None);
        }
        for (lane, l) in self.lanes.iter().enumerate() {
            if self.alias[lane] && rows[lane] >= 2 {
                let data = &host[lane].steps[step].qkv;
                let mut body = vec![f32::from_bits(F32_SENTINEL); RING_ROWS * QKV];
                body[..data.len()].copy_from_slice(data);
                l.inputs.write(gpu, Some(&f32_bytes(&body)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step plan: the host bookkeeping the arch crate does around the launches
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
struct LaneStep {
    rows: usize,
    /// Position of the lane's first row.
    pos: usize,
    /// Live Q8 slot (ring index) the step reads.
    live: usize,
    /// First slot of the half a verify writes (the half without the live slot).
    base: usize,
    /// First row of the lane's segment in the packed scratches.
    seg: usize,
}

/// The half of the ring a verify writes: the one without the live slot
/// (`Qwen4State::capture_base`).
fn capture_base(live: usize) -> usize {
    if live < RING_ROWS {
        RING_ROWS
    } else {
        0
    }
}

/// Per-lane carried bookkeeping: position and live slot.
#[derive(Clone, Copy, Debug)]
struct Carry {
    pos: usize,
    live: usize,
}

/// The step plan of `rows` (per logical lane) with segments packed in
/// `seg_order`.
fn plan_step(rows: &[usize], seg_order: &[usize], carry: &[Carry]) -> Vec<LaneStep> {
    let mut seg = vec![0usize; rows.len()];
    let mut next = 0usize;
    for &l in seg_order {
        seg[l] = next;
        next += rows[l];
    }
    (0..rows.len())
        .map(|l| LaneStep {
            rows: rows[l],
            pos: carry[l].pos,
            live: carry[l].live,
            base: capture_base(carry[l].live),
            seg: seg[l],
        })
        .collect()
}

/// The commit after a step: positions advance by the rows, a verify (>= 2
/// rows) makes its captured slot live; an AR row updates the live slot in place.
fn commit(plan: &[LaneStep], carry: &mut [Carry]) {
    for (l, p) in plan.iter().enumerate() {
        carry[l].pos += p.rows;
        if p.rows >= 2 {
            carry[l].live = p.base + p.rows - 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Lane views and the launch arguments
// ---------------------------------------------------------------------------

/// The tensor views of one lane at one step.
struct Vw {
    rows: usize,
    pos: usize,
    proj: GpuTensor,
    proj2: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    gate_s: GpuTensor,
    beta_s: GpuTensor,
    rec: GpuTensor,
    z: GpuTensor,
    out: GpuTensor,
    rot: GpuTensor,
    hist: GpuTensor,
    inputs: GpuTensor,
    conv_out: GpuTensor,
    gate_c: GpuTensor,
    beta_c: GpuTensor,
    live: GpuTensor,
    states: GpuTensor,
}

impl Vw {
    fn new(arm: &Arm, lane: usize, s: &LaneStep) -> Self {
        let r = s.rows;
        assert!(s.seg + r <= arm.cap, "segment outside the scratch");
        let lb = &arm.lanes[lane];
        let proj2 = arm.projection2.view(s.seg * QKV, r * QKV);
        Self {
            rows: r,
            pos: s.pos,
            proj: if arm.alias[lane] && r >= 2 {
                lb.inputs.view(0, r * QKV)
            } else {
                arm.projection.view(s.seg * QKV, r * QKV)
            },
            q: proj2.sub_offset(0, QK),
            k: proj2.sub_offset(QK, QK),
            v: proj2.sub_offset(2 * QK, VALUE),
            proj2,
            a: arm.a.view(s.seg * NV, r * NV),
            b: arm.b.view(s.seg * NV, r * NV),
            gate_s: arm.gate.view(s.seg * NV, r * NV),
            beta_s: arm.beta.view(s.seg * NV, r * NV),
            rec: arm.recurrent.view(s.seg * VALUE, r * VALUE),
            z: arm.z.view(s.seg * VALUE, r * VALUE),
            out: arm.out.view(s.seg * VALUE, r * VALUE),
            // `view(op.rotation, 0, ..)`: every lane's rotation scratch starts
            // at row 0 (nothing reads it).
            rot: arm.rot.view(0, r * VALUE),
            hist: lb.history.view(0, RING * QKV),
            inputs: lb.inputs.view(0, r * QKV),
            conv_out: lb.recurrence.view(0, r * QKV),
            gate_c: lb.recurrence.view(r * QKV, r * NV),
            beta_c: lb.recurrence.view(r * QKV + r * NV, r * NV),
            live: lb.ring.view(s.live * SLOT, SLOT),
            states: lb.ring.view(s.base * SLOT, r * SLOT),
        }
    }
}

/// The AR lane's per-row-branch arguments (`gated_delta_core_lanes` input).
fn ar_lane<'a>(v: &'a Vw, w: &'a Weights) -> GdnCoreLane<'a> {
    assert_eq!(v.rows, 1);
    GdnCoreLane {
        conv: GatedDeltaConv {
            input: &v.proj,
            kernel: &w.conv,
            history: &v.hist,
            output: &v.proj2,
            next_history: &v.hist,
            channels: QKV,
            history_rows: RING,
            kernel_size: 4,
            cursor: v.pos % RING,
            row_index: 0,
        },
        params: GatedDeltaParams {
            a: &v.a,
            b: &v.b,
            a_log: &w.a_log,
            dt_bias: &w.dt_bias,
            gate: &v.gate_s,
            beta: &v.beta_s,
        },
        step: GatedDeltaStep {
            q: &v.q,
            k: &v.k,
            v: &v.v,
            gate: &v.gate_s,
            beta: &v.beta_s,
            state: &v.live,
            format: GdnStateFormat::Q8,
            residual: None,
            output: &v.rec,
            key_heads: NK,
            value_heads: NV,
            key_dim: HD,
            value_dim: HD,
            position: v.pos,
        },
        gate: GatedDeltaGate {
            recurrent_output: &v.rec,
            z: &v.z,
            norm: &w.norm,
            output: &v.out,
            value_heads: NV,
            value_dim: HD,
        },
    }
}

/// The verify lane's capture-chain arguments (`gated_delta_core_lanes_rows`
/// input), as `execute_gated_delta_net_core_lanes` builds them.
fn verify_lane<'a>(v: &'a Vw, w: &'a Weights) -> GdnCoreRowsLane<'a> {
    let rows = v.rows;
    GdnCoreRowsLane {
        projection: &v.proj,
        conv: GatedDeltaConvBatched {
            input: &v.inputs,
            kernel: &w.conv,
            history: &v.hist,
            output: &v.conv_out,
            next_history: &v.hist,
            rows,
            channels: QKV,
            history_rows: RING,
            kernel_size: 4,
            start_cursor: v.pos % RING,
        },
        params: GatedDeltaParamsBatched {
            a: &v.a,
            b: &v.b,
            a_log: &w.a_log,
            dt_bias: &w.dt_bias,
            gate: &v.gate_c,
            beta: &v.beta_c,
            rows,
            heads: NV,
        },
        step: GatedDeltaStepBatched {
            projection: &v.conv_out,
            gate: &v.gate_c,
            beta: &v.beta_c,
            state: &v.live,
            format: GdnStateFormat::Q8,
            residual: None,
            output: &v.rec,
            row_states: Some(&v.states),
            rows,
            qkv_width: QKV,
            key_heads: NK,
            value_heads: NV,
            key_dim: HD,
            value_dim: HD,
            position: v.pos,
        },
        gate: GatedDeltaGateBatched {
            recurrent_output: &v.rec,
            z: &v.z,
            norm: &w.norm,
            output: &v.out,
            rows,
            value_heads: NV,
            value_dim: HD,
        },
    }
}

/// The per-lane chain of one verify lane, as the capture branch of
/// `execute_gated_delta_net_hc_impl` issues it (`rotate`: the FWHT-basis
/// output projection's rotating gate).
fn verify_per_lane(gpu: &mut Gpu, l: &GdnCoreRowsLane<'_>, rot: &GpuTensor, rotate: bool) {
    // As `execute_gated_delta_net_hc_impl`: no copy when the in-projection already wrote the
    // capture's `inputs` (`HIPFIRE_QWEN4_GDN_CAPTURE_DIRECT`, `projection` IS `conv.input`).
    if l.projection.buf.as_ptr() != l.conv.input.buf.as_ptr() {
        gpu.copy_d2d(l.projection, l.conv.input, l.conv.input.byte_size())
            .expect("capture input copy");
    }
    gated_delta_conv_params_batched(gpu, &l.conv, &l.params).expect("conv + params");
    gated_delta_step_batched(gpu, &l.step).expect("capture step");
    if rotate {
        gated_delta_gate_batched_rotate(gpu, &l.gate, rot).expect("gate + rotate");
    } else {
        gated_delta_gate_batched(gpu, &l.gate).expect("gate");
    }
}

/// The per-row branch of one AR lane.
fn ar_per_lane(gpu: &mut Gpu, l: &GdnCoreLane<'_>, rot: &GpuTensor, rotate: bool) {
    gated_delta_conv_params(gpu, &l.conv, &l.params, NV).expect("AR conv + params");
    gated_delta_step_gated(gpu, &l.step, &l.gate, rotate.then_some(rot)).expect("AR step + gate");
}

/// Reference: every lane's per-lane chain in `launch_order` (logical lanes).
fn run_reference(
    gpu: &mut Gpu,
    w: &Weights,
    arm: &Arm,
    plan: &[LaneStep],
    launch_order: &[usize],
    rotate: bool,
) {
    for &lane in launch_order {
        let v = Vw::new(arm, lane, &plan[lane]);
        if v.rows == 1 {
            ar_per_lane(gpu, &ar_lane(&v, w), &v.rot, rotate);
        } else {
            verify_per_lane(gpu, &verify_lane(&v, w), &v.rot, rotate);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Control {
    /// The first two verify lanes name each other's capture ring.
    SwapRing,
    /// The first two verify lanes name each other's live slot.
    SwapLive,
    /// The last verify lane's Q8 frame is one too high.
    WrongFrame,
    /// The first verify lane claims one row fewer than its data holds.
    RowsShort,
    /// The last AR lane's Q8 frame is one too high (the one-row lanes route
    /// of a mixed group).
    ArWrongFrame,
    /// The last verify lane's convolution cursor is one too high (mod 3).
    WrongCursor,
}

impl Control {
    const ALL: [Control; 6] = [
        Control::SwapRing,
        Control::SwapLive,
        Control::WrongFrame,
        Control::RowsShort,
        Control::ArWrongFrame,
        Control::WrongCursor,
    ];
    fn name(self) -> &'static str {
        match self {
            Control::SwapRing => "swap_ring_pointers",
            Control::SwapLive => "swap_live_slots",
            Control::WrongFrame => "wrong_q8_frame",
            Control::RowsShort => "rows_claimed_short",
            Control::ArWrongFrame => "ar_wrong_q8_frame",
            Control::WrongCursor => "wrong_conv_cursor",
        }
    }
}

/// What a grouped run launched.
#[derive(Default, Clone, Debug)]
struct Grouping {
    ar_grouped: bool,
    verify_grouped: bool,
    ar_lanes: usize,
    verify_lanes: usize,
    /// Admissible sets a route refused (a finding).
    refusals: Vec<String>,
}

impl Grouping {
    /// The route a step took, per class: `rows-lanes` (verify set grouped),
    /// `ar-lanes` (AR set grouped), both joined by `+`, or `per-lane`.
    fn route(&self) -> &'static str {
        match (self.verify_grouped, self.ar_grouped) {
            (true, true) => "rows-lanes+ar-lanes",
            (true, false) => "rows-lanes",
            (false, true) => "ar-lanes",
            (false, false) => "per-lane",
        }
    }
}

/// Candidate: the interpreter's grouping over the table `table_order`: two or
/// more AR lanes through `gated_delta_core_lanes`, two or more verify lanes
/// through `gated_delta_core_lanes_rows`, the rest per lane. `control` corrupts
/// the verify lanes' arguments (the oracle must then report a mismatch).
fn run_candidate(
    gpu: &mut Gpu,
    w: &Weights,
    arm: &Arm,
    plan: &[LaneStep],
    table_order: &[usize],
    rotate: bool,
    control: Option<Control>,
) -> Result<Grouping, String> {
    let views: Vec<(usize, Vw)> = table_order
        .iter()
        .map(|&l| (l, Vw::new(arm, l, &plan[l])))
        .collect();
    let ar: Vec<&(usize, Vw)> = views.iter().filter(|(_, v)| v.rows == 1).collect();
    let verify: Vec<&(usize, Vw)> = views.iter().filter(|(_, v)| v.rows >= 2).collect();
    let mut keep_states: Vec<GpuTensor> = Vec::new();
    let mut rows_short_views: Vec<GpuTensor> = Vec::new();
    let mut ar_lanes: Vec<GdnCoreLane<'_>> = ar.iter().map(|(_, v)| ar_lane(v, w)).collect();
    let mut verify_lanes: Vec<GdnCoreRowsLane<'_>> =
        verify.iter().map(|(_, v)| verify_lane(v, w)).collect();
    // Corrupted views of the controls: declared before the lanes that borrow them.
    if verify_lanes.len() >= 2 {
        match control {
            None => {}
            Some(Control::SwapRing) => {
                let bufs = |l: usize| &arm.lanes[l];
                let (l0, l1) = (verify[0].0, verify[1].0);
                let (p0, p1) = (&plan[l0], &plan[l1]);
                let span = p0.rows.max(p1.rows);
                keep_states.push(bufs(l1).ring.view(p1.base * SLOT, span * SLOT));
                keep_states.push(bufs(l0).ring.view(p0.base * SLOT, span * SLOT));
            }
            Some(Control::SwapLive) => {
                keep_states.push(verify[1].1.live.sub_offset(0, SLOT));
                keep_states.push(verify[0].1.live.sub_offset(0, SLOT));
            }
            Some(Control::WrongFrame)
            | Some(Control::RowsShort)
            | Some(Control::ArWrongFrame)
            | Some(Control::WrongCursor) => {}
        }
    }
    // (the views of the rows control were pushed above)
    if verify_lanes.len() >= 2 {
        match control {
            Some(Control::SwapRing) => {
                verify_lanes[0].step.row_states = Some(&keep_states[0]);
                verify_lanes[1].step.row_states = Some(&keep_states[1]);
            }
            Some(Control::SwapLive) => {
                verify_lanes[0].step.state = &keep_states[0];
                verify_lanes[1].step.state = &keep_states[1];
            }
            Some(Control::WrongFrame) => {
                let last = verify_lanes.len() - 1;
                verify_lanes[last].step.position += 1;
            }
            Some(Control::WrongCursor) => {
                let last = verify_lanes.len() - 1;
                let c = &mut verify_lanes[last].conv;
                c.start_cursor = (c.start_cursor + 1) % RING;
            }
            Some(Control::RowsShort) => {
                // Lane 0 claims rows - 1 everywhere (the views shrink with
                // it); its data still holds the full window.
                let v = &verify[0].1;
                let r = v.rows;
                if r >= 3 {
                    let lb = &arm.lanes[verify[0].0];
                    let rr = r - 1;
                    rows_short_views.push(lb.inputs.view(0, rr * QKV));
                    rows_short_views.push(lb.recurrence.view(0, rr * QKV));
                    rows_short_views.push(lb.recurrence.view(rr * QKV, rr * NV));
                    rows_short_views.push(lb.recurrence.view(rr * QKV + rr * NV, rr * NV));
                    rows_short_views.push(v.proj.sub_offset(0, rr * QKV));
                    rows_short_views.push(v.a.sub_offset(0, rr * NV));
                    rows_short_views.push(v.b.sub_offset(0, rr * NV));
                    rows_short_views.push(v.rec.sub_offset(0, rr * VALUE));
                    rows_short_views.push(v.z.sub_offset(0, rr * VALUE));
                    rows_short_views.push(v.out.sub_offset(0, rr * VALUE));
                    rows_short_views.push(lb.ring.view(plan[verify[0].0].base * SLOT, rr * SLOT));
                    let s = &rows_short_views;
                    let l = &mut verify_lanes[0];
                    l.projection = &s[4];
                    l.conv.input = &s[0];
                    l.conv.output = &s[1];
                    l.conv.rows = rr;
                    l.params.a = &s[5];
                    l.params.b = &s[6];
                    l.params.gate = &s[2];
                    l.params.beta = &s[3];
                    l.params.rows = rr;
                    l.step.projection = &s[1];
                    l.step.gate = &s[2];
                    l.step.beta = &s[3];
                    l.step.output = &s[7];
                    l.step.row_states = Some(&s[10]);
                    l.step.rows = rr;
                    l.gate.recurrent_output = &s[7];
                    l.gate.z = &s[8];
                    l.gate.output = &s[9];
                    l.gate.rows = rr;
                }
            }
            Some(Control::ArWrongFrame) | None => {}
        }
    }
    if control == Some(Control::ArWrongFrame) && ar_lanes.len() >= 2 {
        let last = ar_lanes.len() - 1;
        ar_lanes[last].step.position += 1;
    }
    let mut grouping = Grouping::default();
    let ar_plan = gated_delta_core_lanes(gpu, &ar_lanes, NV).map_err(|e| e.to_string())?;
    let verify_plan =
        gated_delta_core_lanes_rows(gpu, &verify_lanes, NV).map_err(|e| e.to_string())?;
    // A refusal of an admissible set is a finding, but the step still runs
    // (per lane) so the state the later steps compare against stays defined.
    if ar_lanes.len() >= 2 && ar_plan.is_none() {
        grouping
            .refusals
            .push("the AR lanes route refused an admitted AR set".into());
    }
    if verify_lanes.len() >= 2 && verify_plan.is_none() {
        grouping
            .refusals
            .push("the verify lanes route refused an admitted verify set".into());
    }
    if ar_lanes.len() < 2 && ar_plan.is_some() || verify_lanes.len() < 2 && verify_plan.is_some() {
        grouping.refusals.push("a lone lane was admitted".into());
    }
    if let Some(plan) = ar_plan {
        grouping.ar_grouped = true;
        // As `execute_gated_delta_net_core_lanes` launches it: `launch_rotated`
        // (the fused one-launch kernel under HIPFIRE_QWEN4_GDN_LANES_FUSE), with the
        // output projection's rotation scratch only for a group of AR lanes alone.
        let rot_all = arm.rot.view(0, arm.cap * VALUE);
        let rotation = (rotate && verify_lanes.is_empty()).then_some(&rot_all);
        plan.launch_rotated(gpu, rotation)
            .map_err(|e| e.to_string())?;
        gpu.scratch.prerotated = None;
    } else {
        for (l, (_, v)) in ar_lanes.iter().zip(&ar) {
            ar_per_lane(gpu, l, &v.rot, rotate);
        }
    }
    if let Some(plan) = verify_plan {
        grouping.verify_grouped = true;
        plan.launch(gpu).map_err(|e| e.to_string())?;
    } else {
        for (l, (_, v)) in verify_lanes.iter().zip(&verify) {
            verify_per_lane(gpu, l, &v.rot, rotate);
        }
    }
    grouping.ar_lanes = ar_lanes.len();
    grouping.verify_lanes = verify_lanes.len();
    Ok(grouping)
}

/// Launch kinds the interpreter's grouping issues for `rows` lanes under
/// `table_order`: `(kernel launches, device copies)` of the per-lane chain and
/// of the grouped one (`fuse`: the one-row group is the fused one-launch kernel).
fn expected_counts(rows: &[usize], fuse: bool) -> ((u64, u64), (u64, u64)) {
    let ar = rows.iter().filter(|&&r| r == 1).count() as u64;
    let verify = rows.iter().filter(|&&r| r >= 2).count() as u64;
    let per_lane = (2 * ar + 3 * verify, verify);
    // Two or more AR lanes: the fused one-launch kernel (HIPFIRE_QWEN4_GDN_LANES_FUSE,
    // beta's default) or the two incumbent lane launches.
    let ar_launches = if ar >= 2 {
        if fuse {
            1
        } else {
            2
        }
    } else {
        2 * ar
    };
    let (verify_launches, verify_copies) = if verify >= 2 {
        (3, 0)
    } else {
        (3 * verify, verify)
    };
    (per_lane, (ar_launches + verify_launches, verify_copies))
}

/// [`expected_counts`] where the verify lanes flagged in `alias` (by logical lane) carry their
/// projection rows in their capture `inputs`: the per-lane chain issues no capture copy for
/// them (a lone verify lane takes that chain in the grouped arm too; the grouped kernel's own
/// copy is no dtod either way).
fn expected_counts_aliased(rows: &[usize], fuse: bool, alias: &[bool]) -> ((u64, u64), (u64, u64)) {
    let (mut per_lane, mut grouped) = expected_counts(rows, fuse);
    let verify = rows.iter().filter(|&&r| r >= 2).count();
    let aliased = rows
        .iter()
        .zip(alias)
        .filter(|(&r, &a)| r >= 2 && a)
        .count() as u64;
    per_lane.1 -= aliased;
    if verify < 2 {
        grouped.1 -= aliased;
    }
    (per_lane, grouped)
}

// ---------------------------------------------------------------------------
// Snapshots and comparison
// ---------------------------------------------------------------------------

struct Snap {
    /// `(name, bytes)` of every compared item, in a fixed order.
    items: Vec<(String, Vec<u8>)>,
    /// Guard bytes that lost their sentinel, per buffer.
    guard_bad: Vec<(String, usize)>,
}

fn snapshot(gpu: &Gpu, arm: &Arm) -> Snap {
    gpu.hip.device_synchronize().expect("sync before download");
    let mut items = Vec::new();
    let mut guard_bad = Vec::new();
    let shared: [(&str, &Guarded); 9] = [
        ("proj", &arm.projection),
        ("conv_out_ar", &arm.projection2),
        ("a", &arm.a),
        ("b", &arm.b),
        ("gate_ar", &arm.gate),
        ("beta_ar", &arm.beta),
        ("recurrent", &arm.recurrent),
        ("z", &arm.z),
        ("gdn_out", &arm.out),
    ];
    for (name, g) in shared {
        let all = g.download(gpu);
        guard_bad.push((name.to_string(), g.guard_bad(&all)));
        items.push((name.to_string(), g.body(&all).to_vec()));
    }
    // The rotation scratch is dropped by the lanes routes: only its guards.
    let all = arm.rot.download(gpu);
    guard_bad.push(("rot".to_string(), arm.rot.guard_bad(&all)));
    for (i, l) in arm.lanes.iter().enumerate() {
        for (name, g) in [
            ("hist", &l.history),
            ("ring", &l.ring),
            ("capture_inputs", &l.inputs),
            ("capture_recurrence", &l.recurrence),
        ] {
            let all = g.download(gpu);
            guard_bad.push((format!("lane{i}.{name}"), g.guard_bad(&all)));
            items.push((format!("lane{i}.{name}"), g.body(&all).to_vec()));
        }
    }
    Snap { items, guard_bad }
}

/// `(name, mismatching bytes)` of the items that differ.
fn diff(x: &Snap, y: &Snap) -> Vec<(String, usize)> {
    assert_eq!(x.items.len(), y.items.len(), "snapshot shape");
    let mut bad = Vec::new();
    for ((name, p), (other, q)) in x.items.iter().zip(&y.items) {
        assert_eq!(name, other, "snapshot order");
        let n = p.iter().zip(q).filter(|(a, b)| a != b).count() + p.len().abs_diff(q.len());
        if n != 0 {
            bad.push((name.clone(), n));
        }
    }
    bad
}

/// Bytes of each lane's ring that changed between `before` and `after` outside
/// the one slot the lane's step may write (a verify's captured slot, an AR
/// lane's live slot).
fn stray_ring_writes(before: &Snap, after: &Snap, plan: &[LaneStep]) -> u64 {
    fn ring_of<'a>(s: &'a Snap, name: &str) -> &'a [u8] {
        s.items
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b.as_slice())
            .expect("ring item")
    }
    let mut bad = 0u64;
    for (l, p) in plan.iter().enumerate() {
        let name = format!("lane{l}.ring");
        let (x, y) = (ring_of(before, &name), ring_of(after, &name));
        let writable = if p.rows >= 2 {
            p.base + p.rows - 1
        } else {
            p.live
        };
        for slot in 0..2 * RING_ROWS {
            if slot == writable {
                continue;
            }
            let r = slot * SLOT..(slot + 1) * SLOT;
            bad += x[r.clone()]
                .iter()
                .zip(&y[r])
                .filter(|(a, b)| a != b)
                .count() as u64;
        }
    }
    bad
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

struct CaseSpec {
    name: String,
    n: usize,
    pattern: &'static str,
    variant: &'static str,
    pool: &'static str,
    /// Rows `[lane][step]`.
    rows: Vec<Vec<usize>>,
    /// Lanes TABLE order of the grouped launches.
    table_order: Vec<usize>,
    /// Packed memory order of the lane segments.
    seg_order: Vec<usize>,
    /// Launch order of the per-lane reference.
    ref_order: Vec<usize>,
    positions: Vec<usize>,
    steps: usize,
    control: Option<Control>,
    rotate: bool,
    /// Every even logical lane's verify steps carry their projection rows in their capture
    /// `inputs` (the in-projection wrote the capture in place); odd lanes copy as before, so
    /// grouped sets mix aliased and copied lanes.
    alias: bool,
}

struct CaseResult {
    spec_name: String,
    pass: bool,
    bad_bytes: u64,
    bad_items: Vec<String>,
    guard_bad: u64,
    stray_bad: u64,
    launches_ok: bool,
    /// The control had nothing to corrupt (too few verify lanes / rows).
    skipped: bool,
    /// Steps that took no grouped route (no class held two lanes).
    per_lane_steps: usize,
    /// Distinct routes the steps took (`rows-lanes`, `ar-lanes`,
    /// `rows-lanes+ar-lanes`, `per-lane`), comma separated.
    route: String,
    errors: Vec<String>,
    detail: Value,
}

fn run_case(gpu: &mut Gpu, w: &Weights, rng: &mut Rng, spec: &CaseSpec) -> CaseResult {
    let host: Vec<LaneHost> = (0..spec.n)
        .map(|l| make_lane_host(rng, spec.positions[l], &spec.rows[l]))
        .collect();
    // One row of slack: the rows control lays the candidate's segments out by
    // the claimed rows, so a lane's full view may end one row past the packed total.
    let cap = spec.n * GDN_CORE_LANE_ROWS + 1;
    let alias_mask: Vec<bool> = (0..spec.n).map(|l| spec.alias && l % 2 == 0).collect();
    let ref_arm = Arm::new(gpu, cap, &host, &alias_mask);
    let cand_arm = Arm::new(gpu, cap, &host, &alias_mask);
    let carry0: Vec<Carry> = host
        .iter()
        .map(|h| Carry {
            pos: h.base_pos,
            live: 0,
        })
        .collect();
    let mut ref_carry = carry0.clone();
    let mut cand_carry = carry0;
    let mut bad_bytes = 0u64;
    let mut bad_items: Vec<String> = Vec::new();
    let mut guard_bad = 0u64;
    let mut stray_bad = 0u64;
    let mut counts = ((0u64, 0u64), (0u64, 0u64));
    let mut applied_steps = 0usize;
    let mut errors: Vec<String> = Vec::new();
    let mut grouped_steps = (0usize, 0usize);
    let mut routes: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    let mut per_lane_steps = 0usize;
    for step in 0..spec.steps {
        let rows: Vec<usize> = (0..spec.n).map(|l| spec.rows[l][step]).collect();
        let ref_plan = plan_step(&rows, &spec.seg_order, &ref_carry);
        let cand_plan = plan_step(&rows, &spec.seg_order, &cand_carry);
        // The verify lanes of the table order decide whether a control has
        // anything to corrupt this step (two verify lanes, and for the rows
        // control a first one with at least three rows).
        let verify_ids: Vec<usize> = spec
            .table_order
            .iter()
            .copied()
            .filter(|&l| rows[l] >= 2)
            .collect();
        let applies = match spec.control {
            Some(Control::RowsShort) => verify_ids.len() >= 2 && rows[verify_ids[0]] >= 3,
            Some(Control::ArWrongFrame) => rows.iter().filter(|&&r| r == 1).count() >= 2,
            Some(_) => verify_ids.len() >= 2,
            None => false,
        };
        applied_steps += usize::from(applies);
        let step_control = if applies { spec.control } else { None };
        // The rows control: the candidate's first verify lane claims one row
        // fewer, its segment shrinks with the claim, and its data (still the
        // full window) spills into the next segment, which is staged after.
        let mut cand_layout = cand_plan.clone();
        if applies && spec.control == Some(Control::RowsShort) {
            let mut claim_rows = rows.clone();
            claim_rows[verify_ids[0]] -= 1;
            let claimed = plan_step(&claim_rows, &spec.seg_order, &cand_carry);
            for (layout, c) in cand_layout.iter_mut().zip(&claimed) {
                layout.seg = c.seg;
            }
        }

        ref_arm.stage(gpu, &host, step, &seg_of(&ref_plan), &rows);
        let m0 = memory_effects::snapshot();
        run_reference(gpu, w, &ref_arm, &ref_plan, &spec.ref_order, spec.rotate);
        let d = memory_effects::snapshot().since(m0);
        counts.0 .0 += d.launches;
        counts.0 .1 += d.dtod;
        let want = snapshot(gpu, &ref_arm);

        let cand_seg = seg_of(&cand_layout);
        cand_arm.stage(gpu, &host, step, &cand_seg, &rows);
        let before = snapshot(gpu, &cand_arm);
        let m0 = memory_effects::snapshot();
        let views_plan = &cand_layout;
        match run_candidate(
            gpu,
            w,
            &cand_arm,
            views_plan,
            &spec.table_order,
            spec.rotate,
            step_control,
        ) {
            Ok(g) => {
                grouped_steps.0 += usize::from(g.ar_grouped);
                grouped_steps.1 += usize::from(g.verify_grouped);
                for r in &g.refusals {
                    errors.push(format!("step{step}: {r}"));
                }
                // A set of two or more lanes of one class must take its
                // route (admission has no layout-dependent refusal), a lone
                // lane or a mixed group never merges classes.
                let (ar_n, verify_n) = (
                    rows.iter().filter(|&&r| r == 1).count(),
                    rows.iter().filter(|&&r| r >= 2).count(),
                );
                if g.ar_lanes != ar_n
                    || g.verify_lanes != verify_n
                    || g.ar_grouped != (ar_n >= 2)
                    || g.verify_grouped != (verify_n >= 2)
                {
                    errors.push(format!(
                        "step{step}: route {} for {ar_n} AR + {verify_n} verify lanes",
                        g.route()
                    ));
                }
                per_lane_steps += usize::from(g.route() == "per-lane");
                routes.insert(g.route());
            }
            Err(e) => errors.push(format!("step{step}: {e}")),
        }
        let d = memory_effects::snapshot().since(m0);
        counts.1 .0 += d.launches;
        counts.1 .1 += d.dtod;
        let got = snapshot(gpu, &cand_arm);
        if spec.control.is_none() {
            stray_bad += stray_ring_writes(&before, &got, &cand_plan);
        }

        for (name, n) in diff(&want, &got) {
            bad_bytes += n as u64;
            if bad_items.len() < 24 {
                bad_items.push(format!("step{step}.{name}:{n}"));
            }
        }
        for s in [&want, &got] {
            guard_bad += s.guard_bad.iter().map(|(_, n)| *n as u64).sum::<u64>();
        }
        commit(&ref_plan, &mut ref_carry);
        commit(&cand_plan, &mut cand_carry);
    }
    ref_arm.free(gpu);
    cand_arm.free(gpu);
    // Aggregate launch counts: sum the per-step expectations.
    let fuse = gpu.flags.qwen4_gdn_lanes_fuse_enabled();
    let (mut want_ref, mut want_cand) = ((0u64, 0u64), (0u64, 0u64));
    for step in 0..spec.steps {
        let rows: Vec<usize> = (0..spec.n).map(|l| spec.rows[l][step]).collect();
        let (p, g) = expected_counts_aliased(&rows, fuse, &alias_mask);
        want_ref = (want_ref.0 + p.0, want_ref.1 + p.1);
        want_cand = (want_cand.0 + g.0, want_cand.1 + g.1);
    }
    let skipped = spec.control.is_some() && applied_steps == 0;
    let launches_ok = counts.0 == want_ref && counts.1 == want_cand;
    let detected = bad_bytes > 0;
    let pass = match spec.control {
        None => {
            bad_bytes == 0 && guard_bad == 0 && stray_bad == 0 && launches_ok && errors.is_empty()
        }
        // Nothing to corrupt in any step (too few verify lanes or rows).
        Some(_) if skipped => errors.is_empty(),
        Some(_) => detected && launches_ok && errors.is_empty(),
    };
    let detail = json!({
        "name": spec.name,
        "lanes": spec.n,
        "pattern": spec.pattern,
        "variant": spec.variant,
        "pool": spec.pool,
        "steps": spec.steps,
        "rows": spec.rows,
        "table_order": spec.table_order,
        "segment_order": spec.seg_order,
        "ref_launch_order": spec.ref_order,
        "positions": spec.positions,
        "rotate_gate": spec.rotate,
        "control": spec.control.map(|c| c.name()),
        "control_skipped": skipped,
        "pass": pass,
        "bad_bytes": bad_bytes,
        "bad_items": bad_items,
        "guard_bytes_overwritten": guard_bad,
        "stray_ring_bytes_written": stray_bad,
        "errors": errors,
        "grouped_steps": {"ar": grouped_steps.0, "verify": grouped_steps.1},
        // Steps no class of which has two lanes (a group of one lane never forms:
        // `execute_window` groups two or more steps): per lane by design.
        "per_lane_steps": per_lane_steps,
        "routes": routes.iter().copied().collect::<Vec<_>>(),
        "launches": {
            "per_lane": {"kernels": counts.0.0, "copies": counts.0.1,
                         "expected_kernels": want_ref.0, "expected_copies": want_ref.1},
            "grouped": {"kernels": counts.1.0, "copies": counts.1.1,
                        "expected_kernels": want_cand.0, "expected_copies": want_cand.1},
        },
    });
    CaseResult {
        spec_name: spec.name.clone(),
        pass,
        bad_bytes,
        bad_items,
        guard_bad,
        stray_bad,
        launches_ok,
        skipped,
        per_lane_steps,
        route: routes.iter().copied().collect::<Vec<_>>().join(","),
        errors,
        detail,
    }
}

fn seg_of(plan: &[LaneStep]) -> Vec<usize> {
    plan.iter().map(|p| p.seg).collect()
}

fn print_case(r: &CaseResult, control: bool) {
    let verdict = if r.pass { "PASS" } else { "FAIL" };
    if control {
        println!(
            "control {:<44} {verdict} {} route={} (bad_bytes={}{})",
            r.spec_name,
            if r.skipped { "skipped" } else { "detected" },
            r.route,
            r.bad_bytes,
            if r.launches_ok { "" } else { " WRONG COUNT" }
        );
    } else {
        println!(
            "case    {:<44} {verdict} route={}{} bad_bytes={} guard_bad={} stray_ring={}{}{}{}",
            r.spec_name,
            r.route,
            if r.per_lane_steps == 0 {
                String::new()
            } else {
                format!(" per_lane_steps={}", r.per_lane_steps)
            },
            r.bad_bytes,
            r.guard_bad,
            r.stray_bad,
            if r.launches_ok { "" } else { " WRONG COUNT" },
            if r.errors.is_empty() {
                String::new()
            } else {
                format!(" errors={:?}", &r.errors[..r.errors.len().min(2)])
            },
            if r.bad_items.is_empty() {
                String::new()
            } else {
                format!(" first={:?}", &r.bad_items[..r.bad_items.len().min(4)])
            }
        );
    }
}

// ---------------------------------------------------------------------------
// Admission probes
// ---------------------------------------------------------------------------

/// A probe arm: `n` lanes with fixed rows, packed segments, live slot 0.
struct ProbeSet {
    arm: Arm,
    plan: Vec<LaneStep>,
}

fn probe_set(gpu: &mut Gpu, rng: &mut Rng, rows: &[usize]) -> ProbeSet {
    let host: Vec<LaneHost> = rows.iter().map(|&r| make_lane_host(rng, 5, &[r])).collect();
    let arm = Arm::new(gpu, rows.len() * RING_ROWS, &host, &vec![false; rows.len()]);
    let carry: Vec<Carry> = rows
        .iter()
        .enumerate()
        .map(|(i, _)| Carry {
            pos: 5 + i,
            live: 0,
        })
        .collect();
    let order: Vec<usize> = (0..rows.len()).collect();
    let plan = plan_step(rows, &order, &carry);
    ProbeSet { arm, plan }
}

fn probe(
    gpu: &Gpu,
    name: &str,
    lanes: &[GdnCoreRowsLane<'_>],
    expect_some: bool,
    out: &mut Vec<Value>,
) -> bool {
    let got = gated_delta_core_lanes_rows(gpu, lanes, NV).expect("admission");
    let ok = got.is_some() == expect_some;
    println!(
        "probe   {:<44} {} (admitted={}, expected={})",
        name,
        if ok { "PASS" } else { "FAIL" },
        got.is_some(),
        expect_some
    );
    out.push(json!({
        "name": name, "pass": ok, "admitted": got.is_some(), "expected_admitted": expect_some,
    }));
    ok
}

fn admission_probes(gpu: &mut Gpu, w: &Weights, w2: &Weights, rng: &mut Rng) -> (bool, Vec<Value>) {
    // Lanes 0 and 7 carry the same row count (two), so a field of one can be
    // swapped onto the other without tripping a shape check.
    let rows = [2usize, 3, 4, 5, 6, 7, 8, 2, 3];
    let set = probe_set(gpu, rng, &rows);
    let views: Vec<Vw> = (0..rows.len())
        .map(|l| Vw::new(&set.arm, l, &set.plan[l]))
        .collect();
    let lanes_of = |n: usize| -> Vec<GdnCoreRowsLane<'_>> {
        views[..n].iter().map(|v| verify_lane(v, w)).collect()
    };
    let lanes_at = |idx: &[usize]| -> Vec<GdnCoreRowsLane<'_>> {
        idx.iter().map(|&i| verify_lane(&views[i], w)).collect()
    };
    let mut out = Vec::new();
    let mut ok = true;
    let mut check = |gpu: &Gpu, name: &str, lanes: &[GdnCoreRowsLane<'_>], some: bool| {
        ok &= probe(gpu, name, lanes, some, &mut out);
    };
    check(gpu, "admit_2_lanes", &lanes_of(2), true);
    check(gpu, "admit_8_lanes_ragged_rows", &lanes_of(8), true);
    check(gpu, "refuse_1_lane", &lanes_of(1), false);
    check(gpu, "refuse_9_lanes", &lanes_of(9), false);
    {
        // A 9-row lane (the ring half holds 9).
        let nine = probe_set(gpu, rng, &[9, 2]);
        let nv: Vec<Vw> = (0..2)
            .map(|i| Vw::new(&nine.arm, i, &nine.plan[i]))
            .collect();
        let nl: Vec<GdnCoreRowsLane<'_>> = nv.iter().map(|v| verify_lane(v, w)).collect();
        check(gpu, "refuse_9_row_lane", &nl, false);
        drop(nl);
        drop(nv);
        nine.arm.free(gpu);
    }
    {
        // A one-row lane is an AR lane (the per-row branch), never this route.
        let one = probe_set(gpu, rng, &[1, 2]);
        let ov: Vec<Vw> = (0..2).map(|i| Vw::new(&one.arm, i, &one.plan[i])).collect();
        let ol: Vec<GdnCoreRowsLane<'_>> = ov.iter().map(|v| verify_lane(v, w)).collect();
        check(gpu, "refuse_1_row_lane", &ol, false);
        drop(ol);
        drop(ov);
        one.arm.free(gpu);
    }
    {
        let mut l = lanes_of(3);
        l[1].step.row_states = None;
        check(gpu, "refuse_unarmed_lane_no_ring", &l, false);
    }
    {
        let mut l = lanes_at(&[0, 7]);
        l[1].step.row_states = l[0].step.row_states;
        check(gpu, "refuse_duplicate_ring", &l, false);
    }
    {
        let mut l = lanes_of(3);
        l[1].conv.history = &views[0].hist;
        l[1].conv.next_history = &views[0].hist;
        check(gpu, "refuse_duplicate_history", &l, false);
    }
    {
        let mut l = lanes_of(3);
        l[1].step.state = &views[0].live;
        check(gpu, "refuse_duplicate_live_state", &l, false);
    }
    {
        let mut l = lanes_at(&[0, 7]);
        l[1].gate.output = &views[0].out;
        check(gpu, "refuse_duplicate_gdn_output", &l, false);
    }
    {
        let mut l = lanes_at(&[0, 7]);
        l[1].conv.input = &views[0].inputs;
        check(gpu, "refuse_duplicate_capture_inputs", &l, false);
    }
    {
        let mut l = lanes_of(3);
        l[1].conv.kernel = &w2.conv;
        check(gpu, "refuse_other_conv_weight", &l, false);
    }
    {
        let mut l = lanes_of(3);
        l[1].gate.norm = &w2.norm;
        check(gpu, "refuse_other_norm_weight", &l, false);
    }
    {
        let mut l = lanes_of(3);
        l[1].params.a_log = &w2.a_log;
        check(gpu, "refuse_other_a_log", &l, false);
    }
    {
        let mut l = lanes_of(3);
        l[1].params.dt_bias = &w2.dt_bias;
        check(gpu, "refuse_other_dt_bias", &l, false);
    }
    {
        let mut l = lanes_of(3);
        l[1].step.projection = &views[1].inputs;
        check(gpu, "refuse_step_input_not_conv_output", &l, false);
    }
    {
        let mut l = lanes_of(3);
        l[1].step.gate = &views[1].beta_c;
        check(gpu, "refuse_step_gate_not_param_gate", &l, false);
    }
    {
        let mut l = lanes_of(3);
        l[1].gate.recurrent_output = &views[1].out;
        check(gpu, "refuse_recurrent_output_mismatch", &l, false);
    }
    {
        // Gate scratch of one lane far from the others (AR lanes sit between
        // verify lanes in a mixed group): the gate lanes kernel needs no
        // adjacency, so the set is admitted.
        let far = set.arm.out.view(60 * VALUE, views[1].rows * VALUE);
        let mut l = lanes_of(3);
        l[1].gate.output = &far;
        check(gpu, "admit_nonadjacent_gdn_output", &l, true);
    }
    {
        // A lane whose convolution is consistent at one row fewer than its step.
        let rr = views[1].rows - 1;
        let input = views[1].inputs.sub_offset(0, rr * QKV);
        let output = views[1].conv_out.sub_offset(0, rr * QKV);
        let mut l = lanes_of(3);
        l[1].conv.input = &input;
        l[1].conv.output = &output;
        l[1].conv.rows = rr;
        check(gpu, "refuse_rows_mismatch_conv_vs_step", &l, false);
    }
    {
        // A lane reading rows another lane writes (its z is lane 0's gdn
        // output span, stretched over its own rows).
        let z = set
            .arm
            .out
            .view(set.plan[0].seg * VALUE, views[1].rows * VALUE);
        let mut l = lanes_of(3);
        l[1].gate.z = &z;
        check(gpu, "refuse_read_of_another_lanes_write", &l, false);
    }
    {
        let l = lanes_of(2);
        Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes_rows = false;
        check(gpu, "refuse_rows_flag_off", &l, false);
        Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes_rows = true;
        check(gpu, "admit_after_rows_flag_back_on", &l, true);
        // The rows flag needs the lanes flag too (HIPFIRE_QWEN4_GDN_LANES=0 kills both).
        Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes = false;
        check(gpu, "refuse_lanes_flag_off", &l, false);
        Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes = true;
        check(gpu, "admit_after_lanes_flag_back_on", &l, true);
    }
    drop(views);
    set.arm.free(gpu);
    (ok, out)
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Microseconds per call of `f`, event timed over `iters` calls (one run).
fn time_run(gpu: &mut Gpu, iters: usize, f: &mut impl FnMut(&mut Gpu)) -> f64 {
    gpu.hip.device_synchronize().expect("sync before timing");
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..iters {
        f(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("synchronize stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64;
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    ms * 1000.0 / iters as f64
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    let m = s.len() / 2;
    if s.len() % 2 == 1 {
        s[m]
    } else {
        (s[m - 1] + s[m]) / 2.0
    }
}

/// One timed window: the lanes' `rows` at one packed layer step, the per-lane
/// chain against the grouped launches (`runs` alternating runs of `iters`
/// replays each, after warmup; medians), with the launch counts one layer step
/// issues measured by the memory-effects counters.
fn bench(
    gpu: &mut Gpu,
    w: &Weights,
    rng: &mut Rng,
    rows: &[usize],
    iters: usize,
    runs: usize,
) -> Value {
    let n = rows.len();
    let host: Vec<LaneHost> = (0..n)
        .map(|l| make_lane_host(rng, 3 + l, &[rows[l]]))
        .collect();
    let arm = Arm::new(gpu, n * GDN_CORE_LANE_ROWS + 1, &host, &vec![false; n]);
    let carry: Vec<Carry> = (0..n)
        .map(|l| Carry {
            pos: 3 + l,
            live: 0,
        })
        .collect();
    let order: Vec<usize> = (0..n).collect();
    let plan = plan_step(rows, &order, &carry);
    arm.stage(gpu, &host, 0, &seg_of(&plan), rows);
    let mut per_lane = |gpu: &mut Gpu| run_reference(gpu, w, &arm, &plan, &order, false);
    let mut grouped = |gpu: &mut Gpu| {
        run_candidate(gpu, w, &arm, &plan, &order, false, None).expect("grouped timing run");
    };
    // Launches per layer step and the route the grouped arm took.
    let m0 = memory_effects::snapshot();
    per_lane(gpu);
    let d_per = memory_effects::snapshot().since(m0);
    let m0 = memory_effects::snapshot();
    let route = run_candidate(gpu, w, &arm, &plan, &order, false, None)
        .expect("grouped counting run")
        .route();
    let d_grouped = memory_effects::snapshot().since(m0);
    for _ in 0..5 {
        per_lane(gpu);
        grouped(gpu);
    }
    let (mut per_runs, mut grouped_runs) = (Vec::new(), Vec::new());
    for _ in 0..runs {
        per_runs.push(time_run(gpu, iters, &mut per_lane));
        grouped_runs.push(time_run(gpu, iters, &mut grouped));
    }
    arm.free(gpu);
    let (per, want) = expected_counts(rows, gpu.flags.qwen4_gdn_lanes_fuse_enabled());
    let (per_us, grouped_us) = (median(&per_runs), median(&grouped_runs));
    json!({
        "lanes": n,
        "rows": rows,
        "total_rows": rows.iter().sum::<usize>(),
        "iters_per_run": iters,
        "runs": runs,
        "grouped_route": route,
        "per_lane_us_per_layer_median": per_us,
        "grouped_us_per_layer_median": grouped_us,
        "per_lane_us_runs": per_runs,
        "grouped_us_runs": grouped_runs,
        "speedup": per_us / grouped_us,
        "per_lane_launches": {"kernels": d_per.launches, "copies": d_per.dtod,
                              "expected_kernels": per.0, "expected_copies": per.1},
        "grouped_launches": {"kernels": d_grouped.launches, "copies": d_grouped.dtod,
                             "expected_kernels": want.0, "expected_copies": want.1},
    })
}

fn print_bench(t: &Value) {
    println!(
        "bench   N={} rows={} ({} rows): per-lane {:.1} us, grouped {:.1} us per layer ({:.2}x, median of {} x {} iters) route={} launches per-lane {}+{}cp grouped {}+{}cp",
        t["lanes"],
        t["rows"],
        t["total_rows"],
        t["per_lane_us_per_layer_median"].as_f64().unwrap_or(0.0),
        t["grouped_us_per_layer_median"].as_f64().unwrap_or(0.0),
        t["speedup"].as_f64().unwrap_or(0.0),
        t["runs"],
        t["iters_per_run"],
        t["grouped_route"],
        t["per_lane_launches"]["kernels"],
        t["per_lane_launches"]["copies"],
        t["grouped_launches"]["kernels"],
        t["grouped_launches"]["copies"],
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

struct Cli {
    out: Option<std::path::PathBuf>,
    steps: usize,
    lanes: Vec<usize>,
    seed: u64,
    controls: bool,
    timing: bool,
    time_rows: Vec<usize>,
    reps: usize,
    /// `--bench ITERS`: only the per-layer timing table (8 lanes x `time_rows`
    /// plus the mixed window), ITERS replays per run, `bench_runs` runs.
    bench: Option<usize>,
    bench_runs: usize,
    /// `--alias-capture`: every even lane's verify rows are projected into its capture `inputs`
    /// in place (`HIPFIRE_QWEN4_GDN_CAPTURE_DIRECT`), so grouped sets mix aliased and copied
    /// lanes; the matrix only (no controls, no timing).
    alias: bool,
}

fn parse_list(s: &str, what: &str, max: usize) -> Vec<usize> {
    let v: Vec<usize> = s
        .split(',')
        .map(|x| x.trim().parse().unwrap_or_else(|_| usage(what)))
        .collect();
    if v.iter().any(|&n| !(1..=max).contains(&n)) {
        usage(&format!("{what} entries must be in 1..={max}"));
    }
    v
}

fn parse_cli() -> Cli {
    let mut cli = Cli {
        out: None,
        steps: 4,
        lanes: vec![1, 2, 3, 4, 8],
        seed: 0x6D6C_B0A5,
        controls: true,
        timing: true,
        time_rows: vec![2, 3, 4],
        reps: 50,
        bench: None,
        bench_runs: 7,
        alias: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |what: &str| {
            args.next()
                .unwrap_or_else(|| usage(&format!("{what} needs a value")))
        };
        match a.as_str() {
            "--out" => cli.out = Some(value("--out").into()),
            "--steps" => {
                cli.steps = value("--steps")
                    .parse()
                    .unwrap_or_else(|_| usage("--steps"))
            }
            "--seed" => cli.seed = value("--seed").parse().unwrap_or_else(|_| usage("--seed")),
            "--reps" => cli.reps = value("--reps").parse().unwrap_or_else(|_| usage("--reps")),
            "--lanes" => cli.lanes = parse_list(&value("--lanes"), "--lanes", GDN_CORE_LANES_MAX),
            "--time-rows" => {
                cli.time_rows = parse_list(&value("--time-rows"), "--time-rows", GDN_CORE_LANE_ROWS)
            }
            "--bench" => {
                cli.bench = Some(
                    value("--bench")
                        .parse()
                        .unwrap_or_else(|_| usage("--bench")),
                );
                cli.time_rows = vec![1, 2, 4];
            }
            "--bench-runs" => {
                cli.bench_runs = value("--bench-runs")
                    .parse()
                    .unwrap_or_else(|_| usage("--bench-runs"))
            }
            "--skip-controls" => cli.controls = false,
            "--skip-timing" => cli.timing = false,
            "--alias-capture" => {
                cli.alias = true;
                cli.controls = false;
                cli.timing = false;
            }
            "-h" | "--help" => usage(""),
            other => usage(&format!("unknown argument {other}")),
        }
    }
    if cli.steps == 0 || cli.reps == 0 {
        usage("--steps and --reps must be >= 1");
    }
    if cli.bench == Some(0) || cli.bench_runs < 5 {
        usage("--bench ITERS must be >= 1 and --bench-runs >= 5");
    }
    cli
}

fn usage(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("gdn_lanes_rows_oracle: {msg}");
    }
    eprintln!(
        "usage: gdn_lanes_rows_oracle [--out DIR] [--steps S=4] [--lanes 1,2,3,4,8] [--seed U64] \
         [--skip-controls] [--skip-timing] [--time-rows 2,3,4] [--reps 50] [--alias-capture] \
         [--bench ITERS [--bench-runs 7]]"
    );
    std::process::exit(if msg.is_empty() { 0 } else { 2 });
}

/// Base positions of lane i for a pool; rotated by case so lanes see different
/// cursors (`position % 3`) and frames. `steps` keeps every frame within u32
/// (every step advances a lane by at most 8 rows).
fn pool(kind: &str, steps: usize) -> Vec<usize> {
    match kind {
        // Small, every cursor value, ragged.
        "small" => vec![0, 1, 2, 3, 4, 5, 6, 7],
        "mixed" => vec![7, 100, 2, 99, 12345, 3, 4096, 1],
        // u16 boundary, 2^18, > 2^20, 2^24, near the u32 end (the last row of
        // the last step lands exactly on u32::MAX).
        "large" => vec![
            65_534,
            65_535,
            65_536,
            262_143,
            1_000_003,
            16_777_215,
            4_294_967_000,
            u32::MAX as usize - (GDN_CORE_LANE_ROWS * steps - 1),
        ],
        // Every lane at the same position: lanes differ only by data.
        "same" => vec![5; 8],
        _ => unreachable!(),
    }
}

fn shuffled(rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        v.swap(i, rng.below(i + 1));
    }
    v
}

/// Rows `[lane][step]` of a pattern.
fn rows_for(pattern: &str, n: usize, steps: usize, rng: &mut Rng) -> Vec<Vec<usize>> {
    (0..n)
        .map(|l| {
            (0..steps)
                .map(|s| match pattern {
                    "all1" => 1,
                    "all2" => 2,
                    "all4" => 4,
                    "all8" => GDN_CORE_LANE_ROWS,
                    // The reference mixed window of the step packet: AR and verify
                    // lanes interleaved, every row count 1..=8 present; the pattern
                    // rotates by one lane per step so every lane takes every role.
                    "task_mixed" => [1, 4, 1, 4, 2, 1, 3, 8][(l + s) % 8],
                    "ragged" => 1 + rng.below(GDN_CORE_LANE_ROWS),
                    "verify_ragged" => 2 + rng.below(GDN_CORE_LANE_ROWS - 1),
                    // AR lanes (one row) among verify lanes of 2..=8 rows.
                    "mixed" => {
                        if l % 2 == 0 {
                            1
                        } else {
                            2 + (3 * l + s) % (GDN_CORE_LANE_ROWS - 1)
                        }
                    }
                    _ => unreachable!(),
                })
                .collect()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_spec(
    rng: &mut Rng,
    n: usize,
    pattern: &'static str,
    variant: &'static str,
    pool_kind: &'static str,
    steps: usize,
    control: Option<Control>,
    rotate: bool,
) -> CaseSpec {
    let base = pool(pool_kind, steps);
    let rotate_by = (n + variant.len() + pattern.len()) % base.len();
    let positions: Vec<usize> = (0..n).map(|i| base[(i + rotate_by) % base.len()]).collect();
    let (table_order, seg_order, ref_order): (Vec<usize>, Vec<usize>, Vec<usize>) = match variant {
        // The table follows the memory order.
        "ident" => ((0..n).collect(), (0..n).collect(), (0..n).collect()),
        // The table runs against the memory order (the launcher sorts the
        // gate spans by address).
        "reverse" => (
            (0..n).rev().collect(),
            (0..n).collect(),
            (0..n).rev().collect(),
        ),
        "shuffle" => (shuffled(rng, n), shuffled(rng, n), shuffled(rng, n)),
        _ => unreachable!(),
    };
    let tag = control.map_or(String::new(), |c| format!(".{}", c.name()));
    CaseSpec {
        name: format!("N{n}.{pattern}.{variant}.{pool_kind}{tag}"),
        n,
        pattern,
        variant,
        pool: pool_kind,
        rows: rows_for(pattern, n, steps, rng),
        table_order,
        seg_order,
        ref_order,
        positions,
        steps,
        control,
        rotate,
        alias: false,
    }
}

fn main() {
    let cli = parse_cli();
    let mut gpu = Gpu::init().expect("Gpu::init");
    let default_enabled = gpu.flags.qwen4_gdn_lanes_enabled();
    let rows_default_enabled = gpu.flags.qwen4_gdn_lanes_rows_enabled();
    // The oracle runs the candidate routes whatever the defaults are (the rows
    // route is gated on HIPFIRE_QWEN4_GDN_LANES_ROWS; the one-row group runs its
    // fused kernel under HIPFIRE_QWEN4_GDN_LANES_FUSE, left at the process value).
    {
        let flags = Arc::make_mut(&mut gpu.flags);
        flags.qwen4_gdn_lanes = true;
        flags.qwen4_gdn_lanes_rows = true;
    }
    if gpu.arch != "gfx1201" || !gpu.flags.qwen4_gdn_lanes_rows_enabled() {
        eprintln!(
            "gdn_lanes_rows_oracle: the verify lanes route is not available on {} (needs exact gfx1201)",
            gpu.arch
        );
        std::process::exit(2);
    }
    println!(
        "arch={} lanes_default_enabled={} rows_default_enabled={} fuse_enabled={} steps={} seed={:#x}",
        gpu.arch,
        default_enabled,
        rows_default_enabled,
        gpu.flags.qwen4_gdn_lanes_fuse_enabled(),
        cli.steps,
        cli.seed
    );
    let mut rng = Rng(cli.seed);
    let wh = WeightsHost::new(&mut rng);
    let w = Weights::upload(&mut gpu, &wh);
    let w2 = Weights::upload(&mut gpu, &wh);

    if let Some(iters) = cli.bench {
        // Timing only: 8 lanes of uniform rows, then the mixed AR + verify window.
        let mut windows: Vec<Vec<usize>> = cli
            .time_rows
            .iter()
            .map(|&r| vec![r; GDN_CORE_LANES_MAX])
            .collect();
        windows.push(vec![1, 4, 1, 4, 2, 1, 3, 8]);
        let mut table = Vec::new();
        let mut ok = true;
        for rows in &windows {
            let t = bench(&mut gpu, &w, &mut rng, rows, iters, cli.bench_runs);
            print_bench(&t);
            let measured = |k: &str| {
                (
                    t[k]["kernels"].as_u64() == t[k]["expected_kernels"].as_u64(),
                    t[k]["copies"].as_u64() == t[k]["expected_copies"].as_u64(),
                )
            };
            let counts_ok = measured("per_lane_launches") == (true, true)
                && measured("grouped_launches") == (true, true);
            let route_ok = t["grouped_route"] != "per-lane";
            if !counts_ok || !route_ok {
                println!("bench   FAIL: launch counts / route of the window do not match");
                ok = false;
            }
            table.push(t);
        }
        w.free(&mut gpu);
        w2.free(&mut gpu);
        let summary = json!({
            "tool": "gdn_lanes_rows_oracle --bench",
            "arch": gpu.arch,
            "fuse_enabled": gpu.flags.qwen4_gdn_lanes_fuse_enabled(),
            "iters_per_run": iters,
            "runs": cli.bench_runs,
            "pass": ok,
            "bench": table,
        });
        let text = serde_json::to_string_pretty(&summary).expect("json");
        if let Some(dir) = &cli.out {
            std::fs::create_dir_all(dir).expect("create --out dir");
            std::fs::write(dir.join("bench.json"), &text).expect("write bench.json");
        }
        println!("{text}");
        println!(
            "gdn_lanes_rows_oracle --bench: {} arch={}",
            if ok { "PASS" } else { "FAIL" },
            gpu.arch
        );
        std::process::exit(if ok { 0 } else { 1 });
    }

    let mut failures = 0usize;
    let mut case_json = Vec::new();
    let mut control_json = Vec::new();
    let mut counts = (0usize, 0usize);

    let matrix: [(&'static str, &'static str, &'static str); 14] = [
        ("all1", "ident", "small"),
        ("all2", "reverse", "mixed"),
        ("all4", "ident", "mixed"),
        ("all4", "shuffle", "large"),
        ("all8", "shuffle", "large"),
        ("ragged", "shuffle", "mixed"),
        ("verify_ragged", "reverse", "large"),
        ("mixed", "ident", "small"),
        ("ragged", "ident", "same"),
        ("mixed", "reverse", "large"),
        ("mixed", "shuffle", "mixed"),
        ("task_mixed", "ident", "small"),
        ("task_mixed", "shuffle", "large"),
        ("all1", "shuffle", "large"),
    ];
    for &n in &cli.lanes {
        for (i, (pattern, variant, pool_kind)) in matrix.into_iter().enumerate() {
            let mut spec = build_spec(
                &mut rng,
                n,
                pattern,
                variant,
                pool_kind,
                cli.steps,
                None,
                i % 2 == 0,
            );
            spec.alias = cli.alias;
            if cli.alias {
                spec.name.push_str(".alias");
            }
            let r = run_case(&mut gpu, &w, &mut rng, &spec);
            print_case(&r, false);
            counts.0 += 1;
            if !r.pass {
                failures += 1;
            }
            case_json.push(r.detail);
        }
    }

    if cli.controls {
        // Controls corrupt two lanes of one class: verify-only groups at 2, 3
        // and 8 lanes, and mixed groups (AR lanes interleaved with verify
        // lanes: both classes grouped at once) at 4 and 8 lanes, as far as the
        // requested lane counts reach.
        let widest = cli.lanes.iter().copied().max().unwrap_or(0);
        let control_sets: [(&'static str, [usize; 3]); 3] = [
            ("verify_ragged", [2, 3, 8]),
            ("mixed", [4, 8, 8]),
            ("task_mixed", [8, 8, 8]),
        ];
        for (pattern, lane_counts) in control_sets {
            let mut seen = Vec::new();
            for n in lane_counts {
                if n > widest || seen.contains(&n) {
                    continue;
                }
                seen.push(n);
                for control in Control::ALL {
                    // The AR-lane control needs AR lanes.
                    if control == Control::ArWrongFrame && pattern == "verify_ragged" {
                        continue;
                    }
                    let spec = build_spec(
                        &mut rng,
                        n,
                        pattern,
                        "shuffle",
                        "mixed",
                        cli.steps.min(2).max(1),
                        Some(control),
                        false,
                    );
                    let r = run_case(&mut gpu, &w, &mut rng, &spec);
                    print_case(&r, true);
                    counts.1 += 1;
                    if !r.pass {
                        failures += 1;
                    }
                    control_json.push(r.detail);
                }
            }
        }
    }

    let (probes_ok, probe_json) = admission_probes(&mut gpu, &w, &w2, &mut rng);
    if !probes_ok {
        failures += 1;
    }

    let mut timing_json = Vec::new();
    if cli.timing {
        for &rows in &cli.time_rows {
            let t = bench(
                &mut gpu,
                &w,
                &mut rng,
                &vec![rows; GDN_CORE_LANES_MAX],
                cli.reps,
                5,
            );
            print_bench(&t);
            timing_json.push(t);
        }
    }
    w.free(&mut gpu);
    w2.free(&mut gpu);

    let pass = failures == 0;
    let summary = json!({
        "tool": "gdn_lanes_rows_oracle",
        "arch": gpu.arch,
        "lanes_default_enabled": default_enabled,
        "rows_default_enabled": rows_default_enabled,
        "fuse_enabled": gpu.flags.qwen4_gdn_lanes_fuse_enabled(),
        "shape": {
            "key_heads": NK, "value_heads": NV, "head_dim": HD, "qkv": QKV,
            "conv_ring": [RING, QKV], "q8_slot_bytes": SLOT, "ring_rows_per_half": RING_ROWS,
        },
        "steps": cli.steps,
        "seed": cli.seed,
        "lane_counts": cli.lanes,
        "cases": counts.0,
        "controls": counts.1,
        "failures": failures,
        "pass": pass,
        "launch_counts": {
            "per_lane_per_verify_lane": "3 kernels + 1 device copy",
            "per_lane_per_ar_lane": "2 kernels",
            "grouped_verify_set": 3,
            "grouped_ar_set": 2,
        },
        "case_results": case_json,
        "control_results": control_json,
        "admission_probes": probe_json,
        "timing": timing_json,
    });
    let text = serde_json::to_string_pretty(&summary).expect("json");
    if let Some(dir) = &cli.out {
        std::fs::create_dir_all(dir).expect("create --out dir");
        std::fs::write(dir.join("summary.json"), &text).expect("write summary.json");
    }
    println!("{text}");
    println!(
        "gdn_lanes_rows_oracle: {} arch={} cases={} controls={} failures={}",
        if pass { "PASS" } else { "FAIL" },
        gpu.arch,
        counts.0,
        counts.1,
        failures
    );
    std::process::exit(if pass { 0 } else { 1 });
}

/// Host-only self-tests of the harness (`cargo test --features lab --example
/// gdn_lanes_rows_oracle`): no GPU is opened.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_images_and_violations() {
        for (raw, guard, len) in [(false, 5usize, 7usize), (true, 9, 13)] {
            let elem = if raw { 1 } else { 4 };
            let body: Vec<u8> = (0..len * elem).map(|i| i as u8).collect();
            let mut image = Guarded::image_of(raw, guard, len, Some(&body));
            assert_eq!(image.len(), (len + 2 * guard) * elem);
            assert_eq!(guard_violations(raw, guard, len, &image), 0);
            image[guard * elem] ^= 0xff;
            assert_eq!(guard_violations(raw, guard, len, &image), 0);
            let last = image.len() - 1;
            image[0] ^= 1;
            image[last] ^= 1;
            image[guard * elem - 1] ^= 1;
            image[(guard + len) * elem] ^= 1;
            assert_eq!(guard_violations(raw, guard, len, &image), 4);
        }
    }

    #[test]
    fn capture_base_flips_the_ring_half() {
        assert_eq!(capture_base(0), RING_ROWS);
        assert_eq!(capture_base(RING_ROWS - 1), RING_ROWS);
        assert_eq!(capture_base(RING_ROWS), 0);
        assert_eq!(capture_base(2 * RING_ROWS - 1), 0);
    }

    #[test]
    fn plan_packs_segments_in_memory_order_and_commits_like_the_state() {
        let rows = [3usize, 1, 8, 2];
        let carry = vec![
            Carry { pos: 10, live: 0 },
            Carry { pos: 11, live: 4 },
            Carry {
                pos: 12,
                live: RING_ROWS + 2,
            },
            Carry { pos: 13, live: 7 },
        ];
        let plan = plan_step(&rows, &[2, 0, 3, 1], &carry);
        assert_eq!(plan[2].seg, 0);
        assert_eq!(plan[0].seg, 8);
        assert_eq!(plan[3].seg, 11);
        assert_eq!(plan[1].seg, 13);
        assert_eq!(plan[0].base, RING_ROWS);
        assert_eq!(plan[2].base, 0);
        let mut next = carry.clone();
        commit(&plan, &mut next);
        // Verify lanes make their captured slot live, the AR lane keeps its slot.
        assert_eq!(next[0].live, RING_ROWS + 2);
        assert_eq!(next[1].live, 4);
        assert_eq!(next[2].live, 7);
        assert_eq!(next[3].live, RING_ROWS + 1);
        assert_eq!(
            next.iter().map(|c| c.pos).collect::<Vec<_>>(),
            [13, 12, 20, 15]
        );
    }

    #[test]
    fn expected_counts_follow_the_interpreter_grouping() {
        // Per lane: 3 kernels + 1 copy per verify lane, 2 kernels per AR lane.
        assert_eq!(expected_counts(&[2, 3, 8], true), ((9, 3), (3, 0)));
        assert_eq!(expected_counts(&[1, 1, 1], false), ((6, 0), (2, 0)));
        // The fused one-row group is one launch.
        assert_eq!(expected_counts(&[1, 1, 1], true), ((6, 0), (1, 0)));
        assert_eq!(expected_counts(&[1; 8], true), ((16, 0), (1, 0)));
        // A lone lane of a set runs per lane.
        assert_eq!(expected_counts(&[1, 4], true), ((5, 1), (5, 1)));
        assert_eq!(expected_counts(&[1, 1, 4], true), ((7, 1), (4, 1)));
        assert_eq!(expected_counts(&[1, 4, 5], true), ((8, 2), (5, 0)));
        assert_eq!(
            expected_counts(&[1, 4, 1, 4, 2, 1, 3, 8], true),
            ((21, 5), (4, 0))
        );
        assert_eq!(expected_counts(&[7], true), ((3, 1), (3, 1)));
    }

    #[test]
    fn routes_are_named_per_class_and_controls_are_distinct() {
        let route = |ar, verify| {
            Grouping {
                ar_grouped: ar,
                verify_grouped: verify,
                ..Grouping::default()
            }
            .route()
        };
        assert_eq!(route(false, false), "per-lane");
        assert_eq!(route(true, false), "ar-lanes");
        assert_eq!(route(false, true), "rows-lanes");
        assert_eq!(route(true, true), "rows-lanes+ar-lanes");
        let names: std::collections::BTreeSet<&str> =
            Control::ALL.iter().map(|c| c.name()).collect();
        assert_eq!(names.len(), Control::ALL.len());
    }

    /// A mixed group (AR lanes interleaved with verify lanes in the packed
    /// rows) leaves holes between the verify lanes' segments: the gate lanes
    /// kernel must not need them adjacent.
    #[test]
    fn mixed_groups_interleave_segments() {
        let rows = [1usize, 5, 1, 4];
        let carry = vec![Carry { pos: 0, live: 0 }; 4];
        let plan = plan_step(&rows, &[0, 1, 2, 3], &carry);
        assert_eq!(plan[1].seg + plan[1].rows, plan[2].seg);
        assert_ne!(plan[1].seg + plan[1].rows, plan[3].seg);
    }

    #[test]
    fn row_patterns_stay_in_their_ranges() {
        let mut rng = Rng(3);
        for pattern in [
            "all1",
            "all2",
            "all4",
            "all8",
            "ragged",
            "verify_ragged",
            "mixed",
            "task_mixed",
        ] {
            let rows = rows_for(pattern, 8, 6, &mut rng);
            assert_eq!(rows.len(), 8);
            for lane in &rows {
                assert_eq!(lane.len(), 6);
                assert!(lane.iter().all(|r| (1..=GDN_CORE_LANE_ROWS).contains(r)));
                if pattern == "verify_ragged" {
                    assert!(lane.iter().all(|&r| r >= 2));
                }
            }
        }
        let mixed = rows_for("mixed", 4, 3, &mut rng);
        assert!(mixed[0].iter().all(|&r| r == 1) && mixed[1].iter().all(|&r| r >= 2));
    }

    #[test]
    fn positions_fit_u32_and_cover_every_cursor() {
        for steps in [1usize, 2, 4, 9] {
            for kind in ["small", "mixed", "large", "same"] {
                let p = pool(kind, steps);
                assert_eq!(p.len(), 8, "{kind}");
                for &base in &p {
                    assert!(
                        base + GDN_CORE_LANE_ROWS * steps - 1 <= u32::MAX as usize,
                        "{kind} {base}"
                    );
                }
            }
        }
        let cursors: std::collections::BTreeSet<usize> =
            pool("small", 4).iter().map(|p| p % RING).collect();
        assert_eq!(cursors.len(), RING);
    }

    #[test]
    fn specs_are_permutations() {
        let mut rng = Rng(7);
        for n in 1..=GDN_CORE_LANES_MAX {
            for (pattern, variant, kind) in [
                ("all2", "ident", "small"),
                ("ragged", "reverse", "mixed"),
                ("mixed", "shuffle", "large"),
            ] {
                let s = build_spec(&mut rng, n, pattern, variant, kind, 4, None, false);
                for order in [&s.table_order, &s.seg_order, &s.ref_order] {
                    let mut o = order.clone();
                    o.sort_unstable();
                    assert_eq!(o, (0..n).collect::<Vec<_>>(), "{}", s.name);
                }
                assert_eq!(s.rows.len(), n);
                assert_eq!(s.positions.len(), n);
            }
        }
    }

    #[test]
    fn diff_counts_mismatching_bytes_and_stray_writes_skip_the_allowed_slot() {
        let snap = |a: u8| Snap {
            items: vec![
                ("lane0.hist".to_string(), vec![1, 2, 3, a]),
                ("lane1.hist".to_string(), vec![9, 9]),
            ],
            guard_bad: Vec::new(),
        };
        assert!(diff(&snap(4), &snap(4)).is_empty());
        assert_eq!(
            diff(&snap(4), &snap(5)),
            vec![("lane0.hist".to_string(), 1)]
        );

        let ring = |touched: &[usize]| {
            let mut bytes = vec![0u8; 2 * RING_ROWS * SLOT];
            for &slot in touched {
                bytes[slot * SLOT + 5] = 1;
            }
            Snap {
                items: vec![("lane0.ring".to_string(), bytes)],
                guard_bad: Vec::new(),
            }
        };
        let plan = [LaneStep {
            rows: 3,
            pos: 0,
            live: 0,
            base: RING_ROWS,
            seg: 0,
        }];
        // The captured slot (base + rows - 1) is free to change; any other is stray.
        assert_eq!(
            stray_ring_writes(&ring(&[]), &ring(&[RING_ROWS + 2]), &plan),
            0
        );
        assert_eq!(stray_ring_writes(&ring(&[]), &ring(&[0]), &plan), 1);
        assert_eq!(
            stray_ring_writes(&ring(&[]), &ring(&[RING_ROWS + 1]), &plan),
            1
        );
    }

    #[test]
    fn slot_layout_matches_the_q8_slot() {
        assert_eq!(SLOT, 811_008);
        assert_eq!(QKV, 10_240);
        let mut rng = Rng(1);
        let host = make_lane_host(&mut rng, 3, &[2, 5]);
        assert_eq!(host.state.len(), SLOT);
        assert_eq!(host.hist.len(), RING * QKV);
        assert_eq!(host.steps[1].qkv.len(), 5 * QKV);
        assert_eq!(host.steps[0].z.len(), 2 * VALUE);
        let scales = &host.state[NV * HD * HD..];
        assert!(scales
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .all(|s| s > 0.0 && s.is_finite()));
    }
}
