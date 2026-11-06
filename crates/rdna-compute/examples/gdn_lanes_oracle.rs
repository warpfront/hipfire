// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness oracle of the multi-lane Flash-Next GDN decode Core
//! (`gated_delta_core_lanes`: `gated_delta_conv_params_bf16_f32_lanes` +
//! `gated_delta_step_gate_norm128_q8_lanes`, two launches for up to 8 lanes)
//! against the per-lane singleton launch pair the interpreter issues for one
//! one-row AR lane (`gated_delta_conv_params` + unrotated
//! `gated_delta_step_gated`, exactly `layer_ops.rs` execute_gated_delta_net
//! per-row branch: arguments taken from the very same `GdnCoreLane`).
//!
//! One synthetic Flash-Next GDN layer: key heads 16, value heads 48, head dim
//! 128 (q|k|v conv channels 10240, ring 3 x 10240 F32, Q8 state 48 x 128 x 132
//! B = i8 codes `[head][value][128]` then F32 scales). All lanes share the
//! conv weight, `a_log`, `dt_bias` and gated-norm weight and live in ONE row
//! scratch each (lane slot r owns row r of the projection / conv output / a /
//! b / gate / beta / recurrent / z / gdn_output scratches, as in the lane
//! step), with their own ring, Q8 slot, conv cursor (`position % 3`) and Q8
//! frame (`position`).
//!
//! Per case: N lanes (1..=8; N = 1 pads the lanes launch with one decoy lane
//! because admission refuses a lone lane, and compares only the real one),
//! `--steps` consecutive steps with state carried (every step gets fresh
//! random input rows), ragged positions (small, one `[7,100,2,...]` pool,
//! `u16` / `2^24` / near-`u32::MAX` ones, and all-equal), per-lane random Q8
//! state and ring, and a lane ORDER variant (identity / reversed / shuffled
//! table slots; the per-lane reference launches in the matching order).
//!
//! Compared as BYTES, per step, for every compared lane: the conv output row
//! (qkv), gate, beta, the recurrent-output row (never stored by the fused
//! step: stays sentinel in both arms), the gated output row, the input rows
//! (must be untouched), and the lane's whole ring and Q8 slot. All outputs
//! are sentinel-filled (NaN payload) before each step, every scratch and
//! state buffer carries guard bands that must stay sentinel in both arms.
//!
//! Also: launch counts (`hip_bridge::memory_effects`, per-lane = 2N per step,
//! lanes = 2 per step), a rotation-invariance arm (the singleton with the
//! decode rotation `rotate_into` asked must store the same gdn output / state
//! as the unrotated one the lanes route drops), negative controls the oracle
//! MUST detect (two lanes' Q8 slots swapped, two lanes' rings swapped, one
//! lane's Q8 frame off by one, one lane's conv cursor off by one) and
//! admission probes (`gated_delta_core_lanes` returns `None` for a lone lane,
//! 9 lanes, a duplicated state / ring / output buffer, a different shared
//! weight pointer, a q that does not start the conv output, gate or recurrent
//! pointer mismatches, and the flag off; `Some` for admitted 2 and 8 lane
//! sets).
//!
//! `--fuse` (`HIPFIRE_QWEN4_GDN_LANES_FUSE`): the candidate is `GdnCoreLanesPlan::launch_rotated`
//! (`gated_delta_step_gate_norm128_q8_lanes_fuse`: the conv + gate-parameter launch as the
//! step block's prologue, the output projection's `mq_rotate_x` as its epilogue, ONE launch per
//! step). It is compared against the same per-lane singleton reference, and in addition: the
//! rotated rows written into the batched rotation scratch (lane slot s = row s) must equal a real
//! `rotate_x_mq_batched` of the lanes' gated outputs and (cases with the rotation arm) the
//! singleton's decode rotation, the next matching `rotate_x_mq_batched` must skip (the armed
//! `prerotated` claim: no launch), the arrival counters must be back to zero after every step,
//! variant `same_pos` runs the prologue-only form (no rotation: rotated rows stay sentinel, no
//! claim), and a flag-off probe asserts `launch_rotated` is the two incumbent launches and
//! rotates nothing. `--bench ITERS` (implies `--fuse`; run it as a timing job): wall time per
//! Core + rotation of the incumbent chain against the fused launch, then exit.
//!
//! `cargo run --release -p rdna-compute --features lab --example gdn_lanes_oracle --
//!    [--out DIR] [--steps S] [--lanes 1,2,3,4,8] [--seed U64] [--skip-controls] [--fuse]
//!    [--bench ITERS]`
//!
//! Prints one line per case / control / probe, a final `gdn_lanes_oracle:
//! PASS|FAIL` line and a JSON summary (stdout, and `DIR/summary.json` with
//! `--out`). Exit 0 = PASS, 1 = a mismatch / undetected control / wrong
//! launch count / wrong admission, 2 = the GPU cannot run the route.

use hip_bridge::memory_effects;
use rdna_compute::tensor_ops::{
    gated_delta_conv_params, gated_delta_core_lanes, gated_delta_step_gated, GatedDeltaConv,
    GatedDeltaGate, GatedDeltaParams, GatedDeltaStep, GdnCoreLane, GdnStateFormat,
    GDN_CORE_LANES_MAX,
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
const STATE_BYTES: usize = NV * HD * (HD + 4);
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
    // Both guards start on an element boundary: the pattern phase is the
    // offset from the guard's own start.
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

    /// Bytes of body elements `[offset, offset + len)`.
    fn rows<'b>(&self, all: &'b [u8], offset: usize, len: usize) -> &'b [u8] {
        let b = self.body(all);
        &b[offset * self.elem()..(offset + len) * self.elem()]
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
    qkv: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    z: Vec<f32>,
}

/// One logical lane: initial state, position and per-step inputs.
struct LaneHost {
    ring: Vec<f32>,
    state: Vec<u8>,
    base_pos: usize,
    steps: Vec<StepIn>,
}

fn make_lane_host(rng: &mut Rng, base_pos: usize, steps: usize) -> LaneHost {
    let ring = f32s(rng, RING * QKV, 1.0);
    let mut state = Vec::with_capacity(STATE_BYTES);
    for _ in 0..NV * HD * HD {
        state.push(((rng.next() % 255) as i32 - 127) as i8 as u8);
    }
    for _ in 0..NV * HD {
        state.extend_from_slice(&(0.001 + rng.unit() * 0.02).to_le_bytes());
    }
    assert_eq!(state.len(), STATE_BYTES);
    LaneHost {
        ring,
        state,
        base_pos,
        steps: (0..steps)
            .map(|_| StepIn {
                qkv: f32s(rng, QKV, 1.5),
                a: f32s(rng, NV, 2.0),
                b: f32s(rng, NV, 2.0),
                z: f32s(rng, VALUE, 1.5),
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// One arm: row scratch + per-lane state
// ---------------------------------------------------------------------------

struct Arm {
    rows: usize,
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
    /// Per logical lane.
    ring: Vec<Guarded>,
    state: Vec<Guarded>,
}

impl Arm {
    fn new(gpu: &mut Gpu, rows: usize, host: &[LaneHost]) -> Self {
        let sent = |gpu: &mut Gpu, w: usize| Guarded::new_f32(gpu, rows * w, None);
        Self {
            rows,
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
            ring: host
                .iter()
                .map(|h| Guarded::new_f32(gpu, RING * QKV, Some(&h.ring)))
                .collect(),
            state: host
                .iter()
                .map(|h| Guarded::new_raw(gpu, &h.state))
                .collect(),
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
            ring,
            state,
            ..
        } = self;
        for g in [
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
        ]
        .into_iter()
        .chain(ring)
        .chain(state)
        {
            gpu.free_tensor(g.t).expect("free arm");
        }
    }

    /// Fill the input scratches (rows of the lanes in `row_of`, sentinel in the
    /// others) and reset every output scratch to the sentinel.
    fn stage(&self, gpu: &Gpu, host: &[LaneHost], step: usize, row_of: &[(usize, usize)]) {
        let fill = |width: usize, pick: fn(&StepIn) -> &Vec<f32>| -> Vec<u8> {
            let mut body = vec![f32::from_bits(F32_SENTINEL); self.rows * width];
            for &(lane, row) in row_of {
                body[row * width..(row + 1) * width].copy_from_slice(pick(&host[lane].steps[step]));
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
    }
}

/// The tensor views of one lane at one row.
struct LaneViews {
    proj: GpuTensor,
    proj2: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    rec: GpuTensor,
    z: GpuTensor,
    out: GpuTensor,
    rot: GpuTensor,
    ring: GpuTensor,
    state: GpuTensor,
}

impl LaneViews {
    fn new(arm: &Arm, lane: usize, row: usize) -> Self {
        assert!(row < arm.rows, "row outside the scratch");
        let proj2 = arm.projection2.view(row * QKV, QKV);
        Self {
            proj: arm.projection.view(row * QKV, QKV),
            q: proj2.sub_offset(0, QK),
            k: proj2.sub_offset(QK, QK),
            v: proj2.sub_offset(2 * QK, VALUE),
            proj2,
            a: arm.a.view(row * NV, NV),
            b: arm.b.view(row * NV, NV),
            gate: arm.gate.view(row * NV, NV),
            beta: arm.beta.view(row * NV, NV),
            rec: arm.recurrent.view(row * VALUE, VALUE),
            z: arm.z.view(row * VALUE, VALUE),
            out: arm.out.view(row * VALUE, VALUE),
            rot: arm.rot.view(row * VALUE, VALUE),
            ring: arm.ring[lane].view(0, RING * QKV),
            state: arm.state[lane].view(0, STATE_BYTES),
        }
    }
}

/// The per-lane call arguments of `execute_gated_delta_net_core_lanes` /
/// the per-row branch of `execute_gated_delta_net_hc_impl` for one lane.
fn make_lane<'a>(v: &'a LaneViews, w: &'a Weights, position: usize) -> GdnCoreLane<'a> {
    GdnCoreLane {
        conv: GatedDeltaConv {
            input: &v.proj,
            kernel: &w.conv,
            history: &v.ring,
            output: &v.proj2,
            next_history: &v.ring,
            channels: QKV,
            history_rows: RING,
            kernel_size: 4,
            cursor: position % RING,
            row_index: 0,
        },
        params: GatedDeltaParams {
            a: &v.a,
            b: &v.b,
            a_log: &w.a_log,
            dt_bias: &w.dt_bias,
            gate: &v.gate,
            beta: &v.beta,
        },
        step: GatedDeltaStep {
            q: &v.q,
            k: &v.k,
            v: &v.v,
            gate: &v.gate,
            beta: &v.beta,
            state: &v.state,
            format: GdnStateFormat::Q8,
            residual: None,
            output: &v.rec,
            key_heads: NK,
            value_heads: NV,
            key_dim: HD,
            value_dim: HD,
            position,
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

// ---------------------------------------------------------------------------
// Arms' launches
// ---------------------------------------------------------------------------

/// Reference: the per-lane singleton launch pair of each lane in
/// `launch_order`; lane `L` sits at row `L`. `rotate` asks the decode rotation
/// of the step (the Whole-phase product call), which the lanes route drops.
fn run_per_lane(
    gpu: &mut Gpu,
    w: &Weights,
    arm: &Arm,
    host: &[LaneHost],
    launch_order: &[usize],
    step: usize,
    rotate: bool,
) {
    for &lane in launch_order {
        let views = LaneViews::new(arm, lane, lane);
        let l = make_lane(&views, w, host[lane].base_pos + step);
        gated_delta_conv_params(gpu, &l.conv, &l.params, NV).expect("per-lane conv + params");
        gated_delta_step_gated(gpu, &l.step, &l.gate, rotate.then_some(&views.rot))
            .expect("per-lane step + gated norm");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Control {
    /// Slots 0 and 1 name each other's Q8 slot.
    SwapState,
    /// Slots 0 and 1 name each other's conv ring.
    SwapRing,
    /// The last slot's Q8 frame is one too high.
    WrongFrame,
    /// Slot 0's conv cursor is one too high.
    WrongCursor,
}

impl Control {
    const ALL: [Control; 4] = [
        Control::SwapState,
        Control::SwapRing,
        Control::WrongFrame,
        Control::WrongCursor,
    ];
    fn name(self) -> &'static str {
        match self {
            Control::SwapState => "swap_state_pointers",
            Control::SwapRing => "swap_ring_pointers",
            Control::WrongFrame => "wrong_q8_frame",
            Control::WrongCursor => "wrong_conv_cursor",
        }
    }
}

/// Which lanes launch the candidate takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fuse {
    /// The two incumbent launches (`GdnCoreLanesPlan::launch`).
    Off,
    /// `launch_rotated` with a rotation scratch: one launch, rotated rows written, claim armed.
    Rot,
    /// `launch_rotated` without rotation: one launch (conv/params prologue only).
    NoRot,
}

/// Candidate: ONE `gated_delta_core_lanes` launch pair over the table
/// `order` (slot -> logical lane; slot s is row s). `control` corrupts one
/// lane's arguments (the oracle must then report a mismatch).
fn run_lanes(
    gpu: &mut Gpu,
    w: &Weights,
    arm: &Arm,
    host: &[LaneHost],
    order: &[usize],
    step: usize,
    control: Option<Control>,
    fuse: Fuse,
) {
    let views: Vec<LaneViews> = order
        .iter()
        .enumerate()
        .map(|(slot, &lane)| LaneViews::new(arm, lane, slot))
        .collect();
    let mut lanes: Vec<GdnCoreLane<'_>> = order
        .iter()
        .zip(&views)
        .map(|(&lane, v)| make_lane(v, w, host[lane].base_pos + step))
        .collect();
    match control {
        None => {}
        Some(Control::SwapState) => {
            lanes[0].step.state = &views[1].state;
            lanes[1].step.state = &views[0].state;
        }
        Some(Control::SwapRing) => {
            lanes[0].conv.history = &views[1].ring;
            lanes[0].conv.next_history = &views[1].ring;
            lanes[1].conv.history = &views[0].ring;
            lanes[1].conv.next_history = &views[0].ring;
        }
        Some(Control::WrongFrame) => {
            let last = lanes.len() - 1;
            lanes[last].step.position += 1;
        }
        Some(Control::WrongCursor) => {
            lanes[0].conv.cursor = (lanes[0].conv.cursor + 1) % RING;
        }
    }
    let plan = gated_delta_core_lanes(gpu, &lanes, NV)
        .expect("admission")
        .expect("the lanes route refused an admitted lane set");
    match fuse {
        Fuse::Off => plan.launch(gpu).expect("lanes launch"),
        Fuse::NoRot => {
            let rotated = plan.launch_rotated(gpu, None).expect("fused lanes launch");
            assert!(!rotated, "no rotation asked, none reported");
        }
        Fuse::Rot => {
            let rot = arm.rot.view(0, arm.rows * VALUE);
            let rotated = plan
                .launch_rotated(gpu, Some(&rot))
                .expect("fused lanes launch");
            assert!(rotated, "the fused launch refused an admitted rotation");
        }
    }
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

/// `compared`: `(logical lane, row)` in ascending lane order.
fn snapshot(gpu: &Gpu, arm: &Arm, compared: &[(usize, usize)]) -> Snap {
    gpu.hip.device_synchronize().expect("sync before download");
    let mut items = Vec::new();
    let mut guard_bad = Vec::new();
    let shared: [(&str, &Guarded, usize); 9] = [
        ("proj", &arm.projection, QKV),
        ("conv_out", &arm.projection2, QKV),
        ("a", &arm.a, NV),
        ("b", &arm.b, NV),
        ("gate", &arm.gate, NV),
        ("beta", &arm.beta, NV),
        ("recurrent", &arm.recurrent, VALUE),
        ("z", &arm.z, VALUE),
        ("gdn_out", &arm.out, VALUE),
    ];
    for (name, g, width) in shared {
        let all = g.download(gpu);
        guard_bad.push((name.to_string(), g.guard_bad(&all)));
        for &(lane, row) in compared {
            items.push((
                format!("lane{lane}.{name}"),
                g.rows(&all, row * width, width).to_vec(),
            ));
        }
    }
    {
        let all = arm.rot.download(gpu);
        guard_bad.push(("rot".to_string(), arm.rot.guard_bad(&all)));
    }
    for &(lane, _) in compared {
        let all = arm.ring[lane].download(gpu);
        guard_bad.push((format!("lane{lane}.ring"), arm.ring[lane].guard_bad(&all)));
        items.push((
            format!("lane{lane}.ring"),
            arm.ring[lane].body(&all).to_vec(),
        ));
        let all = arm.state[lane].download(gpu);
        guard_bad.push((format!("lane{lane}.q8"), arm.state[lane].guard_bad(&all)));
        items.push((
            format!("lane{lane}.q8"),
            arm.state[lane].body(&all).to_vec(),
        ));
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

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

struct CaseSpec {
    name: String,
    /// Compared lanes (the real ones).
    n: usize,
    /// Lanes in the lanes launch (`max(n, 2)`; the rest are decoys).
    table: usize,
    variant: &'static str,
    /// Slot -> logical lane of the lanes launch.
    order: Vec<usize>,
    /// Launch order of the per-lane reference (logical lanes `0..n`).
    ref_order: Vec<usize>,
    positions: Vec<usize>,
    steps: usize,
    control: Option<Control>,
    check_rotate: bool,
    fuse: Fuse,
}

struct CaseResult {
    spec_name: String,
    pass: bool,
    bad_bytes: u64,
    bad_items: Vec<String>,
    guard_bad: u64,
    rotate_bad: Option<u64>,
    ref_launches: u64,
    lanes_launches: u64,
    launches_ok: bool,
    detail: Value,
}

/// Number of bytes that differ between two equally long byte rows.
fn byte_diff(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len(), "compared rows have one length");
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// The fused arm's checks after one candidate step (`cand` has `table` rows): the arrival
/// counters are back to zero, and
/// * `Rot`: the next matching `rotate_x_mq_batched` skips (the claim), the rotated rows equal a
///   real `rotate_x_mq_batched` of the lanes' gated outputs, and (when `rot_arm` ran the
///   singleton with the decode rotation) the compared lanes' rows equal the singleton's;
/// * `NoRot`: no rotated byte was written and no claim is armed.
///
/// Returns `(mismatching rotated bytes, other violations)`.
fn check_fused(
    gpu: &mut Gpu,
    cand: &Arm,
    truth: Option<&Guarded>,
    rot_arm: Option<&Arm>,
    fuse: Fuse,
    cand_rows: &[(usize, usize)],
) -> (u64, u64) {
    let mut rot_bad = 0u64;
    let mut other = 0u64;
    let table = cand.rows;
    match fuse {
        Fuse::Off => {}
        Fuse::Rot => {
            let out_all = cand.out.view(0, table * VALUE);
            let rot_all = cand.rot.view(0, table * VALUE);
            let m0 = memory_effects::snapshot();
            gpu.rotate_x_mq_batched(&out_all, &rot_all, VALUE, table)
                .expect("rotate after fused launch");
            if memory_effects::snapshot().since(m0).launches != 0 {
                other += 1;
            }
            let truth = truth.expect("truth buffer");
            let truth_all = truth.view(0, table * VALUE);
            gpu.rotate_x_mq_batched(&out_all, &truth_all, VALUE, table)
                .expect("truth rotation");
            gpu.hip.device_synchronize().expect("sync");
            let got = cand.rot.download(gpu);
            let want = truth.download(gpu);
            rot_bad += byte_diff(
                cand.rot.rows(&got, 0, table * VALUE),
                truth.rows(&want, 0, table * VALUE),
            ) as u64;
            if let Some(rot_arm) = rot_arm {
                let singleton = rot_arm.rot.download(gpu);
                for &(lane, slot) in cand_rows {
                    if lane < rot_arm.rows {
                        rot_bad += byte_diff(
                            cand.rot.rows(&got, slot * VALUE, VALUE),
                            rot_arm.rot.rows(&singleton, lane * VALUE, VALUE),
                        ) as u64;
                    }
                }
            }
        }
        Fuse::NoRot => {
            gpu.hip.device_synchronize().expect("sync");
            let got = cand.rot.download(gpu);
            let pattern = sentinel_pattern(false);
            let body = cand.rot.body(&got);
            rot_bad += body
                .iter()
                .enumerate()
                .filter(|(i, b)| **b != pattern[i % pattern.len()])
                .count() as u64;
            if gpu.scratch.prerotated.is_some() {
                other += 1;
            }
        }
    }
    let counters = gpu
        .scratch
        .gdn_pair_counters
        .as_ref()
        .expect("the fused launch allocates its counters");
    let mut raw = vec![0u8; counters.size()];
    gpu.hip
        .memcpy_dtoh(&mut raw, counters)
        .expect("download counters");
    other += raw.iter().filter(|b| **b != 0).count() as u64;
    (rot_bad, other)
}

fn run_case(gpu: &mut Gpu, w: &Weights, rng: &mut Rng, spec: &CaseSpec) -> CaseResult {
    let host: Vec<LaneHost> = spec
        .positions
        .iter()
        .map(|&p| make_lane_host(rng, p, spec.steps))
        .collect();
    assert_eq!(host.len(), spec.table);
    let ref_arm = Arm::new(gpu, spec.n, &host);
    let rot_arm = spec.check_rotate.then(|| Arm::new(gpu, spec.n, &host));
    let cand_arm = Arm::new(gpu, spec.table, &host);
    let truth = (spec.fuse == Fuse::Rot).then(|| Guarded::new_f32(gpu, spec.table * VALUE, None));
    let ref_rows: Vec<(usize, usize)> = (0..spec.n).map(|l| (l, l)).collect();
    let cand_rows: Vec<(usize, usize)> = spec
        .order
        .iter()
        .enumerate()
        .map(|(slot, &lane)| (lane, slot))
        .collect();
    let cand_compared: Vec<(usize, usize)> = (0..spec.n)
        .map(|l| *cand_rows.iter().find(|(lane, _)| *lane == l).unwrap())
        .collect();
    let mut bad_bytes = 0u64;
    let mut bad_items: Vec<String> = Vec::new();
    let mut guard_bad = 0u64;
    let mut rotate_bad = spec.check_rotate.then_some(0u64);
    let (mut fuse_rot_bad, mut fuse_other) = (0u64, 0u64);
    let (mut ref_launches, mut lanes_launches) = (0u64, 0u64);
    for step in 0..spec.steps {
        ref_arm.stage(gpu, &host, step, &ref_rows);
        let m0 = memory_effects::snapshot();
        run_per_lane(gpu, w, &ref_arm, &host, &spec.ref_order, step, false);
        ref_launches += memory_effects::snapshot().since(m0).launches;
        let want = snapshot(gpu, &ref_arm, &ref_rows);

        // The singleton with the decode rotation: the fused arm's rotated rows' reference.
        if let Some(rot) = &rot_arm {
            rot.stage(gpu, &host, step, &ref_rows);
            run_per_lane(gpu, w, rot, &host, &spec.ref_order, step, true);
        }
        let rotated = rot_arm.as_ref().map(|rot| snapshot(gpu, rot, &ref_rows));

        cand_arm.stage(gpu, &host, step, &cand_rows);
        let m0 = memory_effects::snapshot();
        run_lanes(
            gpu,
            w,
            &cand_arm,
            &host,
            &spec.order,
            step,
            spec.control,
            spec.fuse,
        );
        lanes_launches += memory_effects::snapshot().since(m0).launches;
        let got = snapshot(gpu, &cand_arm, &cand_compared);

        for (name, n) in diff(&want, &got) {
            bad_bytes += n as u64;
            if bad_items.len() < 24 {
                bad_items.push(format!("step{step}.{name}:{n}"));
            }
        }
        for s in [&want, &got] {
            guard_bad += s.guard_bad.iter().map(|(_, n)| *n as u64).sum::<u64>();
        }
        if let Some(rotated) = &rotated {
            let n: usize = diff(&want, rotated).iter().map(|(_, n)| *n).sum();
            *rotate_bad.as_mut().unwrap() += n as u64;
        }
        if spec.fuse != Fuse::Off {
            let (r, o) = check_fused(
                gpu,
                &cand_arm,
                truth.as_ref(),
                rot_arm.as_ref(),
                spec.fuse,
                &cand_compared,
            );
            fuse_rot_bad += r;
            fuse_other += o;
            if (r != 0 || o != 0) && bad_items.len() < 24 {
                bad_items.push(format!("step{step}.fused_rot:{r}.other:{o}"));
            }
        }
    }
    ref_arm.free(gpu);
    cand_arm.free(gpu);
    if let Some(r) = rot_arm {
        r.free(gpu);
    }
    if let Some(t) = truth {
        gpu.free_tensor(t.t).expect("free truth");
    }
    let want_ref = 2 * spec.n as u64 * spec.steps as u64;
    let want_lanes = if spec.fuse == Fuse::Off { 2 } else { 1 } * spec.steps as u64;
    let launches_ok = ref_launches == want_ref && lanes_launches == want_lanes;
    let detected = bad_bytes > 0;
    let pass = match spec.control {
        None => {
            bad_bytes == 0
                && guard_bad == 0
                && rotate_bad.unwrap_or(0) == 0
                && fuse_rot_bad == 0
                && fuse_other == 0
                && launches_ok
        }
        Some(_) => detected && launches_ok,
    };
    let detail = json!({
        "name": spec.name,
        "lanes": spec.n,
        "table_lanes": spec.table,
        "variant": spec.variant,
        "fuse": format!("{:?}", spec.fuse),
        "steps": spec.steps,
        "slot_to_lane": spec.order,
        "ref_launch_order": spec.ref_order,
        "positions": spec.positions,
        "control": spec.control.map(|c| c.name()),
        "pass": pass,
        "bad_bytes": bad_bytes,
        "bad_items": bad_items,
        "guard_bytes_overwritten": guard_bad,
        "rotate_arm_bad_bytes": rotate_bad,
        "fused_rotated_bad_bytes": fuse_rot_bad,
        "fused_other_violations": fuse_other,
        "launches": {
            "per_lane_total": ref_launches,
            "per_lane_expected": want_ref,
            "per_lane_per_step": ref_launches / spec.steps.max(1) as u64,
            "lanes_total": lanes_launches,
            "lanes_expected": want_lanes,
            "lanes_per_step": lanes_launches / spec.steps.max(1) as u64,
        },
    });
    CaseResult {
        spec_name: spec.name.clone(),
        pass,
        bad_bytes: bad_bytes + fuse_rot_bad + fuse_other,
        bad_items,
        guard_bad,
        rotate_bad,
        ref_launches,
        lanes_launches,
        launches_ok,
        detail,
    }
}

fn print_case(r: &CaseResult, control: bool) {
    let verdict = if r.pass { "PASS" } else { "FAIL" };
    if control {
        println!(
            "control {:<40} {verdict} (detected: bad_bytes={} launches ref={} lanes={}{})",
            r.spec_name,
            r.bad_bytes,
            r.ref_launches,
            r.lanes_launches,
            if r.launches_ok { "" } else { " WRONG COUNT" }
        );
    } else {
        println!(
            "case    {:<40} {verdict} bad_bytes={} guard_bad={} rotate_bad={} launches per-lane={} lanes={}{}{}",
            r.spec_name,
            r.bad_bytes,
            r.guard_bad,
            r.rotate_bad.map_or("n/a".to_string(), |n| n.to_string()),
            r.ref_launches,
            r.lanes_launches,
            if r.launches_ok { "" } else { " WRONG COUNT" },
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

fn lanes_of<'a>(views: &'a [LaneViews], w: &'a Weights, n: usize) -> Vec<GdnCoreLane<'a>> {
    views[..n]
        .iter()
        .enumerate()
        .map(|(i, v)| make_lane(v, w, 5 + i))
        .collect()
}

fn probe(
    gpu: &Gpu,
    name: &str,
    lanes: &[GdnCoreLane<'_>],
    expect_some: bool,
    out: &mut Vec<Value>,
) -> bool {
    let got = gated_delta_core_lanes(gpu, lanes, NV).expect("admission");
    let ok = got.is_some() == expect_some;
    println!(
        "probe   {:<40} {} (admitted={}, expected={})",
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
    let host: Vec<LaneHost> = (0..GDN_CORE_LANES_MAX)
        .map(|l| make_lane_host(rng, l, 1))
        .collect();
    let arm = Arm::new(gpu, GDN_CORE_LANES_MAX, &host);
    let views: Vec<LaneViews> = (0..GDN_CORE_LANES_MAX)
        .map(|l| LaneViews::new(&arm, l, l))
        .collect();
    let mut out = Vec::new();
    let mut ok = true;
    let mut check = |gpu: &Gpu, name: &str, lanes: &[GdnCoreLane<'_>], some: bool| {
        ok &= probe(gpu, name, lanes, some, &mut out);
    };
    check(gpu, "admit_2_lanes", &lanes_of(&views, w, 2), true);
    check(gpu, "admit_8_lanes", &lanes_of(&views, w, 8), true);
    check(gpu, "refuse_1_lane", &lanes_of(&views, w, 1), false);
    {
        let mut nine = lanes_of(&views, w, 8);
        nine.push(make_lane(&views[0], w, 5));
        check(gpu, "refuse_9_lanes", &nine, false);
    }
    {
        let mut l = lanes_of(&views, w, 3);
        l[1].step.state = &views[0].state;
        check(gpu, "refuse_duplicate_q8_state", &l, false);
    }
    {
        let mut l = lanes_of(&views, w, 3);
        l[1].conv.history = &views[0].ring;
        l[1].conv.next_history = &views[0].ring;
        check(gpu, "refuse_duplicate_ring", &l, false);
    }
    {
        let mut l = lanes_of(&views, w, 3);
        l[1].gate.output = &views[0].out;
        check(gpu, "refuse_duplicate_gdn_output", &l, false);
    }
    {
        let mut l = lanes_of(&views, w, 3);
        l[1].conv.kernel = &w2.conv;
        check(gpu, "refuse_other_conv_weight", &l, false);
    }
    {
        let mut l = lanes_of(&views, w, 3);
        l[1].gate.norm = &w2.norm;
        check(gpu, "refuse_other_norm_weight", &l, false);
    }
    {
        let mut l = lanes_of(&views, w, 3);
        l[1].params.a_log = &w2.a_log;
        check(gpu, "refuse_other_a_log", &l, false);
    }
    {
        let shifted = views[1].proj2.sub_offset(1, QK);
        let mut l = lanes_of(&views, w, 3);
        l[1].step.q = &shifted;
        check(gpu, "refuse_q_not_at_conv_output", &l, false);
    }
    {
        let mut l = lanes_of(&views, w, 3);
        l[1].step.gate = &views[1].beta;
        check(gpu, "refuse_step_gate_not_param_gate", &l, false);
    }
    {
        let mut l = lanes_of(&views, w, 3);
        l[1].gate.recurrent_output = &views[1].out;
        check(gpu, "refuse_recurrent_output_mismatch", &l, false);
    }
    {
        let l = lanes_of(&views, w, 2);
        Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes = false;
        check(gpu, "refuse_flag_off", &l, false);
        Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes = true;
        check(gpu, "admit_after_flag_back_on", &l, true);
    }
    drop(views);
    arm.free(gpu);
    (ok, out)
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
    /// Run the candidate through `launch_rotated` with `HIPFIRE_QWEN4_GDN_LANES_FUSE`.
    fuse: bool,
    /// `--bench ITERS`: time the incumbent chain against the fused launch (implies `--fuse`),
    /// then exit.
    bench: Option<usize>,
}

fn parse_cli() -> Cli {
    let mut cli = Cli {
        out: None,
        steps: 4,
        lanes: vec![1, 2, 3, 4, 8],
        seed: 0x6D6C_A9E5,
        controls: true,
        fuse: false,
        bench: None,
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
            "--lanes" => {
                cli.lanes = value("--lanes")
                    .split(',')
                    .map(|s| s.trim().parse().unwrap_or_else(|_| usage("--lanes")))
                    .collect();
                if cli
                    .lanes
                    .iter()
                    .any(|&n| !(1..=GDN_CORE_LANES_MAX).contains(&n))
                {
                    usage("--lanes entries must be in 1..=8");
                }
            }
            "--skip-controls" => cli.controls = false,
            "--fuse" => cli.fuse = true,
            "--bench" => {
                cli.bench = Some(
                    value("--bench")
                        .parse()
                        .unwrap_or_else(|_| usage("--bench")),
                );
                cli.fuse = true;
            }
            "-h" | "--help" => usage(""),
            other => usage(&format!("unknown argument {other}")),
        }
    }
    if cli.steps == 0 {
        usage("--steps must be >= 1");
    }
    cli
}

fn usage(msg: &str) -> ! {
    if !msg.is_empty() {
        eprintln!("gdn_lanes_oracle: {msg}");
    }
    eprintln!(
        "usage: gdn_lanes_oracle [--out DIR] [--steps S=4] [--lanes 1,2,3,4,8] [--seed U64] [--skip-controls] [--fuse]"
    );
    std::process::exit(if msg.is_empty() { 0 } else { 2 });
}

/// Base positions of lane i for a pool; rotated by case so lanes see different
/// cursors (`position % 3`) and frames. `steps` keeps every frame within u32.
fn pool(kind: &str, steps: usize) -> Vec<usize> {
    match kind {
        // Small, every cursor value, ragged.
        "small" => vec![0, 1, 2, 3, 4, 5, 6, 7],
        "mixed" => vec![7, 100, 2, 99, 12345, 3, 4096, 1],
        // u16 boundary, 2^18, > 2^20, 2^24, near the u32 end (the last step
        // lands exactly on u32::MAX).
        "large" => vec![
            65_534,
            65_535,
            65_536,
            262_143,
            1_000_003,
            16_777_215,
            4_294_967_000,
            u32::MAX as usize - (steps - 1),
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

fn build_spec(
    rng: &mut Rng,
    n: usize,
    variant: &'static str,
    pool_kind: &'static str,
    steps: usize,
    control: Option<Control>,
    check_rotate: bool,
) -> CaseSpec {
    let table = n.max(2);
    let base = pool(pool_kind, steps);
    let rotate_by = (n + variant.len()) % base.len();
    let positions: Vec<usize> = (0..table)
        .map(|i| base[(i + rotate_by) % base.len()])
        .collect();
    let (order, ref_order): (Vec<usize>, Vec<usize>) = match variant {
        "ident" | "same_pos" => ((0..table).collect(), (0..n).collect()),
        "reverse" => ((0..table).rev().collect(), (0..n).rev().collect()),
        "shuffle" => (shuffled(rng, table), shuffled(rng, n)),
        _ => unreachable!(),
    };
    let tag = control.map_or(String::new(), |c| format!(".{}", c.name()));
    CaseSpec {
        name: format!("N{n}.{variant}.{pool_kind}{tag}"),
        n,
        table,
        variant,
        order,
        ref_order,
        positions,
        steps,
        control,
        check_rotate,
        fuse: Fuse::Off,
    }
}

/// Flag off (the default): `launch_rotated` is the two incumbent launches, writes no rotated
/// byte and arms no claim.
fn fuse_flag_off_probe(gpu: &mut Gpu, w: &Weights, rng: &mut Rng) -> (bool, Value) {
    let host: Vec<LaneHost> = (0..2).map(|l| make_lane_host(rng, l, 1)).collect();
    let arm = Arm::new(gpu, 2, &host);
    let rows = [(0usize, 0usize), (1, 1)];
    arm.stage(gpu, &host, 0, &rows);
    Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes_fuse = false;
    gpu.scratch.prerotated = None;
    let (rotated, launches) = {
        let views: Vec<LaneViews> = (0..2).map(|l| LaneViews::new(&arm, l, l)).collect();
        let lanes = lanes_of(&views, w, 2);
        let rot = arm.rot.view(0, 2 * VALUE);
        let m0 = memory_effects::snapshot();
        let plan = gated_delta_core_lanes(gpu, &lanes, NV)
            .expect("admission")
            .expect("the lanes route refused an admitted lane set");
        let rotated = plan
            .launch_rotated(gpu, Some(&rot))
            .expect("flag-off launch");
        (rotated, memory_effects::snapshot().since(m0).launches)
    };
    gpu.hip.device_synchronize().expect("sync");
    let all = arm.rot.download(gpu);
    let pattern = sentinel_pattern(false);
    let untouched = arm
        .rot
        .body(&all)
        .iter()
        .enumerate()
        .all(|(i, b)| *b == pattern[i % pattern.len()]);
    let claim = gpu.scratch.prerotated.is_some();
    Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes_fuse = true;
    arm.free(gpu);
    let ok = !rotated && launches == 2 && untouched && !claim;
    println!(
        "probe   {:<40} {} (rotated={rotated}, launches={launches}, rot_untouched={untouched}, claim={claim})",
        "fuse_flag_off_is_the_incumbent_pair",
        if ok { "PASS" } else { "FAIL" },
    );
    (
        ok,
        json!({
            "name": "fuse_flag_off_is_the_incumbent_pair", "pass": ok,
            "rotated": rotated, "launches": launches, "rot_untouched": untouched, "claim": claim,
        }),
    )
}

/// Wall time per Core + output rotation, back to back on one stream (`iters` per sample, `reps`
/// alternating A/B samples after one discarded): the incumbent chain (two lanes launches +
/// `rotate_x_mq_batched`) against the fused launch (its claimed rotation skips the standalone one).
fn bench_lanes(
    gpu: &mut Gpu,
    w: &Weights,
    rng: &mut Rng,
    n: usize,
    iters: usize,
    reps: usize,
) -> Value {
    let table = n.max(2);
    let host: Vec<LaneHost> = (0..table).map(|l| make_lane_host(rng, l, 1)).collect();
    let arm = Arm::new(gpu, table, &host);
    let rows: Vec<(usize, usize)> = (0..table).map(|l| (l, l)).collect();
    arm.stage(gpu, &host, 0, &rows);
    let order: Vec<usize> = (0..table).collect();
    let out_all = arm.out.view(0, table * VALUE);
    let rot_all = arm.rot.view(0, table * VALUE);
    let mut samples: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for rep in 0..=reps {
        for k in 0..3 {
            let mode = [Fuse::Off, Fuse::Rot, Fuse::NoRot][(rep + k) % 3];
            gpu.hip.device_synchronize().expect("sync");
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                run_lanes(gpu, w, &arm, &host, &order, 0, None, mode);
                gpu.rotate_x_mq_batched(&out_all, &rot_all, VALUE, table)
                    .expect("rotate");
            }
            gpu.hip.device_synchronize().expect("sync");
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            if rep > 0 {
                samples[[Fuse::Off, Fuse::Rot, Fuse::NoRot]
                    .iter()
                    .position(|m| *m == mode)
                    .unwrap()]
                .push(us);
            }
        }
    }
    drop((out_all, rot_all));
    arm.free(gpu);
    let stat = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[v.len() / 2], v[0])
    };
    let (off_med, off_min) = stat(&mut samples[0]);
    let (on_med, on_min) = stat(&mut samples[1]);
    let (nr_med, nr_min) = stat(&mut samples[2]);
    println!(
        "bench   lanes={table:<2} chain(3 launches) median={off_med:.2}us min={off_min:.2}us | fused+rotation(1 launch) median={on_med:.2}us min={on_min:.2}us | fused prologue only + rotate launch(2 launches) median={nr_med:.2}us min={nr_min:.2}us | saved median={:.2}us min-vs-min={:.2}us (prologue-only: {:.2}us)",
        off_med - on_med,
        off_min - on_min,
        off_med - nr_med
    );
    json!({
        "lanes": table,
        "chain_median_us": off_med, "chain_min_us": off_min, "chain_samples_us": samples[0],
        "fused_median_us": on_med, "fused_min_us": on_min, "fused_samples_us": samples[1],
        "fused_norot_median_us": nr_med, "fused_norot_min_us": nr_min, "fused_norot_samples_us": samples[2],
        "saved_median_us": off_med - on_med, "saved_min_us": off_min - on_min,
    })
}

fn main() {
    let cli = parse_cli();
    let mut gpu = Gpu::init().expect("Gpu::init");
    let default_enabled = gpu.flags.qwen4_gdn_lanes_enabled();
    // The oracle runs the candidate route whatever the default is.
    Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes = true;
    if !gpu.flags.qwen4_gdn_lanes_enabled() {
        eprintln!(
            "gdn_lanes_oracle: the lanes route is not available on {} (needs exact gfx1151 / gfx1201)",
            gpu.arch
        );
        std::process::exit(2);
    }
    if cli.fuse {
        Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes_fuse = true;
        if !gpu.flags.qwen4_gdn_lanes_fuse_enabled() {
            eprintln!(
                "gdn_lanes_oracle: the fused lanes route is not available on {} (needs exact gfx1201)",
                gpu.arch
            );
            std::process::exit(2);
        }
        // The fused launch's arrival counters are zeroed once, outside every counted launch.
        rdna_compute::tensor_ops::ensure_gdn_pair_counters(&mut gpu).expect("counters");
    }
    println!(
        "arch={} lanes_default_enabled={} steps={} seed={:#x}",
        gpu.arch, default_enabled, cli.steps, cli.seed
    );
    let mut rng = Rng(cli.seed);
    let wh = WeightsHost::new(&mut rng);
    let w = Weights::upload(&mut gpu, &wh);
    let w2 = Weights::upload(&mut gpu, &wh);

    let mut failures = 0usize;
    let mut case_json = Vec::new();
    let mut control_json = Vec::new();
    let mut counts = (0usize, 0usize);

    for &n in &cli.lanes {
        let variants: [(&'static str, &'static str); 4] = [
            ("ident", "small"),
            ("reverse", "mixed"),
            ("shuffle", "large"),
            ("same_pos", "same"),
        ];
        for (vi, (variant, pool_kind)) in variants.into_iter().enumerate() {
            let mut spec = build_spec(&mut rng, n, variant, pool_kind, cli.steps, None, vi == 0);
            spec.fuse = match (cli.fuse, variant) {
                (false, _) => Fuse::Off,
                (true, "same_pos") => Fuse::NoRot,
                (true, _) => Fuse::Rot,
            };
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
        // Controls need two lanes to corrupt; run them at 2, 3 and 8 lanes
        // (as far as the requested lane counts reach).
        let widest = cli.lanes.iter().copied().max().unwrap_or(0);
        let control_lanes: Vec<usize> = [2usize, 3, 8]
            .into_iter()
            .filter(|&n| n <= widest)
            .collect();
        for n in control_lanes {
            for control in Control::ALL {
                let mut spec = build_spec(
                    &mut rng,
                    n,
                    "shuffle",
                    "mixed",
                    cli.steps.min(2),
                    Some(control),
                    false,
                );
                spec.fuse = if cli.fuse { Fuse::Rot } else { Fuse::Off };
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

    if let Some(iters) = cli.bench {
        let mut results = Vec::new();
        for &n in &cli.lanes {
            results.push(bench_lanes(&mut gpu, &w, &mut rng, n, iters, 7));
        }
        let text = serde_json::to_string_pretty(&json!({
            "tool": "gdn_lanes_oracle --bench", "arch": gpu.arch, "iters_per_sample": iters,
            "samples": 7, "results": results,
        }))
        .expect("json");
        if let Some(dir) = &cli.out {
            std::fs::create_dir_all(dir).expect("create --out dir");
            std::fs::write(dir.join("bench.json"), &text).expect("write bench.json");
        }
        println!("{text}");
        w.free(&mut gpu);
        w2.free(&mut gpu);
        std::process::exit(0);
    }
    let (probes_ok, mut probe_json) = admission_probes(&mut gpu, &w, &w2, &mut rng);
    if !probes_ok {
        failures += 1;
    }
    if cli.fuse {
        let (ok, js) = fuse_flag_off_probe(&mut gpu, &w, &mut rng);
        if !ok {
            failures += 1;
        }
        probe_json.push(js);
    }
    w.free(&mut gpu);
    w2.free(&mut gpu);

    let pass = failures == 0;
    let summary = json!({
        "tool": "gdn_lanes_oracle",
        "arch": gpu.arch,
        "lanes_default_enabled": default_enabled,
        "shape": {
            "key_heads": NK, "value_heads": NV, "head_dim": HD, "qkv": QKV,
            "conv_ring": [RING, QKV], "q8_state_bytes": STATE_BYTES,
        },
        "steps": cli.steps,
        "seed": cli.seed,
        "lane_counts": cli.lanes,
        "cases": counts.0,
        "controls": counts.1,
        "failures": failures,
        "pass": pass,
        "launch_counts": {
            "per_lane_launches_per_step": "2 * N",
            "lanes_launches_per_step": if cli.fuse { 1 } else { 2 },
            "fused": cli.fuse,
        },
        "case_results": case_json,
        "control_results": control_json,
        "admission_probes": probe_json,
    });
    let text = serde_json::to_string_pretty(&summary).expect("json");
    if let Some(dir) = &cli.out {
        std::fs::create_dir_all(dir).expect("create --out dir");
        std::fs::write(dir.join("summary.json"), &text).expect("write summary.json");
    }
    println!("{text}");
    println!(
        "gdn_lanes_oracle: {} arch={} cases={} controls={} failures={}",
        if pass { "PASS" } else { "FAIL" },
        gpu.arch,
        counts.0,
        counts.1,
        failures
    );
    std::process::exit(if pass { 0 } else { 1 });
}

/// Host-only self-tests of the harness (`cargo test --features lab --example
/// gdn_lanes_oracle`): no GPU is opened.
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
            // The body is not a guard: changing it is not a violation.
            image[guard * elem] ^= 0xff;
            assert_eq!(guard_violations(raw, guard, len, &image), 0);
            // Every guard byte is watched, head and tail.
            let last = image.len() - 1;
            image[0] ^= 1;
            image[last] ^= 1;
            image[guard * elem - 1] ^= 1;
            image[(guard + len) * elem] ^= 1;
            assert_eq!(guard_violations(raw, guard, len, &image), 4);
            // A sentinel body is all sentinel.
            let blank = Guarded::image_of(raw, guard, len, None);
            assert_eq!(guard_violations(raw, guard, len, &blank), 0);
            let pattern = sentinel_pattern(raw);
            assert!(blank
                .iter()
                .enumerate()
                .all(|(i, b)| *b == pattern[i % pattern.len()]));
        }
    }

    #[test]
    fn nan_sentinel_survives_the_f32_round_trip() {
        let bytes = F32_SENTINEL.to_le_bytes();
        let f = f32::from_le_bytes(bytes);
        assert!(f.is_nan());
        assert_eq!(f32_bytes(&[f]), bytes);
        assert_eq!(f32::from_bits(F32_SENTINEL).to_bits(), F32_SENTINEL);
    }

    #[test]
    fn positions_fit_u32_and_cover_every_cursor() {
        for steps in [1usize, 2, 4, 9] {
            for kind in ["small", "mixed", "large", "same"] {
                let p = pool(kind, steps);
                assert_eq!(p.len(), 8, "{kind}");
                for &base in &p {
                    assert!(base + steps - 1 <= u32::MAX as usize, "{kind} {base}");
                }
            }
        }
        let cursors: std::collections::BTreeSet<usize> =
            pool("small", 4).iter().map(|p| p % RING).collect();
        assert_eq!(cursors.len(), RING);
        let cursors: std::collections::BTreeSet<usize> =
            pool("large", 4).iter().map(|p| p % RING).collect();
        assert_eq!(cursors.len(), RING);
    }

    #[test]
    fn specs_are_permutations_with_decoys_only_for_one_lane() {
        let mut rng = Rng(7);
        for n in 1..=GDN_CORE_LANES_MAX {
            for (variant, kind) in [
                ("ident", "small"),
                ("reverse", "mixed"),
                ("shuffle", "large"),
                ("same_pos", "same"),
            ] {
                let s = build_spec(&mut rng, n, variant, kind, 4, None, false);
                assert_eq!(s.table, n.max(2));
                assert_eq!(s.positions.len(), s.table);
                let mut order = s.order.clone();
                order.sort_unstable();
                assert_eq!(order, (0..s.table).collect::<Vec<_>>(), "{}", s.name);
                let mut ref_order = s.ref_order.clone();
                ref_order.sort_unstable();
                assert_eq!(ref_order, (0..n).collect::<Vec<_>>(), "{}", s.name);
                assert!(s.table == n || (n == 1 && s.table == 2));
            }
        }
    }

    #[test]
    fn diff_counts_mismatching_bytes_and_names_the_item() {
        let snap = |a: u8| Snap {
            items: vec![
                ("lane0.q8".to_string(), vec![1, 2, 3, a]),
                ("lane1.q8".to_string(), vec![9, 9]),
            ],
            guard_bad: Vec::new(),
        };
        assert!(diff(&snap(4), &snap(4)).is_empty());
        assert_eq!(diff(&snap(4), &snap(5)), vec![("lane0.q8".to_string(), 1)]);
    }

    #[test]
    fn state_layout_matches_the_q8_slot() {
        assert_eq!(STATE_BYTES, 811_008);
        assert_eq!(QKV, 10_240);
        let mut rng = Rng(1);
        let host = make_lane_host(&mut rng, 3, 2);
        assert_eq!(host.state.len(), STATE_BYTES);
        assert_eq!(host.ring.len(), RING * QKV);
        assert_eq!(host.steps.len(), 2);
        assert_eq!(host.steps[0].qkv.len(), QKV);
        // Scales are positive and finite.
        let scales = &host.state[NV * HD * HD..];
        assert!(scales
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .all(|s| s > 0.0 && s.is_finite()));
    }
}
