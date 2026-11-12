// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tier C oracle of the gfx942 GDN Core lanes route (`Gfx942Device::qwen4_gdn_lanes_verify`,
//! module `qwen4_g942_gdn_lanes_verify`: `..._conv_params_k4`, `..._step_f32`, `..._gate_batched`).
//!
//! A lane's GDN Core on gfx942 is, per lane, `copy_d2d(projection -> capture inputs)` (verify
//! lanes under the armed ring), `gated_delta_conv_params_batched`, `gated_delta_step_batched`
//! (the persistent capture launch with `row_states`, or the one-row AR launch) and
//! `gated_delta_gate_batched`. The lanes route runs the same chain for a whole group of lanes in
//! three launches, each lane on its own rows (ragged: verify windows of 2..=8 rows and one-row AR
//! lanes in one group). Every word the chains write must be BYTE-identical:
//! the capture-input copy, the convolution ring, the convolution output, gate and beta, the
//! capture ring (all slots; only slot `rows - 1` is written), the live state (untouched under a
//! ring, updated in place for an AR lane), the recurrent output and the gated output.
//!
//! Each case is a list of per-lane rows (`uniform4`: 8 x 4 rows, the C8 K=3 window; `ragged`,
//! `mixed` with one-row AR lanes, `all_ar`, `single`, `wide`: 8 x 8 rows) run for `--steps`
//! carried windows with a fresh conv cursor per lane per window; a captured lane aliasing its
//! projection and capture input (the direct in-projection) skips the copy in both arms.
//! Negative controls the comparator MUST detect: one flipped bit of the candidate's gated
//! output, lanes' pointers swapped, the last lane not launched, and one lane's conv cursor
//! off by one.
//!
//! `cargo run --release -p rdna-compute --example gdn_lanes_verify_f32_gfx942_oracle --
//!    [--cases uniform4,ragged,...] [--steps 3] [--seed U64] [--skip-controls]`
//!
//! `HIPFIRE_QWEN4_G942_GDN_LANES_VERIFY_IMAGE=/absolute/device.elf` runs the candidate on that
//! code object (the Route-A twin proof); the reference chain is the production per-lane route
//! (the verify module / tensor_ops source). Exit 0 = PASS, 1 = a mismatch or an undetected
//! control, 2 = not exact gfx942.

use rdna_compute::cdna::gfx942::{
    Qwen4GdnLanesVerifyLane, Qwen4GdnLanesVerifyShape, QWEN4_G942_GDN_LANES_VERIFY_MAX,
};
use rdna_compute::tensor_ops::{
    gated_delta_conv_params_batched, gated_delta_gate_batched, gated_delta_step_batched,
    GatedDeltaConvBatched, GatedDeltaGateBatched, GatedDeltaParamsBatched, GatedDeltaStepBatched,
    GdnStateFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const NK: usize = 16;
const NV: usize = 48;
const HD: usize = 128;
const QKV: usize = 2 * NK * HD + NV * HD;
const VALUE: usize = NV * HD;
const STATE: usize = NV * HD * HD;
const RING_SLOTS: usize = 8;
const SENTINEL: u32 = 0xFFFF_FFFF;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

fn rand_vec(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.range(lo, hi)).collect()
}

/// `(differing words, first differing index)`.
fn word_diff(a: &[f32], b: &[f32]) -> (usize, Option<usize>) {
    assert_eq!(a.len(), b.len());
    let mut count = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() != y.to_bits() {
            count += 1;
            first.get_or_insert(i);
        }
    }
    (count, first)
}

fn htod_f32(gpu: &Gpu, tensor: &GpuTensor, data: &[f32]) {
    assert_eq!(tensor.numel(), data.len());
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.bind_thread().expect("bind");
    gpu.hip.memcpy_htod(&tensor.buf, bytes).expect("htod");
}

fn dtoh(gpu: &Gpu, tensor: &GpuTensor) -> Vec<f32> {
    gpu.download_f32(tensor).expect("dtoh")
}

fn upload_bf16(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bits: Vec<u16> = data.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let tensor = gpu
        .alloc_tensor(&[bits.len()], DType::BF16)
        .expect("alloc bf16");
    let bytes = unsafe { std::slice::from_raw_parts(bits.as_ptr() as *const u8, bits.len() * 2) };
    gpu.bind_thread().expect("bind");
    gpu.hip.memcpy_htod(&tensor.buf, bytes).expect("htod bf16");
    tensor
}

/// One arm's buffers of one lane (reference and candidate each own a full set).
struct LaneBufs {
    rows: usize,
    captured: bool,
    /// The lane's projection and capture input are one tensor (a direct in-projection).
    direct: bool,
    projection: GpuTensor,
    inputs: GpuTensor,
    /// `[conv output | gate | beta]` rows, as the capture's retained recurrence input.
    recurrence: GpuTensor,
    kernel: GpuTensor,
    history: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    a_log: GpuTensor,
    dt_bias: GpuTensor,
    state: GpuTensor,
    ring: GpuTensor,
    recurrent_output: GpuTensor,
    z: GpuTensor,
    norm: GpuTensor,
    gdn_output: GpuTensor,
}

impl LaneBufs {
    fn new(gpu: &mut Gpu, rows: usize, direct: bool, weights: &Weights, init: &LaneInit) -> Self {
        let captured = rows >= 2;
        let sentinel_rows = |gpu: &mut Gpu, n: usize| {
            gpu.upload_f32(&vec![f32::from_bits(SENTINEL); n], &[n])
                .expect("sentinel")
        };
        let projection = gpu.zeros(&[rows * QKV], DType::F32).unwrap();
        let inputs = if direct && captured {
            projection.clone()
        } else {
            sentinel_rows(gpu, rows * QKV)
        };
        let recurrence_n = rows * (QKV + 2 * NV);
        Self {
            rows,
            captured,
            direct: direct && captured,
            projection,
            inputs,
            recurrence: sentinel_rows(gpu, recurrence_n),
            kernel: upload_bf16(gpu, &weights.kernel),
            history: gpu.upload_f32(&init.history, &[3 * QKV]).unwrap(),
            a: gpu.zeros(&[rows * NV], DType::F32).unwrap(),
            b: gpu.zeros(&[rows * NV], DType::F32).unwrap(),
            a_log: upload_bf16(gpu, &weights.a_log),
            dt_bias: upload_bf16(gpu, &weights.dt_bias),
            state: gpu.upload_f32(&init.state, &[STATE]).unwrap(),
            ring: sentinel_rows(gpu, RING_SLOTS * STATE),
            recurrent_output: sentinel_rows(gpu, rows * VALUE),
            z: gpu.zeros(&[rows * VALUE], DType::F32).unwrap(),
            norm: upload_bf16(gpu, &weights.norm),
            gdn_output: sentinel_rows(gpu, rows * VALUE),
        }
    }

    fn conv_output(&self) -> GpuTensor {
        self.recurrence.sub_offset(0, self.rows * QKV)
    }

    fn gate(&self) -> GpuTensor {
        self.recurrence.sub_offset(self.rows * QKV, self.rows * NV)
    }

    fn beta(&self) -> GpuTensor {
        self.recurrence
            .sub_offset(self.rows * QKV + self.rows * NV, self.rows * NV)
    }

    fn all(self) -> Vec<GpuTensor> {
        let mut all = vec![
            self.projection,
            self.recurrence,
            self.kernel,
            self.history,
            self.a,
            self.b,
            self.a_log,
            self.dt_bias,
            self.state,
            self.ring,
            self.recurrent_output,
            self.z,
            self.norm,
            self.gdn_output,
        ];
        if !self.direct {
            all.push(self.inputs);
        }
        all
    }
}

struct Weights {
    kernel: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
}

struct LaneInit {
    history: Vec<f32>,
    state: Vec<f32>,
}

/// The window's per-lane host data (identical for both arms).
struct Window {
    projection: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    z: Vec<f32>,
    cursor: usize,
}

fn lane_rows(case: &str) -> Vec<usize> {
    match case {
        "uniform4" => vec![4; 8],
        "uniform2" => vec![2; 8],
        "ragged" => vec![4, 2, 3, 8, 4, 5, 2, 6],
        "mixed" => vec![4, 1, 4, 1, 3, 1, 2, 4],
        "all_ar" => vec![1; 8],
        "single" => vec![3],
        "pair" => vec![2, 1],
        "wide" => vec![8; 8],
        other => panic!("unknown case {other}"),
    }
}

/// The per-lane production chain of one lane (what `execute_gated_delta_net_hc_impl` runs for a
/// Core op, in the order and with the arguments the dispatcher's capture / AR branch uses).
fn run_reference(gpu: &mut Gpu, lane: &LaneBufs, cursor: usize) {
    let rows = lane.rows;
    let conv_output = lane.conv_output();
    let (gate, beta) = (lane.gate(), lane.beta());
    if lane.captured && !lane.direct {
        gpu.copy_d2d(&lane.projection, &lane.inputs, lane.inputs.byte_size())
            .expect("capture-input copy");
    }
    let conv_input = if lane.captured {
        &lane.inputs
    } else {
        &lane.projection
    };
    let conv = GatedDeltaConvBatched {
        input: conv_input,
        kernel: &lane.kernel,
        history: &lane.history,
        output: &conv_output,
        next_history: &lane.history,
        rows,
        channels: QKV,
        history_rows: 3,
        kernel_size: 4,
        start_cursor: cursor,
    };
    let params = GatedDeltaParamsBatched {
        a: &lane.a,
        b: &lane.b,
        a_log: &lane.a_log,
        dt_bias: &lane.dt_bias,
        gate: &gate,
        beta: &beta,
        rows,
        heads: NV,
    };
    gated_delta_conv_params_batched(gpu, &conv, &params).expect("per-lane conv + params");
    gated_delta_step_batched(
        gpu,
        &GatedDeltaStepBatched {
            projection: &conv_output,
            gate: &gate,
            beta: &beta,
            state: &lane.state,
            format: GdnStateFormat::F32,
            residual: None,
            output: &lane.recurrent_output,
            row_states: lane.captured.then_some(&lane.ring),
            rows,
            qkv_width: QKV,
            key_heads: NK,
            value_heads: NV,
            key_dim: HD,
            value_dim: HD,
            position: 0,
        },
    )
    .expect("per-lane recurrence");
    gated_delta_gate_batched(
        gpu,
        &GatedDeltaGateBatched {
            recurrent_output: &lane.recurrent_output,
            z: &lane.z,
            norm: &lane.norm,
            output: &lane.gdn_output,
            rows,
            value_heads: NV,
            value_dim: HD,
        },
    )
    .expect("per-lane gated norm");
}

fn candidate_lanes<'a>(
    bufs: &'a [LaneBufs],
    views: &'a [(GpuTensor, GpuTensor, GpuTensor)],
    cursors: &[usize],
) -> Vec<Qwen4GdnLanesVerifyLane<'a>> {
    bufs.iter()
        .zip(views)
        .zip(cursors)
        .map(
            |((lane, (conv_output, gate, beta)), &cursor)| Qwen4GdnLanesVerifyLane {
                projection: &lane.projection,
                inputs: (lane.captured && !lane.direct).then_some(&lane.inputs),
                kernel: &lane.kernel,
                history: &lane.history,
                conv_output,
                a: &lane.a,
                b: &lane.b,
                a_log: &lane.a_log,
                dt_bias: &lane.dt_bias,
                gate,
                beta,
                state: &lane.state,
                ring: lane.captured.then_some(&lane.ring),
                recurrent_output: &lane.recurrent_output,
                z: &lane.z,
                norm: &lane.norm,
                gdn_output: &lane.gdn_output,
                rows: lane.rows,
                start_cursor: cursor,
            },
        )
        .collect()
}

/// Per-lane word diffs of everything a Core chain writes: `(name, differing words)`.
fn compare(gpu: &Gpu, a: &LaneBufs, b: &LaneBufs) -> Vec<(&'static str, usize, Option<usize>)> {
    let pairs: [(&'static str, &GpuTensor, &GpuTensor); 10] = [
        ("inputs", &a.inputs, &b.inputs),
        ("history", &a.history, &b.history),
        ("recurrence", &a.recurrence, &b.recurrence),
        ("state", &a.state, &b.state),
        ("ring", &a.ring, &b.ring),
        ("recurrent_output", &a.recurrent_output, &b.recurrent_output),
        ("gdn_output", &a.gdn_output, &b.gdn_output),
        ("projection", &a.projection, &b.projection),
        ("a", &a.a, &b.a),
        ("z", &a.z, &b.z),
    ];
    pairs
        .into_iter()
        .map(|(name, x, y)| {
            let (d, first) = word_diff(&dtoh(gpu, x), &dtoh(gpu, y));
            (name, d, first)
        })
        .collect()
}

fn run_case(gpu: &mut Gpu, case: &str, steps: usize, controls: bool, seed: u64) -> bool {
    let rows_per_lane = lane_rows(case);
    let lanes = rows_per_lane.len();
    let mut rng = Rng(seed ^ ((lanes as u64) << 20) ^ (case.len() as u64) << 48 ^ 0xC0DE);
    let mut ok = true;
    let mut report = |what: &str, good: bool, detail: String| {
        if !good {
            ok = false;
        }
        println!(
            "[verify lanes case={case} rows={rows_per_lane:?}] {what}: {detail} => {}",
            if good { "ok" } else { "MISMATCH" }
        );
    };
    let weights = Weights {
        kernel: rand_vec(&mut rng, QKV * 4, -1.0, 1.0),
        a_log: rand_vec(&mut rng, NV, -1.0, 1.0),
        dt_bias: rand_vec(&mut rng, NV, -1.0, 1.0),
        norm: rand_vec(&mut rng, HD, 0.5, 1.5),
    };
    let inits: Vec<LaneInit> = (0..lanes)
        .map(|_| LaneInit {
            history: rand_vec(&mut rng, 3 * QKV, -1.0, 1.0),
            state: rand_vec(&mut rng, STATE, -0.5, 0.5),
        })
        .collect();
    // Every third captured lane aliases its projection and capture input (the direct route).
    let direct: Vec<bool> = (0..lanes).map(|l| l % 3 == 2).collect();
    let mut reference: Vec<LaneBufs> = (0..lanes)
        .map(|l| LaneBufs::new(gpu, rows_per_lane[l], direct[l], &weights, &inits[l]))
        .collect();
    let mut cand: Vec<LaneBufs> = (0..lanes)
        .map(|l| LaneBufs::new(gpu, rows_per_lane[l], direct[l], &weights, &inits[l]))
        .collect();
    let views: Vec<(GpuTensor, GpuTensor, GpuTensor)> = cand
        .iter()
        .map(|lane| (lane.conv_output(), lane.gate(), lane.beta()))
        .collect();
    let shape = Qwen4GdnLanesVerifyShape {
        key_heads: NK,
        value_heads: NV,
        key_dim: HD,
        value_dim: HD,
        conv_kernel: 4,
    };
    let reset_cand = |gpu: &mut Gpu, cand: &[LaneBufs], reference: &[LaneBufs]| {
        // The candidate's written words back to the sentinel / the reference's pre-window bytes.
        for (c, r) in cand.iter().zip(reference) {
            let _ = (c, r);
        }
        let _ = gpu;
    };
    let _ = reset_cand;
    for window in 0..steps {
        let host: Vec<Window> = (0..lanes)
            .map(|l| {
                let rows = rows_per_lane[l];
                Window {
                    projection: rand_vec(&mut rng, rows * QKV, -1.0, 1.0),
                    a: rand_vec(&mut rng, rows * NV, -3.0, 3.0),
                    b: rand_vec(&mut rng, rows * NV, -3.0, 3.0),
                    z: rand_vec(&mut rng, rows * VALUE, -2.0, 2.0),
                    cursor: (rng.next() % 3) as usize,
                }
            })
            .collect();
        let sentinel_ring = vec![f32::from_bits(SENTINEL); RING_SLOTS * STATE];
        for (arm, set) in [&mut reference, &mut cand].into_iter().enumerate() {
            let _ = arm;
            for (l, lane) in set.iter().enumerate() {
                let w = &host[l];
                let rows = lane.rows;
                htod_f32(gpu, &lane.projection, &w.projection);
                htod_f32(gpu, &lane.a, &w.a);
                htod_f32(gpu, &lane.b, &w.b);
                htod_f32(gpu, &lane.z, &w.z);
                if !lane.direct {
                    htod_f32(
                        gpu,
                        &lane.inputs,
                        &vec![f32::from_bits(SENTINEL); rows * QKV],
                    );
                }
                htod_f32(
                    gpu,
                    &lane.recurrence,
                    &vec![f32::from_bits(SENTINEL); rows * (QKV + 2 * NV)],
                );
                htod_f32(gpu, &lane.ring, &sentinel_ring);
                let sentinel_out = vec![f32::from_bits(SENTINEL); rows * VALUE];
                htod_f32(gpu, &lane.recurrent_output, &sentinel_out);
                htod_f32(gpu, &lane.gdn_output, &sentinel_out);
            }
        }
        // Snapshot the candidate's pre-window history / state for the controls.
        let pre_history: Vec<Vec<f32>> = cand.iter().map(|l| dtoh(gpu, &l.history)).collect();
        let pre_state: Vec<Vec<f32>> = cand.iter().map(|l| dtoh(gpu, &l.state)).collect();
        let cursors: Vec<usize> = host.iter().map(|w| w.cursor).collect();
        // Reference: the per-lane chain of every lane.
        for (lane, w) in reference.iter().zip(&host) {
            run_reference(gpu, lane, w.cursor);
        }
        // Candidate: the lanes route.
        {
            let lane_args = candidate_lanes(&cand, &views, &cursors);
            let mut device = gpu.try_gfx942().expect("exact gfx942");
            for chunk in lane_args.chunks(QWEN4_G942_GDN_LANES_VERIFY_MAX) {
                device
                    .qwen4_gdn_lanes_verify(chunk, shape)
                    .expect("lanes verify launch");
            }
        }
        gpu.hip.device_synchronize().expect("sync");
        let mut total = 0usize;
        let mut worst: Option<(usize, &'static str, usize)> = None;
        let mut written = 0usize;
        for l in 0..lanes {
            for (name, d, first) in compare(gpu, &reference[l], &cand[l]) {
                total += d;
                if d > 0 && worst.is_none() {
                    worst = Some((l, name, first.unwrap()));
                }
            }
            if cand[l].captured {
                written += dtoh(gpu, &cand[l].ring)
                    .iter()
                    .filter(|w| w.to_bits() != SENTINEL)
                    .count();
            }
        }
        let want_written: usize = cand.iter().filter(|l| l.captured).count() * STATE;
        report(
            &format!("window {window} lanes route vs per-lane chains"),
            total == 0 && written == want_written,
            format!(
                "differing words {total} (first {worst:?}), ring words written {written} \
                 (want {want_written})"
            ),
        );
        if controls && window == 0 {
            // 1. One flipped bit in the candidate's gated output of the last lane.
            let last = lanes - 1;
            let mut bad = dtoh(gpu, &cand[last].gdn_output);
            let at = bad.len() - 1;
            bad[at] = f32::from_bits(bad[at].to_bits() ^ 1);
            let d = word_diff(&dtoh(gpu, &reference[last].gdn_output), &bad).0;
            report(
                "control: flipped gated-output bit",
                d == 1,
                format!("diff words {d}"),
            );
            if lanes >= 2 && rows_per_lane[0] == rows_per_lane[1] {
                // 2. Lane 0 and 1 with their pointers swapped (same rows): outputs must differ.
                for lane in [&cand[0], &cand[1]] {
                    htod_f32(
                        gpu,
                        &lane.gdn_output,
                        &vec![f32::from_bits(SENTINEL); lane.rows * VALUE],
                    );
                }
                let mut swapped = candidate_lanes(&cand, &views, &cursors);
                let (g0, g1) = (swapped[0].gdn_output, swapped[1].gdn_output);
                swapped[0].gdn_output = g1;
                swapped[1].gdn_output = g0;
                // Restore the pre-window histories / states the first launch advanced.
                for (l, lane) in cand.iter().enumerate() {
                    htod_f32(gpu, &lane.history, &pre_history[l]);
                    htod_f32(gpu, &lane.state, &pre_state[l]);
                }
                gpu.try_gfx942()
                    .expect("exact gfx942")
                    .qwen4_gdn_lanes_verify(
                        &swapped[..swapped.len().min(QWEN4_G942_GDN_LANES_VERIFY_MAX)],
                        shape,
                    )
                    .expect("swapped launch");
                gpu.hip.device_synchronize().expect("sync");
                let d = word_diff(
                    &dtoh(gpu, &reference[0].gdn_output),
                    &dtoh(gpu, &cand[0].gdn_output),
                )
                .0 + word_diff(
                    &dtoh(gpu, &reference[1].gdn_output),
                    &dtoh(gpu, &cand[1].gdn_output),
                )
                .0;
                report(
                    "control: lanes' outputs swapped",
                    d > 0,
                    format!("diff words {d}"),
                );
            }
            // 3. The last lane not launched: its gated output stays the sentinel.
            for (l, lane) in cand.iter().enumerate() {
                htod_f32(gpu, &lane.history, &pre_history[l]);
                htod_f32(gpu, &lane.state, &pre_state[l]);
            }
            htod_f32(
                gpu,
                &cand[last].gdn_output,
                &vec![f32::from_bits(SENTINEL); cand[last].rows * VALUE],
            );
            if lanes >= 2 {
                let all = candidate_lanes(&cand, &views, &cursors);
                gpu.try_gfx942()
                    .expect("exact gfx942")
                    .qwen4_gdn_lanes_verify(&all[..lanes - 1], shape)
                    .expect("short launch");
                gpu.hip.device_synchronize().expect("sync");
                let d = word_diff(
                    &dtoh(gpu, &reference[last].gdn_output),
                    &dtoh(gpu, &cand[last].gdn_output),
                )
                .0;
                report(
                    "control: last lane not launched",
                    d > 0,
                    format!("diff words {d}"),
                );
            }
            // 4. One lane's conv cursor off by one: its convolution ring must differ.
            for (l, lane) in cand.iter().enumerate() {
                htod_f32(gpu, &lane.history, &pre_history[l]);
                htod_f32(gpu, &lane.state, &pre_state[l]);
            }
            let mut skewed = cursors.clone();
            skewed[0] = (skewed[0] + 1) % 3;
            {
                let all = candidate_lanes(&cand, &views, &skewed);
                gpu.try_gfx942()
                    .expect("exact gfx942")
                    .qwen4_gdn_lanes_verify(
                        &all[..all.len().min(QWEN4_G942_GDN_LANES_VERIFY_MAX)],
                        shape,
                    )
                    .expect("skewed launch");
            }
            gpu.hip.device_synchronize().expect("sync");
            let d = word_diff(
                &dtoh(gpu, &reference[0].history),
                &dtoh(gpu, &cand[0].history),
            )
            .0;
            report(
                "control: conv cursor off by one",
                d > 0,
                format!("diff words {d}"),
            );
            // Put the candidate back into the reference's end-of-window state.
            for (l, lane) in cand.iter().enumerate() {
                for (dst, src) in [
                    (&lane.history, &reference[l].history),
                    (&lane.state, &reference[l].state),
                ] {
                    htod_f32(gpu, dst, &dtoh(gpu, src));
                }
                for (dst, src) in [
                    (&lane.gdn_output, &reference[l].gdn_output),
                    (&lane.recurrent_output, &reference[l].recurrent_output),
                    (&lane.ring, &reference[l].ring),
                    (&lane.recurrence, &reference[l].recurrence),
                ] {
                    htod_f32(gpu, dst, &dtoh(gpu, src));
                }
            }
        }
        // Carry: a captured lane's next window starts from this window's final state.
        for l in 0..lanes {
            if reference[l].captured {
                let r = dtoh(gpu, &reference[l].ring);
                let slot = reference[l].rows - 1;
                let final_state = &r[slot * STATE..(slot + 1) * STATE];
                htod_f32(gpu, &reference[l].state, final_state);
                htod_f32(gpu, &cand[l].state, final_state);
            }
        }
    }
    for lane in reference.into_iter().chain(cand) {
        for t in lane.all() {
            gpu.free_tensor(t).ok();
        }
    }
    ok
}

fn main() {
    let mut cases: Vec<String> = [
        "uniform4", "ragged", "mixed", "all_ar", "single", "pair", "wide",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let mut steps = 3usize;
    let mut seed = 0x6D69_3330_3078_7663u64;
    let mut controls = true;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--cases" => {
                cases = value()
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect()
            }
            "--steps" => steps = value().parse().expect("steps"),
            "--seed" => seed = value().parse().expect("seed"),
            "--skip-controls" => controls = false,
            other => panic!("unknown argument {other}"),
        }
    }
    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.try_gfx942().is_none() {
        eprintln!(
            "gdn_lanes_verify_f32_gfx942_oracle: needs exact gfx942, this device is {}",
            gpu.arch
        );
        std::process::exit(2);
    }
    if std::env::var("HIPFIRE_QWEN4_GDN_PIPE").as_deref() == Ok("0") {
        eprintln!(
            "gdn_lanes_verify_f32_gfx942_oracle: HIPFIRE_QWEN4_GDN_PIPE=0 changes the reference \
             kernel; unset it"
        );
        std::process::exit(2);
    }
    let image = match std::env::var("HIPFIRE_QWEN4_G942_GDN_LANES_VERIFY_IMAGE") {
        Ok(path) if !path.is_empty() => {
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            format!("HIPFIRE_QWEN4_G942_GDN_LANES_VERIFY_IMAGE={path} ({bytes} bytes)")
        }
        _ => "HIPFIRE_QWEN4_G942_GDN_LANES_VERIFY_IMAGE unset (hipcc source)".to_string(),
    };
    println!(
        "device {} | GDN shapes: key heads {NK}, value heads {NV}, {HD}x{HD}, qkv width {QKV} | \
         steps {steps} | candidate route: {image}",
        gpu.arch
    );
    let mut all = true;
    for case in &cases {
        all &= run_case(&mut gpu, case, steps, controls, seed);
    }
    println!(
        "gdn_lanes_verify_f32_gfx942_oracle: {} [route: {image}]",
        if all { "PASS" } else { "FAIL" }
    );
    std::process::exit(if all { 0 } else { 1 });
}
