// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tier C oracle of the gfx942 batched GDN decode recurrence
//! (`Gfx942Device::qwen4_gdn_lanes_step_f32`, kernel `qwen4_gdn_lanes_step_f32_gfx942`,
//! `kernels/src/qwen4_gdn_lanes_f32.gfx942.hip`) against the per-lane launches it
//! replaces: one `gated_delta_step_batched` per lane (the production F32 route on gfx942,
//! `gated_delta_step_pipe128_f32`, rows = 1 for decode, rows = 4 for an MTP window).
//!
//! Flash-Next GDN shapes: 16 key heads, 48 value heads, 128 x 128, qkv width 10,240,
//! F32 state 3 MiB per layer per lane.
//!
//! Per case (lanes 1 / 2 / 8 / 33 / 64, rows 1 / 4, `--steps` consecutive steps with the
//! state carried and fresh random inputs each step):
//!   * the lanes' projection / gate / beta rows and output rows live in lane-major pools in
//!     a SHUFFLED slot order with spare slots between them (the lane table must not assume a
//!     uniform stride; the spare output slots must stay sentinel in both arms, the spare
//!     input slots are never read);
//!   * every output is sentinel-filled (NaN payload) before a step: after the reference arm no
//!     sentinel word may remain in a lane's output (it really ran) and states must have moved;
//!   * both arms start from identical random states (checked word for word before running);
//!   * after each step every lane's whole state and the whole output pool are compared word
//!     for word (`f32::to_bits`) between the arms;
//!   * negative controls the comparator and the route MUST detect: one flipped bit in the
//!     candidate's last state word and in an output word (exactly one differing word each); the
//!     candidate launched with two lanes' states swapped; the candidate launched without its
//!     last lane (cases of two or more lanes).
//!
//! `--time LANES,..` (default off): one 36-layer pass (36 layers, each with its own states /
//! inputs / outputs, rows = 1) of the per-lane launches against the batched launches, median
//! and min..max of `--reps` timed passes after warm-up passes, cache scrubbed between reps.
//!
//! `cargo run --release -p rdna-compute --example gdn_lanes_f32_gfx942_oracle --
//!    [--lanes 1,2,8,33,64] [--rows 1,4] [--steps 2] [--seed U64] [--time 8,64]
//!    [--reps 9] [--warm 3] [--skip-controls] [--skip-correctness]`
//!
//! Prints one line per case / control and a final `gdn_lanes_f32_gfx942_oracle: PASS|FAIL`.
//! Exit 0 = PASS, 1 = a mismatch / undetected control, 2 = not exact gfx942 (or
//! `HIPFIRE_QWEN4_GDN_PIPE=0`, which would change the reference kernel).

use rdna_compute::cdna::gfx942::{Qwen4GdnLanesF32Lane, Qwen4GdnLanesF32Shape};
use rdna_compute::tensor_ops::{gated_delta_step_batched, GatedDeltaStepBatched, GdnStateFormat};
use rdna_compute::{DType, Gpu, GpuTensor};

const NK: usize = 16;
const NV: usize = 48;
const HD: usize = 128;
const QKV: usize = 2 * NK * HD + NV * HD;
const VALUE: usize = NV * HD;
const STATE: usize = NV * HD * HD;
const LAYERS: usize = 36;
const SPARE: usize = 3;
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
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn rand_vec(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.range(lo, hi)).collect()
}

fn sentinel_vec(n: usize) -> Vec<f32> {
    vec![f32::from_bits(SENTINEL); n]
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

fn htod(gpu: &Gpu, tensor: &GpuTensor, data: &[f32]) {
    assert_eq!(tensor.numel(), data.len());
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.bind_thread().expect("bind");
    gpu.hip.memcpy_htod(&tensor.buf, bytes).expect("htod");
}

fn dtoh(gpu: &Gpu, tensor: &GpuTensor) -> Vec<f32> {
    gpu.download_f32(tensor).expect("dtoh")
}

fn step_args<'a>(
    projection: &'a GpuTensor,
    gate: &'a GpuTensor,
    beta: &'a GpuTensor,
    state: &'a GpuTensor,
    output: &'a GpuTensor,
    rows: usize,
) -> GatedDeltaStepBatched<'a> {
    GatedDeltaStepBatched {
        projection,
        gate,
        beta,
        state,
        format: GdnStateFormat::F32,
        residual: None,
        output,
        row_states: None,
        rows,
        qkv_width: QKV,
        key_heads: NK,
        value_heads: NV,
        key_dim: HD,
        value_dim: HD,
        position: 0,
    }
}

fn shape(rows: usize) -> Qwen4GdnLanesF32Shape {
    Qwen4GdnLanesF32Shape {
        rows,
        key_heads: NK,
        value_heads: NV,
        key_dim: HD,
        value_dim: HD,
    }
}

struct Cli {
    lanes: Vec<usize>,
    rows: Vec<usize>,
    steps: usize,
    seed: u64,
    time: Vec<usize>,
    reps: usize,
    warm: usize,
    controls: bool,
    correctness: bool,
}

fn csv(v: &str) -> Vec<usize> {
    v.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("integer list"))
        .collect()
}

fn parse_cli() -> Cli {
    let mut cli = Cli {
        lanes: vec![1, 2, 8, 33, 64],
        rows: vec![1, 4],
        steps: 2,
        seed: 0x6D69_3330_3078_6C32,
        time: vec![],
        reps: 9,
        warm: 3,
        controls: true,
        correctness: true,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--lanes" => cli.lanes = csv(&value()),
            "--rows" => cli.rows = csv(&value()),
            "--steps" => cli.steps = value().parse().expect("steps"),
            "--seed" => cli.seed = value().parse().expect("seed"),
            "--time" => cli.time = csv(&value()),
            "--reps" => cli.reps = value().parse().expect("reps"),
            "--warm" => cli.warm = value().parse().expect("warm"),
            "--skip-controls" => cli.controls = false,
            "--skip-correctness" => cli.correctness = false,
            other => panic!("unknown argument {other}"),
        }
    }
    cli
}

/// One lane-set arm: `lanes` states, the lane-major pools in shuffled slot order.
struct Arm {
    states: Vec<GpuTensor>,
    out_pool: GpuTensor,
}

fn run_case(
    gpu: &mut Gpu,
    lanes: usize,
    rows: usize,
    steps: usize,
    controls: bool,
    seed: u64,
) -> bool {
    let mut rng = Rng(seed ^ ((lanes as u64) << 32) ^ ((rows as u64) << 48));
    let pool = lanes + SPARE;
    // Lane l runs in pool slot `slot[l]`; the SPARE leftover slots hold no lane.
    let mut slot: Vec<usize> = (0..pool).collect();
    for i in (1..pool).rev() {
        slot.swap(i, rng.below(i + 1));
    }
    let proj_n = rows * QKV;
    let gate_n = rows * NV;
    let out_n = rows * VALUE;
    let proj_pool = gpu
        .upload_f32(&vec![0.0; pool * proj_n], &[pool * proj_n])
        .unwrap();
    let gate_pool = gpu
        .upload_f32(&vec![0.0; pool * gate_n], &[pool * gate_n])
        .unwrap();
    let beta_pool = gpu
        .upload_f32(&vec![0.0; pool * gate_n], &[pool * gate_n])
        .unwrap();
    let mut arms: Vec<Arm> = Vec::new();
    let init: Vec<Vec<f32>> = (0..lanes)
        .map(|_| rand_vec(&mut rng, STATE, -0.5, 0.5))
        .collect();
    for _ in 0..2 {
        let states = init
            .iter()
            .map(|s| gpu.upload_f32(s, &[STATE]).unwrap())
            .collect();
        let out_pool = gpu
            .upload_f32(&sentinel_vec(pool * out_n), &[pool * out_n])
            .unwrap();
        arms.push(Arm { states, out_pool });
    }
    let cand = arms.pop().unwrap();
    let refr = arms.pop().unwrap();
    let view = |t: &GpuTensor, per: usize, l: usize| t.sub_offset(slot[l] * per, per);
    let proj: Vec<GpuTensor> = (0..lanes).map(|l| view(&proj_pool, proj_n, l)).collect();
    let gate: Vec<GpuTensor> = (0..lanes).map(|l| view(&gate_pool, gate_n, l)).collect();
    let beta: Vec<GpuTensor> = (0..lanes).map(|l| view(&beta_pool, gate_n, l)).collect();
    let out_ref: Vec<GpuTensor> = (0..lanes).map(|l| view(&refr.out_pool, out_n, l)).collect();
    let out_cand: Vec<GpuTensor> = (0..lanes).map(|l| view(&cand.out_pool, out_n, l)).collect();
    let spare_slots: Vec<usize> = slot[lanes..].to_vec();

    let mut ok = true;
    let mut report = |what: &str, good: bool, detail: String| {
        if !good {
            ok = false;
        }
        println!(
            "[lanes={lanes} rows={rows}] {what}: {detail} => {}",
            if good { "ok" } else { "MISMATCH" }
        );
    };

    // Identical starts.
    let mut start_diff = 0;
    for l in 0..lanes {
        start_diff += word_diff(&init[l], &dtoh(gpu, &refr.states[l])).0;
        start_diff += word_diff(&init[l], &dtoh(gpu, &cand.states[l])).0;
    }
    report(
        "identical start states",
        start_diff == 0,
        format!("diff words {start_diff}"),
    );

    for step in 0..steps {
        let p = rand_vec(&mut rng, pool * proj_n, -1.0, 1.0);
        let g = rand_vec(&mut rng, pool * gate_n, -3.0, -0.01);
        let b = rand_vec(&mut rng, pool * gate_n, 0.02, 0.98);
        htod(gpu, &proj_pool, &p);
        htod(gpu, &gate_pool, &g);
        htod(gpu, &beta_pool, &b);
        htod(gpu, &refr.out_pool, &sentinel_vec(pool * out_n));
        htod(gpu, &cand.out_pool, &sentinel_vec(pool * out_n));
        let before: Vec<Vec<f32>> = (0..lanes).map(|l| dtoh(gpu, &refr.states[l])).collect();
        // Reference: one production launch per lane.
        for l in 0..lanes {
            gated_delta_step_batched(
                gpu,
                &step_args(
                    &proj[l],
                    &gate[l],
                    &beta[l],
                    &refr.states[l],
                    &out_ref[l],
                    rows,
                ),
            )
            .expect("per-lane launch");
        }
        // Candidate: one launch.
        let lane_args: Vec<Qwen4GdnLanesF32Lane<'_>> = (0..lanes)
            .map(|l| Qwen4GdnLanesF32Lane {
                projection: &proj[l],
                gate: &gate[l],
                beta: &beta[l],
                state: &cand.states[l],
                output: &out_cand[l],
            })
            .collect();
        gpu.try_gfx942()
            .expect("exact gfx942")
            .qwen4_gdn_lanes_step_f32(&lane_args, shape(rows))
            .expect("lanes launch");
        gpu.hip.device_synchronize().expect("sync");

        let (mut moved, mut state_diff, mut first_bad) = (0usize, 0usize, None);
        for l in 0..lanes {
            let r = dtoh(gpu, &refr.states[l]);
            let c = dtoh(gpu, &cand.states[l]);
            moved += word_diff(&before[l], &r).0;
            let (d, f) = word_diff(&r, &c);
            if d > 0 && first_bad.is_none() {
                first_bad = Some((l, f.unwrap()));
            }
            state_diff += d;
        }
        let out_r = dtoh(gpu, &refr.out_pool);
        let out_c = dtoh(gpu, &cand.out_pool);
        let (out_diff, out_first) = word_diff(&out_r, &out_c);
        let mut unwritten = 0usize;
        for l in 0..lanes {
            let lo = slot[l] * out_n;
            unwritten += out_r[lo..lo + out_n]
                .iter()
                .filter(|w| w.to_bits() == SENTINEL)
                .count();
        }
        let spare_ok = spare_slots.iter().all(|&s| {
            let lo = s * out_n;
            out_r[lo..lo + out_n]
                .iter()
                .all(|w| w.to_bits() == SENTINEL)
                && out_c[lo..lo + out_n]
                    .iter()
                    .all(|w| w.to_bits() == SENTINEL)
        });
        let good = state_diff == 0
            && out_diff == 0
            && moved > 0
            && unwritten == 0
            && spare_ok
            && start_diff == 0;
        report(
            &format!("step {step} lanes launch vs per-lane launches"),
            good,
            format!(
                "state diff words {state_diff} (first {first_bad:?}) | output pool diff words \
                 {out_diff} (first {out_first:?}) | reference moved {moved} state words, \
                 sentinel left in lane outputs {unwritten}, spare slots untouched {spare_ok}"
            ),
        );

        if controls && step + 1 == steps {
            // Control 1: one flipped bit in the candidate's last state word / an output word.
            let last = lanes - 1;
            let r = dtoh(gpu, &refr.states[last]);
            let mut c = dtoh(gpu, &cand.states[last]);
            c[STATE - 1] = f32::from_bits(c[STATE - 1].to_bits() ^ 1);
            let (d, _) = word_diff(&r, &c);
            let mut oc = out_c.clone();
            oc[slot[last] * out_n + out_n / 2] =
                f32::from_bits(oc[slot[last] * out_n + out_n / 2].to_bits() ^ 0x0040_0000);
            let (od, _) = word_diff(&out_r, &oc);
            report(
                "control: one flipped bit (state word, output word)",
                d == 1 && od == 1,
                format!("comparator sees {d} and {od} differing words (want 1 and 1)"),
            );
            // Controls 2 and 3 run the candidate again from the step's start states.
            if lanes >= 2 {
                for (name, swap, drop_last) in [
                    ("two lanes' states swapped", true, false),
                    ("last lane dropped", false, true),
                ] {
                    for l in 0..lanes {
                        htod(gpu, &cand.states[l], &before[l]);
                    }
                    htod(gpu, &cand.out_pool, &sentinel_vec(pool * out_n));
                    let mut order: Vec<usize> = (0..lanes).collect();
                    let mut args: Vec<Qwen4GdnLanesF32Lane<'_>> = Vec::new();
                    if swap {
                        order.swap(0, 1);
                    }
                    for (i, &l) in order.iter().enumerate() {
                        if drop_last && i + 1 == lanes {
                            continue;
                        }
                        // `swap`: lane 0's inputs run on lane 1's state and vice versa.
                        args.push(Qwen4GdnLanesF32Lane {
                            projection: &proj[i],
                            gate: &gate[i],
                            beta: &beta[i],
                            state: &cand.states[l],
                            output: &out_cand[i],
                        });
                    }
                    gpu.try_gfx942()
                        .unwrap()
                        .qwen4_gdn_lanes_step_f32(&args, shape(rows))
                        .expect("control launch");
                    gpu.hip.device_synchronize().unwrap();
                    let mut bad = 0usize;
                    for l in 0..lanes {
                        bad +=
                            word_diff(&dtoh(gpu, &refr.states[l]), &dtoh(gpu, &cand.states[l])).0;
                    }
                    bad += word_diff(&out_r, &dtoh(gpu, &cand.out_pool)).0;
                    report(
                        &format!("control: {name}"),
                        bad > 0,
                        format!("{bad} differing words (must be detected, want > 0)"),
                    );
                }
            }
        }
    }
    drop((proj, gate, beta, out_ref, out_cand));
    for t in refr.states.into_iter().chain(cand.states) {
        gpu.free_tensor(t).ok();
    }
    for t in [
        proj_pool,
        gate_pool,
        beta_pool,
        refr.out_pool,
        cand.out_pool,
    ] {
        gpu.free_tensor(t).ok();
    }
    ok
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
    }
}

/// 36 layers x `lanes` lanes, rows = 1: the per-lane launch chain against the batched one.
fn time_case(gpu: &mut Gpu, lanes: usize, reps: usize, warm: usize, seed: u64) {
    let mut rng = Rng(seed ^ 0x7131 ^ ((lanes as u64) << 24));
    let rows = 1;
    struct Layer {
        proj: GpuTensor,
        gate: GpuTensor,
        beta: GpuTensor,
        out: GpuTensor,
        states: Vec<GpuTensor>,
    }
    let layers: Vec<Layer> = (0..LAYERS)
        .map(|_| Layer {
            proj: gpu
                .upload_f32(&rand_vec(&mut rng, lanes * QKV, -1.0, 1.0), &[lanes * QKV])
                .unwrap(),
            gate: gpu
                .upload_f32(&rand_vec(&mut rng, lanes * NV, -3.0, -0.01), &[lanes * NV])
                .unwrap(),
            beta: gpu
                .upload_f32(&rand_vec(&mut rng, lanes * NV, 0.02, 0.98), &[lanes * NV])
                .unwrap(),
            out: gpu
                .upload_f32(&vec![0.0; lanes * VALUE], &[lanes * VALUE])
                .unwrap(),
            states: (0..lanes)
                .map(|_| {
                    gpu.upload_f32(&rand_vec(&mut rng, STATE, -0.5, 0.5), &[STATE])
                        .unwrap()
                })
                .collect(),
        })
        .collect();
    let lane_views: Vec<Vec<[GpuTensor; 4]>> = layers
        .iter()
        .map(|y| {
            (0..lanes)
                .map(|l| {
                    [
                        y.proj.sub_offset(l * QKV, QKV),
                        y.gate.sub_offset(l * NV, NV),
                        y.beta.sub_offset(l * NV, NV),
                        y.out.sub_offset(l * VALUE, VALUE),
                    ]
                })
                .collect()
        })
        .collect();
    let scrub = gpu.alloc_tensor(&[1usize << 28], DType::F32).unwrap();
    let lane_args: Vec<Vec<Qwen4GdnLanesF32Lane<'_>>> = layers
        .iter()
        .zip(&lane_views)
        .map(|(y, views)| {
            views
                .iter()
                .zip(&y.states)
                .map(|(v, state)| Qwen4GdnLanesF32Lane {
                    projection: &v[0],
                    gate: &v[1],
                    beta: &v[2],
                    state,
                    output: &v[3],
                })
                .collect()
        })
        .collect();

    let e0 = gpu.hip.event_create().unwrap();
    let e1 = gpu.hip.event_create().unwrap();
    let time_pass = |gpu: &mut Gpu, batched: bool| -> f64 {
        gpu.bind_thread().unwrap();
        gpu.hip.memset(&scrub.buf, 0x5A, scrub.byte_size()).unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.hip.event_record(&e0, None).unwrap();
        for (y, layer) in layers.iter().enumerate() {
            if batched {
                gpu.try_gfx942()
                    .unwrap()
                    .qwen4_gdn_lanes_step_f32(&lane_args[y], shape(rows))
                    .unwrap();
            } else {
                for l in 0..lanes {
                    let v = &lane_views[y][l];
                    gated_delta_step_batched(
                        gpu,
                        &step_args(&v[0], &v[1], &v[2], &layer.states[l], &v[3], rows),
                    )
                    .unwrap();
                }
            }
        }
        gpu.hip.event_record(&e1, None).unwrap();
        gpu.hip.event_synchronize(&e1).unwrap();
        gpu.hip.event_elapsed_ms(&e0, &e1).unwrap() as f64
    };
    let mut results = Vec::new();
    for batched in [false, true] {
        for _ in 0..warm {
            time_pass(gpu, batched);
        }
        let mut ms: Vec<f64> = (0..reps).map(|_| time_pass(gpu, batched)).collect();
        let (mn, mx) = (
            ms.iter().cloned().fold(f64::MAX, f64::min),
            ms.iter().cloned().fold(0.0, f64::max),
        );
        results.push((median(&mut ms), mn, mx));
    }
    println!(
        "[time lanes={lanes} rows=1 layers={LAYERS}] 36-layer pass ms median [min..max] over {reps} reps \
         (warm {warm}, 1 GiB scrub per rep): per-lane launches {:.3} [{:.3}..{:.3}] ({} launches) | \
         one launch per layer {:.3} [{:.3}..{:.3}] ({} launches) | speedup {:.2}x",
        results[0].0,
        results[0].1,
        results[0].2,
        LAYERS * lanes,
        results[1].0,
        results[1].1,
        results[1].2,
        LAYERS,
        results[0].0 / results[1].0,
    );
    gpu.hip.event_destroy(e0).ok();
    gpu.hip.event_destroy(e1).ok();
    drop(lane_args);
    drop(lane_views);
    for y in layers {
        for t in y.states.into_iter().chain([y.proj, y.gate, y.beta, y.out]) {
            gpu.free_tensor(t).ok();
        }
    }
    gpu.free_tensor(scrub).ok();
}

fn main() {
    let cli = parse_cli();
    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.try_gfx942().is_none() {
        eprintln!(
            "gdn_lanes_f32_gfx942_oracle: needs exact gfx942, this device is {}",
            gpu.arch
        );
        std::process::exit(2);
    }
    if std::env::var("HIPFIRE_QWEN4_GDN_PIPE").as_deref() == Ok("0") {
        eprintln!(
            "gdn_lanes_f32_gfx942_oracle: HIPFIRE_QWEN4_GDN_PIPE=0 selects persistent256 as the \
             per-lane reference; unset it"
        );
        std::process::exit(2);
    }
    println!(
        "device {} | GDN shapes: key heads {NK}, value heads {NV}, {HD}x{HD}, qkv width {QKV}, \
         state {} MiB/lane/layer | steps {}",
        gpu.arch,
        STATE * 4 >> 20,
        cli.steps
    );
    let mut all = true;
    if cli.correctness {
        for &rows in &cli.rows {
            for &lanes in &cli.lanes {
                all &= run_case(&mut gpu, lanes, rows, cli.steps, cli.controls, cli.seed);
            }
        }
    }
    for &lanes in &cli.time {
        time_case(&mut gpu, lanes, cli.reps, cli.warm, cli.seed);
    }
    println!(
        "gdn_lanes_f32_gfx942_oracle: {}",
        if all { "PASS" } else { "FAIL" }
    );
    std::process::exit(if all { 0 } else { 1 });
}
