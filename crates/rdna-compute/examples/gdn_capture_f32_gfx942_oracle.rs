// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tier C oracle of the gfx942 GDN verify-row capture and its rollback (F32 state).
//!
//! The AR singleton advances the F32 recurrent state with `gated_delta_step_pipe128_f32` (the
//! production uncaptured route on gfx942). Native MTP verify windows run the CAPTURE kernel
//! `gated_delta_step_halves_state128_persistent256_capture_f32` (the final state lands in ring
//! slot `rows - 1`, the pre-window state stays intact) and a partial accept re-runs the kept
//! rows with `gated_delta_rollback_layers_f32` (the same persistent body over the retained
//! recurrence inputs). Both must be BYTE-identical to the pipe128 chain on the same rows, or a
//! lane's state would depend on whether its window was verified in one piece or replayed.
//!
//! Per case (rows 2 / 3 / 4, `--steps` consecutive windows, F32 Flash-Next shapes 16 key heads,
//! 48 value heads, 128 x 128):
//!   * capture: pipe128 over `rows` rows vs the capture kernel: the output rows, the final state
//!     (ring slot `rows - 1`) and the untouched pre-window state, word for word;
//!   * rollback: for every `keep` in `1..=rows`, pipe128 over the first `keep` rows vs the
//!     rollback kernel replaying `keep` of the captured `rows` rows from the pre-window slot, at
//!     a non-zero ring offset (`to`): the final slot word for word, and the other slots untouched;
//!   * the next window starts from the rolled-back state (carried), so an error compounds;
//!   * negative controls the comparator MUST detect: one flipped bit in the captured final state,
//!     a rollback replaying `keep - 1` rows compared against the `keep`-row reference, and the
//!     capture's source state overwritten.
//!
//! `cargo run --release -p rdna-compute --example gdn_capture_f32_gfx942_oracle --
//!    [--rows 2,3,4,8] [--lanes 1,2,8,33] [--steps 3] [--seed U64] [--skip-controls]`
//!
//! The batched capture recurrence of many lanes (`Gfx942Device::qwen4_gdn_lanes_step_capture_f32`,
//! kernel `qwen4_gdn_lanes_step_capture_f32_gfx942`) is compared word for word with the per-lane
//! capture launches it replaces, for every `--lanes` x `--rows` case: each lane's whole ring (only
//! slot `rows - 1` written), output rows and untouched live state, over carried windows, with
//! the lane-swap and unlaunched-lane controls.
//!
//! Prints one line per case and a final `gdn_capture_f32_gfx942_oracle: PASS|FAIL`.
//! Exit 0 = PASS, 1 = a mismatch / undetected control, 2 = not exact gfx942 (or
//! `HIPFIRE_QWEN4_GDN_PIPE=0`, which would change the reference kernel).

use rdna_compute::tensor_ops::{
    gated_delta_rollback_layers, gated_delta_step_batched, GatedDeltaRollbackLayers,
    GatedDeltaStepBatched, GdnStateFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const NK: usize = 16;
const NV: usize = 48;
const HD: usize = 128;
const QKV: usize = 2 * NK * HD + NV * HD;
const VALUE: usize = NV * HD;
const STATE: usize = NV * HD * HD;

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

fn step<'a>(
    projection: &'a GpuTensor,
    gate: &'a GpuTensor,
    beta: &'a GpuTensor,
    state: &'a GpuTensor,
    output: &'a GpuTensor,
    row_states: Option<&'a GpuTensor>,
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
        row_states,
        rows,
        qkv_width: QKV,
        key_heads: NK,
        value_heads: NV,
        key_dim: HD,
        value_dim: HD,
        position: 0,
    }
}

fn run_case(gpu: &mut Gpu, rows: usize, steps: usize, controls: bool, seed: u64) -> bool {
    let mut rng = Rng(seed ^ ((rows as u64) << 40));
    let mut ok = true;
    let mut report = |what: &str, good: bool, detail: String| {
        if !good {
            ok = false;
        }
        println!(
            "[rows={rows}] {what}: {detail} => {}",
            if good { "ok" } else { "MISMATCH" }
        );
    };
    // The recurrence input of the window, laid out as the capture ring retains it:
    // the convolution output rows [rows, QKV], then gate and beta [rows, NV] each.
    let recurrence_n = rows * (QKV + 2 * NV);
    let recurrence = gpu.zeros(&[recurrence_n], DType::F32).unwrap();
    let proj = recurrence.sub_offset(0, rows * QKV);
    let gate = recurrence.sub_offset(rows * QKV, rows * NV);
    let beta = recurrence.sub_offset(rows * QKV + rows * NV, rows * NV);
    let out_ref = gpu.zeros(&[rows * VALUE], DType::F32).unwrap();
    let out_cap = gpu.zeros(&[rows * VALUE], DType::F32).unwrap();
    let discard = gpu.zeros(&[rows * VALUE], DType::F32).unwrap();
    // The reference chain's state and the captured chain's live state (never written by a
    // capture): both start identical.
    let init = rand_vec(&mut rng, STATE, -0.5, 0.5);
    let ref_state = gpu.upload_f32(&init, &[STATE]).unwrap();
    let live = gpu.upload_f32(&init, &[STATE]).unwrap();
    // Ring: slot 0 is the pre-window state of a rollback; the capture writes slot `rows - 1` of
    // its own view; rollbacks land at slot `to + keep - 1` (to = 1).
    let ring_slots = rows + 2;
    let ring = gpu.zeros(&[ring_slots * STATE], DType::F32).unwrap();
    let cap_view = ring.sub_offset(0, rows * STATE);
    // The rollback table: [recurrence pointer, ring pointer] of the one layer.
    let table = gpu.zeros(&[4], DType::F32).unwrap();
    {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&(recurrence.buf.as_ptr() as u64).to_le_bytes());
        bytes.extend_from_slice(&(ring.buf.as_ptr() as u64).to_le_bytes());
        gpu.bind_thread().unwrap();
        gpu.hip.memcpy_htod(&table.buf, &bytes).unwrap();
    }
    let rollback = |gpu: &mut Gpu, keep: usize, from: usize, to: usize| {
        gated_delta_rollback_layers(
            gpu,
            &GatedDeltaRollbackLayers {
                table: &table,
                discard: &discard,
                format: GdnStateFormat::F32,
                layers: 1,
                rows,
                keep,
                from,
                to,
                qkv_width: QKV,
                key_heads: NK,
                value_heads: NV,
                position: 0,
            },
        )
        .expect("rollback launch");
    };

    for window in 0..steps {
        let p = rand_vec(&mut rng, rows * QKV, -1.0, 1.0);
        let g = rand_vec(&mut rng, rows * NV, -3.0, -0.01);
        let b = rand_vec(&mut rng, rows * NV, 0.02, 0.98);
        let mut all = p.clone();
        all.extend_from_slice(&g);
        all.extend_from_slice(&b);
        htod_f32(gpu, &recurrence, &all);
        let pre = dtoh(gpu, &live);
        // Reference: the production pipe128 chain over every row.
        let ref_before = dtoh(gpu, &ref_state);
        gated_delta_step_batched(
            gpu,
            &step(&proj, &gate, &beta, &ref_state, &out_ref, None, rows),
        )
        .expect("pipe128 launch");
        // Candidate: the capture kernel from the live state, the ring slot `rows - 1` takes the
        // final state; the live state is not written.
        gated_delta_step_batched(
            gpu,
            &step(&proj, &gate, &beta, &live, &out_cap, Some(&cap_view), rows),
        )
        .expect("capture launch");
        gpu.hip.device_synchronize().expect("sync");
        let r_state = dtoh(gpu, &ref_state);
        let ring_host = dtoh(gpu, &ring);
        let c_final = &ring_host[(rows - 1) * STATE..rows * STATE];
        let (state_diff, first) = word_diff(&r_state, c_final);
        let (out_diff, out_first) = word_diff(&dtoh(gpu, &out_ref), &dtoh(gpu, &out_cap));
        let live_diff = word_diff(&pre, &dtoh(gpu, &live)).0;
        let moved = word_diff(&ref_before, &r_state).0;
        report(
            &format!("window {window} capture vs pipe128"),
            state_diff == 0 && out_diff == 0 && live_diff == 0 && moved > 0,
            format!(
                "final-state diff words {state_diff} (first {first:?}), output diff words \
                 {out_diff} (first {out_first:?}), live state changed words {live_diff}, \
                 state moved words {moved}"
            ),
        );
        if controls && window == 0 {
            // A flipped bit in the captured state must be one detected word.
            let mut bad = c_final.to_vec();
            bad[STATE - 1] = f32::from_bits(bad[STATE - 1].to_bits() ^ 1);
            let (d, _) = word_diff(&r_state, &bad);
            report("control: flipped captured bit", d == 1, format!("diff words {d}"));
            // The capture compared against the wrong (stale) reference state.
            let d = word_diff(&ref_before, c_final).0;
            report("control: stale reference", d > 0, format!("diff words {d}"));
        }

        // Rollback replays: slot 0 holds the pre-window state; for each keep the rollback
        // writes slot `1 + keep - 1 = keep` and must equal the pipe128 chain over `keep` rows.
        htod_f32(gpu, &ring.sub_offset(0, STATE), &pre);
        let mut rolled_last = None;
        for keep in 1..=rows {
            let before_ring = dtoh(gpu, &ring);
            rollback(gpu, keep, 0, 1);
            gpu.hip.device_synchronize().expect("sync");
            let after_ring = dtoh(gpu, &ring);
            let k_state = gpu.upload_f32(&pre, &[STATE]).unwrap();
            let k_out = gpu.zeros(&[keep * VALUE], DType::F32).unwrap();
            let kp = recurrence.sub_offset(0, keep * QKV);
            // The kept rows' gate / beta are the first `keep` rows of the rows-wide blocks.
            let kg = recurrence.sub_offset(rows * QKV, keep * NV);
            let kb = recurrence.sub_offset(rows * QKV + rows * NV, keep * NV);
            gated_delta_step_batched(
                gpu,
                &step(&kp, &kg, &kb, &k_state, &k_out, None, keep),
            )
            .expect("pipe128 prefix launch");
            gpu.hip.device_synchronize().expect("sync");
            let want = dtoh(gpu, &k_state);
            let got = &after_ring[keep * STATE..(keep + 1) * STATE];
            let (d, f) = word_diff(&want, got);
            // Only slot `keep` may change (slot 0 is the source, the others are not its target).
            let others: usize = (0..ring_slots)
                .filter(|&s| s != keep)
                .map(|s| {
                    word_diff(
                        &before_ring[s * STATE..(s + 1) * STATE],
                        &after_ring[s * STATE..(s + 1) * STATE],
                    )
                    .0
                })
                .sum();
            report(
                &format!("window {window} rollback keep {keep} vs pipe128 prefix"),
                d == 0 && others == 0,
                format!("diff words {d} (first {f:?}), other slots changed words {others}"),
            );
            if controls && window == 0 && keep > 1 {
                // Replaying one row less is not the keep-row state: the comparator must see it.
                let short = gpu.upload_f32(&pre, &[STATE]).unwrap();
                let sk = keep - 1;
                let sp = recurrence.sub_offset(0, sk * QKV);
                let sg = recurrence.sub_offset(rows * QKV, sk * NV);
                let sb = recurrence.sub_offset(rows * QKV + rows * NV, sk * NV);
                let s_out = gpu.zeros(&[sk * VALUE], DType::F32).unwrap();
                gated_delta_step_batched(
                    gpu,
                    &step(&sp, &sg, &sb, &short, &s_out, None, sk),
                )
                .expect("short prefix launch");
                gpu.hip.device_synchronize().expect("sync");
                let d = word_diff(&dtoh(gpu, &short), got).0;
                report(
                    &format!("control: keep {keep} vs keep-1 reference"),
                    d > 0,
                    format!("diff words {d}"),
                );
                gpu.free_tensor(short).ok();
                gpu.free_tensor(s_out).ok();
            }
            if keep == rows {
                rolled_last = Some(got.to_vec());
            }
            gpu.free_tensor(k_state).ok();
            gpu.free_tensor(k_out).ok();
        }
        // keep == rows replays the whole window: its state equals the capture's final state.
        if let Some(full) = rolled_last {
            let d = word_diff(c_final, &full).0;
            report(
                &format!("window {window} full replay vs capture"),
                d == 0,
                format!("diff words {d}"),
            );
        }
        // The next window starts from the full-window state in BOTH chains (carried): the live
        // state takes the capture's final slot.
        htod_f32(gpu, &live, c_final);
    }
    for t in [
        recurrence, out_ref, out_cap, discard, ref_state, live, ring, table,
    ] {
        gpu.free_tensor(t).ok();
    }
    ok
}

const SENTINEL: u32 = 0xFFFF_FFFF;

/// The batched capture recurrence of `lanes` lanes (`qwen4_gdn_lanes_step_capture_f32_gfx942`)
/// against the per-lane capture launches it replaces (`gated_delta_step_batched` with
/// `row_states`, the persistent256 capture kernel the per-lane verify chain launches): every
/// lane's whole ring (the final slot written, every other slot still sentinel), its output rows
/// and its untouched live state, word for word, over `steps` carried windows.
fn run_lanes_case(
    gpu: &mut Gpu,
    lanes: usize,
    rows: usize,
    steps: usize,
    controls: bool,
    seed: u64,
) -> bool {
    use rdna_compute::cdna::gfx942::{Qwen4GdnLanesCaptureF32Lane, Qwen4GdnLanesF32Shape};
    let mut rng = Rng(seed ^ ((lanes as u64) << 20) ^ ((rows as u64) << 48) ^ 0x5A5A);
    let mut ok = true;
    let mut report = |what: &str, good: bool, detail: String| {
        if !good {
            ok = false;
        }
        println!(
            "[capture lanes={lanes} rows={rows}] {what}: {detail} => {}",
            if good { "ok" } else { "MISMATCH" }
        );
    };
    let rec_n = rows * (QKV + 2 * NV);
    let ring_n = rows * STATE;
    let sentinel = vec![f32::from_bits(SENTINEL); ring_n];
    let sentinel_out = vec![f32::from_bits(SENTINEL); rows * VALUE];
    let init: Vec<Vec<f32>> = (0..lanes)
        .map(|_| rand_vec(&mut rng, STATE, -0.5, 0.5))
        .collect();
    let rec: Vec<GpuTensor> = (0..lanes).map(|_| gpu.zeros(&[rec_n], DType::F32).unwrap()).collect();
    let live_ref: Vec<GpuTensor> =
        init.iter().map(|s| gpu.upload_f32(s, &[STATE]).unwrap()).collect();
    let live_cand: Vec<GpuTensor> =
        init.iter().map(|s| gpu.upload_f32(s, &[STATE]).unwrap()).collect();
    let ring_ref: Vec<GpuTensor> =
        (0..lanes).map(|_| gpu.upload_f32(&sentinel, &[ring_n]).unwrap()).collect();
    let ring_cand: Vec<GpuTensor> =
        (0..lanes).map(|_| gpu.upload_f32(&sentinel, &[ring_n]).unwrap()).collect();
    let out_ref: Vec<GpuTensor> = (0..lanes)
        .map(|_| gpu.upload_f32(&sentinel_out, &[rows * VALUE]).unwrap())
        .collect();
    let out_cand: Vec<GpuTensor> = (0..lanes)
        .map(|_| gpu.upload_f32(&sentinel_out, &[rows * VALUE]).unwrap())
        .collect();
    let proj: Vec<GpuTensor> = rec.iter().map(|r| r.sub_offset(0, rows * QKV)).collect();
    let gate: Vec<GpuTensor> =
        rec.iter().map(|r| r.sub_offset(rows * QKV, rows * NV)).collect();
    let beta: Vec<GpuTensor> = rec
        .iter()
        .map(|r| r.sub_offset(rows * QKV + rows * NV, rows * NV))
        .collect();
    let shape = Qwen4GdnLanesF32Shape {
        rows,
        key_heads: NK,
        value_heads: NV,
        key_dim: HD,
        value_dim: HD,
    };
    for window in 0..steps {
        for l in 0..lanes {
            let mut all = rand_vec(&mut rng, rows * QKV, -1.0, 1.0);
            all.extend(rand_vec(&mut rng, rows * NV, -3.0, -0.01));
            all.extend(rand_vec(&mut rng, rows * NV, 0.02, 0.98));
            htod_f32(gpu, &rec[l], &all);
            htod_f32(gpu, &ring_ref[l], &sentinel);
            htod_f32(gpu, &ring_cand[l], &sentinel);
            htod_f32(gpu, &out_ref[l], &sentinel_out);
            htod_f32(gpu, &out_cand[l], &sentinel_out);
        }
        let pre: Vec<Vec<f32>> = (0..lanes).map(|l| dtoh(gpu, &live_ref[l])).collect();
        // Reference: one production capture launch per lane.
        for l in 0..lanes {
            gated_delta_step_batched(
                gpu,
                &step(&proj[l], &gate[l], &beta[l], &live_ref[l], &out_ref[l], Some(&ring_ref[l]), rows),
            )
            .expect("per-lane capture launch");
        }
        // Candidate: one launch.
        let lane_args: Vec<Qwen4GdnLanesCaptureF32Lane<'_>> = (0..lanes)
            .map(|l| Qwen4GdnLanesCaptureF32Lane {
                projection: &proj[l],
                gate: &gate[l],
                beta: &beta[l],
                state: &live_cand[l],
                ring: &ring_cand[l],
                output: &out_cand[l],
            })
            .collect();
        gpu.try_gfx942()
            .expect("exact gfx942")
            .qwen4_gdn_lanes_step_capture_f32(&lane_args, shape)
            .expect("capture lanes launch");
        gpu.hip.device_synchronize().expect("sync");
        let (mut ring_diff, mut out_diff, mut live_diff, mut written, mut first_bad) =
            (0usize, 0usize, 0usize, 0usize, None);
        for l in 0..lanes {
            let r = dtoh(gpu, &ring_ref[l]);
            let c = dtoh(gpu, &ring_cand[l]);
            let (d, f) = word_diff(&r, &c);
            if d > 0 && first_bad.is_none() {
                first_bad = Some((l, f.unwrap()));
            }
            ring_diff += d;
            // Only slot `rows - 1` was written: every other word is still the sentinel.
            written += r.iter().filter(|w| w.to_bits() != SENTINEL).count();
            out_diff += word_diff(&dtoh(gpu, &out_ref[l]), &dtoh(gpu, &out_cand[l])).0;
            live_diff += word_diff(&pre[l], &dtoh(gpu, &live_cand[l])).0
                + word_diff(&pre[l], &dtoh(gpu, &live_ref[l])).0;
        }
        report(
            &format!("window {window} batched capture vs per-lane capture"),
            ring_diff == 0 && out_diff == 0 && live_diff == 0 && written == lanes * STATE,
            format!(
                "ring diff words {ring_diff} (first {first_bad:?}), output diff words {out_diff}, \
                 live state changed words {live_diff}, ring words written {written} (want {})",
                lanes * STATE
            ),
        );
        if controls && window == 0 {
            // A flipped bit in one lane's captured final state: exactly one differing word.
            let mut bad = dtoh(gpu, &ring_cand[lanes - 1]);
            let at = ring_n - 1;
            bad[at] = f32::from_bits(bad[at].to_bits() ^ 1);
            let d = word_diff(&dtoh(gpu, &ring_ref[lanes - 1]), &bad).0;
            report("control: flipped captured bit", d == 1, format!("diff words {d}"));
            if lanes >= 2 {
                // The candidate launched over the lanes in swapped ring order: the rings
                // must differ from the reference (the comparison sees lane identity).
                htod_f32(gpu, &ring_cand[0], &sentinel);
                htod_f32(gpu, &ring_cand[1], &sentinel);
                let mut swapped: Vec<Qwen4GdnLanesCaptureF32Lane<'_>> = (0..lanes)
                    .map(|l| Qwen4GdnLanesCaptureF32Lane {
                        projection: &proj[l],
                        gate: &gate[l],
                        beta: &beta[l],
                        state: &live_cand[l],
                        ring: &ring_cand[l],
                        output: &out_cand[l],
                    })
                    .collect();
                let (r0, r1) = (swapped[0].ring, swapped[1].ring);
                swapped[0].ring = r1;
                swapped[1].ring = r0;
                gpu.try_gfx942()
                    .expect("exact gfx942")
                    .qwen4_gdn_lanes_step_capture_f32(&swapped, shape)
                    .expect("swapped launch");
                gpu.hip.device_synchronize().expect("sync");
                let d = word_diff(&dtoh(gpu, &ring_ref[0]), &dtoh(gpu, &ring_cand[0])).0
                    + word_diff(&dtoh(gpu, &ring_ref[1]), &dtoh(gpu, &ring_cand[1])).0;
                report("control: lanes' rings swapped", d > 0, format!("diff words {d}"));
                // The candidate launched without its last lane leaves that ring unwritten.
                htod_f32(gpu, &ring_cand[lanes - 1], &sentinel);
                let short: Vec<Qwen4GdnLanesCaptureF32Lane<'_>> = (0..lanes - 1)
                    .map(|l| Qwen4GdnLanesCaptureF32Lane {
                        projection: &proj[l],
                        gate: &gate[l],
                        beta: &beta[l],
                        state: &live_cand[l],
                        ring: &ring_cand[l],
                        output: &out_cand[l],
                    })
                    .collect();
                gpu.try_gfx942()
                    .expect("exact gfx942")
                    .qwen4_gdn_lanes_step_capture_f32(&short, shape)
                    .expect("short launch");
                gpu.hip.device_synchronize().expect("sync");
                let d = word_diff(
                    &dtoh(gpu, &ring_ref[lanes - 1]),
                    &dtoh(gpu, &ring_cand[lanes - 1]),
                )
                .0;
                report("control: last lane not launched", d > 0, format!("diff words {d}"));
            }
        }
        // Carry: both chains' next window starts from this window's final state.
        for l in 0..lanes {
            let r = dtoh(gpu, &ring_ref[l]);
            let final_state = &r[(rows - 1) * STATE..rows * STATE];
            htod_f32(gpu, &live_ref[l], final_state);
            htod_f32(gpu, &live_cand[l], final_state);
        }
    }
    for t in rec
        .into_iter()
        .chain(live_ref)
        .chain(live_cand)
        .chain(ring_ref)
        .chain(ring_cand)
        .chain(out_ref)
        .chain(out_cand)
    {
        gpu.free_tensor(t).ok();
    }
    ok
}

/// Which arm the verify-module entries run on: the `HIPFIRE_QWEN4_G942_GDN_VERIFY_IMAGE` code
/// object when the hook is set, else the hipcc source of module `qwen4_g942_gdn_verify`.
fn verify_arm() -> String {
    match std::env::var("HIPFIRE_QWEN4_G942_GDN_VERIFY_IMAGE") {
        Ok(path) if !path.is_empty() => format!("image {path}"),
        _ => "source (hipcc, module qwen4_g942_gdn_verify)".to_string(),
    }
}

fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16
}

fn rand_bf16(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<u16> {
    (0..n).map(|_| bf16_bits(rng.range(lo, hi))).collect()
}

fn htod_u16(gpu: &Gpu, tensor: &GpuTensor, data: &[u16]) {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    gpu.bind_thread().expect("bind");
    gpu.hip.memcpy_htod(&tensor.buf, bytes).expect("htod");
}

/// The verify module's convolution + gate-parameter kernel (`qwen4_g942_gdn_verify_conv_params_k4`,
/// through `gated_delta_conv_params_batched`, on the arm [`verify_arm`] names) against the
/// `tensor_ops` JIT kernel it copies (`gated_delta_conv_bf16_f32_batched_k4`, launched directly):
/// the convolution output rows, the next history ring, and the gate / beta parameter rows, word
/// for word, once with separate history / next-history buffers and once with them aliased (the
/// production call). Controls: a flipped output bit and the candidate at the next ring cursor.
fn run_conv_case(gpu: &mut Gpu, rows: usize, cursor: usize, controls: bool, seed: u64) -> bool {
    use hip_bridge::KernargBlob;
    use rdna_compute::tensor_ops::{
        gated_delta_conv_params_batched, lab_ensure_tensor_ops_kernel, GatedDeltaConvBatched,
        GatedDeltaParamsBatched,
    };
    const KERNEL: &str = "gated_delta_conv_bf16_f32_batched_k4";
    let arm = verify_arm();
    let mut rng = Rng(seed ^ ((rows as u64) << 36) ^ ((cursor as u64) << 52) ^ 0xC0DE);
    let mut ok = true;
    let mut report = |what: &str, good: bool, detail: String| {
        if !good {
            ok = false;
        }
        println!(
            "[conv_params_k4 rows={rows} cursor={cursor}] [arm: {arm}] {what}: {detail} => {}",
            if good { "ok" } else { "MISMATCH" }
        );
    };
    let channels = QKV;
    let heads = NV;
    let input = rand_vec(&mut rng, rows * channels, -2.0, 2.0);
    let weight = rand_bf16(&mut rng, channels * 4, -1.0, 1.0);
    let history = rand_vec(&mut rng, 3 * channels, -2.0, 2.0);
    let a = rand_vec(&mut rng, rows * heads, -3.0, 3.0);
    let b = rand_vec(&mut rng, rows * heads, -3.0, 3.0);
    let a_log = rand_bf16(&mut rng, heads, -1.0, 1.0);
    let dt_bias = rand_bf16(&mut rng, heads, -1.0, 1.0);
    let sentinel = f32::from_bits(0xFFFF_FFFF);
    let t_input = gpu.upload_f32(&input, &[rows * channels]).unwrap();
    let t_weight = gpu.zeros(&[channels * 4], DType::BF16).unwrap();
    htod_u16(gpu, &t_weight, &weight);
    let t_a = gpu.upload_f32(&a, &[rows * heads]).unwrap();
    let t_b = gpu.upload_f32(&b, &[rows * heads]).unwrap();
    let t_a_log = gpu.zeros(&[heads], DType::BF16).unwrap();
    htod_u16(gpu, &t_a_log, &a_log);
    let t_dt = gpu.zeros(&[heads], DType::BF16).unwrap();
    htod_u16(gpu, &t_dt, &dt_bias);
    lab_ensure_tensor_ops_kernel(gpu, KERNEL).expect("tensor_ops kernel");
    let conv_blocks = channels.div_ceil(256);
    let grid = conv_blocks * rows.div_ceil(16) + (rows * heads).div_ceil(256);
    // One arm pair per history layout: (ref, cand) with `alias` = history IS next_history.
    let mut candidate_out: Option<Vec<f32>> = None;
    let mut reference_out: Option<Vec<f32>> = None;
    for alias in [false, true] {
        let mk = |gpu: &mut Gpu| {
            let hist = gpu.upload_f32(&history, &[3 * channels]).unwrap();
            let next = if alias {
                None
            } else {
                Some(gpu.upload_f32(&vec![sentinel; 3 * channels], &[3 * channels]).unwrap())
            };
            let out = gpu.upload_f32(&vec![sentinel; rows * channels], &[rows * channels]).unwrap();
            let gate = gpu.upload_f32(&vec![sentinel; rows * heads], &[rows * heads]).unwrap();
            let beta = gpu.upload_f32(&vec![sentinel; rows * heads], &[rows * heads]).unwrap();
            (hist, next, out, gate, beta)
        };
        let (r_hist, r_next, r_out, r_gate, r_beta) = mk(gpu);
        let (c_hist, c_next, c_out, c_gate, c_beta) = mk(gpu);
        // Reference: the tensor_ops kernel, launched directly.
        let r_next_t = r_next.as_ref().unwrap_or(&r_hist);
        let mut args = KernargBlob::new();
        for t in [&t_input, &t_weight, &r_hist, &r_out, r_next_t] {
            args.push_ptr(t.buf.as_ptr());
        }
        for v in [rows as i32, channels as i32, 3, 4, cursor as i32, 0, 0] {
            args.push_i32(v);
        }
        for t in [&t_a, &t_b, &t_a_log, &t_dt, &r_gate, &r_beta] {
            args.push_ptr(t.buf.as_ptr());
        }
        args.push_i32((rows * heads) as i32);
        args.push_i32(heads as i32);
        args.pad_to(16);
        gpu.launch_kernel_blob(KERNEL, [grid as u32, 1, 1], [256, 1, 1], 0, args.as_mut_slice())
            .expect("reference launch");
        // Candidate: the public launcher (verify module on gfx942).
        let c_next_t = c_next.as_ref().unwrap_or(&c_hist);
        let run_cand = |gpu: &mut Gpu, start_cursor: usize| {
            gated_delta_conv_params_batched(
                gpu,
                &GatedDeltaConvBatched {
                    input: &t_input,
                    kernel: &t_weight,
                    history: &c_hist,
                    output: &c_out,
                    next_history: c_next_t,
                    rows,
                    channels,
                    history_rows: 3,
                    kernel_size: 4,
                    start_cursor,
                },
                &GatedDeltaParamsBatched {
                    a: &t_a,
                    b: &t_b,
                    a_log: &t_a_log,
                    dt_bias: &t_dt,
                    gate: &c_gate,
                    beta: &c_beta,
                    rows,
                    heads,
                },
            )
            .expect("candidate launch");
        };
        run_cand(gpu, cursor);
        gpu.hip.device_synchronize().expect("sync");
        let diff = |gpu: &Gpu, r: &GpuTensor, c: &GpuTensor| word_diff(&dtoh(gpu, r), &dtoh(gpu, c));
        let (d_out, f_out) = diff(gpu, &r_out, &c_out);
        let (d_next, f_next) = diff(gpu, r_next_t, c_next_t);
        let (d_gate, _) = diff(gpu, &r_gate, &c_gate);
        let (d_beta, _) = diff(gpu, &r_beta, &c_beta);
        let written = dtoh(gpu, &r_out)
            .iter()
            .filter(|w| w.to_bits() != 0xFFFF_FFFF)
            .count();
        report(
            &format!("{} history vs tensor_ops kernel", if alias { "aliased" } else { "separate" }),
            d_out == 0 && d_next == 0 && d_gate == 0 && d_beta == 0 && written == rows * channels,
            format!(
                "output diff words {d_out} (first {f_out:?}), next-history diff words {d_next} \
                 (first {f_next:?}), gate diff {d_gate}, beta diff {d_beta}, output words written \
                 {written}/{}",
                rows * channels
            ),
        );
        if controls && !alias {
            let mut bad = dtoh(gpu, &c_out);
            bad[rows * channels - 1] = f32::from_bits(bad[rows * channels - 1].to_bits() ^ 1);
            let d = word_diff(&dtoh(gpu, &r_out), &bad).0;
            report("control: flipped output bit", d == 1, format!("diff words {d}"));
            // The candidate at the next ring cursor: its history reads / tail ring differ.
            let (w_hist, w_next, w_out, w_gate, w_beta) = mk(gpu);
            let w_next_t = w_next.as_ref().unwrap_or(&w_hist);
            gated_delta_conv_params_batched(
                gpu,
                &GatedDeltaConvBatched {
                    input: &t_input,
                    kernel: &t_weight,
                    history: &w_hist,
                    output: &w_out,
                    next_history: w_next_t,
                    rows,
                    channels,
                    history_rows: 3,
                    kernel_size: 4,
                    start_cursor: (cursor + 1) % 3,
                },
                &GatedDeltaParamsBatched {
                    a: &t_a,
                    b: &t_b,
                    a_log: &t_a_log,
                    dt_bias: &t_dt,
                    gate: &w_gate,
                    beta: &w_beta,
                    rows,
                    heads,
                },
            )
            .expect("wrong-cursor launch");
            gpu.hip.device_synchronize().expect("sync");
            let d = word_diff(&dtoh(gpu, &r_next_t.sub_offset(0, 3 * channels)), &dtoh(gpu, w_next_t)).0
                + word_diff(&dtoh(gpu, &r_out), &dtoh(gpu, &w_out)).0;
            report("control: candidate at the wrong ring cursor", d > 0, format!("diff words {d}"));
            for t in [w_hist, w_out, w_gate, w_beta].into_iter().chain(w_next) {
                gpu.free_tensor(t).ok();
            }
        }
        candidate_out.get_or_insert_with(|| dtoh(gpu, &c_out));
        reference_out.get_or_insert_with(|| dtoh(gpu, &r_out));
        for t in [r_hist, r_out, r_gate, r_beta, c_hist, c_out, c_gate, c_beta]
            .into_iter()
            .chain(r_next)
            .chain(c_next)
        {
            gpu.free_tensor(t).ok();
        }
    }
    for t in [t_input, t_weight, t_a, t_b, t_a_log, t_dt] {
        gpu.free_tensor(t).ok();
    }
    ok
}

/// The verify module's gated norm (`qwen4_g942_gdn_verify_gate_batched`, through
/// `gated_delta_gate_batched`, on the arm [`verify_arm`] names) against the `tensor_ops` JIT kernel
/// it copies (`gated_delta_gate_bf16_f32_batched`, launched directly): the gated output and the
/// in-place BF16-rounded recurrent rows, word for word; controls as the convolution cases.
fn run_gate_case(gpu: &mut Gpu, rows: usize, controls: bool, seed: u64) -> bool {
    use hip_bridge::KernargBlob;
    use rdna_compute::tensor_ops::{
        gated_delta_gate_batched, lab_ensure_tensor_ops_kernel, GatedDeltaGateBatched,
    };
    const KERNEL: &str = "gated_delta_gate_bf16_f32_batched";
    let arm = verify_arm();
    let mut rng = Rng(seed ^ ((rows as u64) << 36) ^ 0x6A7E);
    let mut ok = true;
    let mut report = |what: &str, good: bool, detail: String| {
        if !good {
            ok = false;
        }
        println!(
            "[gate_batched rows={rows}] [arm: {arm}] {what}: {detail} => {}",
            if good { "ok" } else { "MISMATCH" }
        );
    };
    let n = rows * VALUE;
    let recurrent = rand_vec(&mut rng, n, -3.0, 3.0);
    let z = rand_vec(&mut rng, n, -3.0, 3.0);
    let norm = rand_bf16(&mut rng, 128, 0.2, 2.0);
    let sentinel = vec![f32::from_bits(0xFFFF_FFFF); n];
    let t_z = gpu.upload_f32(&z, &[n]).unwrap();
    let t_norm = gpu.zeros(&[128], DType::BF16).unwrap();
    htod_u16(gpu, &t_norm, &norm);
    let r_rec = gpu.upload_f32(&recurrent, &[n]).unwrap();
    let c_rec = gpu.upload_f32(&recurrent, &[n]).unwrap();
    let r_out = gpu.upload_f32(&sentinel, &[n]).unwrap();
    let c_out = gpu.upload_f32(&sentinel, &[n]).unwrap();
    lab_ensure_tensor_ops_kernel(gpu, KERNEL).expect("tensor_ops kernel");
    let mut args = KernargBlob::new();
    for t in [&r_rec, &t_z, &t_norm, &r_out] {
        args.push_ptr(t.buf.as_ptr());
    }
    for v in [rows as i32, NV as i32, 128] {
        args.push_i32(v);
    }
    args.pad_to(16);
    gpu.launch_kernel_blob(KERNEL, [(rows * NV) as u32, 1, 1], [32, 1, 1], 0, args.as_mut_slice())
        .expect("reference launch");
    gated_delta_gate_batched(
        gpu,
        &GatedDeltaGateBatched {
            recurrent_output: &c_rec,
            z: &t_z,
            norm: &t_norm,
            output: &c_out,
            rows,
            value_heads: NV,
            value_dim: 128,
        },
    )
    .expect("candidate launch");
    gpu.hip.device_synchronize().expect("sync");
    let (d_out, f_out) = word_diff(&dtoh(gpu, &r_out), &dtoh(gpu, &c_out));
    let (d_rec, f_rec) = word_diff(&dtoh(gpu, &r_rec), &dtoh(gpu, &c_rec));
    let moved = word_diff(&recurrent, &dtoh(gpu, &r_rec)).0;
    let written = dtoh(gpu, &r_out).iter().filter(|w| w.to_bits() != 0xFFFF_FFFF).count();
    report(
        "gated norm vs tensor_ops kernel",
        d_out == 0 && d_rec == 0 && written == n && moved > 0,
        format!(
            "output diff words {d_out} (first {f_out:?}), recurrent diff words {d_rec} (first \
             {f_rec:?}), output words written {written}/{n}, recurrent rows rounded {moved}"
        ),
    );
    if controls {
        let mut bad = dtoh(gpu, &c_out);
        bad[n - 1] = f32::from_bits(bad[n - 1].to_bits() ^ 1);
        let d = word_diff(&dtoh(gpu, &r_out), &bad).0;
        report("control: flipped output bit", d == 1, format!("diff words {d}"));
        // The candidate with a different z row: its output must differ from the reference.
        let t_z2 = gpu.upload_f32(&rand_vec(&mut rng, n, -3.0, 3.0), &[n]).unwrap();
        let w_rec = gpu.upload_f32(&recurrent, &[n]).unwrap();
        let w_out = gpu.upload_f32(&sentinel, &[n]).unwrap();
        gated_delta_gate_batched(
            gpu,
            &GatedDeltaGateBatched {
                recurrent_output: &w_rec,
                z: &t_z2,
                norm: &t_norm,
                output: &w_out,
                rows,
                value_heads: NV,
                value_dim: 128,
            },
        )
        .expect("wrong-z launch");
        gpu.hip.device_synchronize().expect("sync");
        let d = word_diff(&dtoh(gpu, &r_out), &dtoh(gpu, &w_out)).0;
        report("control: candidate on another z", d > 0, format!("diff words {d}"));
        for t in [t_z2, w_rec, w_out] {
            gpu.free_tensor(t).ok();
        }
    }
    for t in [t_z, t_norm, r_rec, c_rec, r_out, c_out] {
        gpu.free_tensor(t).ok();
    }
    ok
}

fn main() {
    let mut rows_list = vec![2usize, 3, 4, 8];
    let mut lanes_list = vec![1usize, 2, 8, 33];
    let mut steps = 3usize;
    let mut seed = 0x6D69_3330_3078_6370u64;
    let mut controls = true;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--lanes" => {
                lanes_list = value()
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse().expect("integer list"))
                    .collect()
            }
            "--rows" => {
                rows_list = value()
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse().expect("integer list"))
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
            "gdn_capture_f32_gfx942_oracle: needs exact gfx942, this device is {}",
            gpu.arch
        );
        std::process::exit(2);
    }
    if std::env::var("HIPFIRE_QWEN4_GDN_PIPE").as_deref() == Ok("0") {
        eprintln!(
            "gdn_capture_f32_gfx942_oracle: HIPFIRE_QWEN4_GDN_PIPE=0 selects persistent256 as \
             the reference; unset it"
        );
        std::process::exit(2);
    }
    println!(
        "device {} | GDN shapes: key heads {NK}, value heads {NV}, {HD}x{HD}, qkv width {QKV} | \
         steps {steps}",
        gpu.arch
    );
    // The candidate routes: every launched entry is the hipcc source unless a twin image hook
    // names a code object (`HIPFIRE_QWEN4_G942_GDN_VERIFY_IMAGE`: the capture / rollback /
    // convolution + parameters / gated-norm entries of the verify chain;
    // `HIPFIRE_QWEN4_G942_GDN_LANES_CAPTURE_IMAGE`: the batched capture lanes kernel). A set hook
    // that does not load fails the first launch (non-zero exit); the route line and the final
    // status say which objects the PASS covers. The pipe128 REFERENCE is the tensor_ops source
    // route in every case.
    let mut images: Vec<String> = Vec::new();
    for hook in [
        "HIPFIRE_QWEN4_G942_GDN_VERIFY_IMAGE",
        "HIPFIRE_QWEN4_G942_GDN_LANES_CAPTURE_IMAGE",
    ] {
        match std::env::var(hook) {
            Ok(path) if !path.is_empty() => {
                let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                images.push(format!("{hook}={path} ({bytes} bytes)"));
            }
            _ => images.push(format!("{hook} unset (hipcc source)")),
        }
    }
    println!("candidate route: {}", images.join("; "));
    let conv_rows = [1usize, 2, 3, 4, 8, 16, 17];
    let gate_rows = [1usize, 2, 4, 8];
    let mut all = true;
    for &rows in &rows_list {
        all &= run_case(&mut gpu, rows, steps, controls, seed);
    }
    for &rows in &rows_list {
        for &lanes in &lanes_list {
            all &= run_lanes_case(&mut gpu, lanes, rows, steps, controls, seed);
        }
    }
    for &rows in &conv_rows {
        for cursor in 0..3 {
            all &= run_conv_case(&mut gpu, rows, cursor, controls, seed);
        }
    }
    for &rows in &gate_rows {
        all &= run_gate_case(&mut gpu, rows, controls, seed);
    }
    println!(
        "gdn_capture_f32_gfx942_oracle: {} [route: {}]",
        if all { "PASS" } else { "FAIL" },
        images.join("; ")
    );
    std::process::exit(if all { 0 } else { 1 });
}
