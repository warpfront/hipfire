// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! SPEC-AR-FLOOR S2a gate 3: a qwen35 DFlash request that the driver-owned
//! floor moves between speculative windows, AR tokens and forced advances
//! must decode exactly what pure AR decodes, and leave the same state.
//!
//! Each schedule drives the real `DflashSpeculator::{step, ar_step,
//! on_forced_advance}` through a scripted interleaving and compares, against
//! pure AR (`ar_step` only, the same forced advances at the same positions):
//!
//! - the token ids at every position;
//! - at every schedule checkpoint: the target KV rows `[0, pos)` (keys,
//!   values, scales), the DeltaNet S / scale / conv / error-feedback state,
//!   the hidden ring's valid rows plus head/written, the interleaved
//!   target-hidden rows of the live positions, and the `thlog` frontier.
//!
//! A sabotage arm (one AR token fed a wrong seed) must be detected.
//! Schedules: spec ×3 → AR ×4 → spec ×1 → AR ×20 → spec ×6 → forced advance
//! → spec ×2 → AR ×2, and the reverse (forced advance first).
//!
//! `#[ignore]`d: needs a real HIP GPU and the models under `~/.hipfire/models`
//! (skipped when missing). One load per process:
//!
//! ```text
//! cargo test --release -p hipfire-generate --test spec_floor_schedule_hw -- --ignored --exact qwen35_dflash_floor_schedules_equal_pure_ar
//! ```
//!
//! - `SPEC_FLOOR_HW_TARGET` / `SPEC_FLOOR_HW_DRAFT`: model file names under
//!   `~/.hipfire/models` or absolute paths (default `qwen3.8-27b.mq4-xts` +
//!   `qwen3.8-27b-dflash.mq4v2-sym.hfq`; the MQ3-XT cell uses
//!   `qwen3.8-27b.mq3-xt`).
//! - `SPEC_FLOOR_HW_ROUTE=pm4|hip` (default `pm4`): the AR executor the load
//!   gets (the retained PM4 default, as the daemon enables it on a DFlash
//!   load where admitted, or the HIP AR graph).

use hipfire_arch_qwen35::dflash_spec::DflashSpeculator;
use hipfire_arch_qwen35::speculative::ModelSlot;
use hipfire_runtime::loader_api::SpecLoadCfg;
use hipfire_runtime::spec::{PrefillOutcome, SpecTarget, Speculator};
use rdna_compute::GpuTensor;
use std::path::PathBuf;

const PROMPT_LEN: usize = 300;
/// A think-close and the newline after it: the forced tokens of a think-cap
/// close (as in `spec_forced_advance_seal_hw`).
const FORCED: [u32; 2] = [151668, 271];

#[derive(Clone, Copy, Debug)]
enum Op {
    Spec(usize),
    Ar(usize),
    Forced,
}

fn model(env: &str, default: &str) -> Option<String> {
    let name = std::env::var(env).unwrap_or_else(|_| default.to_owned());
    let path = if name.starts_with('/') {
        PathBuf::from(&name)
    } else {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".hipfire/models")
            .join(&name)
    };
    if !path.is_file() {
        eprintln!("skip: {} missing", path.display());
        return None;
    }
    Some(path.to_string_lossy().into_owned())
}

fn prompt() -> Vec<u32> {
    let mut state = 0x5ea1u64;
    (0..PROMPT_LEN)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (1000 + (state >> 33) % 99_000) as u32
        })
        .collect()
}

fn h64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn download(gpu: &rdna_compute::Gpu, t: &GpuTensor) -> Vec<u8> {
    let mut bytes = vec![0u8; t.buf.size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).expect("dtoh");
    bytes
}

/// `rows` rows of `row_bytes` from a row-major buffer, row `r` at `index(r)`.
fn rows_digest(
    bytes: &[u8],
    row_bytes: usize,
    rows: impl Iterator<Item = usize>,
    index: impl Fn(usize) -> usize,
) -> u64 {
    let mut h = 0u64;
    for r in rows {
        let at = index(r) * row_bytes;
        h = h.rotate_left(7) ^ h64(&bytes[at..at + row_bytes]);
    }
    h
}

/// One line per state surface at position `pos`.
fn digest(
    gpu: &rdna_compute::Gpu,
    slot: &ModelSlot,
    spec: &DflashSpeculator,
    pos: usize,
) -> Vec<String> {
    gpu.hip.device_synchronize().expect("sync");
    let mut out = Vec::new();
    let kv = &slot.kv_cache;
    let cap = kv.physical_cap.max(1);
    for (name, tensors) in [
        ("kv.k", &kv.k_gpu),
        ("kv.v", &kv.v_gpu),
        ("kv.k_scales", &kv.k_scales),
        ("kv.v_scales", &kv.v_scales),
    ] {
        for (layer, t) in tensors.iter().enumerate() {
            let size = t.buf.size();
            if size == 0 {
                continue;
            }
            let bytes = download(gpu, t);
            // Position-major rows: only the committed prefix is state (a
            // verify writes rejected-draft rows past it).
            let h = if size % cap == 0 {
                rows_digest(&bytes, size / cap, 0..pos.min(cap), |r| r)
            } else {
                h64(&bytes)
            };
            out.push(format!("{name}[{layer}] {h:016x}"));
        }
    }
    let dn = &slot.dn_state;
    for (name, tensors) in [
        ("dn.s", &dn.s_matrices),
        ("dn.scale", &dn.s_scales),
        ("dn.conv", &dn.conv_states),
        ("dn.ef", &dn.s_ef_residual),
    ] {
        for (layer, t) in tensors.iter().enumerate() {
            if t.buf.size() > 0 {
                out.push(format!("{name}[{layer}] {:016x}", h64(&download(gpu, t))));
            }
        }
    }
    let df = spec.lane_state().expect("DFlash lane state");
    let rb = &df.hidden_rb;
    let live = rb.written.min(rb.max_positions);
    let row = rb.hidden_dim * 4;
    for (layer, t) in rb.layer_bufs.iter().enumerate() {
        let bytes = download(gpu, t);
        let h = rows_digest(&bytes, row, 0..live, |r| {
            (rb.head + rb.max_positions - live + r) % rb.max_positions
        });
        out.push(format!("ring[{layer}] {h:016x}"));
    }
    out.push(format!("ring.cursor {}/{}", rb.head, rb.written));
    let feature = rb.hidden_dim * rb.extract_layers.len() * 4;
    let th = &df.draft_scratch.target_hidden;
    if feature > 0 && th.buf.size() % feature == 0 {
        let modulus = th.buf.size() / feature;
        let bytes = download(gpu, th);
        let from = pos.saturating_sub(modulus);
        let h = rows_digest(&bytes, feature, from..pos, |p| p % modulus);
        out.push(format!("interleaved {h:016x}"));
    }
    out.push(format!("thlog {:?}", df.draft_scratch.thlog.mark()));
    out
}

struct Arm {
    /// Token at each decode position from the prompt end (the prefill's first
    /// token first).
    tokens: Vec<u32>,
    /// (position, digest) at each checkpoint.
    digests: Vec<(usize, Vec<String>)>,
    forced_at: Vec<usize>,
}

struct Rig<'a> {
    gpu: &'a mut rdna_compute::Gpu,
    spec: &'a mut Box<dyn Speculator>,
    target: &'a mut dyn SpecTarget,
    prompt: Vec<u32>,
}

impl Rig<'_> {
    fn dflash_digest(&mut self, pos: usize) -> Vec<String> {
        let slot = self
            .target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .expect("qwen35 ModelSlot");
        let spec = self
            .spec
            .drafter_any_mut()
            .and_then(|any| any.downcast_ref::<DflashSpeculator>())
            .expect("DflashSpeculator");
        digest(self.gpu, slot, spec, pos)
    }

    fn prefill(&mut self) -> u32 {
        let toks = self.prompt.clone();
        match self
            .spec
            .prefill(self.gpu, self.target, &toks, &toks, 0, false, None, &|| {
                false
            }) {
            Ok(PrefillOutcome::Ready { first_token }) => first_token,
            other => panic!("prefill: {other:?}"),
        }
    }

    fn ar(&mut self, pos: usize, seed: u32, emitted: &[u32]) -> u32 {
        self.spec
            .ar_step(self.gpu, self.target, pos, seed, emitted)
            .unwrap_or_else(|e| panic!("ar_step at {pos}: {e}"))
    }

    fn forced(&mut self, pos: usize, seed: u32) {
        let handled = self
            .spec
            .on_forced_advance(self.gpu, self.target, &[seed, FORCED[0]], pos, &|| false)
            .unwrap_or_else(|e| panic!("forced advance at {pos}: {e}"));
        assert!(
            handled,
            "the DFlash speculator advances forced tokens itself"
        );
    }

    /// The scripted arm; `sabotage` feeds AR token number `n` a wrong seed.
    fn schedule(&mut self, ops: &[Op], sabotage: Option<usize>) -> Arm {
        let first = self.prefill();
        let mut pos = self.prompt.len();
        let mut seed = first;
        let mut emitted = vec![first];
        let mut arm = Arm {
            tokens: vec![first],
            digests: Vec::new(),
            forced_at: Vec::new(),
        };
        let mut ar_count = 0;
        for &op in ops {
            match op {
                Op::Spec(windows) => {
                    for _ in 0..windows {
                        let step = self
                            .spec
                            .step(
                                self.gpu,
                                self.target,
                                pos,
                                seed,
                                &emitted,
                                None,
                                0.0,
                                usize::MAX,
                            )
                            .unwrap_or_else(|e| panic!("step at {pos}: {e}"));
                        assert!(!step.emit.is_empty(), "verify emitted nothing at {pos}");
                        pos += step.emit.len();
                        emitted.extend_from_slice(&step.emit);
                        arm.tokens.extend_from_slice(&step.emit);
                        seed = step.next_seed;
                    }
                }
                Op::Ar(tokens) => {
                    for _ in 0..tokens {
                        ar_count += 1;
                        let fed = if sabotage == Some(ar_count) {
                            seed ^ 1
                        } else {
                            seed
                        };
                        let next = self.ar(pos, fed, &emitted);
                        pos += 1;
                        emitted.push(next);
                        arm.tokens.push(next);
                        seed = next;
                    }
                }
                Op::Forced => {
                    arm.forced_at.push(pos);
                    self.forced(pos, seed);
                    pos += 2;
                    emitted.extend_from_slice(&FORCED);
                    arm.tokens.extend_from_slice(&FORCED);
                    seed = FORCED[1];
                }
            }
            let d = self.dflash_digest(pos);
            arm.digests.push((pos, d));
        }
        arm
    }

    /// Pure AR over the same positions, with the schedule's forced advances.
    fn reference(&mut self, of: &Arm) -> Arm {
        let first = self.prefill();
        let mut pos = self.prompt.len();
        let end = pos + of.tokens.len() - 1;
        let mut seed = first;
        let mut emitted = vec![first];
        let mut arm = Arm {
            tokens: vec![first],
            digests: Vec::new(),
            forced_at: Vec::new(),
        };
        while pos < end {
            if of.forced_at.contains(&pos) {
                arm.forced_at.push(pos);
                self.forced(pos, seed);
                pos += 2;
                emitted.extend_from_slice(&FORCED);
                arm.tokens.extend_from_slice(&FORCED);
                seed = FORCED[1];
            } else {
                let next = self.ar(pos, seed, &emitted);
                pos += 1;
                emitted.push(next);
                arm.tokens.push(next);
                seed = next;
            }
            if of.digests.iter().any(|(at, _)| *at == pos) {
                let d = self.dflash_digest(pos);
                arm.digests.push((pos, d));
            }
        }
        arm
    }
}

/// Mismatching tokens plus mismatching digest lines of `arm` against `reference`.
fn mismatches(label: &str, arm: &Arm, reference: &Arm) -> usize {
    let mut count = 0;
    if arm.tokens.len() != reference.tokens.len() {
        eprintln!(
            "{label}: token count {} vs pure AR {}",
            arm.tokens.len(),
            reference.tokens.len()
        );
        count += 1;
    }
    for (i, (a, b)) in arm.tokens.iter().zip(&reference.tokens).enumerate() {
        if a != b {
            eprintln!("{label}: token {i}: {a} vs pure AR {b}");
            count += 1;
        }
    }
    for (pos, lines) in &arm.digests {
        let Some((_, want)) = reference.digests.iter().find(|(at, _)| at == pos) else {
            eprintln!("{label}: no pure-AR checkpoint at {pos}");
            count += 1;
            continue;
        };
        for (a, b) in lines.iter().zip(want) {
            if a != b {
                eprintln!("{label}: pos {pos}: {a} vs pure AR {b}");
                count += 1;
            }
        }
    }
    eprintln!(
        "{label}: {} tokens, {} checkpoints, {count} mismatches",
        arm.tokens.len(),
        arm.digests.len()
    );
    count
}

#[test]
#[ignore = "needs a HIP GPU and a qwen3.8-27b target with its DFlash draft (SPEC_FLOOR_HW_TARGET/DRAFT)"]
fn qwen35_dflash_floor_schedules_equal_pure_ar() {
    let (Some(path), Some(draft)) = (
        model("SPEC_FLOOR_HW_TARGET", "qwen3.8-27b.mq4-xts"),
        model("SPEC_FLOOR_HW_DRAFT", "qwen3.8-27b-dflash.mq4v2-sym.hfq"),
    ) else {
        return;
    };
    let pm4 = std::env::var("SPEC_FLOOR_HW_ROUTE").map_or(true, |route| route != "hip");
    let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
    let spec_cfg = SpecLoadCfg {
        dflash: Some(true),
        mtp: Some(false),
        dspark: Some(false),
        ..SpecLoadCfg::default()
    };
    let mut m = hipfire_loader::load_model(
        &path,
        PROMPT_LEN + 1024,
        Some(&draft),
        None,
        None,
        None,
        1,
        spec_cfg,
        &mut gpu,
    )
    .expect("load");
    let enabled = gpu.replay.configure_model_default(pm4);
    if pm4 && gpu.arch == "gfx1201" {
        gpu.replay
            .set_pm4_gfx12_dispatch_pacing(rdna_compute::replay::gfx1201_pm4_pacing_from_config());
    }
    eprintln!(
        "spec_floor_schedule_hw: arch={} route={} retained={enabled} transport={}",
        gpu.arch,
        if pm4 { "pm4" } else { "hip" },
        gpu.replay.transport_name()
    );
    let spec = m.speculator.as_mut().expect("speculator loaded");
    assert_eq!(spec.name(), "dflash", "speculator route");
    assert!(spec.supports_ar_step(), "DFlash AR step");
    let carrier = hipfire_loader::carrier_for(m.arch_id).expect("carrier");
    let mut guard = carrier
        .spec_target_guard(&mut m.state)
        .expect("spec target guard");
    let target = guard.slot().expect("spec target");
    let mut rig = Rig {
        gpu: &mut gpu,
        spec,
        target,
        prompt: prompt(),
    };

    let forward = [
        Op::Spec(3),
        Op::Ar(4),
        Op::Spec(1),
        Op::Ar(20),
        Op::Spec(6),
        Op::Forced,
        Op::Spec(2),
        Op::Ar(2),
    ];
    let reverse = [
        Op::Forced,
        Op::Ar(2),
        Op::Spec(2),
        Op::Ar(20),
        Op::Spec(1),
        Op::Ar(4),
        Op::Spec(3),
        Op::Spec(6),
    ];
    let mut total = 0;
    for (label, ops) in [("forward", &forward[..]), ("reverse", &reverse[..])] {
        let arm = rig.schedule(ops, None);
        let reference = rig.reference(&arm);
        total += mismatches(label, &arm, &reference);
    }
    // Sensitivity control: one AR token fed a wrong seed must show up.
    let sabotaged = rig.schedule(&forward, Some(3));
    let reference = rig.reference(&sabotaged);
    let detected = mismatches("sabotage", &sabotaged, &reference);
    let replay = rig.gpu.replay.replay_observation();
    eprintln!(
        "spec_floor_schedule_hw: replay count={} failed={}",
        replay.count, replay.failed
    );
    assert_eq!(total, 0, "a floor schedule differs from pure AR");
    assert!(detected > 0, "the sabotage control was not detected");
}
