// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S1 slow-link layer-major prefill: focused hardware tests.
//!
//! The `#[ignore]`d tests need a HIP GPU, a discrete narrow-link card and the
//! canonical MQ4 Flash-Next artifact named by `HIPFIRE_SLOWLINK_HW_MODEL`.
//! Placement is the loader's own: the S1 admission's joint placement (the
//! largest resident layer count that actually fits, ten preferred but never a
//! floor, the rest host-mapped); the tests refuse to run when the profile was
//! not admitted or its scalar admission witness is not a maximal-fits proof.
//! They force the route through the one production switch
//! (`HIPFIRE_QWEN4_SLOWLINK_PREFILL=1`, set before the first config read) and
//! run only inside a fabric hold with `--test-threads=1`:
//!
//! ```text
//! fabric-run.sh --class correct --card xtx --mem-gb <actual peak> \
//!   --hostmem-gb <actual unreclaimable> --hostmem-lock --max-secs 600 -- \
//!   slowlink_prefill_hw --ignored --test-threads=1 --exact <test>
//! ```
//!
//! `HIPFIRE_SLOWLINK_HW_ROWS` (default 3400, four full 768-row product tiles
//! and a tail; it must exceed the 1,536-row named incumbent-bypass bound) sets
//! the prompt length. Every product test loads the `Product` purpose and
//! judges its receipts against that profile's tile (`purpose_tile_rows`, read
//! back from the bundle), never the generic `Limits::default()`; the
//! primitives test loads `PrimitiveSelftest` (512 rows) and proves kernel
//! correctness only. The 16384-row one-window proof is the gate's `state`
//! mode (`examples/qwen4_slowlink_gate.rs`). The un-ignored tests are
//! GPU-free and prove the shared receipt validation rejects what the plan
//! requires it to reject.

use hipfire_arch_qwen4::state_parity::{
    slowlink_load_bundle, slowlink_logit_rows_summary, slowlink_state_digest, SlowlinkLoaded,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::path::PathBuf;

#[path = "common/slowlink_receipt.rs"]
mod slowlink_receipt;
use slowlink_receipt::*;

const MODEL_ENV: &str = "HIPFIRE_SLOWLINK_HW_MODEL";
const ROWS_ENV: &str = "HIPFIRE_SLOWLINK_HW_ROWS";
const ROUTE_ENV: &str = "HIPFIRE_QWEN4_SLOWLINK_PREFILL";
const MAX_SEQ: usize = 16640;
const DECODE: usize = 16;

fn prompt(seed: u64, len: usize) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (1000 + (state >> 33) % 99_000) as u32
        })
        .collect()
}

/// Why a hardware test must not load the model here: outside a fabric hold
/// (`FABRIC_JOB` names the live job) or without `--test-threads=1`.
fn hold_refusal(fabric_job: &str, threads_one: bool) -> Option<String> {
    if !threads_one {
        return Some("each test loads Flash-Next: run with --test-threads=1".into());
    }
    if fabric_job.trim().is_empty() {
        return Some("no fabric hold (FABRIC_JOB unset)".into());
    }
    None
}

fn require_hold() {
    let args: Vec<String> = std::env::args().collect();
    let threads_one = std::env::var("RUST_TEST_THREADS").is_ok_and(|v| v == "1")
        || args.iter().any(|a| a == "--test-threads=1")
        || args
            .windows(2)
            .any(|w| w[0] == "--test-threads" && w[1] == "1");
    let job = std::env::var("FABRIC_JOB").unwrap_or_default();
    if let Some(why) = hold_refusal(&job, threads_one) {
        panic!("refusing to load Flash-Next: {why}");
    }
}

struct Hw {
    gpu: Gpu,
    bundle: hipfire_arch_qwen4::bundle::Qwen4Bundle,
    logits: GpuTensor,
    vocab: usize,
    eos: u32,
    rows: usize,
    /// The admitted purpose profile's tile, read from the loaded bundle.
    tile: u64,
}

type Purpose = hipfire_arch_qwen4::slowlink_admission::AdmissionPurpose;

fn load(purpose: Purpose) -> Hw {
    require_hold();
    // Before the first config read of this process.
    match std::env::var(ROUTE_ENV) {
        Ok(v) => assert_eq!(
            v, "1",
            "{ROUTE_ENV}={v} conflicts with the forced layer-major tests"
        ),
        Err(_) => std::env::set_var(ROUTE_ENV, "1"),
    }
    let model = std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{MODEL_ENV} must name the MQ4 Flash-Next artifact"));
    let rows = std::env::var(ROWS_ENV)
        .ok()
        .map_or(3400, |v| v.parse::<usize>().expect("rows"));
    // The named incumbent bypass covers requests of at most 1,536 rows, so a
    // forced S1 request must be longer than that (several product tiles).
    assert!(
        rows > MAX_TILE_ROWS as usize,
        "the prompt must exceed the named incumbent-bypass bound"
    );
    let mut gpu = Gpu::init().expect("GPU");
    let SlowlinkLoaded {
        bundle,
        vocab,
        eos_token,
        slowlink_profile,
        resident_layers,
        num_layers,
        placement,
        ..
    } = slowlink_load_bundle(&mut gpu, &model, MAX_SEQ, None, purpose).expect("load");
    assert!(
        slowlink_profile,
        "the forced route was not admitted (placement {placement}, resident layers {resident_layers})"
    );
    // No resident floor: the actual count must be the maximal count that fits
    // (selected fits, one more does not, a host layer remains), judged from
    // the scalar witness of the shared quote against the bundle's own counts.
    {
        let profile = bundle.slowlink_profile().expect("an admitted S1 profile");
        let witness = admission_witness(profile);
        let bad = admission_witness_problems(
            Some(&witness),
            &admission_actual(Some(profile), resident_layers, num_layers),
        );
        assert!(
            bad.is_empty(),
            "the admission is not a maximal-fits proof: {bad:?}\nwitness: {witness}"
        );
        println!("S1 admission witness: {witness}");
    }
    // The bundle's actual admitted profile must be this purpose's, at exactly
    // the tile the loader derives for it.
    let (admitted_purpose, tile) = {
        let profile = bundle.slowlink_profile().expect("an admitted S1 profile");
        (profile.purpose, profile.tile_rows as u64)
    };
    assert_eq!(
        admitted_purpose, purpose,
        "the load admitted another purpose"
    );
    assert_eq!(
        tile,
        hipfire_arch_qwen4::slowlink_admission::purpose_tile_rows(purpose) as u64,
        "the admitted tile is not the purpose's tile"
    );
    let logits = gpu.zeros(&[vocab], DType::F32).expect("logits");
    Hw {
        gpu,
        bundle,
        logits,
        vocab,
        eos: eos_token,
        rows,
        tile,
    }
}

impl Hw {
    fn prefill(&mut self, ids: &[u32]) -> (Value, Value, Option<Value>) {
        self.bundle.reset(&mut self.gpu).expect("reset");
        self.bundle
            .prefill_final(&mut self.gpu, ids, 0, &self.logits)
            .expect("prefill");
        let receipt = self.bundle.slowlink_receipt();
        let state = slowlink_state_digest(&self.gpu, &self.bundle).expect("state digest");
        let logits = slowlink_logit_rows_summary(&self.gpu, &self.logits, 1, self.vocab, 16)
            .expect("logit summary");
        (state, logits, receipt)
    }

    fn decode(&mut self, count: usize) -> Vec<u32> {
        let row = self.gpu.download_f32(&self.logits).expect("download");
        let mut pending = Some(hipfire_runtime::llama::argmax(&row));
        let mut ids = Vec::new();
        while ids.len() < count {
            let token = self
                .bundle
                .forward_token_or_argmax(&mut self.gpu, pending.take(), &self.logits)
                .expect("decode");
            ids.push(token);
            if token == self.eos {
                break;
            }
        }
        ids
    }

    fn finish(self) {
        let Hw {
            mut gpu,
            bundle,
            logits,
            ..
        } = self;
        gpu.free_tensor(logits).expect("free logits");
        bundle.free_gpu(&mut gpu).expect("free bundle");
    }
}

/// A receipt is valid only at the admitted purpose's tile (`tile`, read from
/// the loaded bundle and checked against `purpose_tile_rows`), never the generic
/// default upper bound.
fn assert_valid(receipt: &Option<Value>, rows: usize, tile: u64) {
    let limits = Limits {
        require_single_window: rows as u64 == TARGET_ROWS,
        expected_tile_rows: Some(tile),
        ..Limits::default()
    };
    let bad = validate(receipt.as_ref(), Expect::LayerMajor, &limits);
    assert!(
        bad.is_empty(),
        "receipt problems: {bad:?}\nreceipt: {receipt:?}"
    );
}

/// The production panel transport, dispatch and ordered combine, on the
/// admitted profile's real host-expert sources (never a CPU model).
#[test]
#[ignore = "needs a HIP GPU, a narrow-link host and HIPFIRE_SLOWLINK_HW_MODEL"]
fn slowlink_primitives_pass_on_the_admitted_profile() {
    // The self-test runs on its own purpose's profile (512 rows, with the
    // diagnostic reserve): kernel correctness only, not a product-shape claim.
    let mut hw = load(Purpose::PrimitiveSelftest);
    let report = hw
        .bundle
        .slowlink_primitive_selftest(&mut hw.gpu, &PRIMITIVE_CASES)
        .expect("self-test on the admitted S1 profile");
    let cases = report["cases"].as_array().expect("cases");
    for name in PRIMITIVE_CASES {
        let case = cases
            .iter()
            .find(|c| c["name"] == name)
            .unwrap_or_else(|| panic!("self-test did not run {name}"));
        assert_eq!(case["status"], "pass", "{name}: {case}");
        assert_eq!(case["mismatches"].as_u64(), Some(0), "{name}: {case}");
    }
    hw.finish();
}

/// Same-route determinism across repeated epochs at the same device
/// addresses, and one submitted payload sweep per request each time (no
/// refetch, no growth): the lookahead pages are reused with delayed readers.
#[test]
#[ignore = "needs a HIP GPU, a narrow-link host and HIPFIRE_SLOWLINK_HW_MODEL"]
fn slowlink_repeated_epochs_are_deterministic_with_one_sweep_each() {
    let mut hw = load(Purpose::Product);
    let ids = prompt(0x51, hw.rows);
    let first = hw.prefill(&ids);
    assert_valid(&first.2, hw.rows, hw.tile);
    let submitted = u64_at(first.2.as_ref().unwrap(), "submitted_bytes");
    for epoch in 1..4 {
        let again = hw.prefill(&ids);
        assert_valid(&again.2, hw.rows, hw.tile);
        assert_eq!(
            again.0["combined"], first.0["combined"],
            "epoch {epoch}: state digest"
        );
        assert_eq!(
            again.1["combined"], first.1["combined"],
            "epoch {epoch}: final logits"
        );
        assert_eq!(
            u64_at(again.2.as_ref().unwrap(), "submitted_bytes"),
            submitted,
            "epoch {epoch}: submitted bytes changed (refetch or leak)"
        );
    }
    hw.finish();
}

/// Each named failure aborts the request, restores or contains the state,
/// and a clean re-run reproduces the clean digests (no donor restore, no
/// poisoned pages).
#[test]
#[ignore = "needs a HIP GPU, a narrow-link host and HIPFIRE_SLOWLINK_HW_MODEL"]
fn slowlink_injected_failures_abort_cleanly_and_rerun_matches() {
    let mut hw = load(Purpose::Product);
    let ids = prompt(0x52, hw.rows);
    let clean = hw.prefill(&ids);
    assert_valid(&clean.2, hw.rows, hw.tile);
    for point in FAILURE_POINTS {
        hw.bundle.reset(&mut hw.gpu).expect("reset");
        hw.bundle.slowlink_inject_failure(Some(point)).expect("arm");
        let failed = hw.bundle.prefill_final(&mut hw.gpu, &ids, 0, &hw.logits);
        assert!(
            failed.is_err(),
            "{point}: the injected failure did not fail the request"
        );
        let consumed = hw
            .bundle
            .slowlink_receipt()
            .and_then(|r| r.get("injection_consumed").and_then(Value::as_bool));
        assert_eq!(consumed, Some(true), "{point}: injection not consumed");
        hw.bundle.slowlink_inject_failure(None).expect("clear");
        let again = hw.prefill(&ids);
        assert_valid(&again.2, hw.rows, hw.tile);
        assert_eq!(
            again.0["combined"], clean.0["combined"],
            "{point}: state after abort + rerun"
        );
        assert_eq!(
            again.1["combined"], clean.1["combined"],
            "{point}: logits after abort + rerun"
        );
    }
    hw.finish();
}

/// Decode after the S1 prefill: the retained-body replay (the only decode
/// body replay Qwen4 production has) must reproduce the eager ids, with every
/// measured token replayed.
#[test]
#[ignore = "needs a HIP GPU, a narrow-link host and HIPFIRE_SLOWLINK_HW_MODEL"]
fn slowlink_decode_after_prefill_replays_the_eager_ids() {
    let mut hw = load(Purpose::Product);
    let ids = prompt(0x53, hw.rows);
    hw.gpu.replay.configure_model_default(true);
    hw.prefill(&ids);
    let row = hw.gpu.download_f32(&hw.logits).expect("download");
    let mut pending = Some(hipfire_runtime::llama::argmax(&row));
    let mut replay_ids = Vec::new();
    let mut warm = 0;
    while hw.gpu.replay.state() != rdna_compute::replay::ReplayState::Ready && warm < 6 {
        replay_ids.push(
            hw.bundle
                .forward_token_or_argmax(&mut hw.gpu, pending.take(), &hw.logits)
                .expect("warm"),
        );
        warm += 1;
    }
    assert_eq!(
        hw.gpu.replay.state(),
        rdna_compute::replay::ReplayState::Ready,
        "retained body never became Ready"
    );
    hw.gpu.replay.begin_replay_observation_window();
    for _ in 0..DECODE {
        replay_ids.push(
            hw.bundle
                .forward_token_or_argmax(&mut hw.gpu, pending.take(), &hw.logits)
                .expect("replay"),
        );
    }
    let observed = hw.gpu.replay.replay_observation();
    assert_eq!(
        observed.count, DECODE as u64,
        "not every measured token replayed"
    );
    assert!(!observed.failed);
    hw.gpu.replay.configure_model_default(false);
    hw.prefill(&ids);
    let eager = hw.decode(replay_ids.len());
    assert_eq!(
        eager, replay_ids,
        "retained replay differs from eager decode after S1 prefill"
    );
    hw.finish();
}

// ── GPU-free ───────────────────────────────────────────────────────────

#[test]
fn slowlink_hold_guard_refuses_loads_outside_a_hold() {
    assert!(hold_refusal("", true).is_some());
    assert!(hold_refusal("123:456:xtx", false).is_some());
    assert!(hold_refusal("123:456:xtx", true).is_none());
}

#[test]
fn slowlink_receipt_a_missing_receipt_or_route_never_validates() {
    assert!(!validate(None, Expect::LayerMajor, &Limits::default()).is_empty());
    let partial = json!({"schema": RECEIPT_SCHEMA, "windows": []});
    assert!(!validate(Some(&partial), Expect::LayerMajor, &Limits::default()).is_empty());
}

#[test]
fn slowlink_receipt_a_short_forced_bypass_must_name_its_reason() {
    let incumbent = json!({
        "schema": RECEIPT_SCHEMA, "route": "incumbent", "selected_by": "env1", "bypass_reason": null,
        "entry_pos": 0, "prompt_rows": 512, "windows": [{"start": 0, "rows": 512}], "launch_cuts": [],
        "tile_rows": 512, "retained_scratch_rows": 512, "sweeps": 0, "resident_layers": 10, "host_layers": 38,
        "host_expert_bytes_total": 0, "submitted_bytes": 0, "useful_bytes": 0, "unused_prefetched_bytes": 0,
        "reserve": {"tile": 0, "arena": 0, "ring": 0, "lookahead": 0, "tables": 0, "total": 0},
        "spans_ms": {"wall": 1.0, "copy": 0.0, "compute": 1.0, "route_barrier": 0.0, "nonexpert": 1.0},
        "overlap": {"cross_layer_intersections": 0, "cross_layer_overlap_ms": 0.0, "copy_idle_ms": 0.0,
                    "page_full_stalls": 0, "copy_starved_stalls": 0},
        "peak_vram_bytes": 1,
    });
    assert!(!validate_forced_short(Some(&incumbent), &Limits::default()).is_empty());
    let mut named = incumbent;
    named["bypass_reason"] = json!("request shorter than the S1 qualification");
    assert!(validate_forced_short(Some(&named), &Limits::default()).is_empty());
}

/// The hardware product proof is judged against the admitted purpose's tile,
/// never the generic `Limits::default()` (shape-agnostic, historical units): a
/// product receipt at the old 1,536 rows or at the primitive 512 rows fails.
#[test]
fn slowlink_receipt_product_hardware_limits_name_the_admitted_purpose_tile() {
    use hipfire_arch_qwen4::slowlink_admission::{purpose_tile_rows, AdmissionPurpose};
    let receipt = |tile: u64| {
        json!({
            "schema": RECEIPT_SCHEMA, "route": "layer-major", "selected_by": "env1", "bypass_reason": null,
            "entry_pos": 0, "prompt_rows": 3400, "windows": [{"start": 0, "rows": 3400}], "launch_cuts": [],
            "tile_rows": tile, "retained_scratch_rows": tile, "sweeps": 1, "resident_layers": 10, "host_layers": 38,
            "host_expert_bytes_total": 50_803_507_200u64, "submitted_bytes": 50_803_507_200u64,
            "useful_bytes": 50_000_000_000u64, "unused_prefetched_bytes": 803_507_200u64,
            "reserve": {"tile": 1u64 << 30, "arena": 1u64 << 29, "ring": 167_772_160u64,
                        "lookahead": 1_336_000_000u64, "tables": 1u64 << 20, "total": 4_000_000_000u64},
            "spans_ms": {"wall": 1.0, "copy": 0.5, "compute": 0.5, "route_barrier": 0.0, "nonexpert": 0.1},
            "overlap": {"cross_layer_intersections": 1, "cross_layer_overlap_ms": 1.0, "copy_idle_ms": 0.0,
                        "page_full_stalls": 0, "copy_starved_stalls": 0},
            "peak_vram_bytes": 1,
        })
    };
    let product = purpose_tile_rows(AdmissionPurpose::Product) as u64;
    let primitive = purpose_tile_rows(AdmissionPurpose::PrimitiveSelftest) as u64;
    assert_eq!((product, primitive), (768, 512));
    let bound = |tile: u64| Limits {
        expected_tile_rows: Some(tile),
        ..Limits::default()
    };
    assert!(validate(Some(&receipt(768)), Expect::LayerMajor, &bound(product)).is_empty());
    for wrong in [1536u64, 1024, primitive] {
        assert!(
            !validate(Some(&receipt(wrong)), Expect::LayerMajor, &bound(product)).is_empty(),
            "a product receipt at {wrong} rows must fail"
        );
    }
    // The shape-agnostic default stays only for historical units.
    assert!(validate(Some(&receipt(1536)), Expect::LayerMajor, &Limits::default()).is_empty());
}

/// Adaptive residency has no resident floor: nine, zero and eleven resident
/// layers close on the trunk and validate; ten is only the preferred count.
#[test]
fn slowlink_receipt_adaptive_resident_counts_validate_without_a_floor() {
    let receipt = |resident: u64| {
        let host = TRUNK_LAYERS - resident;
        let bytes = host * COLD_BYTES_PER_HOST_LAYER;
        json!({
            "schema": RECEIPT_SCHEMA, "route": "layer-major", "selected_by": "env1", "bypass_reason": null,
            "entry_pos": 0, "prompt_rows": 3400, "windows": [{"start": 0, "rows": 3400}], "launch_cuts": [],
            "tile_rows": 768, "retained_scratch_rows": 768, "sweeps": 1,
            "resident_layers": resident, "host_layers": host,
            "host_expert_bytes_total": bytes, "submitted_bytes": bytes,
            "useful_bytes": bytes / 2, "unused_prefetched_bytes": bytes / 2,
            "reserve": {"tile": 1u64 << 30, "arena": 1u64 << 29, "ring": 167_772_160u64,
                        "lookahead": 1_336_000_000u64, "tables": 1u64 << 20, "total": 4_000_000_000u64},
            "spans_ms": {"wall": 1.0, "copy": 0.5, "compute": 0.5, "route_barrier": 0.0, "nonexpert": 0.1},
            "overlap": {"cross_layer_intersections": 1, "cross_layer_overlap_ms": 1.0, "copy_idle_ms": 0.0,
                        "page_full_stalls": 0, "copy_starved_stalls": 0},
            "peak_vram_bytes": 1,
        })
    };
    let limits = Limits {
        expected_tile_rows: Some(768),
        ..Limits::default()
    };
    for resident in [0u64, 9, 10, 11] {
        let bad = validate(Some(&receipt(resident)), Expect::LayerMajor, &limits);
        assert!(bad.is_empty(), "resident {resident}: {bad:?}");
    }
    // Counts that do not close on the trunk, and no host layer at all, fail.
    let mut open = receipt(9);
    open["host_layers"] = json!(38);
    assert!(!validate(Some(&open), Expect::LayerMajor, &limits).is_empty());
    assert!(!validate(Some(&receipt(TRUNK_LAYERS)), Expect::LayerMajor, &limits).is_empty());
    assert_eq!(PREFERRED_RESIDENT_LAYERS, 10);
}

/// The SP short-prompt hook's `sp` object is optional (the knob-off receipt
/// has none and validates unchanged); present, it must be coherent.
#[test]
fn slowlink_receipt_sp_object_is_optional_and_checked_when_present() {
    let receipt = |route: &str| {
        json!({
            "schema": RECEIPT_SCHEMA, "route": route, "selected_by": "auto", "bypass_reason": null,
            "entry_pos": 0, "prompt_rows": 1024, "windows": [{"start": 0, "rows": 1024}], "launch_cuts": [768, 1024],
            "tile_rows": 768, "retained_scratch_rows": 768, "sweeps": 1, "resident_layers": 10, "host_layers": 38,
            "host_expert_bytes_total": 50_803_507_200u64, "submitted_bytes": 20_000_000_000u64,
            "useful_bytes": 19_000_000_000u64, "unused_prefetched_bytes": 1_000_000_000u64,
            "reserve": {"tile": 1u64 << 30, "arena": 1u64 << 29, "ring": 167_772_160u64,
                        "lookahead": 1_336_000_000u64, "tables": 1u64 << 20, "total": 4_000_000_000u64},
            "spans_ms": {"wall": 1.0, "copy": 0.5, "compute": 0.5, "route_barrier": 0.0, "nonexpert": 0.1},
            "overlap": {"cross_layer_intersections": 0, "cross_layer_overlap_ms": 0.0, "copy_idle_ms": 0.0,
                        "page_full_stalls": 0, "copy_starved_stalls": 0},
            "peak_vram_bytes": 1,
        })
    };
    let limits = Limits {
        expected_tile_rows: Some(768),
        ..Limits::default()
    };
    let window = |engaged: bool, bypass: Value, layers: u64| {
        json!({"rows": 1024, "engaged": engaged, "bypass": bypass, "layers": layers,
               "streamed_bytes": 100u64, "streamed_experts": 5u64, "cpu_experts": 7u64, "cpu_slots": 9u64})
    };
    let sp = |armed: bool, decline: Value, unavailable: Value, windows: Vec<Value>| {
        let engaged = windows
            .iter()
            .filter(|w| w["engaged"] == json!(true))
            .count();
        json!({"knob": true, "armed": armed, "decline": decline, "decline_reason": decline.clone(),
               "unavailable": unavailable, "engaged_windows": engaged, "windows": windows})
    };
    // No sp key: the knob-off receipt validates exactly as before.
    let off = receipt("layer-major");
    assert!(validate(Some(&off), Expect::LayerMajor, &limits).is_empty());
    assert!(sp_problems(&off).is_empty());
    // An armed hook that ran its window.
    let mut ran = receipt("layer-major");
    ran["sp"] = sp(
        true,
        Value::Null,
        Value::Null,
        vec![window(true, Value::Null, 38)],
    );
    assert!(
        validate(Some(&ran), Expect::LayerMajor, &limits).is_empty(),
        "{:?}",
        validate(Some(&ran), Expect::LayerMajor, &limits)
    );
    // A declined hook names why, and its window never engaged.
    let mut declined = receipt("layer-major");
    declined["sp"] = sp(
        false,
        json!("sp: arch not calibrated"),
        Value::Null,
        vec![window(false, Value::Null, 0)],
    );
    assert!(validate(Some(&declined), Expect::LayerMajor, &limits).is_empty());
    // A bypassed window of an armed hook names its reason.
    let mut bypassed = receipt("layer-major");
    bypassed["sp"] = sp(
        true,
        Value::Null,
        Value::Null,
        vec![window(
            false,
            json!("sp_bypass: 17000 rows exceeds the 5376-row maximum"),
            0,
        )],
    );
    assert!(validate(Some(&bypassed), Expect::LayerMajor, &limits).is_empty());
    // The incumbent receipt of an armed hook names the hook's state, engaging nothing.
    let mut incumbent = receipt("incumbent");
    incumbent["reserve"] =
        json!({"tile": 0, "arena": 0, "ring": 0, "lookahead": 0, "tables": 0, "total": 0});
    incumbent["sweeps"] = json!(0);
    incumbent["sp"] = sp(true, Value::Null, Value::Null, Vec::new());
    assert!(sp_problems(&incumbent).is_empty());
    // Incoherent objects are refused.
    let broken = |mutate: &dyn Fn(&mut Value)| {
        let mut r = receipt("layer-major");
        r["sp"] = sp(
            true,
            Value::Null,
            Value::Null,
            vec![window(true, Value::Null, 38)],
        );
        mutate(&mut r["sp"]);
        sp_problems(&r)
    };
    assert!(
        !broken(&|s| s["armed"] = json!(false)).is_empty(),
        "engaged while unarmed"
    );
    assert!(
        !broken(&|s| s["decline"] = json!("sp: route")).is_empty(),
        "armed with a decline"
    );
    assert!(
        !broken(&|s| s["engaged_windows"] = json!(0)).is_empty(),
        "count mismatch"
    );
    assert!(
        !broken(&|s| s["knob"] = json!(false)).is_empty(),
        "knob must be true"
    );
    assert!(
        !broken(&|s| s["windows"][0]["bypass"] = json!("x")).is_empty(),
        "engaged with a bypass"
    );
    assert!(
        !broken(&|s| s["windows"][0]["layers"] = json!(0)).is_empty(),
        "engaged, no layer"
    );
    assert!(!broken(&|s| {
        s.as_object_mut().unwrap().remove("windows");
    })
    .is_empty());
    let mut not_object = receipt("layer-major");
    not_object["sp"] = json!(true);
    assert!(!sp_problems(&not_object).is_empty());
    let mut unarmed_silent = receipt("layer-major");
    unarmed_silent["sp"] = sp(false, Value::Null, Value::Null, Vec::new());
    assert!(
        !sp_problems(&unarmed_silent).is_empty(),
        "unarmed without a reason"
    );
    let mut engaged_incumbent = receipt("incumbent");
    engaged_incumbent["sp"] = sp(
        true,
        Value::Null,
        Value::Null,
        vec![window(true, Value::Null, 38)],
    );
    assert!(
        !sp_problems(&engaged_incumbent).is_empty(),
        "SP cannot engage on the incumbent"
    );
}
