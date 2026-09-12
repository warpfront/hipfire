// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G4.10 dense-rollback GPU contract tests (ignored: real HIP GPU + fixtures).
//!
//! Drives the real `ar::generate` entry point against two fixtures from
//! `HIPFIRE_DENSE_FIXTURE` (a directory; falls back to `~/.hipfire/models`):
//!
//! - `qwen3-0.6b.hf4` (LLaMA carrier) exercises the G4.10(b) LLaMA AR loop;
//! - `lfm2.5-1.2b.mq4` exercises the G4.10(a) dense-family fault seams.
//!
//! Per fixture the test pins the full serve contract:
//!
//! - arm the after-prefill / after-first-decode fault via the (a) hooks →
//!   exactly one correlated terminal (one `error`, never `done`), attested
//!   `rolled_back=true`, no assistant-cache store;
//! - the next clean request is envelope-equal to a fresh-daemon (fresh load)
//!   result — rollback restored serve state exactly;
//! - unload/reload returns VRAM to the warmed baseline.
//!
//! A third test pins the reset-phase contract: when rollback itself cannot
//! be attested, the terminal is still exactly one visible error carrying
//! the reset context with `rolled_back=false`.
//!
//! `generate` writes to `&mut std::io::Stdout`, so captures redirect fd 1
//! to a temp file (no new dependencies: one `extern "C"` block).
//! All tests share one process-global lock: VRAM accounting and fd 1 are
//! device-/process-global and cannot run concurrently.

#![allow(clippy::all)]

use hipfire_engine::terminal::*;
use hipfire_generate::ar::GenerationRoute;
use hipfire_generate::common::{
    arm_generation_fault_after_first_decode, arm_generation_fault_after_prefill,
    emit_fail_closed_error, fail_closed_device_sync, fail_closed_epilogue_after_sync,
};
use hipfire_runtime::loader_api::{CaskConfig, SpecLoadCfg};
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

extern "C" {
    fn dup(oldfd: RawFd) -> RawFd;
    fn dup2(oldfd: RawFd, newfd: RawFd) -> RawFd;
    fn close(fd: RawFd) -> std::ffi::c_int;
}

static DENSE_GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    DENSE_GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Slack for device-global free-byte assertions: far below one resident
/// model, so a genuinely leaked load still fails loudly.
const DENSE_VRAM_SLACK_BYTES: usize = 64 << 20;

/// Warm-up upload that pays the ROCm first-device-allocation reservation
/// before any VRAM baseline is taken. The first `hipMalloc` in a HIP
/// process permanently reserves a fixed driver-side VM/setup block that no
/// process-tracked owner can free: measured 153,092,096 bytes (146.00 MiB)
/// on gfx1201/R9700. Warming here (by name, not as slack) keeps the
/// unload/reload assertions strict. Idempotent within a process.
const DENSE_ROCM_FIRST_ALLOC_WARMUP_BYTES: usize = 1 << 20;

fn dense_fixture_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HIPFIRE_DENSE_FIXTURE") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home).join(".hipfire").join("models")
}

fn dense_fixture(name: &str) -> Option<PathBuf> {
    let path = dense_fixture_dir().join(name);
    if path.is_file() {
        Some(path)
    } else {
        eprintln!(
            "skip: dense fixture {name} not found under {}",
            dense_fixture_dir().display()
        );
        None
    }
}

fn dense_gpu() -> Option<rdna_compute::Gpu> {
    match rdna_compute::Gpu::init() {
        Ok(gpu) => Some(gpu),
        Err(_) => {
            eprintln!("skip: no GPU");
            None
        }
    }
}

fn dense_free_vram_bytes(gpu: &rdna_compute::Gpu) -> usize {
    gpu.hip.get_vram_info().expect("dense test VRAM query").0
}

fn dense_warm_first_alloc(gpu: &mut rdna_compute::Gpu) {
    let warm = gpu
        .upload_raw(
            &vec![0u8; DENSE_ROCM_FIRST_ALLOC_WARMUP_BYTES],
            &[DENSE_ROCM_FIRST_ALLOC_WARMUP_BYTES],
        )
        .expect("dense first-alloc warm-up upload");
    gpu.free_tensor(warm)
        .expect("dense first-alloc warm-up free");
    gpu.drain_pool();
}

fn dense_spec_off() -> SpecLoadCfg {
    SpecLoadCfg {
        ngram_draft: Some(false),
        dflash: Some(false),
        mtp: Some(false),
        dspark: Some(false),
        ..Default::default()
    }
}

fn dense_load(gpu: &mut rdna_compute::Gpu, path: &PathBuf) -> hipfire_loader::LoadedModel {
    hipfire_loader::load_model(
        &path.to_string_lossy(),
        512,
        None,
        None,
        None,
        None,
        &CaskConfig::default(),
        1,
        dense_spec_off(),
        gpu,
    )
    .unwrap_or_else(|e| panic!("load dense fixture {}: {e}", path.display()))
}

/// Redirect fd 1 to a fresh temp file; returns the saved fd + path.
fn start_stdout_capture(tag: &str) -> (RawFd, PathBuf) {
    use std::io::Write as _;
    std::io::stdout().flush().expect("flush before capture");
    let saved = unsafe { dup(1) };
    assert!(saved >= 0, "dup stdout");
    let path = std::env::temp_dir().join(format!(
        "dense_rollback_gpu_{tag}_{}.jsonl",
        std::process::id()
    ));
    let file = std::fs::File::create(&path).expect("capture file");
    assert!(
        unsafe { dup2(file.as_raw_fd(), 1) } >= 0,
        "redirect stdout to capture"
    );
    // `file` closed here; fd 1 holds its own reference.
    (saved, path)
}

fn finish_stdout_capture(saved: RawFd, path: &PathBuf) -> Vec<u8> {
    use std::io::Write as _;
    std::io::stdout().flush().expect("flush captured stdout");
    assert!(unsafe { dup2(saved, 1) } >= 0, "restore stdout");
    unsafe {
        close(saved);
    }
    std::io::stdout().flush().expect("flush restored stdout");
    std::fs::read(path).expect("read capture")
}

fn parse_lines(output: &[u8]) -> Vec<serde_json::Value> {
    std::str::from_utf8(output)
        .expect("UTF-8 events")
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).expect("JSON event"))
        .collect()
}

/// Timing fields are wall-clock derived and legitimately jitter run to run;
/// `id`/`attempt_id` are request identity (each turn runs under its own
/// key by construction). Everything else (token bytes, structure,
/// terminals) must match exactly.
fn normalize_envelope(mut v: serde_json::Value) -> serde_json::Value {
    if let Some(obj) = v.as_object_mut() {
        for key in [
            "tok_s",
            "prefill_ms",
            "prefill_tok_s",
            "decode_tok_s",
            "ttft_ms",
            "total_ms",
            "latency_ms",
            "prefill_s",
            "decode_s",
            "id",
            "attempt_id",
        ] {
            obj.remove(key);
        }
    }
    v
}

fn fresh_attempt(id: &str, attempt: u64) {
    batch_clear_all_terminals();
    clear_terminal_control();
    set_active_attempt_id(attempt);
    activate_terminal_control(id, attempt);
    assert_eq!(hipfire_generate::ar::active_generation_route(), None);
}

/// Spawn a committer that acks the two-phase `commit_ready` handshake so a
/// clean `generate` run terminates. Returns the join handle + stop flag.
fn spawn_committer(
    id: &str,
    attempt: u64,
) -> (
    std::thread::JoinHandle<()>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_clone = stop.clone();
    let id = id.to_string();
    let handle = std::thread::spawn(move || {
        while !stop_clone.load(std::sync::atomic::Ordering::SeqCst) {
            apply_terminal_control("commit", &id, attempt);
            std::thread::sleep(Duration::from_millis(20));
        }
    });
    (handle, stop)
}

#[allow(clippy::too_many_arguments)]
fn run_generate(
    m: &mut hipfire_loader::LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    tag: &str,
    id: &str,
    attempt: u64,
    commit: bool,
) -> Vec<u8> {
    fresh_attempt(id, attempt);
    let committer = if commit {
        Some(spawn_committer(id, attempt))
    } else {
        None
    };
    let (saved, path) = start_stdout_capture(tag);
    {
        let mut stdout = std::io::stdout();
        hipfire_generate::ar::generate(
            m,
            gpu,
            None,
            &mut stdout,
            id,
            "Say hello in one short sentence.",
            None,
            false,
            0.0,
            1.0,
            None,
            None,
            0.0,
            8,
            1.0,
            64,
            0.0,
            0.0,
            usize::MAX,
            "",
            1,
            hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
            None,
            None,
            None,
            None,
            hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            &[],
            None,
            false,
            None,
            12345,
        );
    }
    let bytes = finish_stdout_capture(saved, &path);
    if let Some((handle, stop)) = committer {
        stop.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = handle.join();
    }
    let _ = std::fs::remove_file(&path);
    arm_generation_fault_after_prefill(false);
    arm_generation_fault_after_first_decode(false);
    bytes
}

fn assert_single_attested_error(
    output: &[u8],
    id: &str,
    attempt: u64,
    cache_empty_before: bool,
    m: &hipfire_loader::LoadedModel,
) {
    let events = parse_lines(output);
    let errors: Vec<&serde_json::Value> =
        events.iter().filter(|e| e["type"] == "error").collect();
    assert_eq!(
        errors.len(),
        1,
        "exactly one correlated terminal: {events:?}"
    );
    assert!(
        events.iter().all(|e| e["type"] != "done"),
        "fault turn never emits done: {events:?}"
    );
    let err = errors[0];
    assert_eq!(err["id"], id);
    assert_eq!(err["attempt_id"], attempt);
    assert_eq!(err["rolled_back"], true);
    assert!(
        !claim_wire_terminal(id, attempt),
        "fault turn must claim exactly one terminal"
    );
    assert_eq!(
        m.asst_turn_cache.is_empty(),
        cache_empty_before,
        "fault turn stores no assistant cache"
    );
}

fn assert_envelope_equal(baseline: &[u8], retry: &[u8]) {
    let base_events = parse_lines(baseline);
    let retry_events = parse_lines(retry);
    assert_eq!(
        base_events.len(),
        retry_events.len(),
        "same envelope count after rollback"
    );
    let base_norm: Vec<serde_json::Value> = base_events
        .into_iter()
        .map(normalize_envelope)
        .collect();
    let retry_norm: Vec<serde_json::Value> = retry_events
        .into_iter()
        .map(normalize_envelope)
        .collect();
    assert_eq!(
        base_norm, retry_norm,
        "post-rollback turn is envelope-equal to the fresh turn"
    );
    // The generated token bytes themselves are identical.
    let base_text: String = base_norm
        .iter()
        .filter(|e| e["type"] == "token")
        .filter_map(|e| e["text"].as_str())
        .collect();
    let retry_text: String = retry_norm
        .iter()
        .filter(|e| e["type"] == "token")
        .filter_map(|e| e["text"].as_str())
        .collect();
    assert_eq!(
        base_text, retry_text,
        "generated bytes identical after rollback"
    );
}

/// Full fault contract for one fixture: clean baseline, both fault hooks,
/// retry equality, unload/reload VRAM back to the warmed baseline.
fn exercise_dense_fixture(name: &str, attempts: u64) {
    let _guard = lock();
    let Some(path) = dense_fixture(name) else {
        return;
    };
    let Some(mut gpu) = dense_gpu() else {
        return;
    };
    dense_warm_first_alloc(&mut gpu);
    let free_warmed = dense_free_vram_bytes(&gpu);

    // Fresh-daemon baseline: load, one clean turn.
    let mut m = dense_load(&mut gpu, &path);
    let id_base = format!("dense-{attempts}");
    let baseline = run_generate(
        &mut m,
        &mut gpu,
        &format!("{name}-base"),
        &id_base,
        attempts,
        true,
    );
    let base_events = parse_lines(&baseline);
    assert!(
        base_events.iter().any(|e| e["type"] == "done"),
        "clean turn ends in done: {base_events:?}"
    );

    // After-prefill fault: single attested terminal, no cache store.
    let id_prefill = format!("dense-{attempts}-prefill");
    let cache_empty_before = m.asst_turn_cache.is_empty();
    arm_generation_fault_after_prefill(true);
    let prefill_out = run_generate(
        &mut m,
        &mut gpu,
        &format!("{name}-prefill"),
        &id_prefill,
        attempts + 1,
        false,
    );
    assert_single_attested_error(&prefill_out, &id_prefill, attempts + 1, cache_empty_before, &m);

    // After-first-decode fault: same contract.
    let id_decode = format!("dense-{attempts}-decode");
    let cache_empty_before = m.asst_turn_cache.is_empty();
    arm_generation_fault_after_first_decode(true);
    let decode_out = run_generate(
        &mut m,
        &mut gpu,
        &format!("{name}-decode"),
        &id_decode,
        attempts + 2,
        false,
    );
    assert_single_attested_error(&decode_out, &id_decode, attempts + 2, cache_empty_before, &m);

    // Next request on the rolled-back state matches the fresh turn.
    let id_retry = format!("dense-{attempts}-retry");
    let retry = run_generate(
        &mut m,
        &mut gpu,
        &format!("{name}-retry"),
        &id_retry,
        attempts + 3,
        true,
    );
    assert_envelope_equal(&baseline, &retry);

    // Unload returns VRAM to the warmed baseline...
    hipfire_loader::unload_model(m, &mut gpu).expect("unload dense fixture");
    gpu.drain_pool();
    let free_after_unload = dense_free_vram_bytes(&gpu);
    assert!(
        free_after_unload + DENSE_VRAM_SLACK_BYTES >= free_warmed,
        "{name} leaked VRAM across unload: warmed {free_warmed} -> {free_after_unload}"
    );

    // ...and a fresh-daemon reload reproduces the baseline turn exactly.
    let mut m2 = dense_load(&mut gpu, &path);
    let id_fresh = format!("dense-{attempts}-fresh");
    let fresh = run_generate(
        &mut m2,
        &mut gpu,
        &format!("{name}-fresh"),
        &id_fresh,
        attempts + 4,
        true,
    );
    assert_envelope_equal(&baseline, &fresh);

    hipfire_loader::unload_model(m2, &mut gpu).expect("unload reloaded fixture");
    gpu.drain_pool();
    let free_final = dense_free_vram_bytes(&gpu);
    assert!(
        free_final + DENSE_VRAM_SLACK_BYTES >= free_warmed,
        "{name} leaked VRAM across reload unload: warmed {free_warmed} -> {free_final}"
    );
    batch_clear_all_terminals();
    clear_terminal_control();
    set_active_attempt_id(0);
}

/// LLaMA carrier (G4.10(b) loop): prefill/decode faults roll back and the
/// next request reproduces the fresh-daemon turn.
#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_DENSE_FIXTURE with qwen3-0.6b.hf4 and lfm2.5-1.2b.mq4"]
fn dense_llama_fault_rolls_back_and_retry_matches_fresh() {
    exercise_dense_fixture("qwen3-0.6b.hf4", 71_000);
}

/// LFM2-MoE dense family (G4.10(a) seams): same end-to-end contract.
#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_DENSE_FIXTURE with qwen3-0.6b.hf4 and lfm2.5-1.2b.mq4"]
fn dense_lfm_fault_rolls_back_and_retry_matches_fresh() {
    exercise_dense_fixture("lfm2.5-1.2b.mq4", 72_000);
}

/// Reset-phase failure stays visible and unattested: a rollback that cannot
/// be attested still emits exactly one error carrying the reset context
/// with `rolled_back=false`, never `done`.
#[test]
#[ignore = "requires real HIP GPU + HIPFIRE_DENSE_FIXTURE with qwen3-0.6b.hf4 and lfm2.5-1.2b.mq4"]
fn dense_reset_phase_error_visible_and_unattested() {
    let _guard = lock();
    if dense_fixture("qwen3-0.6b.hf4").is_none() {
        return;
    }
    let Some(mut gpu) = dense_gpu() else {
        return;
    };
    dense_warm_first_alloc(&mut gpu);
    let id = "dense-reset-phase";
    let attempt = 73_000_u64;
    fresh_attempt(id, attempt);
    let mut output = Vec::new();
    {
        let _route =
            hipfire_generate::ar::GenerationRouteScope::enter(GenerationRoute::LlamaAr, id);
        hipfire_generate::ar::emit_generation_start(
            GenerationRoute::LlamaAr,
            &mut output,
            id,
            false,
        );
        // A genuinely unattested epilogue through production constructors:
        // a real device sync attests the drain, but the prior carries a
        // reset-phase failure no rollback can attest away.
        let sync = fail_closed_device_sync(&mut gpu);
        let ep = fail_closed_epilogue_after_sync(
            Err("lfm2_decode_batch.reset: injected reset failure".to_string()),
            sync,
        );
        assert!(!ep.rolled_back, "reset failure must not attest");
        emit_fail_closed_error(
            &mut output,
            Some(id),
            "llama decode (forward_scratch) failed: injected reset failure",
            "internal",
            false,
            &ep,
        );
    }
    let events = parse_lines(&output);
    assert_eq!(
        events.len(),
        2,
        "gen_start + one error, never done: {events:?}"
    );
    assert_eq!(events[1]["type"], "error");
    assert_eq!(events[1]["rolled_back"], false);
    let message = events[1]["message"].as_str().expect("error message");
    assert!(
        message.contains("lfm2_decode_batch.reset: injected reset failure"),
        "reset context must stay visible: {message}"
    );
    assert!(
        !claim_wire_terminal(id, attempt),
        "unattested reset error must claim exactly one terminal"
    );
    batch_clear_all_terminals();
    clear_terminal_control();
    set_active_attempt_id(0);
}
