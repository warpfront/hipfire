// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire engine daemon — JSON lines over stdin/stdout.
//! The Bun CLI spawns this process and communicates via IPC.
//! Usage: daemon (reads JSON from stdin, writes JSON to stdout)
//!
//! Daemons sharing a physical GPU are excluded by machine-wide flock(2) locks
//! keyed on the card's resolved identity (UUID, else PCI address). Daemons on
//! distinct GPUs may coexist, even in one HOME.
//!
//! Protocol:
//!   → {"type":"load","model":"path.hfq","params":{"max_seq":4096}}
//!   ← {"type":"loaded","arch":"qwen3_5","dim":4096,"layers":32,"vocab":248320,"vl":true}
//!   → {"type":"generate","id":"r1","prompt":"Hello","temperature":0.3,"max_tokens":512}
//!   → {"type":"generate","id":"r1","prompt":"Describe this","image":"/path/to/img.png","temperature":0.3,"max_tokens":512}
//!   ← {"type":"gen_start","id":"r1","attempt_id":1,...}
//!   ← {"type":"prefill_progress","id":"r1","attempt_id":1,"done":2048,"total":9000}
//!       Prompt-prefill progress, one line per completed chunk while `done < total`
//!       (none for a single-chunk prompt or at `done == total`; counts exclude a
//!       cache-hit prefix). Ordered after `gen_start` and before the first
//!       `token`/`reasoning`. Default on; `HIPFIRE_PREFILL_PROGRESS=0` suppresses it.
//!       Clients that do not know the type ignore it.
//!   ← {"type":"token","id":"r1","text":"The"}
//!   ← {"type":"done","id":"r1","tokens":42,"tok_s":44.5}
//!   → {"type":"unload"}
//!   ← {"type":"unloaded"}

use base64::Engine;
use hipfire_config::developer_var;
use hipfire_config::devices::{Claim, DeviceSelection, ObservedDevice};
use hipfire_engine::emit::*;
use hipfire_engine::prompt::*;
use hipfire_engine::redline::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_engine::wire_seed::parse_wire_seed;
#[cfg(feature = "serve-fault-inject")]
#[cfg(feature = "serve-fault-inject")]
use hipfire_generate::ar::take_fault_after_prefill;
use hipfire_generate::ar::{
    ckpt_interval, ckpt_max, ckpt_resume_enabled, deepseek4_spec_requested,
    deepseek4_spec_requested_from_policy, emit_qwen_ar_done, generate, llama_prefill_sample_seed,
    model_retry_reset_eligible, qwen_ar_cache_action, qwen_ar_done_value,
    qwen_ar_drain_pending_into_router, qwen_ar_eviction_prefill_chunk_limit, qwen_ar_finish_route,
    qwen_ar_forward_fail_action, qwen_ar_forward_fail_message, qwen_ar_observe_and_route,
    qwen_ar_raw_commit_token, qwen_ar_route_filter_text, qwen_ar_route_think_events,
    reset_core_arch_key, select_generation_route, truncate_checkpoints, write_error,
    GenerationRoute, GenerationRouteInputs, QwenArForwardFailAction, QwenArRawCommitDisposition,
    QwenArRouteFinish, QwenArSemanticProducer, QwenArTerminalCause,
};
use hipfire_generate::batch::{
    attach_qwen_ep_batch_receipt_evidence, drive_lfm_continuous_batch,
    drive_qwen35_ep_continuous_batch, drive_qwen_continuous_batch, emit_uncorrelated_error,
    is_batch_request_eligible, is_qwen_ep_batch_request_eligible,
    lfm_prefill_cancellable_or_fallback,
};
use hipfire_generate::redline::{
    handle_railgun_g0_dflash_cycle, handle_redline_dflash_verify_shadow_pm4,
    handle_redline_dispatch_profile, handle_redline_dspark_shadow_pm4, handle_redline_greedy_trace,
    handle_redline_pm4_prefix_profile, handle_redline_prefix_shadow, handle_redline_probe_aql,
    handle_redline_shadow, redline_append_tensor_slice, redline_deepseek4_snapshot,
    redline_dspark_shadow_block, redline_dspark_verify_guard, redline_dspark_verify_snapshot,
    redline_is_dense_lfm, redline_lfm2moe_snapshot, redline_pm4_prefix_profile_deepseek4,
    redline_prepare_retained_fixture, redline_prime_deepseek4, redline_prime_dspark_shadow_arm,
    redline_prime_qwen, redline_prime_retained_fixture, redline_qwen_debug_hashes,
    redline_qwen_snapshot, redline_reset_deepseek4, redline_reset_lfm2moe, redline_reset_qwen,
    redline_run_deepseek4_decode, redline_run_direct_fixture, redline_run_dspark_capture_arm,
    redline_run_dspark_direct_arm, redline_run_dspark_replay_arm, redline_shadow_deepseek4,
    redline_shadow_dspark_verify_pm4, redline_snapshot, RedlineDeepseek4Snapshot, RedlineDsparkArm,
    RedlineDsparkReplayArm, RedlineDsparkVerifySnapshot, RedlineLfm2MoeSnapshot,
    RedlineQwenSnapshot, RedlineSnapshot,
};
use hipfire_runtime::emit_text::{
    currently_in_think, extract_tool_calls_from_text, ThinkOutputRouter, ThinkRouteEvent,
    ToolOutputRouter, ToolRouteError, ToolRouteEvent,
};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::llama;
use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::sampler::{self, SamplerConfig};
use hipfire_runtime::spec::accept_greedy_prefix;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;
mod bench;
mod emu_daemon;
use emu_daemon::{gpu_memory_text, validate};
mod gpu_lock;
use gpu_lock::GpuLocks;
mod effort_probe;
mod load_summary;
mod npu_route;
mod process_init;
mod request_guards;
mod slots;
mod vision_ladder;
mod weight_host_tenant;
use vision_ladder::{apply_vision_mode_gate, resolve_vision_ladder};
#[cfg(test)]
pub(crate) static TERMINAL_TEST_LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(()));
use hipfire_generate::vision::{GenerateVLParams, ImageSource};
use hipfire_loader::{AsstTurnCache, Eviction, LoadedModel};
use hipfire_runtime::spec::{
    ClientEvent, EmitOutcome, EvictRetain, FinishSummary, PrefillOutcome, SpecAdvance, SpecEmit,
    SpecTarget, Speculator, StopReason,
};

/// Formats the independent Qwen decode-batch path can actually execute.
/// Must stay aligned with `lm_head_batched` + `prepare_decode_batch_inputs`
/// in hipfire-arch-qwen35 — unsupported lm_head or F32 embedding must never
/// advertise `continuous_batch_capable` or enter the batch route.
pub type CaskConfig = hipfire_runtime::loader_api::CaskConfig;
#[allow(dead_code)]
fn emit_error_no_id(stdout: &mut impl std::io::Write, message: impl std::fmt::Display) {
    emit_uncorrelated_error(stdout, None, &message.to_string(), "internal", false, false);
}

/// The two receipt lines for one load, in print order.
fn load_phase_lines(line: &str, json: &serde_json::Value) -> [String; 2] {
    [
        format!("hipfire load phases: {line}"),
        format!("hipfire.load_phases.v1 {json}"),
    ]
}

/// Finish the load receipt and print it (stderr, two lines) right before a
/// successful `loaded` ACK. Measurement only; the ACK JSON is unchanged.
fn emit_load_phases() {
    let (line, json) = rdna_compute::load_phases::finish_load();
    for text in load_phase_lines(&line, &json) {
        eprintln!("{text}");
    }
}

/// Retire the reader-side batch admission on every generate exit. Batch
/// drivers clear entries themselves; the guard is a no-op in that case, while
/// singleton/image/error paths cannot leave a reusable key stuck announced.
struct BatchTerminalCleanup {
    id: String,
    attempt_id: u64,
    admission: Option<BatchGeneration>,
}
impl Drop for BatchTerminalCleanup {
    fn drop(&mut self) {
        if let Some(admission) = self.admission {
            batch_clear_terminal_at_generation(&self.id, self.attempt_id, admission);
        }
    }
}

/// Emit a terminal error for a generate key announced by the reader.
///
/// Admission failures happen before the singleton terminal transaction is
/// activated, so bind the existing keyed registry entry through
/// [`BatchAttemptScope`] and let the normal active-attempt writer claim it.
/// Retire the key only after the writer has claimed and emitted the error.
fn emit_batch_admission_error(
    stdout: &mut impl std::io::Write,
    id: &str,
    attempt_id: u64,
    admission: BatchGeneration,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    {
        let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
        emit_active_attempt_error(stdout, Some(id), message, class, retryable, rolled_back);
        let _ = stdout.flush();
    }
    batch_clear_terminal_at_generation(id, attempt_id, admission);
}
/// Parse attempt_id from a JSON number only (u64 or non-neg i64).
/// Decimal strings are rejected — no further coercion.
fn parse_wire_attempt_id(value: Option<&serde_json::Value>) -> Option<u64> {
    let value = value?;
    if let Some(n) = value.as_u64() {
        return Some(n);
    }
    if let Some(n) = value.as_i64() {
        if n >= 0 {
            return Some(n as u64);
        }
    }
    None
}
/// Require a present, nonzero numeric attempt_id on the wire.
///
/// Zero is reserved for uncorrelated control-plane envelopes. It must never
/// enter singleton or continuous-batch generation ownership.
fn require_wire_attempt_id(value: Option<&serde_json::Value>) -> Result<u64, &'static str> {
    match value {
        None => Err("missing attempt_id"),
        Some(_) => match parse_wire_attempt_id(value) {
            None => Err("malformed attempt_id"),
            Some(0) => Err("attempt_id must be nonzero"),
            Some(id) => Ok(id),
        },
    }
}
/// Announce a generate key only after the command has a valid nonzero attempt.
///
/// `None` means the request is malformed or uses reserved attempt zero and
/// must continue to main for one uncorrelated validation error. `Some((..., None))`
/// is a duplicate live key; `Some((..., Some(token)))` owns a fresh admission.
fn announce_generate_terminal(
    msg: &serde_json::Value,
) -> Option<(&str, u64, Option<BatchGeneration>)> {
    if msg.get("type").and_then(|v| v.as_str()) != Some("generate") {
        return None;
    }
    let id = msg.get("id").and_then(|v| v.as_str())?;
    let attempt_id = parse_wire_attempt_id(msg.get("attempt_id"))?;
    if attempt_id == 0 {
        return None;
    }
    Some((id, attempt_id, batch_announce_terminal(id, attempt_id)))
}

// ── serve-fault-inject (test-only; compiled out of production) ─────────
// Per-request fault arming lives in hipfire_generate::common::RequestFaultGuard.
#[cfg(feature = "serve-fault-inject")]
fn write_test_state_snapshot(
    stdout: &mut impl std::io::Write,
    m: Option<&LoadedModel>,
    gpu: &rdna_compute::Gpu,
    state_epoch: u64,
) {
    let (
        arch,
        eligible,
        seq_pos,
        conversation_len,
        kv_hash,
        kv_bytes,
        recurrent_hash,
        recurrent_bytes,
        drafter_reset,
        checkpoint_empty,
        adaptive_clean,
        asst_cache_empty,
        prefix_cache_clean,
    ) = match m {
        Some(m) => {
            let arch = reset_core_arch_key(m.arch_id);
            let eligible: Vec<&'static str> =
                hipfire_runtime::reset_core::fault_inject_eligible_routes(arch).to_vec();
            let (kv_hash, kv_bytes, recurrent_hash, recurrent_bytes) = match m.qwen35() {
                Some(bundle) => match redline_qwen_snapshot(gpu, bundle) {
                    Ok(snap) => (
                        format!("{:016x}", redline_hash(&snap.kv)),
                        snap.kv.len(),
                        format!("{:016x}", redline_hash(&snap.recurrent)),
                        snap.recurrent.len(),
                    ),
                    Err(_) => (
                        "unavailable".to_string(),
                        0usize,
                        "unavailable".to_string(),
                        0usize,
                    ),
                },
                _ => (
                    "unavailable".to_string(),
                    0usize,
                    "unavailable".to_string(),
                    0usize,
                ),
            };
            // Live Speculator evidence only. Missing evidence fail-closes dirty
            // — never invent clean from vestigial m.prefill/dflash_checkpoints.
            let (drafter_reset, checkpoint_empty) = match m.speculator.as_ref() {
                Some(s) => match s.reset_state_evidence() {
                    Some(ev) => (ev.drafter_reset, ev.checkpoint_empty),
                    None => (false, false),
                },
                // No live drafter ⇒ drafter residual N/A / clean; host rings still
                // report empty via prefill/dflash free on rollback.
                None => (
                    true,
                    m.prefill_checkpoints.is_empty() && m.dflash_checkpoints.is_empty(),
                ),
            };
            let adaptive_clean = m
                .kv_adaptive
                .as_ref()
                .map(|ad| !ad.is_poisoned())
                .unwrap_or(true);
            let asst_cache_empty = m.asst_turn_cache.is_empty();
            // Prefix-cache residual for qwen is conversation_tokens + asst
            // turn cache; clean when both empty (fresh/reset).
            let prefix_cache_clean = m.conversation_tokens.is_empty() && asst_cache_empty;
            (
                arch,
                eligible,
                m.seq_pos,
                m.conversation_tokens.len(),
                kv_hash,
                kv_bytes,
                recurrent_hash,
                recurrent_bytes,
                drafter_reset,
                checkpoint_empty,
                adaptive_clean,
                asst_cache_empty,
                prefix_cache_clean,
            )
        }
        None => (
            "none",
            Vec::new(),
            0usize,
            0usize,
            "unavailable".to_string(),
            0usize,
            "unavailable".to_string(),
            0usize,
            true,
            true,
            true,
            true,
            true,
        ),
    };

    let graph_clean = gpu.graphs.captured_graph.is_none()
        && gpu.graphs.graph_exec.is_none()
        && gpu.graphs.verify_graph_count() == 0
        && gpu.graphs.replay_graph_count() == 0;
    let obs = gpu.replay.replay_observation();
    let replay_clean = !obs.failed && obs.count == 0;

    let payload = serde_json::json!({
        "type": "test_state_snapshot",
        "schema_version": 1,
        "arch": arch,
        "eligible_routes": eligible,
        "state_epoch": state_epoch,
        "seq_pos": seq_pos,
        "conversation_len": conversation_len,
        "kv_hash": kv_hash,
        "kv_bytes": kv_bytes,
        "recurrent_hash": recurrent_hash,
        "recurrent_bytes": recurrent_bytes,
        "graph_clean": graph_clean,
        "replay_clean": replay_clean,
        "drafter_reset": drafter_reset,
        "checkpoint_empty": checkpoint_empty,
        "adaptive_clean": adaptive_clean,
        "asst_cache_empty": asst_cache_empty,
        "prefix_cache_clean": prefix_cache_clean,
    });
    let _ = writeln!(stdout, "{}", payload);
    let _ = stdout.flush();
}

/// Pure `gen_start.contract_version` selection used by the live generate path.
/// Qwen AR (5/6) and Muse Glimmer (14) advertise v2; DS4 (9) and every other
/// arch stay unset.
/// Muse Glimmer already emits the v2-shaped two-phase terminal
/// (`commit_ready` -> `commit` -> byte-identical `done`), and its tool calls
/// are staged as canonical `calls` on that terminal. Only the v2 fold reads
/// them: the legacy path builds tool calls solely from mid-stream `tool_calls`
/// events, which Glimmer does not emit, so on legacy a tool turn arrived with
/// `finish_reason=tool_calls` and an empty payload.
const GLIMMER_SEMANTIC_CONTRACT_VERSION: u32 = 2;

/// Production Malformed error envelope for Qwen DFlash epilogue + tests.
fn qwen_dflash_malformed_error_value(
    id: &str,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
    attempt_id: u64,
) -> serde_json::Value {
    serde_json::json!({
        "type": "error",
        "id": id,
        "message": message,
        "class": class,
        "retryable": retryable,
        "rolled_back": rolled_back,
        "attempt_id": attempt_id,
    })
}

#[allow(dead_code)]
fn gpu_block_attractor_token(
    gpu: &rdna_compute::Gpu,
    logits_buf: &hip_bridge::DeviceBuffer,
    history: &[u32],
    tok_id: u32,
    window: usize,
    threshold: usize,
) {
    if window == 0 || threshold == 0 {
        return;
    }
    let start = history.len().saturating_sub(window);
    let count = history[start..].iter().filter(|&&t| t == tok_id).count();
    if count >= threshold {
        let bytes: [u8; 4] = f32::NEG_INFINITY.to_ne_bytes();
        let _ = gpu
            .hip
            .memcpy_htod_offset(logits_buf, (tok_id as usize) * 4, &bytes);
    }
}
/// Query what HIP exposes: one entry per logical device. The UUID is read
/// only where identity comes from HIP rather than the KFD topology.
fn observe_hip_devices(with_uuid: bool) -> Result<Vec<ObservedDevice>, String> {
    let hip = hip_bridge::HipRuntime::load().map_err(|e| e.to_string())?;
    let count = hip.device_count().map_err(|e| e.to_string())?;
    if count < 1 {
        return Err("no visible GPUs to reserve".into());
    }
    (0..count)
        .map(|id| {
            Ok(ObservedDevice {
                logical: id as usize,
                arch: hip.get_arch(id).map_err(|e| e.to_string())?,
                pci_bus_id: hip.device_pci_bus_id(id).map_err(|e| e.to_string())?,
                uuid: with_uuid
                    .then(|| hip.device_uuid(id).map_err(|e| e.to_string()))
                    .transpose()?,
            })
        })
        .collect()
}

/// Complete the GPU reservation after visibility is installed and check the
/// identity of every logical device HIP now exposes.
///
/// `hardware.devices` already reserved its cards while resolving; here each
/// logical device must be exactly the resolved card (arch and PCI bus ID).
/// Without it (raw inherited filters or none), the HIP-visible cards are
/// identified — by PCI address in the KFD topology on native Linux, from HIP
/// UUID and PCI address on WSL2/ROCDXG and native Windows — and reserved
/// all-or-nothing.
fn reserve_and_verify_gpus(locks: &mut GpuLocks) -> Result<(), String> {
    use hipfire_config::devices::{DeviceRoot, IdentitySource};
    let root = DeviceRoot::system();
    let source = root.identity_source();
    let observed = observe_hip_devices(source == IdentitySource::Hip)?;
    if let Some(active) = hipfire_config::devices::active_devices() {
        active.verify_visible(&observed)?;
        for (seen, device) in observed.iter().zip(&active.devices) {
            eprintln!(
                "[devices] verified logical {}: HIP {} {} = index {} {} rocr={}",
                seen.logical,
                seen.arch,
                seen.pci_bus_id,
                device.index,
                device.lock_identity(),
                device.rocr_selector()
            );
        }
    } else {
        let devices = hipfire_config::devices::identify_observed(&root, &observed)?;
        locks.claim_all(&devices)?;
        let source = match source {
            IdentitySource::Kfd => "",
            IdentitySource::Hip => " (identity from HIP: no KFD topology)",
        };
        for (seen, device) in observed.iter().zip(&devices) {
            eprintln!(
                "[devices] logical {}: HIP {} {} = index {} {}{source}",
                seen.logical,
                seen.arch,
                seen.pci_bus_id,
                device.index,
                device.lock_identity()
            );
        }
    }
    locks.write_pid_file()
}

/// Cap on the *encoded* base64 string length the daemon will accept on the
/// IPC. ~40 MB encoded → ~30 MB raw image bytes (4/3 expansion).
const MAX_BASE64_ENCODED_LEN: usize = 40 * 1024 * 1024;

/// hunt3 H-D: upper bound on a request-driven `max_seq` (1M). A defense-in-
/// depth clamp only — it caps an unvalidated 10M `max_seq` that would otherwise
/// drive a multi-GB KV allocation and OOM the daemon at load. It is NOT a
/// VRAM-aware guard: a load that requests exactly this on a non-eviction config
/// can still OOM at allocation; that VRAM validation is out of scope here.
const MAX_REQUESTED_SEQ: usize = 1024 * 1024;

/// #666 G1: lower bound on a request-driven `max_seq`, symmetric with
/// `MAX_REQUESTED_SEQ`. A resolved `max_seq` of 0 reaches the
/// `saddle-core/src/kv.rs` `physical_cap (0) must be in (0, max_seq_len=0]`
/// assertion, which panics and kills the daemon, taking the resident model
/// with it. The `load` arm refuses 0 fail-closed before any teardown or KV
/// allocation; the assertion stays as a last-resort invariant.
const MIN_REQUESTED_SEQ: usize = 1;

/// Typed active-attempt error writer used by generation failure paths and tests.
fn write_typed_error(
    stdout: &mut impl std::io::Write,
    id: &str,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    hipfire_generate::dense::emit_active_attempt_error(
        stdout,
        Some(id),
        message,
        class,
        retryable,
        rolled_back,
    );
}

/// The request's wire `seed`, or `None` after refusing an out-of-domain one
/// (negative, fractional, non-numeric) with a typed `validation` error — a
/// client fault, never an `internal` one.
/// Report a batch drive result; a poisoned (unattested) drive retires the
/// scheduler until unload/reload. Checked teardown: reset_all was already
/// attempted in fail_all.
fn apply_batch_drive_result(
    res: Result<(), BatchDriveError>,
    tag: &str,
    batch_scheduler: &mut Option<ContinuousBatchScheduler>,
    continuous_batch_size: &mut usize,
    batch_poisoned: &mut Option<String>,
) {
    match res {
        Ok(()) => {}
        Err(BatchDriveError::Gpu(e)) => eprintln!("{tag} drive failed (attested): {e}"),
        Err(BatchDriveError::Poisoned(e)) => {
            eprintln!(
                "{tag} drive poisoned (unattested): {e} — generation poisoned until unload/reload"
            );
            *batch_scheduler = None;
            *continuous_batch_size = 1;
            *batch_poisoned = Some(e);
            batch_clear_all_terminals();
        }
    }
}

fn client_seed_or_refuse(
    stdout: &mut impl std::io::Write,
    id: &str,
    seed: Option<&serde_json::Value>,
) -> Option<Option<u64>> {
    match parse_wire_seed(seed) {
        Ok(seed) => Some(seed),
        Err(reason) => {
            write_typed_error(stdout, id, &reason, "validation", false, false);
            None
        }
    }
}

/// Which tower a media-carrying request needs, for the refusal text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MediaTower {
    Vision,
    Audio,
}

impl MediaTower {
    fn noun(self) -> &'static str {
        match self {
            MediaTower::Vision => "vision",
            MediaTower::Audio => "audio",
        }
    }
}

/// Refuse media sent to a model with no encoder for it: the client asked for a
/// capability this load lacks (`unsupported`), not a server fault. The tower is
/// named because the two are separate artifacts on paper and one load can have
/// either (arch 18 keeps both in the one `.vl` sidecar).
fn refuse_media_without_tower(
    stdout: &mut impl std::io::Write,
    id: &str,
    tower: MediaTower,
    vision_gated_off: Option<&str>,
) {
    let message = media_refusal_message(tower, vision_gated_off);
    write_typed_error(stdout, id, &message, "unsupported", false, false);
}

/// The refusal text, split out from the writer so the tower noun and the knob
/// that dropped the sidecar are directly testable.
fn media_refusal_message(tower: MediaTower, vision_gated_off: Option<&str>) -> String {
    let noun = tower.noun();
    match vision_gated_off {
        Some(sidecar) => format!(
            "model has no {noun} encoder loaded: vision_mode is off and the tower sidecar {sidecar} was skipped; run `hipfire config set vision_mode auto` (or `resident`/`on`) and reload"
        ),
        None => format!("model has no {noun} encoder"),
    }
}

/// The vision-only spelling the generate route uses.
fn refuse_image_without_vision(
    stdout: &mut impl std::io::Write,
    id: &str,
    vision_gated_off: Option<&str>,
) {
    refuse_media_without_tower(stdout, id, MediaTower::Vision, vision_gated_off);
}

/// Which tower an `embed` request asks for — the wire's `images_base64` /
/// `audios_base64` / `videos_base64` lists, or the one-image `image_base64`
/// legacy field. Vision wins when a request carries both, so a tower-less
/// load refuses the input the caller listed first.
///
/// A video asks for the VISION tower: arch 18's video leg has no tower of its
/// own, it runs the image tower once per frame.
///
/// Wire shape only: this arm must not name an arch type, so it answers "did
/// the caller ask for a tower", not "is a tower resident" (the encode body in
/// `hipfire_generate::embed` owns that, and decodes the bytes).
fn embed_request_media(msg: &serde_json::Value) -> Option<MediaTower> {
    let input = msg.get("input").and_then(|v| v.as_object())?;
    let list_present = |key: &str| {
        input
            .get(key)
            .and_then(|v| v.as_array())
            .is_some_and(|media| !media.is_empty())
    };
    if list_present("images_base64") || input.get("image_base64").is_some_and(|v| !v.is_null()) {
        return Some(MediaTower::Vision);
    }
    if list_present("audios_base64") {
        return Some(MediaTower::Audio);
    }
    if list_present("videos_base64") {
        return Some(MediaTower::Vision);
    }
    None
}

/// Resident KV backend metadata for `loaded` ACK and `diag`.
/// Set only on successful publish; cleared on unload / empty handoff.
#[derive(Clone, Debug)]
struct ResidentKvDiag {
    request: String,
    backend: String,
    reason: Option<String>,
    legacy: bool,
    warning: Option<String>,
    mode: Option<String>,
}

impl ResidentKvDiag {
    fn from_admission(
        request: hipfire_loader::admission::KvBackendRequest,
        backend: hipfire_runtime::kv_backend::KvBackend,
        automatic_reason: Option<&str>,
    ) -> Self {
        use hipfire_loader::admission::KvBackendRequest;
        use hipfire_runtime::kv_backend::KvBackend;
        let request_str = match request {
            KvBackendRequest::Automatic => "automatic",
            KvBackendRequest::Explicit(_) => "explicit",
        };
        let legacy = matches!(backend, KvBackend::Legacy);
        let warn_reason = if legacy && matches!(request, KvBackendRequest::Automatic) {
            automatic_reason
        } else {
            None
        };
        let warning = hipfire_loader::admission::legacy_warning(backend, warn_reason);
        if let Some(w) = &warning {
            eprintln!("{w}");
        }
        let reason = if legacy {
            match request {
                KvBackendRequest::Automatic => automatic_reason.map(str::to_string),
                KvBackendRequest::Explicit(_) => Some("explicit operator choice".to_string()),
            }
        } else {
            None
        };
        Self {
            request: request_str.to_string(),
            backend: backend.as_str().to_string(),
            reason,
            legacy,
            warning,
            mode: None,
        }
    }

    /// Fixed Q8 slot arena always admits legacy storage.
    fn for_slot(explicit: bool) -> Self {
        use hipfire_loader::admission::KvBackendRequest;
        use hipfire_runtime::kv_backend::KvBackend;
        let request = if explicit {
            KvBackendRequest::Explicit(KvBackend::Legacy)
        } else {
            KvBackendRequest::Automatic
        };
        let automatic_reason = if explicit {
            None
        } else {
            Some("fixed Q8 slot arena has no VMM owner")
        };
        Self::from_admission(request, KvBackend::Legacy, automatic_reason)
    }
}

/// Pure gate for the deferred EP (tp>1) load handoff.
///
/// After a new EP model is constructed, the prior model is unloaded. The new
/// model may be published only when that prior unload succeeds (or there was
/// no prior model — caller passes `Ok(())` in that case). A failed prior
/// unload must never install/emit `loaded` for the new model.
fn ep_deferred_may_publish(prior_unload: &Result<(), String>) -> bool {
    prior_unload.is_ok()
}

/// Hard-error text when deferred prior unload fails (and optional new-model
/// rollback also fails). Always names the prior failure; appends rollback
/// failure when present so neither is log-and-ignored.
fn ep_deferred_handoff_error_message(prior_err: &str, rollback_err: Option<&str>) -> String {
    match rollback_err {
        None => format!("prior unload failed: {prior_err}"),
        Some(rb) => {
            format!("prior unload failed: {prior_err}; new-model rollback also failed: {rb}")
        }
    }
}

/// Whether the deferred-EP load path must run `ensure_vmm_ready_for_load`
/// before constructing a new EP model.
///
/// Only when there is no live prior model (`model_present == false`): a
/// failed deferred prior unload leaves `model=None` while VMM arenas may
/// still be pending. Do NOT preflight when a live deferred prior still
/// occupies `model` — that path tears down after successful new load.
fn ep_deferred_needs_vmm_preflight(load_tp: usize, model_present: bool) -> bool {
    load_tp > 1 && !model_present
}

/// Suffix for an admitted-load failure. Single-device/pp loads are
/// unload-first, so after admission the prior model is already retired and a
/// construction or staging failure leaves `model=None`; say so explicitly.
/// A deferred tp>1 load keeps its prior model and gets no suffix.
fn no_model_loaded_suffix(model_present: bool) -> &'static str {
    if model_present {
        ""
    } else {
        "; no model loaded (the prior model was unloaded before this load)"
    }
}

/// Print a friendly, user-actionable message when Gpu::init fails. Matches
/// the panic shape we used to emit (which dumped a Rust backtrace and the
/// raw HipError debug-format) but turns it into a concrete next-step list.
/// The most common cause on Windows (#112) is HIP SDK present but no
/// AMD GPU driver visible to the runtime; on Linux it is usually missing
/// `libamdhip64.so` or kernel-side amdgpu / kfd not loaded.
fn report_gpu_init_failure(err: &hip_bridge::HipError) {
    eprintln!();
    eprintln!("hipfire: failed to initialize GPU runtime.");
    eprintln!("  HIP error: {} (code {})", err.message, err.code);
    eprintln!();
    if cfg!(target_os = "windows") {
        eprintln!("  Most common Windows cause: HIP SDK is loaded but no");
        eprintln!("  AMD GPU is visible to the runtime. Verify:");
        eprintln!("    1. AMD Adrenalin driver is installed and current.");
        eprintln!("    2. AMD HIP SDK 6.2 or newer is installed:");
        eprintln!("       https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html");
        eprintln!("    3. `amdhip64.dll` is reachable (HIP_PATH set or DLL on PATH).");
        eprintln!("    4. Reboot after driver / SDK install if you have not yet.");
    } else {
        eprintln!("  Most common Linux causes:");
        eprintln!("    1. amdgpu kernel module not loaded (check `lsmod | grep amdgpu`).");
        eprintln!("    2. /dev/kfd missing or not readable by the current user");
        eprintln!("       (add to the `render` group; reboot).");
        eprintln!("    3. ROCm not installed or libamdhip64.so missing");
        eprintln!("       (check `ldconfig -p | grep amdhip64`).");
    }
    eprintln!();
    eprintln!("  Run `hipfire diag` for a full environment report.");
}

/// Validate and install process policy, resolving `hardware.devices` against
/// the KFD topology before any GPU runtime initializes. `claim` reserves each
/// resolved card as part of that same step.
fn install_process_config(
    config: hipfire_config::ProcessConfig,
    claim: &mut hipfire_config::devices::ClaimFn<'_>,
) -> Result<(), String> {
    config.validate().map_err(|error| error.to_string())?;
    let selection = hipfire_config::devices::apply_device_visibility(&config, claim)
        .map_err(|error| error.to_string())?;
    match &selection {
        DeviceSelection::Resolved { devices, visibility } => eprintln!(
            "[devices] hardware.devices={:?} -> {} (ROCR_VISIBLE_DEVICES={} HIP_VISIBLE_DEVICES={})",
            config.legacy_value("HIPFIRE_DEVICES").unwrap_or_default(),
            devices
                .iter()
                .map(|device| format!("index {} {} {} {}", device.index, device.arch, device.lock_identity(), device.bdf))
                .collect::<Vec<_>>()
                .join(", "),
            visibility.rocr,
            visibility.hip
        ),
        DeviceSelection::Inherited(visibility) => eprintln!(
            "[devices] inherited ROCR_VISIBLE_DEVICES={} HIP_VISIBLE_DEVICES={}",
            visibility.rocr, visibility.hip
        ),
        DeviceSelection::Unfiltered => {}
    }
    let runtime = hipfire_runtime::config::RuntimeConfig::from_process_config(&config);
    hipfire_config::install_process_config(config)
        .map_err(|_| "process configuration was already initialized".to_owned())?;
    hipfire_runtime::config::init_with(runtime)
        .map_err(|_| "runtime process configuration was already initialized".to_owned())
}

/// Read the first protocol message before GPU initialization. Native clients
/// send `configure`; older/direct clients may send `load`, in which case the
/// daemon resolves local TOML plus compatibility env itself and preserves the
/// load as the first regular command. Also returns the model the client
/// starts the daemon for, if it names one: `configure`'s optional `model`, or
/// the startup `load`'s.
fn receive_startup_config(
    stdout: &mut impl Write,
) -> Result<
    Option<(
        hipfire_config::ProcessConfig,
        Option<DaemonMsg>,
        bool,
        Option<std::path::PathBuf>,
    )>,
    String,
> {
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        if lock
            .read_line(&mut line)
            .map_err(|error| error.to_string())?
            == 0
        {
            return Ok(None);
        }
        if line.trim().is_empty() {
            continue;
        }
        let msg = match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(msg) => msg,
            Err(error) => {
                emit_uncorrelated_error(
                    stdout,
                    None,
                    &format!("invalid JSON: {error}"),
                    "validation",
                    false,
                    false,
                );
                stdout.flush().map_err(|error| error.to_string())?;
                continue;
            }
        };
        let model = msg
            .get("model")
            .and_then(|value| value.as_str())
            .map(std::path::PathBuf::from);
        if msg.get("type").and_then(|value| value.as_str()) == Some("configure") {
            let config = serde_json::from_value::<hipfire_config::ProcessConfig>(
                msg.get("config")
                    .cloned()
                    .ok_or_else(|| "configure message is missing config".to_owned())?,
            )
            .map_err(|error| format!("invalid process configuration: {error}"))?;
            config.validate().map_err(|error| error.to_string())?;
            return Ok(Some((config, None, true, model)));
        }
        let model =
            model.filter(|_| msg.get("type").and_then(|value| value.as_str()) == Some("load"));
        let config =
            hipfire_config::load_local_process_config().map_err(|error| error.to_string())?;
        let pending = match announce_generate_terminal(&msg) {
            Some((id, attempt_id, Some(admission))) => {
                tracing::debug!(
                    request_id = id,
                    attempt_id,
                    "announced startup generate request"
                );
                DaemonMsg::RegularWithAdmission(msg, admission)
            }
            Some((id, attempt_id, None)) => {
                eprintln!(
                    "[batch] duplicate startup generate dropped id={} attempt_id={}; preserving live registry",
                    id, attempt_id
                );
                continue;
            }
            None => DaemonMsg::Regular(msg),
        };
        return Ok(Some((config, Some(pending), false, model)));
    }
}

fn main() {
    process_init::init();
    tracing::info!(pid = std::process::id(), "daemon starting");

    let args = npu_route::args_or_exit_on_version();

    // --precompile packages the exact-source registry beside this executable.
    // The installed index, not a historical model/format hand list, defines
    // which objects can be used without a device compiler.
    if args.iter().any(|a| a == "--precompile") {
        let process_config = hipfire_config::load_local_process_config().unwrap_or_else(|error| {
            eprintln!("FATAL: invalid process configuration: {error}");
            std::process::exit(1);
        });
        // Packaging needs only the arch and reserves no card.
        install_process_config(process_config, &mut |_| Ok(Claim::Claimed)).unwrap_or_else(
            |error| {
                eprintln!("FATAL: failed to install process configuration: {error}");
                std::process::exit(1);
            },
        );
        // A separate --module probe below exercises the real GPU launch. Normal
        // packaging needs the active architecture but no writable cold directory
        // during Gpu::init: pack_to publishes to the executable-neighbor path.
        let mut gpu = match rdna_compute::Gpu::init() {
            Ok(g) => g,
            Err(e) => {
                report_gpu_init_failure(&e);
                std::process::exit(1);
            }
        };
        // Exercise the production Gpu::rmsnorm_f32 route, whose installed
        // module key is rmsnorm_f32 (the batched route uses rmsnorm).
        if let Some(position) = args.iter().position(|arg| arg == "--module") {
            if args.get(position + 1).map(String::as_str) != Some("rmsnorm_f32") {
                eprintln!("ERROR: --precompile --module currently supports only rmsnorm_f32");
                std::process::exit(2);
            }
            let result = (|| -> rdna_compute::HipResult<Vec<f32>> {
                let input = gpu.upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8])?;
                let weight = gpu.upload_f32(&[1.0, 0.5, 1.5, 2.0, 0.25, 0.75, 1.0, 1.25], &[8])?;
                let output = gpu.zeros(&[8], rdna_compute::DType::F32)?;
                gpu.rmsnorm_f32(&input, &weight, &output, 1e-5)?;
                let values = gpu.download_f32(&output)?;
                gpu.free_tensor(input)?;
                gpu.free_tensor(weight)?;
                gpu.free_tensor(output)?;
                Ok(values)
            })()
            .unwrap_or_else(|error| {
                eprintln!("ERROR: rmsnorm_f32 execution failed: {error}");
                std::process::exit(1);
            });
            let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let weight = [1.0f32, 0.5, 1.5, 2.0, 0.25, 0.75, 1.0, 1.25];
            let scale = (input.iter().map(|v| v * v).sum::<f32>() / 8.0 + 1e-5)
                .sqrt()
                .recip();
            if result
                .iter()
                .zip(input.iter().zip(weight))
                .any(|(&got, (&x, w))| (got - x * w * scale).abs() > 1e-5)
            {
                eprintln!("ERROR: rmsnorm_f32 numerical output differs from reference: {result:?}");
                std::process::exit(1);
            }
            eprintln!(
                "precompile: rmsnorm_f32 execution succeeded on {}: output_bits={:?}",
                gpu.arch,
                result
                    .iter()
                    .map(|v| format!("{:08x}", v.to_bits()))
                    .collect::<Vec<_>>()
            );
            return;
        }
        let output = std::env::current_exe()
            .expect("cannot resolve daemon executable")
            .parent()
            .expect("daemon executable has no parent")
            .join("kernels")
            .join("compiled")
            .join(&gpu.arch);
        let extra_flags =
            rdna_compute::FeatureFlags::from_active_config(&gpu.arch).hipcc_extra_flags;
        let entries = rdna_compute::kernel_registry::entries(&gpu.arch, &extra_flags)
            .unwrap_or_else(|error| {
                eprintln!(
                    "ERROR: no indexed kernel registry for {}: {error:?}",
                    gpu.arch
                );
                std::process::exit(1);
            });
        let count = entries.len();
        let mut compiler = rdna_compute::KernelCompiler::new(&gpu.arch, extra_flags)
            .unwrap_or_else(|error| {
                eprintln!("ERROR: kernel packer initialization failed: {error}");
                std::process::exit(1);
            });
        for entry in entries {
            compiler
                .pack_to(
                    entry.module,
                    entry.source(),
                    &entry
                        .symbols
                        .iter()
                        .map(|symbol| (*symbol).to_owned())
                        .collect::<Vec<_>>(),
                    &output,
                )
                .unwrap_or_else(|error| {
                    eprintln!("ERROR: packaging {} failed: {error}", entry.module);
                    std::process::exit(1);
                });
        }
        eprintln!(
            "precompile: packaged {count} indexed modules for {}",
            gpu.arch
        );
        return;
    }

    // Resolve hardware.devices, reserve the cards it names and install ROCr/HIP
    // visibility in one step, then check what HIP exposes before any GPU
    // context is created.

    let mut stdout = std::io::stdout();
    let Some((process_config, pending_message, acknowledge_config, startup_model)) =
        receive_startup_config(&mut stdout).unwrap_or_else(|error| {
            eprintln!("FATAL: failed to resolve startup configuration: {error}");
            std::process::exit(1);
        })
    else {
        return;
    };
    let (weight_host, mut gpu_locks) = weight_host_tenant::open_locks().unwrap_or_else(|error| {
        eprintln!("FATAL: {error}");
        std::process::exit(1);
    });
    install_process_config(process_config, &mut |device| gpu_locks.try_claim(device))
        .unwrap_or_else(|error| {
            eprintln!("FATAL: failed to install process configuration: {error}");
            std::process::exit(1);
        });
    // The model the client starts this daemon for decides the host-memory
    // switches the HIP runtime reads once, when it first loads just below.
    if let Some(model) = &startup_model {
        hipfire_loader::prepare_host_memory_for(model);
    }
    reserve_and_verify_gpus(&mut gpu_locks).unwrap_or_else(|error| {
        eprintln!("FATAL: {error}");
        std::process::exit(1);
    });
    // Uncertified driver models (WSL2/ROCDXG, native Windows) refuse an
    // explicit Redline backend outright; the per-load default falls back to
    // the HIP graph instead (load handler).
    if let Some(refusal) = hipfire_config::devices::redline_platform_refusal() {
        let backend = hipfire_config::process_value("HIPFIRE_REPLAY_BACKEND")
            .unwrap_or_default()
            .to_ascii_lowercase();
        if matches!(backend.as_str(), "redline" | "shadow") {
            eprintln!("FATAL: replay.backend={backend} refused: {refusal}");
            std::process::exit(1);
        }
    }
    if acknowledge_config {
        writeln!(
            stdout,
            r#"{{"type":"configured","schema_version":{}}}"#,
            hipfire_config::CONFIG_SCHEMA_VERSION
        )
        .unwrap_or_else(|error| {
            eprintln!("FATAL: failed to acknowledge process configuration: {error}");
            std::process::exit(1);
        });
        stdout.flush().unwrap_or_else(|error| {
            eprintln!("FATAL: failed to flush process configuration acknowledgement: {error}");
            std::process::exit(1);
        });
    }

    let mut gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            report_gpu_init_failure(&e);
            std::process::exit(1);
        }
    };
    let mut model: Option<LoadedModel> = None;
    // Monotonic cold-reset epoch; bumped only after successful synchronized reset.
    // Echoed on reset acks so Engine::reset can reject non-increasing epochs.
    let mut state_epoch: u64 = 0;
    // PFlash speculative-prefill state. None unless the load message
    // includes a `prefill_drafter` path AND `prefill_compression` != "off".
    // Lives alongside `model` so unload_model + this state are paired
    // teardowns.
    let mut pflash_state: Option<hipfire_pflash::pflash::PflashState> = None;
    // Set when the most recent load had a tower sidecar available but
    // `vision_mode=off` skipped it, so an image request can name the knob
    // instead of claiming the model has no vision encoder at all.
    let mut vision_gated_off: Option<String> = None;
    // The PflashConfig captured at load time. Per-request `prefill_*`
    // params override individual fields; the rest fall back to these
    // load-time defaults. Cleared alongside `pflash_state`.
    let mut pflash_cfg: Option<hipfire_pflash::pflash::PflashConfig> = None;
    // Hetero PFlash: when prefill_drafter_device differs from the target,
    // the drafter weights/KV/scratch live on a sibling device. The compress
    // output is a host-side Vec<u32>, so no peer-copy is needed — generate
    // routes maybe_compress_prompt to this handle, decode stays on target.
    // None means the drafter shares the target gpu (single-card, unchanged).
    let mut pflash_drafter_gpu: Option<rdna_compute::Gpu> = None;
    // Continuous-batch host scheduler + GPU batch state (if available).
    // Initialized on successful load when `continuous_batch_size` > 1 and
    // the loaded route staged lane state (including Qwen4 single/EP/PP fn-lanes).
    // None => sequential fallback.
    let mut continuous_batch_size: usize = 1;
    let mut batch_scheduler: Option<ContinuousBatchScheduler> = None;
    let mut batch_poisoned: Option<String> = None;
    // VMM batch route params, set only when the model staged `vmm_store`; None => flag-off routes.
    let mut vmm_batch: Option<hipfire_engine::scheduler::VmmBatchParams> = None;
    // Experimental multi-slot backend: alternate model owner (one SlotEngine/weight set).
    // None => ordinary LoadedModel path. Continuous-batching integration is deferred.
    // Arc allows request workers to hold the model alive only while active; reset/unload/swap refuse while active.
    let mut slot_backend: Option<std::sync::Arc<slots::SlotBackend>> = None;
    // KV backend ACK fields for the currently published model/slot. Cleared on
    // unload and empty handoff; preserved when a replacement fails and the prior
    // resident remains loaded.
    let mut resident_kv: Option<ResidentKvDiag> = None;

    // Background stdin reader. Drains stdin into an mpsc channel so
    // the main loop can pull non-blockingly between messages. Abort /
    // commit control messages are NOT forwarded; the reader handles
    // them inline via `apply_terminal_control` against the active
    // `(id, attempt_id)` transaction. This is the channel that makes
    // client-side cancellation actually stop an in-flight prefill —
    // without it, the main loop is blocked on GPU compute and wouldn't
    // even read the abort line until after the prefill completed.
    let (msg_tx, msg_rx) = mpsc::channel::<DaemonMsg>();
    if let Some(message) = pending_message {
        hipfire_engine::scheduler::note_daemon_msg_queued(&message);
        let _ = msg_tx.send(message);
    }
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let lock = stdin.lock();
        for line in lock.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(&line) {
                Ok(msg) => {
                    let msg_type = msg.get("type").and_then(|v| v.as_str());
                    if msg_type == Some("abort") || msg_type == Some("commit") {
                        // Correlated terminal control: require exact id + numeric
                        // attempt_id. Stale/malformed controls are ignored.
                        let id = msg.get("id").and_then(|v| v.as_str());
                        let attempt_id = msg.get("attempt_id").and_then(|v| v.as_u64());
                        if let (Some(id), Some(attempt_id), Some(kind)) = (id, attempt_id, msg_type)
                        {
                            tracing::info!(
                                request_id = id,
                                attempt_id,
                                command = kind,
                                "daemon control command received"
                            );
                            apply_terminal_control(kind, id, attempt_id);
                            batch_apply_terminal_control(kind, id, attempt_id);
                        }
                        continue;
                    }
                    if msg.get("type").and_then(|v| v.as_str()) == Some("force_answer") {
                        if let Some(id) = msg.get("id").and_then(|v| v.as_str()) {
                            tracing::info!(
                                request_id = id,
                                command = "force_answer",
                                "daemon control command received"
                            );
                            eprintln!("[daemon-force-answer] received force_answer for id={}", id);
                            *force_answer_for_id().lock().unwrap() = Some(id.to_string());
                        }
                        continue;
                    }
                    // Batch: carry the reader-minted admission token beside
                    // the internal message. Never rediscover ownership from
                    // the wire key after queueing.
                    let queued = match announce_generate_terminal(&msg) {
                        Some((_id, _attempt_id, Some(admission))) => {
                            DaemonMsg::RegularWithAdmission(msg, admission)
                        }
                        Some((id, attempt_id, None)) => {
                            eprintln!(
                                "[batch] duplicate generate dropped id={} attempt_id={}; preserving live registry",
                                id, attempt_id
                            );
                            continue;
                        }
                        None => DaemonMsg::Regular(msg),
                    };

                    hipfire_engine::scheduler::note_daemon_msg_queued(&queued);
                    if msg_tx.send(queued).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    if msg_tx.send(DaemonMsg::ParseError(e.to_string())).is_err() {
                        break;
                    }
                }
            }
        }
    });
    let mut inbox = DaemonInbox::new(msg_rx);
    loop {
        request_guards::exit_if_gpu_poisoned(&mut stdout);
        let Ok(daemon_msg) = inbox.recv() else {
            break;
        };
        let (msg, admission_from_message, mut singleton_transfer_from_message) = match daemon_msg {
            DaemonMsg::Regular(m) => (m, None, None),
            DaemonMsg::RegularWithAdmission(m, admission) => (m, Some(admission), None),
            DaemonMsg::SingletonWithAdmission(m, transfer) => {
                let admission = transfer.admission();
                (m, Some(admission), Some(transfer))
            }
            DaemonMsg::ParseError(e) => {
                tracing::warn!(error = %e, "daemon received invalid JSON");
                emit_uncorrelated_error(
                    &mut stdout,
                    None,
                    &format!("invalid JSON: {e}"),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                continue;
            }
        };

        let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let idle = !batch_scheduler
            .as_ref()
            .is_some_and(|s| s.active_count() > 0);
        let quiet = weight_host.is_some() && idle && !matches!(msg_type, "ping" | "diag");
        if quiet && weight_host_tenant::reconcile_lease(&mut model, &mut gpu) {
            (
                resident_kv,
                batch_scheduler,
                vmm_batch,
                continuous_batch_size,
            ) = (None, None, None, 1);
        }
        let request_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let command_span = tracing::info_span!(
            "daemon_command",
            command = msg_type,
            request_id = request_id
        );
        let _command_guard = command_span.enter();
        tracing::debug!("daemon command received");

        match msg_type {
            "configure" => {
                emit_uncorrelated_error(
                    &mut stdout,
                    None,
                    "process configuration is immutable after daemon startup",
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
            }
            "load" => {
                // Held for the whole arm: every exit (success after the
                // receipt, error ACK, refusal `continue`) ends the recording.
                let _load_phase_scope = rdna_compute::load_phases::begin_load_scope();
                // A new load: drop the previous load's decision notes.
                hipfire_config::load_summary::begin();
                // `params.pp` > 1 is explicit; otherwise `topology.pp` decides.
                // Resolve once, before the weight-host envelope, slot caps and
                // every pp gate below read the request.
                let mut resolved_msg = msg.clone();
                let load_pp = match hipfire_config::topology::resolve_load_pp(&mut resolved_msg) {
                    Ok(pp) => pp,
                    Err(refusal) => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &refusal,
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                let msg = resolved_msg;
                if let Some(refusal) = weight_host_tenant::load_refusal(weight_host.is_some(), &msg)
                {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        &refusal,
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                // FIX #1 (transactional EP load): the unload of the prior model
                // is deferred for the EP (tp>1) path until AFTER the new load
                // succeeds, so a partial EP load failure leaves the prior model
                // intact (and load_model_ep's staging guard frees the partial
                // ranks). For the single-GPU / pp path the prior model is
                // unloaded eagerly here as before (load_model uses the daemon's
                // `gpu` directly, so it can't be deferred without a major
                // refactor). `tp` is parsed authoritatively below; peek it here.
                let load_tp = msg
                    .get("params")
                    .and_then(|p| p.get("tp"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                let parsed_continuous_batch_size = parse_continuous_batch_size(msg.get("params"));
                let parsed_vmm_batch =
                    hipfire_engine::scheduler::parse_vmm_batch_params(msg.get("params"));
                let experimental_multi_slot = msg
                    .get("params")
                    .and_then(|p| p.get("experimental_multi_slot"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                // Omitted max_seq is resolved from model metadata and card
                // capacity by source admission. Explicit zero still refuses
                // before teardown (the fixed slot owner retains its 4096 default).
                let requested_seq = msg
                    .get("params")
                    .and_then(|p| p.get("max_seq"))
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
                if requested_seq.is_some_and(|n| n < MIN_REQUESTED_SEQ) {
                    let e = format!(
                        "load refused: max_seq {} below floor {MIN_REQUESTED_SEQ}",
                        requested_seq.unwrap()
                    );
                    emit_uncorrelated_error(&mut stdout, None, &e, "validation", false, false);
                    let _ = stdout.flush();
                    continue;
                }
                if experimental_multi_slot {
                    // Experimental slot backend is an alternate model owner, not a batch-mode switch.
                    // Validate mutually exclusive knobs before any GPU work.
                    if let Some(err) = slots::validate_load_caps(&msg, &gpu.arch) {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &err,
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    let slot_kv_k = msg
                        .get("params")
                        .and_then(|p| p.get("kv_k"))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty());
                    let slot_kv_v = msg
                        .get("params")
                        .and_then(|p| p.get("kv_v"))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty());
                    if slot_kv_k.is_some() || slot_kv_v.is_some() {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            "slots do not support --kv-k/--kv-v",
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    let slot_kv_explicit = msg
                        .get("params")
                        .and_then(|p| p.get("kv_backend"))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .is_some();
                    // Refuse model swap while slot requests active; do not keep old Arc alive via workers.
                    if slot_backend.as_ref().is_some_and(|b| b.active_count() > 0) {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            "load refused: slot requests active",
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    // Unload prior backends safely before loading the slot engine (exactly one weight copy).
                    // Drop any prior slot backend only after active check.
                    if let Some(slot) = slot_backend.take() {
                        match std::sync::Arc::try_unwrap(slot) {
                            Err(slot) => {
                                slot_backend = Some(slot);
                                emit_uncorrelated_error(
                                    &mut stdout,
                                    None,
                                    "load refused: slot requests active (Arc live)",
                                    "validation",
                                    false,
                                    false,
                                );
                                let _ = stdout.flush();
                                continue;
                            }
                            Ok(slot) => {
                                if let Err(reason) = slot.shutdown() {
                                    emit_uncorrelated_error(
                                        &mut stdout,
                                        None,
                                        &format!("prior slot shutdown failed: {reason}"),
                                        "internal",
                                        false,
                                        false,
                                    );
                                    let _ = stdout.flush();
                                    continue;
                                }
                                batch_clear_all_terminals();
                                resident_kv = None;
                            }
                        }
                    }
                    // Fixed Q8 slot arena has no VMM owner — always legacy.
                    // Emit once here; prior model/slot is about to be torn down.
                    let slot_kv_diag = ResidentKvDiag::for_slot(slot_kv_explicit);
                    // Tear down PFlash / ordinary model (eager; experimental requires pp=tp=1 so no EP deferral).
                    if let Some(mut pf) = pflash_state.take() {
                        if let Some(mut dg) = pflash_drafter_gpu.take() {
                            dg.bind_thread_or_warn();
                            pf.unload_drafter(&mut dg);
                            gpu.bind_thread_or_warn();
                        } else {
                            pf.unload_drafter(&mut gpu);
                        }
                    }
                    pflash_cfg = None;
                    if let Some(m) = model.take() {
                        if let Err(err) = hipfire_loader::unload_model(m, &mut gpu) {
                            emit_uncorrelated_error(
                                &mut stdout,
                                None,
                                &format!("prior unload failed: {err}"),
                                "internal",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                        // Prior ordinary model is gone; clear resident until slot publishes.
                        resident_kv = None;
                    } else if let Err(err) = hipfire_loader::ensure_vmm_ready_for_load(&mut gpu) {
                        emit_uncorrelated_error(&mut stdout, None, &err, "internal", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                    // Continuous-batch state must be cleared — slot backend is not batched.
                    batch_scheduler = None;
                    continuous_batch_size = 1;
                    batch_poisoned = None;
                    vmm_batch = None;

                    let path = msg.get("model").and_then(|v| v.as_str()).unwrap_or("");
                    if path.is_empty() {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            "load: missing model path",
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    // Floor already admitted pre-teardown above.
                    let requested_max_seq = requested_seq.unwrap_or(4096);
                    let max_seq = requested_max_seq.min(MAX_REQUESTED_SEQ);
                    let n_slots = msg
                        .get("params")
                        .and_then(|p| p.get("experimental_multi_slot_slots"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4) as usize;
                    let cap_tokens = msg
                        .get("params")
                        .and_then(|p| p.get("experimental_multi_slot_ctx"))
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                        .unwrap_or(max_seq);
                    let prefill_chunk = msg
                        .get("params")
                        .and_then(|p| p.get("experimental_multi_slot_prefill_chunk"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(1024) as usize;
                    let slot_params = slots::SlotLoadParams::from_load_msg(&msg);
                    // Vision tower: the SAME ladder the ordinary arm applies
                    // (`vision_mode=off` is a hard override). The slot engine
                    // discovers the `.vl` sibling itself, so it has to be told
                    // the mode — otherwise the documented text-only default
                    // still pays the tower's ~1 GB on this route.
                    let (slot_vision_mode, slot_vision, _) = resolve_vision_ladder(&msg);
                    match slots::SlotBackend::load(
                        path,
                        n_slots,
                        cap_tokens,
                        prefill_chunk,
                        slot_params.mtp_k,
                        slot_params.mtp_mode,
                        &slot_params.kv_mode_raw,
                        &slot_params.kv_backend,
                        slot_params.dflash_draft,
                        slot_params.dflash_required,
                        &slot_vision_mode,
                        slot_vision,
                        slot_params.max_batch_tokens,
                    ) {
                        Ok(backend) => {
                            let arch = backend.arch_str().to_string();
                            let dim = backend.dim();
                            let layers = backend.layers();
                            let vocab = backend.vocab();
                            let vl = backend.is_vl();
                            let audio = backend.is_audio();
                            let video = backend.is_video();
                            let mtp_mode_effective = backend.mtp_mode_effective();
                            // Same probe + ack fields as the single-slot path,
                            // computed from the template this route renders.
                            backend
                                .effort_probe()
                                .log_failure(backend.chat_template_source());
                            let effort_fields = backend.effort_probe().ack_fields();
                            // Ensure ordinary model stays None — exactly one weight copy.
                            model = None;
                            slot_backend = Some(std::sync::Arc::new(backend));
                            resident_kv = Some(ResidentKvDiag {
                                mode: Some(slot_params.kv_mode_resolved.to_owned()),
                                ..slot_kv_diag.clone()
                            });
                            let slot_summary = load_summary::summarize(
                                &hipfire_config::load_summary::take(),
                                &load_summary::SummaryState {
                                    model: std::path::Path::new(path)
                                        .file_name()
                                        .and_then(|name| name.to_str())
                                        .unwrap_or(path),
                                    arch: &arch,
                                    kv_mode: slot_params.kv_mode_resolved,
                                    kv_backend: &slot_kv_diag.backend,
                                    kv_request: &slot_kv_diag.request,
                                    max_seq: cap_tokens,
                                    bound: "slots",
                                    model_ctx: None,
                                    card_cap: None,
                                    speculator: (mtp_mode_effective == "on").then_some("mtp"),
                                    pp: 1,
                                    ep: false,
                                },
                            );
                            eprintln!("{}", slot_summary.line);
                            // Per contract: continuous_batch_capable false, cache_capable true, reasoning_contract qwen_jinja, plus experimental flag.
                            let mut ack = serde_json::json!({
                                "type": "loaded",
                                "arch": arch,
                                "dim": dim,
                                "layers": layers,
                                "vocab": vocab,
                                "vl": vl,
                                "audio": audio,
                                "video": video,
                                "reasoning_contract": "qwen_jinja",
                                "cache_capable": true,
                                "retry_reset_eligible": false,
                                "continuous_batch_capable": false,
                                "experimental_multi_slot": true,
                                "kv_backend_request": slot_kv_diag.request,
                                "kv_backend": slot_kv_diag.backend,
                                "kv_backend_reason": slot_kv_diag.reason,
                                "kv_backend_legacy": slot_kv_diag.legacy,
                                "kv_backend_warning": slot_kv_diag.warning,
                                "kv_mode": slot_params.kv_mode_resolved,
                                "pp": 1,
                                "mtp_mode_effective": mtp_mode_effective,
                            });
                            ack.as_object_mut()
                                .expect("loaded ack is a JSON object")
                                .insert("load_summary".to_string(), slot_summary.to_json());
                            ack.as_object_mut()
                                .expect("loaded ack is a JSON object")
                                .extend(effort_fields);
                            emit_load_phases();
                            let _ = writeln!(stdout, "{ack}");
                            let _ = stdout.flush();
                        }
                        Err(e) => {
                            emit_uncorrelated_error(
                                &mut stdout,
                                None,
                                &format!("load failed: {e}"),
                                "internal",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                        }
                    }
                    continue;
                }
                // Ordinary load: refuse while slot requests active, otherwise checked shutdown
                if slot_backend.as_ref().is_some_and(|b| b.active_count() > 0) {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        "load refused: slot requests active",
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                if let Some(slot) = slot_backend.take() {
                    match std::sync::Arc::try_unwrap(slot) {
                        Err(slot) => {
                            slot_backend = Some(slot);
                            emit_uncorrelated_error(
                                &mut stdout,
                                None,
                                "load refused: slot requests active (Arc live)",
                                "validation",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                        Ok(slot) => {
                            if let Err(reason) = slot.shutdown() {
                                emit_uncorrelated_error(
                                    &mut stdout,
                                    None,
                                    &format!("prior slot shutdown failed: {reason}"),
                                    "internal",
                                    false,
                                    false,
                                );
                                let _ = stdout.flush();
                                continue;
                            }
                            batch_clear_all_terminals();
                            resident_kv = None;
                        }
                    }
                }

                let path = msg.get("model").and_then(|v| v.as_str()).unwrap_or("");
                // Only an explicit request is clamped; omission lets
                // admission choose the trained/model-card minimum instead
                // of the old fixed daemon default.
                let requested_max_seq = requested_seq.unwrap_or(0);
                let max_seq = requested_max_seq.min(MAX_REQUESTED_SEQ);
                if requested_max_seq > MAX_REQUESTED_SEQ {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"info","message":"requested max_seq {} exceeds ceiling {} — clamped"}}"#,
                        requested_max_seq, MAX_REQUESTED_SEQ
                    );
                    let _ = stdout.flush();
                }
                // Optional DFlash draft model path. When supplied AND the target
                // is a Qwen3.5 arch (5 or 6), we load draft weights + scratch
                // alongside the target and the temp=0 generate fast path routes
                // through `spec_step_dflash` for the 1.7-2.5× speedup on the
                // 27B target. Non-matching archs / missing draft file are
                // logged but don't fail the load.
                //
                // `dflash_mode=off` is a hard daemon-side override: even if a
                // draft path was passed, skip the load. CLI-side gating is the
                // primary path (saves the wire round-trip for the draft path
                // string), but this guard makes the flag durable when the
                // daemon is driven by a non-hipfire-CLI client.
                let dflash_mode = msg
                    .get("params")
                    .and_then(|p| p.get("dflash_mode"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("auto");
                // `HIPFIRE_DFLASH_DRAFT` (documented in AGENTS.md §7) forces a
                // draft path for clients that cannot pass `params.draft` (the
                // serve prewarm / HTTP-reload path has no --model-draft flag):
                // non-empty → wins over params.draft; explicitly EMPTY → opt out
                // of draft loading entirely; unset → params.draft as before.
                let env_draft = developer_var("HIPFIRE_DFLASH_DRAFT").ok();
                let raw_draft: Option<String> = match env_draft.as_deref() {
                    Some("") => None,
                    Some(p) => Some(p.to_string()),
                    None => msg
                        .get("params")
                        .and_then(|p| p.get("draft"))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string()),
                };
                // DFlash is not served on an EP/TP mesh (tp>1 refuses a draft
                // below). An `auto` mode only pairs the registry draft when one
                // exists, so on tp>1 it drops the pairing instead of failing
                // the load; an explicit `on` still reaches that refusal.
                let draft_tp = msg
                    .get("params")
                    .and_then(|p| p.get("tp"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1);
                let draft_path = if dflash_mode == "off" {
                    if let Some(d) = &raw_draft {
                        eprintln!("[hipfire-daemon] dflash_mode=off — skipping draft load ({d})");
                    }
                    hipfire_config::load_summary::note(
                        hipfire_config::load_summary::Facet::SpecOff,
                        match &raw_draft {
                            Some(d) => format!(
                                "DFlash off: speculation.dflash=off (draft {} not loaded)",
                                Path::new(d).file_name().map_or_else(
                                    || d.clone(),
                                    |n| n.to_string_lossy().into_owned()
                                )
                            ),
                            None => "DFlash off: speculation.dflash=off".to_string(),
                        },
                    );
                    None
                } else if dflash_mode == "auto" && draft_tp > 1 {
                    if let Some(d) = raw_draft {
                        eprintln!(
                            "[hipfire-daemon] dflash_mode=auto with tp={draft_tp} — DFlash is not served on EP/TP; skipping draft load ({d})"
                        );
                        hipfire_config::load_summary::note(
                            hipfire_config::load_summary::Facet::SpecOff,
                            format!(
                                "DFlash off: not served on EP/TP (tp={draft_tp}, dflash_mode=auto)"
                            ),
                        );
                    }
                    None
                } else {
                    if raw_draft.is_none() {
                        hipfire_config::load_summary::note(
                            hipfire_config::load_summary::Facet::SpecOff,
                            format!(
                                "DFlash off: no draft paired (speculation.dflash={dflash_mode})"
                            ),
                        );
                    }
                    raw_draft
                };
                // Shared Qwen3.5-VL tower sidecar (`qwen3.8-27b-vision.hfq`).
                // Same override ladder as the DFlash draft: `HIPFIRE_VISION_SIDECAR`
                // non-empty wins over `params.vision`; explicitly EMPTY opts out;
                // unset → `params.vision` as sent. Arch-free string plumbing —
                // admission validates (arch 5|6, tower tensor present) and the
                // Qwen35 carrier loads the tower from it.
                //
                // `vision_mode=off` is a hard daemon-side override,
                // mirroring the `dflash_mode=off` guard above: even an explicit
                // sidecar is skipped, so an `off` load never pays the +~1 GB
                // tower VRAM (the schema default is `auto`). CLI-side gating is the primary path; this guard
                // makes the flag durable for non-hipfire-CLI clients.
                //
                // One implementation for both load arms: `resolve_vision_ladder`
                // (next to `apply_vision_mode_gate`).
                let (vision_mode, vision_path, gated_off) = resolve_vision_ladder(&msg);
                vision_gated_off = gated_off;
                // Gemma 4 EAGLE drafter (arch-22 `gemma4_unified_assistant`).
                // Deliberately a SEPARATE param from `params.draft` (the
                // qwen3.5 DFlash knob) so a DFlash .hfq can never be routed
                // into the EAGLE loader by accident. `params.spec` = draft
                // length; 1..=5 accepted (see gemma4_eagle_spec_len).
                let gemma4_drafter = msg
                    .get("params")
                    .and_then(|p| p.get("drafter"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let gemma4_draft_len = if gemma4_drafter.is_some() {
                    let spec_raw = msg
                        .get("params")
                        .and_then(|p| p.get("spec"))
                        .and_then(|v| v.as_u64());
                    match hipfire_loader::gemma4_eagle_spec_len(spec_raw) {
                        Ok(n) => n,
                        Err(e) => {
                            emit_uncorrelated_error(
                                &mut stdout,
                                None,
                                &e,
                                "validation",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    }
                } else {
                    hipfire_loader::GEMMA4_EAGLE_DRAFT_LEN
                };
                // Path to a head overlay (`hipfire-quantize --head-only`),
                // resolved by the CLI from the registry's `heads` map. Empty
                // means "use the head baked into the model file".
                let head_path = msg
                    .get("params")
                    .and_then(|p| p.get("head"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let kv_mode_override = msg
                    .get("params")
                    .and_then(|p| p.get("kv_mode"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let kv_backend_override = msg
                    .get("params")
                    .and_then(|p| p.get("kv_backend"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let kv_k_override = msg
                    .get("params")
                    .and_then(|p| p.get("kv_k"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let kv_v_override = msg
                    .get("params")
                    .and_then(|p| p.get("kv_v"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                // lifecycle: deprecated since 0.4.0, removal 0.5.0 — Givens asym KV and the asymN/turboN aliases are superseded by fwht3
                // One warning line when the load names a deprecated KV format
                // (kv_mode falls back to HIPFIRE_KV_MODE exactly as the carriers do).
                if let Some(warning) = [
                    Some(
                        kv_mode_override
                            .as_deref()
                            .unwrap_or(hipfire_runtime::config::get().kv_mode.as_str()),
                    ),
                    kv_k_override.as_deref(),
                ]
                .into_iter()
                .flatten()
                .find_map(hipfire_runtime::kv_mode::deprecated_kv_name_warning)
                {
                    eprintln!("[hipfire-daemon] {warning}");
                }

                // Per-load adaptive-KV selector (mirrors kv_mode). Overrides the
                // HIPFIRE_KV_ADAPTIVE env. off|conservative|balanced|aggressive|
                // advanced:k=..,v=.. — resolved in load_model (param > env > off).
                let kv_adaptive_override = msg
                    .get("params")
                    .and_then(|p| p.get("kv_adaptive"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());

                // MTP speculative decode config. `mtp_mode` gates weight
                // discovery at load time (off=skip, on=error-if-missing,
                // auto=scan+log). `mtp_k` sets the draft window size.
                let mtp_mode = msg
                    .get("params")
                    .and_then(|p| p.get("mtp_mode"))
                    .and_then(|v| v.as_str())
                    .unwrap_or(&hipfire_runtime::config::get().mtp_mode)
                    .to_string();
                let mtp_k: Option<usize> = msg
                    .get("params")
                    .and_then(|p| p.get("mtp_k"))
                    .and_then(|v| v.as_u64())
                    .map(|value| value as usize)
                    .or(hipfire_runtime::config::get().mtp_k);
                let mtp_path = request_guards::mtp_sidecar_path(&msg, &mtp_mode);

                // DFlash adaptive verify-block (opt-in, default off): clamp(ceil(τ̂)+2,
                // 2, full) over the trailing 8 verify cycles, full below 2k ctx.
                // Mutually exclusive with the retained-PM4 route (fixed B=16 shape);
                // HIPFIRE_DFLASH_ADAPTIVE_B=0 forces fixed.
                let adaptive_b = msg
                    .get("params")
                    .and_then(|p| p.get("dflash_adaptive_b"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let Some((dflash_block, dflash_block_auto)) = request_guards::dflash_block(
                    &mut stdout,
                    &msg,
                    parsed_vmm_batch.is_some(),
                    parsed_continuous_batch_size,
                ) else {
                    continue;
                };

                // Model-free n-gram policy normally arrives as per-load params
                // resolved by the CLI. Direct protocol clients inherit the
                // daemon's typed process policy instead of ambient env.
                let spec_cfg = hipfire_runtime::loader_api::SpecLoadCfg {
                    ngram_draft: msg
                        .get("params")
                        .and_then(|p| p.get("ngram_draft"))
                        .and_then(|v| v.as_bool())
                        .or(Some(hipfire_runtime::config::get().ngram_draft)),
                    ngram_k: msg
                        .get("params")
                        .and_then(|p| p.get("ngram_k"))
                        .and_then(|v| v.as_u64())
                        .map(|k| k as usize)
                        .or(Some(hipfire_runtime::config::get().ngram_k)),
                    ngram_min_count: msg
                        .get("params")
                        .and_then(|p| p.get("ngram_min_count"))
                        .and_then(|v| v.as_u64())
                        .map(|c| c as u32)
                        .or(Some(hipfire_runtime::config::get().ngram_min_count)),
                    // DDTree draft tuning — same load-param mechanism as ngram_k:
                    // CLI `--ddtree-budget` / `--ddtree-topk` → these load params,
                    // env-wins-else-param in the loader.
                    ddtree_budget: msg
                        .get("params")
                        .and_then(|p| p.get("ddtree_budget"))
                        .and_then(|v| v.as_u64())
                        .map(|b| b as usize),
                    ddtree_topk: msg
                        .get("params")
                        .and_then(|p| p.get("ddtree_topk"))
                        .and_then(|v| v.as_u64())
                        .map(|k| k as usize),
                    // DSpark draft module: the CLI lowers `speculation` into a
                    // `dspark_mode` string. off→Some(false) (skip load+build),
                    // on→Some(true) (force), auto/absent→None (load-if-sidecar).
                    dspark: msg
                        .get("params")
                        .and_then(|p| p.get("dspark_mode"))
                        .and_then(|v| v.as_str())
                        .and_then(hipfire_config::parse_tri_state),
                    dspark_conf_threshold: msg
                        .get("params")
                        .and_then(|p| p.get("dspark_conf_threshold"))
                        .and_then(|v| v.as_f64())
                        .map(|t| t as f32),
                    // DFlash mirrors mtp: on = fail closed on a missing/unloadable draft.
                    dflash: hipfire_config::parse_tri_state(dflash_mode),
                    mtp: hipfire_config::parse_tri_state(&mtp_mode),
                    mtp_k,
                    dflash_adaptive_b: Some(adaptive_b),
                    dflash_block: Some(dflash_block),
                    dflash_block_auto,
                };

                // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
                // 0.1.7: TriAttention / CASK eviction protocol fields. When
                // `cask_sidecar` is set, `load_model` sizes the KV cache to a
                // *physical_cap* (budget+beta+safety, clamped to max_seq) instead
                // of the full max_seq, and wires an `Eviction` policy that the
                // generate loop calls after every prefill-chunk / decode-forward.
                // That decouples advertised context length from VRAM footprint —
                // a 128K max_seq can run in ~1K-slot physical buffer when the
                // operator opts in.
                let cask_sidecar = msg
                    .get("params")
                    .and_then(|p| p.get("cask_sidecar"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let cask_enabled = msg
                    .get("params")
                    .and_then(|p| p.get("cask"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let cask_budget = msg
                    .get("params")
                    .and_then(|p| p.get("cask_budget"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(512) as usize;
                let cask_beta = msg
                    .get("params")
                    .and_then(|p| p.get("cask_beta"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128) as usize;
                let cask_handoff_tokens = msg
                    .get("params")
                    .and_then(|p| p.get("cask_handoff_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let cask_core_frac = msg
                    .get("params")
                    .and_then(|p| p.get("cask_core_frac"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32;
                let cask_fold_m = msg
                    .get("params")
                    .and_then(|p| p.get("cask_fold_m"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2) as usize;
                // CASK/TriAttention eviction is deprecated (removal in 0.5.0):
                // it still loads, but every opt-in gets exactly one warning line.
                if cask_sidecar.is_some() || cask_enabled {
                    eprintln!(
                        "[hipfire-daemon] warning: CASK is deprecated and will be removed in 0.5.0; not supported (cask_sidecar/cask set at load)"
                    );
                }
                // Known-broken combo guard: CASK m-folding + DFlash spec decode
                // degenerates into single-token loops after the first eviction
                // (the m-folded synthetic K/V rows are off the draft's trained
                // hidden-state distribution). Until that's fixed at the library
                // level, downgrade m-folding to plain TriAttention drop-eviction
                // when a draft is attached. User's context window + eviction
                // cadence still work; just the fold step is skipped.
                let cask_m_folding_effective = if cask_enabled && draft_path.is_some() {
                    eprintln!(
                        "[hipfire-daemon] cask:true + draft: both set — downgrading to plain TriAttention drop-eviction (CASK m-fold + DFlash is a known-broken combo; see feedback_cask_mfold_dflash_broken.md)",
                    );
                    false
                } else {
                    cask_enabled
                };
                let cask = CaskConfig {
                    sidecar: cask_sidecar,
                    cask_m_folding: cask_m_folding_effective,
                    handoff_tokens: cask_handoff_tokens,
                    budget: cask_budget,
                    beta: cask_beta,
                    core_frac: cask_core_frac,
                    fold_m: cask_fold_m,
                };

                // MMQ per-weight screening (#87): detect outlier rows that
                // cause Q8_1 precision loss and fall back to WMMA for those
                // weights. Disabled by default; enable with mmq_screen=true
                // (or HIPFIRE_MMQ_SCREEN=1) when adding new quant formats.
                if let Some(v) = msg
                    .get("params")
                    .and_then(|p| p.get("mmq_screen"))
                    .and_then(|v| v.as_bool())
                {
                    gpu.mmq_screen.enabled = v;
                }
                if let Some(v) = msg
                    .get("params")
                    .and_then(|p| p.get("mmq_screen_threshold"))
                    .and_then(|v| v.as_f64())
                {
                    gpu.mmq_screen.threshold = v as f32;
                }

                // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
                // ── PFlash load-time params (Phase 4.0 #93) ──────────────
                //
                // Parse compression knobs per PRD §5.3.2. None of these
                // affect the target load itself; they only configure the
                // optional drafter that PFlash uses for prompt scoring.
                // Drafter loading happens AFTER target load succeeds so
                // we can use the target's tokenizer for the compat check.
                let pflash_mode_str = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_compression"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("off")
                    .to_string();
                let pflash_threshold = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_threshold"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32768) as usize;
                let pflash_keep_ratio = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_keep_ratio"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.05) as f32;
                let pflash_alpha = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_alpha"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.85) as f32;
                let pflash_min_keep = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_min_keep"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2048) as usize;
                let pflash_sink = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_sink"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(256) as usize;
                let pflash_recent = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_recent"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1024) as usize;
                let pflash_block = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_block"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128) as usize;
                let pflash_drafter = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_drafter"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                // -1 = drafter shares the target gpu (default). >=0 routes
                // the drafter to that HIP device for hetero compress.
                let pflash_drafter_device: i32 = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_drafter_device"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(-1) as i32;
                let pflash_profile = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_profile"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let pflash_sparse_threshold = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_sparse_threshold"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32768) as usize;

                // Validate load-time PFlash params before they reach
                // PflashConfig + load_drafter. Same range rules the
                // per-request override path uses; without these, a
                // bad load-time value would silently be accepted and
                // panic the daemon at the first generate request.
                let pflash_load_err: Option<String> =
                    if !(pflash_keep_ratio > 0.0 && pflash_keep_ratio <= 1.0) {
                        Some(format!(
                            "prefill_keep_ratio={pflash_keep_ratio} not in (0, 1]"
                        ))
                    } else if pflash_block == 0 {
                        Some("prefill_block must be > 0".to_string())
                    } else {
                        None
                    };
                // PFlash is deprecated (removal in 0.5.0): it still loads, but
                // every opt-in (prefill_compression != off) gets one warning line.
                if pflash_mode_str != "off" {
                    eprintln!(
                        "[hipfire-daemon] warning: PFlash is deprecated and will be removed in 0.5.0; not supported, prefix caching supersedes it (prefill_compression={pflash_mode_str} set at load)"
                    );
                }

                // Pipeline-parallel degree (Stage 7 of #58). Default 1 =
                // single-GPU (no behavior change). pp > 1 routes through
                // Gpus + *_multi paths and refuses VL / DFlash / CASK /
                // PFlash at load time. v1 supports Qwen3.5 dense + MoE
                // only — see load_model_pp for the arch_id check.
                let pp = load_pp;
                // Expert-parallel degree (EP, task #26). tp>1 shards routed
                // experts across ranks via load_model_ep. Mutually exclusive
                // with pp; v1 refuses DFlash. See docs/plans/daemon-ep-wiring.md.
                let tp = msg
                    .get("params")
                    .and_then(|p| p.get("tp"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                if tp > 1 && pp > 1 {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        hipfire_loader::bench_topology::PP_EP_EXCLUSIVE_MESSAGE,
                        "unsupported",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                if tp > 1 && draft_path.is_some() {
                    emit_uncorrelated_error(&mut stdout, None, "EP serving (tp>1) does not support DFlash drafters in v1; reload without a draft.", "unsupported", false, false);
                    let _ = stdout.flush();
                    continue;
                }
                if tp > 1 && gemma4_drafter.is_some() {
                    emit_uncorrelated_error(&mut stdout, None, "EP serving (tp>1) does not support the gemma4 EAGLE drafter; reload without params.drafter.", "unsupported", false, false);
                    let _ = stdout.flush();
                    continue;
                }
                if pp > 1 {
                    if gemma4_drafter.is_some() {
                        emit_uncorrelated_error(&mut stdout, None, "gemma4 EAGLE spec-decode requires pp=1 (arch_id=13 has no pipeline-parallel path); reload without params.drafter.", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                    if draft_path.is_some()
                        && developer_var("HIPFIRE_PP_DFLASH").ok().as_deref() != Some("1")
                    {
                        emit_uncorrelated_error(&mut stdout, None, "DFlash speculative decode requires pp=1 in v1 (set HIPFIRE_PP_DFLASH=1 to opt into the experimental pp>1 PRD path; note PR2-4 of docs/plans/hetero-pflash-dflash.prd are not yet implemented — the load message will accept but generate will not run cross-card spec-decode). See issue #58 v1.1 roadmap.", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                    if cask.sidecar.is_some() {
                        emit_uncorrelated_error(&mut stdout, None, "CASK / TriAttention eviction requires pp=1 in v1; see issue #58 v1.1 roadmap", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                    if (pflash_drafter.is_some() || pflash_mode_str != "off")
                        && developer_var("HIPFIRE_PP_PFLASH").ok().as_deref() != Some("1")
                    {
                        emit_uncorrelated_error(&mut stdout, None, "PFlash prefill compression requires pp=1 in v1 (set HIPFIRE_PP_PFLASH=1 to opt into the experimental pp>1 PoC); see issue #58 v1.1 roadmap", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                }

                let state_quant_override = msg
                    .get("params")
                    .and_then(|p| p.get("state_quant"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let deepseek4_experts_per_token = msg
                    .get("params")
                    .and_then(|p| p.get("deepseek4_experts_per_token"))
                    .and_then(|v| v.as_u64())
                    .map(|value| value as usize);
                let deepseek4_compute_placement = match msg
                    .get("params")
                    .and_then(|p| p.get("deepseek4_compute_placement"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("single")
                    .parse::<hipfire_config::Deepseek4ComputePlacement>()
                {
                    Ok(placement) => placement,
                    Err(error) => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &format!("invalid DeepSeek V4 compute placement: {error}"),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                // DeepSeek V4 experts-per-token override requires tp=1 (moved
                // before admission so the refusal leaves the prior model intact).
                if tp > 1 && deepseek4_experts_per_token.is_some() {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        "DeepSeek V4 experts-per-token override requires tp=1",
                        "unsupported",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }

                // ── G2 source-aware admission (read-only; before teardown) ──
                // Classify the incoming source and decide the effective topology
                // BEFORE any destructive side effect so a refusal leaves the
                // prior model usable. The retained SourceAdmission is consumed
                // by the load route below — no re-open, no re-classify.
                let qwen4_admission_options = hipfire_loader::admission::SourceAdmissionOptions {
                    spec: spec_cfg,
                    kv_adaptive: hipfire_loader::admission::qwen4_kv_adaptive_requested(
                        kv_adaptive_override.as_deref(),
                    ),
                    eagle_drafter: gemma4_drafter.is_some(),
                    cask: cask.sidecar.is_some(),
                    state_quant: state_quant_override.is_some(),
                    non_single_compute: !matches!(
                        deepseek4_compute_placement,
                        hipfire_config::Deepseek4ComputePlacement::Single
                    ),
                    expert_count_override: deepseek4_experts_per_token.is_some(),
                    pflash: pflash_drafter.is_some() || pflash_mode_str != "off",
                };
                let backend_request =
                    match hipfire_loader::admission::KvBackendRequest::from_override(
                        kv_backend_override.as_deref(),
                    ) {
                        Ok(request) => request,
                        Err(e) => {
                            emit_uncorrelated_error(
                                &mut stdout,
                                None,
                                &e,
                                "validation",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    };
                let open_admission_started = std::time::Instant::now();
                let mut admission = match hipfire_loader::admission::admit_source_with_options(
                    path,
                    tp,
                    pp,
                    backend_request,
                    draft_path.as_deref(),
                    gpu.arch.as_str(),
                    vision_path.as_deref(),
                    &vision_mode,
                    head_path.as_deref(),
                    max_seq,
                    hipfire_loader::admission::KvBackendHints {
                        kv_mode: kv_mode_override.as_deref(),
                        kv_k: kv_k_override.as_deref(),
                        kv_v: kv_v_override.as_deref(),
                        kv_adaptive: kv_adaptive_override.as_deref(),
                        cask: Some(&cask),
                        deepseek4_heterogeneous: !matches!(
                            deepseek4_compute_placement,
                            hipfire_config::Deepseek4ComputePlacement::Single
                        ),
                        vmm_runtime_available: gpu.vmm_recommended_granularity().is_ok(),
                        free_vram_bytes: gpu.device_mem_info().ok().map(|(free, _)| free),
                        qwen_default_q8: hipfire_loader::admission::qwen_default_q8_enabled(),
                    },
                    qwen4_admission_options,
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        // Refusal ledger: admission is read-only — zero teardown,
                        // allocation, VMM init, remap, carrier entry, collective,
                        // or cache mutation; the prior model remains loaded.
                        emit_uncorrelated_error(&mut stdout, None, &e, "validation", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                };
                rdna_compute::load_phases::add(
                    "open_admission",
                    rdna_compute::load_phases::ns_since(open_admission_started),
                    0,
                );
                // The continuous-batch request staged after this load, so a
                // Qwen4 expert-parallel load reserves the AR lane store's
                // bytes before it splits the experts. Only the exact VMM
                // route stages Qwen4 lanes; `vmm_batch` is false for the
                // non-exact route and without `serve_vmm_batch`.
                admission.batch_request = hipfire_loader::admission::BatchLoadRequest {
                    vmm_batch: parsed_vmm_batch.is_some_and(|params| !params.nonexact),
                    continuous_batch_size: parsed_continuous_batch_size,
                    row_budget: parsed_vmm_batch.map_or(0, |params| params.max_batch_tokens),
                };
                // Only tp>1 EP loads defer prior retirement. Every single-device
                // or pp load (Qwen4/Flash-Next included) is unload-first: after
                // admission the prior model is torn down and VMM must be clean
                // before construction, so a later failure leaves no model.
                let defer_prior_unload = load_tp > 1;
                let max_seq = admission.max_seq;
                let sequence_reason = admission.sequence_reason;
                let legacy_measured = admission.legacy_measured;
                // Exactly one legacy warning per admitted trunk; loader admitted
                // path does not emit (daemon owns the boundary).
                let pending_kv_diag = ResidentKvDiag::from_admission(
                    backend_request,
                    admission.kv_backend,
                    admission.kv_backend_reason.as_deref(),
                );

                // Unload previous if any. PFlash drafter goes first so
                // its tensors join the pool before unload_model drains
                // it -- otherwise free_tensor would queue them into the
                // pool just-emptied by drain_pool with no follow-up
                // drain, leaving drafter VRAM resident across the next
                // load (the explicit "unload" handler has the same
                // ordering for the same reason).
                //
                // FIX (transactional pflash teardown): pflash_state is part of
                // the PRIOR model (it holds that model's PFlash drafter). For
                // the deferred tp>1 EP path it must NOT be torn down here —
                // otherwise a partial EP load failure (whose FIX #1 deferral
                // keeps `model` alive) would leave the surviving prior model
                // stripped of its drafter. Defer it to the success branch
                // alongside the deferred model unload. For load_tp <= 1 the
                // prior model is unloaded eagerly, so tear pflash down here in
                // the original order. (EP archs are ds4/minimax and refuse
                // PFlash drafters, so on a SUCCESSFUL tp>1 load this just frees
                // the outgoing model's drafter at the deferred site.)
                if !defer_prior_unload {
                    if let Some(mut pf) = pflash_state.take() {
                        if let Some(mut dg) = pflash_drafter_gpu.take() {
                            dg.bind_thread_or_warn();
                            pf.unload_drafter(&mut dg); // sibling-device drafter: free on its own handle, then drop
                            gpu.bind_thread_or_warn();
                        } else {
                            pf.unload_drafter(&mut gpu);
                        }
                    }
                    pflash_cfg = None;
                    if let Some(m) = model.take() {
                        if let Err(err) = hipfire_loader::unload_model(m, &mut gpu) {
                            emit_uncorrelated_error(
                                &mut stdout,
                                None,
                                &format!("prior unload failed: {err}"),
                                "internal",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            resident_kv = None;
                            continue;
                        }
                        // Prior ordinary model is gone until the new load publishes.
                        resident_kv = None;
                    } else if let Err(err) = hipfire_loader::ensure_vmm_ready_for_load(&mut gpu) {
                        emit_uncorrelated_error(&mut stdout, None, &err, "internal", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                }
                // EP path: when no live prior model remains (fresh daemon, or
                // after deferred prior unload failed and left model=None with
                // pending VMM), refuse to construct a new EP model until
                // orphan teardown clears. Skip when a live deferred prior
                // still sits in `model` — unload stays deferred until after
                // successful new-model construction.
                if ep_deferred_needs_vmm_preflight(load_tp, model.is_some()) {
                    if let Err(err) = hipfire_loader::ensure_vmm_ready_for_load(&mut gpu) {
                        emit_uncorrelated_error(&mut stdout, None, &err, "internal", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                }
                let loaded = if tp > 1 {
                    hipfire_loader::load_model_ep_admitted(
                        admission,
                        path,
                        max_seq,
                        tp,
                        kv_mode_override.as_deref(),
                        kv_k_override.as_deref(),
                        kv_v_override.as_deref(),
                        state_quant_override.as_deref(),
                        mtp_path.as_deref(),
                        spec_cfg,
                    )
                } else {
                    hipfire_loader::load_admitted_with_gemma4_drafter(
                        admission,
                        path,
                        max_seq,
                        deepseek4_experts_per_token,
                        deepseek4_compute_placement,
                        draft_path.as_deref(),
                        gemma4_drafter.as_deref(),
                        gemma4_draft_len,
                        mtp_path.as_deref(),
                        kv_mode_override.as_deref(),
                        kv_k_override.as_deref(),
                        kv_adaptive_override.as_deref(),
                        state_quant_override.as_deref(),
                        &cask,
                        pp,
                        spec_cfg,
                        &mut gpu,
                    )
                };
                match loaded {
                    Ok(mut m) => {
                        let max_seq = m.max_seq;
                        let resolved = m.sequence.as_ref();
                        let seq_bound = resolved.map_or(
                            if requested_seq.is_some() {
                                "user"
                            } else if legacy_measured {
                                "legacy-fit"
                            } else {
                                "legacy"
                            },
                            |s| s.bound,
                        );
                        let model_ctx = resolved.map(|s| s.model_ctx);
                        let card_cap = resolved.map(|s| s.card_cap);
                        let seq_kv = resolved.map_or_else(
                            || kv_mode_override.as_deref().unwrap_or("auto"),
                            |s| s.kv_mode,
                        );
                        let seq_reason_json = serde_json::to_string(&sequence_reason).unwrap();
                        if let Some(s) = resolved {
                            eprintln!("max_seq: {max_seq} (bound={seq_bound}; model_ctx={}, card_cap={}, kv={seq_kv})", s.model_ctx, s.card_cap);
                        } else {
                            eprintln!("max_seq: {max_seq} (bound={seq_bound}; model_ctx=unknown, card_cap=unknown, kv={seq_kv})");
                        }
                        if let Some(reason) = sequence_reason {
                            eprintln!("{reason}");
                        }
                        // Deferred EP retirement lives AFTER continuous-batch
                        // staging below: the prior resident in `model` (and
                        // its PFlash drafter) must survive every fallible
                        // stage of the new model so a staging failure rolls
                        // back only `m` and keeps the prior usable.
                        let arch = hipfire_loader::arch_label(m.arch_id);
                        let drafter = m.speculator.as_ref().map(|speculator| speculator.name());
                        let redline_default = hipfire_runtime::config::retained_redline_default(
                            &gpu.arch,
                            arch,
                            path,
                            pp,
                            tp, hipfire_loader::admission::effective_ep(m.arch_id),
                            drafter.is_some(),
                        ) && hipfire_config::devices::redline_platform_refusal().map_or(true, |refusal| {
                            eprintln!("[redline] retained default refused: {refusal}; using the HIP graph");
                            false
                        });
                        if gpu.replay.configure_model_default(redline_default) && redline_default {
                            eprintln!(
                                "[redline] enabling fail-closed retained default on {} \
                                 (model_arch={arch}, drafter={}, transport={})",
                                gpu.arch,
                                drafter.unwrap_or("off"),
                                gpu.replay.transport_name()
                            );
                        }
                        // NOP pacing is certified for the gfx1201 Qwen3.5-dense
                        // retained tape only; MQ4R and every other retained
                        // default keep their unpaced tapes.
                        let gfx1201_dense_pacing = redline_default
                            && gpu.arch == "gfx1201"
                            && arch == "qwen3_5"
                            && !hipfire_runtime::config::mq4r_redline_default(
                                &gpu.arch, path, pp, tp,
                            );
                        gpu.replay
                            .set_pm4_gfx12_dispatch_pacing(if gfx1201_dense_pacing {
                                rdna_compute::replay::gfx1201_pm4_pacing_from_config()
                            } else {
                                Default::default()
                            });
                        let vl = m.has_vision_encoder();
                        let audio = m.has_audio_encoder();
                        let video = m.has_video_route();
                        let (dim, layers, vocab) = m.ack_dims();

                        // MTP config from load params; `None` (auto) keeps the carrier-resolved K (Qwen4 native MTP) else the skeleton MTP_K_DEFAULT.
                        // `auto` is gated by the producer capability declared on the
                        // model's carrier (`Carrier::mtp_measured_ar_floor`): a producer
                        // without a measured AR floor resolves `auto` to `off`, so no
                        // downstream reader rearms it from head/weights presence
                        // (`mtp_weights_present`, the DSpark sidecar). `on` is explicit
                        // and unchanged.
                        m.mtp_mode = hipfire_config::mtp_effective_mode_name(
                            &mtp_mode,
                            m.mtp_measured_ar_floor(),
                        )
                        .to_string();
                        m.mtp_k = mtp_k.unwrap_or(m.mtp_k);
                        // Record whether MTP/DSpark weights are present in the loaded
                        // model. This is metadata only: it never decides arming. The
                        // effective `m.mtp_mode` above (after the producer-capability
                        // gate) decides, so presence alone cannot rearm `auto`
                        // (`deepseek4_spec_requested_from_policy` additionally needs the
                        // declared floor). Sources:
                        //   - DeepSeek V4: the trunk's bundled `mtp_layer` (loaded only
                        //     for an explicit `on`, see `DeepseekV4Config::load_mtp`).
                        //   - DeepSeek V4: a DSpark sidecar (a separate selector; the
                        //     route is picked by the installed drafter, not this flag).
                        //   - Qwen3.5/3.6: a native MTP (NextN) head loaded by the
                        //     loader (`qwen35_mtp_head`, from a bundled `.mq4-mtp`
                        //     trailer or a `.mtp` sidecar), loaded only for an explicit
                        //     `on`. The loader already set `m.mtp_weights_present =
                        //     true` in that case; OR it in here so the ds4 probe doesn't
                        //     clobber it back to false for a qwen35 model.
                        let ds4_mtp = m
                            .deepseek4()
                            .map(|b| b.weights.mtp_layer.is_some() || b.weights.dspark.is_some())
                            .unwrap_or(false);
                        m.mtp_weights_present =
                            // `mtp_weights_present` is set by the loader to exactly
                            // `qwen35_mtp_head.is_some()`, so the old middle term was
                            // redundant AND required the daemon to name an arch crate it
                            // does not depend on.
                            ds4_mtp || m.mtp_weights_present;

                        // ── Optional DPM stabilization (perf instrumentation) ──
                        //
                        // Pins the GPU at high sclk/mclk so the first `generate`
                        // request doesn't pay the 1-10s DPM ramp from idle. Same
                        // `HIPFIRE_DPM_WARMUP_SECS` env the in-process bench tools
                        // honor (`bench_qwen35_mq4`, `dflash_spec_demo`,
                        // `bench_stream_overlap`); see
                        // `crates/rdna-compute/src/dispatch.rs::dpm_warmup` and
                        // `docs/methodology/perf-benchmarking.md`.
                        //
                        // Runs AFTER weight upload but BEFORE the `loaded` ack so
                        // the contract becomes "loaded means daemon is fully ready
                        // including DPM-pinned." Critical for probe-side timing:
                        // if warmup ran AFTER the ack, the probe would receive
                        // `loaded`, immediately send `generate`, and the daemon
                        // (still warming up in this handler) wouldn't process the
                        // generate until warmup finished — folding the warmup
                        // into the probe-measured TTFT and breaking
                        // `tok_s = total_tokens / wall_ms`. With warmup before the
                        // ack, the probe sees `loaded` only when the daemon is
                        // truly ready, and TTFT measures real prefill alone.
                        //
                        // Default OFF (production daemon load latency unchanged).
                        if let Ok(secs_str) = developer_var("HIPFIRE_DPM_WARMUP_SECS") {
                            if let Ok(secs) = secs_str.parse::<f32>() {
                                if secs > 0.0 {
                                    let _dpm_warmup = rdna_compute::load_phases::span("dpm_warmup");
                                    if let Err(e) = gpu.dpm_warmup(secs) {
                                        eprintln!("[daemon] dpm_warmup failed (non-fatal): {e:?}");
                                    }
                                }
                            }
                        }

                        // ── Continuous batch staging (must be before `loaded` ack) ──
                        // Moved verbatim to `hipfire_loader::batch_staging`: it
                        // constructed Qwen35/Lfm2/EP batch state, which is why the
                        // daemon named arch types here. `LoadedModel` already owned
                        // the typed fields it writes; only the construction leaked.
                        // The scheduler is built here because it lives in
                        // `hipfire-engine`, above the loader.
                        let staging = match hipfire_loader::batch_staging::stage_continuous_batch(
                            &mut m,
                            &mut gpu,
                            parsed_continuous_batch_size,
                            hipfire_generate::batch::vmm_staging_request(parsed_vmm_batch),
                        ) {
                            Ok(staging) => staging,
                            Err(stage_err) => {
                                // Staging left no usable pool (batch unpublished,
                                // sequential drained or untrusted): roll back the
                                // whole staged model and never emit `loaded`.
                                // For the deferred tp>1 path the prior model is
                                // still intact in `model`; only `m` is torn down.
                                let (vram_free, vram_total) =
                                    gpu.hip.get_vram_info().unwrap_or((0, 0));
                                let free_mb = vram_free / (1024 * 1024);
                                let total_mb = vram_total / (1024 * 1024);
                                let rollback_err = match hipfire_loader::unload_model(m, &mut gpu) {
                                    Ok(()) => None,
                                    Err(e) => Some(e),
                                };
                                let mut msg = format!(
                                    "load failed: continuous batch staging failed: {stage_err}. GPU: {}{}",
                                    gpu_memory_text(&gpu, free_mb, total_mb),
                                    no_model_loaded_suffix(model.is_some())
                                );
                                if let Some(rb) = rollback_err {
                                    msg.push_str(&format!(
                                        "; staged-model rollback also failed: {rb}"
                                    ));
                                }
                                emit_uncorrelated_error(
                                    &mut stdout,
                                    None,
                                    &msg,
                                    "gpu",
                                    false,
                                    false,
                                );
                                let _ = stdout.flush();
                                continue;
                            }
                        };
                        // FIX #1 (deferred model retirement, after staging):
                        // the new model is fully staged — NOW retire the
                        // prior model before publishing. Single-GPU/pp models
                        // are unloaded eagerly above; this branch handles
                        // deferred tp>1 EP loads. Prior PFlash drafter is
                        // part of that prior model, so tear it down first in
                        // the same drafter-before-unload order used elsewhere.
                        //
                        // Transactional: if prior unload fails, do NOT install
                        // or emit `loaded` for the new model. Explicitly unload
                        // the newly built model, clear associated fresh state,
                        // and emit a hard error covering prior failure and any
                        // rollback failure.
                        if defer_prior_unload {
                            if let Some(mut pf) = pflash_state.take() {
                                if let Some(mut dg) = pflash_drafter_gpu.take() {
                                    dg.bind_thread_or_warn();
                                    pf.unload_drafter(&mut dg); // sibling-device drafter: free on its own handle, then drop
                                    gpu.bind_thread_or_warn();
                                } else {
                                    pf.unload_drafter(&mut gpu);
                                }
                            }
                            pflash_cfg = None;
                            let prior_unload = if let Some(old) = model.take() {
                                hipfire_loader::unload_model(old, &mut gpu)
                            } else {
                                Ok(())
                            };
                            if !ep_deferred_may_publish(&prior_unload) {
                                let prior_err = prior_unload
                                    .err()
                                    .unwrap_or_else(|| "prior unload failed".to_string());
                                // Roll back the newly built EP model — GpuTensor
                                // has no Drop; must free explicitly.
                                let rollback_err = match hipfire_loader::unload_model(m, &mut gpu) {
                                    Ok(()) => None,
                                    Err(e) => Some(e),
                                };
                                // model stays None; pflash already cleared above.
                                resident_kv = None;
                                let msg = ep_deferred_handoff_error_message(
                                    &prior_err,
                                    rollback_err.as_deref(),
                                );
                                emit_uncorrelated_error(
                                    &mut stdout,
                                    None,
                                    &msg,
                                    "gpu",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        }
                        let staged_batch_capable = staging.capable;
                        let staged_batch_scheduler = staging.capable.then(|| {
                            ContinuousBatchScheduler::new(staging.slots, staging.lane_capacity)
                        });
                        let staged_ep_batch = staging.ep;
                        let staged_ep_slots = staging.ep_slots;
                        let staged_ep_lane_cap = staging.ep_lane_cap;
                        continuous_batch_size = if staged_batch_capable {
                            parsed_continuous_batch_size
                        } else {
                            1
                        };
                        batch_scheduler = staged_batch_scheduler;
                        vmm_batch = if staging.vmm { parsed_vmm_batch } else { None };
                        // `cache_capable` is the daemon's prompt-cache source of truth.
                        // Gemma 4 (13) and Qwen4 (16) only with their attached
                        // session cache (`ArchModel::session_cache_attached`).
                        let cache_capable = match m.arch_id {
                            5 | 6 | 9 | 10 | 12 | 14 => true,
                            13 | 16 => m
                                .state
                                .as_deref()
                                .is_some_and(|state| state.session_cache_attached()),
                            _ => false,
                        };
                        let retry_reset_eligible = model_retry_reset_eligible(m.arch_id);
                        let continuous_batch_capable = staged_batch_capable;
                        let reasoning_contract = hipfire_loader::carrier_for(m.arch_id)
                            .map(|c| c.caps().reasoning_contract.wire_name())
                            .unwrap_or("unsupported");
                        // Contract-gated effort probe; failures are logged once and acked.
                        let (
                            reasoning_effort_native,
                            reasoning_efforts_json,
                            reasoning_effort_probe_error_json,
                        ) = effort_probe::loaded_ack_parts(
                            reasoning_contract,
                            m.tokenizer.as_ref(),
                            m.chat_template.as_deref(),
                            m.chat_template_source.as_deref(),
                        );
                        let backend_reason_json = serde_json::to_string(&pending_kv_diag.reason)
                            .expect("backend reason is serializable");
                        let backend_warning_json = serde_json::to_string(&pending_kv_diag.warning)
                            .expect("backend warning is serializable");
                        // Publish resident KV metadata with the loaded ACK.
                        resident_kv = Some(ResidentKvDiag {
                            mode: Some(seq_kv.to_owned()),
                            ..pending_kv_diag.clone()
                        });
                        // One INFO line: what is on/off and why, from the
                        // engine's own decision notes plus state held here.
                        let summary = load_summary::summarize(
                            &hipfire_config::load_summary::take(),
                            &load_summary::SummaryState {
                                model: std::path::Path::new(path)
                                    .file_name()
                                    .and_then(|name| name.to_str())
                                    .unwrap_or(path),
                                arch,
                                kv_mode: seq_kv,
                                kv_backend: &pending_kv_diag.backend,
                                kv_request: &pending_kv_diag.request,
                                max_seq,
                                bound: seq_bound,
                                model_ctx,
                                card_cap,
                                speculator: m.speculator.as_ref().map(|s| s.name()),
                                pp: m.pp,
                                ep: staged_ep_batch,
                            },
                        );
                        eprintln!("{}", summary.line);
                        // Load ack reports effective storage, not only the request.
                        if staged_ep_batch {
                            emit_load_phases();
                            let ack = format!(
                                r#"{{"type":"loaded","arch":"{}","dim":{},"layers":{},"vocab":{},"vl":{},"pp":{},"audio":{},"video":{},"reasoning_contract":"{}","reasoning_effort_native":{},"reasoning_efforts":{},"reasoning_effort_probe_error":{},"cache_capable":{},"retry_reset_eligible":{},"continuous_batch_capable":{},"continuous_batch_slots":{},"continuous_batch_lane_capacity":{},"continuous_batch_parallelism":"expert_parallel","continuous_batch_rank_count":4,"continuous_batch_reduce":"peer_rooted_f32","kv_backend_request":"{}","kv_backend":"{}","kv_backend_reason":{},"kv_backend_legacy":{},"kv_backend_warning":{},"max_seq":{},"max_seq_bound":"{}","max_seq_reason":{},"model_ctx":{},"card_cap":{},"kv_mode":"{}"}}"#,
                                arch,
                                dim,
                                layers,
                                vocab,
                                vl,
                                m.pp,
                                audio,
                                video,
                                reasoning_contract,
                                reasoning_effort_native,
                                reasoning_efforts_json,
                                reasoning_effort_probe_error_json,
                                cache_capable,
                                retry_reset_eligible,
                                continuous_batch_capable,
                                staged_ep_slots,
                                staged_ep_lane_cap,
                                pending_kv_diag.request,
                                pending_kv_diag.backend,
                                backend_reason_json,
                                pending_kv_diag.legacy,
                                backend_warning_json,
                                max_seq,
                                seq_bound,
                                seq_reason_json,
                                serde_json::to_string(&model_ctx).unwrap(),
                                serde_json::to_string(&card_cap).unwrap(),
                                seq_kv,
                            );
                            let ack = load_summary::ack_with_summary(ack, &summary);
                            let _ = writeln!(stdout, "{ack}");
                        } else {
                            let ack = format!(
                                r#"{{"type":"loaded","arch":"{}","dim":{},"layers":{},"vocab":{},"vl":{},"pp":{},"audio":{},"video":{},"reasoning_contract":"{}","reasoning_effort_native":{},"reasoning_efforts":{},"reasoning_effort_probe_error":{},"cache_capable":{},"retry_reset_eligible":{},"continuous_batch_capable":{},"kv_backend_request":"{}","kv_backend":"{}","kv_backend_reason":{},"kv_backend_legacy":{},"kv_backend_warning":{},"max_seq":{},"max_seq_bound":"{}","max_seq_reason":{},"model_ctx":{},"card_cap":{},"kv_mode":"{}","mtp_mode_effective":"{}"}}"#,
                                arch,
                                dim,
                                layers,
                                vocab,
                                vl,
                                m.pp,
                                audio,
                                video,
                                reasoning_contract,
                                reasoning_effort_native,
                                reasoning_efforts_json,
                                reasoning_effort_probe_error_json,
                                cache_capable,
                                retry_reset_eligible,
                                continuous_batch_capable,
                                pending_kv_diag.request,
                                pending_kv_diag.backend,
                                backend_reason_json,
                                pending_kv_diag.legacy,
                                backend_warning_json,
                                max_seq,
                                seq_bound,
                                seq_reason_json,
                                serde_json::to_string(&model_ctx).unwrap(),
                                serde_json::to_string(&card_cap).unwrap(),
                                seq_kv,
                                m.mtp_mode,
                            );
                            let ack = load_summary::ack_with_summary(ack, &summary);
                            let ack = hipfire_generate::batch::vmm_loaded_ack(
                                ack,
                                &mut m,
                                vmm_batch,
                                staging.slots,
                                staging.row_budget,
                                &mut gpu,
                            );
                            emit_load_phases();
                            let _ = writeln!(stdout, "{ack}");
                        }
                        // ── PFlash drafter load (Phase 4.0) ──────────────
                        //
                        // Only attempt when mode != off AND a drafter path
                        // was provided. Failures here are NON-FATAL: log
                        // the reason and continue with PFlash disabled so
                        // the operator gets a clear "model is up, but
                        // compression isn't" signal rather than losing
                        // the entire session.
                        //
                        // EP guard (load_tp > 1): the EP path serves through
                        // `hipfire_generate::qwen::generate_ep`, which bypasses PFlash entirely (no
                        // EP server reads PFlash drafters).
                        // Loading a drafter here would just pin GPU memory it
                        // never reads until unload, so skip the load outright.
                        // Warn once if the operator actually supplied a drafter
                        // so the silent no-op is visible.
                        if load_tp > 1 {
                            if pflash_drafter.is_some() && pflash_mode_str != "off" {
                                eprintln!(
                                    "[pflash] WARN: ignoring PFlash drafter on EP (tp={}) model \
                                     — hipfire_generate::qwen::generate_ep bypasses PFlash; drafter would only waste GPU memory",
                                    load_tp
                                );
                            }
                        } else if let Some(ref pf_drafter_path) = pflash_drafter {
                            if pflash_mode_str != "off" {
                                if let Some(ref reason) = pflash_load_err {
                                    let _ = writeln!(
                                        stdout,
                                        r#"{{"type":"pflash_load_failed","reason":"invalid load param: {}"}}"#,
                                        reason.replace('"', "'")
                                    );
                                    let _ = stdout.flush();
                                    model = Some(m);
                                    batch_poisoned = None;
                                    continue;
                                }
                                let pf_cfg = hipfire_pflash::pflash::PflashConfig {
                                    mode: hipfire_pflash::pflash::PflashMode::parse(
                                        &pflash_mode_str,
                                    )
                                    .unwrap_or(hipfire_pflash::pflash::PflashMode::Off),
                                    threshold_tokens: pflash_threshold,
                                    keep_ratio: pflash_keep_ratio,
                                    alpha: pflash_alpha,
                                    min_keep_tokens: pflash_min_keep,
                                    sink_tokens: pflash_sink,
                                    recent_tokens: pflash_recent,
                                    block_size: pflash_block,
                                    profile: pflash_profile,
                                    drafter_path: Some(pf_drafter_path.clone()),
                                    sparse_threshold: pflash_sparse_threshold,
                                };
                                let mut pf_state =
                                    hipfire_pflash::pflash::PflashState::new(&pf_cfg);
                                // Pull the target tokenizer out of the loaded model
                                // for the compat check. Both Qwen3.5 and plain
                                // Qwen3 paths expose `tokenizer` on LoadedModel.
                                let tgt_tok_ref = m.tokenizer.as_ref();
                                if let Some(tok) = tgt_tok_ref {
                                    let pf_max_kv = max_seq.max(2048);
                                    // Hetero: when prefill_drafter_device >= 0 and isn't
                                    // device 0 (target), allocate a sibling Gpu handle so
                                    // drafter weights/KV/scratch live on the secondary
                                    // card. Compress output is host-side, so decode stays
                                    // on target. -1 / 0 => share target gpu (unchanged).
                                    let mut sibling: Option<rdna_compute::Gpu> = None;
                                    if pflash_drafter_device > 0 {
                                        match rdna_compute::Gpu::init_with_device(
                                            pflash_drafter_device,
                                        ) {
                                            Ok(g) => sibling = Some(g),
                                            Err(e) => {
                                                let _ = writeln!(
                                                    stdout,
                                                    r#"{{"type":"pflash_load_failed","reason":"drafter device {} init: {}"}}"#,
                                                    pflash_drafter_device,
                                                    e.to_string().replace('"', "'")
                                                );
                                            }
                                        }
                                    }
                                    let dg: &mut rdna_compute::Gpu =
                                        sibling.as_mut().unwrap_or(&mut gpu);
                                    dg.bind_thread_or_warn();
                                    match hipfire_pflash::pflash::load_drafter(
                                        &mut pf_state,
                                        dg,
                                        std::path::Path::new(pf_drafter_path),
                                        tok,
                                        pf_max_kv,
                                    ) {
                                        Ok(()) => {
                                            eprintln!("[pflash] LOADED drafter={} dev={} mode={} compat={} keep={} thr={}",
                                                pf_drafter_path, pflash_drafter_device, pflash_mode_str,
                                                pf_state.tokenizer_compat, pflash_keep_ratio, pflash_threshold);
                                            let _ = writeln!(
                                                stdout,
                                                r#"{{"type":"pflash","mode":"{}","drafter":"{}","drafter_device":{},"tokenizer_compat":{},"keep_ratio":{},"threshold":{}}}"#,
                                                pflash_mode_str,
                                                pf_drafter_path,
                                                pflash_drafter_device,
                                                pf_state.tokenizer_compat,
                                                pflash_keep_ratio,
                                                pflash_threshold
                                            );
                                            pflash_state = Some(pf_state);
                                            pflash_cfg = Some(pf_cfg);
                                            pflash_drafter_gpu = sibling; // persist sibling across requests (None if shared)
                                        }
                                        Err(e) => {
                                            eprintln!("[pflash] LOAD FAILED: {}", e);
                                            let _ = writeln!(
                                                stdout,
                                                r#"{{"type":"pflash_load_failed","reason":"{}"}}"#,
                                                e.to_string().replace('"', "'")
                                            );
                                        }
                                    }
                                } else {
                                    let _ = writeln!(
                                        stdout,
                                        r#"{{"type":"pflash_load_failed","reason":"target tokenizer unavailable"}}"#
                                    );
                                }
                            }
                        }

                        model = Some(m);
                        batch_poisoned = None;
                    }
                    Err(e) => {
                        let (vram_free, vram_total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
                        let free_mb = vram_free / (1024 * 1024);
                        let total_mb = vram_total / (1024 * 1024);
                        // serde-escape: raw HipError debug contains { } and "
                        // which corrupt the JSONL protocol if interpolated raw.
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &format!(
                                "load failed: {e}. GPU: {}{}",
                                gpu_memory_text(&gpu, free_mb, total_mb),
                                no_model_loaded_suffix(model.is_some())
                            ),
                            "gpu",
                            false,
                            false,
                        );
                    }
                }
                let _ = stdout.flush();
            }
            "generate" => {
                let gen_attempt_id = match require_wire_attempt_id(msg.get("attempt_id")) {
                    Ok(id) => id,
                    Err(reason) => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            msg.get("id").and_then(|v| v.as_str()),
                            &format!("generate {reason}"),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                set_active_attempt_id(gen_attempt_id);
                let _attempt_guard = ActiveAttemptGuard;
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("0");
                let singleton_handoff = singleton_transfer_from_message.is_some();
                let admission = match admission_from_message {
                    Some(admission) => admission,
                    None => {
                        eprintln!(
                            "[batch] generate dispatch missing reader admission id={} attempt_id={}",
                            id, gen_attempt_id
                        );
                        emit_uncorrelated_error(
                            &mut stdout,
                            Some(id),
                            "generate missing internal batch admission token",
                            "internal",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                // An embedding checkpoint has no generation route: its carrier
                // answers only `embed`. Refuse here — AFTER attempt_id and the
                // admission token are bound, so the refusal is a keyed
                // `emit_batch_admission_error` the caller's pending entry
                // routes (an attempt-0 uncorrelated error would land on a
                // channel no generate caller waits on and the request would
                // hang). The mirror of `embed`'s refusal and of
                // `img_generate`'s non-diffusion refusal below; reaching the
                // generic body instead downcasts the wrong bundle and panics.
                if model
                    .as_ref()
                    .is_some_and(|m| !hipfire_loader::wire_verb(m.arch_id).is_generate())
                {
                    emit_batch_admission_error(
                        &mut stdout,
                        id,
                        gen_attempt_id,
                        admission,
                        "generate refused: the loaded model is a qwen3.5 embedding checkpoint — use the `embed` verb",
                        "validation",
                        false,
                        false,
                    );
                    continue;
                }

                #[cfg(feature = "serve-fault-inject")]
                let _fault_guard =
                    hipfire_generate::common::RequestFaultGuard::arm_from_request(&msg);
                // Experimental slot backend dispatches before the ordinary model path.
                // This preserves byte-for-byte default behavior when absent, and in experimental
                // mode owns exactly one SlotEngine/weight set with no ordinary-model fallback.
                // Spawn a bounded request worker so the main loop continues accepting independent generates.
                if let Some(slot) = slot_backend.clone() {
                    // Bound thread creation BEFORE spawning (spec §5.3:
                    // "a guard acquired inside an already spawned thread
                    // does not bound thread creation"). The worker's own
                    // acquire_guard remains the hard atomic bound; this
                    // pre-check keeps an arrival burst from spawning a
                    // thread per rejected request.
                    if slot.active_count() >= 32 {
                        hipfire_engine::emit::emit_active_attempt_error(
                            &mut stdout,
                            Some(id),
                            "too many concurrent slot requests (bounded worker limit hit)",
                            // Overload, not validation: slot saturation is a
                            // transient capacity condition → 429 + Retry-After.
                            "overload",
                            true,
                            false,
                        );
                        let _ = stdout.flush();
                        batch_clear_terminal(id, gen_attempt_id);
                        continue;
                    }
                    let msg_clone = msg.clone();
                    let id_owned = id.to_string();
                    let slot_clone = slot.clone();
                    let admission = admission;
                    // Bounded: the backend's active counter bounds concurrency;
                    // engine itself is the only GPU worker, so workers serialize on engine submit.
                    std::thread::spawn(move || {
                        // Each worker uses its own stdout handle; every event is one serde JSON line.
                        let mut worker_stdout = std::io::stdout();
                        let _ = slot_clone.handle_generate(
                            &msg_clone,
                            &mut worker_stdout,
                            &id_owned,
                            gen_attempt_id,
                            admission,
                        );
                        let _ = worker_stdout.flush();
                    });
                    continue;
                }
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        emit_batch_admission_error(
                            &mut stdout,
                            id,
                            gen_attempt_id,
                            admission,
                            "no model loaded",
                            "validation",
                            false,
                            false,
                        );
                        continue;
                    }
                };
                // Fail-closed: a diffusion checkpoint trunk
                // (arch 40 or 45) is a loadable component, never a chat model.
                // Text generate on it must refuse — not fall through to the
                // token path with a vocab-0 skeleton tokenizer.
                if hipfire_loader::img_route(m.arch_id).is_diffusion() {
                    emit_batch_admission_error(
                        &mut stdout,
                        id,
                        gen_attempt_id,
                        admission,
                        "text generate refused: loaded model is a diffusion checkpoint (arch 40/45) — use img_generate",
                        "validation",
                        false,
                        false,
                    );
                    continue;
                }
                if let Some(reason) = batch_poisoned.as_ref() {
                    emit_batch_admission_error(
                        &mut stdout,
                        id,
                        gen_attempt_id,
                        admission,
                        &format!(
                            "continuous batch GPU state poisoned; unload/reload required: {reason}"
                        ),
                        "gpu",
                        false,
                        false,
                    );
                    continue;
                }
                // Sticky GPU fault (700/719) latched by an arch op: the HIP
                // context is dead, so any prefill/decode would fail the same
                // way. Fail fast instead of burning a doomed prefill per
                // request — the gate saw the same 719 repeat across requests
                // after a mid-decode realign fault. Only a process restart
                // re-establishes a live context (model reload does not reset
                // the primary context), so the latch is never cleared here.
                if let Some(poison) = hipfire_runtime::reset_core::gpu_poison() {
                    emit_batch_admission_error(
                        &mut stdout,
                        id,
                        gen_attempt_id,
                        admission,
                        &format!(
                            "GPU context dead after sticky HipError({}) at {}; process restart required",
                            poison.code, poison.site,
                        ),
                        "gpu",
                        false,
                        false,
                    );
                    continue;
                }

                // Whole-request PLE serving guard (Qwen4 with PLE rows; `None`
                // otherwise): held from admission to the end of this arm, so
                // background PLE prewarm readers pause for setup, prefill and
                // decode of the ordinary, speculative, bench and serve routes
                // and for the whole continuous-batch drive below (lane
                // requests that join it from the inbox included). RAII: every
                // `continue`, error, cancel and panic unwind drops it.
                let _ple_serving = m.ple_serving_guard();
                // Fresh terminal-control transaction for ordinary generates.
                // A think-barrier message instead restores the exact singleton
                // snapshot captured before the driver's outer guard dropped.
                if let Some(transfer) = singleton_transfer_from_message.take() {
                    if !adopt_singleton_transfer(id, gen_attempt_id, transfer) {
                        emit_uncorrelated_error(
                            &mut stdout,
                            Some(id),
                            "singleton handoff ownership is invalid",
                            "internal",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                } else {
                    activate_terminal_control(id, gen_attempt_id);
                }
                let _terminal_control_guard = TerminalControlGuard;
                let mut batch_scope = if singleton_handoff {
                    BatchAttemptScope::enter_singleton(gen_attempt_id)
                } else {
                    BatchAttemptScope::enter_for_generation(id, gen_attempt_id, admission)
                };
                let _batch_cleanup = BatchTerminalCleanup {
                    id: id.to_owned(),
                    attempt_id: gen_attempt_id,
                    admission: (!singleton_handoff).then_some(admission),
                };
                gpu.replay.begin_replay_observation_window();
                let prompt = msg
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Hello");
                let prompt_norm = hipfire_runtime::tokenizer::maybe_normalize_prompt(prompt);
                let prompt: &str = &prompt_norm;
                if hipfire_runtime::config::get().prompt_token_heat {
                    if let Some(tok) = m.tokenizer.as_ref() {
                        tok.dump_prompt_heat(prompt);
                    }
                }
                let system = msg.get("system").and_then(|v| v.as_str());
                let image = msg.get("image").and_then(|v| v.as_str());
                let image_base64 = msg.get("image_base64").and_then(|v| v.as_str());

                // Structured-tools + structured-messages support (Phase 1 of
                // Jinja-everywhere migration). When present, both fields are
                // routed through `JinjaChatFrame::render_messages` so the
                // model sees the upstream template's `{% if tools %}` and
                // multi-turn branches (XML/JSON tool-call format per arch,
                // tool-response role mapping, etc.).
                //
                // Backward compat: when neither is present, legacy
                // `prompt`+`system` continues to drive a synthesized
                // [system?, user] slice — byte-identical to today's
                // `JinjaChatFrame::render()` single-turn path.
                //
                // Parse errors emit a structured error event and skip the
                // request (rather than silently dropping the fields).
                let tools_json: Option<Vec<serde_json::Value>> = match msg.get("tools") {
                    Some(v) => match serde_json::from_value::<Vec<serde_json::Value>>(v.clone()) {
                        Ok(t) => (!t.is_empty()).then_some(t),
                        Err(e) => {
                            hipfire_generate::dense::emit_active_attempt_error(
                                &mut stdout,
                                Some(id),
                                &format!(
                                    "invalid tools field: {}",
                                    e.to_string().replace('"', "'"),
                                ),
                                "validation",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    },
                    None => None,
                };
                // Each message's content is normalized like `prompt` (shared
                // with the VMM batch admission so both render identical bytes).
                let messages_history: Option<Vec<hipfire_runtime::prompt_frame::Message>> =
                    match hipfire_engine::prompt::parse_generate_messages(&msg) {
                        Ok(m) => m,
                        Err(e) => {
                            hipfire_generate::dense::emit_active_attempt_error(
                                &mut stdout,
                                Some(id),
                                &format!("invalid messages field: {}", e.replace('"', "'")),
                                "validation",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    };
                // OpenAI `stop` (string or array form); an invalid value is rejected.
                let Some(stop_seqs) = hipfire_generate::ar::stop_or_reject(&mut stdout, id, &msg)
                else {
                    continue;
                };

                // Sampling defaults (shared with the batch and multi-slot
                // routes via `resolve_temp_top_p`): explicit request field >
                // the .hfq-baked author recommendation (m.rec_temperature /
                // m.rec_top_p, from HfqFile::recommended_sampling) > the
                // carrier's arch ladder (qwen35 `temp=0.3, top_p=0.8`;
                // DeepSeek V4 Flash `1.0/1.0`). The CLI's curated registry
                // `recommended_settings` reach this handler as explicit
                // request fields (CLI explicit-send guard).
                let defaults = hipfire_loader::carrier_for(m.arch_id)
                    .map(|c| c.sampling_defaults())
                    .unwrap_or_default();
                let (temp, top_p) = hipfire_engine::scheduler::resolve_temp_top_p(
                    &msg,
                    m.arch_id,
                    m.rec_temperature,
                    m.rec_top_p,
                );
                let max_tokens = msg
                    .get("max_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(hipfire_engine::scheduler::DEFAULT_GENERATE_MAX_TOKENS);
                let _fit = request_guards::fit_max_tokens(&msg);
                // CACTUS acceptance-boost δ — OPT-IN (request `cactus_delta`), 0.0
                // default = lossless/distribution-preserving. >0 is deliberately lossy
                // (higher acceptance τ, KL-bounded distortion) and applies only to a
                // CACTUS-capable sampled verify (deepseek4 DSpark / qwen35 DFlash);
                // other drafters ignore it. Never a default.
                // OpenAI logprobs. The gateway has already validated the range
                // (0..=20, requires logprobs=true) and rejects rather than clamps,
                // so a value arriving here is well-formed. None leaves every token
                // envelope byte-identical to before.
                let logprobs_top_k: Option<usize> = msg
                    .get("logprobs")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                    .then(|| {
                        msg.get("top_logprobs")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0) as usize
                    })
                    .filter(|k| *k > 0);
                let cactus_delta = msg
                    .get("cactus_delta")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                // Default 1.0 (off). Matches llama.cpp `--repeat-penalty 1.0`
                // and HF transformers `generate(repetition_penalty=1.0)`
                // defaults. The prior 1.3 default suppressed legitimately
                // repeated formatting tokens (e.g. `' **'` for bullets,
                // indentation patterns) on multi-step reasoning prompts,
                // pushing structured chain-of-thought trajectories off the
                // model's well-trained path into a self-doubt / number-
                // hallucination attractor on 9B Qwen3.5 at greedy decode.
                // Root cause writeup: issue #258 comment "Bug B root cause"
                // and docs/investigations/2026-05-15-9b-reasoning-loop/.
                // Clients can still opt in to a non-1.0 value per request.
                // LFM2.5-MoE (arch_id 11): Liquid's card recommends
                // repetition_penalty=1.05; default to it (others stay 1.0/off).
                let default_repeat_penalty = defaults.repeat_penalty;
                // Accept HF-style `repetition_penalty` as a request ALIAS for our
                // `repeat_penalty` field, used only when the canonical key is
                // absent. (OpenAI/HF clients send `repetition_penalty`.)
                let repeat_penalty = msg
                    .get("repeat_penalty")
                    .or_else(|| msg.get("repetition_penalty"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(default_repeat_penalty) as f32;
                // OpenAI-compatible `reasoning_effort` (also accept our custom
                // `thinking_mode` alias) — ThinkMode is consumed by arch_id=9
                // (DeepSeek DSML thinking), while raw `reasoning_effort` is
                // plumbed to Jinja as `reasoning_effort` for Qwen3.8 (arch 5/6).
                // `auto`/absent stays truly undefined (not null) so the Qwen3.8
                // template defaults to `xhigh` only when undefined; unsupported
                // values are preserved verbatim (template raises) rather than
                // being silently remapped to low/medium/xhigh.
                let think_mode = msg
                    .get("reasoning_effort")
                    .or_else(|| msg.get("thinking_mode"))
                    .and_then(|v| v.as_str())
                    .map(ThinkMode::from_str)
                    .unwrap_or(ThinkMode::NonThink);
                // Raw effort for Jinja `reasoning_effort` — exact, no lowercasing,
                // no empty-filter. `auto`/absent => undefined, `none`/`off`/`chat`
                // => disabled+undefined, all other exact strings (including
                // empty, case-mismatched) pass verbatim so the Qwen3.8 template
                // raises rather than silently normalizing or falling back.
                let raw_reasoning_effort: Option<&str> = msg
                    .get("reasoning_effort")
                    .or_else(|| msg.get("thinking_mode"))
                    .and_then(|v| v.as_str());
                // Typed thinking flag: when present it is authoritative for Jinja
                // enablement. HTTP normalization always sends it; direct JSONL
                // clients may omit it and fall back to legacy effort/cap inference.
                let thinking_enabled: Option<bool> =
                    msg.get("thinking_enabled").and_then(|v| v.as_bool());
                let repeat_window = msg
                    .get("repeat_window")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128) as usize;
                // OpenAI subtractive penalties. The CLI forwards raw
                // `presence_penalty`/`frequency_penalty` (0.0 = off). Unlike the
                // recency-weighted multiplicative `repeat_penalty`, these are
                // flat across the (now long) window, which is what breaks the
                // block-level repetition loops on long reasoning generations.
                // Clamp negatives to 0 (negative would REWARD repetition).
                // Fallback ladder: explicit request `presence_penalty` >
                // .hfq-baked `m.rec_presence_penalty` > 0.0 (off). The .hfq's
                // generation_config does not carry presence_penalty today, so
                // m.rec_presence_penalty is always None on the load path; the
                // field is wired so a curated registry card value still flows in
                // as an explicit request field (CLI explicit-send guard). presence_penalty IS honored by the sampler.
                let presence_penalty = (msg
                    .get("presence_penalty")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(m.rec_presence_penalty.unwrap_or(0.0) as f64)
                    as f32)
                    .max(0.0);
                let frequency_penalty = (msg
                    .get("frequency_penalty")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32)
                    .max(0.0);
                // Request-driven top_k / min_p (W7 P2). Fallback ladder:
                // explicit request field > .hfq/registry-baked rec_top_k /
                // rec_min_p > None. None reproduces the legacy sampler exactly
                // (top-K candidate gather of 20, no min-p cut). top_k <= 0 is
                // treated as "unset" (None) so 0 never collapses to argmax.
                let top_k: Option<u32> = msg
                    .get("top_k")
                    .and_then(|v| v.as_u64())
                    .map(|k| k as u32)
                    .or_else(|| m.rec_top_k.map(|k| k as u32))
                    .filter(|&k| k > 0);
                let min_p: Option<f32> = msg
                    .get("min_p")
                    .and_then(|v| v.as_f64())
                    .map(|p| p as f32)
                    .or(m.rec_min_p)
                    .filter(|&p| p > 0.0);
                // Experimental: inject a nudge string at a specific generated-
                // token count. The nudge tokens get forward-fed through the KV
                // cache so the model "sees" them as part of its own trajectory,
                // and are emitted to stdout so the client stream includes them.
                // Used to test whether telling a thinking model "time's up"
                // gets it to close </think> and commit to an answer.
                //
                // GATED: off by default. The feature has a real UX hazard — if
                // the alert fires after </think> has already closed, the nudge
                // leaks into the visible answer. Only honor the params when the
                // operator has explicitly opted in via config
                // (`experimental_budget_alert: true` → HIPFIRE_EXPERIMENTAL_
                // BUDGET_ALERT=1 set by the CLI). Research use only; not a
                // stable contract.
                let experimental_ok = hipfire_runtime::config::get().experimental_budget_alert;
                let budget_alert_at_tok = if experimental_ok {
                    msg.get("budget_alert_at_tok")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize
                } else {
                    0
                };
                let budget_alert_text = if experimental_ok {
                    msg.get("budget_alert_text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                } else {
                    String::new()
                };
                // Budget for tokens emitted INSIDE the model's <think>...</think>
                // block. 0 = uncapped (model thinks until it naturally closes).
                // This is an independent explicit cap (never derived from
                // effort) — 0 means uncapped, 1 means immediately closed.
                // Legacy direct JSONL without `thinking_enabled` still infers
                // disable from `max_think==1` via the helper's fallback, but
                // new clients send `thinking_enabled` as authority and keep
                // `max_think_tokens` independent.
                //
                // When the cap is reached the generate path splices a think
                // close through the same KV-write + commit path as a sampled
                // token, and the model answers with the remaining max_tokens
                // budget. Qwen4 (AR and MTP) splices the template's own
                // "\n</think>\n\n"; the Qwen3.5 AR/DFlash loops splice
                // `HIPFIRE_THINK_CONTINUATION` (default "</think>\n\n"); the
                // multi-slot engine masks to the close instead. The Qwen3.5
                // dense-TP and MoE-EP producers still fail the request
                // ("think token budget exceeded").
                let max_think_tokens = msg
                    .get("max_think_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                // Derive Jinja `enable_thinking` and `reasoning_effort` via
                // pure helper. `thinking_enabled` is authoritative when
                // present; legacy effort/max_think inference is preserved
                // only for direct old JSONL clients that omit it.
                let (enable_thinking_jinja, reasoning_effort_jinja) =
                    qwen_jinja_reasoning(thinking_enabled, raw_reasoning_effort, max_think_tokens);
                // Controls the ChatML framing after the assistant role header.
                // Propagated through both text and Qwen3.5-VL paths.
                let assistant_prefix = match msg
                    .get("assistant_prefix")
                    .and_then(|v| v.as_str())
                    .unwrap_or("plain")
                {
                    "open_think" => hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink,
                    "closed_think" => hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink,
                    _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                };

                let has_image = image_base64.is_some() || image.is_some();
                let vision_route = hipfire_loader::vision_route(m.arch_id);
                // Covers qwen35-vl (arch 5/6 bundle), dots-ocr (arch 8) AND
                // lfm2-vl (arch-11 bundle) in one declared-capability probe.
                let has_vl = m.has_vision_encoder();
                let tp = m.ep.is_some();
                let tp_penalties =
                    m.ep.as_ref()
                        .is_some_and(|ep| ep.inner.applies_request_penalties());
                if request_guards::refuse_ep_request(
                    &mut stdout,
                    id,
                    &msg,
                    tp,
                    tp_penalties,
                    has_image,
                ) {
                    continue;
                }

                if has_image {
                    let _ =
                        batch_transfer_abort_to_singleton_and_clear(id, gen_attempt_id, admission);
                    batch_scope.rebind_for(gen_attempt_id);
                }
                if has_image && !has_vl {
                    refuse_image_without_vision(&mut stdout, id, vision_gated_off.as_deref());
                } else if has_image && has_vl {
                    // DEFENSIVE: VL is single-image, single-turn only. The
                    // CLI rejects images in non-last turns, but a raw
                    // JSONL client could send a second image on turn 2+.
                    // If seq_pos > 0 here, a previous conversation's KV
                    // entries are live — running vision_forward and
                    // splicing visual tokens into that context would
                    // produce garbage. Force a reset so VL always starts
                    // from a clean KV state.
                    //
                    // Must mirror the "reset" command handler (line ~2098).
                    // VL runs on qwen35-vl (arch_id 5|6), dots-ocr (arch_id 8)
                    // and lfm2-vl (arch_id 11); other arch states are None
                    // here — but clear them anyway for defense-in-depth in
                    // case a future arch adds VL support.
                    if m.seq_pos > 0 {
                        eprintln!("[daemon/vl] non-zero seq_pos ({}) at VL dispatch — resetting conversation", m.seq_pos);
                        m.seq_pos = 0;
                        m.conversation_tokens.clear();
                        hipfire_generate::common::free_checkpoints(
                            &mut m.prefill_checkpoints,
                            &mut gpu,
                        );
                        hipfire_generate::common::free_checkpoints(
                            &mut m.dflash_checkpoints,
                            &mut gpu,
                        );
                        // The DFlash checkpoint ring now lives inside the
                        // speculator (m.dflash_checkpoints is vestigial/empty),
                        // so free THAT ring on conversation reset too — else its
                        // GPU snapshots persist until the next prefill-miss.
                        if let Some(s) = m.speculator.as_mut() {
                            if let Err(e) = s.reset(&mut gpu) {
                                hipfire_generate::dense::emit_active_attempt_error(
                                    &mut stdout,
                                    Some(id),
                                    &format!("vision conversation reset failed: {e}"),
                                    "gpu",
                                    true,
                                    false,
                                );
                                continue;
                            }
                        }
                        // qwen35(-vl) recurrent state lives in the bundle
                        // (qwen35 arch bundle, accessed via ArchModel). There is no
                        // `LoadedModel.dn_state` — it was removed as vestigial
                        // (always None); the live DeltaNet state is inside the
                        // bundle. `m.kv_cache` is likewise vestigial on this path.
                        if let Err(e) =
                            hipfire_generate::common::reset_qwen35_recurrent(m, &mut gpu)
                        {
                            hipfire_generate::dense::emit_active_attempt_error(
                                &mut stdout,
                                Some(id),
                                &format!("vision recurrent reset failed: {e}"),
                                "gpu",
                                true,
                                false,
                            );
                            continue;
                        }
                        if let Some(b) = m.llama_mut() {
                            b.kv.compact_offset = 0;
                        }
                        if let Some(b) = m.dots_ocr_mut() {
                            b.state.reset();
                        }
                        if let Some(b) = m.qwen2_mut() {
                            b.state.reset();
                        }
                        if let Some(b) = m.deepseek4_mut() {
                            b.state.reset();
                        }
                        // lfm2-vl (arch 11): KV + conv state live in the
                        // lfm2moe bundle. generate_lfm2_vl cold-resets again
                        // before its own prefill, so this arm is
                        // defense-in-depth parity with the other VL arches —
                        // without it a failed dispatch between guard and
                        // generate body would leave stale state behind.
                        if let Some(b) = m.lfm2moe_mut() {
                            if let Err(e) = b.state.reset(&mut gpu) {
                                hipfire_generate::dense::emit_active_attempt_error(
                                    &mut stdout,
                                    Some(id),
                                    &format!("vision lfm2moe reset failed: {e:?}"),
                                    "gpu",
                                    true,
                                    false,
                                );
                                continue;
                            }
                        }
                        if let Some(ad) = m.kv_adaptive.as_mut() {
                            if let Some(s) = m.state.as_mut() {
                                if s.arch_key() == "qwen35" {
                                    if let Some(kv) = s.kv_cache_mut() {
                                        ad.reset_with_cache(&mut gpu, kv);
                                    } else {
                                        ad.reset();
                                    }
                                } else {
                                    ad.reset();
                                }
                            } else {
                                ad.reset();
                            }
                        }
                    }
                    if image_base64.is_some() && image.is_some() {
                        eprintln!(
                            "[daemon/vl] both image and image_base64 provided — using image_base64"
                        );
                    }
                    let source = if let Some(b64) = image_base64 {
                        if b64.len() > MAX_BASE64_ENCODED_LEN {
                            write_typed_error(
                                &mut stdout,
                                id,
                                &format!(
                                    "image payload exceeds maximum encoded size ({} bytes)",
                                    MAX_BASE64_ENCODED_LEN,
                                ),
                                "validation",
                                false,
                                false,
                            );
                            continue;
                        }
                        ImageSource::Base64(b64)
                    } else {
                        ImageSource::Path(image.unwrap())
                    };
                    // Plan-mandated Phase-1 stopgap (docs/plans/completions_vision.md §2.1):
                    // VL dispatch defaults `max_think_tokens` to 256 when the
                    // client doesn't specify one. Caps runaway thinking
                    // without needing the full `ThinkState` extraction. Text
                    // path keeps unwrap_or(0) — it has different defaults
                    // controlled per-model on the CLI side.
                    let vl_max_think_tokens = if max_think_tokens == 0 {
                        256
                    } else {
                        max_think_tokens
                    };
                    // Same tiered derivation as the text path below: explicit
                    // wire `seed` wins, else attempt key + counter entropy.
                    // Out-of-domain seeds (negative, fractional, non-numeric)
                    // are rejected — never silently treated as unseeded.
                    let Some(client_seed) = client_seed_or_refuse(&mut stdout, id, msg.get("seed"))
                    else {
                        continue;
                    };
                    let vl_request_seed =
                        request_seed_for(&AttemptKey::new(id, gen_attempt_id), client_seed);
                    let params = GenerateVLParams {
                        id,
                        prompt,
                        system_prompt: system,
                        image_source: source,
                        temp,
                        top_p,
                        max_tokens,
                        repeat_penalty,
                        repeat_window,
                        max_think_tokens: vl_max_think_tokens,
                        assistant_prefix,
                        seed: vl_request_seed,
                    };
                    match vision_route {
                        hipfire_loader::VisionRoute::DotsOcr => {
                            hipfire_generate::vision::generate_vl_dots_ocr(
                                m,
                                &mut gpu,
                                &mut stdout,
                                &params,
                            )
                        }
                        hipfire_loader::VisionRoute::Lfm2Vl => {
                            hipfire_generate::vision::generate_lfm2_vl(
                                m,
                                &mut gpu,
                                &mut stdout,
                                &params,
                            )
                        }
                        _ => {
                            hipfire_generate::vision::generate_vl(m, &mut gpu, &mut stdout, &params)
                        }
                    }
                } else {
                    // Per-request PflashConfig: clone the load-time cfg
                    // and apply any per-request overrides from `params`.
                    // None when no drafter was configured at load --
                    // generate() then takes the identity path.
                    //
                    // Out-of-range overrides (keep_ratio outside (0, 1],
                    // block_size == 0) would otherwise reach asserts inside
                    // select_spans / scoring and panic the entire daemon.
                    // Reject the request with an explicit error event so
                    // the client gets a clean signal and the daemon stays up.
                    let mut pf_override_err: Option<String> = None;
                    let pf_cfg_owned = pflash_cfg.as_ref().map(|base| {
                        let mut c = base.clone();
                        if let Some(s) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_compression"))
                            .and_then(|v| v.as_str())
                        {
                            if let Some(m) = hipfire_pflash::pflash::PflashMode::parse(s) {
                                c.mode = m;
                            }
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_threshold"))
                            .and_then(|v| v.as_u64())
                        {
                            c.threshold_tokens = v as usize;
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_keep_ratio"))
                            .and_then(|v| v.as_f64())
                        {
                            let r = v as f32;
                            if !(r > 0.0 && r <= 1.0) {
                                pf_override_err =
                                    Some(format!("prefill_keep_ratio={r} not in (0, 1]"));
                            } else {
                                c.keep_ratio = r;
                            }
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_min_keep"))
                            .and_then(|v| v.as_u64())
                        {
                            c.min_keep_tokens = v as usize;
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_sink"))
                            .and_then(|v| v.as_u64())
                        {
                            c.sink_tokens = v as usize;
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_recent"))
                            .and_then(|v| v.as_u64())
                        {
                            c.recent_tokens = v as usize;
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_block"))
                            .and_then(|v| v.as_u64())
                        {
                            let b = v as usize;
                            if b == 0 {
                                pf_override_err = Some("prefill_block must be > 0".to_string());
                            } else {
                                c.block_size = b;
                            }
                        }
                        c
                    });
                    if let Some(reason) = pf_override_err {
                        hipfire_generate::dense::emit_active_attempt_error(
                            &mut stdout,
                            Some(id),
                            &format!("invalid pflash override: {}", reason.replace('"', "'"),),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    // ── Continuous batch admission (tightened to actual route) ──
                    // EP Qwen35 expert-parallel continuous batch is TP=4, 4×gfx1201;
                    // sequential EP (tp=2|4) serves every request that is not
                    // batch-eligible. Check EP eligibility first so the batch
                    // route wins when staged and requested.
                    let serve_continuous_batch = parse_serve_continuous_batch(&msg);
                    let pflash_active = pf_cfg_owned.as_ref().is_some_and(|c| {
                        !matches!(c.mode, hipfire_pflash::pflash::PflashMode::Off)
                    });
                    let ep_batch_eligible =
                        if !singleton_handoff && batch_scheduler.is_some() && m.ep.is_some() {
                            is_qwen_ep_batch_request_eligible(
                                &msg,
                                m,
                                continuous_batch_size,
                                serve_continuous_batch,
                                pflash_active,
                            )
                        } else {
                            false
                        };
                    if ep_batch_eligible {
                        let _ = batch_transition_to_queued(id, gen_attempt_id, admission);
                        if batch_check_abort(id, gen_attempt_id, admission) {
                            let _scope = BatchAttemptScope::enter_for_generation(
                                id,
                                gen_attempt_id,
                                admission,
                            );
                            hipfire_generate::ar::emit_generation_start(
                                hipfire_generate::ar::GenerationRoute::QwenAr,
                                &mut stdout,
                                id,
                                false,
                            );
                            hipfire_generate::ar::emit_active_route_cancel(&mut stdout, id, 0);
                            batch_clear_terminal_at_generation(id, gen_attempt_id, admission);
                            continue;
                        }
                        let sampling = resolve_batch_sampling(&msg, m);
                        let prompt_owned =
                            batch_single_user_content(&msg).unwrap_or_else(|| prompt.to_string());
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_owned,
                            system,
                            assistant_prefix,
                            m.tokenizer.as_ref().unwrap(),
                            m.chat_template.as_ref(),
                            max_think_tokens,
                            messages_history.as_deref(),
                            enable_thinking_jinja,
                            reasoning_effort_jinja.as_deref(),
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                emit_batch_admission_error(
                                    &mut stdout,
                                    id,
                                    gen_attempt_id,
                                    admission,
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        if started_in_think {
                            let _ = batch_transfer_abort_to_singleton_and_clear(
                                id,
                                gen_attempt_id,
                                admission,
                            );
                        } else {
                            if prompt_tokens.is_empty() || prompt_tokens.len() >= m.max_seq {
                                emit_batch_admission_error(
                                    &mut stdout,
                                    id,
                                    gen_attempt_id,
                                    admission,
                                    "prompt exceeds lane capacity or empty",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                            // Explicit wire `seed` must reach the lane RNG on
                            // the batched route too; out-of-domain values are
                            // rejected loudly, never silently unseeded.
                            let Some(client_seed) =
                                client_seed_or_refuse(&mut stdout, id, msg.get("seed"))
                            else {
                                batch_clear_terminal_at_generation(id, gen_attempt_id, admission);
                                continue;
                            };
                            let pending = BatchPendingRequest {
                                key: AttemptKey::new(id, gen_attempt_id),
                                admission,
                                original_msg: msg.clone(),
                                prompt: prompt_owned.clone(),
                                prompt_tokens: prompt_tokens.clone(),
                                started_in_think,
                                system: system.map(|s| s.to_string()),
                                assistant_prefix,
                                max_think_tokens,
                                max_tokens,
                                client_seed,
                                sampling: sampling.clone(),
                            };
                            if let Some(sched) = batch_scheduler.as_mut() {
                                let enq_ok = sched.enqueue(pending);
                                if !enq_ok {
                                    eprintln!("[batch][EP] duplicate enqueue rejected id={} attempt_id={}; preserving live registry", id, gen_attempt_id);
                                    continue;
                                }
                                {
                                    let _scope = BatchAttemptScope::enter_for_generation(
                                        id,
                                        gen_attempt_id,
                                        admission,
                                    );
                                    hipfire_generate::ar::emit_generation_start(
                                        hipfire_generate::ar::GenerationRoute::QwenAr,
                                        &mut stdout,
                                        id,
                                        false,
                                    );
                                }
                                let drive_res = drive_qwen35_ep_continuous_batch(
                                    sched,
                                    m,
                                    &mut stdout,
                                    &mut inbox,
                                );
                                apply_batch_drive_result(
                                    drive_res,
                                    "[batch][EP]",
                                    &mut batch_scheduler,
                                    &mut continuous_batch_size,
                                    &mut batch_poisoned,
                                );
                            }
                            continue;
                        }
                    }
                    // Sequential EP now serves Qwen35-MoE (tp=2|4) via
                    // `generate_ep`: a request that is not batch-eligible
                    // falls through to the sequential path below instead of
                    // failing closed. The batch route above still wins when
                    // staged and requested with `serve_continuous_batch`.
                    let vmm_route = hipfire_generate::batch::vmm_route_active(vmm_batch, m);
                    // Qwen4 batches only through its staged fn-lanes route.
                    let qwen4_batch = hipfire_loader::continuous_batch_route(m.arch_id)
                        == Some(hipfire_loader::ContinuousBatchRoute::Qwen4);
                    let qwen4_lanes = qwen4_batch && vmm_route;
                    let batch_eligible = if !singleton_handoff
                        && batch_scheduler.is_some()
                        && (!qwen4_batch || qwen4_lanes)
                    {
                        hipfire_generate::batch::batch_request_eligible_for_route(
                            vmm_route,
                            &msg,
                            m,
                            continuous_batch_size,
                            serve_continuous_batch,
                            pflash_active,
                            &mut inbox,
                        )
                    } else {
                        false
                    };
                    if batch_eligible {
                        // Current request was already announced by the reader; promote to Queued.
                        let _ = batch_transition_to_queued(id, gen_attempt_id, admission);
                        // If already aborted, emit cancelled and do not enqueue.
                        if batch_check_abort(id, gen_attempt_id, admission) {
                            let _scope = BatchAttemptScope::enter_for_generation(
                                id,
                                gen_attempt_id,
                                admission,
                            );
                            hipfire_generate::ar::emit_generation_start(
                                if qwen4_lanes {
                                    hipfire_generate::ar::GenerationRoute::Qwen4Ar
                                } else {
                                    hipfire_generate::ar::GenerationRoute::QwenAr
                                },
                                &mut stdout,
                                id,
                                false,
                            );
                            hipfire_generate::ar::emit_active_route_cancel(&mut stdout, id, 0);
                            batch_clear_terminal_at_generation(id, gen_attempt_id, admission);
                            continue;
                        }
                        let sampling = resolve_batch_sampling(&msg, m);
                        // Render prompt once at admission and store tokens/started flag; do not render twice at lane assignment.
                        let prompt_owned =
                            batch_single_user_content(&msg).unwrap_or_else(|| prompt.to_string());
                        // Qwen4 lanes: the driver renders exactly at assignment (its own
                        // template path, not the Qwen3.5 batch render); admission defers
                        // prompt tokens and skips Qwen3.5 length/think validation.
                        let render = if qwen4_lanes {
                            Ok((Vec::new(), false))
                        } else {
                            batch_render_prompt_tokens(
                                &prompt_owned,
                                system,
                                assistant_prefix,
                                m.tokenizer.as_ref().unwrap(),
                                m.chat_template.as_ref(),
                                max_think_tokens,
                                messages_history.as_deref(),
                                enable_thinking_jinja,
                                reasoning_effort_jinja.as_deref(),
                            )
                        };
                        let (prompt_tokens, started_in_think) = match render {
                            Ok(v) => v,
                            Err(e) => {
                                emit_batch_admission_error(
                                    &mut stdout,
                                    id,
                                    gen_attempt_id,
                                    admission,
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        // VMM route: a sampled request with no batch lane in its singleton mode at
                        // this prompt length takes the singleton route before enqueue announces it.
                        let vmm_singleton_now = vmm_route
                            && !hipfire_generate::batch::vmm_request_fits_batch(
                                &msg,
                                m,
                                &sampling,
                                vmm_batch,
                                batch_scheduler.as_ref().map_or(0, |s| s.max_batch),
                                &gpu.arch,
                                prompt_tokens.len(),
                                max_tokens,
                            );
                        if (started_in_think && !vmm_route) || vmm_singleton_now {
                            // Fixed-lane route: think-open prompts are sequential
                            // barriers; VMM route: see above. Transfer any
                            // pre-latched abort exactly once (transfer retires the
                            // batch entry and holds a tombstone until singleton cleanup).
                            let _ = batch_transfer_abort_to_singleton_and_clear(
                                id,
                                gen_attempt_id,
                                admission,
                            );
                            if vmm_singleton_now {
                                // As on the not-eligible path: the sequential producer takes the singleton claim.
                                batch_scope.rebind_for(gen_attempt_id);
                            }
                            // Fall through to sequential generate below (do not enqueue).
                        } else {
                            if !qwen4_lanes
                                && (prompt_tokens.is_empty() || prompt_tokens.len() >= m.max_seq)
                            {
                                emit_batch_admission_error(
                                    &mut stdout,
                                    id,
                                    gen_attempt_id,
                                    admission,
                                    "prompt exceeds lane capacity or empty",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                            // Explicit wire `seed` must reach the lane RNG on
                            // the batched route too; out-of-domain values are
                            // rejected loudly, never silently unseeded.
                            let Some(client_seed) =
                                client_seed_or_refuse(&mut stdout, id, msg.get("seed"))
                            else {
                                batch_clear_terminal_at_generation(id, gen_attempt_id, admission);
                                continue;
                            };
                            let pending = BatchPendingRequest {
                                key: AttemptKey::new(id, gen_attempt_id),
                                admission,
                                original_msg: msg.clone(),
                                prompt: prompt_owned.clone(),
                                prompt_tokens: prompt_tokens.clone(),
                                started_in_think,
                                system: system.map(|s| s.to_string()),
                                assistant_prefix,
                                max_think_tokens,
                                max_tokens,
                                client_seed,
                                sampling: sampling.clone(),
                            };
                            if let Some(sched) = batch_scheduler.as_mut() {
                                let arch = m.arch_id;
                                if arch == 11 {
                                    let enq_ok = sched.enqueue(pending);
                                    if !enq_ok {
                                        eprintln!(
                                            "[batch] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                            id, gen_attempt_id
                                        );
                                        continue;
                                    }
                                    {
                                        let _scope = BatchAttemptScope::enter_for_generation(
                                            id,
                                            gen_attempt_id,
                                            admission,
                                        );
                                        hipfire_generate::ar::emit_generation_start(
                                            hipfire_generate::ar::GenerationRoute::LfmAr,
                                            &mut stdout,
                                            id,
                                            false,
                                        );
                                    }
                                    let drive_res = drive_lfm_continuous_batch(
                                        sched,
                                        &mut gpu,
                                        m,
                                        &mut stdout,
                                        &mut inbox,
                                    );
                                    apply_batch_drive_result(
                                        drive_res,
                                        "[batch]",
                                        &mut batch_scheduler,
                                        &mut continuous_batch_size,
                                        &mut batch_poisoned,
                                    );
                                } else if arch == 5 || arch == 6 || qwen4_lanes {
                                    let enq_ok = sched.enqueue(pending);
                                    if !enq_ok {
                                        eprintln!(
                                            "[batch] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                            id, gen_attempt_id
                                        );
                                        continue;
                                    }
                                    // Qwen4 lanes announce `Qwen4Ar` from the driver after
                                    // its exact render (started_in_think is known only then).
                                    if !qwen4_lanes {
                                        let _scope = BatchAttemptScope::enter_for_generation(
                                            id,
                                            gen_attempt_id,
                                            admission,
                                        );
                                        hipfire_generate::ar::emit_generation_start(
                                            hipfire_generate::ar::GenerationRoute::QwenAr,
                                            &mut stdout,
                                            id,
                                            started_in_think,
                                        );
                                    }
                                    let drive_res =
                                        hipfire_generate::batch::drive_staged_continuous_batch(
                                            sched,
                                            &mut gpu,
                                            m,
                                            vmm_batch.filter(|_| vmm_route),
                                            &mut stdout,
                                            &mut inbox,
                                        );
                                    apply_batch_drive_result(
                                        drive_res,
                                        "[batch]",
                                        &mut batch_scheduler,
                                        &mut continuous_batch_size,
                                        &mut batch_poisoned,
                                    );
                                } else {
                                    eprintln!(
                                        "[batch] impossible arch {} reached scheduler — fail closed",
                                        arch
                                    );
                                    let _scope = BatchAttemptScope::enter_for_generation(
                                        id,
                                        gen_attempt_id,
                                        admission,
                                    );
                                    let ep = hipfire_generate::common::RollbackEpilogue {
                                        rolled_back: true,
                                        context: None,
                                    };
                                    hipfire_generate::common::emit_fail_closed_error(
                                        &mut stdout,
                                        Some(id),
                                        &format!("batch not supported for arch {}", arch),
                                        "validation",
                                        false,
                                        &ep,
                                    );
                                    batch_clear_terminal_at_generation(
                                        id,
                                        gen_attempt_id,
                                        admission,
                                    );
                                    batch_scheduler = None;
                                    continuous_batch_size = 1;
                                    batch_poisoned = Some(format!("impossible arch {}", arch));
                                    batch_clear_all_terminals();
                                }
                            }
                            continue;
                        }
                    } else {
                        // Ordinary sequential mode transfers the reader-owned
                        // batch key here. A dedicated singleton handoff has
                        // already retired it and must only rebind the scope.
                        if !singleton_handoff {
                            let _ = batch_transfer_abort_to_singleton_and_clear(
                                id,
                                gen_attempt_id,
                                admission,
                            );
                        }
                        batch_scope.rebind_for(gen_attempt_id);
                    }
                    // Did the request explicitly set a non-temperature sampling
                    // control? (gates temp>0 spec routing — see generate()).
                    let user_explicit_sampling = [
                        "top_p",
                        "top_k",
                        "min_p",
                        "repeat_penalty",
                        "presence_penalty",
                        "frequency_penalty",
                    ]
                    .iter()
                    .any(|k| msg.get(*k).is_some());
                    // Per-request sampler entropy (replaces the historical
                    // fixed 0x13579BDF that made same-prompt requests
                    // byte-identical at temp>0 on the sequential noslots path).
                    // Explicit wire `seed` wins (deterministic per seed alone —
                    // attempt identity is NOT mixed in, so two HTTP requests
                    // with the same seed reproduce); otherwise hipfire-engine
                    // mixes the attempt key with a process-global counter and
                    // boot nonce so reused client keys still get distinct
                    // streams. Out-of-domain seeds (negative, fractional,
                    // non-numeric) are rejected — never silently treated as
                    // unseeded.
                    let Some(client_seed) = client_seed_or_refuse(&mut stdout, id, msg.get("seed"))
                    else {
                        continue;
                    };
                    let request_seed =
                        request_seed_for(&AttemptKey::new(id, gen_attempt_id), client_seed);
                    hipfire_generate::batch::arm_promotion(
                        vmm_route && batch_scheduler.is_some(),
                        &msg,
                        m,
                        continuous_batch_size,
                        serve_continuous_batch,
                        pflash_active,
                        max_tokens,
                        max_think_tokens,
                        client_seed,
                        assistant_prefix,
                    );
                    generate(
                        m,
                        &mut gpu,
                        pflash_drafter_gpu.as_mut(),
                        &mut stdout,
                        id,
                        prompt,
                        system,
                        user_explicit_sampling,
                        temp,
                        top_p,
                        top_k,
                        min_p,
                        cactus_delta,
                        max_tokens,
                        repeat_penalty,
                        repeat_window,
                        presence_penalty,
                        frequency_penalty,
                        budget_alert_at_tok,
                        &budget_alert_text,
                        max_think_tokens,
                        assistant_prefix,
                        pflash_state.as_mut(),
                        pf_cfg_owned.as_ref(),
                        tools_json.as_deref(),
                        messages_history.as_deref(),
                        think_mode,
                        &stop_seqs, // hunt3 M-F
                        reasoning_effort_jinja.as_deref(),
                        enable_thinking_jinja,
                        logprobs_top_k,
                        request_seed,
                    );
                    if let Some(drive_res) = hipfire_generate::batch::drive_promoted(
                        batch_scheduler.as_mut(),
                        &mut gpu,
                        m,
                        vmm_batch,
                        &mut stdout,
                        &mut inbox,
                    ) {
                        apply_batch_drive_result(
                            drive_res,
                            "[batch]",
                            &mut batch_scheduler,
                            &mut continuous_batch_size,
                            &mut batch_poisoned,
                        );
                    }
                }
                if let Some(marker) = gpu.replay.replay_observation_marker(id) {
                    eprintln!("{marker}");
                }
            }

            "decide" => {
                let lanes_active = batch_scheduler
                    .as_ref()
                    .is_some_and(|s| s.active_count() > 0);
                let (reply, reset) = hipfire_generate::decide::handle_decide_message(
                    model.as_mut(),
                    &mut gpu,
                    &msg,
                    lanes_active,
                    slot_backend.is_some(),
                );
                // `reset`: the decide ran and rolled the model back; bump like `reset`.
                if reset {
                    state_epoch = state_epoch.saturating_add(1);
                }
                let _ = writeln!(stdout, "{reply}");
                let _ = stdout.flush();
            }

            "reset" => {
                // attempt_id is mandatory and must be echoed exactly on the ack.
                // Reject before mutating host/GPU state.
                let reset_attempt_id = match require_wire_attempt_id(msg.get("attempt_id")) {
                    Ok(id) => id,
                    Err(reason) => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &format!("reset {reason}"),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                // Experimental slot backend: alternate owner, reset via engine. Refuse while active, otherwise checked teardown, clear keyed entries, ack only success.
                if let Some(slot) = slot_backend.as_ref() {
                    if slot.active_count() > 0 {
                        hipfire_generate::dense::write_error_envelope(
                            &mut stdout,
                            None,
                            "reset refused: slot requests active",
                            "validation",
                            false,
                            false,
                            reset_attempt_id,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    match slot.reset() {
                        Ok(()) => {
                            batch_clear_all_terminals();
                            state_epoch = state_epoch.saturating_add(1);
                            let ack = serde_json::json!({
                                "type": "reset",
                                "rolled_back": true,
                                "state_epoch": state_epoch,
                                "seq_pos": 0,
                                "conversation_len": 0,
                                "attempt_id": reset_attempt_id,
                                "retry_reset_eligible": false,
                            });
                            let _ = writeln!(stdout, "{ack}");
                        }
                        Err(e) => {
                            hipfire_generate::dense::write_error_envelope(
                                &mut stdout,
                                None,
                                &format!("reset failed: {e}"),
                                "transient",
                                true,
                                false,
                                reset_attempt_id,
                            );
                        }
                    }
                    let _ = stdout.flush();
                    continue;
                }
                // Reset conversation state without unloading the model.
                // Single production epilogue owns ordering + graph/replay
                // invalidate + sync attestation (same path as fail-closed turns).
                if let Some(m) = &mut model {
                    // Batch guard: reset is forbidden while any lane is active (would corrupt disjoint KV/state).
                    if batch_scheduler
                        .as_ref()
                        .is_some_and(|s| s.active_count() > 0)
                    {
                        hipfire_generate::dense::write_error_envelope(
                            &mut stdout,
                            None,
                            "reset refused: continuous batch lanes active",
                            "validation",
                            false,
                            false,
                            reset_attempt_id,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    if developer_var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
                        eprintln!("[qwen-cache RESET] daemon received reset — clearing conversation_tokens (was {})", m.conversation_tokens.len());
                    }
                    // Attested reset ownership: EP models reset through the
                    // shared mesh epilogue (`ep_reset_after_abort`, which owns
                    // every EpArch variant incl. Qwen35 sequential + batch),
                    // single-GPU through the production rollback. No separate
                    // algorithm either way — the full per-rank reset happens
                    // here at request time, never deferred to a later cold
                    // reset. The mesh EP arm leaves the host semantic cache,
                    // so a daemon reset (full cold reset) drops it explicitly
                    // — the single-GPU path already clears it inside its
                    // rollback.
                    let ep = if m.ep.is_some() {
                        let ep = hipfire_generate::common::reset_mesh_request_state(m, None);
                        m.asst_turn_cache.clear();
                        ep
                    } else {
                        hipfire_generate::common::production_fail_closed_rollback(
                            m, &mut gpu, None, None,
                        )
                    };
                    if !ep.rolled_back {
                        let detail = ep
                            .context
                            .as_deref()
                            .unwrap_or("rollback could not be attested");
                        hipfire_generate::dense::write_error_envelope(
                            &mut stdout,
                            None,
                            &format!("reset failed: {detail}"),
                            "transient",
                            true,
                            false,
                            reset_attempt_id,
                        );
                        continue;
                    }
                    // Host counters must already be zero before ack (set in epilogue).
                    debug_assert_eq!(m.seq_pos, 0);
                    state_epoch = state_epoch.saturating_add(1);
                    // Clear batch scheduler host state on successful cold reset
                    if let Some(sched) = batch_scheduler.as_mut() {
                        let _ = sched.fail_all_active();
                    }
                    batch_clear_all_terminals();
                    let ack = serde_json::json!({
                        "type": "reset",
                        "rolled_back": true,
                        "state_epoch": state_epoch,
                        "seq_pos": 0,
                        "conversation_len": 0,
                        "attempt_id": reset_attempt_id,
                        "retry_reset_eligible": model_retry_reset_eligible(m.arch_id),
                    });
                    let _ = writeln!(stdout, "{ack}");
                } else {
                    hipfire_generate::dense::write_error_envelope(
                        &mut stdout,
                        None,
                        "no model loaded",
                        "validation",
                        false,
                        false,
                        reset_attempt_id,
                    );
                }
                let _ = stdout.flush();
            }

            "unload" => {
                // Experimental slot backend owns its own weight copy; unload it exclusively. Refuse while active, otherwise checked teardown, clear keyed entries, ack only success. Do not allow old Arc workers to keep prior model alive.
                if slot_backend.is_some() {
                    if slot_backend.as_ref().is_some_and(|b| b.active_count() > 0) {
                        let attempt = msg.get("attempt_id").and_then(|v| v.as_u64()).unwrap_or(0);
                        hipfire_generate::dense::write_error_envelope(
                            &mut stdout,
                            None,
                            "unload refused: slot requests active",
                            "validation",
                            false,
                            false,
                            attempt,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    if let Some(slot) = slot_backend.take() {
                        match std::sync::Arc::try_unwrap(slot) {
                            Err(slot) => {
                                slot_backend = Some(slot);
                                let attempt =
                                    msg.get("attempt_id").and_then(|v| v.as_u64()).unwrap_or(0);
                                hipfire_generate::dense::write_error_envelope(
                                    &mut stdout,
                                    None,
                                    "unload refused: slot requests active (Arc live)",
                                    "validation",
                                    false,
                                    false,
                                    attempt,
                                );
                                let _ = stdout.flush();
                            }
                            Ok(slot) => match slot.shutdown() {
                                Ok(()) => {
                                    batch_scheduler = None;
                                    continuous_batch_size = 1;
                                    batch_poisoned = None;
                                    batch_clear_all_terminals();
                                    resident_kv = None;
                                    let _ = writeln!(stdout, "{}", r#"{"type":"unloaded"}"#);
                                    let _ = stdout.flush();
                                }
                                Err(reason) => {
                                    let attempt =
                                        msg.get("attempt_id").and_then(|v| v.as_u64()).unwrap_or(0);
                                    hipfire_generate::dense::write_error_envelope(
                                        &mut stdout,
                                        None,
                                        &format!("unload failed: {reason}"),
                                        "internal",
                                        false,
                                        false,
                                        attempt,
                                    );
                                    let _ = stdout.flush();
                                }
                            },
                        }
                    } else {
                        resident_kv = None;
                        let _ = writeln!(stdout, "{}", r#"{"type":"unloaded"}"#);
                        let _ = stdout.flush();
                    }
                    continue;
                }
                // Batch guard: unload is forbidden while lanes active.
                if batch_scheduler
                    .as_ref()
                    .is_some_and(|s| s.active_count() > 0)
                {
                    let attempt = msg.get("attempt_id").and_then(|v| v.as_u64()).unwrap_or(0);
                    hipfire_generate::dense::write_error_envelope(
                        &mut stdout,
                        None,
                        "unload refused: continuous batch lanes active",
                        "validation",
                        false,
                        false,
                        attempt,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                // PFlash drafter goes FIRST: its weights/scratch/KV
                // tensors are released via Gpu::free_tensor, which only
                // queues into the GPU pool. The actual hipFree happens
                // inside unload_model -> drain_pool. Calling
                // unload_drafter AFTER unload_model would leave the
                // drafter buffers cached in the just-emptied pool with
                // no drain to follow, so the VRAM stays resident until
                // the next load message arrives. Order matters here.
                vision_gated_off = None;
                if let Some(mut pf) = pflash_state.take() {
                    if let Some(mut dg) = pflash_drafter_gpu.take() {
                        dg.bind_thread_or_warn();
                        pf.unload_drafter(&mut dg);
                        gpu.bind_thread_or_warn();
                    } else {
                        pf.unload_drafter(&mut gpu);
                    }
                }
                pflash_cfg = None;
                let unload_result = if let Some(m) = model.take() {
                    hipfire_loader::unload_model(m, &mut gpu)
                } else {
                    // No model: still retry any process-global pending VMM arenas.
                    hipfire_loader::ensure_vmm_ready_for_load(&mut gpu)
                };
                match unload_result {
                    Ok(()) => {
                        resident_kv = None;
                        let _ = writeln!(stdout, r#"{{"type":"unloaded"}}"#);
                    }
                    Err(err) => {
                        // Model handle is already taken; clear resident claim.
                        resident_kv = None;
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &format!("unload incomplete: {err}; VMM arenas retained for retry"),
                            "internal",
                            false,
                            false,
                        );
                    }
                }
                batch_scheduler = None;
                continuous_batch_size = 1;
                batch_clear_all_terminals();
                let _ = stdout.flush();
            }
            "ping" => {
                let _ = writeln!(stdout, r#"{{"type":"pong"}}"#);
                let _ = stdout.flush();
            }

            // ── Image generation wire ─────────────────
            // `img_load` is deliberately NOT a separate command: the ordinary
            // `load` message routes a FLUX trunk pack to the arch-40/45
            // carrier. Refuse the name so a client typo never silently
            // no-ops.
            "img_load" => {
                emit_uncorrelated_error(
                    &mut stdout,
                    msg.get("id").and_then(|v| v.as_str()),
                    "img_load is not a command: load the pack with {\"type\":\"load\",\"model\":\"<base>-transformer.hfq\"} (arch 40/45), then send img_generate",
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
            }

            "img_generate" => {
                // Contract: monotonic img_progress 0..steps,
                // exactly one img_done; fail-closed errors for bad
                // width/height/steps/seed/sampler and for non-diffusion
                // loads (the mirror of the text-generate refusal above).
                let id = msg
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("0")
                    .to_string();
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            Some(id.as_str()),
                            "no model loaded",
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                if !hipfire_loader::img_route(m.arch_id).is_diffusion() {
                    emit_uncorrelated_error(
                        &mut stdout,
                        Some(id.as_str()),
                        "img_generate refused: loaded model is not a diffusion checkpoint (arch 40/45) — use generate",
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                // The body lives in hipfire_generate::img so main.rs names
                // no arch-crate type (keeps daemon_arch_refs at 0).
                hipfire_generate::img::generate_img(m, &mut gpu, &mut stdout, &id, &msg);
                let _ = stdout.flush();
            }

            "embed" => {
                // Embedding encode (arch 17, the qwen3.5 multi-vector
                // encoder; arch 18, EmbeddingGemma-2). The img_generate shape:
                // dispatch + refusals here, the body in hipfire-generate so
                // main.rs names no arch-crate type.
                let id = msg
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("0")
                    .to_string();
                // Multi-slot backend dispatches before the ordinary model
                // path, for the same reason the generate arm does: in slot mode
                // the daemon's `model` slot is empty by design and the engine
                // owns the one weight copy. A worker per request keeps the read
                // loop free, so concurrent embeds reach the engine together and
                // it can merge them into one forward.
                if let Some(slot) = slot_backend.clone() {
                    // Bound thread creation BEFORE spawning, same reason the
                    // generate arm does: a guard inside an already-spawned
                    // thread does not bound thread creation. The worker's
                    // acquire_guard in handle_embed remains the hard bound.
                    // Embed replies correlate by (id, 0), so the uncorrelated
                    // emitter is the right channel here (attempt TLS is 0).
                    if slot.active_count() >= 32 {
                        emit_uncorrelated_error(
                            &mut stdout,
                            Some(id.as_str()),
                            "too many concurrent slot requests (bounded worker limit hit)",
                            "overload",
                            true,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    let msg_clone = msg.clone();
                    let id_owned = id.clone();
                    let slot_clone = slot.clone();
                    std::thread::spawn(move || {
                        let mut worker_stdout = std::io::stdout();
                        slot_clone.handle_embed(&msg_clone, &mut worker_stdout, &id_owned);
                        let _ = worker_stdout.flush();
                    });
                    continue;
                }
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            Some(id.as_str()),
                            "no model loaded",
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                if hipfire_loader::wire_verb(m.arch_id).is_generate() {
                    emit_uncorrelated_error(
                        &mut stdout,
                        Some(id.as_str()),
                        "embed refused: loaded model is not an embedding checkpoint \
                         (arch 17/18) — use generate",
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                // A media-carrying embed needs its tower. Whether a tower is
                // resident is arch-side state (`bundle.vision` / `bundle.audio`)
                // this arm must not name; the load-time knob it DOES own is
                // named here — `vision_mode=off` dropped the sidecar (which
                // carries both towers) that this request needs. Placed after
                // the verb refusal so a generation model still answers "use
                // generate" (the actionable fix there).
                if let Some(tower) = embed_request_media(&msg) {
                    if vision_gated_off.is_some() {
                        refuse_media_without_tower(
                            &mut stdout,
                            id.as_str(),
                            tower,
                            vision_gated_off.as_deref(),
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                }
                hipfire_generate::embed::generate_embed(m, &mut gpu, &mut stdout, &id, &msg);
                let _ = stdout.flush();
            }

            "diag" => {
                let (vram_free, vram_total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
                let hip_ver = gpu.hip.runtime_version().unwrap_or((0, 0));
                let has_model = model.is_some() || slot_backend.is_some();
                let model_arch = model
                    .as_ref()
                    .map(|m| hipfire_loader::arch_label(m.arch_id))
                    .unwrap_or("none");
                // Count pre-compiled kernels
                let kernel_dir = std::env::current_exe()
                    .ok()
                    .and_then(|e| {
                        e.parent()
                            .map(|p| p.join("kernels").join("compiled").join(&gpu.arch))
                    })
                    .filter(|p| p.is_dir());
                let (hsaco_count, hash_count) = kernel_dir
                    .map(|d| {
                        let hsaco = std::fs::read_dir(&d)
                            .map(|r| {
                                r.filter(|e| {
                                    e.as_ref()
                                        .ok()
                                        .map(|e| {
                                            e.path()
                                                .extension()
                                                .map(|x| x == "hsaco")
                                                .unwrap_or(false)
                                        })
                                        .unwrap_or(false)
                                })
                                .count()
                            })
                            .unwrap_or(0);
                        let hash = std::fs::read_dir(&d)
                            .map(|r| {
                                r.filter(|e| {
                                    e.as_ref()
                                        .ok()
                                        .map(|e| {
                                            e.path()
                                                .extension()
                                                .map(|x| x == "hash")
                                                .unwrap_or(false)
                                        })
                                        .unwrap_or(false)
                                })
                                .count()
                            })
                            .unwrap_or(0);
                        (hsaco, hash)
                    })
                    .unwrap_or((0, 0));
                let kv = resident_kv.as_ref();
                let kv_backend_json = serde_json::to_string(&kv.map(|k| k.backend.as_str()))
                    .expect("kv_backend serializable");
                let kv_request_json = serde_json::to_string(&kv.map(|k| k.request.as_str()))
                    .expect("kv_backend_request serializable");
                let kv_reason_json = serde_json::to_string(&kv.and_then(|k| k.reason.as_deref()))
                    .expect("kv_backend_reason serializable");
                let kv_legacy = kv.map(|k| k.legacy).unwrap_or(false);
                let kv_warning_json = serde_json::to_string(&kv.and_then(|k| k.warning.as_deref()))
                    .expect("kv_backend_warning serializable");
                let kv_mode_json = serde_json::to_string(&kv.and_then(|k| k.mode.as_deref()))
                    .expect("kv_mode serializable");
                let mut diag_line = format!(
                    r#"{{"type":"diag","arch":"{}","hip_version":"{}.{}","vram_free_mb":{},"vram_total_mb":{},"model_loaded":{},"model_arch":"{}","kernels":{},"kernel_hashes":{},"kv_backend":{},"kv_backend_request":{},"kv_backend_reason":{},"kv_backend_legacy":{},"kv_backend_warning":{},"kv_mode":{}}}"#,
                    gpu.arch,
                    hip_ver.0,
                    hip_ver.1,
                    vram_free / (1024 * 1024),
                    vram_total / (1024 * 1024),
                    has_model,
                    model_arch,
                    hsaco_count,
                    hash_count,
                    kv_backend_json,
                    kv_request_json,
                    kv_reason_json,
                    kv_legacy,
                    kv_warning_json,
                    kv_mode_json,
                );
                // Windows only: which amdhip64*.dll actually loaded (System32's
                // driver copy can shadow an SDK's) and the raw runtime/driver
                // version ints. null when unavailable. Same splice as
                // `append_diag_emu` below.
                #[cfg(windows)]
                {
                    let lib_json = serde_json::to_string(&gpu.hip.mapped_hip_library())
                        .expect("hip_runtime_library serializable");
                    let rt_json = serde_json::to_string(&gpu.hip.runtime_version_raw().ok())
                        .expect("hip_runtime_version_raw serializable");
                    let drv_json = serde_json::to_string(&gpu.hip.driver_version().ok())
                        .expect("hip_driver_version_raw serializable");
                    diag_line.pop();
                    diag_line.push_str(&format!(
                        r#","hip_runtime_library":{lib_json},"hip_runtime_version_raw":{rt_json},"hip_driver_version_raw":{drv_json}}}"#
                    ));
                }
                emu_daemon::append_diag_emu(&gpu, &mut diag_line);
                let _ = writeln!(stdout, "{diag_line}");
                let _ = stdout.flush();
            }

            "emu_validate" => validate(&mut gpu, &msg, &mut stdout, (&model, &slot_backend)),
            "npu_trunk_route" => npu_route::handle(&msg, &mut stdout),

            "bench_prefill" => bench::handle_bench_prefill(&mut model, &mut gpu, &msg, &mut stdout),

            "bench_decode" => bench::handle_bench_decode(&mut model, &mut gpu, &msg, &mut stdout),

            "redline_probe_aql" => {
                handle_redline_probe_aql(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_dspark_shadow_pm4" => {
                handle_redline_dspark_shadow_pm4(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_dflash_verify_shadow_pm4" => {
                handle_redline_dflash_verify_shadow_pm4(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "railgun_g0_dflash_cycle" => {
                handle_railgun_g0_dflash_cycle(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_shadow_aql" | "redline_shadow_pm4" => {
                handle_redline_shadow(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_greedy_trace" => {
                handle_redline_greedy_trace(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_dispatch_profile" => {
                handle_redline_dispatch_profile(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_pm4_prefix_profile" => {
                handle_redline_pm4_prefix_profile(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_prefix_shadow" => {
                handle_redline_prefix_shadow(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "profile" => {
                // Compile the arch's registry inventory into the kernel cache so
                // the profiler has every packaged kernel, not only this model's.
                // CPU-only: no module is loaded and a loaded route is untouched.
                if let Err(e) = gpu.precompile_registry() {
                    eprintln!("profile: registry precompile incomplete: {e}");
                }
                let (cap, kernels) = gpu.profile();
                let kernels_json: Vec<String> = kernels.iter().map(|k| k.to_json()).collect();
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"profile","gpu":{},"kernels":[{}]}}"#,
                    cap.to_json(),
                    kernels_json.join(",")
                );
                let _ = stdout.flush();
            }

            #[cfg(feature = "serve-fault-inject")]
            "test_state_snapshot" => {
                write_test_state_snapshot(&mut stdout, model.as_ref(), &gpu, state_epoch);
            }

            _ => {
                tracing::warn!(command = msg_type, "daemon received unknown command");
                emit_uncorrelated_error(
                    &mut stdout,
                    None,
                    &format!("unknown type: {}", msg_type),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
            }
        }
    }
    // stdin closed: persist the retained-PM4 dispatch ledger's replay counters
    // (`HIPFIRE_REDLINE_LEDGER`; a no-op when unset).
    rdna_compute::pm4_ledger::flush_all();
}

#[cfg(test)]
mod tests {
    /// The load receipt is exactly two stderr lines, human summary first,
    /// then the compact `hipfire.load_phases.v1` JSON, from one snapshot.
    #[test]
    fn load_phase_receipt_is_two_lines_in_order() {
        rdna_compute::load_phases::begin_load();
        rdna_compute::load_phases::add("open_admission", 1_500_000, 0);
        let (line, json) = rdna_compute::load_phases::finish_load();
        let lines = super::load_phase_lines(&line, &json);
        assert!(
            lines[0].starts_with("hipfire load phases: total="),
            "{}",
            lines[0]
        );
        assert!(lines[0].contains("open_admission=1.5ms/1"), "{}", lines[0]);
        let payload = lines[1]
            .strip_prefix("hipfire.load_phases.v1 ")
            .expect("second line is the JSON receipt");
        assert!(!payload.contains('\n'), "compact JSON");
        let parsed: serde_json::Value = serde_json::from_str(payload).expect("valid JSON");
        assert_eq!(parsed["schema"], "hipfire.load_phases.v1");
        assert_eq!(parsed, json);
        assert!(lines.iter().all(|l| !l.contains('\n')));
        // finish_load ended the load: a later record is dropped.
        rdna_compute::load_phases::add("late", 1, 0);
        assert!(!rdna_compute::load_phases::summary_line().contains("late="));
    }

    /// Source-level: the generate arm takes the whole-request PLE guard
    /// before any setup or route (batch enqueue, drive, singleton generate)
    /// and the bench verbs take it right after resolving the model.
    #[test]
    fn request_routes_hold_the_ple_serving_guard() {
        let src = include_str!("main.rs");
        let prod = &src[..src.find("#[cfg(test)]\nmod tests {").expect("test module")];
        let arm = prod.find("\"generate\" => {").expect("generate arm");
        let arm_end = arm + prod[arm..].find("\"decide\" => {").expect("decide arm");
        let arm = &prod[arm..arm_end];
        let guard = arm
            .find("let _ple_serving = m.ple_serving_guard();")
            .expect("guard");
        for later in [
            "activate_terminal_control(id, gen_attempt_id)",
            "batch_request_eligible_for_route(",
            "drive_staged_continuous_batch(",
            "drive_lfm_continuous_batch(",
            "hipfire_generate::batch::drive_promoted(",
            "                    generate(\n",
        ] {
            let at = arm.find(later).unwrap_or_else(|| panic!("{later} missing"));
            assert!(guard < at, "guard must precede {later}");
        }
        let bench = include_str!("bench.rs");
        assert_eq!(
            bench
                .matches("let _ple_serving = m.ple_serving_guard();")
                .count(),
            2
        );
        for handler in ["fn handle_bench_prefill(", "fn handle_bench_decode("] {
            let body = &bench[bench.find(handler).expect(handler)..];
            let guard = body
                .find("let _ple_serving = m.ple_serving_guard();")
                .expect("bench guard");
            assert!(guard < body.find("m.bench_topology(").expect("topology"));
        }
    }

    #[test]
    fn load_pp_resolves_config_fallback_before_weight_host_refusal() {
        let mut msg = serde_json::json!({"type": "load", "params": {"tp": 1}});
        assert_eq!(
            hipfire_config::topology::resolve_load_pp_with(&mut msg, |_| Ok(2)),
            Ok(2)
        );
        assert!(super::weight_host_tenant::load_refusal(true, &msg)
            .is_some_and(|refusal| refusal.contains("tp/pp > 1")));
    }

    #[test]
    fn admitted_load_failure_names_missing_model_only_when_unload_first() {
        // Unload-first (single-device/pp, incl. Qwen4): prior already retired.
        assert!(super::no_model_loaded_suffix(false).contains("no model loaded"));
        // Deferred tp>1: prior model survives a failed load.
        assert_eq!(super::no_model_loaded_suffix(true), "");
    }

    use super::{
        announce_generate_terminal, apply_vision_mode_gate, client_seed_or_refuse,
        embed_request_media, emit_batch_admission_error, media_refusal_message,
        refuse_image_without_vision, require_wire_attempt_id, MediaTower, TERMINAL_TEST_LOCK,
    };
    use hipfire_engine::emit::{emit_active_attempt_error, emit_uncorrelated_error};
    use hipfire_engine::terminal::{
        batch_announce_terminal, batch_clear_all_terminals, batch_clear_terminal,
        batch_terminal_control, clear_terminal_control, set_active_attempt_id, terminal_generation,
        AttemptKey, BatchAttemptScope,
    };

    #[test]
    fn vision_mode_off_drops_even_an_explicit_sidecar() {
        // Hard override, mirroring the `dflash_mode=off` draft guard: a
        // default load never pays the tower VRAM, however the path arrived.
        assert_eq!(
            apply_vision_mode_gate("off", Some("/models/qwen3.8-27b-vision.hfq".into())),
            None
        );
        assert_eq!(apply_vision_mode_gate("off", None), None);
    }

    #[test]
    fn vision_mode_auto_resident_and_on_pass_the_ladder_result_through() {
        for mode in ["auto", "resident", "on"] {
            assert_eq!(
                apply_vision_mode_gate(mode, Some("/models/qwen3.8-27b-vision.hfq".into())),
                Some("/models/qwen3.8-27b-vision.hfq".to_owned())
            );
            assert_eq!(apply_vision_mode_gate(mode, None), None);
        }
    }

    /// The embed arm's media predicate decides which refusal a media request
    /// gets, so its spelling sensitivity is load-bearing: the canonical
    /// `images_base64` / `audios_base64` lists and the one-image
    /// `image_base64` legacy field count, and the *wrong* list spelling does
    /// not — that request is refused by field name in
    /// `hipfire_generate::embed`, and answering it with the tower-knob refusal
    /// instead would point the caller at the wrong fix.
    #[test]
    fn an_embed_request_reports_which_tower_it_asked_for() {
        let media = |prompt: serde_json::Value| embed_request_media(&prompt);
        assert_eq!(
            media(serde_json::json!({"input": {"text": "t", "images_base64": ["QUJD"]}})),
            Some(MediaTower::Vision)
        );
        assert_eq!(
            media(serde_json::json!({"input": {"text": "t", "image_base64": "QUJD"}})),
            Some(MediaTower::Vision)
        );
        assert_eq!(
            media(serde_json::json!({"input": {"text": "t", "audios_base64": ["QUJD"]}})),
            Some(MediaTower::Audio)
        );
        // Video asks for the VISION tower: arch 18's video leg runs the image
        // tower once per frame, it has no tower of its own.
        assert_eq!(
            media(serde_json::json!({"input": {"text": "t", "videos_base64": [["QUJD"]]}})),
            Some(MediaTower::Vision)
        );
        assert_eq!(
            media(serde_json::json!({"input": {"text": "t", "videos_base64": []}})),
            None
        );
        assert_eq!(
            media(serde_json::json!({"input": {"text": "t", "videos": [["QUJD"]]}})),
            None
        );
        // Both media: vision first, the tower whose refusal the caller sees.
        assert_eq!(
            media(serde_json::json!({
                "input": {"images_base64": ["QUJD"], "audios_base64": ["REVG"]}
            })),
            Some(MediaTower::Vision)
        );
        assert_eq!(media(serde_json::json!({"input": {"text": "t"}})), None);
        assert_eq!(
            media(serde_json::json!({"input": {"text": "t", "images_base64": []}})),
            None
        );
        assert_eq!(
            media(serde_json::json!({"input": {"text": "t", "images": ["QUJD"]}})),
            None
        );
        assert_eq!(media(serde_json::json!({"input": "t"})), None);
        assert_eq!(media(serde_json::json!({"input": null})), None);
    }

    /// The refusal names the tower the caller asked for, and the knob that
    /// dropped the one sidecar both towers live in.
    #[test]
    fn a_media_refusal_names_the_tower_the_caller_asked_for() {
        for (tower, noun) in [(MediaTower::Vision, "vision"), (MediaTower::Audio, "audio")] {
            let gated = media_refusal_message(tower, Some("/models/m.vl"));
            assert!(gated.contains(&format!("no {noun} encoder")), "{gated}");
            assert!(gated.contains("vision_mode is off"), "{gated}");
            assert!(gated.contains("/models/m.vl"), "{gated}");
            let bare = media_refusal_message(tower, None);
            assert_eq!(bare, format!("model has no {noun} encoder"), "{bare}");
        }
    }

    fn error_events(output: &[u8]) -> Vec<serde_json::Value> {
        std::str::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    /// Client faults in the generate arm carry a client class (HTTP 400 at
    /// the gateway), never `internal` (HTTP 500).
    #[test]
    fn generate_client_faults_are_typed_not_internal() {
        let _lock = TERMINAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // One live generate attempt per refusal, as the generate arm has.
        let announce = |id: &str| {
            clear_terminal_control();
            batch_clear_all_terminals();
            set_active_attempt_id(7);
            announce_generate_terminal(
                &serde_json::json!({"type": "generate", "id": id, "attempt_id": 7}),
            );
        };
        for seed in [
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("abc"),
        ] {
            announce("req-seed");
            let _scope = BatchAttemptScope::enter_for("req-seed", 7);
            let mut output = Vec::new();
            assert_eq!(
                client_seed_or_refuse(&mut output, "req-seed", Some(&seed)),
                None
            );
            let events = error_events(&output);
            assert_eq!(events.len(), 1, "{seed}");
            assert_eq!(events[0]["type"], "error");
            assert_eq!(events[0]["id"], "req-seed");
            assert_eq!(events[0]["attempt_id"], 7);
            assert_eq!(events[0]["class"], "validation", "{seed}");
        }
        announce("req-seed");
        let mut output = Vec::new();
        assert_eq!(
            client_seed_or_refuse(&mut output, "req-seed", Some(&serde_json::json!(42))),
            Some(Some(42))
        );
        assert_eq!(
            client_seed_or_refuse(&mut output, "req-seed", None),
            Some(None)
        );
        assert!(output.is_empty(), "a valid or absent seed emits nothing");

        for gated in [None, Some("/models/x-vision.hfq")] {
            announce("req-img");
            let _scope = BatchAttemptScope::enter_for("req-img", 7);
            let mut output = Vec::new();
            refuse_image_without_vision(&mut output, "req-img", gated);
            let events = error_events(&output);
            assert_eq!(events.len(), 1);
            assert_eq!(events[0]["class"], "unsupported");
            assert!(events[0]["message"]
                .as_str()
                .unwrap()
                .starts_with("model has no vision encoder"));
        }
        clear_terminal_control();
        batch_clear_all_terminals();
        set_active_attempt_id(0);
    }

    #[test]
    fn zero_generate_attempt_is_rejected_before_lifecycle_admission() {
        let _lock = TERMINAL_TEST_LOCK.lock().unwrap();
        clear_terminal_control();
        batch_clear_all_terminals();
        set_active_attempt_id(0);

        let zero = serde_json::json!({
            "type": "generate",
            "id": "req-zero",
            "attempt_id": 0,
        });
        assert_eq!(announce_generate_terminal(&zero), None);
        assert_eq!(
            require_wire_attempt_id(zero.get("attempt_id")),
            Err("attempt_id must be nonzero")
        );
        assert!(
            batch_terminal_control()
                .mu
                .lock()
                .unwrap()
                .entries
                .is_empty(),
            "reserved attempt zero must never enter the batch registry"
        );
        assert!(
            terminal_generation("req-zero", 0).is_none(),
            "reserved attempt zero must never activate singleton state"
        );

        // The rejected raw request still gets one observable, routeable
        // validation envelope. Its id is retained while attempt zero remains
        // the explicit uncorrelated channel.
        let reason = require_wire_attempt_id(zero.get("attempt_id")).unwrap_err();
        let mut output = Vec::new();
        emit_uncorrelated_error(
            &mut output,
            Some("req-zero"),
            &format!("generate {reason}"),
            "validation",
            false,
            false,
        );
        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], "error");
        assert_eq!(events[0]["id"], "req-zero");
        assert_eq!(events[0]["attempt_id"], 0);
        assert_eq!(events[0]["class"], "validation");
        assert_eq!(events[0]["message"], "generate attempt_id must be nonzero");

        // A normal nonzero request keeps the existing admission path.
        let valid = serde_json::json!({
            "type": "generate",
            "id": "req-valid",
            "attempt_id": 7,
        });
        assert_eq!(require_wire_attempt_id(valid.get("attempt_id")), Ok(7));
        assert!(matches!(
            announce_generate_terminal(&valid),
            Some(("req-valid", 7, Some(_)))
        ));
        assert!(batch_terminal_control()
            .mu
            .lock()
            .unwrap()
            .entries
            .contains_key(&AttemptKey::new("req-valid", 7)));
        batch_clear_terminal("req-valid", 7);
        clear_terminal_control();
        batch_clear_all_terminals();
        set_active_attempt_id(0);
    }

    #[test]
    fn admitted_generate_errors_are_keyed_and_cleanup_after_emit() {
        let _lock = TERMINAL_TEST_LOCK.lock().unwrap();
        clear_terminal_control();
        batch_clear_all_terminals();
        set_active_attempt_id(0);

        // The early no-model path owns the reader-announced batch key even
        // before singleton activation. Verify the writer claims that live
        // entry before cleanup retires it.
        let early_id = "admission-no-model";
        let early_attempt = 70_001;
        let early_msg = serde_json::json!({
            "type": "generate",
            "id": early_id,
            "attempt_id": early_attempt,
        });
        assert!(matches!(
            announce_generate_terminal(&early_msg),
            Some((id, attempt, Some(_)))
        ));
        let mut early_out = Vec::new();
        {
            let _scope = BatchAttemptScope::enter_for(early_id, early_attempt);
            let key = AttemptKey::new(early_id, early_attempt);
            {
                let state = batch_terminal_control().mu.lock().unwrap();
                assert!(state
                    .entries
                    .get(&key)
                    .is_some_and(|entry| !entry.terminal_claimed));
            }
            emit_active_attempt_error(
                &mut early_out,
                Some(early_id),
                "no model loaded",
                "validation",
                false,
                false,
            );
            {
                let state = batch_terminal_control().mu.lock().unwrap();
                assert!(
                    state
                        .entries
                        .get(&key)
                        .is_some_and(|entry| entry.terminal_claimed),
                    "writer must claim the live keyed entry before cleanup"
                );
            }
        }
        batch_clear_terminal(early_id, early_attempt);
        let early_events: Vec<serde_json::Value> = std::str::from_utf8(&early_out)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(early_events.len(), 1);
        assert_eq!(early_events[0]["id"], early_id);
        assert_eq!(early_events[0]["attempt_id"], early_attempt);
        assert_ne!(early_events[0]["attempt_id"], 0);

        // The shared admission helper covers both batch render and lane
        // capacity rejection without emitting an attempt-zero envelope.
        for (offset, message) in [
            (1_u64, "render failed: malformed prompt"),
            (2_u64, "prompt exceeds lane capacity or empty"),
        ] {
            let id = format!("admission-batch-{offset}");
            let attempt = 70_001 + offset;
            let admission = batch_announce_terminal(&id, attempt).expect("batch admission");
            let mut out = Vec::new();
            emit_batch_admission_error(
                &mut out,
                &id,
                attempt,
                admission,
                message,
                "validation",
                false,
                false,
            );
            let events: Vec<serde_json::Value> = std::str::from_utf8(&out)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0]["id"], id);
            assert_eq!(events[0]["attempt_id"], attempt);
            assert_ne!(events[0]["attempt_id"], 0);
            assert_eq!(events[0]["class"], "validation");
            assert!(
                batch_announce_terminal(&id, attempt).is_some(),
                "key must be retired only after the emitted terminal"
            );
            batch_clear_terminal(&id, attempt);
        }

        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
    }

    /// `scripts/agentic_gate_driver.py` emits the requests checked in at
    /// `scripts/tests/fixtures/agentic-gate-requests.jsonl` (a pytest pins the
    /// driver output to that file). Every generate line must pass the daemon's
    /// real wire validation: a nonzero numeric attempt_id, an accepted reader
    /// admission, the typed reasoning fields the generate arm parses, and the
    /// explicit sampling fields the daemon resolves. A request the daemon
    /// would refuse (the pre-protocol gate sent no attempt_id) fails here.
    #[test]
    fn agentic_gate_driver_requests_pass_daemon_wire_validation() {
        let _lock = TERMINAL_TEST_LOCK.lock().unwrap();
        clear_terminal_control();
        batch_clear_all_terminals();
        set_active_attempt_id(0);

        let fixture = include_str!("../../../scripts/tests/fixtures/agentic-gate-requests.jsonl");
        let mut seen_ids = std::collections::HashSet::new();
        let mut last_attempt = 0u64;
        let mut generates = 0usize;
        let mut loads = 0usize;
        let mut unloads = 0usize;
        for (n, line) in fixture.lines().enumerate() {
            let msg: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("fixture line {}: invalid JSON: {e}", n + 1));
            match msg["type"].as_str() {
                Some("load") => {
                    loads += 1;
                    assert!(msg["model"].is_string());
                    assert!(msg["params"]["max_seq"].as_u64().is_some());
                    continue;
                }
                Some("unload") => {
                    unloads += 1;
                    continue;
                }
                Some("generate") => generates += 1,
                other => panic!("fixture line {}: unexpected type {other:?}", n + 1),
            }

            // Wire attempt id: the same validator the generate arm runs first.
            let attempt = require_wire_attempt_id(msg.get("attempt_id"))
                .unwrap_or_else(|e| panic!("fixture line {}: {e}", n + 1));
            assert!(
                attempt > last_attempt,
                "attempt ids must increase per request"
            );
            last_attempt = attempt;
            let id = msg["id"].as_str().expect("string id");
            assert!(seen_ids.insert(id.to_string()), "duplicate request id {id}");

            // Reader-side admission accepts it as a fresh live key.
            match announce_generate_terminal(&msg) {
                Some((got_id, got_attempt, Some(_))) => {
                    assert_eq!((got_id, got_attempt), (id, attempt));
                }
                other => panic!("fixture line {}: announce refused: {other:?}", n + 1),
            }
            batch_clear_terminal(id, attempt);

            // Typed fields the generate arm reads.
            assert!(msg["prompt"].is_string() && msg["system"].is_string());
            assert!(msg["max_tokens"].as_u64().is_some_and(|v| v > 0));
            let thinking_enabled = msg
                .get("thinking_enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or_else(|| panic!("fixture line {}: thinking_enabled must be bool", n + 1));
            let prefix = msg["assistant_prefix"].as_str().expect("assistant_prefix");
            assert_eq!(
                prefix,
                if thinking_enabled {
                    "open_think"
                } else {
                    "closed_think"
                },
                "prefix follows the serve path's Qwen Jinja framing"
            );
            assert!(
                msg.get("max_think_tokens").is_none(),
                "serve omits max_think_tokens unless an explicit positive cap is set"
            );
            let max_think_tokens = msg
                .get("max_think_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let (enable_thinking, effort) = hipfire_engine::prompt::qwen_jinja_reasoning(
                Some(thinking_enabled),
                msg.get("reasoning_effort").and_then(|v| v.as_str()),
                max_think_tokens,
            );
            assert_eq!(enable_thinking, thinking_enabled);
            assert_eq!(effort, None);

            // Explicit sampling wins over the model/arch defaults.
            let (temp, top_p) =
                hipfire_engine::scheduler::resolve_temp_top_p(&msg, 0, Some(0.7), Some(0.7));
            assert_eq!((temp, top_p), (0.0, 1.0));
            assert_eq!(msg["repeat_penalty"].as_f64(), Some(1.0));
        }
        assert_eq!((loads, unloads), (1, 1));
        assert!(generates >= 2, "fixture must cover multi-turn requests");

        // The pre-protocol gate request shape is refused by the same validator.
        let legacy = serde_json::json!({ "type": "generate", "id": "c0_t1", "thinking": false });
        assert_eq!(
            require_wire_attempt_id(legacy.get("attempt_id")),
            Err("missing attempt_id")
        );
        assert_eq!(announce_generate_terminal(&legacy), None);

        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
    }
}
