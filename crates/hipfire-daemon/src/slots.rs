// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Experimental multi-slot daemon backend — alternate model owner, NOT a
//! continuous-batching mode.
//!
//! This module is a **daemon-owned** backend that owns exactly one `SlotEngine`
//! (one weight copy) when `experimental_multi_slot=true`. It is an *alternate*
//! to the ordinary `LoadedModel` path, not a switch that enables batching on
//! that model.
//!
//! Continuous-batching integration is **deferred** — this backend deliberately
//! does not integrate with `ContinuousBatchScheduler`, `qwen35/batch.rs`, or
//! `hipfire-generate` batch code. The `continuous_batch_capable` advertisement
//! remains `false` while this backend is active, and any future batch
//! integration must be an explicit, separate cutover that preserves the
//! invariant: ordinary/default daemon load/generate remains byte-for-byte on the
//! existing daemon path when `experimental_multi_slot` is absent.
//!
//! Load-time CPU preflight opens the HFQ, requires arch_id 5|6, records whether
//! the HFQ has a vision tower, parses Qwen config/tokenizer/chat template, then
//! spawns `SlotEngine`. Generate validates supported wire fields and builds
//! prompt/convo/continuation with the CLI `slots.rs` semantics. Image turns
//! are CPU-preprocessed here and submitted with `VisualData` for the engine's
//! sequential M-RoPE path.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use hipfire_arch_qwen35::spec_emit::Qwen35Emit;
use hipfire_engine::emit::{emit_reasoning_token, emit_visible_token, stage_terminal_tool_calls};
use hipfire_engine::terminal::{
    batch_bind_active, batch_check_abort, batch_clear_terminal_at_generation,
    batch_mark_ready_with_pending, batch_transition_to_queued, batch_wait_decision,
    BatchAttemptScope, BatchGeneration, LaneTicket, CLIENT_TERMINAL_COMMIT_TIMEOUT,
};
use hipfire_generate::ar::{
    emit_active_route_cancel, emit_active_route_done, emit_generation_start, GenerationRoute,
    GenerationRouteScope,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::prompt_frame::{
    continuation_suffix_tool_results, qwen35_grammar_on, AssistantPrefix, ChatFrame,
    JinjaChatFrame, Message, Role, ThinkMode, ToolCall,
};
use hipfire_runtime::serve::{
    Continuation, DoneReason, Event, SlotEngineConfig, SubmitRequest, VisualData,
};
use hipfire_runtime::spec::{ClientEvent, SpecEmitCtx};
use hipfire_runtime::tokenizer::Tokenizer;

fn emit_qwen_ar_slot_error<W: std::io::Write>(
    stdout: &mut W,
    id: &str,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    hipfire_generate::ar::emit_generation_error(
        GenerationRoute::QwenAr,
        stdout,
        Some(id),
        message,
        class,
        retryable,
        rolled_back,
    );
}

/// The per-arch multi-slot engine behind the four-method surface
/// `handle_generate` drives (`submit` / `close` / `reset` / `shutdown`).
///
/// This enum is THE daemon-side arch dispatch point for slot mode. A new
/// model family with a slot engine adds a variant here and a spawn arm —
/// nothing else in this file may name a concrete engine type. The wire
/// types (`SubmitRequest`, `Event`, `Continuation`, `VisualData`) already
/// live family-neutral in `hipfire_runtime::serve`, so the variant only
/// binds the engine itself.
///
/// Deliberately an enum, not a trait: with one implementation a trait would
/// guess at the abstraction boundary. When a second family exists and the
/// variants want behavior beyond these four calls, graduate this into a
/// trait shaped by that port (the same driver-style rule the run-loop
/// extraction follows — see `hipfire-arch-qwen35::serve_engine`).
/// The per-arch multi-slot engine behind the four-method surface
/// `handle_generate` drives (`submit` / `close` / `reset` / `shutdown`).
///
/// Slot-mode arch dispatch goes through `hipfire_loader::carrier_for`:
/// the carrier that claims `arch_id` overrides `Carrier::spawn_slot_engine`
/// and returns an arch-erased `Box<dyn SlotEngineHandle>`. `loader` depends
/// on the arch crates, so the concrete engine is boxed there without a
/// loader -> arch -> loader cycle — this file never names a concrete engine
/// type. A new model family with a slot engine only adds a `spawn_slot_engine`
/// override on its carrier; nothing here changes.
type AnySlotEngine = Box<dyn hipfire_runtime::serve::SlotEngineHandle>;

/// Resolve the carrier for `arch_id` and spawn its multi-slot engine. The
/// family-neutral `SlotEngineConfig` carries every serve.* knob; the carrier
/// maps it onto its own engine config. `Err` names the arch when no carrier
/// provides a slot engine.
fn spawn_slot_engine(arch_id: u32, cfg: SlotEngineConfig) -> Result<AnySlotEngine, String> {
    let carrier = hipfire_loader::carrier_for(arch_id).ok_or_else(|| {
        format!("no carrier claims arch_id {arch_id} — slot mode cannot dispatch")
    })?;
    carrier
        .spawn_slot_engine(cfg)
        .map_err(|e| format!("no multi-slot engine for arch_id {arch_id}: {e}"))
}

/// Slot-engine load parameters parsed from a `load` message. The daemon's
/// sequential arm keeps its own parsing; this owns the multi-slot route's.
pub struct SlotLoadParams {
    pub mtp_k: usize,
    /// The `params.mtp_mode` tri-state. `mtp_k > 0` cannot distinguish `auto`
    /// from an explicit `on`, so this rides beside it into `EngineConfig`
    /// for the host-spill decline.
    pub mtp_mode: hipfire_config::MtpMode,
    /// Raw `params.kv_mode` (empty = env/config), resolved engine-side
    /// through `QWEN35_SLOTS_POLICY`.
    pub kv_mode_raw: String,
    /// Effective KV backend. The capability gate already refused anything but
    /// legacy; it rides into `EngineConfig` so `Rig::build` fails closed on a
    /// bypassed gate.
    pub kv_backend: String,
    /// Display name of the tier the engine will run, for the loaded ACK and
    /// kv_mode diag.
    pub kv_mode_resolved: &'static str,
    /// Shared `PagePool` for the multi-slot engine (auto-sized when pages=0).
    pub multi_slot_paged: bool,
    pub multi_slot_paged_pages: usize,
    pub multi_slot_paged_backend: String,
    pub dflash_draft: Option<PathBuf>,
    pub dflash_required: bool,
    /// Optional `params.max_batch_tokens` override. When present it wins over
    /// the `serve.max_batch_tokens` process-config value — the CLI serve path
    /// clamps the built-in default to the multi-slot scratch capacity and
    /// passes the result on the wire, and the daemon must honour that clamp
    /// rather than re-reading the unclamped OnceLock snapshot.
    pub max_batch_tokens: Option<usize>,
}

impl SlotLoadParams {
    pub fn from_load_msg(msg: &serde_json::Value) -> Self {
        let param = |k: &str| msg.get("params").and_then(|p| p.get(k));
        let param_str = |k: &str| param(k).and_then(|v| v.as_str());
        // `mtp_mode` on/auto only requests a depth here (`mtp_k` > 0); `off`
        // is 0. Whether the head arms is decided later by the producer-
        // capability gate (`SlotEngine::spawn`: `auto` arms only a producer
        // with a measured AR floor, `on` always). mtp_k sets depth. An integer param
        // is explicit; the CLI's `"auto"` resolves to HIPFIRE_MTP_K/config, else
        // MTP_K_DEFAULT (3); an absent param (non-CLI client) means 4.
        // The tri-state survives beside mtp_k: the engine's host-spill
        // decline reads the mode, not the depth. Absent normalizes to
        // "off" (default Off); the mtp_k rule below (empty/off -> 0) is
        // verbatim.
        let mtp_mode_raw = param_str("mtp_mode").unwrap_or("off");
        let mtp_mode = hipfire_config::MtpMode::parse(mtp_mode_raw);
        let mtp_k = if mtp_mode_raw.is_empty() || mtp_mode_raw == "off" {
            0
        } else {
            match param("mtp_k") {
                None => 4,
                Some(v) => v.as_u64().map(|k| k as usize).unwrap_or_else(|| {
                    hipfire_runtime::config::get()
                        .mtp_k
                        .unwrap_or(hipfire_config::MTP_K_DEFAULT)
                }),
            }
        };
        // Same precedence as the sequential loader: load param > env/config.
        let kv_mode_raw = param_str("kv_mode").unwrap_or("").to_string();
        let kv_backend = param_str("kv_backend").unwrap_or("legacy").to_string();
        let kv_mode_resolved = {
            let raw = if kv_mode_raw.is_empty() {
                hipfire_runtime::config::get().kv_mode.clone()
            } else {
                kv_mode_raw.clone()
            };
            hipfire_runtime::kv_mode::qwen_k_display_name(
                hipfire_runtime::kv_mode::resolve(
                    &raw,
                    &hipfire_runtime::kv_mode::QWEN35_SLOTS_POLICY,
                )
                .mode,
            )
        };
        // DFlash2 draft: HIPFIRE_DFLASH_DRAFT (non-empty wins, empty opts out)
        // > params.draft; dflash_mode=off suppresses either way.
        let dflash_mode = param_str("dflash_mode").unwrap_or("auto");
        let raw_draft: Option<String> = match hipfire_config::developer_var("HIPFIRE_DFLASH_DRAFT")
            .ok()
            .as_deref()
        {
            Some("") => None,
            Some(p) => Some(p.to_string()),
            None => param_str("draft")
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        };
        let dflash_draft = if dflash_mode == "off" {
            if let Some(d) = raw_draft {
                eprintln!("[hipfire-daemon] dflash_mode=off — skipping draft load ({d})");
            }
            None
        } else {
            raw_draft.map(PathBuf::from)
        };
        Self {
            mtp_k,
            mtp_mode,
            kv_mode_raw,
            kv_backend,
            kv_mode_resolved,
            multi_slot_paged: param("multi_slot_paged")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            multi_slot_paged_pages: param("multi_slot_paged_pages")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(0),
            multi_slot_paged_backend: param("multi_slot_paged_backend")
                .and_then(|v| v.as_str())
                .unwrap_or("arena")
                .to_string(),
            dflash_draft,
            dflash_required: dflash_mode == "on",
            max_batch_tokens: param("max_batch_tokens")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
        }
    }
}

/// The multi-slot route renders `hfq.chat_template()` directly.
const SLOT_CHAT_TEMPLATE_SOURCE: &str =
    "embedded HFQ tokenizer_config.chat_template (multi-slot route renders it directly)";

/// Daemon-owned slot backend. Owns one weight copy via SlotEngine.
pub struct SlotBackend {
    chat_template: Option<String>,
    /// `reasoning_effort` capability of the template this route actually
    /// renders, probed once at load.
    effort_probe: crate::effort_probe::EffortProbe,
    /// Provenance of `chat_template` (`None` iff there is no template).
    chat_template_source: Option<&'static str>,
    engine: AnySlotEngine,
    tokenizer: Tokenizer,
    arch_str: String,
    /// `params.mtp_mode` after the producer-capability gate (what the engine
    /// serves: `auto` on a producer without a measured AR floor is `off`).
    mtp_mode_effective: &'static str,
    dim: usize,
    layers: usize,
    vocab: usize,
    is_vl: bool,
    /// Post-gate audio-tower / video-leg residency, published verbatim in
    /// the `loaded` ack (`audio` / `video`) so `/v1/models` can advertise
    /// the modalities the serve actually accepts.
    is_audio: bool,
    is_video: bool,
    vision_config: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionConfig>,
    /// Per-slot context cap handed to the engine at load. Admission-side
    /// only (the engine re-enforces it per token); kept here so an
    /// over-budget request can be rejected with an actionable message
    /// before it occupies a slot.
    cap_tokens: usize,
    /// Sampling-default inputs shared with the sequential route
    /// ([`resolve_slot_sampling`]): carrier arch ladder + `.hfq` recs.
    arch_id: u32,
    rec_temperature: Option<f32>,
    rec_top_p: Option<f32>,
    active: AtomicUsize,
    tool_grammar: bool,
    pending_tools: Mutex<PendingToolBroker>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PendingToolCall {
    id: String,
    name: String,
}

#[derive(Clone, Debug)]
struct PendingToolTurn {
    session: u64,
    convo: Vec<u64>,
    calls: Vec<PendingToolCall>,
}

#[derive(Default)]
struct PendingToolBroker {
    by_session: HashMap<u64, PendingToolTurn>,
    by_call: HashMap<String, u64>,
}

#[derive(Debug)]
struct ToolResultClaim {
    session: u64,
    results: Vec<String>,
}

impl PendingToolBroker {
    fn remove_session(&mut self, session: u64) {
        if let Some(turn) = self.by_session.remove(&session) {
            for call in turn.calls {
                self.by_call.remove(&call.id);
            }
        }
    }

    fn register(&mut self, session: u64, convo: Vec<u64>, calls: &[ToolCall]) {
        self.remove_session(session);
        let calls = calls
            .iter()
            .filter_map(|call| {
                call.id.as_ref().map(|id| PendingToolCall {
                    id: id.clone(),
                    name: call.name.clone(),
                })
            })
            .collect::<Vec<_>>();
        if calls.is_empty() {
            return;
        }
        for call in &calls {
            self.by_call.insert(call.id.clone(), session);
        }
        self.by_session.insert(
            session,
            PendingToolTurn {
                session,
                convo,
                calls,
            },
        );
    }

    /// Resolve and consume one complete tool-result turn atomically. Consuming
    /// before submit makes retries fail closed; the caller closes the session
    /// if admission or generation subsequently fails.
    fn claim(
        &mut self,
        messages: &[Message],
        convo: &[u64],
    ) -> Result<Option<ToolResultClaim>, String> {
        let results = trailing_tool_results(messages)?;
        if results.is_empty() {
            return Ok(None);
        }
        let mut sessions = HashSet::new();
        for (_, id, _) in &results {
            let session = self
                .by_call
                .get(id)
                .copied()
                .ok_or_else(|| format!("unknown or stale tool_call_id {id:?}"))?;
            sessions.insert(session);
        }
        if sessions.len() != 1 {
            return Err("tool results span multiple pending slot sessions".to_string());
        }
        let session = *sessions.iter().next().expect("one session");
        let pending = self
            .by_session
            .get(&session)
            .ok_or_else(|| "pending tool turn disappeared".to_string())?;
        if pending.session != session || pending.convo != convo {
            return Err("tool results do not match the pending conversation".to_string());
        }
        if pending.calls.len() != results.len() {
            return Err("tool-result turn must answer every pending call exactly once".to_string());
        }
        let mut by_id = HashMap::new();
        for (name, id, content) in results {
            if by_id.insert(id.clone(), (name, content)).is_some() {
                return Err(format!("duplicate tool_call_id {id:?}"));
            }
        }
        let mut ordered = Vec::with_capacity(pending.calls.len());
        for call in &pending.calls {
            let Some((name, content)) = by_id.remove(&call.id) else {
                return Err(format!("missing result for tool_call_id {:?}", call.id));
            };
            if name.as_deref().is_some_and(|name| name != call.name) {
                return Err(format!(
                    "tool result name does not match pending call {:?}",
                    call.id
                ));
            }
            ordered.push(content);
        }
        self.remove_session(session);
        Ok(Some(ToolResultClaim {
            session,
            results: ordered,
        }))
    }
}

/// JSON shape for the swap-tier snapshot carried by the load ack and every
/// `done` event. `tier` is derived from the effective budgets, so a disabled
/// or read-only disk tier reads `host`/`none` rather than the configured value.
pub(crate) fn swap_stats_json(stats: &hipfire_runtime::serve::EngineStats) -> serde_json::Value {
    let tier = match (stats.swap_host_budget > 0, stats.swap_disk_budget > 0) {
        (true, true) => "host+disk",
        (true, false) => "host",
        (false, true) => "disk",
        (false, false) => "none",
    };
    serde_json::json!({
        "tier": tier,
        "evictions": stats.evictions,
        "restores": stats.restores,
        "failures": stats.swap_failures,
        "host_budget": stats.swap_host_budget,
        "host_used": stats.swap_host_used,
        "disk_budget": stats.swap_disk_budget,
        "disk_used": stats.swap_disk_used,
    })
}

/// Default swap directory for a daemon that did not set `serve.swap_dir`.
///
/// Per process, under the user's runtime directory when one exists, so two
/// daemons (multi-GPU or test ports) never share spill files. The temp
/// fallback encodes the pid for the same reason. `SwapStore` still creates it
/// 0700, requires this user to own it, and locks it for the store's lifetime.
fn default_swap_dir() -> PathBuf {
    let xdg = std::env::var_os("XDG_RUNTIME_DIR");
    let temp = std::env::temp_dir();
    sweep_dead_swap_dirs(xdg.as_deref(), &temp);
    default_swap_dir_from(xdg.as_deref(), &temp, std::process::id())
}

/// Remove per-pid swap dirs left by crashed daemons. `$XDG_RUNTIME_DIR` is
/// usually cleaned on logout; the `$TMPDIR` fallback is not, so a crash there
/// would otherwise leak a directory of snapshots per run.
#[cfg(target_os = "linux")]
fn sweep_dead_swap_dirs(xdg_runtime: Option<&std::ffi::OsStr>, temp: &std::path::Path) {
    let (parent, prefix) = match xdg_runtime.filter(|dir| !dir.is_empty()) {
        Some(runtime) => (PathBuf::from(runtime).join("hipfire").join("swap"), ""),
        None => (temp.to_path_buf(), "hipfire-swap-"),
    };
    sweep_dead_pid_dirs(&parent, prefix);
}

#[cfg(not(target_os = "linux"))]
fn sweep_dead_swap_dirs(_xdg_runtime: Option<&std::ffi::OsStr>, _temp: &std::path::Path) {}

#[cfg(target_os = "linux")]
fn sweep_dead_pid_dirs(parent: &std::path::Path, prefix: &str) -> usize {
    use std::os::unix::fs::MetadataExt;
    let Ok(entries) = std::fs::read_dir(parent) else {
        return 0;
    };
    let current = std::process::id();
    let uid = std::fs::metadata("/proc/self").ok().map(|meta| meta.uid());
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(pid) = name
            .strip_prefix(prefix)
            .and_then(|rest| rest.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == current || std::path::Path::new("/proc").join(pid.to_string()).exists() {
            continue;
        }
        let Ok(meta) = std::fs::symlink_metadata(entry.path()) else {
            continue;
        };
        if meta.file_type().is_symlink() || !meta.is_dir() {
            continue;
        }
        if uid.is_some_and(|uid| meta.uid() != uid) {
            continue;
        }
        if std::fs::remove_dir_all(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

fn default_swap_dir_from(
    xdg_runtime: Option<&std::ffi::OsStr>,
    temp: &std::path::Path,
    pid: u32,
) -> PathBuf {
    if let Some(runtime) = xdg_runtime.filter(|dir| !dir.is_empty()) {
        return PathBuf::from(runtime)
            .join("hipfire")
            .join("swap")
            .join(pid.to_string());
    }
    temp.join(format!("hipfire-swap-{pid}"))
}

/// Resolve the swap policy from the process config, falling back to the
/// registered defaults when a key is absent. The CLI projects `serve --swap-*`
/// into this same config layer, so flags, environment, and TOML cannot drift.
fn resolve_swap_policy(process: &hipfire_config::ProcessConfig) -> (u64, u64, PathBuf) {
    let config_u64 = |key: &str| {
        process.values.get(key).and_then(|value| match value {
            hipfire_config::ConfigValue::Integer(i) => u64::try_from(*i).ok(),
            _ => None,
        })
    };
    let host = config_u64("serve.swap_host_bytes")
        .unwrap_or(hipfire_runtime::swap::DEFAULT_HOST_BUDGET_BYTES);
    let disk = config_u64("serve.swap_disk_bytes")
        .unwrap_or(hipfire_runtime::swap::DEFAULT_DISK_BUDGET_BYTES);
    let dir = process
        .values
        .get("serve.swap_dir")
        .and_then(|value| match value {
            hipfire_config::ConfigValue::String(s) if !s.is_empty() => Some(PathBuf::from(s)),
            _ => None,
        })
        .unwrap_or_else(default_swap_dir);
    (host, disk, dir)
}

impl SlotBackend {
    /// CPU preflight then GPU load. Called only when experimental_multi_slot load is requested.
    /// `kv_mode_raw`/`kv_backend` are the load's effective KV selector and
    /// backend (already gated by `validate_load_caps` to the static tier
    /// ladder and `legacy`); they ride into `EngineConfig` so `Rig::build`
    /// fails closed if a tier-less or non-legacy value arrives.
    ///
    /// `vision_mode` / `vision` are the load's resolved vision-tower ladder
    /// (`params.vision_mode` + `params.vision`/`HIPFIRE_VISION_SIDECAR`,
    /// already gated by the caller). They exist because the multi-slot route
    /// used to resolve the `.vl` sibling on its own, ignoring both — so a VL
    /// trunk paid the tower's ~1 GB under the documented text-only default,
    /// and an explicit `serve --vision` was dropped here.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        model_path: &str,
        n_slots: usize,
        cap_tokens: usize,
        prefill_chunk: usize,
        mtp_k: usize,
        mtp_mode: hipfire_config::MtpMode,
        kv_mode_raw: &str,
        kv_backend: &str,
        multi_slot_paged: bool,
        multi_slot_paged_pages: usize,
        multi_slot_paged_backend: &str,
        dflash_draft: Option<PathBuf>,
        dflash_required: bool,
        vision_mode: &str,
        vision: Option<String>,
        max_batch_tokens_override: Option<usize>,
    ) -> Result<Self, String> {
        // CPU preflight: open HFQ, arch, VL, config, tokenizer.
        let preflight = cpu_preflight(model_path, vision_mode, vision.as_deref())?;
        let arch_id = preflight.arch_id;
        let arch_str = preflight.arch_str.clone();
        // The engine applies the same gate to its depth (`SlotEngine::spawn`);
        // this is the verdict the load ack reports.
        let mtp_mode_effective = hipfire_config::mtp_effective_mode(
            mtp_mode,
            hipfire_loader::carrier_for(arch_id)
                .is_some_and(|carrier| carrier.mtp_measured_ar_floor()),
        )
        .name();
        let dim = preflight.dim;
        let layers = preflight.layers;
        let chat_template = preflight.chat_template;
        let vocab = preflight.vocab;
        let tokenizer = preflight.tokenizer;
        let is_vl = preflight.is_vl;
        let is_audio = preflight.audio;
        let is_video = preflight.video;
        let vision_config = preflight.vision_config;
        let rec_temperature = preflight.rec_temperature;
        let rec_top_p = preflight.rec_top_p;
        // Read prefix cache config (spec §4.5–4.6). Keys are registered in
        // hipfire-config as serve.prefix_cache / serve.prefix_cache_max_bytes
        // with env HIPFIRE_SERVE_PREFIX_CACHE*. Default false/0.
        let prefix_cache = hipfire_config::active_or_local_process_config()
            .values
            .get("serve.prefix_cache")
            .and_then(|v| {
                if let hipfire_config::ConfigValue::Bool(b) = v {
                    Some(*b)
                } else {
                    None
                }
            })
            .unwrap_or(false);
        let prefix_cache_max_bytes = hipfire_config::active_or_local_process_config()
            .values
            .get("serve.prefix_cache_max_bytes")
            .and_then(|v| {
                if let hipfire_config::ConfigValue::Integer(i) = v {
                    Some(*i as u64)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        // Read the global trunk-row budget and minimum prefill quantum (spec
        // §5.2 S2 / §5.3 S3). Keys are registered in hipfire-config as
        // serve.max_batch_tokens / serve.prefill_min_tokens with env
        // HIPFIRE_SERVE_MAX_BATCH_TOKENS / HIPFIRE_SERVE_PREFILL_MIN_TOKENS.
        // Defaults 4096/1 (the registered config defaults).
        let max_batch_tokens = max_batch_tokens_override
            .or_else(|| {
                hipfire_config::active_or_local_process_config()
                    .values
                    .get("serve.max_batch_tokens")
                    .and_then(|v| {
                        if let hipfire_config::ConfigValue::Integer(i) = v {
                            Some(*i as usize)
                        } else {
                            None
                        }
                    })
            })
            .unwrap_or(4096);
        let prefill_min_tokens = hipfire_config::active_or_local_process_config()
            .values
            .get("serve.prefill_min_tokens")
            .and_then(|v| {
                if let hipfire_config::ConfigValue::Integer(i) = v {
                    Some(*i as usize)
                } else {
                    None
                }
            })
            .unwrap_or(1);
        // Read the bounded waiting room config (spec §5.3 S3). Keys are
        // registered in hipfire-config as serve.max_queue /
        // serve.max_queue_bytes / serve.queue_timeout_ms with env
        // HIPFIRE_SERVE_MAX_QUEUE / HIPFIRE_SERVE_MAX_QUEUE_BYTES /
        // HIPFIRE_SERVE_QUEUE_TIMEOUT_MS. Defaults are the registered
        // `DefaultValue::Integer` entries in hipfire-config's field table:
        // serve.max_queue = 64, serve.max_queue_bytes = 268435456 (256 MiB),
        // serve.queue_timeout_ms = 600000 (10 min). The CLI startup guard rejects
        // serve.max_queue=0 for multi-slot; Rig::build also refuses
        // WaitQueue::new(0, _) as a backstop.
        let process_config = hipfire_config::active_or_local_process_config();
        let wait_max_count =
            integer_or_registered_default(&process_config.values, "serve.max_queue") as usize;
        let wait_max_bytes =
            integer_or_registered_default(&process_config.values, "serve.max_queue_bytes") as u64;
        let queue_timeout_ms =
            integer_or_registered_default(&process_config.values, "serve.queue_timeout_ms") as u64;
        // Read the structured-output jump-forward flag (spec §7.3 G3).
        // Key registered as serve.structured_jump_forward with env
        // HIPFIRE_SERVE_STRUCTURED_JUMP_FORWARD. Default false — no
        // behavior change on the constrained-decode path.
        let structured_jump_forward = hipfire_config::active_or_local_process_config()
            .values
            .get("serve.structured_jump_forward")
            .and_then(|v| {
                if let hipfire_config::ConfigValue::Bool(b) = v {
                    Some(*b)
                } else {
                    None
                }
            })
            .unwrap_or(false);

        let vl_path = preflight.vl_path;
        let tool_grammar = qwen35_grammar_on(
            hipfire_config::developer_var("HIPFIRE_QWEN35_GRAMMAR")
                .ok()
                .as_deref(),
            model_path,
        );

        let n_slots = n_slots.max(1);
        let cap_tokens = cap_tokens.max(1);
        let prefill_chunk = prefill_chunk.max(1).min(cap_tokens);

        let (swap_host_bytes, swap_disk_bytes, swap_dir) =
            resolve_swap_policy(&hipfire_config::active_or_local_process_config());
        eprintln!(
            "serve: swap policy host-budget={swap_host_bytes} B disk-budget={swap_disk_bytes} B dir={}",
            swap_dir.display()
        );

        // Arch dispatch point: the carrier that claims `arch_id` spawns its
        // multi-slot engine via `Carrier::spawn_slot_engine`. Everything
        // downstream drives the family-neutral `SlotEngineHandle` surface
        // only.
        let engine = spawn_slot_engine(
            arch_id,
            SlotEngineConfig {
                model_path: PathBuf::from(model_path),
                n_slots,
                cap_tokens,
                prefill_chunk,
                host_budget_bytes: swap_host_bytes,
                swap_dir,
                swap_disk_bytes,
                mtp_k,
                mtp_mode,
                is_vl,
                vl_path,
                vision_mode: vision_mode.to_string(),
                kv_mode_raw: kv_mode_raw.to_string(),
                kv_backend: kv_backend.to_string(),
                multi_slot_paged,
                multi_slot_paged_pages,
                multi_slot_paged_backend: multi_slot_paged_backend.to_string(),
                prefix_cache,
                prefix_cache_max_bytes,
                max_batch_tokens,
                prefill_min_tokens,
                wait_max_count,
                wait_max_bytes,
                queue_timeout_ms,
                structured_jump_forward,
                dflash_draft,
                dflash_required,
            },
        )?;

        // This route renders the HFQ's embedded template directly (no loader
        // overrides / built-in fallback), so that is the template probed and
        // the source reported.
        let effort_probe =
            crate::effort_probe::probe_loaded_effort(Some(&tokenizer), chat_template.as_deref());
        let chat_template_source = chat_template.as_ref().map(|_| SLOT_CHAT_TEMPLATE_SOURCE);
        Ok(Self {
            engine,
            effort_probe,
            chat_template_source,
            tokenizer,
            arch_str,
            mtp_mode_effective,
            dim,
            chat_template,
            layers,
            vocab,
            is_vl,
            is_audio,
            is_video,
            vision_config,
            cap_tokens,
            arch_id,
            rec_temperature,
            rec_top_p,
            active: AtomicUsize::new(0),
            tool_grammar,
            pending_tools: Mutex::new(PendingToolBroker::default()),
        })
    }

    pub fn arch_str(&self) -> &str {
        &self.arch_str
    }
    /// The MTP mode this slot engine serves after the producer-capability gate.
    pub fn mtp_mode_effective(&self) -> &'static str {
        self.mtp_mode_effective
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn layers(&self) -> usize {
        self.layers
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn is_vl(&self) -> bool {
        self.is_vl
    }
    pub fn is_audio(&self) -> bool {
        self.is_audio
    }
    pub fn is_video(&self) -> bool {
        self.is_video
    }

    /// Load-time `reasoning_effort` probe of the rendered template.
    pub fn effort_probe(&self) -> &crate::effort_probe::EffortProbe {
        &self.effort_probe
    }

    /// Provenance of the rendered chat template, for the probe diagnostic.
    pub fn chat_template_source(&self) -> Option<&'static str> {
        self.chat_template_source
    }

    pub fn active_count(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    fn acquire_guard(&self) -> Option<SlotGuard<'_>> {
        let prev = self.active.fetch_add(1, Ordering::SeqCst);
        if prev >= 32 {
            self.active.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(SlotGuard { backend: self })
    }

    /// Drop broker state for sessions the engine swept on its own. Called at
    /// the start of each generate/embed request and on `Accepted` (an admit
    /// can sweep), so a swept tool session's pending entries do not linger.
    fn reap_closed_sessions(&self) {
        let closed = self.engine.take_closed_sessions();
        if closed.is_empty() {
            return;
        }
        let mut pending = self
            .pending_tools
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for session in closed {
            pending.remove_session(session);
        }
    }

    fn close_session(&self, session: u64) {
        self.pending_tools
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove_session(session);
        let _ = self.engine.close(session);
    }
    /// Reset idle sessions. May reject if requests are in flight. Checked teardown.
    pub fn reset(&self) -> Result<(), String> {
        if self.active_count() > 0 {
            return Err("reset refused: slot requests active".to_string());
        }
        let res = self.engine.reset();
        if res.is_ok() {
            // Clear keyed registry on successful teardown is done by daemon, but also clear here for safety.
            hipfire_engine::terminal::batch_clear_all_terminals();
            let mut pending = self
                .pending_tools
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            pending.by_session.clear();
            pending.by_call.clear();
        }
        res
    }

    /// Checked consuming shutdown for unload / model swap. The caller must
    /// first own the sole Arc and verify no request guards are active.
    pub fn shutdown(self) -> Result<(), String> {
        if self.active_count() > 0 {
            return Err("unload refused: slot requests active".to_string());
        }
        self.engine.shutdown()
    }

    /// Embedding handler for experimental slot mode: parse the `embed` wire
    /// body, hold an activity guard for the duration (so an unload or model
    /// swap cannot tear the engine down mid-encode), run it on the engine, and
    /// write the same `embedded` envelope the sequential route writes — the
    /// CLI, the HTTP layer and the container client need no change to reach
    /// the slot path.
    ///
    /// One wire request is one encode request. Concurrency comes from the
    /// daemon spawning a worker per request (as the generate route does) and
    /// from the engine merging the requests that arrive together into one
    /// batched forward.
    pub fn handle_embed<W: std::io::Write>(
        &self,
        msg: &serde_json::Value,
        stdout: &mut W,
        id: &str,
    ) {
        // Bound thread creation before spawning, same shape and reason as the
        // generate route: a burst must not spawn a thread per rejected
        // request, and the worker's own guard remains the hard bound.
        //
        // Embed replies are correlated by (id, attempt_id=0) — the worker
        // never enters a BatchAttemptScope, so `emit_active_attempt_error`
        // would write NOTHING (TLS 0 is the reserved uncorrelated channel)
        // and the caller would wait forever. The uncorrelated emitter is the
        // one whose envelope matches the pending key the caller registered.
        if self.active_count() >= 32 {
            hipfire_engine::emit::emit_uncorrelated_error(
                stdout,
                Some(id),
                "too many concurrent slot requests (bounded worker limit hit)",
                "overload",
                true,
                false,
            );
            return;
        }
        let Some(_guard) = self.acquire_guard() else {
            hipfire_engine::emit::emit_uncorrelated_error(
                stdout,
                Some(id),
                "too many concurrent slot requests (bounded worker limit hit)",
                "overload",
                true,
                false,
            );
            return;
        };
        self.reap_closed_sessions();
        hipfire_generate::embed::generate_embed_slot(&*self.engine, stdout, id, msg);
    }

    /// Generate handler for experimental slot mode.
    ///
    /// Requires `experimental_multi_slot=true` on the wire, validates
    /// supported wire fields (images allowed when the loaded model has a
    /// vision encoder), builds prompt/convo/continuation, submits with
    /// sampling, streams token/reasoning, handles abort via keyed registry +
    /// close, and performs two-phase terminal commit with byte-identical done.
    pub fn handle_generate<W: std::io::Write>(
        &self,
        msg: &serde_json::Value,
        stdout: &mut W,
        id: &str,
        attempt_id: u64,
        admission: BatchGeneration,
    ) -> Result<(), String> {
        // Guard active count and ensure keyed entry cleared on every exit.
        let Some(_guard) = self.acquire_guard() else {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "too many concurrent slot requests (bounded worker limit hit)",
                // Overload, not validation: all slots busy is a transient
                // capacity condition → HTTP 429 + Retry-After, not a
                // client-fixable 400.
                "overload",
                true,
                false,
            );
            let _ = stdout.flush();
            batch_clear_terminal_at_generation(id, attempt_id, admission);
            return Ok(());
        };
        // Ensure keyed registry entry cleared on exit (including error paths).
        struct ClearOnExit<'a> {
            id: String,
            attempt_id: u64,
            admission: BatchGeneration,
            _marker: std::marker::PhantomData<&'a ()>,
        }
        impl Drop for ClearOnExit<'_> {
            fn drop(&mut self) {
                batch_clear_terminal_at_generation(&self.id, self.attempt_id, self.admission);
            }
        }
        let _clear = ClearOnExit {
            id: id.to_string(),
            attempt_id,
            admission,
            _marker: std::marker::PhantomData,
        };
        hipfire_engine::terminal::set_active_attempt_id(attempt_id);
        let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
        self.reap_closed_sessions();

        // Require experimental marker.
        if !is_experimental_generate(msg) {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "experimental_multi_slot=true required for slot backend",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }

        // Defensive validation of unsupported wire fields.
        if let Some(err) = validate_generate_caps(msg) {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                &err,
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }

        let (temperature, top_p, max_tokens, fit) =
            match resolve_slot_sampling(msg, self.arch_id, self.rec_temperature, self.rec_top_p) {
                Ok(resolved) => resolved,
                Err(err) => {
                    hipfire_engine::emit::emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &err,
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return Ok(());
                }
            };
        let top_k = msg.get("top_k").and_then(|v| v.as_i64()).unwrap_or(0);
        if !(0..=i32::MAX as i64).contains(&top_k) {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "top_k must fit a non-negative 32-bit integer",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        // The parallel sampler compiles 20- and 64-wide kernels; anything
        // larger would be silently capped to a 64-candidate pool, so reject
        // it up front instead. 0 keeps the engine default (20).
        if top_k > 64 {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "invalid top_k: must be <= 64 (the sampler's widest compiled \
                 kernel); use top_p / min_p for tighter shaping",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        let client_seed = match hipfire_engine::wire_seed::parse_wire_seed(msg.get("seed")) {
            Ok(seed) => seed,
            Err(reason) => {
                hipfire_engine::emit::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("seed: {reason}"),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return Ok(());
            }
        };
        let seed = hipfire_engine::scheduler::request_seed_for(
            &hipfire_engine::terminal::AttemptKey::new(id, attempt_id),
            client_seed,
        );

        // Token penalties, mirroring the sequential path's request parsing
        // (main.rs): HF-style `repetition_penalty` is accepted as an alias,
        // the window defaults to 128 like the sequential daemon, and the
        // window is clamped to the engine's buffer capacity. min_p rides
        // along — the slot sampler's kernel supports it.
        let presence_penalty = msg
            .get("presence_penalty")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            .max(0.0) as f32;
        let frequency_penalty = msg
            .get("frequency_penalty")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            .max(0.0) as f32;
        let repeat_penalty = msg
            .get("repeat_penalty")
            .or_else(|| msg.get("repetition_penalty"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0)
            .max(1.0) as f32;
        let repeat_window = msg
            .get("repeat_window")
            .and_then(|v| v.as_u64())
            .unwrap_or(128)
            // Mirror of the engine's REPEAT_WINDOW_MAX (the engine clamps
            // again on admit; this keeps the emitted accepted-request view
            // honest about the window that will actually apply).
            .min(2048) as usize;
        let min_p = msg
            .get("min_p")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            .max(0.0) as f32;

        let max_think_tokens = msg
            .get("max_think_tokens")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                msg.get("params")
                    .and_then(|p| p.get("max_think_tokens"))
                    .and_then(|v| v.as_u64())
            });

        // Finite caps (max_think_tokens >= 2) are ENFORCED on this route:
        // once the cursor consumes the budget inside a think span, the
        // grammar mask allows only the think close, forcing the span to end
        // through the normal commit path (vLLM thinking_token_budget
        // parity). Enforcement requires the tokenizer's special close id;
        // engines without one fall back to uncapped thinking (documented).

        // --- Projected reasoning authority ---
        let has_think = self.tokenizer.special_token_id("<think>").is_some();
        let thinking_enabled_opt = msg.get("thinking_enabled").and_then(|v| v.as_bool());
        let assistant_prefix_opt = msg.get("assistant_prefix").and_then(|v| v.as_str());
        // Raw effort rung, spelled exactly as the sequential route consumes it
        // (`reasoning_effort`, with the project-custom `thinking_mode` alias).
        // Kept verbatim — no lowercasing and no rung validation here — so the
        // template stays the authority on which rungs it accepts. A rejected
        // rung surfaces as a render error (validation), never a silent
        // downgrade.
        let raw_reasoning_effort = msg
            .get("reasoning_effort")
            .or_else(|| msg.get("thinking_mode"))
            .and_then(|v| v.as_str());
        if max_think_tokens == Some(1)
            && (thinking_enabled_opt == Some(true) || assistant_prefix_opt == Some("open_think"))
        {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "max_think_tokens=1 contradicts enabled/open-think framing",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        // Validate assistant_prefix values
        let assistant_prefix_valid = matches!(
            assistant_prefix_opt,
            None | Some("open_think") | Some("closed_think") | Some("plain")
        );
        if !assistant_prefix_valid {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "assistant_prefix must be open_think|closed_think|plain",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        // Derive enable_thinking and expected prefix according to authority.
        // thinking_enabled and assistant_prefix are authority; max_think_tokens is fallback only.
        let (enable_thinking, expected_prefix): (bool, AssistantPrefix) =
            if let Some(enabled) = thinking_enabled_opt {
                let exp = if enabled {
                    if !has_think {
                        // No think token but thinking requested -> contradiction fail closed
                        hipfire_engine::emit::emit_active_attempt_error(
                            stdout,
                            Some(id),
                            "thinking_enabled true but model has no <think> token",
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        return Ok(());
                    }
                    AssistantPrefix::OpenThink
                } else if has_think {
                    AssistantPrefix::ClosedThink
                } else {
                    AssistantPrefix::Plain
                };
                // Validate assistant_prefix if present
                if let Some(ap_str) = assistant_prefix_opt {
                    let actual = match ap_str {
                        "open_think" => AssistantPrefix::OpenThink,
                        "closed_think" => AssistantPrefix::ClosedThink,
                        _ => AssistantPrefix::Plain,
                    };
                    if actual != exp {
                        hipfire_engine::emit::emit_active_attempt_error(
                            stdout,
                            Some(id),
                            &format!(
                                "thinking_enabled {enabled} contradicts assistant_prefix {ap_str}"
                            ),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        return Ok(());
                    }
                }
                (enabled, exp)
            } else if let Some(ap_str) = assistant_prefix_opt {
                let actual = match ap_str {
                    "open_think" => AssistantPrefix::OpenThink,
                    "closed_think" => AssistantPrefix::ClosedThink,
                    _ => AssistantPrefix::Plain,
                };
                // assistant_prefix is authority when thinking_enabled absent
                let enabled = match actual {
                    AssistantPrefix::OpenThink => {
                        if !has_think {
                            hipfire_engine::emit::emit_active_attempt_error(
                                stdout,
                                Some(id),
                                "assistant_prefix open_think but model has no <think> token",
                                "validation",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            return Ok(());
                        }
                        true
                    }
                    AssistantPrefix::ClosedThink => false,
                    AssistantPrefix::Plain => false,
                };
                // Also validate against max_think fallback if present and contradictory?
                // max_think is fallback only, so when assistant_prefix present we ignore it, but if max_think==1 vs open_think contradiction we still prefer authority and ignore fallback.
                (enabled, actual)
            } else {
                // Fallback: `thinking_enabled` and `assistant_prefix` are both
                // absent, so defer to the shared sequential derivation — an
                // explicit disable rung (`none`/`off`/`chat`) turns thinking
                // off, and the immediate-close sentinel (1) does too. Same
                // result as before, plus the disable rungs the sequential and
                // batch routes already honour; still gated on this model
                // actually carrying a `<think>` token.
                let (enabled, _) = hipfire_engine::prompt::qwen_jinja_reasoning(
                    None,
                    raw_reasoning_effort,
                    max_think_tokens.unwrap_or(0) as usize,
                );
                let enabled = enabled && has_think;
                let exp = if enabled {
                    AssistantPrefix::OpenThink
                } else if has_think {
                    AssistantPrefix::ClosedThink
                } else {
                    AssistantPrefix::Plain
                };
                (enabled, exp)
            };
        // Additional contradiction: if max_think==1 but thinking_enabled true? Already handled as authority wins, but spec says reject contradictions. If fallback path derived enabled true but max_think==1 would have been false, but since thinking_enabled absent we use max_think, so not contradictory.
        // If has_think false, ensure plain regardless.

        // The effort rung handed to the Jinja template; see
        // `slot_jinja_effort` for the suppression rule and why it is not keyed
        // on this route's `enable_thinking`.
        let jinja_effort =
            slot_jinja_effort(thinking_enabled_opt, raw_reasoning_effort, max_think_tokens);

        // Consume only the gateway-projected tool contract. Raw OpenAI
        // tool_choice never reaches this owner.
        let tools = match projected_tools(msg) {
            Ok(tools) => tools,
            Err(reason) => {
                hipfire_engine::emit::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &reason,
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return Ok(());
            }
        };
        if let Err(reason) = validate_projected_tool_policy(msg, tools) {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                &reason,
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        if let Err(reason) = validate_tool_schema_bounds(tools) {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                &reason,
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        let messages = match project_jinja_messages(msg) {
            Ok(messages) => messages,
            Err(reason) => {
                hipfire_engine::emit::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &reason,
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return Ok(());
            }
        };
        // Tool-choice projection may add a transient system instruction to
        // the rendered prompt.  The gateway also carries its normalized view
        // from immediately before that mutation; use it for stable ownership
        // across the subsequent tool-result request (whose choice is commonly
        // `none`).  Direct daemon callers retain the historical fallback.
        let conversation_messages = if let Some(identity) = msg.get("conversation_messages") {
            let identity_request = serde_json::json!({ "messages": identity });
            match project_jinja_messages(&identity_request) {
                Ok(messages) => messages,
                Err(reason) => {
                    hipfire_engine::emit::emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!("invalid projected conversation identity: {reason}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return Ok(());
                }
            }
        } else {
            messages.clone()
        };
        let convo = build_convo_from_messages(&conversation_messages);
        let last_user = messages
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .expect("projection requires a user message");

        let slot_images = match extract_slot_images(msg) {
            Ok(image) => image,
            Err(reason) => {
                hipfire_engine::emit::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &reason,
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return Ok(());
            }
        };
        if !slot_images.is_empty() && !self.is_vl() {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                "model has no vision encoder",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }

        let mut visual_data = None;
        let (prompt_tokens, prefix) = if !slot_images.is_empty() {
            // The projected message view carries the system text; fall back to
            // the top-level `system` field for prompt-only requests.
            let system = messages
                .iter()
                .find(|m| m.role == Role::System)
                .map(|m| m.content.as_str())
                .or_else(|| msg.get("system").and_then(|v| v.as_str()));
            match build_slot_vl_prompt(
                self,
                &slot_images,
                system,
                &last_user.content,
                expected_prefix,
            ) {
                Ok((tokens, vd)) => {
                    // The VL builder frames the assistant turn itself from
                    // `expected_prefix`, so that is authoritative here.
                    visual_data = Some(vd);
                    (tokens, expected_prefix)
                }
                Err(reason) => {
                    hipfire_engine::emit::emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &reason,
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return Ok(());
                }
            }
        } else if let Some(template) = self.chat_template.as_deref() {
            let frame = JinjaChatFrame {
                tokenizer: &self.tokenizer,
                template,
                system: None,
                user: "",
                enable_thinking,
                bos_token: None,
                reasoning_strength: None,
                reasoning_effort: jinja_effort,
            };
            let rendered = match frame.render_messages(&messages, tools, None) {
                Ok(rendered) => rendered,
                Err(reason) => {
                    hipfire_engine::emit::emit_active_attempt_error(
                        stdout,
                        Some(id),
                        &format!("multi_slot Jinja render failed: {reason}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    return Ok(());
                }
            };
            // The template owns the generation suffix: read the assistant
            // framing back from what it emitted. A thinking-on request that
            // the template answers with a bare `assistant\n` (standard
            // Qwen3-family behaviour — the model opens `<think>` itself)
            // frames `Plain`, not `ClosedThink`.
            let framing = hipfire_engine::emit::render_assistant_prefix(&rendered);
            (self.tokenizer.encode(&rendered), framing)
        } else {
            if tools.is_some()
                || messages
                    .iter()
                    .any(|message| message.role == Role::Tool || !message.tool_calls.is_empty())
            {
                hipfire_engine::emit::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    "tool requests require a model chat_template in experimental multi-slot",
                    "unsupported",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return Ok(());
            }
            let (system, turns, _) = project_messages(msg).expect("validated projection");
            let history: Vec<(Role, &str)> = turns
                .iter()
                .map(|(role, text)| (*role, text.as_str()))
                .collect();
            let frame = ChatFrame {
                tokenizer: &self.tokenizer,
                system: system.as_deref(),
                user: &last_user.content,
                assistant_prefix: expected_prefix,
                raw: false,
            };
            // `ChatFrame` frames the assistant turn itself from
            // `expected_prefix`, so that is authoritative here.
            (frame.build_multi_turn(&history), expected_prefix)
        };

        // Every branch above now reports the framing the prompt actually
        // ends in. There is nothing left to reconcile: the Jinja render owns
        // the generation suffix, and the two builders frame it themselves.
        let started_in_think = matches!(prefix, AssistantPrefix::OpenThink);
        let prompt_len = prompt_tokens.len();
        // A client that omitted max_tokens gets the default clamped to the
        // room left after the prompt, exactly like the sequential route.
        let max_tokens = hipfire_generate::common::fit_max_tokens_if(
            fit,
            max_tokens,
            prompt_len,
            self.cap_tokens,
        );
        // vLLM-style admission: prompt + generation budget must fit the
        // slot's context cap. The engine's per-token ctx guard would stop
        // the decode at the cap anyway, but that surfaces as a silently
        // truncated answer; an explicit rejection up front tells the client
        // the request was too large and by how much.
        if prompt_len.saturating_add(max_tokens) > self.cap_tokens {
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                &format!(
                    "request requires context {prompt_len} prompt + {max_tokens} max_tokens = {}, \
                     exceeding the slot context cap of {} tokens — reduce max_tokens or shorten the prompt",
                    // saturating: a direct-wire max_tokens = usize::MAX would
                    // overflow a plain `prompt_len + max_tokens` and panic in
                    // an overflow-checked build while the request is live.
                    prompt_len.saturating_add(max_tokens),
                    self.cap_tokens
                ),
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        let tool_claim = match self
            .pending_tools
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .claim(&messages, &convo)
        {
            Ok(claim) => claim,
            Err(reason) => {
                hipfire_engine::emit::emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &reason,
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return Ok(());
            }
        };
        let claimed_session = tool_claim.as_ref().map(|claim| claim.session);
        // Extract JSON Schema for structured output (spec §7 G1/G2). The
        // schema was already compiled and validated in validate_generate_caps;
        // here we extract the raw schema Value to pass to the engine, which
        // recompiles it into a per-request SchemaMatcher on admit.
        let json_schema = match msg
            .get("response_format")
            .and_then(|v| v.get("type"))
            .and_then(|v| v.as_str())
        {
            Some("json_schema") => msg
                .get("response_format")
                .and_then(|v| v.get("json_schema"))
                .and_then(|v| v.get("schema"))
                .cloned(),
            Some("json_object") => Some(serde_json::json!({ "type": "object" })),
            _ => None,
        };
        let continuation = if visual_data.is_some() {
            Continuation::Cold
        } else if let Some(claim) = tool_claim {
            Continuation::ToolResults {
                tokens: continuation_suffix_tool_results(&self.tokenizer, &claim.results, prefix),
                session: claim.session,
            }
        } else if convo.len() >= 3 {
            Continuation::UserTurn(hipfire_runtime::prompt_frame::continuation_suffix(
                &self.tokenizer,
                &last_user.content,
                prefix,
            ))
        } else {
            Continuation::Cold
        };

        // Own the route and its exact `(id, attempt)` start latch for the
        // complete post-start lifecycle. Pre-start validation above remains
        // intentionally on the uncorrelated direct error path.
        let _route_scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, id);
        emit_generation_start(GenerationRoute::QwenAr, stdout, id, started_in_think);
        if batch_check_abort(id, attempt_id, admission) {
            if let Some(session) = claimed_session {
                self.close_session(session);
            }
            emit_active_route_cancel(stdout, id, 0);
            batch_clear_terminal_at_generation(id, attempt_id, admission);
            return Ok(());
        }
        // Transition queued (stdin reader announced). Bind will happen after Accepted.
        let _ = batch_transition_to_queued(id, attempt_id, admission);
        let (tx, rx) = mpsc::channel::<Event>();
        // Cancellation identity (spec §4.6 C6): a request parked in the
        // engine's waiting room has no session id yet, so the abort path
        // cancels it by this tag. (id, attempt_id) is unique per attempt.
        let request_tag = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            id.hash(&mut h);
            attempt_id.hash(&mut h);
            h.finish()
        };
        let req = SubmitRequest {
            prompt_tokens,
            convo: convo.clone(),
            continuation,
            max_tokens,
            temperature,
            top_p,
            top_k: top_k as i32,
            seed,
            repeat_window,
            repeat_penalty,
            presence_penalty,
            frequency_penalty,
            min_p,
            logprobs_top_k: (msg.get("logprobs").and_then(|v| v.as_bool()) == Some(true)).then(
                || {
                    msg.get("top_logprobs")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize
                },
            ),
            visual_data,
            json_schema,
            started_in_think,
            // Enforced thinking budget: absent/0 = uncapped; the 1-sentinel
            // never enters a think span so the budget cannot engage.
            think_budget: max_think_tokens
                .filter(|&n| n >= 2)
                .map(|n| n as usize)
                .unwrap_or(usize::MAX),
            // Canonical pending-input bytes (spec §5.3): the engine's wait
            // queue charges its byte cap against this, so daemon-side
            // waiters must carry the real prompt weight — a hardcoded 0
            // (floored to 1 downstream) made the byte cap unbindable.
            queue_bytes: canonical_prompt_bytes(msg),
            request_tag,
            reply: tx,
        };
        if let Err(e) = self.engine.submit(req) {
            if let Some(session) = claimed_session {
                self.close_session(session);
            }
            emit_qwen_ar_slot_error(
                stdout,
                id,
                &format!("multi_slot submit: {e}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }

        // Stream loop with keyed terminal binding
        let mut accepted_session: Option<u64> = None;
        let mut accepted_ticket: Option<LaneTicket> = None;
        let mut produced: usize = 0;
        let mut cached_tokens: usize = 0;
        let mut prefill_tokens: usize = prompt_len;
        let t_start = Instant::now();
        // The parser's initial think state follows the prompt's framing, not
        // the request's `enable_thinking`. `enable_thinking` is a Jinja input:
        // a thinking-on Qwen3-family render frames no opener and the model
        // emits `<think>` itself, so the parser must start outside the
        // reasoning span and open it on that token. Equating the two rejected
        // every thinking-on request.
        let think_mode = if started_in_think {
            ThinkMode::Low
        } else {
            ThinkMode::NonThink
        };
        // `validate_generate_caps` already rejected a malformed field; this is
        // the validated request-local list the streaming matcher consumes.
        let stop =
            hipfire_runtime::stop_sequence::parse_stop_field(msg.get("stop")).unwrap_or_default();
        let mut emitter = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &self.tokenizer,
            eos: self.tokenizer.eos_id,
            im_end: self.tokenizer.special_token_id("<|im_end|>"),
            tools,
            enable_grammar: self.tool_grammar,
            stop,
            max_think: 0,
            max_tokens,
            assistant_prefix: prefix,
            think_mode,
            decoded_vocab: None,
        });
        let render_events =
            |events: Vec<ClientEvent>,
             stdout: &mut W,
             tool_calls: &mut Vec<ToolCall>,
             logprob: &Option<(f32, Vec<(u32, f32)>)>,
             tokenizer: &hipfire_runtime::tokenizer::Tokenizer| {
                for event in events {
                    match event {
                        ClientEvent::Token(text) => {
                            if let Some((lp, top)) = logprob {
                                let top_json: Vec<serde_json::Value> = top
                                    .iter()
                                    .map(|(tok_id, tok_lp)| {
                                        serde_json::json!({
                                            "token": tokenizer.decode(&[*tok_id]),
                                            "logprob": tok_lp,
                                            "bytes": tokenizer.decode_bytes(&[*tok_id]),
                                        })
                                    })
                                    .collect();
                                let envelope = serde_json::json!({
                                    "type": "token",
                                    "id": id,
                                    "text": text,
                                    "logprob": lp,
                                    "top_logprobs": top_json,
                                    "attempt_id": hipfire_engine::terminal::active_attempt_id(),
                                });
                                let _ = writeln!(stdout, "{}", envelope);
                                let _ = stdout.flush();
                            } else {
                                emit_visible_token(stdout, id, &text);
                            }
                        }
                        ClientEvent::Reasoning(text) => emit_reasoning_token(stdout, id, &text),
                        ClientEvent::Committed { .. } => {}
                        ClientEvent::ToolCalls(mut calls) => tool_calls.append(&mut calls),
                    }
                }
            };
        let mut terminal_tool_calls = Vec::new();
        let mut first_token = true;

        let mut done_reason: Option<(DoneReason, usize)> = None;
        let mut stop_matched = false;
        let mut rejected: Option<(hipfire_runtime::serve::RejectClass, String)> = None;
        // Set when the engine channel dropped without a terminal event —
        // the engine thread died (panic/GPU fault). Must NOT be reported as
        // a normal stop: the client would see a clean-looking truncated
        // answer while the session leaks resident on its slot.
        let mut engine_died = false;

        loop {
            if batch_check_abort(id, attempt_id, admission) {
                drop(rx);
                // A request parked in the engine's waiting room has no
                // session to close — cancel it by tag so it leaves the queue
                // (and its queue-byte charge) immediately instead of being
                // admitted against a dead receiver later (spec §4.6 C6/A11).
                self.engine.cancel_waiting(request_tag);
                if let Some(sess) = accepted_session.take() {
                    self.close_session(sess);
                } else if let Some(session) = claimed_session {
                    self.close_session(session);
                }
                emit_active_route_cancel(stdout, id, produced);
                return Ok(());
            }
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(ev) => match ev {
                    Event::Accepted {
                        session,
                        reused,
                        prefill,
                    } => {
                        // An admit may have swept parked sessions; prune their
                        // broker state before this request can name one.
                        self.reap_closed_sessions();
                        if let Some(expected) =
                            claimed_session.filter(|expected| *expected != session)
                        {
                            self.close_session(expected);
                            accepted_session = Some(session);
                            rejected = Some((
                                hipfire_runtime::serve::RejectClass::Internal,
                                "tool-result reentry was admitted on the wrong slot session"
                                    .to_string(),
                            ));
                            break;
                        }
                        accepted_session = Some(session);
                        cached_tokens = reused;
                        prefill_tokens = prefill;
                        let ticket = LaneTicket {
                            lane: (session % 1024) as usize,
                            generation: session,
                            admission,
                        };
                        accepted_ticket = Some(ticket);
                        let _ = batch_bind_active(id, attempt_id, admission, ticket);
                    }
                    Event::Token {
                        id: tok_id,
                        logprob,
                        top_logprobs,
                    } => {
                        let outcome = if first_token {
                            first_token = false;
                            emitter.begin(tok_id)
                        } else {
                            emitter.observe(tok_id)
                        };
                        let lp = logprob.map(|lp| (lp, top_logprobs.unwrap_or_default()));
                        render_events(
                            outcome.events,
                            stdout,
                            &mut terminal_tool_calls,
                            &lp,
                            &self.tokenizer,
                        );
                        produced += 1;
                        // A user stop matched in this token's visible text.
                        // The matcher has already withheld the stop bytes;
                        // end the turn as `stop` and close the session (the
                        // KV now holds unrendered text past the stop, so a
                        // later turn must cold-reprefill rather than extend
                        // a history the client never saw).
                        if matches!(
                            outcome.stop,
                            Some(hipfire_runtime::spec::StopReason::StopSequence)
                        ) {
                            stop_matched = true;
                            done_reason = Some((DoneReason::Eos, produced));
                            break;
                        }
                    }
                    Event::Done { reason, generated } => {
                        done_reason = Some((reason, generated));
                        break;
                    }
                    Event::Rejected { class, reason } => {
                        rejected = Some((class, reason));
                        break;
                    }
                },
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    engine_died = true;
                    break;
                }
            }
        }

        if engine_died {
            // The engine thread dropped its end without Done/Rejected. Close
            // the session (its resident state is untrustworthy) and fail
            // loudly — never a normal "stop" with partial tokens.
            drop(rx);
            if let Some(sess) = accepted_session.take() {
                let _ = self.engine.close(sess);
            }
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                &format!(
                    "multi_slot engine terminated without completing request {} \
                     (engine thread exited; {produced} tokens were produced)",
                    attempt_id
                ),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }

        if let Some((class, reason)) = rejected {
            drop(rx);
            if let Some(sess) = accepted_session.take() {
                self.close_session(sess);
            } else if let Some(session) = claimed_session {
                self.close_session(session);
            }
            let (kind, retryable) = match class {
                hipfire_runtime::serve::RejectClass::Overload => ("overload", true),
                hipfire_runtime::serve::RejectClass::Validation => ("validation", false),
                hipfire_runtime::serve::RejectClass::Internal => ("internal", false),
                hipfire_runtime::serve::RejectClass::Cancel => ("cancel", false),
            };
            hipfire_engine::emit::emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("multi_slot rejected: {reason}"),
                kind,
                retryable,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        if accepted_session.is_none() {
            if let Some(session) = claimed_session {
                self.close_session(session);
            }
            emit_qwen_ar_slot_error(
                stdout,
                id,
                "multi_slot stream ended before session acceptance",
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }
        // A matched stop is terminal: release the session now (the engine
        // cancels the in-flight request and frees the slot) so the client's
        // next turn cannot extend KV past text that was never emitted.
        if stop_matched {
            if let Some(session) = accepted_session.take() {
                self.close_session(session);
            } else if let Some(session) = claimed_session {
                self.close_session(session);
            }
        }
        let summary = emitter.finish();
        render_events(
            summary.events,
            stdout,
            &mut terminal_tool_calls,
            &None,
            &self.tokenizer,
        );
        // Multi-slot keeps failing closed on an unfinished tool call at any
        // exit, under its previous `malformed_protocol` label.
        let unsafe_reason = match summary.finish_reason {
            "truncated_tool_call" => Some("malformed_protocol"),
            r @ "malformed_protocol" => Some(r),
            _ => None,
        };
        // A turn that ended inside `<think>` is a reasoning-only stop. Its
        // session is closed at commit instead of kept for reuse: the slot KV
        // holds an unclosed think span that no rendered history reproduces.
        let open_think = summary.open_think;
        if let Some(unsafe_reason) = unsafe_reason {
            if let Some(sess) = accepted_session.take() {
                self.close_session(sess);
            } else if let Some(session) = claimed_session {
                self.close_session(session);
            }
            emit_qwen_ar_slot_error(
                stdout,
                id,
                &format!("unsafe multi_slot terminal: {unsafe_reason}"),
                "generation",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        }

        let (reason, generated) = done_reason.unwrap_or((DoneReason::Eos, produced));
        if matches!(reason, DoneReason::ClientGone) {
            if let Some(sess) = accepted_session.take() {
                self.close_session(sess);
            } else if let Some(session) = claimed_session {
                self.close_session(session);
            }
            emit_active_route_cancel(stdout, id, produced);
            return Ok(());
        }

        // Stage normal done payload and await commit via keyed registry.
        let finish_reason = match reason {
            DoneReason::MaxTokens if !summary.decoded_eot => "length",
            _ if open_think => "stop",
            _ => summary.finish_reason,
        };
        if finish_reason != "tool_calls" {
            terminal_tool_calls.clear();
        } else {
            let Some(session) = accepted_session else {
                emit_qwen_ar_slot_error(
                    stdout,
                    id,
                    "tool-call terminal missing accepted slot session",
                    "internal",
                    false,
                    false,
                );
                let _ = stdout.flush();
                return Ok(());
            };
            for (index, call) in terminal_tool_calls.iter_mut().enumerate() {
                call.id = Some(format!("call_hf_{session:016x}_{attempt_id:016x}_{index}"));
            }
        }
        let total_s = t_start.elapsed().as_secs_f64();
        let tok_s = if total_s > 0.0 {
            produced as f64 / total_s
        } else {
            0.0
        };
        let mut pending_done = serde_json::json!({
            "type": "done",
            "id": id,
            "attempt_id": attempt_id,
            "finish_reason": finish_reason,
            "tokens": generated,
            "prompt_tokens": cached_tokens + prefill_tokens,
            "cached_tokens": cached_tokens,
            "prefill_tokens": prefill_tokens,
            "tok_s": (tok_s * 10.0).round() / 10.0,
            // Pushed on every done because a polled stats request would queue
            // behind generation; the gateway exports the latest snapshot.
            "engine_stats": self.engine.stats(),
        });
        stage_terminal_tool_calls(&mut pending_done, finish_reason, &terminal_tool_calls);
        // Live swap-tier counters for /health. The engine refreshes its stats
        // each step, so this is the state as of (at worst) the current step.
        pending_done["swap"] = swap_stats_json(&self.engine.stats());

        // Mark ready with exact pending done, publish commit_ready, poll keyed Commit/Abort with normal timeout
        let Some(ticket) = accepted_ticket else {
            if let Some(sess) = accepted_session.take() {
                self.close_session(sess);
            } else if let Some(session) = claimed_session {
                self.close_session(session);
            }
            emit_qwen_ar_slot_error(
                stdout,
                id,
                "slot terminal without a bound lane ticket",
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return Ok(());
        };
        let _ =
            batch_mark_ready_with_pending(id, attempt_id, admission, ticket, pending_done.clone());
        // commit_ready is the staged done with only `type` changed.
        pending_done["type"] = serde_json::Value::String("commit_ready".to_string());
        let _ = writeln!(stdout, "{}", pending_done);
        pending_done["type"] = serde_json::Value::String("done".to_string());
        let _ = stdout.flush();

        let decision =
            batch_wait_decision(id, attempt_id, admission, CLIENT_TERMINAL_COMMIT_TIMEOUT);
        match decision {
            hipfire_engine::terminal::ClientTerminalDecision::Commit => {
                if let Some(session) = accepted_session {
                    if open_think {
                        self.close_session(session);
                    } else {
                        self.pending_tools
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .register(session, convo.clone(), &terminal_tool_calls);
                    }
                }
                // Emit byte-identical done through the route adapter so the
                // keyed claim and route-start latch are retired together.
                emit_active_route_done(stdout, id, &pending_done);
            }
            hipfire_engine::terminal::ClientTerminalDecision::Abort => {
                if let Some(sess) = accepted_session.take() {
                    self.close_session(sess);
                } else if let Some(session) = claimed_session {
                    self.close_session(session);
                }
                emit_active_route_cancel(stdout, id, produced);
            }
        }
        Ok(())
    }
}

struct SlotGuard<'a> {
    backend: &'a SlotBackend,
}
impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        self.backend.active.fetch_sub(1, Ordering::SeqCst);
    }
}

// ── CPU preflight ────────────────────────────────────────────────────────────

struct Preflight {
    /// Raw HFQ arch id — the factory's dispatch key (5|6 = qwen3_5 today).
    arch_id: u32,
    arch_str: String,
    dim: usize,
    layers: usize,
    vocab: usize,
    tokenizer: Tokenizer,
    chat_template: Option<String>,
    is_vl: bool,
    /// Post-gate audio-tower and video-leg residency (arch 18 only; both
    /// ride the one `.vl` sidecar the vision gate already resolved).
    audio: bool,
    video: bool,
    vision_config: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionConfig>,
    /// Discovered .vl sidecar path. When set, the slot engine loads vision
    /// weights from this file instead of the trunk HFQ.
    vl_path: Option<PathBuf>,
    /// `.hfq`-baked author sampling recommendation (`generation_config`).
    rec_temperature: Option<f32>,
    rec_top_p: Option<f32>,
}

fn cpu_preflight(
    model_path: &str,
    vision_mode: &str,
    vision: Option<&str>,
) -> Result<Preflight, String> {
    // A carrier that owns the shape of its own slot checkpoint answers first.
    // arch 17's encoder is why this exists: its checkpoint may be a
    // safetensors directory (`HfqFile::open` cannot read one) and its geometry
    // lives in HF config metadata, not the qwen35 HFQ layout. `None` = no
    // carrier claims the path, so the HFQ preflight below owns both the
    // reading and the refusals.
    if let Some(over) = hipfire_loader::slot_preflight(std::path::Path::new(model_path)) {
        let p = over?;
        // The carrier reports the tower it FOUND (an explicit path, or the
        // `.vl` sibling); the load's ladder owns whether it is used — the
        // same rule the sequential arm applies (`vision_mode=off` is a hard
        // override; `on` requires a tower; `auto` and `resident` discover and
        // never fail closed — `resident` only changes how a found tower is
        // held, see `TowerResidency::from_vision_mode`). Without this gate the
        // mode never reached a carrier-delegated preflight at all, so a
        // multi-slot load asking for `auto` came back text-only and refused
        // media.
        let (vl_path, is_vl) = match vision_mode {
            "off" => (None, false),
            _ => {
                let explicit = vision.filter(|v| !v.is_empty()).map(PathBuf::from);
                match explicit.or(p.vl_path) {
                    Some(v) => (Some(v), true),
                    None if vision_mode == "on" && !p.is_vl => {
                        return Err(format!(
                            "vision_mode=on: no vision tower resolved for {model_path} \
                             (no explicit sidecar and no <stem>.vl sibling found)"
                        ))
                    }
                    None => (None, p.is_vl),
                }
            }
        };
        return Ok(Preflight {
            arch_id: p.arch_id,
            arch_str: p.arch_str,
            dim: p.dim,
            layers: p.layers,
            vocab: p.vocab,
            tokenizer: p.tokenizer,
            // Generate-only fields, and this route has no generate verb: no
            // chat template, no sampling recommendation, no VL splice. The
            // tower is loaded and driven by the arch crate itself.
            chat_template: None,
            is_vl,
            // Both towers ride the ONE `.vl` pack, so the mode gate that
            // stripped `vl_path` strips audio and the video leg with it.
            audio: p.audio && vl_path.is_some(),
            video: p.video && vl_path.is_some(),
            vision_config: None,
            vl_path,
            rec_temperature: None,
            rec_top_p: None,
        });
    }

    let hfq =
        HfqFile::open(std::path::Path::new(model_path)).map_err(|e| format!("open model: {e}"))?;
    validate_arch_id(hfq.arch_id)?;

    // ── .vl sidecar discovery ──────────────────────────────────────────
    // A .vl file is a standalone HFQM container with only vision tower
    // tensors + config.vision_config metadata. Discovery order:
    //   1. the caller's explicit sidecar (`params.vision` /
    //      `HIPFIRE_VISION_SIDECAR`), already filtered by `vision_mode`
    //   2. `<stem>.vl` sibling next to the trunk model
    // When a .vl file is found, it takes precedence over inline vision
    // tensors. If no .vl file exists, fall back to inline (backward compat).
    //
    // `vision_mode` is the SAME documented ladder the sequential load arm
    // applies (`apply_vision_mode_gate`): `off` never wires a
    // sidecar — not even an explicit one — so a text load pays no tower
    // VRAM; `on` requires a tower to resolve; `auto` and `resident` discover
    // and never fail closed (`resident` uploads the found tower whole, the
    // others stream it). The multi-slot arm used to
    // discover the sibling unconditionally, so a VL trunk paid the tower's
    // ~1 GB even under `vision_mode=off`, and `serve --vision`
    // was dropped on this route entirely.
    let has_inline_vision = is_vision_hfq(&hfq);
    let vl_path: Option<PathBuf> = if vision_mode == "off" {
        None
    } else {
        vision
            .map(PathBuf::from)
            .or_else(|| discover_vl_sidecar(model_path))
    };
    if vision_mode == "on" && vl_path.is_none() && !has_inline_vision {
        return Err(format!(
            "vision_mode=on: no vision tower resolved for {model_path} \
             (no explicit sidecar and no <stem>.vl sibling found)"
        ));
    }
    let is_vl = vl_path.is_some() || has_inline_vision;

    // Read vision_config from the .vl file when present, else from the trunk.
    let vision_config = if is_vl {
        if let Some(vl) = &vl_path {
            let vl_hfq =
                HfqFile::open(vl).map_err(|e| format!("open .vl file {}: {e}", vl.display()))?;
            Some(
                hipfire_arch_qwen35_vl::qwen35_vl::vision_config_from_hfq(&vl_hfq)
                    .ok_or_else(|| ".vl file missing vision_config in metadata".to_string())?,
            )
        } else {
            Some(
                hipfire_arch_qwen35_vl::qwen35_vl::vision_config_from_hfq(&hfq)
                    .ok_or_else(|| "VL model missing vision_config in metadata".to_string())?,
            )
        }
    } else {
        None
    };
    let config = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq)
        .map_err(|e| format!("qwen config: {e}"))?;
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| format!("tokenizer: {e}"))?;
    if is_vl {
        for name in ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"] {
            if tokenizer.special_token_id(name).is_none() {
                return Err(format!("VL tokenizer missing {name} special token"));
            }
        }
    }
    let chat_template = hfq.chat_template();
    if chat_template
        .as_deref()
        .is_some_and(|template| template.trim().is_empty())
    {
        return Err("empty chat_template".to_string());
    }
    let arch_str = hipfire_loader::arch_label(hfq.arch_id).to_string();
    let rec = hfq.recommended_sampling();
    Ok(Preflight {
        rec_temperature: rec.as_ref().and_then(|r| r.temperature),
        rec_top_p: rec.as_ref().and_then(|r| r.top_p),
        arch_id: hfq.arch_id,
        arch_str,
        dim: config.dim,
        layers: config.n_layers,
        vocab: config.vocab_size,
        tokenizer,
        chat_template,
        is_vl,
        // The HFQ slot route (chat arches) has no embed media legs.
        audio: false,
        video: false,
        vision_config,
        vl_path,
    })
}

// ── Validation helpers (pure) ────────────────────────────────────────────────

/// Refuse an arch the daemon's own (qwen35-HFQ-shaped) slot preflight cannot
/// describe. Only reached for a path no carrier claimed through
/// `Carrier::preflight_slot_model` — arch 17's encoder answers there, so its
/// `validate_arch_id` refusal never fires.
pub fn validate_arch_id(arch_id: u32) -> Result<(), String> {
    if matches!(arch_id, 5 | 6) {
        Ok(())
    } else {
        Err(format!(
            "experimental multi-slot supports arch_id 5|6 (qwen3_5) plus any arch whose \
             carrier provides a slot engine, got {arch_id}"
        ))
    }
}

pub fn is_vision_hfq(hfq: &HfqFile) -> bool {
    hfq.tensor_data("model.visual.patch_embed.proj.weight")
        .is_some()
}

/// Discover a `.vl` sidecar file for the given trunk model path.
///
/// Discovery order:
/// 1. `HIPFIRE_VL_FILE` env var — explicit override (any path)
/// 2. `<stem>.vl` sibling next to the trunk model, where `<stem>` also
///    strips `.hfq` and a quant suffix (`model.mq4v2.hfq` → `model.vl`)
///
/// Returns `None` when no .vl file is found (caller falls back to inline
/// vision tensors in the trunk for backward compat).
///
/// The probe list lives in [`hipfire_runtime::sidecar`] so the daemon, the
/// loader and the CLI capability advertisement cannot drift apart — they
/// already had three copies, and the `.vl` copies had lost the quant-suffix
/// stripping the `.mtp` probe has.
fn discover_vl_sidecar(model_path: &str) -> Option<PathBuf> {
    hipfire_runtime::sidecar::resolve_vl_sidecar(model_path)
}

/// Capability gate for an experimental multi-slot load, run before any GPU
/// work. `gpu_arch` is the initialized GPU's arch string (the caller reads
/// it off the live `Gpu`), so the fp8 admission is exact on every host —
/// tests parameterize it by string instead of assuming the host's card.
pub fn validate_load_caps(msg: &serde_json::Value, gpu_arch: &str) -> Option<String> {
    // continuous_batch_size
    if let Some(v) = msg
        .get("params")
        .and_then(|p| p.get("continuous_batch_size"))
        .and_then(|v| v.as_u64())
    {
        if v > 1 {
            return Some(
                "experimental_multi_slot and continuous_batch_size>1 are mutually exclusive"
                    .to_string(),
            );
        }
    }
    if let Some(v) = msg.get("continuous_batch_size").and_then(|v| v.as_u64()) {
        if v > 1 {
            return Some(
                "experimental_multi_slot and continuous_batch_size>1 are mutually exclusive"
                    .to_string(),
            );
        }
    }
    let tp = msg
        .get("params")
        .and_then(|p| p.get("tp"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1);
    let pp = msg
        .get("params")
        .and_then(|p| p.get("pp"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1);
    if tp != 1 || pp != 1 {
        return Some("experimental multi-slot requires pp=tp=1".to_string());
    }
    // The slot kernels own a fixed Q8 KV/state path plus the DFlash2
    // spec-decode sidecar (`params.draft` + `dflash_mode`). Other
    // speculative/eviction sidecars stay refused rather than silently
    // ignored.
    let params = msg.get("params");
    if params
        .and_then(|p| p.get("drafter"))
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.is_empty())
    {
        return Some("spec/drafter not supported in experimental multi-slot".to_string());
    }
    for (key, label) in [("prefill_compression", "PFlash")] {
        if params
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_str())
            .is_some_and(|v| !v.is_empty() && v != "off")
        {
            return Some(format!("{label} not supported in experimental multi-slot"));
        }
    }
    if params
        .and_then(|p| p.get("ngram_draft"))
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        return Some("n-gram speculation not supported in experimental multi-slot".to_string());
    }
    if params.and_then(|p| p.get("cask")).and_then(|v| v.as_bool()) == Some(true)
        || params
            .and_then(|p| p.get("cask_sidecar"))
            .and_then(|v| v.as_str())
            .is_some_and(|v| !v.is_empty())
    {
        return Some("CASK not supported in experimental multi-slot".to_string());
    }
    if params
        .and_then(|p| p.get("kv_adaptive"))
        .and_then(|v| v.as_str())
        .is_some_and(|v| !v.is_empty() && v != "off")
    {
        return Some("adaptive KV not supported in experimental multi-slot".to_string());
    }
    // The multi-slot DFlash2 path runs the default proposal width only
    // (`auto` resolves to it: this route never stages VMM batching).
    match hipfire_runtime::dflash::DflashBlockPolicy::from_param(
        params.and_then(|p| p.get("dflash_block")),
    ) {
        Err(e) => return Some(e),
        Ok(policy)
            if policy.resolve(false, 1) != hipfire_runtime::dflash::DFLASH_BLOCK_UNBATCHED =>
        {
            return Some(format!(
                "speculation.dflash_block other than {} not supported in experimental multi-slot",
                hipfire_runtime::dflash::DFLASH_BLOCK_UNBATCHED
            ));
        }
        Ok(_) => {}
    }
    // The slot engine resolves the full static KV ladder (q8, asym{2,3,4},
    // fwht{2,3,4}) plus the flat 2-byte tiers bf16/f16; fp8 is admitted on
    // exact gfx1201 only (descriptor-aware fp8 write/attend kernels exist
    // there). Rejected here — loudly, before any GPU work — rather than
    // silently downgraded to the q8 default or failed deep inside the
    // engine thread. The arch comes from the caller's initialized GPU so
    // this gate is exact on every host (tests parameterize it by string).
    if let Some(raw) = params
        .and_then(|p| p.get("kv_mode"))
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
    {
        let accepted_list = "q8|asym2|asym3|asym4|fwht2|fwht3|fwht4|bf16|f16, \
                             fp8 on gfx1201 only; 'auto'/unset = q8";
        match hipfire_runtime::kv_mode::resolve_qwen35_slots(raw, gpu_arch) {
            // Arch gate (or any future hard error): refuse with the named
            // message verbatim — it already names the arch and the fix.
            Err(e) => {
                return Some(format!(
                    "experimental multi-slot refuses kv_mode='{raw}': {e} \
                     (accepted: {accepted_list})"
                ));
            }
            // Unrecognized / unaccepted string: the pure resolver fell back
            // to the default WITH a warning — refuse here too, so a typo'd
            // tier is a loud load error, not a silent q8 downgrade.
            Ok(resolved) if resolved.warning.is_some() => {
                return Some(format!(
                    "experimental multi-slot does not support kv_mode='{raw}' \
                     (accepted: {accepted_list})"
                ));
            }
            Ok(_) => {}
        }
    }
    if let Some(v) = params
        .and_then(|p| p.get("kv_backend"))
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
    {
        match v.parse::<KvBackend>() {
            // Arena route: the fixed `SlotPool` slabs.
            Ok(KvBackend::Legacy) => {}
            // VMM on this engine is the shared q8 page pool
            // (`serve.multi_slot_paged_backend=vmm`), not a per-slot owner.
            // That pool has no fp8 kernels.
            Ok(KvBackend::Vmm) => {
                let raw = params
                    .and_then(|p| p.get("kv_mode"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let tier_ok = matches!(
                    hipfire_runtime::kv_mode::resolve_qwen35_slots(raw, gpu_arch),
                    Ok(r) if r.warning.is_none()
                        && matches!(r.mode, hipfire_runtime::kv_mode::KvMode::Q8)
                );
                if !tier_ok {
                    return Some(format!(
                        "multi-slot kv_backend=vmm is the shared VMM page pool and supports kv_mode=q8 only; got kv_mode={raw:?}"
                    ));
                }
            }
            Err(e) => return Some(e.to_string()),
        }
    }
    None
}

pub fn is_experimental_generate(msg: &serde_json::Value) -> bool {
    if msg.get("experimental_multi_slot").and_then(|v| v.as_bool()) == Some(true) {
        return true;
    }
    if msg
        .get("params")
        .and_then(|p| p.get("experimental_multi_slot"))
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        return true;
    }
    false
}

/// Effective `(temperature, top_p, max_tokens, max_tokens_fit)` for a slot
/// generate request. Defaults are the sequential route's
/// ([`hipfire_engine::scheduler::resolve_temp_top_p`],
/// [`hipfire_engine::scheduler::DEFAULT_GENERATE_MAX_TOKENS`], wire
/// `max_tokens_fit`); a present-but-malformed `max_tokens` is refused.
pub(crate) fn resolve_slot_sampling(
    msg: &serde_json::Value,
    arch_id: u32,
    rec_temperature: Option<f32>,
    rec_top_p: Option<f32>,
) -> Result<(f32, f32, usize, bool), String> {
    let (temperature, top_p) =
        hipfire_engine::scheduler::resolve_temp_top_p(msg, arch_id, rec_temperature, rec_top_p);
    let max_tokens = match msg.get("max_tokens") {
        None | Some(serde_json::Value::Null) => {
            hipfire_engine::scheduler::DEFAULT_GENERATE_MAX_TOKENS
        }
        Some(value) => match value.as_u64().and_then(|v| usize::try_from(v).ok()) {
            Some(v) if v > 0 => v,
            _ => return Err("max_tokens must be a positive integer".to_owned()),
        },
    };
    let fit = msg.get("max_tokens_fit").and_then(|v| v.as_bool()) == Some(true);
    Ok((temperature, top_p, max_tokens, fit))
}

/// The `reasoning_effort` rung handed to the Jinja template on this route.
///
/// Suppression mirrors `hipfire_engine::prompt::qwen_jinja_reasoning`: an
/// explicit `thinking_enabled = false`, a disable rung, or the immediate-close
/// sentinel (`max_think_tokens = 1`) leave the variable undefined so the
/// template's own default applies. `auto` is the client's "no preference"
/// spelling and is left undefined too. The rung itself is passed verbatim —
/// the template is the authority on which values it accepts, and a rejected
/// one fails the render as validation.
///
/// Deliberately NOT keyed on the route's `enable_thinking`: that flag
/// describes the `<think>`-token ChatML framing, and a template that opens
/// reasoning through a different token (e.g. `<ifm|think>`) keeps it false —
/// keying on it would silence the dial on exactly the models that use it.
fn slot_jinja_effort<'a>(
    thinking_enabled: Option<bool>,
    raw_effort: Option<&'a str>,
    max_think_tokens: Option<u64>,
) -> Option<&'a str> {
    let disable_rung = matches!(raw_effort, Some("none") | Some("off") | Some("chat"));
    if thinking_enabled == Some(false) || max_think_tokens == Some(1) || disable_rung {
        return None;
    }
    raw_effort.filter(|rung| *rung != "auto")
}

pub fn validate_generate_caps(msg: &serde_json::Value) -> Option<String> {
    // Optional numeric controls must not silently fall back to defaults when
    // present with the wrong JSON type. The parsing path uses as_f64/as_u64;
    // validate the shape here before those Option chains erase intent.
    for key in [
        "temperature",
        "top_p",
        "repeat_penalty",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "min_p",
    ] {
        if let Some(value) = msg.get(key) {
            if !value.is_null() && value.as_f64().is_none() {
                return Some(format!("{key} must be a number"));
            }
        }
    }
    for key in ["top_k", "repeat_window", "max_think_tokens"] {
        if let Some(value) = msg.get(key) {
            if !value.is_null() && value.as_u64().is_none() {
                return Some(format!("{key} must be a non-negative integer"));
            }
        }
    }
    if let Some(value) = msg.get("params").and_then(|p| p.get("max_think_tokens")) {
        if !value.is_null() && value.as_u64().is_none() {
            return Some("params.max_think_tokens must be a non-negative integer".to_string());
        }
    }
    // Images and tools are each supported in experimental multi-slot, but
    // not together: the VL prompt path splices image pads into a ChatFrame
    // user body and cannot render a tool contract. Reject the combination
    // rather than silently dropping the tools (spec §7.1).
    let has_image = msg.get("image").is_some_and(|v| !v.is_null())
        || msg.get("image_base64").is_some_and(|v| !v.is_null())
        || msg.get("image_url").is_some_and(|v| !v.is_null())
        || msg
            .get("messages")
            .and_then(|v| v.as_array())
            .is_some_and(|messages| {
                messages.iter().any(|message| {
                    message
                        .get("content")
                        .and_then(|v| v.as_array())
                        .is_some_and(|parts| {
                            parts.iter().any(|part| {
                                part.get("type").and_then(|v| v.as_str()) == Some("image_url")
                            })
                        })
                })
            });
    if has_image
        && msg
            .get("tools")
            .and_then(|v| v.as_array())
            .is_some_and(|tools| !tools.is_empty())
    {
        return Some(
            "images and tools together are not supported in experimental multi-slot".to_string(),
        );
    }
    // Stop sequences are honoured by the slot emitter's streaming matcher;
    // validate the wire shape here (string / array, count and character caps)
    // so a malformed field is a 400, not a silently dropped constraint.
    if let Err(e) = hipfire_runtime::stop_sequence::parse_stop_field(msg.get("stop")) {
        return Some(e);
    }
    // logprobs/top_logprobs are supported on the slot path: the engine
    // forces AR and attaches per-token log-probabilities. Validate the
    // OpenAI contract here; malformed values remain 400s.
    if let Some(v) = msg.get("logprobs") {
        if !v.is_boolean() {
            return Some("logprobs must be a boolean".to_string());
        }
    }
    if let Some(v) = msg.get("top_logprobs").filter(|v| !v.is_null()) {
        let n = match v.as_u64() {
            Some(n) => n,
            None => return Some("top_logprobs must be an integer between 0 and 20".to_string()),
        };
        if msg.get("logprobs").and_then(|v| v.as_bool()) != Some(true) {
            return Some("top_logprobs requires logprobs to be true".to_string());
        }
        if n > 20 {
            return Some("top_logprobs must be an integer between 0 and 20".to_string());
        }
    }
    // response_format: the ONLY supported type is json_schema (spec §7.1:
    // "reject missing requested semantics. An optimization bypass is
    // allowed, a silent semantic downgrade is not"). Any other type — e.g.
    // json_object — used to fall through validation and run unconstrained.
    let rf_type = msg
        .get("response_format")
        .and_then(|v| v.get("type"))
        .and_then(|v| v.as_str());
    match rf_type {
        None => {}
        Some("json_object") => {
            // Permissive JSON mode is enforced as the unconstrained object
            // schema; the gateway forwards it as json_schema, and a direct
            // daemon caller gets the same compiled mask.
            let schema = serde_json::json!({ "type": "object" });
            if let Err(e) =
                saddle_core::grammar::json::json_schema::CompiledSchema::compile(&schema)
            {
                return Some(format!("response_format json_object: {e}"));
            }
        }
        Some("json_schema") => {
            // Compile the schema with the saddle-core JSON Schema subset
            // compiler (spec §7 G1). Unsupported keywords, refs, regex,
            // recursion, and combinators are rejected here before the
            // request is submitted — no GPU work, no cache/session mutation
            // (spec §5.4 S4). A valid subset schema passes; the engine
            // recompiles on admit to build the per-request SchemaMatcher
            // cursor (spec §7.2 G2).
            let schema = msg
                .get("response_format")
                .and_then(|v| v.get("json_schema"))
                .and_then(|v| v.get("schema"));
            let schema = match schema {
                Some(s) => s,
                None => {
                    return Some(
                        "response_format json_schema requires a schema object \
                         (spec §7 G1)"
                            .to_string(),
                    );
                }
            };
            if let Err(e) = saddle_core::grammar::json::json_schema::CompiledSchema::compile(schema)
            {
                return Some(format!("response_format json_schema: {e}"));
            }
        }
        Some(other) => {
            return Some(format!(
                "response_format type '{other}' is not supported in \
                 experimental multi-slot (only json_schema)"
            ));
        }
    }
    // Token penalties and min_p are implemented by the slot sampler (in-kernel
    // repeat/presence/frequency over a per-slot recent-token window; min_p in
    // the top-p tail), so non-neutral values are honored, not refused. Only
    // out-of-range values are rejected here.
    for (key, lo, hi) in [
        ("repeat_penalty", 1.0, 2.0),
        // `repetition_penalty` is the OpenAI alias the sampler read resolves
        // (`.or_else(repetition_penalty)`); it must be range-checked too —
        // otherwise `{"repetition_penalty": 0.5}` silently clamps to 1.0 and
        // `{"repetition_penalty": 1e9}` reaches the sampler unclamped.
        ("repetition_penalty", 1.0, 2.0),
        ("presence_penalty", 0.0, 2.0),
        ("frequency_penalty", 0.0, 2.0),
        ("min_p", 0.0, 1.0),
        // temperature/top_p are consumed by the sampler with no daemon-side
        // check; only the HTTP gateway validated them. Mirror the ranges so
        // a direct-wire client cannot push -5 / 1e300 (f32::INFINITY) in.
        ("temperature", 0.0, 2.0),
    ] {
        if let Some(v) = msg.get(key).and_then(|v| v.as_f64()) {
            if !(lo..=hi).contains(&v) {
                return Some(format!("{key} must be within [{lo}, {hi}]"));
            }
        }
    }
    // top_p is (0, 1] — open at 0, so it cannot ride the inclusive table.
    if let Some(v) = msg.get("top_p").and_then(|v| v.as_f64()) {
        if !(v > 0.0 && v <= 1.0) {
            return Some("top_p must be within (0, 1]".to_string());
        }
    }
    // `reasoning_effort` is ACCEPTED: it is plumbed to the Jinja template (and
    // used to derive the enable/disable pair) exactly as on the sequential and
    // batch routes. A rung the model's template rejects surfaces as a render
    // error, which is validation — not a silent downgrade.
    // max_think_tokens >= 2 is ACCEPTED here (was refused): the engine now
    // enforces finite think budgets — the grammar cursor force-closes the
    // span at the budget (vLLM thinking_token_budget parity). Enforced
    // behavior must never be refused at the door.
    // Fields the wire accepts but the engine never reads must be refused,
    // not silently dropped: `n: 2` returning one completion is a silent
    // semantic downgrade (spec §7.1). The check reads the value as a NUMBER,
    // not as a u64: `n: 2.5` / `n: -1` deserialise as f64, and an `as_u64`
    // gate let both through as a silent single completion — the same
    // downgrade, spelled differently. Anything present that is not exactly
    // numeric 1 is refused (a string `n` cannot be honoured either).
    match msg.get("n") {
        None => {}
        Some(v) if v.is_null() => {}
        Some(v) if v.as_f64() == Some(1.0) => {}
        Some(_) => return Some("n != 1 not supported in experimental multi-slot".to_string()),
    }
    if msg.get("best_of").is_some_and(|v| !v.is_null()) {
        return Some("best_of not supported in experimental multi-slot".to_string());
    }
    if msg.get("logit_bias").is_some_and(|v| !v.is_null()) {
        return Some("logit_bias not supported in experimental multi-slot".to_string());
    }
    // `echo: false` (the OpenAI default, which this route honours by simply
    // not echoing) passes; `echo: true` and any non-boolean spelling (whose
    // intent cannot be read) are refused rather than silently ignored.
    match msg.get("echo") {
        None => {}
        Some(v) if v.is_null() => {}
        Some(v) if v.as_bool() == Some(false) => {}
        Some(_) => return Some("echo not supported in experimental multi-slot".to_string()),
    }
    if msg.get("suffix").is_some_and(|v| !v.is_null()) {
        return Some("suffix not supported in experimental multi-slot".to_string());
    }
    None
}

/// Integer config value for `key`, falling back to the registered default in
/// hipfire-config's field table (single source of truth: a hard-coded
/// fallback here drifts when the registered default changes).
///
/// Panics if `key` is not a registered integer field; that is a programmer
/// error caught by `waiting_room_fallbacks_are_the_registered_defaults`.
fn integer_or_registered_default(layer: &hipfire_config::ConfigLayer, key: &str) -> i64 {
    if let Some(hipfire_config::ConfigValue::Integer(i)) = layer.values.get(key) {
        return *i;
    }
    match hipfire_config::field(key).map(|f| f.default) {
        Some(hipfire_config::DefaultValue::Integer(i)) => i,
        _ => panic!("config key {key} is not a registered integer field"),
    }
}

// ── Message projection (pure) ────────────────────────────────────────────────

/// Canonical pending-input bytes charged to the admission queue (spec §5.3).
///
/// Charges the FULL serialized request, not just the prompt text: a large
/// `response_format.json_schema.schema`, `logit_bias` map, or `suffix`
/// payload otherwise bypasses the waiting-room byte cap entirely while
/// still occupying the same queue slot and memory.
pub fn canonical_prompt_bytes(msg: &serde_json::Value) -> u64 {
    serde_json::to_vec(msg)
        .map(|v| v.len() as u64)
        .unwrap_or(u64::MAX)
}

/// FNV-1a 64 hash of user turn text.
pub fn turn_hash(s: &str) -> u64 {
    turn_hash_named(None, s)
}

/// Hash with optional name for conversation identity.
pub fn turn_hash_named(name: Option<&str>, content: &str) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    if let Some(n) = name {
        for b in n.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xFF;
        h = h.wrapping_mul(0x100000001b3);
    }
    for b in content.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Tagged hash of all system messages. Prepended to convo to make system differences cold-miss.
pub fn system_hash(system: Option<&str>) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    for b in b"system:" {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    if let Some(s) = system {
        for b in s.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Project messages/prompt into (system, turns, last_user) per CLI slots semantics.
pub fn project_messages(
    msg: &serde_json::Value,
) -> Result<(Option<String>, Vec<(Role, String)>, String), String> {
    if let Some(arr) = msg.get("messages").and_then(|v| v.as_array()) {
        let mut system: Option<String> = None;
        let mut turns: Vec<(Role, String)> = Vec::new();
        for m in arr {
            let text = m
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            match m.get("role").and_then(|v| v.as_str()) {
                Some("system") => match system.as_mut() {
                    Some(existing) => {
                        existing.push('\n');
                        existing.push_str(&text);
                    }
                    None => system = Some(text),
                },
                Some("assistant") => turns.push((Role::Assistant, text)),
                Some("tool") => turns.push((Role::User, text)),
                _ => turns.push((Role::User, text)),
            }
        }
        let last_user = match turns.iter().rposition(|(r, _)| *r == Role::User) {
            Some(i) => turns.remove(i).1,
            None => return Err("messages must contain at least one user message".to_string()),
        };
        Ok((system, turns, last_user))
    } else {
        // Fallback to prompt + optional system
        let prompt = msg
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Hello")
            .to_owned();
        let system = msg
            .get("system")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());
        Ok((system, Vec::new(), prompt))
    }
}
fn project_jinja_messages(msg: &serde_json::Value) -> Result<Vec<Message>, String> {
    let owned_fallback;
    let messages = match msg.get("messages").and_then(|value| value.as_array()) {
        Some(messages) => messages.as_slice(),
        None => {
            owned_fallback = vec![serde_json::json!({
                "role": "user",
                "content": msg
                    .get("prompt")
                    .and_then(|value| value.as_str())
                    .unwrap_or("Hello"),
            })];
            owned_fallback.as_slice()
        }
    };
    let mut projected = Vec::with_capacity(messages.len());
    for message in messages {
        let role = match message.get("role").and_then(|value| value.as_str()) {
            Some("system") => Role::System,
            Some("assistant") => Role::Assistant,
            Some("tool") => Role::Tool,
            Some("user") | None => Role::User,
            Some(other) => {
                return Err(format!(
                    "message role {other:?} is not supported in experimental multi-slot"
                ))
            }
        };
        let tool_calls = message
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .map(|calls| {
                calls
                    .iter()
                    .map(|call| {
                        serde_json::from_value::<ToolCall>(call.clone())
                            .map_err(|error| format!("invalid assistant tool call: {error}"))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        if !tool_calls.is_empty() && role != Role::Assistant {
            return Err("tool_calls are only valid on assistant messages".to_string());
        }
        let tool_call_id = message
            .get("tool_call_id")
            .and_then(serde_json::Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_owned);
        if role == Role::Tool && tool_call_id.is_none() {
            return Err("tool message requires a non-empty tool_call_id".to_string());
        }
        projected.push(Message {
            role,
            content: message
                .get("content")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_owned(),
            reasoning_content: message
                .get("reasoning_content")
                .and_then(|value| value.as_str())
                .map(str::to_owned),
            name: message
                .get("name")
                .and_then(|value| value.as_str())
                .map(str::to_owned),
            rendered_name: None,
            tool_calls,
            tool_call_id,
            tool_plan: message
                .get("tool_plan")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned(),
        });
    }
    if !projected.iter().any(|message| message.role == Role::User) {
        return Err("messages must contain at least one user message".to_string());
    }
    Ok(projected)
}

fn projected_tools(msg: &serde_json::Value) -> Result<Option<&[serde_json::Value]>, String> {
    match msg.get("tools") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Array(tools)) if tools.is_empty() => Ok(None),
        Some(serde_json::Value::Array(tools)) => Ok(Some(tools)),
        Some(_) => Err("projected tools must be an array".to_string()),
    }
}

fn validate_projected_tool_policy(
    msg: &serde_json::Value,
    tools: Option<&[serde_json::Value]>,
) -> Result<(), String> {
    match msg.get("tool_choice_policy") {
        None => Ok(()),
        Some(serde_json::Value::String(policy)) if policy == "auto" => Ok(()),
        Some(serde_json::Value::String(policy)) if policy == "none" => {
            if tools.is_some() {
                Err("projected tool_choice none must not carry tools".to_string())
            } else {
                Ok(())
            }
        }
        Some(serde_json::Value::String(policy)) if policy == "required" => tools
            .filter(|tools| !tools.is_empty())
            .map(|_| ())
            .ok_or_else(|| "projected tool_choice required needs tools".to_string()),
        Some(serde_json::Value::Object(policy)) => {
            let name = policy
                .get("function")
                .and_then(serde_json::Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| "invalid projected function tool_choice".to_string())?;
            let tools =
                tools.ok_or_else(|| "projected function tool_choice needs tools".to_string())?;
            if tools.iter().all(|tool| {
                tool.get("function")
                    .unwrap_or(tool)
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    == Some(name)
            }) {
                Ok(())
            } else {
                Err("projected function tool_choice contains a different tool".to_string())
            }
        }
        Some(_) => Err("invalid projected tool_choice policy".to_string()),
    }
}

/// Refuse a tools array that would exceed the grammar compiler's input
/// bounds. `Matcher::with_config` builds the tool grammar with an internal
/// `expect` on these same bounds, so an over-bound request that slipped
/// through here would panic the per-request worker thread (and the whole
/// process on a panic=abort build). Validating at the trust boundary turns
/// it into a typed 400. Mirrors `saddle_core::grammar::json`'s accounting:
/// tool count, plus the byte total of every tool name and `required` entry.
fn validate_tool_schema_bounds(tools: Option<&[serde_json::Value]>) -> Result<(), String> {
    use saddle_core::grammar::json::{MAX_SCHEMA_BYTES, MAX_TOOL_SCHEMAS};
    let Some(tools) = tools else { return Ok(()) };
    if tools.len() > MAX_TOOL_SCHEMAS {
        return Err(format!(
            "tools: {} entries exceeds the grammar bound of {MAX_TOOL_SCHEMAS}",
            tools.len()
        ));
    }
    let mut total_bytes = 0usize;
    for (i, tool) in tools.iter().enumerate() {
        let func = tool.get("function").unwrap_or(tool);
        let name = func.get("name").and_then(serde_json::Value::as_str);
        if let Some(n) = name {
            if n.is_empty() {
                return Err(format!("tools[{i}]: empty tool name"));
            }
            total_bytes += n.len();
        }
        if let Some(required) = func
            .get("parameters")
            .and_then(|p| p.get("required"))
            .and_then(serde_json::Value::as_array)
        {
            for r in required {
                if let Some(s) = r.as_str() {
                    total_bytes += s.len();
                }
            }
        }
    }
    if total_bytes > MAX_SCHEMA_BYTES {
        return Err(format!(
            "tools: {total_bytes} bytes of names/required exceeds the grammar \
             bound of {MAX_SCHEMA_BYTES}"
        ));
    }
    Ok(())
}

fn trailing_tool_results(
    messages: &[Message],
) -> Result<Vec<(Option<String>, String, String)>, String> {
    if messages
        .last()
        .is_none_or(|message| message.role != Role::Tool)
    {
        return Ok(Vec::new());
    }
    let assistant = messages
        .iter()
        .rposition(|message| message.role == Role::Assistant && !message.tool_calls.is_empty())
        .ok_or_else(|| "tool results have no preceding assistant tool calls".to_string())?;
    let tail = &messages[assistant + 1..];
    if tail.is_empty() || tail.iter().any(|message| message.role != Role::Tool) {
        return Err(
            "tool-result messages must directly follow the assistant tool-call turn".to_string(),
        );
    }
    tail.iter()
        .map(|message| {
            Ok((
                message.name.clone(),
                message
                    .tool_call_id
                    .clone()
                    .ok_or_else(|| "tool result missing tool_call_id".to_string())?,
                message.content.clone(),
            ))
        })
        .collect()
}

/// Build convo hashes from turns + last_user (user turns only) with system and name identity.
/// Prepends tagged system hash; each user hash includes name.
pub fn build_convo(turns: &[(Role, String)], last_user: &str) -> Vec<u64> {
    build_convo_with_system(None, turns, last_user, None)
}

pub fn build_convo_with_system(
    system: Option<&str>,
    turns: &[(Role, String)],
    last_user: &str,
    last_user_name: Option<&str>,
) -> Vec<u64> {
    let mut convo: Vec<u64> =
        Vec::with_capacity(turns.iter().filter(|(r, _)| *r == Role::User).count() + 2);
    convo.push(system_hash(system));
    for (r, t) in turns {
        if *r == Role::User {
            convo.push(turn_hash_named(None, t));
        }
    }
    convo.push(turn_hash_named(last_user_name, last_user));
    convo
}

/// Build convo directly from Message slice (includes names and system).
pub fn build_convo_from_messages(messages: &[Message]) -> Vec<u64> {
    let system = messages
        .iter()
        .find(|m| m.role == Role::System)
        .map(|m| m.content.as_str());
    let mut convo = Vec::new();
    convo.push(system_hash(system));
    for m in messages {
        if m.role == Role::User {
            convo.push(turn_hash_named(m.name.as_deref(), &m.content));
        }
    }
    convo
}

/// Encoded base64 cap matching the sequential daemon (~40 MiB → ~30 MiB raw).
const MAX_BASE64_ENCODED_LEN: usize = 40 * 1024 * 1024;
/// Hard cap on images per request. Each image is a tower pass; more than a
/// handful is outside the serving contract and multiplies VRAM for the
/// per-image patch tensors.
const MAX_IMAGES_PER_REQUEST: usize = 8;

#[derive(Debug)]
enum SlotImage {
    Path(String),
    Base64(String),
}

fn validate_slot_image_payload(which: &str, b64: &str) -> Result<SlotImage, String> {
    if b64.len() > MAX_BASE64_ENCODED_LEN {
        return Err(format!(
            "{which} payload exceeds maximum encoded size ({} bytes)",
            MAX_BASE64_ENCODED_LEN
        ));
    }
    if b64.is_empty() {
        return Err(format!("{which} is empty"));
    }
    Ok(SlotImage::Base64(b64.to_string()))
}

fn extract_slot_images(msg: &serde_json::Value) -> Result<Vec<SlotImage>, String> {
    // OpenAI-style multi-image array (the CLI forwards one entry per
    // image_url content part, in request order).
    if let Some(items) = msg.get("images_base64") {
        let items = items
            .as_array()
            .ok_or_else(|| "images_base64 must be an array of base64 strings".to_string())?;
        if items.len() > MAX_IMAGES_PER_REQUEST {
            return Err(format!(
                "at most {MAX_IMAGES_PER_REQUEST} images are supported per request"
            ));
        }
        let mut out = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let b64 = item
                .as_str()
                .ok_or_else(|| format!("images_base64[{i}] must be a base64 string"))?;
            out.push(validate_slot_image_payload("image", b64)?);
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }
    let image = msg.get("image").and_then(|v| v.as_str());
    let image_base64 = msg.get("image_base64").and_then(|v| v.as_str());
    let image_url = msg.get("image_url").and_then(|v| v.as_str());
    if image_base64.is_some() && image.is_some() {
        eprintln!("[daemon/vl] both image and image_base64 provided — using image_base64");
    }
    if let Some(b64) = image_base64 {
        return Ok(vec![validate_slot_image_payload("image", b64)?]);
    }
    if let Some(path) = image {
        return Ok(vec![SlotImage::Path(path.to_string())]);
    }
    if let Some(url) = image_url {
        // Only data-URLs are meaningful here (the CLI validates and forwards
        // OpenAI-style image_url parts as image_base64). Apply the same size
        // cap as image_base64 — this is a daemon-wire field, so without the
        // cap a direct client could force an unbounded base64 decode — and
        // give remote URLs an actionable message instead of a confusing
        // base64-decode failure.
        if url.len() > MAX_BASE64_ENCODED_LEN {
            return Err(format!(
                "image payload exceeds maximum encoded size ({} bytes)",
                MAX_BASE64_ENCODED_LEN
            ));
        }
        let trimmed = url.trim_start();
        if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
            return Err(
                "remote image URLs are not supported — download the image and \
                 send it inline (data URL or base64)"
                    .to_string(),
            );
        }
        return Ok(vec![validate_slot_image_payload("image", url)?]);
    }
    Ok(Vec::new())
}

fn splice_vl_user_body(
    vision_start_id: u32,
    image_pad_id: u32,
    vision_end_id: u32,
    images: &[(usize, usize, usize)],
    nl: &[u32],
    q_tokens: &[u32],
) -> Vec<u32> {
    let total_visual: usize = images.iter().map(|&(_, _, n)| n).sum();
    let mut user_body =
        Vec::with_capacity(total_visual + images.len() * (nl.len() + 2) + q_tokens.len());
    for &(_, _, n_visual_tokens) in images {
        user_body.push(vision_start_id);
        user_body.extend(std::iter::repeat(image_pad_id).take(n_visual_tokens));
        user_body.push(vision_end_id);
        user_body.extend_from_slice(nl);
    }
    user_body.extend_from_slice(q_tokens);
    user_body
}

/// Build the 3D M-RoPE table for one or more image spans. `images` carries
/// `(grid_h, grid_w, n_visual_tokens)` in prompt order; the prompt's
/// `<|image_pad|>` runs must match that order and shape exactly.
fn build_slot_mrope(
    prompt_ids: &[u32],
    image_pad_id: u32,
    images: &[(usize, usize, usize)],
    spatial_merge_size: usize,
) -> Result<(Vec<[i32; 3]>, i32), String> {
    if images.is_empty() || spatial_merge_size == 0 {
        return Err("VL request has no visual tokens".to_string());
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < prompt_ids.len() {
        if prompt_ids[i] != image_pad_id {
            i += 1;
            continue;
        }
        let start = i;
        while i < prompt_ids.len() && prompt_ids[i] == image_pad_id {
            i += 1;
        }
        runs.push((start, i - start));
    }
    if runs.len() != images.len() {
        return Err(format!(
            "{} image-pad run(s) for {} image(s)",
            runs.len(),
            images.len()
        ));
    }
    let mut spans = Vec::with_capacity(images.len());
    for ((start, len), &(grid_h, grid_w, n_visual)) in runs.iter().zip(images) {
        if *len != n_visual {
            return Err(format!("image-pad run {len} != visual tokens {n_visual}"));
        }
        let merged = (grid_h / spatial_merge_size) * (grid_w / spatial_merge_size);
        if merged != n_visual {
            return Err(format!(
                "merged grid {merged} != spliced visual tokens {n_visual}"
            ));
        }
        spans.push(hipfire_arch_qwen35_vl::mrope::ImageSpan {
            start: *start,
            len: *len,
            grid_h,
            grid_w,
        });
    }
    let built = hipfire_arch_qwen35_vl::mrope::build_mrope_positions(
        prompt_ids.len(),
        &spans,
        spatial_merge_size,
    );
    if built.positions.len() != prompt_ids.len() {
        return Err(format!(
            "build_mrope_positions returned {} positions for {} tokens",
            built.positions.len(),
            prompt_ids.len()
        ));
    }
    Ok((built.positions, built.rope_delta))
}

fn decode_image_base64(b64: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    let raw_b64 = if let Some(rest) = b64.strip_prefix("data:") {
        match rest.split_once(',') {
            Some((_, after)) => after,
            None => return Err("malformed data URL: missing ',' separator".to_string()),
        }
    } else {
        b64
    };
    Engine::decode(&base64::engine::general_purpose::STANDARD, raw_b64)
        .map_err(|e| format!("failed to decode base64 image data: {e}"))
}

fn build_slot_vl_prompt(
    backend: &SlotBackend,
    images: &[SlotImage],
    system: Option<&str>,
    prompt: &str,
    assistant_prefix: AssistantPrefix,
) -> Result<(Vec<u32>, VisualData), String> {
    if images.is_empty() {
        return Err("VL request has no images".to_string());
    }
    let vc = backend
        .vision_config
        .as_ref()
        .ok_or_else(|| "VL model missing vision_config".to_string())?;
    let tokenizer = &backend.tokenizer;
    let image_pad_id = tokenizer
        .special_token_id("<|image_pad|>")
        .ok_or_else(|| "VL tokenizer missing <|image_pad|>".to_string())?;
    let vision_start_id = tokenizer
        .special_token_id("<|vision_start|>")
        .ok_or_else(|| "VL tokenizer missing <|vision_start|>".to_string())?;
    let vision_end_id = tokenizer
        .special_token_id("<|vision_end|>")
        .ok_or_else(|| "VL tokenizer missing <|vision_end|>".to_string())?;

    let mut prepared = Vec::with_capacity(images.len());
    for image in images {
        let (pixels, img_h, img_w) = match image {
            SlotImage::Path(path) => hipfire_arch_qwen35_vl::image::load_and_preprocess(
                std::path::Path::new(path),
                vc.patch_size,
                vc.spatial_merge_size,
            )?,
            SlotImage::Base64(b64) => {
                let bytes = decode_image_base64(b64)?;
                hipfire_arch_qwen35_vl::image::load_and_preprocess_from_bytes(
                    &bytes,
                    vc.patch_size,
                    vc.spatial_merge_size,
                )?
            }
        };
        let grid_h = img_h / vc.patch_size;
        let grid_w = img_w / vc.patch_size;
        let n_patches = grid_h * grid_w;
        let n_visual_tokens = n_patches / (vc.spatial_merge_size * vc.spatial_merge_size);
        if n_visual_tokens == 0 {
            return Err("image produced no visual tokens".to_string());
        }
        let patches = hipfire_arch_qwen35_vl::image::extract_patches(
            &pixels,
            3,
            img_h,
            img_w,
            vc.patch_size,
            vc.temporal_patch_size,
            vc.spatial_merge_size,
        );
        prepared.push((patches, grid_h, grid_w, n_visual_tokens));
    }
    let shapes: Vec<(usize, usize, usize)> = prepared
        .iter()
        .map(|&(_, grid_h, grid_w, n)| (grid_h, grid_w, n))
        .collect();
    let total_visual: usize = shapes.iter().map(|&(_, _, n)| n).sum();

    let nl = tokenizer.encode("\n");
    let q_tokens = tokenizer.encode(prompt);
    let user_body = splice_vl_user_body(
        vision_start_id,
        image_pad_id,
        vision_end_id,
        &shapes,
        &nl,
        &q_tokens,
    );
    let prompt_tokens = ChatFrame {
        tokenizer,
        system,
        user: "",
        assistant_prefix,
        raw: false,
    }
    .build_with_user_tokens(&user_body);

    let (mrope_positions, rope_delta) =
        build_slot_mrope(&prompt_tokens, image_pad_id, &shapes, vc.spatial_merge_size)?;
    let images = prepared
        .into_iter()
        .map(
            |(patches, grid_h, grid_w, n_visual_tokens)| hipfire_runtime::serve::VisualImage {
                patches,
                grid_h,
                grid_w,
                n_visual_tokens,
            },
        )
        .collect::<Vec<_>>();
    // Preserve the historical single-image payload exactly; the engine's
    // single-image fast path reads `patches`/`grid_*`.
    if images.len() == 1 {
        let image = images.into_iter().next().expect("one image");
        Ok((
            prompt_tokens,
            VisualData {
                patches: image.patches,
                grid_h: image.grid_h,
                grid_w: image.grid_w,
                n_visual_tokens: total_visual,
                images: Vec::new(),
                mrope_positions,
                rope_delta,
            },
        ))
    } else {
        Ok((
            prompt_tokens,
            VisualData {
                patches: Vec::new(),
                grid_h: 0,
                grid_w: 0,
                n_visual_tokens: total_visual,
                images,
                mrope_positions,
                rope_delta,
            },
        ))
    }
}

// ── Tests (pure, no GPU) ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_engine::terminal::{batch_clear_all_terminals, batch_terminal_generation};
    use hipfire_runtime::prompt_frame::Role;
    use serde_json::json;

    /// The waiting-room fallbacks used when the process config lacks a key
    /// must equal the registered defaults (a hard-coded 30000 ms once
    /// drifted from the registered 600000 ms).
    #[test]
    fn waiting_room_fallbacks_are_the_registered_defaults() {
        let empty = hipfire_config::ConfigLayer::default();
        for (key, expected) in [
            ("serve.max_queue", 64),
            ("serve.max_queue_bytes", 268_435_456),
            ("serve.queue_timeout_ms", 600_000),
        ] {
            let registered = hipfire_config::field(key).expect("registered key").default;
            assert!(
                matches!(registered, hipfire_config::DefaultValue::Integer(i) if i == expected),
                "{key}: registered default is {registered:?}, expected {expected}"
            );
            assert_eq!(
                integer_or_registered_default(&empty, key),
                expected,
                "{key}"
            );
        }
        let mut set = hipfire_config::ConfigLayer::default();
        set.set(
            "serve.queue_timeout_ms",
            hipfire_config::ConfigValue::Integer(1234),
        )
        .unwrap();
        assert_eq!(
            integer_or_registered_default(&set, "serve.queue_timeout_ms"),
            1234
        );
    }

    #[test]
    fn arch_validation_accepts_5_and_6() {
        assert!(validate_arch_id(5).is_ok());
        assert!(validate_arch_id(6).is_ok());
        assert!(validate_arch_id(7).is_err());
        assert!(validate_arch_id(9).is_err());
    }
    #[test]
    fn slot_mtp_parse_preserves_tri_state_and_k_default() {
        // The tri-state must survive beside mtp_k: auto and an explicit on
        // are indistinguishable downstream when only mtp_k travels, and a
        // check alone cannot restore the distinction.
        let parse = |params: serde_json::Value| {
            SlotLoadParams::from_load_msg(&serde_json::json!({ "params": params }))
        };
        // Absent and off: Off with k forced to 0 (verbatim rule).
        for params in [
            serde_json::json!({}),
            serde_json::json!({ "mtp_mode": "off", "mtp_k": 7 }),
            serde_json::json!({ "mtp_mode": "", "mtp_k": 7 }),
        ] {
            let p = parse(params);
            assert_eq!(p.mtp_mode, hipfire_config::MtpMode::Off);
            assert_eq!(p.mtp_k, 0);
        }
        // auto and on keep the existing k default-4 rule verbatim.
        let auto = parse(serde_json::json!({ "mtp_mode": "auto" }));
        assert_eq!(auto.mtp_mode, hipfire_config::MtpMode::Auto);
        assert_eq!(auto.mtp_k, 4);
        let on = parse(serde_json::json!({ "mtp_mode": "on" }));
        assert_eq!(on.mtp_mode, hipfire_config::MtpMode::On);
        assert_eq!(on.mtp_k, 4);
        // Explicit k still wins on both armed modes.
        let auto_k = parse(serde_json::json!({ "mtp_mode": "auto", "mtp_k": 2 }));
        assert_eq!(auto_k.mtp_mode, hipfire_config::MtpMode::Auto);
        assert_eq!(auto_k.mtp_k, 2);
        let on_k = parse(serde_json::json!({ "mtp_mode": "on", "mtp_k": 2 }));
        assert_eq!(on_k.mtp_mode, hipfire_config::MtpMode::On);
        assert_eq!(on_k.mtp_k, 2);
    }

    /// P10: a generate with no sampling fields resolves on the multi-slot
    /// route to exactly what the sequential route resolves (arch ladder /
    /// `.hfq` recs, default max_tokens 4096, the `max_tokens_fit` clamp) —
    /// not the old slot-local 0.0 / 1.0 / 512 / no-fit.
    #[test]
    fn slot_sampling_defaults_match_sequential_route() {
        use hipfire_engine::scheduler::{resolve_temp_top_p, DEFAULT_GENERATE_MAX_TOKENS};
        let bare = json!({"type": "generate", "max_tokens_fit": true});
        for (rec_t, rec_p) in [(None, None), (Some(0.6), Some(0.95)), (Some(0.7), None)] {
            let (seq_t, seq_p) = resolve_temp_top_p(&bare, 5, rec_t, rec_p);
            let (t, p, max_tokens, fit) = resolve_slot_sampling(&bare, 5, rec_t, rec_p).unwrap();
            assert_eq!((t, p), (seq_t, seq_p));
            assert_eq!(max_tokens, DEFAULT_GENERATE_MAX_TOKENS);
            assert!(fit);
        }
        // Pinned values: qwen35 ladder, then per-knob .hfq recs.
        assert_eq!(resolve_slot_sampling(&bare, 5, None, None).unwrap().0, 0.3);
        assert_eq!(resolve_slot_sampling(&bare, 5, None, None).unwrap().1, 0.8);
        let partial = resolve_slot_sampling(&bare, 5, Some(0.7), None).unwrap();
        assert_eq!((partial.0, partial.1), (0.7, 0.8));
        // Explicit fields win over recs, and an explicit max_tokens is never
        // marked for fitting.
        let explicit = json!({"max_tokens": 64, "temperature": 0.0, "top_p": 1.0});
        assert_eq!(
            resolve_slot_sampling(&explicit, 5, Some(0.6), Some(0.95)).unwrap(),
            (0.0, 1.0, 64, false)
        );
        assert!(resolve_slot_sampling(&json!({"max_tokens": 0}), 5, None, None).is_err());
    }

    #[test]
    fn generate_caps_allows_images() {
        let m = json!({"image": "/tmp/a.png", "experimental_multi_slot": true});
        assert!(validate_generate_caps(&m).is_none());
        let m2 = json!({"image_base64": "abcd", "experimental_multi_slot": true});
        assert!(validate_generate_caps(&m2).is_none());
    }

    #[test]
    fn generate_caps_accepts_tool_turns_and_refuses_tools_with_images() {
        // Tool turns are SUPPORTED on the slot route (upstream ad6004ac0:
        // the daemon projects role=tool into the Jinja message list and the
        // tool-call parser emits `tool_calls`). The gate that used to refuse
        // a bare tool message is gone; the surviving refusal is the
        // images+tools COMBINATION (the VL prompt path splices image pads
        // into a user body and cannot render a tool contract).
        let tool_turn = json!({"messages": [{"role": "tool", "content": "result",
                                             "tool_call_id": "call_1"}],
                               "experimental_multi_slot": true});
        assert!(
            validate_generate_caps(&tool_turn).is_none(),
            "tool-result turns are supported and must not be refused here"
        );
        let assistant_call = json!({"messages": [{"role": "assistant", "tool_calls": [
                                        {"id": "1", "function": {"name": "f"}}]}],
                                    "experimental_multi_slot": true});
        assert!(validate_generate_caps(&assistant_call).is_none());
        let both = json!({"image_base64": "abcd", "tools": [{"type": "function"}],
                          "experimental_multi_slot": true});
        let reason = validate_generate_caps(&both)
            .expect("images+tools must stay refused (no VL tool contract)");
        assert!(reason.contains("images and tools"), "unexpected: {reason}");
    }

    #[test]
    fn extract_slot_image_prefers_base64() {
        let msg = json!({"image": "/tmp/a.png", "image_base64": "abcd"});
        match extract_slot_images(&msg).unwrap().as_slice() {
            [SlotImage::Base64(s)] => assert_eq!(s, "abcd"),
            other => panic!("expected base64, got {other:?}"),
        }
    }

    #[test]
    fn extract_slot_images_keeps_request_order() {
        let msg = json!({"images_base64": ["aaaa", "bbbb"]});
        match extract_slot_images(&msg).unwrap().as_slice() {
            [SlotImage::Base64(a), SlotImage::Base64(b)] => {
                assert_eq!(a, "aaaa");
                assert_eq!(b, "bbbb");
            }
            other => panic!("expected two base64 images, got {other:?}"),
        }
    }

    #[test]
    fn splice_vl_user_body_wraps_pads() {
        let body = splice_vl_user_body(1, 2, 3, &[(4, 4, 4)], &[10], &[20, 21]);
        assert_eq!(body, vec![1, 2, 2, 2, 2, 3, 10, 20, 21]);
    }

    #[test]
    fn build_slot_mrope_marks_image_span() {
        let pad = 9u32;
        let mut ids = vec![1, 2, pad, pad, pad, pad, 3, 4];
        let (pos, delta) = build_slot_mrope(&ids, pad, &[(4, 4, 4)], 2).unwrap();
        assert_eq!(pos.len(), ids.len());
        assert_eq!(pos[0], [0, 0, 0]);
        assert_eq!(pos[2], [2, 2, 2]); // first visual: t=cursor, h=cursor+0, w=cursor+0
        assert_ne!(delta, 0);
        ids.push(pad);
        assert!(build_slot_mrope(&ids, pad, &[(4, 4, 4)], 2).is_err());
    }

    #[test]
    fn build_slot_mrope_rejects_degenerate_and_mismatched_inputs() {
        let pad = 9u32;
        let ids = vec![1, pad, pad, pad, pad, 3];
        // Merged grid ((8/2)*(8/2)=16) disagrees with the spliced token count (4).
        assert!(build_slot_mrope(&ids, pad, &[(8, 8, 4)], 2).is_err());
        // Zero visual tokens.
        assert!(build_slot_mrope(&ids, pad, &[(4, 4, 0)], 2).is_err());
        // Degenerate spatial merge size.
        assert!(build_slot_mrope(&ids, pad, &[(4, 4, 4)], 0).is_err());
        // Two pad runs are two images; their spans are legal in request order.
        let split = vec![pad, pad, 5, pad, pad];
        let (pos, _) = build_slot_mrope(&split, pad, &[(4, 2, 2), (4, 2, 2)], 2).unwrap();
        assert_eq!(pos.len(), split.len());
        assert_eq!(pos[0], [0, 0, 0]);
        assert_eq!(pos[3], [3, 3, 3]);
        // A prompt with no pad at all.
        let no_pad = vec![1, 2, 3];
        assert!(build_slot_mrope(&no_pad, pad, &[(4, 4, 4)], 2).is_err());
    }

    #[test]
    fn extract_slot_image_wraps_top_level_image_url_as_base64() {
        let msg = json!({"image_url": "data:image/png;base64,aGVsbG8="});
        match extract_slot_images(&msg).unwrap().as_slice() {
            [SlotImage::Base64(s)] => assert_eq!(s, "data:image/png;base64,aGVsbG8="),
            other => panic!("expected base64, got {other:?}"),
        }
        let none = json!({"prompt": "hi"});
        assert!(extract_slot_images(&none).unwrap().is_empty());
    }

    #[test]
    fn extract_slot_image_rejects_oversized_base64() {
        let big = "A".repeat(MAX_BASE64_ENCODED_LEN + 1);
        let msg = json!({ "image_base64": big });
        let err = extract_slot_images(&msg).unwrap_err();
        assert!(err.contains("maximum encoded size"), "unexpected: {err}");
    }

    /// The effort dial reaches the template only when the client actually
    /// asked for reasoning. Keyed on the request, not on the route's
    /// `<think>`-token framing: a template that opens reasoning with another
    /// token keeps that framing false and must still get the rung.
    #[test]
    fn slot_effort_reaches_the_template_except_when_reasoning_is_off() {
        // A rung is passed through verbatim, including ones the template may
        // reject (that failure belongs to the render, as validation).
        assert_eq!(slot_jinja_effort(None, Some("high"), None), Some("high"));
        assert_eq!(slot_jinja_effort(None, Some("xhigh"), None), Some("xhigh"));
        assert_eq!(
            slot_jinja_effort(Some(true), Some("low"), None),
            Some("low")
        );
        // `auto` and absent mean "no preference": leave it undefined so the
        // template's own default applies.
        assert_eq!(slot_jinja_effort(None, Some("auto"), None), None);
        assert_eq!(slot_jinja_effort(None, None, None), None);
        // Reasoning off — explicitly, by a disable rung, or by the
        // immediate-close sentinel — suppresses the dial entirely.
        assert_eq!(slot_jinja_effort(Some(false), Some("high"), None), None);
        assert_eq!(slot_jinja_effort(None, Some("none"), None), None);
        assert_eq!(slot_jinja_effort(None, Some("off"), None), None);
        assert_eq!(slot_jinja_effort(None, Some("chat"), None), None);
        assert_eq!(slot_jinja_effort(Some(true), Some("high"), Some(1)), None);
        // A finite budget is not a disable: the rung survives it.
        assert_eq!(
            slot_jinja_effort(Some(true), Some("high"), Some(64)),
            Some("high")
        );
    }

    #[test]
    fn decode_image_base64_strips_data_urls_and_rejects_garbage() {
        assert_eq!(
            decode_image_base64("data:image/png;base64,aGVsbG8=").unwrap(),
            b"hello"
        );
        assert_eq!(decode_image_base64("aGVsbG8=").unwrap(), b"hello");
        // A data URL without the ',' separator fails closed.
        assert!(decode_image_base64("data:image/png;base64").is_err());
        assert!(decode_image_base64("not base64 !!!").is_err());
    }

    #[test]
    fn generate_caps_accepts_tools_valid_stop_and_logprobs() {
        let m = json!({"tools": [{"type":"function"}], "experimental_multi_slot": true});
        assert!(validate_generate_caps(&m).is_none());
        for rf in [
            json!({"type": "json_object"}),
            json!({"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}),
        ] {
            let m = json!({"response_format": rf, "experimental_multi_slot": true});
            assert!(
                validate_generate_caps(&m).is_none(),
                "response_format {rf} must be accepted"
            );
        }
        for stop in [json!(["END"]), json!("END"), json!([]), json!("")] {
            let m = json!({"stop": stop, "experimental_multi_slot": true});
            assert!(
                validate_generate_caps(&m).is_none(),
                "stop={stop:?} must be accepted (empty is neutral, non-empty is enforced)"
            );
        }
        for bad in [
            json!([1]),
            json!("x".repeat(65)),
            json!(["a", "b", "c", "d", "e"]),
        ] {
            let m = json!({"stop": bad, "experimental_multi_slot": true});
            assert!(
                validate_generate_caps(&m).is_some(),
                "stop={bad:?} must be refused by the shared parser"
            );
        }
        for logprobs in [
            json!({"logprobs": true, "experimental_multi_slot": true}),
            json!({"logprobs": true, "top_logprobs": 5, "experimental_multi_slot": true}),
            json!({"logprobs": false, "experimental_multi_slot": true}),
        ] {
            assert!(
                validate_generate_caps(&logprobs).is_none(),
                "valid logprobs must be accepted: {logprobs}"
            );
        }
        for bad in [
            json!({"logprobs": "yes", "experimental_multi_slot": true}),
            json!({"top_logprobs": 5, "experimental_multi_slot": true}),
            json!({"logprobs": true, "top_logprobs": 21, "experimental_multi_slot": true}),
        ] {
            assert!(
                validate_generate_caps(&bad).is_some(),
                "malformed logprobs must be refused: {bad}"
            );
        }
        // Finite think caps are ENFORCED (the grammar cursor force-closes the
        // span at the budget) and are therefore never refused at the door —
        // including the `1` no-thinking sentinel, whose contradiction with an
        // open-think framing is raised later, in `handle_generate`'s projected
        // reasoning authority, not by this wire-cap gate.
        for cap in [json!(1), json!(5)] {
            let m = json!({"max_think_tokens": cap, "experimental_multi_slot": true});
            assert!(
                validate_generate_caps(&m).is_none(),
                "max_think_tokens={cap} is enforced, not refused"
            );
        }
    }
    #[test]
    fn generate_caps_rejects_every_non_unit_spelling_of_n_and_echo() {
        // `n` is refused unless it is exactly the number 1: an `as_u64` gate
        // silently accepted `2.5`/`-1`/"2" as a single completion (the
        // semantic downgrade the refusal exists to prevent).
        for bad in [json!(2), json!(2.5), json!(-1), json!(0), json!("2")] {
            let m = json!({"n": bad, "experimental_multi_slot": true});
            assert!(
                validate_generate_caps(&m).is_some(),
                "n={bad} must be refused, not silently downgraded to one completion"
            );
        }
        assert!(
            validate_generate_caps(&json!({"n": 1, "experimental_multi_slot": true})).is_none()
        );
        assert!(
            validate_generate_caps(&json!({"n": null, "experimental_multi_slot": true})).is_none()
        );
        // `echo: false` is the OpenAI default this route honours by not
        // echoing; `true` and non-boolean spellings are refused.
        assert!(
            validate_generate_caps(&json!({"echo": false, "experimental_multi_slot": true}))
                .is_none()
        );
        for bad in [json!(true), json!(1)] {
            let m = json!({"echo": bad, "experimental_multi_slot": true});
            assert!(
                validate_generate_caps(&m).is_some(),
                "echo={bad} must be refused"
            );
        }
    }

    #[test]
    fn generate_caps_accepts_valid_json_schema_rejects_unsupported() {
        // A valid subset schema (object with typed properties) is accepted at
        // validate_generate_caps — the saddle-core CompiledSchema compiler
        // succeeds (spec §7 G1).
        let m = json!({
            "response_format": {"type": "json_schema", "json_schema": {"name": "test", "schema": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}},
            "experimental_multi_slot": true
        });
        assert!(
            validate_generate_caps(&m).is_none(),
            "valid json_schema response_format should be accepted"
        );
        // `response_format: {"type":"text"}` is OpenAI's DEFAULT type and
        // means "unconstrained". The gateway maps it to "no response_format"
        // before the wire (verified: E15 accepts it with HTTP 200), so a
        // `text` type arriving at the DAEMON door is a second, ambiguous
        // meaning and is refused rather than silently reinterpreted.
        let m2 = json!({
            "response_format": {"type": "text"},
            "experimental_multi_slot": true
        });
        let reason = validate_generate_caps(&m2)
            .expect("a text response_format must be refused at the daemon door");
        assert!(reason.contains("is not supported"), "unexpected: {reason}");
        // Unsupported $ref is rejected before submit (spec §7 G1, §5.4 S4).
        let m3 = json!({
            "response_format": {"type": "json_schema", "json_schema": {"name": "test", "schema": {"$ref": "#/$defs/foo"}}},
            "experimental_multi_slot": true
        });
        let reason = validate_generate_caps(&m3);
        assert!(reason.is_some(), "unsupported $ref must be rejected");
        assert!(
            reason.unwrap().contains("response_format json_schema"),
            "rejection must be typed as a json_schema error"
        );
        // Missing schema object is rejected.
        let m4 = json!({
            "response_format": {"type": "json_schema", "json_schema": {"name": "test"}},
            "experimental_multi_slot": true
        });
        assert!(
            validate_generate_caps(&m4).is_some(),
            "json_schema without a schema object must be rejected"
        );
    }

    #[test]
    fn load_caps_rejects_continuous_and_tp_pp() {
        let m = json!({"params": {"continuous_batch_size": 2}});
        assert!(validate_load_caps(&m, "gfx1100").is_some());
        let m2 = json!({"params": {"tp": 2}});
        assert!(validate_load_caps(&m2, "gfx1100").is_some());
        let m3 = json!({"params": {"pp": 2}});
        assert!(validate_load_caps(&m3, "gfx1100").is_some());
        let m4 = json!({"params": {"drafter": "some.hfq"}});
        assert!(validate_load_caps(&m4, "gfx1100").is_some());
        let m5 = json!({"params": {"prefill_compression": "on"}});
        assert!(validate_load_caps(&m5, "gfx1100").is_some());
    }

    #[test]
    fn load_caps_admit_only_the_default_dflash_block() {
        for ok in [
            json!({"params": {}}),
            json!({"params": {"dflash_block": 16}}),
            json!({"params": {"dflash_block": "auto"}}),
        ] {
            assert!(
                validate_load_caps(&ok, "gfx1100").is_none(),
                "{ok} must be admitted"
            );
        }
        for bad in [
            json!({"params": {"dflash_block": 8}}),
            json!({"params": {"dflash_block": 99}}),
        ] {
            assert!(
                validate_load_caps(&bad, "gfx1100").is_some(),
                "{bad} must be refused"
            );
        }
    }

    #[test]
    fn turn_hash_is_fnv1a() {
        // Known FNV-1a for "a" is 0xaf63dc4c8601ec8c etc. Just test determinism.
        assert_ne!(turn_hash("hello"), turn_hash("hello "));
        assert_eq!(turn_hash("test"), turn_hash("test"));
    }

    #[test]
    fn turn_hash_name_changes_identity() {
        assert_ne!(
            turn_hash_named(Some("alice"), "hello"),
            turn_hash_named(None, "hello")
        );
        assert_ne!(
            turn_hash_named(Some("alice"), "hello"),
            turn_hash_named(Some("bob"), "hello")
        );
    }

    #[test]
    fn system_hash_differs() {
        assert_ne!(system_hash(Some("sys1")), system_hash(Some("sys2")));
        assert_ne!(system_hash(None), system_hash(Some("sys")));
    }

    #[test]
    fn project_messages_builds_convo() {
        let msg = json!({
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "follow up"}
            ]
        });
        let (system, turns, last) = project_messages(&msg).unwrap();
        assert_eq!(system.as_deref(), Some("you are helpful"));
        assert_eq!(turns.len(), 2); // hi -> assistant hello, then follow up removed as last
        assert_eq!(turns[0].0, Role::User);
        assert_eq!(turns[0].1, "hi");
        assert_eq!(turns[1].0, Role::Assistant);
        assert_eq!(last, "follow up");
        let convo = build_convo_with_system(system.as_deref(), &turns, &last, None);
        // system hash + 2 user hashes
        assert_eq!(convo.len(), 3);
        assert_eq!(convo[0], system_hash(Some("you are helpful")));
        assert_eq!(convo[1], turn_hash("hi"));
        assert_eq!(convo[2], turn_hash("follow up"));
    }

    #[test]
    fn project_messages_requires_user() {
        let msg = json!({"messages": [{"role": "assistant", "content": "hi"}]});
        assert!(project_messages(&msg).is_err());
    }

    #[test]
    fn project_messages_fallback_prompt() {
        let msg = json!({"prompt": "Hello", "system": "sys"});
        let (system, turns, last) = project_messages(&msg).unwrap();
        assert_eq!(system.as_deref(), Some("sys"));
        assert!(turns.is_empty());
        assert_eq!(last, "Hello");
        let convo = build_convo_with_system(system.as_deref(), &turns, &last, None);
        assert_eq!(convo[0], system_hash(Some("sys")));
    }
    #[test]
    fn jinja_projection_preserves_supported_message_roles() {
        let messages = project_jinja_messages(&json!({
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer", "reasoning_content": "reason",
                 "tool_calls": [{"id":"call_1","name":"echo","arguments":{"text":"x"}}]},
                {"role":"tool","content":"x","name":"echo","tool_call_id":"call_1"}
            ]
        }))
        .unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[2].reasoning_content.as_deref(), Some("reason"));
        assert_eq!(messages[2].tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(messages[3].role, Role::Tool);
        assert_eq!(messages[3].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn pending_tool_broker_orders_parallel_results_and_consumes_once() {
        let base = project_jinja_messages(&json!({
            "messages": [{"role":"user","content":"use both"}]
        }))
        .unwrap();
        let convo = build_convo_from_messages(&base);
        let calls = vec![
            ToolCall {
                id: Some("call_a".into()),
                name: "first".into(),
                arguments: json!({}),
                rendered_body: None,
            },
            ToolCall {
                id: Some("call_b".into()),
                name: "second".into(),
                arguments: json!({}),
                rendered_body: None,
            },
        ];
        let mut broker = PendingToolBroker::default();
        broker.register(17, convo.clone(), &calls);
        let messages = project_jinja_messages(&json!({
            "messages": [
                {"role":"user","content":"use both"},
                {"role":"assistant","content":"","tool_calls":[
                    {"id":"call_a","name":"first","arguments":{}},
                    {"id":"call_b","name":"second","arguments":{}}
                ]},
                {"role":"tool","name":"second","tool_call_id":"call_b","content":"B"},
                {"role":"tool","name":"first","tool_call_id":"call_a","content":"A"}
            ]
        }))
        .unwrap();
        let claim = broker.claim(&messages, &convo).unwrap().expect("claim");
        assert_eq!(claim.session, 17);
        assert_eq!(claim.results, ["A", "B"]);
        assert!(
            broker.claim(&messages, &convo).is_err(),
            "reentry ids are one-shot"
        );
    }

    #[test]
    fn pending_tool_broker_rejects_cross_session_and_name_mismatch() {
        let base = project_jinja_messages(&json!({
            "messages": [{"role":"user","content":"use tool"}]
        }))
        .unwrap();
        let convo = build_convo_from_messages(&base);
        let mut broker = PendingToolBroker::default();
        broker.register(
            1,
            convo.clone(),
            &[ToolCall {
                id: Some("call_a".into()),
                name: "first".into(),
                arguments: json!({}),
                rendered_body: None,
            }],
        );
        broker.register(
            2,
            convo.clone(),
            &[ToolCall {
                id: Some("call_b".into()),
                name: "second".into(),
                arguments: json!({}),
                rendered_body: None,
            }],
        );
        let mixed = project_jinja_messages(&json!({"messages":[
            {"role":"user","content":"use tool"},
            {"role":"assistant","content":"","tool_calls":[
                {"id":"call_a","name":"first","arguments":{}},
                {"id":"call_b","name":"second","arguments":{}}
            ]},
            {"role":"tool","tool_call_id":"call_a","content":"A"},
            {"role":"tool","tool_call_id":"call_b","content":"B"}
        ]}))
        .unwrap();
        assert!(broker
            .claim(&mixed, &convo)
            .unwrap_err()
            .contains("multiple"));

        let wrong_name = project_jinja_messages(&json!({"messages":[
            {"role":"user","content":"use tool"},
            {"role":"assistant","content":"","tool_calls":[{"id":"call_a","name":"first","arguments":{}}]},
            {"role":"tool","name":"wrong","tool_call_id":"call_a","content":"A"}
        ]})).unwrap();
        assert!(broker
            .claim(&wrong_name, &convo)
            .unwrap_err()
            .contains("name"));
        assert!(
            broker.by_call.contains_key("call_a"),
            "failed claims must not consume state"
        );
    }

    #[test]
    fn experimental_load_requires_its_fixed_q8_non_speculative_contract() {
        let supported = json!({"params": {
            "continuous_batch_size": 1,
            "tp": 1,
            "pp": 1,
            "kv_mode": "q8",
            "kv_backend": "legacy",
            "kv_adaptive": "off",
            "dflash_mode": "off",
            "mtp_mode": "off",
            "prefill_compression": "off",
            "ngram_draft": false,
            "cask": false
        }});
        assert_eq!(validate_load_caps(&supported, "gfx1100"), None);
        // The full static KV ladder plus the flat 2-byte native tiers is
        // accepted (engine resolves it through the slots site policy); only
        // strings the policy rejects — garbage — and fp8 off gfx1201 are
        // refused here, loudly, before any GPU work. The gate is exact per
        // arch string, not per host card, so these run on any CI machine.
        for kv in [
            "asym3", "asym2", "asym4", "fwht2", "fwht3", "fwht4", "bf16", "f16", "auto",
        ] {
            assert_eq!(
                validate_load_caps(&json!({"params": {"kv_mode": kv}}), "gfx1100"),
                None,
                "ladder tier {kv} must be accepted"
            );
        }
        // fp8: gfx1201 admits, every other arch refuses with a named error.
        assert_eq!(
            validate_load_caps(&json!({"params": {"kv_mode": "fp8"}}), "gfx1201"),
            None,
            "fp8 must be admitted on gfx1201"
        );
        for arch in ["gfx1100", "gfx1101", "gfx1151", "gfx1200", "gfx942"] {
            let refused = validate_load_caps(&json!({"params": {"kv_mode": "fp8"}}), arch)
                .expect("fp8 off gfx1201 must be refused");
            assert!(refused.contains("fp8"), "{refused}");
            assert!(refused.contains(arch), "{refused}");
            assert!(refused.contains("gfx1201"), "{refused}");
        }
        for params in [
            json!({"kv_mode": "garbage"}),
            json!({"ngram_draft": true}),
            json!({"cask": true}),
            json!({"drafter": "some-drafter"}),
            json!({"prefill_compression": "on"}),
        ] {
            assert!(
                validate_load_caps(&json!({"params": params}), "gfx1201").is_some(),
                "unsupported params passed: {params}"
            );
        }
        // The VMM KV route is admitted for q8; a rotated tier is refused by
        // the route's own descriptor-kernel gate.
        assert_eq!(
            validate_load_caps(
                &json!({"params": {"kv_backend": "vmm", "kv_mode": "q8"}}),
                "gfx1100"
            ),
            None,
            "vmm + q8 must be admitted"
        );
        assert!(
            validate_load_caps(
                &json!({"params": {"kv_backend": "vmm", "kv_mode": "fwht3"}}),
                "gfx1100"
            )
            .is_some(),
            "vmm + rotated KV must be refused"
        );
        assert!(
            validate_load_caps(
                &json!({"params": {"kv_backend": "vmm", "kv_mode": "fp8"}}),
                "gfx1201"
            )
            .is_some(),
            "the shared VMM page pool has no fp8 kernels"
        );
        // MTP is now accepted in multi-slot (sequential per-slot path).
        assert_eq!(
            validate_load_caps(&json!({"params": {"mtp_mode": "on"}}), "gfx1100"),
            None,
            "mtp_mode on should be accepted in multi-slot"
        );
        // DFlash2 is accepted: params.draft + dflash_mode auto/on route to
        // the slot engine's spec-decode path.
        for params in [
            json!({"dflash_mode": "auto"}),
            json!({"dflash_mode": "on"}),
            json!({"dflash_mode": "off"}),
            json!({"draft": "/path/to/draft.hfq"}),
            json!({"draft": "/path/to/draft.hfq", "dflash_mode": "on"}),
        ] {
            assert_eq!(
                validate_load_caps(&json!({"params": params}), "gfx1100"),
                None,
                "dflash params must be accepted: {params}"
            );
        }
    }

    #[test]
    fn generate_caps_rejects_silently_ignored_controls() {
        assert!(validate_generate_caps(&json!({"temperature": 0.7, "top_p": 0.9})).is_none());
        // Penalties are honored by the slot sampler now: in-range values pass,
        // out-of-range values are rejected.
        for request in [
            json!({"repeat_penalty": 1.05}),
            json!({"presence_penalty": 1.5}),
            json!({"min_p": 0.05}),
            // The effort dial is honoured on this route: it reaches the Jinja
            // template, so it is accepted at the door (a rung the template
            // rejects fails the render instead).
            json!({"reasoning_effort": "high"}),
            json!({"reasoning_effort": "xhigh"}),
        ] {
            assert!(
                validate_generate_caps(&request).is_none(),
                "supported request refused: {request}"
            );
        }
        for request in [
            json!({"repeat_penalty": 2.5}),
            json!({"presence_penalty": -0.1}),
            json!({"min_p": 1.5}),
        ] {
            assert!(
                validate_generate_caps(&request).is_some(),
                "unsupported request passed: {request}"
            );
        }
        assert!(validate_generate_caps(&json!({
            "tools": [{"type":"function","function":{"name":"echo"}}],
            "messages": [
                {"role":"user","content":"echo hi"},
                {"role":"assistant","content":"","tool_calls":[{
                    "id":"call_0","name":"echo","arguments":{"text":"hi"}
                }]},
                {"role":"tool","tool_call_id":"call_0","content":"hi"}
            ]
        }))
        .is_none());
    }

    #[test]
    fn generate_caps_rejects_malformed_numeric_controls() {
        for request in [
            json!({"temperature": "hot"}),
            json!({"top_p": true}),
            json!({"top_k": 1.5}),
            json!({"repeat_window": -1}),
            json!({"presence_penalty": []}),
            json!({"max_think_tokens": "many"}),
            json!({"params": {"max_think_tokens": -2}}),
        ] {
            assert!(
                validate_generate_caps(&request).is_some(),
                "malformed numeric control silently passed: {request}"
            );
        }
    }

    #[test]
    fn build_convo_from_messages_includes_names_and_system() {
        let msgs = vec![
            Message {
                role: Role::System,
                content: "sys".to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: "hi".to_string(),
                reasoning_content: None,
                name: Some("alice".to_string()),
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ];
        let c1 = build_convo_from_messages(&msgs);
        let c2 = build_convo_from_messages(&[
            Message {
                role: Role::System,
                content: "sys".to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: "hi".to_string(),
                reasoning_content: None,
                name: Some("bob".to_string()),
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ]);
        assert_ne!(c1, c2);
    }
    #[test]
    fn qwen_ar_slot_post_start_errors_release_route_and_reuse_generation() {
        let _lock = crate::TERMINAL_TEST_LOCK.lock().unwrap();
        let cases = [
            (
                "submit",
                "multi_slot submit: engine unavailable",
                "internal",
            ),
            (
                "rejected",
                "multi_slot rejected: engine rejected result",
                "internal",
            ),
            (
                "summary",
                "model emitted tool calls on a tool-disabled multi-slot request",
                "internal",
            ),
        ];

        for (offset, (name, message, class)) in cases.into_iter().enumerate() {
            let id = format!("slot-{name}-reuse");
            let attempt_id = 92_000 + offset as u64;
            batch_clear_all_terminals();
            hipfire_engine::terminal::set_active_attempt_id(attempt_id);
            assert_eq!(hipfire_generate::ar::active_generation_route(), None);

            let admission_a = hipfire_engine::terminal::batch_announce_terminal(&id, attempt_id)
                .expect("slot admission A");
            let mut output = Vec::new();
            {
                let _batch_scope =
                    BatchAttemptScope::enter_for_generation(&id, attempt_id, admission_a);
                let _route_scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, &id);
                emit_generation_start(GenerationRoute::QwenAr, &mut output, &id, false);
                emit_qwen_ar_slot_error(&mut output, &id, message, class, false, false);
            }

            let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
                .expect("slot error UTF-8")
                .lines()
                .map(|line| serde_json::from_str(line).expect("slot error JSON"))
                .collect();
            assert_eq!(events.len(), 2, "{name} terminal envelope count");
            assert_eq!(events[0]["type"], "gen_start", "{name} start event");
            assert_eq!(events[0]["id"], id, "{name} start id");
            assert_eq!(events[0]["attempt_id"], attempt_id, "{name} start attempt");
            assert_eq!(events[1]["type"], "error", "{name} error event");
            assert_eq!(events[1]["id"], id, "{name} error id");
            assert_eq!(events[1]["attempt_id"], attempt_id, "{name} error attempt");
            assert_eq!(
                hipfire_generate::ar::active_generation_route(),
                None,
                "{name} route cleanup"
            );
            assert_eq!(events[1]["class"], class, "{name} error class");
            assert_eq!(events[1]["retryable"], false, "{name} retryability");
            assert_eq!(events[1]["rolled_back"], false, "{name} rollback");
            assert_eq!(
                batch_terminal_generation(&id, attempt_id),
                Some(admission_a),
                "{name} admission remains until ClearOnExit"
            );

            // ClearOnExit owns A's opaque token. A stale second drop must
            // never remove a same-key B admission.
            assert!(batch_clear_terminal_at_generation(
                &id,
                attempt_id,
                admission_a
            ));
            let admission_b = hipfire_engine::terminal::batch_announce_terminal(&id, attempt_id)
                .expect("slot admission B");
            assert_ne!(admission_a, admission_b, "{name} fresh admission");
            assert!(
                !batch_clear_terminal_at_generation(&id, attempt_id, admission_a),
                "{name} stale ClearOnExit"
            );
            assert_eq!(
                batch_terminal_generation(&id, attempt_id),
                Some(admission_b),
                "{name} B survived stale cleanup"
            );

            {
                let _batch_scope =
                    BatchAttemptScope::enter_for_generation(&id, attempt_id, admission_b);
                let _route_scope = GenerationRouteScope::enter(GenerationRoute::QwenAr, &id);
                emit_generation_start(GenerationRoute::QwenAr, &mut output, &id, false);
                emit_qwen_ar_slot_error(&mut output, &id, message, class, false, false);
            }
            let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
                .expect("reused slot UTF-8")
                .lines()
                .map(|line| serde_json::from_str(line).expect("reused slot JSON"))
                .collect();
            assert_eq!(
                events
                    .iter()
                    .filter(|event| event["type"] == "gen_start")
                    .count(),
                2,
                "{name} fresh start after error"
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| event["type"] == "error")
                    .count(),
                2,
                "{name} one error per generation"
            );
            assert_eq!(events[2]["attempt_id"], attempt_id, "{name} reuse attempt");
            assert_eq!(events[3]["class"], class, "{name} reuse error class");
            assert!(batch_clear_terminal_at_generation(
                &id,
                attempt_id,
                admission_b
            ));
        }
        batch_clear_all_terminals();
        hipfire_engine::terminal::set_active_attempt_id(0);
    }

    #[test]
    fn swap_policy_resolves_defaults_and_cli_overrides() {
        let resolved = hipfire_config::resolve(Vec::<hipfire_config::NamedLayer>::new()).unwrap();
        let mut process = hipfire_config::ProcessConfig::from_resolved(&resolved).unwrap();
        let (host, disk, dir) = resolve_swap_policy(&process);
        assert_eq!(host, hipfire_runtime::swap::DEFAULT_HOST_BUDGET_BYTES);
        assert_eq!(disk, hipfire_runtime::swap::DEFAULT_DISK_BUDGET_BYTES);
        assert_eq!(dir, default_swap_dir());

        process
            .values
            .set_cli("serve.swap_host_bytes", "1048576")
            .unwrap();
        process
            .values
            .set_cli("serve.swap_disk_bytes", "0")
            .unwrap();
        process
            .values
            .set_cli("serve.swap_dir", "/tmp/hipfire-swap-test")
            .unwrap();
        let (host, disk, dir) = resolve_swap_policy(&process);
        assert_eq!(host, 1_048_576);
        assert_eq!(disk, 0);
        assert_eq!(dir, PathBuf::from("/tmp/hipfire-swap-test"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn dead_pid_swap_dirs_are_swept() {
        let parent = std::env::temp_dir().join("hipfire-deadpid-test");
        let _ = std::fs::remove_dir_all(&parent);
        std::fs::create_dir_all(&parent).unwrap();
        let dead = parent.join("hipfire-swap-4000000");
        let live = parent.join(format!("hipfire-swap-{}", std::process::id()));
        let keep = parent.join("keep");
        for dir in [&dead, &live, &keep] {
            std::fs::create_dir_all(dir).unwrap();
        }
        assert_eq!(sweep_dead_pid_dirs(&parent, "hipfire-swap-"), 1);
        assert!(!dead.exists(), "a dead pid's dir is removed");
        assert!(live.exists(), "the current pid's dir is kept");
        assert!(keep.exists(), "names outside the pattern are kept");
        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn default_swap_dir_is_per_process_and_prefers_the_runtime_dir() {
        use std::ffi::OsStr;
        let xdg = default_swap_dir_from(
            Some(OsStr::new("/run/user/1000")),
            std::path::Path::new("/tmp"),
            4242,
        );
        assert_eq!(xdg, PathBuf::from("/run/user/1000/hipfire/swap/4242"));
        let temp = default_swap_dir_from(None, std::path::Path::new("/var/tmp"), 4242);
        assert_eq!(temp, PathBuf::from("/var/tmp/hipfire-swap-4242"));
        let empty =
            default_swap_dir_from(Some(OsStr::new("")), std::path::Path::new("/var/tmp"), 7);
        assert_eq!(empty, PathBuf::from("/var/tmp/hipfire-swap-7"));
    }
}
