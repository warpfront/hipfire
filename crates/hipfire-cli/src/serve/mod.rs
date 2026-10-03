// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Serve host lifecycle and shared state.
//!
//! Concern: process/daemon spawning, admission control, model lifecycle, idle eviction,
//! and the `serve` subcommand entrypoints. Centralises all state that must be
//! shared across HTTP workers so transport changes stay isolated.

use crate::{
    apply_kv_axis_overrides, config_bool, config_f64, config_i64, config_string, config_u64,
    find_daemon, find_model_path, http_get_json, list_local_models, load_params, probe_host,
    pull_command, resolve_mtp_sidecar, resolved_for_model, resolved_global, ListArgs, Paths,
    PullArgs, ServeArgs, StopArgs,
};
use anyhow::{anyhow, bail, Context, Result};
use hipfire_client::Engine;
use hipfire_config::{load_catalog, load_global, resolve, ConfigLayer, ConfigSource, NamedLayer};
use hipfire_registry::{load as load_registry, RegistryV1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command},
    sync::{Arc, Condvar, Mutex},
    thread,
    time::{Duration, Instant},
};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

pub(crate) mod metrics;

pub mod complete;
pub mod http;

#[derive(Debug)]
pub(crate) struct ServeMeta {
    pub(crate) current_model: Option<String>,
    pub(crate) loading_model: Option<String>,
    pub(crate) instance_token: String,
    pub(crate) requests_served: u64,
    pub(crate) retries_attempted: u64,
    pub(crate) retries_succeeded: u64,
    pub(crate) recent_tok_s: Option<f64>,
    pub(crate) started: Instant,
    pub(crate) last_activity: Instant,
    /// Daemon health published to `/health`. Lives here, not on
    /// `ServeRuntime`, because a respawn and reload hold the runtime lock.
    pub(crate) engine_state: EngineState,
    /// Effective context of the resident model — the `max_seq` it was actually
    /// loaded with (KV capacity), not the registry policy and not the trained
    /// window. `0` while nothing is resident.
    ///
    /// Mirrored here, next to `current_model` and set at the same two sites
    /// (`ensure_model` / `clear_resident`), because `/health` must be able to
    /// answer while a model load holds `ServeShared::runtime`: that mutex is
    /// held for the whole load (prewarm and request-time loads alike), so
    /// reading load facts through it would block `/health` for the duration —
    /// the endpoint every readiness probe, the TUI and `serve_harness` poll
    /// with a sub-second timeout (`docs/SERVE.md` § "Detached readiness").
    pub(crate) n_ctx: u64,
    /// Facts about the resident model from the daemon's load ack. Same reason
    /// as `n_ctx` for living here rather than on `ServeRuntime`: it is
    /// published state, not working state, and `/v1/models` may not need the
    /// runtime lock to serve an entry.
    pub(crate) loaded: LoadedInfo,
}

impl ServeMeta {
    /// Fresh serve state: no model, no load-ack facts, daemon up.
    pub(crate) fn new(instance_token: String) -> Self {
        Self {
            current_model: None,
            loading_model: None,
            instance_token,
            requests_served: 0,
            retries_attempted: 0,
            retries_succeeded: 0,
            recent_tok_s: None,
            started: Instant::now(),
            last_activity: Instant::now(),
            engine_state: EngineState::Up,
            n_ctx: 0,
            loaded: LoadedInfo::default(),
        }
    }
}

/// Whether the daemon behind serve can take requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EngineState {
    Up,
    /// The daemon exited; serve is respawning it and reloading the model.
    Restarting,
    /// Serve cannot answer requests and exits: the daemon could not be
    /// respawned, or multi-slot serve could not load its model.
    Down,
}

impl EngineState {
    /// `/health` `status` value.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Up => "ok",
            Self::Restarting => "restarting",
            Self::Down => "unhealthy",
        }
    }
}

fn set_engine_state(meta: &Mutex<ServeMeta>, state: EngineState) {
    meta.lock()
        .unwrap_or_else(|error| error.into_inner())
        .engine_state = state;
}

/// How serve starts a daemon, kept so a dead one can be replaced.
pub(crate) struct EngineSpawner {
    pub(crate) daemon: PathBuf,
    pub(crate) process_config: hipfire_config::ProcessConfig,
    /// Spawn attempts per recovery before serve gives up.
    pub(crate) attempts: u32,
    /// Wait before the second attempt; doubles after each failure.
    pub(crate) backoff: Duration,
}

impl EngineSpawner {
    /// `model`: the model the new daemon is started for
    /// ([`Engine::spawn_configured`]).
    pub(crate) fn spawn(&self, model: Option<&Path>) -> Result<Engine> {
        let engine =
            Engine::spawn_configured(&self.daemon, &BTreeMap::new(), &self.process_config, model)?;
        engine.ping()?;
        Ok(engine)
    }
}

/// Who named the model being loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelOrigin {
    /// The operator: `serve.default_model` / `--model` pre-warm, or the reload
    /// of the resident model after a daemon restart.
    Operator,
    /// The `model` field of an HTTP request.
    Request,
}

/// What an HTTP request's `model` field may make serve do.
pub(crate) struct RequestModelPolicy {
    /// `serve.allow_request_pull`: download a registry model that is not local.
    pub(crate) allow_pull: bool,
    /// `serve.allow_request_paths`: load any readable file the request names.
    pub(crate) allow_paths: bool,
    /// The operator's pre-warm model, which requests may always name.
    pub(crate) operator_model: Option<String>,
}

pub(crate) fn finish_prewarm(meta: &mut ServeMeta, succeeded: bool) {
    meta.loading_model = None;
    if succeeded {
        meta.last_activity = Instant::now();
    }
}

/// Load the operator's model (`serve.default_model` / `--model`) at startup.
///
/// Single-slot serve logs a failure and keeps serving: the first request
/// loads the model instead. Multi-slot serve fails closed
/// ([`ServeRuntime::fail_closed_multi_slot`]) and returns the error, for the
/// caller to exit on while it still holds `runtime`.
pub(crate) fn prewarm(
    runtime: &mut ServeRuntime,
    meta: &Mutex<ServeMeta>,
    model: &str,
) -> Result<()> {
    let result = runtime.ensure_model(model, meta, ModelOrigin::Operator);
    {
        let mut meta = meta.lock().unwrap_or_else(|error| error.into_inner());
        finish_prewarm(&mut meta, result.is_ok());
    }
    match result {
        Ok(_) => eprintln!("[hipfire] pre-warmed {model}"),
        Err(error) if runtime.multi_slot_enabled => {
            return Err(runtime.fail_closed_multi_slot(
                meta,
                &format!("pre-warm of {model}"),
                error,
            ));
        }
        Err(error) => eprintln!("[hipfire] pre-warm failed: {error:#}; serving lazily"),
    }
    Ok(())
}

pub(crate) fn idle_model_expired(meta: &ServeMeta, idle_timeout: Duration) -> bool {
    meta.loading_model.is_none()
        && meta.current_model.is_some()
        && meta.last_activity.elapsed() >= idle_timeout
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ServePidRecord {
    pub(crate) pid: u32,
    #[serde(default)]
    pub(crate) start_time: Option<u64>,
    #[serde(default)]
    pub(crate) port: Option<u16>,
    #[serde(default)]
    pub(crate) token: Option<String>,
    #[serde(skip)]
    pub(crate) legacy: bool,
}

/// Facts about the resident model taken from the daemon's load ack.
///
/// The ack has always carried these (`{"type":"loaded","arch":…,"dim":…,
/// "layers":…,"vocab":…,"vl":…}`); serve read `arch`/`cache_capable`/
/// `continuous_batch_capable` and dropped the rest. `vl` is the daemon's
/// `LoadedModel::has_vision_encoder()` — a vision tower is present in this
/// load (qwen3.5-VL tower, dots.ocr, or the lfm2-vl tower), the same
/// carrier-level probe the generate path's image gate uses. A tower
/// configured but skipped by `vision_mode=off` reports `false`; so does a
/// text-only checkpoint of a VL-capable arch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LoadedInfo {
    pub(crate) vision: bool,
    /// Hidden size (`dim`) — llama.cpp's `meta.n_embd`.
    pub(crate) n_embd: u64,
    /// Vocabulary size (`vocab`) — llama.cpp's `meta.n_vocab`.
    pub(crate) n_vocab: u64,
}

pub(crate) struct ServeRuntime {
    pub(crate) engine: Engine,
    pub(crate) paths: Paths,
    pub(crate) registry: RegistryV1,
    pub(crate) current_path: Option<PathBuf>,
    pub(crate) current_arch: Option<String>,
    pub(crate) current_reasoning_contract: saddle_core::caps::ReasoningContract,
    pub(crate) current_reasoning_effort_native: bool,
    pub(crate) current_reasoning_efforts: Vec<String>,
    pub(crate) continuous_batch_capable: bool,
    pub(crate) current_max_seq: u64,
    pub(crate) cache_capable: bool,
    /// Set when an attempt failed mid-generation without a rollback: the
    /// slot session is mid-turn, so the next attempt cold-resets before
    /// generating (see `poisons_session_state` in `complete.rs`). Cleared
    /// by that reset and by `clear_resident`.
    pub(crate) needs_session_reset: bool,
    pub(crate) kv_override: Option<String>,
    pub(crate) kv_k_override: Option<String>,
    pub(crate) kv_v_override: Option<String>,
    pub(crate) kv_backend_override: Option<String>,
    /// Explicit vision-tower sidecar (`serve --vision`) projected as
    /// `params["vision"]` on every model load, winning over the registry
    /// `vision` slot and `HIPFIRE_VISION_SIDECAR`; skipped while
    /// `vision_mode=off`.
    pub(crate) vision_override: Option<PathBuf>,
    pub(crate) tp: Option<u64>,
    pub(crate) continuous_batch_size: u64,
    /// Experimental daemon multi-slot mode (`serve.multi_slot`). Default off.
    /// Projects load/generate wire markers only; CLI holds no GPU slot backend.
    pub(crate) multi_slot_enabled: bool,
    pub(crate) multi_slot_slots: u64,
    pub(crate) multi_slot_ctx: u64,
    pub(crate) multi_slot_prefill_chunk: u64,
    pub(crate) spawner: EngineSpawner,
    /// The `model` string the resident model was loaded under; reloaded after
    /// a daemon restart.
    pub(crate) resident_model: Option<String>,
    pub(crate) request_policy: RequestModelPolicy,
    /// Resolved (possibly scratch-clamped) `serve.max_batch_tokens`, forwarded
    /// on the multi-slot load params so the daemon's slot engine honours the
    /// same budget the CLI validated.
    pub(crate) max_batch_tokens: u64,
}

pub(crate) struct ServeShared {
    pub(crate) runtime: Mutex<ServeRuntime>,
    pub(crate) meta: Mutex<ServeMeta>,
    pub(crate) max_request_bytes: u64,
    pub(crate) admission: Arc<Admission>,
    pub(crate) idle_timeout: Duration,
    pub(crate) retry_enabled: bool,
    pub(crate) retry_backoff: Duration,
    /// Test seam: when set, invoked instead of `thread::sleep` during retry backoff.
    pub(crate) backoff_hook: Mutex<Option<Arc<dyn Fn(Duration) + Send + Sync>>>,
    /// Prometheus counters and histograms for `/metrics`. Lock-free, so a
    /// scrape never contends with a request.
    pub(crate) metrics: metrics::Metrics,
    /// Maximum stalled-consumer interval (spec §5.4
    /// `serve.stream_stall_timeout_ms`) on the multi-slot route: a streaming
    /// client that leaves the response channel full this long is aborted and
    /// its admission permit released. `None` on the standard route, whose
    /// streaming contract is unchanged.
    pub(crate) stream_stall_timeout: Option<Duration>,
    /// Route capability advertisement (spec §9.1 observability; OpenAI
    /// discovery). Built ONCE at startup from the same resolved config the
    /// daemon reads, so what `/health` advertises is what the slot engine
    /// was built with — never a separate source of truth. Immutable:
    /// route capabilities are configuration, not live state.
    pub(crate) capabilities: serde_json::Value,
}

/// Build the route capability advertisement from the resolved serve
/// configuration (spec §9.1). Pure so tests can pin the payload shape.
/// Values come from the SAME config resolution the daemon's slot engine
/// reads, so this never drifts from what the engine was built with.
#[allow(clippy::too_many_arguments)]
pub(crate) fn route_capabilities(
    multi_slot: bool,
    multi_slot_slots: u64,
    multi_slot_ctx: u64,
    multi_slot_prefill_chunk: u64,
    prefix_cache: bool,
    prefix_cache_max_bytes: u64,
    structured_jump_forward: bool,
    max_batch_tokens: u64,
    prefill_min_tokens: u64,
    max_queue: u64,
    max_queue_bytes: u64,
    queue_timeout_ms: u64,
    stream_stall_timeout_ms: Option<u64>,
    max_request_bytes: u64,
) -> serde_json::Value {
    // Structured output is a property of the multi-slot route (the strict
    // json_schema subset + framing-aware cursor live in the slot engine);
    // the standard route does not enforce response_format.
    let structured_output = multi_slot;
    serde_json::json!({
        "openai_compatible": true,
        "mode": if multi_slot { "multi-slot" } else { "standard" },
        "streaming": true,
        "multi_slot": multi_slot,
        "multi_slot_slots": multi_slot_slots,
        "multi_slot_ctx": multi_slot_ctx,
        "multi_slot_prefill_chunk": multi_slot_prefill_chunk,
        "prefix_cache": prefix_cache,
        "prefix_cache_max_bytes": prefix_cache_max_bytes,
        "structured_output": structured_output,
        "structured_output_subset": if structured_output {
            serde_json::json!("json-schema-strict-v1")
        } else {
            serde_json::Value::Null
        },
        "structured_jump_forward": structured_jump_forward,
        "max_batch_tokens": max_batch_tokens,
        "prefill_min_tokens": prefill_min_tokens,
        "max_queue": max_queue,
        "max_queue_bytes": max_queue_bytes,
        "queue_timeout_ms": queue_timeout_ms,
        "stream_stall_timeout_ms": stream_stall_timeout_ms,
        "max_request_bytes": max_request_bytes,
        // Honest refusal list: fields this route rejects BEFORE generation
        // (typed 400s). Mirrors `multi_slot_request_supported` (gateway) +
        // `validate_generate_caps` (daemon), which together refuse every
        // entry below — a client that only reads the advertisement must not
        // be surprised by a 400 for a field it was never told about.
        // `images+tools` is a rejected COMBINATION, not a field, so it is
        // spelled out separately.
        "refused_request_fields": if multi_slot {
            serde_json::json!([
                "stop",
                "logprobs",
                "top_logprobs",
                "n",
                "best_of",
                "logit_bias",
                "echo",
                "suffix",
                "reasoning_effort",
                "response_format:json_object",
                "tools+image",
            ])
        } else {
            serde_json::json!([])
        },
    })
}

#[derive(Debug, Default)]
pub(crate) struct AdmissionState {
    /// Batch-eligible requests currently admitted. The single-daemon backend
    /// caps this at one; the multi-slot engine admits `capacity` at once.
    eligible: usize,
    /// A batch-ineligible request holds the backend exclusively.
    ineligible_busy: bool,
    queued: usize,
    /// Total canonical pending-input bytes held by queued requests (spec §5.3).
    /// A queue count alone is insufficient; this bounds total bytes so a few
    /// large prompts cannot exhaust memory while staying under `max_queue`.
    queued_bytes: u64,
    batch_model: Option<String>,
}

pub(crate) struct Admission {
    state: Mutex<AdmissionState>,
    available: Condvar,
    notify: Notify,
    max_queue: usize,
    timeout: Duration,
    /// Aggregate canonical pending-input byte budget (spec §5.3
    /// `serve.max_queue_bytes`). Zero disables the byte cap (only valid when
    /// multi-slot is off; the startup guard rejects `max_queue==0` for
    /// multi-slot, and this field is positive from config validation).
    max_queue_bytes: u64,
    /// How many batch-eligible requests may be in flight at once. One for the
    /// single-daemon backend -- this gate is what protects it -- and the slot
    /// count when the multi-slot engine is active, which has its own admission
    /// behind it.
    capacity: usize,
}

impl std::fmt::Debug for Admission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        f.debug_struct("Admission")
            .field("state", &*state)
            .field("max_queue", &self.max_queue)
            .field("max_queue_bytes", &self.max_queue_bytes)
            .field("timeout", &self.timeout)
            .field("capacity", &self.capacity)
            .finish()
    }
}

#[derive(Debug)]
pub(crate) struct AdmissionError {
    message: String,
    retry_after_seconds: u64,
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AdmissionError {}

#[derive(Debug)]
pub(crate) struct AdmissionGuard {
    admission: Arc<Admission>,
    is_eligible: bool,
    model: Option<String>,
    /// Canonical pending-input bytes this request charged to the queue
    /// (spec §5.3). Released exactly once at ACQUIRE time (the queued
    /// byte charge converts to the in-flight charge) across
    /// cancel/timeout/normal paths. Zero when the request was admitted
    /// immediately (never queued).
    queue_bytes: u64,
}

impl AdmissionGuard {
    /// Canonical pending-input bytes carried by this permit (spec §5.3).
    /// The daemon submit path reads this to propagate the unified permit.
    pub(crate) fn queue_bytes(&self) -> u64 {
        self.queue_bytes
    }

    pub(crate) fn is_eligible(&self) -> bool {
        self.is_eligible
    }
}

struct AdmissionWaiter {
    admission: Arc<Admission>,
    active: bool,
    /// Bytes this waiter charged to `queued_bytes`; released on drop if active.
    bytes: u64,
}

impl AdmissionWaiter {
    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for AdmissionWaiter {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut state = self
            .admission
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.queued = state.queued.saturating_sub(1);
        state.queued_bytes = state.queued_bytes.saturating_sub(self.bytes);
        drop(state);
        self.admission.available.notify_all();
        self.admission.notify.notify_waiters();
    }
}

impl Admission {
    pub(crate) fn new(max_queue: usize, timeout: Duration) -> Self {
        Self::new_with_capacity_and_bytes(max_queue, timeout, 1, 0)
    }
    pub(crate) fn new_with_capacity(max_queue: usize, timeout: Duration, capacity: usize) -> Self {
        Self::new_with_capacity_and_bytes(max_queue, timeout, capacity, 0)
    }

    /// Construct with an aggregate queue byte budget (spec §5.3
    /// `serve.max_queue_bytes`). `max_queue_bytes == 0` disables the byte
    /// cap (only valid when multi-slot is off).
    pub(crate) fn new_with_capacity_and_bytes(
        max_queue: usize,
        timeout: Duration,
        capacity: usize,
        max_queue_bytes: u64,
    ) -> Self {
        Self {
            state: Mutex::new(AdmissionState::default()),
            available: Condvar::new(),
            notify: Notify::new(),
            max_queue,
            timeout,
            max_queue_bytes,
            capacity: capacity.max(1),
        }
    }

    pub(crate) fn is_model_compatible(current: &Option<String>, requested: Option<&str>) -> bool {
        match (current, requested) {
            (None, _) => true,
            (Some(cur), Some(req)) => cur == req,
            (Some(_), None) => true,
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total canonical pending-input bytes currently held by queued
    /// requests (spec §5.3).
    pub(crate) fn queued_bytes(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .queued_bytes
    }

    pub(crate) fn max_queue_bytes(&self) -> u64 {
        self.max_queue_bytes
    }

    /// Check the aggregate byte budget before queuing (spec §5.3).
    /// Returns `Ok(())` if the bytes fit, or a typed `QueueFull` error.
    fn check_byte_budget(&self, state: &AdmissionState, bytes: u64) -> Result<(), AdmissionError> {
        if self.max_queue_bytes == 0 {
            return Ok(());
        }
        let new_total = state.queued_bytes.saturating_add(bytes);
        if new_total > self.max_queue_bytes {
            return Err(AdmissionError {
                message: format!(
                    "serve queue byte budget exceeded ({}+{} > {})",
                    state.queued_bytes, bytes, self.max_queue_bytes
                ),
                retry_after_seconds: self.retry_after_seconds(),
            });
        }
        Ok(())
    }

    pub(crate) fn acquire(self: &Arc<Self>) -> std::result::Result<AdmissionGuard, AdmissionError> {
        self.acquire_for(false, None)
    }

    pub(crate) fn acquire_for(
        self: &Arc<Self>,
        is_eligible: bool,
        model: Option<&str>,
    ) -> std::result::Result<AdmissionGuard, AdmissionError> {
        self.acquire_for_with_bytes(is_eligible, model, 0)
    }

    /// Synchronous acquire with a canonical pending-input byte charge
    /// (spec §5.3). `request_bytes` is charged to the aggregate queue byte
    /// budget while waiting and released on the guard's Drop.
    pub(crate) fn acquire_for_with_bytes(
        self: &Arc<Self>,
        is_eligible: bool,
        model: Option<&str>,
        request_bytes: u64,
    ) -> std::result::Result<AdmissionGuard, AdmissionError> {
        let model_owned = model.map(|s| s.to_owned());
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        // Fast path when no queued waiters and resource is available.
        if is_eligible {
            if !state.ineligible_busy
                && state.eligible < self.capacity
                && state.queued == 0
                && Self::is_model_compatible(&state.batch_model, model_owned.as_deref())
            {
                state.eligible += 1;
                if state.batch_model.is_none() {
                    state.batch_model = model_owned.clone();
                }
                return Ok(AdmissionGuard {
                    admission: Arc::clone(self),
                    is_eligible: true,
                    model: model_owned,
                    queue_bytes: 0,
                });
            }
        } else if state.eligible == 0 && !state.ineligible_busy && state.queued == 0 {
            state.ineligible_busy = true;
            return Ok(AdmissionGuard {
                admission: Arc::clone(self),
                is_eligible: false,
                model: None,
                queue_bytes: 0,
            });
        }
        if self.max_queue != 0 && state.queued >= self.max_queue {
            return Err(self.queue_full_error(state.queued));
        }
        self.check_byte_budget(&state, request_bytes)?;
        state.queued = state.queued.saturating_add(1);
        state.queued_bytes = state.queued_bytes.saturating_add(request_bytes);
        let started = Instant::now();
        loop {
            if self.timeout.is_zero() {
                state = self
                    .available
                    .wait(state)
                    .unwrap_or_else(|error| error.into_inner());
            } else {
                let remaining = self.timeout.saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    state.queued = state.queued.saturating_sub(1);
                    state.queued_bytes = state.queued_bytes.saturating_sub(request_bytes);
                    return Err(self.wait_timeout_error());
                }
                let (next, wait) = self
                    .available
                    .wait_timeout(state, remaining)
                    .unwrap_or_else(|error| error.into_inner());
                state = next;
                if wait.timed_out() {
                    let can_acquire = if is_eligible {
                        !state.ineligible_busy
                            && state.eligible < self.capacity
                            && Self::is_model_compatible(&state.batch_model, model_owned.as_deref())
                    } else {
                        state.eligible == 0 && !state.ineligible_busy
                    };
                    if !can_acquire {
                        state.queued = state.queued.saturating_sub(1);
                        state.queued_bytes = state.queued_bytes.saturating_sub(request_bytes);
                        return Err(self.wait_timeout_error());
                    }
                }
            }
            let can_acquire = if is_eligible {
                !state.ineligible_busy
                    && state.eligible < self.capacity
                    && Self::is_model_compatible(&state.batch_model, model_owned.as_deref())
            } else {
                state.eligible == 0 && !state.ineligible_busy
            };
            if can_acquire {
                state.queued = state.queued.saturating_sub(1);
                state.queued_bytes = state.queued_bytes.saturating_sub(request_bytes);
                if is_eligible {
                    state.eligible += 1;
                    if state.batch_model.is_none() {
                        state.batch_model = model_owned.clone();
                    }
                    return Ok(AdmissionGuard {
                        admission: Arc::clone(self),
                        is_eligible: true,
                        model: model_owned.clone(),
                        queue_bytes: request_bytes,
                    });
                } else {
                    state.ineligible_busy = true;
                    return Ok(AdmissionGuard {
                        admission: Arc::clone(self),
                        is_eligible: false,
                        model: None,
                        queue_bytes: request_bytes,
                    });
                }
            }
        }
    }

    pub(crate) fn inflight(&self) -> usize {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.eligible + usize::from(state.ineligible_busy) + state.queued
    }

    /// `Retry-After` for a 503. The waiter has already spent the whole queue
    /// timeout, and the running generation may end at any moment, so the hint
    /// stays short instead of repeating that wait.
    pub(crate) fn retry_after_seconds(&self) -> u64 {
        self.timeout.as_secs().clamp(1, 30)
    }

    fn queue_full_error(&self, queued: usize) -> AdmissionError {
        AdmissionError {
            message: format!(
                "server busy: serve queue full (depth {queued}/{}); retry later",
                self.max_queue
            ),
            retry_after_seconds: self.retry_after_seconds(),
        }
    }

    fn wait_timeout_error(&self) -> AdmissionError {
        AdmissionError {
            message: format!(
                "server busy: serve queue wait exceeded {}ms while another generation ran \
                 (serve.queue_timeout_ms); retry later",
                self.timeout.as_millis()
            ),
            retry_after_seconds: self.retry_after_seconds(),
        }
    }

    pub(crate) async fn acquire_async(
        self: &Arc<Self>,
        cancel: CancellationToken,
    ) -> std::result::Result<AdmissionGuard, AdmissionError> {
        self.acquire_for_async(false, None, cancel).await
    }

    pub(crate) async fn acquire_for_async(
        self: &Arc<Self>,
        is_eligible: bool,
        model: Option<&str>,
        cancel: CancellationToken,
    ) -> std::result::Result<AdmissionGuard, AdmissionError> {
        self.acquire_for_async_with_bytes(is_eligible, model, 0, cancel)
            .await
    }

    /// Async acquire with a canonical pending-input byte charge (spec §5.3).
    /// `request_bytes` is charged to the aggregate queue byte budget while
    /// waiting and released exactly once — on `AdmissionWaiter::drop` if the
    /// wait is cancelled/timed-out, or on `AdmissionGuard::drop` after
    /// successful admission.
    pub(crate) async fn acquire_for_async_with_bytes(
        self: &Arc<Self>,
        is_eligible: bool,
        model: Option<&str>,
        request_bytes: u64,
        cancel: CancellationToken,
    ) -> std::result::Result<AdmissionGuard, AdmissionError> {
        let model_owned = model.map(str::to_owned);
        if cancel.is_cancelled() {
            return Err(AdmissionError {
                message: "cancelled".to_string(),
                retry_after_seconds: self.retry_after_seconds(),
            });
        }
        {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            let can_acquire = if is_eligible {
                !state.ineligible_busy
                    && state.eligible < self.capacity
                    && state.queued == 0
                    && Self::is_model_compatible(&state.batch_model, model_owned.as_deref())
            } else {
                state.eligible == 0 && !state.ineligible_busy && state.queued == 0
            };
            if can_acquire {
                if is_eligible {
                    state.eligible += 1;
                    if state.batch_model.is_none() {
                        state.batch_model = model_owned.clone();
                    }
                    return Ok(AdmissionGuard {
                        admission: Arc::clone(self),
                        is_eligible: true,
                        model: model_owned,
                        queue_bytes: 0,
                    });
                }
                state.ineligible_busy = true;
                return Ok(AdmissionGuard {
                    admission: Arc::clone(self),
                    is_eligible: false,
                    model: None,
                    queue_bytes: 0,
                });
            }
            if self.max_queue != 0 && state.queued >= self.max_queue {
                return Err(self.queue_full_error(state.queued));
            }
            self.check_byte_budget(&state, request_bytes)?;
            state.queued = state.queued.saturating_add(1);
            state.queued_bytes = state.queued_bytes.saturating_add(request_bytes);
        }
        let mut queued = AdmissionWaiter {
            admission: Arc::clone(self),
            active: true,
            bytes: request_bytes,
        };

        let started = Instant::now();
        loop {
            // Register before inspecting state so a guard drop cannot notify
            // between the state check and creation of the wait future.
            let notified = self.notify.notified();
            if cancel.is_cancelled() {
                return Err(AdmissionError {
                    message: "cancelled".to_string(),
                    retry_after_seconds: self.retry_after_seconds(),
                    });
            }
            {
                let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
                let can_acquire = if is_eligible {
                    !state.ineligible_busy
                        && state.eligible < self.capacity
                        && Self::is_model_compatible(&state.batch_model, model_owned.as_deref())
                } else {
                    state.eligible == 0 && !state.ineligible_busy
                };
                if can_acquire {
                    state.queued = state.queued.saturating_sub(1);
                    state.queued_bytes = state.queued_bytes.saturating_sub(request_bytes);
                    queued.disarm();
                    if is_eligible {
                        state.eligible += 1;
                        if state.batch_model.is_none() {
                            state.batch_model = model_owned.clone();
                        }
                        return Ok(AdmissionGuard {
                            admission: Arc::clone(self),
                            is_eligible: true,
                            model: model_owned.clone(),
                            queue_bytes: request_bytes,
                        });
                    }
                    state.ineligible_busy = true;
                    return Ok(AdmissionGuard {
                        admission: Arc::clone(self),
                        is_eligible: false,
                        model: None,
                        queue_bytes: request_bytes,
                    });
                }
            }

            let remaining = self.timeout.saturating_sub(started.elapsed());
            if !self.timeout.is_zero() && remaining.is_zero() {
                return Err(self.wait_timeout_error());
            }
            if self.timeout.is_zero() {
                tokio::select! {
                    _ = notified => {}
                    _ = cancel.cancelled() => {}
                }
            } else {
                tokio::select! {
                    _ = notified => {}
                    _ = cancel.cancelled() => {}
                    _ = tokio::time::sleep(remaining) => {}
                }
            }
        }
    }
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        let mut state = self
            .admission
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.is_eligible {
            state.eligible = state.eligible.saturating_sub(1);
            if state.eligible == 0 {
                state.batch_model = None;
            }
        } else {
            state.ineligible_busy = false;
        }
        self.admission.available.notify_all();
        self.admission.notify.notify_waiters();
    }
}

/// Conservative batch eligibility for independent continuous-batch decode.
///
/// Eligible only for Qwen (arch 5/6) or dense LFM2 (`lfm` arch identity),
/// stateless text with no tools/images/stops/spec/adaptive/prefix behavior.
/// TP policy: Qwen admits ordinary tp=1 or pure expert-parallel tp=4 when the
/// daemon advertises batch capability; dense LFM remains tp=1 only. All other
/// tp/arch combinations fall back to sequential. Check is intentionally strict
/// and synchronous; model arch is taken from `current_arch` when available,
/// otherwise inferred from the requested model name containing `qwen` or `lfm`.
///
/// Message-shape gate matches daemon admission: absent/empty `messages` are
/// eligible; otherwise only exactly one `user` message with plain string
/// content (no tool_calls, no multipart/array content).
pub(crate) fn is_batch_eligible_request(
    body: &serde_json::Value,
    tp: Option<u64>,
    current_arch: Option<&str>,
    daemon_batch_capable: bool,
) -> bool {
    if !daemon_batch_capable {
        return false;
    }
    // Qwen or LFM2 only. Prefer runtime arch when known.
    let (is_qwen, is_lfm) = if let Some(arch) = current_arch {
        let arch_l = arch.to_ascii_lowercase();
        (arch_l.contains("qwen"), arch_l.contains("lfm"))
    } else if let Some(model) = body.get("model").and_then(|v| v.as_str()) {
        let model_l = model.to_ascii_lowercase();
        (model_l.contains("qwen"), model_l.contains("lfm"))
    } else {
        (false, false)
    };
    if !is_qwen && !is_lfm {
        return false;
    }
    // TP policy: Qwen tp=1 ordinary or tp=4 pure EP; dense LFM tp=1 only.
    let tp_degree = tp.unwrap_or(1);
    let tp_ok = if is_qwen {
        tp_degree == 1 || tp_degree == 4
    } else {
        tp_degree == 1
    };
    if !tp_ok {
        return false;
    }
    // Stateless text: no tools, no images, no stops, no spec, no adaptive, no prefix.
    if body
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty())
    {
        return false;
    }
    if body.get("tool_choice").is_some() {
        return false;
    }
    // Images via explicit image_base64.
    if body.get("image_base64").is_some() {
        return false;
    }
    // Message history must match daemon single-user plain-string shape.
    if !batch_messages_are_single_user(body) {
        return false;
    }
    if body.get("stop").is_some() {
        return false;
    }
    // Speculation / adaptive / prefix behavior disqualifies.
    for key in [
        "speculation",
        "dflash_mode",
        "mtp_mode",
        "ngram_draft",
        "prefill_sparse_threshold",
        "kv_adaptive",
        "prefix",
    ] {
        if body.get(key).is_some() {
            return false;
        }
    }
    true
}

/// HTTP/OpenAI `messages` are batch-eligible only when absent/empty (prompt
/// path) or exactly one user turn with plain string content. Multi-turn,
/// system/assistant/tool roles, tool_call payloads, and multipart/image
/// content stay on the sequential route — same contract as the daemon.
pub(crate) fn batch_messages_are_single_user(body: &serde_json::Value) -> bool {
    let Some(messages) = body.get("messages") else {
        return true;
    };
    let Some(arr) = messages.as_array() else {
        return false;
    };
    if arr.is_empty() {
        return true;
    }
    if arr.len() != 1 {
        return false;
    }
    let m0 = &arr[0];
    if m0.get("role").and_then(|v| v.as_str()) != Some("user") {
        return false;
    }
    // Tool-call payloads on the sole message force sequential (batch v1 has
    // no tools path). Empty / missing tool_calls is fine.
    if let Some(tc) = m0.get("tool_calls") {
        if tc.as_array().is_some_and(|a| !a.is_empty()) || tc.is_object() {
            return false;
        }
    }
    // Content must be a plain string (reject multipart/array/image parts).
    match m0.get("content") {
        Some(serde_json::Value::String(_)) => true,
        // Missing content is not a plain user string turn.
        None => false,
        // Arrays (multipart/text+image), objects, numbers, bool, null.
        Some(_) => false,
    }
}

pub(crate) fn serve_command(paths: &Paths, mut args: ServeArgs) -> Result<()> {
    let (_, resolved) = resolved_global(paths, true)?;
    let default_host = config_string(&resolved, "serve.host")?;
    let default_port = config_u64(&resolved, "serve.port")? as u16;
    let (host, port, positional_model) =
        resolve_serve_positionals(paths, &args.positionals, &default_host, default_port)?;
    if let Some(positional_model) = positional_model {
        if args
            .model
            .as_ref()
            .is_some_and(|model| model != &positional_model)
        {
            bail!("serve model specified more than once");
        }
        args.model = Some(positional_model);
    }
    if args.detach && !args.foreground_child {
        return detach_serve(paths, &args, &host, port);
    }
    if let Some(vision) = args.vision.as_ref() {
        if !vision.is_file() {
            bail!("vision sidecar not found: {}", vision.display());
        }
    }
    serve_foreground(paths, &args, &host, port, resolved)
}
pub(crate) fn resolve_serve_positionals(
    paths: &Paths,
    values: &[String],
    default_host: &str,
    default_port: u16,
) -> Result<(String, u16, Option<String>)> {
    let registry = load_registry(&paths.registry).registry;
    let mut host = None;
    let mut port = None;
    let mut model = None;
    for value in values {
        if let Ok(value_port) = value.parse::<u16>() {
            if port.replace(value_port).is_some() {
                bail!("serve port specified more than once");
            }
            continue;
        }
        if let Some((value_host, value_port)) = parse_host_port(value)? {
            if host.replace(value_host).is_some() || port.replace(value_port).is_some() {
                bail!("serve bind specified more than once");
            }
            continue;
        }
        let is_model =
            registry.model(value).is_some() || find_model_path(paths, &registry, value).is_some();
        if is_model && model.is_none() {
            model = Some(value.clone());
        } else if host.replace(value.clone()).is_some() {
            bail!("serve host specified more than once");
        }
    }
    Ok((
        host.unwrap_or_else(|| default_host.to_owned()),
        port.unwrap_or(default_port),
        model,
    ))
}

pub(crate) fn parse_host_port(value: &str) -> Result<Option<(String, u16)>> {
    if let Some(stripped) = value.strip_prefix('[') {
        if let Some((host, port)) = stripped.split_once("]:") {
            return Ok(Some((
                host.to_owned(),
                port.parse().context("invalid serve port")?,
            )));
        }
    }
    if value.matches(':').count() == 1 {
        if let Some((host, port)) = value.rsplit_once(':') {
            if let Ok(port) = port.parse::<u16>() {
                return Ok(Some((host.to_owned(), port)));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
pub(crate) fn parse_bind(
    address: Option<&str>,
    port: Option<u16>,
    default_host: &str,
    default_port: u16,
) -> Result<(String, u16)> {
    let Some(address) = address else {
        return Ok((default_host.to_owned(), port.unwrap_or(default_port)));
    };
    if let Ok(port_only) = address.parse::<u16>() {
        return Ok((default_host.to_owned(), port_only));
    }
    if let Some(stripped) = address.strip_prefix('[') {
        if let Some((host, port_text)) = stripped.split_once("]:") {
            return Ok((
                host.to_owned(),
                port_text.parse().context("invalid serve port")?,
            ));
        }
    }
    if address.matches(':').count() == 1 {
        if let Some((host, port_text)) = address.rsplit_once(':') {
            if let Ok(parsed) = port_text.parse::<u16>() {
                return Ok((host.to_owned(), parsed));
            }
        }
    }
    Ok((address.to_owned(), port.unwrap_or(default_port)))
}

pub(crate) fn detach_serve(paths: &Paths, args: &ServeArgs, host: &str, port: u16) -> Result<()> {
    fs::create_dir_all(&paths.root)?;
    let log_path = paths.root.join("serve.log");
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?;
    let executable = env::current_exe().context("failed to resolve native hipfire binary")?;
    let mut command = Command::new(executable);
    command
        .arg("serve")
        .arg(host)
        .arg(port.to_string())
        .arg("--foreground-child")
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    if args.no_prewarm {
        command.arg("--no-prewarm");
    }
    if let Some(model) = &args.model {
        command.arg("--model").arg(model);
    }
    if let Some(mode) = &args.kv_mode {
        command.arg("--kv-mode").arg(mode);
    }
    if let Some(k) = &args.kv_k {
        command.arg("--kv-k").arg(k);
    }
    if let Some(v) = &args.kv_v {
        command.arg("--kv-v").arg(v);
    }
    if let Some(backend) = &args.kv_backend {
        command.arg("--kv-backend").arg(backend);
    }
    if let Some(seconds) = args.idle_timeout {
        command.arg("--idle-timeout").arg(seconds.to_string());
    }
    if let Some(tp) = args.tp {
        command.arg("--tp").arg(tp.to_string());
    }
    if let Some(batch) = args.continuous_batch_size {
        command
            .arg("--continuous-batch-size")
            .arg(batch.to_string());
    }
    let mut child = command.spawn().context("failed to detach native serve")?;
    let probe_host = match host {
        "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    };
    for _ in 0..600 {
        if let Some(status) = child.try_wait()? {
            bail!(
                "native serve exited before readiness ({status}); see {}",
                log_path.display()
            );
        }
        if health_ready(probe_host, port) {
            println!(
                "hipfire serve running at http://{}:{} (PID {}, log {})",
                host,
                port,
                child.id(),
                log_path.display()
            );
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    bail!(
        "native serve did not become ready within 60s; PID {}, see {}",
        child.id(),
        log_path.display()
    )
}

pub(crate) fn health_ready(host: &str, port: u16) -> bool {
    let url = if host.contains(':') {
        format!("http://[{host}]:{port}/health")
    } else {
        format!("http://{host}:{port}/health")
    };
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(100)))
        .http_status_as_error(false)
        .build()
        .into();
    agent
        .get(&url)
        .call()
        .is_ok_and(|response| response.status().is_success())
}

pub(crate) fn serve_foreground(
    paths: &Paths,
    args: &ServeArgs,
    host: &str,
    port: u16,
    global: hipfire_config::ResolvedConfig,
) -> Result<()> {
    let daemon = find_daemon(paths).ok_or_else(|| anyhow!("daemon binary not found"))?;
    let registry = load_registry(&paths.registry).registry;
    let spawner = EngineSpawner {
        daemon,
        process_config: hipfire_config::ProcessConfig::from_resolved(&global)?,
        attempts: ENGINE_RESPAWN_ATTEMPTS,
        backoff: ENGINE_RESPAWN_BACKOFF,
    };
    let default_model = args
        .model
        .clone()
        .unwrap_or(config_string(&global, "serve.default_model")?);
    // Started for the pre-warm model, when it is on disk.
    let prewarm_path = if args.no_prewarm {
        None
    } else {
        find_model_path(paths, &registry, &default_model)
    };
    let engine = spawner.spawn(prewarm_path.as_deref())?;
    let max_request_bytes = config_u64(&global, "serve.max_request_bytes")?;
    let max_queue = config_u64(&global, "serve.max_queue")? as usize;
    let queue_timeout = Duration::from_millis(config_u64(&global, "serve.queue_timeout_ms")?);
    let continuous_batch_size = args
        .continuous_batch_size
        .unwrap_or(config_u64(&global, "serve.continuous_batch_size")?);
    if continuous_batch_size == 0 || continuous_batch_size > 256 {
        bail!("--continuous-batch-size must be between 1 and 256");
    }
    let multi_slot_enabled = config_bool(&global, "serve.multi_slot")?;
    let multi_slot_slots = config_u64(&global, "serve.multi_slot_slots").unwrap_or(4);
    let multi_slot_ctx = config_u64(&global, "serve.multi_slot_ctx").unwrap_or(8192);
    let multi_slot_prefill_chunk =
        config_u64(&global, "serve.multi_slot_prefill_chunk").unwrap_or(1024);
    // Serving cache/scheduler contract keys (spec §9.1). Read here for
    // startup validation only; the runtime does not yet consume them.
    let max_batch_tokens = config_u64(&global, "serve.max_batch_tokens")?;
    let prefill_min_tokens = config_u64(&global, "serve.prefill_min_tokens")?;
    // Aggregate queue byte budget and stalled-stream deadline (spec §5.3,
    // §5.4). Both are multi-slot route contracts: the standard route keeps
    // its count-only queue and its unbounded streaming wait.
    let max_queue_bytes = if multi_slot_enabled {
        config_u64(&global, "serve.max_queue_bytes")?
    } else {
        0
    };
    let stream_stall_timeout = if multi_slot_enabled {
        Some(Duration::from_millis(config_u64(
            &global,
            "serve.stream_stall_timeout_ms",
        )?))
    } else {
        None
    };
    // Cache/scheduler route flags (spec §9.1). Read with the SAME config
    // resolution the daemon applies so /health advertises the engine's
    // actual build options.
    let prefix_cache = config_bool(&global, "serve.prefix_cache")?;
    let prefix_cache_max_bytes = config_u64(&global, "serve.prefix_cache_max_bytes")?;
    let structured_jump_forward = config_bool(&global, "serve.structured_jump_forward")?;
    // Multi-slot scratch capacity is `prefill_chunk * n_slots` trunk rows. The
    // built-in `serve.max_batch_tokens` default (4096) exceeds that for the
    // documented slot counts (1..3 slots x 1024) and would refuse every such
    // start. When the key is still the built-in default, clamp it to the
    // scratch capacity rather than failing; a user-set value keeps the strict
    // refuse (an explicit oversized budget is a real misconfiguration).
    let max_batch_tokens = if multi_slot_enabled {
        let scratch_rows = multi_slot_prefill_chunk.saturating_mul(multi_slot_slots);
        let is_builtin = global
            .get("serve.max_batch_tokens")
            .map(|v| matches!(v.source, ConfigSource::BuiltIn))
            .unwrap_or(true);
        if is_builtin && max_batch_tokens > scratch_rows {
            eprintln!(
                "serve: built-in max_batch_tokens {max_batch_tokens} exceeds multi-slot scratch capacity {scratch_rows} ({multi_slot_slots} x {multi_slot_prefill_chunk}); clamping to {scratch_rows}"
            );
            scratch_rows
        } else {
            max_batch_tokens
        }
    } else {
        max_batch_tokens
    };
    // Multi-slot is an alternate daemon-owned Qwen35 mode, not continuous batching.
    // Combining them is rejected until a future integration lands. The robust
    // multi-slot mode also rejects an uncapped queue (serve.max_queue=0) rather
    // than reinterpreting zero (spec §5.3), and the global trunk-row budget must
    // be at least the minimum prefill quantum (spec §5.2, §5.3).
    if let Err(message) = validate_multi_slot_startup(
        multi_slot_enabled,
        continuous_batch_size,
        max_queue as u64,
        max_batch_tokens,
        prefill_min_tokens,
        multi_slot_prefill_chunk,
        multi_slot_slots,
        prefix_cache,
        prefix_cache_max_bytes,
    ) {
        bail!("{message}");
    }
    let retry_enabled = config_bool(&global, "serve.retry_enabled")?;
    let retry_backoff = Duration::from_millis(config_u64(&global, "serve.retry_backoff_ms")?);
    let idle_timeout = Duration::from_secs(
        args.idle_timeout
            .unwrap_or(config_u64(&global, "serve.idle_timeout_seconds")?),
    );
    let request_policy = RequestModelPolicy {
        allow_pull: config_bool(&global, "serve.allow_request_pull")?,
        allow_paths: config_bool(&global, "serve.allow_request_paths")?,
        operator_model: Some(default_model.clone()),
    };
    let instance_token = serve_instance_token();
    // Admission width: experimental multi-slot projects N concurrent daemon
    // sessions; continuous batch admits up to continuous_batch_size. Take the
    // larger so the HTTP gate does not serialise what the backend can run.
    let slot_concurrency = if multi_slot_enabled {
        multi_slot_slots.max(1) as usize
    } else {
        1
    };
    if multi_slot_enabled {
        eprintln!(
            "serve: experimental multi-slot mode ({} slots, {} ctx) — daemon-owned, continuous batching deferred",
            multi_slot_slots, multi_slot_ctx
        );
    }
    // Capability advertisement snapshot (see `route_capabilities`): built
    // once here, after startup validation, from the resolved values below.
    let capabilities = route_capabilities(
        multi_slot_enabled,
        multi_slot_slots,
        multi_slot_ctx,
        multi_slot_prefill_chunk,
        prefix_cache,
        prefix_cache_max_bytes,
        structured_jump_forward,
        max_batch_tokens,
        prefill_min_tokens,
        max_queue as u64,
        max_queue_bytes,
        queue_timeout.as_millis() as u64,
        stream_stall_timeout.map(|t| t.as_millis() as u64),
        max_request_bytes,
    );
    let shared = Arc::new(ServeShared {
        capabilities,
        metrics: metrics::Metrics::default(),
        runtime: Mutex::new(ServeRuntime {
            engine,
            paths: paths.clone(),
            registry: registry.clone(),
            current_path: None,
            current_arch: None,
            current_reasoning_contract: saddle_core::caps::ReasoningContract::Unsupported,
            current_reasoning_effort_native: false,
            current_reasoning_efforts: Vec::new(),
            continuous_batch_capable: false,
            current_max_seq: 0,
            cache_capable: false,
            needs_session_reset: false,
            kv_override: args.kv_mode.clone(),
            kv_k_override: args.kv_k.clone(),
            kv_v_override: args.kv_v.clone(),
            kv_backend_override: args.kv_backend.clone(),
            vision_override: args.vision.clone(),
            tp: args.tp,
            continuous_batch_size,
            multi_slot_enabled,
            multi_slot_slots,
            multi_slot_ctx,
            multi_slot_prefill_chunk,
            spawner,
            resident_model: None,
            request_policy,
            max_batch_tokens,
        }),
        meta: Mutex::new(ServeMeta::new(instance_token.clone())),
        max_request_bytes,
        admission: Arc::new(Admission::new_with_capacity_and_bytes(
            max_queue,
            queue_timeout,
            slot_concurrency.max(continuous_batch_size as usize),
            max_queue_bytes,
        )),
        idle_timeout,
        retry_enabled,
        retry_backoff,
        backoff_hook: Mutex::new(None),
        stream_stall_timeout,
    });
    let bind = format_bind(host, port);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind(&bind))
        .map_err(|error| anyhow!("failed to bind {bind}: {error}"))?;
    fs::create_dir_all(&paths.root)?;
    let pid_path = paths.root.join("serve.pid");
    let pid_record = ServePidRecord {
        pid: std::process::id(),
        start_time: proc_start_time(std::process::id()),
        port: Some(port),
        token: Some(instance_token),
        legacy: false,
    };
    fs::write(
        &pid_path,
        format!("{}\n", serde_json::to_string(&pid_record)?),
    )?;
    let cleanup = pid_path.clone();
    // SIGINT/SIGTERM stops accepting and lets requests in flight finish
    // (bounded, see http::serve_listener_until); a second signal exits now.
    let shutdown = CancellationToken::new();
    let signal_shutdown = shutdown.clone();
    ctrlc::set_handler(move || {
        if signal_shutdown.is_cancelled() {
            let _ = fs::remove_file(&cleanup);
            std::process::exit(0);
        }
        eprintln!(
            "[hipfire] shutting down: finishing requests in flight (signal again to exit now)"
        );
        signal_shutdown.cancel();
    })
    .context("failed to install serve signal handler")?;
    eprintln!("[hipfire] native serve listening on http://{bind}");
    if !args.no_prewarm {
        let shared = Arc::clone(&shared);
        let pid_path = pid_path.clone();
        thread::spawn(move || {
            shared
                .meta
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .loading_model = Some(default_model.clone());
            let mut runtime = shared
                .runtime
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Err(error) = prewarm(&mut runtime, &shared.meta, &default_model) {
                // Exit with `runtime` still held, so no request waiting on it
                // starts a load of its own.
                eprintln!("[hipfire] {error:#}; exiting");
                let _ = fs::remove_file(&pid_path);
                std::process::exit(1);
            }
        });
    }
    // Daemon supervision: a daemon that exited (crash, panic, or a sticky GPU
    // fault, after which it exits 75) is respawned and the resident model
    // reloaded. If it cannot be respawned, serve exits so a service manager
    // restarts it instead of answering every request with an error.
    {
        let shared = Arc::clone(&shared);
        let pid_path = pid_path.clone();
        thread::spawn(move || loop {
            thread::sleep(ENGINE_POLL);
            if let Err(error) = supervise_engine(&shared) {
                eprintln!("[hipfire] {error:#}; exiting");
                let _ = fs::remove_file(&pid_path);
                std::process::exit(1);
            }
        });
    }
    if !shared.idle_timeout.is_zero() {
        let shared = Arc::clone(&shared);
        thread::spawn(move || loop {
            thread::sleep(Duration::from_secs(1));
            let expired = {
                let meta = shared
                    .meta
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                idle_model_expired(&meta, shared.idle_timeout)
            };
            if !expired {
                continue;
            }
            let unloaded = {
                let mut runtime = shared
                    .runtime
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                // Inflight check INSIDE the runtime lock (TOCTOU): the old
                // check-then-lock window let a request acquire its permit
                // between the check and the unload — unloading the model
                // out from under a live request. Under the lock, a request
                // either has its permit (inflight != 0 → skip) or has not
                // started (unload precedes it).
                if shared.admission.inflight() != 0 {
                    continue;
                }
                if runtime.current_path.is_some() {
                    let result = runtime.engine.unload();
                    if result.is_ok() {
                        runtime.clear_resident(&shared.meta);
                    }
                    result
                } else {
                    Ok(())
                }
            };
            if unloaded.is_ok() {
                let mut meta = shared
                    .meta
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                meta.current_model = None;
                meta.loading_model = None;
                meta.last_activity = Instant::now();
                eprintln!("[hipfire] unloaded idle model");
            }
        });
    }
    runtime.block_on(crate::serve::http::serve_listener_until(
        listener,
        Arc::clone(&shared),
        shutdown,
    ))?;
    let _ = fs::remove_file(pid_path);
    // A request still inside the daemon after the drain bound must not hold
    // the exit: Runtime::drop would wait for its blocking worker.
    runtime.shutdown_timeout(Duration::from_secs(1));
    Ok(())
}

pub(crate) fn format_bind(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

// lifecycle: deprecated since 0.4.0, removal 0.5.0 — MQ4R route selection by file extension; 0.5.0 selects by HFQ metadata (no runtime warning: only way to reach the route today)
pub(crate) fn should_prewarm_qwen_mq4r_decode(
    path: &Path,
    loaded: &serde_json::Value,
    tp: Option<u64>,
) -> bool {
    let qwen = loaded
        .get("arch")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|arch| arch.starts_with("qwen3_5"));
    let mq4r = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mq4r"));
    qwen && mq4r && tp.unwrap_or(1) == 1
}

pub(crate) fn prewarm_qwen_mq4r_decode(engine: &mut Engine) -> Result<()> {
    let response = engine.request(&serde_json::json!({
        "type": "bench_decode",
        "context_tokens": 64,
        "iterations": 32,
    }))?;
    match response.get("type").and_then(serde_json::Value::as_str) {
        Some("decode_result") => {
            let tok_s = response
                .get("tok_s")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            eprintln!("[hipfire] pre-warmed Qwen MQ4R decode route ({tok_s:.1} tok/s)");
            Ok(())
        }
        Some("error") => bail!(
            "Qwen MQ4R decode pre-warm failed: {}",
            response
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("daemon returned an unspecified error")
        ),
        other => bail!(
            "Qwen MQ4R decode pre-warm expected decode_result, received {}",
            other.unwrap_or("missing type")
        ),
    }
}

impl ServeRuntime {
    /// Make `model` the resident model, loading it if needed.
    ///
    /// A dead daemon is respawned first. For `ModelOrigin::Request`, a
    /// registry model that is not on disk is pulled only with
    /// `serve.allow_request_pull`, and a file outside the operator's model set
    /// is loaded only with `serve.allow_request_paths`; otherwise the request
    /// gets "model not found". The load never depends on the request's
    /// `max_tokens`: the daemon's `max_seq` cannot grow on reload, so the
    /// request budget is fitted or refused against it instead.
    pub(crate) fn ensure_model(
        &mut self,
        model: &str,
        meta: &Mutex<ServeMeta>,
        origin: ModelOrigin,
    ) -> Result<hipfire_config::ResolvedConfig> {
        if self.engine.exited() {
            self.restart_engine(meta)?;
            set_engine_state(meta, EngineState::Up);
        }
        let (tag, entry) = crate::registry_entry_for_path(&self.paths, &self.registry, model)
            .map(|(tag, entry)| (Some(tag.to_owned()), Some(entry)))
            .unwrap_or((None, None));
        let mut path = find_model_path(&self.paths, &self.registry, model);
        if path.is_none() && entry.is_some() {
            if origin == ModelOrigin::Request && !self.request_policy.allow_pull {
                return Err(ModelNotFound(format!(
                    "model not found locally: {model}; run `hipfire pull {model}` on the server \
                     first (downloads named by a request are off: serve.allow_request_pull)"
                ))
                .into());
            }
            pull_command(
                &self.paths,
                PullArgs {
                    model: model.to_owned(),
                    force: false,
                },
            )?;
            path = entry.map(|entry| self.paths.models.join(&entry.file));
        }
        let path = path.ok_or_else(|| ModelNotFound(format!("model not found locally: {model}")))?;
        let resolved = resolved_for_model(&self.paths, model, tag.as_deref(), entry)?;
        if self.current_path.as_ref() != Some(&path) {
            if origin == ModelOrigin::Request && !self.request_may_load(&path, entry) {
                return Err(ModelNotFound(format!(
                    "model not found: {model} (a request may name only installed models; \
                     loading other files is off: serve.allow_request_paths)"
                ))
                .into());
            }
            let max_tokens = config_u64(&resolved, "generation.max_tokens")?;
            let mut params = load_params(
                &resolved,
                entry,
                &self.paths.models,
                &path,
                max_tokens,
                self.kv_override.as_deref(),
                self.kv_backend_override.as_deref(),
                tag.as_deref(),
                false,
                // serve has no --head yet; models load their own head.
                None,
            )?;
            apply_kv_axis_overrides(
                &mut params,
                self.kv_k_override.as_deref(),
                self.kv_v_override.as_deref(),
            )?;
            resolve_mtp_sidecar(
                &mut params,
                entry,
                &self.paths.models,
                &path,
                model,
                tag.as_deref(),
            );
            if let Some(vision) = self.vision_override.as_ref() {
                // Forwarded in every mode; the daemon's `vision_mode=off` gate decides.
                params["vision"] = serde_json::json!(vision.display().to_string());
            }
            if let Some(tp) = self.tp {
                params["tp"] = serde_json::json!(tp);
            }
            params["continuous_batch_size"] = serde_json::json!(self.continuous_batch_size);
            // Experimental multi-slot: daemon owns SlotEngine instead of ordinary
            // LoadedModel. Future continuous-batch integration is deferred.
            if self.multi_slot_enabled {
                params["experimental_multi_slot"] = serde_json::json!(true);
                params["experimental_multi_slot_slots"] = serde_json::json!(self.multi_slot_slots);
                params["experimental_multi_slot_ctx"] = serde_json::json!(self.multi_slot_ctx);
                params["experimental_multi_slot_prefill_chunk"] =
                    serde_json::json!(self.multi_slot_prefill_chunk);
                // Forward the resolved (possibly clamped-to-scratch) trunk-row
                // budget so the daemon's slot engine does not re-read the
                // unclamped OnceLock `serve.max_batch_tokens` default and
                // refuse the documented 1-3 slot configuration.
                params["max_batch_tokens"] = serde_json::json!(self.max_batch_tokens);
                // This alternate backend's allocation cap is authoritative.
                // Do not advertise the ordinary model's larger max_seq to
                // request budgeting and then stop early at the slot boundary.
                params["max_seq"] = serde_json::json!(self.multi_slot_ctx);
            }
            let requested_max_seq = params["max_seq"].as_u64().unwrap_or(0);
            // For tp<=1 the daemon unloads the resident model before loading
            // the new one, so after a failed load nothing is resident. At
            // tp>1 it defers that unload until the new model is built, and a
            // failure leaves the old model in place.
            let loaded = match self.engine.load(&path, params) {
                Ok(loaded) => loaded,
                Err(error) => {
                    if self.tp.unwrap_or(1) <= 1 {
                        self.clear_resident(meta);
                    }
                    return Err(error.into());
                }
            };
            if !self.multi_slot_enabled && should_prewarm_qwen_mq4r_decode(&path, &loaded, self.tp)
            {
                // The new model is resident but not recorded yet; clear the
                // old record so the next request reloads.
                if let Err(error) = prewarm_qwen_mq4r_decode(&mut self.engine) {
                    self.clear_resident(meta);
                    return Err(error);
                }
            }
            self.cache_capable = loaded
                .get("cache_capable")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            self.current_path = Some(path);
            self.current_arch = loaded
                .get("arch")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            self.current_reasoning_contract = loaded
                .get("reasoning_contract")
                .and_then(serde_json::Value::as_str)
                .and_then(saddle_core::caps::ReasoningContract::from_wire_name)
                .unwrap_or(saddle_core::caps::ReasoningContract::Unsupported);
            self.current_reasoning_effort_native = loaded
                .get("reasoning_effort_native")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            self.current_reasoning_efforts = loaded
                .get("reasoning_efforts")
                .and_then(serde_json::Value::as_array)
                .map(|array| {
                    array
                        .iter()
                        .filter_map(|value| value.as_str().map(|string| string.to_owned()))
                        .collect()
                })
                .unwrap_or_default();
            self.continuous_batch_capable = loaded
                .get("continuous_batch_capable")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            self.current_max_seq = loaded
                .get("max_seq")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(requested_max_seq);
            let published = LoadedInfo {
                vision: loaded
                    .get("vl")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
                n_embd: loaded
                    .get("dim")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                n_vocab: loaded
                    .get("vocab")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
            };
            // Report the model the way it was requested. A path-form
            // request now resolves its registry entry (for sidecars and
            // tag policy), but clients — serve_harness's warm probe among
            // them — compare `/health.model` against the path they asked
            // for; a tag only stands in when the request was a tag.
            let served_name = if Path::new(model).is_absolute() || model.contains('/') {
                model.to_owned()
            } else {
                tag.unwrap_or_else(|| model.to_owned())
            };
            self.resident_model = Some(model.to_owned());
            // The served facts land in `ServeMeta`, not `ServeRuntime`: this
            // method runs with the runtime lock held for the whole load, and
            // `/health` (100 ms readiness probes, the TUI's 450 ms poll,
            // `serve_harness`'s 1 s poll) must keep answering while it does.
            let mut meta = meta.lock().unwrap_or_else(|error| error.into_inner());
            meta.current_model = Some(served_name);
            meta.n_ctx = self.current_max_seq;
            meta.loaded = published;
        }
        Ok(resolved)
    }

    /// Whether a request may load `path`, which its `model` field resolved
    /// to. Without `serve.allow_request_paths`, only models the operator put
    /// in place qualify: files under the models directory (symlinked entries
    /// included), the resolved registry artifact, paths registered in the
    /// model catalog, and the pre-warm model.
    fn request_may_load(&self, path: &Path, entry: Option<&hipfire_registry::ModelEntry>) -> bool {
        if self.request_policy.allow_paths {
            return true;
        }
        let Ok(target) = fs::canonicalize(path) else {
            return false;
        };
        let same = |candidate: &Path| fs::canonicalize(candidate).is_ok_and(|c| c == target);
        fs::canonicalize(&self.paths.models).is_ok_and(|models| target.starts_with(models))
            || entry.is_some_and(|entry| same(&self.paths.models.join(&entry.file)))
            || crate::local_model_paths(&self.paths, &self.registry)
                .unwrap_or_default()
                .iter()
                .any(|candidate| same(candidate))
            || load_catalog(&self.paths.config).is_ok_and(|catalog| {
                catalog
                    .catalog
                    .models
                    .values()
                    .filter_map(|record| record.path.as_deref())
                    .any(same)
            })
            || self
                .request_policy
                .operator_model
                .as_deref()
                .and_then(|model| find_model_path(&self.paths, &self.registry, model))
                .is_some_and(|operator| same(&operator))
    }

    /// Drop every field that names the resident model, serve-side and in
    /// `/health`, so the next request reloads. Used when the daemon is known
    /// to hold nothing (idle unload, failed switch, failed pre-warm, daemon
    /// restart) or its state cannot be trusted (failed forced reset).
    /// Ported from #787.
    pub(crate) fn clear_resident(&mut self, meta: &Mutex<ServeMeta>) {
        self.current_path = None;
        self.current_arch = None;
        self.current_reasoning_contract = saddle_core::caps::ReasoningContract::Unsupported;
        self.current_reasoning_effort_native = false;
        self.current_reasoning_efforts = Vec::new();
        self.continuous_batch_capable = false;
        self.current_max_seq = 0;
        self.cache_capable = false;
        self.needs_session_reset = false;
        self.resident_model = None;
        let mut meta = meta.lock().unwrap_or_else(|error| error.into_inner());
        meta.current_model = None;
        meta.n_ctx = 0;
        meta.loaded = LoadedInfo::default();
    }

    /// Replace a daemon that exited: reap it (freeing its GPU memory), then
    /// spawn a new one with bounded, backed-off retries. Leaves
    /// `engine_state` at `Restarting` on success (the caller sets `Up` once it
    /// is done with the new daemon) and `Down` on failure. Returns the model
    /// that was resident, for the caller to reload.
    pub(crate) fn restart_engine(&mut self, meta: &Mutex<ServeMeta>) -> Result<Option<String>> {
        set_engine_state(meta, EngineState::Restarting);
        let status = self.engine.terminate();
        let resident = self.resident_model.clone();
        let resident_path = self.current_path.clone();
        self.clear_resident(meta);
        eprintln!("[hipfire] daemon exited ({status}); restarting it");
        let attempts = self.spawner.attempts.max(1);
        let mut backoff = self.spawner.backoff;
        for attempt in 1..=attempts {
            match self.spawner.spawn(resident_path.as_deref()) {
                Ok(engine) => {
                    self.engine = engine;
                    eprintln!("[hipfire] daemon restarted (attempt {attempt}/{attempts})");
                    return Ok(resident);
                }
                Err(error) => {
                    eprintln!("[hipfire] daemon restart attempt {attempt}/{attempts} failed: {error:#}");
                }
            }
            if attempt < attempts {
                thread::sleep(backoff);
                backoff = backoff.saturating_mul(2);
            }
        }
        // A later successful restart still reloads what was resident.
        self.resident_model = resident;
        set_engine_state(meta, EngineState::Down);
        bail!("daemon exited and could not be restarted after {attempts} attempts")
    }

    /// Fail multi-slot serve closed after the operator's model failed to load
    /// (`what`: the pre-warm, or the reload after a daemon respawn). The slot
    /// engine is multi-slot serve's only backend. Such a load, typically
    /// slot-engine allocations that do not fit beside another process on the
    /// card, fails the same way when a request retries it, so serving lazily
    /// would answer every request 500 while `/health` said `ok`. Marks serve
    /// unhealthy and stops the daemon; the caller exits with the returned error.
    pub(crate) fn fail_closed_multi_slot(
        &mut self,
        meta: &Mutex<ServeMeta>,
        what: &str,
        error: anyhow::Error,
    ) -> anyhow::Error {
        set_engine_state(meta, EngineState::Down);
        self.engine.terminate();
        anyhow!(
            "multi-slot {what} failed: {error:#}; multi-slot serve has no backend without its \
             slot engine, so it fails closed instead of serving lazily"
        )
    }
}

/// How often the supervisor checks whether the daemon is still running.
pub(crate) const ENGINE_POLL: Duration = Duration::from_secs(1);
/// Respawn attempts per daemon failure, 1+2+4+8 s apart.
const ENGINE_RESPAWN_ATTEMPTS: u32 = 5;
const ENGINE_RESPAWN_BACKOFF: Duration = Duration::from_secs(1);

/// One supervisor pass: if the daemon exited, respawn it and reload the model
/// that was resident. `/health` reports `restarting` (503) until the reload
/// finishes. Skips the pass while a request or load holds the runtime; that
/// holder checks the daemon itself. Errors when the daemon cannot be
/// respawned, or when multi-slot serve cannot reload its model.
pub(crate) fn supervise_engine(shared: &ServeShared) -> Result<()> {
    let mut runtime = match shared.runtime.try_lock() {
        Ok(runtime) => runtime,
        Err(std::sync::TryLockError::WouldBlock) => return Ok(()),
        Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
    };
    if !runtime.engine.exited() {
        return Ok(());
    }
    if let Some(model) = runtime.restart_engine(&shared.meta)? {
        shared
            .meta
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .loading_model = Some(model.clone());
        let reloaded = runtime.ensure_model(&model, &shared.meta, ModelOrigin::Operator);
        {
            let mut meta = shared
                .meta
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            finish_prewarm(&mut meta, reloaded.is_ok());
        }
        match reloaded {
            Ok(_) => eprintln!("[hipfire] reloaded {model} after the daemon restart"),
            Err(error) if runtime.multi_slot_enabled => {
                return Err(runtime.fail_closed_multi_slot(
                    &shared.meta,
                    &format!("reload of {model} after the daemon restart"),
                    error,
                ));
            }
            Err(error) => {
                eprintln!("[hipfire] reload of {model} after the daemon restart failed: {error:#}")
            }
        }
    }
    set_engine_state(&shared.meta, EngineState::Up);
    Ok(())
}

/// Startup validation for the serving admission path (spec §5.2, §5.3, §9.1).
///
/// Rejects: (1) `serve.max_batch_tokens < serve.prefill_min_tokens` so a
/// nonzero prefill quantum always fits the global trunk-row budget; (2)
/// experimental multi-slot combined with `continuous_batch_size > 1`; (3)
/// `serve.max_queue == 0` when multi-slot is enabled — an uncapped queue is
/// rejected rather than reinterpreted in robust multi-slot mode.
pub(crate) fn validate_multi_slot_startup(
    multi_slot_enabled: bool,
    continuous_batch_size: u64,
    max_queue: u64,
    max_batch_tokens: u64,
    prefill_min_tokens: u64,
    prefill_chunk: u64,
    n_slots: u64,
    prefix_cache: bool,
    prefix_cache_max_bytes: u64,
) -> Result<(), String> {
    if prefill_min_tokens == 0 {
        return Err("serve.prefill_min_tokens must be at least 1".to_owned());
    }
    // The global trunk-row budget must be at least the minimum prefill quantum
    // so a nonzero prefill service quantum can always be allocated (spec §5.2,
    // §5.3). This holds regardless of multi-slot mode.
    if max_batch_tokens < prefill_min_tokens {
        return Err(format!(
            "serve.max_batch_tokens ({max_batch_tokens}) must be >= \
             serve.prefill_min_tokens ({prefill_min_tokens}); the global trunk-row \
             budget cannot be smaller than the minimum prefill quantum"
        ));
    }
    if multi_slot_enabled && prefill_min_tokens > prefill_chunk {
        return Err(format!(
            "serve.prefill_min_tokens ({prefill_min_tokens}) must be <= serve.multi_slot_prefill_chunk ({prefill_chunk}); otherwise one decode row plus the minimum prefill quantum exceeds the slot scratch"
        ));
    }
    // The engine refuses prefix_cache=true with max_bytes=0 (spec §4.5 — a
    // zero-byte pool is a guaranteed load failure, not a silently-disabled
    // cache). Catch it at startup rather than per-request HTTP 500.
    if prefix_cache && prefix_cache_max_bytes == 0 {
        return Err(
            "serve.prefix_cache requires serve.prefix_cache_max_bytes > 0".to_owned(),
        );
    }
    if multi_slot_enabled {
        let scratch_rows = prefill_chunk
            .checked_mul(n_slots)
            .ok_or_else(|| "multi-slot scratch row capacity overflow".to_owned())?;
        if max_batch_tokens > scratch_rows {
            return Err(format!(
                "serve.max_batch_tokens ({max_batch_tokens}) must be <= multi-slot scratch capacity ({scratch_rows} = {n_slots} slots x {prefill_chunk} rows)"
            ));
        }
    }
    if multi_slot_enabled {
        if continuous_batch_size > 1 {
            return Err(
                "serve.multi_slot cannot be combined with continuous_batch_size > 1; \
                 continuous batching integration is deferred"
                    .to_owned(),
            );
        }
        // The robust multi-slot mode rejects an uncapped queue rather than
        // reinterpreting zero or silently inheriting an unbounded queue (spec
        // §5.3). serve.max_queue=0 historically means uncapped; that is no
        // longer acceptable when multi-slot admission can overlap work.
        if max_queue == 0 {
            return Err(
                "serve.max_queue must be non-zero when serve.multi_slot is enabled; \
                 an uncapped queue is rejected rather than reinterpreted in robust \
                 multi-slot mode (spec §5.3)"
                    .to_owned(),
            );
        }
    }
    Ok(())
}

/// A request the client must fix. The HTTP layer answers it with 400 by
/// TYPE, never by wording: build it with [`invalid_request!`] or return it
/// with [`bail_invalid!`], and the message may say anything.
#[derive(Debug)]
pub(crate) struct InvalidRequest(pub(crate) String);

impl std::fmt::Display for InvalidRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidRequest {}

/// A model the request names that this server will not serve (404).
#[derive(Debug)]
pub(crate) struct ModelNotFound(pub(crate) String);

impl std::fmt::Display for ModelNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ModelNotFound {}

/// `anyhow::Error` carrying an [`InvalidRequest`] (HTTP 400).
macro_rules! invalid_request {
    ($($arg:tt)*) => {
        ::anyhow::Error::new($crate::serve::InvalidRequest(format!($($arg)*)))
    };
}
pub(crate) use invalid_request;

/// `return Err(invalid_request!(..))`: the `bail!` for client faults.
macro_rules! bail_invalid {
    ($($arg:tt)*) => {
        return Err($crate::serve::invalid_request!($($arg)*).into())
    };
}
pub(crate) use bail_invalid;

pub(crate) fn serve_instance_token() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut digest = Sha256::new();
    digest.update(std::process::id().to_le_bytes());
    digest.update(now.to_le_bytes());
    format!("{:x}", digest.finalize())
}

pub(crate) fn proc_start_time(pid: u32) -> Option<u64> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_comm = stat.rsplit_once(") ")?.1;
    after_comm.split_whitespace().nth(19)?.parse().ok()
}

pub(crate) fn pid_owns_listen_port(pid: u32, port: u16) -> Option<bool> {
    let mut listen_inodes = BTreeSet::new();
    let port_hex = format!("{port:04X}");
    let mut read_any = false;
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let Ok(raw) = fs::read_to_string(table) else {
            continue;
        };
        read_any = true;
        for line in raw.lines().skip(1) {
            let columns = line.split_whitespace().collect::<Vec<_>>();
            if columns.len() < 10 || columns[3] != "0A" {
                continue;
            }
            let Some((_, local_port)) = columns[1].rsplit_once(':') else {
                continue;
            };
            if local_port.eq_ignore_ascii_case(&port_hex) {
                listen_inodes.insert(columns[9].to_owned());
            }
        }
    }
    if !read_any {
        return None;
    }
    if listen_inodes.is_empty() {
        return Some(false);
    }
    let entries = fs::read_dir(format!("/proc/{pid}/fd")).ok()?;
    for entry in entries.flatten() {
        let Ok(target) = fs::read_link(entry.path()) else {
            continue;
        };
        let target = target.to_string_lossy();
        if let Some(inode) = target
            .strip_prefix("socket:[")
            .and_then(|value| value.strip_suffix(']'))
        {
            if listen_inodes.contains(inode) {
                return Some(true);
            }
        }
    }
    Some(false)
}

pub(crate) fn validate_serve_pid(
    record: &ServePidRecord,
    host: &str,
    fallback_port: u16,
) -> Result<()> {
    let proc_dir = PathBuf::from(format!("/proc/{}", record.pid));
    if !proc_dir.is_dir() {
        bail!("tracked serve PID {} is no longer alive", record.pid);
    }
    let cmdline = fs::read(proc_dir.join("cmdline")).unwrap_or_default();
    let cmdline = String::from_utf8_lossy(&cmdline).replace('\0', " ");
    if !cmdline.contains("hipfire") || !cmdline.contains("serve") {
        bail!("PID {} is not a hipfire serve process", record.pid);
    }
    if let Some(expected) = record.start_time {
        if proc_start_time(record.pid) != Some(expected) {
            bail!("PID {} was reused after serve.pid was written", record.pid);
        }
    }

    let port = record.port.unwrap_or(fallback_port);
    let owns_port = pid_owns_listen_port(record.pid, port);
    if owns_port == Some(false) {
        bail!(
            "PID {} does not own the tracked serve port {port}",
            record.pid
        );
    }
    let health_matches = record.token.as_deref().is_some_and(|expected| {
        http_get_json(host, port, "/health").is_some_and(|health| {
            health.get("pid").and_then(serde_json::Value::as_u64) == Some(record.pid as u64)
                && health.get("token").and_then(serde_json::Value::as_str) == Some(expected)
        })
    });
    if owns_port == Some(true) || health_matches || record.legacy && owns_port.is_none() {
        Ok(())
    } else {
        bail!(
            "could not prove ownership of PID {} with port or health token",
            record.pid
        )
    }
}

pub(crate) fn stop_command(paths: &Paths, args: StopArgs) -> Result<()> {
    let pid_path = paths.root.join("serve.pid");
    match fs::read_to_string(&pid_path) {
        Ok(raw) => {
            let record = parse_pid_record(&raw)
                .ok_or_else(|| anyhow!("invalid serve.pid; refusing to signal"))?;
            let resolved = resolved_global(paths, true)
                .ok()
                .map(|(_, resolved)| resolved);
            let host = resolved
                .as_ref()
                .and_then(|resolved| config_string(resolved, "serve.host").ok())
                .unwrap_or_else(|| "127.0.0.1".into());
            let fallback_port = args
                .port
                .or_else(|| {
                    resolved.as_ref().and_then(|resolved| {
                        config_u64(resolved, "serve.port")
                            .ok()
                            .and_then(|port| u16::try_from(port).ok())
                    })
                })
                .unwrap_or(11435);
            if let Err(error) = validate_serve_pid(&record, probe_host(&host), fallback_port) {
                fs::remove_file(&pid_path)?;
                if !args.force {
                    bail!("{error}; removed stale pidfile without signaling");
                }
                eprintln!(
                    "warning: {error}; refusing direct PID signal and continuing forced reap"
                );
            } else {
                let status = Command::new("kill")
                    .arg("-TERM")
                    .arg(record.pid.to_string())
                    .status()
                    .context("failed to invoke kill")?;
                if !status.success() {
                    bail!("failed to stop native serve PID {}", record.pid);
                }
                for _ in 0..50 {
                    if !Path::new(&format!("/proc/{}", record.pid)).exists() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                let _ = fs::remove_file(&pid_path);
                println!("hipfire serve stopped (PID {})", record.pid);
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            println!("hipfire serve is not running");
        }
        Err(error) => return Err(error).context("failed to read serve.pid"),
    }
    if args.force || args.all {
        let (_, resolved) = resolved_global(paths, true)?;
        let port = args
            .port
            .unwrap_or(config_u64(&resolved, "serve.port")? as u16);
        let _ = Command::new("pkill").args(["-x", "daemon"]).status();
        if args.all {
            let _ = Command::new("pkill")
                .args(["-f", "target/release/hipfire-quantize"])
                .status();
        }
        let _ = Command::new("fuser")
            .args(["-k", &format!("{port}/tcp")])
            .status();
        println!("reaped orphan daemon processes and freed port {port}");
    }
    Ok(())
}

pub(crate) fn parse_pid_record(raw: &str) -> Option<ServePidRecord> {
    if let Ok(pid) = raw.trim().parse() {
        return Some(ServePidRecord {
            pid,
            start_time: None,
            port: None,
            token: None,
            legacy: true,
        });
    }
    let mut record = serde_json::from_str::<ServePidRecord>(raw).ok()?;
    record.legacy = record.start_time.is_none() && record.port.is_none() && record.token.is_none();
    Some(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Commands, Paths, ServeArgs, StopArgs};
    use clap::Parser;
    use hipfire_config::{
        load_global, resolve, ConfigLayer, ConfigPaths, ConfigSource, NamedLayer,
    };
    use hipfire_registry::{RegistryPaths, RegistryV1};
    use std::{
        env, fs,
        path::{Path, PathBuf},
        sync::{mpsc, Arc, Mutex},
        thread,
        time::{Duration, Instant},
    };
    fn test_paths(label: &str) -> Paths {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hipfire-cli-{label}-{}-{nonce}",
            std::process::id()
        ));
        let config = ConfigPaths::under(&root);
        Paths {
            models: config.models.clone(),
            registry: RegistryPaths {
                cache: root.join("registry.cache.json"),
            },
            root,
            config,
        }
    }

    fn idle_test_meta() -> ServeMeta {
        ServeMeta {
            current_model: Some("model.hfq".to_owned()),
            loading_model: Some("model.hfq".to_owned()),
            last_activity: Instant::now() - Duration::from_secs(600),
            ..ServeMeta::new("test".to_owned())
        }
    }

    #[test]
    fn idle_timeout_does_not_evict_a_loading_model() {
        let meta = idle_test_meta();
        assert!(!idle_model_expired(&meta, Duration::from_secs(300)));
    }

    #[test]
    fn successful_prewarm_starts_a_fresh_idle_window() {
        let mut meta = idle_test_meta();
        finish_prewarm(&mut meta, true);
        assert!(meta.loading_model.is_none());
        assert!(!idle_model_expired(&meta, Duration::from_secs(300)));
        assert!(meta.last_activity.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn bind_and_pid_compatibility_parsers_cover_legacy_shapes() {
        assert_eq!(
            parse_bind(Some("127.0.0.1:12000"), None, "0.0.0.0", 11435).unwrap(),
            ("127.0.0.1".into(), 12000)
        );
        assert_eq!(
            parse_bind(Some("[::1]:12001"), None, "0.0.0.0", 11435).unwrap(),
            ("::1".into(), 12001)
        );
        let legacy = parse_pid_record("42\n").unwrap();
        assert_eq!(legacy.pid, 42);
        assert!(legacy.legacy);
        let json = parse_pid_record(r#"{"pid":43,"token":"old"}"#).unwrap();
        assert_eq!(json.pid, 43);
        assert_eq!(json.token.as_deref(), Some("old"));
        assert!(!json.legacy);
    }

    #[test]
    fn serve_accepts_legacy_positionals_and_native_overrides() {
        let parsed = Cli::try_parse_from([
            "hipfire",
            "serve",
            "qwen3.6:35b-a3b-mq4r",
            "127.0.0.1",
            "11520",
            "--kv-mode",
            "q8",
            "--kv-backend",
            "vmm",
            "--idle-timeout",
            "0",
            "--tp",
            "2",
        ])
        .unwrap();
        let Some(Commands::Serve(args)) = parsed.command else {
            panic!("expected serve command")
        };
        assert_eq!(args.positionals.len(), 3);
        assert_eq!(args.kv_mode.as_deref(), Some("q8"));
        assert_eq!(args.kv_backend.as_deref(), Some("vmm"));
        assert_eq!(args.idle_timeout, Some(0));
        assert_eq!(args.tp, Some(2));
    }

    #[test]
    fn admission_queue_is_bounded_and_times_out() {
        let admission = Arc::new(Admission::new(1, Duration::from_millis(200)));
        let holder = admission.acquire().unwrap();
        let queued_admission = Arc::clone(&admission);
        let (sender, receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let guard = queued_admission.acquire().unwrap();
            sender.send(()).unwrap();
            drop(guard);
        });
        for _ in 0..100 {
            if admission.inflight() == 2 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        let saturated = admission.acquire().unwrap_err();
        assert!(saturated.message.contains("queue full"));
        drop(holder);
        receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();

        let admission = Arc::new(Admission::new(1, Duration::from_millis(5)));
        let _holder = admission.acquire().unwrap();
        let timeout = admission.acquire().unwrap_err();
        assert!(timeout.message.contains("wait exceeded"));
        assert_eq!(admission.inflight(), 1);
    }

    #[test]
    fn admission_retry_after_stays_short_for_long_queue_waits() {
        let secs = |timeout| Admission::new(1, timeout).retry_after_seconds();
        // The default 10-minute wait must not tell clients to back off 10 minutes.
        assert_eq!(secs(Duration::from_secs(600)), 30);
        assert_eq!(secs(Duration::from_secs(5)), 5);
        // Unbounded wait (0) and sub-second waits still send a valid header.
        assert_eq!(secs(Duration::ZERO), 1);
        assert_eq!(secs(Duration::from_millis(5)), 1);
    }

    #[test]
    fn async_admission_cancellation_removes_waiter() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            let admission = Arc::new(Admission::new(1, Duration::from_secs(5)));
            let holder = admission.acquire().unwrap();
            let cancel = CancellationToken::new();
            let waiter_admission = Arc::clone(&admission);
            let waiter_cancel = cancel.clone();
            let waiter =
                tokio::spawn(async move { waiter_admission.acquire_async(waiter_cancel).await });
            while admission.inflight() != 2 {
                tokio::task::yield_now().await;
            }
            cancel.cancel();
            let error = tokio::time::timeout(Duration::from_millis(500), waiter)
                .await
                .expect("cancelled waiter completes")
                .expect("waiter task")
                .expect_err("cancelled admission fails");
            assert!(error.message.contains("cancelled"));
            assert_eq!(admission.inflight(), 1);
            drop(holder);
            assert_eq!(admission.inflight(), 0);
        });
    }

    #[test]
    fn dropping_async_admission_future_removes_waiter() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            let admission = Arc::new(Admission::new(1, Duration::from_secs(5)));
            let holder = admission.acquire().unwrap();
            let waiter_admission = Arc::clone(&admission);
            let waiter = tokio::spawn(async move {
                waiter_admission
                    .acquire_async(CancellationToken::new())
                    .await
            });
            while admission.inflight() != 2 {
                tokio::task::yield_now().await;
            }
            waiter.abort();
            let error = waiter.await.expect_err("aborted waiter task");
            assert!(error.is_cancelled());
            assert_eq!(
                admission.inflight(),
                1,
                "dropped admission future left a phantom queued request"
            );
            drop(holder);
            assert_eq!(admission.inflight(), 0);
        });
    }

    #[test]
    fn async_admission_observes_guard_release_without_lost_wake() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            let admission = Arc::new(Admission::new(1, Duration::from_secs(5)));
            let holder = admission.acquire().unwrap();
            let waiter_admission = Arc::clone(&admission);
            let waiter = tokio::spawn(async move {
                waiter_admission
                    .acquire_async(CancellationToken::new())
                    .await
            });
            while admission.inflight() != 2 {
                tokio::task::yield_now().await;
            }
            drop(holder);
            let admitted = tokio::time::timeout(Duration::from_millis(500), waiter)
                .await
                .expect("notified waiter completes")
                .expect("waiter task")
                .expect("waiter admitted");
            assert_eq!(admission.inflight(), 1);
            drop(admitted);
            assert_eq!(admission.inflight(), 0);
        });
    }

    #[test]
    fn admission_eligible_concurrent_up_to_capacity() {
        let admission = Arc::new(Admission::new_with_capacity(8, Duration::from_secs(1), 2));
        let g1 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        let g2 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        assert_eq!(admission.inflight(), 2);
        // Third eligible same model should queue, then timeout quickly.
        let admission2 = Arc::clone(&admission);
        let handle =
            thread::spawn(move || admission2.acquire_for(true, Some("qwen3.5:7b")).unwrap());
        thread::sleep(Duration::from_millis(50));
        assert_eq!(admission.inflight(), 3); // 2 held + 1 queued
        drop(g1);
        let g3 = handle.join().unwrap();
        assert_eq!(admission.inflight(), 2);
        drop(g2);
        drop(g3);
        assert_eq!(admission.inflight(), 0);
    }

    #[test]
    fn admission_ineligible_is_exclusive() {
        let admission = Arc::new(Admission::new_with_capacity(8, Duration::from_secs(1), 2));
        let g1 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        // Ineligible must wait for eligible to finish, even though capacity not full.
        let admission2 = Arc::clone(&admission);
        let handle = thread::spawn(move || admission2.acquire().unwrap());
        thread::sleep(Duration::from_millis(50));
        assert!(admission.inflight() == 2); // 1 eligible + 1 queued ineligible
        drop(g1);
        let g_inelig = handle.join().unwrap();
        assert_eq!(admission.inflight(), 1);
        // While ineligible holds, eligible must wait.
        let admission3 = Arc::clone(&admission);
        let handle2 =
            thread::spawn(move || admission3.acquire_for(true, Some("qwen3.5:7b")).unwrap());
        thread::sleep(Duration::from_millis(50));
        assert_eq!(admission.inflight(), 2);
        drop(g_inelig);
        let g2 = handle2.join().unwrap();
        assert_eq!(admission.inflight(), 1);
        drop(g2);
    }

    #[test]
    fn admission_model_lease_prevents_cross_model_batch() {
        let admission = Arc::new(Admission::new_with_capacity(8, Duration::from_secs(1), 2));
        let g1 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        // Different model cannot share batch lanes, must wait for exclusive.
        let admission2 = Arc::clone(&admission);
        let handle =
            thread::spawn(move || admission2.acquire_for(true, Some("qwen3.5:14b")).unwrap());
        thread::sleep(Duration::from_millis(50));
        assert_eq!(admission.inflight(), 2); // 1 held + 1 queued due to model mismatch
        drop(g1);
        let g2 = handle.join().unwrap();
        assert_eq!(admission.inflight(), 1);
        drop(g2);
    }

    #[test]
    fn batch_eligibility_conservative_checks() {
        // Eligible: tp=1 qwen with one plain user string.
        let body =
            serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":"hi"}]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Eligible: absent messages (prompt path) for Qwen.
        let body = serde_json::json!({"model":"qwen3.5:7b","prompt":"hi"});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Eligible: empty messages array for Qwen.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Tools disqualify.
        let body = serde_json::json!({"model":"qwen3.5:7b","tools":[{"type":"function","function":{"name":"x"}}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Image / multipart content disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:"}}]}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Multipart text-only array content also disqualifies (must be plain string).
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // system+user history disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[
            {"role":"system","content":"be brief"},
            {"role":"user","content":"hi"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // user+assistant multi-turn disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":"hello"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Tool-call content on the sole message disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[{
            "role":"user",
            "content":"hi",
            "tool_calls":[{"id":"c0","type":"function","function":{"name":"x","arguments":"{}"}}]
        }]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Qwen tp=4 pure EP is eligible when daemon admits batch.
        let body =
            serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":"hi"}]});
        assert!(is_batch_eligible_request(
            &body,
            Some(4),
            Some("qwen35"),
            true
        ));
        // Qwen tp=2 (and any non-1/non-4) disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b"});
        assert!(!is_batch_eligible_request(
            &body,
            Some(2),
            Some("qwen35"),
            true
        ));
        // Non-qwen disqualifies.
        let body = serde_json::json!({"model":"deepseek4:671b"});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("deepseek4"),
            true
        ));
        // Daemon load says batch incapable: HTTP admission must not invent it.
        let body =
            serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":"hi"}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            false
        ));
        // Eligible: tp=1 LFM2 dense with one plain user string when daemon admits batch.
        let body =
            serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":"hi"}]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // Eligible: absent messages for LFM.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","prompt":"hi"});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // Eligible: empty messages for LFM.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects system+user the same way.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[
            {"role":"system","content":"be brief"},
            {"role":"user","content":"hi"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects user+assistant.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":"hello"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects tool-call content.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[{
            "role":"user",
            "content":"hi",
            "tool_calls":[{"id":"c0","type":"function","function":{"name":"x","arguments":"{}"}}]
        }]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects image/multipart content.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":[
            {"type":"text","text":"describe"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj"}}
        ]}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM tp=4 disqualifies (dense remains tp=1 only).
        let body =
            serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":"hi"}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(4),
            Some("lfm2"),
            true
        ));
        // Daemon load says batch incapable: LFM HTTP admission must not invent it.
        let body =
            serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":"hi"}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            false
        ));
    }

    #[test]
    fn batch_messages_shape_matches_daemon_contract() {
        // Absent / empty → eligible shape.
        assert!(batch_messages_are_single_user(&serde_json::json!({})));
        assert!(batch_messages_are_single_user(
            &serde_json::json!({"messages":[]})
        ));
        // Exactly one plain user string → eligible.
        assert!(batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"user","content":"hi"}]
        })));
        // system+user, user+assistant → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[
                {"role":"system","content":"sys"},
                {"role":"user","content":"hi"}
            ]
        })));
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":"yo"}
            ]
        })));
        // Non-user sole role → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"system","content":"sys"}]
        })));
        // tool_calls payload → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{
                "role":"user",
                "content":"hi",
                "tool_calls":[{"id":"c0"}]
            }]
        })));
        // Multipart / image array content → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]
        })));
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"data:"}}
            ]}]
        })));
        // Non-array messages → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":"not-an-array"
        })));
    }

    #[test]
    fn qwen_mq4r_decode_prewarm_is_fail_closed_to_the_exact_route() {
        let qwen = serde_json::json!({ "arch": "qwen3_5_moe" });
        let qwen_dense = serde_json::json!({ "arch": "qwen3_5" });
        let deepseek = serde_json::json!({ "arch": "deepseek4" });
        assert!(should_prewarm_qwen_mq4r_decode(
            Path::new("qwen3.6-35b-a3b.mq4r"),
            &qwen,
            None,
        ));
        assert!(should_prewarm_qwen_mq4r_decode(
            Path::new("QWEN3.6-9B.MQ4R"),
            &qwen_dense,
            Some(1),
        ));
        assert!(!should_prewarm_qwen_mq4r_decode(
            Path::new("qwen3.6-35b-a3b.mq4"),
            &qwen,
            None,
        ));
        assert!(!should_prewarm_qwen_mq4r_decode(
            Path::new("deepseek-v4-flash.mq4r"),
            &deepseek,
            None,
        ));
        assert!(!should_prewarm_qwen_mq4r_decode(
            Path::new("qwen3.6-35b-a3b.mq4r"),
            &qwen,
            Some(2),
        ));
    }

    #[test]
    fn reasoning_contract_handshake_parsing() {
        use saddle_core::caps::ReasoningContract;
        assert_eq!(
            ReasoningContract::from_wire_name("qwen_jinja"),
            Some(ReasoningContract::QwenJinja)
        );
        assert_eq!(
            ReasoningContract::from_wire_name("deepseek4"),
            Some(ReasoningContract::DeepSeek4)
        );
        assert_eq!(
            ReasoningContract::from_wire_name("gemma_boolean"),
            Some(ReasoningContract::GemmaBoolean)
        );
        assert_eq!(
            ReasoningContract::from_wire_name("muse_glimmer"),
            Some(ReasoningContract::MuseGlimmer)
        );
        assert_eq!(
            ReasoningContract::from_wire_name("unsupported"),
            Some(ReasoningContract::Unsupported)
        );
        assert_eq!(ReasoningContract::from_wire_name("unknown"), None);
        assert_eq!(
            ReasoningContract::from_wire_name("").unwrap_or(ReasoningContract::Unsupported),
            ReasoningContract::Unsupported
        );
        let parsed =
            ReasoningContract::from_wire_name("bogus").unwrap_or(ReasoningContract::Unsupported);
        assert_eq!(parsed, ReasoningContract::Unsupported);
        for contract in [
            ReasoningContract::Unsupported,
            ReasoningContract::QwenJinja,
            ReasoningContract::DeepSeek4,
            ReasoningContract::GemmaBoolean,
            ReasoningContract::MuseGlimmer,
        ] {
            let wire = contract.wire_name();
            assert_eq!(ReasoningContract::from_wire_name(wire), Some(contract));
        }
        let runtime_default = ReasoningContract::Unsupported;
        assert_eq!(runtime_default, ReasoningContract::Unsupported);
        let loaded = serde_json::json!({
            "reasoning_effort_native": true,
            "reasoning_efforts": ["low", "medium", "xhigh"]
        });
        assert_eq!(
            loaded
                .get("reasoning_effort_native")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            true
        );
        let efforts: Vec<String> = loaded
            .get("reasoning_efforts")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_owned()))
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(
            efforts,
            vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()]
        );
        let empty_loaded = serde_json::json!({});
        assert_eq!(
            empty_loaded
                .get("reasoning_effort_native")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            false
        );
        let empty_efforts: Vec<String> = empty_loaded
            .get("reasoning_efforts")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_owned()))
                    .collect()
            })
            .unwrap_or_default();
        assert!(empty_efforts.is_empty());
    }

    #[test]
    fn multi_slot_startup_rejects_continuous_batch_gt_one() {
        // Args: (multi_slot, continuous_batch_size, max_queue, max_batch_tokens, prefill_min_tokens)
        assert!(validate_multi_slot_startup(false, 1, 64, 4096, 1, 1024, 4, false, 0).is_ok());
        assert!(validate_multi_slot_startup(false, 8, 64, 4096, 1, 1024, 4, false, 0).is_ok());
        assert!(validate_multi_slot_startup(true, 1, 64, 4096, 1, 1024, 4, false, 0).is_ok());
        let err = validate_multi_slot_startup(true, 2, 64, 4096, 1, 1024, 4, false, 0).unwrap_err();
        assert!(err.contains("continuous_batch_size > 1"), "{err}");
        assert!(err.contains("deferred"), "{err}");
        let err = validate_multi_slot_startup(true, 16, 64, 4096, 1, 1024, 4, false, 0).unwrap_err();
        assert!(err.contains("serve.multi_slot"), "{err}");
    }

    #[test]
    fn multi_slot_startup_rejects_uncapped_queue() {
        // serve.max_queue=0 is uncapped historically; robust multi-slot rejects it.
        let err = validate_multi_slot_startup(true, 1, 0, 4096, 1, 1024, 4, false, 0).unwrap_err();
        assert!(err.contains("serve.max_queue"), "{err}");
        assert!(err.contains("non-zero"), "{err}");
        // Off multi-slot, an uncapped queue is still accepted (old behavior preserved).
        assert!(validate_multi_slot_startup(false, 1, 0, 4096, 1, 1024, 4, false, 0).is_ok());
    }

    #[test]
    fn multi_slot_startup_rejects_budget_below_prefill_min() {
        // max_batch_tokens < prefill_min_tokens is rejected regardless of multi-slot.
        let err = validate_multi_slot_startup(false, 1, 64, 0, 1, 1024, 4, false, 0).unwrap_err();
        assert!(err.contains("serve.max_batch_tokens"), "{err}");
        assert!(err.contains("serve.prefill_min_tokens"), "{err}");
        let err = validate_multi_slot_startup(true, 1, 64, 1, 2, 1024, 4, false, 0).unwrap_err();
        assert!(err.contains("serve.max_batch_tokens"), "{err}");
        // Equal values are accepted.
        assert!(validate_multi_slot_startup(true, 1, 64, 4, 4, 1024, 4, false, 0).is_ok());
    }

    #[test]
    fn multi_slot_startup_rejects_prefill_min_above_chunk() {
        let err = validate_multi_slot_startup(true, 1, 64, 2048, 2048, 1024, 2, false, 0).unwrap_err();
        assert!(
            err.contains("must be <= serve.multi_slot_prefill_chunk"),
            "{err}"
        );
    }

    #[test]
    fn multi_slot_startup_rejects_budget_above_scratch_rows() {
        let err = validate_multi_slot_startup(true, 1, 64, 2049, 1, 1024, 2, false, 0).unwrap_err();
        assert!(err.contains("scratch capacity (2048"), "{err}");
    }

    #[test]
    fn multi_slot_startup_rejects_zero_prefill_minimum() {
        let err = validate_multi_slot_startup(true, 1, 64, 2048, 0, 1024, 2, false, 0).unwrap_err();
        assert!(err.contains("at least 1"), "{err}");
    }

    // ---- Queue byte budget (spec §5.3) ----

    #[test]
    fn queue_byte_budget_rejects_when_bytes_exceed_the_cap() {
        // max_queue_bytes = 100; a request with 60 bytes fits, but two don't.
        let admission = Arc::new(Admission::new_with_capacity_and_bytes(
            10,
            Duration::from_secs(5),
            1,
            100,
        ));
        let _holder = admission.acquire().unwrap();
        // Second request with 60 bytes queues (0 + 60 ≤ 100).
        let adm2 = Arc::clone(&admission);
        let handle =
            thread::spawn(move || adm2.acquire_for_with_bytes(true, Some("m"), 60).unwrap());
        // Wait for it to queue.
        for _ in 0..100 {
            if admission.queued_bytes() > 0 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(admission.queued_bytes(), 60);
        // Now a third request with 60 bytes: 60 + 60 = 120 > 100 → reject.
        let err = admission
            .acquire_for_with_bytes(true, Some("m"), 60)
            .unwrap_err();
        assert!(err.message.contains("byte budget"), "{}", err.message);
        drop(_holder);
        drop(handle.join().unwrap());
    }

    #[test]
    fn queue_byte_budget_zero_disables_byte_cap() {
        // max_queue_bytes = 0 means no byte cap (old behavior preserved
        // when multi-slot is off).
        let admission = Arc::new(Admission::new_with_capacity_and_bytes(
            2,
            Duration::from_millis(10),
            1,
            0,
        ));
        let _holder = admission.acquire().unwrap();
        // Large bytes should not be rejected by the byte budget.
        let err = admission
            .acquire_for_with_bytes(true, Some("m"), 999_999_999)
            .unwrap_err();
        // Should be a timeout (queue wait), not a QueueFull byte error.
        assert!(err.message.contains("queue wait exceeded"), "{}", err.message);
    }

    // ---- Permit released exactly once (spec §5.3) ----

    #[test]
    fn permit_released_exactly_once_on_normal_path() {
        let admission = Arc::new(Admission::new_with_capacity_and_bytes(
            4,
            Duration::from_secs(5),
            2,
            1024,
        ));
        let guard = admission
            .acquire_for_with_bytes(true, Some("m"), 100)
            .unwrap();
        assert_eq!(admission.inflight(), 1);
        assert_eq!(admission.queued_bytes(), 0); // bytes released from queue on admit
        drop(guard);
        assert_eq!(admission.inflight(), 0);
    }

    #[test]
    fn permit_released_exactly_once_on_cancel_path() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            let admission = Arc::new(Admission::new_with_capacity_and_bytes(
                4,
                Duration::from_secs(5),
                1,
                1024,
            ));
            let _holder = admission.acquire().unwrap();
            let cancel = CancellationToken::new();
            let adm = Arc::clone(&admission);
            let waiter_cancel = cancel.clone();
            let waiter = tokio::spawn(async move {
                adm.acquire_for_async_with_bytes(true, Some("m"), 200, waiter_cancel)
                    .await
            });
            // Wait for the waiter to queue.
            for _ in 0..100 {
                if admission.queued_bytes() == 200 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(admission.queued_bytes(), 200);
            cancel.cancel();
            let err = tokio::time::timeout(Duration::from_millis(500), waiter)
                .await
                .expect("cancelled waiter completes")
                .expect("waiter task")
                .expect_err("cancelled admission fails");
            assert_eq!(err.message, "cancelled");
            // Queued bytes must be released by AdmissionWaiter::drop.
            assert_eq!(admission.queued_bytes(), 0);
            assert_eq!(admission.inflight(), 1); // only the holder
        });
    }

    #[test]
    fn permit_released_exactly_once_on_timeout_path() {
        let admission = Arc::new(Admission::new_with_capacity_and_bytes(
            4,
            Duration::from_millis(10),
            1,
            1024,
        ));
        let _holder = admission.acquire().unwrap();
        let err = admission
            .acquire_for_with_bytes(true, Some("m"), 300)
            .unwrap_err();
        assert!(err.message.contains("queue wait exceeded"), "{}", err.message);
        // Queued bytes must be released on timeout.
        assert_eq!(admission.queued_bytes(), 0);
        assert_eq!(admission.inflight(), 1);
    }

    #[test]
    fn admission_guard_carries_queue_bytes() {
        let admission = Arc::new(Admission::new_with_capacity_and_bytes(
            4,
            Duration::from_secs(5),
            1,
            1024,
        ));
        // Fast path: no queueing, queue_bytes = 0.
        let guard = admission
            .acquire_for_with_bytes(true, Some("m"), 500)
            .unwrap();
        assert_eq!(guard.queue_bytes(), 0);
        assert!(guard.is_eligible());
        drop(guard);

        // Queued path: queue_bytes = charged bytes.
        let holder = admission
            .acquire_for_with_bytes(true, Some("m"), 0)
            .unwrap();
        let adm2 = Arc::clone(&admission);
        let handle =
            thread::spawn(move || adm2.acquire_for_with_bytes(true, Some("m"), 400).unwrap());
        // Wait for queue, then release holder so the waiter acquires.
        for _ in 0..100 {
            if admission.queued_bytes() == 400 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        drop(holder);
        let guard2 = handle.join().unwrap();
        assert_eq!(guard2.queue_bytes(), 400);
        drop(guard2);
        assert_eq!(admission.queued_bytes(), 0);
    }

    // ---- Admission errors ----

    #[test]
    fn queue_full_error_names_the_full_queue() {
        // max_queue=1: one holder + one queued = full. Third gets QueueFull.
        let admission = Arc::new(Admission::new_with_capacity_and_bytes(
            1,
            Duration::from_secs(5),
            1,
            0,
        ));
        let _holder = admission.acquire().unwrap();
        // Spawn a waiter that fills the queue slot.
        let adm2 = Arc::clone(&admission);
        let handle = thread::spawn(move || {
            let _g = adm2.acquire().unwrap();
        });
        // Wait for the waiter to queue.
        for _ in 0..100 {
            if admission.inflight() == 2 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        // Now the queue is full (1 holder + 1 queued). Third gets QueueFull.
        let err = admission.acquire().unwrap_err();
        assert!(err.message.contains("serve queue full"), "{}", err.message);
        drop(_holder);
        let _ = handle.join();
    }

    #[test]
    fn queue_timeout_error_names_the_wait() {
        let admission = Arc::new(Admission::new_with_capacity_and_bytes(
            1,
            Duration::from_millis(5),
            1,
            0,
        ));
        let _holder = admission.acquire().unwrap();
        let err = admission.acquire().unwrap_err();
        assert!(err.message.contains("queue wait exceeded"), "{}", err.message);
    }

    #[test]
    fn cancelled_admission_reports_cancelled() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            let admission = Arc::new(Admission::new(1, Duration::from_secs(5)));
            let _holder = admission.acquire().unwrap();
            let cancel = CancellationToken::new();
            cancel.cancel();
            let err = admission.acquire_async(cancel).await.unwrap_err();
            assert_eq!(err.message, "cancelled");
        });
    }
}

#[cfg(test)]
mod capabilities_tests {
    use super::*;

    fn route() -> serde_json::Value {
        route_capabilities(
            /*multi_slot*/ true,
            2,
            50000,
            1024,
            /*prefix_cache*/ true,
            536870912,
            /*jump_forward*/ true,
            8192,
            1,
            64,
            268435456,
            30000,
            Some(30_000),
            64 << 20,
        )
    }

    /// The multi-slot route advertises every capability a client needs to
    /// discover before using it (spec §9.1): slots, cache, structured
    /// output subset, queue bounds, and the honest refusal list.
    #[test]
    fn multi_slot_route_advertises_capabilities() {
        let caps = route();
        assert_eq!(caps["mode"], "multi-slot");
        assert_eq!(caps["multi_slot"], true);
        assert_eq!(caps["multi_slot_slots"], 2);
        assert_eq!(caps["multi_slot_ctx"], 50000);
        assert_eq!(caps["prefix_cache"], true);
        assert_eq!(caps["prefix_cache_max_bytes"], 536870912);
        assert_eq!(caps["structured_output"], true);
        assert_eq!(caps["structured_output_subset"], "json-schema-strict-v1");
        assert_eq!(caps["structured_jump_forward"], true);
        assert_eq!(caps["max_batch_tokens"], 8192);
        assert_eq!(caps["max_queue"], 64);
        assert_eq!(caps["queue_timeout_ms"], 30000);
        assert_eq!(caps["openai_compatible"], true);
        let refused = caps["refused_request_fields"].as_array().unwrap();
        assert!(!refused.iter().any(|v| v == "tools"), "tools are supported");
        for field in [
            "stop",
            "logprobs",
            "top_logprobs",
            "n",
            "best_of",
            "logit_bias",
            "echo",
            "suffix",
            "reasoning_effort",
            "response_format:json_object",
            "tools+image",
        ] {
            assert!(
                refused.iter().any(|v| v == field),
                "refusal list must contain {field}"
            );
        }
    }

    /// The standard route must NOT advertise multi-slot/structured-output
    /// capabilities it does not have — advertisement is honest, not
    /// aspirational (spec §6 X2: no silent capability inflation).
    #[test]
    fn standard_route_advertises_honest_absence() {
        let caps = route_capabilities(
            false,
            4,
            8192,
            1024,
            false,
            0,
            false,
            4096,
            1,
            64,
            268435456,
            30000,
            None,
            64 << 20,
        );
        assert_eq!(caps["mode"], "standard");
        assert_eq!(caps["multi_slot"], false);
        assert_eq!(caps["structured_output"], false);
        assert!(caps["structured_output_subset"].is_null());
        assert_eq!(caps["prefix_cache"], false);
        let refused = caps["refused_request_fields"].as_array().unwrap();
        assert!(refused.is_empty(), "the standard route refuses none of these fields");
        assert!(caps["stream_stall_timeout_ms"].is_null());
    }
}
