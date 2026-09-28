// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `POST /v1/systemone` — Jev-compatible decide. The daemon is the single
//! validation authority; this layer selects the model, serialises against
//! chat traffic, and shapes Jev's response. Spec §6.

use super::http::{json_response, openai_error, request_id, BoxBody};
use super::{AdmissionGuard, ServeShared};
use hyper::{header, Response};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn is_local_model(runtime: &super::ServeRuntime, model: &str) -> bool {
    crate::registry_entry_for_path(&runtime.paths, &runtime.registry, model).is_some()
        || crate::find_model_path(&runtime.paths, &runtime.registry, model).is_some()
}

/// Outcome of the synchronous decide attempt loop, handed back to the async
/// caller over a oneshot channel. Plain data (not `Response<BoxBody>`) so the
/// blocking side never has to reason about HTTP framing.
enum DecideOutcome {
    Ok {
        model: String,
        answers: serde_json::Value,
        usage: serde_json::Value,
        timing: serde_json::Value,
    },
    Err {
        status: u16,
        message: String,
        required_max_seq: Option<u64>,
    },
}

/// Same `{error: {message, type}}` shape as `openai_error`, plus
/// `required_max_seq` inside the error object when the daemon supplied one —
/// a Jev client needs it to decide whether retrying with more context is
/// even possible.
fn decide_error_response(
    message: &str,
    status: u16,
    required_max_seq: Option<u64>,
) -> Response<BoxBody> {
    let error_type = if (400..500).contains(&status) {
        "invalid_request_error"
    } else {
        "server_error"
    };
    let mut error = serde_json::json!({ "message": message, "type": error_type });
    if let Some(seq) = required_max_seq {
        error["required_max_seq"] = serde_json::json!(seq);
    }
    json_response(serde_json::json!({ "error": error }), status)
}

/// The whole decide attempt loop, synchronous end to end: model resolution,
/// `ensure_model` (which can block on a GPU load), the daemon round trip via
/// `Engine::request` (a blocking `mpsc::Receiver::recv`), and the one-shot
/// `required_max_seq` reload-and-retry. Must run entirely off the tokio
/// executor (inside `spawn_blocking`) and must own `guard` for its whole
/// duration: the admission slot has to stay held until the daemon call(s)
/// actually finish, not merely until hyper drops the handler future on a
/// client disconnect. `_guard` is intentionally unused past being held.
fn run_decide(
    shared: &ServeShared,
    body: serde_json::Value,
    _guard: AdmissionGuard,
    cancelled: &AtomicBool,
) -> DecideOutcome {
    let requested = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    for attempt in 0..2 {
        let (engine, model_echo) = {
            let mut runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
            let target = if !requested.is_empty() && is_local_model(&runtime, &requested) {
                requested.clone()
            } else {
                match runtime.current_path.as_ref() {
                    Some(p) => p.display().to_string(),
                    None => {
                        return DecideOutcome::Err {
                            status: 503,
                            message: "no model loaded: name a local hipfire model in `model`, \
                                      or load one first"
                                .to_string(),
                            required_max_seq: None,
                        }
                    }
                }
            };
            if let Err(e) = runtime.ensure_model(&target, &shared.meta, None) {
                return DecideOutcome::Err {
                    status: 500,
                    message: format!("{e:#}"),
                    required_max_seq: None,
                };
            }
            let echo = runtime
                .current_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or(target);
            (runtime.engine.clone(), echo)
        };

        let id = request_id();
        let mut msg = serde_json::json!({"type": "decide", "id": id});
        for key in ["state", "questions", "_debug_no_snapshot"] {
            if let Some(v) = body.get(key) {
                msg[key] = v.clone();
            }
        }
        let reply = match engine.request(&msg) {
            Ok(v) => v,
            Err(e) => {
                return DecideOutcome::Err {
                    status: 500,
                    message: format!("daemon: {e}"),
                    required_max_seq: None,
                }
            }
        };
        if reply.get("type").and_then(|v| v.as_str()) != Some("decided") {
            return DecideOutcome::Err {
                status: 500,
                message: format!("unexpected daemon reply: {reply}"),
                required_max_seq: None,
            };
        }

        // Keep the model warm: this decide reached the daemon and got a
        // `decided` reply — success or embedded error — so the idle-eviction
        // clock resets just like a completed chat generate does
        // (complete.rs's `meta.last_activity = Instant::now()` after a
        // successful `engine_clone.generate`).
        {
            let mut meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
            meta.requests_served = meta.requests_served.saturating_add(1);
            meta.last_activity = std::time::Instant::now();
        }

        if let Some(err) = reply.get("error") {
            let status = err.get("status").and_then(|v| v.as_u64()).unwrap_or(500) as u16;
            let message = err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("decide failed")
                .to_string();
            let required_max_seq = err.get("required_max_seq").and_then(|v| v.as_u64());

            if attempt == 0 && !cancelled.load(Ordering::Relaxed) {
                if let Some(n) = required_max_seq {
                    let mut runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
                    let target = runtime
                        .current_path
                        .as_ref()
                        .map(|p| p.display().to_string());
                    if let Some(target) = target {
                        match runtime.ensure_model(&target, &shared.meta, Some(n)) {
                            Ok(_) => {
                                drop(runtime);
                                continue;
                            }
                            Err(reload_err) => {
                                drop(runtime);
                                return DecideOutcome::Err {
                                    status: 500,
                                    message: format!(
                                        "decide: reload for required_max_seq={n} failed: \
                                         {reload_err:#}"
                                    ),
                                    required_max_seq: None,
                                };
                            }
                        }
                    }
                }
            }
            return DecideOutcome::Err {
                status,
                message,
                required_max_seq,
            };
        }

        return DecideOutcome::Ok {
            model: model_echo,
            answers: reply["answers"].clone(),
            usage: reply["usage"].clone(),
            timing: reply["timing"].clone(),
        };
    }
    DecideOutcome::Err {
        status: 500,
        message: "decide: context growth retry exhausted".to_string(),
        required_max_seq: None,
    }
}

pub(crate) async fn handle_decide(
    shared: Arc<ServeShared>,
    body: serde_json::Value,
    guard: AdmissionGuard,
    cancelled: Arc<AtomicBool>,
) -> Response<BoxBody> {
    let (tx, rx) = tokio::sync::oneshot::channel::<DecideOutcome>();
    tokio::task::spawn_blocking(move || {
        let outcome = run_decide(&shared, body, guard, &cancelled);
        let _ = tx.send(outcome);
    });
    // `rx.await` merely observes completion; dropping this future (hyper
    // cancelling the handler on client disconnect) detaches from the
    // spawned blocking task without aborting it, and `guard` — moved into
    // that closure above — is released only when it returns.
    match rx.await {
        Ok(DecideOutcome::Ok {
            model,
            answers,
            usage,
            timing,
        }) => {
            let out = serde_json::json!({
                "model": model,
                "answers": answers,
                "usage": usage,
            });
            let mut resp = json_response(out, 200);
            if let Ok(v) = header::HeaderValue::from_str(&timing.to_string()) {
                resp.headers_mut().insert("x-hipfire-timing", v);
            }
            resp
        }
        Ok(DecideOutcome::Err {
            status,
            message,
            required_max_seq,
        }) => decide_error_response(&message, status, required_max_seq),
        Err(_) => openai_error("decide worker disconnected", 500),
    }
}
