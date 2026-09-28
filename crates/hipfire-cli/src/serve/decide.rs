// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `POST /v1/systemone` — Jev-compatible decide. The daemon is the single
//! validation authority; this layer selects the model, serialises against
//! chat traffic, and shapes Jev's response. Spec §6.

use super::http::{admission_error_response, json_response, openai_error, request_id, BoxBody};
use super::ServeShared;
use hyper::{header, Response};
use std::sync::Arc;

fn is_local_model(runtime: &super::ServeRuntime, model: &str) -> bool {
    crate::registry_entry_for_path(&runtime.paths, &runtime.registry, model).is_some()
        || crate::find_model_path(&runtime.paths, &runtime.registry, model).is_some()
}

pub(crate) async fn handle_decide(
    shared: Arc<ServeShared>,
    mut body: serde_json::Value,
) -> Response<BoxBody> {
    let guard = match shared.admission.acquire() {
        Ok(g) => g,
        Err(e) => return admission_error_response(&e),
    };
    let _guard = guard;
    let requested = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    for attempt in 0..2 {
        // Resolve the model under the runtime lock, then release it before
        // the blocking daemon round trip so /v1/chat/completions is not
        // starved of the lock while a decide request is in flight.
        let (engine, model_echo) = {
            let mut runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
            let target = if !requested.is_empty() && is_local_model(&runtime, &requested) {
                requested.clone()
            } else {
                match runtime.current_path.as_ref() {
                    Some(p) => p.display().to_string(),
                    None => return openai_error(
                        "no model loaded: name a local hipfire model in `model`, or load one first",
                        503,
                    ),
                }
            };
            if let Err(e) = runtime.ensure_model(&target, &shared.meta, None) {
                return openai_error(&format!("{e:#}"), 500);
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
        let reply = match tokio::task::spawn_blocking(move || engine.request(&msg)).await {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return openai_error(&format!("daemon: {e}"), 500),
            Err(e) => return openai_error(&format!("decide worker failed: {e}"), 500),
        };
        if reply.get("type").and_then(|v| v.as_str()) != Some("decided") {
            return openai_error(&format!("unexpected daemon reply: {reply}"), 500);
        }
        if let Some(err) = reply.get("error") {
            let status = err.get("status").and_then(|v| v.as_u64()).unwrap_or(500) as u16;
            let message = err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("decide failed");
            if let (0, Some(n)) = (
                attempt,
                err.get("required_max_seq").and_then(|v| v.as_u64()),
            ) {
                let mut runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
                let target = runtime
                    .current_path
                    .as_ref()
                    .map(|p| p.display().to_string());
                if let Some(target) = target {
                    if runtime.ensure_model(&target, &shared.meta, Some(n)).is_ok() {
                        drop(runtime);
                        body["model"] = serde_json::json!(target);
                        continue;
                    }
                }
            }
            return openai_error(message, status);
        }
        {
            let mut meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
            meta.requests_served += 1;
        }
        let out = serde_json::json!({
            "model": model_echo,
            "answers": reply["answers"],
            "usage": reply["usage"],
        });
        let mut resp = json_response(out, 200);
        if let Ok(v) = header::HeaderValue::from_str(&reply["timing"].to_string()) {
            resp.headers_mut().insert("x-hipfire-timing", v);
        }
        return resp;
    }
    openai_error("decide: context growth retry exhausted", 500)
}
