// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-free A/B of chat-template prefix-cache behaviour for thinking models.
//!
//! Legacy three-positional-argument experiment is retained but is
//! NON-AUTHORITATIVE for production rendering: it builds its own minijinja
//! env with `UndefinedBehavior::Strict`, while production
//! `JinjaChatFrame::render_messages` uses `Lenient` plus the HF-spaced
//! `tojson` override, current-date injection, and serialized-`Message`
//! context (`prompt_frame.rs:970-1017, 1049-1095`). That divergence makes
//! it an INVALID oracle for Onyx, where optional keys and mapping-valued
//! ATEM arguments decide success. Use `--harmony` for the authoritative
//! GPU-free gate, which calls production `render_messages` and
//! `build_cached_history_jinja` directly (no `rdna_compute::Gpu`, no HIP).
//!
//! Usage (legacy, non-authoritative):
//!   template_cache_eval <model.hfq> <template.jinja> <preserve_thinking:true|false>
//!
//! Usage (authoritative Harmony/Onyx gate):
//!   template_cache_eval <model.mq4> --harmony [--perturb none|drop-reasoning-envelope|corrupt-tool-body-token]
//!
//! Usage (GPU-free Qwen XML tool-turn splice oracle; replays a 3-request agent conversation
//! through the production Qwen4 Jinja render, plain vs `ToolTurnSplice::WholeBody`):
//!   template_cache_eval --qwen-tool-turns <metadata.json>

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::prompt_frame::{
    build_cached_history_jinja, build_cached_history_jinja_with, cached_producer_reasoning_text,
    hf_tojson, CachedAssistantBody, CachedAssistantToolBody, CachedAssistantTurn, JinjaChatFrame,
    Message, Role, ToolCall, ToolTurnSplice,
};
use hipfire_runtime::tokenizer::Tokenizer;
use minijinja::{context, Environment, Error, ErrorKind, Value};
use minijinja_contrib::pycompat::unknown_method_callback;
use std::path::Path;

// ---------------------------------------------------------------------------
// Legacy render (Strict) — kept for backwards compat, labelled non-authoritative
// ---------------------------------------------------------------------------
fn render(
    template: &str,
    bos_token: &str,
    messages: &serde_json::Value,
    preserve_thinking: bool,
) -> Result<String, String> {
    let mut env = Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });
    env.add_template("chat", template)
        .map_err(|e| format!("parse: {e}"))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("lookup: {e}"))?;
    let empty: Vec<serde_json::Value> = Vec::new();
    let ctx = context! {
        messages => Value::from_serialize(messages),
        add_generation_prompt => true,
        enable_thinking => true,
        preserve_thinking => preserve_thinking,
        bos_token => bos_token,
        tools => Value::from_serialize(&empty),
        documents => Value::from_serialize(&empty),
        tool_call_kwargs => Value::from_serialize(&serde_json::Map::new()),
    };
    tmpl.render(ctx).map_err(|e| format!("render: {e}"))
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// Harmony oracle types
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Perturbation {
    None,
    DropReasoningEnvelope,
    CorruptToolBodyToken,
}

impl Perturbation {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "drop-reasoning-envelope" => Some(Self::DropReasoningEnvelope),
            "corrupt-tool-body-token" => Some(Self::CorruptToolBodyToken),
            _ => None,
        }
    }
}

#[derive(Debug, serde::Serialize)]
struct OracleRow {
    case: &'static str,
    prior_len: usize,
    prompt_len: usize,
    lcp: usize,
    forward_extension: bool,
}

fn extension_row(case: &'static str, prior_kv: &[u32], next_prompt: &[u32]) -> OracleRow {
    let l = lcp(prior_kv, next_prompt);
    let forward_extension = l == prior_kv.len();
    OracleRow {
        case,
        prior_len: prior_kv.len(),
        prompt_len: next_prompt.len(),
        lcp: l,
        forward_extension,
    }
}

/// Rewrite the Onyx template's nested `tc.function.*` accessors to the flat
/// `tc.name` / `tc.arguments` shape that `prompt_frame::ToolCall` serializes,
/// and add the `tc.rendered_body` verbatim branch. This mirrors the
/// arch-14-only loader rewrite (slice E) so the oracle exercises the same
/// bytes the daemon will see. If the template already uses flat accessors
/// the replacements are a no-op.
fn rewrite_onyx_template_for_flat_toolcall(template: &str) -> String {
    let mut out = template.to_string();
    // Flat accessors
    out = out.replace("tc.function.name", "tc.name");
    out = out.replace("tc.function.arguments", "tc.arguments");
    // Verbatim splice branch for cached tool bodies.
    // Original: {{- render_atem(tc) -}}
    // Rewritten: {%- if tc.rendered_body is defined and tc.rendered_body -%}{{ tc.rendered_body }}{%- else -%}{{ render_atem(tc) }}{%- endif -%}
    // Handle both trimmed and non-trimmed variants.
    let needle = "{{- render_atem(tc) -}}";
    let replacement = "{%- if tc.rendered_body is defined and tc.rendered_body -%}{{ tc.rendered_body }}{%- else -%}{{ render_atem(tc) }}{%- endif -%}";
    if out.contains(needle) {
        out = out.replace(needle, replacement);
    } else {
        // Fallback: without dash whitespace control
        out = out.replace("{{ render_atem(tc) }}", replacement);
        out = out.replace("{{- render_atem(tc) }}", replacement);
        out = out.replace("{{ render_atem(tc) -}}", replacement);
    }
    out
}

fn hf_json_for_value(v: &serde_json::Value) -> String {
    // Use production hf_tojson for byte-identical spacing.
    let mv = Value::from_serialize(v);
    hf_tojson(mv).unwrap_or_else(|_| serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string()))
}

fn render_atem_string(name: &str, arguments: &serde_json::Value) -> String {
    let mut s = String::new();
    s.push_str("<atem:function_calls>\n<atem:invoke name=\"");
    s.push_str(name);
    s.push_str("\">\n");
    if let Some(map) = arguments.as_object() {
        for (k, v) in map {
            s.push_str("<atem:parameter name=\"");
            s.push_str(k);
            s.push_str("\">");
            if v.is_boolean() {
                s.push_str(if v.as_bool().unwrap() {
                    "true"
                } else {
                    "false"
                });
            } else if v.is_null() {
                s.push_str("null");
            } else if v.is_object() || v.is_array() {
                s.push_str(&hf_json_for_value(v));
            } else if let Some(st) = v.as_str() {
                s.push_str(st);
            } else if v.is_number() {
                s.push_str(&v.to_string());
            } else {
                s.push_str(&v.to_string());
            }
            s.push_str("</atem:parameter>\n");
        }
    }
    s.push_str("</atem:invoke>\n</atem:function_calls>");
    s
}

fn run_harmony_oracle(model_path: &Path, perturb: Perturbation) -> Result<Vec<OracleRow>, String> {
    // Load HFQ and production tokenizer/template.
    let hfq = HfqFile::open(model_path).map_err(|e| format!("open hfq: {e}"))?;
    let tok =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| format!("tokenizer: {e}"))?;
    let raw_template = hfq
        .chat_template()
        .ok_or_else(|| "hfq has no chat_template".to_string())?;
    let template = rewrite_onyx_template_for_flat_toolcall(&raw_template);

    // Common conversation snippets (deterministic).
    let q1 = "What is the capital of France?";
    let q2 = "Now what is the capital of Germany?";
    let reasoning = "The user wants the capital of France. In the standard RYB model those are not relevant; answer Paris.";
    let answer = "The capital of France is Paris.";
    // Tool fixtures
    let tool_args = serde_json::json!({
        "location": "Paris",
        "options": {"units": "celsius", "days": [1, 2]},
        "include_alerts": true,
        "fallback": null
    });
    let tool_name = "weather.get_forecast";
    let tool_atem = render_atem_string(tool_name, &tool_args);
    let tool_result_content = "{\"location\":\"Paris\",\"units\":\"celsius\",\"days\":[{\"day\":1,\"high\":18},{\"day\":2,\"high\":20}],\"alerts\":[],\"fallback_used\":null}";

    let tools_json: Vec<serde_json::Value> = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Get weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "options": {"type": "object", "properties": {"units": {"type": "string"}, "days": {"type": "array", "items": {"type": "integer"}}}},
                    "include_alerts": {"type": "boolean"},
                    "fallback": {"type": ["string", "null"]}
                },
                "required": ["location", "options", "include_alerts", "fallback"],
                "additionalProperties": false
            }
        }
    })];
    let tools_changed_json: Vec<serde_json::Value> = {
        let mut v = tools_json.clone();
        v.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "calc.add",
                "description": "Add numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"]
                }
            }
        }));
        v
    };

    let make_frame = |enable_thinking: bool| JinjaChatFrame {
        tokenizer: &tok,
        template: &template,
        system: None,
        user: "",
        enable_thinking,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    };

    let mut rows: Vec<OracleRow> = Vec::new();

    // ---- normal-thinking-on ----
    {
        let frame_on = make_frame(true);
        let user1 = Message {
            role: Role::User,
            content: q1.to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let req1 = vec![user1.clone()];
        let prompt1_str = frame_on
            .render_messages(&req1, Some(&tools_json), None)
            .map_err(|e| format!("normal-thinking-on req1 render: {e}"))?;
        // Prior KV: prompt1 + generated reasoning+content (omitting final <|eot|>)
        let generation = format!(
            " to=self<|message|>{}<|eom|><|start|>assistant to=user<|message|>{}",
            reasoning, answer
        );
        let prior_kv = tok.encode(&(prompt1_str.clone() + &generation));

        // Next prompt via splice.
        // Perturb: drop-reasoning-envelope omits reasoning from both Message and cache.
        let (msg_reasoning, cached_reasoning) = if perturb == Perturbation::DropReasoningEnvelope {
            (None, None)
        } else {
            (
                Some(reasoning.to_string()),
                Some(CachedAssistantBody {
                    token_ids: tok.encode(reasoning),
                    text: reasoning.to_string(),
                }),
            )
        };
        let next_msgs = vec![
            Message {
                role: Role::User,
                content: q1.to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Assistant,
                content: answer.to_string(),
                reasoning_content: msg_reasoning,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: q2.to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ];
        let cached_turn = CachedAssistantTurn {
            reasoning: cached_reasoning,
            tools: Vec::new(),
            content: Some(CachedAssistantBody {
                token_ids: tok.encode(answer),
                text: answer.to_string(),
            }),
        };
        let next_prompt =
            build_cached_history_jinja(&frame_on, &next_msgs, Some(&tools_json), |m| {
                if m.role == Role::Assistant && m.content == answer {
                    Some(cached_turn.clone())
                } else {
                    None
                }
            })
            .map_err(|e| format!("normal-thinking-on build_cached: {e}"))?;
        rows.push(extension_row("normal-thinking-on", &prior_kv, &next_prompt));
    }

    // ---- normal-thinking-off ----
    {
        // Prior is still generated with thinking on; next frame is thinking off but history stays.
        let frame_on = make_frame(true);
        let frame_off = make_frame(false);
        let user1 = Message {
            role: Role::User,
            content: q1.to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let req1 = vec![user1.clone()];
        let prompt1_str = frame_on
            .render_messages(&req1, Some(&tools_json), None)
            .map_err(|e| format!("normal-thinking-off req1 render: {e}"))?;
        let generation = format!(
            " to=self<|message|>{}<|eom|><|start|>assistant to=user<|message|>{}",
            reasoning, answer
        );
        let prior_kv = tok.encode(&(prompt1_str + &generation));

        let (msg_reasoning, cached_reasoning) = if perturb == Perturbation::DropReasoningEnvelope {
            (None, None)
        } else {
            (
                Some(reasoning.to_string()),
                Some(CachedAssistantBody {
                    token_ids: tok.encode(reasoning),
                    text: reasoning.to_string(),
                }),
            )
        };
        let next_msgs = vec![
            Message {
                role: Role::User,
                content: q1.to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Assistant,
                content: answer.to_string(),
                reasoning_content: msg_reasoning,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: q2.to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ];
        let cached_turn = CachedAssistantTurn {
            reasoning: cached_reasoning,
            tools: Vec::new(),
            content: Some(CachedAssistantBody {
                token_ids: tok.encode(answer),
                text: answer.to_string(),
            }),
        };
        let next_prompt =
            build_cached_history_jinja(&frame_off, &next_msgs, Some(&tools_json), |m| {
                if m.role == Role::Assistant && m.content == answer {
                    Some(cached_turn.clone())
                } else {
                    None
                }
            })
            .map_err(|e| format!("normal-thinking-off build_cached: {e}"))?;
        rows.push(extension_row(
            "normal-thinking-off",
            &prior_kv,
            &next_prompt,
        ));
    }

    // ---- no-system-message ----
    {
        // No system turn at all; default system block is auto-injected by template.
        let frame_on = make_frame(true);
        // Same q1/q2 but we explicitly test that a conversation with zero system
        // messages renders without error and still forward-extends.
        let user1 = Message {
            role: Role::User,
            content: "Hello".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let req1 = vec![user1.clone()];
        let prompt1_str = frame_on
            .render_messages(&req1, Some(&tools_json), None)
            .map_err(|e| format!("no-system-message req1 render: {e}"))?;
        // Ensure the two guarded branches (current_date / strftime_now) did not raise:
        // if either raised, render_messages would have Err'd and we'd have bailed.
        let generation = format!(
            " to=self<|message|>{}<|eom|><|start|>assistant to=user<|message|>{}",
            reasoning, answer
        );
        let prior_kv = tok.encode(&(prompt1_str.clone() + &generation));
        let next_msgs = vec![
            Message {
                role: Role::User,
                content: "Hello".to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Assistant,
                content: answer.to_string(),
                reasoning_content: if perturb == Perturbation::DropReasoningEnvelope {
                    None
                } else {
                    Some(reasoning.to_string())
                },
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: q2.to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ];
        let cached_reasoning = if perturb == Perturbation::DropReasoningEnvelope {
            None
        } else {
            Some(CachedAssistantBody {
                token_ids: tok.encode(reasoning),
                text: reasoning.to_string(),
            })
        };
        let cached_turn = CachedAssistantTurn {
            reasoning: cached_reasoning,
            tools: Vec::new(),
            content: Some(CachedAssistantBody {
                token_ids: tok.encode(answer),
                text: answer.to_string(),
            }),
        };
        let next_prompt =
            build_cached_history_jinja(&frame_on, &next_msgs, Some(&tools_json), |m| {
                if m.role == Role::Assistant && m.content == answer {
                    Some(cached_turn.clone())
                } else {
                    None
                }
            })
            .map_err(|e| format!("no-system-message build_cached: {e}"))?;
        rows.push(extension_row("no-system-message", &prior_kv, &next_prompt));
    }

    // ---- tool-roundtrip ----
    {
        let frame_on = make_frame(true);
        let user1 = Message {
            role: Role::User,
            content: "Call the weather tool for Paris.".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let req1 = vec![user1.clone()];
        let prompt1_str = frame_on
            .render_messages(&req1, Some(&tools_json), None)
            .map_err(|e| format!("tool-roundtrip req1 render: {e}"))?;
        let atem_body = tool_atem.clone();
        let generation = format!(
            " to=self<|message|>{}<|eom|><|start|>assistant to={}<|message|>{}",
            reasoning, tool_name, atem_body
        );
        let prior_kv = tok.encode(&(prompt1_str + &generation));

        // Prepare cached tool body tokens; corrupt if perturb asks.
        let mut tool_body_ids = tok.encode(&atem_body);
        if perturb == Perturbation::CorruptToolBodyToken && !tool_body_ids.is_empty() {
            tool_body_ids[0] = tool_body_ids[0].wrapping_add(1);
        }
        let reasoning_body = if perturb == Perturbation::DropReasoningEnvelope {
            None
        } else {
            Some(CachedAssistantBody {
                token_ids: tok.encode(reasoning),
                text: reasoning.to_string(),
            })
        };
        // For drop-reasoning-envelope perturb we also keep tool body (only reasoning dropped)
        // For corrupt-tool-body-token we keep reasoning but corrupt tool.
        let cached_turn = CachedAssistantTurn {
            reasoning: if perturb == Perturbation::DropReasoningEnvelope {
                None
            } else {
                reasoning_body
            },
            tools: vec![CachedAssistantToolBody {
                recipient: tool_name.to_string(),
                token_ids: tool_body_ids,
            }],
            content: None,
        };
        let tool_call = ToolCall {
            id: Some("call_0".to_string()),
            name: tool_name.to_string(),
            arguments: tool_args.clone(),
            rendered_body: None,
        };
        let next_msgs = vec![
            Message {
                role: Role::User,
                content: "Call the weather tool for Paris.".to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Assistant,
                content: String::new(),
                reasoning_content: if perturb == Perturbation::DropReasoningEnvelope {
                    None
                } else {
                    Some(reasoning.to_string())
                },
                name: None,
                rendered_name: None,
                tool_calls: vec![tool_call],
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Tool,
                content: tool_result_content.to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: Some("call_0".to_string()),
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: "Summarize the forecast.".to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ];
        let next_prompt =
            build_cached_history_jinja(&frame_on, &next_msgs, Some(&tools_json), |m| {
                if m.role == Role::Assistant && !m.tool_calls.is_empty() {
                    Some(cached_turn.clone())
                } else {
                    None
                }
            })
            .map_err(|e| format!("tool-roundtrip build_cached: {e}"))?;
        rows.push(extension_row("tool-roundtrip", &prior_kv, &next_prompt));
    }

    // ---- tool-set-changed ----
    {
        let frame_on = make_frame(true);
        let user1 = Message {
            role: Role::User,
            content: "Call the weather tool for Paris.".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let req1 = vec![user1.clone()];
        let prompt1_str = frame_on
            .render_messages(&req1, Some(&tools_json), None)
            .map_err(|e| format!("tool-set-changed req1 render: {e}"))?;
        let atem_body = tool_atem.clone();
        let generation = format!(
            " to=self<|message|>{}<|eom|><|start|>assistant to={}<|message|>{}",
            reasoning, tool_name, atem_body
        );
        let prior_kv = tok.encode(&(prompt1_str + &generation));

        let tool_body_ids = tok.encode(&atem_body);
        let cached_turn = CachedAssistantTurn {
            reasoning: Some(CachedAssistantBody {
                token_ids: tok.encode(reasoning),
                text: reasoning.to_string(),
            }),
            tools: vec![CachedAssistantToolBody {
                recipient: tool_name.to_string(),
                token_ids: tool_body_ids,
            }],
            content: None,
        };
        let tool_call = ToolCall {
            id: Some("call_0".to_string()),
            name: tool_name.to_string(),
            arguments: tool_args.clone(),
            rendered_body: None,
        };
        let next_msgs = vec![
            Message {
                role: Role::User,
                content: "Call the weather tool for Paris.".to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Assistant,
                content: String::new(),
                reasoning_content: Some(reasoning.to_string()),
                name: None,
                rendered_name: None,
                tool_calls: vec![tool_call],
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Tool,
                content: tool_result_content.to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: Some("call_0".to_string()),
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: "Summarize the forecast.".to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ];
        // Use CHANGED tool set for next prompt — system block differs, so LCP must collapse.
        // Even if perturb corrupts tool body, this case remains collapsed; perturb corrupts tool-roundtrip only.
        let mut next_prompt =
            build_cached_history_jinja(&frame_on, &next_msgs, Some(&tools_changed_json), |m| {
                if m.role == Role::Assistant && !m.tool_calls.is_empty() {
                    Some(cached_turn.clone())
                } else {
                    None
                }
            })
            .map_err(|e| format!("tool-set-changed build_cached: {e}"))?;
        // If corrupt perturb, we already corrupted tool-roundtrip; keep this one uncorrupted
        // to preserve its expected collapse behaviour independent of perturb.
        // No need to adjust; collapse is due to system block.
        let _ = &mut next_prompt; // silence unused_mut if not needed
        rows.push(extension_row("tool-set-changed", &prior_kv, &next_prompt));
    }

    Ok(rows)
}

// ---------------------------------------------------------------------------
// Qwen tool-turn oracle (`--qwen-tool-turns`): GPU-free replay of a 3-request
// agent conversation through the production Qwen4 Jinja render, comparing the
// per-request prior consumed stream (prompt ++ generated body ++ <|im_end|>)
// with the `Envelopes` (plain re-render) and `WholeBody` cached renders.
// ---------------------------------------------------------------------------
struct ToolTurnRow {
    thinking: &'static str,
    tok: &'static str,
    request: usize,
    prior_len: usize,
    lcp_plain: usize,
    lcp_wholebody: usize,
    strict_plain: bool,
    strict_wholebody: bool,
}

fn tt_msg(role: Role, content: &str, reasoning: Option<&str>) -> Message {
    Message {
        role,
        content: content.to_string(),
        reasoning_content: reasoning.map(|s| s.to_string()),
        name: None,
        rendered_name: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tool_plan: String::new(),
    }
}

fn tt_call_text(name: &str, key: &str, value: &str) -> String {
    format!(
        "<tool_call>\n<function={name}>\n<parameter={key}>\n{value}\n</parameter>\n</function>\n</tool_call>"
    )
}

/// Generated-body ids. Canonical = one whole-text encode. Non-canonical =
/// each line encoded separately with newlines as their own encode, so BPE
/// merges across line boundaries (`\n\n`, `>\n`, ...) differ from the
/// whole-text encode — the tokenization a sampler can emit but the encoder
/// would not reproduce. A body with no merge across a line boundary (e.g. a
/// single tool call with no blank line) encodes identically per line, so each
/// line is then additionally split at its midpoint; the result is asserted to
/// differ from the canonical encode either way.
fn tt_body_ids(tok: &Tokenizer, text: &str, canonical: bool) -> Result<Vec<u32>, String> {
    let whole = tok.encode(text);
    if canonical {
        return Ok(whole);
    }
    let encode_lines = |split_midpoint: bool| -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::new();
        for (i, line) in text.split('\n').enumerate() {
            if i > 0 {
                ids.extend(tok.encode("\n"));
            }
            if line.is_empty() {
                continue;
            }
            if split_midpoint {
                let mut mid = line.len() / 2;
                while !line.is_char_boundary(mid) {
                    mid += 1;
                }
                let (a, b) = line.split_at(mid);
                for piece in [a, b] {
                    if !piece.is_empty() {
                        ids.extend(tok.encode(piece));
                    }
                }
            } else {
                ids.extend(tok.encode(line));
            }
        }
        ids
    };
    for split_midpoint in [false, true] {
        let ids = encode_lines(split_midpoint);
        if ids != whole {
            return Ok(ids);
        }
    }
    Err(format!(
        "non-canonical tokenization equals canonical for body {text:?}; the oracle would not discriminate"
    ))
}

fn tt_filler(prefix: &str, lines: usize) -> String {
    (0..lines)
        .map(|i| {
            format!(
                "{prefix}_{i}: fn handler_{i}(input: u32) -> u32 {{ input.wrapping_mul({}) }}",
                i + 3
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn simulate_qwen_tool_turns(
    tok: &Tokenizer,
    template: &str,
    thinking: bool,
    canonical: bool,
    tools: &[serde_json::Value],
) -> Result<Vec<ToolTurnRow>, String> {
    let frame = JinjaChatFrame {
        tokenizer: tok,
        template,
        system: None,
        user: "",
        enable_thinking: thinking,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    };
    let im_end = tok
        .special_token_id("<|im_end|>")
        .ok_or("tokenizer has no <|im_end|>")?;
    // (reasoning text, prose before the call, call name, arg key, arg value, tool result)
    let turns: [(&str, &str, &str, &str, &str, String); 2] = [
        (
            "The user wants the failing test fixed. I should read the source file first.",
            "",
            "read_file",
            "path",
            "src/lib.rs",
            tt_filler("file", 9),
        ),
        (
            "The file looks fine at a glance. Now I should run the tests to see the failure.",
            "I'll run the tests.",
            "run_tests",
            "filter",
            "parse",
            tt_filler("test", 9),
        ),
    ];
    let mut messages = vec![
        tt_msg(Role::System, "You are a coding agent.", None),
        tt_msg(Role::User, "Fix the failing parse test.", None),
    ];
    let render_ids = |msgs: &[Message]| -> Result<(Vec<u32>, bool), String> {
        let text = frame.render_messages(msgs, Some(tools), None)?;
        Ok((tok.encode(&text), text.ends_with("<think>\n")))
    };
    let (mut prompt, started_in_think) = render_ids(&messages)?;
    if started_in_think != thinking {
        return Err(format!(
            "generation prompt primer ({started_in_think}) disagrees with enable_thinking ({thinking})"
        ));
    }
    // asst_turn_cache stand-in: keyed by the call's identity (name + arguments).
    let mut entries: Vec<(String, serde_json::Value, CachedAssistantTurn)> = Vec::new();
    let mut rows = Vec::new();
    for (k, (reasoning, prose, name, key, value, result)) in turns.iter().enumerate() {
        let call = tt_call_text(name, key, value);
        let prose_part = if prose.is_empty() {
            String::new()
        } else {
            format!("{prose}\n\n")
        };
        let body_text = if thinking {
            format!("{reasoning}\n</think>\n\n{prose_part}{call}")
        } else {
            format!("{prose_part}{call}")
        };
        let body = tt_body_ids(tok, &body_text, canonical)?;
        let mut consumed = prompt.clone();
        consumed.extend_from_slice(&body);
        consumed.push(im_end);
        // Production cache entry shape (`qwen_cached_turn_entry`).
        let reasoning_text = cached_producer_reasoning_text(tok, &body, started_in_think);
        let entry = CachedAssistantTurn {
            reasoning: reasoning_text.clone().map(|text| CachedAssistantBody {
                token_ids: Vec::new(),
                text,
            }),
            tools: Vec::new(),
            content: Some(CachedAssistantBody {
                token_ids: body.clone(),
                text: String::new(),
            }),
        };
        let arguments = serde_json::json!({ *key: *value });
        entries.push((name.to_string(), arguments.clone(), entry));
        // API echo of the turn: visible content, reasoning channel, parsed call.
        let mut assistant = tt_msg(Role::Assistant, prose, reasoning_text.as_deref());
        assistant.tool_calls = vec![ToolCall {
            id: Some(format!("call_{k}")),
            name: name.to_string(),
            arguments,
            rendered_body: None,
        }];
        let mut tool_result = tt_msg(Role::Tool, result, None);
        tool_result.tool_call_id = Some(format!("call_{k}"));
        messages.push(assistant);
        messages.push(tool_result);

        let lookup = |m: &Message| -> Option<CachedAssistantTurn> {
            if m.role != Role::Assistant {
                return None;
            }
            let c = m.tool_calls.first()?;
            entries
                .iter()
                .find(|(n, a, _)| *n == c.name && *a == c.arguments)
                .map(|(_, _, e)| e.clone())
        };
        let plain = build_cached_history_jinja_with(
            &frame,
            &messages,
            Some(tools),
            ToolTurnSplice::Envelopes,
            lookup,
        )?;
        let whole = build_cached_history_jinja_with(
            &frame,
            &messages,
            Some(tools),
            ToolTurnSplice::WholeBody,
            lookup,
        )?;
        let lcp_plain = lcp(&consumed, &plain);
        let lcp_wholebody = lcp(&consumed, &whole);
        rows.push(ToolTurnRow {
            thinking: if thinking { "on" } else { "off" },
            tok: if canonical {
                "canonical"
            } else {
                "noncanonical"
            },
            request: k + 2,
            prior_len: consumed.len(),
            lcp_plain,
            lcp_wholebody,
            strict_plain: lcp_plain == consumed.len() && plain.len() > consumed.len(),
            strict_wholebody: lcp_wholebody == consumed.len() && whole.len() > consumed.len(),
        });
        // The engine feeds the WholeBody render as the next request's prompt.
        prompt = whole;
    }
    Ok(rows)
}

fn run_qwen_tool_turns(meta_path: &Path) -> Result<Vec<ToolTurnRow>, String> {
    let json =
        std::fs::read_to_string(meta_path).map_err(|e| format!("read {meta_path:?}: {e}"))?;
    let tok = Tokenizer::from_hfq_metadata(&json).map_err(|e| format!("tokenizer: {e:?}"))?;
    let meta: serde_json::Value =
        serde_json::from_str(&json).map_err(|e| format!("metadata json: {e}"))?;
    let template = meta["tokenizer_config"]["chat_template"]
        .as_str()
        .ok_or("metadata has no tokenizer_config.chat_template string")?;
    let tools = vec![
        serde_json::json!({"type": "function", "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {"type": "object",
                "properties": {"path": {"type": "string", "description": "Workspace-relative path"}},
                "required": ["path"]}}}),
        serde_json::json!({"type": "function", "function": {
            "name": "run_tests",
            "description": "Run the test suite.",
            "parameters": {"type": "object",
                "properties": {"filter": {"type": "string", "description": "Test name filter"}},
                "required": ["filter"]}}}),
    ];
    eprintln!(
        "tool result sizes: {} / {} tokens",
        tok.encode(&tt_filler("file", 9)).len(),
        tok.encode(&tt_filler("test", 9)).len()
    );
    let mut rows = Vec::new();
    for thinking in [false, true] {
        for canonical in [true, false] {
            rows.extend(simulate_qwen_tool_turns(
                &tok, template, thinking, canonical, &tools,
            )?);
        }
    }
    Ok(rows)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Qwen tool-turn mode: template_cache_eval --qwen-tool-turns <metadata.json>
    if let Some(idx) = args.iter().position(|a| a == "--qwen-tool-turns") {
        let Some(meta_path) = args.get(idx + 1) else {
            eprintln!("usage: template_cache_eval --qwen-tool-turns <metadata.json>");
            std::process::exit(2);
        };
        match run_qwen_tool_turns(Path::new(meta_path)) {
            Ok(rows) => {
                let mut failing = 0usize;
                for r in &rows {
                    println!(
                        "qwen_tool_turns thinking={} tok={} request={} prior_len={} lcp_plain={} \
                         lcp_wholebody={} strict_extension_plain={} strict_extension_wholebody={}",
                        r.thinking,
                        r.tok,
                        r.request,
                        r.prior_len,
                        r.lcp_plain,
                        r.lcp_wholebody,
                        r.strict_plain,
                        r.strict_wholebody
                    );
                    if !r.strict_wholebody {
                        failing += 1;
                    }
                }
                if failing == 0 {
                    println!("oracle_result=PASS");
                    std::process::exit(0);
                }
                eprintln!(
                    "{failing} WholeBody row(s) are not a strict extension of the consumed stream"
                );
                println!("oracle_result=FAIL");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("qwen tool-turn oracle error: {e}");
                std::process::exit(2);
            }
        }
    }

    // Harmony mode: template_cache_eval <model.mq4> --harmony [--perturb ...]
    let harmony_idx = args.iter().position(|a| a == "--harmony");
    if let Some(idx) = harmony_idx {
        let model_path = args
            .get(1)
            .expect("harmony usage: template_cache_eval <model.mq4> --harmony [--perturb ...]");
        if idx != 2 {
            eprintln!("harmony usage: template_cache_eval <model.mq4> --harmony [--perturb none|drop-reasoning-envelope|corrupt-tool-body-token]");
            std::process::exit(2);
        }
        let perturb = if let Some(p_idx) = args.iter().position(|a| a == "--perturb") {
            let v = args.get(p_idx + 1).expect("--perturb requires a value");
            Perturbation::from_str(v).unwrap_or_else(|| {
                eprintln!("unknown --perturb value: {v} (expected none|drop-reasoning-envelope|corrupt-tool-body-token)");
                std::process::exit(2);
            })
        } else {
            Perturbation::None
        };
        // Also support --perturb=VALUE form
        let perturb = args
            .iter()
            .find_map(|a| a.strip_prefix("--perturb="))
            .and_then(Perturbation::from_str)
            .unwrap_or(perturb);

        // Verdict, deliberately unambiguous for scripting:
        //   `oracle_result=PASS`        + rc 0  -> green. This is the ONLY green token.
        //   `oracle_result=FAIL`        + rc 1  -> a case did not match its expectation.
        //   `oracle_result=INSENSITIVE` + rc 3  -> a negative control failed to go red, i.e.
        //                                          the gate cannot detect the bug it exists for.
        //   rc 2                                -> usage / internal error.
        //
        // Under `--perturb`, FAIL is the SUCCESSFUL outcome of the negative control: the caller
        // requires a non-zero rc AND the absence of `oracle_result=PASS`. Printing PASS while red
        // (the previous behaviour) would fool any script that greps for PASS.
        match run_harmony_oracle(Path::new(model_path), perturb) {
            Ok(rows) => {
                // Expectation truth table. `tool-set-changed` is expected NOT to forward-extend:
                // a different tool list rewrites the system block, so the LCP must collapse and
                // the daemon must cold-reset. That collapse is correct behaviour, not a failure.
                let expected = |case: &str| case != "tool-set-changed";
                let mut mismatches: Vec<&str> = Vec::new();
                for r in &rows {
                    println!(
                        "oracle={} prior_len={} prompt_len={} lcp={} forward_extension={}",
                        r.case, r.prior_len, r.prompt_len, r.lcp, r.forward_extension
                    );
                    if r.forward_extension != expected(r.case) {
                        mismatches.push(r.case);
                    }
                }

                if perturb != Perturbation::None {
                    if mismatches.is_empty() {
                        eprintln!(
                            "perturb={perturb:?} changed nothing — the oracle cannot observe this \
                             corruption, so it is not a gate"
                        );
                        println!("oracle_result=INSENSITIVE");
                        std::process::exit(3);
                    }
                    eprintln!(
                        "perturb={perturb:?} correctly flipped: {}",
                        mismatches.join(", ")
                    );
                    println!("oracle_result=FAIL");
                    std::process::exit(1);
                }

                if mismatches.is_empty() {
                    println!("oracle_result=PASS");
                    std::process::exit(0);
                }
                eprintln!("cases not matching expectation: {}", mismatches.join(", "));
                println!("oracle_result=FAIL");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("harmony oracle error: {e}");
                std::process::exit(2);
            }
        }
    }

    // Legacy three-positional-argument experiment (non-authoritative)
    if args.len() >= 4 && !args.contains(&"--harmony".to_string()) {
        eprintln!(
            "warning: legacy three-argument form is NON-AUTHORITATIVE for Onyx; it uses Strict undefined and lacks HF tojson/current_date. Use --harmony for the production oracle."
        );
    }
    let model_path = args
        .get(1)
        .expect("usage: <model.hfq> <template.jinja> <preserve:true|false>  OR  <model.mq4> --harmony [--perturb ...]");
    let tmpl_path = args.get(2).expect("template path");
    let preserve = args.get(3).map(|s| s == "true").unwrap_or(false);

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let tok = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let bos_bytes = tok.decode_bytes(&[tok.bos_id]);
    let bos = String::from_utf8_lossy(&bos_bytes).to_string();
    let template = std::fs::read_to_string(tmpl_path).expect("read template");

    // Conversation pieces (short, deterministic).
    let q1 = "Name the three primary colors in one short sentence.";
    let reasoning = "The user wants the three primary colors. In the standard RYB model those are red, yellow, and blue.";
    let answer = "The three primary colors are red, yellow, and blue.";
    let q2 = "Now name the three secondary colors in one short sentence.";

    let t1_msgs = serde_json::json!([{ "role": "user", "content": q1, "tool_calls": [] }]);
    let t1_prompt = render(&template, &bos, &t1_msgs, preserve).expect("t1 render");
    let asst_body = format!("{reasoning}\n</think>\n\n{answer}");
    let t1_kv_text = format!("{t1_prompt}{asst_body}");
    let t1_kv = tok.encode(&t1_kv_text);

    let t2_msgs = serde_json::json!([
        { "role": "user", "content": q1, "tool_calls": [] },
        { "role": "assistant", "content": answer, "reasoning_content": reasoning, "tool_calls": [] },
        { "role": "user", "content": q2, "tool_calls": [] },
    ]);
    let t2_text = render(&template, &bos, &t2_msgs, preserve).expect("t2 render");
    let t2 = tok.encode(&t2_text);

    let l = lcp(&t1_kv, &t2);
    let forward_ext = l == t1_kv.len();
    let pct = if t1_kv.is_empty() {
        0.0
    } else {
        100.0 * l as f64 / t1_kv.len() as f64
    };

    println!(
        "template       : {}",
        Path::new(tmpl_path).file_name().unwrap().to_string_lossy()
    );
    println!("preserve_thinking: {preserve}");
    println!("turn1_kv tokens : {}", t1_kv.len());
    println!("turn2    tokens : {}", t2.len());
    println!("lcp            : {l}  ({pct:.1}% of turn1_kv)");
    println!("forward_extension(100% cache): {forward_ext}");
    if !forward_ext {
        let lo = l.saturating_sub(6);
        let a_hi = (l + 6).min(t1_kv.len());
        let b_hi = (l + 6).min(t2.len());
        println!("  diverge@ {l}: kv …{:?}", tok.decode(&t1_kv[lo..a_hi]));
        println!("           t2 …{:?}", tok.decode(&t2[lo..b_hi]));
    }
    // Also emit machine-checkable row for legacy (not gated)
    println!(
        "legacy case=legacy prior_len={} prompt_len={} lcp={} forward_extension={}",
        t1_kv.len(),
        t2.len(),
        l,
        forward_ext
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcp_empty_and_prefix() {
        assert_eq!(lcp(&[], &[]), 0);
        assert_eq!(lcp(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(lcp(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(lcp(&[1, 2], &[1, 2, 3, 4]), 2);
        assert_eq!(lcp(&[5, 6, 7], &[1, 2, 3]), 0);
    }

    #[test]
    fn extension_row_computes_forward() {
        let r = extension_row("test", &[1, 2, 3], &[1, 2, 3, 4, 5]);
        assert_eq!(r.prior_len, 3);
        assert_eq!(r.prompt_len, 5);
        assert_eq!(r.lcp, 3);
        assert!(r.forward_extension);
        let r2 = extension_row("test", &[1, 2, 3], &[1, 9, 3]);
        assert_eq!(r2.lcp, 1);
        assert!(!r2.forward_extension);
    }

    #[test]
    fn perturbation_parsing() {
        assert_eq!(Perturbation::from_str("none"), Some(Perturbation::None));
        assert_eq!(
            Perturbation::from_str("drop-reasoning-envelope"),
            Some(Perturbation::DropReasoningEnvelope)
        );
        assert_eq!(
            Perturbation::from_str("corrupt-tool-body-token"),
            Some(Perturbation::CorruptToolBodyToken)
        );
        assert_eq!(Perturbation::from_str("bogus"), None);
    }

    #[test]
    fn rewrite_onyx_template_rewrites_flat_accessors() {
        let orig = "tc.function.name and tc.function.arguments and {{- render_atem(tc) -}}";
        let rewritten = rewrite_onyx_template_for_flat_toolcall(orig);
        assert!(rewritten.contains("tc.name"));
        assert!(rewritten.contains("tc.arguments"));
        assert!(!rewritten.contains("tc.function.name"));
        assert!(rewritten.contains("tc.rendered_body"));
        assert!(rewritten.contains("render_atem"));
    }

    #[test]
    fn rewrite_is_noop_when_already_flat() {
        let orig = "already flat tc.name and no atem";
        let rewritten = rewrite_onyx_template_for_flat_toolcall(orig);
        assert_eq!(rewritten, orig);
    }

    #[test]
    fn hf_json_spacing_matches_production() {
        // HfJsonFormatter uses ", " and ": "
        let v = serde_json::json!({"a": 1, "b": [1, 2]});
        let s = hf_json_for_value(&v);
        // Should contain ": " and ", "
        assert!(s.contains(": "));
        assert!(s.contains(", "));
        // Compact serde_json would be {"a":1,"b":[1,2]} without spaces after colon? Actually serde compact is without spaces.
        // Ensure not compact.
        assert_ne!(s, serde_json::to_string(&v).unwrap());
    }

    #[test]
    fn render_atem_produces_expected_wrapper() {
        let args = serde_json::json!({
            "location": "Paris",
            "include_alerts": true,
            "fallback": null
        });
        let s = render_atem_string("weather.get_forecast", &args);
        assert!(s.contains("<atem:function_calls>"));
        assert!(s.contains("<atem:invoke name=\"weather.get_forecast\">"));
        assert!(s.contains("<atem:parameter name=\"location\">Paris</atem:parameter>"));
        assert!(s.contains("<atem:parameter name=\"include_alerts\">true</atem:parameter>"));
        assert!(s.contains("<atem:parameter name=\"fallback\">null</atem:parameter>"));
    }
}
