// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 producer wire contract, without a GPU.
//!
//! `generate_gemma4` used to end a turn with a bare `done` that carried no
//! `finish_reason` or `prompt_tokens` and skipped the `commit_ready`
//! handshake, so serve never built an HTTP terminal: non-stream requests got
//! 500 "generation completed without a response body" and streams ended with
//! no finish/usage chunk. These tests drive the producer's real emission
//! helpers (route start, [`Gemma4TurnText`], [`gemma4_commit_terminal`]) into
//! a buffer and check the exact wire a serve client folds.

use hipfire_engine::terminal::{
    activate_terminal_control, apply_terminal_control, clear_terminal_control,
    set_active_attempt_id, terminal_control,
};
use hipfire_generate::ar::{emit_generation_start, GenerationRoute, GenerationRouteScope};
use hipfire_generate::dense::{
    gemma4_commit_terminal, gemma4_finish_reason, gemma4_pending_done, gemma4_template_messages,
    gemma4_tools_unsupported_reason, Gemma4TurnStats, Gemma4TurnText,
};
use hipfire_runtime::prompt_frame::{Message, Role, ToolCall};
use serde_json::{json, Value};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

/// The terminal-control singleton is process-global; serialize every test.
fn begin_test() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_terminal_control();
    set_active_attempt_id(0);
    guard
}

/// Answer `decision` ("commit" / "abort") once the turn reports ready.
fn spawn_client(id: &str, attempt: u64, decision: &'static str) -> std::thread::JoinHandle<()> {
    let id = id.to_owned();
    std::thread::spawn(move || {
        for _ in 0..2000 {
            let ready = terminal_control().mu.lock().ok().and_then(|g| {
                g.active
                    .as_ref()
                    .filter(|a| a.id == id && a.attempt_id == attempt)
                    .map(|a| a.ready)
            });
            if ready == Some(true) {
                apply_terminal_control(decision, &id, attempt);
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("gemma4 turn never reported commit_ready");
    })
}

fn stats(generated: usize) -> Gemma4TurnStats {
    Gemma4TurnStats {
        generated,
        prompt_tokens: 31,
        prefill_tokens: 24,
        cached_tokens: 7,
        prefill_ms: 12.0,
        decode_ms: 100.0,
        ttft_ms: 14.0,
        total_ms: 112.0,
    }
}

struct Turn {
    events: Vec<Value>,
    committed: bool,
}

/// One full producer turn: route start, `fragments` through the answer
/// emitter, then the two-phase terminal answered with `decision`.
fn run_turn(
    id: &str,
    attempt: u64,
    fragments: &[&str],
    tools: bool,
    ended_on_stop: bool,
    decision: &'static str,
) -> Turn {
    activate_terminal_control(id, attempt);
    set_active_attempt_id(attempt);
    let _route = GenerationRouteScope::enter(GenerationRoute::Gemma4Ar, id);
    let mut out = Vec::new();
    emit_generation_start(GenerationRoute::Gemma4Ar, &mut out, id, false);
    let mut text = Gemma4TurnText::new(false, 0, tools);
    for fragment in fragments {
        text.push(&mut out, id, fragment, || None)
            .expect("fragment routes");
    }
    text.finish(&mut out, id, "", ended_on_stop)
        .expect("turn finishes");
    let generated = fragments.len();
    let pending = gemma4_pending_done(
        id,
        gemma4_finish_reason(ended_on_stop, text.calls().len()),
        &stats(generated),
    );
    let client = spawn_client(id, attempt, decision);
    let committed = gemma4_commit_terminal(&mut out, id, pending, text.calls(), generated);
    client.join().unwrap();
    clear_terminal_control();
    set_active_attempt_id(0);
    let events = std::str::from_utf8(&out)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).expect("wire line is JSON"))
        .collect();
    Turn { events, committed }
}

fn of_type<'a>(events: &'a [Value], ty: &str) -> Vec<&'a Value> {
    events.iter().filter(|e| e["type"] == ty).collect()
}

fn without_type(event: &Value) -> Value {
    let mut event = event.clone();
    event.as_object_mut().unwrap().remove("type");
    event
}

#[test]
fn stop_turn_stages_a_commit_ready_with_finish_and_usage_then_the_same_done() {
    let _lock = begin_test();
    let turn = run_turn(
        "g4-stop",
        41,
        &["391", "\n\nExplanation."],
        false,
        true,
        "commit",
    );
    assert!(turn.committed);
    let types: Vec<&str> = turn
        .events
        .iter()
        .map(|e| e["type"].as_str().unwrap())
        .collect();
    assert_eq!(
        types,
        ["gen_start", "token", "token", "commit_ready", "done"]
    );

    let start = &turn.events[0];
    assert_eq!(start["id"], "g4-stop");
    assert_eq!(start["attempt_id"], 41);
    // Legacy contract: Gemma never advertises semantic v2.
    assert!(
        start.get("contract_version").is_none_or(Value::is_null),
        "{start}"
    );

    let text: String = of_type(&turn.events, "token")
        .iter()
        .map(|e| e["text"].as_str().unwrap())
        .collect();
    assert_eq!(text, "391\n\nExplanation.");

    let ready = of_type(&turn.events, "commit_ready")[0];
    assert_eq!(ready["finish_reason"], "stop");
    assert_eq!(ready["prompt_tokens"], 31);
    assert_eq!(ready["prefill_tokens"], 24);
    assert_eq!(ready["cached_tokens"], 7);
    assert_eq!(ready["tokens"], 2);
    assert_eq!(ready["attempt_id"], 41);
    assert!(ready.get("calls").is_none());
    // The client rejects a done whose payload diverges from the staged one.
    let done = of_type(&turn.events, "done")[0];
    assert_eq!(without_type(done), without_type(ready));
}

#[test]
fn truncated_turn_reports_length() {
    let _lock = begin_test();
    let turn = run_turn(
        "g4-length",
        42,
        &["The little", " cat"],
        false,
        false,
        "commit",
    );
    assert!(turn.committed);
    assert_eq!(
        of_type(&turn.events, "commit_ready")[0]["finish_reason"],
        "length"
    );
    assert_eq!(of_type(&turn.events, "done")[0]["finish_reason"], "length");
}

#[test]
fn native_tool_call_is_withheld_from_content_and_staged_as_calls() {
    let _lock = begin_test();
    let turn = run_turn(
        "g4-tool",
        43,
        &[
            "<|tool_call>",
            "call:get_temperature{city:",
            "<|\"|>Oslo<|\"|>}",
            "<tool_call|>",
        ],
        true,
        true,
        "commit",
    );
    assert!(turn.committed);
    assert!(
        of_type(&turn.events, "token").is_empty(),
        "{:?}",
        turn.events
    );
    let ready = of_type(&turn.events, "commit_ready")[0];
    assert_eq!(ready["finish_reason"], "tool_calls");
    assert_eq!(
        ready["calls"],
        json!([{"name": "get_temperature", "arguments": {"city": "Oslo"}}])
    );
    assert_eq!(
        without_type(of_type(&turn.events, "done")[0]),
        without_type(ready)
    );
}

#[test]
fn tool_call_cut_by_max_tokens_is_never_released() {
    let _lock = begin_test();
    let turn = run_turn(
        "g4-tool-cut",
        44,
        &[
            "Checking. ",
            "<|tool_call>call:get_temperature{city:<|\"|>Os",
        ],
        true,
        false,
        "commit",
    );
    let ready = of_type(&turn.events, "commit_ready")[0];
    assert_eq!(ready["finish_reason"], "length");
    assert!(ready.get("calls").is_none());
    let text: String = of_type(&turn.events, "token")
        .iter()
        .map(|e| e["text"].as_str().unwrap())
        .collect();
    assert_eq!(text, "Checking. ");
}

#[test]
fn tools_off_passes_tool_markup_through_unchanged() {
    let _lock = begin_test();
    let raw = "<|tool_call>call:f{}<tool_call|>";
    let turn = run_turn("g4-notools", 45, &[raw], false, true, "commit");
    let text: String = of_type(&turn.events, "token")
        .iter()
        .map(|e| e["text"].as_str().unwrap())
        .collect();
    assert_eq!(text, raw);
    assert_eq!(
        of_type(&turn.events, "commit_ready")[0]["finish_reason"],
        "stop"
    );
}

#[test]
fn aborted_turn_emits_the_aborted_pair_and_no_success_done() {
    let _lock = begin_test();
    let turn = run_turn("g4-abort", 46, &["partial"], false, true, "abort");
    assert!(!turn.committed);
    assert_eq!(of_type(&turn.events, "aborted").len(), 1);
    let dones = of_type(&turn.events, "done");
    assert_eq!(dones.len(), 1);
    assert_eq!(dones[0]["finish_reason"], "aborted");
}

#[test]
fn malformed_closed_tool_call_is_an_error_not_content() {
    let _lock = begin_test();
    let mut out = Vec::new();
    let mut text = Gemma4TurnText::new(false, 0, true);
    let error = text
        .push(
            &mut out,
            "g4-bad",
            "<|tool_call>call:f{city:<tool_call|>",
            || None,
        )
        .unwrap_err();
    assert!(!error.0.is_empty());
    assert!(out.is_empty(), "no partial call text reaches the wire");
}

#[test]
fn tools_refusal_names_the_missing_template_capability() {
    let tool_template = "{{ '<|tool_call>call:' }}";
    assert_eq!(
        gemma4_tools_unsupported_reason(Some(tool_template), true),
        None
    );
    assert_eq!(
        gemma4_tools_unsupported_reason(Some("{{ bos_token }}"), true),
        Some("this model's chat template has no tool-call format")
    );
    assert!(gemma4_tools_unsupported_reason(None, true)
        .unwrap()
        .contains("no chat template"));
    assert!(gemma4_tools_unsupported_reason(Some(tool_template), false)
        .unwrap()
        .contains("HIPFIRE_JINJA_CHAT=0"));
}

fn message(role: Role, content: &str) -> Message {
    Message {
        role,
        content: content.to_owned(),
        reasoning_content: None,
        name: None,
        rendered_name: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tool_plan: String::new(),
    }
}

/// The serve smoke's system + tools + call + result history, in the shape
/// the daemon receives it (OpenAI `arguments` still a JSON string).
fn smoke_history() -> Vec<Message> {
    let mut assistant = message(Role::Assistant, "");
    assistant.tool_calls = vec![ToolCall {
        id: Some("call_smoke_1".to_owned()),
        name: "get_temperature".to_owned(),
        arguments: Value::String(r#"{"city":"Oslo"}"#.to_owned()),
        rendered_body: None,
    }];
    let mut result = message(Role::Tool, r#"{"city":"Oslo","temperature_c":7}"#);
    result.tool_call_id = Some("call_smoke_1".to_owned());
    vec![
        message(Role::System, "You are a concise weather assistant."),
        message(
            Role::User,
            "What is the temperature in Oslo? Use get_temperature.",
        ),
        assistant,
        result,
    ]
}

#[test]
fn template_messages_carry_openai_function_calls_and_resolved_tool_names() {
    let values = gemma4_template_messages(&smoke_history()).unwrap();
    assert_eq!(
        values[2]["tool_calls"],
        json!([{
            "id": "call_smoke_1",
            "type": "function",
            "function": {"name": "get_temperature", "arguments": {"city": "Oslo"}},
        }])
    );
    assert_eq!(values[3]["name"], "get_temperature");
    assert_eq!(values[1].get("tool_calls"), Some(&json!([])));
    assert!(values[1].get("name").is_none());

    let mut bad = smoke_history();
    bad[2].tool_calls[0].arguments = Value::String("not json".to_owned());
    assert!(gemma4_template_messages(&bad).is_err());
}

/// The Gemma 4 template's tool-call and tool-response walk (verbatim
/// structure from the embedded template) renders the projected history; the
/// plain `Message` serialization has no `function` key and fails to render.
#[test]
fn projected_history_renders_through_the_gemma_tool_walk() {
    let template = concat!(
        "{%- for message in messages -%}",
        "{%- if message['role'] != 'tool' -%}",
        "<|turn>{{ message['role'] }}\n",
        "{%- if message.get('tool_calls') -%}",
        "{%- for tool_call in message.get('tool_calls') -%}",
        "{%- set function = tool_call['function'] -%}",
        "{{- '<|tool_call>call:' + function['name'] + '{' -}}",
        "{%- if function['arguments'] is mapping -%}",
        "{%- for key, value in function['arguments'] | dictsort -%}",
        "{{- key -}}:<|\"|>{{- value -}}<|\"|>",
        "{%- endfor -%}",
        "{%- else -%}{{- raise_exception('arguments must be a mapping') -}}",
        "{%- endif -%}",
        "{{- '}<tool_call|>' -}}",
        "{%- endfor -%}",
        "{%- endif -%}",
        "{{- message['content'] -}}<turn|>\n",
        "{%- else -%}",
        "{%- set ns = namespace(name=message.get('name') or 'unknown') -%}",
        "<|tool_response>response:{{ ns.name }}{{ message['content'] }}<tool_response|>",
        "{%- endif -%}",
        "{%- endfor -%}",
    );
    // `bos_token` is explicit, so the tokenizer is never consulted.
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hf_json(
        r#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"added_tokens":[]}"#,
    )
    .expect("minimal tokenizer");
    let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
        tokenizer: &tokenizer,
        template,
        system: None,
        user: "",
        enable_thinking: false,
        bos_token: Some("<bos>"),
        reasoning_strength: None,
        reasoning_effort: None,
    };
    let history = smoke_history();
    let rendered = frame
        .render_message_values(gemma4_template_messages(&history).unwrap(), None, None)
        .unwrap();
    assert!(
        rendered.contains("<|tool_call>call:get_temperature{city:<|\"|>Oslo<|\"|>}<tool_call|>"),
        "{rendered}"
    );
    assert!(
        rendered.contains("<|tool_response>response:get_temperature{\"city\":\"Oslo\""),
        "{rendered}"
    );
    assert!(frame.render_messages(&history, None, None).is_err());
}
