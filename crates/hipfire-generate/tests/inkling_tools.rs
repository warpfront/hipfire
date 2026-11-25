// SPDX-License-Identifier: Apache-2.0
//! Inkling-2-Mini structured tools: golden prompts, the invocation parser and
//! the terminal decision.
//!
//! `tools.json` expectations were produced by transformers 5.8.0
//! `render_jinja_template(..., tools=..., add_generation_prompt=True)` over the
//! byte-exact embedded template; `hipfire_messages` is the same conversation in
//! the daemon's `Message` shape. The tokenizer is a byte-level stand-in with
//! the real Inkling special tokens at their real ids, so every byte of an
//! invocation body is its own token.

use hipfire_generate::dense::{inkling_template_messages, inkling_terminal, inkling_tool_names};
use hipfire_runtime::emit_text::{MessageFrameMarkers, MessageFrameRouter, ThinkRouteEvent};
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message};
use hipfire_runtime::tokenizer::Tokenizer;

const TEMPLATE: &str =
    include_str!("../../hipfire-runtime/tests/fixtures/inkling2/chat_template.jinja");
const CASES: &str = include_str!("../../hipfire-runtime/tests/fixtures/inkling2/tools.json");

const SPECIALS: &[(u32, &str)] = &[
    (199999, "<|endoftext|>"),
    (200000, "<|message_user|>"),
    (200001, "<|message_model|>"),
    (200002, "<|message_system|>"),
    (200003, "<|message_tool|>"),
    (200004, "<|content_text|>"),
    (200006, "<|content_model_end_sampling|>"),
    (200008, "<|content_thinking|>"),
    (200010, "<|end_message|>"),
    (200024, "<|content_xml|>"),
    (200049, "<|content_invoke_tool_json|>"),
    (200057, "<|content_invoke_tool_text|>"),
];
const BYTE_BASE: u32 = 100;

fn byte_char(b: u8) -> char {
    let mut bs: Vec<u32> = (0x21..=0x7E)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let i = bs.iter().position(|x| *x == b as u32).unwrap();
    char::from_u32(cs[i]).unwrap()
}

fn tokenizer() -> Tokenizer {
    let mut vocab = serde_json::Map::new();
    for b in 0u32..256 {
        vocab.insert(byte_char(b as u8).to_string(), (BYTE_BASE + b).into());
    }
    let mut added = Vec::new();
    for &(id, content) in SPECIALS {
        vocab.insert(content.into(), id.into());
        added.push(serde_json::json!({"id": id, "content": content, "special": true}));
    }
    let j = serde_json::json!({
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": added,
    });
    Tokenizer::from_hf_json(&j.to_string()).expect("byte-level stand-in tokenizer")
}

fn cases() -> Vec<serde_json::Value> {
    serde_json::from_str(CASES).unwrap()
}

fn render(
    t: &Tokenizer,
    messages: &[Message],
    tools: &[serde_json::Value],
) -> Result<String, String> {
    let frame = JinjaChatFrame {
        tokenizer: t,
        template: TEMPLATE,
        system: None,
        user: "",
        enable_thinking: true,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    };
    inkling_template_messages(messages)
        .and_then(|values| frame.render_message_values(values, Some(tools), None))
}

#[test]
fn tool_prompts_match_hf_apply_chat_template() {
    let t = tokenizer();
    let cases = cases();
    assert!(cases.len() >= 3);
    for c in &cases {
        let messages: Vec<Message> = serde_json::from_value(c["hipfire_messages"].clone()).unwrap();
        let tools = c["tools"].as_array().unwrap();
        let got = render(&t, &messages, tools).unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        assert_eq!(got, c["expected"].as_str().unwrap(), "case {}", c["name"]);
    }
}

#[test]
fn tool_result_without_a_resolvable_name_is_refused() {
    let t = tokenizer();
    let c = &cases()[1];
    let mut messages: Vec<Message> = serde_json::from_value(c["hipfire_messages"].clone()).unwrap();
    let last = messages.last_mut().unwrap();
    last.tool_call_id = Some("call_unknown".into());
    let err = render(&t, &messages, c["tools"].as_array().unwrap()).unwrap_err();
    assert!(err.contains("tool_call_id"), "{err}");
}

#[test]
fn tool_names_follow_the_template() {
    let tools = cases()[0]["tools"].as_array().unwrap().clone();
    assert_eq!(inkling_tool_names(&tools), ["get_weather", "read_file"]);
    let bare = [serde_json::json!({"name": "bare", "parameters": {}})];
    assert_eq!(inkling_tool_names(&bare), ["bare"]);
}

struct Turn {
    router: MessageFrameRouter,
    events: Vec<ThinkRouteEvent>,
}

/// Feed `text` (specials atomic, every other byte its own token) through a
/// tools router declaring `get_weather` and `read_file`.
fn route(t: &Tokenizer, text: &str) -> Turn {
    let markers = MessageFrameMarkers::from_tokenizer(t).unwrap();
    let mut router =
        MessageFrameRouter::with_tools(markers, vec!["get_weather".into(), "read_file".into()]);
    let mut events = Vec::new();
    for id in t.encode(text) {
        router.push_token(t, id, &mut events);
    }
    router.finish_into(&mut events);
    Turn { router, events }
}

fn invoke(name: &str, body: &str) -> String {
    format!("<|message_model|>{name}<|content_invoke_tool_json|>{body}<|end_message|>")
}

fn visible(turn: &Turn) -> String {
    turn.events
        .iter()
        .map(|e| match e {
            ThinkRouteEvent::Reasoning(s) | ThinkRouteEvent::Content(s) => s.as_str(),
        })
        .collect()
}

#[test]
fn a_valid_call_is_buffered_validated_and_released_on_eos() {
    let t = tokenizer();
    let text = format!(
        "<|message_model|><|content_thinking|>Need weather.<|end_message|>{}",
        invoke(
            "get_weather",
            r#"{"name":"get_weather","args":{"city":"Paris","units":"c"}}"#
        )
    );
    let turn = route(&t, &text);
    // The body never reaches an event; only the reasoning does.
    assert_eq!(visible(&turn), "Need weather.");
    assert_eq!(turn.router.tool_error(), None);
    let (finish, calls) = inkling_terminal(true, &turn.router).unwrap();
    assert_eq!(finish, "tool_calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        calls[0].arguments,
        serde_json::json!({"city": "Paris", "units": "c"})
    );
    // A length cut releases nothing, even after a complete call.
    let (finish, calls) = inkling_terminal(false, &turn.router).unwrap();
    assert_eq!((finish, calls.len()), ("length", 0));
}

#[test]
fn parallel_calls_keep_order() {
    let t = tokenizer();
    let text = format!(
        "{}{}",
        invoke(
            "read_file",
            r#"{"name":"read_file","args":{"path":"a.rs"}}"#
        ),
        invoke(
            "read_file",
            r#"{"name":"read_file","args":{"path":"b.rs"}}"#
        )
    );
    let turn = route(&t, &text);
    let (finish, calls) = inkling_terminal(true, &turn.router).unwrap();
    assert_eq!(finish, "tool_calls");
    let paths: Vec<&str> = calls
        .iter()
        .map(|c| c.arguments["path"].as_str().unwrap())
        .collect();
    assert_eq!(paths, ["a.rs", "b.rs"]);
}

#[test]
fn malformed_calls_fail_closed() {
    let t = tokenizer();
    let bad = [
        ("unknown tool", invoke("rm", r#"{"name":"rm","args":{}}"#)),
        (
            "name differs from header",
            invoke("get_weather", r#"{"name":"read_file","args":{"path":"x"}}"#),
        ),
        (
            "string args",
            invoke(
                "get_weather",
                r#"{"name":"get_weather","args":"{\"city\":\"x\"}"}"#,
            ),
        ),
        (
            "array args",
            invoke("get_weather", r#"{"name":"get_weather","args":[1]}"#),
        ),
        (
            "extra key",
            invoke(
                "get_weather",
                r#"{"name":"get_weather","args":{},"id":"c"}"#,
            ),
        ),
        (
            "invalid json",
            invoke("get_weather", r#"{"name":"get_weather","args":{"#),
        ),
        ("non-object body", invoke("get_weather", "[]")),
        (
            "text invocation",
            "<|message_model|>get_weather<|content_invoke_tool_text|>city=Paris<|end_message|>"
                .to_owned(),
        ),
        (
            "invocation without header",
            "<|content_invoke_tool_json|>{}<|end_message|>".to_owned(),
        ),
        (
            "next message before close",
            "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\"\
             <|message_model|><|content_text|>hi<|end_message|>"
                .to_owned(),
        ),
    ];
    for (label, text) in bad {
        let turn = route(&t, &text);
        assert!(turn.router.tool_error().is_some(), "{label}: accepted");
        assert!(inkling_terminal(true, &turn.router).is_err(), "{label}");
        assert!(inkling_terminal(false, &turn.router).is_err(), "{label}");
        // Nothing of the invocation leaks into the visible channels.
        assert!(!visible(&turn).contains("get_weather"), "{label}: leaked");
    }
}

#[test]
fn every_byte_prefix_of_a_call_is_never_released() {
    // Cut the turn after each byte of a valid invocation: EOS there fails
    // closed, a length cut releases nothing, and no prefix leaks.
    let t = tokenizer();
    let full = invoke(
        "get_weather",
        r#"{"name":"get_weather","args":{"city":"Zürich"}}"#,
    );
    let ids = t.encode(&full);
    let markers = MessageFrameMarkers::from_tokenizer(&t).unwrap();
    for cut in 0..ids.len() {
        let mut router =
            MessageFrameRouter::with_tools(markers.clone(), vec!["get_weather".into()]);
        let mut events = Vec::new();
        for &id in &ids[..cut] {
            router.push_token(&t, id, &mut events);
        }
        router.finish_into(&mut events);
        assert!(events.is_empty(), "cut {cut}: {events:?}");
        assert_eq!(
            inkling_terminal(false, &router).map(|(f, c)| (f, c.len())),
            Ok(("length", 0)),
            "cut {cut}"
        );
        if router.in_tool_call() {
            assert!(inkling_terminal(true, &router).is_err(), "cut {cut}");
        }
    }
    // The whole invocation validates.
    let turn = route(&t, &full);
    assert_eq!(inkling_terminal(true, &turn.router).unwrap().1.len(), 1);
}

#[test]
fn without_tools_an_invocation_stays_content() {
    let t = tokenizer();
    let markers = MessageFrameMarkers::from_tokenizer(&t).unwrap();
    let mut router = MessageFrameRouter::new(markers);
    let mut events = Vec::new();
    for id in t.encode(&invoke(
        "get_weather",
        r#"{"name":"get_weather","args":{}}"#,
    )) {
        router.push_token(&t, id, &mut events);
    }
    router.finish_into(&mut events);
    let turn = Turn { router, events };
    assert_eq!(visible(&turn), r#"{"name":"get_weather","args":{}}"#);
    let (finish, calls) = inkling_terminal(true, &turn.router).unwrap();
    assert_eq!((finish, calls.len()), ("stop", 0));
}
