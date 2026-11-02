// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 AR vs native-MTP client semantics for `tools` and `stop`.
//!
//! The two Qwen4 routes classify committed tokens with different producers:
//! `generate_qwen4_ar` drives `QwenArSemanticProducer` (via
//! `generate_ar_with_forward`), native MTP drives `Qwen35Emit` (the Qwen4
//! carrier's spec emitter, tool grammar off). Greedy MTP commits the AR argmax
//! stream, so for the wire to stay byte-identical with MTP on or off the two
//! producers must turn the same token stream into the same visible text,
//! reasoning, tool calls, finish reason and stop position.

use hipfire_arch_qwen35::spec_emit::Qwen35Emit;
use hipfire_generate::ar::QwenArSemanticProducer;
use hipfire_runtime::prompt_frame::{AssistantPrefix, ThinkMode};
use hipfire_runtime::spec::{ClientEvent, SpecEmitCtx};
use hipfire_runtime::tokenizer::Tokenizer;

const IM_END: u32 = 1;
const EOS: u32 = 9;

/// GPT-2 byte_to_unicode (mirrors the production tokenizer table).
fn byte_to_gpt2_char(b: u8) -> char {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend((b'!' as u32)..=(b'~' as u32));
    bs.extend(0xA1u32..=0xACu32);
    bs.extend(0xAEu32..=0xFFu32);
    let mut cs = bs.clone();
    let mut n = 0u32;
    for byte in 0u32..=255 {
        if !bs.contains(&byte) {
            bs.push(byte);
            cs.push(256 + n);
            n += 1;
        }
    }
    let at = bs.iter().position(|&x| x == b as u32).unwrap();
    char::from_u32(cs[at]).unwrap()
}

/// Byte-level BPE (no merges) plus the ChatML / think specials: every text
/// encodes one token per byte, so stops and tool markup straddle tokens.
fn tokenizer() -> Tokenizer {
    let mut vocab = vec![
        r#""<|im_start|>": 0"#.to_owned(),
        r#""<|im_end|>": 1"#.to_owned(),
        r#""<think>": 2"#.to_owned(),
        r#""</think>": 3"#.to_owned(),
        r#""<|endoftext|>": 9"#.to_owned(),
    ];
    for b in 0u32..=255 {
        let ch = serde_json::to_string(&byte_to_gpt2_char(b as u8).to_string()).unwrap();
        vocab.push(format!("{ch}: {}", 100 + b));
    }
    let json = format!(
        r#"{{"model": {{"type": "BPE", "vocab": {{ {} }}, "merges": []}},
            "added_tokens": [
                {{"id": 0, "content": "<|im_start|>", "special": true}},
                {{"id": 1, "content": "<|im_end|>", "special": true}},
                {{"id": 2, "content": "<think>", "special": true}},
                {{"id": 3, "content": "</think>", "special": true}},
                {{"id": 9, "content": "<|endoftext|>", "special": true}}
            ]}}"#,
        vocab.join(", ")
    );
    Tokenizer::from_hf_json(&json).expect("test tokenizer")
}

/// What the client observes for one turn.
#[derive(Debug, PartialEq)]
struct Turn {
    text: String,
    reasoning: String,
    calls: Vec<(String, serde_json::Value)>,
    finish_reason: &'static str,
    /// Tokens committed before the producer ended the turn.
    committed: usize,
}

fn calls_of(calls: &[hipfire_runtime::prompt_frame::ToolCall]) -> Vec<(String, serde_json::Value)> {
    calls
        .iter()
        .map(|c| (c.name.clone(), c.arguments.clone()))
        .collect()
}

/// `generate_ar_with_forward`'s per-token classify + stop rule.
fn ar_turn(ids: &[u32], tools: bool, stop: &[String], open_think: bool) -> Turn {
    let tok = tokenizer();
    let mut producer =
        QwenArSemanticProducer::new_with_tool_protocol("p", open_think, tools).with_stop(stop);
    let (mut sink, mut conversation, mut streamed, mut seq_pos) =
        (Vec::new(), Vec::new(), Vec::new(), 0usize);
    let mut committed = 0;
    for &id in ids {
        let mut bytes = Vec::new();
        tok.decode_token_bytes_into(id, &mut bytes);
        let stopped = producer
            .commit_and_observe(
                &mut sink,
                &mut conversation,
                &mut streamed,
                &mut seq_pos,
                id,
                &bytes,
            )
            .expect("classify");
        committed += 1;
        if stopped || id == EOS || tok.is_terminator(id) {
            break;
        }
    }
    let (finish, _) = producer.finish(&mut sink, false).expect("finish");
    let (mut text, mut reasoning) = (String::new(), String::new());
    for line in String::from_utf8(sink).unwrap().lines() {
        let event: serde_json::Value = serde_json::from_str(line).unwrap();
        match event["type"].as_str() {
            Some("token") => text.push_str(event["text"].as_str().unwrap()),
            Some("reasoning") => reasoning.push_str(event["text"].as_str().unwrap()),
            _ => {}
        }
    }
    Turn {
        text,
        reasoning,
        calls: calls_of(&finish.wire_tool_calls),
        finish_reason: finish.finish_reason,
        committed,
    }
}

/// Native MTP's emitter, built exactly as `generate_dflash` builds it for a
/// Qwen4 load (`enable_grammar: false`).
fn mtp_turn(
    ids: &[u32],
    tools: Option<&[serde_json::Value]>,
    stop: &[String],
    open_think: bool,
) -> Turn {
    let tok = tokenizer();
    let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
        tokenizer: &tok,
        eos: EOS,
        im_end: Some(IM_END),
        tools,
        enable_grammar: false,
        stop: stop.to_vec(),
        max_think: 0,
        max_tokens: 4096,
        assistant_prefix: if open_think {
            AssistantPrefix::OpenThink
        } else {
            AssistantPrefix::Plain
        },
        think_mode: ThinkMode::NonThink,
        decoded_vocab: None,
    });
    let mut events = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        let outcome = if i == 0 {
            emit.begin(id)
        } else {
            emit.observe(id)
        };
        events.extend(outcome.events);
        if outcome.stop.is_some() {
            break;
        }
    }
    let committed = emit.streamed_tokens().len();
    let finish = emit.finish();
    events.extend(finish.events);
    let (mut text, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
    for event in events {
        match event {
            ClientEvent::Token(t) => text.push_str(&t),
            ClientEvent::Reasoning(t) => reasoning.push_str(&t),
            ClientEvent::ToolCalls(c) => calls = calls_of(&c),
            _ => {}
        }
    }
    Turn {
        text,
        reasoning,
        calls,
        finish_reason: finish.finish_reason,
        committed,
    }
}

fn tool_defs() -> Vec<serde_json::Value> {
    vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    })]
}

/// Encode the model output, then end the turn with `<|im_end|>`.
fn stream(text: &str) -> Vec<u32> {
    let mut ids = tokenizer().encode(text);
    ids.push(IM_END);
    ids
}

/// Both routes agree; returns the shared turn for content assertions.
fn both(text: &str, tools: bool, stop: &[&str], open_think: bool) -> Turn {
    let ids = stream(text);
    let stop: Vec<String> = stop.iter().map(|s| (*s).to_owned()).collect();
    let defs = tool_defs();
    let ar = ar_turn(&ids, tools, &stop, open_think);
    let mtp = mtp_turn(&ids, tools.then_some(defs.as_slice()), &stop, open_think);
    assert_eq!(ar, mtp, "AR vs MTP diverged for {text:?} stop={stop:?}");
    ar
}

const XML_CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

#[test]
fn tool_call_turn_is_identical_on_ar_and_mtp() {
    let turn = both(&format!("Checking.\n\n{XML_CALL}"), true, &[], false);
    assert_eq!(turn.finish_reason, "tool_calls");
    assert_eq!(
        turn.calls,
        vec![(
            "get_weather".to_owned(),
            serde_json::json!({"city": "Paris"})
        )]
    );
    assert!(!turn.text.contains("<tool_call>"), "{:?}", turn.text);

    // Reasoning before the call (thinking-on template opens <think>).
    let turn = both(
        &format!("need the weather\n</think>\n\n{XML_CALL}"),
        true,
        &[],
        true,
    );
    assert_eq!(turn.finish_reason, "tool_calls");
    assert_eq!(turn.reasoning.trim(), "need the weather");
}

#[test]
fn tool_markup_without_tools_is_plain_content_on_both_routes() {
    // `tools` absent: neither producer parses calls (AR used to always parse).
    let turn = both(XML_CALL, false, &[], false);
    assert_eq!(turn.finish_reason, "stop");
    assert!(turn.calls.is_empty());
    assert!(turn.text.contains("<tool_call>"));
}

#[test]
fn stop_sequences_truncate_identically_on_ar_and_mtp() {
    // Spans byte tokens; neither the stop text nor anything after it leaks.
    let turn = both("Hello END tail", false, &["END"], false);
    assert_eq!((turn.text.as_str(), turn.finish_reason), ("Hello ", "stop"));
    assert_eq!(turn.committed, "Hello END".len());

    // Earliest match across several stops wins.
    let turn = both("one, two; three", false, &["three", ";", "zzz"], false);
    assert_eq!(turn.text, "one, two");

    // A possible-but-false prefix is released, and an unmatched stop is inert.
    let turn = both("ENx END", false, &["END"], false);
    assert_eq!(turn.text, "ENx ");
    let turn = both("abc EN", false, &["END"], false);
    assert_eq!((turn.text.as_str(), turn.finish_reason), ("abc EN", "stop"));
    assert_eq!(turn.committed, "abc EN".len() + 1, "ran to <|im_end|>");

    // Reasoning is not matched; the answer is.
    let turn = both("a\nb</think>answer\nmore", false, &["\n"], true);
    assert_eq!(
        (turn.reasoning.as_str(), turn.text.as_str()),
        ("a\nb", "answer")
    );
}

#[test]
fn stop_inside_a_tool_call_drops_the_call_on_both_routes() {
    let turn = both(&format!("pre {XML_CALL}"), true, &["Paris"], false);
    assert_eq!(turn.finish_reason, "stop");
    assert!(turn.calls.is_empty());
    assert_eq!(turn.text.trim_end(), "pre");

    // A stop after a completed call keeps the call.
    let turn = both(
        &format!("{XML_CALL}\nDONE trailing"),
        true,
        &["DONE"],
        false,
    );
    assert_eq!(turn.finish_reason, "tool_calls");
    assert_eq!(turn.calls.len(), 1);
}
