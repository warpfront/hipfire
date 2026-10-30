// SPDX-License-Identifier: Apache-2.0
//! Raon-1-mini chat template / tool-call / reasoning grammar checks.
//!
//! Expected bytes in `fixtures/raon_tokenizer/template.json` were produced by
//! `transformers` `apply_chat_template` (add_generation_prompt=True) over the
//! official `chat_template.jinja` (copied beside it). The tokenizer here is a
//! byte-level stand-in: the template never reads `bos_token`, so rendering is
//! tokenizer-independent.

use hipfire_runtime::emit_text::{ThinkOutputRouter, ThinkRouteEvent};
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::tool_call::{Qwen35XmlParser, ToolCallParser};

const TEMPLATE: &str = include_str!("fixtures/raon_tokenizer/chat_template.jinja");
const CASES: &str = include_str!("fixtures/raon_tokenizer/template.json");

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
    vocab.insert("<|im_start|>".into(), 0.into());
    vocab.insert("<|im_end|>".into(), 1.into());
    vocab.insert("<think>".into(), 2.into());
    vocab.insert("</think>".into(), 3.into());
    for b in 0u32..256 {
        vocab.insert(byte_char(b as u8).to_string(), (100 + b).into());
    }
    let j = serde_json::json!({
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [
            {"id": 0, "content": "<|im_start|>", "special": true},
            {"id": 1, "content": "<|im_end|>", "special": true},
            {"id": 2, "content": "<think>", "special": true},
            {"id": 3, "content": "</think>", "special": true}
        ]
    });
    Tokenizer::from_hf_json(&j.to_string()).expect("byte-level stand-in tokenizer")
}

fn render(t: &Tokenizer, case: &serde_json::Value) -> Result<String, String> {
    let messages: Vec<Message> = serde_json::from_value(case["messages"].clone()).unwrap();
    let tools: Option<Vec<serde_json::Value>> = case["tools"].as_array().cloned();
    let frame = JinjaChatFrame {
        tokenizer: t,
        template: TEMPLATE,
        system: None,
        user: "",
        enable_thinking: case["enable_thinking"].as_bool().unwrap(),
        bos_token: Some(""),
        reasoning_strength: None,
        reasoning_effort: None,
    };
    frame.render_messages(&messages, tools.as_deref(), None)
}

#[test]
fn raon_template_matches_hf_apply_chat_template() {
    let t = tokenizer();
    let cases: Vec<serde_json::Value> = serde_json::from_str(CASES).unwrap();
    assert!(cases.len() >= 10);
    for c in &cases {
        let got = render(&t, c).unwrap_or_else(|e| panic!("case {} failed: {e}", c["name"]));
        let want = c["expected"].as_str().unwrap();
        assert_eq!(got, want, "case {} not byte-identical to HF", c["name"]);
    }
}

#[test]
fn raon_template_generation_suffixes() {
    let t = tokenizer();
    let cases: Vec<serde_json::Value> = serde_json::from_str(CASES).unwrap();
    for c in &cases {
        let got = render(&t, c).unwrap();
        let on = c["enable_thinking"].as_bool().unwrap();
        let want = if on {
            "<|im_start|>assistant\n<think>\n"
        } else {
            "<|im_start|>assistant\n<think></think>\n\n"
        };
        assert!(
            got.ends_with(want),
            "case {}: bad suffix {:?}",
            c["name"],
            &got[got.len().saturating_sub(50)..]
        );
        assert!(
            !got.starts_with("<|endoftext|>"),
            "no BOS in rendered prompt"
        );
    }
}

#[test]
fn raon_template_tool_block_is_xml() {
    let t = tokenizer();
    let cases: Vec<serde_json::Value> = serde_json::from_str(CASES).unwrap();
    let c = cases.iter().find(|c| c["name"] == "tool_history").unwrap();
    let got = render(&t, c).unwrap();
    assert!(
        got.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"")
    );
    assert!(got.contains(
        "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n서울\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=add>\n"
    ));
    assert!(got.contains("<|im_start|>user\n<tool_response>\n{\"temp\": 21}\n</tool_response>\n<tool_response>\n3\n</tool_response><|im_end|>\n"));
}

#[test]
fn raon_template_toolcall_xml_parser_roundtrip() {
    let t = tokenizer();
    let cases: Vec<serde_json::Value> = serde_json::from_str(CASES).unwrap();
    let c = cases.iter().find(|c| c["name"] == "tool_history").unwrap();
    let got = render(&t, c).unwrap();
    // Take the assistant turn exactly as the template wrote it.
    let start = got.find("<|im_start|>assistant\n").unwrap();
    let end = got[start..].find("<|im_end|>").unwrap() + start;
    let body = got[start + "<|im_start|>assistant\n".len()..end]
        .strip_prefix("<think></think>\n\n")
        .unwrap();
    let r = Qwen35XmlParser::new().parse(body);
    assert_eq!(r.prose.as_deref(), Some("Let me check."));
    assert_eq!(r.tool_calls.len(), 2);
    assert_eq!(r.tool_calls[0].name, "get_weather");
    assert_eq!(r.tool_calls[0].arguments["city"], "서울");
    assert_eq!(r.tool_calls[0].arguments["unit"], "celsius");
    assert_eq!(r.tool_calls[1].name, "add");
    assert_eq!(r.tool_calls[1].arguments["a"], "1");
    assert!(r.tool_calls.iter().all(|c| !c.repaired));
}

#[test]
fn raon_template_toolcall_multiline_param_and_no_prose() {
    let text = "<tool_call>\n<function=write>\n<parameter=path>\n/a/b.txt\n</parameter>\n<parameter=body>\nline1\nline2 <b>x</b>\n</parameter>\n</function>\n</tool_call>";
    let r = Qwen35XmlParser::new().parse(text);
    assert_eq!(r.prose, None);
    assert_eq!(r.tool_calls.len(), 1);
    assert_eq!(r.tool_calls[0].arguments["path"], "/a/b.txt");
    assert_eq!(r.tool_calls[0].arguments["body"], "line1\nline2 <b>x</b>");
}

fn route(started: bool, chunks: &[&str]) -> (String, String) {
    let mut r = ThinkOutputRouter::new(started);
    let mut ev = Vec::new();
    for c in chunks {
        r.push_into(c, &mut ev);
    }
    r.finish_into(&mut ev);
    let (mut reasoning, mut content) = (String::new(), String::new());
    for e in ev {
        match e {
            ThinkRouteEvent::Reasoning(s) => reasoning.push_str(&s),
            ThinkRouteEvent::Content(s) => content.push_str(&s),
        }
    }
    (reasoning, content)
}

#[test]
fn raon_template_reasoning_started_in_think() {
    // Prompt ended with `<think>\n`; the model emits reasoning then `</think>`.
    let full = "Let me think.\n</think>\n\nThe answer is 4.";
    let (r, c) = route(true, &[full]);
    assert_eq!(r, "Let me think.\n");
    assert_eq!(c, "The answer is 4.");
    // Chunk-boundary invariance, including splitting inside the closer.
    let chunks: Vec<&str> = full
        .split_inclusive(|ch| ch == '<' || ch == '/' || ch == 'k')
        .collect();
    assert_eq!(route(true, &chunks), (r, c));
    // No closer: everything stays reasoning.
    let (r, c) = route(true, &["still thinking"]);
    assert_eq!((r.as_str(), c.as_str()), ("still thinking", ""));
}

#[test]
fn raon_template_reasoning_empty_think_prefix_is_content() {
    // enable_thinking=false prompt ends `<think></think>\n\n`: generation has no think span.
    let (r, c) = route(false, &["plain answer"]);
    assert_eq!((r.as_str(), c.as_str()), ("", "plain answer"));
}
