// SPDX-License-Identifier: Apache-2.0
//! Message-framed reasoning protocol (`ReasoningContract::MessageFrames`,
//! Inkling-2-Mini): token-indexed routing and the template's effort control.
//!
//! `fixtures/inkling2/chat_template.jinja` is the byte-exact
//! `tokenizer_config.chat_template` embedded in `inkling2-mini-mq4.hfq`
//! (sha256 `53a9035f…cc16dc`). Expected renders in
//! `fixtures/inkling2/template.json` were produced by transformers 5.8.0
//! `render_jinja_template(..., add_generation_prompt=True)` over that file.
//!
//! The tokenizer is a byte-level stand-in carrying the real Inkling special
//! tokens at their real ids. With no merges every byte is its own token, so a
//! multi-byte character always spans several tokens.

use hipfire_runtime::emit_text::{MessageFrameMarkers, MessageFrameRouter, ThinkRouteEvent};
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message};
use hipfire_runtime::tokenizer::Tokenizer;

const TEMPLATE: &str = include_str!("fixtures/inkling2/chat_template.jinja");
const CASES: &str = include_str!("fixtures/inkling2/template.json");

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
    (200049, "<|content_invoke_tool_json|>"),
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

fn special(name: &str) -> u32 {
    SPECIALS.iter().find(|(_, s)| *s == name).unwrap().0
}

/// Ordinary byte tokens spelling `text`, never special ids.
fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(|b| BYTE_BASE + b as u32).collect()
}

struct Turn {
    events: Vec<ThinkRouteEvent>,
}

impl Turn {
    fn reasoning(&self) -> String {
        self.channel(true)
    }
    fn content(&self) -> String {
        self.channel(false)
    }
    fn channel(&self, reasoning: bool) -> String {
        self.events
            .iter()
            .filter_map(|e| match (e, reasoning) {
                (ThinkRouteEvent::Reasoning(t), true) | (ThinkRouteEvent::Content(t), false) => {
                    Some(t.as_str())
                }
                _ => None,
            })
            .collect()
    }
}

/// Feed `ids` one token at a time, as the decode loop does, then finish.
fn route(t: &Tokenizer, ids: &[u32]) -> Turn {
    let markers = MessageFrameMarkers::from_tokenizer(t).expect("inkling markers");
    let mut router = MessageFrameRouter::new(markers);
    let mut events = Vec::new();
    for &id in ids {
        router.push_token(t, id, &mut events);
    }
    router.finish_into(&mut events);
    Turn { events }
}

fn assert_no_markers(turn: &Turn) {
    for e in &turn.events {
        let (ThinkRouteEvent::Reasoning(s) | ThinkRouteEvent::Content(s)) = e;
        assert!(!s.is_empty(), "empty event: {:?}", turn.events);
        for &(_, name) in SPECIALS {
            assert!(!s.contains(name), "marker {name} leaked into {e:?}");
        }
    }
}

/// The shape of the gfx1201 battery's arithmetic answer (run-3 rows.json),
/// with a multiplication sign that spans two byte tokens.
const THINK: &str = "Compute: 60 mph × 2.5 h = 150 miles. Final number 210 miles.";
const ANSWER: &str = "Distance = speed × time\n\nTotal: **210 miles** 🚗";

fn framed(kind: &str, body: &str) -> String {
    format!("<|message_model|>{kind}{body}<|end_message|>")
}

#[test]
fn markers_resolve_from_special_tokens_only() {
    let t = tokenizer();
    let m = MessageFrameMarkers::from_tokenizer(&t).unwrap();
    assert_eq!(
        (m.message_open, m.message_close, m.reasoning, m.text),
        (200001, 200010, 200008, 200004)
    );
    assert!(m.protocol.windows(2).all(|w| w[0] < w[1]));
    assert!(m
        .protocol
        .contains(&special("<|content_model_end_sampling|>")));
    assert!(m.protocol.contains(&special("<|message_user|>")));
    assert!(!m.protocol.contains(&m.message_open));

    let mut vocab = serde_json::Map::new();
    vocab.insert("a".into(), 0.into());
    vocab.insert("<|message_model|>".into(), 1.into());
    let qwenish = Tokenizer::from_hf_json(
        &serde_json::json!({
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": [{"id": 1, "content": "<|message_model|>", "special": true}],
        })
        .to_string(),
    )
    .unwrap();
    assert_eq!(MessageFrameMarkers::from_tokenizer(&qwenish), None);
}

#[test]
fn thinking_then_answer_split_on_protocol_tokens() {
    let t = tokenizer();
    let text = format!(
        "{}{}",
        framed("<|content_thinking|>", THINK),
        framed("<|content_text|>", ANSWER)
    );
    let ids = t.encode(&text);
    assert!(ids.contains(&special("<|content_thinking|>")));
    let turn = route(&t, &ids);
    assert_eq!(turn.reasoning(), THINK);
    assert_eq!(turn.content(), ANSWER);
    assert_no_markers(&turn);
    // The first event is reasoning: no content leaks ahead of the answer frame.
    assert!(matches!(turn.events[0], ThinkRouteEvent::Reasoning(_)));
}

#[test]
fn split_characters_are_held_until_complete() {
    let t = tokenizer();
    let body = "é中🎉";
    let ids = t.encode(&framed("<|content_thinking|>", body));
    // Byte-level stand-in: one token per byte, so each character spans tokens.
    assert_eq!(ids.len(), 3 + body.len());
    let turn = route(&t, &ids);
    assert_eq!(turn.reasoning(), body);
    // One event per completed character, never a replacement character.
    let pieces: Vec<&str> = turn
        .events
        .iter()
        .map(|e| match e {
            ThinkRouteEvent::Reasoning(s) => s.as_str(),
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    assert_eq!(pieces, ["é", "中", "🎉"]);
}

#[test]
fn marker_inside_a_character_keeps_bytes_in_their_span() {
    let t = tokenizer();
    // "中" is E4 B8 AD: the model closes the thinking message after two bytes.
    let mut ids = vec![
        special("<|message_model|>"),
        special("<|content_thinking|>"),
    ];
    ids.extend(bytes("ok "));
    ids.extend([BYTE_BASE + 0xE4, BYTE_BASE + 0xB8]);
    ids.push(special("<|end_message|>"));
    ids.extend(t.encode(&framed("<|content_text|>", "fine")));
    let turn = route(&t, &ids);
    assert_eq!(turn.reasoning(), "ok \u{FFFD}");
    assert_eq!(turn.content(), "fine");
}

#[test]
fn adjacent_markers_and_empty_messages_emit_nothing() {
    let t = tokenizer();
    let text = format!(
        "{}{}{}",
        framed("<|content_thinking|>", ""),
        framed("<|content_text|>", "Hi"),
        "<|end_message|><|end_message|>"
    );
    let turn = route(&t, &t.encode(&text));
    // One event per byte token of "Hi"; the empty frames and the doubled
    // closer contribute nothing.
    assert_eq!(
        turn.events,
        [
            ThinkRouteEvent::Content("H".into()),
            ThinkRouteEvent::Content("i".into())
        ]
    );
}

#[test]
fn message_open_without_close_starts_a_new_message() {
    let t = tokenizer();
    let text = "<|message_model|><|content_thinking|>plan<|message_model|><|content_text|>done";
    let turn = route(&t, &t.encode(text));
    assert_eq!(
        turn.events,
        [
            ThinkRouteEvent::Reasoning("p".into()),
            ThinkRouteEvent::Reasoning("l".into()),
            ThinkRouteEvent::Reasoning("a".into()),
            ThinkRouteEvent::Reasoning("n".into()),
            ThinkRouteEvent::Content("d".into()),
            ThinkRouteEvent::Content("o".into()),
            ThinkRouteEvent::Content("n".into()),
            ThinkRouteEvent::Content("e".into()),
        ]
    );
}

#[test]
fn multiple_blocks_keep_their_order() {
    let t = tokenizer();
    let text = [
        framed("<|content_thinking|>", "A"),
        framed("<|content_text|>", "B"),
        framed("<|content_thinking|>", "C"),
        framed("<|content_text|>", "D"),
    ]
    .concat();
    let turn = route(&t, &t.encode(&text));
    assert_eq!(
        turn.events,
        [
            ThinkRouteEvent::Reasoning("A".into()),
            ThinkRouteEvent::Content("B".into()),
            ThinkRouteEvent::Reasoning("C".into()),
            ThinkRouteEvent::Content("D".into()),
        ]
    );
}

#[test]
fn turn_end_inside_thinking_keeps_it_reasoning() {
    // EOS (or a length cut) before `<|end_message|>`: the open message is
    // still reasoning and no content is invented.
    let t = tokenizer();
    let turn = route(
        &t,
        &t.encode("<|message_model|><|content_thinking|>still going"),
    );
    assert_eq!(turn.reasoning(), "still going");
    assert_eq!(turn.content(), "");

    // A closed message does not end the turn: text after it is content.
    let turn = route(
        &t,
        &t.encode(&format!(
            "{}after",
            framed("<|content_thinking|>", "thought")
        )),
    );
    assert_eq!(turn.reasoning(), "thought");
    assert_eq!(turn.content(), "after");
}

#[test]
fn truncation_flushes_a_partial_character_and_an_unkinded_header() {
    let t = tokenizer();
    // Length cut after two of the three bytes of "中".
    let mut ids = t.encode("<|message_model|><|content_text|>x");
    ids.extend([BYTE_BASE + 0xE4, BYTE_BASE + 0xB8]);
    let turn = route(&t, &ids);
    assert_eq!(turn.content(), "x\u{FFFD}");

    // Cut right after the message opener: nothing to show.
    let turn = route(&t, &t.encode("<|message_model|>"));
    assert!(turn.events.is_empty());

    // A header that never names a content kind carried text: keep it.
    let turn = route(&t, &t.encode("<|message_model|>stray"));
    assert_eq!(turn.events, [ThinkRouteEvent::Content("stray".into())]);
    let turn = route(&t, &t.encode("<|message_model|>stray<|end_message|>"));
    assert_eq!(turn.events, [ThinkRouteEvent::Content("stray".into())]);
}

#[test]
fn plain_output_without_frames_is_content() {
    let t = tokenizer();
    let turn = route(&t, &t.encode("just text"));
    assert_eq!(turn.content(), "just text");
    assert_eq!(turn.reasoning(), "");
}

#[test]
fn marker_text_spelled_by_ordinary_tokens_stays_text() {
    let t = tokenizer();
    let mut ids = t.encode("<|message_model|><|content_text|>");
    ids.extend(bytes("use <|content_thinking|> literally"));
    ids.push(special("<|end_message|>"));
    let turn = route(&t, &ids);
    assert_eq!(turn.content(), "use <|content_thinking|> literally");
    assert_eq!(turn.reasoning(), "");
}

#[test]
fn other_protocol_tokens_are_never_rendered() {
    let t = tokenizer();
    // A stray role header inside the answer is dropped, not shown.
    let turn = route(
        &t,
        &t.encode("<|message_model|><|content_text|>a<|message_user|>b<|end_message|>"),
    );
    assert_eq!(turn.content(), "ab");
    assert_no_markers(&turn);

    // Another content kind: the name is header metadata, the body content.
    let turn = route(
        &t,
        &t.encode(
            "<|message_model|>get_time<|content_invoke_tool_json|>{\"name\":\"get_time\"}<|end_message|>",
        ),
    );
    assert_eq!(turn.content(), "{\"name\":\"get_time\"}");
    assert_no_markers(&turn);
}

#[test]
fn streamed_events_equal_one_shot_decode_of_each_span() {
    // Non-streamed responses concatenate the same events; each channel must
    // equal decoding that span's token ids in one call.
    let t = tokenizer();
    let think_ids = t.encode(THINK);
    let answer_ids = t.encode(ANSWER);
    let mut ids = vec![
        special("<|message_model|>"),
        special("<|content_thinking|>"),
    ];
    ids.extend(&think_ids);
    ids.extend([
        special("<|end_message|>"),
        special("<|message_model|>"),
        special("<|content_text|>"),
    ]);
    ids.extend(&answer_ids);
    ids.push(special("<|end_message|>"));
    let turn = route(&t, &ids);
    assert_eq!(turn.reasoning(), t.decode(&think_ids));
    assert_eq!(turn.content(), t.decode(&answer_ids));
}

#[test]
fn finish_is_terminal_and_idempotent() {
    let t = tokenizer();
    let markers = MessageFrameMarkers::from_tokenizer(&t).unwrap();
    let mut router = MessageFrameRouter::new(markers);
    let mut events = Vec::new();
    for id in t.encode("<|message_model|><|content_text|>ab") {
        router.push_token(&t, id, &mut events);
    }
    router.finish_into(&mut events);
    let n = events.len();
    router.finish_into(&mut events);
    for id in t.encode("late") {
        router.push_token(&t, id, &mut events);
    }
    assert_eq!(events.len(), n);
}

fn render(t: &Tokenizer, messages: &[Message], effort: Option<&str>) -> Result<String, String> {
    JinjaChatFrame {
        tokenizer: t,
        template: TEMPLATE,
        system: None,
        user: "",
        enable_thinking: true,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: effort,
    }
    .render_messages(messages, None, None)
}

#[test]
fn template_effort_matches_hf_apply_chat_template() {
    let t = tokenizer();
    let cases: Vec<serde_json::Value> = serde_json::from_str(CASES).unwrap();
    assert!(cases.len() >= 7);
    for c in &cases {
        let messages: Vec<Message> = serde_json::from_value(c["messages"].clone()).unwrap();
        let got = render(&t, &messages, c["reasoning_effort"].as_str())
            .unwrap_or_else(|e| panic!("case {}: {e}", c["name"]));
        assert_eq!(got, c["expected"].as_str().unwrap(), "case {}", c["name"]);
    }
}

#[test]
fn thinking_off_is_effort_none_and_default_is_unchanged() {
    let t = tokenizer();
    let user: Vec<Message> =
        serde_json::from_value(serde_json::json!([{"role": "user", "content": "What is 2+2?"}]))
            .unwrap();
    let default = render(&t, &user, None).unwrap();
    let off = render(&t, &user, Some("none")).unwrap();
    assert!(default.contains("Thinking effort level: 0.9<|end_message|>"));
    assert!(off.contains("Thinking effort level: 0<|end_message|>"));
    // Explicit `high` is the template default: identical prompt bytes, so an
    // unset effort and `high` produce identical token ids.
    assert_eq!(render(&t, &user, Some("high")).unwrap(), default);
    // `enable_thinking` is not a variable of this template.
    let ignored = JinjaChatFrame {
        tokenizer: &t,
        template: TEMPLATE,
        system: None,
        user: "",
        enable_thinking: false,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    }
    .render_messages(&user, None, None)
    .unwrap();
    assert_eq!(ignored, default);
    // Unknown effort strings raise in the template; nothing is remapped.
    assert!(render(&t, &user, Some("bogus")).is_err());
}

#[test]
fn template_effort_probe_reports_native_ladder() {
    let t = tokenizer();
    let cap = hipfire_runtime::prompt_frame::probe_effort_capability(&t, TEMPLATE);
    assert!(cap.native);
    assert_eq!(cap.probe_error, None);
    assert_eq!(cap.supported, ["low", "medium", "high", "xhigh", "max"]);
}
