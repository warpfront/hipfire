// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! The DeepSeek V4 and Cohere2-MoE speculative emitters stream visible text
//! through `TokenTextStream`: a character whose UTF-8 bytes span several
//! tokens reaches the client whole, never as U+FFFD, including a character
//! still split when generation stops.

use hipfire_runtime::spec::{ClientEvent, SpecEmit, SpecEmitCtx};
use hipfire_runtime::tokenizer::Tokenizer;

const EOS: u32 = 7;

/// Byte-level BPE: token `100 + b` is the single byte `b` (GPT-2 byte-to-unicode
/// alphabet, no merges), plus a special `eos` at id 7.
fn byte_level_tokenizer() -> Tokenizer {
    let mut printable: Vec<u32> = (u32::from(b'!')..=u32::from(b'~')).collect();
    printable.extend(0xA1..=0xAC);
    printable.extend(0xAE..=0xFF);
    let mut shifted = 0;
    let mut entries = vec![format!(r#""eos": {EOS}"#)];
    for byte in 0u32..=255 {
        let ch = if printable.contains(&byte) {
            char::from_u32(byte).unwrap()
        } else {
            shifted += 1;
            char::from_u32(255 + shifted).unwrap()
        };
        let key = serde_json::to_string(&ch.to_string()).unwrap();
        entries.push(format!("{key}: {}", 100 + byte));
    }
    let json = format!(
        r#"{{"model": {{"type": "BPE", "vocab": {{ {} }}, "merges": []}},
            "added_tokens": [{{"id": {EOS}, "content": "eos", "special": true}}]}}"#,
        entries.join(", ")
    );
    Tokenizer::from_hf_json(&json).expect("byte-level tokenizer")
}

fn ctx(tokenizer: &Tokenizer) -> SpecEmitCtx<'_> {
    SpecEmitCtx {
        tokenizer,
        eos: EOS,
        im_end: None,
        tools: None,
        enable_grammar: false,
        stop: Vec::new(),
        max_think: 0,
        max_tokens: 256,
        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
        decoded_vocab: None,
    }
}

/// Feed `text` byte by byte, then finish; return the visible text.
fn stream_visible(mut emit: Box<dyn SpecEmit + '_>, text: &[u8]) -> String {
    let mut visible = String::new();
    let mut take = |events: Vec<ClientEvent>| {
        for event in events {
            if let ClientEvent::Token(t) = event {
                visible.push_str(&t);
            }
        }
    };
    let mut ids = text.iter().map(|&b| 100 + u32::from(b));
    take(emit.begin(ids.next().unwrap()).events);
    for id in ids {
        take(emit.observe(id).events);
    }
    take(emit.finish().events);
    visible
}

#[test]
fn ds4_spec_emitter_streams_split_characters_whole() {
    let tokenizer = byte_level_tokenizer();
    let text = "emoji 🎉 and 中文 done";
    let emit = hipfire_arch_deepseek4::spec_emit::Deepseek4Emit::from_ctx(ctx(&tokenizer));
    assert_eq!(stream_visible(emit, text.as_bytes()), text);
    // Generation stopped inside a character: the complete prefix arrives and
    // the truncated tail is flushed at finish instead of vanishing.
    let cut = &"ok 🎉".as_bytes()[..5];
    let emit = hipfire_arch_deepseek4::spec_emit::Deepseek4Emit::from_ctx(ctx(&tokenizer));
    assert_eq!(stream_visible(emit, cut), "ok \u{FFFD}");
}

#[test]
fn cohere2moe_spec_emitter_streams_split_characters_whole() {
    let tokenizer = byte_level_tokenizer();
    let text = "emoji 🎉 and 中文 done";
    let emit = hipfire_arch_cohere2moe::spec_emit::Cohere2MoeEmit::from_ctx(ctx(&tokenizer));
    assert_eq!(stream_visible(emit, text.as_bytes()), text);
}

/// P16 usage policy pin (0.4.1: documented, not changed). Spec decode counts a
/// token into `done.tokens` (→ `usage.completion_tokens`) exactly when
/// `spec_outcome_seed_committable` accepts the emitter outcome. Qwen3.5 emits a
/// `Committed` event for EOS, so the terminator is counted; DeepSeek V4 returns
/// no events for EOS, so it is not. The AR loops follow the same split (Qwen AR
/// increments before classify; DS4 AR / LFM batch stop before counting EOS).
#[test]
fn spec_usage_counts_eos_for_qwen_not_ds4() {
    let tokenizer = byte_level_tokenizer();
    let counted = |mut emit: Box<dyn SpecEmit + '_>| {
        let outcomes = [emit.begin(100 + u32::from(b'h')), emit.observe(100 + u32::from(b'i')), emit.observe(EOS)];
        outcomes
            .iter()
            .filter(|o| hipfire_generate::qwen::spec_outcome_seed_committable(o))
            .count()
    };
    let qwen = hipfire_arch_qwen35::spec_emit::Qwen35Emit::from_ctx(ctx(&tokenizer));
    assert_eq!(counted(qwen), 3, "Qwen spec: 2 text tokens + EOS");
    let ds4 = hipfire_arch_deepseek4::spec_emit::Deepseek4Emit::from_ctx(ctx(&tokenizer));
    assert_eq!(counted(ds4), 2, "DS4 spec: 2 text tokens, EOS excluded");
}
