// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate: a tool-result turn extends the session that made those calls, and
//! nothing else.
//!
//! A tool-result turn adds no user turn, so its conversation key is identical
//! to the session it answers — and to every duplicate of that conversation.
//! Matching on the key alone therefore cannot distinguish them, and the loser
//! of that coin flip gets `<tool_response>` blocks spliced into a KV that never
//! emitted the calls. The caller names the session instead; this asserts the
//! engine honours the name and re-prefills rather than substituting a stand-in
//! when the name does not fit.
//!
//! Run: `cargo run --release -p hipfire-runtime --features lab,deltanet
//!       --example test_serve_tool_reentry -- [model.mq4r]`

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::serve_engine::{EngineConfig, SlotEngine};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::prompt_frame::{
        continuation_suffix_tool_results, AssistantPrefix, ChatFrame,
    };
    use hipfire_runtime::serve::{Continuation, Event, SubmitRequest};
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::channel;

    const MAX_TOKENS: usize = 8;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        let home = std::env::var("HOME").expect("HOME not set");
        format!("{home}/.hipfire/models/qwen3.6-35b-a3b.mq4r")
    });
    println!("=== test_serve_tool_reentry ===\nmodel: {model_path}");

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    drop(hfq);

    let engine = SlotEngine::spawn(EngineConfig {
        model_path: PathBuf::from(&model_path),
        n_slots: 2,
        cap_tokens: 2048,
        prefill_chunk: 512,
        host_budget_bytes: 8 * 1024 * 1024 * 1024,
        swap_dir: std::env::temp_dir().join("hipfire-tool-reentry-swap"),
        kv_mode: "q8".to_string(),
        kv_backend: "contiguous".to_string(),
    })
    .expect("engine");

    fn turn_hash(s: &str) -> u64 {
        let mut h = 0xcbf29ce484222325_u64;
        for b in s.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    let ask = "List one European capital.";
    let prompt = ChatFrame {
        tokenizer: &tokenizer,
        system: None,
        user: ask,
        assistant_prefix: AssistantPrefix::ClosedThink,
        raw: false,
    }
    .build();
    let convo = vec![turn_hash(ask)];
    let results = continuation_suffix_tool_results(
        &tokenizer,
        &["{\"temp_c\": 19}".to_string()],
        AssistantPrefix::ClosedThink,
    );

    let submit = |continuation: Continuation, convo: Vec<u64>| -> (u64, usize, usize) {
        let (tx, rx) = channel::<Event>();
        engine
            .submit(SubmitRequest {
                prompt_tokens: prompt.clone(),
                convo,
                continuation,
                max_tokens: MAX_TOKENS,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
                reply: tx,
            })
            .expect("submit");
        let (mut session, mut reused, mut prefill) = (u64::MAX, 0, 0);
        while let Ok(ev) = rx.recv() {
            match ev {
                Event::Accepted {
                    session: s,
                    reused: r,
                    prefill: p,
                } => {
                    session = s;
                    reused = r;
                    prefill = p;
                }
                Event::Token { .. } => {}
                Event::Done { .. } => break,
                Event::Rejected { reason } => panic!("rejected: {reason}"),
            }
        }
        assert_ne!(session, u64::MAX, "engine never accepted the request");
        (session, reused, prefill)
    };

    // Turn 1 opens the session whose (imagined) tool calls the rest answer.
    let (session_a, _, prefill_a) = submit(Continuation::Cold, convo.clone());
    println!("turn 1: session {session_a}, prefilled {prefill_a}");
    assert!(prefill_a > 0, "a cold turn must prefill the prompt");

    // A second, independent session on the SAME conversation. Under a
    // key-only match this is an equally good reentry candidate — and, being
    // the most recently used, the one an LRU tie-break would hand back.
    let (session_b, _, _) = submit(Continuation::Cold, convo.clone());
    println!("turn 2: duplicate conversation opened as session {session_b}");
    assert_ne!(
        session_a, session_b,
        "the duplicate must be its own session"
    );

    // Naming session A extends A: its stored tokens are reused, only the
    // tool-result suffix is prefilled.
    let (named, reused, prefill) = submit(
        Continuation::ToolResults {
            tokens: results.clone(),
            session: session_a,
        },
        convo.clone(),
    );
    println!("turn 3: named session {session_a} -> {named}, reused {reused}, prefill {prefill}");
    assert_eq!(named, session_a, "the named session must serve the turn");
    assert!(
        reused >= prompt.len(),
        "reentry must reuse the whole stored prompt ({reused} < {})",
        prompt.len()
    );
    assert!(
        prefill > 0,
        "the tool-result suffix still has to be prefilled"
    );

    // An id that no longer resolves must NOT fall through to the duplicate.
    let unknown = session_b + 1_000;
    let (fallback, reused_unknown, _) = submit(
        Continuation::ToolResults {
            tokens: results.clone(),
            session: unknown,
        },
        convo.clone(),
    );
    println!("turn 4: unknown session {unknown} -> {fallback}, reused {reused_unknown}");
    assert_ne!(fallback, session_a, "a stale id must not reach session A");
    assert_ne!(fallback, session_b, "a stale id must not reach session B");

    // A live session whose conversation has moved on is equally not a match.
    let (moved_on, reused_moved, _) = submit(
        Continuation::ToolResults {
            tokens: results,
            session: session_a,
        },
        vec![turn_hash("a different conversation entirely")],
    );
    println!("turn 5: mismatched conversation -> {moved_on}, reused {reused_moved}");
    assert_ne!(
        moved_on, session_a,
        "the conversation key still has to agree with the named session"
    );

    println!("PASS: tool-result reentry is by name, and fails closed");
}
