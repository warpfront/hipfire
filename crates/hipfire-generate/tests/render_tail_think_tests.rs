// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Render tail think tests.
//!
//! Moved out of `hipfire-daemon`'s `main.rs`. Compiled into a bin crate these
//! never appeared as their own test target; as integration tests they are
//! reported individually.

#![allow(unused_imports, dead_code, clippy::all)]

use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_generate::ar::*;
use hipfire_generate::batch::*;
use hipfire_generate::common::*;

    
    use hipfire_generate::{common::asst_turn_fingerprint, common::normalize_asst_turn_for_fingerprint};
    use hipfire_runtime::prompt_frame::AssistantPrefix;

    #[test]
    fn qwen_jinja_think_tail_primes_reasoning_channel() {
        assert!(render_tail_opens_think("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn speculative_emitter_uses_rendered_think_state() {
        assert!(matches!(
            spec_assistant_prefix(true),
            AssistantPrefix::OpenThink
        ));
        assert!(matches!(
            spec_assistant_prefix(false),
            AssistantPrefix::Plain
        ));
    }

    #[test]
    fn plain_closed_and_user_literal_tails_do_not_prime() {
        assert!(!render_tail_opens_think("<|im_start|>assistant\n"));
        assert!(!render_tail_opens_think(
            "<|im_start|>assistant\n<think>\n</think>\n"
        ));
        assert!(!render_tail_opens_think(
            "<|im_start|>user\nliteral <think><|im_end|>\n<|im_start|>assistant\n"
        ));
    }

    /// The assistant framing is read back from the rendered suffix, not from
    /// the tokenizer's token inventory. A thinking-on Qwen3-family render ends
    /// in a bare `assistant\n` — the model opens `<think>` itself — so the
    /// framing is `Plain`, never `ClosedThink`.
    ///
    /// Regression: the multi-slot route derived `ClosedThink` whenever the
    /// tokenizer registered `<think>`, then failed every thinking-on request
    /// closed with "reasoning prefix mismatch: expected OpenThink got
    /// ClosedThink".
    #[test]
    fn rendered_framing_is_read_from_the_suffix() {
        assert!(matches!(
            render_assistant_prefix("<|im_start|>assistant\n"),
            AssistantPrefix::Plain
        ));
        assert!(matches!(
            render_assistant_prefix("<|im_start|>assistant\n<think>\n"),
            AssistantPrefix::OpenThink
        ));
        assert!(matches!(
            render_assistant_prefix("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            AssistantPrefix::ClosedThink
        ));
        // Tail-only: a literal closer in user content must not reclassify.
        assert!(matches!(
            render_assistant_prefix(
                "<|im_start|>user\nan answer </think><|im_end|>\n<|im_start|>assistant\n"
            ),
            AssistantPrefix::Plain
        ));
    }

    #[test]
    fn assistant_cache_fingerprint_matches_client_visible_content() {
        let raw = "hidden reasoning</think>\n\nvisible answer<|im_end|>";
        let normalized = hipfire_generate::common::normalize_asst_turn_for_fingerprint(raw);
        assert_eq!(normalized, "visible answer");
        assert_eq!(
            hipfire_generate::common::asst_turn_fingerprint(&normalized, &[]),
            hipfire_generate::common::asst_turn_fingerprint("visible answer", &[])
        );
    }
