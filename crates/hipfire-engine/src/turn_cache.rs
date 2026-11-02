// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Assistant-turn cache publication — the one store seam every generation
//! loop (AR, spec, EP, batch, lane batch) uses to publish an emitted turn
//! into `AsstTurnCache` keyed by the client-visible fingerprint.

use crate::emit::canonical_json;
use hipfire_runtime::prompt_frame::ToolCall;

/// Stable fingerprint over an assistant turn — pair of (text content,
/// tool_calls canonical JSON). Output is identical for two messages
/// that have the same content+tool_calls regardless of how the
/// surrounding bytes (e.g. whitespace inside JSON args) were rendered
/// upstream. Used by the V4F prefix-cache to identify "this is the
/// same assistant turn the model previously emitted, so reuse the
/// emitted token IDs verbatim instead of re-encoding via the DSML
/// renderer + BPE (which is not bijective)."
pub fn asst_turn_fingerprint(content: &str, tool_calls: &[ToolCall]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    "assistant".hash(&mut h);
    if tool_calls.is_empty() {
        // Pure-text turn — content IS the message. Trim whitespace
        // to absorb minor formatting drift between store (model's
        // verbatim emission) and lookup (whatever the client preserved).
        content.trim().hash(&mut h);
    } else {
        // Mixed turn (text + tool_calls) or pure tool_call. Hash ONLY
        // the tool_calls — pi-coding-agent (and most OpenAI-compat
        // clients) sends `content: null` on assistant messages that
        // carry tool_calls, even when the model originally emitted
        // prose ahead of the tool block (e.g. "Let me check the
        // structure first.<｜DSML｜tool_calls>…"). The store-side
        // sees the prose in `emit_text_buf`; the lookup-side sees
        // content=`""`. Excluding content from the fingerprint when
        // tool_calls is non-empty matches the client's effective
        // identity for tool-call turns and lets the cache hit.
        //
        // Collision risk: two distinct turns with identical
        // tool_calls hash to the same key; the later store wins,
        // and a replay of the earlier turn replays the later turn's
        // tokens. In practice this only matters when the model emits
        // the SAME tool_call twice with different surrounding prose
        // in the same conversation — uncommon for agent flows, and
        // the worst-case effect is the model seeing slightly altered
        // prose in its own history.
    }
    for tc in tool_calls {
        tc.name.hash(&mut h);
        // Serialize args in a CANONICAL form: walk the Value tree and
        // emit objects with keys sorted lexically (recursively). The
        // upstream `serde_json::Map` uses insertion order — fine for
        // round-tripping a single payload, but two clients (or two
        // parser passes on the same payload) can yield different
        // insertion orders for the same logical args. Without
        // canonicalization those two turns hash to DIFFERENT keys,
        // dropping cache hit rate on otherwise-identical tool calls.
        let args = canonical_json(&tc.arguments);
        args.hash(&mut h);
    }
    h.finish()
}

/// Cache-store decision derived from a family terminal. Production and tests
/// share it so `store` is never asserted apart from the sink mutation.
#[derive(Debug, Clone)]
pub struct TurnCacheAction {
    pub store: bool,
    pub fingerprint_text: String,
    pub tool_calls: Vec<ToolCall>,
}

impl TurnCacheAction {
    /// Terminal that must never publish (malformed, fail-closed).
    pub fn skip() -> Self {
        Self {
            store: false,
            fingerprint_text: String::new(),
            tool_calls: Vec::new(),
        }
    }
}

/// Publish `cached_seq` through `insert` (production `AsstTurnCache` or a
/// test map). Returns the fingerprint key when a store happened.
pub fn apply_turn_cache_action<F>(
    mut insert: F,
    action: &TurnCacheAction,
    cached_seq: Vec<u32>,
) -> Option<u64>
where
    F: FnMut(u64, Vec<u32>),
{
    if !action.store || cached_seq.is_empty() {
        return None;
    }
    let fp = asst_turn_fingerprint(&action.fingerprint_text, &action.tool_calls);
    insert(fp, cached_seq);
    Some(fp)
}
