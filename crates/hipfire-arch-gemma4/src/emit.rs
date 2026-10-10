// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 output routing: the thought-channel router that splits decoded
//! text into reasoning and visible answer, shared by the AR loop and the
//! speculative emitter.

use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::spec::{
    ClientEvent, EmitOutcome, FinishSummary, SpecEmit, SpecEmitCtx, StopReason,
};
use hipfire_runtime::tokenizer::{TokenTextStream, Tokenizer};

/// Gemma thought-channel router (minimal, follows Glimmer patterns and checked-in Gemma template).
///
/// Gemma4 template uses `<|channel>thought\n` ... `\n<channel|>` to wrap reasoning.
/// When `enable_thinking` is false the Jinja prompt already emits a closed empty
/// `thought` channel (`<|channel>thought\n<channel|>`), so the router starts in
/// Answer mode. When true it starts awaiting the thought opening. Content inside
/// the thought channel is emitted as semantic `reasoning` events;
/// content outside is visible answer tokens. `max_think_tokens` is an orthogonal
/// force-close cap (0 = uncapped) that, when reached, flushes pending reasoning
/// and transitions to Answer, mirroring Glimmer's `max_think_tokens` handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaChannel {
    AwaitingThought,
    Reasoning,
    Answer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmaEmit {
    Reasoning(String),
    Token(String),
}

pub struct GemmaThoughtRouter {
    pub state: GemmaChannel,
    pub pending: String,
    pub reasoning_tokens: usize,
    pub max_think_tokens: usize,
    pub enable_thinking: bool,
    pub just_forced: bool,
}

impl GemmaThoughtRouter {
    pub fn new(enable_thinking: bool, max_think_tokens: usize) -> Self {
        let state = if enable_thinking {
            GemmaChannel::AwaitingThought
        } else {
            GemmaChannel::Answer
        };
        Self {
            state,
            pending: String::new(),
            reasoning_tokens: 0,
            max_think_tokens,
            enable_thinking,
            just_forced: false,
        }
    }

    pub fn push(&mut self, frag: &str) -> (Vec<GemmaEmit>, bool) {
        if frag.is_empty() {
            return (Vec::new(), false);
        }
        self.pending.push_str(frag);
        let mut out = Vec::new();
        let entered_as_reasoning = self.state == GemmaChannel::Reasoning;
        loop {
            match self.state {
                GemmaChannel::AwaitingThought => {
                    const CHANNEL_OPEN: &str = "<|channel>";
                    const THOUGHT_OPEN: &str = "<|channel>thought";

                    // A thought channel is optional and may be split across
                    // decoded fragments. Hold only while the bytes seen so far
                    // can still become the canonical opening header.
                    if THOUGHT_OPEN.starts_with(&self.pending) {
                        break;
                    }
                    if self.pending.starts_with(THOUGHT_OPEN) {
                        let mut header_end = THOUGHT_OPEN.len();
                        if self.pending[header_end..].starts_with('\n') {
                            header_end += 1;
                        }
                        self.pending.drain(..header_end);
                        self.state = GemmaChannel::Reasoning;
                        continue;
                    }

                    // The response schema makes the thought header optional.
                    // Once the buffered bytes cannot form that header, route
                    // them as answer content. Some Gemma4 checkpoints emit an
                    // orphan `<|channel>` before an otherwise valid answer;
                    // consume that control token without dropping its payload.
                    if self.pending.starts_with(CHANNEL_OPEN) {
                        self.pending.drain(..CHANNEL_OPEN.len());
                        if self.pending.starts_with('\n') {
                            self.pending.drain(..1);
                        }
                    }
                    self.state = GemmaChannel::Answer;
                    continue;
                }
                GemmaChannel::Reasoning => {
                    if let Some(pos) = self.pending.find("<channel|>") {
                        if pos > 0 {
                            let text = self.pending[..pos].to_string();
                            self.pending.drain(..pos);
                            if !text.is_empty() {
                                out.push(GemmaEmit::Reasoning(text));
                            }
                            continue;
                        }
                        let end = "<channel|>".len();
                        self.pending.drain(..end);
                        if self.pending.starts_with('\n') {
                            self.pending.drain(..1);
                        }
                        self.state = GemmaChannel::Answer;
                        continue;
                    } else {
                        let hold = gemma_longest_marker_suffix(&self.pending);
                        let emit_len = self.pending.len().saturating_sub(hold);
                        if emit_len > 0 {
                            let text = self.pending[..emit_len].to_string();
                            self.pending.drain(..emit_len);
                            if !text.is_empty() {
                                out.push(GemmaEmit::Reasoning(text));
                            }
                        }
                        break;
                    }
                }
                GemmaChannel::Answer => {
                    // Answer must chunk-safely strip any Gemma channel
                    // control markers, including canonical <|channel|>
                    // forms, orphan <|channel> variants, and markers
                    // arriving after a forced max-think transition.
                    // Preserve payload before/after each marker and hold
                    // a suffix that could still become a marker.
                    const ANSWER_MARKERS: &[&str] = &[
                        "<|channel>thought",
                        "<|channel>",
                        "<|channel|>",
                        "<channel|>",
                        "<|turn>",
                        "<turn|>",
                    ];
                    loop {
                        let hold = gemma_longest_marker_suffix(&self.pending);
                        let search_len = self.pending.len().saturating_sub(hold);
                        let searchable = &self.pending[..search_len];
                        let mut best_pos: Option<usize> = None;
                        let mut best_len = 0usize;
                        for &m in ANSWER_MARKERS {
                            if let Some(pos) = searchable.find(m) {
                                if best_pos.is_none()
                                    || pos < best_pos.unwrap()
                                    || (pos == best_pos.unwrap() && m.len() > best_len)
                                {
                                    best_pos = Some(pos);
                                    best_len = m.len();
                                }
                            }
                        }
                        if let Some(pos) = best_pos {
                            if pos > 0 {
                                let text = self.pending[..pos].to_string();
                                self.pending.drain(..pos);
                                if !text.is_empty() {
                                    out.push(GemmaEmit::Token(text));
                                }
                                continue;
                            }
                            self.pending.drain(..best_len);
                            if self.pending.starts_with('\n') {
                                self.pending.drain(..1);
                            }
                            if self.pending.is_empty() {
                                break;
                            }
                            continue;
                        }
                        if search_len > 0 {
                            let text = self.pending[..search_len].to_string();
                            self.pending.drain(..search_len);
                            if !text.is_empty() {
                                out.push(GemmaEmit::Token(text));
                            }
                        }
                        break;
                    }
                    break;
                }
            }
        }
        let emitted_reasoning = out.iter().any(|e| matches!(e, GemmaEmit::Reasoning(_)));
        let should_count = entered_as_reasoning || emitted_reasoning;
        if should_count
            && self.max_think_tokens != 0
            && self.reasoning_tokens < self.max_think_tokens
        {
            self.reasoning_tokens += 1;
            if self.reasoning_tokens >= self.max_think_tokens
                && self.state == GemmaChannel::Reasoning
            {
                if !self.pending.is_empty() {
                    let tail = std::mem::take(&mut self.pending);
                    if !gemma_is_marker_prefix(&tail) && !tail.is_empty() {
                        out.push(GemmaEmit::Reasoning(tail));
                    }
                }
                self.state = GemmaChannel::Answer;
                self.just_forced = true;
            }
        }
        (out, false)
    }

    pub fn flush(&mut self) -> Vec<GemmaEmit> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        if self.state == GemmaChannel::AwaitingThought && gemma_is_marker_prefix(&self.pending) {
            self.pending.clear();
            return Vec::new();
        }
        let text = std::mem::take(&mut self.pending);
        if gemma_is_marker_prefix(&text) {
            return Vec::new();
        }
        vec![if self.state == GemmaChannel::Reasoning {
            GemmaEmit::Reasoning(text)
        } else {
            GemmaEmit::Token(text)
        }]
    }
}

pub fn gemma_is_marker_prefix(s: &str) -> bool {
    const MARKERS: &[&str] = &[
        "<|channel>thought",
        "<|channel>",
        "<|channel|>",
        "<channel|>",
        "<|turn>",
        "<turn|>",
    ];
    MARKERS.iter().any(|m| m.starts_with(s))
}

/// [`SpecEmit`] for Gemma 4: the AR loop's emission (incremental decode
/// through the [`GemmaThoughtRouter`]) over speculatively committed tokens.
/// A stop token ends the turn uncommitted, as in AR.
pub struct GemmaSpecEmit<'a> {
    tokenizer: &'a Tokenizer,
    stops: Vec<u32>,
    router: GemmaThoughtRouter,
    text: TokenTextStream,
    committed: usize,
}

impl<'a> GemmaSpecEmit<'a> {
    /// Stops on the target EOS, `im_end` (the config EOS) and Gemma's
    /// `<end_of_turn>` id 106; thinking is on unless `think_mode` is
    /// `NonThink`.
    pub fn new(ctx: SpecEmitCtx<'a>) -> Self {
        let mut stops = vec![ctx.eos, 106];
        stops.extend(ctx.im_end);
        Self {
            tokenizer: ctx.tokenizer,
            stops,
            router: GemmaThoughtRouter::new(
                !matches!(ctx.think_mode, ThinkMode::NonThink),
                ctx.max_think,
            ),
            text: TokenTextStream::new(),
            committed: 0,
        }
    }
}

fn client_events(emits: Vec<GemmaEmit>, events: &mut Vec<ClientEvent>) {
    events.extend(emits.into_iter().map(|e| match e {
        GemmaEmit::Reasoning(text) => ClientEvent::Reasoning(text),
        GemmaEmit::Token(text) => ClientEvent::Token(text),
    }));
}

impl SpecEmit for GemmaSpecEmit<'_> {
    fn begin(&mut self, first_token: u32) -> EmitOutcome {
        self.observe(first_token)
    }

    fn observe(&mut self, token: u32) -> EmitOutcome {
        if self.stops.contains(&token) {
            return EmitOutcome {
                events: Vec::new(),
                stop: Some(StopReason::Eos),
            };
        }
        let frag = self.text.push(self.tokenizer, token);
        let mut events = Vec::new();
        client_events(self.router.push(&frag).0, &mut events);
        events.push(ClientEvent::Committed {
            id: token,
            idx: self.committed,
        });
        self.committed += 1;
        EmitOutcome { events, stop: None }
    }

    fn finish(mut self: Box<Self>) -> FinishSummary {
        let tail = self.text.flush();
        let mut events = Vec::new();
        client_events(self.router.push(&tail).0, &mut events);
        client_events(self.router.flush(), &mut events);
        FinishSummary {
            events,
            finish_reason: "stop",
            ..FinishSummary::default()
        }
    }
}

#[cfg(test)]
mod gemma_thought_router_tests {
    use super::{gemma_is_marker_prefix, GemmaChannel, GemmaEmit, GemmaThoughtRouter};

    fn route(enable_thinking: bool, chunks: &[&str]) -> (String, String, GemmaChannel) {
        let mut router = GemmaThoughtRouter::new(enable_thinking, 0);
        let mut visible = String::new();
        let mut reasoning = String::new();
        for chunk in chunks {
            for event in router.push(chunk).0 {
                match event {
                    GemmaEmit::Reasoning(text) => reasoning.push_str(&text),
                    GemmaEmit::Token(text) => visible.push_str(&text),
                }
            }
        }
        for event in router.flush() {
            match event {
                GemmaEmit::Reasoning(text) => reasoning.push_str(&text),
                GemmaEmit::Token(text) => visible.push_str(&text),
            }
        }
        (visible, reasoning, router.state)
    }

    #[test]
    fn gemma_router_routes_canonical_thought_then_answer() {
        let (visible, reasoning, state) = route(
            true,
            &["<|channel>", "thought", "\nplan<channel|>\nanswer<turn|>"],
        );
        assert_eq!(reasoning, "plan");
        assert_eq!(visible, "answer");
        assert_eq!(state, GemmaChannel::Answer);
    }

    #[test]
    fn gemma_router_recovers_orphan_channel_before_answer() {
        let (visible, reasoning, state) =
            route(true, &["<|channel>", "\n", "```python\nprint('ok')\n```"]);
        assert_eq!(visible, "```python\nprint('ok')\n```");
        assert!(reasoning.is_empty());
        assert_eq!(state, GemmaChannel::Answer);
    }

    #[test]
    fn gemma_router_orphan_channel_is_chunk_boundary_invariant() {
        let full = "<|channel>\nanswer";
        let expected = route(true, &[full]);
        for split in 1..full.len() {
            if full.is_char_boundary(split) {
                assert_eq!(route(true, &[&full[..split], &full[split..]]), expected);
            }
        }
    }

    #[test]
    fn gemma_router_thinking_request_can_emit_direct_answer() {
        let (visible, reasoning, state) = route(true, &["direct answer"]);
        assert_eq!(visible, "direct answer");
        assert!(reasoning.is_empty());
        assert_eq!(state, GemmaChannel::Answer);
    }

    #[test]
    fn gemma_router_drops_only_an_unfinished_control_marker_at_eos() {
        let (visible, reasoning, state) = route(true, &["<|chan"]);
        assert!(visible.is_empty());
        assert!(reasoning.is_empty());
        assert_eq!(state, GemmaChannel::AwaitingThought);
    }

    #[test]
    fn gemma_marker_prefix_does_not_classify_marker_plus_payload() {
        assert!(gemma_is_marker_prefix("<|chan"));
        assert!(gemma_is_marker_prefix("<|channel>"));
        assert!(!gemma_is_marker_prefix("<|channel>\nanswer"));
    }

    fn route_with_cap(
        enable_thinking: bool,
        max_think_tokens: usize,
        chunks: &[&str],
    ) -> (String, String, GemmaChannel) {
        let mut router = GemmaThoughtRouter::new(enable_thinking, max_think_tokens);
        let mut visible = String::new();
        let mut reasoning = String::new();
        for chunk in chunks {
            for event in router.push(chunk).0 {
                match event {
                    GemmaEmit::Reasoning(text) => reasoning.push_str(&text),
                    GemmaEmit::Token(text) => visible.push_str(&text),
                }
            }
        }
        for event in router.flush() {
            match event {
                GemmaEmit::Reasoning(text) => reasoning.push_str(&text),
                GemmaEmit::Token(text) => visible.push_str(&text),
            }
        }
        (visible, reasoning, router.state)
    }

    fn assert_no_markers(s: &str) {
        for m in &[
            "<|channel>thought",
            "<|channel>",
            "<|channel|>",
            "<channel|>",
            "<|turn>",
            "<turn|>",
        ] {
            assert!(!s.contains(m), "visible leaked marker {:?} in {:?}", m, s);
        }
    }

    #[test]
    fn gemma_router_thinking_off_strips_all_channel_markers_every_split() {
        // Thinking-off starts in Answer; every channel/turn marker must be
        // stripped chunk-safely regardless of split.
        let full = "pre<|channel>mid<channel|>post<|channel|>inner<|channel>thought\nX<channel|>tail<|turn>end<turn|>after";
        let expected = route_with_cap(false, 0, &[full]);
        assert_no_markers(&expected.0);
        // Adjacent payload must survive: markers stripped, text joined.
        assert_eq!(expected.0, "premidpostinnerXtailendafter");
        for split in 1..full.len() {
            if !full.is_char_boundary(split) {
                continue;
            }
            let got = route_with_cap(false, 0, &[&full[..split], &full[split..]]);
            assert_eq!(got, expected, "mismatch at split {}", split);
            assert_no_markers(&got.0);
        }
        // Also verify orphan <|channel> with newline framing is stripped.
        let full2 = "<|channel>\nanswer";
        let exp2 = route_with_cap(false, 0, &[full2]);
        assert_eq!(exp2.0, "answer");
        for split in 1..full2.len() {
            if !full2.is_char_boundary(split) {
                continue;
            }
            assert_eq!(
                route_with_cap(false, 0, &[&full2[..split], &full2[split..]]),
                exp2
            );
        }
        // Canonical <|channel|> in thinking-off must also be stripped.
        let full3 = "A<|channel|>B";
        let exp3 = route_with_cap(false, 0, &[full3]);
        assert_eq!(exp3.0, "AB");
        assert_no_markers(&exp3.0);
        for split in 1..full3.len() {
            if !full3.is_char_boundary(split) {
                continue;
            }
            assert_eq!(
                route_with_cap(false, 0, &[&full3[..split], &full3[split..]]),
                exp3
            );
        }
    }

    #[test]
    fn gemma_router_forced_close_strips_markers_after_transition_every_split() {
        // Force max-think after one reasoning push, then ensure every
        // subsequent channel/turn marker in Answer is stripped at every
        // split, preserving adjacent payload.
        let pre = "<|channel>thought\nAAA<channel|>";
        let post = "BBB<|channel>CCC<channel|>DDD<|channel|>EEE<|turn>FFF<turn|>GGG";
        let full = format!("{}{}", pre, post);
        // full = "<|channel>thought\nAAA<channel|>BBB<|channel>CCC<channel|>DDD<|channel|>EEE<|turn>FFF<turn|>GGG"
        // With max_think=1 the router forces to Answer after the first
        // reasoning push; the trailing <channel|> that closes thought and
        // all markers inside post must be stripped, not leaked.
        let expected = route_with_cap(true, 1, &[&full]);
        assert!(expected.1.contains("AAA") || expected.0.contains("AAA"));
        assert_no_markers(&expected.0);
        // Answer payload should be the post text with markers removed.
        // Post without markers: "BBBCCCDDDEEEFFFGGG"
        assert_eq!(expected.0, "BBBCCCDDDEEEFFFGGG");
        // For split invariance after forced close, keep the reasoning header
        // as one chunk and only split the post payload. Splitting the header
        // itself changes per-push reasoning counting and is not required to
        // be invariant for this test.
        for split in 0..=post.len() {
            if split != 0 && !post.is_char_boundary(split) {
                continue;
            }
            let got = if split == 0 || split == post.len() {
                route_with_cap(true, 1, &[&full])
            } else {
                let c1 = &post[..split];
                let c2 = &post[split..];
                route_with_cap(true, 1, &[pre, c1, c2])
            };
            assert_eq!(got.0, expected.0, "forced mismatch at post split {}", split);
            assert_no_markers(&got.0);
        }
        // Also test forced transition where pending marker is split across
        // the forced boundary: reasoning chunk ends with partial marker prefix.
        let full2 = "<|channel>thought\nRR<channel|>XX<|channel>YY";
        let exp2 = route_with_cap(true, 1, &[full2]);
        assert_no_markers(&exp2.0);
        // Split only the post part after the forced close to keep reasoning counting stable
        let pre2 = "<|channel>thought\nRR<channel|>";
        let post2 = "XX<|channel>YY";
        let exp2_post = route_with_cap(true, 1, &[full2]);
        for split in 0..=post2.len() {
            if split != 0 && !post2.is_char_boundary(split) {
                continue;
            }
            let got = if split == 0 || split == post2.len() {
                route_with_cap(true, 1, &[full2])
            } else {
                route_with_cap(true, 1, &[pre2, &post2[..split], &post2[split..]])
            };
            assert_eq!(
                got.0, exp2.0,
                "forced split2 mismatch at post split {}",
                split
            );
            assert_no_markers(&got.0);
        }
        // Verify that a marker arriving strictly after forced transition
        // as a separate push is still stripped at every internal split.
        for payload in &[
            "hello<channel|>world",
            "hello<|channel>world",
            "hello<|channel|>world",
            "hello<turn|>world",
            "hello<|turn>world",
        ] {
            let expected_payload = (*payload)
                .replace("<|channel>thought", "")
                .replace("<|channel>", "")
                .replace("<|channel|>", "")
                .replace("<channel|>", "")
                .replace("<|turn>", "")
                .replace("<turn|>", "");
            for split in 0..=payload.len() {
                if split != 0 && !payload.is_char_boundary(split) {
                    continue;
                }
                // Build payload split after forced transition.
                let full = if split == 0 || split == payload.len() {
                    format!("<|channel>thought\nZ<channel|>{}", payload)
                } else {
                    // Simulate payload split across two pushes after forced.
                    // Create a single concatenated string and test split invariance
                    // via the full-string split test already done; here just
                    // verify the payload alone after forced.
                    let c1 = &payload[..split];
                    let c2 = &payload[split..];
                    let mut rr = GemmaThoughtRouter::new(true, 1);
                    let _ = rr.push("<|channel>thought\nZ");
                    let mut vis = String::new();
                    for ev in rr.push("<channel|>").0 {
                        if let GemmaEmit::Token(t) = ev {
                            vis.push_str(&t);
                        }
                    }
                    for ev in rr.push(c1).0 {
                        if let GemmaEmit::Token(t) = ev {
                            vis.push_str(&t);
                        }
                    }
                    for ev in rr.push(c2).0 {
                        if let GemmaEmit::Token(t) = ev {
                            vis.push_str(&t);
                        }
                    }
                    for ev in rr.flush() {
                        if let GemmaEmit::Token(t) = ev {
                            vis.push_str(&t);
                        }
                    }
                    assert_eq!(
                        vis, expected_payload,
                        "payload {:?} split {}",
                        payload, split
                    );
                    assert_no_markers(&vis);
                    continue;
                };
                let got = route_with_cap(true, 1, &[&full]);
                assert!(got.0.contains(&expected_payload) || got.0 == expected_payload);
                assert_no_markers(&got.0);
            }
        }
    }
}

pub fn gemma_longest_marker_suffix(s: &str) -> usize {
    const MARKERS: &[&str] = &[
        "<|channel>thought",
        "<|channel>",
        "<|channel|>",
        "<channel|>",
        "<|turn>",
        "<turn|>",
    ];
    for len in (1..=s.len()).rev() {
        if !s.is_char_boundary(s.len() - len) {
            continue;
        }
        let suffix = &s[s.len() - len..];
        if MARKERS.iter().any(|m| m.starts_with(suffix)) {
            return len;
        }
    }
    0
}
