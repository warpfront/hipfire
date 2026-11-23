// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The wire-protocol seam of the engine loop and its ChatML implementation
//! (Qwen-style `<think>` / `<tool_call>` markup, `<|im_end|>` turns). The
//! emitter is `QwenArSemanticProducer` unchanged; a family picks its protocol
//! by declaration, the loop never names one.

use super::{Emit, Verdict};
use crate::ar::{
    qwen_ar_cache_action, qwen_ar_raw_commit_token, qwen_ar_think_budget_step,
    QwenArRawCommitDisposition, QwenArRouteFinish, QwenArSemanticProducer, QwenArThinkBudgetAction,
};
use crate::common::emit_committed_event;
use hipfire_engine::terminal::check_abort;
use hipfire_engine::turn_cache::apply_turn_cache_action;
use hipfire_loader::AsstTurnCache;
use hipfire_runtime::prompt_frame::ToolCall;
use hipfire_runtime::tokenizer::Tokenizer;
use std::io::Stdout;
use std::time::Instant;

/// The host state a request's emitter commits into.
pub struct Host<'a> {
    pub id: &'a str,
    pub out: &'a mut Stdout,
    pub conversation: &'a mut Vec<u32>,
    pub seq_pos: &'a mut usize,
    pub tokenizer: &'a Tokenizer,
    pub prompt: &'a [u32],
    pub t0: Instant,
}

/// A wire protocol, as a family's route declares it for one request.
pub trait WireProtocol {
    /// Whether the rendered prompt left the model inside a reasoning span.
    fn started_in_think(&self) -> bool;
    fn open<'a>(&'a self, host: Host<'a>) -> Box<dyn ProtocolEmit + 'a>;
}

/// One request's emitter: [`Emit`] for the loop, plus the terminal.
pub trait ProtocolEmit: Emit {
    /// Flush the stream; the finish reason and the calls the terminal carries.
    fn finish(&mut self, length: bool) -> Result<(&'static str, Vec<ToolCall>), String>;
    fn out(&mut self) -> &mut Stdout;
    /// Forget the committed history (a rolled-back request).
    fn clear_history(&mut self);
    /// After the client committed the terminal: store the turn for reuse.
    fn commit_turn(&mut self, cache: &mut AsstTurnCache);
}

/// ChatML as the Qwen AR routes speak it.
pub struct ChatMl<'a> {
    pub started_in_think: bool,
    pub tools: bool,
    pub stop: &'a [String],
    pub min_tokens: usize,
    pub max_think_tokens: usize,
    /// Ids that end the turn besides the tokenizer's terminators.
    pub eos: &'a [u32],
}

impl WireProtocol for ChatMl<'_> {
    fn started_in_think(&self) -> bool {
        self.started_in_think
    }

    fn open<'a>(&'a self, host: Host<'a>) -> Box<dyn ProtocolEmit + 'a> {
        Box::new(ChatMlEmit {
            stream: ChatMlStream::new(self, host.id),
            spec: self,
            host,
            streamed: Vec::new(),
        })
    }
}

/// One request's ChatML stream state, apart from the host state it commits
/// into: the engine's scheduler keeps it per request across its steps.
pub struct ChatMlStream {
    /// Taken by the finish (it consumes the producer).
    semantic: Option<QwenArSemanticProducer>,
    think_budget: hipfire_runtime::emit_text::ThinkBudget,
    stream_bytes: Vec<u8>,
    finished: Option<(QwenArRouteFinish, String)>,
}

impl ChatMlStream {
    pub fn new(spec: &ChatMl<'_>, id: &str) -> Self {
        Self {
            semantic: Some(
                QwenArSemanticProducer::new_with_tool_protocol(
                    id,
                    spec.started_in_think,
                    spec.tools,
                )
                .with_stop(spec.stop)
                .with_min_tokens(spec.min_tokens, 0),
            ),
            think_budget: hipfire_runtime::emit_text::ThinkBudget::new(spec.max_think_tokens),
            stream_bytes: Vec::new(),
            finished: None,
        }
    }

    /// Commit and emit `token` (it has run) into `host`, appending it to
    /// `streamed`; returns how it classified and any template tokens to
    /// force next.
    pub fn emit(
        &mut self,
        spec: &ChatMl<'_>,
        host: &mut Host<'_>,
        streamed: &mut Vec<u32>,
        token: u32,
    ) -> Result<(Verdict, Vec<u32>), String> {
        let elapsed = host.t0.elapsed().as_millis() as u64;
        let tokenizer = host.tokenizer;
        let fed = self.stream_bytes.len();
        tokenizer.decode_token_bytes_into(token, &mut self.stream_bytes);
        let new_bytes = &self.stream_bytes[fed..];
        let Host {
            id,
            out,
            conversation,
            seq_pos,
            ..
        } = host;
        let semantic = self.semantic.as_mut().ok_or("emit after finish")?;
        let filter_stop = semantic
            .commit_and_classify(
                *out,
                token,
                || {
                    let position = qwen_ar_raw_commit_token(
                        conversation,
                        streamed,
                        seq_pos,
                        token,
                        QwenArRawCommitDisposition::ClassifiedVisible,
                    );
                    (position, new_bytes)
                },
                |position, out| emit_committed_event(out, id, token, position, elapsed),
            )
            .map_err(|e| format!("semantic classify failed: {e}"))?;
        // An EOS id below `min_tokens` is an ordinary token.
        if semantic.last_stops_active
            && (spec.eos.contains(&token) || tokenizer.is_terminator(token))
        {
            return Ok((Verdict::Eos, Vec::new()));
        }
        if filter_stop {
            return Ok((Verdict::Stop, Vec::new()));
        }
        Ok(
            match qwen_ar_think_budget_step(
                &mut self.think_budget,
                &self.stream_bytes,
                spec.started_in_think,
                tokenizer,
            ) {
                QwenArThinkBudgetAction::Continue => (Verdict::Continue, Vec::new()),
                QwenArThinkBudgetAction::Close(close) => (Verdict::Continue, close),
                QwenArThinkBudgetAction::Stop => (Verdict::Stop, Vec::new()),
            },
        )
    }

    /// Flush the stream; the finish reason and the calls the terminal carries.
    pub fn finish(
        &mut self,
        out: &mut Stdout,
        length: bool,
    ) -> Result<(&'static str, Vec<ToolCall>), String> {
        let semantic = self.semantic.take().ok_or("one finish per request")?;
        let (finish, visible) = semantic
            .finish(out, length)
            .map_err(|e| format!("semantic finish failed: {e}"))?;
        let terminal = (finish.finish_reason, finish.wire_tool_calls.clone());
        self.finished = Some((finish, visible));
        Ok(terminal)
    }

    /// The verbatim emitted span (`streamed`) under the fingerprint every
    /// Qwen AR route stores (`qwen_cached_turn_entry`), so the next turn
    /// splices these ids.
    pub fn commit_turn(
        &self,
        spec: &ChatMl<'_>,
        tokenizer: &Tokenizer,
        streamed: &[u32],
        cache: &mut AsstTurnCache,
    ) {
        let Some((finish, visible)) = &self.finished else {
            return;
        };
        let action = qwen_ar_cache_action(finish, visible);
        if !action.store {
            return;
        }
        let im_end = tokenizer.encode("<|im_end|>");
        let im_end = (im_end.len() == 1).then(|| im_end[0]);
        let nl: std::collections::HashSet<u32> = tokenizer.encode("\n").into_iter().collect();
        let seq = crate::qwen::qwen_dflash_cache_seq(streamed, im_end, &nl);
        let started_in_think = spec.started_in_think;
        let _ = apply_turn_cache_action(
            |fp, seq| {
                cache.insert(
                    fp,
                    crate::qwen::qwen_cached_turn_entry(tokenizer, seq, started_in_think),
                );
            },
            &action,
            seq,
        );
    }
}

struct ChatMlEmit<'a> {
    spec: &'a ChatMl<'a>,
    host: Host<'a>,
    stream: ChatMlStream,
    streamed: Vec<u32>,
}

impl Emit for ChatMlEmit<'_> {
    fn commit_prompt(&mut self) {
        *self.host.seq_pos = self.host.prompt.len();
        self.host.conversation.extend_from_slice(self.host.prompt);
    }

    fn emit(&mut self, token: u32) -> Result<(Verdict, Vec<u32>), String> {
        self.stream
            .emit(self.spec, &mut self.host, &mut self.streamed, token)
    }

    fn history(&self) -> &[u32] {
        self.host.conversation
    }

    fn aborted(&self) -> bool {
        check_abort(self.host.id)
    }
}

impl ProtocolEmit for ChatMlEmit<'_> {
    fn finish(&mut self, length: bool) -> Result<(&'static str, Vec<ToolCall>), String> {
        self.stream.finish(self.host.out, length)
    }

    fn out(&mut self) -> &mut Stdout {
        self.host.out
    }

    fn clear_history(&mut self) {
        *self.host.seq_pos = 0;
        self.host.conversation.clear();
    }

    fn commit_turn(&mut self, cache: &mut AsstTurnCache) {
        self.stream
            .commit_turn(self.spec, self.host.tokenizer, &self.streamed, cache);
    }
}
