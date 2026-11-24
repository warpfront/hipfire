// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The Gemma 4 wire protocol of the engine loop: the legacy route's stop set,
//! thought router and tool splitter ([`Gemma4TurnText`]) unchanged.

use super::chatml::{Host, ProtocolEmit, WireProtocol};
use super::{DoneMetrics, Emit, Verdict};
use crate::dense::{gemma4_finish_reason, Gemma4TurnText};
use hipfire_engine::terminal::check_abort;
use hipfire_loader::AsstTurnCache;
use hipfire_runtime::prompt_frame::ToolCall;
use hipfire_runtime::tokenizer::TokenTextStream;
use std::io::Stdout;

/// Gemma 4 as `generate_gemma4` speaks it, for one request.
pub struct Gemma4Wire<'a> {
    pub enable_thinking: bool,
    pub max_think_tokens: usize,
    pub tools: bool,
    /// Ids that end the turn (`<eos>`, `<turn|>`, 106, `<|tool_response>`).
    pub stop: &'a [u32],
}

impl WireProtocol for Gemma4Wire<'_> {
    fn started_in_think(&self) -> bool {
        false
    }

    fn open<'a>(&'a self, host: Host<'a>) -> Box<dyn ProtocolEmit + 'a> {
        Box::new(Gemma4Emit {
            stream: Gemma4Stream::new(self),
            spec: self,
            host,
        })
    }

    /// The legacy terminal's usage fields: the whole prompt, the wall time,
    /// and `tok_s` as the decode rate.
    fn stamp_done(&self, done: &mut serde_json::Value, metrics: &DoneMetrics) {
        done["prompt_tokens"] = metrics.prompt_tokens.into();
        done["total_ms"] = ((metrics.total_ms * 10.0).round() / 10.0).into();
        done["tok_s"] = done["decode_tok_s"].clone();
    }
}

/// One request's Gemma 4 stream state, apart from the host state it commits
/// into.
pub struct Gemma4Stream {
    /// Multi-byte characters span tokens: decode incrementally.
    text: TokenTextStream,
    turn: Gemma4TurnText,
}

impl Gemma4Stream {
    pub fn new(spec: &Gemma4Wire<'_>) -> Self {
        Self {
            text: TokenTextStream::new(),
            turn: Gemma4TurnText::new(spec.enable_thinking, spec.max_think_tokens, spec.tools),
        }
    }

    /// A stop id ends the turn uncommitted; any other token is committed and
    /// routed to the wire.
    pub fn emit(
        &mut self,
        spec: &Gemma4Wire<'_>,
        host: &mut Host<'_>,
        token: u32,
    ) -> Result<(Verdict, Vec<u32>), String> {
        if spec.stop.contains(&token) {
            return Ok((Verdict::Eos, Vec::new()));
        }
        let frag = self.text.push(host.tokenizer, token);
        self.turn
            .push(host.out, host.id, &frag, || None)
            .map_err(|error| format!("malformed Gemma 4 tool call: {error}"))?;
        host.conversation.push(token);
        *host.seq_pos += 1;
        Ok((Verdict::Continue, Vec::new()))
    }

    /// Flush the held text; a cut turn drops a half-written tool call.
    pub fn finish(
        &mut self,
        out: &mut Stdout,
        id: &str,
        length: bool,
    ) -> Result<(&'static str, Vec<ToolCall>), String> {
        self.turn
            .finish(out, id, &self.text.flush(), !length)
            .map_err(|error| format!("malformed Gemma 4 tool call: {error}"))?;
        let calls = self.turn.calls();
        Ok((gemma4_finish_reason(!length, calls.len()), calls.to_vec()))
    }
}

struct Gemma4Emit<'a> {
    spec: &'a Gemma4Wire<'a>,
    host: Host<'a>,
    stream: Gemma4Stream,
}

impl Emit for Gemma4Emit<'_> {
    fn commit_prompt(&mut self) {
        *self.host.seq_pos = self.host.prompt.len();
        self.host.conversation.extend_from_slice(self.host.prompt);
    }

    fn emit(&mut self, token: u32) -> Result<(Verdict, Vec<u32>), String> {
        self.stream.emit(self.spec, &mut self.host, token)
    }

    fn history(&self) -> &[u32] {
        self.host.conversation
    }

    fn aborted(&self) -> bool {
        check_abort(self.host.id)
    }
}

impl ProtocolEmit for Gemma4Emit<'_> {
    fn finish(&mut self, length: bool) -> Result<(&'static str, Vec<ToolCall>), String> {
        self.stream.finish(self.host.out, self.host.id, length)
    }

    fn out(&mut self) -> &mut Stdout {
        self.host.out
    }

    fn clear_history(&mut self) {
        *self.host.seq_pos = 0;
        self.host.conversation.clear();
    }

    /// The Gemma 4 route keeps no assistant-turn cache.
    fn commit_turn(&mut self, _cache: &mut AsstTurnCache) {}
}
