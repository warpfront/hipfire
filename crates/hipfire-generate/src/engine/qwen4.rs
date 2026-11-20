// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 AR on the engine loop: the Single bundle's forward on a one-owner
//! mesh (`Qwen4SingleMesh`), session state through the bundle's
//! `SessionDriver`, emit through the unchanged `QwenArSemanticProducer`.

use super::{
    admit_budget, drive, DoneMetrics, EngineFamily, Exit, Finish, Next, RequestRun, Verdict,
};
use crate::ar::{
    emit_generation_start, qwen_ar_cache_action, qwen_ar_raw_commit_token,
    qwen_ar_think_budget_step, GenerationRoute, QwenArRawCommitDisposition, QwenArSemanticProducer,
    QwenArThinkBudgetAction,
};
use crate::common::{
    emit_committed_event, emit_fail_closed_error, emit_spec_cancel_after_rollback,
    production_fail_closed_rollback,
};
use hipfire_arch_qwen4::bundle::{Qwen4Bundle, Qwen4SingleMesh};
use hipfire_engine::emit::{emit_active_attempt_error, stage_terminal_tool_calls};
use hipfire_engine::terminal::{await_client_terminal_commit, check_abort, ClientTerminalDecision};
use hipfire_engine::turn_cache::apply_turn_cache_action;
use hipfire_loader::LoadedModel;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::sampler::SamplerConfig;
use hipfire_runtime::session_cache::SessionRoute;
use rdna_compute::{DType, GpuTensor};
use std::any::Any;
use std::ops::Range;
use std::time::Instant;

const ROUTE: GenerationRoute = GenerationRoute::Qwen4Ar;

/// The request controls the Qwen4 AR route hands the engine.
pub struct ArRequest<'a> {
    pub sampler: SamplerConfig,
    pub max_tokens: usize,
    pub max_think_tokens: usize,
    pub stop: &'a [String],
    pub min_tokens: usize,
    pub tools: bool,
    pub seed: u32,
}

fn bundle(m: &mut LoadedModel) -> Result<&mut Qwen4Bundle, String> {
    m.state
        .as_deref_mut()
        .and_then(|state| (state as &mut dyn Any).downcast_mut::<Qwen4Bundle>())
        .ok_or_else(|| "qwen4 engine: the Qwen4 bundle disappeared".to_string())
}

/// Qwen4 AR of the rendered `prompt` on the engine loop. Emits exactly one
/// terminal; every error and abort exit after `gen_start` rolls back once.
pub fn generate_ar(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &[u32],
    started_in_think: bool,
    request: ArRequest<'_>,
) {
    let refuse = |stdout: &mut std::io::Stdout, message: &str, class: &str| {
        emit_active_attempt_error(stdout, Some(id), message, class, false, false);
    };
    let budget = match admit_budget(
        prompt.len(),
        request.max_tokens,
        m.max_seq,
        crate::common::fit_max_tokens,
    ) {
        Ok(budget) => budget,
        Err(message) if request.max_tokens == 0 => return refuse(stdout, &message, "validation"),
        Err(message) => return refuse(stdout, &format!("qwen4 {message}"), "context_length"),
    };
    let (n_layers, vocab, eos) = match bundle(m) {
        Ok(b) => (
            b.config.num_hidden_layers,
            b.config.vocab_size,
            b.config.eos_token_id,
        ),
        Err(message) => return refuse(stdout, &message, "internal"),
    };
    if let Some(state) = m.state.as_deref_mut() {
        state.session_request_write_limit(prompt.len().saturating_add(budget).saturating_add(1));
    }
    let reused = crate::qwen::plan_qwen4_session(m, prompt, SessionRoute::Ar);
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    Gpus::lend_single(gpu, n_layers, |gpus| {
        let tensors = (|| {
            let device = &mut gpus.devices[0];
            Ok::<_, String>([
                device
                    .zeros(&[vocab], DType::F32)
                    .map_err(|e| e.to_string())?,
                device
                    .zeros(&[vocab], DType::F32)
                    .map_err(|e| e.to_string())?,
                device
                    .zeros(&[std::mem::size_of::<u32>()], DType::Raw)
                    .map_err(|e| e.to_string())?,
            ])
        })();
        let [prompt_logits, decode_logits, top1] = match tensors {
            Ok(tensors) => tensors,
            Err(error) => {
                return refuse(stdout, &format!("qwen4 engine buffers: {error}"), "gpu");
            }
        };
        let mut family = Qwen4Engine {
            id,
            prompt,
            budget,
            eos,
            started_in_think,
            m,
            gpus: &mut *gpus,
            mesh: None,
            prompt_logits: &prompt_logits,
            decode_logits: &decode_logits,
            top1: &top1,
            out: &mut *stdout,
            semantic: Some(
                QwenArSemanticProducer::new_with_tool_protocol(id, started_in_think, request.tools)
                    .with_stop(request.stop)
                    .with_min_tokens(request.min_tokens, 0),
            ),
            streamed: Vec::new(),
            stream_bytes: Vec::new(),
            think_budget: hipfire_runtime::emit_text::ThinkBudget::new(request.max_think_tokens),
            t0: Instant::now(),
            prefill_end: None,
        };
        let mut run = RequestRun::new(prompt.len(), reused, budget, request.sampler, request.seed);
        emit_generation_start(ROUTE, family.out, id, started_in_think);
        let exit = match family.begin(reused) {
            Ok(()) => drive(&mut run, &mut family),
            Err(error) => Exit::Failed(error),
        };
        family.finish(exit, &run, reused);
        drop(family);
        for tensor in [prompt_logits, decode_logits, top1] {
            let _ = gpus.devices[0].free_tensor(tensor);
        }
    })
}

struct Qwen4Engine<'a> {
    id: &'a str,
    prompt: &'a [u32],
    budget: usize,
    eos: u32,
    started_in_think: bool,
    m: &'a mut LoadedModel,
    gpus: &'a mut Gpus,
    mesh: Option<Qwen4SingleMesh>,
    prompt_logits: &'a GpuTensor,
    decode_logits: &'a GpuTensor,
    top1: &'a GpuTensor,
    out: &'a mut std::io::Stdout,
    /// Taken by the finish (it consumes the producer).
    semantic: Option<QwenArSemanticProducer>,
    streamed: Vec<u32>,
    stream_bytes: Vec<u8>,
    think_budget: hipfire_runtime::emit_text::ThinkBudget,
    t0: Instant,
    prefill_end: Option<Instant>,
}

impl Qwen4Engine<'_> {
    /// Provision the one-owner mesh and begin the prompt's session.
    fn begin(&mut self, reused: usize) -> Result<(), String> {
        let bundle = bundle(self.m)?;
        let mesh = Qwen4SingleMesh::provision(bundle, self.gpus).map_err(|e| e.to_string())?;
        self.mesh = Some(mesh);
        bundle
            .single_mesh_prefill_begin(self.gpus, self.prompt, reused)
            .map_err(|e| e.to_string())
    }

    /// The one rollback of every error and abort exit: fail closed, attest,
    /// emit the terminal.
    fn roll_back(&mut self, error: Option<String>, generated: usize) {
        let epilogue =
            production_fail_closed_rollback(self.m, &mut self.gpus.devices[0], None, None);
        match error {
            None => emit_spec_cancel_after_rollback(self.out, self.id, generated, &epilogue),
            Some(message) => emit_fail_closed_error(
                self.out,
                Some(self.id),
                &format!("{} engine: {message}", ROUTE.name()),
                "gpu",
                false,
                &epilogue,
            ),
        }
    }

    fn finish(&mut self, exit: Exit, run: &RequestRun, reused: usize) {
        let finish = match exit {
            Exit::Finished(finish) => finish,
            Exit::Aborted => return self.roll_back(None, run.generated()),
            Exit::Failed(error) => return self.roll_back(Some(error), run.generated()),
        };
        let semantic = self.semantic.take().expect("one finish per request");
        let (route_finish, visible) = match semantic.finish(self.out, finish == Finish::Length) {
            Ok(done) => done,
            Err(error) => {
                return self.roll_back(Some(format!("semantic finish failed: {error}")), 0)
            }
        };
        let end = Instant::now();
        let prefill_end = self.prefill_end.unwrap_or(end);
        let metrics = DoneMetrics {
            prompt_tokens: self.prompt.len(),
            cached_tokens: reused,
            generated: run.generated(),
            prefill_ms: ms(prefill_end - self.t0),
            decode_ms: ms(end - prefill_end),
            total_ms: ms(end - self.t0),
        };
        let mut done = metrics.done_value(self.id, route_finish.finish_reason);
        stage_terminal_tool_calls(
            &mut done,
            route_finish.finish_reason,
            &route_finish.wire_tool_calls,
        );
        if await_client_terminal_commit(self.out, self.id, &done) != ClientTerminalDecision::Commit
        {
            return self.roll_back(None, run.generated());
        }
        crate::qwen::commit_session(self.m);
        let action = qwen_ar_cache_action(&route_finish, &visible);
        if action.store {
            let tok = self
                .m
                .tokenizer
                .as_ref()
                .expect("a Qwen4 load has a tokenizer");
            let im_end = tok.encode("<|im_end|>");
            let im_end = (im_end.len() == 1).then(|| im_end[0]);
            let nl: std::collections::HashSet<u32> = tok.encode("\n").into_iter().collect();
            let seq = crate::qwen::qwen_dflash_cache_seq(&self.streamed, im_end, &nl);
            let cache = &mut self.m.asst_turn_cache;
            let _ = apply_turn_cache_action(
                |fp, seq| {
                    cache.insert(
                        fp,
                        crate::qwen::qwen_cached_turn_entry(tok, seq, self.started_in_think),
                    );
                },
                &action,
                seq,
            );
        }
        crate::ar::emit_active_route_done_value(self.out, &done);
    }
}

fn ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

impl EngineFamily for Qwen4Engine<'_> {
    fn prefill(&mut self, rows: Range<usize>) -> Result<(usize, Option<Vec<f32>>), String> {
        let mesh = self
            .mesh
            .as_mut()
            .ok_or("qwen4 engine: mesh not provisioned")?;
        let end = bundle(self.m)?
            .single_mesh_prefill_step(mesh, self.gpus, self.prompt, self.prompt_logits)
            .map_err(|e| format!("prefill: {e}"))?;
        if end != rows.end {
            return Ok((end, None));
        }
        let device = &mut self.gpus.devices[0];
        let row = device
            .download_f32(self.prompt_logits)
            .map_err(|e| format!("prefill logits download: {e}"))?;
        self.m.seq_pos = self.prompt.len();
        self.m.conversation_tokens.extend_from_slice(self.prompt);
        self.prefill_end = Some(Instant::now());
        // Best effort: map context ahead for the whole budget before decode.
        if let Some(state) = self.m.state.as_deref_mut() {
            state.map_ahead_for_decode(device, self.prompt.len(), self.budget, 0);
        }
        Ok((end, Some(row)))
    }

    fn decode(&mut self, token: u32, top1: bool) -> Result<Next, String> {
        let mesh = self
            .mesh
            .as_mut()
            .ok_or("qwen4 engine: mesh not provisioned")?;
        let argmax = bundle(self.m)?
            .single_mesh_decode(
                mesh,
                self.gpus,
                token,
                self.decode_logits,
                top1.then_some(self.top1),
            )
            .map_err(|e| format!("decode: {e}"))?;
        match argmax {
            Some(next) => Ok(Next::Top1(next)),
            None => self.gpus.devices[0]
                .download_f32(self.decode_logits)
                .map(Next::Logits)
                .map_err(|e| format!("decode logits download: {e}")),
        }
    }

    fn emit(&mut self, token: u32) -> Result<(Verdict, Vec<u32>), String> {
        let elapsed = self.t0.elapsed().as_millis() as u64;
        let tokenizer = self
            .m
            .tokenizer
            .as_ref()
            .ok_or("qwen4 engine: no tokenizer")?;
        let fed = self.stream_bytes.len();
        tokenizer.decode_token_bytes_into(token, &mut self.stream_bytes);
        let new_bytes = &self.stream_bytes[fed..];
        let (conversation, seq_pos, streamed, id) = (
            &mut self.m.conversation_tokens,
            &mut self.m.seq_pos,
            &mut self.streamed,
            self.id,
        );
        let semantic = self
            .semantic
            .as_mut()
            .ok_or("qwen4 engine: emit after finish")?;
        let filter_stop = semantic
            .commit_and_classify(
                self.out,
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
        let tokenizer = self
            .m
            .tokenizer
            .as_ref()
            .ok_or("qwen4 engine: no tokenizer")?;
        if semantic.last_stops_active && (token == self.eos || tokenizer.is_terminator(token)) {
            return Ok((Verdict::Eos, Vec::new()));
        }
        if filter_stop {
            return Ok((Verdict::Stop, Vec::new()));
        }
        Ok(
            match qwen_ar_think_budget_step(
                &mut self.think_budget,
                &self.stream_bytes,
                self.started_in_think,
                tokenizer,
            ) {
                QwenArThinkBudgetAction::Continue => (Verdict::Continue, Vec::new()),
                QwenArThinkBudgetAction::Close(close) => (Verdict::Continue, close),
                QwenArThinkBudgetAction::Stop => (Verdict::Stop, Vec::new()),
            },
        )
    }

    fn history(&self) -> &[u32] {
        &self.m.conversation_tokens
    }

    fn aborted(&self) -> bool {
        check_abort(self.id)
    }
}
