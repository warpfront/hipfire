// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Raon-1-mini-preview (arch 19, `GenerationRoute::RaonAr` / `RaonSpec`) serving
//! body.
//!
//! A deliberately small adapter. Everything with a wire contract is the shared
//! engine's: [`crate::ar::generate_ar_with_forward`] owns the
//! `gen_start` → events → one terminal lifecycle, the
//! [`QwenArSemanticProducer`](crate::ar::QwenArSemanticProducer) think router /
//! XML tool router / stop matcher, the explicit `max_think_tokens` budget, the
//! client commit handshake and abort/rollback. This module owns only what is
//! Raon-specific:
//!
//! * the prompt: the checkpoint's own Jinja template, rendered through
//!   [`JinjaChatFrame`] and classified by the shared
//!   [`render_assistant_prefix`] (thinking-on ends `<think>\n`, so the stream
//!   starts in reasoning; thinking-off renders the closed
//!   `<think></think>\n\n` pair, so it starts visible; a second opener is never
//!   appended);
//! * the `generation_config.json` sampling defaults and the generated-only
//!   presence penalty ([`RaonSampling::resolve`]);
//! * the two bundle callbacks (batched prefill in bounded chunks, one decode
//!   step per token) that hand logits to the shared body;
//! * the speculative adapter [`generate_raon_spec`]: the shared DFlash2 body
//!   ([`crate::qwen::generate_dflash`]) serves the same turn, from the same
//!   prompt tokens and sampling, under the same wire contract.
//!
//! Wire contract: semantic v2. The route emits typed `reasoning` / `token` /
//! tool-call events through the router-backed producer, so the arch-19
//! carrier MUST advertise `semantic_contract_version: Some(2)`; `gen_start`
//! reads it from `carrier_for(19).caps()` (see the contract test below).

use hipfire_loader::LoadedModel;
use hipfire_runtime::prompt_frame::{AssistantPrefix, JinjaChatFrame, Message, Role};
use hipfire_runtime::tokenizer::Tokenizer;
use std::io::Write;

use crate::ar::{generate_ar_with_forward, GenerationRoute};
use hipfire_engine::emit::{render_assistant_prefix, render_tail_opens_think};

/// Request sampling knobs exactly as `generate` received them.
#[derive(Debug, Clone, Copy)]
pub struct RaonSampling {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// The request (or the registry card via the client) explicitly set a
    /// non-temperature sampling control. Model defaults never override these.
    pub user_explicit: bool,
}

/// [`RaonSampling`] after model defaults and the generated-history window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedRaonSampling {
    pub top_k: Option<u32>,
    pub presence_penalty: f32,
    pub repeat_window: usize,
}

impl RaonSampling {
    /// Apply `generation_config.json` defaults (`top_k`, `presence_penalty`)
    /// when the request is a sampled one that set nothing explicit, and size
    /// the penalty window to cover the whole generation.
    ///
    /// * Greedy (`temp <= 0`) is an explicit benchmark setting: no default is
    ///   ever added, so presence cannot perturb an argmax run.
    /// * A request that set any non-temperature control keeps exactly what it
    ///   sent; the card's values reach it through the client/registry layer.
    /// * Penalties see only generated tokens (the body is told so via
    ///   `penalty_generated_only`); a window narrower than the generation would
    ///   forget early tokens, so the window is at least `max_tokens`.
    pub fn resolve(
        &self,
        defaults: &hipfire_arch_raon::GenerationDefaults,
        max_tokens: usize,
    ) -> ResolvedRaonSampling {
        let sampled = self.temp > 1e-6;
        let fill = sampled && !self.user_explicit;
        let top_k = match self.top_k {
            Some(k) => Some(k),
            None if fill => Some(defaults.top_k),
            None => None,
        };
        let presence_penalty = if fill && self.presence_penalty == 0.0 {
            defaults.presence_penalty
        } else {
            self.presence_penalty
        };
        ResolvedRaonSampling {
            top_k,
            presence_penalty,
            repeat_window: self.repeat_window.max(max_tokens),
        }
    }

    /// Whether the AR producer runs a non-neutral repeat/presence/frequency
    /// penalty for this request once [`Self::resolve`] has applied the model
    /// defaults. Route selection reads this, not the request's raw penalties:
    /// a sampled request that set nothing explicit is penalized by the card's
    /// presence default, and the DFlash2 verifiers implement no penalties.
    pub fn penalized_after_defaults(
        &self,
        defaults: &hipfire_arch_raon::GenerationDefaults,
    ) -> bool {
        self.repeat_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.resolve(defaults, 0).presence_penalty != 0.0
    }
}

/// A rendered, tokenized prompt plus the stream's starting channel.
#[derive(Debug, Clone, PartialEq)]
pub struct RaonPrompt {
    pub text: String,
    pub tokens: Vec<u32>,
    /// The live assistant prefix leaves the model inside an open `<think>`.
    pub started_in_think: bool,
}

fn message(role: Role, content: &str) -> Message {
    Message {
        role,
        content: content.to_string(),
        reasoning_content: None,
        name: None,
        rendered_name: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tool_plan: String::new(),
    }
}

/// Render the official template over the whole conversation to text.
fn render_raon_text(
    tokenizer: &Tokenizer,
    template: &str,
    system_prompt: Option<&str>,
    prompt: &str,
    messages_history: Option<&[Message]>,
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
) -> Result<String, String> {
    let frame = JinjaChatFrame {
        tokenizer,
        template,
        system: system_prompt,
        user: prompt,
        enable_thinking,
        bos_token: Some(""),
        reasoning_strength: None,
        reasoning_effort: None,
    };
    let synthesized: Vec<Message>;
    let messages: &[Message] = match messages_history {
        Some(h) if !h.is_empty() => h,
        _ => {
            let mut v = Vec::with_capacity(2);
            if let Some(sys) = system_prompt.filter(|s| !s.trim().is_empty()) {
                v.push(message(Role::System, sys));
            }
            v.push(message(Role::User, prompt));
            synthesized = v;
            &synthesized
        }
    };
    let tools = tools.filter(|t| !t.is_empty());
    frame
        .render_messages(messages, tools, None)
        .map_err(|e| format!("raon chat template render failed: {e}"))
}

/// Render the official template over the whole conversation and tokenize it.
///
/// `started_in_think` comes from the rendered live assistant turn
/// ([`render_assistant_prefix`]), never from `enable_thinking`, the request's
/// `assistant_prefix`, or the tokenizer's token inventory.
pub fn render_raon_prompt(
    tokenizer: &Tokenizer,
    template: &str,
    system_prompt: Option<&str>,
    prompt: &str,
    messages_history: Option<&[Message]>,
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
) -> Result<RaonPrompt, String> {
    let text = render_raon_text(
        tokenizer,
        template,
        system_prompt,
        prompt,
        messages_history,
        tools,
        enable_thinking,
    )?;
    let started_in_think = render_assistant_prefix(&text) == AssistantPrefix::OpenThink;
    let tokens = tokenizer.encode(&text);
    Ok(RaonPrompt {
        text,
        tokens,
        started_in_think,
    })
}

fn raon_error(stdout: &mut impl Write, id: &str, message: &str, class: &str) {
    crate::dense::emit_active_attempt_error(stdout, Some(id), message, class, false, false);
    let _ = stdout.flush();
}

/// Serve one Raon-1-mini turn. Validation refusals are correlated pre-start
/// errors; everything after `gen_start` is the shared body's lifecycle.
#[allow(clippy::too_many_arguments)]
pub fn generate_raon(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    messages_history: Option<&[Message]>,
    tools: Option<&[serde_json::Value]>,
    sampling: RaonSampling,
    max_tokens: usize,
    max_think_tokens: usize,
    stop: &[String],
    enable_thinking: bool,
) {
    let Some((vocab, eos_tokens, defaults, cap)) = m.raon().map(|b| {
        (
            b.config.vocab,
            b.eos.clone(),
            b.config.generation.clone(),
            b.max_seq(),
        )
    }) else {
        raon_error(
            stdout,
            id,
            "raon AR dispatch requires a loaded raon1_mini bundle",
            "internal",
        );
        return;
    };
    if max_tokens == 0 {
        raon_error(stdout, id, "max_tokens must be > 0", "validation");
        return;
    }
    let rendered = match (m.tokenizer.as_ref(), m.chat_template.as_deref()) {
        (None, _) => Err("tokenizer not loaded".to_string()),
        (_, None) => Err(
            "raon1_mini requires its embedded chat template (re-quantize with the \
             checkpoint's chat_template.jinja); no ChatML fallback exists for this arch"
                .to_string(),
        ),
        (Some(tokenizer), Some(template)) => render_raon_prompt(
            tokenizer,
            template,
            system_prompt,
            prompt,
            messages_history,
            tools,
            enable_thinking,
        ),
    };
    let RaonPrompt {
        tokens: prompt_tokens,
        started_in_think,
        ..
    } = match rendered {
        Ok(p) => p,
        Err(message) => {
            raon_error(stdout, id, &message, "validation");
            return;
        }
    };
    if prompt_tokens.is_empty() {
        raon_error(
            stdout,
            id,
            "raon prompt rendered to zero tokens",
            "validation",
        );
        return;
    }
    // One decode forward per committed token, including the last: +1 trailer.
    let max_tokens = crate::common::fit_max_tokens(max_tokens, prompt_tokens.len() + 1, cap);
    if prompt_tokens
        .len()
        .checked_add(max_tokens)
        .and_then(|n| n.checked_add(1))
        .is_none_or(|n| n > cap)
    {
        raon_error(
            stdout,
            id,
            &format!(
                "prompt exceeds context capacity: prompt={} + max_tokens={} + trailer=1 > \
                 capacity={cap} — reload model with a larger max_seq",
                prompt_tokens.len(),
                max_tokens
            ),
            "context_length",
        );
        return;
    }

    let resolved = sampling.resolve(&defaults, max_tokens);
    // Prior turns reach the model through the rendered prompt, never through
    // retained state (PLAN 3.5: no prefix hit until RDA+conv snapshots exist).
    // The device reset itself runs inside the prefill callback, after
    // `gen_start`, so a failed reset takes the one fail-closed terminal.
    m.seq_pos = 0;
    m.conversation_tokens.clear();

    generate_ar_with_forward(
        m,
        gpu,
        stdout,
        id,
        GenerationRoute::RaonAr,
        &prompt_tokens,
        vocab,
        &eos_tokens,
        sampling.temp,
        sampling.top_p,
        resolved.top_k,
        sampling.min_p,
        max_tokens,
        sampling.repeat_penalty,
        resolved.repeat_window,
        resolved.presence_penalty,
        sampling.frequency_penalty,
        max_think_tokens,
        started_in_think,
        stop,
        tools.is_some_and(|t| !t.is_empty()),
        // No prefix reuse: every prompt token is computed.
        0,
        // The GPU penalty prepass is not wired for this arch: penalized greedy
        // downloads the row and samples on the host.
        false,
        true,
        |model, device, tokens, logits| {
            let bundle = model
                .raon_mut()
                .ok_or_else(|| "raon bundle disappeared before prefill".to_string())?;
            bundle.reset(device)?;
            let total = tokens.len();
            let mut done = 0usize;
            for chunk in tokens.chunks(bundle.prefill_chunk_rows()) {
                bundle.prefill(device, chunk)?;
                done += chunk.len();
                hipfire_runtime::prefill_progress::report(done, total);
            }
            copy_logits(device, bundle.logits(), logits, vocab)
        },
        |model, device, token, logits| {
            let token = match token {
                Some(t) => t,
                // Greedy token: argmax kernel on the decode stream and a 4-byte
                // readback ordered on it (no per-token null-stream hop, no
                // per-token malloc/free).
                None => rdna_compute::tensor_ops::argmax_f32_host_stream_ordered(
                    device,
                    &logits.sub_offset(0, vocab),
                )
                .map_err(|e| format!("raon logits argmax failed: {e}"))?,
            };
            let bundle = model
                .raon_mut()
                .ok_or_else(|| "raon bundle disappeared during decode".to_string())?;
            bundle.decode_step(device, token)?;
            copy_logits(device, bundle.logits(), logits, vocab)?;
            Ok(token)
        },
        |_| {},
    );
}

/// The shared speculative body classifies the opening channel from the
/// rendered tail ([`render_tail_opens_think`]); the AR route classifies the
/// live turn ([`render_assistant_prefix`]). A render on which the two
/// disagree would start the two routes in different channels, so it is
/// served by the AR route.
fn spec_start_channel_matches_ar(rendered: &str) -> bool {
    (render_assistant_prefix(rendered) == AssistantPrefix::OpenThink)
        == render_tail_opens_think(rendered)
}

/// Whether the shared speculative body renders this request to exactly the
/// prompt [`generate_raon`] renders. It always uses the model's Jinja
/// template when `HIPFIRE_JINJA_CHAT` allows it and the model has one, and
/// falls back to plain ChatML on a render error: a request that would take
/// either of those differences, or whose start channel is classified
/// differently, must run on the AR route.
fn spec_prompt_matches_ar(
    m: &LoadedModel,
    system_prompt: Option<&str>,
    prompt: &str,
    messages_history: Option<&[Message]>,
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
) -> bool {
    if hipfire_config::developer_var("HIPFIRE_JINJA_CHAT")
        .ok()
        .as_deref()
        == Some("0")
    {
        return false;
    }
    let (Some(tokenizer), Some(template)) = (m.tokenizer.as_ref(), m.chat_template.as_deref())
    else {
        return false;
    };
    render_raon_text(
        tokenizer,
        template,
        system_prompt,
        prompt,
        messages_history,
        tools,
        enable_thinking,
    )
    .is_ok_and(|text| spec_start_channel_matches_ar(&text))
}

/// Serve one Raon-1-mini turn on `GenerationRoute::RaonSpec`: the shared
/// DFlash2 body ([`crate::qwen::generate_dflash`]) drives the spec loop and
/// owns the wire lifecycle (`gen_start` contract v2, reasoning/tool events
/// through the arch carrier's `SpecEmit`, stop handling, one terminal), and
/// this adapter makes it commit the turn [`generate_raon`] would:
///
/// * the prompt is the same render: blank system prompts, empty histories and
///   empty tool lists are dropped exactly as [`render_raon_text`] drops them,
///   no `reasoning_effort` is forwarded (the template reads none), and a
///   request the shared body would render differently
///   ([`spec_prompt_matches_ar`]) takes the AR route;
/// * there is no prefix reuse (as on the AR route): the device state, the
///   conversation mirror and the assistant-turn cache are cleared first, so
///   the shared body plans a cold prefill of the full render and never
///   splices cached tokens into it;
/// * the sampling is the AR route's: `top_k` after the card defaults
///   ([`RaonSampling::resolve`]); route selection already kept every request
///   that the defaults penalize on the AR route.
///
/// A speculative context-capacity miss emitted nothing, and falls through to
/// [`generate_raon`].
#[allow(clippy::too_many_arguments)]
pub fn generate_raon_spec(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    messages_history: Option<&[Message]>,
    tools: Option<&[serde_json::Value]>,
    sampling: RaonSampling,
    cactus_delta: f32,
    request_seed: u64,
    max_tokens: usize,
    max_think_tokens: usize,
    stop: &[String],
    enable_thinking: bool,
) {
    let ar = |m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu, stdout: &mut std::io::Stdout| {
        generate_raon(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            messages_history,
            tools,
            sampling,
            max_tokens,
            max_think_tokens,
            stop,
            enable_thinking,
        )
    };
    let Some(defaults) = m.raon().map(|b| b.config.generation.clone()) else {
        return ar(m, gpu, stdout);
    };
    if !spec_prompt_matches_ar(
        m,
        system_prompt,
        prompt,
        messages_history,
        tools,
        enable_thinking,
    ) {
        return ar(m, gpu, stdout);
    }
    let resolved = sampling.resolve(&defaults, max_tokens);
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    m.asst_turn_cache.clear();
    if crate::qwen::generate_dflash(
        m,
        gpu,
        stdout,
        id,
        prompt,
        system_prompt.filter(|s| !s.trim().is_empty()),
        max_tokens,
        max_think_tokens,
        // The Jinja render owns the generation suffix once it succeeds, and
        // `spec_prompt_matches_ar` proved that it does.
        AssistantPrefix::Plain,
        None,
        None,
        tools.filter(|t| !t.is_empty()),
        messages_history.filter(|h| !h.is_empty()),
        stop,
        sampling.temp,
        sampling.top_p,
        resolved.top_k,
        sampling.min_p.unwrap_or(0.0),
        sampling.repeat_penalty,
        resolved.repeat_window,
        resolved.presence_penalty,
        sampling.frequency_penalty,
        cactus_delta,
        request_seed,
        None,
        enable_thinking,
    ) {
        return;
    }
    ar(m, gpu, stdout)
}

fn copy_logits(
    gpu: &mut rdna_compute::Gpu,
    src: &rdna_compute::GpuTensor,
    dst: &rdna_compute::GpuTensor,
    vocab: usize,
) -> Result<(), String> {
    gpu.memcpy_dtod_auto(&dst.buf, &src.buf, vocab * std::mem::size_of::<f32>())
        .map_err(|e| format!("raon logits copy failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ar::{
        active_generation_route, emit_active_route_done_value, emit_generation_start,
        generation_route_adapter, select_generation_route, GenerationRouteInputs,
        GenerationRouteScope, QwenArRawCommitDisposition, QwenArSemanticProducer,
    };
    use hipfire_engine::terminal::{
        activate_terminal_control, clear_terminal_control, set_active_attempt_id,
    };
    use hipfire_runtime::prompt_frame::ToolCall;

    const TEMPLATE: &str =
        include_str!("../../hipfire-runtime/tests/fixtures/raon_tokenizer/chat_template.jinja");

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

    /// Byte-level stand-in tokenizer (the template never reads the vocab).
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
        Tokenizer::from_hf_json(&j.to_string()).expect("stand-in tokenizer")
    }

    fn parse_events(sink: &[u8]) -> Vec<serde_json::Value> {
        std::str::from_utf8(sink)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn render(enable_thinking: bool, tools: Option<&[serde_json::Value]>) -> RaonPrompt {
        render_raon_prompt(
            &tokenizer(),
            TEMPLATE,
            Some("Be brief."),
            "hi",
            None,
            tools,
            enable_thinking,
        )
        .expect("render")
    }

    fn inputs(arch_id: u32) -> GenerationRouteInputs {
        GenerationRouteInputs {
            arch_id,
            ep: false,
            dense_tp: false,
            pp: 1,
            has_speculator: false,
            speculator_is_mtp: false,
            speculator_is_dflash2: false,
            deepseek4_spec_requested: false,
            ngram_can_sample: false,
            temp: 1.0,
            user_explicit_sampling: false,
            min_p: None,
            nonneutral_penalties: false,
            force_ar_chat: false,
            temp_spec_env_off: false,
            fast_sample_on: true,
            supports_temp_swor: false,
            supports_chain_nucleus_verify: false,
            kv_adaptive: false,
        }
    }

    #[test]
    fn raon_route_selection_inventory_and_capabilities() {
        assert_eq!(
            select_generation_route(&inputs(19)),
            GenerationRoute::RaonAr
        );
        // A speculator that is not the DFlash2 drafter, greedy temp or sampling
        // penalties never change the route.
        for (has_speculator, temp, pen) in
            [(true, 0.0, false), (false, 0.0, true), (true, 1.0, true)]
        {
            let i = GenerationRouteInputs {
                has_speculator,
                temp,
                nonneutral_penalties: pen,
                ..inputs(19)
            };
            assert_eq!(select_generation_route(&i), GenerationRoute::RaonAr);
        }
        assert!(GenerationRoute::ALL.contains(&GenerationRoute::RaonAr));
        assert_eq!(GenerationRoute::RaonAr.name(), "raon_ar");
        assert!(GenerationRoute::RaonAr.supports_tools());
        assert!(GenerationRoute::RaonAr.supports_stop());
        assert!(!GenerationRoute::RaonAr.runs_qwen35_ar_body());
        let adapter = generation_route_adapter(GenerationRoute::RaonAr).expect("adapter");
        assert_eq!(adapter.route, GenerationRoute::RaonAr);
        // No other arch lands on a Raon route.
        for arch in [0, 5, 6, 9, 11, 15, 16, 17, 18, 20] {
            for route in [
                select_generation_route(&inputs(arch)),
                select_generation_route(&dflash2_inputs(arch, 0.0)),
            ] {
                assert_ne!(route, GenerationRoute::RaonAr);
                assert_ne!(route, GenerationRoute::RaonSpec);
            }
        }
    }

    /// A loaded DFlash2 speculator (greedy or sampled with the selector-chain
    /// nucleus verify) on a Raon load.
    fn dflash2_inputs(arch_id: u32, temp: f32) -> GenerationRouteInputs {
        GenerationRouteInputs {
            has_speculator: true,
            speculator_is_dflash2: true,
            supports_temp_swor: true,
            supports_chain_nucleus_verify: true,
            temp,
            ..inputs(arch_id)
        }
    }

    #[test]
    fn raon_dflash2_takes_the_spec_route_when_the_request_is_eligible() {
        for temp in [0.0f32, 1.0] {
            let request = dflash2_inputs(19, temp);
            assert_eq!(
                select_generation_route(&request),
                GenerationRoute::RaonSpec,
                "{request:?}"
            );
            // The DFlash2 verifiers apply no token penalty; adaptive KV and the
            // force-AR switches keep the AR producer; so does a load without
            // the drafter.
            for refused in [
                GenerationRouteInputs {
                    nonneutral_penalties: true,
                    ..request
                },
                GenerationRouteInputs {
                    kv_adaptive: true,
                    ..request
                },
                GenerationRouteInputs {
                    force_ar_chat: true,
                    ..request
                },
                GenerationRouteInputs {
                    temp_spec_env_off: true,
                    ..request
                },
                GenerationRouteInputs {
                    has_speculator: false,
                    speculator_is_dflash2: false,
                    ..request
                },
                GenerationRouteInputs {
                    speculator_is_dflash2: false,
                    speculator_is_mtp: true,
                    ..request
                },
            ] {
                assert_eq!(
                    select_generation_route(&refused),
                    GenerationRoute::RaonAr,
                    "{refused:?}"
                );
            }
        }
        // A sampled request needs a drafter that verifies at temperature with
        // the selector-chain nucleus law.
        for incapable in [
            GenerationRouteInputs {
                supports_temp_swor: false,
                ..dflash2_inputs(19, 1.0)
            },
            GenerationRouteInputs {
                supports_chain_nucleus_verify: false,
                ..dflash2_inputs(19, 1.0)
            },
        ] {
            assert_eq!(
                select_generation_route(&incapable),
                GenerationRoute::RaonAr,
                "{incapable:?}"
            );
        }
        // Greedy needs neither capability.
        let greedy = GenerationRouteInputs {
            supports_temp_swor: false,
            supports_chain_nucleus_verify: false,
            ..dflash2_inputs(19, 0.0)
        };
        assert_eq!(select_generation_route(&greedy), GenerationRoute::RaonSpec);
    }

    #[test]
    fn raon_spec_route_inventory_matches_the_ar_route_contract() {
        let spec = GenerationRoute::RaonSpec;
        assert!(GenerationRoute::ALL.contains(&spec));
        assert_eq!(spec.name(), "raon_spec");
        assert!(spec.supports_tools());
        assert!(spec.supports_stop());
        assert!(!spec.runs_qwen35_ar_body());
        let adapter = generation_route_adapter(spec).expect("adapter");
        assert_eq!(adapter.route, spec);
        // Same gen_start contract as the AR route.
        let mut start = Vec::new();
        (adapter.start)(&mut start, "raon-spec", true);
        let mut ar_start = Vec::new();
        (generation_route_adapter(GenerationRoute::RaonAr)
            .expect("adapter")
            .start)(&mut ar_start, "raon-spec", true);
        assert_eq!(start, ar_start);
    }

    #[test]
    fn raon_defaults_that_penalize_keep_sampled_requests_on_ar() {
        let d = hipfire_arch_raon::GenerationDefaults::default();
        assert!(d.presence_penalty != 0.0, "the card default is penalized");
        let base = RaonSampling {
            temp: 1.0,
            top_p: 0.95,
            top_k: None,
            min_p: None,
            repeat_penalty: 1.0,
            repeat_window: 128,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user_explicit: false,
        };
        // Unset sampled request: the card's presence default applies.
        assert!(base.penalized_after_defaults(&d));
        // Greedy and explicit-neutral requests are unpenalized.
        assert!(!RaonSampling { temp: 0.0, ..base }.penalized_after_defaults(&d));
        assert!(!RaonSampling {
            user_explicit: true,
            ..base
        }
        .penalized_after_defaults(&d));
        // Raw non-neutral controls always count.
        for penalized in [
            RaonSampling {
                temp: 0.0,
                repeat_penalty: 1.1,
                ..base
            },
            RaonSampling {
                temp: 0.0,
                presence_penalty: 0.5,
                ..base
            },
            RaonSampling {
                temp: 0.0,
                frequency_penalty: 0.5,
                ..base
            },
        ] {
            assert!(penalized.penalized_after_defaults(&d), "{penalized:?}");
        }
    }

    #[test]
    fn raon_spec_start_channel_agrees_with_the_ar_classifier_on_the_template() {
        assert!(spec_start_channel_matches_ar(&render(true, None).text));
        assert!(spec_start_channel_matches_ar(&render(false, None).text));
        // A literal `<think>` in user content stays out of both classifiers.
        let history = vec![message(Role::User, "what is <think> in xml?")];
        for thinking in [true, false] {
            let p = render_raon_prompt(
                &tokenizer(),
                TEMPLATE,
                None,
                "what is <think> in xml?",
                Some(&history),
                None,
                thinking,
            )
            .unwrap();
            assert!(spec_start_channel_matches_ar(&p.text), "{:?}", p.text);
            assert_eq!(p.started_in_think, thinking);
        }
        // A seeded opener the tail classifier would miss is served by AR.
        assert!(!spec_start_channel_matches_ar(
            "<|im_start|>assistant\n<think>Let me"
        ));
    }

    #[test]
    fn raon_spec_render_drops_what_the_ar_render_drops() {
        let blank =
            render_raon_text(&tokenizer(), TEMPLATE, Some("  "), "hi", None, None, true).unwrap();
        let none = render_raon_text(&tokenizer(), TEMPLATE, None, "hi", None, None, true).unwrap();
        assert_eq!(blank, none, "a blank system prompt is not rendered");
        let empty_history = render_raon_text(
            &tokenizer(),
            TEMPLATE,
            None,
            "hi",
            Some(&[]),
            Some(&[]),
            true,
        )
        .unwrap();
        assert_eq!(empty_history, none, "empty history and tools are dropped");
    }

    #[test]
    fn raon_thinking_on_prefix_opens_think_and_off_is_closed_visible() {
        let on = render(true, None);
        assert!(
            on.text.ends_with("<|im_start|>assistant\n<think>\n"),
            "{:?}",
            on.text
        );
        assert!(on.started_in_think);
        assert_eq!(on.text.matches("<think>").count(), 1, "no second opener");
        assert!(!on.tokens.is_empty());

        let off = render(false, None);
        assert!(
            off.text
                .ends_with("<|im_start|>assistant\n<think></think>\n\n"),
            "{:?}",
            off.text
        );
        assert!(!off.started_in_think, "the closed span is visible mode");
        assert_eq!(off.text.matches("<think>").count(), 1, "no second opener");
    }

    #[test]
    fn raon_history_think_markup_does_not_reclassify_live_turn() {
        // A prior assistant turn with a closed think block and user text that
        // contains a literal `<think>` must not flip the live-turn answer.
        let history = vec![
            message(Role::User, "what is <think> in xml?"),
            Message {
                reasoning_content: Some("old reasoning".into()),
                ..message(Role::Assistant, "It is a tag.")
            },
            message(Role::User, "thanks"),
        ];
        let off = render_raon_prompt(
            &tokenizer(),
            TEMPLATE,
            None,
            "thanks",
            Some(&history),
            None,
            false,
        )
        .unwrap();
        assert!(!off.started_in_think);
        let on = render_raon_prompt(
            &tokenizer(),
            TEMPLATE,
            None,
            "thanks",
            Some(&history),
            None,
            true,
        )
        .unwrap();
        assert!(on.started_in_think);
    }

    #[test]
    fn raon_tools_render_xml_block() {
        let tool = serde_json::json!({"type":"function","function":{
            "name":"get_weather","description":"w",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}});
        let p = render(true, Some(std::slice::from_ref(&tool)));
        assert!(p.text.contains("<tools>"), "{:?}", p.text);
        assert!(p.text.contains("<function=example_function_name>"));
        assert!(p.started_in_think);
    }

    #[test]
    fn raon_sampling_defaults_apply_to_unset_sampled_requests_only() {
        let d = hipfire_arch_raon::GenerationDefaults::default();
        let base = RaonSampling {
            temp: 1.0,
            top_p: 0.95,
            top_k: None,
            min_p: None,
            repeat_penalty: 1.0,
            repeat_window: 128,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user_explicit: false,
        };
        // Unset sampled request: card top_k 20 + presence 1.5; window covers
        // the whole generation.
        let r = base.resolve(&d, 4096);
        assert_eq!(r.top_k, Some(20));
        assert_eq!(r.presence_penalty, 1.5);
        assert_eq!(r.repeat_window, 4096);
        // Short generations keep the request's window.
        assert_eq!(base.resolve(&d, 16).repeat_window, 128);
        // Explicit greedy: nothing is added (argmax runs unperturbed).
        let g = RaonSampling { temp: 0.0, ..base };
        let r = g.resolve(&d, 256);
        assert_eq!((r.top_k, r.presence_penalty), (None, 0.0));
        // Explicit controls win, including an explicit presence of 0.
        let e = RaonSampling {
            user_explicit: true,
            top_k: Some(40),
            ..base
        };
        let r = e.resolve(&d, 256);
        assert_eq!((r.top_k, r.presence_penalty), (Some(40), 0.0));
        let e = RaonSampling {
            user_explicit: true,
            presence_penalty: 0.5,
            ..base
        };
        assert_eq!(e.resolve(&d, 256).presence_penalty, 0.5);
    }

    #[test]
    fn raon_presence_penalty_sees_generated_tokens_only() {
        // The body hands `sample_cpu` the generated history. Token 7 occurs in
        // the prompt only, token 3 was generated: only 3 is penalized.
        use hipfire_runtime::sampler::{sample_cpu, SamplerConfig};
        let cfg = SamplerConfig {
            temperature: 0.0,
            presence_penalty: 1.5,
            repeat_window: 64,
            ..SamplerConfig::greedy()
        };
        let base = vec![0.0f32, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.9];
        let prompt = [7u32, 7, 7];
        let generated = [3u32];
        let mut with_prompt = base.clone();
        let t_prompt = sample_cpu(
            &mut with_prompt,
            &[&prompt[..], &generated[..]].concat(),
            &cfg,
        );
        let mut gen_only = base.clone();
        let t_gen = sample_cpu(&mut gen_only, &generated, &cfg);
        assert_eq!(
            t_prompt, 3,
            "prompt token 7 penalized when prompt is in history"
        );
        assert_eq!(gen_only[7], 1.9, "prompt-only token untouched");
        assert_eq!(gen_only[3], 0.5, "generated token penalized by presence");
        assert_eq!(t_gen, 7);
    }

    /// Drive the real shared producer exactly as `generate_ar_with_forward`
    /// does, from a canned token stream.
    struct Run {
        events: Vec<serde_json::Value>,
        finish_reason: &'static str,
        calls: Vec<ToolCall>,
    }

    fn run_stream(
        id: &str,
        attempt: u64,
        started_in_think: bool,
        tools: bool,
        chunks: &[&str],
        hit_length_cap: bool,
    ) -> Run {
        let mut sink: Vec<u8> = Vec::new();
        clear_terminal_control();
        set_active_attempt_id(attempt);
        activate_terminal_control(id, attempt);
        let finish;
        {
            let _scope = GenerationRouteScope::enter(GenerationRoute::RaonAr, id);
            assert_eq!(active_generation_route(), Some(GenerationRoute::RaonAr));
            emit_generation_start(GenerationRoute::RaonAr, &mut sink, id, started_in_think);
            let mut semantic =
                QwenArSemanticProducer::new_with_tool_protocol(id, started_in_think, tools);
            let (mut conv, mut streamed, mut pos) = (Vec::new(), Vec::new(), 0usize);
            for (i, chunk) in chunks.iter().enumerate() {
                let token = 5000 + i as u32;
                let stopped = semantic
                    .commit_raw(
                        &mut sink,
                        token,
                        QwenArRawCommitDisposition::ClassifiedVisible,
                        || {
                            let p = crate::ar::qwen_ar_raw_commit_token(
                                &mut conv,
                                &mut streamed,
                                &mut pos,
                                token,
                                QwenArRawCommitDisposition::ClassifiedVisible,
                            );
                            (p, chunk.as_bytes())
                        },
                        |_, _| {},
                    )
                    .unwrap();
                if stopped {
                    break;
                }
            }
            let (f, _visible) = semantic.finish(&mut sink, hit_length_cap).unwrap();
            let mut done = serde_json::json!({
                "type": "done", "id": id, "tokens": chunks.len(),
                "finish_reason": f.finish_reason, "attempt_id": attempt,
            });
            hipfire_engine::emit::stage_terminal_tool_calls(
                &mut done,
                f.finish_reason,
                &f.wire_tool_calls,
            );
            emit_active_route_done_value(&mut sink, &done);
            // A racing second terminal for the same (id, attempt) is dropped.
            emit_active_route_done_value(&mut sink, &done);
            finish = (f.finish_reason, f.wire_tool_calls);
        }
        clear_terminal_control();
        set_active_attempt_id(0);
        Run {
            events: parse_events(&sink),
            finish_reason: finish.0,
            calls: finish.1,
        }
    }

    const EOS: &str = "<|im_end|>";

    #[test]
    fn raon_stream_begins_with_gen_start_and_has_one_terminal() {
        let _g = crate::ar::generation_test_lock();
        for started_in_think in [true, false] {
            let run = run_stream(
                "raon-start",
                93_001,
                started_in_think,
                false,
                &["hello", " world", EOS],
                false,
            );
            let first = &run.events[0];
            assert_eq!(first["type"], "gen_start");
            assert_eq!(first["id"], "raon-start");
            assert_eq!(first["attempt_id"], 93_001);
            assert_eq!(first["started_in_think"], started_in_think);
            assert_eq!(
                run.events
                    .iter()
                    .filter(|e| e["type"] == "gen_start")
                    .count(),
                1
            );
            // Nothing precedes gen_start; exactly one correlated terminal.
            let terminals: Vec<_> = run.events.iter().filter(|e| e["type"] == "done").collect();
            assert_eq!(terminals.len(), 1, "{:?}", run.events);
            assert_eq!(terminals[0]["attempt_id"], 93_001);
            assert_eq!(run.events.last().unwrap()["type"], "done");
            assert_eq!(run.finish_reason, "stop");
        }
    }

    #[test]
    fn raon_reasoning_routes_until_close_and_open_think_is_never_relabeled() {
        let _g = crate::ar::generation_test_lock();
        // Thinking on: starts in reasoning, closes, then answers.
        let run = run_stream(
            "raon-think",
            93_002,
            true,
            false,
            &["plan it", "\n</think>\n\n", "42", EOS],
            false,
        );
        let reasoning: String = run
            .events
            .iter()
            .filter(|e| e["type"] == "reasoning")
            .map(|e| e["text"].as_str().unwrap())
            .collect();
        let content: String = run
            .events
            .iter()
            .filter(|e| e["type"] == "token")
            .map(|e| e["text"].as_str().unwrap())
            .collect();
        assert_eq!(reasoning.trim(), "plan it");
        assert_eq!(content, "42");
        assert_eq!(run.finish_reason, "stop");

        // max_tokens inside reasoning: reasoning only, finish `length`, no
        // visible content, no tool calls.
        let cut = run_stream(
            "raon-think-cut",
            93_003,
            true,
            true,
            &["still thinking <tool_call> not a call", " and more"],
            true,
        );
        assert!(
            cut.events.iter().all(|e| e["type"] != "token"),
            "{:?}",
            cut.events
        );
        assert!(cut.events.iter().any(|e| e["type"] == "reasoning"));
        assert_eq!(cut.finish_reason, "length");
        assert!(cut.calls.is_empty());

        // EOS inside reasoning: graceful stop, reasoning only.
        let open = run_stream(
            "raon-think-open",
            93_004,
            true,
            false,
            &["only thoughts", EOS],
            false,
        );
        assert!(open.events.iter().all(|e| e["type"] != "token"));
        assert_eq!(open.finish_reason, "stop");
    }

    #[test]
    fn raon_tool_call_extracted_from_canned_xml_stream() {
        let _g = crate::ar::generation_test_lock();
        let run = run_stream(
            "raon-tools",
            93_005,
            false,
            true,
            &[
                "Checking.\n\n<tool_call>\n<function=get_weather>\n",
                "<parameter=city>\n서울\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n",
                "</function>\n</tool_call>",
                EOS,
            ],
            false,
        );
        assert_eq!(run.finish_reason, "tool_calls");
        assert_eq!(run.calls.len(), 1);
        assert_eq!(run.calls[0].name, "get_weather");
        assert_eq!(run.calls[0].arguments["city"], "서울");
        assert_eq!(run.calls[0].arguments["unit"], "celsius");
        // The call markup never reaches the visible stream.
        let content: String = run
            .events
            .iter()
            .filter(|e| e["type"] == "token")
            .map(|e| e["text"].as_str().unwrap())
            .collect();
        assert!(!content.contains("<tool_call>"), "{content:?}");
        assert!(content.contains("Checking."));
        let done = run.events.last().unwrap();
        assert_eq!(done["finish_reason"], "tool_calls");
        assert_eq!(done["calls"][0]["name"], "get_weather");
        assert_eq!(run.events.iter().filter(|e| e["type"] == "done").count(), 1);
    }

    #[test]
    fn raon_truncated_tool_call_is_not_released() {
        let _g = crate::ar::generation_test_lock();
        let run = run_stream(
            "raon-tools-cut",
            93_006,
            false,
            true,
            &["<tool_call>\n<function=get_weather>\n<parameter=city>\nSe"],
            true,
        );
        assert_eq!(run.finish_reason, "length");
        assert!(run.calls.is_empty());
    }

    #[test]
    fn raon_pre_start_refusal_has_no_gen_start_and_same_key_is_reusable() {
        let _g = crate::ar::generation_test_lock();
        let id = "raon-refusal";
        let mut sink = Vec::new();
        clear_terminal_control();
        set_active_attempt_id(93_007);
        activate_terminal_control(id, 93_007);
        {
            let _scope = GenerationRouteScope::enter(GenerationRoute::RaonAr, id);
            raon_error(&mut sink, id, "max_tokens must be > 0", "validation");
        }
        let events = parse_events(&sink);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], "error");
        assert_eq!(events[0]["attempt_id"], 93_007);
        assert!(events.iter().all(|e| e["type"] != "gen_start"));
        clear_terminal_control();
        set_active_attempt_id(0);

        // The next request (new attempt) starts cleanly with gen_start first.
        let run = run_stream("raon-refusal", 93_008, true, false, &["x", EOS], false);
        assert_eq!(run.events[0]["type"], "gen_start");
        assert_eq!(run.events[0]["attempt_id"], 93_008);
    }

    /// The producer above emits typed semantic events, so the carrier must
    /// advertise the matching contract and no Qwen DFlash/batch capability.
    #[test]
    fn raon_carrier_caps_match_the_route_producer() {
        assert_eq!(
            crate::common::gen_start_contract_version_for_arch(19),
            Some(2)
        );
        let caps = hipfire_loader::carrier_for(19)
            .expect("raon carrier")
            .caps();
        assert!(
            !caps.is_qwen_dflash(),
            "arch 19 must never look like Qwen DFlash"
        );
        assert!(!caps.supports_continuous_batch);
    }
}
