// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vision and OCR generation — Qwen3.5-VL and dots.ocr.
//!
//! Per-architecture generation bodies lifted verbatim from `crates/hipfire-daemon/src/main.rs`
//! (wave 5 / D3). See `lib.rs` for layering rationale.

use base64::Engine;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative;
use hipfire_arch_qwen35_vl::image;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_engine::emit::{emit_reasoning_token, emit_visible_token};
use hipfire_engine::scheduler::block_attractor_unclosed_cpu;
use hipfire_engine::terminal::{
    active_attempt_id, await_client_terminal_commit, check_abort,
    emit_aborted_terminal_after_abort, ClientTerminalDecision,
};
use hipfire_engine::turn_cache::{apply_turn_cache_action, TurnCacheAction};

fn emit_active_attempt_error(
    stdout: &mut impl std::io::Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    crate::ar::emit_active_route_error(stdout, id, message, class, retryable, rolled_back);
}

fn write_error(stdout: &mut impl std::io::Write, id: &str, message: &str) {
    crate::ar::emit_active_route_error(stdout, Some(id), message, "internal", false, false);
}

use hipfire_loader::{AsstTurnCache, LoadedModel};
use hipfire_runtime::emit_text::{ThinkOutputRouter, ThinkRouteEvent};
use hipfire_runtime::eos_filter::{EosFilter, FilterAction};
use hipfire_runtime::sampler::{self, SamplerConfig};
use hipfire_runtime::spec::{PrefillOutcome, SpecTarget};
use std::any::Any;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

// ── Local verbatim copies of daemon-shared helpers ───────────────────────────
// `free_checkpoints` and `emit_committed_event` are shared across all generate
// paths in the daemon (25 and 18 call sites). They cannot live in
// `hipfire-engine` (the former touches `hipfire_arch_qwen35::speculative::DeltaNetSnapshot`,
// an arch type) and the generate crate cannot depend back on the daemon.
// This local copy keeps the moved VL bodies byte-identical without introducing a
// circular dependency. The daemon retains its own identical copy.

fn free_checkpoints(
    cks: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    gpu: &mut rdna_compute::Gpu,
) {
    for (_, snap) in cks.drain(..) {
        snap.free_gpu(gpu);
    }
}

fn emit_committed_event(
    stdout: &mut (impl std::io::Write + ?Sized),
    id: &str,
    tok_id: u32,
    pos: usize,
    t_ms: u64,
) {
    if !hipfire_config::developer_bool("HIPFIRE_EMIT_TOKEN_IDS", false) {
        return;
    }
    // Build through `serde_json::json!` for the same reason
    // `emit_error_with_id` does: `id` is user-supplied and a single `"`
    // or `\` in it would corrupt the line, breaking the client's JSONL
    // parser for every subsequent event on the same connection.
    let envelope = serde_json::json!({
        "type": "committed",
        "id": id,
        "tok_id": tok_id,
        "pos": pos,
        "t_ms": t_ms,
        "attempt_id": active_attempt_id(),
    });
    let _ = writeln!(stdout, "{}", envelope);
}

/// Feed one decode-step's new bytes through the v2 typed emission contract:
/// EosFilter (UTF-8 boundary + EOT marker suppression) → ThinkOutputRouter
/// (think-channel split) → `token` / `reasoning` envelopes.
///
/// Arch 5/6 advertises `gen_start.contract_version = 2`, under which the CLI
/// appends token text to `content` VERBATIM — no marker scan (complete.rs
/// SemanticEventFold). The producer therefore owns think-splitting and marker
/// suppression; before this helper the VL loop emitted every decoded byte as
/// a bare token envelope, so image turns shipped the whole think block plus
/// literal `</think>` / `<|im_end|>` markers inside `content` with zero
/// `reasoning_content` (2026-08-27 ledger, post-thinking garble).
///
/// Mirrors `ar::qwen_ar_observe_and_route` minus the ToolOutputRouter: VL
/// image turns carry no tool contract, so reserved-looking text stays visible
/// instead of being buffered or failing closed.
pub(crate) fn vl_route_decode_text(
    stdout: &mut impl std::io::Write,
    id: &str,
    filter: &mut EosFilter,
    think: &mut ThinkOutputRouter,
    new_bytes: &[u8],
) -> bool {
    vl_route_decode_text_into(stdout, id, filter, think, new_bytes, None)
}

/// Emit routed events for `text`; the visible (answer-channel) text is also
/// appended to `visible` when the caller keeps it (the assistant-turn cache
/// key is the visible text the client will echo back).
fn vl_emit_routed(
    stdout: &mut impl std::io::Write,
    id: &str,
    think: &mut ThinkOutputRouter,
    text: &str,
    mut visible: Option<&mut String>,
) {
    let mut routed = Vec::new();
    think.push_into(text, &mut routed);
    for ev in routed {
        match ev {
            ThinkRouteEvent::Reasoning(t) => emit_reasoning_token(stdout, id, &t),
            ThinkRouteEvent::Content(t) => {
                if let Some(acc) = visible.as_deref_mut() {
                    acc.push_str(&t);
                }
                emit_visible_token(stdout, id, &t)
            }
        }
    }
}

/// [`vl_route_decode_text`] that also collects the visible text.
pub(crate) fn vl_route_decode_text_into(
    stdout: &mut impl std::io::Write,
    id: &str,
    filter: &mut EosFilter,
    think: &mut ThinkOutputRouter,
    new_bytes: &[u8],
    visible: Option<&mut String>,
) -> bool {
    let (bytes, is_stop) = match filter.observe(new_bytes) {
        FilterAction::Emit(b) => (b, false),
        FilterAction::EmitAndStop(b) => (b, true),
        FilterAction::Hold => return false,
        FilterAction::Stop => return true,
    };
    if let Ok(text) = std::str::from_utf8(&bytes) {
        if !text.is_empty() {
            vl_emit_routed(stdout, id, think, text, visible);
        }
    }
    is_stop
}

pub(crate) fn vl_finish_think_routing(
    stdout: &mut impl std::io::Write,
    id: &str,
    filter: &mut EosFilter,
    think: &mut ThinkOutputRouter,
) {
    vl_finish_think_routing_into(stdout, id, filter, think, None)
}

/// [`vl_finish_think_routing`] that also collects the visible text.
pub(crate) fn vl_finish_think_routing_into(
    stdout: &mut impl std::io::Write,
    id: &str,
    filter: &mut EosFilter,
    think: &mut ThinkOutputRouter,
    mut visible: Option<&mut String>,
) {
    let pending = filter.flush_pending();
    if !pending.is_empty() {
        if let Ok(text) = std::str::from_utf8(&pending) {
            if !text.is_empty() {
                vl_emit_routed(stdout, id, think, text, visible.as_deref_mut());
            }
        }
    }
    let mut routed = Vec::new();
    think.finish_into(&mut routed);
    for ev in routed {
        match ev {
            ThinkRouteEvent::Reasoning(t) => emit_reasoning_token(stdout, id, &t),
            ThinkRouteEvent::Content(t) => {
                if let Some(acc) = visible.as_deref_mut() {
                    acc.push_str(&t);
                }
                emit_visible_token(stdout, id, &t)
            }
        }
    }
}

pub enum ImageSource<'a> {
    Path(&'a str),
    Base64(&'a str),
}

pub struct GenerateVLParams<'a> {
    pub id: &'a str,
    pub prompt: &'a str,
    pub system_prompt: Option<&'a str>,
    /// The request's images in conversation order: one per image slot of
    /// `messages_history`, or the lone image of a prompt-only request. dots-OCR
    /// and LFM2-VL render a single image and refuse more.
    pub image_sources: Vec<ImageSource<'a>>,
    /// The whole conversation, live user turn last, with an
    /// [`IMAGE_SLOT`](hipfire_runtime::prompt_frame::IMAGE_SLOT) at each
    /// image's position. `None` for a prompt-only request.
    pub messages_history: Option<&'a [hipfire_runtime::prompt_frame::Message]>,
    /// Chat-template inputs, as on the text path.
    pub enable_thinking: bool,
    pub reasoning_effort: Option<&'a str>,
    pub temp: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub max_think_tokens: usize,
    pub assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    /// Per-request sampler seed (see `hipfire_engine::request_seed_for`).
    pub seed: u32,
}

impl<'a> GenerateVLParams<'a> {
    /// The one image of a single-image route, or the refusal text.
    fn single_image(&self) -> Result<&ImageSource<'a>, String> {
        match self.image_sources.as_slice() {
            [only] => Ok(only),
            [] => Err("vision request carries no image".to_owned()),
            many => Err(format!(
                "this vision model renders one image per request, got {}",
                many.len()
            )),
        }
    }
}

pub fn vl_no_eviction_kv_cap(physical_cap: usize, max_seq: usize, adaptive_engaged: bool) -> usize {
    if adaptive_engaged {
        max_seq
    } else {
        physical_cap
    }
}

/// Fail-closed rollback for uncommitted Qwen35-VL state — the VL analogue of
/// the text-AR `reset_ar_uncommitted_state!` macro (same scope: host cursors,
/// recurrent/conv state, KV offset, adaptive, prefill checkpoint ring), plus
/// the shared graph/replay-invalidate + device-sync tail so the returned
/// [`crate::common::RollbackEpilogue`] attests the device is drained.
///
/// Recurrent state resets via the canonical [`qwen35::DeltaNetState::reset`]
/// (attested — a HIP failure reports `rolled_back=false` instead of being
/// swallowed — covering every buffer class including future additions).
/// Takes the live request pieces as disjoint borrows because the bundle
/// itself stays borrowed by the caller (`&mut LoadedModel` is unreachable).
/// Cancellation (which additionally clears the DFlash ring, speculator, and
/// assistant-turn cache) uses [`vl_cancel_after_rollback`] instead.
pub(crate) fn vl_rollback_uncommitted(
    gpu: &mut rdna_compute::Gpu,
    dn: &mut qwen35::DeltaNetState,
    kv: &mut hipfire_runtime::llama::KvCache,
    kv_adaptive: &mut Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
) -> crate::common::RollbackEpilogue {
    use crate::common::{
        fail_closed_device_sync, fail_closed_epilogue_after_sync,
        fail_closed_invalidate_graphs_and_replay, push_reset_err,
    };
    *seq_pos = 0;
    conversation_tokens.clear();
    let mut first_err: Option<String> = None;
    if let Err(e) = dn.reset(gpu) {
        push_reset_err(&mut first_err, "dn.reset", e);
    }
    kv.compact_offset = 0;
    if let Some(ad) = kv_adaptive.as_mut() {
        if !ad.is_poisoned() {
            ad.reset_with_cache(gpu, kv);
        }
    }
    free_checkpoints(prefill_checkpoints, gpu);
    fail_closed_invalidate_graphs_and_replay(gpu);
    let prior = match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    };
    let epilogue = fail_closed_epilogue_after_sync(prior, fail_closed_device_sync(gpu));
    if epilogue.rolled_back {
        gpu.replay.begin_replay_observation_window();
    }
    epilogue
}

pub(crate) fn vl_adaptive_downshift_fail_closed(
    kv_adaptive: &mut Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    seq_pos: &mut usize,
    gpu: &mut rdna_compute::Gpu,
    kv: &mut hipfire_runtime::llama::KvCache,
    dn: &mut qwen35::DeltaNetState,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    stdout: &mut std::io::Stdout,
    id: &str,
    phase: &str,
) -> bool {
    let Some(ad) = kv_adaptive.as_mut() else {
        return false;
    };
    let committed = *seq_pos;
    match ad.maybe_downshift(gpu, kv, committed) {
        Ok(applied) => {
            for step in &applied {
                eprintln!(
                    "[adaptive-kv] downshift @ pos {} ({}): {:?} (K={:?} V={:?})",
                    committed, phase, step, ad.cur_k, ad.cur_v
                );
            }
            false
        }
        Err(e) => {
            eprintln!(
                "[adaptive-kv] maybe_downshift error @ pos {} ({}): {:?} — poisoning model",
                committed, phase, e
            );
            // maybe_downshift already poisons on partial failure; rollback
            // leaves poison sticky (reset_with_cache skipped when poisoned).
            let ep = vl_rollback_uncommitted(
                gpu,
                dn,
                kv,
                kv_adaptive,
                seq_pos,
                conversation_tokens,
                prefill_checkpoints,
            );
            crate::common::emit_fail_closed_error(
                stdout,
                Some(id),
                &format!("adaptive KV transition failed during {phase}: {e}"),
                "internal",
                false,
                &ep,
            );
            true
        }
    }
}

pub(crate) fn vl_forward_fail(
    stdout: &mut std::io::Stdout,
    id: &str,
    phase: &str,
    err: impl std::fmt::Display,
    gpu: &mut rdna_compute::Gpu,
    dn: &mut qwen35::DeltaNetState,
    kv: &mut hipfire_runtime::llama::KvCache,
    kv_adaptive: &mut Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
) {
    let ep = vl_rollback_uncommitted(
        gpu,
        dn,
        kv,
        kv_adaptive,
        seq_pos,
        conversation_tokens,
        prefill_checkpoints,
    );
    crate::common::emit_fail_closed_error(
        stdout,
        Some(id),
        &format!("VL {phase}: {err}"),
        "internal",
        false,
        &ep,
    );
}

/// Cancel epilogue for Qwen35-VL: canonical rollback first, then the
/// route-aware cancelled pair — or a fail-closed error when rollback could
/// not be attested (never a silent cancel over dirty state, never a `done`).
pub(crate) fn vl_cancel_after_rollback(
    stdout: &mut std::io::Stdout,
    id: &str,
    generated: usize,
    gpu: &mut rdna_compute::Gpu,
    dn: &mut qwen35::DeltaNetState,
    kv: &mut hipfire_runtime::llama::KvCache,
    kv_adaptive: &mut Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    dflash_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    asst_turn_cache: &mut hipfire_loader::AsstTurnCache,
    speculator: &mut Option<Box<dyn hipfire_runtime::spec::Speculator>>,
) {
    let ep = {
        use crate::common::{
            fail_closed_device_sync, fail_closed_epilogue_after_sync,
            fail_closed_invalidate_graphs_and_replay, push_reset_err,
        };
        // Full production scope (mirrors
        // `production_fail_closed_rollback` minus the `&mut LoadedModel`
        // pieces the caller keeps borrowed): host cursors, assistant-turn
        // cache, recurrent state, both checkpoint rings, speculator,
        // graph/replay invalidation, device sync.
        *seq_pos = 0;
        conversation_tokens.clear();
        asst_turn_cache.clear();
        let mut first_err: Option<String> = None;
        if let Err(e) = dn.reset(gpu) {
            push_reset_err(&mut first_err, "dn.reset", e);
        }
        kv.compact_offset = 0;
        if let Some(ad) = kv_adaptive.as_mut() {
            if !ad.is_poisoned() {
                ad.reset_with_cache(gpu, kv);
            }
        }
        free_checkpoints(prefill_checkpoints, gpu);
        free_checkpoints(dflash_checkpoints, gpu);
        if let Some(s) = speculator.as_mut() {
            if let Err(e) = s.reset(gpu) {
                push_reset_err(&mut first_err, "spec.reset", e);
            }
        }
        fail_closed_invalidate_graphs_and_replay(gpu);
        let prior = match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        };
        let ep = fail_closed_epilogue_after_sync(prior, fail_closed_device_sync(gpu));
        if ep.rolled_back {
            gpu.replay.begin_replay_observation_window();
        }
        ep
    };
    crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
}

/// Fail-closed rollback for the dots.ocr Qwen2 decoder state.
///
/// The vision tower is stateless one-shot (encode → splice → free), so the
/// only request-mutated state is the text decoder: [`qwen2::Qwen2State`],
/// whose `reset()` is an infallible O(1) cursor rewind (slots are
/// overwritten in place). Dots turns never touch `LoadedModel` host cursors,
/// checkpoint rings, the assistant-turn cache, or the speculator slot on the
/// AR path — the shared graph/replay-invalidate + device-sync tail attests
/// the device is drained. Same terminal discipline as the VL funnels above:
/// error → correlated fail-closed error, cancel → cancelled pair (or a
/// fail-closed error when unattested), never a normal `done`.
pub(crate) fn dots_ar_rollback(
    state: &mut qwen2::Qwen2State,
    gpu: &mut rdna_compute::Gpu,
) -> crate::common::RollbackEpilogue {
    use crate::common::{
        fail_closed_device_sync, fail_closed_epilogue_after_sync,
        fail_closed_invalidate_graphs_and_replay,
    };
    state.reset();
    fail_closed_invalidate_graphs_and_replay(gpu);
    let epilogue = fail_closed_epilogue_after_sync(Ok(()), fail_closed_device_sync(gpu));
    if epilogue.rolled_back {
        gpu.replay.begin_replay_observation_window();
    }
    epilogue
}

pub(crate) fn dots_ar_fail(
    state: &mut qwen2::Qwen2State,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    message: &str,
) {
    let ep = dots_ar_rollback(state, gpu);
    crate::common::emit_fail_closed_error(stdout, Some(id), message, "internal", false, &ep);
}

pub(crate) fn dots_ar_cancel(
    state: &mut qwen2::Qwen2State,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    generated: usize,
) {
    let ep = dots_ar_rollback(state, gpu);
    crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
}

/// Fail-closed rollback for the dots.ocr n-gram spec loop, which holds the
/// bundle and speculator outside `LoadedModel`. Same order as
/// [`crate::common::production_fail_closed_rollback_live`]: recurrent rewind
/// via the [`SpecTarget`] hook, drafter reset, graph/replay invalidation,
/// device sync, replay-observation reopen.
pub(crate) fn dots_spec_rollback(
    bundle: &mut hipfire_arch_dots_ocr::DotsOcrBundle,
    spec: &mut dyn hipfire_runtime::spec::Speculator,
    gpu: &mut rdna_compute::Gpu,
) -> crate::common::RollbackEpilogue {
    use crate::common::{
        fail_closed_device_sync, fail_closed_epilogue_after_sync,
        fail_closed_invalidate_graphs_and_replay, push_reset_err,
    };
    let mut first_err: Option<String> = None;
    if let Err(e) = bundle.reset_recurrent(gpu) {
        push_reset_err(&mut first_err, "reset_recurrent", e);
    }
    if let Err(e) = spec.reset(gpu) {
        push_reset_err(&mut first_err, "spec.reset", e);
    }
    fail_closed_invalidate_graphs_and_replay(gpu);
    let prior = match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    };
    let epilogue = fail_closed_epilogue_after_sync(prior, fail_closed_device_sync(gpu));
    if epilogue.rolled_back {
        gpu.replay.begin_replay_observation_window();
    }
    epilogue
}

pub(crate) fn dots_spec_fail(
    bundle: &mut hipfire_arch_dots_ocr::DotsOcrBundle,
    spec: &mut dyn hipfire_runtime::spec::Speculator,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    message: &str,
) {
    let ep = dots_spec_rollback(bundle, spec, gpu);
    crate::common::emit_fail_closed_error(stdout, Some(id), message, "internal", false, &ep);
}

pub(crate) fn dots_spec_cancel(
    bundle: &mut hipfire_arch_dots_ocr::DotsOcrBundle,
    spec: &mut dyn hipfire_runtime::spec::Speculator,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    generated: usize,
) {
    let ep = dots_spec_rollback(bundle, spec, gpu);
    crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
}

/// The text one image occupies in the rendered chat prompt: the vision frame
/// around a SINGLE `<|image_pad|>`. [`expand_image_pads`] widens the pad to
/// the image's visual-token count once its grid is known, so the chat template
/// (and the assistant-turn cache) never sees a thousand pad tokens.
pub(crate) const VL_IMAGE_MARKER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// Geometry of one image after smart-resize: the pre-merge patch grid and the
/// number of visual tokens it expands to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VlGrid {
    pub grid_h: usize,
    pub grid_w: usize,
    pub n_visual: usize,
}

/// Widen each single `<|image_pad|>` of `tokens` to that image's visual-token
/// count (`visual_counts`, in image order). The prompt must carry exactly one
/// pad per image.
pub(crate) fn expand_image_pads(
    tokens: &[u32],
    image_pad_id: u32,
    visual_counts: &[usize],
) -> Result<Vec<u32>, String> {
    let pads = tokens.iter().filter(|&&t| t == image_pad_id).count();
    if pads != visual_counts.len() {
        return Err(format!(
            "the rendered prompt has {pads} image position(s) for {} image(s)",
            visual_counts.len()
        ));
    }
    if visual_counts.contains(&0) {
        return Err("an image produced no visual tokens".to_owned());
    }
    let extra: usize = visual_counts.iter().map(|n| n - 1).sum();
    let mut out = Vec::with_capacity(tokens.len() + extra);
    let mut next = 0usize;
    for &token in tokens {
        if token == image_pad_id {
            out.extend(std::iter::repeat(image_pad_id).take(visual_counts[next]));
            next += 1;
        } else {
            out.push(token);
        }
    }
    Ok(out)
}

/// `(start, len)` of every maximal run of `<|image_pad|>` in `tokens`.
pub(crate) fn image_pad_runs(tokens: &[u32], image_pad_id: u32) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] != image_pad_id {
            i += 1;
            continue;
        }
        let start = i;
        while i < tokens.len() && tokens[i] == image_pad_id {
            i += 1;
        }
        runs.push((start, i - start));
    }
    runs
}

/// How many leading rows of the resident conversation this prompt can keep.
///
/// Rows are positional (KV) or cumulative (DeltaNet), so only a PURE forward
/// extension is reusable: the resident tokens must be a strict prefix of the
/// new prompt and the live cursor must sit exactly at their end. The
/// exact-match edge (nothing new to prefill) would re-apply the last token's
/// non-commutative recurrent update, so it starts cold, as does a boundary
/// that would cut through an image's pad run. Everything else — a diverging
/// history, an edited message, eviction — is `0` (cold start).
///
/// Every image's rows carry the same `<|image_pad|>` id, so equal tokens do not
/// make equal images: the resident images' content fingerprints
/// (`resident_images`, in order) must equal the prompt's first images
/// (`images`), else a replaced same-size image would silently reuse the old
/// image's rows. A DeltaNet state cannot rewind to an earlier image's start,
/// so a changed image means a cold start, not a shorter reuse.
pub(crate) fn vl_plan_reuse(
    prior: &[u32],
    prompt: &[u32],
    seq_pos: usize,
    reuse_allowed: bool,
    image_pad_id: u32,
    resident_images: &[[u8; 32]],
    images: &[[u8; 32]],
) -> usize {
    if !reuse_allowed || prior.is_empty() || seq_pos != prior.len() {
        return 0;
    }
    let lcp = prior.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    if lcp != prior.len() || lcp >= prompt.len() {
        return 0;
    }
    if prompt[lcp - 1] == image_pad_id && prompt[lcp] == image_pad_id {
        return 0;
    }
    let resident_runs = image_pad_runs(prior, image_pad_id).len();
    if resident_runs > 0
        && (resident_images.len() != resident_runs
            || images.len() < resident_runs
            || resident_images != &images[..resident_runs])
    {
        return 0;
    }
    lcp
}

/// Per-row `(t, h, w)` rope phases of the WHOLE conversation: a pure function
/// of the token sequence (each row's phase depends only on the rows before
/// it), so a turn that extends a resident conversation recomputes the same
/// phases for the rows it keeps and continues from them.
///
/// `tokens` carry every image expanded to its visual-token run; `grids` are
/// the images in order.
pub(crate) fn vl_mrope_positions(
    tokens: &[u32],
    image_pad_id: u32,
    grids: &[VlGrid],
    spatial_merge_size: usize,
) -> Result<hipfire_arch_qwen35_vl::mrope::MropePositions, String> {
    use hipfire_arch_qwen35_vl::mrope::{build_mrope_positions, ImageSpan};
    if spatial_merge_size == 0 {
        return Err("spatial_merge_size is zero".to_owned());
    }
    let runs = image_pad_runs(tokens, image_pad_id);
    if runs.len() != grids.len() {
        return Err(format!(
            "{} image-pad run(s) in the prompt for {} image(s)",
            runs.len(),
            grids.len()
        ));
    }
    let mut spans = Vec::with_capacity(runs.len());
    for (&(start, len), grid) in runs.iter().zip(grids) {
        // The merged grid must account for exactly the spliced visual tokens,
        // else `build_mrope_positions` pushes a different count than the
        // prompt has and every downstream index is off.
        let merged = (grid.grid_h / spatial_merge_size) * (grid.grid_w / spatial_merge_size);
        if len != grid.n_visual || merged != grid.n_visual {
            return Err(format!(
                "image run of {len} rows does not match its {}x{} grid ({merged} merged, {} expected)",
                grid.grid_h, grid.grid_w, grid.n_visual
            ));
        }
        spans.push(ImageSpan {
            start,
            len,
            grid_h: grid.grid_h,
            grid_w: grid.grid_w,
        });
    }
    let built = build_mrope_positions(tokens.len(), &spans, spatial_merge_size);
    if built.positions.len() != tokens.len() {
        return Err(format!(
            "build_mrope_positions returned {} positions for {} tokens",
            built.positions.len(),
            tokens.len()
        ));
    }
    Ok(built)
}

/// Rope context for a request that prefills `tokens[base..]` over a resident
/// `[0, base)` — `base` is the reused prefix, `0` on a cold start. Phases come
/// from the whole conversation ([`vl_mrope_positions`]), so a later turn
/// resumes at `previous_max + 1` rather than at the token index.
pub(crate) fn build_vl_mrope_ctx(
    tokens: &[u32],
    image_pad_id: u32,
    grids: &[VlGrid],
    spatial_merge_size: usize,
    base: usize,
    config: &qwen35::Qwen35Config,
) -> Result<qwen35::MropeCtx, String> {
    if base > tokens.len() {
        return Err(format!(
            "mrope base {base} is past the prompt ({} rows)",
            tokens.len()
        ));
    }
    let built = vl_mrope_positions(tokens, image_pad_id, grids, spatial_merge_size)?;
    eprintln!(
        "[daemon/vl] mrope: images={} rows={} base={base} rope_delta={} section={:?}",
        grids.len(),
        tokens.len(),
        built.rope_delta,
        config.mrope_section
    );
    Ok(qwen35::MropeCtx::new(
        config,
        base,
        built.positions[base..].to_vec(),
        built.rope_delta,
    ))
}

/// The rendered prompt of a vision turn, one `<|image_pad|>` per image.
pub(crate) struct VlPrompt {
    pub tokens: Vec<u32>,
    /// The generation turn opens inside `<think>` (read off the render tail).
    pub started_in_think: bool,
    /// The tokens come from the chat template over the whole conversation, so
    /// the resident conversation can be a prefix of them.
    pub whole_conversation: bool,
}

pub(crate) struct VlRenderInputs<'a> {
    pub tokenizer: &'a hipfire_runtime::tokenizer::Tokenizer,
    pub chat_template: Option<&'a str>,
    pub jinja_enabled: bool,
    pub history: Option<&'a [hipfire_runtime::prompt_frame::Message]>,
    pub n_images: usize,
    pub system: Option<&'a str>,
    pub prompt: &'a str,
    pub enable_thinking: bool,
    pub reasoning_effort: Option<&'a str>,
    pub assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    /// `<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>`.
    pub image_ids: [u32; 3],
    /// The resident conversation: assistant turns replay their emitted tokens
    /// (instead of a template re-render) only when there is one to continue.
    pub resident_tokens: &'a [u32],
    pub turn_cache_enabled: bool,
}

/// Render the conversation a vision request carries.
///
/// With a chat template the whole history goes through the same Jinja frame the
/// text path uses, each image slot swapped for [`VL_IMAGE_MARKER`] at its
/// message position, and earlier assistant turns replay the tokens the model
/// emitted. Without one (or with the template disabled) only a single user turn
/// can be framed — the hand-built ChatML turn a prompt-only request always used
/// — and anything longer is refused rather than silently truncated.
pub(crate) fn vl_render_prompt(
    inputs: &VlRenderInputs<'_>,
    asst_turn_cache: &mut AsstTurnCache,
) -> Result<VlPrompt, String> {
    use hipfire_runtime::prompt_frame::{
        count_image_slots, expand_image_slots, ChatFrame, JinjaChatFrame, Role, ToolTurnSplice,
    };
    let tokenizer = inputs.tokenizer;
    let [vision_start_id, image_pad_id, vision_end_id] = inputs.image_ids;
    let legacy = |system: Option<&str>, text: &str| -> Result<VlPrompt, String> {
        if inputs.n_images != 1 {
            return Err(format!(
                "this model has no usable chat template, so a vision request renders one image; got {}",
                inputs.n_images
            ));
        }
        let nl = tokenizer.encode("\n");
        let q_tokens = tokenizer.encode(text);
        let mut user_body: Vec<u32> = Vec::with_capacity(q_tokens.len() + nl.len() + 3);
        user_body.push(vision_start_id);
        user_body.push(image_pad_id);
        user_body.push(vision_end_id);
        user_body.extend_from_slice(&nl);
        user_body.extend_from_slice(&q_tokens);
        let tokens = ChatFrame {
            tokenizer,
            system,
            user: "", // unused: the tokens are passed directly
            assistant_prefix: inputs.assistant_prefix,
            raw: false,
        }
        .build_with_user_tokens(&user_body);
        // started_in_think mirrors the ChatFrame builder's own conditions: the
        // `<think>` opener lands in the prompt only for OpenThink AND a
        // tokenizer that carries the special token.
        let started_in_think = matches!(
            inputs.assistant_prefix,
            hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
        ) && tokenizer.special_token_id("<think>").is_some();
        Ok(VlPrompt {
            tokens,
            started_in_think,
            whole_conversation: false,
        })
    };
    let Some(history) = inputs.history else {
        return legacy(inputs.system, inputs.prompt);
    };
    let slots = count_image_slots(history);
    if slots != inputs.n_images {
        return Err(format!(
            "messages carry {slots} image position(s) but the request has {} image(s)",
            inputs.n_images
        ));
    }
    let Some(template) = inputs.chat_template.filter(|_| inputs.jinja_enabled) else {
        let turns = history.iter().filter(|m| m.role != Role::System).count();
        let last_user = history.iter().rev().find(|m| m.role == Role::User);
        let (true, Some(last_user)) = (turns == 1, last_user) else {
            return Err(
                "multi-turn image conversations need the model's chat template (it is missing or \
                 HIPFIRE_JINJA_CHAT=0)"
                    .to_owned(),
            );
        };
        let system = history
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.as_str())
            .or(inputs.system);
        let text = last_user
            .content
            .replace(hipfire_runtime::prompt_frame::IMAGE_SLOT, "");
        return legacy(system, text.trim_start_matches('\n'));
    };
    let marked = expand_image_slots(history, VL_IMAGE_MARKER);
    let frame = JinjaChatFrame {
        tokenizer,
        template,
        system: inputs.system,
        user: inputs.prompt,
        enable_thinking: inputs.enable_thinking,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: inputs.reasoning_effort,
    };
    let rendered = frame
        .render_messages(&marked, None, None)
        .map_err(|e| format!("jinja render: {e}"))?;
    let started_in_think = hipfire_engine::emit::render_tail_opens_think(&rendered);
    let cold = tokenizer.encode(&rendered);
    let tokens = if inputs.turn_cache_enabled && !inputs.resident_tokens.is_empty() {
        match crate::qwen::qwen_jinja_cached_history_tokens(
            &frame,
            asst_turn_cache,
            &cold,
            &marked,
            None,
            ToolTurnSplice::Envelopes,
            "vl",
            None,
        ) {
            Ok(tokens) => tokens,
            Err(e) => {
                eprintln!("[daemon/vl] cached-history render failed ({e}) — cold render");
                cold
            }
        }
    } else {
        cold
    };
    Ok(VlPrompt {
        tokens,
        started_in_think,
        whole_conversation: true,
    })
}

/// One request image, decoded just far enough to know its geometry.
struct VlImage {
    bytes: Vec<u8>,
    img_h: usize,
    img_w: usize,
    grid: VlGrid,
    /// sha256 of the encoded image bytes.
    fingerprint: [u8; 32],
}

/// Read one image source and derive its post-resize geometry from the format
/// header — no pixel decode, so an image that is already in the resident
/// conversation costs nothing beyond this.
fn vl_load_image(
    source: &ImageSource<'_>,
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<VlImage, String> {
    let bytes = match source {
        ImageSource::Path(path) => {
            eprintln!("[VL-DEBUG] preprocessing image: path: {path}");
            std::fs::read(path).map_err(|e| format!("failed to open image {path}: {e}"))?
        }
        ImageSource::Base64(b64) => {
            let bytes = decode_image_bytes(b64)?;
            eprintln!(
                "[VL-DEBUG] preprocessing image: <{}-byte buffer>",
                bytes.len()
            );
            bytes
        }
    };
    let (img_h, img_w) =
        image::preprocessed_dims_from_bytes(&bytes, patch_size, spatial_merge_size)?;
    eprintln!("[VL-DEBUG] preprocessed: {img_w}x{img_h}");
    let grid_h = img_h / patch_size;
    let grid_w = img_w / patch_size;
    let n_visual = grid_h * grid_w / (spatial_merge_size * spatial_merge_size);
    let fingerprint = image::image_fingerprint(&bytes);
    Ok(VlImage {
        bytes,
        fingerprint,
        img_h,
        img_w,
        grid: VlGrid {
            grid_h,
            grid_w,
            n_visual,
        },
    })
}

/// Why encoding one image failed.
enum VlEncodeFail {
    /// The request is at fault (bad image); no decoder state was touched.
    Request(String),
    /// A device failure after work was enqueued: roll the conversation back.
    Forward(&'static str, String),
}

/// Run one image through the vision tower, returning `[n_visual][dim]` f32
/// embedding rows. `allow_vcn` lets a lone image take the pooled VCN decode
/// path (the session lease is single-holder, so a multi-image request decodes
/// on the CPU).
fn vl_encode_image(
    gpu: &mut rdna_compute::Gpu,
    weights: &qwen35_vl::VisionWeights,
    config: &qwen35_vl::VisionConfig,
    image: &VlImage,
    allow_vcn: bool,
    stdout: &mut std::io::Stdout,
    id: &str,
) -> Result<Vec<f32>, VlEncodeFail> {
    let cpu_patches = |image: &VlImage| -> Result<Vec<f32>, VlEncodeFail> {
        let (pixels, fb_h, fb_w) = image::load_and_preprocess_from_bytes(
            &image.bytes,
            config.patch_size,
            config.spatial_merge_size,
        )
        .map_err(VlEncodeFail::Request)?;
        // The header-derived dims and the decoded dims come from the same
        // `smart_resize`; if they ever disagree, feeding `extract_patches`
        // mismatched geometry would corrupt silently.
        if (fb_h, fb_w) != (image.img_h, image.img_w) {
            return Err(VlEncodeFail::Request(format!(
                "image resize mismatch (header {}x{} vs decoded {fb_w}x{fb_h}) — refusing to encode mismatched geometry",
                image.img_w, image.img_h
            )));
        }
        Ok(image::extract_patches(
            &pixels,
            3,
            image.img_h,
            image.img_w,
            config.patch_size,
            config.temporal_patch_size,
            config.spatial_merge_size,
        ))
    };
    #[cfg(feature = "vcn-jpeg")]
    {
        // VCN pre-pass (opt-in via `image.decode`): pooled libva decode plus
        // resized target dims. `VcnDecoded` holds the shared session lease; at
        // most one lives per request and `vcn_to_patches` consumes it right
        // after its terminal sync.
        let vcn = if allow_vcn && image::resolve_image_decode() != image::ImageDecode::Cpu {
            image::vcn_decode(&image.bytes, config.patch_size, config.spatial_merge_size)
        } else {
            None
        };
        if let Some(d) = vcn.filter(|d| (d.img_h, d.img_w) == (image.img_h, image.img_w)) {
            // By value: `vcn_to_patches` consumes the session lease after its
            // terminal sync, so the surface is reusable before the tower runs.
            match image::vcn_to_patches(
                gpu,
                d,
                config.patch_size,
                config.temporal_patch_size,
                config.spatial_merge_size,
            ) {
                Ok(vp) => {
                    // Owned input: `vision_forward_patches` frees `patches`
                    // after patch embedding — nothing to release here, on
                    // either outcome.
                    return qwen35_vl::vision_forward_patches(
                        gpu, weights, config, vp.patches, vp.grid_h, vp.grid_w,
                    )
                    .map_err(|e| VlEncodeFail::Forward("vision_forward_vcn", e.to_string()));
                }
                Err(e) if e.fallback_safe() => {
                    // `Recoverable`: proven pre-enqueue or after a successful
                    // terminal sync, so this GPU is healthy.
                    eprintln!("[daemon/vl] VCN patch build failed ({e}) — CPU fallback");
                }
                Err(e) => {
                    // `TerminalSync`: the work was NOT cancelled and the device
                    // is suspect — outstanding kernels may still read/write the
                    // pooled surface and the request allocations. Report, flush
                    // and terminate WITHOUT destructors or GPU cleanup; the
                    // supervisor must restart the daemon.
                    let msg = format!(
                        "VL vcn_terminal_sync: {e} (GPU work outstanding and uncancelled; daemon terminating)"
                    );
                    eprintln!("[daemon/vl] {msg}");
                    write_error(stdout, id, &msg);
                    let _ = stdout.flush();
                    std::process::exit(1);
                }
            }
        }
    }
    #[cfg(not(feature = "vcn-jpeg"))]
    {
        // `image.decode = vcn|auto` requests the VCN path, but this binary was
        // built without the `vcn-jpeg` cargo feature. CPU decode is correct —
        // warn once so the operator intent never silently no-ops.
        let _ = (allow_vcn, &stdout, id);
        static VCN_FEATURE_WARNED: std::sync::Once = std::sync::Once::new();
        if image::resolve_image_decode() != image::ImageDecode::Cpu {
            VCN_FEATURE_WARNED.call_once(|| {
                eprintln!(
                    "[daemon/vl] image.decode requests VCN but this binary lacks the `vcn-jpeg` feature — CPU fallback"
                );
            });
        }
    }
    let patches = cpu_patches(image)?;
    qwen35_vl::vision_forward(
        gpu,
        weights,
        config,
        &patches,
        image.grid.grid_h,
        image.grid.grid_w,
    )
    .map_err(|e| VlEncodeFail::Forward("vision_forward", e.to_string()))
}
/// Strip an optional `data:...;base64,` prefix and base64-decode image bytes.
/// Shared by the VCN pre-pass and the CPU fallback below so both see the
/// same bytes (and the same errors).
fn decode_image_bytes(b64: &str) -> Result<Vec<u8>, String> {
    // A `data:` URL missing the comma separator is malformed — surface that
    // explicitly rather than letting it fall through to a misleading
    // "invalid byte 'd' at index 0" base64 error.
    let raw_b64 = if let Some(rest) = b64.strip_prefix("data:") {
        match rest.split_once(',') {
            Some((_, after)) => after,
            None => return Err("malformed data URL: missing ',' separator".to_string()),
        }
    } else {
        b64
    };
    Engine::decode(&base64::engine::general_purpose::STANDARD, raw_b64)
        .map_err(|e| format!("failed to decode base64 image data: {e}"))
}

/// Why the batched VL prefill loop in [`generate_vl`] stopped early.
enum VlBatchedStop {
    /// Client abort polled between chunks.
    Abort,
    /// A chunk (or the chunk plan) failed: phase label + message.
    Forward(&'static str, String),
    /// `vl_adaptive_downshift_fail_closed` already rolled back and emitted
    /// the terminal error.
    Reported,
}

pub fn generate_vl(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) {
    let route = crate::ar::GenerationRoute::QwenAr;
    let _route_scope = crate::ar::GenerationRouteScope::enter(route, params.id);
    let GenerateVLParams {
        id,
        prompt,
        system_prompt,
        ref image_sources,
        messages_history,
        enable_thinking,
        reasoning_effort,
        temp,
        top_p,
        max_tokens,
        repeat_penalty,
        repeat_window,
        max_think_tokens,
        assistant_prefix,
        seed,
    } = *params;
    // Render the conversation (CPU only) BEFORE the stream opener: the opener's
    // `started_in_think` is read off the rendered prompt tail, like the text
    // path's. A render failure is reported right after the opener.
    let jinja_enabled = hipfire_config::developer_var("HIPFIRE_JINJA_CHAT")
        .ok()
        .as_deref()
        != Some("0");
    // The text path's prompt-cache kill switch also gates continuing a vision
    // conversation in place; eviction renumbers physical rows, so it never does.
    let turn_cache_enabled = hipfire_config::developer_var("HIPFIRE_QWEN_PROMPT_CACHE")
        .ok()
        .as_deref()
        != Some("0")
        && m.eviction.is_none();
    let rendered = match m.tokenizer.as_ref() {
        None => Err("tokenizer not loaded".to_owned()),
        Some(tokenizer) => {
            // Vision special-token IDs resolved from the tokenizer rather than
            // hardcoded constants. Different VL-capable Qwen variants ship with
            // different IDs for these tokens; a hardcoded mismatch silently
            // splices the wrong tokens into the prompt. Required at load time —
            // panic loudly here so the failure is at first-VL-request, not after
            // a successful but wrong forward pass.
            let special = |name: &str| {
                tokenizer
                    .special_token_id(name)
                    .unwrap_or_else(|| panic!("VL tokenizer missing {name} special token"))
            };
            let image_ids = [
                special("<|vision_start|>"),
                special("<|image_pad|>"),
                special("<|vision_end|>"),
            ];
            vl_render_prompt(
                &VlRenderInputs {
                    tokenizer,
                    chat_template: m.chat_template.as_deref(),
                    jinja_enabled,
                    history: messages_history,
                    n_images: image_sources.len(),
                    system: system_prompt,
                    prompt,
                    enable_thinking,
                    reasoning_effort,
                    assistant_prefix,
                    image_ids,
                    resident_tokens: &m.conversation_tokens,
                    turn_cache_enabled,
                },
                &mut m.asst_turn_cache,
            )
        }
    };
    // Stream-contract opener. MUST be the first event on this request's
    // stream: the HTTP CLI's StreamContractGate rejects any later event that
    // arrives without a preceding gen_start for this id — which stranded
    // image turns after the encoder finished ("no response bytes", wedged
    // slot; 2026-08-27 ledger finding b). Text-path generate() has emitted
    // this since the e99583afa-class fixes.
    let started_in_think = rendered.as_ref().is_ok_and(|p| p.started_in_think);
    crate::ar::emit_generation_start(route, stdout, id, started_in_think);
    let VlPrompt {
        tokens: rendered_tokens,
        whole_conversation,
        ..
    } = match rendered {
        Ok(prompt) => prompt,
        Err(e) => {
            emit_active_attempt_error(stdout, Some(id), &e, "validation", false, false);
            return;
        }
    };
    // INVARIANT: all early returns before the `vision_forward` call (the
    // first expensive GPU allocation in this function) use `write_error`
    // and return without owning any GPU buffers. If you add a GPU
    // allocation above this line, you MUST clean it up on every early
    // return path — the current early returns are safe because they
    // only hold CPU-side data (tokenizer refs, preprocess output).
    //
    // hunt3 M-E: seed the process-global CPU sampler RNG per request. The VL
    // path samples exclusively via sampler::sample_cpu, which draws from this
    // global; without the per-request reset it carried RNG state across
    // requests (and across earlier text-path requests) → cross-request
    // nondeterminism. Seeded by hipfire-engine::request_seed_for (wire `seed`
    // wins, else attempt key + counter), matching the sequential text path.
    hipfire_runtime::llama::reset_cpu_sampler_rng(seed);
    // Adaptive KV poison is sticky until unload/reload. Refuse VL generation so a
    // partial tier transition cannot continue writing into mixed-tier state.
    // Mirror generate() — reset preserves poison, so VL must refuse independently.
    if let Some(ad) = m.kv_adaptive.as_ref() {
        if ad.is_poisoned() {
            let reason = ad
                .poison_reason()
                .unwrap_or("adaptive KV is poisoned; unload/reload required");
            write_error(stdout, id, reason);
            return;
        }
    }
    let vision_config = m.vision_config().unwrap().clone();

    // Image geometry (header only — no pixel decode). It sizes each image's
    // visual-token run, so the whole conversation can be positioned and
    // compared with the resident one before any image is encoded.
    let mut images: Vec<VlImage> = Vec::with_capacity(image_sources.len());
    for source in image_sources {
        match vl_load_image(
            source,
            vision_config.patch_size,
            vision_config.spatial_merge_size,
        ) {
            Ok(image) => images.push(image),
            Err(e) => {
                write_error(stdout, id, &e);
                return;
            }
        }
    }
    let grids: Vec<VlGrid> = images.iter().map(|image| image.grid).collect();
    let image_fingerprints: Vec<[u8; 32]> = images.iter().map(|image| image.fingerprint).collect();
    let (image_pad_id, nl, im_end) = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        (
            tokenizer
                .special_token_id("<|image_pad|>")
                .expect("checked while rendering"),
            tokenizer.encode("\n"),
            tokenizer.encode("<|im_end|>"),
        )
    };
    let visual_counts: Vec<usize> = grids.iter().map(|grid| grid.n_visual).collect();
    let prompt_full = match expand_image_pads(&rendered_tokens, image_pad_id, &visual_counts) {
        Ok(tokens) => tokens,
        Err(e) => {
            emit_active_attempt_error(stdout, Some(id), &e, "validation", false, false);
            return;
        }
    };

    // Continue the resident conversation when this prompt purely extends it
    // (the history of an ongoing chat); otherwise start cold. A reused prefix
    // keeps its KV rows and DeltaNet state — and its images' embeddings — so
    // only the new rows (and the images among them) are encoded and prefilled.
    let reuse = if whole_conversation {
        vl_plan_reuse(
            &m.conversation_tokens,
            &prompt_full,
            m.seq_pos,
            turn_cache_enabled,
            image_pad_id,
            &m.vl_image_fingerprints,
            &image_fingerprints,
        )
    } else {
        0
    };
    if reuse == 0 {
        if m.seq_pos > 0 || !m.conversation_tokens.is_empty() {
            eprintln!(
                "[daemon/vl] resident conversation ({} rows) is not a prefix of this prompt — cold start",
                m.seq_pos
            );
            let reset = crate::common::cold_reset_keep_turn_cache(m, gpu);
            if !reset.rolled_back {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!(
                        "vision conversation reset failed: {}",
                        reset.context.unwrap_or_default()
                    ),
                    "gpu",
                    true,
                    false,
                );
                return;
            }
        }
    } else if let Some(s) = m.speculator.as_mut() {
        // A vision turn does not advance the speculator's own rings; drop them
        // so they cannot describe rows they never saw.
        if let Err(e) = s.reset(gpu) {
            emit_active_attempt_error(
                stdout,
                Some(id),
                &format!("vision conversation reset failed: {e}"),
                "gpu",
                true,
                false,
            );
            return;
        }
    }
    let prompt_tokens: Vec<u32> = prompt_full[reuse..].to_vec();
    // The images whose rows are NOT already resident.
    let encode_idx: Vec<usize> = image_pad_runs(&prompt_full, image_pad_id)
        .iter()
        .enumerate()
        .filter(|(_, (start, _))| *start >= reuse)
        .map(|(index, _)| index)
        .collect();
    let n_visual_tokens: usize = encode_idx.iter().map(|&i| grids[i].n_visual).sum();
    eprintln!(
        "[daemon/vl] prompt: rows={} reused={reuse} new={} images={} encoding={}",
        prompt_full.len(),
        prompt_tokens.len(),
        grids.len(),
        encode_idx.len()
    );

    let tokenizer = m.tokenizer.as_ref().unwrap();
    let Some(b) = m.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
    }) else {
        unreachable!()
    };
    let config = &b.config;
    let weights = &b.weights;
    let scratch = &b.scratch;
    let kv = &mut b.kv_cache;
    let dn = &mut b.dn_state;
    let vision_weights = b.vision_weights.as_ref().unwrap();

    // KV-budget guard — tier-aware without eviction, absolute window with.
    // Adaptive admits against max_seq (floor-tier guarantee); non-adaptive keeps
    // physical_cap. Reserves trailer slots so natural im_end can write ChatML \n.
    // Checked on the real prefill length BEFORE the vision tower runs, the most
    // expensive part of a VL prefill.
    let trailer = nl.len();
    let absolute_pos_vl = m.seq_pos.saturating_add(kv.compact_offset);
    let adaptive_engaged = m.kv_adaptive.is_some();
    let no_evict_cap = vl_no_eviction_kv_cap(m.physical_cap, m.max_seq, adaptive_engaged);
    let max_tokens = if m.eviction.is_none() {
        crate::common::fit_max_tokens(
            max_tokens,
            m.seq_pos + prompt_tokens.len() + trailer,
            no_evict_cap,
        )
    } else {
        crate::common::fit_max_tokens(
            max_tokens,
            absolute_pos_vl + prompt_tokens.len() + trailer,
            m.max_seq,
        )
    };
    let over_budget = if m.eviction.is_none() {
        m.seq_pos
            .saturating_add(prompt_tokens.len())
            .saturating_add(max_tokens)
            .saturating_add(trailer)
            > no_evict_cap
    } else {
        absolute_pos_vl
            .saturating_add(prompt_tokens.len())
            .saturating_add(max_tokens)
            .saturating_add(trailer)
            > m.max_seq
    };
    if over_budget {
        write_error(stdout, id, &format!(
            "request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > cap={} — reload model with a larger max_seq",
            m.seq_pos, prompt_tokens.len(), max_tokens, trailer,
            if m.eviction.is_none() { no_evict_cap } else { m.max_seq },
        ));
        return;
    }

    // 3D mrope phases for this request, from the whole conversation, BEFORE any
    // GPU work so a validation bail is cheap.
    //
    // Disabled while eviction is armed: TriAttention renumbers physical slots
    // mid-prefill (`m.seq_pos = new_phys`), and `MropeCtx::positions` is indexed
    // by physical position. Rather than silently mis-indexing, that
    // configuration keeps today's 1D behavior.
    let mrope_ctx = if m.eviction.is_some() {
        if !grids.is_empty() {
            eprintln!("[daemon/vl] mrope disabled (eviction armed) — falling back to 1D positions");
        }
        None
    } else {
        match build_vl_mrope_ctx(
            &prompt_full,
            image_pad_id,
            &grids,
            vision_config.spatial_merge_size,
            reuse,
            config,
        ) {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                write_error(stdout, id, &format!("vision rope phases: {e}"));
                return;
            }
        }
    };
    let mrope = mrope_ctx.as_ref();

    // Now safe to run the expensive GPU vision encoder, one image at a time
    // (only the images not already resident). A lone image may take the pooled
    // VCN decode; everything else extracts patches on the CPU and uploads.
    let allow_vcn = encode_idx.len() == 1;
    let t_encode = Instant::now();
    let mut visual_tokens: Vec<f32> = Vec::with_capacity(n_visual_tokens * config.dim);
    for &index in &encode_idx {
        match vl_encode_image(
            gpu,
            vision_weights,
            &vision_config,
            &images[index],
            allow_vcn,
            stdout,
            id,
        ) {
            Ok(rows) => visual_tokens.extend_from_slice(&rows),
            Err(VlEncodeFail::Request(e)) => {
                write_error(stdout, id, &e);
                return;
            }
            Err(VlEncodeFail::Forward(phase, e)) => {
                vl_forward_fail(
                    stdout,
                    id,
                    phase,
                    e,
                    gpu,
                    dn,
                    kv,
                    &mut m.kv_adaptive,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                );
                return;
            }
        }
    }
    drop(images);
    if !encode_idx.is_empty() {
        eprintln!(
            "[daemon/vl] vision encode: {} image(s), {n_visual_tokens} visual tokens in {} ms",
            encode_idx.len(),
            t_encode.elapsed().as_millis()
        );
    }
    if visual_tokens.len() != n_visual_tokens * config.dim {
        vl_forward_fail(
            stdout,
            id,
            "vision_forward",
            format!(
                "the tower returned {} values for {n_visual_tokens} visual tokens of width {}",
                visual_tokens.len(),
                config.dim
            ),
            gpu,
            dn,
            kv,
            &mut m.kv_adaptive,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
        );
        return;
    }

    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };
    let prefill_tokens = prompt_tokens.len();
    let t0 = Instant::now();
    // Test-only fault point (G4.8 lifecycle evidence), armed in-process via
    // `common::arm_vision_fault`: `"prefill"` fails this request through the
    // fail-closed error epilogue after prefill; `"decode"` fails it on the
    // first decode iteration. Unarmed (production) => one thread-local load.
    let vl_fault: Option<&'static str> = crate::common::peek_vision_fault();

    // Mirror the text path: <think>/</think> as paired open/close. The
    // previous implementation queried "💭" twice (open == close) which
    // collapsed depth tracking and made `in_think` always-false; the
    // force-close splice also encoded the open emoji, doubling the
    // unclosed depth instead of closing it.
    let think_pair = match (
        tokenizer.special_token_id("<think>"),
        tokenizer.special_token_id("</think>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };

    // ── Prefill route: batched (default) vs per-token ───────────────────────
    // Chosen BEFORE any prefill work. The batched route runs the prompt through
    // `forward_prefill_batch` chunks with the vision embeddings spliced over the
    // `<|image_pad|>` rows and the SAME absolute mrope phases the per-token arm
    // uses (`qwen35::vl_prefill`); it needs a dense, batch-admissible model and
    // no eviction. Anything else (and `HIPFIRE_VL_PREFILL=token`) keeps the
    // per-token loop below. Decode is unchanged either way.
    let vl_token_forced = hipfire_config::developer_var("HIPFIRE_VL_PREFILL")
        .is_ok_and(|v| v.trim().eq_ignore_ascii_case("token"));
    let vl_batched_why: Result<(), String> = if vl_token_forced {
        Err("HIPFIRE_VL_PREFILL=token".to_string())
    } else if m.eviction.is_some() {
        Err("eviction armed".to_string())
    } else if m.ep.is_some() || m.pp > 1 {
        Err("multi-GPU topology".to_string())
    } else if prompt_tokens.len() < 2 {
        Err("prompt below the 2-row batch minimum".to_string())
    } else {
        qwen35::vl_prefill::vl_batched_prefill_admitted(gpu, weights, config, kv, dn)
    };
    let vl_batched = vl_batched_why.is_ok();
    match &vl_batched_why {
        Ok(()) => eprintln!("[daemon/vl] prefill: batched"),
        Err(why) => eprintln!("[daemon/vl] prefill: per-token ({why})"),
    }
    if vl_batched {
        let spans = qwen35::vl_prefill::image_spans(
            &prompt_tokens,
            image_pad_id,
            m.seq_pos,
            n_visual_tokens,
        );
        let guard = match qwen35::vl_prefill::install(
            gpu,
            qwen35::vl_prefill::VlPrefillInputs {
                spans,
                prompt_start: m.seq_pos,
                prompt_len: prompt_tokens.len(),
                embeds: &visual_tokens,
                dim: config.dim,
                mrope,
            },
        ) {
            Ok(g) => g,
            Err(e) => {
                vl_forward_fail(
                    stdout,
                    id,
                    "vl_prefill install",
                    e,
                    gpu,
                    dn,
                    kv,
                    &mut m.kv_adaptive,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                );
                return;
            }
        };
        // Outer chunking follows the ar.rs no-eviction branch: the admitted
        // ordinary ceiling, or the hard `PREFILL_MAX_BATCH` under adaptive KV
        // so the controller's downshift boundaries stay exact.
        let chunk_max = if m.kv_adaptive.is_some() {
            qwen35::PREFILL_MAX_BATCH
        } else {
            let deduction = qwen35::prefill::request_kv_growth_deduction(
                gpu,
                kv,
                m.seq_pos + prompt_tokens.len(),
            );
            match qwen35::prefill::ordinary_prefill_chunk_limit_deducting(
                gpu, weights, config, dn, kv, deduction,
            ) {
                Ok(limit) => limit,
                Err(e) => {
                    eprintln!("[daemon/vl] chunk-limit query failed ({e}); keeping legacy ceiling");
                    qwen35::prefill_max_batch(gpu)
                }
            }
        };
        let mut stop: Option<VlBatchedStop> = None;
        let mut start = 0usize;
        while start < prompt_tokens.len() {
            // Client-cancel poll between chunks (the encoder phase cannot be
            // interrupted; the first iteration picks up an abort sent during it).
            if check_abort(id) {
                stop = Some(VlBatchedStop::Abort);
                break;
            }
            let remaining = prompt_tokens.len() - start;
            // Exact chunk planner: never leaves a singleton tail, which the
            // batched entry would hand to the per-token fallback (refused
            // for VL rows).
            let Some(outer) = qwen35::prefill::next_exact_prefill_chunk_len(remaining, chunk_max)
            else {
                stop = Some(VlBatchedStop::Forward(
                    "forward_prefill_batch (vl)",
                    format!("chunk plan cannot split {remaining} rows under ceiling {chunk_max}"),
                ));
                break;
            };
            let end = start + outer;
            if let Err(e) = qwen35::forward_prefill_batch(
                gpu,
                weights,
                config,
                &prompt_tokens[start..end],
                m.seq_pos,
                kv,
                dn,
                scratch,
                None,
                None,
                None,
                None,
            ) {
                stop = Some(VlBatchedStop::Forward(
                    "forward_prefill_batch (vl)",
                    e.to_string(),
                ));
                break;
            }
            m.seq_pos += outer;
            // Adaptive KV: downshift BETWEEN chunks the moment the start-tier
            // buffer fills (same boundary the per-token loop and the text path use).
            if vl_adaptive_downshift_fail_closed(
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                gpu,
                kv,
                dn,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                stdout,
                id,
                "vl-prefill",
            ) {
                stop = Some(VlBatchedStop::Reported);
                break;
            }
            start = end;
            crate::common::emit_prefill_progress(
                stdout,
                id,
                Some(active_attempt_id()),
                end,
                prompt_tokens.len(),
            );
        }
        // Drop the scope before decode (and before any terminal below).
        let released = guard.release(gpu);
        match stop {
            Some(VlBatchedStop::Abort) => {
                // Same canonical rollback + cancelled pair as the per-token loop.
                vl_cancel_after_rollback(
                    stdout,
                    id,
                    0,
                    gpu,
                    dn,
                    kv,
                    &mut m.kv_adaptive,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                    &mut m.dflash_checkpoints,
                    &mut m.asst_turn_cache,
                    &mut m.speculator,
                );
                return;
            }
            Some(VlBatchedStop::Forward(phase, msg)) => {
                vl_forward_fail(
                    stdout,
                    id,
                    phase,
                    msg,
                    gpu,
                    dn,
                    kv,
                    &mut m.kv_adaptive,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                );
                return;
            }
            // `vl_adaptive_downshift_fail_closed` already rolled back and
            // emitted the terminal error.
            Some(VlBatchedStop::Reported) => return,
            None => {}
        }
        if let Err(e) = released {
            vl_forward_fail(
                stdout,
                id,
                "vl_prefill release",
                e,
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
            );
            return;
        }
    }

    // Per-token prefill with vision token embedding for image_pad positions
    // (`HIPFIRE_VL_PREFILL=token`, eviction, MoE / non-batch-admissible
    // models). We advance m.seq_pos in-loop and call maybe_evict /
    // maybe_downshift after every committed write. Lazy VMM map/growth
    // failures and adaptive transition errors are request-scoped (no panic,
    // no later token emit). Empty when the batched route above already ran.
    let per_token_prompt: &[u32] = if vl_batched { &[] } else { &prompt_tokens };
    let mut visual_idx = 0usize;
    for &token in per_token_prompt.iter() {
        // Client-cancel poll. The encoder phase before this loop cannot be
        // interrupted mid-kernel, so prefill's first iteration is where an
        // abort signalled during vision_forward takes effect. The terminal
        // MUST be the canonical cancelled pair: falling through to the done
        // handshake on an aborted attempt strands the serve admission guard
        // permanently (2026-08-27 ledger finding c — slot wedged ≥3 min on
        // every mid-encode disconnect before these polls existed).
        if check_abort(id) {
            // Abort with uncommitted prefill state: roll back through the
            // canonical epilogue before the cancelled pair so the next
            // request re-prefills from clean state (no deferred reset).
            vl_cancel_after_rollback(
                stdout,
                id,
                0,
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
            );
            return;
        }
        if token == image_pad_id && visual_idx < n_visual_tokens {
            let emb = &visual_tokens[visual_idx * config.dim..(visual_idx + 1) * config.dim];
            if let Err(e) = qwen35::forward_scratch_embed_mrope(
                gpu, weights, config, emb, m.seq_pos, kv, dn, scratch, mrope,
            ) {
                vl_forward_fail(
                    stdout,
                    id,
                    "forward_scratch_embed (prefill)",
                    e,
                    gpu,
                    dn,
                    kv,
                    &mut m.kv_adaptive,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                );
                return;
            }
            visual_idx += 1;
        } else if let Err(e) = qwen35::forward_scratch_mrope(
            gpu, weights, config, token, m.seq_pos, kv, dn, scratch, mrope,
        ) {
            vl_forward_fail(
                stdout,
                id,
                "forward_scratch (prefill)",
                e,
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
            );
            return;
        }
        m.seq_pos += 1;
        if let Some(ref ev) = m.eviction {
            match ev.maybe_evict(gpu, kv, m.seq_pos) {
                Ok(Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                })) => {
                    m.seq_pos = new_phys;
                }
                Ok(None) => {}
                Err(e) => {
                    vl_forward_fail(
                        stdout,
                        id,
                        "maybe_evict (prefill)",
                        e,
                        gpu,
                        dn,
                        kv,
                        &mut m.kv_adaptive,
                        &mut m.seq_pos,
                        &mut m.conversation_tokens,
                        &mut m.prefill_checkpoints,
                    );
                    return;
                }
            }
        }
        // Adaptive KV: downshift BETWEEN prefill tokens the moment the
        // start-tier buffer fills so a long multi-chunk visual+text prompt
        // cannot overflow current-stride capacity before decode begins.
        if vl_adaptive_downshift_fail_closed(
            &mut m.kv_adaptive,
            &mut m.seq_pos,
            gpu,
            kv,
            dn,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            stdout,
            id,
            "vl-prefill",
        ) {
            return;
        }
    }

    m.conversation_tokens.extend_from_slice(&prompt_tokens);
    m.vl_image_fingerprints = image_fingerprints;

    // Adaptive KV: post-prefill catch-up before first sample/decode write.
    if vl_adaptive_downshift_fail_closed(
        &mut m.kv_adaptive,
        &mut m.seq_pos,
        gpu,
        kv,
        dn,
        &mut m.conversation_tokens,
        &mut m.prefill_checkpoints,
        stdout,
        id,
        "vl-post-prefill",
    ) {
        return;
    }
    // Injected prefill fault (test hook): prefill committed decoder state,
    // so fail through the canonical error epilogue exactly like a real
    // forward failure.
    if vl_fault == Some("prefill") {
        vl_forward_fail(
            stdout,
            id,
            "forward_scratch (injected-fault)",
            "injected vision fault: prefill",
            gpu,
            dn,
            kv,
            &mut m.kv_adaptive,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
        );
        return;
    }

    // hunt3 M-D: repeat-penalty / n-gram-block history must be scoped to the
    // GENERATED tokens only (mirrors the text path's `ngram_scope_start` set to
    // conversation_tokens.len() after prefill). Passing the full conversation
    // makes the trailing window prompt-dominated, suppressing the names/numbers
    // a VL transcription task must reproduce.
    let vl_ngram_scope_start = m.conversation_tokens.len();

    // Generate. CPU-side sampling — VL path predates the GPU sampler
    // and downloads logits each step:
    //   - first sample: top-p only (no repeat penalty);
    //   - subsequent samples: repeat penalty, then top-p sample.
    //
    // Unlike ordinary text generation, do not apply the positional 3..6-gram
    // ban here. OCR/layout output legitimately repeats table and markup
    // sequences. The configured LoopGuard remains available for pathological
    // full-loop termination, and the text paths retain their n-gram policies.
    //
    // Attractor-block uses CPU-side mutation of the downloaded logits
    // vector (`block_attractor_unclosed_cpu`) instead of the previous
    // GPU memcpy + redownload — saves a full vocab-sized DMA per token.
    let mut logits = match gpu.download_f32(&scratch.logits) {
        Ok(v) => v,
        Err(e) => {
            vl_forward_fail(
                stdout,
                id,
                "download_f32 (post-prefill)",
                e,
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
            );
            return;
        }
    };
    if let Some((open, close)) = think_pair {
        block_attractor_unclosed_cpu(&mut logits, &m.conversation_tokens, open, close, 20, 2);
    }
    let vl_cfg_first = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty: 1.0,
        repeat_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        blocked_tokens: Vec::new(),
        // `sample_cpu` honours top_k / min_p, but `GenerateVLParams` does not
        // carry the request's values yet, so VL keeps the top-20 default.
        top_k: None,
        min_p: None,
    };
    let vl_cfg = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        blocked_tokens: Vec::new(),
        top_k: None,
        min_p: None,
    };
    let mut next_token = sampler::sample_cpu(&mut logits, &[], &vl_cfg_first);
    let t_prefill = Instant::now();
    let mut generated = 0;
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    // Typed-emission state for the v2 stream contract (see
    // `vl_route_decode_text`): EosFilter owns UTF-8 boundaries + EOT marker
    // suppression, ThinkOutputRouter owns the reasoning/content channel
    // split. `emitted_bytes` above becomes the bytes-FED-to-filter cursor —
    // the filter holds back partial UTF-8/marker tails internally, so the
    // old valid-prefix arithmetic is gone.
    let mut vl_filter = EosFilter::new(hipfire_runtime::eos_filter::qwen35_eos_filter_config());
    let mut vl_think = ThinkOutputRouter::new(started_in_think);
    // Think-depth tracking via token IDs (not UTF-8 rfind).
    // The previous implementation decoded the full streamed output to a
    // string and ran rfind on every token — O(N²) total, fragile to
    // tokenizer changes. Since `think_pair` already gives us the
    // open/close token IDs, we can track depth incrementally in O(1).
    let mut think_depth: usize = 0; // number of unmatched opens seen
    let mut think_count: usize = 0; // tokens emitted while depth > 0

    // N-gram loop detector — mirrors the text path. Catches answer-phase
    // attractor loops that the think cap and repeat penalty miss.
    let loop_guard =
        hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get());
    // A terminator ended the turn (vs. the budget running out).
    let mut decoded_eot = false;
    // The answer-channel text the client will echo back as this turn.
    let mut visible_text = String::new();

    'vl_generate: while generated < max_tokens {
        // Decode-side client-cancel poll — roll back synchronously (same
        // rule as the prefill poll above); the stale comment's deferred
        // "next dispatch reset" no longer applies.
        if check_abort(id) {
            vl_cancel_after_rollback(
                stdout,
                id,
                generated,
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
            );
            return;
        }
        // Injected decode fault (test hook): fails on the first iteration
        // through the canonical error epilogue.
        if vl_fault == Some("decode") {
            vl_forward_fail(
                stdout,
                id,
                "forward_scratch (decode, injected-fault)",
                "injected vision fault: decode",
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
            );
            return;
        }
        // Commit KV for this sampled token BEFORE any client-visible emit so a
        // lazy VMM map/growth failure cannot stream an uncommitted token.
        // Order: forward → seq_pos++ → evict → downshift → then
        // generated/conversation/committed/token text. On failure: cold-reset
        // + request error only (no failed token). Terminators break after a
        // successful commit (same as AR).
        if let Err(e) = qwen35::forward_scratch_mrope(
            gpu, weights, config, next_token, m.seq_pos, kv, dn, scratch, mrope,
        ) {
            vl_forward_fail(
                stdout,
                id,
                "forward_scratch (decode)",
                e,
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
            );
            return;
        }
        m.seq_pos += 1;
        if let Some(ref ev) = m.eviction {
            match ev.maybe_evict(gpu, kv, m.seq_pos) {
                Ok(Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                })) => {
                    m.seq_pos = new_phys;
                }
                Ok(None) => {}
                Err(e) => {
                    vl_forward_fail(
                        stdout,
                        id,
                        "maybe_evict (decode)",
                        e,
                        gpu,
                        dn,
                        kv,
                        &mut m.kv_adaptive,
                        &mut m.seq_pos,
                        &mut m.conversation_tokens,
                        &mut m.prefill_checkpoints,
                    );
                    return;
                }
            }
        }
        if vl_adaptive_downshift_fail_closed(
            &mut m.kv_adaptive,
            &mut m.seq_pos,
            gpu,
            kv,
            dn,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            stdout,
            id,
            "vl-decode",
        ) {
            return;
        }

        generated += 1;
        m.conversation_tokens.push(next_token);
        streamed_tokens.push(next_token);
        emit_committed_event(
            stdout,
            id,
            next_token,
            generated - 1,
            t0.elapsed().as_millis() as u64,
        );

        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
        let new_bytes = &all_bytes[emitted_bytes..];
        emitted_bytes = all_bytes.len();
        if vl_route_decode_text_into(
            stdout,
            id,
            &mut vl_filter,
            &mut vl_think,
            new_bytes,
            Some(&mut visible_text),
        ) || next_token == config.eos_token
            || im_end_token == Some(next_token)
            || tokenizer.is_terminator(next_token)
        {
            decoded_eot = true;
            break;
        }

        if let Some(hipfire_runtime::loop_guard::StopReason::NgramRepeat { count, .. }) =
            loop_guard.check(&streamed_tokens)
        {
            let window_len = loop_guard.window_len(streamed_tokens.len());
            let _ = writeln!(
                stdout,
                r#"{{"type":"info","id":"{}","message":"ngram loop detected (4gram repeated {}× in last {} tokens) — forcing EOS"}}"#,
                id, count, window_len,
            );
            let _ = stdout.flush();
            break;
        }

        logits = match gpu.download_f32(&scratch.logits) {
            Ok(v) => v,
            Err(e) => {
                vl_forward_fail(
                    stdout,
                    id,
                    "download_f32 (decode)",
                    e,
                    gpu,
                    dn,
                    kv,
                    &mut m.kv_adaptive,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                );
                return;
            }
        };
        // hunt3 M-D: scope repeat-penalty history to generated-only.
        // Exact transcription legitimately repeats HTML/Markdown n-grams
        // (`<tr>`, `<td>`, table delimiters, boilerplate). Hard no-repeat
        // blocking corrupts those outputs by forcing a lower-ranked token
        // whenever a 3..6-gram recurs. Keep the ordinary configured repeat
        // penalty in `sample_cpu`, but do not mutate VL logits with an
        // unconditional no-repeat constraint.
        let vl_ngram_scope = &m.conversation_tokens[vl_ngram_scope_start..];
        if let Some((open, close)) = think_pair {
            block_attractor_unclosed_cpu(&mut logits, &m.conversation_tokens, open, close, 20, 2);
        }

        next_token = sampler::sample_cpu(&mut logits, vl_ngram_scope, &vl_cfg);

        if max_think_tokens > 0 {
            if let Some((open, close)) = think_pair {
                // Incremental think-depth tracking via token IDs — O(1)
                // per token instead of the previous O(N²) decode+rfind.
                if next_token == open {
                    think_depth += 1;
                    think_count = 1;
                } else if next_token == close {
                    think_depth = think_depth.saturating_sub(1);
                    if think_depth == 0 {
                        think_count = 0;
                    }
                } else if think_depth > 0 {
                    think_count += 1;
                }

                if think_depth > 0 && think_count >= max_think_tokens {
                    let close_tokens = tokenizer.encode("</think>\n");
                    let budget_left = max_tokens.saturating_sub(generated);
                    let take = close_tokens.len().min(budget_left);
                    for &t in &close_tokens[..take] {
                        // KV write before any emit — same contract as main decode.
                        if let Err(e) = qwen35::forward_scratch_mrope(
                            gpu, weights, config, t, m.seq_pos, kv, dn, scratch, mrope,
                        ) {
                            vl_forward_fail(
                                stdout,
                                id,
                                "forward_scratch (vl-think-close)",
                                e,
                                gpu,
                                dn,
                                kv,
                                &mut m.kv_adaptive,
                                &mut m.seq_pos,
                                &mut m.conversation_tokens,
                                &mut m.prefill_checkpoints,
                            );
                            return;
                        }
                        m.seq_pos += 1;
                        if let Some(ref ev) = m.eviction {
                            match ev.maybe_evict(gpu, kv, m.seq_pos) {
                                Ok(Some(hipfire_runtime::triattn::EvictionResult {
                                    new_physical: new_phys,
                                    ..
                                })) => {
                                    m.seq_pos = new_phys;
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    vl_forward_fail(
                                        stdout,
                                        id,
                                        "maybe_evict (vl-think-close)",
                                        e,
                                        gpu,
                                        dn,
                                        kv,
                                        &mut m.kv_adaptive,
                                        &mut m.seq_pos,
                                        &mut m.conversation_tokens,
                                        &mut m.prefill_checkpoints,
                                    );
                                    return;
                                }
                            }
                        }
                        if vl_adaptive_downshift_fail_closed(
                            &mut m.kv_adaptive,
                            &mut m.seq_pos,
                            gpu,
                            kv,
                            dn,
                            &mut m.conversation_tokens,
                            &mut m.prefill_checkpoints,
                            stdout,
                            id,
                            "vl-think-close",
                        ) {
                            return;
                        }
                        m.conversation_tokens.push(t);
                        streamed_tokens.push(t);
                        // hunt3 H-F: emit the committed-token event for force-closed
                        // </think> tokens too, BEFORE `generated += 1`, so the
                        // committed pos stays in lockstep with the streamed count
                        // under HIPFIRE_EMIT_TOKEN_IDS=1. The VL main loop uses
                        // `generated - 1` after its increment; here `generated`
                        // (pre-increment) is the same value.
                        emit_committed_event(
                            stdout,
                            id,
                            t,
                            generated,
                            t0.elapsed().as_millis() as u64,
                        );

                        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                        let new_bytes = &all_bytes[emitted_bytes..];
                        emitted_bytes = all_bytes.len();
                        // Same typed routing as the main decode site: the
                        // forced `</think>` closer is consumed by the router
                        // (channel flips to content), never emitted literally.
                        if vl_route_decode_text_into(
                            stdout,
                            id,
                            &mut vl_filter,
                            &mut vl_think,
                            new_bytes,
                            Some(&mut visible_text),
                        ) {
                            generated += 1;
                            decoded_eot = true;
                            break 'vl_generate;
                        }
                        generated += 1;
                    }
                    think_count = 0;
                    think_depth = 0; // Must reset — the close tokens
                                     // above bypass the incremental tracker, so depth
                                     // is still > 0 here. Without this, any subsequent
                                     // non-open/close token would re-trigger the cap.
                    if generated >= max_tokens {
                        break;
                    }
                    logits = match gpu.download_f32(&scratch.logits) {
                        Ok(v) => v,
                        Err(e) => {
                            vl_forward_fail(
                                stdout,
                                id,
                                "download_f32 (vl-think-close)",
                                e,
                                gpu,
                                dn,
                                kv,
                                &mut m.kv_adaptive,
                                &mut m.seq_pos,
                                &mut m.conversation_tokens,
                                &mut m.prefill_checkpoints,
                            );
                            return;
                        }
                    };
                    block_attractor_unclosed_cpu(
                        &mut logits,
                        &m.conversation_tokens,
                        open,
                        close,
                        20,
                        2,
                    );
                    // hunt3 M-D: generated-only repeat-penalty scope.
                    next_token = sampler::sample_cpu(
                        &mut logits,
                        &m.conversation_tokens[vl_ngram_scope_start..],
                        &vl_cfg,
                    );
                }
            }
        }
    }

    // ChatML \n boundary — run through forward to keep KV cache + DeltaNet in sync
    if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
        for &t in &nl {
            if let Err(e) = qwen35::forward_scratch_mrope(
                gpu, weights, config, t, m.seq_pos, kv, dn, scratch, mrope,
            ) {
                vl_forward_fail(
                    stdout,
                    id,
                    "forward_scratch (vl-trailer)",
                    e,
                    gpu,
                    dn,
                    kv,
                    &mut m.kv_adaptive,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                );
                return;
            }
            m.seq_pos += 1;
            if let Some(ref ev) = m.eviction {
                match ev.maybe_evict(gpu, kv, m.seq_pos) {
                    Ok(Some(hipfire_runtime::triattn::EvictionResult {
                        new_physical: new_phys,
                        ..
                    })) => {
                        m.seq_pos = new_phys;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        vl_forward_fail(
                            stdout,
                            id,
                            "maybe_evict (vl-trailer)",
                            e,
                            gpu,
                            dn,
                            kv,
                            &mut m.kv_adaptive,
                            &mut m.seq_pos,
                            &mut m.conversation_tokens,
                            &mut m.prefill_checkpoints,
                        );
                        return;
                    }
                }
            }
            if vl_adaptive_downshift_fail_closed(
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                gpu,
                kv,
                dn,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                stdout,
                id,
                "vl-trailer",
            ) {
                return;
            }
            m.conversation_tokens.push(t);
        }
    }

    if check_abort(id) {
        // Post-loop abort latch: same synchronous rollback as the in-loop
        // polls — the done handshake below must never run on an aborted
        // attempt.
        vl_cancel_after_rollback(
            stdout,
            id,
            generated,
            gpu,
            dn,
            kv,
            &mut m.kv_adaptive,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            &mut m.speculator,
        );
        return;
    }
    // Flush any trailing partial think marker as ordinary text in its
    // current channel (text-AR finish parity) before the terminal.
    vl_finish_think_routing_into(
        stdout,
        id,
        &mut vl_filter,
        &mut vl_think,
        Some(&mut visible_text),
    );
    let t_end = Instant::now();
    let total_s = t_end.duration_since(t0).as_secs_f64();
    let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
    let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
    let tok_s = if total_s > 0.0 {
        generated as f64 / total_s
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_s > 0.0 {
        prefill_tokens as f64 / prefill_s
    } else {
        0.0
    };
    let decode_tok_s = if decode_s > 0.0 {
        generated as f64 / decode_s
    } else {
        0.0
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (tok_s * 10.0).round() / 10.0,
        "prefill_tokens": prefill_tokens,
        "cached_tokens": reuse,
        "prefill_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "ttft_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "attempt_id": active_attempt_id(),
        "finish_reason": crate::common::length_or_stop(generated, max_tokens, decoded_eot),
    });
    eprintln!(
        "[daemon/vl] turn: prefilled {prefill_tokens} rows over {reuse} resident in {:.0} ms, decoded {generated} tokens in {:.0} ms",
        prefill_s * 1000.0,
        decode_s * 1000.0
    );
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => {
            // Publish the emitted assistant turn so the next turn's history
            // render replays these exact tokens: the client echoes the visible
            // text, and a template re-render of that text would not reproduce
            // the tokens (reasoning, BPE) the KV already holds. Natural stops
            // only — a length-capped or loop-cut turn is not what a re-render
            // would continue from.
            if whole_conversation && decoded_eot {
                let nl_set: std::collections::HashSet<u32> = nl.iter().copied().collect();
                let cached_seq =
                    crate::qwen::qwen_dflash_cache_seq(&streamed_tokens, im_end_token, &nl_set);
                let action = TurnCacheAction {
                    store: true,
                    fingerprint_text: crate::common::normalize_asst_turn_for_fingerprint(
                        &visible_text,
                    ),
                    tool_calls: Vec::new(),
                };
                let tok = m.tokenizer.as_ref().unwrap();
                let _ = apply_turn_cache_action(
                    |fp, seq| {
                        m.asst_turn_cache.insert(
                            fp,
                            crate::qwen::qwen_cached_turn_entry(tok, seq, started_in_think),
                        );
                    },
                    &action,
                    cached_seq,
                );
            }
            crate::ar::emit_active_route_done_value(stdout, &pending_done)
        }
        ClientTerminalDecision::Abort => {
            // Abort after the turn committed: roll back like the in-loop
            // polls (and text AR), or the cancelled turn stays in context.
            vl_cancel_after_rollback(
                stdout,
                id,
                generated,
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
            );
        }
    }
}

pub fn generate_vl_dots_ocr(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) {
    use hipfire_arch_dots_ocr::image as dots_image;
    let route = crate::ar::GenerationRoute::DotsOcr;
    let _route_scope = crate::ar::GenerationRouteScope::enter(route, params.id);
    // Stream-contract opener — same HTTP-gate rationale as generate_vl above.
    crate::ar::emit_generation_start(route, stdout, params.id, false);
    let t0 = Instant::now();
    let GenerateVLParams {
        id,
        prompt,
        max_tokens,
        ..
    } = *params;

    // 1. Preprocess image (CPU; no model borrow yet so error returns are clean).
    let image_source = match params.single_image() {
        Ok(source) => source,
        Err(e) => {
            write_error(stdout, id, &e);
            return;
        }
    };
    let img = match image_source {
        ImageSource::Path(path) => {
            eprintln!("[dots-ocr] preprocessing image: {path}");
            dots_image::preprocess_image(Path::new(path))
        }
        ImageSource::Base64(b64) => {
            // Strip an optional `data:<mime>;base64,` URL prefix.
            let raw_b64 = match b64.strip_prefix("data:") {
                Some(rest) => match rest.split_once(',') {
                    Some((_, after)) => after,
                    None => {
                        write_error(stdout, id, "malformed data URL: missing ',' separator");
                        return;
                    }
                },
                None => &b64[..],
            };
            eprintln!(
                "[dots-ocr] preprocessing base64 image (<{}-byte payload>)",
                raw_b64.len()
            );
            match Engine::decode(&base64::engine::general_purpose::STANDARD, raw_b64) {
                Ok(bytes) => dots_image::preprocess_image_bytes(&bytes),
                Err(e) => {
                    write_error(stdout, id, &format!("dots.ocr: base64 decode failed: {e}"));
                    return;
                }
            }
        }
    };
    let img = match img {
        Ok(i) => i,
        Err(e) => {
            write_error(
                stdout,
                id,
                &format!("dots.ocr image preprocess failed: {e}"),
            );
            return;
        }
    };
    let n_visual = img.n_visual_tokens();
    let n_patches = img.n_patches();
    eprintln!(
        "[dots-ocr] grid {}x{}, {} patches → {} visual tokens",
        img.grid_h, img.grid_w, n_patches, n_visual
    );

    let max_seq = m.max_seq;

    // 2. Model state (disjoint field borrows of `m`).
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let config = m.dots_ocr().unwrap().config.clone();
    let text_cfg = config.text.clone();
    let dim = text_cfg.hidden_size;
    // Weights/state via raw pointers to allow owned config while keeping disjoint borrows.
    let bundle_ptr: *mut hipfire_arch_dots_ocr::DotsOcrBundle =
        match m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_dots_ocr::DotsOcrBundle>()
        }) {
            Some(b) => b as *mut _,
            None => unreachable!(),
        };
    let weights = unsafe { &(*bundle_ptr).weights };
    let state = unsafe { &mut (*bundle_ptr).state };
    // 3. Build the prompt (HF-exact framing; imgpad count == n_visual by construction).
    let prompt_ids = dots_ocr::build_prompt_ids(tokenizer, prompt, n_visual);
    if prompt_ids.len().saturating_add(max_tokens) > max_seq {
        write_error(stdout, id, &format!(
            "dots.ocr request ({} prompt + {} gen) exceeds KV budget ({}); reload with a larger --max-seq",
            prompt_ids.len(), max_tokens, max_seq));
        return;
    }
    if check_abort(id) {
        // Cancel before prefill: decoder state is pristine but route through
        // the fail-closed epilogue for a uniform cancel/reuse contract.
        dots_ar_cancel(state, gpu, stdout, id, 0);
        return;
    }

    // 4. Vision encoder → merged visual tokens.
    let patch_cols = img.patches.len() / n_patches;
    let patches_gpu = match gpu.upload_f32(&img.patches, &[n_patches, patch_cols]) {
        Ok(t) => t,
        Err(e) => {
            write_error(stdout, id, &format!("dots.ocr patch upload failed: {e:?}"));
            return;
        }
    };
    let merged_gpu = match dots_ocr::vision_forward(
        gpu,
        &weights.vision,
        &config.vision,
        &patches_gpu,
        img.grid_h,
        img.grid_w,
        || check_abort(id),
    ) {
        Ok(Some(t)) => t,
        Ok(None) => {
            let _ = gpu.free_tensor(patches_gpu);
            dots_ar_cancel(state, gpu, stdout, id, 0);
            return;
        }
        Err(e) => {
            let _ = gpu.free_tensor(patches_gpu);
            dots_ar_fail(
                state,
                gpu,
                stdout,
                id,
                &format!("dots.ocr vision_forward failed: {e:?}"),
            );
            return;
        }
    };
    let _ = gpu.free_tensor(patches_gpu);
    let merged = match gpu.download_f32(&merged_gpu) {
        Ok(v) => v,
        Err(e) => {
            let _ = gpu.free_tensor(merged_gpu);
            dots_ar_fail(
                state,
                gpu,
                stdout,
                id,
                &format!("dots.ocr merger download failed: {e:?}"),
            );
            return;
        }
    };
    let _ = gpu.free_tensor(merged_gpu);
    // Hard guard: merger output count MUST equal the imgpad-slot count, or
    // the splice silently corrupts the text context (PRD §"Vision token splicing").
    if merged.len() != n_visual * dim {
        dots_ar_fail(
            state,
            gpu,
            stdout,
            id,
            &format!(
            "dots.ocr: merger produced {} values but prompt has {} <|imgpad|> slots × {} dims = {}",
            merged.len(), n_visual, dim, n_visual * dim),
        );
        return;
    }

    // 5. Prefill: build the [seq × dim] embedding matrix (token-embedding
    // rows for text positions, spliced vision-merger rows at IMGPAD slots)
    // and run it through the batched prefill in one pass. Only the ~215
    // text positions need a GPU embedding lookup; the 4880 visual rows are
    // already host-resident in `merged`.
    state.reset();
    let t_prefill = Instant::now();
    let mut embeds = vec![0f32; prompt_ids.len() * dim];
    let emb_scratch = match gpu.alloc_tensor(&[dim], rdna_compute::DType::F32) {
        Ok(t) => t,
        Err(e) => {
            dots_ar_fail(
                state,
                gpu,
                stdout,
                id,
                &format!("dots.ocr embed scratch alloc failed: {e:?}"),
            );
            return;
        }
    };
    let mut visual_idx = 0usize;
    let mut embed_err: Option<String> = None;
    for (pos, &token) in prompt_ids.iter().enumerate() {
        if check_abort(id) {
            let _ = gpu.free_tensor(emb_scratch);
            dots_ar_cancel(state, gpu, stdout, id, 0);
            return;
        }
        if token == dots_ocr::IMGPAD_ID {
            embeds[pos * dim..(pos + 1) * dim]
                .copy_from_slice(&merged[visual_idx * dim..(visual_idx + 1) * dim]);
            visual_idx += 1;
        } else {
            // Dispatch the token-embedding lookup on the actual embedding
            // format. HFQ dots.ocr ships Q8_0 embeddings, but the
            // safetensors/Dir loader uploads F32 — hardcoding the Q8 kernel
            // here misreads F32 bytes as Q8 blocks, corrupting every text
            // token's embedding (the model then ignores the prompt). Mirrors
            // the per-format dispatch in `llama::forward`.
            let lookup = hipfire_runtime::llama::embedding_lookup_dispatch(
                gpu,
                weights.text.embd_format,
                &weights.text.token_embd,
                &emb_scratch,
                token,
                dim,
            );
            if let Err(e) = lookup {
                embed_err = Some(format!("embedding lookup: {e:?}"));
                break;
            }
            match gpu.download_f32(&emb_scratch) {
                Ok(row) => embeds[pos * dim..(pos + 1) * dim].copy_from_slice(&row),
                Err(e) => {
                    embed_err = Some(format!("embedding download: {e:?}"));
                    break;
                }
            }
        }
    }
    let _ = gpu.free_tensor(emb_scratch);
    if check_abort(id) {
        dots_ar_cancel(state, gpu, stdout, id, 0);
        return;
    }
    if let Some(e) = embed_err {
        dots_ar_fail(
            state,
            gpu,
            stdout,
            id,
            &format!("dots.ocr prefill embed build failed: {e}"),
        );
        return;
    }
    if let Err(e) =
        qwen2::forward_prefill_batch_embeds(gpu, &weights.text, &text_cfg, state, &embeds)
    {
        dots_ar_fail(
            state,
            gpu,
            stdout,
            id,
            &format!("dots.ocr batched prefill failed: {e:?}"),
        );
        return;
    }
    if check_abort(id) {
        dots_ar_cancel(state, gpu, stdout, id, 0);
        return;
    }
    // Test-only fault point (G4.8 lifecycle evidence), armed in-process via
    // `common::arm_dots_fault`: `"prefill"` fails this request through the
    // fail-closed error epilogue once prefill has committed decoder state;
    // `"argmax"` / `"decode"` fail at those seams. Unarmed => no behavior change.
    let dots_fault: Option<&'static str> = crate::common::peek_dots_fault();
    if dots_fault == Some("prefill") {
        dots_ar_fail(
            state,
            gpu,
            stdout,
            id,
            "dots.ocr batched prefill failed: injected fault (injected fault: prefill)",
        );
        return;
    }
    let prefill_tokens = prompt_ids.len();
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    // 6. Decode. Opt-in n-gram speculative decode when a speculator was built at
    // load (HIPFIRE_NGRAM_DRAFT=1, arch_id=8 gate in `spec_build`); else the
    // bespoke greedy AR loop below. The vision prefill above already advanced the
    // dots-ocr Qwen2 state (`ModelState::DotsOcr`), so both paths decode from the
    // same warm state — only the drafting differs. The n-gram verify always falls back to
    // the target's greedy argmax, so spec output is byte-identical to AR; only τ
    // (speed) changes. The prefill bindings above (`tokenizer`/`config`/`state`/…)
    // are released here so the speculative branch can take `&mut m`; the AR path
    // re-borrows them below.
    if m.speculator.is_some() {
        decode_vl_dots_ocr_ngram(
            m,
            gpu,
            stdout,
            id,
            &prompt_ids,
            max_tokens,
            t0,
            prefill_tokens,
            prefill_s,
        );
        return;
    }
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let config = m.dots_ocr().unwrap().config.clone();
    let text_cfg = config.text.clone();
    let bundle_ptr: *mut hipfire_arch_dots_ocr::DotsOcrBundle =
        match m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_dots_ocr::DotsOcrBundle>()
        }) {
            Some(b) => b as *mut _,
            None => unreachable!(),
        };
    let weights = unsafe { &(*bundle_ptr).weights };
    let state = unsafe { &mut (*bundle_ptr).state };
    // Greedy decode, streaming in the daemon JSONL protocol.
    let eos_set: Vec<u32> = if text_cfg.eos_token_ids.is_empty() {
        vec![text_cfg.eos_token_id]
    } else {
        text_cfg.eos_token_ids.clone()
    };
    if dots_fault == Some("argmax") {
        dots_ar_fail(
            state,
            gpu,
            stdout,
            id,
            "dots.ocr argmax failed: injected fault (injected fault: argmax)",
        );
        return;
    }
    let mut next = match gpu.argmax_f32(&state.logits, text_cfg.vocab_size) {
        Ok(t) => t,
        Err(e) => {
            dots_ar_fail(
                state,
                gpu,
                stdout,
                id,
                &format!("dots.ocr argmax failed: {e:?}"),
            );
            return;
        }
    };
    let t_gen = Instant::now();
    let mut streamed: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    let mut generated = 0usize;
    // No ngram loop-guard here: dots.ocr layout-JSON legitimately repeats
    // short structures (`<td>…</td>`, `"category":`, bracket patterns), and
    // the default guard force-stops mid-table (observed: truncation at 391
    // tokens on a table-heavy page). The proven ocr_e2e path decodes
    // straight to EOS without a guard; see DotsOcr::loop_guard_overrides.

    let mut decoded_eot = false;
    while generated < max_tokens {
        if check_abort(id) {
            dots_ar_cancel(state, gpu, stdout, id, generated);
            return;
        }
        if dots_fault == Some("decode") {
            dots_ar_fail(
                state,
                gpu,
                stdout,
                id,
                "dots.ocr decode failed: injected fault (injected fault: decode)",
            );
            return;
        }
        if eos_set.contains(&next) {
            decoded_eot = true;
            break;
        }
        emit_committed_event(stdout, id, next, generated, t0.elapsed().as_millis() as u64);
        generated += 1;
        streamed.push(next);

        // Incremental UTF-8 streaming — only emit complete code points.
        let all_bytes = tokenizer.decode_bytes(&streamed);
        let new_bytes = &all_bytes[emitted_bytes..];
        let valid_len = match std::str::from_utf8(new_bytes) {
            Ok(_) => new_bytes.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_len > 0 {
            let text = std::str::from_utf8(&new_bytes[..valid_len]).unwrap();
            let _ = writeln!(
                stdout,
                r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                id,
                serde_json::to_string(&text).unwrap_or_default(),
                active_attempt_id()
            );
            let _ = stdout.flush();
            emitted_bytes += valid_len;
        }

        match qwen2::forward_step_greedy(gpu, &weights.text, &text_cfg, state, next) {
            Ok(t) => next = t,
            Err(e) => {
                dots_ar_fail(
                    state,
                    gpu,
                    stdout,
                    id,
                    &format!("dots.ocr decode failed: {e:?}"),
                );
                return;
            }
        }
    }

    if check_abort(id) {
        dots_ar_cancel(state, gpu, stdout, id, generated);
        return;
    }

    let decode_s = t_gen.elapsed().as_secs_f64();
    let total_s = t0.elapsed().as_secs_f64();
    let tok_s = if total_s > 0.0 {
        generated as f64 / total_s
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_s > 0.0 {
        prefill_tokens as f64 / prefill_s
    } else {
        0.0
    };
    let decode_tok_s = if decode_s > 0.0 {
        generated as f64 / decode_s
    } else {
        0.0
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (tok_s * 10.0).round() / 10.0,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "ttft_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "attempt_id": active_attempt_id(),
        "finish_reason": crate::common::length_or_stop(generated, max_tokens, decoded_eot),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => {
            crate::ar::emit_active_route_done_value(stdout, &pending_done)
        }
        ClientTerminalDecision::Abort => {
            crate::ar::emit_active_route_cancel(stdout, id, generated);
        }
    }
}

pub fn decode_vl_dots_ocr_ngram(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    max_tokens: usize,
    t0: Instant,
    prefill_tokens: usize,
    prefill_s: f64,
) {
    use hipfire_arch_dots_ocr::DotsOcrBundle;
    // Move the live decoder state into a SpecTarget bundle; restored on return.
    let mut bundle = *(m.state.take().unwrap() as Box<dyn std::any::Any>)
        .downcast::<DotsOcrBundle>()
        .unwrap();
    let mut spec = m.speculator.take().unwrap();
    // `m.tokenizer` is a disjoint field → coexists with the takes above and the
    // restore below; the loop never touches `m`.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    run_dots_ocr_ngram_loop(
        &mut bundle,
        spec.as_mut(),
        tokenizer,
        gpu,
        stdout,
        id,
        prompt_ids,
        max_tokens,
        t0,
        prefill_tokens,
        prefill_s,
    );
    m.state = Some(Box::new(bundle));
    m.speculator = Some(spec);
}
pub fn run_dots_ocr_ngram_loop(
    bundle: &mut hipfire_arch_dots_ocr::DotsOcrBundle,
    spec: &mut dyn hipfire_runtime::spec::Speculator,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    max_tokens: usize,
    t0: Instant,
    prefill_tokens: usize,
    prefill_s: f64,
) {
    let eos_set: Vec<u32> = if bundle.config.text.eos_token_ids.is_empty() {
        vec![bundle.config.text.eos_token_id]
    } else {
        bundle.config.text.eos_token_ids.clone()
    };
    let block_size = spec.block_size();
    let ctx_capacity = spec.ctx_capacity();

    // Prime the n-gram drafter + fetch the first token WITHOUT re-running the
    // (vision-conditioned) target prefill. `cache_hit=true` + an empty suffix
    // makes `ChainSpeculator::prefill` skip the target advance —
    // `spec_advance(&[], prompt_len, reset=false)` just argmaxes the live
    // post-vision-prefill logits — and only `drafter.prefill_seed(prompt_ids)`.
    // It also lazily builds the verify scratch (required before the first `step`).
    let first_token = match spec.prefill(
        gpu,
        bundle,
        prompt_ids,
        &[],
        prompt_ids.len(),
        true,
        None,
        &|| check_abort(id),
    ) {
        Ok(PrefillOutcome::Ready { first_token }) => first_token,
        Ok(PrefillOutcome::Aborted) => {
            // Client cancel during n-gram prefill: roll back drafter/target
            // state, then the cancelled pair (no success done / commit_ready).
            dots_spec_cancel(bundle, spec, gpu, stdout, id, 0);
            return;
        }
        Err(e) => {
            dots_spec_fail(
                bundle,
                spec,
                gpu,
                stdout,
                id,
                &format!("dots.ocr spec prefill: {e}"),
            );
            return;
        }
    };

    let t_gen = Instant::now();
    let mut streamed: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    let mut generated = 0usize;
    // n-gram context (committed generated tail; the drafter holds the prompt
    // internally via prefill_seed).
    let mut emitted: Vec<u32> = Vec::new();
    let mut position = prompt_ids.len();
    let mut seed_token = first_token;
    // τ accounting (accepted drafts / windows) — mirrors the text spec path so
    // the done envelope reports acceptance for diagnosing spec-vs-AR perf.
    let mut spec_cycles = 0usize;
    let mut spec_accepted = 0usize;
    // Tokens to stream this iteration. First window = the prefill seed alone
    // (mirrors the AR loop emitting the first argmax), then the accepted
    // committed tail from each `spec.step` (seed re-echo already stripped).
    let mut window: Vec<u32> = vec![first_token];

    let mut decoded_eot = false;
    // The context ran out before the budget did: still a truncation.
    let mut ctx_exhausted = false;
    'outer: loop {
        for &tok in &window {
            if generated >= max_tokens {
                break 'outer;
            }
            // EOS is never streamed (matches the AR loop's pre-emit break).
            if eos_set.contains(&tok) {
                decoded_eot = true;
                break 'outer;
            }
            emit_committed_event(stdout, id, tok, generated, t0.elapsed().as_millis() as u64);
            generated += 1;
            streamed.push(tok);
            emitted.push(tok);
            // Incremental UTF-8 streaming — only emit complete code points
            // (byte-identical to the AR path).
            let all_bytes = tokenizer.decode_bytes(&streamed);
            let new_bytes = &all_bytes[emitted_bytes..];
            let valid_len = match std::str::from_utf8(new_bytes) {
                Ok(_) => new_bytes.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid_len > 0 {
                let text = std::str::from_utf8(&new_bytes[..valid_len]).unwrap();
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                    id,
                    serde_json::to_string(&text).unwrap_or_default(),
                    active_attempt_id()
                );
                let _ = stdout.flush();
                emitted_bytes += valid_len;
            }
        }
        if generated >= max_tokens {
            break;
        }
        // Decode-side cancel: roll back drafter/target state, emit the
        // cancelled pair and stop — falling through to the done handshake on
        // an aborted attempt strands the serve admission guard (2026-08-27
        // ledger finding-c class; same rule as the prefill-cancel site above).
        if check_abort(id) {
            dots_spec_cancel(bundle, spec, gpu, stdout, id, generated);
            return;
        }
        // Context-overflow guard (matches generate_spec): one window writes up
        // to `block_size` KV slots.
        if position.saturating_add(block_size) >= ctx_capacity {
            ctx_exhausted = true;
            break;
        }
        let max_emit = max_tokens.saturating_sub(generated);
        // Test-only fault point (G4.8 lifecycle evidence), armed in-process via
        // `common::arm_dots_fault("spec")`: fails through the same fail-closed
        // path as a real verify failure. Unarmed => no behavior change.
        if crate::common::peek_dots_fault() == Some("spec") {
            dots_spec_fail(
                bundle,
                spec,
                gpu,
                stdout,
                id,
                "dots.ocr spec_step: injected fault (injected fault: spec)",
            );
            return;
        }
        let step = match spec.step(
            gpu, bundle, position, seed_token, &emitted, None, 0.0, max_emit,
        ) {
            Ok(s) => s,
            Err(e) => {
                // Fail closed: roll back and return with the error terminal.
                // The old `break` fell through to the success `done`
                // handshake below, emitting error + done for one request.
                dots_spec_fail(
                    bundle,
                    spec,
                    gpu,
                    stdout,
                    id,
                    &format!("dots.ocr spec_step: {e}"),
                );
                return;
            }
        };
        spec_cycles += 1;
        spec_accepted += step.accepted;
        // Advance by the emitted-tail length (= accepted + 1), per the spec.rs
        // `emit_len_drives_advance` contract; the target already wrote KV for the
        // whole tail in `verify_block`.
        position += step.emit.len();
        seed_token = step.next_seed;
        window.clear();
        window.extend_from_slice(&step.emit);
    }

    let decode_s = t_gen.elapsed().as_secs_f64();
    let total_s = t0.elapsed().as_secs_f64();
    let tok_s = if total_s > 0.0 {
        generated as f64 / total_s
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_s > 0.0 {
        prefill_tokens as f64 / prefill_s
    } else {
        0.0
    };
    let decode_tok_s = if decode_s > 0.0 {
        generated as f64 / decode_s
    } else {
        0.0
    };
    let tau = if spec_cycles > 0 {
        spec_accepted as f64 / spec_cycles as f64
    } else {
        0.0
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (tok_s * 10.0).round() / 10.0,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "ttft_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "dflash": true,
        "tau": (tau * 100.0).round() / 100.0,
        "cycles": spec_cycles,
        "attempt_id": active_attempt_id(),
        "finish_reason": if ctx_exhausted {
            "length"
        } else {
            crate::common::length_or_stop(generated, max_tokens, decoded_eot)
        },
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => {
            crate::ar::emit_active_route_done_value(stdout, &pending_done)
        }
        ClientTerminalDecision::Abort => {
            emit_aborted_terminal_after_abort(stdout, id, generated);
        }
    }
}

pub fn generate_dots_ocr_text(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    _system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
) {
    let route = crate::ar::GenerationRoute::DotsOcr;
    let _route_scope = crate::ar::GenerationRouteScope::enter(route, id);
    crate::ar::emit_generation_start(route, stdout, id, false);
    let _ = (temp, top_p); // greedy decode for now; sampling left for future work
    let t0 = Instant::now();

    let max_seq = m.max_seq;

    // Model state (disjoint field borrows of `m`).
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let config = m.dots_ocr().unwrap().config.clone();
    let text_cfg = config.text.clone();
    let dim = text_cfg.hidden_size;
    // Weights/state via raw pointers to allow owned config while keeping disjoint borrows.
    let bundle_ptr: *mut hipfire_arch_dots_ocr::DotsOcrBundle =
        match m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_dots_ocr::DotsOcrBundle>()
        }) {
            Some(b) => b as *mut _,
            None => unreachable!(),
        };
    let weights = unsafe { &(*bundle_ptr).weights };
    let state = unsafe { &mut (*bundle_ptr).state };
    // Tokenize the text prompt directly (no image tokens).
    let prompt_ids = tokenizer.encode(prompt);
    if prompt_ids.len().saturating_add(max_tokens) > max_seq {
        write_error(stdout, id, &format!(
            "dots.ocr text request ({} prompt + {} gen) exceeds KV budget ({}); reload with a larger --max-seq",
            prompt_ids.len(), max_tokens, max_seq));
        return;
    }

    // Prefill: build the [seq × dim] embedding matrix via per-token
    // embedding lookup dispatch, then run through batched prefill.
    state.reset();
    let t_prefill = Instant::now();
    let mut embeds = vec![0f32; prompt_ids.len() * dim];
    let emb_scratch = match gpu.alloc_tensor(&[dim], rdna_compute::DType::F32) {
        Ok(t) => t,
        Err(e) => {
            dots_ar_fail(
                state,
                gpu,
                stdout,
                id,
                &format!("dots.ocr embed scratch alloc failed: {e:?}"),
            );
            return;
        }
    };
    let mut embed_err: Option<String> = None;
    for (pos, &token) in prompt_ids.iter().enumerate() {
        // Cancel poll (parity with the image-conditioned AR loop): abort
        // during the embedding build rolls back before any token is live.
        if check_abort(id) {
            let _ = gpu.free_tensor(emb_scratch);
            dots_ar_cancel(state, gpu, stdout, id, 0);
            return;
        }
        let lookup = hipfire_runtime::llama::embedding_lookup_dispatch(
            gpu,
            weights.text.embd_format,
            &weights.text.token_embd,
            &emb_scratch,
            token,
            dim,
        );
        if let Err(e) = lookup {
            embed_err = Some(format!("embedding lookup: {e:?}"));
            break;
        }
        match gpu.download_f32(&emb_scratch) {
            Ok(row) => embeds[pos * dim..(pos + 1) * dim].copy_from_slice(&row),
            Err(e) => {
                embed_err = Some(format!("embedding download: {e:?}"));
                break;
            }
        }
    }
    let _ = gpu.free_tensor(emb_scratch);
    if let Some(e) = embed_err {
        dots_ar_fail(
            state,
            gpu,
            stdout,
            id,
            &format!("dots.ocr prefill embed build failed: {e}"),
        );
        return;
    }
    if let Err(e) =
        qwen2::forward_prefill_batch_embeds(gpu, &weights.text, &text_cfg, state, &embeds)
    {
        dots_ar_fail(
            state,
            gpu,
            stdout,
            id,
            &format!("dots.ocr batched prefill failed: {e:?}"),
        );
        return;
    }
    if check_abort(id) {
        dots_ar_cancel(state, gpu, stdout, id, 0);
        return;
    }
    let prefill_tokens = prompt_ids.len();
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    // Greedy decode, streaming in the daemon JSONL protocol.
    let eos_set: Vec<u32> = if text_cfg.eos_token_ids.is_empty() {
        vec![text_cfg.eos_token_id]
    } else {
        text_cfg.eos_token_ids.clone()
    };
    let mut next = match gpu.argmax_f32(&state.logits, text_cfg.vocab_size) {
        Ok(t) => t,
        Err(e) => {
            dots_ar_fail(
                state,
                gpu,
                stdout,
                id,
                &format!("dots.ocr argmax failed: {e:?}"),
            );
            return;
        }
    };
    let t_gen = Instant::now();
    let mut streamed: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    let mut generated = 0usize;

    let mut decoded_eot = false;
    while generated < max_tokens {
        // Cancel poll: the image path has one per iteration; the text path
        // previously ran to `max_tokens` uninterruptibly.
        if check_abort(id) {
            dots_ar_cancel(state, gpu, stdout, id, generated);
            return;
        }
        if eos_set.contains(&next) {
            decoded_eot = true;
            break;
        }
        emit_committed_event(stdout, id, next, generated, t0.elapsed().as_millis() as u64);
        generated += 1;
        streamed.push(next);

        // Incremental UTF-8 streaming — only emit complete code points.
        let all_bytes = tokenizer.decode_bytes(&streamed);
        let new_bytes = &all_bytes[emitted_bytes..];
        let valid_len = match std::str::from_utf8(new_bytes) {
            Ok(_) => new_bytes.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_len > 0 {
            let text = std::str::from_utf8(&new_bytes[..valid_len]).unwrap();
            let _ = writeln!(
                stdout,
                r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                id,
                serde_json::to_string(&text).unwrap_or_default(),
                active_attempt_id()
            );
            let _ = stdout.flush();
            emitted_bytes += valid_len;
        }

        match qwen2::forward_step_greedy(gpu, &weights.text, &text_cfg, state, next) {
            Ok(t) => next = t,
            Err(e) => {
                dots_ar_fail(
                    state,
                    gpu,
                    stdout,
                    id,
                    &format!("dots.ocr decode failed: {e:?}"),
                );
                return;
            }
        }
    }

    // Post-loop abort latch (parity with the image path): a disconnect during
    // the final iteration must cancel, never fall through to `done`.
    if check_abort(id) {
        dots_ar_cancel(state, gpu, stdout, id, generated);
        return;
    }

    let decode_s = t_gen.elapsed().as_secs_f64();
    let total_s = t0.elapsed().as_secs_f64();
    let tok_s = if total_s > 0.0 {
        generated as f64 / total_s
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_s > 0.0 {
        prefill_tokens as f64 / prefill_s
    } else {
        0.0
    };
    let decode_tok_s = if decode_s > 0.0 {
        generated as f64 / decode_s
    } else {
        0.0
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (tok_s * 10.0).round() / 10.0,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "ttft_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "attempt_id": active_attempt_id(),
        "finish_reason": crate::common::length_or_stop(generated, max_tokens, decoded_eot),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => {
            crate::ar::emit_active_route_done_value(stdout, &pending_done)
        }
        ClientTerminalDecision::Abort => {
            emit_aborted_terminal_after_abort(stdout, id, generated);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{vl_finish_think_routing, vl_route_decode_text};
    use hipfire_runtime::emit_text::ThinkOutputRouter;
    use hipfire_runtime::eos_filter::EosFilter;

    /// Drive byte chunks through the VL typed-emission pipeline and collect
    /// (channel, text) wire events parsed back from the JSONL lines.
    fn drive(started_in_think: bool, chunks: &[&[u8]]) -> Vec<(String, String)> {
        let mut filter = EosFilter::new(hipfire_runtime::eos_filter::qwen35_eos_filter_config());
        let mut think = ThinkOutputRouter::new(started_in_think);
        let mut out = Vec::new();
        for chunk in chunks {
            let _ = vl_route_decode_text(&mut out, "t-id", &mut filter, &mut think, chunk);
        }
        vl_finish_think_routing(&mut out, "t-id", &mut filter, &mut think);
        let out = String::from_utf8_lossy(&out);
        out.lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .map(|v| {
                (
                    v.get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    v.get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            })
            .collect()
    }

    fn joined(events: &[(String, String)], channel: &str) -> String {
        events
            .iter()
            .filter(|(t, _)| t == channel)
            .map(|(_, s)| s.as_str())
            .collect()
    }

    #[test]
    fn thinking_turn_splits_reasoning_from_content() {
        // OpenThink prefix: generation starts inside think (no opener in the
        // generated bytes); closer flips to content; trailing EOT suppressed.
        let events = drive(
            true,
            &[
                b"The user wants a description.\n\n",
                b"</think>\n\n",
                b"A Shiba Inu dog naps on its back.",
                b"<|im_end|>",
            ],
        );
        assert_eq!(
            joined(&events, "reasoning"),
            "The user wants a description.\n\n"
        );
        assert_eq!(
            joined(&events, "token"),
            "A Shiba Inu dog naps on its back."
        );
        let all = events.iter().map(|(_, s)| s.clone()).collect::<String>();
        assert!(!all.contains("</think>"), "closer leaked: {all:?}");
        assert!(!all.contains("<|im_end|>"), "EOT leaked: {all:?}");
    }

    #[test]
    fn marker_split_across_chunks_still_routes() {
        // `</thi` + `nk>` arriving as separate decode deltas (token boundary)
        // must not leak a partial marker as visible text.
        let events = drive(true, &[b"plan</thi", b"nk>\nanswer", b"<|im_end|>"]);
        assert_eq!(joined(&events, "reasoning"), "plan");
        assert_eq!(joined(&events, "token"), "answer");
    }

    #[test]
    fn plain_prefix_spontaneous_think_markers_route() {
        // assistant_prefix=Plain and the model opens think itself.
        let events = drive(
            false,
            &[
                b"<think>private</think>",
                b"\n\nvisible answer",
                b"<|im_end|>",
            ],
        );
        assert_eq!(joined(&events, "reasoning"), "private");
        assert_eq!(joined(&events, "token"), "visible answer");
    }

    #[test]
    fn no_think_turn_is_pure_content() {
        let events = drive(false, &[b"Pointy teeths wow.", b"<|im_end|>"]);
        assert!(joined(&events, "reasoning").is_empty());
        assert_eq!(joined(&events, "token"), "Pointy teeths wow.");
    }

    #[test]
    fn hostile_request_id_stays_json_escaped() {
        // The old hand-rolled envelope spliced `id` into the JSON unescaped;
        // the typed emitters must survive a quote-bearing id.
        let mut filter = EosFilter::new(hipfire_runtime::eos_filter::qwen35_eos_filter_config());
        let mut think = ThinkOutputRouter::new(false);
        let mut out = Vec::new();
        let _ = vl_route_decode_text(&mut out, "bad\"id\\", &mut filter, &mut think, b"text");
        vl_finish_think_routing(&mut out, "bad\"id\\", &mut filter, &mut think);
        let out = String::from_utf8_lossy(&out);
        for line in out.lines() {
            assert!(
                serde_json::from_str::<serde_json::Value>(line).is_ok(),
                "{line}"
            );
        }
    }

    #[test]
    fn ordinary_emit_returns_false() {
        let mut filter = EosFilter::new(hipfire_runtime::eos_filter::qwen35_eos_filter_config());
        let mut think = ThinkOutputRouter::new(false);
        let mut out = Vec::new();
        let stopped = vl_route_decode_text(&mut out, "t-id", &mut filter, &mut think, b"hello");
        assert!(!stopped, "ordinary emit should return false");
        vl_finish_think_routing(&mut out, "t-id", &mut filter, &mut think);
        let out_s = String::from_utf8_lossy(&out);
        assert!(out_s.contains("hello"), "payload should emit: {out_s:?}");
    }

    #[test]
    fn stop_marker_returns_true_and_emits_preceding_text_without_leakage() {
        let mut filter = EosFilter::new(hipfire_runtime::eos_filter::qwen35_eos_filter_config());
        let mut think = ThinkOutputRouter::new(false);
        let mut out = Vec::new();
        let payload = b"visible answer<|im_end|>";
        let stopped = vl_route_decode_text(&mut out, "t-id", &mut filter, &mut think, payload);
        assert!(stopped, "EOT marker should signal stop");
        vl_finish_think_routing(&mut out, "t-id", &mut filter, &mut think);
        let out_s = String::from_utf8_lossy(&out);
        assert!(
            out_s.contains("visible answer"),
            "preceding text should emit: {out_s:?}"
        );
        assert!(
            !out_s.contains("<|im_end|>"),
            "marker must not leak: {out_s:?}"
        );
    }

    #[test]
    fn trailing_held_prefix_flushed_at_finish() {
        // ` <` is a partial prefix of `<|im_end|>`; EosFilter holds it mid-stream
        // and only at finish should it be treated as ordinary prose and routed
        // through ThinkOutputRouter.
        let mut filter = EosFilter::new(hipfire_runtime::eos_filter::qwen35_eos_filter_config());
        let mut think = ThinkOutputRouter::new(false);
        let mut out = Vec::new();
        let stopped = vl_route_decode_text(&mut out, "t-id", &mut filter, &mut think, b"hello <");
        assert!(!stopped);
        // Before finish, the held `<` should not yet be emitted.
        let mid = String::from_utf8_lossy(&out).to_string();
        let mid_events: Vec<(String, String)> = mid
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .map(|v| {
                (
                    v.get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    v.get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            })
            .collect();
        let mid_text = joined(&mid_events, "token");
        assert_eq!(
            mid_text, "hello ",
            "partial prefix should be held, not emitted yet: {mid_text:?}"
        );
        vl_finish_think_routing(&mut out, "t-id", &mut filter, &mut think);
        let out_s = String::from_utf8_lossy(&out);
        let events: Vec<(String, String)> = out_s
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .map(|v| {
                (
                    v.get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    v.get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            })
            .collect();
        assert_eq!(
            joined(&events, "token"),
            "hello <",
            "held prefix should flush at finish"
        );
    }
}

// ── LFM2-VL (arch 11) ────────────────────────────────────────────────────────
//
// SigLIP2-NaFlex tower + projector executed by `hipfire_arch_lfm2_vl`; the
// text loop, stream contract, sampling, and terminal handshake are copied
// from `dense::generate_lfm2moe` so an image turn behaves exactly like a
// long-prompted text turn once the visual features are spliced. See
// docs/specs/2026-08-27-lfm2-vl-vision-runtime.md.

use hipfire_arch_lfm2_vl as lfm2vl;

/// Expand the single `<image>` id in `prompt_ids` into the marker structure
/// HF's processor produces (narrow spec §1.5): `<|image_start|>`,
/// per-tile `<|img_row_R_col_C|>` + 256 placeholders, `<|img_thumbnail|>` +
/// thumbnail placeholders, `<|image_end|>` (markers only when present in
/// this tokenizer; multi-tile REQUIRES its markers and fails closed).
fn expand_image_placeholders(
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    prompt_ids: &[u32],
    image_token_id: u32,
    prepared: &lfm2vl::Prepared,
    cfg: &lfm2vl::VisionConfig,
) -> Result<Vec<u32>, String> {
    let start_count = prompt_ids.iter().filter(|&&t| t == image_token_id).count();
    if start_count != 1 {
        return Err(format!(
            "expected exactly one <image> token in rendered prompt, found {start_count} — \
             the artifact tokenizer is missing the VL added-token; requantize or use a \
             VL-capable tokenizer"
        ));
    }
    let special = |s: &str| -> Option<u32> { tokenizer.special_token_id(s) };
    let multi_tile = prepared.grid_rows > 1 || prepared.grid_cols > 1;

    let mut body: Vec<u32> = Vec::new();
    let start_id = special("<|image_start|>");
    let end_id = special("<|image_end|>");

    let tiles: Vec<(usize, usize, usize)> = if multi_tile {
        // row-major tile list aligned with Prepared.sub_images ordering.
        prepared
            .sub_images
            .iter()
            .take(prepared.grid_rows * prepared.grid_cols)
            .enumerate()
            .map(|(i, s)| {
                (
                    i / prepared.grid_cols + 1,
                    i % prepared.grid_cols + 1,
                    cfg.tokens_for_grid(s.gh(cfg), s.gw(cfg)),
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    // HF's processor ALWAYS wraps the placeholder run in <|image_start|> …
    // <|image_end|> (single-tile included — pinned source §1.5), so both
    // markers are unconditionally required from the tokenizer.
    body.push(start_id.ok_or("tokenizer missing <|image_start|>")?);

    if multi_tile {
        for &(row, col, ntok) in &tiles {
            let marker = special(&format!("<|img_row_{row}_col_{col}|>"))
                .ok_or_else(|| format!("tokenizer missing <|img_row_{row}_col_{col}|>"))?;
            body.push(marker);
            for _ in 0..ntok {
                body.push(image_token_id);
            }
        }
        if cfg.use_thumbnail {
            let thumb_marker = special("<|img_thumbnail|>")
                .ok_or("multi-tile image requires <|img_thumbnail|> but tokenizer lacks it")?;
            body.push(thumb_marker);
            for _ in 0..cfg.tokens_for_grid(
                prepared.sub_images.last().unwrap().gh(cfg),
                prepared.sub_images.last().unwrap().gw(cfg),
            ) {
                body.push(image_token_id);
            }
        }
    } else {
        for _ in 0..cfg.tokens_for_grid(
            prepared.sub_images[0].gh(cfg),
            prepared.sub_images[0].gw(cfg),
        ) {
            body.push(image_token_id);
        }
    }
    body.push(end_id.ok_or("tokenizer missing <|image_end|>")?);

    // splice over the single <image>
    let idx = prompt_ids
        .iter()
        .position(|&t| t == image_token_id)
        .expect("position checked above");
    let mut out = Vec::with_capacity(prompt_ids.len() + body.len() - 1);
    out.extend_from_slice(&prompt_ids[..idx]);
    out.extend_from_slice(&body);
    out.extend_from_slice(&prompt_ids[idx + 1..]);

    let total = out.iter().filter(|&&t| t == image_token_id).count();
    if total != prepared.total_tokens(cfg) {
        return Err(format!(
            "placeholder/feature mismatch after expansion: {total} placeholders vs {} projected tokens",
            prepared.total_tokens(cfg)
        ));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_lfm2_vl(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) {
    let route = crate::ar::GenerationRoute::LfmAr;
    let _route_scope = crate::ar::GenerationRouteScope::enter(route, params.id);
    // Stream contract opener must precede every validation result, including
    // tokenizer and vision-capability errors.
    crate::ar::emit_generation_start(route, stdout, params.id, false);
    let GenerateVLParams {
        id,
        prompt,
        system_prompt,
        temp,
        top_p,
        max_tokens,
        max_think_tokens,
        seed,
        ..
    } = *params;
    if m.tokenizer.is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "tokenizer not loaded",
            "validation",
            false,
            false,
        );
        return;
    }
    let vision_cfg = match m.lfm2_vision() {
        Some((vc, _)) => vc.clone(),
        None => {
            emit_active_attempt_error(
                stdout,
                Some(id),
                "model has no vision encoder",
                "validation",
                false,
                false,
            );
            return;
        }
    };

    // Full-turn clock: preprocess + tower encode + prefill + decode. The
    // tower dominates image turns (~7–10 s of the ~22 s wall on gfx1101),
    // so a total that excludes it would misreport the turn.
    let t_turn = Instant::now();

    // ── Preprocess (CPU decode + resize/split/thumbnail) ──
    let image_source = match params.single_image() {
        Ok(source) => source,
        Err(e) => {
            emit_active_attempt_error(stdout, Some(id), &e, "validation", false, false);
            return;
        }
    };
    let prepared = match image_source {
        ImageSource::Path(path) => {
            lfm2vl::load_and_preprocess(std::path::Path::new(path), &vision_cfg)
        }
        ImageSource::Base64(b64) => {
            let raw_b64 = if let Some(rest) = b64.strip_prefix("data:") {
                match rest.split_once(',') {
                    Some((_, after)) => after,
                    None => {
                        emit_active_attempt_error(
                            stdout,
                            Some(id),
                            "malformed data URL: missing ',' separator",
                            "validation",
                            false,
                            false,
                        );
                        return;
                    }
                }
            } else {
                b64
            };
            match Engine::decode(&base64::engine::general_purpose::STANDARD, raw_b64) {
                Ok(bytes) => lfm2vl::load_and_preprocess_from_bytes(&bytes, &vision_cfg),
                Err(e) => Err(format!("failed to decode base64 image data: {e}")),
            }
        }
    };
    let prepared = match prepared {
        Ok(p) => p,
        Err(e) => {
            emit_active_attempt_error(stdout, Some(id), &e, "validation", false, false);
            return;
        }
    };
    let n_visual_tokens = prepared.total_tokens(&vision_cfg);
    eprintln!(
        "[daemon/vl-lfm2] image preprocessed: {} sub-image(s), {}x{} grid(s), {} visual tokens",
        prepared.sub_images.len(),
        prepared.grid_rows,
        prepared.grid_cols,
        n_visual_tokens
    );

    // ── Prompt build: normal ChatML jinja frame with a literal <image> ──
    let image_token_id = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        match tokenizer.special_token_id("<image>") {
            Some(t) => t,
            None => {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    "tokenizer has no <image> token — not a VL tokenizer",
                    "validation",
                    false,
                    false,
                );
                return;
            }
        }
    };
    // Same `primed_think` convention as the text path: emit the template's
    // think opener as display-only text so downstream reasoning/content
    // consumers see a well-formed think block.
    let mut primed_think = false;
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let template = m.chat_template.as_ref();
        let user_content = format!("<image>{prompt}");
        let rendered = match template {
            Some(template) => hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: &user_content,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
                reasoning_strength: None,
                reasoning_effort: None,
            }
            .render(),
            None => Err("no chat template embedded in artifact".to_string()),
        };
        match rendered {
            Ok(text) => {
                primed_think = hipfire_engine::emit::render_tail_opens_think(&text);
                tokenizer.encode(&text)
            }
            Err(e) => {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("vl prompt render failed: {e}"),
                    "validation",
                    false,
                    false,
                );
                return;
            }
        }
    };
    let prompt_ids = match expand_image_placeholders(
        m.tokenizer.as_ref().unwrap(),
        &prompt_ids,
        image_token_id,
        &prepared,
        &vision_cfg,
    ) {
        Ok(p) => p,
        Err(e) => {
            emit_active_attempt_error(stdout, Some(id), &e, "validation", false, false);
            return;
        }
    };

    // Capacity guard BEFORE GPU work. VL forces the cold-reset path (below),
    // so seq_pos is always 0 here.
    let cap = m.lfm2moe().unwrap().state.max_seq;
    if prompt_ids.len().saturating_add(max_tokens) > cap {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!(
                "prompt exceeds context capacity: prompt={} (incl. {} visual) + max_tokens={} > capacity={}",
                prompt_ids.len(),
                n_visual_tokens,
                max_tokens,
                cap
            ),
            "context_length",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    // Cross-conversation cold reset (mirrors dense::generate_lfm2moe).
    let _ = m.lfm2moe_mut().unwrap().state.reset(gpu);
    m.seq_pos = 0;
    m.conversation_tokens.clear();

    // ── Vision encode → projected [n_visual, hidden] rows ──
    let visual_tokens = match m.lfm2_vision() {
        Some((_, vw)) => lfm2vl::vision_forward(gpu, vw, &vision_cfg, &prepared),
        None => {
            write_error(stdout, id, "model has no vision encoder");
            return;
        }
    };
    let visual_tokens = match visual_tokens {
        Ok(v) => v,
        Err(e) => {
            write_error(stdout, id, &e);
            return;
        }
    };
    // Release-mode fail-closed length check (the narrow spec §2.3 claims
    // "fails loud BEFORE splicing"): debug_assert compiles out in release,
    // and a mismatch would otherwise panic the daemon on the emb slice below
    // instead of erroring the request. Unreachable while the splitter and
    // the tower share tokens_for_grid, but the splice must not trust that
    // by accident.
    if visual_tokens.len() != n_visual_tokens * vision_cfg.out_hidden_size {
        write_error(
            stdout,
            id,
            &format!(
                "vision/projector row mismatch: {} floats for {} tokens × {} dims",
                visual_tokens.len(),
                n_visual_tokens,
                vision_cfg.out_hidden_size
            ),
        );
        return;
    }

    // EOS-class stop set (verbatim rationale lives at the text path copy).
    let stop_toks = hipfire_arch_lfm2moe::carrier::stop_tokens(
        m.tokenizer.as_ref().unwrap(),
        m.lfm2moe().unwrap().eos_tok,
    );

    // ── Prefill with visual embeddings spliced at placeholder positions ──
    let mut last_logits: Vec<f32> = Vec::new();
    let prefill_t0 = Instant::now();
    {
        let b = m.lfm2moe_mut().unwrap();
        let cfg_t = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let dim = vision_cfg.out_hidden_size;
        let mut vis_idx = 0usize;
        let mut position = state.n_tokens as u32;
        for &tok in &prompt_ids {
            if check_abort(id) {
                // Client-cancel poll (mirrors the generate_vl hardening on
                // feat/qwen35-vl): the vision-encode phase before this loop
                // cannot be interrupted mid-kernel, so prefill is where an
                // abort signalled during the tower takes effect. Emit the
                // canonical cancelled pair and return — breaking out here
                // would fall through to the decode loop relying on its
                // top-of-loop abort check to avoid sampling empty logits,
                // and would push the full prompt into conversation_tokens
                // against a partially-filled KV.
                crate::ar::emit_active_route_cancel(stdout, id, 0);
                return;
            }
            let res = if tok == image_token_id && vis_idx < n_visual_tokens {
                let emb = &visual_tokens[vis_idx * dim..(vis_idx + 1) * dim];
                vis_idx += 1;
                hipfire_arch_lfm2moe::forward::prefill_embed_step(
                    cfg_t, weights, state, gpu, emb, position,
                )
            } else {
                hipfire_arch_lfm2moe::forward::decode_step(
                    cfg_t, weights, state, gpu, tok, position,
                )
            };
            match res {
                Ok(logits) => last_logits = logits,
                Err(e) => {
                    write_error(stdout, id, &format!("lfm2-vl prefill failed: {e:?}"));
                    return;
                }
            }
            position += 1;
        }
    }
    for &tok in &prompt_ids {
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = prefill_t0.elapsed().as_millis().max(1);

    // ── Decode loop (identical sampling/stop semantics to generate_lfm2moe) ──
    // Per-request sampler seed from hipfire-engine::request_seed_for (wire
    // `seed` wins, else attempt key + counter). Matches generate_vl / text
    // paths — never wall-clock entropy.
    let mut rng = hipfire_arch_deepseek4::sampling::Xorshift::new(seed as u64);

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    let mut text_stream = hipfire_runtime::tokenizer::TokenTextStream::new();

    // Same `primed_think` convention as the text path (generate_lfm2moe):
    // emit the template's `<think>` opener as display-only text so the
    // serve reasoning/content split sees a well-formed block.
    if primed_think {
        crate::dense::emit_dense_token_text(stdout, id, "\x3Cthink\x3E");
    }

    // Same tool-span handling as the text path: the LFM tool markers and
    // their body are suppressed from the stream and parsed into
    // `tool_calls` at the terminal; all tokens still reach KV + history.
    let tool_open_tok = m
        .tokenizer
        .as_ref()
        .and_then(|t| t.special_token_id("\x3C\x7Ctool_call_start\x7C\x3E"));
    let tool_close_tok = m
        .tokenizer
        .as_ref()
        .and_then(|t| t.special_token_id("\x3C\x7Ctool_call_end\x7C\x3E"));
    let mut in_tool_span = false;
    let mut tool_span_tokens: Vec<u32> = Vec::new();
    let mut tool_body_tokens: Vec<u32> = Vec::new();
    let mut tool_calls: Vec<hipfire_runtime::prompt_frame::ToolCall> = Vec::new();
    let mut decoded_eot = false;
    loop {
        if check_abort(id) {
            eprintln!("[daemon/vl-lfm2] aborted mid-decode");
            break;
        }
        if generated_count >= max_tokens {
            break;
        }
        let next_tok =
            hipfire_arch_deepseek4::sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng);
        if stop_toks.contains(&next_tok) {
            decoded_eot = true;
            break;
        }
        let tokenizer = m.tokenizer.as_ref().unwrap();
        // Turn-end markers are checked on the token's own text.
        if matches!(
            tokenizer.decode(&[next_tok]).trim(),
            "<|endoftext|>" | "</s>" | "<|im_end|>" | "<|startoftext|>"
        ) {
            decoded_eot = true;
            break;
        }
        // Tool-call span handling — suppresses marker + body tokens from the
        // stream; everything still enters conversation_tokens and the decode
        // step (mirrors dense::generate_lfm2moe).
        let in_span_boundary = if tool_open_tok.is_some_and(|t| next_tok == t) && !in_tool_span {
            in_tool_span = true;
            tool_span_tokens.clear();
            tool_span_tokens.push(next_tok);
            tool_body_tokens.clear();
            true
        } else if in_tool_span {
            tool_span_tokens.push(next_tok);
            if tool_close_tok.is_some_and(|t| next_tok == t) {
                in_tool_span = false;
                let body_text = tokenizer.decode(&tool_body_tokens);
                tool_calls.extend(hipfire_runtime::emit_text::parse_lfm_tool_call_body(
                    &body_text,
                ));
            } else {
                tool_body_tokens.push(next_tok);
            }
            true
        } else {
            false
        };
        if !in_span_boundary {
            // Multi-byte characters span tokens: emit only completed text.
            crate::dense::emit_dense_token_text(stdout, id, &text_stream.push(tokenizer, next_tok));
        }
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        if check_abort(id) {
            break;
        }
        let step = {
            let b = m.lfm2moe_mut().unwrap();
            let position = b.state.n_tokens as u32;
            hipfire_arch_lfm2moe::forward::decode_step(
                &b.config,
                &b.weights,
                &mut b.state,
                gpu,
                next_tok,
                position,
            )
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                write_error(stdout, id, &format!("lfm2-vl decode failed: {e:?}"));
                return;
            }
        }
    }
    crate::dense::emit_dense_token_text(stdout, id, &text_stream.flush());
    // Unclosed tool span: re-emit the raw text (treat-as-text fallback);
    // a truncated body never becomes a call.
    if in_tool_span {
        crate::dense::emit_dense_token_text(
            stdout,
            id,
            &m.tokenizer.as_ref().unwrap().decode(&tool_span_tokens),
        );
    }
    let decode_ms = decode_t0.elapsed().as_millis().max(1);

    // Post-loop abort latch — MANDATORY before the two-phase commit
    // handshake. If the client cancelled/disconnected mid-decode,
    // `await_client_terminal_commit` would block forever waiting for a
    // commit that can never arrive and wedge the single slot (the exact
    // failure recorded in the 2026-08-27 serve ledger). Emits the CANONICAL
    // cancelled-terminal pair via `emit_active_route_cancel` (wire `aborted` +
    // `aborted_done`) — serve's stream reader only releases an HTTP handler
    // on the recognized terminal dialect, so a raw custom event here would
    // hold the admission guard forever.
    if check_abort(id) {
        crate::ar::emit_active_route_cancel(stdout, id, generated_count);
        return;
    }

    m.seq_pos = m.lfm2moe().unwrap().state.n_tokens;

    let tok_s: f64 = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    // `tool_calls` finish when a complete span produced executable calls —
    // same convention as the text path.
    let finish_reason = if !tool_calls.is_empty() {
        "tool_calls"
    } else {
        crate::common::length_or_stop(generated_count, max_tokens, decoded_eot)
    };
    let mut pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated_count,
        "tok_s": (tok_s * 100.0).round() / 100.0,
        "prefill_ms": prefill_ms,
        "total_ms": t_turn.elapsed().as_millis().max(1),
        "attempt_id": active_attempt_id(),
        "finish_reason": finish_reason,
    });
    hipfire_engine::emit::stage_terminal_tool_calls(&mut pending_done, finish_reason, &tool_calls);
    // Legacy contract: the `tool_calls` event must precede commit_ready.
    hipfire_engine::emit::emit_tool_calls_event(stdout, id, &tool_calls);
    let _ = stdout.flush();
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => {
            crate::ar::emit_active_route_done_value(stdout, &pending_done)
        }
        ClientTerminalDecision::Abort => {
            // Same release contract as the post-loop latch: the terminal pair
            // must be the recognized wire dialect or serve holds its
            // admission guard forever.
            crate::ar::emit_active_route_cancel(stdout, id, generated_count);
        }
    }
}

/// Image conversations keep their history (#1079): the whole conversation is
/// rendered through the chat template with every image at its message position,
/// earlier assistant turns replay their emitted tokens, and a turn that extends
/// the resident conversation reuses it — rope phases included.
#[cfg(test)]
mod vl_history_tests {
    use super::{
        expand_image_pads, image_pad_runs, vl_mrope_positions, vl_plan_reuse, vl_render_prompt,
        VlGrid, VlRenderInputs,
    };
    use hipfire_engine::turn_cache::{apply_turn_cache_action, TurnCacheAction};
    use hipfire_loader::AsstTurnCache;
    use hipfire_runtime::prompt_frame::{AssistantPrefix, Message, Role, IMAGE_SLOT};
    use hipfire_runtime::tokenizer::Tokenizer;

    const IM_END: u32 = 1;
    const VS: u32 = 4;
    const PAD: u32 = 5;
    const VE: u32 = 6;
    const BYTE_BASE: u32 = 100;
    /// Visual tokens per image in the synthetic conversations.
    const N_VISUAL: usize = 3;
    const TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    fn byte_char(b: u8) -> char {
        let mut bs: Vec<u32> = ((b'!' as u32)..=(b'~' as u32)).collect();
        bs.extend(0xA1u32..=0xACu32);
        bs.extend(0xAEu32..=0xFFu32);
        let mut cs = bs.clone();
        let mut n = 0u32;
        for byte in 0u32..=255 {
            if !bs.contains(&byte) {
                bs.push(byte);
                cs.push(256 + n);
                n += 1;
            }
        }
        let i = bs.iter().position(|&x| x == b as u32).unwrap();
        char::from_u32(cs[i]).unwrap()
    }

    /// Byte-level BPE with one merge (`h e`): the canonical encode of "he" is
    /// one token while a producer may emit it as two byte tokens, so a
    /// re-encoded history turn differs observably from the emitted IDs.
    fn tokenizer() -> Tokenizer {
        let specials = [
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
            "<|endoftext|>",
            "<|reserved_0|>",
            "<|reserved_1|>",
            "<|reserved_2|>",
        ];
        let mut entries: Vec<String> = specials
            .iter()
            .enumerate()
            .map(|(i, s)| format!(r#""{s}": {i}"#))
            .collect();
        entries.push(r#""he": 50"#.to_string());
        for b in 0u32..=255 {
            let escaped = serde_json::to_string(&byte_char(b as u8).to_string()).unwrap();
            entries.push(format!("{escaped}: {}", BYTE_BASE + b));
        }
        let added: Vec<String> = specials
            .iter()
            .enumerate()
            .map(|(i, s)| format!(r#"{{"id": {i}, "content": "{s}", "special": true}}"#))
            .collect();
        let json = format!(
            r#"{{"model": {{"type": "BPE", "vocab": {{ {} }}, "merges": ["h e"]}}, "added_tokens": [{}]}}"#,
            entries.join(", "),
            added.join(", ")
        );
        Tokenizer::from_hf_json(&json).expect("test tokenizer")
    }

    fn msg(role: Role, content: &str) -> Message {
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

    /// A user turn carrying an image before its text, as the gateway projects it.
    fn image_user(text: &str) -> Message {
        msg(Role::User, &format!("{IMAGE_SLOT}\n{text}"))
    }

    fn render(
        tok: &Tokenizer,
        template: Option<&str>,
        history: &[Message],
        n_images: usize,
        resident: &[u32],
        cache: &mut AsstTurnCache,
    ) -> Result<super::VlPrompt, String> {
        vl_render_prompt(
            &VlRenderInputs {
                tokenizer: tok,
                chat_template: template,
                jinja_enabled: true,
                history: Some(history),
                n_images,
                system: None,
                prompt: "",
                enable_thinking: false,
                reasoning_effort: None,
                assistant_prefix: AssistantPrefix::Plain,
                image_ids: [VS, PAD, VE],
                resident_tokens: resident,
                turn_cache_enabled: true,
            },
            cache,
        )
    }

    fn decode(tok: &Tokenizer, tokens: &[u32]) -> String {
        String::from_utf8_lossy(&tok.decode_bytes(tokens)).into_owned()
    }

    #[test]
    fn an_image_in_history_keeps_every_turn_and_each_image_at_its_message() {
        let tok = tokenizer();
        let history = vec![
            msg(Role::System, "be brief"),
            image_user("remember purple hippo"),
            msg(Role::Assistant, "got it"),
            image_user("and this one"),
            msg(Role::Assistant, "nice"),
            msg(Role::User, "what was the code word?"),
        ];
        let mut cache = AsstTurnCache::new_from_env();
        // Two historical images do not error, and the last user turn is text.
        let prompt = render(&tok, Some(TEMPLATE), &history, 2, &[], &mut cache).unwrap();
        assert!(prompt.whole_conversation);
        let pads: Vec<usize> = prompt
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == PAD)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(pads.len(), 2, "one pad per image, expanded later");
        let before_first = decode(&tok, &prompt.tokens[..pads[0]]);
        let between = decode(&tok, &prompt.tokens[pads[0]..pads[1]]);
        let after_second = decode(&tok, &prompt.tokens[pads[1]..]);
        // System prompt, then the first image, then ITS question and the answer;
        // the second image sits in the next user turn; the question comes last.
        assert!(before_first.contains("be brief") && !before_first.contains("purple hippo"));
        assert!(between.contains("remember purple hippo") && between.contains("got it"));
        assert!(!between.contains("and this one"));
        assert!(after_second.contains("and this one") && after_second.contains("nice"));
        assert!(
            after_second.ends_with("what was the code word?<|im_end|>\n<|im_start|>assistant\n")
        );
        // The text-only last turn still sees both earlier images in the prompt.
        let counts = [N_VISUAL, N_VISUAL];
        let expanded = expand_image_pads(&prompt.tokens, PAD, &counts).unwrap();
        assert_eq!(expanded.len(), prompt.tokens.len() + 2 * (N_VISUAL - 1));
        assert_eq!(expanded.iter().filter(|&&t| t == PAD).count(), 2 * N_VISUAL);
    }

    #[test]
    fn image_slots_and_images_must_agree() {
        let tok = tokenizer();
        let history = vec![
            image_user("one"),
            msg(Role::Assistant, "ok"),
            image_user("two"),
        ];
        let mut cache = AsstTurnCache::new_from_env();
        for wrong in [0usize, 1, 3] {
            let err = render(&tok, Some(TEMPLATE), &history, wrong, &[], &mut cache)
                .err()
                .expect("a mismatch must be refused, not silently truncated");
            assert!(err.contains("2 image position(s)"), "{err}");
        }
        assert!(render(&tok, Some(TEMPLATE), &history, 2, &[], &mut cache).is_ok());
    }

    #[test]
    fn without_a_template_only_a_single_turn_is_framed_and_longer_is_refused() {
        let tok = tokenizer();
        let mut cache = AsstTurnCache::new_from_env();
        let single = vec![msg(Role::System, "sys"), image_user("describe")];
        let prompt = render(&tok, None, &single, 1, &[], &mut cache).unwrap();
        assert!(!prompt.whole_conversation);
        let ids = &prompt.tokens;
        let vs = ids.iter().position(|&t| t == VS).expect("vision start");
        assert_eq!(&ids[vs..vs + 3], &[VS, PAD, VE]);
        assert!(decode(&tok, ids).contains("describe"));
        let multi = vec![
            image_user("one"),
            msg(Role::Assistant, "ok"),
            msg(Role::User, "two"),
        ];
        let err = render(&tok, None, &multi, 1, &[], &mut cache)
            .err()
            .unwrap();
        assert!(err.contains("chat template"), "{err}");
    }

    #[test]
    fn expand_image_pads_widens_each_pad_in_order_and_checks_counts() {
        let tokens = [1, PAD, 2, PAD, 3];
        let out = expand_image_pads(&tokens, PAD, &[3, 2]).unwrap();
        assert_eq!(out, vec![1, PAD, PAD, PAD, 2, PAD, PAD, 3]);
        assert_eq!(image_pad_runs(&out, PAD), vec![(1, 3), (5, 2)]);
        assert!(expand_image_pads(&tokens, PAD, &[3]).is_err());
        assert!(expand_image_pads(&tokens, PAD, &[3, 2, 1]).is_err());
        assert!(expand_image_pads(&tokens, PAD, &[3, 0]).is_err());
    }

    #[test]
    fn only_a_pure_forward_extension_is_reused() {
        let prior = vec![1, 2, PAD, PAD, 3, 4];
        let longer = vec![1, 2, PAD, PAD, 3, 4, 5, 6];
        let one = [[1u8; 32]];
        let plan = |prior: &[u32], prompt: &[u32], seq_pos: usize, allowed: bool| {
            vl_plan_reuse(prior, prompt, seq_pos, allowed, PAD, &one, &one)
        };
        assert_eq!(plan(&prior, &longer, 6, true), 6);
        // Reuse disabled (eviction / kill switch), nothing resident, a cursor that
        // is not at the end of the resident tokens: cold.
        assert_eq!(plan(&prior, &longer, 6, false), 0);
        assert_eq!(plan(&[], &longer, 0, true), 0);
        assert_eq!(plan(&prior, &longer, 5, true), 0);
        // Re-sent identical prompt: nothing to prefill, the last token's
        // recurrent update must not be applied twice.
        assert_eq!(plan(&prior, &prior, 6, true), 0);
        // A diverging or edited history is not an extension.
        assert_eq!(plan(&prior, &[1, 9, PAD, PAD, 3, 4, 5], 6, true), 0);
        assert_eq!(plan(&prior, &[1, 2, PAD, PAD, 3], 6, true), 0);
        // A boundary inside an image's pad run is never a reuse point.
        let cut = vec![1, 2, PAD, PAD];
        assert_eq!(plan(&cut, &[1, 2, PAD, PAD, PAD, 3], 4, true), 0);
        // Same tokens but a different image: cold.
        assert_eq!(
            vl_plan_reuse(&prior, &longer, 6, true, PAD, &one, &[[2u8; 32]]),
            0
        );
    }

    #[test]
    fn mrope_phases_are_a_function_of_the_whole_conversation_and_resume_after_the_image() {
        let a = VlGrid {
            grid_h: 4,
            grid_w: 4,
            n_visual: 4,
        };
        let b = VlGrid {
            grid_h: 4,
            grid_w: 8,
            n_visual: 8,
        };
        let mut turn1 = vec![10, 11, VS, PAD, PAD, PAD, PAD, VE, 12, 13];
        let p1 = vl_mrope_positions(&turn1, PAD, &[a], 2).unwrap();
        // Turn 2 appends text and a second image (base > 0 for the new rows).
        let base = turn1.len();
        turn1.extend([14, 15, VS]);
        turn1.extend(std::iter::repeat(PAD).take(8));
        turn1.extend([VE, 16]);
        let turn2 = turn1;
        let p2 = vl_mrope_positions(&turn2, PAD, &[a, b], 2).unwrap();
        // Appending never moves an earlier row.
        assert_eq!(&p2.positions[..base], p1.positions.as_slice());
        // The first new row resumes at `previous_max + 1` (the image compressed
        // 4 rows into a 2-wide grid), not at its token index.
        let prev_max = p1.positions.iter().flatten().copied().max().unwrap();
        assert_eq!(p2.positions[base], [prev_max + 1; 3]);
        assert_eq!(p2.positions[base][0], base as i32 + p1.rope_delta);
        assert!(p1.rope_delta < 0, "an image compresses positions");
        // The second image: one `t`, a 2x4 merged grid of (h, w).
        let second = image_pad_runs(&turn2, PAD)[1].0;
        let origin = p2.positions[second][0];
        assert_eq!(p2.positions[second], [origin, origin, origin]);
        assert_eq!(p2.positions[second + 7], [origin, origin + 1, origin + 3]);
        // The rope context of the new rows (base > 0) addresses them directly and
        // decodes at `pos + rope_delta` past the prompt.
        let ctx = hipfire_arch_qwen35::qwen35::MropeCtx {
            base,
            positions: p2.positions[base..].to_vec(),
            rope_delta: p2.rope_delta,
            section: [11, 11, 10],
        };
        assert_eq!(ctx.pos3(base), p2.positions[base]);
        assert_eq!(ctx.pos3(turn2.len() - 1), p2.positions[turn2.len() - 1]);
        assert_eq!(
            ctx.pos3(turn2.len()),
            [turn2.len() as i32 + p2.rope_delta; 3]
        );
        // Geometry that does not match the prompt is refused, not mis-positioned.
        assert!(vl_mrope_positions(&turn2, PAD, &[a], 2).is_err());
        let wrong = VlGrid {
            grid_h: 4,
            grid_w: 4,
            n_visual: 5,
        };
        assert!(vl_mrope_positions(&turn2, PAD, &[a, wrong], 2).is_err());
    }

    /// What the daemon does for a turn: render, widen the pads, plan the reuse.
    struct Turn {
        prompt: Vec<u32>,
        reuse: usize,
        encoded_images: Vec<usize>,
    }

    /// Content fingerprint of the n-th distinct test image.
    fn fp(n: u8) -> [u8; 32] {
        [n; 32]
    }

    fn plan_turn(
        tok: &Tokenizer,
        history: &[Message],
        resident: &[u32],
        resident_fps: &[[u8; 32]],
        fps: &[[u8; 32]],
        cache: &mut AsstTurnCache,
    ) -> Turn {
        let n_images = history
            .iter()
            .map(|m| m.content.matches(IMAGE_SLOT).count())
            .sum();
        let rendered = render(tok, Some(TEMPLATE), history, n_images, resident, cache).unwrap();
        let counts = vec![N_VISUAL; n_images];
        let prompt = expand_image_pads(&rendered.tokens, PAD, &counts).unwrap();
        let reuse = vl_plan_reuse(
            resident,
            &prompt,
            resident.len(),
            true,
            PAD,
            resident_fps,
            fps,
        );
        let encoded_images = image_pad_runs(&prompt, PAD)
            .iter()
            .enumerate()
            .filter(|(_, (start, _))| *start >= reuse)
            .map(|(i, _)| i)
            .collect();
        Turn {
            prompt,
            reuse,
            encoded_images,
        }
    }

    /// The tokens a finished turn leaves resident, and the cache entry the
    /// daemon publishes after commit.
    fn finish_turn(
        tok: &Tokenizer,
        prompt: &[u32],
        reply: &str,
        cache: &mut AsstTurnCache,
    ) -> Vec<u32> {
        // The producer emits one byte-level token per byte, then `<|im_end|>`.
        let mut streamed: Vec<u32> = reply.bytes().map(|b| BYTE_BASE + b as u32).collect();
        streamed.push(IM_END);
        let nl = tok.encode("\n");
        let nl_set: std::collections::HashSet<u32> = nl.iter().copied().collect();
        let cached_seq = crate::qwen::qwen_dflash_cache_seq(&streamed, Some(IM_END), &nl_set);
        let action = TurnCacheAction {
            store: true,
            fingerprint_text: crate::common::normalize_asst_turn_for_fingerprint(reply),
            tool_calls: Vec::new(),
        };
        apply_turn_cache_action(
            |fp, seq| {
                cache.insert(fp, crate::qwen::qwen_cached_turn_entry(tok, seq, false));
            },
            &action,
            cached_seq,
        )
        .expect("a clean stop stores its turn");
        let mut resident = prompt.to_vec();
        resident.extend(streamed);
        resident.extend(nl);
        resident
    }

    #[test]
    fn an_ongoing_image_conversation_continues_in_place_and_encodes_only_new_images() {
        let tok = tokenizer();
        let mut cache = AsstTurnCache::new_from_env();
        let mut history = vec![msg(Role::System, "be brief"), image_user("hello there")];
        // Turn 1: cold. The image is encoded.
        let t1 = plan_turn(&tok, &history, &[], &[], &[fp(1)], &mut cache);
        assert_eq!((t1.reuse, t1.encoded_images.as_slice()), (0, &[0][..]));
        let reply1 = "he said hi";
        let resident1 = finish_turn(&tok, &t1.prompt, reply1, &mut cache);

        // Turn 2: the client resends the image in history and adds a text turn.
        history.push(msg(Role::Assistant, reply1));
        history.push(msg(Role::User, "and a follow up"));
        let t2 = plan_turn(&tok, &history, &resident1, &[fp(1)], &[fp(1)], &mut cache);
        assert_eq!(
            t2.reuse,
            resident1.len(),
            "the whole first turn stays resident"
        );
        assert!(
            t2.encoded_images.is_empty(),
            "the resent image is not encoded again"
        );
        assert_eq!(&t2.prompt[..t2.reuse], resident1.as_slice());
        let resident2 = finish_turn(&tok, &t2.prompt, "he agreed", &mut cache);

        // Turn 3: a second image arrives with the next question; only it is
        // encoded, the first stays resident.
        history.push(msg(Role::Assistant, "he agreed"));
        history.push(image_user("a second picture"));
        let t3 = plan_turn(
            &tok,
            &history,
            &resident2,
            &[fp(1)],
            &[fp(1), fp(2)],
            &mut cache,
        );
        assert_eq!(t3.reuse, resident2.len());
        assert_eq!(t3.encoded_images, vec![1]);

        // Without the published turn the template re-render does not reproduce
        // the emitted tokens (BPE), so the conversation could not be reused.
        let mut empty = AsstTurnCache::new_from_env();
        let cold = plan_turn(
            &tok,
            &history[..4],
            &resident1,
            &[fp(1)],
            &[fp(1)],
            &mut empty,
        );
        assert_eq!(cold.reuse, 0, "no verbatim replay, no pure extension");

        // An edited earlier message is not an extension either.
        let mut edited = history[..4].to_vec();
        edited[1] = image_user("hello there!!");
        let miss = plan_turn(&tok, &edited, &resident1, &[fp(1)], &[fp(1)], &mut cache);
        assert_eq!(miss.reuse, 0);
        assert_eq!(
            miss.encoded_images,
            vec![0],
            "a cold start re-encodes every image"
        );

        // Same words, same grid, DIFFERENT image bytes at the same position
        // (a re-captured screenshot, a regenerate flow): the tokens are equal
        // but the resident rows belong to the old image, so nothing is reused.
        let swapped = plan_turn(
            &tok,
            &history[..4],
            &resident1,
            &[fp(1)],
            &[fp(7)],
            &mut cache,
        );
        assert_eq!(
            swapped.reuse, 0,
            "a replaced image must not reuse the old rows"
        );
        assert_eq!(swapped.encoded_images, vec![0]);
        let same = plan_turn(
            &tok,
            &history[..4],
            &resident1,
            &[fp(1)],
            &[fp(1)],
            &mut cache,
        );
        assert_eq!(same.reuse, resident1.len(), "identical bytes keep the rows");
        // A new image after the resident one does not disturb the resident one.
        let grown = plan_turn(
            &tok,
            &history,
            &resident2,
            &[fp(1)],
            &[fp(1), fp(2)],
            &mut cache,
        );
        assert_eq!(grown.reuse, resident2.len());
        // Bookkeeping that disagrees with the resident tokens is a cold start.
        let stale = plan_turn(&tok, &history, &resident2, &[], &[fp(1), fp(2)], &mut cache);
        assert_eq!(stale.reuse, 0);
    }

    /// A client that drops the image (a text-only conversation) after an image
    /// conversation: the text path's render has no image rows, so it diverges
    /// from the resident tokens at or before the first image row and never
    /// satisfies its pure-extension test (`lcp == prior_len`) — it resumes from
    /// a checkpoint at or before the divergence or starts cold, never past an
    /// image row.
    #[test]
    fn a_text_request_after_an_image_conversation_diverges_before_the_image_rows() {
        let tok = tokenizer();
        let mut cache = AsstTurnCache::new_from_env();
        let history = vec![msg(Role::System, "be brief"), image_user("hello there")];
        let t1 = plan_turn(&tok, &history, &[], &[], &[fp(1)], &mut cache);
        let resident = finish_turn(&tok, &t1.prompt, "he said hi", &mut cache);
        let first_pad = resident.iter().position(|&t| t == PAD).unwrap();
        let text_only = vec![
            msg(Role::System, "be brief"),
            msg(Role::User, "hello there"),
            msg(Role::Assistant, "he said hi"),
            msg(Role::User, "and a follow up"),
        ];
        let rendered = render(&tok, Some(TEMPLATE), &text_only, 0, &resident, &mut cache).unwrap();
        assert!(!rendered.tokens.contains(&PAD) && !rendered.tokens.contains(&VS));
        let lcp = resident
            .iter()
            .zip(&rendered.tokens)
            .take_while(|(a, b)| a == b)
            .count();
        assert!(lcp <= first_pad, "diverges before the first image row");
        assert!(
            lcp < resident.len(),
            "never a pure extension of the image conversation"
        );
    }
}
