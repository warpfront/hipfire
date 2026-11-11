// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! VMM continuous-batching route: the singleton's per-request semantics
//! carried onto batch lanes.
//!
//! * **Conversation continuity.** The singleton keeps exactly one
//!   conversation resident (`conversation_tokens`, the bundle KV/DeltaNet,
//!   `prefill_checkpoints`) and decides each turn's reuse in `ar.rs`
//!   `generate`: forward-extension hit, checkpoint resume, else cold. Batch
//!   lanes run concurrent conversations, so every finished conversation —
//!   batched, promoted or singleton (parked between turns) — is kept as a
//!   [`VmmPrefixEntry`] in the store's prefix pool: its owner with every
//!   token forwarded (ChatML trailer included), its token stream and its
//!   resume checkpoints. Admission, on a lane or on the singleton, runs that
//!   same decision against the entry it reuses most, so a turn is warm or
//!   cold — and therefore byte-for-byte — as the singleton serving that
//!   conversation alone.
//! * **Think control.** `max_think_tokens`, `reasoning.max_total_tokens`,
//!   the force-answer signal, its latch and the post-latch answer bound,
//!   applied per lane at the singleton decode loop's decision point
//!   ([`ThinkCtl::step`]).
//!
//! The pool is active only where the singleton's own cache is the AR
//! route's: no speculator or MTP spec lanes (their routes keep their own
//! caches), Jinja rendering, prompt cache on, no eviction/adaptive KV.

use crate::ar::{ckpt_interval, ckpt_max, ckpt_resume_enabled, truncate_checkpoints};
use hipfire_arch_qwen35::forward_slots::vmm::{
    Qwen35RequestState, Qwen35VmmStore, VmmPrefixEntry, VmmRequestInit,
};
use hipfire_arch_qwen35::qwen35::DeltaNetState;
use hipfire_arch_qwen35::speculative::{self, DeltaNetSnapshot};
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_loader::LoadedModel;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::slot_batch::RequestEpoch;
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};

/// DeltaNet resume checkpoints `(position, snapshot)`, the singleton's
/// `prefill_checkpoints` ring.
pub type Checkpoints = Vec<(usize, DeltaNetSnapshot)>;

/// Epoch of an owner that is not admitted to any lane.
const UNADMITTED: RequestEpoch = RequestEpoch {
    request_tag: 0,
    owner_generation: 0,
};

static STAMP: AtomicU64 = AtomicU64::new(1);

fn next_stamp() -> u64 {
    STAMP.fetch_add(1, Ordering::Relaxed)
}

// ── Singleton prompt-cache decision ───────────────────────────────────

/// Reuse of one prior conversation for a rendered prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reuse {
    /// Pure forward extension: keep `[0..lcp)`, prefill `rendered[lcp..]`.
    Extend(usize),
    /// Divergence or exact re-send: restore checkpoint `idx` (at `pos`),
    /// prefill `rendered[pos..]`.
    Resume { idx: usize, pos: usize },
    /// No reusable prefix: prefill the whole render from zero state.
    Cold,
}

impl Reuse {
    /// Prompt position the prefill starts at (the reported cached tokens).
    pub fn start(self) -> usize {
        match self {
            Reuse::Extend(lcp) => lcp,
            Reuse::Resume { pos, .. } => pos,
            Reuse::Cold => 0,
        }
    }
}

/// `ar.rs` `generate`'s prompt-cache decision for `rendered` against a prior
/// conversation and its checkpoints (single GPU, no eviction, qwen35).
pub fn plan_reuse(
    prior: &[u32],
    checkpoints: &[(usize, DeltaNetSnapshot)],
    rendered: &[u32],
) -> Reuse {
    plan_reuse_at(
        prior,
        checkpoints.iter().map(|(p, _)| *p),
        rendered,
        ckpt_resume_enabled(),
    )
}

/// [`plan_reuse`] over checkpoint positions (ascending, as the ring keeps
/// them): pure forward extension (`lcp == prior.len() < rendered.len()`)
/// reuses the prefix; divergence or an exact re-send (DeltaNet cannot rewind,
/// so a re-sent final token must not be applied twice) resumes from the
/// latest checkpoint at or before the LCP that leaves at least one token to
/// prefill, else goes cold.
fn plan_reuse_at<I>(prior: &[u32], positions: I, rendered: &[u32], resume_enabled: bool) -> Reuse
where
    I: DoubleEndedIterator<Item = usize> + ExactSizeIterator,
{
    let lcp = prior
        .iter()
        .zip(rendered)
        .take_while(|(a, b)| a == b)
        .count();
    if lcp < prior.len() || lcp == rendered.len() {
        if !resume_enabled {
            return Reuse::Cold;
        }
        positions
            .enumerate()
            .rev()
            .find(|&(_, p)| p <= lcp && p < rendered.len())
            .map_or(Reuse::Cold, |(idx, pos)| Reuse::Resume { idx, pos })
    } else {
        Reuse::Extend(lcp)
    }
}

/// Committed pool entry whose decision reuses the most of `rendered`
/// (newest on ties); `None` when nothing is reusable.
fn best_entry(pool: &[VmmPrefixEntry], rendered: &[u32]) -> Option<(usize, Reuse)> {
    pool.iter()
        .enumerate()
        .filter(|(_, e)| e.pending.is_none())
        .map(|(i, e)| (i, plan_reuse(&e.tokens, &e.checkpoints, rendered), e.stamp))
        .filter(|(_, r, _)| r.start() > 0)
        .max_by_key(|(_, r, stamp)| (r.start(), *stamp))
        .map(|(i, r, _)| (i, r))
}

/// Snapshot `dn` into `checkpoints` at the singleton's cadence.
pub fn checkpoint(
    checkpoints: &mut Checkpoints,
    dn: &DeltaNetState,
    kv: &hipfire_runtime::llama::KvCache,
    gpu: &mut rdna_compute::Gpu,
    pos: usize,
) {
    if ckpt_resume_enabled() {
        speculative::take_dn_checkpoint(checkpoints, dn, kv, gpu, pos, ckpt_interval(), ckpt_max());
    }
}

pub fn free_checkpoints(checkpoints: Checkpoints, gpu: &mut rdna_compute::Gpu) {
    for (_, snap) in checkpoints {
        snap.free_gpu(gpu);
    }
}

// ── Pool ownership ────────────────────────────────────────────────────

fn bundle_mut(state: &mut Option<Box<dyn ArchModel>>) -> Option<&mut Qwen35Bundle> {
    state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<Qwen35Bundle>())
}

fn store(m: &LoadedModel) -> Option<&Qwen35VmmStore> {
    m.qwen35().and_then(|b| b.vmm_store.as_ref())
}

/// The pool serves this model: VMM store staged with AR lanes only, no
/// singleton speculator, Jinja rendering, prompt cache on, plain KV.
pub fn pool_enabled(m: &LoadedModel) -> bool {
    m.speculator.is_none()
        && m.eviction.is_none()
        && m.kv_adaptive.is_none()
        && m.pp <= 1
        && m.ep.is_none()
        && m.chat_template.is_some()
        && hipfire_config::developer_var("HIPFIRE_JINJA_CHAT")
            .ok()
            .as_deref()
            != Some("0")
        && hipfire_config::developer_var("HIPFIRE_QWEN_PROMPT_CACHE")
            .ok()
            .as_deref()
            != Some("0")
        && store(m).is_some_and(|s| s.spec_engine().is_none())
}

/// Pool enabled and holding at least one committed conversation.
pub fn pool_active(m: &LoadedModel) -> bool {
    pool_enabled(m) && store(m).is_some_and(|s| s.prefix_pool.iter().any(|e| e.pending.is_none()))
}

/// Any conversation is held anywhere (pool entry, including ones awaiting
/// their client commit). Batch admission's analogue of the singleton's
/// "resident conversation is non-empty" render condition.
pub fn pool_nonempty(m: &LoadedModel) -> bool {
    store(m).is_some_and(|s| !s.prefix_pool.is_empty())
}

/// A kept conversation awaits its client commit. The singleton serves the
/// next request only after the previous turn's decision (its conversation
/// and verbatim assistant turn land then); batch admission holds while
/// this is true for the same reason — serve answers the client at
/// `commit_ready`, so its next turn can arrive before the commit.
pub fn pool_pending(m: &LoadedModel) -> bool {
    store(m).is_some_and(|s| s.prefix_pool.iter().any(|e| e.pending.is_some()))
}

/// Drop the least recently stored committed entry. `false` when none.
fn evict_one(store: &mut Qwen35VmmStore, gpu: &mut rdna_compute::Gpu) -> bool {
    let Some(i) = store
        .prefix_pool
        .iter()
        .enumerate()
        .filter(|(_, e)| e.pending.is_none())
        .min_by_key(|(_, e)| e.stamp)
        .map(|(i, _)| i)
    else {
        return false;
    };
    if let Err(e) = store.prefix_pool.remove(i).free_gpu(gpu) {
        eprintln!("[vmm-prefix] evict free: {e}");
    }
    true
}

/// Keep a finished conversation; past the width the oldest committed
/// entries are dropped.
pub fn retain(
    store: &mut Qwen35VmmStore,
    gpu: &mut rdna_compute::Gpu,
    state: Qwen35RequestState,
    tokens: Vec<u32>,
    checkpoints: Checkpoints,
    pending: Option<(String, u64)>,
) {
    store.prefix_pool.push(VmmPrefixEntry {
        state,
        tokens,
        checkpoints,
        pending,
        stamp: next_stamp(),
    });
    while store.prefix_pool.len() > store.max_slots() {
        if !evict_one(store, gpu) {
            break;
        }
    }
}

/// The client decided the attempt whose finished conversation was kept:
/// commit makes it reusable, abort drops it (the singleton rolls back an
/// aborted turn's conversation).
pub fn settle(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    id: &str,
    attempt_id: u64,
    commit: bool,
) {
    let Some(store) = bundle_mut(&mut m.state).and_then(|b| b.vmm_store.as_mut()) else {
        return;
    };
    let Some(i) = store.prefix_pool.iter().position(|e| {
        e.pending
            .as_ref()
            .is_some_and(|(pid, a)| pid == id && *a == attempt_id)
    }) else {
        return;
    };
    if commit {
        store.prefix_pool[i].pending = None;
        store.prefix_pool[i].stamp = next_stamp();
    } else if let Err(e) = store.prefix_pool.remove(i).free_gpu(gpu) {
        eprintln!("[vmm-prefix] drop on abort: {e}");
    }
}

/// Free every kept conversation (cold reset / fail-closed rollback).
pub fn clear(state: &mut Option<Box<dyn ArchModel>>, gpu: &mut rdna_compute::Gpu) {
    let Some(store) = bundle_mut(state).and_then(|b| b.vmm_store.as_mut()) else {
        return;
    };
    for e in std::mem::take(&mut store.prefix_pool) {
        if let Err(err) = e.free_gpu(gpu) {
            eprintln!("[vmm-prefix] clear: {err}");
        }
    }
}

/// Copy `src`'s DeltaNet state into `dst` (common buffer extent: same
/// shapes, allocations may be pool-rounded differently). Synchronizes.
fn copy_dn(
    gpu: &mut rdna_compute::Gpu,
    dst: &DeltaNetState,
    src: &DeltaNetState,
) -> Result<(), String> {
    if dst.s_matrices.len() != src.s_matrices.len()
        || dst.s_scales.len() != src.s_scales.len()
        || dst.conv_states.len() != src.conv_states.len()
        || dst.s_ef_residual.len() != src.s_ef_residual.len()
    {
        return Err("copy_dn: DeltaNet layouts differ".into());
    }
    let pairs = dst
        .s_matrices
        .iter()
        .zip(&src.s_matrices)
        .chain(dst.s_scales.iter().zip(&src.s_scales))
        .chain(dst.conv_states.iter().zip(&src.conv_states))
        .chain(dst.s_ef_residual.iter().zip(&src.s_ef_residual));
    for (d, s) in pairs {
        if d.shape != s.shape {
            return Err("copy_dn: DeltaNet tensor shapes differ".into());
        }
        let n = d.buf.size().min(s.buf.size());
        gpu.memcpy_dtod_at_auto(&d.buf, 0, &s.buf, 0, n)
            .map_err(|e| format!("copy_dn: {e}"))?;
    }
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("copy_dn: sync: {e}"))
}

/// A fresh owner shaped like the bundle's. On allocation failure the oldest
/// committed entries are dropped one at a time and the allocation retried.
pub fn alloc_owner(
    gpu: &mut rdna_compute::Gpu,
    b: &mut Qwen35Bundle,
    epoch: RequestEpoch,
    slot: usize,
    init: impl Fn() -> VmmRequestInit,
) -> Result<Qwen35RequestState, String> {
    loop {
        match Qwen35RequestState::new_like(
            gpu,
            &b.config,
            &b.kv_cache,
            &b.dn_state,
            epoch,
            slot,
            init(),
        ) {
            Ok(s) => return Ok(s),
            Err(e) => {
                let evicted = b.vmm_store.as_mut().is_some_and(|s| evict_one(s, gpu));
                if !evicted {
                    return Err(e);
                }
            }
        }
    }
}

fn parked_init() -> VmmRequestInit {
    VmmRequestInit {
        prompt_len: 0,
        stop_ids: Vec::new(),
        sampler: hipfire_runtime::sampler::SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            blocked_tokens: Vec::new(),
            top_k: None,
            min_p: None,
        },
        rng_state: 0,
        history: Vec::new(),
    }
}

/// Graph/replay recordings bake the bundle's KV/DN addresses: drop them
/// after the bundle's owners changed (as promotion does).
fn rearm(gpu: &mut rdna_compute::Gpu) {
    crate::common::fail_closed_invalidate_graphs_and_replay(gpu);
    gpu.replay.rearm_after_layout_growth();
}

/// Move the bundle's resident conversation into the pool. The bundle keeps
/// a fresh owner and zeroed DeltaNet state; the singleton host cursors are
/// cleared. No-op on an empty or inconsistent resident.
fn park_bundle(
    gpu: &mut rdna_compute::Gpu,
    b: &mut Qwen35Bundle,
    seq_pos: &mut usize,
    conversation: &mut Vec<u32>,
    checkpoints: &mut Checkpoints,
    dflash_checkpoints: &mut Checkpoints,
) -> Result<(), String> {
    if conversation.is_empty() {
        return Ok(());
    }
    if *seq_pos != conversation.len() || b.kv_cache.compact_offset != 0 {
        return Err(format!(
            "resident not parkable: seq_pos={} conversation={} compact_offset={}",
            seq_pos,
            conversation.len(),
            b.kv_cache.compact_offset
        ));
    }
    let mut st = alloc_owner(gpu, b, UNADMITTED, 0, parked_init)?;
    if let Err(e) = st.copy_dn_from(gpu, &b.dn_state) {
        let freed = st.free_gpu(gpu);
        return Err(format!("{e}; free: {freed:?}"));
    }
    std::mem::swap(&mut st.kv, &mut b.kv_cache);
    st.position = *seq_pos;
    let reset = b.dn_state.reset(gpu);
    let tokens = std::mem::take(conversation);
    let cks = std::mem::take(checkpoints);
    *seq_pos = 0;
    crate::common::free_checkpoints(dflash_checkpoints, gpu);
    rearm(gpu);
    let store = b.vmm_store.as_mut().ok_or("VMM store not staged")?;
    retain(store, gpu, st, tokens, cks, None);
    reset.map_err(|e| format!("bundle DeltaNet reset after park: {e}"))
}

/// Park the singleton's resident conversation (after a singleton turn and
/// before a batch drive), so lanes and later singleton turns find it.
pub fn park_resident(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu) {
    if !pool_enabled(m) || m.conversation_tokens.is_empty() {
        return;
    }
    let LoadedModel {
        state,
        seq_pos,
        conversation_tokens,
        prefill_checkpoints,
        dflash_checkpoints,
        ..
    } = m;
    let Some(b) = bundle_mut(state) else {
        return;
    };
    if let Err(e) = park_bundle(
        gpu,
        b,
        seq_pos,
        conversation_tokens,
        prefill_checkpoints,
        dflash_checkpoints,
    ) {
        eprintln!("[vmm-prefix] park resident: {e}");
    }
}

/// Singleton prompt cache (`ar.rs` `generate`): when a pool entry's
/// decision reuses more of `rendered` than the resident conversation's,
/// park the resident and make that entry resident. The caller's ordinary
/// LCP/resume/cold logic then runs unchanged against it.
pub fn singleton_adopt(
    state: &mut Option<Box<dyn ArchModel>>,
    seq_pos: &mut usize,
    conversation: &mut Vec<u32>,
    checkpoints: &mut Checkpoints,
    dflash_checkpoints: &mut Checkpoints,
    gpu: &mut rdna_compute::Gpu,
    rendered: &[u32],
) {
    let Some(b) = bundle_mut(state) else {
        return;
    };
    let Some(store) = b.vmm_store.as_mut() else {
        return;
    };
    let resident = if conversation.is_empty() {
        0
    } else {
        plan_reuse(conversation, checkpoints, rendered).start()
    };
    let Some((idx, reuse)) = best_entry(&store.prefix_pool, rendered) else {
        return;
    };
    if reuse.start() <= resident {
        return;
    }
    let entry = store.prefix_pool.remove(idx);
    if let Err(e) = park_bundle(
        gpu,
        b,
        seq_pos,
        conversation,
        checkpoints,
        dflash_checkpoints,
    ) {
        // The resident stays (and stays resident); the entry goes back.
        eprintln!("[vmm-prefix] singleton adopt: {e}");
        if let Some(store) = b.vmm_store.as_mut() {
            store.prefix_pool.push(entry);
        }
        return;
    }
    let VmmPrefixEntry {
        state: mut st,
        tokens,
        checkpoints: cks,
        ..
    } = entry;
    if let Err(e) = copy_dn(gpu, &b.dn_state, &st.dn) {
        // Bundle DeltaNet is undefined: zero it and serve cold.
        eprintln!("[vmm-prefix] singleton adopt: {e}; cold");
        let _ = b.dn_state.reset(gpu);
        free_checkpoints(cks, gpu);
        let _ = st.free_gpu(gpu);
        return;
    }
    std::mem::swap(&mut st.kv, &mut b.kv_cache);
    *seq_pos = tokens.len();
    *conversation = tokens;
    *checkpoints = cks;
    rearm(gpu);
    if let Err(e) = st.free_gpu(gpu) {
        eprintln!("[vmm-prefix] singleton adopt free: {e}");
    }
}

/// Batch admission: the lane owner for `rendered` — the pool entry whose
/// decision reuses most, restored to its reuse point (`Extend` keeps the
/// prefix, `Resume` restores the checkpoint), else a fresh owner. Returns
/// the owner (re-initialized for this request, not admitted), its
/// checkpoints and the prefill start (cached tokens).
pub fn lane_owner(
    gpu: &mut rdna_compute::Gpu,
    b: &mut Qwen35Bundle,
    rendered: &[u32],
    epoch: RequestEpoch,
    slot: usize,
    init: impl Fn() -> VmmRequestInit,
) -> Result<(Qwen35RequestState, Checkpoints, usize), String> {
    let reused = b.vmm_store.as_mut().and_then(|store| {
        best_entry(&store.prefix_pool, rendered).map(|(i, r)| (store.prefix_pool.remove(i), r))
    });
    if let Some((entry, reuse)) = reused {
        let VmmPrefixEntry {
            state: mut st,
            checkpoints: mut cks,
            ..
        } = entry;
        let start = match reuse {
            Reuse::Extend(lcp) => lcp,
            Reuse::Resume { idx, pos } => {
                if let Err(e) = cks[idx].1.restore_to(&mut st.dn, gpu) {
                    free_checkpoints(cks, gpu);
                    let freed = st.free_gpu(gpu);
                    return Err(format!("prefix resume restore: {e}; free: {freed:?}"));
                }
                truncate_checkpoints(&mut cks, idx + 1, gpu);
                pos
            }
            Reuse::Cold => unreachable!("best_entry yields reusable entries only"),
        };
        let init = init();
        st.epoch = epoch;
        st.slot = slot;
        st.position = start;
        st.pending_seed = None;
        st.poisoned = false;
        st.prompt_len = init.prompt_len;
        st.stop_ids = init.stop_ids;
        st.sampler = init.sampler;
        st.rng_state = init.rng_state;
        st.history = init.history;
        return Ok((st, cks, start));
    }
    Ok((alloc_owner(gpu, b, epoch, slot, init)?, Vec::new(), 0))
}

// ── Think control ─────────────────────────────────────────────────────

/// Post-latch answer bound (`ar.rs`): hard EOS this many tokens after the
/// think cap latched.
pub fn post_latch_answer_budget() -> usize {
    hipfire_config::developer_var("HIPFIRE_POST_LATCH_ANSWER_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(768)
}

/// Per-request think-budget state of the singleton AR decode loop.
#[derive(Clone, Debug, Default)]
pub struct ThinkCtl {
    pub max_think_tokens: usize,
    pub max_total_think: usize,
    pub post_latch_budget: usize,
    pub think_count: usize,
    pub prev_in_think: bool,
    pub total_think_tokens: usize,
    pub force_answer_latched: bool,
    pub latch_gen_mark: Option<usize>,
}

/// What the decode loop does after a committed token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkAction {
    Continue,
    /// Hard EOS (total-think margin or post-latch bound).
    Eos,
    /// Force-close the think span: commit this many continuation tokens.
    Close(usize),
}

impl ThinkCtl {
    pub fn new(max_think_tokens: usize) -> Self {
        Self {
            max_think_tokens,
            max_total_think: hipfire_runtime::config::get().max_total_think_tokens,
            post_latch_budget: post_latch_answer_budget(),
            ..Self::default()
        }
    }

    /// Request carries a think budget (per-block or total).
    pub fn budgeted(&self) -> bool {
        self.max_think_tokens > 0 || self.max_total_think > 0
    }

    /// `<think>` is blocked at sampling once force-answer latched.
    pub fn blocks_think_open(&self) -> bool {
        self.force_answer_latched
    }

    /// The singleton loop's think block after committing a token that was
    /// neither a stop nor EOS: `stream` yields every committed byte (decoded
    /// only when a budget or force-answer is in play), `generated` is the
    /// committed count, `close_len` the continuation's token count.
    pub fn step(
        &mut self,
        id: &str,
        stream: impl FnOnce() -> Vec<u8>,
        started_in_think: bool,
        generated: usize,
        max_tokens: usize,
        close_len: usize,
    ) -> ThinkAction {
        let force_answer_now = hipfire_engine::terminal::check_force_answer(id);
        if force_answer_now {
            self.force_answer_latched = true;
        }
        if !(self.max_think_tokens > 0
            || force_answer_now
            || self.force_answer_latched
            || self.max_total_think > 0)
        {
            return ThinkAction::Continue;
        }
        let bytes = stream();
        let raw_str = std::str::from_utf8(&bytes).unwrap_or("");
        let in_think = hipfire_runtime::emit_text::currently_in_think(raw_str, started_in_think);
        if in_think {
            self.total_think_tokens += 1;
        }
        if self.max_total_think > 0 && self.total_think_tokens >= self.max_total_think {
            self.force_answer_latched = true;
        }
        if self.force_answer_latched && self.latch_gen_mark.is_none() {
            self.latch_gen_mark = Some(generated);
        }
        if self.max_total_think > 0
            && in_think
            && self.total_think_tokens >= self.max_total_think + 256
        {
            eprintln!(
                "[think-cap] id={} — total think {} exceeded cap {}+256 while still thinking; forcing EOS",
                id, self.total_think_tokens, self.max_total_think
            );
            return ThinkAction::Eos;
        }
        if let Some(mark) = self.latch_gen_mark {
            if generated.saturating_sub(mark) >= self.post_latch_budget {
                eprintln!(
                    "[think-cap] id={} — {} tokens since think-cap latch without finishing; forcing EOS",
                    id,
                    generated.saturating_sub(mark)
                );
                return ThinkAction::Eos;
            }
        }
        if self.max_think_tokens > 0 {
            if in_think {
                if !self.prev_in_think {
                    self.think_count = 1;
                } else {
                    self.think_count += 1;
                }
            } else {
                self.think_count = 0;
            }
            self.prev_in_think = in_think;
        }
        let budget_hit = self.max_think_tokens > 0 && self.think_count >= self.max_think_tokens;
        let latched_now = crate::common::latch_request_think_cap(
            budget_hit,
            generated,
            &mut self.force_answer_latched,
            &mut self.latch_gen_mark,
        );
        if in_think && (budget_hit || force_answer_now || self.force_answer_latched) {
            if latched_now {
                eprintln!(
                    "[think-cap] id={} — per-request think cap {} reached; closing <think>",
                    id, self.max_think_tokens
                );
            } else if force_answer_now {
                eprintln!(
                    "[force-answer] id={} — closing <think> mid-turn to commit to the answer",
                    id
                );
            }
            self.think_count = 0;
            self.prev_in_think = false;
            return ThinkAction::Close(close_len.min(max_tokens.saturating_sub(generated)));
        }
        ThinkAction::Continue
    }
}

/// Singleton conversation state a promoted request carries into its lane.
#[derive(Default)]
pub struct PromotedConv {
    /// Think-budget state at the promotion boundary (`None`: a fresh one
    /// from the request's `max_think_tokens`).
    pub think: Option<ThinkCtl>,
    pub checkpoints: Checkpoints,
    /// Prompt tokens the singleton reused (`cached_tokens` of the done).
    pub cached_tokens: usize,
    /// Prompt tokens the singleton prefilled (`prefill_tokens` of the done);
    /// 0 = the lane's `prompt_len`.
    pub prefill_tokens: usize,
}
