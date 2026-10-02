// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt and hipfire contributors

//! DFlash2 speculative decode for the multi-slot serve engine.
//!
//! The slot engine owns the target's batched forward; the DFlash draft runs
//! per-slot between the batched steps:
//!
//! 1. `dflash_slot_draft_step` — after a slot's prefill completes (or after
//!    the previous verify committed), run the DFlash2 draft forward over the
//!    slot's `target_hidden` context + the noise-embedded block
//!    `[seed, MASK × (B-1)]`, then the candidate selector proposes B−1 draft
//!    tokens. The returned [`DflashSlotDraft`] carries the verify row tokens
//!    `[seed, drafts…]` the caller injects into the next `SlotBatch`.
//! 2. `dflash_slot_verify_accept` — after the batched forward, norm the
//!    slot's verify rows out of `pbs.x_batch`, run the trunk lm_head over
//!    them, greedy-accept the draft prefix, repair the DeltaNet state on a
//!    partial accept (snapshot restore + tape replay), and scatter the kept
//!    rows' extract-layer hiddens from the step's staging into the slot's
//!    `target_hidden` ring.
//!
//! Differences vs the sequential `DflashSpeculator` path
//! (`dflash_spec.rs`):
//!
//! * no `HiddenStateRingBuffer` / `target_hidden_host` — the batched
//!   forward's `x_batch` IS the post-layer hidden source; extract layers are
//!   captured into shared staging by [`crate::forward_slots::SpecVerifyCapture`]
//!   during the forward and scattered into the slot's interleaved
//!   `target_hidden` after accept;
//! * no DDTree, no PLD spine, no CACTUS, no ngram-block, no repeat-penalty —
//!   the slot path is the DFlash2 selector chain only (the only draft shape
//!   admitted by [`load_dflash_shared`]);
//! * greedy verify only (temp==0 admission is enforced by the caller, the
//!   same gate MTP uses);
//! * windowed draft context is honoured for all-sliding DFlash2 drafts
//!   (their seed backfill is a no-op); split drafts fall back to Legacy
//!   capped mode because `draft_seed_backfill` needs a host hidden shadow
//!   the slot path does not maintain.

use hip_bridge::HipResult;
use hipfire_runtime::dflash::{self, DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::spec::accept_greedy_prefix;
use rdna_compute::slot_pool::SlotId;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

use crate::forward_slots::SpecHiddenCapture;
use crate::qwen35::{DeltaNetState, PrefillBatchScratch, Qwen35Config, Qwen35Weights};
use crate::speculative::{DeltaNetSnapshot, GdnTape, VerifyScratch};

/// Draft-side state shared by every DFlash slot: the draft weights and the
/// resolved draft-context policy. One per engine load; `None` when DFlash
/// is off. (The draft's lm_head is the trunk's `weights.output` — the draft
/// artifact carries no separate output projection.)
pub struct DflashShared {
    pub draft_config: DflashConfig,
    pub draft_weights: DflashWeights,
    /// Runtime block size B (verify rows per cycle = B).
    pub block_size: usize,
    /// Draft-context window: `Some(w)` for all-sliding windowed drafts,
    /// `None` for Legacy (capped) mode.
    pub window: Option<usize>,
    /// Row capacity of each slot's `target_hidden` ring / ctx K/V caches.
    /// Legacy mode: `min(requested_ctx, HIPFIRE_DFLASH_CTX_CAP | 8192)`.
    /// Windowed: `w`.
    pub ctx_capacity: usize,
    /// The engine's per-slot token capacity (`cap_tokens`) — the draft's
    /// hard context bound. In windowed mode this is `new_windowed`'s
    /// `max_ctx`: the committed length `l` grows past the window `w` (the
    /// rings wrap), so `max_ctx_len` must be the physical cap, NOT `w`.
    /// Passing `w` here froze `l` at the window and tripped
    /// `draft_forward_opts`'s `assert!(l <= max_ctx_len)` on any request
    /// whose committed length crossed it.
    pub physical_ctx: usize,
    /// Per-extract-layer staging buffers the batched forward fills via
    /// [`SpecHiddenCapture`]; sized `max_batch × dim` each. Shared across
    /// slots because staging holds the step's rows (slots scatter their own
    /// row range out of it after accept).
    pub hidden_staging: Vec<GpuTensor>,
    /// Row capacity of each staging buffer (= `max_batch` of the engine's
    /// `PrefillBatchScratch`).
    pub staging_rows: usize,
}

impl DflashShared {
    /// Build the `SpecHiddenCapture` view the batched forward consumes.
    /// `dim` is the target hidden size (row stride of `x_batch`).
    pub fn hidden_capture(&self, dim: usize) -> SpecHiddenCapture<'_> {
        SpecHiddenCapture {
            staging: &self.hidden_staging,
            extract_layers: &self.draft_config.target_layer_ids,
            dim,
        }
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.draft_weights.free_gpu(gpu);
        for t in self.hidden_staging {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Per-slot DFlash2 state. Allocated lazily on the slot's first post-prefill
/// draft step (mirrors `mtp_states`); dropped when the slot is released.
pub struct DflashSlotState {
    /// Draft-side scratch: `target_hidden` interleaved ring, per-layer ctx
    /// K/V caches, block K/V concat buffers, thlog watermarks.
    pub scratch: DflashScratch,
    /// Trunk verify scratch: `final_hidden` (post-norm verify rows),
    /// `logits`, `rot` (FWHT), `argmax`. `prefill_batch` stays `None` — the
    /// slot engine's own `pbs` runs the verify forward.
    pub verify_scratch: VerifyScratch,
    /// Trunk DN snapshot taken before each verify forward; restored on
    /// partial accept before the tape replay.
    pub trunk_snap: DeltaNetSnapshot,
    /// Session this draft context belongs to (None = fresh/unseeded). A
    /// continuation request on the same session reuses the seeded prefix;
    /// any other occupant forces a reset.
    pub session: Option<u64>,
    /// Rows of `target_hidden` already seeded/committed — the thlog
    /// watermark mirror used to detect prefix reuse.
    pub seeded_through: usize,
    /// In-flight draft output between the draft step and the verify accept.
    pub draft: Option<DflashSlotDraft>,
}

impl DflashSlotState {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.scratch.free_gpu(gpu);
        self.verify_scratch.free_gpu(gpu);
        self.trunk_snap.free_gpu(gpu);
        if let Some(d) = self.draft {
            d.free_gpu(gpu);
        }
    }
}

/// One cycle's draft output: the verify-row tokens and the device buffers
/// the draft forward produced (kept alive until the accept step consumes
/// them — the selector's per-row data is host-side, but `verify_hidden` /
/// `verify_logits` / `verify_rot` / `verify_argmax` are the draft lm_head
/// scratch the accept step reuses for the trunk verify).
pub struct DflashSlotDraft {
    /// `[seed, drafts…]` — the rows the caller injects into `SlotBatch`.
    pub verify_tokens: Vec<u32>,
    /// Absolute position of verify row 0 (the seed's position).
    pub cur_pos: usize,
    /// Post-norm draft hidden rows `[B-1 × dim]` (draft lm_head input rows
    /// 1..B); reused as the trunk verify's post-norm staging.
    pub verify_hidden: GpuTensor,
    /// Draft lm_head logits `[B-1 × vocab]`; reused for trunk verify logits.
    pub verify_logits: GpuTensor,
    /// FWHT-rotated draft hidden `[B-1 × hidden_k]`; reused for trunk rot.
    pub verify_rot: GpuTensor,
    /// Argmax scratch `[B-1]`; reused for trunk verify argmax.
    pub verify_argmax: GpuTensor,
}

impl DflashSlotDraft {
    /// Number of verify rows this draft contributes to the step batch.
    pub fn n_verify(&self) -> usize {
        self.verify_tokens.len()
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.verify_hidden);
        let _ = gpu.free_tensor(self.verify_logits);
        let _ = gpu.free_tensor(self.verify_rot);
        let _ = gpu.free_tensor(self.verify_argmax);
    }
}

/// Load the shared DFlash2 draft state for the slot engine. Mirrors
/// `load_dflash_state` (dflash_spec.rs) minus the sequential-only pieces
/// (hidden ring, host shadow, DDTree, PM4 tape, verify prefill scratch).
///
/// `requested_ctx` is the engine's per-slot token capacity (`cap_tokens`).
/// `max_batch` sizes the shared hidden staging (one step's rows).
///
/// Returns `Err` on any load/alloc failure — the caller treats it like the
/// sequential path does (`dflash_mode=on` fails the load, `auto` warns and
/// serves AR).
pub fn load_dflash_shared(
    gpu: &mut Gpu,
    draft_path: &str,
    target_config: &Qwen35Config,
    requested_ctx: usize,
    max_batch: usize,
) -> Result<DflashShared, String> {
    let draft_hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("{e}"))?;
    let draft_config = DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "draft: failed to parse DflashConfig from HFQ metadata".to_string())?;

    // Window resolution mirrors load_dflash_state: env override wins, else
    // the draft-declared window, else Legacy. Split (non-all-sliding) drafts
    // can't run windowed here — their seed backfill needs a host hidden
    // shadow the slot path doesn't keep — so they fall back to Legacy.
    let window = match hipfire_config::developer_var("HIPFIRE_DFLASH_WINDOW")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(0) => None,
        Some(w) => {
            if let Some(declared) = draft_config.declared_window {
                if declared != w {
                    eprintln!(
                        "  DFlash window override {w} != draft-declared sliding_window \
                         {declared} — the draft was trained at {declared}; acceptance may \
                         degrade (output stays verify-exact)"
                    );
                }
            }
            Some(w)
        }
        None => draft_config.declared_window,
    };
    let window = match window {
        Some(w) if !draft_config.all_layers_sliding => {
            eprintln!(
                "  DFlash windowed mode ({w}) disabled on the slots path: split drafts \
                 need a host hidden shadow for seed backfill — using Legacy capped mode"
            );
            None
        }
        w => w,
    };
    let ctx_capacity = match window {
        Some(w) => w,
        None => match hipfire_config::developer_var("HIPFIRE_DFLASH_CTX_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            Some(0) => requested_ctx,
            Some(cap) => requested_ctx.min(cap),
            None => requested_ctx.min(crate::dflash_spec::DEFAULT_DFLASH_CTX_CAP),
        },
    };
    if let Some(w) = window {
        eprintln!(
            "  DFlash2 draft windowed (slots): all {} layers sliding at W={w} \
             (draft VRAM pinned at W; HIPFIRE_DFLASH_WINDOW=0 for Legacy)",
            draft_config.n_layers
        );
    } else if ctx_capacity < requested_ctx {
        eprintln!(
            "  DFlash draft ctx capped: {requested_ctx} -> {ctx_capacity} rows \
             (draft-side VRAM scales with this; HIPFIRE_DFLASH_CTX_CAP=0 for uncapped)"
        );
    }

    macro_rules! or_free {
        ($e:expr, $ctx:expr $(, $owned:expr)* $(,)?) => {
            match $e {
                Ok(v) => v,
                Err(e) => {
                    $($owned.free_gpu(gpu);)*
                    let ctx: &str = $ctx;
                    return Err(if ctx.is_empty() {
                        format!("{e}")
                    } else {
                        format!("{ctx}: {e}")
                    });
                }
            }
        };
    }

    let draft_weights = or_free!(DflashWeights::load(gpu, &draft_hfq, &draft_config), "");
    if !draft_weights.has_candidate_selector() {
        draft_weights.free_gpu(gpu);
        return Err(
            "slots DFlash requires a DFlash2 draft (candidate selector); \
             legacy DFlash drafts are sequential-path only"
                .into(),
        );
    }
    let block_size = draft_config.runtime_block_size();
    if block_size != draft_config.block_size {
        eprintln!(
            "  DFlash2 runtime block: {} -> {} (selector/conv path is length-generic)",
            draft_config.block_size, block_size
        );
    }
    // Extract layers must index real target layers — same guard as
    // load_dflash_state (checkpoint [5,19,33,47,61] captured verbatim).
    for &lid in &draft_config.target_layer_ids {
        if lid >= target_config.n_layers {
            draft_weights.free_gpu(gpu);
            return Err(format!(
                "draft target_layer_ids contains {lid} >= num_target_layers {}",
                target_config.n_layers
            ));
        }
    }
    // One staging buffer per extract layer, each holding a whole step's rows
    // (the capture copies all `n` rows of `x_batch` per extract layer).
    let mut hidden_staging = Vec::with_capacity(draft_config.target_layer_ids.len());
    for &lid in &draft_config.target_layer_ids {
        match gpu.alloc_tensor(&[max_batch * target_config.dim], DType::F32) {
            Ok(t) => hidden_staging.push(t),
            Err(e) => {
                for t in hidden_staging.drain(..) {
                    let _ = gpu.free_tensor(t);
                }
                draft_weights.free_gpu(gpu);
                return Err(format!("hidden staging (extract layer {lid}): {e}"));
            }
        }
    }
    Ok(DflashShared {
        draft_config,
        draft_weights,
        block_size,
        physical_ctx: requested_ctx,
        window,
        ctx_capacity,
        hidden_staging,
        staging_rows: max_batch,
    })
}

/// Allocate a slot's DFlash2 state. `seed` is the slot's first decoded token
/// (the prefill's last-row argmax/sample); `position` is the committed length
/// after that token (prompt_len + 1 at first call — the seed's own row is
/// committed by the verify cycle that consumes it, so `position` is the
/// prefill length and the seed rides as verify row 0).
pub fn new_dflash_slot_state(
    gpu: &mut Gpu,
    shared: &DflashShared,
    target_config: &Qwen35Config,
    dn_state: &DeltaNetState,
) -> Result<DflashSlotState, String> {
    let b = shared.block_size;
    let scratch = match shared.window {
        Some(w) => DflashScratch::new_windowed(
            gpu,
            &shared.draft_config,
            b,
            w,
            // All-sliding drafts ignore w_full; pass the window for both.
            w,
            // max_ctx is the draft's HARD context bound (the committed
            // length `l` grows past the window `w` — the rings wrap), so it
            // must be the physical per-slot cap, not `w`. Passing `w` here
            // froze `l` at the window and tripped draft_forward_opts's
            // `assert!(l <= max_ctx_len)` once a request crossed it.
            shared.physical_ctx,
            shared.draft_weights.has_mq,
        ),
        None => DflashScratch::new_with_mq(
            gpu,
            &shared.draft_config,
            b,
            shared.ctx_capacity,
            shared.draft_weights.has_mq,
        ),
    }
    .map_err(|e| format!("DflashScratch: {e}"))?;
    let hidden_k = target_config.dim.next_power_of_two();
    let verify_scratch = VerifyScratch::new(
        gpu,
        b,
        target_config.dim,
        target_config.vocab_size,
        hidden_k,
    )
    .map_err(|e| format!("VerifyScratch: {e}"))?;
    let trunk_snap =
        DeltaNetSnapshot::new_for(gpu, dn_state).map_err(|e| format!("DeltaNetSnapshot: {e}"))?;
    Ok(DflashSlotState {
        scratch,
        verify_scratch,
        trunk_snap,
        session: None,
        seeded_through: 0,
        draft: None,
    })
}

/// Reset the slot's draft context for a new occupant: drop the thlog
/// watermarks so the next prefill re-seeds `target_hidden` from row 0.
/// The ctx K/V rings are position-keyed and get overwritten row-by-row —
/// no explicit clear needed (same invariant the sequential path relies on).
pub fn reset_dflash_slot(st: &mut DflashSlotState, session: u64) {
    st.scratch.reset_upload_tracking();
    st.session = Some(session);
    st.seeded_through = 0;
    st.draft = None;
}

/// `target_hidden` is interleaved `[pos % modulus][layer][h]` — one
/// `scatter_hidden_block_to_interleaved`-style copy per extract layer.
pub fn scatter_staging_rows_to_interleaved(
    gpu: &mut Gpu,
    shared: &DflashShared,
    st: &mut DflashSlotState,
    row_off: usize,
    pos: usize,
    n: usize,
) -> HipResult<()> {
    let h = shared.draft_config.hidden;
    let ne = shared.draft_config.num_extract();
    let modulus = st.scratch.ctx_modulus();
    // Legacy (modulus == usize::MAX) writes at the ABSOLUTE position into a
    // `ctx_capacity`-row target_hidden. Bound it here so an over-cap write is
    // a typed Err (the caller retires spec / falls back to AR), not the
    // release `assert!` panic `sub_offset` would raise — the serve-hang class.
    if modulus == usize::MAX {
        // target_hidden is a flat [rows * ne * h] f32 buffer.
        let rows = st.scratch.target_hidden.numel() / (ne * h);
        if pos + n > rows {
            return Err(hip_bridge::HipError::new(
                1,
                &format!(
                    "dflash legacy scatter: pos {pos} + n {n} exceeds target_hidden \
                     rows {rows} (ctx_capacity)"
                ),
            ));
        }
    }
    for (i, staging) in shared.hidden_staging.iter().enumerate() {
        // Source: this extract layer's staging, rows row_off..row_off+n.
        let src = staging.sub_offset(row_off * h, n * h);
        // Dest: interleaved ring — row r lands at slot (pos + r) % modulus,
        // layer plane i. Walk ring segments so a wrap stays contiguous.
        let mut done = 0usize;
        while done < n {
            let abs = pos + done;
            let slot = if modulus == usize::MAX {
                abs
            } else {
                abs % modulus
            };
            let seg = if modulus == usize::MAX {
                n - done
            } else {
                (modulus - slot).min(n - done)
            };
            // Rows are strided in dst (ne*h apart) but contiguous in src —
            // copy row-by-row within the segment.
            for r in 0..seg {
                let dst_row = st.scratch.target_hidden.sub_offset(
                    (if modulus == usize::MAX {
                        abs + r
                    } else {
                        (abs + r) % modulus
                    }) * ne
                        * h
                        + i * h,
                    h,
                );
                let src_row = src.sub_offset((done + r) * h, h);
                gpu.hip
                    .memcpy_dtod_at(&dst_row.buf, 0, &src_row.buf, 0, h * 4)?;
            }
            done += seg;
        }
    }
    Ok(())
}

/// Draft phase for one DFlash slot: run the DFlash2 draft forward over the
/// slot's committed context + the noise-embedded block, then the candidate
/// selector proposes B−1 draft tokens. Returns the draft output whose
/// `verify_tokens` the caller injects into the next `SlotBatch`.
///
/// `seed` is the last committed token; `position` the committed length
/// (both from the caller's `PendingWork`). Saves the trunk DN snapshot for
/// the verify phase's partial-accept rollback.
pub fn dflash_slot_draft_step(
    gpu: &mut Gpu,
    shared: &DflashShared,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    st: &mut DflashSlotState,
    seed: u32,
    position: usize,
) -> Result<DflashSlotDraft, String> {
    let b = shared.block_size;
    let h = shared.draft_config.hidden;
    let dim = config.dim;
    let vocab = config.vocab_size;

    // Block: [seed, MASK × (B-1)] — the draft fills the mask slots.
    let mut block = vec![shared.draft_config.mask_token_id; b];
    block[0] = seed;

    // Noise embeddings: target embed over the block, straight into
    // draft_scratch.x (the fused batched path when the format allows,
    // per-row lookups otherwise — mirrors spec_step_dflash §2).
    if !crate::speculative::build_dflash_noise_embeddings_weights(
        gpu,
        weights,
        &block,
        h,
        &mut st.scratch,
    )
    .map_err(|e| format!("noise embeddings: {e}"))?
    {
        for (i, &tok) in block.iter().enumerate() {
            let dst = st.scratch.x.sub_offset(i * h, h);
            match weights.embd_format {
                hipfire_runtime::llama::EmbeddingFormat::HFQ4G256 => gpu
                    .embedding_lookup_hfq4g256(&weights.token_embd, &dst, tok, h)
                    .map_err(|e| format!("noise embed: {e}"))?,
                hipfire_runtime::llama::EmbeddingFormat::HFQ4G128 => gpu
                    .embedding_lookup_hfq4g128(&weights.token_embd, &dst, tok, h)
                    .map_err(|e| format!("noise embed: {e}"))?,
                hipfire_runtime::llama::EmbeddingFormat::Q8_0 => gpu
                    .embedding_lookup_q8(&weights.token_embd, &dst, tok, h)
                    .map_err(|e| format!("noise embed: {e}"))?,
                hipfire_runtime::llama::EmbeddingFormat::F32 => gpu
                    .embedding_lookup(&weights.token_embd, &dst, tok, h)
                    .map_err(|e| format!("noise embed: {e}"))?,
                _ => {
                    return Err(
                        "dflash slot: unsupported target embedding format for noise lookup".into(),
                    )
                }
            }
        }
    }

    // Positions: Q = [position .. position+B) (slot KV has no compaction —
    // compact_offset is always 0 on the slots path); K = committed abs
    // positions (contiguous — no CASK eviction on slots) + the same block
    // slots.
    //
    // `effective_ctx_len` is capped at `scratch.max_ctx_len` (= the window
    // `w` for all-sliding DFlash2, the ctx cap for Legacy). Without the cap
    // the unbounded `abs_positions` log lets `l` cross `w` once a request's
    // committed length exceeds the window, tripping `draft_forward_opts`'s
    // `assert!(l <= max_ctx_len)` — a panic in the engine loop. The windowed
    // contract is to DEGRADE τ past the window (the K span is the last `w`
    // rows), never to assert. The sequential path gets the same cap from
    // `ctx_slice = Some(window)`; the slot path has no ctx_slice, so the cap
    // lives here.
    let effective_ctx_len = st
        .scratch
        .thlog
        .abs_positions()
        .len()
        .min(position)
        .min(st.scratch.max_ctx_len);
    let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
    let mut positions_k: Vec<i32> = Vec::with_capacity(effective_ctx_len + b);
    {
        let th_abs = st.scratch.thlog.abs_positions();
        let start_idx = th_abs.len().saturating_sub(effective_ctx_len);
        positions_k.extend_from_slice(&th_abs[start_idx..]);
        for p in 0..b {
            positions_k.push(position as i32 + p as i32);
        }
    }

    // Draft forward: target_hidden already lives on GPU (populated by the
    // post-accept scatter / prefill seed), so `target_hidden=None` skips the
    // H2D upload — same fast path as the sequential ctx_slice=None arm.
    dflash::draft_forward_opts(
        gpu,
        &shared.draft_weights,
        &shared.draft_config,
        None,
        None,
        &positions_q,
        &positions_k,
        b,
        effective_ctx_len,
        &mut st.scratch,
        false,
    )
    .map_err(|e| format!("dflash2 draft forward: {e}"))?;

    // Trunk lm_head over draft hidden rows 1..B (the mask slots), then the
    // candidate selector proposes B−1 tokens anchored on the seed.
    let batch = b - 1;
    let hidden_rows = st.scratch.x.sub_offset(h, batch * h);
    // First cycle allocates the per-cycle device buffers; later cycles
    // reuse them (sized to the block).
    let mut draft = match st.draft.take() {
        Some(d) => d,
        None => {
            let hidden_k = dim.next_power_of_two();
            let verify_hidden = gpu
                .alloc_tensor(&[batch * dim], DType::F32)
                .map_err(|e| format!("dflash verify_hidden alloc: {e}"))?;
            let verify_logits = gpu
                .alloc_tensor(&[batch * vocab], DType::F32)
                .map_err(|e| format!("dflash verify_logits alloc: {e}"))?;
            let verify_rot = gpu
                .alloc_tensor(&[batch * hidden_k], DType::F32)
                .map_err(|e| format!("dflash verify_rot alloc: {e}"))?;
            let verify_argmax = gpu
                .alloc_tensor(&[batch], DType::F32)
                .map_err(|e| format!("dflash verify_argmax alloc: {e}"))?;
            DflashSlotDraft {
                verify_tokens: Vec::new(),
                cur_pos: 0,
                verify_hidden,
                verify_logits,
                verify_rot,
                verify_argmax,
            }
        }
    };
    draft.cur_pos = position;

    let logits_view = draft.verify_logits.sub_offset(0, batch * vocab);
    crate::mtp_spec::mtp_trunk_verify_lm_head(
        gpu,
        &weights.output,
        &hidden_rows,
        &draft.verify_rot,
        &logits_view,
        batch,
        dim,
        vocab,
    )
    .map_err(|e| format!("draft lm_head: {e}"))?;

    // Greedy selector proposal (slots path is temp==0 only — admission
    // enforced by the caller, same gate as MTP).
    let proposal = dflash::propose_candidates_device(
        gpu,
        &shared.draft_weights,
        &st.scratch,
        &hidden_rows,
        &logits_view,
        batch,
        seed,
        0.0,
        None,
    )
    .map_err(|e| format!("selector proposal: {e}"))?;

    draft.verify_tokens.clear();
    draft.verify_tokens.push(seed);
    draft
        .verify_tokens
        .extend_from_slice(&proposal.tokens[..batch.min(proposal.tokens.len())]);
    debug_assert_eq!(draft.verify_tokens.len(), b);
    Ok(draft)
}

/// Verify/accept phase for one DFlash slot, after the batched forward that
/// carried the slot's `verify_tokens` rows.
///
/// `hidden_row_offset` is the flat row index where this slot's verify rows
/// begin in the step's `x_batch` (from `batch.m_per_slot` prefix sums).
/// `verify_tape` is the engine's shared `GdnTape` (stride = B per slot).
/// `commit_budget` mirrors MTP: `min(remaining max_tokens, remaining ctx)`.
///
/// Returns `(committed, hit_eos)` where `committed` excludes the seed and
/// includes the bonus — exactly MTP's convention.
#[allow(clippy::too_many_arguments)]
pub fn dflash_slot_verify_accept(
    gpu: &mut Gpu,
    shared: &DflashShared,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &mut DeltaNetState,
    pbs: &PrefillBatchScratch,
    st: &mut DflashSlotState,
    draft: &mut DflashSlotDraft,
    pos: usize,
    hidden_row_offset: usize,
    verify_tape: &GdnTape,
    tape_stride: usize,
    slot: SlotId,
    eos_token_id: u32,
    aux_eot_id: Option<u32>,
    commit_budget: usize,
) -> Result<Vec<u32>, String> {
    let dim = config.dim;
    let vocab = config.vocab_size;
    let b = shared.block_size;
    let n_verify = draft.n_verify();
    debug_assert_eq!(n_verify, b);

    // Post-output-norm the slot's verify rows out of x_batch — the trunk
    // lm_head consumes post-norm hidden (same convention as MTP's
    // mtp_batched_verify_accept_from_batch).
    let verify_raw = pbs
        .x_batch
        .sub_offset(hidden_row_offset * dim, n_verify * dim);
    let verify_hidden = st.verify_scratch.final_hidden.sub_offset(0, n_verify * dim);
    gpu.rmsnorm_batched(
        &verify_raw,
        &weights.output_norm,
        &verify_hidden,
        n_verify,
        dim,
        config.norm_eps,
    )
    .map_err(|e| format!("verify norm: {e}"))?;

    // Trunk lm_head over all B verify rows.
    let logits_view = st.verify_scratch.logits.sub_offset(0, n_verify * vocab);
    crate::mtp_spec::mtp_trunk_verify_lm_head(
        gpu,
        &weights.output,
        &verify_hidden,
        &st.verify_scratch.rot,
        &logits_view,
        n_verify,
        dim,
        vocab,
    )
    .map_err(|e| format!("verify lm_head: {e}"))?;

    // Argmax per verify row, one packed D2H.
    let argmax_v = st.verify_scratch.argmax.sub_offset(0, n_verify);
    gpu.argmax_f32_batched(&logits_view, &argmax_v, vocab, n_verify)
        .map_err(|e| format!("verify argmax: {e}"))?;
    let mut argmax_host: Vec<i32> = vec![0; n_verify];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(argmax_host.as_mut_ptr() as *mut u8, n_verify * 4)
        };
        gpu.hip
            .memcpy_dtoh(bytes, &argmax_v.buf)
            .map_err(|e| format!("verify argmax dtoh: {e}"))?;
    }
    let target_pick: Vec<u32> = argmax_host.into_iter().map(|x| x as u32).collect();

    // Greedy accept: drafts = verify_tokens[1..] (the B−1 proposals);
    // target_pick[i] is the trunk's pick at verify row i. committed =
    // accepted drafts + bonus (seed excluded — it was committed last cycle).
    let accepted =
        accept_greedy_prefix(&draft.verify_tokens[1..], &target_pick, Some(eos_token_id));
    let mut committed = accepted.committed;
    let mut hit_eos = accepted.hit_eos;

    // Clamp to the commit budget and truncate on an auxiliary eot inside the
    // kept prefix — mirrors mtp_batched_verify_accept_from_batch exactly so
    // `advance` (committed.len()) stays the rows the store actually holds.
    let mut keep = committed.len().min(commit_budget.max(1));
    for (i, t) in committed[..keep].iter().enumerate() {
        if i + 1 < keep && Some(*t) == aux_eot_id {
            keep = i + 1;
            hit_eos = true;
            break;
        }
    }
    committed.truncate(keep);
    let advance = committed.len();
    debug_assert!(advance >= 1 && advance <= b);

    // DN repair: full accept without EOS leaves the verify-advanced state
    // exactly right; anything less restores the pre-verify snapshot and
    // replays the taped DeltaNet activations over the kept rows.
    let full_accept_no_eos = advance == b && !hit_eos;
    if !full_accept_no_eos {
        st.trunk_snap
            .restore_to(dn_state, gpu)
            .map_err(|e| format!("dflash dn restore: {e}"))?;
        crate::mtp_spec::mtp_dn_repair_from_tape(
            gpu,
            weights,
            config,
            dn_state,
            verify_tape,
            slot.0 * tape_stride,
            advance,
        )
        .map_err(|e| format!("dflash dn repair: {e}"))?;
    }

    // Commit the kept rows' extract-layer hiddens into the slot's
    // target_hidden ring and advance the thlog. Rows 0..advance of the
    // slot's verify range are the kept ones (seed + accepted drafts; the
    // bonus row's hidden is NOT committed — it becomes next cycle's seed
    // and its hidden is captured by next cycle's verify row 0).
    scatter_staging_rows_to_interleaved(gpu, shared, st, hidden_row_offset, pos, advance)
        .map_err(|e| format!("hidden scatter: {e}"))?;
    st.scratch.thlog.append_committed(pos, advance, 0);
    st.seeded_through = pos + advance;

    draft.verify_tokens.clear();
    Ok(committed)
}
