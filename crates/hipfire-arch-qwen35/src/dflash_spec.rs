// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Qwen3.5 DFlash / DDTree speculative-decode state and `Speculator` impl.
//!
//! Contents: [`DflashState`] (the loaded draft weights/scratch + target
//! snapshot/tape + optional [`DdtreeState`]), [`load_dflash_state`] (its
//! load-time constructor), the [`DflashSpeculator`] impl (which owns
//! `DflashState` + the divergent-render checkpoint ring) behind the arch-generic
//! [`Speculator`] trait, and [`build_dflash_speculator`] (its env-resolving
//! constructor). All types here are qwen35 + runtime types — no loader types —
//! so the loader only calls in; it never owns the DFlash mechanics.
use crate::dflash_verify_pm4::{DflashVerifyPm4, DFLASH_VERIFY_PM4_BLOCK};
use crate::qwen35::{self, DeltaNetState, Qwen35Config, Qwen35Weights, StateQuant};
use crate::speculative::{
    apply_eviction_retain_to_draft, apply_host_ar_trunc, sample_categorical,
    scatter_hidden_block_to_interleaved, scatter_hidden_prefill_to_interleaved,
    seed_target_hidden_dense_tp2_abortable, seed_target_hidden_from_prompt_abortable,
    seed_target_hidden_from_prompt_abortable_parts, seed_target_hidden_suffix_abortable,
    seed_target_hidden_suffix_abortable_parts, softmax_temp_into, spec_step_ddtree_batched,
    spec_step_dflash_dense_tp2, spec_step_dflash_with_railgun, xorshift_next_unit, DdtreeScratch,
    DeltaNetSnapshot, DenseTpTargetView, DflashLookup, DflashTargetParts, GdnTape,
    HiddenStateRingBuffer, ModelSlot, RailgunStep, SpecStepResult, VerifyScratch,
};
use hipfire_runtime::dflash::{
    draft_seed_backfill_reads_host, DflashConfig, DflashScratch, DflashWeights, DraftCtxMode,
    TargetHiddenLogMark,
};
use hipfire_runtime::dflash_adaptive_block::DflashAdaptiveBlock;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::spec::{
    request_rng_state, terminal_prefix_replay, EvictRetain, PrefillOutcome, SpecGrammar,
    SpecRequestConfig, SpecStep, SpecTarget, Speculator,
};
use rdna_compute::{Gpu, GpuTensor};
use std::path::Path;
use std::sync::Arc;

/// Extract layers the retained B=16 DFlash2 verify route admits.
const DFLASH_VERIFY_PM4_EXTRACT_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];

// ─── DDTree side state ────────────────────────────────────────────────

/// Side state for DDTree-mode speculative decoding.
pub struct DdtreeState {
    pub post_seed_snap: DeltaNetSnapshot,
    pub scratch: DdtreeScratch,
    pub budget: usize,
    pub topk: usize,
}

// ─── DFlash state ─────────────────────────────────────────────────────

/// Optional DFlash speculative-decoding state.
pub struct DflashState {
    pub draft_config: DflashConfig,
    /// Shared draft weights: the singleton and VMM engine lanes hold
    /// references to the SAME resident GPU weights; only the final owner frees
    /// them ([`release_shared_dflash_weights`]).
    pub draft_weights: Arc<DflashWeights>,
    pub draft_scratch: DflashScratch,
    pub hidden_rb: HiddenStateRingBuffer,
    pub verify_scratch: VerifyScratch,
    pub target_snap: DeltaNetSnapshot,
    pub gdn_tape: GdnTape,
    pub ctx_capacity: usize,
    pub block_size: usize,
    pub ddtree: Option<DdtreeState>,
    /// Retained-PM4 route for the fixed B=16 chain target-verify forward.
    /// Always present; `Disabled` when admission fails (no controller held).
    pub verify_pm4: DflashVerifyPm4,
}

impl DflashState {
    /// Destructured without `..` on purpose: a field added later that owns GPU
    /// memory becomes a compile error here instead of a per-load leak.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let DflashState {
            draft_config: _,
            draft_weights,
            draft_scratch,
            hidden_rb,
            verify_scratch,
            target_snap,
            gdn_tape,
            ctx_capacity: _,
            block_size: _,
            ddtree,
            mut verify_pm4,
        } = self;
        // Must prove no retained IB is in flight before any captured owner is
        // freed. Unknown quiescence → intentional leak; daemon restart is the
        // containment path.
        if let Err(reason) = verify_pm4.shutdown() {
            eprintln!("dflash verify PM4: refusing free after unknown quiescence: {reason}");
            std::mem::forget((
                verify_pm4,
                draft_weights,
                draft_scratch,
                hidden_rb,
                verify_scratch,
                target_snap,
                gdn_tape,
                ddtree,
            ));
            let _ = gpu;
            return;
        }
        drop(verify_pm4);
        release_shared_dflash_weights(gpu, draft_weights);
        draft_scratch.free_gpu(gpu);
        hidden_rb.free_gpu(gpu);
        verify_scratch.free_gpu(gpu);
        target_snap.free_gpu(gpu);
        gdn_tape.free_gpu(gpu);
        if let Some(dd) = ddtree {
            dd.post_seed_snap.free_gpu(gpu);
            dd.scratch.free_gpu(gpu);
        }
    }

    /// Free the scratch a VMM lane never needs and that a promoted singleton
    /// owner still carries: the draft scratch's FWHT rotation buffers
    /// (`mq_x_rot`/`mq_x_rot_f16`, borrowed from the engine's shared draft
    /// scratch instead) and `verify_scratch.prefill_batch` (a lane's target
    /// forward runs on the shared trunk scratch). Idempotent.
    ///
    /// Synchronizes first so no queued kernel still reads a buffer being
    /// freed. A captured draft-FFN graph would bake the freed pointers, so
    /// any is destroyed exactly as [`DflashScratch::free_gpu`] does and the
    /// warmed-up sets are cleared (the draft then re-warms on its next use;
    /// VMM lanes never capture).
    pub fn shed_vmm_lane_scratch(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("dflash lane scratch shed: {e}"))?;
        let ds = &mut self.draft_scratch;
        for per_layer in ds.draft_ffn_graphs.iter_mut() {
            for (_, (graph, exec, _blobs)) in per_layer.drain() {
                let _ = gpu.hip.graph_exec_destroy(exec);
                let _ = gpu.hip.graph_destroy(graph);
            }
        }
        for warmed in ds.draft_ffn_warmed_up.iter_mut() {
            warmed.clear();
        }
        if let Some(t) = ds.mq_x_rot.take() {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = ds.mq_x_rot_f16.take() {
            let _ = gpu.free_tensor(t);
        }
        if let Some(pbs) = self.verify_scratch.prefill_batch.take() {
            let _ = pbs.free_gpu(gpu);
        }
        Ok(())
    }
}

/// The per-lane mutable DFlash owners (everything but the shared weights and
/// config). One allocation recipe for the singleton load and VMM lanes.
struct DflashOwners {
    draft_scratch: DflashScratch,
    hidden_rb: HiddenStateRingBuffer,
    verify_scratch: VerifyScratch,
    target_snap: DeltaNetSnapshot,
    gdn_tape: GdnTape,
}

impl DflashOwners {
    fn free_gpu(self, gpu: &mut Gpu) {
        let DflashOwners {
            draft_scratch,
            hidden_rb,
            verify_scratch,
            target_snap,
            gdn_tape,
        } = self;
        gdn_tape.free_gpu(gpu);
        target_snap.free_gpu(gpu);
        verify_scratch.free_gpu(gpu);
        hidden_rb.free_gpu(gpu);
        draft_scratch.free_gpu(gpu);
    }
}

/// Which role a set of DFlash owners plays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DflashOwnerKind {
    /// The singleton speculator: its draft scratch owns the FWHT rotation
    /// buffers (`mq_x_rot`/`mq_x_rot_f16`) and its verify scratch owns a
    /// persistent `PrefillBatchScratch` (the singleton runs the target
    /// verify forward through it).
    Singleton,
    /// A VMM continuous-batching lane: no rotation buffers (the engine's
    /// shared draft scratch lends them per call) and no `prefill_batch` (the
    /// batched verify runs on the shared trunk scratch; the lane's own
    /// `VerifyScratch` only holds its head rows). Every other owner, shape
    /// and allocation order is the singleton's.
    VmmLane,
}

/// Allocate the lane owners in the singleton's order, freeing everything
/// already allocated on any failure. `max_ctx` is the draft scratch's logical
/// context (`new_windowed`'s `max_ctx`; the Legacy scratch rows);
/// `ring_positions` the hidden-ring rows (= host hidden log rows). `kind`
/// selects the scratch the role needs ([`DflashOwnerKind`]).
#[allow(clippy::too_many_arguments)]
fn alloc_dflash_owners(
    gpu: &mut Gpu,
    draft_config: &DflashConfig,
    has_mq: bool,
    block_size: usize,
    ctx_mode: DraftCtxMode,
    max_ctx: usize,
    ring_positions: usize,
    staging_max_batch: usize,
    verify_max_n: usize,
    tape_max_n: usize,
    target_config: &Qwen35Config,
    target_dn: &DeltaNetState,
    kind: DflashOwnerKind,
) -> Result<DflashOwners, String> {
    // `with_mq` allocates the FWHT rotation scratch (mq_x_rot) that
    // `gemm_dispatch` requires for MQ4/MQ3/MQ6 draft weights. A VMM lane
    // leaves it out: the engine's shared draft scratch lends its identically
    // sized buffers for every lane draft call.
    let has_mq = has_mq && kind == DflashOwnerKind::Singleton;
    let draft_scratch = match ctx_mode {
        DraftCtxMode::Windowed { w, w_full } => {
            DflashScratch::new_windowed(gpu, draft_config, block_size, w, w_full, max_ctx, has_mq)
        }
        DraftCtxMode::Legacy => {
            DflashScratch::new_with_mq(gpu, draft_config, block_size, max_ctx, has_mq)
        }
    }
    .map_err(|e| format!("{e}"))?;
    for &lid in &draft_config.target_layer_ids {
        if lid >= target_config.n_layers {
            draft_scratch.free_gpu(gpu);
            return Err(format!(
                "draft target_layer_ids contains {} >= num_target_layers {}",
                lid, target_config.n_layers
            ));
        }
    }
    let hidden_rb = match HiddenStateRingBuffer::new_for_layers(
        gpu,
        &draft_config.target_layer_ids,
        target_config.dim,
        ring_positions,
        staging_max_batch,
    ) {
        Ok(v) => v,
        Err(e) => {
            draft_scratch.free_gpu(gpu);
            return Err(format!("HiddenStateRingBuffer::new_for_layers: {e}"));
        }
    };
    let hidden_k = target_config.dim.next_power_of_two();
    let verify_scratch = match kind {
        DflashOwnerKind::Singleton => VerifyScratch::with_prefill(
            gpu,
            verify_max_n,
            target_config.dim,
            target_config.vocab_size,
            hidden_k,
            target_config,
        ),
        DflashOwnerKind::VmmLane => VerifyScratch::new(
            gpu,
            verify_max_n,
            target_config.dim,
            target_config.vocab_size,
            hidden_k,
        ),
    };
    let verify_scratch = match verify_scratch {
        Ok(v) => v,
        Err(e) => {
            hidden_rb.free_gpu(gpu);
            draft_scratch.free_gpu(gpu);
            return Err(format!("VerifyScratch alloc: {e}"));
        }
    };
    let target_snap = match DeltaNetSnapshot::new_for(gpu, target_dn) {
        Ok(v) => v,
        Err(e) => {
            verify_scratch.free_gpu(gpu);
            hidden_rb.free_gpu(gpu);
            draft_scratch.free_gpu(gpu);
            return Err(format!("DeltaNetSnapshot::new_for: {e}"));
        }
    };
    let gdn_tape = match GdnTape::new_for_dflash(gpu, target_config, tape_max_n) {
        Ok(v) => v,
        Err(e) => {
            target_snap.free_gpu(gpu);
            verify_scratch.free_gpu(gpu);
            hidden_rb.free_gpu(gpu);
            draft_scratch.free_gpu(gpu);
            return Err(format!("GdnTape::new_for_dflash: {e}"));
        }
    };
    Ok(DflashOwners {
        draft_scratch,
        hidden_rb,
        verify_scratch,
        target_snap,
        gdn_tape,
    })
}

// ─── DFlash state load ────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
/// Default ceiling for the DFlash draft's context-indexed structures
/// (`target_hidden` [L × extract×hidden], the per-layer K/V caches, the
/// hidden ring, `mq_x_rot`, and the host hidden log). Serve loads default
/// `max_seq` to 32768+, which sized ALL of these to 32K rows — on a 27B
/// target with a 5-layer-extract MQ4 draft that is ~11 GB of draft-side
/// VRAM, vs ~1.4 GB at the ≤4K contexts DFlash benches actually run. The
/// draft only affects acceptance rate (verify is target-gated), so a
/// request that outgrows the cap simply falls back to AR in the daemon —
/// emitted tokens are never at risk. `HIPFIRE_DFLASH_CTX_CAP=0` opts out
/// (legacy uncapped behaviour); any other value overrides the ceiling.
pub const DEFAULT_DFLASH_CTX_CAP: usize = 8192;

// lifecycle: deprecated since 0.4.0, removal 0.5.0 — legacy contiguous DFlash opt-in knobs; the draft-declared window supersedes them
/// One warning line when an operator opts into legacy contiguous DFlash
/// (`HIPFIRE_DFLASH_WINDOW=0`) or overrides its draft cap
/// (`HIPFIRE_DFLASH_CTX_CAP`). Silent when neither knob is set.
fn warn_deprecated_legacy_dflash_knobs() {
    let window_zero = hipfire_config::developer_var("HIPFIRE_DFLASH_WINDOW")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        == Some(0);
    let ctx_cap = hipfire_config::developer_var("HIPFIRE_DFLASH_CTX_CAP").is_ok();
    if window_zero || ctx_cap {
        eprintln!(
            "  warning: HIPFIRE_DFLASH_WINDOW=0 / HIPFIRE_DFLASH_CTX_CAP (legacy contiguous DFlash) are deprecated and will be removed in 0.5.0; use the draft-declared window"
        );
    }
}

pub fn load_dflash_state(
    draft_path: &str,
    ctx_capacity: usize,
    target_config: &Qwen35Config,
    target_dn: &DeltaNetState,
    gpu: &mut Gpu,
    // DDTree draft tuning forwarded by the loader from the unified spec config
    // (CLI `--ddtree-budget` / `--ddtree-topk`). Env wins, else these, else default.
    ddtree_budget_param: Option<usize>,
    ddtree_topk_param: Option<usize>,
    // CASK eviction active for this load. Windowed draft mode refuses the
    // combination (the eviction rebuild re-projects rows the window has
    // already dropped) and falls back to Legacy — gather-compact over the
    // rings is a follow-up.
    eviction_active: bool,
    // Retained-PM4 admission facts owned by the loader/target, not the draft.
    target_weights: &Qwen35Weights,
    kv_is_q8: bool,
    // Target KV is a VMM owner; the gfx1100 split verifier admits VMM KV only.
    kv_is_vmm: bool,
    target_flash_partials_numel: usize,
    single_gpu: bool,
    // True when adaptive KV is engaged for this load (tier-switching cache).
    // Must be false for retained-PM4 admission.
    adaptive_kv: bool,
    // `speculation.dflash_block`, resolved per serving configuration
    // (`DflashBlockPolicy::resolve`). Fixes the proposal width of the
    // singleton AND of every VMM lane built from this state.
    dflash_block: usize,
) -> Result<DflashState, String> {
    // The IU4 and PM FP8 verify opt-ins are two different verify arithmetics;
    // both forced on has no defined route, so the load fails before any
    // allocation (the final decision below uses the runtime block).
    Gpu::iu4_verify_load_decision(&gpu.flags, dflash_block)?;
    let requested_ctx = ctx_capacity;
    // Open the draft container up-front: its declared SWA window is the
    // DEFAULT window (below), so the artifact must be parsed before the
    // windowed-vs-Legacy decision.
    let draft_hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("{e}"))?;
    let draft_config = DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "draft: failed to parse DflashConfig from HFQ metadata".to_string())?;
    if draft_config.target_streams_per_layer != 1 {
        return Err(format!(
            "DFlash draft expects {} target streams per layer; this target captures one",
            draft_config.target_streams_per_layer
        ));
    }

    // Windowed draft context:
    //   - Legacy DFlash (n−1 sliding + last full): layers 0..n−2 attend over
    //     the last `window` rows, the last layer over the entire supported
    //     context (`w_full = requested_ctx`). Draft VRAM pins at W.
    //   - DFlash2 (all layers sliding): every extract layer shares the same
    //     W ring; there is no full-attention last layer. `new_windowed`
    //     already skips the long-reach ring when `all_layers_sliding`.
    // Requests past the window degrade τ instead of hitting Legacy AR fallback.
    //
    // The window DEFAULTS to what the draft artifact declares it was trained
    // with (`config.sliding_window`, honoured only when `use_sliding_window`
    // is true and `layer_types` match a split we implement — see
    // `DflashConfig::declared_window` / `all_layers_sliding`). That is the
    // only width correct by construction.
    //   HIPFIRE_DFLASH_WINDOW=<rows>  explicit override (warns on mismatch)
    //   HIPFIRE_DFLASH_WINDOW=0       explicit Legacy (cap + AR fallback)
    //   unset                         draft-declared window, else Legacy
    warn_deprecated_legacy_dflash_knobs();
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
    let window = match (window, eviction_active) {
        (Some(w), true) => {
            eprintln!(
                "  DFlash windowed mode ({w}) disabled: CASK eviction rebuild is not \
                 ring-aware — falling back to Legacy capped mode"
            );
            None
        }
        (w, _) => w,
    };
    let ctx_capacity = match window {
        Some(w) => w,
        None => match hipfire_config::developer_var("HIPFIRE_DFLASH_CTX_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            Some(0) => ctx_capacity, // explicit opt-out: legacy uncapped
            Some(cap) => ctx_capacity.min(cap),
            None => ctx_capacity.min(DEFAULT_DFLASH_CTX_CAP),
        },
    };
    if let Some(w) = window {
        let from_meta = if draft_config.declared_window == Some(w) {
            " [from draft metadata]"
        } else {
            ""
        };
        if draft_config.all_layers_sliding {
            eprintln!(
                "  DFlash2 draft windowed: all {} layers sliding at W={w} \
                 (no full-attention layer; draft VRAM pinned at W; \
                 HIPFIRE_DFLASH_WINDOW=0 for Legacy){from_meta}",
                draft_config.n_layers
            );
        } else {
            eprintln!(
                "  DFlash draft windowed: SWA W={w} rows on layers 0..n-2, full-attention \
                 last layer over all {} rows (draft VRAM pinned at W; HIPFIRE_DFLASH_WINDOW=0 for Legacy){from_meta}",
                requested_ctx
            );
        }
    } else if ctx_capacity < requested_ctx {
        eprintln!(
            "  DFlash draft ctx capped: {} -> {} rows (draft-side VRAM scales with this; \
             HIPFIRE_DFLASH_CTX_CAP=0 for uncapped, or set a larger cap)",
            requested_ctx, ctx_capacity
        );
    }
    // Every step below owns GPU memory the later ones need. A bare `?` drops
    // those without freeing (no `Drop` on the GPU-owning types), so a failed
    // DFlash load stays resident and the AR fallback it announces then OOMs.
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
    let block_size = draft_config.runtime_block_size(dflash_block);
    if block_size != draft_config.block_size {
        eprintln!(
            "  DFlash2 runtime block: {} -> {} (speculation.dflash_block; selector/conv path is length-generic)",
            draft_config.block_size, block_size
        );
    }
    // IU4 verify (numerics differ from the F16 verify; quality tied on task
    // evals): on by default when this load's proposal width is 8 (a batching
    // serve), `HIPFIRE_CB_VERIFY_IU4` forces/kills, FP8 opt-in wins the
    // default. Batched == singleton stays byte-identical within the config.
    // Only the both-forced case errs, refused above before any allocation.
    gpu.iu4_verify_load = Gpu::iu4_verify_load_decision(&gpu.flags, block_size).unwrap_or(false);
    eprintln!(
        "  DFlash verify IU4: {} (block {block_size}, HIPFIRE_CB_VERIFY_IU4={})",
        if gpu.iu4_verify_load { "on" } else { "off" },
        match gpu.flags.cb_verify_iu4 {
            None => "unset",
            Some(true) => "1",
            Some(false) => "0",
        }
    );
    // DDTree verify batches up to `budget + 1` slots (seed + budget nodes), which
    // can exceed the chain block_size+1. Size verify_scratch / GdnTape / hidden
    // staging for the larger of the two so ddtree-mode serve doesn't overflow
    // ("verify_scratch max_n < b" panic). budget=0 ⇒ chain-only, unchanged.
    // Resolved through FeatureFlags (env override) so the ddtree budget has a
    // single parser shared with the dense path — env wins, else the CLI param,
    // else 0 (chain-only). An explicit `HIPFIRE_DDTREE_BUDGET=0` reads as None
    // (unset) here and falls through to the param, matching the dense semantics.
    let mut ddtree_budget: usize = gpu.flags.ddtree_budget.or(ddtree_budget_param).unwrap_or(0);
    // DFlash2 is chain-only: never construct DDTree (independent-marginal
    // tree would bypass the candidate selector).
    if draft_weights.has_candidate_selector() {
        if ddtree_budget > 0 {
            eprintln!(
                "  DFlash2 candidate selector: DDTree disabled (chain-only; \
                 independent-marginal tree would bypass the selector)"
            );
        }
        ddtree_budget = 0;
    }
    let max_n = (block_size + 1).max(ddtree_budget + 1);
    // The hidden-ring STAGING buffers must hold one staged chunk. Verify
    // cycles stage only `max_n` (= block_size+1) rows. The prompt seed
    // (`seed_target_hidden_from_prompt_abortable`) follows AR's chunk plan and
    // writes chunks wider than staging straight to the ring, but its legacy
    // route (`HIPFIRE_DFLASH_LEGACY_PREFILL=1`) stages chunks of up to
    // `PREFILL_MAX_BATCH` via `write_rows_to_staging`. Sizing staging to only
    // `max_n` overflowed the d2d copy there on any prompt longer than
    // block_size+1 tokens. Size it to the larger of the two so both fit.
    let staging_max_batch = max_n.max(qwen35::PREFILL_MAX_BATCH);
    // `with_mq` (inside `alloc_dflash_owners`) allocates the FWHT rotation
    // scratch (mq_x_rot) that `gemm_dispatch` requires for MQ4/MQ3/MQ6 draft
    // weights. Hidden extraction uses checkpoint target_layer_ids exactly —
    // validated against the target layer range and allocated by the explicit
    // list (not the evenly-spaced fallback), so [5,19,33,47,61] is captured
    // verbatim.
    let (ctx_mode, draft_max_ctx) = match window {
        // Split drafts: last (full-attention) layer's ring spans the whole
        // supported context. All-sliding DFlash2: `new_windowed` ignores
        // w_full and pins every layer at W.
        Some(w) => (
            DraftCtxMode::Windowed {
                w,
                w_full: requested_ctx,
            },
            requested_ctx,
        ),
        None => (DraftCtxMode::Legacy, ctx_capacity),
    };
    let owners = match alloc_dflash_owners(
        gpu,
        &draft_config,
        draft_weights.has_mq,
        block_size,
        ctx_mode,
        draft_max_ctx,
        ctx_capacity,
        staging_max_batch,
        max_n,
        max_n,
        target_config,
        target_dn,
        DflashOwnerKind::Singleton,
    ) {
        Ok(owners) => owners,
        Err(e) => {
            draft_weights.free_gpu(gpu);
            return Err(e);
        }
    };
    let _ = draft_hfq;
    // DDTree (budget read once above, used for scratch sizing).
    let ddtree = if ddtree_budget > 0 {
        let topk: usize = gpu.flags.ddtree_topk.or(ddtree_topk_param).unwrap_or(4);
        let post_seed_snap = or_free!(
            DeltaNetSnapshot::new_for(gpu, target_dn),
            "",
            owners,
            draft_weights,
        );
        let scratch = or_free!(
            DdtreeScratch::new(gpu, ddtree_budget),
            "DdtreeScratch::new",
            post_seed_snap,
            owners,
            draft_weights,
        );
        Some(DdtreeState {
            post_seed_snap,
            scratch,
            budget: ddtree_budget,
            topk,
        })
    } else {
        None
    };
    let DflashOwners {
        draft_scratch,
        hidden_rb,
        verify_scratch,
        target_snap,
        gdn_tape,
    } = owners;
    // Retained-PM4 admission (default-off). Pure gates + env opt-in; Disabled
    // holds no controller so HIP behavior stays byte-identical when not armed.
    let env_opt_in = hipfire_config::developer_var("HIPFIRE_DFLASH_VERIFY_PM4")
        .ok()
        .as_deref()
        == Some("1");
    let moe_router_present = verify_scratch
        .prefill_batch
        .as_ref()
        .map(|p| p.moe_router_logits_batch.is_some())
        .unwrap_or(false);
    let pbs_eligible_b16 = qwen35::prefill_batch_pbs_eligible(
        target_weights,
        target_config,
        target_dn,
        DFLASH_VERIFY_PM4_BLOCK,
        &gpu.arch,
        moe_router_present,
    );
    let pbs_max_batch = verify_scratch
        .prefill_batch
        .as_ref()
        .map(|p| p.max_batch)
        .unwrap_or(0);
    let gfx1100_split_verify_min_ctx =
        qwen35::prefill::gfx1100_split_verify_min_ctx(gpu, target_config).filter(|&threshold| {
            qwen35::prefill::gfx1100_split_verify_admitted(
                gpu,
                target_config,
                kv_is_q8,
                kv_is_vmm,
                DFLASH_VERIFY_PM4_BLOCK,
                threshold.saturating_add(1),
                target_flash_partials_numel,
            )
        });
    let gfx1100_split_verify = gfx1100_split_verify_min_ctx.is_some();
    // The draft's selector/dynamic-conv shape is deliberately NOT a gate: the
    // draft forward is outside the tape, so DFlash2 and legacy DFlash yield an
    // identical target verify body.
    let verify_pm4 = match admit_dflash_verify_pm4(
        env_opt_in,
        &gpu.arch,
        gfx1100_split_verify,
        single_gpu,
        target_config.num_experts,
        kv_is_q8,
        matches!(target_dn.quant, StateQuant::Q8),
        // finish_qwen35_load suppresses generic DFlash under adaptive KV, but
        // admit fail-closed so a future load path cannot arm a static-Q8 tape
        // against a tier-switching cache.
        !adaptive_kv,
        &hidden_rb.extract_layers,
        block_size,
        &draft_config.target_layer_ids,
        ddtree.is_some(),
        verify_scratch.prefill_batch.is_some(),
        pbs_eligible_b16,
        verify_scratch.max_n,
        pbs_max_batch,
        hidden_rb.max_batch,
        gdn_tape.max_n,
    ) {
        // The IU4 verify producers are eager-only; a recorded PM4 tape would
        // refuse them, so an IU4-verify load keeps the HIP verify route.
        Ok(())
            if qwen35::prefill::multi::dflash_iu4_verify_admitted(
                gpu,
                target_weights,
                target_config,
            ) =>
        {
            let reason =
                "IU4 verify is on for this load (eager-only; HIPFIRE_CB_VERIFY_IU4=0 turns it off)";
            eprintln!("  DFlash verify PM4: disabled ({reason})");
            DflashVerifyPm4::disabled(reason)
        }
        Ok(()) => {
            eprintln!(
                "  DFlash verify PM4: armed (B={}, exact {})",
                DFLASH_VERIFY_PM4_BLOCK, gpu.arch
            );
            if let Some(min_ctx) = gfx1100_split_verify_min_ctx {
                // Keep the faster incumbent below the resolved crossover, and
                // do not capture a short-context tape that would silently omit
                // the new partial + merge launches.
                DflashVerifyPm4::armed_after_context(min_ctx)
            } else {
                DflashVerifyPm4::armed()
            }
        }
        Err(reason) => {
            eprintln!("  DFlash verify PM4: disabled ({reason})");
            DflashVerifyPm4::disabled(reason)
        }
    };
    Ok(DflashState {
        draft_config,
        // Wrapped at successful state construction only: every error path
        // above frees the raw weights directly.
        draft_weights: Arc::new(draft_weights),
        draft_scratch,
        hidden_rb,
        verify_scratch,
        target_snap,
        gdn_tape,
        // Windowed mode reports the TARGET's physical capacity: the draft
        // degrades τ past its window instead of refusing, so the spec
        // loop's overflow guard and the daemon's capacity fallback track
        // the true cliff, not the window.
        ctx_capacity: if window.is_some() {
            requested_ctx
        } else {
            ctx_capacity
        },
        block_size,
        ddtree,
        verify_pm4,
    })
}

/// Whether a cold prefill of `prompt_len` rows downloads its hidden rows into
/// a temporary CPU shadow. After the prompt, production singleton/lane decode
/// reads only the GPU-resident draft context (`ctx_slice=None`); the only host
/// reader left is the windowed long-reach backfill, so by default the shadow
/// is captured exactly when [`draft_seed_backfill_reads_host`] holds. The
/// paired all-sliding draft therefore keeps every piece on device.
/// `HIPFIRE_DFLASH_PREFILL_DEVICE=0` restores the per-piece capture.
pub fn dflash_prefill_host_required(ctx_mode: DraftCtxMode, prompt_len: usize) -> bool {
    static DEVICE_PREFILL: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_bool("HIPFIRE_DFLASH_PREFILL_DEVICE", true)
    });
    !*DEVICE_PREFILL || draft_seed_backfill_reads_host(ctx_mode, prompt_len)
}

/// Cold/suffix DFlash prefill finishing steps shared by the singleton and the
/// VMM lane prefill: scatter the seeded hidden rows into the draft's
/// interleaved context (on a hit, only the suffix rows at `prefill_start`;
/// on a miss, all prompt rows from 0), windowed backfill (cold only), thlog
/// prompt seed / resume checkpoint, then the host copy of the post-prompt
/// logits. Order is the singleton's.
#[allow(clippy::too_many_arguments)]
fn dflash_prefill_finish(
    gpu: &mut Gpu,
    df: &mut DflashState,
    host_hidden: &[f32],
    logits: &GpuTensor,
    prompt_tokens: &[u32],
    prefill_len: usize,
    cache_hit: bool,
    prefill_start: usize,
    resume_from: Option<usize>,
) -> Result<Vec<f32>, String> {
    let (scatter_off, scatter_len) = if cache_hit {
        (prefill_start, prefill_len)
    } else {
        (0, prompt_tokens.len())
    };
    if let Err(e) = scatter_hidden_prefill_to_interleaved(
        gpu,
        &df.hidden_rb,
        &df.draft_scratch.target_hidden,
        scatter_off,
        scatter_len,
        scatter_len,
        df.draft_scratch.ctx_modulus(),
    ) {
        eprintln!("[dflash] scatter failed: {e} — falling back to per-cycle upload");
    }
    // Windowed cold prefill longer than W: the 4+1 split's last (full)
    // layer still needs K/V for every prompt row. `draft_seed_backfill`
    // is a no-op for all-sliding DFlash2 (every layer shares the W ring)
    // and for prompt_len <= W. Keep the call — it is safe on both splits.
    if !cache_hit {
        hipfire_runtime::dflash::draft_seed_backfill(
            gpu,
            &df.draft_weights,
            &df.draft_config,
            &mut df.draft_scratch,
            host_hidden,
            prompt_tokens.len(),
        )
        .map_err(|e| {
            hipfire_runtime::reset_core::note_hip_error(&e, "qwen35::dflash_prefill::backfill");
            e.to_string()
        })?;
    }
    df.draft_scratch.thlog.seed_prompt(prompt_tokens.len());
    if let Some(ckpt) = resume_from {
        // Divergent rows [ckpt..len) were just overwritten; drop the draft's
        // projection cursor so the first spec step re-projects from `ckpt`.
        df.draft_scratch.thlog.set_resume_checkpoint(ckpt);
    }
    gpu.download_f32(logits).map_err(|e| {
        hipfire_runtime::reset_core::note_hip_error(&e, "qwen35::dflash_prefill::logits");
        e.to_string()
    })
}

/// Prefill first-token policy shared by the singleton `prefill` and the VMM
/// lane `dflash_prefill_lane_parts` (extracted verbatim from the singleton
/// prefill body, originally dflash_spec.rs lines 890-913).
///
/// `cfg.temp <= 1e-6`: historical host argmax fold (first maximal index wins
/// via strict `>`), no RNG draw. Otherwise: host softmax at `cfg.temp`, then
/// `apply_host_ar_trunc(top_k, top_p)` ONLY when `chain` (DDTree SWOR honors
/// temperature only; origin lines 905-910), then exactly one
/// `xorshift_next_unit(rng)` draw and `sample_categorical` (origin lines
/// 911-912). `cactus_delta` is not consulted. `scratch` holds the probability
/// row (cleared/reserved by `softmax_temp_into`; identical to the singleton's
/// former fresh `Vec`).
pub(crate) fn dflash_first_token(
    logits: &[f32],
    chain: bool,
    cfg: crate::speculative::DflashSampleCfg,
    rng: &mut u64,
    scratch: &mut Vec<f32>,
) -> u32 {
    if cfg.temp <= 1e-6 {
        logits
            .iter()
            .enumerate()
            .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
                if v > bv {
                    (i as u32, v)
                } else {
                    (best, bv)
                }
            })
            .0
    } else {
        softmax_temp_into(logits, cfg.temp, scratch);
        if chain {
            apply_host_ar_trunc(scratch, cfg.top_k, cfg.top_p);
        }
        let u = xorshift_next_unit(rng);
        sample_categorical(scratch, u)
    }
}

// ─── DflashSpeculator ───────────────────────────────────────────────────

/// Developer-only per-cycle DFlash trace (`HIPFIRE_DFLASH_CYCLE_TRACE=<path>`):
/// one JSON object per line appended to `<path>`. Off by default; when unset the
/// only cost on the hot path is one `LazyLock` read. Diagnostic only — it never
/// changes decode behaviour and an open/write failure is swallowed (one
/// `eprintln` at open, tracing disabled).
mod cycle_trace {
    use std::fmt::Write as _;
    use std::io::Write as _;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{LazyLock, Mutex};

    type Sink = Mutex<std::io::BufWriter<std::fs::File>>;

    static SINK: LazyLock<Option<Sink>> = LazyLock::new(|| {
        let path = hipfire_config::developer_var("HIPFIRE_DFLASH_CYCLE_TRACE")
            .ok()
            .filter(|p| !p.is_empty())?;
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(f) => Some(Mutex::new(std::io::BufWriter::new(f))),
            Err(e) => {
                eprintln!(
                    "[dflash] HIPFIRE_DFLASH_CYCLE_TRACE: cannot open {path}: {e}; tracing disabled"
                );
                None
            }
        }
    });
    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn sink() -> Option<&'static Sink> {
        SINK.as_ref()
    }

    pub(super) fn enabled() -> bool {
        sink().is_some()
    }

    /// Next 1-based per-process request ordinal (called from `prefill`).
    pub(super) fn next_seq() -> u64 {
        SEQ.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Append one line and flush. Write errors are ignored (diagnostic only).
    pub(super) fn emit(line: &str) {
        if let Some(sink) = sink() {
            if let Ok(mut w) = sink.lock() {
                let _ = w.write_all(line.as_bytes());
                let _ = w.write_all(b"\n");
                let _ = w.flush();
            }
        }
    }

    fn push_u32s(out: &mut String, v: &[u32]) {
        out.push('[');
        for (i, t) in v.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "{t}");
        }
        out.push(']');
    }

    pub(super) fn prefill_line(
        pid: u32,
        seq: u64,
        prompt_ids: &[u32],
        prefill_start: usize,
        cache_hit: bool,
    ) -> String {
        let mut s = String::with_capacity(96 + prompt_ids.len() * 7);
        let _ = write!(
            s,
            r#"{{"ev":"prefill","pid":{pid},"seq":{seq},"prompt_len":{},"prefill_start":{prefill_start},"cache_hit":{cache_hit},"prompt_ids":"#,
            prompt_ids.len()
        );
        push_u32s(&mut s, prompt_ids);
        s.push('}');
        s
    }

    pub(super) fn forced_line(pid: u32, seq: u64, pos: usize, tokens: &[u32]) -> String {
        let mut s = String::with_capacity(64 + tokens.len() * 7);
        let _ = write!(
            s,
            r#"{{"ev":"forced","pid":{pid},"seq":{seq},"pos":{pos},"tokens":"#
        );
        push_u32s(&mut s, tokens);
        s.push('}');
        s
    }

    pub(super) fn reset_line(pid: u32, seq: u64) -> String {
        format!(r#"{{"ev":"reset","pid":{pid},"seq":{seq}}}"#)
    }

    /// Trailing lookup fields of a `cycle` line (written only when the
    /// lookup is armed; unarmed lines are unchanged).
    pub(super) struct LookupTag {
        pub kind: &'static str,
        pub k: usize,
        pub lookup: usize,
    }

    /// Fields of one `cycle` line. `accepted` is the raw `SpecStepResult`
    /// value; `committed` is the post-`cap_emit` emit (drafted[..accepted] ++
    /// [bonus] unless capped).
    pub(super) struct Cycle<'a> {
        pub pid: u32,
        pub seq: u64,
        pub cyc: u64,
        pub pos: usize,
        pub seed: u32,
        pub b: usize,
        pub max_emit: usize,
        pub temp: f32,
        pub drafted: &'a [u32],
        pub accepted: usize,
        pub committed: &'a [u32],
        pub step_us: u64,
        /// Tree-mode result: appended as a trailing `"ddtree":true`.
        pub ddtree: bool,
        /// Armed-lookup window kind: appended as trailing
        /// `"kind":..,"k":..,"lookup":..`.
        pub lookup: Option<LookupTag>,
        /// Certified-window route of this cycle (`whole16`, `hostsplice`,
        /// `hip:<reason>`): appended as a trailing `"railgun":"<route>"` only
        /// when `HIPFIRE_DFLASH_RAILGUN_WINDOW=1`; absent otherwise, so
        /// unset lines are byte-identical.
        pub railgun: Option<&'static str>,
    }

    impl Cycle<'_> {
        pub(super) fn line(&self) -> String {
            let mut s =
                String::with_capacity(192 + (self.drafted.len() + self.committed.len()) * 7);
            let _ = write!(
                s,
                r#"{{"ev":"cycle","pid":{},"seq":{},"cyc":{},"pos":{},"seed":{},"b":{},"max_emit":{},"temp":"#,
                self.pid, self.seq, self.cyc, self.pos, self.seed, self.b, self.max_emit
            );
            if self.temp.is_finite() {
                let _ = write!(s, "{}", self.temp);
            } else {
                s.push_str("null");
            }
            s.push_str(r#","drafted":"#);
            push_u32s(&mut s, self.drafted);
            let _ = write!(s, r#","accepted":{},"committed":"#, self.accepted);
            push_u32s(&mut s, self.committed);
            let _ = write!(s, r#","step_us":{}"#, self.step_us);
            if self.ddtree {
                s.push_str(r#","ddtree":true"#);
            }
            if let Some(l) = &self.lookup {
                let _ = write!(
                    s,
                    r#","kind":"{}","k":{},"lookup":{}"#,
                    l.kind, l.k, l.lookup
                );
            }
            if let Some(r) = self.railgun {
                let _ = write!(s, r#","railgun":"{r}""#);
            }
            s.push('}');
            s
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn cycle_trace_cycle_line_exact_bytes() {
            let c = Cycle {
                pid: 7,
                seq: 2,
                cyc: 3,
                pos: 100,
                seed: 5,
                b: 4,
                max_emit: 16,
                temp: 0.5,
                drafted: &[9, 8, 7],
                accepted: 2,
                committed: &[9, 8, 6],
                step_us: 1234,
                ddtree: false,
                lookup: None,
                railgun: None,
            };
            assert_eq!(
                c.line(),
                r#"{"ev":"cycle","pid":7,"seq":2,"cyc":3,"pos":100,"seed":5,"b":4,"max_emit":16,"temp":0.5,"drafted":[9,8,7],"accepted":2,"committed":[9,8,6],"step_us":1234}"#
            );
        }

        #[test]
        fn cycle_trace_cycle_line_empty_drafted_and_ddtree() {
            let c = Cycle {
                pid: 1,
                seq: 1,
                cyc: 0,
                pos: 0,
                seed: 3,
                b: 1,
                max_emit: 1,
                temp: 0.0,
                drafted: &[],
                accepted: 0,
                committed: &[4],
                step_us: 0,
                ddtree: true,
                lookup: None,
                railgun: None,
            };
            assert_eq!(
                c.line(),
                r#"{"ev":"cycle","pid":1,"seq":1,"cyc":0,"pos":0,"seed":3,"b":1,"max_emit":1,"temp":0,"drafted":[],"accepted":0,"committed":[4],"step_us":0,"ddtree":true}"#
            );
        }

        #[test]
        fn cycle_trace_other_lines_exact_bytes() {
            assert_eq!(
                prefill_line(9, 1, &[10, 11], 0, false),
                r#"{"ev":"prefill","pid":9,"seq":1,"prompt_len":2,"prefill_start":0,"cache_hit":false,"prompt_ids":[10,11]}"#
            );
            assert_eq!(
                forced_line(9, 1, 42, &[1, 2]),
                r#"{"ev":"forced","pid":9,"seq":1,"pos":42,"tokens":[1,2]}"#
            );
            assert_eq!(reset_line(9, 1), r#"{"ev":"reset","pid":9,"seq":1}"#);
        }

        #[test]
        fn lookup_cycle_trace_suffix_only_when_armed() {
            let mk = |lookup| Cycle {
                pid: 7,
                seq: 2,
                cyc: 3,
                pos: 100,
                seed: 5,
                b: 16,
                max_emit: 64,
                temp: 0.0,
                drafted: &[9, 8],
                accepted: 1,
                committed: &[9, 6],
                step_us: 10,
                ddtree: false,
                lookup,
                railgun: None,
            };
            let plain = mk(None).line();
            assert!(!plain.contains("kind"), "{plain}");
            assert!(plain.ends_with(r#""step_us":10}"#), "{plain}");
            let tagged = mk(Some(LookupTag {
                kind: "copy",
                k: 0,
                lookup: 15,
            }))
            .line();
            assert_eq!(
                tagged,
                format!(
                    "{}{}",
                    &plain[..plain.len() - 1],
                    r#","kind":"copy","k":0,"lookup":15}"#
                )
            );
        }
    }
}

// ─── Verbatim-copy lookup composed into the DFlash2 verify window ───────
//
// Wide-lookup S3b (`release-0.4.3/wide-lookup/PLAN.md` §3). Host-side glue
// between the request-local `MtpCopyContext` / `LookupController`
// (`hipfire_runtime`) and the `DflashLookup` seam of `spec_step_dflash`. All
// GPU-free logic is factored here so it is unit-tested on the CPU; `step`
// only wires it.
//
// `emitted` contract (verified in `hipfire-generate/src/qwen.rs`
// `generate_spec`): the loop pushes the prefill first token and every token
// the emitter retained, so at `step(position, seed, emitted)` the pending
// seed is `emitted.last()`. Hence `prompt ++ emitted` ends with the seed and
// the suffix key of a proposal is "…, seed". A window whose `emitted` does not
// end with `seed` (a grammar-rejected or non-committable token) runs the
// incumbent window with no lookup bookkeeping at all.
mod lookup {
    use crate::speculative::DflashTailSource;
    use hipfire_runtime::ngram_mod::{MtpCopyContext, NgramModConfig};
    use hipfire_runtime::spec_lookup::{
        LookupAction, LookupBudget, LookupController, LookupKind, LookupPolicy, WindowOutcome,
        LOOKUP_MAX_CANDIDATES, LOOKUP_MAX_ROWS,
    };

    /// Copy-lane pool of this lane: the Qwen4 copy lane's triple
    /// (`hipfire_config::ngram_mod_copy_triple()` = `NGRAM_MOD_COPY_TRIPLE`
    /// 5/1/63 plus its `HIPFIRE_NGRAM_MOD_*` overrides) and capacity
    /// (`NgramModConfig::default().capacity`), validated and row-capped exactly
    /// like `qwen4_ngram_mod_config` (`hipfire-arch-qwen4/src/mtp_spec.rs`):
    /// `n_max` is the proposal length, here at most one window's candidates.
    /// A triple the pool cannot run is an error, never a silently different
    /// policy.
    pub(super) fn copy_config(
        (n_match, n_min, n_max): (usize, usize, usize),
    ) -> Result<NgramModConfig, String> {
        if n_match == 0 || n_max == 0 || n_max > 64 || n_min > n_max {
            return Err(format!(
                "invalid n-gram triple n_match={n_match} n_min={n_min} n_max={n_max} (need 1 <= n_match, n_min <= n_max <= 64)"
            ));
        }
        let n_max = n_max.min(LOOKUP_MAX_CANDIDATES);
        Ok(NgramModConfig {
            n_match,
            n_min: n_min.min(n_max),
            n_max,
            ..NgramModConfig::default()
        })
    }

    /// Load-time arming inputs.
    #[derive(Clone, Copy, Debug)]
    pub(super) struct ArmInputs {
        /// Resolved `speculation.dflash_lookup` (`auto` = on).
        pub on: bool,
        /// `HIPFIRE_SPEC_LOOKUP_TAIL` is not `0`.
        pub tail_env: bool,
        pub ddtree: bool,
        pub selector: bool,
        /// Retained-PM4 verify replay admitted (shape-frozen at B=16).
        pub pm4: bool,
        /// The load's runtime verify block.
        pub block_size: usize,
    }

    /// `Ok(tail_enabled)` when lookup may arm, else the reason it stays off.
    /// Only the singleton `DflashSpeculator` calls this: the VMM/CB lanes and
    /// the dense-TP mesh speculator are different types that never arm, and
    /// CACTUS (a per-request setting) is excluded per window in `step`.
    pub(super) fn arming(i: &ArmInputs) -> Result<bool, &'static str> {
        if !i.on {
            return Err("speculation.dflash_lookup is off");
        }
        if i.ddtree {
            return Err("DDTree chain");
        }
        if !i.selector {
            return Err("no DFlash2 candidate selector");
        }
        if i.pm4 {
            return Err("retained PM4 verify is shape-frozen at B=16");
        }
        if i.block_size < 3 {
            return Err("verify block < 3 rows");
        }
        Ok(i.tail_env)
    }

    pub(super) fn load_line(arm: &Result<bool, String>, max_rows: usize) -> String {
        match arm {
            Ok(true) => format!("[dflash] lookup: armed (copy+tail, rows<={max_rows})"),
            Ok(false) => format!("[dflash] lookup: armed (copy, rows<={max_rows})"),
            Err(why) => format!("[dflash] lookup: off ({why})"),
        }
    }

    /// Armed state, or the load line of a lookup that stays off. `None` for
    /// the state is the whole unarmed path: nothing is allocated.
    pub(super) fn build_state(
        i: &ArmInputs,
        triple: (usize, usize, usize),
    ) -> (Option<LookupState>, String) {
        let max_rows = i.block_size.min(LOOKUP_MAX_ROWS);
        let arm = arming(i).map_err(str::to_string).and_then(|tail| {
            let ctx = copy_config(triple)
                .and_then(|c| MtpCopyContext::new(c).map_err(str::to_string))
                .map_err(|e| format!("copy context: {e}"))?;
            Ok((tail, ctx))
        });
        match arm {
            Ok((tail, ctx)) => {
                let line = load_line(&Ok(tail), max_rows);
                (Some(LookupState::new(ctx, tail, max_rows)), line)
            }
            Err(why) => (None, load_line(&Err(why), max_rows)),
        }
    }

    /// Budget at the pre-draft Copy decision (`model_drafts = 0`,
    /// `lookup_available` unset). `native_b` is the incumbent window's rows
    /// (seed included) after the adaptive / `max_emit` logic; `block_size` the
    /// load's runtime block; `max_emit` the remaining client budget;
    /// `context_rows_left` mirrors the generate loop's overflow guard
    /// (`position + width < ctx_capacity`).
    pub(super) fn lookup_budget(
        native_b: usize,
        block_size: usize,
        max_emit: usize,
        position: usize,
        ctx_capacity: usize,
    ) -> LookupBudget {
        LookupBudget {
            native_candidates: native_b.saturating_sub(1),
            model_drafts: 0,
            lookup_available: 0,
            max_rows: block_size.min(LOOKUP_MAX_ROWS),
            remaining_emit: max_emit,
            context_rows_left: ctx_capacity.saturating_sub(position).saturating_sub(1),
        }
    }

    /// Verify rows (seed included) any lookup window of `b` may use.
    pub(super) fn rows_cap(b: &LookupBudget) -> usize {
        b.max_rows
            .min(LOOKUP_MAX_ROWS)
            .min(b.remaining_emit)
            .min(b.context_rows_left)
    }

    /// Clamp a chooser's Copy `take` before any write: `None` when no legal
    /// candidate remains.
    pub(super) fn clamp_copy_take(take: usize, offered: usize, b: &LookupBudget) -> Option<usize> {
        let t = take
            .min(offered)
            .min(LOOKUP_MAX_CANDIDATES)
            .min(rows_cap(b).saturating_sub(1));
        (t >= 1).then_some(t)
    }

    /// Clamp a chooser's Tail `take` after `k` kept learned candidates of a
    /// `drafted_len`-candidate window.
    pub(super) fn clamp_tail_take(
        take: usize,
        offered: usize,
        k: usize,
        drafted_len: usize,
        b: &LookupBudget,
    ) -> Option<usize> {
        let t = take
            .min(offered)
            .min(drafted_len.saturating_sub(k))
            .min(rows_cap(b).saturating_sub(1 + k));
        (k >= 1 && t >= 1).then_some(t)
    }

    /// A price-probe native window: only wider than the incumbent window and
    /// within the same clamps a Copy of that many rows would face.
    pub(super) fn valid_probe_rows(
        rows: usize,
        native_b: usize,
        b: &LookupBudget,
    ) -> Option<usize> {
        (rows >= 2 && rows > native_b && rows <= rows_cap(b)).then_some(rows)
    }

    /// Sequential Tail stopping rule. For `K = 1..drafted.len()` (at least
    /// one tail slot remains) the decision to splice at `K` reads only
    /// `drafted[..K]`: `propose(prefix, cap, out)` fills `out` with the
    /// lookup candidates after the committed history plus that prefix
    /// (`cap = drafted.len() - K`) and `decide(K, offered)` returns the
    /// chosen take. The first `K` that takes wins. On `Some((K, L))` the tail
    /// is `tail[..L]`; `None` leaves `tail` empty.
    pub(super) fn tail_splice<P, D>(
        drafted: &[u32],
        tail: &mut Vec<u32>,
        cand: &mut Vec<u32>,
        mut propose: P,
        mut decide: D,
    ) -> Option<(usize, usize)>
    where
        P: FnMut(&[u32], usize, &mut Vec<u32>),
        D: FnMut(usize, usize) -> Option<usize>,
    {
        tail.clear();
        let n = drafted.len();
        for k in 1..n {
            let cap = n - k;
            cand.clear();
            propose(&drafted[..k], cap, cand);
            let offered = cand.len().min(cap);
            if offered == 0 {
                continue;
            }
            if let Some(take) = decide(k, offered) {
                let take = take.min(offered);
                if take == 0 {
                    continue;
                }
                tail.extend_from_slice(&cand[..take]);
                return Some((k, take));
            }
        }
        None
    }

    /// `DflashTailSource` over the request's copy context and controller.
    pub(super) struct TailSource<'a> {
        pub ctx: &'a mut MtpCopyContext,
        pub ctl: &'a LookupController,
        pub emitted: &'a [u32],
        /// Pre-draft budget template; `native_candidates` is set per window.
        pub base: LookupBudget,
        pub position: usize,
        pub cand: &'a mut Vec<u32>,
        /// `(K, L)` of the splice taken, `None` when the window stayed native.
        pub spliced: Option<(usize, usize)>,
    }

    impl DflashTailSource for TailSource<'_> {
        fn splice(&mut self, drafted: &[u32], tail: &mut Vec<u32>) -> Option<usize> {
            let mut base = self.base;
            base.native_candidates = drafted.len();
            let (ctx, ctl, emitted, position) =
                (&mut *self.ctx, self.ctl, self.emitted, self.position);
            let r = tail_splice(
                drafted,
                tail,
                &mut *self.cand,
                |prefix, cap, out| {
                    if let Some(c) = ctx.propose_after(emitted, prefix, cap) {
                        out.extend_from_slice(c);
                    }
                },
                |k, offered| {
                    let mut b = base;
                    b.model_drafts = k;
                    b.lookup_available = offered;
                    match ctl.plan(b, position) {
                        LookupAction::Tail { take } => {
                            clamp_tail_take(take, offered, k, drafted.len(), &b)
                        }
                        _ => None,
                    }
                },
            );
            self.spliced = r;
            r.map(|(k, _)| k)
        }

        /// Enumerate every decision `splice` could face: `k = 1..n-1` kept
        /// learned candidates and `offered = 1..=n-k` lookup candidates; build
        /// the exact `LookupBudget` `splice` builds, run the existing
        /// `LookupController::plan` and `clamp_tail_take`. Any positive take
        /// refutes the proof. The controller is only read, so the
        /// answer is for the state the following `splice` would see.
        fn declines_every_tail(&self, candidates: usize) -> bool {
            no_tail_action(self.ctl, self.base, self.position, candidates)
        }
    }

    /// `NoTailAction`: true iff no `(k, offered)` decision of a
    /// `candidates`-candidate learned block yields a positive Tail take.
    /// Mirrors `TailSource::splice` + `tail_splice` exactly (same budget
    /// construction, same `plan`, same `clamp_tail_take`); never approximates
    /// prices and never infers from the previous window's outcome.
    pub(super) fn no_tail_action(
        ctl: &LookupController,
        base: LookupBudget,
        position: usize,
        candidates: usize,
    ) -> bool {
        let mut base = base;
        base.native_candidates = candidates;
        for k in 1..candidates {
            for offered in 1..=(candidates - k) {
                let mut b = base;
                b.model_drafts = k;
                b.lookup_available = offered;
                if let LookupAction::Tail { take } = ctl.plan(b, position) {
                    if clamp_tail_take(take, offered, k, candidates, &b).is_some() {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// What `step` decided for this window.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum WindowPlan {
        /// Lookup state unusable this window (seed not at the end of
        /// `emitted`): run the incumbent call and observe nothing.
        Off,
        /// Incumbent block; `tail` = offer the Tail source.
        Native {
            tail: bool,
        },
        Copy {
            take: usize,
        },
        /// Run THIS native window at `rows` (> the incumbent width) to price
        /// the row count a Copy needs.
        Probe {
            rows: usize,
        },
    }

    /// Which window ran.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Ran {
        Native,
        Copy { take: usize },
        Tail { k: usize, l: usize },
    }

    /// `(matched, available)` of the pre-window copy `proposal` against the
    /// window's committed tokens after the seed (`accepted` drafts + bonus).
    /// `None` when nothing was proposed.
    pub(super) fn shadow_match(
        proposal: &[u32],
        committed_after_seed: &[u32],
    ) -> Option<(usize, usize)> {
        if proposal.is_empty() {
            return None;
        }
        let matched = proposal
            .iter()
            .zip(committed_after_seed)
            .take_while(|(p, c)| p == c)
            .count();
        Some((matched, proposal.len()))
    }

    /// Controller observation of one finished window (the shared observation
    /// rules live in `LookupController::observe`). `rows` = verified rows
    /// (`SpecStepResult.drafted.len()`), `accepted` the raw accepted
    /// candidates (clamped to `rows - 1`), `proposal` the pre-window copy
    /// proposal and `committed_after_seed` the window's committed tokens
    /// after the seed. Only a window that ran no lookup rows (Native, incl. a
    /// declined Tail and a price probe) carries a shadow.
    pub(super) fn window_outcome(
        ran: Ran,
        rows: usize,
        accepted: usize,
        proposal: &[u32],
        committed_after_seed: &[u32],
        ctx: usize,
        wall_us: f64,
    ) -> WindowOutcome {
        let accepted = accepted.min(rows.saturating_sub(1));
        let (kind, model_drafts, lookup_len, shadow) = match ran {
            Ran::Native => (
                LookupKind::Native,
                rows.saturating_sub(1),
                0,
                shadow_match(proposal, committed_after_seed),
            ),
            Ran::Copy { take } => (LookupKind::Copy, 0, take, None),
            Ran::Tail { k, l } => (LookupKind::Tail, k, l, None),
        };
        WindowOutcome {
            kind,
            rows,
            model_drafts,
            lookup_len,
            accepted,
            shadow,
            ctx,
            wall_us,
        }
    }

    /// Trailing `"kind","k","lookup"` of the dev cycle-trace line.
    /// `offered` = the pre-window copy proposal length (a Native window's
    /// `lookup` is what it declined to verify).
    pub(super) fn trace_tag(o: &WindowOutcome, offered: usize) -> (&'static str, usize, usize) {
        match o.kind {
            LookupKind::Native => ("native", o.model_drafts, offered),
            LookupKind::Copy => ("copy", o.model_drafts, o.lookup_len),
            LookupKind::Tail => ("tail", o.model_drafts, o.lookup_len),
        }
    }

    /// Request-local lookup state of an armed `DflashSpeculator`.
    pub(super) struct LookupState {
        pub ctx: MtpCopyContext,
        pub ctl: LookupController,
        pub tail: bool,
        pub max_rows: usize,
        prompt_len: usize,
        dirty: bool,
        /// Pre-window copy proposal (reusable; capacity reserved once).
        pub proposal: Vec<u32>,
        /// Tail candidate scratch.
        pub cand: Vec<u32>,
    }

    impl LookupState {
        fn new(ctx: MtpCopyContext, tail: bool, max_rows: usize) -> Self {
            Self {
                ctx,
                ctl: LookupController::new(LookupPolicy::default(), true),
                tail,
                max_rows,
                prompt_len: 0,
                dirty: false,
                proposal: Vec::with_capacity(LOOKUP_MAX_CANDIDATES),
                cand: Vec::with_capacity(LOOKUP_MAX_CANDIDATES),
            }
        }

        /// New request: history = the canonical full prompt, acceptance
        /// tables reset (prices are load-scoped and kept).
        pub fn begin_request(&mut self, prompt: &[u32]) {
            self.ctx.begin_request(prompt);
            self.ctl.begin_request();
            self.prompt_len = prompt.len();
            self.dirty = false;
            self.proposal.clear();
        }

        /// No live request (prefill start / `reset`): nothing proposes.
        pub fn end_request(&mut self) {
            self.ctx.reset_request();
            self.prompt_len = 0;
            self.dirty = false;
            self.proposal.clear();
        }

        /// A rewind / terminal repair may have shrunk or rewritten `emitted`:
        /// rebuild the history from the prompt at the next window.
        pub fn mark_dirty(&mut self) {
            self.dirty = true;
        }

        /// Make the history `prompt ++ emitted`. A shrunk or diverged
        /// `emitted` (rewind, terminal repair, divergent re-render) rebuilds
        /// it from the stored prompt prefix (`begin_request(prompt)` then
        /// `sync_emitted`); otherwise only the new suffix is appended.
        /// `false` before `begin_request`.
        pub fn sync_history(&mut self, emitted: &[u32]) -> bool {
            if !self.ctx.is_begun() {
                return false;
            }
            let hist = self.ctx.history();
            let have = hist.len().saturating_sub(self.prompt_len);
            let stale = self.dirty
                || emitted.len() < have
                || (have > 0 && emitted[have - 1] != hist[hist.len() - 1]);
            if stale {
                let prompt = self.ctx.history()[..self.prompt_len].to_vec();
                self.ctx.begin_request(&prompt);
                self.dirty = false;
            }
            self.ctx.sync_emitted(emitted);
            true
        }

        /// Plan one window: sync the history, take the pre-window copy
        /// proposal (stored in `self.proposal`), then Copy / price-probe /
        /// native(+tail offer). No proposal and no tail offer = the incumbent
        /// call exactly.
        #[allow(clippy::too_many_arguments)]
        pub fn plan_window(
            &mut self,
            emitted: &[u32],
            seed: u32,
            position: usize,
            native_b: usize,
            max_emit: usize,
            block_size: usize,
            ctx_capacity: usize,
        ) -> WindowPlan {
            self.proposal.clear();
            if emitted.last() != Some(&seed) || !self.sync_history(emitted) {
                return WindowPlan::Off;
            }
            let mut budget = lookup_budget(native_b, block_size, max_emit, position, ctx_capacity);
            budget.max_rows = budget.max_rows.min(self.max_rows);
            let rows = rows_cap(&budget);
            let cap = rows.saturating_sub(1).min(LOOKUP_MAX_CANDIDATES);
            if cap >= 1 {
                if let Some(c) = self.ctx.propose(emitted, cap) {
                    self.proposal.extend_from_slice(c);
                }
            }
            if !self.proposal.is_empty() {
                let mut b = budget;
                b.lookup_available = self.proposal.len();
                if let LookupAction::Copy { take } = self.ctl.plan(b, position) {
                    if let Some(t) = clamp_copy_take(take, self.proposal.len(), &b) {
                        return WindowPlan::Copy { take: t };
                    }
                }
                if let Some(r) = self
                    .ctl
                    .price_probe_rows(b, position)
                    .and_then(|r| valid_probe_rows(r, native_b, &b))
                {
                    return WindowPlan::Probe { rows: r };
                }
            }
            WindowPlan::Native {
                tail: self.tail && rows >= 3,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn st(tail: bool) -> LookupState {
            let ctx = MtpCopyContext::new(copy_config((5, 1, 63)).unwrap()).unwrap();
            LookupState::new(ctx, tail, 16)
        }

        fn inputs() -> ArmInputs {
            ArmInputs {
                on: true,
                tail_env: true,
                ddtree: false,
                selector: true,
                pm4: false,
                block_size: 16,
            }
        }

        #[test]
        fn lookup_budget_construction_and_clamps() {
            let b = lookup_budget(16, 16, 1000, 500, 100_000);
            assert_eq!(
                b,
                LookupBudget {
                    native_candidates: 15,
                    model_drafts: 0,
                    lookup_available: 0,
                    max_rows: 16,
                    remaining_emit: 1000,
                    context_rows_left: 100_000 - 500 - 1,
                }
            );
            // Adaptive shrink reduces the native candidates, not the lookup width.
            let b = lookup_budget(5, 16, 1000, 500, 100_000);
            assert_eq!((b.native_candidates, b.max_rows), (4, 16));
            // Configured block above 16 never exceeds 16 rows; below keeps it.
            assert_eq!(lookup_budget(8, 24, 9, 0, 100).max_rows, 16);
            assert_eq!(lookup_budget(8, 8, 9, 0, 100).max_rows, 8);
            // Context mirrors the loop's `position + width < ctx_capacity` guard.
            assert_eq!(lookup_budget(8, 16, 99, 95, 100).context_rows_left, 4);
            assert_eq!(lookup_budget(8, 16, 99, 99, 100).context_rows_left, 0);
            assert_eq!(lookup_budget(8, 16, 99, 400, 100).context_rows_left, 0);
            // Uncapped benches pass usize::MAX through.
            assert_eq!(
                lookup_budget(8, 16, usize::MAX, 0, 100).remaining_emit,
                usize::MAX
            );
            // Degenerate native width never underflows.
            assert_eq!(lookup_budget(0, 16, 4, 0, 100).native_candidates, 0);
        }

        #[test]
        fn lookup_copy_and_tail_takes_clamp_before_any_write() {
            let b = lookup_budget(16, 16, 1000, 0, 100_000);
            assert_eq!(clamp_copy_take(15, 15, &b), Some(15));
            assert_eq!(clamp_copy_take(40, 15, &b), Some(15));
            assert_eq!(clamp_copy_take(15, 9, &b), Some(9));
            assert_eq!(clamp_copy_take(0, 9, &b), None);
            assert_eq!(clamp_copy_take(4, 0, &b), None);
            // remaining_emit: rows <= max_emit, so take <= max_emit - 1.
            let b4 = lookup_budget(16, 16, 4, 0, 100_000);
            assert_eq!(clamp_copy_take(15, 15, &b4), Some(3));
            let b1 = lookup_budget(16, 16, 1, 0, 100_000);
            assert_eq!(clamp_copy_take(15, 15, &b1), None);
            // context rows left.
            let bc = lookup_budget(16, 16, 1000, 97, 100);
            assert_eq!(clamp_copy_take(15, 15, &bc), Some(1));
            let bc0 = lookup_budget(16, 16, 1000, 99, 100);
            assert_eq!(clamp_copy_take(15, 15, &bc0), None);
            // block width 8.
            let b8 = lookup_budget(8, 8, 1000, 0, 100_000);
            assert_eq!(clamp_copy_take(15, 15, &b8), Some(7));
            // Tail: K=3 of 15 drafted leaves 12 slots; rows 1+3+L <= 16.
            assert_eq!(clamp_tail_take(12, 12, 3, 15, &b), Some(12));
            assert_eq!(clamp_tail_take(15, 15, 3, 15, &b), Some(12));
            assert_eq!(clamp_tail_take(5, 2, 3, 15, &b), Some(2));
            assert_eq!(clamp_tail_take(5, 5, 15, 15, &b), None);
            assert_eq!(clamp_tail_take(5, 5, 0, 15, &b), None);
            // K=14: one tail slot.
            assert_eq!(clamp_tail_take(9, 9, 14, 15, &b), Some(1));
            // emit budget: K + L + 1 <= max_emit.
            assert_eq!(clamp_tail_take(9, 9, 2, 15, &b4), Some(1));
            assert_eq!(clamp_tail_take(9, 9, 3, 15, &b4), None);
            // Probe rows: strictly wider than the incumbent, inside the clamps.
            assert_eq!(valid_probe_rows(16, 8, &b), Some(16));
            assert_eq!(valid_probe_rows(8, 8, &b), None);
            assert_eq!(valid_probe_rows(6, 8, &b), None);
            assert_eq!(valid_probe_rows(16, 8, &b4), None);
            assert_eq!(valid_probe_rows(1, 0, &b), None);
        }

        #[test]
        fn lookup_shadow_matched_computation() {
            assert_eq!(shadow_match(&[], &[1, 2]), None);
            assert_eq!(shadow_match(&[1, 2, 3], &[]), Some((0, 3)));
            assert_eq!(shadow_match(&[1, 2, 3], &[9, 2, 3]), Some((0, 3)));
            assert_eq!(shadow_match(&[1, 2, 3], &[1, 2, 9]), Some((2, 3)));
            assert_eq!(shadow_match(&[1, 2, 3], &[1, 2, 3, 7]), Some((3, 3)));
            // Committed shorter than the proposal: matched <= committed.
            assert_eq!(shadow_match(&[1, 2, 3, 4], &[1, 2]), Some((2, 4)));
        }

        #[test]
        fn lookup_outcome_per_kind_accepted_zero_partial_full_and_eos_cap() {
            let prop = [10, 11, 12, 13];
            // Native, accepted 0: the bonus alone, shadow compares it to the proposal.
            let o = window_outcome(Ran::Native, 8, 0, &prop, &[10], 700, 1500.0);
            assert_eq!(
                o,
                WindowOutcome {
                    kind: LookupKind::Native,
                    rows: 8,
                    model_drafts: 7,
                    lookup_len: 0,
                    accepted: 0,
                    shadow: Some((1, 4)),
                    ctx: 700,
                    wall_us: 1500.0,
                }
            );
            // Native partial; the learned run and the proposal diverge after 2.
            let o = window_outcome(Ran::Native, 16, 3, &prop, &[10, 11, 99, 5], 1, 2.0);
            assert_eq!(
                (o.accepted, o.shadow, o.model_drafts),
                (3, Some((2, 4)), 15)
            );
            // Native full accept (accepted = rows-1), no proposal -> no shadow.
            let o = window_outcome(Ran::Native, 4, 3, &[], &[1, 2, 3, 4], 1, 2.0);
            assert_eq!((o.accepted, o.shadow), (3, None));
            // Copy: never a shadow, model unobserved, lookup_len = take.
            for acc in [0, 6, 15] {
                let o = window_outcome(Ran::Copy { take: 15 }, 16, acc, &prop, &prop, 9, 3.0);
                assert_eq!(
                    (o.kind, o.model_drafts, o.lookup_len, o.accepted, o.shadow),
                    (LookupKind::Copy, 0, 15, acc, None)
                );
                assert_eq!(o.rows, 16);
            }
            // Tail(K=3, L=8): accepted inside the learned prefix, at the splice, full.
            for acc in [0usize, 2, 3, 7, 11] {
                let o = window_outcome(Ran::Tail { k: 3, l: 8 }, 12, acc, &prop, &prop, 9, 3.0);
                assert_eq!(
                    (o.kind, o.model_drafts, o.lookup_len, o.accepted, o.shadow),
                    (LookupKind::Tail, 3, 8, acc, None)
                );
            }
            // EOS / max_emit cap: a raw accepted above rows-1 can never be observed.
            let o = window_outcome(Ran::Copy { take: 3 }, 4, 9, &prop, &prop, 0, 1.0);
            assert_eq!(o.accepted, 3);
            let o = window_outcome(Ran::Native, 1, 5, &[], &[1], 0, 1.0);
            assert_eq!((o.accepted, o.model_drafts), (0, 0));
            // Trace tags.
            assert_eq!(
                trace_tag(&window_outcome(Ran::Native, 8, 2, &prop, &[10], 0, 1.0), 4),
                ("native", 7, 4)
            );
            assert_eq!(
                trace_tag(
                    &window_outcome(Ran::Copy { take: 5 }, 6, 2, &prop, &prop, 0, 1.0),
                    9
                ),
                ("copy", 0, 5)
            );
            assert_eq!(
                trace_tag(
                    &window_outcome(Ran::Tail { k: 2, l: 4 }, 7, 2, &prop, &prop, 0, 1.0),
                    9
                ),
                ("tail", 2, 4)
            );
        }

        /// The splice decision at K may read only `drafted[..K]`: the propose
        /// callback sees prefixes of exactly K = 1, 2, ... in order, the
        /// decision is asked once per K that offers candidates, and nothing
        /// is read after the first K that takes.
        #[test]
        fn lookup_tail_source_decides_from_prefix_only() {
            let mut seen: Vec<Vec<u32>> = Vec::new();
            let mut asked: Vec<(usize, usize)> = Vec::new();
            let drafted = [10, 11, 12, 13, 14, 15];
            let mut tail = vec![99];
            let mut cand = Vec::new();
            let r = tail_splice(
                &drafted,
                &mut tail,
                &mut cand,
                |prefix, cap, out| {
                    seen.push(prefix.to_vec());
                    assert_eq!(cap, drafted.len() - prefix.len());
                    // Candidates exist only after the prefix [10, 11, 12].
                    if prefix == [10, 11, 12] {
                        out.extend_from_slice(&[7, 8, 9, 6]);
                    }
                },
                |k, offered| {
                    asked.push((k, offered));
                    Some(2)
                },
            );
            assert_eq!(r, Some((3, 2)));
            assert_eq!(tail, vec![7, 8]);
            assert_eq!(
                seen,
                vec![vec![10], vec![10, 11], vec![10, 11, 12]],
                "never read past the deciding prefix"
            );
            assert_eq!(asked, vec![(3, 3)], "offered clamps to the slots left");
            // Same decision for any drafted[K..]: change everything after K.
            let drafted_b = [10, 11, 12, 1, 2, 3];
            let mut tail_b = Vec::new();
            let r_b = tail_splice(
                &drafted_b,
                &mut tail_b,
                &mut cand,
                |prefix, _cap, out| {
                    if prefix == [10, 11, 12] {
                        out.extend_from_slice(&[7, 8, 9, 6]);
                    }
                },
                |_k, _offered| Some(2),
            );
            assert_eq!(r_b, Some((3, 2)));
            assert_eq!(tail_b, vec![7, 8]);
        }

        #[test]
        fn lookup_tail_source_first_k_wins_declines_and_bounds() {
            let drafted = [1, 2, 3, 4];
            let mut tail = Vec::new();
            let mut cand = Vec::new();
            // Decline everything: None, tail left empty.
            let r = tail_splice(
                &drafted,
                &mut tail,
                &mut cand,
                |_p, _c, out| out.extend_from_slice(&[5, 6]),
                |_k, _o| None,
            );
            assert_eq!((r, tail.len()), (None, 0));
            // A chooser that takes at K=1 stops there (first K wins).
            let mut calls = 0;
            let r = tail_splice(
                &drafted,
                &mut tail,
                &mut cand,
                |_p, _c, out| out.extend_from_slice(&[5, 6, 7, 8]),
                |k, _o| {
                    calls += 1;
                    (k == 1).then_some(9)
                },
            );
            // take clamps to what was offered (cap = 3 slots after K=1).
            assert_eq!((r, calls, tail.clone()), (Some((1, 3)), 1, vec![5, 6, 7]));
            // No lookup candidates at any K: the chooser is never asked.
            let r = tail_splice(
                &drafted,
                &mut tail,
                &mut cand,
                |_p, _c, _out| {},
                |_k, _o| panic!("decide without candidates"),
            );
            assert_eq!(r, None);
            // K = drafted.len() leaves no tail slot, so a window of 1 or 0
            // learned candidates never splices.
            let none: [u32; 0] = [];
            let one = [1u32];
            for d in [&none[..], &one[..]] {
                let r = tail_splice(
                    d,
                    &mut tail,
                    &mut cand,
                    |_p, _c, _o| panic!("propose with no tail slot"),
                    |_k, _o| panic!("decide with no tail slot"),
                );
                assert_eq!(r, None);
            }
            // A zero take is a decline, the search continues.
            let r = tail_splice(
                &drafted,
                &mut tail,
                &mut cand,
                |_p, _c, out| out.push(5),
                |k, _o| Some(if k == 2 { 1 } else { 0 }),
            );
            assert_eq!(r, Some((2, 1)));
        }

        fn mk_history() -> (Vec<u32>, Vec<u32>) {
            // 40 unique ids; the prompt ends in a copy of an earlier run.
            let a: Vec<u32> = (100..140).collect();
            let mut prompt = a.clone();
            prompt.extend_from_slice(&[900, 901]);
            (prompt, a)
        }

        #[test]
        fn lookup_rewind_rebuilds_history_from_prompt() {
            let (prompt, a) = mk_history();
            let mut s = st(true);
            s.begin_request(&prompt);
            let emitted: Vec<u32> = a[..8].to_vec();
            assert!(s.sync_history(&emitted));
            assert_eq!(s.ctx.history().len(), prompt.len() + 8);
            assert_eq!(&s.ctx.history()[..prompt.len()], &prompt[..]);
            assert_eq!(&s.ctx.history()[prompt.len()..], &emitted[..]);
            // Incremental growth appends only the new suffix.
            let grown: Vec<u32> = a[..12].to_vec();
            assert!(s.sync_history(&grown));
            assert_eq!(&s.ctx.history()[prompt.len()..], &grown[..]);
            // emitted shrinks (terminal repair / re-render): rebuilt, not stale.
            let shrunk: Vec<u32> = a[..5].to_vec();
            assert!(s.sync_history(&shrunk));
            assert_eq!(s.ctx.history().len(), prompt.len() + 5);
            assert_eq!(&s.ctx.history()[prompt.len()..], &shrunk[..]);
            // Same length, different content (divergent re-render): rebuilt.
            let diverged: Vec<u32> = vec![a[0], a[1], a[2], a[3], 777];
            assert!(s.sync_history(&diverged));
            assert_eq!(&s.ctx.history()[prompt.len()..], &diverged[..]);
            // A rewind marks the state dirty: rebuilt even though emitted only grew.
            s.mark_dirty();
            let regrown: Vec<u32> = vec![a[0], a[1], a[2], a[3], 777, 778];
            assert!(s.sync_history(&regrown));
            assert_eq!(&s.ctx.history()[prompt.len()..], &regrown[..]);
            assert!(!s.dirty);
            // The rebuilt tables match a fresh context's proposals.
            let mut fresh = MtpCopyContext::new(copy_config((5, 1, 63)).unwrap()).unwrap();
            fresh.begin_request(&prompt);
            let key: Vec<u32> = a[..8].to_vec();
            let want = fresh.propose(&key, 15).map(<[u32]>::to_vec);
            assert!(s.sync_history(&key));
            assert_eq!(s.ctx.propose(&key, 15).map(<[u32]>::to_vec), want);
            assert!(want.is_some());
            // No request: nothing proposes, no history.
            s.end_request();
            assert!(!s.sync_history(&key));
            assert!(s.ctx.history().is_empty());
        }

        #[test]
        fn lookup_plan_window_incumbent_paths() {
            let (prompt, a) = mk_history();
            let mut s = st(true);
            // Before begin_request: lookup off for the window (incumbent call).
            let emitted: Vec<u32> = a[..8].to_vec();
            assert_eq!(
                s.plan_window(&emitted, a[7], 1000, 16, 4096, 16, 100_000),
                WindowPlan::Off
            );
            s.begin_request(&prompt);
            // The seed must be the last emitted token.
            assert_eq!(
                s.plan_window(&emitted, a[3], 1000, 16, 4096, 16, 100_000),
                WindowPlan::Off
            );
            assert!(s.proposal.is_empty());
            // A proposal exists but the controller has no prices/evidence:
            // the incumbent block, Tail offered (it can only splice on measured wins).
            assert_eq!(
                s.plan_window(&emitted, a[7], 1000, 16, 4096, 16, 100_000),
                WindowPlan::Native { tail: true }
            );
            assert_eq!(s.proposal, a[8..23].to_vec());
            // Tail kill switch: still native, no tail offer.
            let mut s2 = st(false);
            s2.begin_request(&prompt);
            assert_eq!(
                s2.plan_window(&emitted, a[7], 1000, 16, 4096, 16, 100_000),
                WindowPlan::Native { tail: false }
            );
            // No proposal: `emitted` of tokens never seen before.
            let mut s3 = st(true);
            s3.begin_request(&prompt);
            let novel: Vec<u32> = vec![5000, 5001, 5002, 5003, 5004, 5005];
            assert_eq!(
                s3.plan_window(&novel, 5005, 1000, 16, 4096, 16, 100_000),
                WindowPlan::Native { tail: true }
            );
            assert!(s3.proposal.is_empty());
            // max_emit 2: no room for a tail (rows < 3), no proposal.
            assert_eq!(
                s3.plan_window(&novel, 5005, 1000, 2, 2, 16, 100_000),
                WindowPlan::Native { tail: false }
            );
        }

        #[test]
        fn lookup_plan_window_takes_copy_after_priced_evidence() {
            let (prompt, a) = mk_history();
            let mut s = st(true);
            s.begin_request(&prompt);
            let emitted: Vec<u32> = a[..8].to_vec();
            // Priced native 16-row windows whose lookup shadow always matches
            // the full proposal while the learned chain accepts little.
            for _ in 0..12 {
                let o = window_outcome(Ran::Native, 16, 1, &a[8..23], &a[8..23], 1000, 40_000.0);
                s.ctl.observe(&o);
            }
            let plan = s.plan_window(&emitted, a[7], 1000, 16, 4096, 16, 100_000);
            match plan {
                WindowPlan::Copy { take } => {
                    assert!((1..=15).contains(&take), "{take}");
                    assert!(take <= s.proposal.len());
                }
                other => panic!("expected a Copy window, got {other:?}"),
            }
            // The same state clamps to max_emit: rows <= max_emit.
            let plan = s.plan_window(&emitted, a[7], 1000, 16, 4, 16, 100_000);
            if let WindowPlan::Copy { take } = plan {
                assert!(take <= 3, "{take}");
            }
        }

        /// `no_tail_action` against an independent enumeration of the decision
        /// space `TailSource::splice` can face (`k = 1..15`, `offered =
        /// 1..=15-k`, the same `plan` + `clamp_tail_take`) on fresh, priced and
        /// tail-trained controllers.
        #[test]
        fn no_tail_action_matches_the_splice_decision_space() {
            let (prompt, a) = mk_history();
            let mut s = st(true);
            s.begin_request(&prompt);
            let base = lookup_budget(16, 16, 64, 1000, 100_000);
            let decides = |ctl: &LookupController, k: usize, offered: usize| -> bool {
                let mut b = base;
                b.native_candidates = 15;
                b.model_drafts = k;
                b.lookup_available = offered;
                match ctl.plan(b, 1000) {
                    LookupAction::Tail { take } => {
                        clamp_tail_take(take, offered, k, 15, &b).is_some()
                    }
                    _ => false,
                }
            };
            let any_take =
                |ctl: &LookupController| (1..15).any(|k| (1..=15 - k).any(|o| decides(ctl, k, o)));
            let mut states = vec![s.ctl.clone()];
            for _ in 0..12 {
                let o = window_outcome(Ran::Native, 16, 1, &a[8..23], &a[8..23], 1000, 40_000.0);
                s.ctl.observe(&o);
            }
            states.push(s.ctl.clone());
            for _ in 0..12 {
                let o = window_outcome(
                    Ran::Tail { k: 3, l: 4 },
                    8,
                    7,
                    &a[8..15],
                    &a[8..15],
                    1000,
                    30_000.0,
                );
                s.ctl.observe(&o);
            }
            states.push(s.ctl.clone());
            for ctl in &states {
                assert_eq!(no_tail_action(ctl, base, 1000, 15), !any_take(ctl));
            }
            // Degenerate windows have no tail slot: always declined.
            assert!(no_tail_action(&states[0], base, 1000, 0));
            assert!(no_tail_action(&states[0], base, 1000, 1));
        }

        #[test]
        fn lookup_arming_reasons_and_load_lines() {
            assert_eq!(arming(&inputs()), Ok(true));
            assert_eq!(
                arming(&ArmInputs {
                    tail_env: false,
                    ..inputs()
                }),
                Ok(false)
            );
            assert_eq!(
                arming(&ArmInputs {
                    on: false,
                    ..inputs()
                }),
                Err("speculation.dflash_lookup is off")
            );
            assert_eq!(
                arming(&ArmInputs {
                    ddtree: true,
                    ..inputs()
                }),
                Err("DDTree chain")
            );
            assert_eq!(
                arming(&ArmInputs {
                    selector: false,
                    ..inputs()
                }),
                Err("no DFlash2 candidate selector")
            );
            assert_eq!(
                arming(&ArmInputs {
                    pm4: true,
                    ..inputs()
                }),
                Err("retained PM4 verify is shape-frozen at B=16")
            );
            assert_eq!(
                arming(&ArmInputs {
                    block_size: 2,
                    ..inputs()
                }),
                Err("verify block < 3 rows")
            );
            assert_eq!(
                load_line(&Ok(true), 16),
                "[dflash] lookup: armed (copy+tail, rows<=16)"
            );
            assert_eq!(
                load_line(&Ok(false), 8),
                "[dflash] lookup: armed (copy, rows<=8)"
            );
            assert_eq!(
                load_line(&Err("DDTree chain".into()), 16),
                "[dflash] lookup: off (DDTree chain)"
            );
        }

        #[test]
        fn lookup_unarmed_builds_no_state() {
            for i in [
                ArmInputs {
                    on: false,
                    ..inputs()
                },
                ArmInputs {
                    ddtree: true,
                    ..inputs()
                },
                ArmInputs {
                    selector: false,
                    ..inputs()
                },
                ArmInputs {
                    pm4: true,
                    ..inputs()
                },
            ] {
                let (state, line) = build_state(&i, (5, 1, 63));
                assert!(state.is_none());
                assert!(line.starts_with("[dflash] lookup: off ("), "{line}");
            }
            let (state, line) = build_state(&inputs(), (5, 1, 63));
            let state = state.expect("armed");
            assert!(state.tail);
            assert_eq!(state.max_rows, 16);
            assert_eq!(line, "[dflash] lookup: armed (copy+tail, rows<=16)");
            assert!(!state.ctx.is_begun());
            assert!(state.proposal.capacity() >= LOOKUP_MAX_CANDIDATES);
            let (state, line) = build_state(
                &ArmInputs {
                    tail_env: false,
                    block_size: 8,
                    ..inputs()
                },
                (5, 1, 63),
            );
            let state = state.expect("armed copy-only");
            assert!(!state.tail);
            assert_eq!(state.max_rows, 8);
            assert_eq!(line, "[dflash] lookup: armed (copy, rows<=8)");
            // An unrunnable triple never arms and says why.
            let (state, line) = build_state(&inputs(), (5, 9, 3));
            assert!(state.is_none());
            assert!(
                line.starts_with("[dflash] lookup: off (copy context: invalid n-gram triple"),
                "{line}"
            );
        }

        #[test]
        fn lookup_copy_config_matches_the_qwen4_copy_lane() {
            let c = copy_config(hipfire_config::NGRAM_MOD_COPY_TRIPLE).unwrap();
            assert_eq!((c.n_match, c.n_min, c.n_max), (5, 1, 15));
            assert_eq!(c.capacity, NgramModConfig::default().capacity);
            let c = copy_config((8, 20, 30)).unwrap();
            assert_eq!((c.n_match, c.n_min, c.n_max), (8, 15, 15));
            assert!(copy_config((0, 1, 3)).is_err());
            assert!(copy_config((5, 1, 65)).is_err());
            assert!(copy_config((5, 4, 3)).is_err());
        }
    }
}

/// Lower a qwen35 `SpecStepResult` onto the arch-generic `SpecStep`.
///
/// The daemon-called `spec_step_*` build `committed = [seed, drafts.., bonus]`,
/// so `committed[1..]` is exactly the daemon's `committed_tail` (the tokens
/// emitted this window) and its length is `accepted + 1` — which is why the
/// unified loop advances `position` by `emit.len()`.
fn lower_qwen35(r: &SpecStepResult) -> SpecStep {
    SpecStep::new(
        r.committed[1..].iter().copied(),
        r.bonus_token,
        r.drafted.len(),
        r.accepted,
    )
}

/// Error of every operation that needs the singleton's lane state while a VMM
/// promotion has moved it out (see [`DflashSpeculator::take_vmm_lane`]).
const SINGLETON_VACANT: &str = "DFlash singleton lane state is vacant; prefill first";

/// DFlash / DDTree speculator: wraps the qwen35 `spec_step_*` chain/tree
/// kernels behind the arch-generic [`Speculator`] trait. Chain-vs-tree is an
/// internal detail resolved at build (`ddtree` presence comes from the loaded
/// `DflashState`).
///
/// Owns the `DflashState` moved out of `LoadedModel.dflash`, plus the divergent-
/// render DeltaNet checkpoint ring folded in from `LoadedModel.dflash_checkpoints`.
///
/// The lane state is VACANT after [`Self::take_vmm_lane`] (a promoted request
/// owns it): exactly one of `df` / `vacant` is `Some`. The speculator then
/// holds only the shared-asset descriptor (a counted weights reference) and
/// rebuilds a fresh state lazily on the next `prefill` ([`Self::ensure_live`]),
/// or reinstalls the snapshot ([`Self::restore_vmm_lane`]); no replacement is
/// allocated while the lane lives.
pub struct DflashSpeculator {
    /// Live singleton owners; `None` while vacant.
    df: Option<DflashState>,
    /// Shared assets while `df` is vacant (holds one `Arc` reference to the
    /// draft weights, released when the state is rebuilt/restored/freed).
    vacant: Option<DflashVmmAssets>,
    rng_state: u64,
    /// Per-request sampling, set via `set_sampling` before each step loop and
    /// applied in the chain-mode `spec_step_dflash` branch of `step`. Default
    /// greedy (temp 0 / top_p 1 / top_k 0 / cactus 0) → argmax-accept, the
    /// historical DFlash posture, so an unconfigured speculator (or the
    /// greedy-only DDTree branches) decode greedily. Mirrors spec-graph's old
    /// inline `generate_dflash` call, which threaded the request temp/top_p/top_k
    /// into the same four `spec_step_dflash` args.
    sample_temp: f32,
    sample_top_p: f32,
    sample_top_k: usize,
    sample_cactus: f32,
    /// Divergent-render checkpoint ring. Populated by `prefill`'s seed when
    /// `resume_enabled`; freed on `reset`/`free`.
    checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    resume_enabled: bool,
    ck_interval: usize,
    ck_cap: usize,
    last_window: Option<DflashWindowMark>,
    /// Adaptive verify-block controller (trailing-τ proposal shrink),
    /// owned by `hipfire_runtime::dflash_adaptive_block` so the policy is
    /// family-free. Disabled (fixed full block) when the load knob opted out,
    /// `HIPFIRE_DFLASH_ADAPTIVE_B=0` wins, or the retained-PM4 verify route
    /// was admitted (its replay is shape-frozen at B=16; see
    /// `build_dflash_speculator`). Reset to full at request start.
    adaptive: DflashAdaptiveBlock,
    /// Resolved adaptive knob passed to `new` (the controller's `enabled` is
    /// private); VMM lanes require it to be false (fixed block).
    adaptive_enabled: bool,
    /// `HIPFIRE_DFLASH_CYCLE_TRACE` request ordinal (process-wide counter value
    /// taken at the last `prefill`) and per-request cycle index. Diagnostic only.
    trace_seq: u64,
    trace_cyc: u64,
    /// Request-local verbatim-copy lookup composed into the verify window
    /// (`speculation.dflash_lookup`). `None` = not armed: the unarmed `step`
    /// is the incumbent window with no lookup state at all.
    lookup: Option<lookup::LookupState>,
    /// Certified railgun DFlash window owner (design A). `None` unless
    /// `HIPFIRE_DFLASH_RAILGUN_WINDOW=1` was read at build: the unset path
    /// never constructs it, so it is the incumbent speculator exactly.
    railgun: Option<crate::dflash_railgun::RailgunWindowOwner>,
}

/// Pre-window mark of the last completed window (terminal-prefix repair).
#[derive(Clone, Copy, Debug)]
pub struct DflashWindowMark {
    pub position: usize,
    pub seed: u32,
    pub target_hidden: TargetHiddenLogMark,
}

impl DflashSpeculator {
    /// `resume_enabled`/`ck_interval`/`ck_cap` mirror the daemon's
    /// `ckpt_resume_enabled()`/`ckpt_interval()`/`ckpt_max()` — passed in by
    /// `build_dflash_speculator` so `new` itself is env-free (and unit-testable).
    /// `adaptive_b` is the already-resolved knob (load param AND env
    /// kill-switch AND the retained-PM4 gate folded by `build_dflash_speculator`)
    /// — likewise env-free and PM4-free here.
    pub fn new(
        df: DflashState,
        resume_enabled: bool,
        ck_interval: usize,
        ck_cap: usize,
        adaptive_b: bool,
    ) -> Self {
        let adaptive = DflashAdaptiveBlock::new(df.block_size, adaptive_b);
        Self {
            df: Some(df),
            vacant: None,
            // Same fixed seed the daemon's DFlash loop used. `set_sampling`
            // re-seeds it to this value per request (matching spec-graph's local
            // `let mut rng_state = 0x13579BDF` per `generate_dflash` call) so a
            // sampled request is deterministic given its seed; greedy decode does
            // not consume it.
            rng_state: 0x13579BDF,
            // Greedy by default until a request calls `set_sampling`.
            sample_temp: 0.0,
            sample_top_p: 1.0,
            sample_top_k: hipfire_runtime::llama::ar_candidate_cap(None),
            sample_cactus: 0.0,
            checkpoints: Vec::new(),
            resume_enabled,
            ck_interval,
            ck_cap,
            last_window: None,
            adaptive,
            adaptive_enabled: adaptive_b,
            trace_seq: 0,
            trace_cyc: 0,
            lookup: None,
            railgun: None,
        }
    }

    /// Borrow the retained-PM4 verify route (report / phase inspection).
    /// `None` while the lane state is vacant.
    pub fn verify_pm4(&self) -> Option<&DflashVerifyPm4> {
        self.df.as_ref().map(|df| &df.verify_pm4)
    }

    /// Mutable borrow for the chain verify path (`Some(&mut df.verify_pm4)`).
    /// `None` while the lane state is vacant.
    pub fn verify_pm4_mut(&mut self) -> Option<&mut DflashVerifyPm4> {
        self.df.as_mut().map(|df| &mut df.verify_pm4)
    }

    /// Resolved DeltaNet checkpoint cadence of this speculator.
    pub fn checkpoint_policy(&self) -> DflashCheckpointPolicy {
        DflashCheckpointPolicy {
            resume_enabled: self.resume_enabled,
            interval: self.ck_interval,
            cap: self.ck_cap,
        }
    }

    /// The lane state is currently moved out ([`Self::take_vmm_lane`]); the
    /// next `prefill` ([`Self::ensure_live`]) rebuilds it.
    pub fn is_vacant(&self) -> bool {
        self.df.is_none()
    }

    /// Arm (or leave off) the verbatim-copy lookup and print the one load
    /// line. `on` = resolved `speculation.dflash_lookup`, `tail_env` = the
    /// `HIPFIRE_SPEC_LOOKUP_TAIL` kill switch is not `0`. Armed only on this
    /// singleton DFlash2 selector chain (no DDTree, no retained-PM4 verify);
    /// the VMM/CB lanes and the dense-TP mesh speculator are other types and
    /// never arm; CACTUS is excluded per window in `step`.
    pub fn configure_lookup(&mut self, on: bool, tail_env: bool) {
        let Some(df) = self.df.as_ref() else {
            self.lookup = None;
            return;
        };
        let inputs = lookup::ArmInputs {
            on,
            tail_env,
            ddtree: df.ddtree.is_some(),
            selector: df.draft_weights.has_candidate_selector(),
            pm4: df.verify_pm4.admitted(),
            block_size: df.block_size,
        };
        let (state, line) = lookup::build_state(&inputs, hipfire_config::ngram_mod_copy_triple());
        eprintln!("{line}");
        self.lookup = state;
    }

    /// Drop an armed lookup once VMM DFlash lanes are staged for this load,
    /// printing `[dflash] lookup: off (<why>)`. Lanes never run lookup and a
    /// request can move between the singleton and a lane mid-decode, so an
    /// armed singleton would make a request's text depend on server load
    /// (the same reason the lanes refuse the adaptive block). No-op when the
    /// lookup was never armed.
    pub fn disarm_lookup(&mut self, why: &str) {
        if self.lookup.take().is_some() {
            eprintln!("{}", lookup::load_line(&Err(why.to_string()), 0));
        }
    }
}

impl Speculator for DflashSpeculator {
    fn name(&self) -> &'static str {
        "dflash"
    }

    fn prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        prefill_tokens: &[u32],
        prefill_start: usize,
        cache_hit: bool,
        resume_from: Option<usize>,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        self.adaptive.reset();
        if let Some(lk) = self.lookup.as_mut() {
            lk.end_request();
        }
        self.last_window = None;
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;

        // Rebuild the lane state when a VMM promotion moved it out (the
        // conversation caches were cleared with it, so this is a cold prefill).
        self.ensure_live(gpu, &slot.config, &slot.dn_state)?;
        let df = self
            .df
            .as_mut()
            .ok_or_else(|| SINGLETON_VACANT.to_string())?;
        if let Some(t) = df.draft_scratch.online.as_mut() {
            t.seed_prompt(prompt_tokens);
        }

        // A CPU shadow exists only while the cold backfill reads it (see
        // `dflash_prefill_host_required`). Cold prompts drop draft
        // upload/projection tracking; cache hits preserve the prior turn's GPU
        // projections and extend its suffix.
        let mut host_hidden = Vec::new();
        let capture_host =
            dflash_prefill_host_required(df.draft_scratch.ctx_mode, prompt_tokens.len());
        if !cache_hit {
            df.draft_scratch.reset_upload_tracking();
        }

        // Seed the target's hidden state into the drafter ring (chunked prefill
        // with hidden extraction). Cache hit → seed only the suffix from
        // `prefill_start`, reusing the prior turn's KV + recurrent state; miss →
        // seed the full prompt (the seed fn resets target state itself).
        let (ck_interval, ck_cap) = (self.ck_interval, self.ck_cap);
        let ckpt_sink = if self.resume_enabled {
            Some(&mut self.checkpoints)
        } else {
            None
        };
        let aborted = if cache_hit {
            seed_target_hidden_suffix_abortable(
                gpu,
                slot,
                &mut df.hidden_rb,
                prefill_tokens,
                prefill_start,
                abort,
                ckpt_sink,
                ck_interval,
                ck_cap,
            )
        } else {
            seed_target_hidden_from_prompt_abortable(
                gpu,
                slot,
                &mut df.hidden_rb,
                capture_host.then_some(&mut host_hidden),
                prefill_tokens,
                abort,
                ckpt_sink,
                ck_interval,
                ck_cap,
            )
        }
        .map_err(|e| {
            hipfire_runtime::reset_core::note_hip_error(&e, "qwen35::dflash_prefill::seed");
            e.to_string()
        })?;
        if aborted {
            // Caller resets conversation state + emits aborted/done; the slot
            // guard restores the target bundle on the way out.
            return Ok(PrefillOutcome::Aborted);
        }

        // Prime/extend the draft's GPU target_hidden buffer, backfill, seed the
        // thlog and read the post-prompt logits (shared with the VMM lane
        // prefill: `dflash_prefill_finish`).
        //
        // First emit = target draw at the final prompt position (seed already
        // ran the per-token forward; scratch.logits holds the post-prompt logits).
        // temp≈0 stays the historical host argmax fold (byte-identical greedy).
        // temp>0 uses the same host nucleus sampler as chain DFlash verify so the
        // post-prefill seed is not a special greedy exception on distribution-
        // preserving requests.
        let first_logits = dflash_prefill_finish(
            gpu,
            df,
            &host_hidden,
            &slot.scratch.logits,
            prompt_tokens,
            prefill_tokens.len(),
            cache_hit,
            prefill_start,
            resume_from,
        )?;
        let first_token = dflash_first_token(
            &first_logits,
            df.ddtree.is_none(),
            crate::speculative::DflashSampleCfg {
                temp: self.sample_temp,
                top_p: self.sample_top_p,
                top_k: self.sample_top_k,
                cactus_delta: self.sample_cactus,
            },
            &mut self.rng_state,
            &mut Vec::new(),
        );
        if cycle_trace::enabled() {
            self.trace_seq = cycle_trace::next_seq();
            self.trace_cyc = 0;
            cycle_trace::emit(&cycle_trace::prefill_line(
                std::process::id(),
                self.trace_seq,
                prompt_tokens,
                prefill_start,
                cache_hit,
            ));
        }
        if let Some(lk) = self.lookup.as_mut() {
            // Canonical full prompt (cache hits included); the first token
            // reaches the history through `emitted`.
            lk.begin_request(prompt_tokens);
        }
        Ok(PrefillOutcome::Ready { first_token })
    }

    /// Forced tokens (think-budget force-close) must land in the drafter's
    /// per-position `target_hidden` cache, not just the target's KV. Seeding via
    /// the same suffix path the prompt-cache HIT uses advances the target WITH
    /// hidden extraction, so the rows exist and `thlog` stays contiguous.
    ///
    /// Skipping this is what previously left an uninitialized (NaN) hole at the
    /// forced positions: the next draft forward read it, produced all-NaN logits,
    /// and `argmax` collapsed to token 0 — τ went to 0 for the rest of the
    /// session and stayed dead across prompt-cache HITs.
    fn on_forced_advance(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        if tokens.is_empty() {
            return Ok(true);
        }
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;
        let df = self
            .df
            .as_mut()
            .ok_or_else(|| SINGLETON_VACANT.to_string())?;
        let aborted = seed_target_hidden_suffix_abortable(
            gpu,
            slot,
            &mut df.hidden_rb,
            tokens,
            start_pos,
            abort,
            None,
            self.ck_interval,
            self.ck_cap,
        )
        .map_err(|e| e.to_string())?;
        if aborted {
            // Caller tears the request down; leaving the rows unwritten is fine
            // because the drafter state is reset on the way out.
            return Ok(true);
        }
        scatter_hidden_block_to_interleaved(
            gpu,
            &df.hidden_rb,
            &df.draft_scratch.target_hidden,
            start_pos,
            tokens.len(),
            tokens.len(),
            df.draft_scratch.ctx_modulus(),
        )
        .map_err(|e| e.to_string())?;
        let co = slot.kv_cache_mut().map(|kv| kv.compact_offset).unwrap_or(0) as i32;
        df.draft_scratch
            .thlog
            .append_committed(start_pos, tokens.len(), co);
        if cycle_trace::enabled() {
            cycle_trace::emit(&cycle_trace::forced_line(
                std::process::id(),
                self.trace_seq,
                start_pos,
                tokens,
            ));
        }
        Ok(true)
    }

    fn supports_ar_step(&self) -> bool {
        self.df.is_some()
    }

    /// AR's single-token decode forward plus this position's hidden rows, then
    /// the same scatter + log append a forced advance does, so the draft
    /// context stays whole for a later request on this prompt cache.
    fn ar_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
    ) -> Result<u32, String> {
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;
        let df = self
            .df
            .as_mut()
            .ok_or_else(|| SINGLETON_VACANT.to_string())?;
        // No retained window: a terminal stop after this token flushes plainly.
        self.last_window = None;
        qwen35::forward_scratch_with_hidden(
            gpu,
            &slot.weights,
            &slot.config,
            seed,
            position,
            &mut slot.kv_cache,
            &mut slot.dn_state,
            &slot.scratch,
            &mut df.hidden_rb,
        )
        .map_err(|e| e.to_string())?;
        scatter_hidden_block_to_interleaved(
            gpu,
            &df.hidden_rb,
            &df.draft_scratch.target_hidden,
            position,
            1,
            1,
            df.draft_scratch.ctx_modulus(),
        )
        .map_err(|e| e.to_string())?;
        let co = slot.kv_cache.compact_offset as i32;
        df.draft_scratch.thlog.append_committed(position, 1, co);
        // The verify's argmax kernel and buffer: same tie-break, no per-token alloc.
        let argmax = df.verify_scratch.argmax.sub_offset(0, 1);
        gpu.argmax_f32_batched(&slot.scratch.logits, &argmax, slot.config.vocab_size, 1)
            .map_err(|e| e.to_string())?;
        crate::speculative::dflash_download_verify_argmax(gpu, &df.verify_scratch, 1)
            .map(|t| t[0])
            .map_err(|e| e.to_string())
    }

    /// Temp>0 verify is distribution-correct on the ddtree-batched arm (SWOR)
    /// and on DFlash2 selector-chain rejection sampling. Legacy chain without
    /// a selector stays greedy-only at this gate.
    fn supports_temp_verify(&self) -> bool {
        match (&self.df, &self.vacant) {
            (Some(df), _) => df.ddtree.is_some() || df.draft_weights.has_candidate_selector(),
            // A vacancy only exists for fixed-chain assets: no DDTree.
            (None, Some(assets)) => assets.weights.has_candidate_selector(),
            (None, None) => false,
        }
    }

    /// Faithful top_p/top_k nucleus only on the DFlash2 candidate-selector chain.
    /// DDTree SWOR returns false so route selection still blocks user-explicit
    /// non-temperature controls for the tree path.
    fn supports_chain_nucleus_verify(&self) -> bool {
        match (&self.df, &self.vacant) {
            (Some(df), _) => df.draft_weights.has_candidate_selector(),
            (None, Some(assets)) => assets.weights.has_candidate_selector(),
            (None, None) => false,
        }
    }

    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        _grammar: Option<&mut dyn SpecGrammar>,
        temp: f32,
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        let t0 = cycle_trace::enabled().then(std::time::Instant::now);
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;
        let df = self
            .df
            .as_mut()
            .ok_or_else(|| SINGLETON_VACANT.to_string())?;

        if max_emit == 0 {
            return Err("DflashSpeculator: max_emit=0 (no remaining output budget)".into());
        }
        // Chain DFlash: emit ≤ b (accepted drafts + bonus). Cap block size so the
        // verify window cannot commit past remaining client budget. b >= 2.
        // emit = accept + 1 ≤ b when seed is excluded. Prefer b = max_emit
        // (uniform for max_emit >= 1); max_accept clamps accept before commit
        // so max_emit == 1 is a true one-token path (accept 0 + bonus).
        let block_override = {
            // Adaptive proposal width at this absolute position: the
            // trailing-τ controller's effective block, gated on context
            // length (full below ADAPTIVE_CTX_GATE where the B×L scan is
            // trivial; full until 8 cycles observed / when opted out),
            // further bounded by the remaining client budget below.
            // Proposing fewer rows only shortens the accepted run —
            // rows 1..=k are the causal prefix rows, never a different
            // prefix (bound documented on DflashAdaptiveBlock).
            let cfg_b = self
                .adaptive
                .effective_at(position)
                .max(2)
                .min(df.block_size.max(2));
            let want = max_emit.max(2);
            let b = cfg_b.min(want);
            if b < cfg_b || b != df.draft_config.block_size {
                Some(b)
            } else {
                None
            }
        };
        // accepted drafts + bonus = emit; max accepted drafts = max_emit - 1.
        let max_accept = Some(max_emit.saturating_sub(1));

        let window_mark = DflashWindowMark {
            position,
            seed,
            target_hidden: df.draft_scratch.thlog.mark(),
        };

        // Two-way dispatch: DDTree-batched (SWOR) when a tree is configured
        // (never for DFlash2 selector — load refused construction), else
        // chain-mode DFlash. Selector chain uses sparse-q rejection at temp>0.
        // The grammar arg is ignored — qwen35 enforces tool-call grammar post-hoc
        // in the daemon.
        // The production chain/tree calls below use ctx_slice=None, so no
        // persistent host shadow exists or is read after prompt backfill.
        let mut host_hidden = Vec::new();
        // Armed-lookup bookkeeping of this window. `lk_ran` is `None` when
        // lookup is inactive for it (unarmed, DDTree, CACTUS, or the pending
        // seed is not `emitted.last()`): the incumbent call, nothing observed.
        let mut lk_ran: Option<lookup::Ran> = None;
        let mut lk_probe = false;
        if let Some(owner) = self.railgun.as_mut() {
            owner.clear_route();
        }
        let mut lk_wall_us = 0.0f64;
        let result = if let Some(dd) = df.ddtree.as_mut() {
            // Tree node budget is structural; max_accept is the commit bound.
            // Keep at least 1 node so the tree builder stays well-formed; the
            // accept clamp drops to 0 drafts when max_emit == 1.
            let tree_budget = dd.budget.min(max_emit.saturating_sub(1).max(1));
            spec_step_ddtree_batched(
                gpu,
                slot,
                &df.draft_weights,
                &df.draft_config,
                &mut df.draft_scratch,
                &mut df.hidden_rb,
                &mut host_hidden,
                &mut df.target_snap,
                &mut dd.post_seed_snap,
                &mut df.gdn_tape,
                &dd.scratch,
                &df.verify_scratch,
                position,
                seed,
                None, // ctx_slice = full history
                tree_budget,
                dd.topk,
                // Request temperature → distribution-preserving SWOR verify at
                // temp>0 (greedy/argmax at temp 0). The ddtree-batched arm is the
                // only DFlash mode with sampled verify; the chain below stays
                // greedy, so `supports_temp_verify` gates serve routing to ddtree.
                temp,
                &mut self.rng_state,
                max_accept,
            )
        } else {
            // Selector chain must error on invalid rewrites rather than silently
            // substituting 0/None — keep the CACTUS guard here (the other
            // rewrites are hard-wired to off/None in this path, but the direct
            // `spec_step_dflash` caller covers them).
            if df.draft_weights.has_candidate_selector() && self.sample_cactus != 0.0 {
                return Err(
                    "selector mode: CACTUS is not supported with DFlash2 candidate selector".into(),
                );
            }
            // Lookup planning (armed only). The unarmed path and any window
            // with no lookup proposal and no Tail offer reach the call below
            // with `block_override` and `lookup = None`: the incumbent call.
            let native_b = block_override.unwrap_or(df.draft_config.block_size);
            let mut lk: Option<&mut lookup::LookupState> = if self.sample_cactus == 0.0 {
                self.lookup.as_mut()
            } else {
                None
            };
            let plan = match lk.as_deref_mut() {
                Some(l) => l.plan_window(
                    emitted,
                    seed,
                    position,
                    native_b,
                    max_emit,
                    df.block_size,
                    df.ctx_capacity,
                ),
                None => lookup::WindowPlan::Off,
            };
            // Copy runs at `1 + take` rows (the learned draft forward runs at
            // that width and its candidates are discarded); a price probe is
            // ONE native window at the wider row count that prices the Copy
            // it would need (its acceptance feeds the lookup shadow only).
            let (call_block, mut ran) = match plan {
                lookup::WindowPlan::Copy { take } => {
                    (Some(1 + take), Some(lookup::Ran::Copy { take }))
                }
                lookup::WindowPlan::Probe { rows } => {
                    lk_probe = true;
                    (Some(rows), Some(lookup::Ran::Native))
                }
                lookup::WindowPlan::Native { .. } => (block_override, Some(lookup::Ran::Native)),
                lookup::WindowPlan::Off => (block_override, None),
            };
            let mut tail_src: Option<lookup::TailSource<'_>> = None;
            let lookup_arg: Option<DflashLookup<'_>> = match (plan, lk.as_deref_mut()) {
                (lookup::WindowPlan::Copy { take }, Some(l)) => {
                    Some(DflashLookup::Copy(&l.proposal[..take]))
                }
                (lookup::WindowPlan::Native { tail: true }, Some(l)) => {
                    let mut base = lookup::lookup_budget(
                        native_b,
                        df.block_size,
                        max_emit,
                        position,
                        df.ctx_capacity,
                    );
                    base.max_rows = base.max_rows.min(l.max_rows);
                    tail_src = Some(lookup::TailSource {
                        ctx: &mut l.ctx,
                        ctl: &l.ctl,
                        emitted,
                        base,
                        position,
                        cand: &mut l.cand,
                        spliced: None,
                    });
                    tail_src.as_mut().map(|s| DflashLookup::Tail(s))
                }
                _ => None,
            };
            let call_t0 = ran.is_some().then(std::time::Instant::now);
            let res = spec_step_dflash_with_railgun(
                gpu,
                slot,
                &df.draft_weights,
                &df.draft_config,
                &mut df.draft_scratch,
                &mut df.hidden_rb,
                &mut host_hidden,
                &mut df.target_snap,
                &df.verify_scratch,
                position,
                seed,
                None, // ctx_slice = full history
                Some(&mut df.gdn_tape),
                // Sampling threaded from the request via `set_sampling` (#477
                // merge re-wire). These four positions reproduce spec-graph's old
                // inline `generate_dflash` call verbatim: temp 0 ⇒ greedy/argmax;
                // temp>0 ⇒ lossless rejection sampling with the IDENTICAL
                // (top_k,top_p) nucleus truncation on draft + target. The DDTree
                // branches above stay greedy (tree-verify is greedy by
                // construction) and ignore these.
                self.sample_temp,
                self.sample_top_p, // top_p (1.0 = no truncation)
                self.sample_top_k, // top_k (0 = top_p-only)
                &mut self.rng_state,
                call_block, // remaining-output budget (Copy / probe: the lookup width)
                None,       // ngram_cache
                emitted,
                self.sample_cactus, // selector already checked — safe to pass through
                None,               // pld_spine
                1.0_f32,            // repeat_penalty (off)
                0,                  // repeat_window
                max_accept,
                Some(&mut df.verify_pm4),
                lookup_arg,
                self.railgun.as_mut().map(|owner| RailgunStep {
                    owner,
                    probe: lk_probe,
                }),
            );
            if let Some(t) = call_t0 {
                lk_wall_us = t.elapsed().as_secs_f64() * 1e6;
            }
            // A Tail offer that spliced ran a Tail window; one that declined
            // ran the incumbent window unchanged (Native, with a shadow).
            if let Some((k, l)) = tail_src.as_ref().and_then(|s| s.spliced) {
                ran = Some(lookup::Ran::Tail { k, l });
            }
            lk_ran = ran;
            res
        };
        // `online` (`HIPFIRE_DFLASH_ONLINE_TUNE`) labels its pending proposal
        // rows with the tokens the TARGET committed after the seed. Those are
        // the true labels for any window kind (a Copy / Tail window verifies
        // lookup rows, but what it commits is still the target's own stream),
        // and skipping a window would leave its pending cycle unlabelled
        // forever (the tuner finalizes strictly in order), so it is fed
        // every chain window exactly as before.
        if df.ddtree.is_none() {
            if let (Ok(r), Some(t)) = (&result, df.draft_scratch.online.as_mut()) {
                t.observe(position, &r.committed[1..], r.accepted);
            }
        }
        // Armed lookup: the controller learns from the window it actually ran
        // (`LookupController::observe` owns the censoring / source-separated
        // rules), and only Native windows feed the adaptive-B controller: a
        // Copy / Tail window's accepted count measures the lookup rows, not
        // the learned chain the controller sizes the block for, and a price
        // probe runs at a width the controller did not choose.
        let mut lk_trace: Option<(&'static str, usize, usize)> = None;
        let mut feed_adaptive = true;
        if let (Some(ran), Ok(r), Some(lk)) = (lk_ran, &result, self.lookup.as_mut()) {
            let o = lookup::window_outcome(
                ran,
                r.drafted.len(),
                r.accepted,
                &lk.proposal,
                r.committed.get(1..).unwrap_or(&[]),
                position,
                lk_wall_us,
            );
            lk.ctl.observe(&o);
            // A price probe is a Native window run wider than the incumbent
            // would have: tag it so receipts never count it as incumbent.
            lk_trace = Some(lookup::trace_tag(&o, lk.proposal.len()))
                .map(|(kind, k, l)| (if lk_probe { "probe" } else { kind }, k, l));
            feed_adaptive = matches!(ran, lookup::Ran::Native) && !lk_probe;
        }

        let lowered = result
            .as_ref()
            .map(lower_qwen35)
            // Defense only — accept stage already committed ≤ max_emit.
            .map(|s| s.cap_emit(max_emit))
            .map_err(|e| e.to_string());
        // Feed the committed accept count into the trailing-τ controller for
        // the NEXT cycle's proposal width (cap_emit already reconciled
        // `accepted` with the truncated emit, so this is what really
        // committed). DDTree has its own structural budget — only the chain
        // proposal width shrinks; the tree budget is never fed here.
        if let (Ok(s), true) = (&lowered, feed_adaptive) {
            let prev = self.adaptive.effective();
            let next = self.adaptive.observe(s.accepted);
            if next != prev {
                // Log with the gate state: below ADAPTIVE_CTX_GATE the internal
                // width follows τ̂ while effective_at() keeps returning the full
                // block, so a serve.log reader must not mistake a logged
                // "shrink" for an applied proposal width (this run's verify
                // width is the `applied` column).
                let applied = self.adaptive.effective_at(position);
                eprintln!(
                    "[dflash] adaptive_b {}: internal {} -> {} (tau_hat {:.2} over last 8, full {}; applied {}, ctx {position}){}",
                    if next < prev { "shrink" } else { "recover" },
                    prev,
                    next,
                    self.adaptive.tau_hat().unwrap_or(0.0),
                    self.adaptive.full(),
                    applied,
                    if applied != next {
                        " — gated below 2k ctx: internal follow only, verify holds full"
                    } else {
                        ""
                    },
                );
            }
        }
        if let Some(t0) = t0 {
            let step_us = t0.elapsed().as_micros() as u64;
            if let (Ok(r), Ok(s)) = (&result, &lowered) {
                let cyc = self.trace_cyc;
                self.trace_cyc += 1;
                cycle_trace::emit(
                    &cycle_trace::Cycle {
                        pid: std::process::id(),
                        seq: self.trace_seq,
                        cyc,
                        pos: position,
                        seed,
                        // `SpecStepResult.drafted` is `[seed, d1..d(b-1)]`:
                        // its length is the verify rows, the candidates
                        // follow the seed.
                        b: r.drafted.len(),
                        max_emit,
                        temp: if df.ddtree.is_some() {
                            temp
                        } else {
                            self.sample_temp
                        },
                        drafted: r.drafted.get(1..).unwrap_or(&[]),
                        accepted: r.accepted,
                        committed: &s.emit,
                        step_us,
                        ddtree: df.ddtree.is_some(),
                        lookup: lk_trace.map(|(kind, k, lookup)| cycle_trace::LookupTag {
                            kind,
                            k,
                            lookup,
                        }),
                        railgun: self.railgun.as_ref().and_then(|o| o.last_route()),
                    }
                    .line(),
                );
            }
        }
        self.last_window = lowered.as_ref().ok().map(|_| window_mark);
        lowered
    }

    fn on_evict(&mut self, gpu: &mut Gpu, retain: &EvictRetain) -> Result<(), String> {
        self.last_window = None;
        if let Some(owner) = self.railgun.as_mut() {
            owner.invalidate();
        }
        // Compact the drafter's cached target-hidden rows to match the target KV
        // after the FlashCASK eviction the daemon already applied to the target.
        let df = self
            .df
            .as_mut()
            .ok_or_else(|| SINGLETON_VACANT.to_string())?;
        let ne = df.draft_config.num_extract();
        let h = df.draft_config.hidden;
        apply_eviction_retain_to_draft(
            gpu,
            &mut df.draft_scratch,
            &retain.retain_mask,
            ne,
            h,
            retain.pre_phys,
        )
        .map_err(|e| e.to_string())
    }

    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if cycle_trace::enabled() {
            cycle_trace::emit(&cycle_trace::reset_line(std::process::id(), self.trace_seq));
        }
        // Drafter-local reset: invalidate cached suffix projections and free the
        // divergent-render checkpoint ring (the target KV/recurrent reset is the
        // daemon's job — it owns the bundle). The adaptive window reseeds to
        // full too — the fail-safe direction (today's fixed-B behaviour).
        // (Vacant: the lane owns the draft scratch; nothing to reset here.)
        if let Some(df) = self.df.as_mut() {
            df.draft_scratch.reset_upload_tracking();
        }
        self.last_window = None;
        if let Some(owner) = self.railgun.as_mut() {
            owner.invalidate();
        }
        self.adaptive.reset();
        if let Some(lk) = self.lookup.as_mut() {
            lk.end_request();
        }
        for (_, snap) in self.checkpoints.drain(..) {
            snap.free_gpu(gpu);
        }
        Ok(())
    }

    fn repair_terminal_prefix(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        window_start: usize,
        window_seed: u32,
        consumed: &[u32],
    ) -> Result<bool, String> {
        if let Some(lk) = self.lookup.as_mut() {
            // `emitted` shrinks to the consumed prefix: rebuild the history.
            lk.mark_dirty();
        }
        if self.df.is_none() {
            return Err(SINGLETON_VACANT.into());
        }
        let mark = self
            .last_window
            .take()
            .ok_or("DflashSpeculator: no completed window available for terminal repair")?;
        if mark.position != window_start || mark.seed != window_seed {
            return Err(format!(
                "DflashSpeculator: terminal repair window mismatch (saved pos={} seed={}, requested pos={} seed={})",
                mark.position, mark.seed, window_start, window_seed
            ));
        }

        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;
        let mut parts = DflashTargetParts::from_slot(slot);
        let policy = DflashCheckpointPolicy {
            resume_enabled: self.resume_enabled,
            interval: self.ck_interval,
            cap: self.ck_cap,
        };
        dflash_repair_terminal_prefix_parts(
            gpu,
            &mut parts,
            self.df
                .as_mut()
                .ok_or_else(|| SINGLETON_VACANT.to_string())?,
            &mut self.checkpoints,
            &policy,
            mark.target_hidden,
            window_start,
            window_seed,
            consumed,
        )?;
        Ok(true)
    }

    fn reset_state_evidence(&self) -> Option<hipfire_runtime::spec::SpecResetEvidence> {
        // Vacant: no drafter state exists, i.e. it is trivially reset.
        let drafter_reset = match self.df.as_ref() {
            Some(df) => {
                let th = &df.draft_scratch.thlog;
                th.uploaded_rows() == 0 && th.proj_cached_rows() == 0 && th.full_cached_rows() == 0
            }
            None => true,
        };
        Some(hipfire_runtime::spec::SpecResetEvidence {
            drafter_reset,
            checkpoint_empty: self.checkpoints.is_empty(),
        })
    }

    fn block_size(&self) -> usize {
        // Effective proposal width (adaptive shrink); full when opted out or
        // seeding. The generate loop's capacity checks + overflow guard read
        // this per cycle, and `step` truncates the draft block to match.
        self.adaptive.effective()
    }

    fn ctx_capacity(&self) -> usize {
        match (&self.df, &self.vacant) {
            (Some(df), _) => df.ctx_capacity,
            (None, Some(assets)) => assets.ctx_capacity,
            (None, None) => 0,
        }
    }

    fn checkpoint_positions(&self) -> Vec<usize> {
        self.checkpoints.iter().map(|(p, _)| *p).collect()
    }

    fn rewind_to(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
    ) -> Result<usize, String> {
        if let Some(lk) = self.lookup.as_mut() {
            // Divergent re-render: the request restarts from `prefill`; any
            // surviving history is rebuilt from the prompt.
            lk.mark_dirty();
        }
        // Restore the target's DeltaNet recurrent state to the checkpoint at
        // `position` and drop the now-stale tail of the ring (mirrors the old
        // divergent-render resume at generate_dflash 4021-4036). Caller rewinds
        // seq_pos / conversation_tokens to match.
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;
        if let Some(idx) = self.checkpoints.iter().rposition(|(p, _)| *p == position) {
            self.checkpoints[idx]
                .1
                .restore_to(&mut slot.dn_state, gpu)
                .map_err(|e| format!("DeltaNetSnapshot::restore_to: {e}"))?;
            for (_, snap) in self.checkpoints.drain(idx + 1..) {
                snap.free_gpu(gpu);
            }
        }
        Ok(position)
    }

    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        // Store the request's sampling config for the chain-mode branch of
        // `step`, and re-seed the sampled-draw RNG from the request seed via
        // the shared helper (0 maps to the 0x13579BDF sentinel; every other
        // seed passes through verbatim so an explicit wire `seed` reproduces
        // its draw sequence exactly).
        self.sample_temp = cfg.temp;
        self.sample_top_p = cfg.top_p;
        self.sample_top_k = cfg.ar_candidate_cap();
        self.sample_cactus = cfg.cactus_delta;
        self.rng_state = request_rng_state(cfg.rng_seed);
        self.last_window = None;
        // Per-request reseed of the adaptive window: a new request opens at
        // the full block; the controller re-shrinks only after a fresh
        // 8-cycle window (no cross-request stale-history shrink).
        self.adaptive.reset();
        if let Some(t) = self
            .df
            .as_mut()
            .and_then(|df| df.draft_scratch.online.as_mut())
        {
            t.reset();
        }
    }

    fn requires_greedy(&self) -> bool {
        // DFlash supports faithful temp>0 decode via lossless rejection sampling
        // (set_sampling + the sampled `spec_step_dflash` path), so it does NOT
        // require greedy verification. The daemon dispatch consults this (via
        // `spec_can_sample`) to decide whether a temp>0 request may take the spec
        // path or must fall to AR — returning `false` here is what lets sampled
        // DFlash engage while greedy-only drafters (MTP/n-gram) stay on AR.
        false
    }

    fn quiesce(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        match self.df.as_mut() {
            Some(df) => df.verify_pm4.shutdown(),
            None => Ok(()),
        }
    }

    fn verify_pm4_report(&self) -> Option<serde_json::Value> {
        Some(match self.df.as_ref() {
            Some(df) => df.verify_pm4.report_json(),
            None => DflashVerifyPm4::disabled("singleton lane state vacant").report_json(),
        })
    }

    fn drafter_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        let DflashSpeculator {
            df,
            vacant,
            checkpoints,
            railgun,
            ..
        } = *self;
        // Prepared windows and the builder's staging/selector buffers first:
        // nothing is in flight between windows.
        if let Some(owner) = railgun {
            owner.release(gpu);
        }
        if let Some(df) = df {
            df.free_gpu(gpu);
        }
        if let Some(assets) = vacant {
            release_shared_dflash_weights(gpu, assets.weights);
        }
        for (_, snap) in checkpoints {
            snap.free_gpu(gpu);
        }
    }
}

/// Construct the DFlash speculator from a freshly-loaded `DflashState`, resolving
/// the env config the daemon's old `generate_dflash` read inline: checkpoint
/// resume (`HIPFIRE_DFLASH_CKPT_RESUME` + no-eviction) and interval/cap
/// (`HIPFIRE_CACHE_CKPT_INTERVAL`/`_MAX`, matching the daemon's
/// `ckpt_interval()`/`ckpt_max()` defaults), plus the adaptive-B kill-switch
/// (`HIPFIRE_DFLASH_ADAPTIVE_B=0` forces the fixed full block, mirroring
/// DSpark's `HIPFIRE_DSPARK_ADAPTIVE_BLOCK=0`). Called once at load.
/// `adaptive_b` is the `SpecLoadCfg::dflash_adaptive_b` load param (None =
/// default off).
pub fn build_dflash_speculator(
    df: DflashState,
    eviction_is_none: bool,
    adaptive_b: bool,
) -> Box<dyn Speculator> {
    let resume_enabled = hipfire_config::developer_var("HIPFIRE_DFLASH_CKPT_RESUME")
        .ok()
        .as_deref()
        != Some("0")
        && eviction_is_none;
    let ck_interval = hipfire_config::developer_var("HIPFIRE_CACHE_CKPT_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048usize)
        .max(256);
    let ck_cap = hipfire_config::developer_var("HIPFIRE_CACHE_CKPT_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize)
        .max(1);
    // Opt-in (load param, default off); HIPFIRE_DFLASH_ADAPTIVE_B=0 still
    // forces the fixed block when the param is on.
    // The load param ANDs into this here so the constructor stays the single
    // env-touching site and `new` stays pure (mirrors
    // build_dspark_speculator).
    let adaptive_b = adaptive_b
        && hipfire_config::developer_var("HIPFIRE_DFLASH_ADAPTIVE_B")
            .ok()
            .as_deref()
            != Some("0");
    // Retained-PM4 verify replay is shape-frozen at B=16: an adaptive
    // controller shrinks windows below DFLASH_VERIFY_PM4_BLOCK, the PM4
    // window's `eligible_shape()` then refuses (`batch == B` only) and every
    // verify silently falls back to partial_hip — PM4 opt-in users would
    // silently lose replay. Fixed block wins for the whole load; adaptive
    // and PM4 replay are mutually exclusive opt-ins by construction.
    let pm4_replay = df.verify_pm4.admitted();
    if adaptive_b && pm4_replay {
        eprintln!(
            "[dflash] adaptive_b: fixed block (retained PM4 verify replay is shape-frozen at B={DFLASH_VERIFY_PM4_BLOCK})"
        );
    }
    let adaptive_b = adaptive_b && !pm4_replay;
    let mut speculator = DflashSpeculator::new(df, resume_enabled, ck_interval, ck_cap, adaptive_b);
    speculator.configure_lookup(
        hipfire_config::dflash_lookup_enabled(),
        hipfire_config::dflash_lookup_tail_enabled(),
    );
    if crate::dflash_railgun::window_var_enabled() {
        speculator.enable_railgun_window();
    }
    Box::new(speculator)
}

impl DflashSpeculator {
    /// Arm the certified railgun window (design A). Called once at build when
    /// `HIPFIRE_DFLASH_RAILGUN_WINDOW=1`; windows the owner does not admit run
    /// the ordinary HIP route with a counted, log-once refusal.
    pub fn enable_railgun_window(&mut self) {
        if self.railgun.is_none() {
            eprintln!(
                "[dflash] railgun window: enabled ({}=1; exact gfx1201 dense 27B, FP8 KV, fixed-16 greedy chain only)",
                crate::dflash_railgun::RAILGUN_WINDOW_VAR
            );
            let mut owner = crate::dflash_railgun::RailgunWindowOwner::new();
            owner.install_builder(Box::new(
                crate::dflash_railgun_builder::RailgunWindowBuilder::from_env(),
            ));
            self.railgun = Some(owner);
        }
    }

    /// The window owner (counters, quarantine, builder install); `None` when
    /// the variable was unset at build.
    pub fn railgun_window_mut(&mut self) -> Option<&mut crate::dflash_railgun::RailgunWindowOwner> {
        self.railgun.as_mut()
    }
}

// ─── Shared-asset VMM lanes (continuous batching) ───────────────────────

/// Release one reference to the shared draft weights. Only the final owner
/// frees the raw GPU weights (`Arc::into_inner` guarantees exactly one caller
/// observes the last reference even under concurrent releases); every other
/// call just drops its reference. There is deliberately no `Drop` impl: GPU
/// frees need a bound `Gpu`.
pub fn release_shared_dflash_weights(gpu: &mut Gpu, weights: Arc<DflashWeights>) {
    if let Some(w) = Arc::into_inner(weights) {
        w.free_gpu(gpu);
    }
}

/// Request-local DeltaNet checkpoint cadence (`DflashSpeculator`'s resolved
/// `resume_enabled` / `ck_interval` / `ck_cap`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflashCheckpointPolicy {
    pub resume_enabled: bool,
    pub interval: usize,
    pub cap: usize,
}

/// Read-only descriptor of the loaded DFlash draft a VMM engine shares:
/// cloned config metadata, a clone of the SAME weights `Arc`, and the
/// singleton's resolved context policy / sizing. Never reparses env.
///
/// NOTE: `weights` is a counted reference. Drop descriptors (or hand them to
/// [`release_shared_dflash_weights`]) before expecting the final owner's free.
#[derive(Clone)]
pub struct DflashVmmAssets {
    pub draft_config: DflashConfig,
    pub weights: Arc<DflashWeights>,
    /// Resolved draft context mode (Legacy / Windowed{w, w_full}).
    pub ctx_mode: DraftCtxMode,
    /// `draft_scratch.max_ctx_len` (logical target context for windowed mode).
    pub draft_max_ctx: usize,
    /// Logical context cap the singleton reports (`DflashState::ctx_capacity`).
    pub ctx_capacity: usize,
    /// Hidden-ring rows (`HiddenStateRingBuffer::max_positions`); also the
    /// host hidden log row capacity.
    pub ring_positions: usize,
    /// Hidden-ring staging rows (`HiddenStateRingBuffer::max_batch`).
    pub staging_rows: usize,
    /// `VerifyScratch::max_n` (block_size + 1 for fixed chain).
    pub verify_max_rows: usize,
    /// `GdnTape::max_n`.
    pub tape_max_rows: usize,
    pub block_size: usize,
    pub checkpoint: DflashCheckpointPolicy,
    /// Proof of fixed-block chain eligibility: no DDTree, no admitted
    /// retained-PM4 route, adaptive block controller disabled.
    pub fixed_chain: bool,
}

impl DflashVmmAssets {
    /// Same shared weights (pointer identity) and same resolved policy as
    /// `df`, i.e. `df` was built from these assets (or the singleton they
    /// describe).
    pub fn matches_state(&self, df: &DflashState) -> bool {
        Arc::ptr_eq(&self.weights, &df.draft_weights)
            && self.ctx_mode == df.draft_scratch.ctx_mode
            && self.draft_max_ctx == df.draft_scratch.max_ctx_len
            && self.ctx_capacity == df.ctx_capacity
            && self.block_size == df.block_size
            && self.ring_positions == df.hidden_rb.max_positions
            && self.staging_rows == df.hidden_rb.max_batch
            && self.verify_max_rows == df.verify_scratch.max_n
            && self.tape_max_rows == df.gdn_tape.max_n
    }
}

/// Allocate a clean DFlash `DflashState` against the shared weights and
/// resolved policy of `assets` (no weight load). Fixed-chain lanes have
/// `ddtree = None` and an explicitly disabled retained verify route. `kind`
/// selects the scratch footprint ([`DflashOwnerKind`]): `VmmLane` omits the
/// draft FWHT rotation buffers and the verify `prefill_batch`; `Singleton` is
/// the full owner set (the speculator's lazily rebuilt state).
pub fn new_dflash_lane_state(
    gpu: &mut Gpu,
    assets: &DflashVmmAssets,
    config: &Qwen35Config,
    dn: &DeltaNetState,
    kind: DflashOwnerKind,
) -> Result<DflashState, String> {
    if !assets.fixed_chain {
        return Err("DFlash VMM lane requires fixed-block chain assets".into());
    }
    let owners = alloc_dflash_owners(
        gpu,
        &assets.draft_config,
        assets.weights.has_mq,
        assets.block_size,
        assets.ctx_mode,
        assets.draft_max_ctx,
        assets.ring_positions,
        assets.staging_rows,
        assets.verify_max_rows,
        assets.tape_max_rows,
        config,
        dn,
        kind,
    )?;
    let DflashOwners {
        draft_scratch,
        hidden_rb,
        verify_scratch,
        target_snap,
        gdn_tape,
    } = owners;
    Ok(DflashState {
        draft_config: assets.draft_config.clone(),
        draft_weights: Arc::clone(&assets.weights),
        draft_scratch,
        hidden_rb,
        verify_scratch,
        target_snap,
        gdn_tape,
        ctx_capacity: assets.ctx_capacity,
        block_size: assets.block_size,
        ddtree: None,
        verify_pm4: DflashVerifyPm4::disabled("VMM continuous-batching lane is eager-only"),
    })
}

/// Lossless owned transfer of a live singleton DFlash lane (mutable owners,
/// committed position, request cursors). Fields are crate-visible so the VMM
/// lane wrapper can destructure it; consume it with `free_gpu` or by moving
/// the fields. Asset identity is carried by `df.draft_weights` (same `Arc`)
/// plus the policy in `df` — see [`DflashVmmAssets::matches_state`].
pub struct DflashLaneSnapshot {
    pub(crate) df: DflashState,
    /// Committed target position at capture.
    pub(crate) position: usize,
    pub(crate) rng_state: u64,
    pub(crate) sample_temp: f32,
    pub(crate) sample_top_p: f32,
    pub(crate) sample_top_k: usize,
    pub(crate) sample_cactus: f32,
    pub(crate) checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub(crate) policy: DflashCheckpointPolicy,
    pub(crate) last_window: Option<DflashWindowMark>,
}

impl DflashLaneSnapshot {
    /// Committed rows (target position) the snapshot was taken at.
    pub fn rows(&self) -> usize {
        self.position
    }

    /// Free every owned GPU object exactly once (the shared weights are
    /// released through [`release_shared_dflash_weights`]).
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let DflashLaneSnapshot {
            df,
            position: _,
            rng_state: _,
            sample_temp: _,
            sample_top_p: _,
            sample_top_k: _,
            sample_cactus: _,
            checkpoints,
            policy: _,
            last_window: _,
        } = self;
        df.free_gpu(gpu);
        for (_, snap) in checkpoints {
            snap.free_gpu(gpu);
        }
    }
}

/// Drain pointer-bearing execution and drop pointer-bound singleton verify
/// graphs before lane owners change hands.
fn quiesce_for_lane_move(gpu: &mut Gpu) -> Result<(), String> {
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("dflash lane move quiesce: {e}"))?;
    gpu.graphs.verify_graph_destroy_all(&gpu.hip, gpu.device_id);
    Ok(())
}

impl DflashSpeculator {
    /// Shared-asset descriptor for a VMM engine. `Err(reason)` when the loaded
    /// speculator is not a fixed-block chain DFlash (DDTree, admitted retained
    /// PM4, or adaptive block controller enabled). While the lane state is
    /// vacant the descriptor it was taken with is returned (a clone of the
    /// same weights `Arc`).
    pub fn vmm_assets(&self) -> Result<DflashVmmAssets, String> {
        let Some(df) = self.df.as_ref() else {
            return self
                .vacant
                .clone()
                .ok_or_else(|| SINGLETON_VACANT.to_string());
        };
        if df.ddtree.is_some() {
            return Err("DFlash VMM lanes require chain mode (DDTree is loaded)".into());
        }
        if df.verify_pm4.admitted() {
            return Err("DFlash VMM lanes refuse an admitted retained-PM4 verify route".into());
        }
        if self.adaptive_enabled {
            return Err("DFlash VMM lanes require the fixed block (adaptive block enabled)".into());
        }
        let ds = &df.draft_scratch;
        Ok(DflashVmmAssets {
            draft_config: df.draft_config.clone(),
            weights: Arc::clone(&df.draft_weights),
            ctx_mode: ds.ctx_mode,
            draft_max_ctx: ds.max_ctx_len,
            ctx_capacity: df.ctx_capacity,
            ring_positions: df.hidden_rb.max_positions,
            staging_rows: df.hidden_rb.max_batch,
            verify_max_rows: df.verify_scratch.max_n,
            tape_max_rows: df.gdn_tape.max_n,
            block_size: df.block_size,
            checkpoint: DflashCheckpointPolicy {
                resume_enabled: self.resume_enabled,
                interval: self.ck_interval,
                cap: self.ck_cap,
            },
            fixed_chain: true,
        })
    }

    /// Move the live lane (draft K/V + hidden ring + verify/tape/snapshot
    /// owners, host log, checkpoints, request cursors) into a snapshot. The
    /// speculator is left VACANT, holding only the shared-asset descriptor:
    /// no replacement state is allocated (a promoted lane lives for the whole
    /// batch; the singleton rebuilds a fresh state on its next `prefill`,
    /// [`Self::ensure_live`], or takes the lane back via
    /// [`Self::restore_vmm_lane`]). A refused request (DDTree / PM4 /
    /// adaptive, or already vacant) and a failed quiesce leave the speculator
    /// untouched. No re-prefill, no projection-watermark reset.
    /// Pointer-bound singleton verify graphs are destroyed.
    pub fn take_vmm_lane(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
    ) -> Result<DflashLaneSnapshot, String> {
        if self.df.is_none() {
            return Err("DFlash singleton lane state is already vacant (moved out)".into());
        }
        let assets = self.vmm_assets()?;
        quiesce_for_lane_move(gpu)?;
        let df = self.df.take().ok_or_else(|| SINGLETON_VACANT.to_string())?;
        self.vacant = Some(assets);
        Ok(DflashLaneSnapshot {
            df,
            position,
            rng_state: self.rng_state,
            sample_temp: self.sample_temp,
            sample_top_p: self.sample_top_p,
            sample_top_k: self.sample_top_k,
            sample_cactus: self.sample_cactus,
            checkpoints: std::mem::take(&mut self.checkpoints),
            policy: DflashCheckpointPolicy {
                resume_enabled: self.resume_enabled,
                interval: self.ck_interval,
                cap: self.ck_cap,
            },
            last_window: self.last_window.take(),
        })
    }

    /// Undo [`Self::take_vmm_lane`]: move the snapshot's owners back. A
    /// vacant speculator reinstalls them and releases the vacancy's asset
    /// reference; a live one frees the state it holds (today's behaviour). A
    /// snapshot that does not match this speculator's shared assets is freed
    /// and refused (never installed).
    pub fn restore_vmm_lane(
        &mut self,
        gpu: &mut Gpu,
        snap: DflashLaneSnapshot,
    ) -> Result<(), String> {
        let matches = match self.vacant.as_ref() {
            Some(assets) => Ok(assets.matches_state(&snap.df)),
            None => self
                .vmm_assets()
                .map(|assets| assets.matches_state(&snap.df)),
        };
        match matches {
            Ok(true) => {}
            Ok(false) => {
                snap.free_gpu(gpu);
                return Err("DFlash lane snapshot does not match this speculator's assets".into());
            }
            Err(e) => {
                snap.free_gpu(gpu);
                return Err(e);
            }
        }
        if let Err(e) = quiesce_for_lane_move(gpu) {
            snap.free_gpu(gpu);
            return Err(e);
        }
        let DflashLaneSnapshot {
            df,
            position: _,
            rng_state,
            sample_temp,
            sample_top_p,
            sample_top_k,
            sample_cactus,
            checkpoints,
            policy: _,
            last_window,
        } = snap;
        if let Some(clean) = self.df.replace(df) {
            clean.free_gpu(gpu);
        }
        if let Some(assets) = self.vacant.take() {
            release_shared_dflash_weights(gpu, assets.weights);
        }
        for (_, old) in self.checkpoints.drain(..) {
            old.free_gpu(gpu);
        }
        self.checkpoints = checkpoints;
        self.rng_state = rng_state;
        self.sample_temp = sample_temp;
        self.sample_top_p = sample_top_p;
        self.sample_top_k = sample_top_k;
        self.sample_cactus = sample_cactus;
        self.last_window = last_window;
        Ok(())
    }

    /// Rebuild the singleton's lane state if a promotion left it vacant: a
    /// fresh full owner set ([`DflashOwnerKind::Singleton`]) sized from `dn`
    /// (the resident target DeltaNet state), with the vacancy's asset
    /// reference released once it is installed. No-op when live. On failure
    /// the speculator stays vacant (nothing leaked) and the error is
    /// returned. `prefill` calls this first; call it directly to rebuild
    /// without prefilling.
    pub fn ensure_live(
        &mut self,
        gpu: &mut Gpu,
        config: &Qwen35Config,
        dn: &DeltaNetState,
    ) -> Result<(), String> {
        let Some(assets) = self.vacant.as_ref() else {
            return Ok(());
        };
        let df = new_dflash_lane_state(gpu, assets, config, dn, DflashOwnerKind::Singleton)?;
        self.df = Some(df);
        if let Some(assets) = self.vacant.take() {
            release_shared_dflash_weights(gpu, assets.weights);
        }
        Ok(())
    }
}

/// Cold DFlash prompt prefill over explicit trunk owners (a VMM lane): the
/// singleton's seed → scatter → backfill → thlog seed → first-logits sequence
/// with the same ring-piece scheduling. The returned first token is
/// `dflash_first_token` (the singleton's prefill first-token policy, chain
/// mode) over the last prompt logits (`target.scratch.logits`): host argmax at
/// `cfg.temp <= 1e-6`, else host softmax + top-k/nucleus and exactly one
/// xorshift draw advancing `rng` (the lane's `rng_state`). `rng` MUST be
/// initialised from the request seed before this call and the advanced value
/// kept; `scratch` is reusable probability scratch. `checkpoints` is only used
/// when `policy.resume_enabled`; `policy` is the CALLER's checkpoint cadence (a
/// lane's, which may take none while nothing reads its checkpoints), not
/// necessarily `assets.checkpoint`. Request cursors other than `checkpoints`
/// (last-window mark, adaptive state) are the caller's.
#[allow(clippy::too_many_arguments)]
pub fn dflash_prefill_lane_parts(
    gpu: &mut Gpu,
    target: &mut DflashTargetParts<'_>,
    assets: &DflashVmmAssets,
    policy: DflashCheckpointPolicy,
    df: &mut DflashState,
    checkpoints: &mut Vec<(usize, DeltaNetSnapshot)>,
    prompt: &[u32],
    cfg: crate::speculative::DflashSampleCfg,
    rng: &mut u64,
    scratch: &mut Vec<f32>,
    abort: &dyn Fn() -> bool,
) -> Result<PrefillOutcome, String> {
    if !assets.matches_state(df) {
        return Err("DFlash lane state does not match the shared assets".into());
    }
    let mut host_hidden = Vec::new();
    let capture_host = dflash_prefill_host_required(df.draft_scratch.ctx_mode, prompt.len());
    df.draft_scratch.reset_upload_tracking();
    let ckpt_sink = if policy.resume_enabled {
        Some(checkpoints)
    } else {
        None
    };
    let aborted = seed_target_hidden_from_prompt_abortable_parts(
        gpu,
        target,
        &mut df.hidden_rb,
        capture_host.then_some(&mut host_hidden),
        prompt,
        abort,
        ckpt_sink,
        policy.interval,
        policy.cap,
    )
    .map_err(|e| {
        hipfire_runtime::reset_core::note_hip_error(&e, "qwen35::dflash_prefill::seed");
        e.to_string()
    })?;
    if aborted {
        return Ok(PrefillOutcome::Aborted);
    }
    let first_logits = dflash_prefill_finish(
        gpu,
        df,
        &host_hidden,
        &target.scratch.logits,
        prompt,
        prompt.len(),
        false,
        0,
        None,
    )?;
    // Shared with the singleton prefill (lines 890-913 originally):
    // `chain = true`, the lane runs the selector chain route.
    let first_token = dflash_first_token(&first_logits, true, cfg, rng, scratch);
    Ok(PrefillOutcome::Ready { first_token })
}

/// Terminal-prefix repair over explicit trunk owners: restore the pre-window
/// target snapshot and thlog mark, then re-seed the consumed prefix
/// (`terminal_prefix_replay(seed, consumed)`) through the singleton suffix
/// forward and scatter/append its hidden rows. `position`/`seed` are the
/// window's start and seed; `thlog_mark` is the mark captured before the
/// window's draft. Window identity validation is the caller's.
#[allow(clippy::too_many_arguments)]
pub fn dflash_repair_terminal_prefix_parts(
    gpu: &mut Gpu,
    target: &mut DflashTargetParts<'_>,
    df: &mut DflashState,
    checkpoints: &mut Vec<(usize, DeltaNetSnapshot)>,
    policy: &DflashCheckpointPolicy,
    thlog_mark: TargetHiddenLogMark,
    position: usize,
    seed: u32,
    consumed: &[u32],
) -> Result<(), String> {
    df.target_snap
        .restore_to(target.dn_state, gpu)
        .map_err(|e| format!("DeltaNetSnapshot::restore_to: {e}"))?;
    df.draft_scratch.thlog.restore(thlog_mark)?;

    // Before the ordinary terminal flush, target state must include the
    // old pending seed and every consumed token except the new pending
    // terminal token. For consumed=[] there is nothing to replay.
    let replay = terminal_prefix_replay(seed, consumed);
    if replay.is_empty() {
        return Ok(());
    }

    let aborted = seed_target_hidden_suffix_abortable_parts(
        gpu,
        target,
        &mut df.hidden_rb,
        &replay,
        position,
        &|| false,
        policy.resume_enabled.then_some(checkpoints),
        policy.interval,
        policy.cap,
    )
    .map_err(|e| e.to_string())?;
    debug_assert!(!aborted, "terminal repair uses a non-aborting callback");
    scatter_hidden_block_to_interleaved(
        gpu,
        &df.hidden_rb,
        &df.draft_scratch.target_hidden,
        position,
        replay.len(),
        replay.len(),
        df.draft_scratch.ctx_modulus(),
    )
    .map_err(|e| e.to_string())?;
    let co = target.kv_cache.compact_offset as i32;
    df.draft_scratch
        .thlog
        .append_committed(position, replay.len(), co);
    Ok(())
}

// ─── Retained-PM4 admission (pure) ────────────────────────────────────

/// Pure admission predicate for the fixed B=16 DFlash retained-PM4 verify route.
///
/// The gates constrain what the tape actually captures — the *target* forward's
/// arch, shape, quantization, and scratch capacities. They deliberately say
/// nothing about the draft's selector or dynamic-conv fields: the draft forward,
/// candidate selection, and draft lm-head are all outside the tape boundary, so
/// DFlash2 and legacy DFlash produce an identical target verify body.
///
/// Each failed condition returns a **distinct** reason string so harness
/// evidence can name the gate. GPU-free: unit-tested without a device.
pub fn admit_dflash_verify_pm4(
    env_opt_in: bool,
    arch: &str,
    gfx1100_split_verify: bool,
    single_gpu: bool,
    num_experts: usize,
    kv_is_q8: bool,
    dn_state_is_q8: bool,
    adaptive_kv_absent: bool,
    runtime_extract_layers: &[usize],
    runtime_block_size: usize,
    target_layer_ids: &[usize],
    ddtree_present: bool,
    prefill_batch_present: bool,
    pbs_eligible_b16: bool,
    verify_max_n: usize,
    pbs_max_batch: usize,
    hidden_rb_max_batch: usize,
    gdn_tape_max_n: usize,
) -> Result<(), String> {
    if !env_opt_in {
        return Err("HIPFIRE_DFLASH_VERIFY_PM4 is not set to 1".into());
    }
    if arch != "gfx1201" && !(arch == "gfx1100" && gfx1100_split_verify) {
        return Err(format!(
            "arch is {arch}, not exact gfx1201 or admitted gfx1100 split verifier"
        ));
    }
    if !single_gpu {
        return Err("multi-GPU load is not admitted".into());
    }
    if num_experts != 0 {
        return Err(format!(
            "MoE target (num_experts={num_experts}) is not admitted"
        ));
    }
    if !kv_is_q8 {
        return Err("KV mode is not Q8".into());
    }
    if !dn_state_is_q8 {
        return Err("DeltaNet state quant is not Q8".into());
    }
    if !adaptive_kv_absent {
        return Err("adaptive KV is engaged".into());
    }
    // Consistency, not identity: whatever layers the draft declares must be the
    // layers the hidden ring actually extracts, or the captured staging writes
    // do not correspond to what the draft will read back.
    if runtime_extract_layers != target_layer_ids {
        return Err(format!(
            "hidden-ring extract layers {runtime_extract_layers:?} != draft target_layer_ids {target_layer_ids:?}"
        ));
    }
    if target_layer_ids.is_empty() {
        return Err("draft declares no target_layer_ids".into());
    }
    if runtime_block_size != DFLASH_VERIFY_PM4_BLOCK {
        return Err(format!(
            "runtime block size is {runtime_block_size}, not {DFLASH_VERIFY_PM4_BLOCK}"
        ));
    }
    if ddtree_present {
        return Err("DDTree is present".into());
    }
    if !prefill_batch_present {
        return Err("verify_scratch.prefill_batch is absent".into());
    }
    if !pbs_eligible_b16 {
        return Err("prefill_batch_pbs_eligible failed at B=16".into());
    }
    if verify_max_n < DFLASH_VERIFY_PM4_BLOCK {
        return Err(format!(
            "verify_scratch.max_n={verify_max_n} < {DFLASH_VERIFY_PM4_BLOCK}"
        ));
    }
    if pbs_max_batch < DFLASH_VERIFY_PM4_BLOCK {
        return Err(format!(
            "prefill_batch.max_batch={pbs_max_batch} < {DFLASH_VERIFY_PM4_BLOCK}"
        ));
    }
    if hidden_rb_max_batch < DFLASH_VERIFY_PM4_BLOCK {
        return Err(format!(
            "hidden_rb.max_batch={hidden_rb_max_batch} < {DFLASH_VERIFY_PM4_BLOCK}"
        ));
    }
    if gdn_tape_max_n < DFLASH_VERIFY_PM4_BLOCK {
        return Err(format!(
            "gdn_tape.max_n={gdn_tape_max_n} < {DFLASH_VERIFY_PM4_BLOCK}"
        ));
    }
    Ok(())
}

// ─── Dense-TP (TP=2) DFlash2 mesh owner ───────────────────────────────
//
// Transactional owner of one full DFlash2 draft replica per target rank plus
// each rank's persistent `[B,H]` TP partial. No draft collective exists: the
// target's post-reduction hidden is already full-width and bit-identical on
// both ranks, so each replica consumes local data and the mesh gate requires
// byte-identical drafted IDs before the target may mutate. Eager-only and
// greedy-only; retained PM4 and HipGraph verify stay disarmed on this path.

/// One rank's draft replica + mesh bookkeeping. `dflash` mirrors the
/// single-GPU `DflashState` layout (chain-only: `ddtree` is always `None`,
/// `verify_pm4` always disabled); `tp_partial` is the persistent `[B,H]`
/// reduction staging for that rank's device.
pub struct DenseTpDflashRankState {
    pub dflash: DflashState,
    pub tp_partial: GpuTensor,
    pub logical_pos: usize,
    pub receipt_generation: u64,
}

/// Two draft replicas (rank order) + the next window generation tag.
pub struct DenseTpDflashSpeculator {
    pub ranks: Vec<DenseTpDflashRankState>,
    pub next_generation: u64,
}

/// Transactionally build both draft replicas. Every rank binds its own device
/// and runs the single-rank load logic under a `DenseTp2EagerGreedy` posture:
/// fixed B=16, DFlash2 all-sliding selector draft, no DDTree/PFlash/CASK/
/// adaptive-KV state, PM4/graph never armed. A later rank's failure frees all
/// earlier ranks' draft + partial state; borrowed target state is untouched.
/// Returns the mesh owner; target rank configs must already exist.
pub fn load_dflash_speculator_dense_tp2(
    draft_path: &str,
    gpus: &mut Gpus,
    target: DenseTpTargetView<'_>,
    ctx_capacity: usize,
    block_size: usize,
) -> Result<DenseTpDflashSpeculator, String> {
    if gpus.devices.len() != 2 || target.configs.len() != 2 || target.dn_states.len() != 2 {
        return Err("dense TP DFlash needs exactly 2 target ranks".into());
    }
    if block_size != 16 {
        return Err(format!(
            "dense TP DFlash requires fixed B=16 (requested block_size={block_size})"
        ));
    }
    for cfg in target.configs.iter() {
        if cfg.num_experts != 0 {
            return Err("dense TP DFlash requires num_experts == 0 (dense target)".into());
        }
    }
    if target.configs[0].dim != target.configs[1].dim
        || target.configs[0].n_layers != target.configs[1].n_layers
    {
        return Err("dense TP DFlash rank configs diverge".into());
    }
    let dim = target.configs[0].dim;
    let vocab = target.configs[0].vocab_size;
    let hidden_k = dim.next_power_of_two();
    let draft_hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("draft open: {e}"))?;
    let draft_config = DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "draft: failed to parse DflashConfig from HFQ metadata".to_string())?;
    if draft_config.target_streams_per_layer != 1 {
        return Err(format!(
            "DFlash draft expects {} target streams per layer; this target captures one",
            draft_config.target_streams_per_layer
        ));
    }
    if !draft_config.all_layers_sliding {
        return Err("dense TP DFlash requires a DFlash2 all-sliding draft".into());
    }
    if draft_config.runtime_block_size(block_size) != 16 {
        return Err(format!(
            "dense TP DFlash requires runtime B=16 (draft runtime {})",
            draft_config.runtime_block_size(block_size)
        ));
    }
    // Window posture mirrors `load_dflash_state` (declared window by default,
    // explicit override or Legacy opt-out via HIPFIRE_DFLASH_WINDOW). Mesh
    // loads never enable eviction, so no ring-aware refusal applies here.
    warn_deprecated_legacy_dflash_knobs();
    let window = match hipfire_config::developer_var("HIPFIRE_DFLASH_WINDOW")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(0) => None,
        Some(w) => {
            if let Some(declared) = draft_config.declared_window {
                if declared != w {
                    eprintln!(
                        "  DFlash-TP2 window override {w} != draft-declared sliding_window \
                         {declared} — acceptance may degrade (output stays verify-exact)"
                    );
                }
            }
            Some(w)
        }
        None => draft_config.declared_window,
    };
    let requested_ctx = ctx_capacity;
    let ctx_capacity = match window {
        Some(_) => requested_ctx,
        None => match hipfire_config::developer_var("HIPFIRE_DFLASH_CTX_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            Some(0) => ctx_capacity,
            Some(cap) => ctx_capacity.min(cap),
            None => ctx_capacity.min(DEFAULT_DFLASH_CTX_CAP),
        },
    };
    let mut ranks: Vec<DenseTpDflashRankState> = Vec::with_capacity(2);
    // Free helper for the transactional unwind: every rank state fully built
    // so far is freed on its own device; the borrowed target is untouched.
    let unwind = |gpus: &mut Gpus, ranks: Vec<DenseTpDflashRankState>| {
        for (rank, s) in ranks.into_iter().enumerate() {
            let _ = gpus.devices[rank].bind_thread();
            let DenseTpDflashRankState {
                dflash,
                tp_partial,
                logical_pos: _,
                receipt_generation: _,
            } = s;
            dflash.free_gpu(&mut gpus.devices[rank]);
            let _ = gpus.devices[rank].free_tensor(tp_partial);
        }
    };
    for rank in 0..2 {
        let gpu = &mut gpus.devices[rank];
        if let Err(e) = gpu.bind_thread() {
            let e = format!("dense TP DFlash rank {rank} bind: {e}");
            unwind(gpus, ranks);
            return Err(e);
        }
        macro_rules! or_unwind {
            ($e:expr, $ctx:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => {
                        let ctx: &str = $ctx;
                        let e = if ctx.is_empty() {
                            format!("{e}")
                        } else {
                            format!("{ctx}: {e}")
                        };
                        unwind(gpus, ranks);
                        return Err(format!("dense TP DFlash rank {rank}: {e}"));
                    }
                }
            };
        }
        let draft_weights = or_unwind!(DflashWeights::load(gpu, &draft_hfq, &draft_config), "");
        if !draft_weights.has_candidate_selector() {
            draft_weights.free_gpu(gpu);
            unwind(gpus, ranks);
            return Err(format!(
                "dense TP DFlash rank {rank}: draft has no candidate selector (DFlash2 required)"
            ));
        }
        let draft_scratch = or_unwind!(
            match window {
                Some(w) => DflashScratch::new_windowed(
                    gpu,
                    &draft_config,
                    block_size,
                    w,
                    requested_ctx,
                    requested_ctx,
                    draft_weights.has_mq,
                ),
                None => DflashScratch::new_with_mq(
                    gpu,
                    &draft_config,
                    block_size,
                    ctx_capacity,
                    draft_weights.has_mq,
                ),
            },
            ""
        );
        // or_unwind! above consumes partially-owned GPU state on failure the
        // same way load_dflash_state's or_free! does; mirror its discipline
        // for the remaining fallible builds by freeing the two owned values
        // before unwinding earlier ranks.
        macro_rules! or_unwind_owned {
            ($e:expr, $ctx:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => {
                        draft_scratch.free_gpu(gpu);
                        draft_weights.free_gpu(gpu);
                        unwind(gpus, ranks);
                        return Err(format!("dense TP DFlash rank {rank}: {}: {e}", $ctx));
                    }
                }
            };
        }
        for lid in draft_config.target_layer_ids.iter() {
            if *lid >= target.configs[rank].n_layers {
                draft_scratch.free_gpu(gpu);
                draft_weights.free_gpu(gpu);
                unwind(gpus, ranks);
                return Err(format!(
                    "dense TP DFlash rank {rank}: draft target_layer_ids contains {lid} >= num_target_layers {}",
                    target.configs[rank].n_layers
                ));
            }
        }
        let hidden_rb = or_unwind_owned!(
            HiddenStateRingBuffer::new_for_layers(
                gpu,
                &draft_config.target_layer_ids,
                dim,
                ctx_capacity + block_size,
                block_size,
            ),
            "HiddenStateRingBuffer::new_for_layers"
        );
        let verify_scratch = or_unwind_owned!(
            VerifyScratch::with_prefill(
                gpu,
                block_size,
                dim,
                vocab,
                hidden_k,
                &target.configs[rank]
            ),
            "VerifyScratch::with_prefill"
        );
        let target_snap = or_unwind_owned!(
            DeltaNetSnapshot::new_for(gpu, &target.dn_states[rank]),
            "DeltaNetSnapshot::new_for"
        );
        let gdn_tape = or_unwind_owned!(
            GdnTape::new_for_config(gpu, &target.configs[rank], block_size),
            "GdnTape::new_for_config"
        );
        let tp_partial = or_unwind_owned!(
            gpu.alloc_tensor(&[block_size * dim], rdna_compute::DType::F32),
            "tp_partial alloc"
        );
        ranks.push(DenseTpDflashRankState {
            dflash: DflashState {
                draft_config: draft_config.clone(),
                draft_weights: Arc::new(draft_weights),
                draft_scratch,
                hidden_rb,
                verify_scratch,
                target_snap,
                gdn_tape,
                ctx_capacity: if window.is_some() {
                    requested_ctx
                } else {
                    ctx_capacity
                },
                block_size,
                ddtree: None,
                verify_pm4: DflashVerifyPm4::disabled("dense TP mesh is eager-only"),
            },
            tp_partial,
            logical_pos: 0,
            receipt_generation: u64::MAX,
        });
    }
    Ok(DenseTpDflashSpeculator {
        ranks,
        next_generation: 0,
    })
}

impl DenseTpDflashSpeculator {
    /// Cold prompt prefill on both ranks; returns the greedy first token from
    /// rank-0 target logits. Never aborts (pass-through `false` guard).
    pub fn prefill(
        &mut self,
        gpus: &mut Gpus,
        target: DenseTpTargetView<'_>,
        prompt: &[u32],
    ) -> Result<u32, String> {
        seed_target_hidden_dense_tp2_abortable(gpus, target, &mut self.ranks, prompt, &|| false)
    }

    /// One traced mesh window: full `SpecStepResult` (seed/drafted/accepted/
    /// bonus/committed) for diagnostics and parity diffing.
    pub fn step_traced(
        &mut self,
        gpus: &mut Gpus,
        target: DenseTpTargetView<'_>,
        position: usize,
        seed: u32,
    ) -> hip_bridge::HipResult<SpecStepResult> {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);
        spec_step_dflash_dense_tp2(gpus, target, &mut self.ranks, position, seed, generation)
    }

    /// Consumer-visible step: the existing `SpecStep` contract (emit =
    /// `committed[1..]`, `next_seed` = bonus). Same lowering as the
    /// single-GPU speculator, without the cross-architecture trait migration.
    pub fn step(
        &mut self,
        gpus: &mut Gpus,
        target: DenseTpTargetView<'_>,
        position: usize,
        seed: u32,
    ) -> Result<SpecStep, String> {
        self.step_traced(gpus, target, position, seed)
            .map(|r| {
                SpecStep::new(
                    r.committed[1..].iter().copied(),
                    r.bonus_token,
                    r.drafted.len(),
                    r.accepted,
                )
            })
            .map_err(|e| e.to_string())
    }

    /// Cold reset between requests: ring/thlog/host/receipt state cleared on
    /// both ranks (target KV/DN ownership stays with the caller). Drains both
    /// rank streams so no window work leaks across the reset.
    pub fn reset_all(&mut self, gpus: &mut Gpus) {
        for (rank, s) in self.ranks.iter_mut().enumerate() {
            if gpus.devices.get(rank).is_none() {
                continue;
            }
            let _ = gpus.devices[rank].bind_thread();
            s.dflash.hidden_rb.reset();
            s.dflash.draft_scratch.reset_upload_tracking();
            s.logical_pos = 0;
            s.receipt_generation = u64::MAX;
        }
        for gpu in gpus.devices.iter() {
            let _ = gpu.bind_thread();
            let _ = gpu.hip.device_synchronize();
        }
    }

    /// Drain both rank streams without mutating mesh state.
    pub fn quiesce_all(&self, gpus: &mut Gpus) -> Result<(), String> {
        for (rank, gpu) in gpus.devices.iter().enumerate() {
            gpu.bind_thread()
                .map_err(|e| format!("dense TP DFlash quiesce rank {rank}: {e}"))?;
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip
                    .stream_synchronize(stream)
                    .map_err(|e| format!("dense TP DFlash quiesce rank {rank}: {e}"))?;
            } else {
                gpu.hip
                    .device_synchronize()
                    .map_err(|e| format!("dense TP DFlash quiesce rank {rank}: {e}"))?;
            }
        }
        Ok(())
    }

    /// Destructured without `..` on purpose (same rule as
    /// `DflashState::free_gpu`): a later GPU-owning field becomes a compile
    /// error here instead of a leak. Frees rank `r` only while device `r` is
    /// bound; target state is borrowed and untouched.
    pub fn free_gpu(self, gpus: &mut Gpus) {
        let DenseTpDflashSpeculator {
            ranks,
            next_generation: _,
        } = self;
        for (rank, s) in ranks.into_iter().enumerate() {
            let _ = gpus.devices[rank].bind_thread();
            let DenseTpDflashRankState {
                dflash,
                tp_partial,
                logical_pos: _,
                receipt_generation: _,
            } = s;
            dflash.free_gpu(&mut gpus.devices[rank]);
            let _ = gpus.devices[rank].free_tensor(tp_partial);
        }
    }
}

#[cfg(test)]
mod admit_dflash_verify_pm4_tests {
    use super::*;

    /// Baseline admitted args; override one field per negative case.
    struct Args {
        env_opt_in: bool,
        arch: &'static str,
        gfx1100_split_verify: bool,
        single_gpu: bool,
        num_experts: usize,
        kv_is_q8: bool,
        dn_state_is_q8: bool,
        adaptive_kv_absent: bool,
        runtime_extract_layers: &'static [usize],
        runtime_block_size: usize,
        target_layer_ids: &'static [usize],
        ddtree_present: bool,
        prefill_batch_present: bool,
        pbs_eligible_b16: bool,
        verify_max_n: usize,
        pbs_max_batch: usize,
        hidden_rb_max_batch: usize,
        gdn_tape_max_n: usize,
    }

    impl Default for Args {
        fn default() -> Self {
            Self {
                env_opt_in: true,
                arch: "gfx1201",
                gfx1100_split_verify: false,
                single_gpu: true,
                num_experts: 0,
                kv_is_q8: true,
                dn_state_is_q8: true,
                adaptive_kv_absent: true,
                runtime_extract_layers: &DFLASH_VERIFY_PM4_EXTRACT_LAYERS,
                runtime_block_size: DFLASH_VERIFY_PM4_BLOCK,
                target_layer_ids: &DFLASH_VERIFY_PM4_EXTRACT_LAYERS,
                ddtree_present: false,
                prefill_batch_present: true,
                pbs_eligible_b16: true,
                verify_max_n: DFLASH_VERIFY_PM4_BLOCK,
                pbs_max_batch: DFLASH_VERIFY_PM4_BLOCK,
                hidden_rb_max_batch: DFLASH_VERIFY_PM4_BLOCK,
                gdn_tape_max_n: DFLASH_VERIFY_PM4_BLOCK,
            }
        }
    }

    fn admit(a: Args) -> Result<(), String> {
        admit_dflash_verify_pm4(
            a.env_opt_in,
            a.arch,
            a.gfx1100_split_verify,
            a.single_gpu,
            a.num_experts,
            a.kv_is_q8,
            a.dn_state_is_q8,
            a.adaptive_kv_absent,
            a.runtime_extract_layers,
            a.runtime_block_size,
            a.target_layer_ids,
            a.ddtree_present,
            a.prefill_batch_present,
            a.pbs_eligible_b16,
            a.verify_max_n,
            a.pbs_max_batch,
            a.hidden_rb_max_batch,
            a.gdn_tape_max_n,
        )
    }

    #[test]
    fn admits_full_conjunction() {
        assert!(admit(Args::default()).is_ok());
    }

    #[test]
    fn rejects_env_off() {
        let err = admit(Args {
            env_opt_in: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "HIPFIRE_DFLASH_VERIFY_PM4 is not set to 1");
    }

    #[test]
    fn rejects_wrong_arch() {
        let err = admit(Args {
            arch: "gfx1100",
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(
            err,
            "arch is gfx1100, not exact gfx1201 or admitted gfx1100 split verifier"
        );
    }

    #[test]
    fn admits_gfx1100_only_with_split_verifier() {
        assert!(admit(Args {
            arch: "gfx1100",
            gfx1100_split_verify: true,
            ..Args::default()
        })
        .is_ok());
    }

    #[test]
    fn rejects_multi_gpu() {
        let err = admit(Args {
            single_gpu: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "multi-GPU load is not admitted");
    }

    #[test]
    fn rejects_moe() {
        let err = admit(Args {
            num_experts: 256,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "MoE target (num_experts=256) is not admitted");
    }

    #[test]
    fn rejects_non_q8_kv() {
        let err = admit(Args {
            kv_is_q8: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "KV mode is not Q8");
    }

    #[test]
    fn rejects_non_q8_dn_state() {
        let err = admit(Args {
            dn_state_is_q8: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "DeltaNet state quant is not Q8");
    }

    #[test]
    fn rejects_adaptive_kv() {
        let err = admit(Args {
            adaptive_kv_absent: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "adaptive KV is engaged");
    }

    #[test]
    fn admits_a_legacy_dflash_draft() {
        // The draft forward is outside the tape, so a draft without the DFlash2
        // selector or dynamic conv still yields an identical target verify body.
        assert!(admit(Args {
            runtime_extract_layers: &[3, 7, 11, 15, 19],
            target_layer_ids: &[3, 7, 11, 15, 19],
            ..Args::default()
        })
        .is_ok());
    }

    #[test]
    fn rejects_empty_extract_layers() {
        let err = admit(Args {
            runtime_extract_layers: &[],
            target_layer_ids: &[],
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "draft declares no target_layer_ids");
    }

    #[test]
    fn rejects_wrong_block_size() {
        let err = admit(Args {
            runtime_block_size: 8,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "runtime block size is 8, not 16");
    }

    #[test]
    fn rejects_extract_layer_disagreement() {
        let err = admit(Args {
            target_layer_ids: &[5, 19, 33, 47, 60],
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(
            err,
            "hidden-ring extract layers [5, 19, 33, 47, 61] != draft target_layer_ids [5, 19, 33, 47, 60]"
        );
    }

    #[test]
    fn rejects_ddtree() {
        let err = admit(Args {
            ddtree_present: true,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "DDTree is present");
    }

    #[test]
    fn rejects_missing_prefill_batch() {
        let err = admit(Args {
            prefill_batch_present: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "verify_scratch.prefill_batch is absent");
    }

    #[test]
    fn rejects_pbs_ineligible() {
        let err = admit(Args {
            pbs_eligible_b16: false,
            ..Args::default()
        })
        .unwrap_err();
        assert_eq!(err, "prefill_batch_pbs_eligible failed at B=16");
    }

    #[test]
    fn rejects_capacity_shortfalls() {
        let cases = [
            (
                Args {
                    verify_max_n: 15,
                    ..Args::default()
                },
                "verify_scratch.max_n=15 < 16",
            ),
            (
                Args {
                    pbs_max_batch: 15,
                    ..Args::default()
                },
                "prefill_batch.max_batch=15 < 16",
            ),
            (
                Args {
                    hidden_rb_max_batch: 15,
                    ..Args::default()
                },
                "hidden_rb.max_batch=15 < 16",
            ),
            (
                Args {
                    gdn_tape_max_n: 15,
                    ..Args::default()
                },
                "gdn_tape.max_n=15 < 16",
            ),
        ];
        for (args, expected) in cases {
            assert_eq!(admit(args).unwrap_err(), expected);
        }
    }
}
