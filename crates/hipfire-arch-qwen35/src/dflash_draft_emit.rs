// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Design A slice A4: the planned DFlash draft (railgun certified window).
//!
//! [`plan_dflash_draft`] runs the **existing eager draft body**
//! (`draft_dflash_forward_rank` + `draft_dflash_block_rank` for the greedy
//! temp==0 selector/argmax draft) under the frozen plan sink
//! (`Gpu::with_plan_sink`) and turns what the sink recorded into a typed
//! window segment: ordered kernels with their exact kernarg bytes, typed D2D
//! copies / fills, declared patch sites, a host delta and a shape family. No
//! kernel is launched and no host watermark is published.
//!
//! What changed relative to the eager body (everything else is the same call
//! sequence, so route decisions, kernels, order, grids and kernarg bytes are
//! the eager ones):
//!
//! * The two position uploads of `draft_forward_based` (`positions_q`,
//!   `positions_k[pos_base..]`, formerly `hipMemcpy` H2D) become D2D copies
//!   out of the owner's pre-uploaded staging buffer
//!   ([`DraftStagingLayout`], packed by [`draft_staged_words`]).
//! * The scalar noise-embedding loop keeps its 16 `embedding_lookup_*`
//!   launches; the seed row's token word is a declared `Scalar32` patch
//!   (`draft_seed`), the mask rows are model constants. (The gfx1100-only
//!   batched noise helper is *not* the gfx1201 singleton route:
//!   `Gpu::draft_collapse_fused_enabled` is exact gfx1100, so a plan on a
//!   fused-route GPU is refused.)
//! * The selector tail replaces `propose_candidates_device`'s two D2H round
//!   trips and host scorer with `topk_values_batched_f32` +
//!   `dflash_selector_greedy_exact` into the dedicated [`SelectorGreedyBlock`],
//!   then one typed copy of the 15 proposals into `verify_tokens[1..16]`.
//!   Without a selector the existing `argmax_f32_batched` runs and its 15
//!   words are copied instead.
//! * `mark_proj_cached` is not called; [`DraftEmission::host_delta`] carries
//!   the transition for the owner to commit after completion.
//!
//! ## Shape families
//!
//! Supported (the measured all-sliding DFlash2 window, `swa_w == full_w`):
//!
//! 1. **steady non-wrap**: `delta = L - max(C, L - W)` in `1..=16`, one fc /
//!    context-fill segment, `L < W` (span grows) or `L >= W` with the
//!    context ring on a slot boundary (one concat copy per K and V).
//! 2. **SWA wrap**: the new rows or the concat span cross the ring boundary:
//!    fc/fill split at `ring_segments` (segment row counts are the family
//!    key), concat splits into two copies per K and V.
//!
//! Refused (typed, before any work): legacy / split-window drafts (a
//! full-attention layer: its kernel route and `seq_len` layout are not
//! declared here), cold seed / reset / eviction / rebuild (`delta` outside
//! `1..=16`: family 4), online tuner, non-batched LM heads, F32 / unsupported
//! embedding tables, the gfx1100 fused-collapse route, `HIPFIRE_DFLASH_ORDERED_DRAFT=0`.
//!
//! ## Patches (node = ordinal among this emission's `Kernel` ops)
//!
//! * `Scalar32`: seed token of embedding row 0 (`draft_seed`); per layer the
//!   sliding-attention `l` (`+36`), tile size `ts` (`+56`), `ctx_span` (`+60`).
//! * `Interior64`: every pointer that lands in a shape-dependent window of
//!   `target_hidden`, `target_hidden_proj`, `positions_k`, `k_ctx_cached[i]`,
//!   `v_ctx_cached[i]` (ring slots). A kernel pointer inside one of those
//!   allocations that no declared candidate explains makes the emission fail.
//! * No `GridAxis`: GEMM / norm / RoPE grids depend on segment row counts,
//!   which are the family key; the attention grid is `[n_heads, B, 1]`.
//! * Copies/fills carry typed [`railgun::window::CopyStep`] expressions.
//!
//! Attention dynamic LDS is `(min(l, room) + block + head_dim) * 4` and has no
//! patch kind: the planned LDS is **saturated at the largest `l` of the
//! declared domain** (`attention_lds_saturated`); the tile size argument is the
//! patched `min(l, room)`, so only the allocation size differs from the eager
//! launch. The phase oracle byte-compares this across the domain.

use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;

use hip_bridge::{DeviceBuffer, HipError, HipResult};
use hipfire_runtime::dflash::{
    self, DflashConfig, DflashScratch, DflashWeights, DraftCtxMode, DraftStagedInputs,
    SelectorDeviceCodebooks, SelectorGreedyBlock,
};
use hipfire_runtime::llama;
use railgun::kernel::{Expr, PatchKind, ScalarField};
use railgun::window::{
    CopyStep, FillStep, HostDeltaSpec, Region, ShapeFamily, TransitionSpec, WindowShape, WindowStep,
};
use railgun::ResourceId;
use rdna_compute::plan_sink::{PlanSink, PlannedKernel, PlannedOp};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::qwen35::Qwen35Weights;
use crate::speculative::{draft_lmhead_batched, VerifyScratch};

/// Draft rows of a Whole16 window.
pub const DRAFT_ROWS: usize = 16;
/// Window scalar the owner supplies: the seed token of this window.
pub const SCALAR_DRAFT_SEED: &str = "draft_seed";

const SLIDING_ATTENTION_SYMBOL: &str = "attention_dflash_sliding_f32";
/// `attention_dflash_sliding_f32` kernarg offsets (`attention.rs`
/// `attention_dflash_sliding_f32`): q,k,v,out ptrs, then b,l,nh,nkv,hd,scale,
/// ts,cs,sw.
const ATTN_OFF_B: usize = 32;
const ATTN_OFF_L: usize = 36;
const ATTN_OFF_HD: usize = 48;
const ATTN_OFF_TS: usize = 56;
const ATTN_OFF_CS: usize = 60;
const ATTN_OFF_SW: usize = 64;
/// 56 KB LDS budget in f32 words (`attention_dflash_sliding_f32`).
const ATTN_LDS_BUDGET_F32: usize = 14_336;
/// Scalar embedding kernels (`embedding_q8`, `embedding_hfq4g256`,
/// `embedding_hfq4g128`): table, out, token@16, dim@20.
const EMBED_OFF_TOKEN: usize = 16;
const EMBED_OFF_DIM: usize = 20;

/// Byte offsets, inside the owner's pre-uploaded device staging buffer, of the
/// sections [`draft_staged_words`] describes. `k_off` must be the LAST
/// section: its length varies with the context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftStagingLayout {
    /// One u32 word: the window's seed token.
    pub seed_off: usize,
    /// `B` u32 words `[seed, mask x (B-1)]` (noise ids).
    pub ids_off: usize,
    /// `B` i32 absolute Q positions.
    pub q_off: usize,
    /// `min(L, full_w) + B` i32 words: the live K-position suffix then the B
    /// noise positions (`positions_k[pos_base..]`).
    pub k_off: usize,
}

/// Host words the owner packs (into `PinnedWindowInputs`) and uploads before
/// the window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftStagedWords {
    pub seed: u32,
    /// `[seed, mask x (B-1)]`: the noise-token id plane the eager route uploads
    /// before `embedding_q8_batched`.
    pub noise_ids: Vec<u32>,
    pub q_positions: Vec<u32>,
    pub k_positions: Vec<u32>,
}

/// The exact position words `draft_dflash_forward_rank` +
/// `draft_forward_based` upload today (`thlog.abs_positions()` preserved,
/// compaction offsets included).
pub fn draft_staged_words(
    abs_positions: &[i32],
    position: usize,
    compact_offset: i32,
    seed: u32,
    mask_token: u32,
    b: usize,
    full_w: usize,
) -> DraftStagedWords {
    let l = abs_positions.len().min(position);
    let live = l.min(full_w);
    let suffix = &abs_positions[abs_positions.len() - live..];
    let q: Vec<u32> = ((position as i32 + compact_offset)
        ..(position as i32 + b as i32 + compact_offset))
        .map(|p| p as u32)
        .collect();
    let mut k: Vec<u32> = suffix.iter().map(|&p| p as u32).collect();
    k.extend((0..b).map(|p| (position as i32 + p as i32 + compact_offset) as u32));
    let mut ids = vec![seed];
    ids.extend(std::iter::repeat(mask_token).take(b - 1));
    DraftStagedWords {
        seed,
        noise_ids: ids,
        q_positions: q,
        k_positions: k,
    }
}

/// One allocation the emission references, in `ResourceId` order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftResource {
    pub id: ResourceId,
    pub name: String,
    pub base: u64,
    pub bytes: u64,
    /// True when kernels reach it through shape-dependent interior pointers.
    pub dynamic: bool,
}

/// What [`plan_dflash_draft`] returns.
#[derive(Clone, Debug)]
pub struct DraftEmission {
    /// The frozen sink ops (kernel patches are attached to each kernel).
    pub ops: Vec<PlannedOp>,
    /// One typed step per op, in order (`Kernel { node }` / `Copy` / `Fill`);
    /// directly usable as `WindowPlanParts::steps`.
    pub steps: Vec<WindowStep>,
    pub resources: Vec<DraftResource>,
    /// `projected_rows -> logical_ctx_rows`, committed by the owner after
    /// completion (`thlog.mark_proj_cached(L)`, which also advances
    /// `full_cached_rows`). No upload watermark: the GPU-resident route never
    /// uploads `target_hidden`.
    pub host_delta: HostDeltaSpec,
    pub family: ShapeFamily,
    /// Concat copies per K and per V (1 or 2): part of the family.
    pub concat_segments: usize,
    /// Inclusive `logical_ctx_rows` domain over which block size and kernel
    /// topology stay those of this plan.
    pub l_domain: RangeInclusive<u64>,
    pub attention_lds_saturated: bool,
    /// Scalars the owner must put in `WindowInputs.scalars`.
    pub scalars_required: Vec<&'static str>,
    /// Resources read after completion (selector block/scores/status, or the
    /// argmax plane).
    pub completion_surfaces: Vec<ResourceId>,
    /// The shape this emission was planned at.
    pub shape: WindowShape,
    pub mask_token: u32,
    pub swa_w: usize,
}

impl DraftEmission {
    pub fn kernel_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|o| matches!(o, PlannedOp::Kernel(_)))
            .count()
    }

    /// `window scalars` for a window with seed `seed`.
    pub fn scalars(&self, seed: u32) -> BTreeMap<String, u64> {
        BTreeMap::from([(SCALAR_DRAFT_SEED.to_owned(), u64::from(seed))])
    }

    pub fn resource(&self, id: ResourceId) -> Option<&DraftResource> {
        self.resources.iter().find(|r| r.id == id)
    }

    /// True iff `shape` belongs to this emission's shape family and domain.
    /// A different family (delta, segment split, concat split) is a different
    /// plan, never a patch.
    pub fn accepts(&self, shape: &WindowShape) -> Result<(), String> {
        let key = draft_family_key(shape)?;
        if key.fc_rows != self.family.projection_rows {
            return Err(format!(
                "projection segments {:?} != planned {:?}",
                key.fc_rows, self.family.projection_rows
            ));
        }
        if key.concat_segments != self.concat_segments {
            return Err(format!(
                "concat segments {} != planned {}",
                key.concat_segments, self.concat_segments
            ));
        }
        if shape.swa_w != self.swa_w as u64 || shape.full_w != self.swa_w as u64 {
            return Err("swa_w/full_w differ from the plan".into());
        }
        if !self.l_domain.contains(&shape.logical_ctx_rows) {
            return Err(format!(
                "logical_ctx_rows {} outside the planned domain {:?}",
                shape.logical_ctx_rows, self.l_domain
            ));
        }
        Ok(())
    }
}

/// The topology-defining part of a window shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftFamilyKey {
    /// fc / hidden_norm segment row counts (= every layer's context-fill step
    /// row counts for the all-sliding draft).
    pub fc_rows: Vec<u32>,
    pub concat_segments: usize,
}

/// Family of `shape` (all-sliding windowed draft, `swa_w == full_w`).
pub fn draft_family_key(shape: &WindowShape) -> Result<DraftFamilyKey, String> {
    let w = usize::try_from(shape.swa_w).map_err(|_| "swa_w exceeds usize".to_string())?;
    let l =
        usize::try_from(shape.logical_ctx_rows).map_err(|_| "ctx rows exceed usize".to_string())?;
    let c = usize::try_from(shape.projected_rows)
        .map_err(|_| "projected rows exceed usize".to_string())?;
    if w == 0 || c > l {
        return Err(format!("degenerate shape: W={w} C={c} L={l}"));
    }
    let fc_start = c.max(l.saturating_sub(w));
    let delta = l - fc_start;
    if !(1..=DRAFT_ROWS).contains(&delta) {
        return Err(format!(
            "delta {delta} outside 1..=16 (cold seed / rebuild: shape family 4)"
        ));
    }
    let fc_rows = dflash::ring_segments(fc_start, l, w)
        .iter()
        .map(|&(_, _, len)| len as u32)
        .collect();
    let concat_segments = dflash::ring_segments(l.saturating_sub(w), l, w).len();
    Ok(DraftFamilyKey {
        fc_rows,
        concat_segments,
    })
}

/// What [`plan_dflash_draft`] needs. The draft runs exactly the arguments of
/// `draft_dflash_block_rank` (`ctx_slice = None`, no FFN graph).
pub struct DraftEmitArgs<'a> {
    pub target_weights: &'a Qwen35Weights,
    pub draft_weights: &'a DflashWeights,
    pub draft_cfg: &'a DflashConfig,
    pub draft_scratch: &'a mut DflashScratch,
    pub verify_scratch: &'a VerifyScratch,
    /// Exact selector codebooks and the dedicated proposal block; required iff
    /// the draft has a candidate selector.
    pub selector: Option<(&'a SelectorDeviceCodebooks, &'a SelectorGreedyBlock)>,
    /// Device staging the owner uploads the window inputs into.
    pub staging: &'a DeviceBuffer,
    pub layout: DraftStagingLayout,
    /// Verify token vector (`pbs.tokens.buf`): words `1..16` receive the 15
    /// proposals.
    pub verify_tokens: &'a DeviceBuffer,
    pub position: usize,
    pub compact_offset: i32,
    pub seed_token: u32,
    pub mask_token: u32,
    /// `WindowShape::max_position` bound of the window.
    pub max_position: u64,
}

// ------------------------------------------------------------ expressions

fn k(v: usize) -> Expr {
    Expr::Const(v as u64)
}
fn l_in() -> Expr {
    Expr::input("logical_ctx_rows")
}
fn c_in() -> Expr {
    Expr::input("projected_rows")
}
fn rem_const(a: Expr, m: usize) -> Expr {
    Expr::rem(a, m as u64).expect("ring modulus is nonzero")
}

fn eval(e: &Expr, shape: &WindowShape, scalars: &BTreeMap<String, u64>) -> Result<u64, String> {
    e.eval(&|n| shape.get(n).or_else(|| scalars.get(n).copied()))
        .map_err(|x| x.to_string())
}

// ------------------------------------------------------------ registry

struct Registry {
    items: Vec<DraftResource>,
}

impl Registry {
    fn add(&mut self, name: &str, buf: &DeviceBuffer, dynamic: bool) -> usize {
        let base = buf.as_ptr() as usize as u64;
        if let Some(i) = self.items.iter().position(|r| r.base == base) {
            self.items[i].dynamic |= dynamic;
            return i;
        }
        let id = ResourceId(self.items.len() as u32);
        self.items.push(DraftResource {
            id,
            name: name.to_owned(),
            base,
            bytes: buf.size() as u64,
            dynamic,
        });
        self.items.len() - 1
    }

    fn find(&self, addr: u64) -> Option<usize> {
        self.items
            .iter()
            .position(|r| addr >= r.base && addr < r.base + r.bytes)
    }

    fn region(&self, addr: u64) -> Result<Region, String> {
        let i = self
            .find(addr)
            .ok_or_else(|| format!("address {addr:#x} is in no registered allocation"))?;
        Ok(Region {
            resource: self.items[i].id,
            byte_offset: Expr::Const(addr - self.items[i].base),
        })
    }
}

fn tensor_bufs<'a>(
    s: &'a DflashScratch,
    vs: &'a VerifyScratch,
) -> Vec<(String, &'a GpuTensor, bool)> {
    let mut v: Vec<(String, &GpuTensor, bool)> = vec![
        ("x".into(), &s.x, false),
        ("x_norm".into(), &s.x_norm, false),
        ("q".into(), &s.q, false),
        ("k_noise".into(), &s.k_noise, false),
        ("v_noise".into(), &s.v_noise, false),
        ("gate".into(), &s.gate, false),
        ("up".into(), &s.up, false),
        ("gate_up".into(), &s.gate_up, false),
        ("attn_out".into(), &s.attn_out, false),
        ("residual".into(), &s.residual, false),
        ("target_hidden".into(), &s.target_hidden, true),
        ("target_hidden_proj".into(), &s.target_hidden_proj, true),
        ("k_cat".into(), &s.k_cat, false),
        ("v_cat".into(), &s.v_cat, false),
        ("positions_q".into(), &s.positions_q, false),
        ("positions_k".into(), &s.positions_k, true),
        ("noise_tokens".into(), &s.noise_tokens, false),
        ("vs_logits".into(), &vs.logits, false),
        ("vs_rot".into(), &vs.rot, false),
        ("vs_argmax".into(), &vs.argmax, false),
    ];
    for (n, t) in [
        ("mq_x_rot", &s.mq_x_rot),
        ("mq_x_rot_f16", &s.mq_x_rot_f16),
        ("conv_temp", &s.conv_temp),
        ("conv_dynamic", &s.conv_dynamic),
        ("selector_proj", &s.selector_proj),
        ("topk_ids", &s.topk_ids),
        ("topk_vals", &s.topk_vals),
        ("k_full_cached", &s.k_full_cached),
        ("v_full_cached", &s.v_full_cached),
        ("k_cat_full", &s.k_cat_full),
        ("v_cat_full", &s.v_cat_full),
    ] {
        if let Some(t) = t {
            v.push((n.to_owned(), t, false));
        }
    }
    for (i, t) in s.k_ctx_cached.iter().enumerate() {
        v.push((format!("k_ctx_cached[{i}]"), t, true));
    }
    for (i, t) in s.v_ctx_cached.iter().enumerate() {
        v.push((format!("v_ctx_cached[{i}]"), t, true));
    }
    v
}

/// Every device allocation the planned draft reads or writes, by name, for
/// the phase oracle's complete-state comparison.
pub fn draft_state_tensors<'a>(
    s: &'a DflashScratch,
    vs: &'a VerifyScratch,
) -> Vec<(String, &'a GpuTensor)> {
    tensor_bufs(s, vs)
        .into_iter()
        .map(|(n, t, _)| (n, t))
        .collect()
}

// ------------------------------------------------------------ candidates

struct PtrCand {
    res: usize,
    ptr: u64,
    off: Expr,
    extent: Expr,
}

struct CopyTemplate {
    src: (usize, Expr),
    dst: (usize, Expr),
    bytes: Expr,
}

fn attn_block(l: usize, hd: usize) -> usize {
    std::cmp::min(256, std::cmp::max(l, hd)).next_power_of_two()
}

fn le_u32(b: &[u8], off: usize) -> Option<u32> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn le_u64(b: &[u8], off: usize) -> Option<u64> {
    b.get(off..off + 8)
        .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
}

fn herr(e: HipError) -> String {
    e.to_string()
}

/// Plan the greedy DFlash draft of the next window. See the module docs.
pub fn plan_dflash_draft(gpu: &mut Gpu, a: DraftEmitArgs<'_>) -> Result<DraftEmission, String> {
    let DraftEmitArgs {
        target_weights,
        draft_weights,
        draft_cfg,
        draft_scratch,
        verify_scratch,
        selector,
        staging,
        layout,
        verify_tokens,
        position,
        compact_offset,
        seed_token,
        mask_token,
        max_position,
    } = a;
    let b = DRAFT_ROWS;
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let kvd = draft_cfg.kv_dim();
    let hd = draft_cfg.head_dim;
    let vocab = draft_cfg.vocab_size;
    let n_layers = draft_cfg.n_layers;
    let batch = b - 1;

    // ---- admission (typed refusals, before any work) ----------------------
    if gpu.plan_mode_active() {
        return Err("plan_dflash_draft: already in plan mode".into());
    }
    if gpu.draft_collapse_fused_enabled() {
        return Err(
            "refused: fused draft-collapse route (exact gfx1100) is not the gfx1201 body".into(),
        );
    }
    // Mirrors build_dflash_noise_embeddings_weights: batched route when admitted.
    let batched_embed =
        crate::speculative::noise_embed_batch_admitted(gpu, target_weights, draft_scratch, b);
    if !dflash::ordered_draft() {
        return Err(
            "refused: HIPFIRE_DFLASH_ORDERED_DRAFT=0 uses synchronous null-stream copies".into(),
        );
    }
    if draft_scratch.online.is_some() {
        return Err("refused: online draft tuner".into());
    }
    // Whole16 is the RUNTIME block (DflashState.block_size == 16); the draft's
    // trained block (draft_cfg.block_size, 8 for the 27B registry draft) is
    // irrelevant to the eager body, which runs `block.len()` rows. max_n only
    // needs the 15 head rows (verify scratch carries B+1 headroom).
    if batch > verify_scratch.max_n || b > draft_scratch.max_block_size {
        return Err(format!(
            "refused: runtime block {b} needs verify max_n >= {batch} (have {}) and scratch max block >= {b} (have {})",
            verify_scratch.max_n, draft_scratch.max_block_size
        ));
    }
    let (w, full_w) = match draft_scratch.ctx_mode {
        DraftCtxMode::Windowed { w, w_full } => (w, w_full),
        DraftCtxMode::Legacy => {
            return Err(
                "refused: legacy (non-windowed) draft: full-attention route is not declared".into(),
            )
        }
    };
    if !draft_cfg.all_layers_sliding || w != full_w {
        return Err(format!(
            "refused: split-window draft (all_layers_sliding={}, W={w}, W_full={full_w}): the full-attention layer's route is not declared",
            draft_cfg.all_layers_sliding
        ));
    }
    match target_weights.embd_format {
        llama::EmbeddingFormat::Q8_0
        | llama::EmbeddingFormat::HFQ4G256
        | llama::EmbeddingFormat::HFQ4G128 => {}
        other => {
            return Err(format!(
                "refused: scalar noise embedding for {other:?} has no declared token patch"
            ))
        }
    }
    let w_out = &target_weights.output;
    if !matches!(
        w_out.gpu_dtype,
        DType::Q8_0
            | DType::HFQ4G256
            | DType::MQ4G256
            | DType::MQ4G256V2
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MQ3G256
            | DType::HFQ6G256
            | DType::MQ6G256
    ) {
        return Err("refused: non-batched LM head (per-row GEMV + host readback)".into());
    }
    let selector_mode = draft_weights.has_candidate_selector();
    if selector_mode != selector.is_some() {
        return Err(
            "selector codebooks/block must be supplied exactly when the draft has a selector"
                .into(),
        );
    }
    if let Some((books, blk)) = selector {
        if blk.rows != batch {
            return Err(format!("selector block rows {} != {batch}", blk.rows));
        }
        if books.vocab != vocab {
            return Err("selector codebook vocab != draft vocab".into());
        }
    }

    let abs_len = draft_scratch.thlog.abs_positions().len();
    let l = abs_len.min(position);
    let c = draft_scratch.thlog.proj_cached_rows();
    let f = draft_scratch.thlog.full_cached_rows();
    if l == 0 || c > l || l > draft_scratch.max_ctx_len {
        return Err(format!(
            "refused: ctx rows L={l} cached C={c} cap {}",
            draft_scratch.max_ctx_len
        ));
    }
    let shape = WindowShape {
        position: position as u64,
        logical_ctx_rows: l as u64,
        projected_rows: c as u64,
        full_cached_rows: f as u64,
        compact_offset: u64::from(compact_offset.unsigned_abs()),
        ring_head: 0,
        swa_w: w as u64,
        full_w: full_w as u64,
        max_position,
        max_tiles: 0,
        verify_rows: b as u64,
        draft_rows: b as u64,
        generation: 0,
    };
    let fam = draft_family_key(&shape)?;
    let fc_start = c.max(l.saturating_sub(w));
    let segs = dflash::ring_segments(fc_start, l, w);
    let cat_segs = dflash::ring_segments(l.saturating_sub(w), l, w);

    // ---- resource registry --------------------------------------------------
    let mut reg = Registry { items: Vec::new() };
    for (name, t, dynamic) in tensor_bufs(draft_scratch, verify_scratch) {
        reg.add(&name, &t.buf, dynamic);
    }
    let res_staging = reg.add("staging", staging, false);
    let res_tokens = reg.add("verify_tokens", verify_tokens, false);
    if let Some((_, blk)) = selector {
        reg.add("selector_seed", blk.seed_buffer(), false);
        reg.add("selector_scores", blk.scores_buffer(), false);
        reg.add("selector_block", blk.block_buffer(), false);
        reg.add("selector_status", blk.status_buffer(), false);
    }
    let base_of = |name: &str| -> Result<(usize, u64), String> {
        let i = reg
            .items
            .iter()
            .position(|r| r.name == name)
            .ok_or_else(|| format!("resource {name} not registered"))?;
        Ok((i, reg.items[i].base))
    };
    let (i_th, b_th) = base_of("target_hidden")?;
    let (i_thp, b_thp) = base_of("target_hidden_proj")?;
    let (i_pk, b_pk) = base_of("positions_k")?;
    let (_i_kn, b_kn) = base_of("k_noise")?;
    let (_i_vn, b_vn) = base_of("v_noise")?;
    let (i_kc, b_kc) = base_of("k_cat")?;
    let (i_vc, b_vc) = base_of("v_cat")?;
    let (i_pq, b_pq) = base_of("positions_q")?;
    let staging_base = reg.items[res_staging].base;
    let _ = res_tokens;
    let mut k_cache: Vec<(usize, u64)> = Vec::new();
    let mut v_cache: Vec<(usize, u64)> = Vec::new();
    for li in 0..n_layers {
        k_cache.push(base_of(&format!("k_ctx_cached[{li}]"))?);
        v_cache.push(base_of(&format!("v_ctx_cached[{li}]"))?);
    }

    // ---- declared dynamic candidates ---------------------------------------
    let shape_ev = |e: &Expr| eval(e, &shape, &BTreeMap::new());
    let fc_start_e = Expr::max(c_in(), Expr::sub_sat(l_in(), k(w)));
    let pos_base_e = Expr::sub_sat(l_in(), k(full_w));
    let mut cands: Vec<PtrCand> = Vec::new();
    let add_cand = |cands: &mut Vec<PtrCand>,
                    res: usize,
                    base: u64,
                    off: Expr,
                    extent: Expr|
     -> Result<(), String> {
        let ptr = base + shape_ev(&off)?;
        if let Some(e) = cands.iter_mut().find(|e| e.res == res && e.ptr == ptr) {
            e.extent = Expr::max(e.extent.clone(), extent);
        } else {
            cands.push(PtrCand {
                res,
                ptr,
                off,
                extent,
            });
        }
        Ok(())
    };
    for &(row0, _slot0, len) in &segs {
        let prefix = (row0 - fc_start) as u64;
        let row_e = if prefix == 0 {
            fc_start_e.clone()
        } else {
            Expr::add(fc_start_e.clone(), Expr::Const(prefix))
        };
        let slot_e = rem_const(row_e.clone(), w);
        add_cand(
            &mut cands,
            i_th,
            b_th,
            Expr::mul(slot_e.clone(), k(ne * h * 4)),
            k(len * ne * h * 4),
        )?;
        add_cand(
            &mut cands,
            i_thp,
            b_thp,
            Expr::mul(slot_e.clone(), k(h * 4)),
            k(len * h * 4),
        )?;
        for li in 0..n_layers {
            add_cand(
                &mut cands,
                k_cache[li].0,
                k_cache[li].1,
                Expr::mul(slot_e.clone(), k(kvd * 4)),
                k(len * kvd * 4),
            )?;
            add_cand(
                &mut cands,
                v_cache[li].0,
                v_cache[li].1,
                Expr::mul(slot_e.clone(), k(kvd * 4)),
                k(len * kvd * 4),
            )?;
        }
        add_cand(
            &mut cands,
            i_pk,
            b_pk,
            Expr::mul(Expr::sub_sat(row_e, pos_base_e.clone()), k(4)),
            k(len * 4),
        )?;
    }

    // ---- declared dynamic copies --------------------------------------------
    let span_e = Expr::min(l_in(), k(w));
    let span_start = l.saturating_sub(w);
    let s_e = rem_const(Expr::sub_sat(l_in(), k(w)), w);
    let mut templates: HashMap<(u64, u64, u64), CopyTemplate> = HashMap::new();
    let add_tpl = |t: &mut HashMap<(u64, u64, u64), CopyTemplate>,
                   key: (u64, u64, u64),
                   tpl: CopyTemplate,
                   base_src: u64,
                   base_dst: u64|
     -> Result<(), String> {
        // The expressions must reproduce the planned addresses at the plan shape.
        let src = base_src + shape_ev(&tpl.src.1)?;
        let dst = base_dst + shape_ev(&tpl.dst.1)?;
        let bytes = shape_ev(&tpl.bytes)?;
        if (src, dst, bytes) != key {
            return Err(format!(
                "copy template expressions evaluate to {:#x}/{:#x}/{} but the body copies {:#x}/{:#x}/{}",
                src, dst, bytes, key.0, key.1, key.2
            ));
        }
        t.insert(key, tpl);
        Ok(())
    };
    // positions_k ← staging
    let k_rows = l.min(full_w) + b;
    add_tpl(
        &mut templates,
        (
            staging_base + layout.k_off as u64,
            b_pk,
            (k_rows * 4) as u64,
        ),
        CopyTemplate {
            src: (res_staging, k(layout.k_off)),
            dst: (i_pk, k(0)),
            bytes: Expr::mul(Expr::add(Expr::min(l_in(), k(full_w)), k(b)), k(4)),
        },
        staging_base,
        b_pk,
    )?;
    let kvd4 = kvd * 4;
    for li in 0..n_layers {
        for (is_k, (cache_res, cache_base), (cat_res, cat_base), noise_base) in [
            (true, k_cache[li], (i_kc, b_kc), b_kn),
            (false, v_cache[li], (i_vc, b_vc), b_vn),
        ] {
            let _ = is_k;
            let mut cat_off = 0usize;
            for (j, &(_r0, slot0, len)) in cat_segs.iter().enumerate() {
                let (src_off, bytes, dst_off) = if cat_segs.len() == 1 {
                    (
                        Expr::mul(s_e.clone(), k(kvd4)),
                        Expr::mul(span_e.clone(), k(kvd4)),
                        k(0),
                    )
                } else if j == 0 {
                    (
                        Expr::mul(s_e.clone(), k(kvd4)),
                        Expr::mul(Expr::sub_sat(k(w), s_e.clone()), k(kvd4)),
                        k(0),
                    )
                } else {
                    (
                        k(0),
                        Expr::mul(s_e.clone(), k(kvd4)),
                        Expr::mul(Expr::sub_sat(k(w), s_e.clone()), k(kvd4)),
                    )
                };
                add_tpl(
                    &mut templates,
                    (
                        cache_base + (slot0 * kvd4) as u64,
                        cat_base + cat_off as u64,
                        (len * kvd4) as u64,
                    ),
                    CopyTemplate {
                        src: (cache_res, src_off),
                        dst: (cat_res, dst_off),
                        bytes,
                    },
                    cache_base,
                    cat_base,
                )?;
                cat_off += len * kvd4;
            }
            // noise rows land after the span
            let span_bytes = (l - span_start) * kvd4;
            add_tpl(
                &mut templates,
                (noise_base, cat_base + span_bytes as u64, (b * kvd4) as u64),
                CopyTemplate {
                    src: (if is_k { _i_kn } else { _i_vn }, k(0)),
                    dst: (cat_res, Expr::mul(span_e.clone(), k(kvd4))),
                    bytes: k(b * kvd4),
                },
                noise_base,
                cat_base,
            )?;
        }
    }

    // ---- run the eager body under the sink ------------------------------------
    let thlog_before = (
        draft_scratch.thlog.uploaded_rows(),
        draft_scratch.thlog.proj_cached_rows(),
        draft_scratch.thlog.full_cached_rows(),
        abs_len,
    );
    let block: Vec<u32> = std::iter::once(seed_token)
        .chain(std::iter::repeat(mask_token).take(b - 1))
        .collect();
    // Design §2: materialize lazy JIT/module loads OUTSIDE plan mode. The
    // selector is the one module the eager draft never launches; everything
    // else the body emits (embedding, topk, argmax, GEMM/norm/rope/attention)
    // was loaded by the eager window of the same shape that precedes a plan.
    // Pending route-module preloads (the launch funnels' first-launch hook)
    // are drained here too.
    gpu.ensure_route_modules_preloaded().map_err(herr)?;
    if selector.is_some() {
        gpu.ensure_dflash_selector_greedy_exact().map_err(herr)?;
    }
    let mut sink = PlanSink::default();
    let staged = DraftStagedInputs {
        buf: staging,
        q_byte_off: layout.q_off,
        k_byte_off: layout.k_off,
    };
    let planned: HipResult<()> = gpu.with_plan_sink(&mut sink, |gpu| {
        // Seed word → the selector block's device seed (its owner stages it).
        if let Some((_, blk)) = selector {
            gpu.memcpy_dtod_at_ordered_async(blk.seed_buffer(), 0, staging, layout.seed_off, 4)?;
        }
        if batched_embed {
            // Eager route: one id upload + one embedding_q8_batched. The ids
            // [seed, mask x15] are staged; the upload becomes a typed D2D copy.
            gpu.memcpy_dtod_at_ordered_async(
                &draft_scratch.noise_tokens.buf,
                0,
                staging,
                layout.ids_off,
                b * 4,
            )?;
            let ids_view = draft_scratch.noise_tokens.sub_offset(0, b);
            let out_view = draft_scratch.x.sub_offset(0, b * h);
            gpu.embedding_lookup_q8_batched(
                &target_weights.token_embd,
                &out_view,
                &ids_view,
                b,
                h,
            )?;
        } else {
            // Scalar noise embeddings: draft_dflash_forward_rank's fallback loop.
            for (i, &tok) in block.iter().enumerate() {
                let dst = draft_scratch.x.sub_offset(i * h, h);
                match target_weights.embd_format {
                    llama::EmbeddingFormat::HFQ4G256 => {
                        gpu.embedding_lookup_hfq4g256(&target_weights.token_embd, &dst, tok, h)?
                    }
                    llama::EmbeddingFormat::HFQ4G128 => {
                        gpu.embedding_lookup_hfq4g128(&target_weights.token_embd, &dst, tok, h)?
                    }
                    llama::EmbeddingFormat::Q8_0 => {
                        gpu.embedding_lookup_q8(&target_weights.token_embd, &dst, tok, h)?
                    }
                    _ => unreachable!("admission checked the embedding format"),
                }
            }
        }
        dflash::draft_forward_planned(gpu, draft_weights, draft_cfg, &staged, b, l, draft_scratch)?;
        let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
        let logits_batch = verify_scratch.logits.sub_offset(0, batch * vocab);
        draft_lmhead_batched(
            gpu,
            w_out,
            verify_scratch,
            &hidden_rows,
            &logits_batch,
            batch,
            h,
        )?;
        if let Some((books, blk)) = selector {
            dflash::enqueue_selector_greedy_device(
                gpu,
                draft_weights,
                draft_scratch,
                books,
                blk,
                &hidden_rows,
                &logits_batch,
            )?;
            // The 15 proposals (block[1..16]) → verify tokens [1..16].
            gpu.memcpy_dtod_at_ordered_async(verify_tokens, 4, blk.block_buffer(), 4, batch * 4)?;
        } else {
            let argmax_buf = verify_scratch.argmax.sub_offset(0, batch);
            gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, batch)?;
            gpu.memcpy_dtod_at_ordered_async(verify_tokens, 4, &argmax_buf.buf, 0, batch * 4)?;
        }
        Ok(())
    });
    if let Some(why) = sink.refusal.take() {
        return Err(format!("plan refused: {why}"));
    }
    planned.map_err(herr)?;
    let thlog_after = (
        draft_scratch.thlog.uploaded_rows(),
        draft_scratch.thlog.proj_cached_rows(),
        draft_scratch.thlog.full_cached_rows(),
        draft_scratch.thlog.abs_positions().len(),
    );
    if thlog_before != thlog_after {
        return Err(format!(
            "planning mutated host watermarks {thlog_before:?} -> {thlog_after:?}"
        ));
    }
    let mut ops = sink.ops;

    // ---- attach patches / convert to typed steps --------------------------------
    let dyn_parent: Vec<bool> = reg.items.iter().map(|r| r.dynamic).collect();
    let mut steps: Vec<WindowStep> = Vec::with_capacity(ops.len());
    let mut node = 0usize;
    let mut embed_seen = 0usize;
    let mut l_domain: Option<RangeInclusive<u64>> = None;
    let mut lds_saturated = false;
    let mut kernels: Vec<String> = Vec::new();
    for op in ops.iter_mut() {
        match op {
            PlannedOp::Kernel(kn) => {
                kernels.push(kn.symbol.clone());
                // pointer patches
                let mut at = 0usize;
                while at + 8 <= kn.kernarg.len() {
                    if let Some(v) = le_u64(&kn.kernarg, at) {
                        if let Some(ri) = reg.find(v) {
                            if dyn_parent[ri] {
                                let Some(cand) = cands.iter().find(|c| c.res == ri && c.ptr == v)
                                else {
                                    return Err(format!(
                                        "{}: kernarg+{at} points {:#x} into dynamic allocation {} with no declared window",
                                        kn.symbol, v, reg.items[ri].name
                                    ));
                                };
                                kn.patches.push(PatchKind::Interior64 {
                                    node,
                                    arg_offset: at as u32,
                                    resource: reg.items[ri].id,
                                    byte_offset: cand.off.clone(),
                                    required_extent: cand.extent.clone(),
                                });
                            }
                        }
                    }
                    at += 8;
                }
                match kn.symbol.as_str() {
                    "embedding_q8" | "embedding_hfq4g256" | "embedding_hfq4g128" => {
                        if le_u32(&kn.kernarg, EMBED_OFF_DIM) != Some(h as u32) {
                            return Err(format!("{}: unexpected dim word", kn.symbol));
                        }
                        let tok = le_u32(&kn.kernarg, EMBED_OFF_TOKEN).unwrap_or(u32::MAX);
                        if embed_seen == 0 {
                            if tok != seed_token {
                                return Err(format!(
                                    "embedding row 0 token {tok} != seed {seed_token}"
                                ));
                            }
                            kn.patches.push(PatchKind::Scalar32 {
                                field: ScalarField {
                                    node,
                                    offset: EMBED_OFF_TOKEN as u32,
                                },
                                expr: Expr::input(SCALAR_DRAFT_SEED),
                                domain: 0..=(vocab as u64 - 1),
                            });
                        } else if tok != mask_token {
                            return Err(format!(
                                "embedding row {embed_seen} token {tok} != mask {mask_token}"
                            ));
                        }
                        embed_seen += 1;
                    }
                    SLIDING_ATTENTION_SYMBOL => {
                        let blk = kn.block[0] as usize;
                        let span = l.min(w);
                        let l_plan = span + b;
                        let room = ATTN_LDS_BUDGET_F32.saturating_sub(blk + hd).max(1);
                        let ts_plan = l_plan.min(room);
                        let check = |off: usize, want: usize, what: &str| -> Result<(), String> {
                            let got = le_u32(&kn.kernarg, off);
                            if got != Some(want as u32) {
                                return Err(format!(
                                    "{SLIDING_ATTENTION_SYMBOL}: {what}@{off} = {got:?}, expected {want}"
                                ));
                            }
                            Ok(())
                        };
                        check(ATTN_OFF_B, b, "b")?;
                        check(ATTN_OFF_L, l_plan, "l")?;
                        check(ATTN_OFF_HD, hd, "head_dim")?;
                        check(ATTN_OFF_TS, ts_plan, "tile_size")?;
                        check(ATTN_OFF_CS, span, "ctx_span")?;
                        check(ATTN_OFF_SW, w, "sliding_window")?;
                        // Domain: block size must stay the planned one.
                        let mut lo = l_plan;
                        while lo > b + 1 && attn_block(lo - 1, hd) == blk {
                            lo -= 1;
                        }
                        let mut hi = l_plan;
                        while hi < w + b && attn_block(hi + 1, hd) == blk {
                            hi += 1;
                        }
                        // The kernel's `l` is `min(L, W) + b`: once the plan
                        // sits on the saturated block (hi reached W + b) every
                        // larger logical context maps to the same `l`, so the
                        // logical domain is unbounded above.
                        let dom_hi = if hi >= w + b {
                            u64::MAX
                        } else {
                            (hi - b) as u64
                        };
                        let dom = (lo - b) as u64..=dom_hi;
                        l_domain = Some(match l_domain.take() {
                            None => dom,
                            Some(prev) => {
                                (*prev.start().max(dom.start()))..=(*prev.end().min(dom.end()))
                            }
                        });
                        let span_pat = Expr::min(l_in(), k(w));
                        kn.patches.push(PatchKind::Scalar32 {
                            field: ScalarField {
                                node,
                                offset: ATTN_OFF_L as u32,
                            },
                            expr: Expr::add(span_pat.clone(), k(b)),
                            domain: lo as u64..=hi as u64,
                        });
                        kn.patches.push(PatchKind::Scalar32 {
                            field: ScalarField {
                                node,
                                offset: ATTN_OFF_TS as u32,
                            },
                            expr: Expr::min(Expr::add(span_pat.clone(), k(b)), k(room)),
                            domain: lo.min(room) as u64..=hi.min(room) as u64,
                        });
                        kn.patches.push(PatchKind::Scalar32 {
                            field: ScalarField {
                                node,
                                offset: ATTN_OFF_CS as u32,
                            },
                            expr: span_pat,
                            domain: (lo - b) as u64..=(hi - b) as u64,
                        });
                        // No LDS patch kind: saturate at the domain's largest tile.
                        let lds_hi = ((hi.min(room) + blk + hd) * 4) as u32;
                        if lds_hi != kn.dynamic_lds {
                            lds_saturated = true;
                            kn.dynamic_lds = lds_hi;
                        }
                    }
                    _ => {}
                }
                steps.push(WindowStep::Kernel { node });
                node += 1;
            }
            PlannedOp::Copy { dst, src, bytes } => {
                let key = (*src, *dst, *bytes);
                let step = if let Some(t) = templates.get(&key) {
                    CopyStep {
                        src: Region {
                            resource: reg.items[t.src.0].id,
                            byte_offset: t.src.1.clone(),
                        },
                        dst: Region {
                            resource: reg.items[t.dst.0].id,
                            byte_offset: t.dst.1.clone(),
                        },
                        bytes: t.bytes.clone(),
                    }
                } else {
                    CopyStep {
                        src: reg.region(*src)?,
                        dst: reg.region(*dst)?,
                        bytes: Expr::Const(*bytes),
                    }
                };
                steps.push(WindowStep::Copy(step));
            }
            PlannedOp::Fill { dst, value, words } => {
                steps.push(WindowStep::Fill(FillStep {
                    dst: reg.region(*dst)?,
                    value: *value,
                    words: Expr::Const(u64::from(*words)),
                }));
            }
        }
    }
    let want_scalar = if batched_embed { 0 } else { b };
    if embed_seen != want_scalar {
        return Err(format!(
            "{embed_seen} scalar embedding launches, expected {want_scalar}"
        ));
    }
    if batched_embed && !kernels.iter().any(|s| s == "embedding_q8_batched") {
        return Err("batched noise route planned no embedding_q8_batched launch".into());
    }
    let attn_count = kernels
        .iter()
        .filter(|s| *s == SLIDING_ATTENTION_SYMBOL)
        .count();
    if attn_count != n_layers {
        return Err(format!(
            "{attn_count} sliding attention launches for {n_layers} layers (a layer took another attention route)"
        ));
    }
    // Every dynamic copy template must have been used by the body.
    let copies_seen: std::collections::HashSet<(u64, u64, u64)> = ops
        .iter()
        .filter_map(|o| match o {
            PlannedOp::Copy { dst, src, bytes } => Some((*src, *dst, *bytes)),
            _ => None,
        })
        .collect();
    if let Some(missing) = templates.keys().find(|k| !copies_seen.contains(k)) {
        return Err(format!(
            "declared dynamic copy {:#x}->{:#x} ({} bytes) never issued by the body",
            missing.0, missing.1, missing.2
        ));
    }
    let _ = (i_pq, b_pq);

    let completion_surfaces = if selector.is_some() {
        ["selector_block", "selector_scores", "selector_status"]
            .iter()
            .map(|n| base_of(n).map(|(i, _)| reg.items[i].id))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![reg.items[base_of("vs_argmax")?.0].id]
    };
    let host_delta = HostDeltaSpec {
        projection_watermark: Some(TransitionSpec {
            from: c_in(),
            to: l_in(),
        }),
        ..HostDeltaSpec::default()
    };
    Ok(DraftEmission {
        ops,
        steps,
        resources: reg.items,
        host_delta,
        family: ShapeFamily {
            kernels,
            projection_rows: fam.fc_rows,
        },
        concat_segments: fam.concat_segments,
        l_domain: l_domain.unwrap_or(1..=w as u64),
        attention_lds_saturated: lds_saturated,
        scalars_required: if batched_embed {
            vec![]
        } else {
            vec![SCALAR_DRAFT_SEED]
        },
        completion_surfaces,
        shape,
        mask_token,
        swa_w: w,
    })
}

/// The oracle-grade binder: evaluate every declared patch and typed copy
/// expression of `em` at `shape`/`scalars` and return concrete ops that
/// `Gpu::run_planned_ops` can issue. (`railgun::window::PreparedWindow::bind`
/// is the production binder; this one exists so the draft's patch table can
/// be proven against the eager body on a GPU without a PM4 queue.)
pub fn bind_draft_ops(
    em: &DraftEmission,
    shape: &WindowShape,
    scalars: &BTreeMap<String, u64>,
) -> Result<Vec<PlannedOp>, String> {
    em.accepts(shape)?;
    let ev = |e: &Expr| eval(e, shape, scalars);
    let base = |r: ResourceId| -> Result<&DraftResource, String> {
        em.resource(r)
            .ok_or_else(|| format!("unknown resource {r:?}"))
    };
    let mut out = Vec::with_capacity(em.ops.len());
    for (op, step) in em.ops.iter().zip(&em.steps) {
        match (op, step) {
            (PlannedOp::Kernel(k0), WindowStep::Kernel { .. }) => {
                let mut kn: PlannedKernel = k0.clone();
                kn.patches.clear();
                for p in &k0.patches {
                    match p {
                        PatchKind::Scalar32 {
                            field,
                            expr,
                            domain,
                        } => {
                            let v = ev(expr)?;
                            if !domain.contains(&v) {
                                return Err(format!(
                                    "{}: scalar {v} outside {domain:?}",
                                    kn.symbol
                                ));
                            }
                            let v32 =
                                u32::try_from(v).map_err(|_| "scalar exceeds u32".to_string())?;
                            let o = field.offset as usize;
                            kn.kernarg[o..o + 4].copy_from_slice(&v32.to_le_bytes());
                        }
                        PatchKind::Interior64 {
                            arg_offset,
                            resource,
                            byte_offset,
                            required_extent,
                            ..
                        } => {
                            let r = base(*resource)?;
                            let off = ev(byte_offset)?;
                            let ext = ev(required_extent)?;
                            if off.checked_add(ext).map_or(true, |e| e > r.bytes) {
                                return Err(format!(
                                    "{}: window {off}+{ext} exceeds {} ({} bytes)",
                                    kn.symbol, r.name, r.bytes
                                ));
                            }
                            let o = *arg_offset as usize;
                            kn.kernarg[o..o + 8].copy_from_slice(&(r.base + off).to_le_bytes());
                        }
                        PatchKind::GridAxis {
                            axis, expr, domain, ..
                        } => {
                            let v = ev(expr)?;
                            if v == 0 || !domain.contains(&v) {
                                return Err(format!("{}: grid {v} outside {domain:?}", kn.symbol));
                            }
                            kn.grid[axis.index()] =
                                u32::try_from(v).map_err(|_| "grid exceeds u32".to_string())?;
                        }
                    }
                }
                out.push(PlannedOp::Kernel(kn));
            }
            (PlannedOp::Copy { .. }, WindowStep::Copy(cs)) => {
                let (s, d) = (base(cs.src.resource)?, base(cs.dst.resource)?);
                let (so, dof, bytes) = (
                    ev(&cs.src.byte_offset)?,
                    ev(&cs.dst.byte_offset)?,
                    ev(&cs.bytes)?,
                );
                if so + bytes > s.bytes || dof + bytes > d.bytes {
                    return Err(format!(
                        "copy {}+{so}->{}+{dof} of {bytes} bytes exceeds an allocation",
                        s.name, d.name
                    ));
                }
                out.push(PlannedOp::Copy {
                    dst: d.base + dof,
                    src: s.base + so,
                    bytes,
                });
            }
            (PlannedOp::Fill { .. }, WindowStep::Fill(fs)) => {
                let d = base(fs.dst.resource)?;
                let off = ev(&fs.dst.byte_offset)?;
                let words = ev(&fs.words)?;
                if off + words * 4 > d.bytes {
                    return Err("fill exceeds its allocation".into());
                }
                out.push(PlannedOp::Fill {
                    dst: d.base + off,
                    value: fs.value,
                    words: u32::try_from(words).map_err(|_| "fill words exceed u32".to_string())?,
                });
            }
            _ => return Err("ops and steps are out of step".into()),
        }
    }
    Ok(out)
}
