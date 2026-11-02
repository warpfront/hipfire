// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Railgun design A slice A5: the verify-window emitter.
//!
//! [`plan_dflash_verify`] authors the DFlash chain verify window (the fixed
//! B-row target verify the HIP-graph body runs) as an ordered list of typed
//! [`PlannedOp`]s through the rdna-compute plan sink, without launching
//! anything:
//!
//! 1. **forward** — `forward_prefill_batch_single_chunk_captured_opts`, the
//!    body `verify_dflash_block_inner_body` warms up / captures / replays,
//!    planned under the *graph-body route policy*: the plan sets
//!    `graphs.capture_mode` (route decisions such as `max_ctx_len =
//!    kv.physical_cap`, the multi-row attention gate and the hidden-ring
//!    staging read it) but never Redline recording, because
//!    `launch_verify_gqa` refuses recording and would pick another body
//!    (`attention.rs:9936-9953`). The launch funnels check the plan flag
//!    first, so nothing is captured or launched.
//! 2. **hidden commit** — `commit_staging_to_ring` split into its GPU emit
//!    ([`HiddenStateRingBuffer::emit_commit_staging`], the fused `commit5`
//!    kernel with `ring_head` as a declared `Scalar32` patch) and its host
//!    delta ([`HiddenStateRingBuffer::commit_delta`]); planned under the
//!    ordinary eager route exactly as the HIP path runs it after the graph.
//! 3. **verify head** — `dflash_enqueue_verify_lm_head_argmax`
//!    (lm_head GEMM + `argmax_f32_batched`), the eager head of the HIP path.
//!    The two readbacks (`dflash_download_verify_argmax`, logits) stay with
//!    the owner after completion.
//!
//! Inputs (`pbs.tokens` / `pbs.positions`, uploaded by
//! `upload_prefill_batch_inputs` / `PinnedWindowInputs` before the IB) are
//! device-resident, so the forward's kernargs are window-invariant by
//! construction of the HIP graph (the same blobs replay at every position).
//! The only per-window scalar is `commit5.head`; the declared patch table is
//! returned with the plan and [`undeclared_varying_words`] proves it against
//! a second plan at another shape.
//!
//! [`run_planned_ops`](rdna_compute::Gpu::run_planned_ops) is the emitted-HIP
//! twin runner used by the phase oracle
//! (`examples/dflash_verify_emit_oracle.rs`).

use crate::qwen35::{self, prefill::fp8_verify_target};
use crate::speculative::{
    dflash_batched_lm_head_supported, dflash_enqueue_verify_lm_head_argmax,
    dflash_moe_verify_graph_lmhead_eligible, dflash_verify_fusion_ctx,
    dflash_verify_hip_graph_route_ok, GdnTape, HiddenStateRingBuffer, ModelSlot, VerifyScratch,
};
use hipfire_dispatch::pipeline::batched::verify_dense_math;
use railgun::kernel::{DeclaredWord, Expr, PatchKind, ScalarField};
use railgun::window::{Transition, WindowHostDelta};
use rdna_compute::dflash_hidden_scatter::DFLASH_HIDDEN_COMMIT5;
use rdna_compute::plan_sink::{diff_plans, take_declared_words, KernargDiff};
use rdna_compute::{Gpu, PlanSink, PlannedOp};
use std::collections::HashSet;
use std::ops::Range;

/// Byte offset of the `head` i32 in the `dflash_hidden_commit5` kernarg
/// (five staging pointers, five ring pointers, then `head`, `n`, `hidden`,
/// `max_pos`; see `Gpu::dflash_hidden_commit5_launch`).
pub const COMMIT5_HEAD_OFFSET: u32 = 80;

/// One planned verify window.
#[derive(Clone, Debug)]
pub struct DflashVerifyPlan {
    /// Forward, then hidden commit, then verify head, in launch order.
    pub ops: Vec<PlannedOp>,
    /// Op ranges of the three phases.
    pub forward: Range<usize>,
    pub commit: Range<usize>,
    pub head: Range<usize>,
    /// Ring head/written transitions the commit proposes (planning does not
    /// publish them).
    pub host_delta: WindowHostDelta,
    /// `gdn_requant_frame_checkpoint()` the plan was authored at, and how
    /// many frames its launches reserved. Planning restores the counter; the
    /// owner advances it only where HIP would (a graph *replay* reserves
    /// none).
    pub gdn_frame_base: u32,
    pub gdn_frames_reserved: u32,
    /// Railgun-declared dynamic words (`GdnFrame`, ...) as `(op index, words)`.
    pub declared_words: Vec<(usize, Vec<DeclaredWord>)>,
    pub rows: usize,
    pub start_pos: usize,
    pub ring_head: usize,
}

impl DflashVerifyPlan {
    /// The kernels of the plan, in order.
    pub fn kernels(&self) -> impl Iterator<Item = &rdna_compute::PlannedKernel> {
        self.ops.iter().filter_map(|op| match op {
            PlannedOp::Kernel(k) => Some(k),
            _ => None,
        })
    }

    /// The declared patch table: `(kernel ordinal, symbol, patches)`.
    pub fn patch_table(&self) -> Vec<(usize, &str, &[PatchKind])> {
        self.kernels()
            .enumerate()
            .filter(|(_, k)| !k.patches.is_empty())
            .map(|(i, k)| (i, k.symbol.as_str(), k.patches.as_slice()))
            .collect()
    }
}

fn hip_err(what: &str, err: hip_bridge::HipError) -> String {
    format!("{what}: {err}")
}

fn plan_err(what: &str, sink: &PlanSink, err: hip_bridge::HipError) -> String {
    match &sink.refusal {
        Some(why) => format!("{what}: plan refused: {why}"),
        None => hip_err(what, err),
    }
}

/// Plan one chain verify window of `draft_tokens.len()` rows at `start_pos`
/// and return its ops (design A §5 / slice A5). See
/// [`plan_dflash_verify_full`] for the phase split, host delta and patch
/// table.
#[allow(clippy::too_many_arguments)]
pub fn plan_dflash_verify(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    verify_scratch: &VerifyScratch,
) -> Result<Vec<PlannedOp>, String> {
    plan_dflash_verify_full(
        gpu,
        target,
        draft_tokens,
        start_pos,
        hidden_rb,
        gdn_tape,
        verify_scratch,
    )
    .map(|plan| plan.ops)
}

/// Plan one chain verify window with its phase ranges, host delta, frame
/// accounting and declared patch sites.
///
/// Preconditions (the same ones the HIP-graph body has): the window takes the
/// graph body (`dflash_verify_hip_graph_route_ok`), the lm_head is a batched
/// dtype, and one ordinary window of this shape has already run so every
/// lazily-built kernel/scratch exists (a `hipModuleLoad`/`hipMalloc` in plan
/// mode refuses the plan). Nothing is launched; the GDN requant-frame counter
/// is restored; `hidden_rb` is not advanced.
///
/// Refuses (`Err`) rather than approximating when: the window would take the
/// eager/IU4/MoE-extended route, the hidden commit is not the fused `commit5`
/// kernel (the copy-loop commit bakes `ring_head` into copy addresses, which
/// no declared patch kind can express), or any HIP API outside blob launch /
/// D2D copy / word memset is reached.
#[allow(clippy::too_many_arguments)]
pub fn plan_dflash_verify_full(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    verify_scratch: &VerifyScratch,
) -> Result<DflashVerifyPlan, String> {
    if gpu.plan_mode_active() {
        return Err("plan_dflash_verify: a plan sink is already active".into());
    }
    let b = draft_tokens.len();
    if b < 2 || b > verify_scratch.max_n {
        return Err(format!(
            "plan_dflash_verify: {b} rows outside 2..={}",
            verify_scratch.max_n
        ));
    }
    if b > hidden_rb.max_batch {
        return Err(format!(
            "plan_dflash_verify: {b} rows exceed hidden staging max_batch {}",
            hidden_rb.max_batch
        ));
    }
    let required_tokens = start_pos
        .checked_add(b)
        .ok_or_else(|| format!("plan_dflash_verify: KV range overflow ({start_pos} + {b})"))?;
    let pbs = verify_scratch
        .prefill_batch
        .as_ref()
        .ok_or("plan_dflash_verify: verify scratch has no PrefillBatchScratch")?;
    if b > pbs.max_batch {
        return Err(format!(
            "plan_dflash_verify: {b} rows exceed PrefillBatchScratch.max_batch {}",
            pbs.max_batch
        ));
    }

    // The same route preludes the HIP body runs before it touches the graph.
    // They are host/mapping work and live outside plan mode.
    qwen35::prefill::release_widened_pbs_for_kv_growth(
        gpu,
        &target.kv_cache,
        &target.config,
        &target.scratch,
        required_tokens,
    )
    .map_err(|e| hip_err("release_widened_pbs_for_kv_growth", e))?;
    target
        .kv_cache
        .ensure_mapped_capacity(gpu, required_tokens)
        .map_err(|e| hip_err("ensure_mapped_capacity", e))?;
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(
            gpu.hip
                .stream_create()
                .map_err(|e| hip_err("stream_create", e))?,
        );
    }

    // The PM FP8 / IU4 verify scopes `verify_dflash_block_inner` opens around
    // the body (route flags only; entered and left outside plan mode).
    let fp8 = fp8_verify_target(&target.weights, &target.config, &target.kv_cache);
    let iu4 =
        qwen35::prefill::multi::dflash_iu4_verify_admitted(gpu, &target.weights, &target.config);
    let prev_fp8 = gpu
        .fp8_verify_enter(fp8)
        .map_err(|e| hip_err("fp8_verify_enter", e))?;
    let prev_iu4 = gpu.iu4_verify_enter(iu4);

    let frame_base = rdna_compute::norm::gdn_requant_frame_checkpoint();
    let phases = plan_phases(
        gpu,
        target,
        draft_tokens,
        start_pos,
        hidden_rb,
        gdn_tape,
        verify_scratch,
        pbs,
        required_tokens,
        frame_base,
    );
    let frame_after = rdna_compute::norm::gdn_requant_frame_checkpoint();
    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_base);

    gpu.iu4_verify_exit(prev_iu4);
    gpu.fp8_verify_exit(prev_fp8);

    let mut plan = phases?;
    plan.gdn_frames_reserved = frame_after.wrapping_sub(frame_base);
    Ok(plan)
}

#[allow(clippy::too_many_arguments)]
fn plan_phases(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    verify_scratch: &VerifyScratch,
    pbs: &qwen35::PrefillBatchScratch,
    required_tokens: usize,
    frame_base: u32,
) -> Result<DflashVerifyPlan, String> {
    let b = draft_tokens.len();
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;

    if !dflash_verify_hip_graph_route_ok(gpu, target, verify_scratch, b, true) {
        return Err(
            "plan_dflash_verify: this window does not take the HIP-graph verify body \
             (env gate, embedding format, pbs eligibility or IU4 scope); the emitted twin \
             would not be the same route"
                .into(),
        );
    }
    let moe_env = hipfire_config::developer_var("HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD").ok();
    let head_dtype = target.weights.output.gpu_dtype;
    if !dflash_batched_lm_head_supported(head_dtype) {
        return Err(format!(
            "plan_dflash_verify: lm_head dtype {head_dtype:?} has no batched head \
             (per-row GEMV + D2H is not plannable)"
        ));
    }
    if dflash_moe_verify_graph_lmhead_eligible(
        target.config.num_experts,
        false,
        false,
        moe_env.as_deref(),
    ) {
        return Err(
            "plan_dflash_verify: the MoE extended verify graph (lm_head+argmax captured) is \
             not planned; dense chain verify only"
                .into(),
        );
    }

    let fusion = dflash_verify_fusion_ctx(gpu, target, false, b, required_tokens);
    let final_hidden = verify_scratch.final_hidden.sub_offset(0, b * dim);
    let math = verify_dense_math(gpu);

    let mut sink = PlanSink::default();
    let mut declared: Vec<(usize, Vec<DeclaredWord>)> = Vec::new();

    // ── Phase 1: the captured forward, graph-body route policy. ───────────
    let prior_capture = gpu.graphs.capture_mode;
    let prior_verify_capturing = gpu.graphs.verify.capturing;
    gpu.graphs.capture_mode = true;
    gpu.graphs.verify.capturing = Some(b);
    let forward_result = gpu.with_plan_sink(&mut sink, |g| {
        g.plan_set_label("verify.forward");
        qwen35::forward_prefill_batch_single_chunk_captured_opts(
            g,
            &target.weights,
            &target.config,
            draft_tokens,
            start_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            pbs,
            Some(hidden_rb),
            Some(&final_hidden),
            gdn_tape,
            None,
            false, // DFlash computes all verify logits from final_hidden
            fusion,
        )
    });
    gpu.graphs.capture_mode = prior_capture;
    gpu.graphs.verify.capturing = prior_verify_capturing;
    declared.extend(take_declared_words());
    forward_result.map_err(|e| plan_err("verify forward", &sink, e))?;
    let forward_end = sink.ops.len();

    // ── Phase 2: hidden-ring commit (GPU emit only; eager route). ─────────
    let fused = gpu
        .with_plan_sink(&mut sink, |g| {
            g.plan_set_label("verify.hidden_commit");
            hidden_rb.emit_commit_staging(g, b)
        })
        .map_err(|e| plan_err("verify hidden commit", &sink, e))?;
    declared.extend(take_declared_words());
    if !fused {
        return Err(
            "plan_dflash_verify: the hidden commit is not the fused commit5 kernel (arch \
             not admitted or HIPFIRE hidden-scatter fusion off); the copy-loop commit bakes \
             ring_head into copy addresses, which no declared patch kind expresses"
                .into(),
        );
    }
    let commit_end = sink.ops.len();

    // ── Phase 3: lm_head + argmax (eager route). ──────────────────────────
    gpu.with_plan_sink(&mut sink, |g| {
        g.plan_set_label("verify.lm_head_argmax");
        dflash_enqueue_verify_lm_head_argmax(
            g,
            &target.weights.output,
            &final_hidden,
            verify_scratch,
            b,
            vocab,
            math,
        )
    })
    .map_err(|e| plan_err("verify head", &sink, e))?;
    declared.extend(take_declared_words());
    let head_end = sink.ops.len();

    // ── Declared patch sites (design A §3). ───────────────────────────────
    let ring_head = hidden_rb.head;
    let max_head = hidden_rb.max_positions.saturating_sub(1) as u64;
    let mut ordinal = 0usize;
    let mut commit_kernels = 0usize;
    for (index, op) in sink.ops.iter_mut().enumerate() {
        let PlannedOp::Kernel(kernel) = op else {
            continue;
        };
        if kernel.symbol == DFLASH_HIDDEN_COMMIT5 {
            if !(forward_end..commit_end).contains(&index) {
                return Err(format!(
                    "plan_dflash_verify: {DFLASH_HIDDEN_COMMIT5} planned outside the commit phase (op {index})"
                ));
            }
            let at = COMMIT5_HEAD_OFFSET as usize;
            let planned_head = kernel
                .kernarg
                .get(at..at + 4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize);
            if planned_head != Some(ring_head) {
                return Err(format!(
                    "plan_dflash_verify: commit5 kernarg head word {planned_head:?} != ring head {ring_head}"
                ));
            }
            kernel.patches.push(PatchKind::Scalar32 {
                field: ScalarField {
                    node: ordinal,
                    offset: COMMIT5_HEAD_OFFSET,
                },
                expr: Expr::input("ring_head"),
                domain: 0..=max_head,
            });
            commit_kernels += 1;
        }
        ordinal += 1;
    }
    if commit_kernels != 1 {
        return Err(format!(
            "plan_dflash_verify: expected exactly one {DFLASH_HIDDEN_COMMIT5}, found {commit_kernels}"
        ));
    }

    let delta = hidden_rb.commit_delta(b);
    let host_delta = WindowHostDelta {
        ring_head: Some(Transition {
            from: delta.head_from as u64,
            to: delta.head_to as u64,
        }),
        ring_written: Some(Transition {
            from: delta.written_from as u64,
            to: delta.written_to as u64,
        }),
        ..WindowHostDelta::default()
    };

    Ok(DflashVerifyPlan {
        ops: std::mem::take(&mut sink.ops),
        forward: 0..forward_end,
        commit: forward_end..commit_end,
        head: commit_end..head_end,
        host_delta,
        gdn_frame_base: frame_base,
        gdn_frames_reserved: 0,
        declared_words: declared,
        rows: b,
        start_pos,
        ring_head,
    })
}

/// Compare two plans authored at two different window shapes (different
/// `start_pos` and/or ring head, same ring/KV capacity). `Err` when the
/// topology differs (op kinds, symbols, grids, LDS, copy/fill extents or
/// addresses); otherwise every differing kernarg dword that is **not** a
/// declared `Scalar32` patch site. An empty result proves the patch table
/// complete for these two shapes.
pub fn undeclared_varying_words(
    a: &DflashVerifyPlan,
    b: &DflashVerifyPlan,
) -> Result<Vec<KernargDiff>, String> {
    let diffs = diff_plans(&a.ops, &b.ops)?;
    let mut declared: HashSet<(usize, u32)> = HashSet::new();
    for kernel in a.kernels().chain(b.kernels()) {
        for patch in &kernel.patches {
            if let PatchKind::Scalar32 { field, .. } = patch {
                declared.insert((field.node, field.offset));
            }
        }
    }
    Ok(diffs
        .into_iter()
        .filter(|d| !declared.contains(&(d.kernel, d.offset as u32)))
        .collect())
}
