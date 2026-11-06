// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched draft forward over several lanes' blocks (the continuous-batching
//! store's DFlash lanes). See [`draft_forward_lanes`].

use super::{
    draft_dtod_at, draft_ffn_layer, gemm_dispatch, ordered_draft, ring_segments, upload_slice_i32,
    DflashConfig, DflashLayerWeights, DflashScratch, DflashWeights, DraftCtxMode,
    DRAFT_LANE_WS_ROWS,
};
use crate::llama::{rotate_x_mq_batched_for, WeightTensor};
use hip_bridge::HipResult;
use rdna_compute::draft_lane_conv::{DraftConvLane, DRAFT_CONV_LANES_MAX};
use rdna_compute::draft_lane_rows::{DraftRowSpan, DRAFT_ROW_SPANS_MAX};
use rdna_compute::{DType, Gpu, GpuTensor};

/// `HIPFIRE_CB_DRAFT_LANE_CONV` (developer var; `0` is the kill switch):
/// whether the DFlash2 prepare/finish convolutions of the batched lane draft
/// run as one lane-batched launch per stage instead of one launch per lane
/// (`Gpu::dynamic_causal_conv_f32_lanes`; exact gfx1201, never during graph
/// capture, replay recording or G0 observation). Every output byte is
/// identical either way; off restores the per-lane launches.
static CB_DRAFT_LANE_CONV: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_DRAFT_LANE_CONV").map_or(true, |v| v.trim() != "0")
});

/// Is the lane-batched draft convolution enabled for this process?
fn draft_lane_conv_enabled() -> bool {
    *CB_DRAFT_LANE_CONV
}

/// `HIPFIRE_CB_DRAFT_LANE_CTX` (developer var; `0` is the kill switch):
/// whether the per-lane `fc` projection and the per-layer context K/V fill of
/// the batched lane draft run as packed, lane-batched stages (gather, grouped
/// rotate + GEMM per singleton arithmetic class, one norm, one RoPE, one
/// scatter) instead of one GEMM chain per lane segment. Exact gfx1201, MQ4G256V2
/// weights, `ordered_draft()`, never during graph capture, replay recording or
/// G0 observation. Every output byte is identical either way; `0` restores the
/// per-lane loops.
static CB_DRAFT_LANE_CTX: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_DRAFT_LANE_CTX").map_or(true, |v| v.trim() != "0")
});

/// Is the lane-batched ctx/fc path enabled for this process?
pub(super) fn draft_lane_ctx_enabled() -> bool {
    *CB_DRAFT_LANE_CTX
}

/// One lane of [`draft_forward_lanes`]: its own draft scratch (context rings,
/// projection watermarks) and the singleton `draft_dflash_forward_rank`
/// geometry (`ctx_slice = None`).
pub struct DraftLane<'a> {
    pub scratch: &'a mut DflashScratch,
    /// Block rows of this lane (`<= scratch.max_block_size`).
    pub b: usize,
    /// Context rows `l` (`min(thlog.abs_positions().len(), position)`).
    pub ctx_len: usize,
    /// `[b]` absolute query positions.
    pub positions_q: &'a [i32],
    /// `[l + b]` absolute key positions.
    pub positions_k: &'a [i32],
}

/// Per-lane geometry of [`draft_forward_lanes`] (the singleton's per-call
/// locals of `draft_forward_opts`).
struct LaneGeo {
    l: usize,
    b: usize,
    /// First row of this lane in the shared row scratch.
    off: usize,
    swa_w: usize,
    full_w: usize,
    windowed: bool,
    all_sliding_windowed: bool,
    pos_base: usize,
    /// `thlog.proj_cached_rows()` at entry.
    cached_rows: usize,
}

/// The key positions a lane uploads: from `l - w_full` on in windowed mode
/// (the windowed `positions_k` holds the full-window span), the whole
/// `[l + b]` array in legacy mode.
fn lane_key_positions<'p>(ls: &DflashScratch, positions_k: &'p [i32], l: usize) -> &'p [i32] {
    match ls.ctx_mode {
        DraftCtxMode::Legacy => positions_k,
        DraftCtxMode::Windowed { w_full, .. } => &positions_k[l.saturating_sub(w_full)..],
    }
}

/// Context fill of one layer of one lane: the incremental K/V projection of
/// the rows past the layer's watermark into the lane's own K/V ring. Verbatim
/// the context block of `draft_forward_opts` (`q_dummy` is the ignored query
/// operand of the K-only RoPE). The FWHT rotation temporaries of the two
/// GEMMs come from `shared` (see [`draft_forward_lanes`]), never from the
/// lane: a lane scratch may be allocated without them.
#[allow(clippy::too_many_arguments)]
fn lane_ctx_fill(
    gpu: &mut Gpu,
    cfg: &DflashConfig,
    layer: &DflashLayerWeights,
    ls: &DflashScratch,
    shared: &DflashScratch,
    q_dummy: &GpuTensor,
    li: usize,
    g: &LaneGeo,
) -> HipResult<()> {
    let h = cfg.hidden;
    let kvd = cfg.kv_dim();
    let hd = cfg.head_dim;
    let eps = cfg.norm_eps;
    let theta = cfg.rope_theta;
    let l = g.l;
    let swa_w = g.swa_w;
    let lay = ctx_layout(cfg, ls, g, li);
    let layer_w = lay.layer_w;
    let mut row = lay.fill_start;
    while row < l {
        let (step, p_slot, c_slot) = ctx_step(row, l, swa_w, layer_w);
        let thp_slice = ls.target_hidden_proj.sub_offset(p_slot * h, step * h);
        let k_slot = lay.k_cache.sub_offset(c_slot * kvd, step * kvd);
        let v_slot = lay.v_cache.sub_offset(c_slot * kvd, step * kvd);
        gemm_dispatch(
            gpu,
            &thp_slice,
            &layer.wk,
            &k_slot,
            step,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;
        gemm_dispatch(
            gpu,
            &thp_slice,
            &layer.wv,
            &v_slot,
            step,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;
        gpu.rmsnorm_batched(
            &k_slot,
            &layer.k_norm,
            &k_slot,
            step * cfg.n_kv_heads,
            hd,
            eps,
        )?;
        let fill_positions = ls.positions_k.sub_offset(row - g.pos_base, step);
        gpu.rope_batched_f32(
            q_dummy,
            &k_slot,
            &fill_positions,
            0,
            cfg.n_kv_heads,
            hd,
            theta,
            step,
        )?;
        row += step;
    }
    Ok(())
}

/// Layer- and lane-specific geometry of a context fill: the K/V ring the layer
/// fills, its modulus, and the first row past every watermark. Shared by the
/// per-lane fill ([`lane_ctx_fill`]) and the lane-batched planner
/// ([`plan_ctx`]) so both split the rows identically.
struct CtxLayout<'a> {
    k_cache: &'a GpuTensor,
    v_cache: &'a GpuTensor,
    layer_w: usize,
    fill_start: usize,
}

fn ctx_layout<'a>(
    cfg: &DflashConfig,
    ls: &'a DflashScratch,
    g: &LaneGeo,
    li: usize,
) -> CtxLayout<'a> {
    let l = g.l;
    let (swa_w, full_w) = (g.swa_w, g.full_w);
    let is_last_layer = li + 1 == cfg.n_layers;
    let is_full_layer = !g.all_sliding_windowed && is_last_layer;
    let layer_w = if is_full_layer { full_w } else { swa_w };
    let (k_cache, v_cache) = if is_full_layer && g.windowed {
        (
            ls.k_full_cached.as_ref().expect("windowed k_full"),
            ls.v_full_cached.as_ref().expect("windowed v_full"),
        )
    } else {
        (&ls.k_ctx_cached[li], &ls.v_ctx_cached[li])
    };
    let span_start = l.saturating_sub(layer_w);
    let wm = if is_full_layer {
        ls.thlog.full_cached_rows()
    } else {
        g.cached_rows
    };
    let fill_start = wm.max(span_start).max(l.saturating_sub(swa_w));
    CtxLayout {
        k_cache,
        v_cache,
        layer_w,
        fill_start,
    }
}

/// One step of the context-fill loop at `row`: `(rows, projected-hidden ring
/// slot, layer K/V ring slot)`, split at both ring boundaries.
fn ctx_step(row: usize, l: usize, swa_w: usize, layer_w: usize) -> (usize, usize, usize) {
    let step = (swa_w.saturating_sub(if swa_w == usize::MAX { 0 } else { row % swa_w }))
        .min(layer_w.saturating_sub(if layer_w == usize::MAX {
            0
        } else {
            row % layer_w
        }))
        .min(l - row);
    let p_slot = if swa_w == usize::MAX {
        row
    } else {
        row % swa_w
    };
    let c_slot = if layer_w == usize::MAX {
        row
    } else {
        row % layer_w
    };
    (step, p_slot, c_slot)
}

/// Per-lane `fc` projection + hidden norm of the rows past the lane's
/// projection watermark into its `target_hidden_proj` ring: the original
/// per-segment route (`gemm_dispatch` + norm). `ls` is the lane scratch; the
/// FWHT temporaries come from `shared`.
#[allow(clippy::too_many_arguments)]
fn lane_fc_fill(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    ls: &DflashScratch,
    shared: &DflashScratch,
    l: usize,
    swa_w: usize,
    cached_rows: usize,
) -> HipResult<()> {
    let h = cfg.hidden;
    let fw = cfg.feature_width();
    let eps = cfg.norm_eps;
    let fc_start = cached_rows.max(l.saturating_sub(swa_w));
    if l.saturating_sub(fc_start) > 0 {
        for (_row0, slot0, len) in ring_segments(fc_start, l, swa_w) {
            let th_slice = ls.target_hidden.sub_offset(slot0 * fw, len * fw);
            let thp_slice = ls.target_hidden_proj.sub_offset(slot0 * h, len * h);
            // Per-lane fc projection (`len` rows): singleton route, outside
            // the batched 64-row scope.
            let scoped = gpu.draft_rows64_enter(false);
            let fc = gemm_dispatch(
                gpu,
                &th_slice,
                &weights.fc,
                &thp_slice,
                len,
                shared.mq_x_rot.as_ref(),
                shared.mq_x_rot_f16.as_ref(),
            );
            gpu.draft_rows64_exit(scoped);
            fc?;
            gpu.rmsnorm_batched(&thp_slice, &weights.hidden_norm, &thp_slice, len, h, eps)?;
        }
    }
    Ok(())
}

// ─── Lane-batched fc / context fill (`HIPFIRE_CB_DRAFT_LANE_CTX`) ─────────
//
// Every ORIGINAL segment (an fc `ring_segments` piece, or one step of the
// ctx-fill loop) keeps the arithmetic class today's per-lane code gives it:
//   * 1 row  -> `gemm_dispatch`'s `n == 1` arm = `weight_gemv`: F32 FWHT
//     rotation + the prerotated MQ4G256V2 GEMV (multirow R2 on this arch).
//   * >=2 rows -> FWHT rotation + F16 convert + overwrite WMMA.
// The two classes are never pooled together and never go through
// `gemm_dispatch` on pooled rows: the WMMA pool runs the overwrite route
// directly in chunks of 2..=63 rows (never a 1-row tail), the scalar pool
// runs the R2 lane-batched GEMV (one z-lane per original 1-row segment).
// Anything not exactly representable falls back, before any write, to the
// original per-lane loops.

/// New rows one lane may contribute to a batched stage (an accepted block of
/// up to `b + 1` committed rows).
const LANE_CTX_MAX_LANE_ROWS: usize = 9;
/// Largest WMMA overwrite chunk (`gemm_mq4g256v2_overwrite_gfx12_dflash`
/// admits `2..=63` rows outside the 64-row scope).
const WMMA_MAX_CHUNK: usize = 63;
/// Most 1-row segments per `gemv_mq4g256v2_multirow_r2_lanes` launch.
const R2_LANES_BATCH_MAX: usize = 8;

/// Device address of element `off` of an F32 tensor (the `DraftRowSpan`
/// pointer fields are byte addresses already advanced to the segment start).
fn elem_ptr(t: &GpuTensor, off: usize) -> u64 {
    t.buf.as_ptr() as u64 + (off * 4) as u64
}

/// One original segment: source/destination row addresses and its row count.
/// `packed` is its first row in the packed workspace (assigned by
/// [`RowPlan::finish`]).
#[derive(Clone, Copy)]
struct RowSeg {
    src: u64,
    dst0: u64,
    dst1: u64,
    positions: u64,
    n: usize,
    packed: usize,
}

/// The original segments of one stage, partitioned into the WMMA pool
/// (`n >= 2`, packed rows `[0, n_w)`) and the scalar pool (`n == 1`, packed
/// rows `[n_w, n_w + n_s)`). Fixed storage: no allocation.
struct RowPlan {
    segs: [RowSeg; DRAFT_ROW_SPANS_MAX],
    ns: usize,
    n_w: usize,
    n_s: usize,
    lane_rows: usize,
}

impl RowPlan {
    fn new() -> Self {
        RowPlan {
            segs: [RowSeg {
                src: 0,
                dst0: 0,
                dst1: 0,
                positions: 0,
                n: 0,
                packed: 0,
            }; DRAFT_ROW_SPANS_MAX],
            ns: 0,
            n_w: 0,
            n_s: 0,
            lane_rows: 0,
        }
    }

    fn begin_lane(&mut self) {
        self.lane_rows = 0;
    }

    /// Append an original segment of the current lane; `false` = over the
    /// per-lane row cap or the span table.
    fn push(&mut self, seg: RowSeg) -> bool {
        debug_assert!(seg.n > 0);
        self.lane_rows += seg.n;
        if seg.n == 0 || self.ns == DRAFT_ROW_SPANS_MAX || self.lane_rows > LANE_CTX_MAX_LANE_ROWS {
            return false;
        }
        self.segs[self.ns] = seg;
        self.ns += 1;
        true
    }

    /// Classify (by ORIGINAL segment length) and assign packed rows. `false`
    /// when the packed rows exceed `max_rows` (the workspace capacity).
    fn finish(&mut self, max_rows: usize) -> bool {
        self.n_w = self.segs[..self.ns]
            .iter()
            .filter(|s| s.n >= 2)
            .map(|s| s.n)
            .sum();
        self.n_s = self.segs[..self.ns].iter().filter(|s| s.n == 1).count();
        if self.n_w + self.n_s > max_rows {
            return false;
        }
        let (mut w_off, mut s_off) = (0usize, self.n_w);
        for s in &mut self.segs[..self.ns] {
            if s.n >= 2 {
                s.packed = w_off;
                w_off += s.n;
            } else {
                s.packed = s_off;
                s_off += 1;
            }
        }
        true
    }

    fn total(&self) -> usize {
        self.n_w + self.n_s
    }
}

fn empty_span() -> DraftRowSpan {
    DraftRowSpan {
        src: 0,
        dst0: 0,
        dst1: 0,
        positions: 0,
        packed_row0: 0,
        rows: 0,
    }
}

/// Gather table: `src` rows (and `positions`) of every segment into its packed rows.
fn gather_spans(plan: &RowPlan) -> [DraftRowSpan; DRAFT_ROW_SPANS_MAX] {
    let mut spans: [DraftRowSpan; DRAFT_ROW_SPANS_MAX] = std::array::from_fn(|_| empty_span());
    for (sp, s) in spans.iter_mut().zip(&plan.segs[..plan.ns]) {
        sp.src = s.src;
        sp.positions = s.positions;
        sp.packed_row0 = s.packed as i32;
        sp.rows = s.n as i32;
    }
    spans
}

/// Scatter table: packed rows of every segment back to its `dst0` (and `dst1`).
fn scatter_spans(plan: &RowPlan) -> [DraftRowSpan; DRAFT_ROW_SPANS_MAX] {
    let mut spans: [DraftRowSpan; DRAFT_ROW_SPANS_MAX] = std::array::from_fn(|_| empty_span());
    for (sp, s) in spans.iter_mut().zip(&plan.segs[..plan.ns]) {
        sp.dst0 = s.dst0;
        sp.dst1 = s.dst1;
        sp.packed_row0 = s.packed as i32;
        sp.rows = s.n as i32;
    }
    spans
}

/// Size of the next WMMA chunk of `rem >= 2` pooled rows: `2..=63`, never
/// leaving a 1-row tail (64 -> 62 + 2, 72 -> 63 + 9).
fn wmma_chunk(rem: usize) -> usize {
    let n = rem.min(WMMA_MAX_CHUNK);
    if rem - n == 1 {
        n - 1
    } else {
        n
    }
}

/// Process-level admission of the batched ctx/fc stages: switch on, exact
/// gfx1201, no graph capture / replay recording / G0 observation, no
/// calibration tap armed, `ordered_draft()`, and the shared workspace plus the
/// FWHT rotation scratch present. Per-stage (per-weight, per-plan) checks are
/// in [`pool_admitted`].
fn lane_rows_admitted(gpu: &Gpu, shared: &DflashScratch, ordered: bool) -> bool {
    draft_lane_ctx_enabled()
        && ordered
        && gpu.arch == "gfx1201"
        && !gpu.graphs.capture_mode
        && !gpu.replay.is_recording()
        && !gpu.replay.is_g0_observing()
        && gpu.active_capture.is_none()
        && shared.lane_ws.is_some()
        && shared.mq_x_rot.is_some()
}

/// May `plan` run through the pooled route of weight `w` and still produce the
/// bytes of the original per-segment route? Decided with no device effect.
fn pool_admitted(gpu: &Gpu, shared: &DflashScratch, w: &WeightTensor, plan: &RowPlan) -> bool {
    let Some(rot) = shared.mq_x_rot.as_ref() else {
        return false;
    };
    if w.gpu_dtype != DType::MQ4G256V2 || w.k == 0 || rot.shape[0] / w.k < DRAFT_LANE_WS_ROWS {
        return false;
    }
    // The original 1-row segments run `weight_gemv`'s R2 GEMV only where the
    // arch's default GEMV row count is 2.
    if plan.n_s > 0 && gpu.arch_caps.gemv_rows_default() != 2 {
        return false;
    }
    // Every original >=2-row segment AND every pooled chunk must be admitted
    // by the overwrite route (else the original route is not the overwrite
    // WMMA either).
    if !plan.segs[..plan.ns]
        .iter()
        .all(|s| s.n < 2 || gpu.draft_overwrite_gfx12_admitted(w.k, s.n))
    {
        return false;
    }
    let mut rem = plan.n_w;
    while rem > 0 {
        let n = wmma_chunk(rem);
        if !gpu.draft_overwrite_gfx12_admitted(w.k, n) {
            return false;
        }
        rem -= n;
    }
    true
}

/// `y[r] = W . x[r]` for every packed row of `plan` (`x` `[rows, w.k]`, `y`
/// `[rows, w.m]`), each pool with its original arithmetic. Only called after
/// [`pool_admitted`]; rotation is per weight (AWQ-aware), never shared
/// between weights.
fn pool_gemm(
    gpu: &mut Gpu,
    shared: &DflashScratch,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    plan: &RowPlan,
) -> HipResult<()> {
    let rot_scratch = shared
        .mq_x_rot
        .as_ref()
        .expect("lane ctx admitted without mq_x_rot");
    let (k, m) = (w.k, w.m);
    let mut r0 = 0usize;
    while r0 < plan.n_w {
        let n = wmma_chunk(plan.n_w - r0);
        let x_chunk = x.sub_offset(r0 * k, n * k);
        let y_chunk = y.sub_offset(r0 * m, n * m);
        let rot = rot_scratch.sub_offset(0, n * k);
        rotate_x_mq_batched_for(gpu, w, &x_chunk, &rot, k, n)?;
        if !gpu.gemm_mq4g256v2_overwrite_gfx12_dflash(&w.buf, &rot, &y_chunk, m, k, n)? {
            return Err(hip_bridge::HipError::new(
                0,
                "lane ctx/fc: overwrite WMMA refused a chunk its admission predicate accepted",
            ));
        }
        r0 += n;
    }
    if plan.n_s > 0 {
        let xs = x.sub_offset(plan.n_w * k, plan.n_s * k);
        let rot = rot_scratch.sub_offset(0, plan.n_s * k);
        rotate_x_mq_batched_for(gpu, w, &xs, &rot, k, plan.n_s)?;
        let mut r = 0usize;
        while r < plan.n_s {
            let c = (plan.n_s - r).min(R2_LANES_BATCH_MAX);
            let rot_c = rot.sub_offset(r * k, c * k);
            let y_c = y.sub_offset((plan.n_w + r) * m, c * m);
            gpu.gemv_mq4g256v2_multirow_r2_lanes(&w.buf, &rot_c, &y_c, m, k, c)?;
            r += c;
        }
    }
    Ok(())
}

/// Plan of the `fc` stage over all lanes (the original `ring_segments` pieces
/// past each lane's projection watermark), or `None` to keep the per-lane
/// loop. `Some` with no segments = nothing to project (the loop would also
/// launch nothing).
fn plan_fc(
    gpu: &Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    shared: &DflashScratch,
    lanes: &[DraftLane<'_>],
) -> Option<RowPlan> {
    let ws = shared.lane_ws.as_ref()?;
    let (h, fw) = (cfg.hidden, cfg.feature_width());
    if weights.fc.k != fw || weights.fc.m != h || fw > ws.input_width {
        return None;
    }
    let mut plan = RowPlan::new();
    for lane in lanes {
        let ls = &*lane.scratch;
        let l = lane.ctx_len;
        let swa_w = match ls.ctx_mode {
            DraftCtxMode::Legacy => usize::MAX,
            DraftCtxMode::Windowed { w, .. } => w,
        };
        let fc_start = ls.thlog.proj_cached_rows().max(l.saturating_sub(swa_w));
        if l.saturating_sub(fc_start) == 0 {
            continue;
        }
        if l - fc_start > LANE_CTX_MAX_LANE_ROWS {
            return None;
        }
        plan.begin_lane();
        for (_row0, slot0, len) in ring_segments(fc_start, l, swa_w) {
            let seg = RowSeg {
                src: elem_ptr(&ls.target_hidden, slot0 * fw),
                dst0: elem_ptr(&ls.target_hidden_proj, slot0 * h),
                dst1: 0,
                positions: 0,
                n: len,
                packed: 0,
            };
            if !plan.push(seg) {
                return None;
            }
        }
    }
    if !plan.finish(ws.rows) || !pool_admitted(gpu, shared, &weights.fc, &plan) {
        return None;
    }
    Some(plan)
}

/// Lane-batched `fc`: gather every lane's `target_hidden` segments, grouped
/// rotate + GEMM per arithmetic class, ONE hidden norm over all packed rows,
/// ONE scatter into the lanes' `target_hidden_proj` rings. No lane state is
/// written before the scatter.
fn fc_batched(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    shared: &DflashScratch,
    plan: &RowPlan,
) -> HipResult<()> {
    let ws = shared.lane_ws.as_ref().expect("lane workspace");
    let h = cfg.hidden;
    let k = cfg.feature_width();
    let gather = gather_spans(plan);
    gpu.draft_rows_gather_f32(&ws.input, None, &gather[..plan.ns], k)?;
    pool_gemm(gpu, shared, &weights.fc, &ws.input, &ws.fc_out, plan)?;
    gpu.rmsnorm_batched(
        &ws.fc_out,
        &weights.hidden_norm,
        &ws.fc_out,
        plan.total(),
        h,
        cfg.norm_eps,
    )?;
    let scatter = scatter_spans(plan);
    gpu.draft_rows_scatter_f32(&ws.fc_out, None, &scatter[..plan.ns], h)
}

/// Plan of layer `li`'s context fill over all lanes (the original watermark /
/// two-modulus steps), or `None` to keep the per-lane loop.
fn plan_ctx(
    gpu: &Gpu,
    cfg: &DflashConfig,
    layer: &DflashLayerWeights,
    shared: &DflashScratch,
    lanes: &[DraftLane<'_>],
    geos: &[LaneGeo],
    li: usize,
) -> Option<RowPlan> {
    let ws = shared.lane_ws.as_ref()?;
    let (h, kvd) = (cfg.hidden, cfg.kv_dim());
    if layer.wk.k != h
        || layer.wk.m != kvd
        || layer.wv.k != h
        || layer.wv.m != kvd
        || h > ws.input_width
    {
        return None;
    }
    let mut plan = RowPlan::new();
    for (lane, g) in lanes.iter().zip(geos) {
        let ls = &*lane.scratch;
        let lay = ctx_layout(cfg, ls, g, li);
        if g.l.saturating_sub(lay.fill_start) == 0 {
            continue;
        }
        if g.l - lay.fill_start > LANE_CTX_MAX_LANE_ROWS {
            return None;
        }
        plan.begin_lane();
        let mut row = lay.fill_start;
        while row < g.l {
            let (step, p_slot, c_slot) = ctx_step(row, g.l, g.swa_w, lay.layer_w);
            let seg = RowSeg {
                src: elem_ptr(&ls.target_hidden_proj, p_slot * h),
                dst0: elem_ptr(lay.k_cache, c_slot * kvd),
                dst1: elem_ptr(lay.v_cache, c_slot * kvd),
                positions: elem_ptr(&ls.positions_k, row - g.pos_base),
                n: step,
                packed: 0,
            };
            if !plan.push(seg) {
                return None;
            }
            row += step;
        }
    }
    if !plan.finish(ws.rows)
        || !pool_admitted(gpu, shared, &layer.wk, &plan)
        || !pool_admitted(gpu, shared, &layer.wv, &plan)
    {
        return None;
    }
    Some(plan)
}

/// Lane-batched context fill of one layer: gather projected-hidden rows and
/// key positions, grouped wk / wv (each with its own rotation), ONE K norm,
/// ONE K-only RoPE over the packed rows, ONE K+V scatter into the lanes' rings.
/// No ring is written before the scatter.
fn ctx_fill_batched(
    gpu: &mut Gpu,
    cfg: &DflashConfig,
    layer: &DflashLayerWeights,
    shared: &DflashScratch,
    plan: &RowPlan,
) -> HipResult<()> {
    if plan.ns == 0 {
        return Ok(());
    }
    let ws = shared.lane_ws.as_ref().expect("lane workspace");
    let (h, kvd, hd) = (cfg.hidden, cfg.kv_dim(), cfg.head_dim);
    let total = plan.total();
    let gather = gather_spans(plan);
    gpu.draft_rows_gather_f32(&ws.input, Some(&ws.positions), &gather[..plan.ns], h)?;
    pool_gemm(gpu, shared, &layer.wk, &ws.input, &ws.k_out, plan)?;
    pool_gemm(gpu, shared, &layer.wv, &ws.input, &ws.v_out, plan)?;
    gpu.rmsnorm_batched(
        &ws.k_out,
        &layer.k_norm,
        &ws.k_out,
        total * cfg.n_kv_heads,
        hd,
        cfg.norm_eps,
    )?;
    gpu.rope_batched_f32(
        &shared.q,
        &ws.k_out,
        &ws.positions,
        0,
        cfg.n_kv_heads,
        hd,
        cfg.rope_theta,
        total,
    )?;
    let scatter = scatter_spans(plan);
    gpu.draft_rows_scatter_f32(&ws.k_out, Some(&ws.v_out), &scatter[..plan.ns], kvd)
}

/// Context-K/V concat and attention of one layer of one lane: the lane's own
/// K/V ring plus its noise rows (taken from the shared row scratch) into the
/// lane's concat buffers, then attention of its query rows into the shared
/// attention output. Verbatim the concat/attention block of
/// `draft_forward_opts`.
fn lane_attention(
    gpu: &mut Gpu,
    cfg: &DflashConfig,
    ls: &DflashScratch,
    shared: &DflashScratch,
    li: usize,
    g: &LaneGeo,
    ordered: bool,
) -> HipResult<()> {
    let kvd = cfg.kv_dim();
    let qd = cfg.q_dim();
    let hd = cfg.head_dim;
    let (b, l) = (g.b, g.l);
    let (swa_w, full_w) = (g.swa_w, g.full_w);
    let is_last_layer = li + 1 == cfg.n_layers;
    let is_full_layer = !g.all_sliding_windowed && is_last_layer;
    let layer_w = if is_full_layer { full_w } else { swa_w };
    let (k_cache_layer, v_cache_layer) = if is_full_layer && g.windowed {
        (
            ls.k_full_cached.as_ref().expect("windowed k_full"),
            ls.v_full_cached.as_ref().expect("windowed v_full"),
        )
    } else {
        (&ls.k_ctx_cached[li], &ls.v_ctx_cached[li])
    };
    let (k_cat_l, v_cat_l) = if is_full_layer && g.windowed {
        (
            ls.k_cat_full.as_ref().expect("windowed k_cat_full"),
            ls.v_cat_full.as_ref().expect("windowed v_cat_full"),
        )
    } else {
        (&ls.k_cat, &ls.v_cat)
    };
    let span_start = l.saturating_sub(layer_w);
    let span = l - span_start;
    let noise_bytes = (b * kvd) * 4;
    let mut cat_off = 0usize;
    for (_row0, slot0, len) in ring_segments(span_start, l, layer_w) {
        let seg_bytes = len * kvd * 4;
        draft_dtod_at(
            gpu,
            ordered,
            &k_cat_l.buf,
            cat_off,
            &k_cache_layer.buf,
            slot0 * kvd * 4,
            seg_bytes,
        )?;
        draft_dtod_at(
            gpu,
            ordered,
            &v_cat_l.buf,
            cat_off,
            &v_cache_layer.buf,
            slot0 * kvd * 4,
            seg_bytes,
        )?;
        cat_off += seg_bytes;
    }
    draft_dtod_at(
        gpu,
        ordered,
        &k_cat_l.buf,
        cat_off,
        &shared.k_noise.buf,
        g.off * kvd * 4,
        noise_bytes,
    )?;
    draft_dtod_at(
        gpu,
        ordered,
        &v_cat_l.buf,
        cat_off,
        &shared.v_noise.buf,
        g.off * kvd * 4,
        noise_bytes,
    )?;
    let q_view = shared.q.sub_offset(g.off * qd, b * qd);
    let out_view = shared.attn_out.sub_offset(g.off * qd, b * qd);
    let is_swa_layer = g.windowed && (g.all_sliding_windowed || !is_full_layer);
    if is_swa_layer {
        gpu.attention_dflash_sliding_f32(
            &q_view,
            k_cat_l,
            v_cat_l,
            &out_view,
            b,
            span + b,
            cfg.n_heads,
            cfg.n_kv_heads,
            hd,
            span,
            swa_w,
        )?;
    } else {
        use crate::llama::{attention_family, DispatchCtx, FullAttnParams, KernelKey};
        let ctx = DispatchCtx::new(gpu);
        let family = attention_family();
        family
            .run_full_attention(
                &ctx,
                gpu,
                &FullAttnParams {
                    key: KernelKey::AttnFullF32,
                    q: &q_view,
                    k: k_cat_l,
                    v: v_cat_l,
                    out: &out_view,
                    n: b,
                    seq_len: span + b,
                    n_heads: cfg.n_heads,
                    n_kv_heads: cfg.n_kv_heads,
                    head_dim: hd,
                },
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }
    Ok(())
}

/// Batched draft forward over several lanes' blocks (one forward of the
/// draft model over all lanes' rows). Row-local stages — every draft weight
/// GEMM (qkv, wo, FFN, conv kernel projections), the norms, the query/noise-K
/// RoPE (per-row positions) — run ONCE over the concatenated rows in
/// `shared`; lane-state stages — the context fill into each lane's own K/V
/// ring, the K/V concat, the attention, and the DFlash2 convolutions (causal
/// across the rows of ONE block) — run per lane on that lane's rows.
///
/// `shared.x[off_i .. off_i + b_i)` MUST hold lane `i`'s noise embeddings on
/// entry (rows packed in lane order); the final-normed hidden rows of every
/// lane are left in `shared.x` at the same offsets. Each lane's draft scratch
/// (K/V rings, `target_hidden_proj`, thlog watermarks) ends bit-identical to
/// a singleton `draft_forward_opts` of that lane iff every shared GEMM is
/// row-count independent for the row counts used (`sum b_i`); the caller owns
/// that proof (exact gfx1201 MQ4 v2 below 64 rows). Requires the non-fused
/// (non-gfx1100) draft route, `ctx_slice = None` geometry and no FFN graph.
///
/// FWHT rotation scratch: the per-lane GEMMs (the `fc` projection and the
/// context-fill `wk`/`wv`) rotate their activation into `shared.mq_x_rot`
/// / `shared.mq_x_rot_f16`, NOT into the lane's own buffers (a lane may be
/// allocated without them). This is byte-exact: the rotation buffer is a
/// per-`gemm_dispatch`-call temporary (each call writes the rotated rows,
/// the GEMM reads them, in the same call and on the single stream, and
/// nothing reads it afterwards), and `gemm_dispatch` chunks by the element
/// count of the buffer, which is identical for every `DflashScratch`
/// (`MQ_X_ROT_CHUNK_ROWS × max(inter, q_dim, feature_width)`,
/// independent of block/context size) — so chunking, kernels and row order
/// are unchanged. A lane that does own a rotation buffer must match the
/// shared one in shape (checked), and MQ draft weights with no shared
/// rotation buffer are an error, not a panic.
///
/// Lane-batched ctx/fc (`HIPFIRE_CB_DRAFT_LANE_CTX`, default on): when the
/// shared scratch carries a [`super::DraftLaneWorkspace`], the device is exact
/// gfx1201 and no capture/replay/G0 is active, the `fc` projection and each
/// layer's context fill of all lanes run as packed stages grouped by the
/// arithmetic class of the ORIGINAL segment (1-row: R2 GEMV; >=2 rows:
/// overwrite WMMA, chunks of 2..=63). Eligibility is decided per stage before
/// any write; otherwise the per-lane loops run unchanged.
pub fn draft_forward_lanes(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    shared: &mut DflashScratch,
    lanes: &mut [DraftLane<'_>],
) -> HipResult<()> {
    if lanes.is_empty() {
        return Ok(());
    }
    // Checked before any launch: the per-lane GEMMs borrow the shared
    // rotation scratch (see the doc comment).
    if weights.has_mq && shared.mq_x_rot.is_none() {
        return Err(hip_bridge::HipError::new(
            0,
            "draft_forward_lanes: MQ draft weights need the shared FWHT rotation scratch (shared.mq_x_rot is None)",
        ));
    }
    if let Some(sh) = shared.mq_x_rot.as_ref() {
        if lanes.iter().any(|ln| {
            ln.scratch
                .mq_x_rot
                .as_ref()
                .is_some_and(|t| t.shape != sh.shape)
        }) {
            return Err(hip_bridge::HipError::new(
                0,
                "draft_forward_lanes: a lane's own FWHT rotation scratch differs in size from the shared one",
            ));
        }
    }
    if gpu.draft_collapse_fused_enabled() {
        return Err(hip_bridge::HipError::new(
            0,
            "draft_forward_lanes: the fused gfx1100 draft route is row-count dependent",
        ));
    }
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let eps = cfg.norm_eps;
    let theta = cfg.rope_theta;
    let inter = cfg.intermediate;
    let total: usize = lanes.iter().map(|ln| ln.b).sum();
    assert!(
        total <= shared.max_block_size,
        "lane rows > shared scratch max"
    );

    // ── Per-lane geometry, key positions, context projection ─────────────
    let ordered = ordered_draft();
    // Owned staging (capacity `max_block_size >= total`, never grown).
    shared.lane_positions_q.clear();
    for lane in lanes.iter() {
        shared.lane_positions_q.extend_from_slice(lane.positions_q);
    }
    if ordered {
        // Every host position upload precedes the draft body, so no
        // synchronous H2D drains the queue between the lane fc GEMMs. The fc
        // GEMMs and hidden norms read no positions; every position reader
        // (context RoPE, query/noise RoPE) still runs after its upload.
        for lane in lanes.iter() {
            assert_eq!(
                lane.positions_k.len(),
                lane.ctx_len + lane.b,
                "positions_k size"
            );
            let ls = &*lane.scratch;
            upload_slice_i32(
                gpu,
                &ls.positions_k,
                lane_key_positions(ls, lane.positions_k, lane.ctx_len),
            )?;
        }
        upload_slice_i32(gpu, &shared.positions_q, &shared.lane_positions_q)?;
    }
    // Lane-batched ctx/fc (`HIPFIRE_CB_DRAFT_LANE_CTX`): admission and the fc
    // row plan are decided here, before any lane state is written. `None`
    // keeps the per-lane loops below.
    let lane_ctx_on = lane_rows_admitted(gpu, &*shared, ordered);
    let fc_plan = if lane_ctx_on {
        plan_fc(gpu, weights, cfg, &*shared, &*lanes)
    } else {
        None
    };
    let mut geos: Vec<LaneGeo> = Vec::with_capacity(lanes.len());
    let mut off = 0usize;
    for lane in lanes.iter_mut() {
        let (b, l) = (lane.b, lane.ctx_len);
        let ls = &mut *lane.scratch;
        assert!(b <= ls.max_block_size, "block_size > scratch max");
        assert!(l <= ls.max_ctx_len, "ctx_len > scratch max");
        assert_eq!(lane.positions_q.len(), b, "positions_q size");
        assert_eq!(lane.positions_k.len(), l + b, "positions_k size");
        let (swa_w, full_w) = match ls.ctx_mode {
            DraftCtxMode::Legacy => (usize::MAX, usize::MAX),
            DraftCtxMode::Windowed { w, w_full } => (w, w_full),
        };
        let windowed = !matches!(ls.ctx_mode, DraftCtxMode::Legacy);
        let pos_base = if windowed {
            l.saturating_sub(full_w)
        } else {
            0
        };
        if windowed
            && full_w != swa_w
            && l > swa_w
            && ls.thlog.full_cached_rows() < l.saturating_sub(swa_w)
        {
            static BACKFILL_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !BACKFILL_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[dflash] windowed: last-layer backfill watermark {} < l−W {} — stale \
                     out-of-window K/V (acceptance degraded; output still verify-exact). \
                     Call draft_seed_backfill after cold seeds.",
                    ls.thlog.full_cached_rows(),
                    l.saturating_sub(swa_w)
                );
            }
        }
        if !ordered {
            upload_slice_i32(
                gpu,
                &ls.positions_k,
                lane_key_positions(ls, lane.positions_k, l),
            )?;
        }
        let cached_rows = ls.thlog.proj_cached_rows();
        if fc_plan.is_none() {
            lane_fc_fill(gpu, weights, cfg, &*ls, &*shared, l, swa_w, cached_rows)?;
        }
        geos.push(LaneGeo {
            l,
            b,
            off,
            swa_w,
            full_w,
            windowed,
            all_sliding_windowed: windowed && cfg.all_layers_sliding,
            pos_base,
            cached_rows,
        });
        off += b;
    }
    // Lane-batched fc over every lane's original segments (planned and
    // admitted before the loop above; nothing was written for the lanes whose
    // fc it covers). Positions were uploaded up front (`ordered`).
    if let Some(plan) = &fc_plan {
        if plan.ns > 0 {
            // Same 64-row scope as the per-lane fc: outside it.
            let scoped = gpu.draft_rows64_enter(false);
            let r = fc_batched(gpu, weights, cfg, &*shared, plan);
            gpu.draft_rows64_exit(scoped);
            r?;
        }
    }
    if !ordered {
        upload_slice_i32(gpu, &shared.positions_q, &shared.lane_positions_q)?;
    }

    // ── Per-layer decoder ────────────────────────────────────────────────
    let rows = total;
    // Lane table of the lane-batched convolution, built once from the lane
    // geometry. `None` (kill switch, non-gfx1201, capture/replay/G0, more
    // lanes than the table holds) keeps the per-lane launches below.
    let conv_lanes: Option<Vec<DraftConvLane>> = (draft_lane_conv_enabled()
        && geos.len() <= DRAFT_CONV_LANES_MAX
        && geos.iter().all(|g_| g_.b > 0)
        && gpu.draft_lane_conv_admitted())
    .then(|| {
        geos.iter()
            .map(|g_| DraftConvLane {
                row0: g_.off,
                rows: g_.b,
            })
            .collect()
    });
    let conv_kgs = || {
        let k = cfg.conv_kernel_size.unwrap_or(2);
        let g = cfg.conv_group_size.unwrap_or(16);
        (k, g, h / g, 2 * k * (h / g))
    };
    for li in 0..cfg.n_layers {
        let layer = &weights.layers[li];
        draft_dtod_at(
            gpu,
            ordered,
            &shared.residual.buf,
            0,
            &shared.x.buf,
            0,
            (rows * h) * 4,
        )?;
        gpu.rmsnorm_batched(&shared.x, &layer.attn_norm, &shared.x_norm, rows, h, eps)?;

        // DFlash2 prepare convolution (causal across the rows of ONE block ⇒
        // one launch per lane; the kernel-coefficient GEMM is row-local).
        let conv = match (&layer.attn_conv_base, &layer.attn_conv_proj) {
            (Some(base), Some(proj)) => match (&shared.conv_dynamic, &shared.conv_temp) {
                (Some(dyn_buf), Some(tmp)) => Some((base, proj, dyn_buf, tmp)),
                _ => None,
            },
            _ => None,
        };
        let qkv_src: &GpuTensor = if let Some((base, proj, dyn_buf, tmp)) = conv {
            let (k, g, _groups, stride) = conv_kgs();
            let dyn_all = dyn_buf.sub_offset(0, rows * stride);
            gemm_dispatch(
                gpu,
                &shared.x_norm,
                proj,
                &dyn_all,
                rows,
                shared.mq_x_rot.as_ref(),
                shared.mq_x_rot_f16.as_ref(),
            )?;
            if let Some(cl) = &conv_lanes {
                gpu.dynamic_causal_conv_f32_lanes(
                    &shared.x_norm,
                    base,
                    dyn_buf,
                    tmp,
                    cl,
                    h,
                    k,
                    g,
                    stride,
                    0,
                )?;
            } else {
                for g_ in &geos {
                    let inp = shared.x_norm.sub_offset(g_.off * h, g_.b * h);
                    let out = tmp.sub_offset(g_.off * h, g_.b * h);
                    let dy = dyn_buf.sub_offset(g_.off * stride, g_.b * stride);
                    gpu.dynamic_causal_conv_f32(&inp, base, &dy, &out, g_.b, h, k, g, stride, 0)?;
                }
            }
            tmp
        } else {
            &shared.x_norm
        };
        gemm_dispatch(
            gpu,
            qkv_src,
            &layer.wq,
            &shared.q,
            rows,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;
        gemm_dispatch(
            gpu,
            qkv_src,
            &layer.wk,
            &shared.k_noise,
            rows,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;
        gemm_dispatch(
            gpu,
            qkv_src,
            &layer.wv,
            &shared.v_noise,
            rows,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;

        // Context K/V of every lane into its own ring.
        // Per-lane context fill is a lane-local GEMM of arbitrary row count: it
        // keeps the singleton's route even inside the batched 64-row scope.
        // Lane-batched ctx fill of this layer (planned and admitted before any
        // write); `None` = the per-lane loop below.
        let ctx_plan = if lane_ctx_on {
            plan_ctx(gpu, cfg, layer, &*shared, &*lanes, &geos, li)
        } else {
            None
        };
        let scoped = gpu.draft_rows64_enter(false);
        let mut filled = Ok(());
        if let Some(plan) = &ctx_plan {
            filled = ctx_fill_batched(gpu, cfg, layer, &*shared, plan);
        } else {
            for (lane, g_) in lanes.iter().zip(&geos) {
                filled = lane_ctx_fill(gpu, cfg, layer, lane.scratch, shared, &shared.q, li, g_);
                if filled.is_err() {
                    break;
                }
            }
        }
        gpu.draft_rows64_exit(scoped);
        filled?;

        gpu.rmsnorm_batched(
            &shared.q,
            &layer.q_norm,
            &shared.q,
            rows * cfg.n_heads,
            hd,
            eps,
        )?;
        gpu.rmsnorm_batched(
            &shared.k_noise,
            &layer.k_norm,
            &shared.k_noise,
            rows * cfg.n_kv_heads,
            hd,
            eps,
        )?;
        gpu.rope_batched_f32(
            &shared.q,
            &shared.k_noise,
            &shared.positions_q,
            cfg.n_heads,
            cfg.n_kv_heads,
            hd,
            theta,
            rows,
        )?;

        for (lane, g_) in lanes.iter().zip(&geos) {
            lane_attention(gpu, cfg, lane.scratch, shared, li, g_, ordered)?;
        }

        // wo lands in x (the non-fused route), then the DFlash2 finish
        // convolution per lane and the attention residual add.
        gemm_dispatch(
            gpu,
            &shared.attn_out,
            &layer.wo,
            &shared.x,
            rows,
            shared.mq_x_rot.as_ref(),
            shared.mq_x_rot_f16.as_ref(),
        )?;
        let res_v = shared.residual.sub_offset(0, rows * h);
        let x_v = shared.x.sub_offset(0, rows * h);
        if let Some((base, _proj, dyn_buf, tmp)) = conv {
            let (k, g, groups, stride) = conv_kgs();
            let base_phase1 = base.sub_offset(k * h, k * h);
            if let Some(cl) = &conv_lanes {
                gpu.dynamic_causal_conv_f32_lanes(
                    &shared.x,
                    &base_phase1,
                    dyn_buf,
                    tmp,
                    cl,
                    h,
                    k,
                    g,
                    stride,
                    k * groups,
                )?;
            } else {
                for g_ in &geos {
                    let inp = shared.x.sub_offset(g_.off * h, g_.b * h);
                    let out = tmp.sub_offset(g_.off * h, g_.b * h);
                    let dy = dyn_buf.sub_offset(g_.off * stride, g_.b * stride);
                    gpu.dynamic_causal_conv_f32(
                        &inp,
                        &base_phase1,
                        &dy,
                        &out,
                        g_.b,
                        h,
                        k,
                        g,
                        stride,
                        k * groups,
                    )?;
                }
            }
            let tmp_v = tmp.sub_offset(0, rows * h);
            gpu.add_f32(&res_v, &tmp_v, &x_v)?;
        } else {
            gpu.add_f32(&res_v, &x_v, &x_v)?;
        }

        // FFN tail.
        if let (Some(base), Some(proj), Some(dyn_buf), Some(tmp)) = (
            &layer.mlp_conv_base,
            &layer.mlp_conv_proj,
            &shared.conv_dynamic,
            &shared.conv_temp,
        ) {
            draft_dtod_at(
                gpu,
                ordered,
                &shared.residual.buf,
                0,
                &shared.x.buf,
                0,
                (rows * h) * 4,
            )?;
            gpu.rmsnorm_batched(&shared.x, &layer.ffn_norm, &shared.x_norm, rows, h, eps)?;
            let (k, g, groups, stride) = conv_kgs();
            let dyn_all = dyn_buf.sub_offset(0, rows * stride);
            gemm_dispatch(
                gpu,
                &shared.x_norm,
                proj,
                &dyn_all,
                rows,
                shared.mq_x_rot.as_ref(),
                shared.mq_x_rot_f16.as_ref(),
            )?;
            if let Some(cl) = &conv_lanes {
                gpu.dynamic_causal_conv_f32_lanes(
                    &shared.x_norm,
                    base,
                    dyn_buf,
                    tmp,
                    cl,
                    h,
                    k,
                    g,
                    stride,
                    0,
                )?;
            } else {
                for g_ in &geos {
                    let inp = shared.x_norm.sub_offset(g_.off * h, g_.b * h);
                    let out = tmp.sub_offset(g_.off * h, g_.b * h);
                    let dy = dyn_buf.sub_offset(g_.off * stride, g_.b * stride);
                    gpu.dynamic_causal_conv_f32(&inp, base, &dy, &out, g_.b, h, k, g, stride, 0)?;
                }
            }
            gemm_dispatch(
                gpu,
                tmp,
                &layer.w_gate,
                &shared.gate,
                rows,
                shared.mq_x_rot.as_ref(),
                shared.mq_x_rot_f16.as_ref(),
            )?;
            gemm_dispatch(
                gpu,
                tmp,
                &layer.w_up,
                &shared.up,
                rows,
                shared.mq_x_rot.as_ref(),
                shared.mq_x_rot_f16.as_ref(),
            )?;
            let gate_v = shared.gate.sub_offset(0, rows * inter);
            let up_v = shared.up.sub_offset(0, rows * inter);
            let gate_up_v = shared.gate_up.sub_offset(0, rows * inter);
            gpu.silu_mul_f32(&gate_v, &up_v, &gate_up_v)?;
            gemm_dispatch(
                gpu,
                &shared.gate_up,
                &layer.w_down,
                &shared.x,
                rows,
                shared.mq_x_rot.as_ref(),
                shared.mq_x_rot_f16.as_ref(),
            )?;
            let base_phase1 = base.sub_offset(k * h, k * h);
            if let Some(cl) = &conv_lanes {
                gpu.dynamic_causal_conv_f32_lanes(
                    &shared.x,
                    &base_phase1,
                    dyn_buf,
                    tmp,
                    cl,
                    h,
                    k,
                    g,
                    stride,
                    k * groups,
                )?;
            } else {
                for g_ in &geos {
                    let inp = shared.x.sub_offset(g_.off * h, g_.b * h);
                    let out = tmp.sub_offset(g_.off * h, g_.b * h);
                    let dy = dyn_buf.sub_offset(g_.off * stride, g_.b * stride);
                    gpu.dynamic_causal_conv_f32(
                        &inp,
                        &base_phase1,
                        &dy,
                        &out,
                        g_.b,
                        h,
                        k,
                        g,
                        stride,
                        k * groups,
                    )?;
                }
            }
            let tmp_v = tmp.sub_offset(0, rows * h);
            gpu.add_f32(&res_v, &tmp_v, &x_v)?;
        } else {
            draft_ffn_layer(gpu, layer, shared, rows, h, eps, false)?;
        }
    }

    // ── Final norm and projection-cache advance ──────────────────────────
    gpu.rmsnorm_batched(&shared.x, &weights.norm, &shared.x, rows, h, eps)?;
    for (lane, g_) in lanes.iter_mut().zip(&geos) {
        lane.scratch.thlog.mark_proj_cached(g_.l);
    }
    Ok(())
}
