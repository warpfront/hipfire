// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-request batched prefill chunk: several independent requests' short
//! row blocks (e.g. MTP verify rows `[seed, drafts..]`) through ONE trunk
//! forward, per request byte-identical to the singleton
//! [`super::forward_prefill_batch`] over that request's rows alone.
//!
//! It runs the singleton chunk body's own stage functions
//! (`forward_batch_chunk_impl`): the row-local stages — embedding, the
//! norm/rotate + projection GEMM stages, the FFN, the final norm — run once
//! over all requests' rows; the stages that read request state — position
//! upload, DeltaNet pre-GDN + recurrence, FullAttention prep + KV write +
//! attend — run per request on that request's rows (a row view of the
//! shared scratch), KV and DeltaNet state. Exactness rests on the shared
//! stages being row-local and selecting the same kernels for the combined
//! row count as for one request's rows: true while every count is in
//! `2..64` (the gfx12 projection GEMMs are row-count independent there —
//! CbSpec probe — and every n-dependent route switch is at >= 64 rows).
//!
//! Aggregate 64..=128 rows (`HIPFIRE_CB_VERIFY_CHUNK128`, exact gfx1201 with
//! a dense MQ4G256V2 target only, see [`multi_chunk_wide_admitted`]): the
//! default `>= 64`-row projection route (IU4/FP8 quantized activations) is not
//! the singleton's arithmetic, so the shared stages run with
//! `DenseBatchMath::SingletonWmma` — the F32 producers plus the explicit
//! exact verify GEMMs, a F16-input / f32-WMMA chain byte-identical to the
//! singleton at every row count. Every request stays below 64 rows on its
//! own (its state-sensitive stages and its singleton equivalent are
//! request-sized); only the combined row count may reach 128.
//!
//! DFlash chain-verify requests (`MultiChunkRequest::fusion == ChainVerify`,
//! with their own hidden ring and tape) keep every route that depends on the
//! request's own row count request-sized: GDN pre/tape fusion and FA prep use
//! the request's `fusion`. The ChainVerify S4 residual arm (`1..=16` rows) is
//! byte-identical to the `Off` route on exact gfx1201, so output projection +
//! FFN run once at the combined count there; on other arches a layer where
//! any request's singleton would take the S4 arm runs those stages per request
//! view instead.

use super::*;

/// One request's rows of a multi-request chunk.
///
/// `fusion` is the request's own [`DflashFusionCtx`]: `Off` for MTP verify and
/// ordinary rows, `ChainVerify` for a DFlash chain verify block. Every
/// route-sensitive stage (GDN pre/tape fusion, FA prep; off gfx1201 also the
/// S4 residual arm) runs with this request's `fusion` and its own row count,
/// exactly as the singleton verify forward over these rows alone. `hidden_rb`, when set, is
/// this request's DFlash extraction ring: the post-layer residual rows of
/// each extract layer are staged and committed (head advanced by the
/// request's row count) exactly as the singleton verify does.
pub struct MultiChunkRequest<'a> {
    pub tokens: &'a [u32],
    pub start_pos: usize,
    pub kv_cache: &'a mut llama::KvCache,
    pub dn_state: &'a mut DeltaNetState,
    /// Rollback tape for this request's rows (the singleton verify's
    /// `gdn_tape`, offset 0); `None` = no capture.
    pub gdn_tape: Option<&'a crate::speculative::GdnTape>,
    /// `Off` (MTP / plain rows) or `ChainVerify` (DFlash chain verify).
    pub fusion: DflashFusionCtx,
    /// This request's hidden-state ring (DFlash extract layers); `None` = no
    /// capture. Its staging must hold the request's rows (`max_batch >= n`).
    pub hidden_rb: Option<&'a mut HiddenStateRingBuffer>,
}

/// The largest aggregate row count the multi-request body supports (the
/// exact wide verify GEMMs cover 64..=128 rows). The route's EFFECTIVE cap is
/// [`multi_chunk_row_cap`]: 128 only with `HIPFIRE_CB_VERIFY_CHUNK128` on exact
/// gfx1201, otherwise [`MULTI_CHUNK_PRODUCT_MAX_ROWS`].
pub const MULTI_CHUNK_MAX_ROWS: usize = 128;

/// The largest aggregate row count on the product arithmetic: below every
/// n-dependent kernel/route switch of the shared stages (`>= 64` rows).
pub const MULTI_CHUNK_PRODUCT_MAX_ROWS: usize = 63;

/// The largest single request: its singleton (`forward_prefill_batch`) over
/// 64+ rows would select the quantized `>= 64`-row projection route, whose
/// bytes the wide exact route does not reproduce, so a lone 64/128-row request
/// is never admitted as a window.
pub const MULTI_CHUNK_MAX_LANE_ROWS: usize = 63;

/// The effective aggregate row cap of this device/process:
/// `Gpu::mq4_verify_chunk_rows` (128 on exact gfx1201 with
/// `HIPFIRE_CB_VERIFY_CHUNK128` and WMMA batch tiles, else 63) bounded to the
/// supported `MULTI_CHUNK_PRODUCT_MAX_ROWS..=MULTI_CHUNK_MAX_ROWS`.
pub fn multi_chunk_row_cap(gpu: &Gpu) -> usize {
    gpu.mq4_verify_chunk_rows().clamp(MULTI_CHUNK_PRODUCT_MAX_ROWS, MULTI_CHUNK_MAX_ROWS)
}

/// May an aggregate chunk of 64+ rows run on this target? The wide route is
/// exact only when (a) the device's effective cap exceeds the product cap,
/// (b) every singleton arm it must match runs the one-tile F16-WMMA chain the
/// exact wide kernels reproduce ([`Gpu::mq4_verify_singleton_chain`]: refused
/// under `HIPFIRE_FP16=0`, `HIPFIRE_LM_HEAD_WMMA=0`,
/// `HIPFIRE_HFQ4G256_LDSSTAGE=1`), and (c) the trunk and head are dense
/// MQ4G256V2: every projection of every layer and the lm_head (no MoE,
/// Lloyd, MQ3/MQ6, Q8 or other dtypes), with `w_down.k == hidden_dim` (the
/// `SingletonWmma` FFN-down shape). Decided before any request state is
/// written; a refusal keeps the cap at [`MULTI_CHUNK_PRODUCT_MAX_ROWS`].
pub fn multi_chunk_wide_admitted(gpu: &Gpu, weights: &Qwen35Weights, config: &Qwen35Config) -> bool {
    multi_chunk_wide_route(&gpu.arch, &gpu.flags) && multi_chunk_dense_mq4v2_target(weights, config)
}

/// The device/flag half of [`multi_chunk_wide_admitted`] (no device needed).
fn multi_chunk_wide_route(arch: &str, flags: &rdna_compute::FeatureFlags) -> bool {
    Gpu::mq4_verify_chunk_rows_for(arch, flags).clamp(MULTI_CHUNK_PRODUCT_MAX_ROWS, MULTI_CHUNK_MAX_ROWS)
        > MULTI_CHUNK_PRODUCT_MAX_ROWS
        && Gpu::mq4_verify_singleton_chain_for(arch, flags)
}

/// The target half of [`multi_chunk_wide_admitted`].
fn multi_chunk_dense_mq4v2_target(weights: &Qwen35Weights, config: &Qwen35Config) -> bool {
    use rdna_compute::DType::MQ4G256V2;
    config.num_experts == 0
        && weights.output.gpu_dtype == MQ4G256V2
        && weights.layers.iter().all(|l| match l {
            LayerWeights::DeltaNet(d) => {
                d.w_down.k == config.hidden_dim
                    && [&d.wqkv, &d.wz, &d.w_alpha, &d.w_beta, &d.wo, &d.w_gate, &d.w_up, &d.w_down]
                        .iter()
                        .all(|w| w.gpu_dtype == MQ4G256V2)
            }
            LayerWeights::FullAttn(f) => {
                f.w_down.k == config.hidden_dim
                    && [&f.wq, &f.wk, &f.wv, &f.wo, &f.w_gate, &f.w_up, &f.w_down]
                        .iter()
                        .all(|w| w.gpu_dtype == MQ4G256V2)
            }
            _ => false,
        })
}

/// The rows one trunk chunk of this target packs: the scratch capacity
/// `cap` when the wide route is admitted for this target, else at most
/// [`MULTI_CHUNK_PRODUCT_MAX_ROWS`] (a scratch sized for the wide route may
/// serve a target that cannot take it).
pub fn multi_chunk_pack_cap(gpu: &Gpu, weights: &Qwen35Weights, config: &Qwen35Config, cap: usize) -> usize {
    pack_cap_for(multi_chunk_wide_admitted(gpu, weights, config), cap)
}

fn pack_cap_for(wide_admitted: bool, cap: usize) -> usize {
    cap.min(if wide_admitted { MULTI_CHUNK_MAX_ROWS } else { MULTI_CHUNK_PRODUCT_MAX_ROWS })
}

/// Rows a shared verify scratch planned for `requested` rows allocates on
/// this target: its effective packing cap ([`multi_chunk_pack_cap`]), at
/// least [`MIN_BATCH`]. A target the wide route does not admit allocates at
/// most [`MULTI_CHUNK_PRODUCT_MAX_ROWS`] rows, whatever the device cap.
pub fn multi_chunk_scratch_rows(gpu: &Gpu, weights: &Qwen35Weights, config: &Qwen35Config, requested: usize) -> usize {
    scratch_rows_for(multi_chunk_wide_admitted(gpu, weights, config), requested)
}

fn scratch_rows_for(wide_admitted: bool, requested: usize) -> usize {
    pack_cap_for(wide_admitted, requested).max(MIN_BATCH)
}

/// Pack whole lanes, in order, into trunk chunks of at most
/// `min(max_rows, MULTI_CHUNK_MAX_ROWS)` rows: `rows[i]` is lane `i`'s row
/// count and `max_rows` the caller's effective cap (63 on the product route;
/// 64..=128 only with the wide exact route admitted). `out` receives one
/// lane-index range per chunk (cleared first; no allocation when `out`
/// already has capacity). A lane is never split, padded or reordered, and no
/// lane may exceed [`MULTI_CHUNK_MAX_LANE_ROWS`]: e.g. eight 16-row lanes pack
/// as `[0..3, 3..6, 6..8]` (48 + 48 + 32 rows) at cap 63, `[0..4, 4..8]`
/// (64 + 64) at cap 64 and `[0..8]` (one 128-row chunk) at cap 128. Errors
/// (leaving `out` empty) when a lane has fewer than [`MIN_BATCH`] rows or more
/// rows than one chunk or one request can hold.
pub fn pack_whole_lanes(
    rows: &[usize],
    max_rows: usize,
    out: &mut Vec<std::ops::Range<usize>>,
) -> Result<(), String> {
    out.clear();
    let cap = max_rows.min(MULTI_CHUNK_MAX_ROWS);
    let lane_cap = cap.min(MULTI_CHUNK_MAX_LANE_ROWS);
    let mut start = 0usize;
    let mut acc = 0usize;
    for (i, &n) in rows.iter().enumerate() {
        if n < MIN_BATCH || n > lane_cap {
            out.clear();
            return Err(format!(
                "pack_whole_lanes: lane {i} has {n} rows, outside {MIN_BATCH}..={lane_cap}"
            ));
        }
        if acc + n > cap {
            out.push(start..i);
            start = i;
            acc = 0;
        }
        acc += n;
    }
    if start < rows.len() {
        out.push(start..rows.len());
    }
    Ok(())
}

/// Most requests (segments) one twin launch carries: every request has at
/// least [`MIN_BATCH`] rows.
const MULTI_CHUNK_MAX_SEGS: usize = MULTI_CHUNK_MAX_ROWS / MIN_BATCH;

use hipfire_dispatch::ops::verify_twins::{self as seg_twins, AttnFp8Seg};

/// `HIPFIRE_CB_SEG_TWINS=0` keeps every request on its own singleton
/// launches (A/B control for the segment twins).
static SEG_TWINS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_SEG_TWINS").map_or(true, |v| v.trim() != "0")
});

/// Shared scratch of a multi-request chunk: the row scratch every request's
/// rows live in, and the segment-twin tables (one per FullAttention layer).
pub struct MultiChunkScratch {
    pub pbs: PrefillBatchScratch,
    attn_segs: GpuTensor,
}

impl MultiChunkScratch {
    /// `max_rows` is clamped to `MIN_BATCH..=multi_chunk_row_cap(gpu)` (the
    /// effective cap: 63 unless the wide verify route is on); the scratch
    /// stores that capacity ([`Self::max_rows`]).
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config, max_rows: usize) -> HipResult<Self> {
        let rows = max_rows.clamp(MIN_BATCH, multi_chunk_row_cap(gpu));
        let pbs = PrefillBatchScratch::new(gpu, config, rows)?;
        let n_fa = config.layer_types.iter().filter(|t| **t == LayerType::FullAttention).count();
        let bytes = n_fa.max(1) * MULTI_CHUNK_MAX_SEGS * std::mem::size_of::<AttnFp8Seg>();
        match gpu.zeros(&[bytes.div_ceil(4)], rdna_compute::DType::F32) {
            Ok(attn_segs) => Ok(Self { pbs, attn_segs }),
            Err(e) => {
                let _ = pbs.free_gpu(gpu);
                Err(e)
            }
        }
    }

    pub fn max_rows(&self) -> usize {
        self.pbs.max_batch
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        self.pbs.free_gpu(gpu)?;
        gpu.free_tensor(self.attn_segs)
    }
}

/// Does this request's singleton attend run `attention_fp8_e4m3_kv_batched`
/// after `kv_cache_write_fp8_e4m3_batched` (the fp8 scalar-batched arm:
/// gfx1201, under 64 rows, context at most the 4096 crossover)? Then the
/// segment twin reproduces it.
fn attn_fp8_twin_eligible(gpu: &Gpu, s: &Qwen35Scratch, kv: &llama::KvCache, start_pos: usize, n: usize) -> bool {
    if !(gpu.arch_caps.is_gfx1201() && kv.quant_fp8 && (MIN_BATCH..64).contains(&n) && start_pos + n <= 4096) {
        return false;
    }
    KvTierPlan::derive(KvTierInputs {
        pos: start_pos,
        flash_mode: s.flash_mode as usize,
        capture_mode: gpu.graphs.capture_mode,
        batch_size: n,
        is_tree: false,
        ..kv.tier_inputs()
    })
    .is_ok_and(|p| {
        p.attend_key == hipfire_dispatch::types::KernelKey::AttnFp8E4m3KvBatchedMasked
            && p.write_key == hipfire_dispatch::types::KernelKey::KvWriteFp8E4m3Batched
    })
}

/// Rows `[r0, r0 + n)` of `pbs` as a scratch of `n` rows. Only row-major
/// fields are re-pointed; they alias `pbs` (never freed through the view).
fn rows_view(pbs: &PrefillBatchScratch, config: &Qwen35Config, r0: usize, n: usize) -> std::mem::ManuallyDrop<PrefillBatchScratch> {
    let dim = config.dim;
    let hidden = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = 2 * k_dim + v_dim;
    let nv = config.linear_num_value_heads;
    let q_dim = config.n_heads * config.head_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let v = |t: &GpuTensor, w: usize| t.sub_offset(r0 * w, n * w);
    // SAFETY: a bitwise copy whose every row-major tensor is then replaced
    // by a non-owning view of the same buffer; `ManuallyDrop` keeps the copy
    // from ever releasing anything `pbs` owns.
    unsafe {
        let mut p = std::mem::ManuallyDrop::new(std::ptr::read(pbs));
        let set = |slot: &mut GpuTensor, t: GpuTensor| std::ptr::write(slot, t);
        p.max_batch = n;
        set(&mut p.x_batch, v(&pbs.x_batch, dim));
        set(&mut p.x_rot_batch, v(&pbs.x_rot_batch, dim));
        set(&mut p.x_norm_batch, v(&pbs.x_norm_batch, dim));
        set(&mut p.dn_qkv_batch, v(&pbs.dn_qkv_batch, qkv_dim));
        set(&mut p.dn_z_batch, v(&pbs.dn_z_batch, v_dim));
        set(&mut p.dn_z_fold_batch, v(&pbs.dn_z_fold_batch, v_dim + 256));
        set(&mut p.dn_alpha_batch, v(&pbs.dn_alpha_batch, nv));
        set(&mut p.dn_beta_batch, v(&pbs.dn_beta_batch, nv));
        set(&mut p.dn_q_raw_batch, v(&pbs.dn_q_raw_batch, k_dim));
        set(&mut p.dn_k_raw_batch, v(&pbs.dn_k_raw_batch, k_dim));
        set(&mut p.dn_v_batch, v(&pbs.dn_v_batch, v_dim));
        set(&mut p.dn_q_batch, v(&pbs.dn_q_batch, v_dim));
        set(&mut p.dn_k_batch, v(&pbs.dn_k_batch, v_dim));
        set(&mut p.dn_attn_out_batch, v(&pbs.dn_attn_out_batch, v_dim));
        set(&mut p.dn_normed_batch, v(&pbs.dn_normed_batch, v_dim));
        set(&mut p.gate_ffn_batch, v(&pbs.gate_ffn_batch, hidden));
        set(&mut p.up_batch, v(&pbs.up_batch, hidden));
        set(&mut p.ffn_hidden_batch, v(&pbs.ffn_hidden_batch, hidden));
        set(&mut p.dn_normed_rot_batch, v(&pbs.dn_normed_rot_batch, v_dim));
        set(&mut p.positions, v(&pbs.positions, 1));
        set(&mut p.rope_positions, v(&pbs.rope_positions, 1));
        set(&mut p.pos3, v(&pbs.pos3, 3));
        set(&mut p.ext_emb_index, v(&pbs.ext_emb_index, 1));
        set(&mut p.tokens, v(&pbs.tokens, 1));
        set(&mut p.fa_q_full_batch, v(&pbs.fa_q_full_batch, 2 * q_dim));
        set(&mut p.fa_q_batch, v(&pbs.fa_q_batch, q_dim));
        set(&mut p.fa_gate_batch, v(&pbs.fa_gate_batch, q_dim));
        set(&mut p.fa_k_batch, v(&pbs.fa_k_batch, kv_dim));
        set(&mut p.fa_v_batch, v(&pbs.fa_v_batch, kv_dim));
        set(&mut p.fa_attn_out_batch, v(&pbs.fa_attn_out_batch, q_dim));
        set(&mut p.fa_attn_out_rot_batch, v(&pbs.fa_attn_out_rot_batch, q_dim));
        p
    }
}

/// Must a layer's output projection and FFN run per request view because a
/// request's singleton takes the ChainVerify S4 residual arm for a consumer of
/// dtype `w_dtype` at its own row count (`s4_residual_fast` admits only
/// `1..=16` rows)?
///
/// Never on exact gfx1201: there the S4 arm is byte-identical to the `Off`
/// route at any row count `<= 63` (row-parallel F16 producers computing the
/// F32 pipeline's values in-register and casting with `convert_f32_to_f16`'s
/// cast; the same one-tile-arithmetic residual GEMM; every other n-dependent
/// route switch is at `>= 64` rows), so the stage runs once at the combined
/// row count. On exact gfx1100 the S4 residual GEMM picks ldsstage / split-K
/// tiers at `n <= 16` that the `Off` route does not reproduce at larger `n`
/// (not proven identical), so it keeps the split.
fn any_lane_s4(gpu: &Gpu, reqs: &[MultiChunkRequest<'_>], w_dtype: rdna_compute::DType) -> bool {
    !gpu.arch_caps.is_gfx1201()
        && reqs
            .iter()
            .any(|r| s4_residual_fast(gpu, r.fusion, w_dtype, &BatchEpilogue::Residual, r.tokens.len()))
}

/// Capture the post-layer residual rows of every request whose ring extracts
/// `layer_idx` (the singleton's post-FFN `write_chunk_rows` of `x_batch`).
fn capture_layer_rows(
    gpu: &mut Gpu,
    reqs: &[MultiChunkRequest<'_>],
    views: &[std::mem::ManuallyDrop<PrefillBatchScratch>],
    layer_idx: usize,
) -> HipResult<()> {
    for (r, view) in reqs.iter().zip(views) {
        if let Some(rb) = r.hidden_rb.as_deref() {
            if let Some(slot) = rb.extract_slot(layer_idx) {
                rb.write_chunk_rows(gpu, slot, &view.x_batch, r.tokens.len())?;
            }
        }
    }
    Ok(())
}

/// Run every request's rows through one trunk forward, per request
/// byte-identical to `forward_prefill_batch_with_pbs_opts` over its rows
/// with the request's own `fusion`, `gdn_tape` (offset 0) and `hidden_rb`.
/// With `hidden_out`, writes post-output-norm hidden rows `[total x dim]` in
/// request order (`HiddenCapture::Verify`, as the singleton verify).
///
/// Shared across requests: embedding, the norm/rotate + input projection
/// GEMMs, and the output projection + FFN (on exact gfx1201, including for
/// ChainVerify lanes, whose S4 arm equals the `Off` route; elsewhere only when
/// no request's singleton takes the ChainVerify S4 residual arm / gfx1100 F16
/// projection route at its own row count). Per request: positions, GDN
/// pre/tape + recurrence, FA prep/KV write/attend with that request's fusion
/// flags, and the ring capture. A layer where a non-gfx1201 request would
/// select a ChainVerify route at its own `n` runs those stages on every
/// request's row view with its own `fusion` and `n` (MTP views with `Off`),
/// never at the combined count.
///
/// Every precondition (rows `2..=63` per request, in total at most the
/// effective cap — `<= 63` on the product route, `64..=128` only when
/// [`multi_chunk_wide_admitted`] holds for this target — capacities,
/// ring/tape bounds, Q8+EF DeltaNet, uncompacted Q8/fp8 KV, no capture or
/// recording) is checked before any request state is mutated; the request is
/// refused before the first launch otherwise. A combined count of 64 or more
/// runs the shared stages with [`DenseBatchMath::SingletonWmma`]; at most 63
/// rows keeps [`DenseBatchMath::Product`] exactly as before.
pub fn forward_prefill_batch_multi(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    s: &Qwen35Scratch,
    scratch: &MultiChunkScratch,
    reqs: &mut [MultiChunkRequest<'_>],
    hidden_out: Option<&GpuTensor>,
) -> HipResult<()> {
    let refuse = |why: &str| Err(HipError::new(0, &format!("forward_prefill_batch_multi: {why}")));
    let pbs = &scratch.pbs;
    let total: usize = reqs.iter().map(|r| r.tokens.len()).sum();
    if reqs.is_empty() || total > multi_chunk_row_cap(gpu) || total > pbs.max_batch || pbs.lean {
        return refuse("row count outside 1..=effective cap (63, or 128 on the wide verify route) / scratch capacity, or lean scratch");
    }
    // Aggregate 64+ rows: the default projection route is quantized and not
    // the singleton's arithmetic; only the exact verify GEMMs of a dense
    // MQ4G256V2 target whose singleton runs the one-tile F16-WMMA chain
    // reproduce it. Refused before any state write.
    let wide = total > MULTI_CHUNK_PRODUCT_MAX_ROWS;
    if wide && !multi_chunk_wide_admitted(gpu, weights, config) {
        return refuse("64+ rows need the exact wide verify route: a dense MQ4G256V2 gfx1201 target whose singleton runs the one-tile F16-WMMA chain");
    }
    let math = if wide { DenseBatchMath::SingletonWmma } else { DenseBatchMath::Product };
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return refuse("graph capture / replay recording is not supported");
    }
    let arch = gpu.arch.clone();
    let mut kv_ends = Vec::with_capacity(reqs.len());
    for r in reqs.iter() {
        let n = r.tokens.len();
        // The singleton runs n == 1 through forward_scratch, not this body.
        if n < MIN_BATCH {
            return refuse("every request needs >= 2 rows");
        }
        // A lone 64+-row request would be the singleton's quantized route.
        if n > MULTI_CHUNK_MAX_LANE_ROWS {
            return refuse("every request needs <= 63 rows");
        }
        if !prefill_batch_pbs_eligible(weights, config, r.dn_state, n, &arch, true) {
            return refuse("request not eligible for the batched prefill body");
        }
        let d = &r.dn_state;
        if d.quant != StateQuant::Q8 || d.s_ef_residual.len() != d.s_matrices.len() {
            return refuse("DeltaNet state must be Q8 with error feedback");
        }
        let kv = &r.kv_cache;
        if !(kv.quant_q8 || kv.quant_fp8) || kv.compact_offset != 0 {
            return refuse("KV must be uncompacted Q8 or fp8");
        }
        if r.gdn_tape.is_some_and(|t| t.max_n < n) {
            return refuse("GDN tape smaller than the request's rows");
        }
        if let Some(rb) = r.hidden_rb.as_deref() {
            // The singleton verify stages `n` rows per extract layer and
            // commits them; a ring it would write straight (n > staging) or
            // cannot hold the rows is not reproduced here.
            if rb.hidden_dim != config.dim
                || rb.layer_bufs.len() != rb.extract_layers.len()
                || rb.staging_bufs.len() != rb.extract_layers.len()
                || rb.extract_layers.iter().any(|&l| l >= config.n_layers)
                || n > rb.max_batch
                || n > rb.max_positions
            {
                return refuse("hidden ring layers/dimension/staging cannot hold the request's rows");
            }
        }
        kv_ends.push(checked_kv_end(r.start_pos, n, "forward_prefill_batch_multi")?);
    }
    if !weights.layers.iter().all(|l| match l {
        LayerWeights::DeltaNet(_) => true,
        LayerWeights::FullAttn(_) => qwen35_layer_batch_admissible(l, config, &arch).is_ok(),
        _ => false,
    }) {
        return refuse("only dense DeltaNet / batched FullAttention layers");
    }
    // Capacity/mapping (warms mapped KV; no request state is written).
    for (r, &end) in reqs.iter_mut().zip(&kv_ends) {
        release_widened_pbs_for_kv_growth(gpu, r.kv_cache, config, s, end)?;
        r.kv_cache.ensure_mapped_capacity(gpu, end)?;
        r.kv_cache.require_mapped_capacity(end)?;
    }

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    // Fusion of every stage run once over the combined rows: those stages
    // read `fusion` only through the gfx1100 F16 projection route (checked
    // per request below, which switches them to per-request views) and the
    // S4 arm (checked per layer), so `Off` is the singleton's route for them.
    let shared_fusion = DflashFusionCtx::Off;
    let sem = BatchSemantics::Sequential;
    // Request row ranges in the shared scratch.
    let mut offs = Vec::with_capacity(reqs.len());
    let mut off = 0usize;
    for r in reqs.iter() {
        offs.push(off);
        off += r.tokens.len();
    }
    let views: Vec<_> = reqs.iter().zip(&offs).map(|(r, &o)| rows_view(pbs, config, o, r.tokens.len())).collect();
    // Any request on the ChainVerify gfx1100 F16 projection route at its own
    // row count: the projection/FFN input stages run per request view.
    let split_proj = reqs.iter().any(|r| mq_f16_projection_fast_route(gpu, r.fusion, r.tokens.len(), dim));

    let tokens: Vec<u32> = reqs.iter().flat_map(|r| r.tokens.iter().copied()).collect();
    batch_chunk_embed_tokens(gpu, weights, &tokens, s, pbs, total, dim, dim * 4, true, false, false, None, None)?;
    for (r, view) in reqs.iter().zip(&views) {
        batch_chunk_upload_positions(gpu, view, sem, r.start_pos, r.tokens.len(), None, false)?;
    }
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
    let arch_has_wmma = q8_wmma_arch;
    let tapes = reqs.iter().any(|r| r.gdn_tape.is_some());
    let ctx = DispatchCtx::new(gpu).with_workload(prefill_dispatch_workload(hidden_out.is_some(), tapes, false));

    // Requests whose singleton attend is the fp8 scalar-batched arm run it
    // as one segment-twin launch per layer (tables staged up front). All
    // twin segments share the singleton `max_seq` (`physical_cap`) word.
    let mut twin_attn: Vec<bool> = reqs
        .iter()
        .map(|r| *SEG_TWINS && attn_fp8_twin_eligible(gpu, s, r.kv_cache, r.start_pos, r.tokens.len()))
        .collect();
    let twin_cap = reqs.iter().zip(&twin_attn).find(|(_, &t)| t).map(|(r, _)| r.kv_cache.physical_cap);
    for (r, t) in reqs.iter().zip(twin_attn.iter_mut()) {
        *t &= Some(r.kv_cache.physical_cap) == twin_cap;
    }
    let twin_ctx: Vec<usize> =
        reqs.iter().zip(&twin_attn).filter(|(_, &t)| t).map(|(r, _)| r.start_pos + r.tokens.len()).collect();
    let twin_rows = reqs.iter().zip(&twin_attn).filter(|(_, &t)| t).map(|(r, _)| r.tokens.len()).max().unwrap_or(0);
    if !twin_ctx.is_empty() {
        let mut fa_idx = 0usize;
        for (layer_idx, ty) in config.layer_types.iter().enumerate() {
            if *ty != LayerType::FullAttention {
                continue;
            }
            let segs: Vec<AttnFp8Seg> = reqs
                .iter()
                .zip(&views)
                .zip(&twin_attn)
                .filter(|(_, &t)| t)
                .map(|((r, view), _)| AttnFp8Seg {
                    q: view.fa_q_batch.buf.as_ptr() as u64,
                    k_cache: r.kv_cache.k_gpu[layer_idx].buf.as_ptr() as u64,
                    v_cache: r.kv_cache.v_gpu[layer_idx].buf.as_ptr() as u64,
                    out: view.fa_attn_out_batch.buf.as_ptr() as u64,
                    positions: view.positions.buf.as_ptr() as u64,
                    n_rows: r.tokens.len() as u64,
                })
                .collect();
            seg_twins::stage_attention_fp8_segs(
                gpu,
                &scratch.attn_segs,
                fa_idx * MULTI_CHUNK_MAX_SEGS,
                &segs,
                config.n_heads,
                config.head_dim,
            )?;
            fa_idx += 1;
        }
    }

    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;
    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // batch_chunk_delta_net_attn, non chunk-scan arm (n < 64).
                if split_proj {
                    for (r, view) in reqs.iter().zip(&views) {
                        batch_chunk_delta_net_input_projection(
                            gpu, layer, config, view, r.tokens.len(), dim, q8_wmma_arch, r.fusion, None,
                            DenseBatchMath::Product,
                        )?;
                    }
                } else {
                    batch_chunk_delta_net_input_projection(
                        gpu, layer, config, pbs, total, dim, q8_wmma_arch, shared_fusion, None, math,
                    )?;
                }
                for (r, view) in reqs.iter_mut().zip(&views) {
                    let n = r.tokens.len();
                    let parents = batch_chunk_delta_net_pre_gdn(
                        gpu, layer, config, view, r.dn_state, n, k_dim, v_dim, n_v_heads, hd, sem, None, r.gdn_tape, 0,
                        delta_layer_idx, r.fusion,
                    )?;
                    if parents.is_some() {
                        return refuse("tree recurrence in a linear verify");
                    }
                    let d = &*r.dn_state;
                    gpu.gated_delta_net_q8_batch_seq(
                        &view.dn_q_batch,
                        &view.dn_k_batch,
                        &view.dn_v_batch,
                        &view.dn_alpha_batch,
                        &view.dn_beta_batch,
                        &d.s_matrices[delta_layer_idx],
                        &d.s_scales[delta_layer_idx],
                        &view.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                        d.ef_residual(delta_layer_idx),
                    )?;
                }
                if split_proj
                    || any_lane_s4(gpu, reqs, layer.wo.gpu_dtype)
                    || any_lane_s4(gpu, reqs, layer.w_down.gpu_dtype)
                {
                    for (r, view) in reqs.iter().zip(&views) {
                        let n = r.tokens.len();
                        batch_chunk_delta_net_output_projection(
                            gpu,
                            layer,
                            config,
                            view,
                            n,
                            n_v_heads,
                            q8_wmma_arch,
                            arch_has_wmma,
                            BatchEpilogue::Residual,
                            r.fusion,
                            GdnScanOut::F32,
                            DenseBatchMath::Product,
                        )?;
                        batch_chunk_delta_net_ffn(
                            gpu, layer, config, view, n, dim, hidden_dim, q8_wmma_arch, arch_has_wmma,
                            BatchEpilogue::Residual, r.fusion, DenseBatchMath::Product,
                        )?;
                    }
                } else {
                    batch_chunk_delta_net_output_projection(
                        gpu,
                        layer,
                        config,
                        pbs,
                        total,
                        n_v_heads,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        shared_fusion,
                        GdnScanOut::F32,
                        math,
                    )?;
                    batch_chunk_delta_net_ffn(
                        gpu, layer, config, pbs, total, dim, hidden_dim, q8_wmma_arch, arch_has_wmma,
                        BatchEpilogue::Residual, shared_fusion, math,
                    )?;
                }
                capture_layer_rows(gpu, reqs, &views, layer_idx)?;
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                // batch_chunk_full_attn_attn with the per-request flags each
                // request's own singleton chunk computes (all n < 64).
                if split_proj {
                    for (r, view) in reqs.iter().zip(&views) {
                        batch_chunk_full_attn_input_projection(
                            gpu, layer, config, view, r.tokens.len(), dim, q8_wmma_arch, r.fusion,
                            DenseBatchMath::Product,
                        )?;
                    }
                } else {
                    batch_chunk_full_attn_input_projection(
                        gpu, layer, config, pbs, total, dim, q8_wmma_arch, shared_fusion, math,
                    )?;
                }
                for ((r, view), &twin) in reqs.iter_mut().zip(&views).zip(&twin_attn) {
                    let n = r.tokens.len();
                    let max_ctx_len = r.start_pos + n;
                    let gfx12_fa_prep = gfx12_fa_prep_admitted(gpu, config, r.fusion, n);
                    let multirow = q8_multirow_attn_admitted(
                        gpu.arch_caps.arch(),
                        r.kv_cache.quant_q8,
                        config.head_dim,
                        n,
                        r.start_pos + n,
                        fa_pertoken_min_ctx(gpu.arch_caps.arch()),
                        false,
                        false,
                        false,
                        false,
                        false,
                    );
                    batch_chunk_full_attn_prepare(
                        gpu, multirow, layer, config, view, s, r.kv_cache, n, r.start_pos, max_ctx_len, &ctx, sem, None,
                        kv_layer_idx, layer_idx, r.fusion, gfx12_fa_prep, false, false,
                    )?;
                    if twin {
                        // The singleton attend's paired write; the attention
                        // itself runs in the segment twin below.
                        for (cache, src) in
                            [(&r.kv_cache.k_gpu[layer_idx], &view.fa_k_batch), (&r.kv_cache.v_gpu[layer_idx], &view.fa_v_batch)]
                        {
                            gpu.kv_cache_write_fp8_e4m3_batched(
                                cache,
                                src,
                                &view.positions,
                                config.n_kv_heads,
                                config.head_dim,
                                n,
                            )?;
                        }
                    } else {
                        batch_chunk_fa_attend(
                            gpu, config, view, s, r.kv_cache, n, r.start_pos, max_ctx_len, &ctx, sem, None, layer_idx,
                            multirow, None, false,
                        )?;
                    }
                }
                if let Some(cap) = twin_cap.filter(|_| !twin_ctx.is_empty()) {
                    seg_twins::attention_fp8_segs(
                        gpu,
                        &scratch.attn_segs,
                        kv_layer_idx * MULTI_CHUNK_MAX_SEGS,
                        twin_rows,
                        &twin_ctx,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        cap,
                    )?;
                }
                if split_proj
                    || any_lane_s4(gpu, reqs, layer.wo.gpu_dtype)
                    || any_lane_s4(gpu, reqs, layer.w_down.gpu_dtype)
                {
                    for (r, view) in reqs.iter().zip(&views) {
                        let n = r.tokens.len();
                        batch_chunk_full_attn_output_projection(
                            gpu,
                            layer,
                            view,
                            n,
                            q8_wmma_arch,
                            arch_has_wmma,
                            BatchEpilogue::Residual,
                            r.fusion,
                            false,
                            None,
                            DenseBatchMath::Product,
                        )?;
                        batch_chunk_full_attn_ffn(
                            gpu, layer, config, view, n, dim, hidden_dim, q8_wmma_arch, arch_has_wmma,
                            BatchEpilogue::Residual, r.fusion, DenseBatchMath::Product,
                        )?;
                    }
                } else {
                    batch_chunk_full_attn_output_projection(
                        gpu,
                        layer,
                        pbs,
                        total,
                        q8_wmma_arch,
                        arch_has_wmma,
                        BatchEpilogue::Residual,
                        shared_fusion,
                        false,
                        None,
                        math,
                    )?;
                    batch_chunk_full_attn_ffn(
                        gpu, layer, config, pbs, total, dim, hidden_dim, q8_wmma_arch, arch_has_wmma,
                        BatchEpilogue::Residual, shared_fusion, math,
                    )?;
                }
                capture_layer_rows(gpu, reqs, &views, layer_idx)?;
                kv_layer_idx += 1;
            }
            _ => return refuse("layer type mismatch"),
        }
    }
    batch_chunk_final_logits(
        gpu,
        weights,
        config,
        s,
        pbs,
        total,
        dim,
        dim * 4,
        hidden_out.map(|t| (t, 0, HiddenCapture::Verify)),
        false,
        true,
        &ctx,
    )?;
    // The singleton chunk loop's per-chunk ring finish: scatter this
    // request's staged rows to its ring head and advance by its row count.
    for r in reqs.iter_mut() {
        let n = r.tokens.len();
        if let Some(rb) = r.hidden_rb.as_deref_mut() {
            rb.finish_prefill_chunk(gpu, n)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod pack_tests {
    use super::{
        multi_chunk_wide_route, pack_whole_lanes, scratch_rows_for, MULTI_CHUNK_MAX_ROWS, MULTI_CHUNK_PRODUCT_MAX_ROWS,
    };
    use rdna_compute::FeatureFlags;

    /// gfx1201 defaults with `HIPFIRE_CB_VERIFY_CHUNK128=1` (the env default of
    /// `HIPFIRE_WMMA_BATCH_TILES` on gfx1201 is on).
    fn chunk128_flags() -> FeatureFlags {
        let mut f = FeatureFlags::for_test("gfx1201");
        f.cb_verify_chunk128 = true;
        f.wmma_batch_tiles = true;
        f
    }

    /// Every switch that moves a singleton (`<= 63`-row) dense MQ4G256V2 arm
    /// off the one-tile F16-WMMA chain the wide kernels reproduce.
    const SINGLETON_ARM_SWITCHES: [(&str, fn(&mut FeatureFlags)); 4] = [
        ("HIPFIRE_FP16=0", |f| f.fp16_disabled = true),
        ("HIPFIRE_LM_HEAD_WMMA=0", |f| f.lm_head_wmma_disabled = true),
        ("HIPFIRE_HFQ4G256_LDSSTAGE=1", |f| f.hfq4g256_ldsstage_wmma = true),
        ("HIPFIRE_WMMA_BATCH_TILES=0", |f| f.wmma_batch_tiles = false),
    ];

    #[test]
    fn wide_route_admitted_by_default_refused_under_each_singleton_switch() {
        assert!(multi_chunk_wide_route("gfx1201", &chunk128_flags()));
        // The chunk128 flag off, or another arch, never opens the wide route.
        assert!(!multi_chunk_wide_route("gfx1201", &FeatureFlags::for_test("gfx1201")));
        assert!(!multi_chunk_wide_route("gfx1100", &chunk128_flags()));
        for (name, set) in SINGLETON_ARM_SWITCHES {
            let mut f = chunk128_flags();
            set(&mut f);
            assert!(!multi_chunk_wide_route("gfx1201", &f), "{name} must refuse wide admission");
        }
    }

    #[test]
    fn scratch_rows_equal_the_effective_cap() {
        // Admitted: the planned rows, up to 128.
        assert_eq!(scratch_rows_for(true, MULTI_CHUNK_MAX_ROWS), MULTI_CHUNK_MAX_ROWS);
        assert_eq!(scratch_rows_for(true, 4096), MULTI_CHUNK_MAX_ROWS);
        assert_eq!(scratch_rows_for(true, 96), 96);
        // Not admitted (any refusing switch, a non-dense target, flag off):
        // exactly the product cap, as before the wide route existed.
        for (name, set) in SINGLETON_ARM_SWITCHES {
            let mut f = chunk128_flags();
            set(&mut f);
            let rows = scratch_rows_for(multi_chunk_wide_route("gfx1201", &f), MULTI_CHUNK_MAX_ROWS);
            assert_eq!(rows, MULTI_CHUNK_PRODUCT_MAX_ROWS, "{name}");
        }
        assert_eq!(scratch_rows_for(false, MULTI_CHUNK_MAX_ROWS), MULTI_CHUNK_PRODUCT_MAX_ROWS);
        assert_eq!(scratch_rows_for(false, 32), 32);
        // Never below the batched-prefill minimum.
        assert_eq!(scratch_rows_for(false, 0), 2);
        assert_eq!(scratch_rows_for(true, 1), 2);
    }

    fn pack(rows: &[usize], cap: usize) -> Vec<std::ops::Range<usize>> {
        let mut out = Vec::new();
        pack_whole_lanes(rows, cap, &mut out).expect("packs");
        out
    }

    #[test]
    fn c8_block16_packs_by_effective_cap() {
        let rows = [16usize; 8];
        // Flag off (product cap 63): 48 + 48 + 32.
        assert_eq!(pack(&rows, MULTI_CHUNK_PRODUCT_MAX_ROWS), vec![0..3, 3..6, 6..8]);
        // Cap 64: 64 + 64.
        assert_eq!(pack(&rows, 64), vec![0..4, 4..8]);
        // Cap 128: one chunk.
        assert_eq!(pack(&rows, MULTI_CHUNK_MAX_ROWS), vec![0..8]);
        // A cap above the supported aggregate is clamped to it.
        assert_eq!(pack(&rows, 4096), vec![0..8]);
    }

    #[test]
    fn ragged_totals_pack_whole_lanes_in_order() {
        // 4 x 16 rows then 6 x 15 rows (154 in all).
        let rows = [16, 16, 16, 16, 15, 15, 15, 15, 15, 15];
        assert_eq!(pack(&rows, 64), vec![0..4, 4..8, 8..10]);
        assert_eq!(pack(&rows, 96), vec![0..6, 6..10]);
        assert_eq!(pack(&rows, 128), vec![0..8, 8..10]);
        // A 63-row lane leaves no room for a second lane at cap 64.
        assert_eq!(pack(&[63, 2], 64), vec![0..1, 1..2]);
        assert_eq!(pack(&[63, 63], 128), vec![0..2]);
        // Nothing split, padded or reordered: every lane appears exactly once.
        let r = pack(&[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], 64);
        assert_eq!(r.first().map(|x| x.start), Some(0));
        assert_eq!(r.last().map(|x| x.end), Some(11));
        assert!(r.windows(2).all(|w| w[0].end == w[1].start));
    }

    #[test]
    fn a_lane_over_63_rows_or_under_2_is_refused() {
        let mut out = vec![0..1];
        assert!(pack_whole_lanes(&[16, 64], 128, &mut out).is_err());
        assert!(out.is_empty());
        assert!(pack_whole_lanes(&[1, 16], 128, &mut out).is_err());
        assert!(out.is_empty());
        // At the product cap a 64-row lane is also refused.
        assert!(pack_whole_lanes(&[64], 63, &mut out).is_err());
    }
}
