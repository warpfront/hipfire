// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched (prefill / multi-row verify) mixers of hybrid-decoder layers whose
//! FFN is the sealed MoE program: the gated DeltaNet body and the gated
//! full-attention prep, attend and output projection, plus their MoE-only
//! route gates (exact IU4 producers, fused GDN preamble, indexed Q/K, chunk
//! scan output plane). Moved verbatim from the Qwen3.5 prefill path. They stay
//! separate from the dense executors in [`super::batched_deltanet`] and
//! [`super::batched_attention`]: the output projection always accumulates into
//! the residual, the PARO Givens arms rotate the un-rotated `x_norm_batch`, and
//! the conv / QK-norm launches are the MoE sequence, so merging them would
//! change numerics. The architecture binds the layer's MoE FFN (the sealed
//! program with its CPU expert splice) as the `moe_ffn` callback.

#![allow(clippy::too_many_arguments)]

use crate::context::DispatchCtx;
use crate::families::gemv::WeightRef;
use crate::ops::delta_net::StateQuant;
use crate::pipeline::batched::*;
use crate::pipeline::batched_attention::*;
use crate::pipeline::batched_deltanet::*;
use crate::pipeline::dump_hidden_localize;
use crate::pipeline::hybrid::HybridDims;
use hip_bridge::{HipError, HipResult};
use rdna_compute::norm::GdnScanOut;
use rdna_compute::prefill_route_capabilities::{self, PrefillRoute};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Resident weights of the gated DeltaNet mixer of a MoE layer.
#[derive(Clone, Copy)]
pub struct DeltaNetMoeLayerView<'a> {
    pub attn_norm: &'a GpuTensor,
    pub wqkv: WeightRef<'a>,
    pub wz: WeightRef<'a>,
    pub w_alpha: WeightRef<'a>,
    pub w_beta: WeightRef<'a>,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub conv_weight: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub wo: WeightRef<'a>,
}

/// Resident weights of the gated full-attention mixer of a MoE layer.
#[derive(Clone, Copy)]
pub struct AttentionMoeLayerView<'a> {
    pub attn_norm: &'a GpuTensor,
    pub wq: WeightRef<'a>,
    pub wk: WeightRef<'a>,
    pub wv: WeightRef<'a>,
    pub q_norm: &'a GpuTensor,
    pub k_norm: &'a GpuTensor,
    pub wo: WeightRef<'a>,
}

/// MQ3G256 / MQ3-Lloyd projections have no MoE-batched route (their stride
/// would be read through the HFQ4-layout arms). The architecture's MoE MQ3
/// guard keeps them on the per-token path; refuse them here too.
fn refuse_mq3_projections(site: &str, dtypes: &[DType]) -> HipResult<()> {
    if dtypes
        .iter()
        .any(|d| matches!(d, DType::MQ3G256 | DType::MQ3G256Lloyd))
    {
        return Err(HipError::new(
            0,
            &format!("{site}: MQ3G256 / MQ3G256Lloyd attention weights have no MoE-batched prefill route — refusing (per-token fallback serves this layer)"),
        ));
    }
    Ok(())
}

/// `HIPFIRE_PREFILL_IU4_PRODUCER_FUSED` (developer typed boolean): the
/// explicit value, read once. Unset is `None`; the capability table has raw
/// exact-gfx1201 inventory and a qualified row there, so unset resolves on
/// at exact gfx1201 and off elsewhere.
fn prefill_iu4_producer_fused_explicit() -> Option<bool> {
    static EXPLICIT: std::sync::LazyLock<Option<bool>> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_typed_bool("HIPFIRE_PREFILL_IU4_PRODUCER_FUSED")
    });
    *EXPLICIT
}

/// Pure admission of the qualified exact IU4 producer fusion: the typed flag
/// resolved through the capability table (`explicit` `1` needs the raw
/// inventory of `arch`, `0` is off, unset follows the exact-gfx1201 qualification), a uniform
/// MQ4G256V2 consumer weight, and the existing producer shape/eager prerequisites
/// (`Gpu::iu4_producer_quant_fused_active`).
fn prefill_iu4_producer_fused_admitted(
    explicit: Option<bool>,
    arch: &str,
    weight_is_mq4v2: bool,
    producer_shape_active: bool,
) -> bool {
    weight_is_mq4v2
        && producer_shape_active
        && prefill_route_capabilities::raw_inventory(arch, PrefillRoute::PrefillIu4ProducerFused)
        && prefill_route_capabilities::enabled(
            explicit,
            arch,
            PrefillRoute::PrefillIu4ProducerFused,
        )
}

/// Qualified (`HIPFIRE_PREFILL_IU4_PRODUCER_FUSED`, unset on at exact gfx1201) FWHT-rotate +
/// `block_i4_128` producer for the MoE FullAttn wo input, which has no
/// epilogue param and always accumulates residual into `x_batch`. Uses the
/// exact wrapper (`rotate_x_mq_i4_gfx12_exact_batched`); the legacy
/// `rotate_x_mq_i4_gfx12_batched` of the dense sites is untouched. The f32
/// `x_rot` store is written so every downstream reader is preserved.
fn try_gfx12_rotate_exact_prepared_no_epilogue(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    x: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if !prefill_iu4_producer_fused_admitted(
        prefill_iu4_producer_fused_explicit(),
        &gpu.arch,
        wo.dtype == DType::MQ4G256V2,
        gpu.mq4g256v2_shared_iu4_supported(k, n) && gpu.iu4_producer_quant_fused_active(n, k),
    ) {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep = gpu.rotate_x_mq_i4_gfx12_exact_batched(x, wo.awq_scale, Some(x_rot), res, k, n)?;
    // The residual GEMM family taps the emitted F32 rotated activation before
    // its GEMM; the prepared consumer does not.
    gpu.maybe_capture_activation(wo.buf, x_rot, n, k);
    // The residual GEMM entry notes the F16 receipt before its scope checks;
    // the prepared consumer does not. Exact route only (the gfx11 legacy
    // prepared route never reaches this helper's launch).
    gpu.iu4_verify_note_f16();
    Ok(Some(prep))
}

/// L4: the FA output gate of the MoE finish (`sigmoid_mul_f32(attn, gate)`) folded into the exact C8 PM
/// rotate that consumes it (`HIPFIRE_PREFILL_IU4_C8_PM` nested under the F3 producer fusion). The
/// gated activation is formed per element in registers (the exact `sigmoid_mul_f32` expression), then
/// the identical AWQ / rotate / C8 path runs: the f32 rotated side output, the sidecar and the
/// capture / receipt notes are those of [`try_gfx12_rotate_exact_prepared_no_epilogue`] after the
/// standalone gate. `attn` is not written; nothing after this helper reads the gated f32 attention
/// buffer on the admitted route (the residual GEMM consumes the prepared sidecar, and the rotated
/// f32 side output is still written for every other reader). `None` (nothing launched, the
/// scratch untouched) keeps the unfused chain: the standalone gate, then the legacy / exact
/// rotate producers.
fn try_gfx12_sigmoid_rotate_exact_prepared_no_epilogue(
    gpu: &mut Gpu,
    wo: &WeightRef<'_>,
    attn: &GpuTensor,
    gate: &GpuTensor,
    x_rot: &GpuTensor,
    k: usize,
    n: usize,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    // The unfused chain would take the legacy prepared rotate (`try_iu4_rotate_prepared_no_epilogue`)
    // before the exact one: not this route.
    if wo.dtype == DType::MQ4G256V2 && gpu.iu4_producer_sidecar_active(n, k) {
        return Ok(None);
    }
    if !prefill_iu4_producer_fused_admitted(
        prefill_iu4_producer_fused_explicit(),
        &gpu.arch,
        wo.dtype == DType::MQ4G256V2,
        gpu.mq4g256v2_shared_iu4_supported(k, n) && gpu.iu4_producer_quant_fused_active(n, k),
    ) {
        return Ok(None);
    }
    if !gpu.iu4_c8_pm_sigmoid_rotate_admitted(attn, gate, wo.awq_scale, x_rot, k, n) {
        return Ok(None);
    }
    let res = gpu.reserve_int4_mmq(k, n)?;
    let prep =
        gpu.mq_sigmoid_rotate_x_i4_c8_pm(attn, gate, wo.awq_scale, Some(x_rot), res, k, n)?;
    gpu.maybe_capture_activation(wo.buf, x_rot, n, k);
    gpu.iu4_verify_note_f16();
    Ok(Some(prep))
}

/// Qualified (`HIPFIRE_PREFILL_IU4_PRODUCER_FUSED`, unset on at exact gfx1201) RMSNorm/FWHT +
/// `block_i4_128` producer for the MoE LA qkvza / FA qkv input. The caller
/// passes whether every consumer weight is MQ4G256V2 and the fused key
/// (`FusedQkvzaMq4G256V2` / `FusedQkvMq4G256V2`) the prepared consumer
/// resolves. The producer launches through the fused-QKV family's exact
/// entry; the legacy dense producer is untouched. The f32 `x_rot` store is
/// written so every downstream reader is preserved.
#[allow(clippy::too_many_arguments)]
fn try_gfx12_rmsnorm_exact_prepared(
    gpu: &mut Gpu,
    x: &GpuTensor,
    norm_weight: &GpuTensor,
    next_linear: &WeightRef<'_>,
    x_rot: &GpuTensor,
    k: usize,
    eps: f32,
    n: usize,
    weights_uniform_mq4v2: bool,
    kind: crate::types::KernelKey,
) -> HipResult<Option<rdna_compute::Int4MmqPrepared>> {
    if !prefill_iu4_producer_fused_admitted(
        prefill_iu4_producer_fused_explicit(),
        &gpu.arch,
        weights_uniform_mq4v2 && next_linear.dtype == DType::MQ4G256V2,
        gpu.mq4g256v2_shared_iu4_supported(k, n) && gpu.iu4_producer_quant_fused_active(n, k),
    ) {
        return Ok(None);
    }
    use crate::families::fused_qkv::FusedQkvIu4RmsnormParams;
    let ctx = DispatchCtx::new(gpu);
    let params = FusedQkvIu4RmsnormParams {
        kind,
        x,
        norm_weight,
        awq: next_linear.awq_scale,
        x_rot,
        k,
        eps,
        batch_size: n,
    };
    let prep = FUSED_QKV
        .emit_rmsnorm_iu4_exact(&ctx, gpu, &params)
        .map_err(hip_err)?;
    Ok(Some(prep))
}

/// `HIPFIRE_GDN_MOE_PRE_FUSED` (`developer.gdn_moe_pre_fused`, a typed
/// boolean): the explicit value, read once. `true`/`1` arms the fused MoE GDN
/// preamble, `false`/`0` is the kill switch. Unset (`None`) follows
/// `prefill_route_capabilities` qualification of the running architecture.
/// Admission still requires the dense preamble's own kill switch and gfx1201
/// capability ([`gdn_moe_pre_fused_ratio`]).
fn gdn_moe_pre_fused_explicit() -> Option<bool> {
    static EXPLICIT: std::sync::LazyLock<Option<bool>> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_typed_bool("HIPFIRE_GDN_MOE_PRE_FUSED")
    });
    *EXPLICIT
}

/// Whether the fused MoE GDN preamble is on for `arch`: the explicit
/// `HIPFIRE_GDN_MOE_PRE_FUSED` value, else qualification. Admission is
/// [`gdn_moe_pre_fused_ratio`].
fn gdn_moe_pre_fused_on(arch: &str) -> bool {
    let explicit = gdn_moe_pre_fused_explicit();
    prefill_route_capabilities::enabled(explicit, arch, PrefillRoute::GdnMoePreFused)
}

/// The dense fused-preamble predicate, plus the MoE opt-in and value-head
/// dimension guard. State quantization is deliberately not an input: the
/// preamble writes F32 inputs for every existing GDN state-update route.
#[allow(clippy::too_many_arguments)]
fn gdn_moe_pre_fused_ratio(
    enabled: bool,
    dense_enabled: bool,
    capable: bool,
    sequential: bool,
    tree: bool,
    tape: bool,
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
    value_head_dim: usize,
    rows: usize,
) -> Option<usize> {
    if !enabled
        || !dense_enabled
        || !capable
        || !sequential
        || tree
        || tape
        || key_heads == 0
        || key_heads >= value_heads
        || value_heads % key_heads != 0
        || head_dim != 128
        || value_head_dim != 128
        || rows == 0
    {
        return None;
    }
    Some(value_heads / key_heads)
}

/// `HIPFIRE_GDN_QK_INDEXED` (`developer.gdn_qk_indexed`, a typed boolean:
/// `true`/`false`/`1`/`0`; any other spelling is rejected when the process
/// configuration is built). The explicit value, read once: `true`/`1` arms the
/// indexed sequential Q8 GDN prefill (raw key-head Q/K, no repeat/copy) on any
/// arch, `false`/`0` is the kill switch. Unset (`None`) follows
/// `prefill_route_capabilities` qualification of the running architecture.
fn gdn_qk_indexed_explicit() -> Option<bool> {
    static EXPLICIT: std::sync::LazyLock<Option<bool>> =
        std::sync::LazyLock::new(|| hipfire_config::developer_typed_bool("HIPFIRE_GDN_QK_INDEXED"));
    *EXPLICIT
}

/// Whether the indexed GDN Q/K route is on for `arch`: the explicit
/// `HIPFIRE_GDN_QK_INDEXED` value, else qualification. Admission is
/// [`gdn_qk_indexed_ratio`].
fn gdn_qk_indexed_on(arch: &str) -> bool {
    let explicit = gdn_qk_indexed_explicit();
    prefill_route_capabilities::enabled(explicit, arch, PrefillRoute::GdnQkIndexed)
}

/// Pure admission of the indexed Q8 batch_seq GDN launch in the MoE LA
/// prefill branch. `Some(qk_head_ratio)` (value heads per key head) when the
/// flag is on, the DeltaNet state is Q8, the batch is sequential (neither tree
/// verify nor `Independent`), the key-head count is positive and divides the
/// value-head count (so value heads >= key heads), and both head dims are 128.
/// `None` keeps the repeat/copy plus the existing launches. An admitted fused
/// preamble takes priority because it already writes interleaved Q/K.
#[allow(clippy::too_many_arguments)]
fn gdn_qk_indexed_ratio(
    enabled: bool,
    pre_fused: bool,
    quant: StateQuant,
    tree: bool,
    independent: bool,
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
    value_head_dim: usize,
) -> Option<usize> {
    if !enabled
        || pre_fused
        || quant != StateQuant::Q8
        || tree
        || independent
        || key_heads == 0
        || value_heads < key_heads
        || value_heads % key_heads != 0
        || head_dim != 128
        || value_head_dim != 128
    {
        return None;
    }
    Some(value_heads / key_heads)
}

/// Route of one MoE LA layer's GDN input preamble and scan. The chunk scan
/// supersedes the fused preamble (`HIPFIRE_GDN_MOE_PRE_FUSED`), which
/// supersedes the indexed Q8 launch (`HIPFIRE_GDN_QK_INDEXED`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GdnMoeRoute {
    ChunkScan,
    PreFused,
    QkIndexed,
    Legacy,
}

fn gdn_moe_route(
    chunk_scan: bool,
    pre_fused_ratio: Option<usize>,
    qk_indexed_ratio: Option<usize>,
) -> GdnMoeRoute {
    if chunk_scan {
        GdnMoeRoute::ChunkScan
    } else if pre_fused_ratio.is_some() {
        GdnMoeRoute::PreFused
    } else if qk_indexed_ratio.is_some() {
        GdnMoeRoute::QkIndexed
    } else {
        GdnMoeRoute::Legacy
    }
}

/// Storage of the chunk-scan `out` plane of a MoE LA layer: BF16 only when the
/// gfx1201 IU4 gated-norm producer (the one with an `_xbf16` twin) is admitted
/// for `wo` (`HIPFIRE_GDN_SCAN_OUT` may still force F32), else F32.
fn gdn_moe_chunk_scan_out(
    gpu: &Gpu,
    layer: &DeltaNetMoeLayerView<'_>,
    config: &HybridDims,
    n: usize,
    n_v_heads: usize,
) -> HipResult<GdnScanOut> {
    if !gpu.iu4_verify_scope
        && gfx12_gdn_quant_fused_admitted(
            gpu,
            &layer.wo,
            n_v_heads,
            config.linear_value_dim,
            layer.wo.k,
            n,
            &BatchEpilogue::Residual,
        )
    {
        gpu.gdn_scan_out(GdnScanOut::Bf16)
    } else {
        Ok(GdnScanOut::F32)
    }
}

/// Batched gated DeltaNet mixer of a MoE layer, then the layer's MoE FFN
/// (`moe_ffn`, which reads and accumulates into `pbs.x_batch`).
/// `gdn_chunk_scan_admitted` is the caller's MoE chunk-scan admission for
/// this layer.
pub fn execute_deltanet_moe_layer_batched(
    gpu: &mut Gpu,
    layer: &DeltaNetMoeLayerView<'_>,
    config: &HybridDims,
    pbs: &BatchScratch<'_>,
    dn_state: &DeltaNetStateView<'_>,
    n: usize,
    dim: usize,
    k_dim: usize,
    v_dim: usize,
    n_v_heads: usize,
    hd: usize,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    gdn_tape: Option<&GdnTapeView<'_>>,
    tape_offset: usize,
    delta_layer_idx: usize,
    start_pos: usize,
    layer_idx: usize,
    commit_stride: Option<usize>,
    gdn_chunk_scan_admitted: bool,
    moe_ffn: impl FnOnce(&mut Gpu) -> HipResult<()>,
) -> HipResult<()> {
    // Layer span (opt-in trace only): attention half vs MoE FFN body. Host
    // wall around work this chunk already runs; no new sync, no order change.
    let t_layer = std::time::Instant::now();
    let t_attn = std::time::Instant::now();
    // Batched MoE LA layer. LA body is the same as DeltaNet
    // (rmsnorm + qkvza + sigmoid_alpha + conv1d + L2norm +
    // repeat_interleave + GDN + gated_norm + wo+residual);
    // only the FFN differs. Duplicated inline for now — can
    // be factored into a `prefill_la_body_batched` helper
    // when dense and MoE LA paths are proven byte-exact.
    // This body is unreachable for MQ3 / MQ3-Lloyd weights —
    // the upstream `mq3_in_moe` guard at the top of
    // `forward_prefill_batch_with_pbs` rejects any MoE layer
    // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
    // mirroring the captured-path guard at line 3367+. So
    // `layer.wqkv.dtype` is restricted here to MQ4G256 /
    // HFQ4G256 / MQ6G256 / HFQ6G256 / Q8_0. Q8 admit landed
    // alongside the moe_ffn router/gate Q8 unlock (A3B's LA
    // attention weights are Q8 — engine quantizer keeps q/k/v/o
    // at Q8 alongside the Q8 router + shared_expert_gate).
    let is_mq = matches!(
        layer.wqkv.dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256V2
            | DType::MQ2G256V2
    );
    let is_6bit = matches!(layer.wqkv.dtype, DType::MQ6G256 | DType::HFQ6G256);
    let is_q8 = matches!(layer.wqkv.dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let is_lowbit = matches!(layer.wqkv.dtype, DType::TQ2G128 | DType::BQ1G128);
    // qt=52 has no MoE-batched kernels (grouped/indexed LUT variants don't
    // exist): refuse loudly here rather than falling through to a uniform
    // fused key. Per-token decode serves Lloyd-MoE via generic dispatch paths.
    if matches!(layer.wqkv.dtype, DType::MQ4G256V2Lloyd) {
        return Err(HipError::new(
            0,
            "execute_deltanet_moe_layer_batched: MQ4G256V2Lloyd attention weights have no MoE-batched prefill kernel — refusing (per-token fallback serves this layer)",
        ));
    }
    refuse_mq3_projections(
        "execute_deltanet_moe_layer_batched",
        &[
            layer.wqkv.dtype,
            layer.wz.dtype,
            layer.w_beta.dtype,
            layer.w_alpha.dtype,
            layer.wo.dtype,
        ],
    )?;
    // Phase 1.5: PARO mode for DeltaNetMoe — wqkv/wz are
    // ParoQ4G128 (each with its own Givens rotation tables);
    // w_alpha/w_beta are F32 (no rotation, no quantization).
    // Dispatch is unfused: rotate+gemm_hfq4g128 for wqkv and wz,
    // direct gemm_f32_batched for w_alpha and w_beta. Same shape
    // outputs as the Q8/MQ4 paths (dn_qkv_batch, dn_z_batch,
    // dn_alpha_batch, dn_beta_batch).
    let is_paro = matches!(layer.wqkv.dtype, DType::ParoQ4G128);
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);

    // JIT/load failure and extent errors must precede the projection and the
    // persistent conv-ring mutation; no legacy retry is valid afterwards.
    let gdn_chunk_views = if gdn_chunk_scan_admitted {
        let scan_out = gdn_moe_chunk_scan_out(gpu, layer, config, n, n_v_heads)?;
        // `HIPFIRE_GDN_CHUNK_PREP_PM` (typed; unset follows the exact-gfx1201 qualification, `0` kills) is nested
        // beneath this admitted chunk-scan layer. `gdn_chunk_prep_pm_admitted` adds the raw
        // exact-gfx1201 inventory and the fixed 16/32-head, 1..=8192-row shape; anything
        // else keeps the incumbent `gdn_chunk_prep`. The MoE projection never writes q/k/v
        // targets, so the PM twin replaces exactly the plain prep.
        let prep = if gpu.gdn_chunk_prep_pm_admitted(n, config.linear_key_heads, n_v_heads) {
            GdnChunkPrepModule::Pm
        } else {
            GdnChunkPrepModule::Incumbent
        };
        Some(gdn_chunk_scan_prepare_views(
            gpu,
            pbs,
            dn_state,
            config,
            delta_layer_idx,
            n,
            k_dim,
            v_dim,
            config.linear_key_heads,
            n_v_heads,
            commit_stride,
            scan_out,
            prep,
        )?)
    } else {
        None
    };
    let moe_qkvza_uniform_mq4v2 = [
        layer.wqkv.dtype,
        layer.wz.dtype,
        layer.w_beta.dtype,
        layer.w_alpha.dtype,
    ] == [DType::MQ4G256V2; 4];
    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    if is_mq {
        if moe_qkvza_uniform_mq4v2 {
            iu4_prep = try_gfx12_rmsnorm_exact_prepared(
                gpu,
                pbs.x_batch,
                layer.attn_norm,
                &layer.wqkv,
                pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
                true,
                crate::types::KernelKey::FusedQkvzaMq4G256V2,
            )?;
        }
        if iu4_prep.is_none() {
            // AWQ-aware: next linear is LA's fused wqkv.
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                pbs.x_batch,
                layer.attn_norm,
                &layer.wqkv,
                pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        }
    } else if is_paro {
        // PARO: need un-rotated x_norm available for per-weight
        // Givens rotation. Write rmsnorm into x_norm_batch (the
        // dedicated normalized buffer); x_rot_batch becomes the
        // per-weight rotation scratch (overwritten per GEMM).
        gpu.rmsnorm_batched(
            pbs.x_batch,
            layer.attn_norm,
            pbs.x_norm_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    } else {
        gpu.rmsnorm_batched(
            pbs.x_batch,
            layer.attn_norm,
            pbs.x_rot_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    }
    if let Some(prep) = &iu4_prep {
        // The family taps every constituent weight on the F32 rotated X and
        // notes the F16 receipt once before the prepared consumer.
        use crate::families::fused_qkv::FusedQkvIu4PreparedParams;
        let ctx = DispatchCtx::new(gpu);
        let params = FusedQkvIu4PreparedParams {
            kind: crate::types::KernelKey::FusedQkvzaMq4G256V2,
            weights: &[
                layer.wqkv.buf,
                layer.wz.buf,
                layer.w_beta.buf,
                layer.w_alpha.buf,
            ],
            x_rot: pbs.x_rot_batch,
            prepared: prep,
            outputs: &[
                pbs.dn_qkv_batch,
                pbs.dn_z_batch,
                pbs.dn_beta_batch,
                pbs.dn_alpha_batch,
            ],
            m: &[layer.wqkv.m, layer.wz.m, layer.w_beta.m, layer.w_alpha.m],
            k: layer.wqkv.k,
            batch_size: n,
        };
        FUSED_QKV
            .run_iu4_prepared(&ctx, gpu, &params)
            .map_err(hip_err)?;
    } else if is_paro {
        // PARO 4-way unfused dispatch. wqkv and wz are
        // ParoQ4G128 with their own Givens rotation tables;
        // w_alpha and w_beta are F32 with no rotation.
        let paro_wqkv = layer.wqkv.rotation.as_ref().unwrap_or_else(|| {
            panic!(
                "ParoQ4G128 wqkv missing paro metadata at LA layer {layer_idx} \
                             — load_paroquant_weight() loader regression?"
            )
        });
        let paro_wz = layer.wz.rotation.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wz missing paro metadata at LA layer {layer_idx}")
        });
        // wqkv: rotate x_norm → x_rot, then HFQ4G128 GEMM.
        gpu.givens_rotate_to(
            pbs.x_norm_batch,
            pbs.x_rot_batch,
            &paro_wqkv.pairs,
            &paro_wqkv.theta,
            &paro_wqkv.scales,
            n,
            dim,
            paro_wqkv.krot,
            128,
        )?;
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq4G128,
            layer.wqkv.buf,
            layer.wqkv.dtype,
            pbs.x_rot_batch,
            pbs.dn_qkv_batch,
            layer.wqkv.m,
            layer.wqkv.k,
            n,
        )?;
        // wz: re-rotate x_norm → x_rot (overwrite), then GEMM.
        gpu.givens_rotate_to(
            pbs.x_norm_batch,
            pbs.x_rot_batch,
            &paro_wz.pairs,
            &paro_wz.theta,
            &paro_wz.scales,
            n,
            dim,
            paro_wz.krot,
            128,
        )?;
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq4G128,
            layer.wz.buf,
            layer.wz.dtype,
            pbs.x_rot_batch,
            pbs.dn_z_batch,
            layer.wz.m,
            layer.wz.k,
            n,
        )?;
        // w_alpha / w_beta: F32, no rotation, direct batched GEMM.
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmF32Batched,
            layer.w_alpha.buf,
            layer.w_alpha.dtype,
            pbs.x_norm_batch,
            pbs.dn_alpha_batch,
            layer.w_alpha.m,
            layer.w_alpha.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmF32Batched,
            layer.w_beta.buf,
            layer.w_beta.dtype,
            pbs.x_norm_batch,
            pbs.dn_beta_batch,
            layer.w_beta.m,
            layer.w_beta.k,
            n,
        )?;
    } else if is_6bit {
        run_fused_qkvza_key(
            gpu,
            crate::types::KernelKey::FusedQkvzaHfq6G256,
            layer.wqkv.buf,
            layer.wz.buf,
            layer.w_beta.buf,
            layer.w_alpha.buf,
            pbs.x_rot_batch,
            pbs.dn_qkv_batch,
            pbs.dn_z_batch,
            pbs.dn_beta_batch,
            pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    } else if is_q8 && q8_wmma_arch {
        // Fused Q8 QKVZA WMMA — assumes all 4 weights share Q8_0
        // stride; mixed Q8/other layers within DNMoe are rejected
        // upstream by `moe_ffn_batched_admissible` (router/gate Q8 OK, but
        // shared_expert + experts must be MQ4) and would otherwise
        // re-introduce Tier-1 stride corruption.
        debug_assert!(
            matches!(layer.wz.dtype, DType::Q8_0)
                && matches!(layer.w_beta.dtype, DType::Q8_0)
                && matches!(layer.w_alpha.dtype, DType::Q8_0),
            "DNMoe LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
        );
        run_fused_qkvza_key(
            gpu,
            crate::types::KernelKey::FusedQkvzaQ8_0,
            layer.wqkv.buf,
            layer.wz.buf,
            layer.w_beta.buf,
            layer.w_alpha.buf,
            pbs.x_rot_batch,
            pbs.dn_qkv_batch,
            pbs.dn_z_batch,
            pbs.dn_beta_batch,
            pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    } else if is_q8 || is_lowbit {
        // #397 Ship 5.2 slice1: four plain Q8 batched GEMMs
        // (wqkv/wz/w_beta/w_alpha), sibling DeltaNet QKVZA path.
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wqkv.dtype),
            layer.wqkv.buf,
            layer.wqkv.dtype,
            pbs.x_rot_batch,
            pbs.dn_qkv_batch,
            layer.wqkv.m,
            layer.wqkv.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wz.dtype),
            layer.wz.buf,
            layer.wz.dtype,
            pbs.x_rot_batch,
            pbs.dn_z_batch,
            layer.wz.m,
            layer.wz.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_beta.dtype),
            layer.w_beta.buf,
            layer.w_beta.dtype,
            pbs.x_rot_batch,
            pbs.dn_beta_batch,
            layer.w_beta.m,
            layer.w_beta.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.w_alpha.dtype),
            layer.w_alpha.buf,
            layer.w_alpha.dtype,
            pbs.x_rot_batch,
            pbs.dn_alpha_batch,
            layer.w_alpha.m,
            layer.w_alpha.k,
            n,
        )?;
    } else {
        run_fused_qkvza_key(
            gpu,
            crate::families::fused_qkv::fused_qkvza_key_for(layer.wqkv.dtype),
            layer.wqkv.buf,
            layer.wz.buf,
            layer.w_beta.buf,
            layer.w_alpha.buf,
            pbs.x_rot_batch,
            pbs.dn_qkv_batch,
            pbs.dn_z_batch,
            pbs.dn_beta_batch,
            pbs.dn_alpha_batch,
            layer.wqkv.m,
            layer.wz.m,
            layer.w_beta.m,
            layer.w_alpha.m,
            layer.wqkv.k,
            n,
        )?;
    }
    let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);
    let pre_fused_ratio = gdn_moe_pre_fused_ratio(
        gdn_moe_pre_fused_on(&gpu.arch),
        gpu.flags.gfx12_gdn_pre_fused,
        gpu.arch_caps.is_gfx1201(),
        matches!(batch_semantics, BatchSemantics::Sequential),
        tree_parents.is_some(),
        gdn_tape.is_some(),
        config.linear_key_heads,
        n_v_heads,
        hd,
        config.linear_value_dim,
        n,
    );
    // The fused preamble already materializes value-head Q/K. It wins over
    // QK_INDEXED, which otherwise consumes the normalized raw key-head planes.
    let qk_indexed_ratio = gdn_qk_indexed_ratio(
        gdn_qk_indexed_on(&gpu.arch),
        pre_fused_ratio.is_some(),
        dn_state.quant,
        tree_parents.is_some(),
        batch_semantics.is_independent(),
        config.linear_key_heads,
        n_v_heads,
        hd,
        config.linear_value_dim,
    );
    let chunk_scan = gdn_chunk_views.is_some();
    let moe_route = gdn_moe_route(chunk_scan, pre_fused_ratio, qk_indexed_ratio);
    if let (GdnMoeRoute::ChunkScan, Some(views)) = (moe_route, gdn_chunk_views.as_ref()) {
        // The MoE projection never writes q/k/v targets, so the plain prep runs
        // (`prep_fused = false`); then KKT -> BF16 mseg/segment scan -> gated norm.
        gdn_chunk_scan_layer(
            gpu,
            views,
            pbs,
            dn_state,
            delta_layer_idx,
            layer.conv_weight,
            layer.dt_bias,
            layer.a_log,
            n,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            false,
            config.linear_key_heads,
            n_v_heads,
        )?;
        return deltanet_moe_output_batched(
            gpu,
            layer,
            config,
            pbs,
            n,
            n_v_heads,
            q8_wmma_arch,
            layer_idx,
            views.scan_out,
            t_layer,
            t_attn,
            moe_ffn,
        );
    }
    if let Some(ratio) = pre_fused_ratio {
        gpu.gdn_pre_batched_gfx1201(
            pbs.dn_beta_batch,
            pbs.dn_alpha_batch,
            layer.dt_bias,
            layer.a_log,
            pbs.dn_qkv_batch,
            layer.conv_weight,
            &dn_state.conv_states[delta_layer_idx],
            pbs.dn_q_raw_batch,
            pbs.dn_k_raw_batch,
            pbs.dn_v_batch,
            pbs.dn_q_batch,
            pbs.dn_k_batch,
            n_v_heads,
            config.linear_key_heads,
            ratio,
            k_dim,
            v_dim,
            n,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
        )?;
    } else {
        gpu.fused_sigmoid_alpha_gate_f32_batched(
            pbs.dn_beta_batch,
            pbs.dn_alpha_batch,
            layer.dt_bias,
            layer.a_log,
            n_v_heads,
            n,
        )?;
        // The tape forces this arm (the fused preamble and chunk scan refuse
        // it), so the DFlash/MTP rollback replays exactly these rows.
        if let Some(tape) = gdn_tape.as_ref() {
            let qkv_row_bytes = tape.qkv_dim * 4;
            let alpha_row_bytes = n_v_heads * 4;
            let off_qkv = tape_offset * qkv_row_bytes;
            let off_a = tape_offset * alpha_row_bytes;
            let copy_qkv = n * qkv_row_bytes;
            let copy_a = n * alpha_row_bytes;
            gpu.memcpy_dtod_at_auto(
                &tape.qkv_bufs[delta_layer_idx].buf,
                off_qkv,
                &pbs.dn_qkv_batch.buf,
                0,
                copy_qkv,
            )?;
            gpu.memcpy_dtod_at_auto(
                &tape.alpha_bufs[delta_layer_idx].buf,
                off_a,
                &pbs.dn_alpha_batch.buf,
                0,
                copy_a,
            )?;
            gpu.memcpy_dtod_at_auto(
                &tape.beta_bufs[delta_layer_idx].buf,
                off_a,
                &pbs.dn_beta_batch.buf,
                0,
                copy_a,
            )?;
        }
        // Same tree-aware dispatch gate as dense LA branch above.
        if let Some(parents) = tree_parents {
            gpu.conv1d_silu_split_tree_f32_n(
                pbs.dn_q_raw_batch,
                pbs.dn_k_raw_batch,
                pbs.dn_v_batch,
                pbs.dn_qkv_batch,
                layer.conv_weight,
                &dn_state.conv_states[delta_layer_idx],
                parents,
                k_dim,
                v_dim,
                n,
            )?;
        } else if let BatchSemantics::Independent { active_mask, .. } = batch_semantics {
            let full_mask = valid_lane_mask(n)?;
            if active_mask == full_mask {
                gpu.conv1d_silu_split_f32_independent(
                    pbs.dn_q_raw_batch,
                    pbs.dn_k_raw_batch,
                    pbs.dn_v_batch,
                    pbs.dn_qkv_batch,
                    layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim,
                    v_dim,
                    n,
                )?;
            } else {
                gpu.conv1d_silu_split_f32_independent_masked(
                    pbs.dn_q_raw_batch,
                    pbs.dn_k_raw_batch,
                    pbs.dn_v_batch,
                    pbs.dn_qkv_batch,
                    layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim,
                    v_dim,
                    n,
                    active_mask,
                )?;
            }
        } else {
            gpu.conv1d_silu_split_f32_n(
                pbs.dn_q_raw_batch,
                pbs.dn_k_raw_batch,
                pbs.dn_v_batch,
                pbs.dn_qkv_batch,
                layer.conv_weight,
                &dn_state.conv_states[delta_layer_idx],
                k_dim,
                v_dim,
                n,
            )?;
        }
        gpu.fused_qk_l2_norm_scale_f32_batched(
            pbs.dn_q_raw_batch,
            pbs.dn_k_raw_batch,
            config.linear_key_heads,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            n,
        )?;
        if qk_indexed_ratio.is_some() {
            // Raw planes are consumed in place; nothing to repeat or copy.
        } else if config.linear_key_heads < n_v_heads {
            let ratio = n_v_heads / config.linear_key_heads;
            gpu.repeat_interleave_qk_f32_batched(
                pbs.dn_q_raw_batch,
                pbs.dn_k_raw_batch,
                pbs.dn_q_batch,
                pbs.dn_k_batch,
                config.linear_key_heads,
                ratio,
                hd,
                n,
            )?;
        } else {
            gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
            gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
        }
    }
    // DIAG: dump GDN inputs (batched, MoE branch)
    if layer_idx == 0 {
        // Dump the buffers the GDN launch actually reads: the raw key-head
        // planes (key-head width) on the indexed route, else the repeated
        // value-head planes.
        let (dump_q, dump_k, qk_dim) = if qk_indexed_ratio.is_some() {
            (
                pbs.dn_q_raw_batch,
                pbs.dn_k_raw_batch,
                config.linear_key_heads * hd,
            )
        } else {
            (pbs.dn_q_batch, pbs.dn_k_batch, n_v_heads * hd)
        };
        dump_hidden_localize(gpu, dump_q, n, start_pos, qk_dim, 0, "q_b");
        dump_hidden_localize(gpu, dump_k, n, start_pos, qk_dim, 0, "k_b");
        dump_hidden_localize(
            gpu,
            pbs.dn_alpha_batch,
            n,
            start_pos,
            n_v_heads,
            0,
            "alpha_b",
        );
        dump_hidden_localize(gpu, pbs.dn_beta_batch, n, start_pos, n_v_heads, 0, "beta_b");
    }
    // Same conv/norm dispatch as dense LA branch above (skipped only when the
    // fused preamble already wrote the interleaved Q/K above). The indexed
    // route also skips the repeat/copy: the raw key-head planes are consumed
    // in place by the indexed Q8 launch below.
    if let Some(parents) = tree_parents {
        // MoE-path tree-verify GDN, dispatched by state quant
        // (mirror of the dense path above).
        match dn_state.quant {
            StateQuant::FP32 => {
                let tape_f32 = pbs.dn_s_tape_f32.expect(
                                "FP32 tree-aware LA requires dn_s_tape_f32 scratch (check PrefillBatchScratch::new)",
                            );
                gpu.gated_delta_net_f32_tree_batch_seq(
                    pbs.dn_q_batch,
                    pbs.dn_k_batch,
                    pbs.dn_v_batch,
                    pbs.dn_alpha_batch,
                    pbs.dn_beta_batch,
                    &dn_state.s_matrices[delta_layer_idx],
                    tape_f32,
                    parents,
                    pbs.dn_attn_out_batch,
                    n,
                    n_v_heads,
                    config.linear_value_dim,
                )?;
            }
            StateQuant::Q8 => {
                let tape_q8 = pbs
                    .dn_s_tape_q8
                    .expect("tree-aware LA requires dn_s_tape_q8 scratch");
                let tape_sc = pbs
                    .dn_s_tape_scales
                    .expect("tree-aware LA requires dn_s_tape_scales scratch");
                gpu.gated_delta_net_q8_tree_batch_seq(
                    pbs.dn_q_batch,
                    pbs.dn_k_batch,
                    pbs.dn_v_batch,
                    pbs.dn_alpha_batch,
                    pbs.dn_beta_batch,
                    &dn_state.s_matrices[delta_layer_idx],
                    &dn_state.s_scales[delta_layer_idx],
                    tape_q8,
                    tape_sc,
                    parents,
                    pbs.dn_attn_out_batch,
                    n,
                    n_v_heads,
                    config.linear_value_dim,
                )?;
            }
            StateQuant::Q4 => {
                return Err(HipError::new(
                    0,
                    "Q4 DeltaNet state + tree-verify (DDTree) is unsupported: \
                                 there is no Q4 tree-tape GDN kernel. Use Q8 or FP32 state \
                                 for tree spec-decode.",
                ));
            }
        }
    } else {
        match dn_state.quant {
            StateQuant::FP32 => {
                if rdna_compute::norm::gdn_chunked() && n > 1 {
                    gpu.gated_delta_net_f32_chunked(
                        pbs.dn_q_batch,
                        pbs.dn_k_batch,
                        pbs.dn_v_batch,
                        pbs.dn_alpha_batch,
                        pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_dim,
                        rdna_compute::norm::gdn_chunk_size(),
                    )?
                } else {
                    gpu.gated_delta_net_f32_batch_seq(
                        pbs.dn_q_batch,
                        pbs.dn_k_batch,
                        pbs.dn_v_batch,
                        pbs.dn_alpha_batch,
                        pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_dim,
                    )?
                }
            }
            StateQuant::Q8 => {
                if let BatchSemantics::Independent { active_mask, .. } = batch_semantics {
                    let full_mask = valid_lane_mask(n)?;
                    if active_mask == full_mask {
                        gpu.gated_delta_net_q8_independent(
                            pbs.dn_q_batch,
                            pbs.dn_k_batch,
                            pbs.dn_v_batch,
                            pbs.dn_alpha_batch,
                            pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_dim,
                            dn_state.ef_residual(delta_layer_idx),
                        )?
                    } else {
                        gpu.gated_delta_net_q8_independent_masked(
                            pbs.dn_q_batch,
                            pbs.dn_k_batch,
                            pbs.dn_v_batch,
                            pbs.dn_alpha_batch,
                            pbs.dn_beta_batch,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx],
                            pbs.dn_attn_out_batch,
                            n,
                            n_v_heads,
                            config.linear_value_dim,
                            dn_state.ef_residual(delta_layer_idx),
                            active_mask,
                        )?
                    }
                } else if let Some(ratio) = qk_indexed_ratio {
                    // Raw (un-repeated) Q/K: the kernel maps value head h to
                    // key head h / ratio. `qk_indexed_ratio` is Some only for
                    // Q8 + sequential + no tree, so this is the same launch as
                    // the arm below minus the repeat/copy.
                    gpu.gated_delta_net_q8_batch_seq_indexed(
                        pbs.dn_q_raw_batch,
                        pbs.dn_k_raw_batch,
                        pbs.dn_v_batch,
                        pbs.dn_alpha_batch,
                        pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_dim,
                        dn_state.ef_residual(delta_layer_idx),
                        ratio,
                    )?
                } else {
                    gpu.gated_delta_net_q8_batch_seq(
                        pbs.dn_q_batch,
                        pbs.dn_k_batch,
                        pbs.dn_v_batch,
                        pbs.dn_alpha_batch,
                        pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_dim,
                        dn_state.ef_residual(delta_layer_idx),
                    )?
                }
            }
            StateQuant::Q4 => gpu.gated_delta_net_q4(
                pbs.dn_q_batch,
                pbs.dn_k_batch,
                pbs.dn_v_batch,
                pbs.dn_alpha_batch,
                pbs.dn_beta_batch,
                &dn_state.s_matrices[delta_layer_idx],
                &dn_state.s_scales[delta_layer_idx],
                pbs.dn_attn_out_batch,
                n,
                n_v_heads,
                config.linear_value_dim,
            )?,
        }
        // DIAG: dump GDN attention output at layer 0
        if layer_idx == 0 {
            dump_hidden_localize(
                gpu,
                pbs.dn_attn_out_batch,
                n,
                start_pos,
                n_v_heads * config.linear_value_dim,
                0,
                "gdn_b",
            );
        }
    }
    deltanet_moe_output_batched(
        gpu,
        layer,
        config,
        pbs,
        n,
        n_v_heads,
        q8_wmma_arch,
        layer_idx,
        GdnScanOut::F32,
        t_layer,
        t_attn,
        moe_ffn,
    )
}

/// Output half of a batched MoE LA layer: gated output norm, `wo` residual,
/// then the MoE FFN body. `scan_out` is the storage of `dn_attn_out_batch`:
/// `Bf16` is accepted only when the gfx1201 IU4 gated-norm producer (which
/// has the `_xbf16` twin) is admitted for `wo`, else the layer fails closed.
#[allow(clippy::too_many_arguments)]
fn deltanet_moe_output_batched(
    gpu: &mut Gpu,
    layer: &DeltaNetMoeLayerView<'_>,
    config: &HybridDims,
    pbs: &BatchScratch<'_>,
    n: usize,
    n_v_heads: usize,
    q8_wmma_arch: bool,
    layer_idx: usize,
    scan_out: GdnScanOut,
    t_layer: std::time::Instant,
    t_attn: std::time::Instant,
    moe_ffn: impl FnOnce(&mut Gpu) -> HipResult<()>,
) -> HipResult<()> {
    // slices-4: whole gated_norm+rotate+quant chain in one producer
    // (`_gfx12` entries on gfx1201, `_gfx11` entries on gfx1100/gfx1151).
    // This MoE path always accumulates residual into x_batch, so the
    // Residual-only admission is structural (same as the no-epilogue C2
    // helper below). When live, the standalone gated_norm_f32 store is
    // skipped; the f32 `x_rot` store is skipped and only the iu4 sidecar
    // is emitted by the fused producer.
    let mut gdn_fused_prep = try_gfx12_gdn_quant_fused_prepared(
        gpu,
        &layer.wo,
        pbs.dn_attn_out_batch,
        scan_out,
        pbs.dn_z_batch,
        layer.norm_weight,
        pbs.dn_normed_rot_batch,
        n_v_heads,
        config.linear_value_dim,
        config.norm_eps,
        layer.wo.k,
        n,
        &BatchEpilogue::Residual,
    )?;
    if scan_out != GdnScanOut::F32 && gdn_fused_prep.is_none() {
        return Err(HipError::new(
            0,
            "execute_deltanet_moe_layer_batched: bf16 GDN scan plane without an _xbf16 producer",
        ));
    }
    if gdn_fused_prep.is_none() {
        gdn_fused_prep = try_gfx11_gdn_quant_fused_prepared_no_epilogue(
            gpu,
            &layer.wo,
            pbs.dn_attn_out_batch,
            pbs.dn_z_batch,
            layer.norm_weight,
            n_v_heads,
            config.linear_value_dim,
            config.norm_eps,
            layer.wo.k,
            n,
        )?;
    }
    if gdn_fused_prep.is_none() {
        gpu.gated_norm_f32_batched(
            pbs.dn_attn_out_batch,
            pbs.dn_z_batch,
            layer.norm_weight,
            pbs.dn_normed_batch,
            n_v_heads,
            config.linear_value_dim,
            config.norm_eps,
            n,
        )?;
    }
    // wo + residual. Q8 wo lands un-rotated (Q8 weights were
    // quantized against un-rotated activations); MQ4/MQ6 wo
    // require FWHT(awq_scale-adjusted) rotation. Mirrors the
    // dense LA wo dispatch (qwen35.rs:5000-5043) — the MQ6
    // branch is required for AWQ A3B where 4/40 LA layers
    // ship MQ6 wo and would otherwise corrupt the residual
    // stream when dispatched through the HFQ4 kernel against
    // 200 B/group MQ6-layout bytes.
    let dn_wo_is_q8 = matches!(layer.wo.dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let dn_wo_is_lowbit = matches!(layer.wo.dtype, DType::TQ2G128 | DType::BQ1G128);
    let dn_wo_is_6bit = matches!(layer.wo.dtype, DType::MQ6G256 | DType::HFQ6G256);
    let dn_wo_is_paro = matches!(layer.wo.dtype, DType::ParoQ4G128);
    // T-B: fused rotate+quantize (always-Residual here — no epilogue param).
    // Skipped when the slices-4 GDN producer already emitted both outputs.
    let mut iu4_wo_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    if gdn_fused_prep.is_none() {
        iu4_wo_prep = try_iu4_rotate_prepared_no_epilogue(
            gpu,
            &layer.wo,
            pbs.dn_normed_batch,
            layer.wo.k,
            n,
        )?;
    }
    let dn_wo_input = if gdn_fused_prep.is_some() || iu4_wo_prep.is_some() {
        pbs.dn_normed_rot_batch
    } else if dn_wo_is_q8 {
        pbs.dn_normed_batch
    } else if dn_wo_is_paro {
        // PARO wo: rotate dn_normed by wo's own Givens tables
        // into dn_normed_rot_batch. Same scratch layout as MQ4
        // (since dn_normed_rot_batch is unused on the Q8 path).
        let paro_wo = layer.wo.rotation.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wo missing paro metadata at LA layer {layer_idx}")
        });
        gpu.givens_rotate_to(
            pbs.dn_normed_batch,
            pbs.dn_normed_rot_batch,
            &paro_wo.pairs,
            &paro_wo.theta,
            &paro_wo.scales,
            n,
            layer.wo.k,
            paro_wo.krot,
            128,
        )?;
        pbs.dn_normed_rot_batch
    } else {
        // F2: AWQ-aware rotate for linear_attn wo (out_proj) input.
        rotate_x_mq_batched_for(
            gpu,
            &layer.wo,
            pbs.dn_normed_batch,
            pbs.dn_normed_rot_batch,
            layer.wo.k,
            n,
        )?;
        pbs.dn_normed_rot_batch
    };
    if dn_wo_is_6bit {
        run_residual_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq6G256Residual,
            layer.wo.buf,
            layer.wo.dtype,
            dn_wo_input,
            pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else if dn_wo_is_q8 && q8_wmma_arch {
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        run_residual_gemm_key(
            gpu,
            crate::types::KernelKey::GemmQ8_0ResidualWmma,
            layer.wo.buf,
            layer.wo.dtype,
            dn_wo_input,
            &x_n,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else if dn_wo_is_q8 || dn_wo_is_lowbit {
        // Non-WMMA Q8: gemm into a scratch then add into x_batch.
        // Reuse `dn_normed_rot_batch` (free since the MQ4 rotate
        // path didn't run here) as the GEMM scratch.
        let scratch = pbs.dn_normed_rot_batch.sub_offset(0, n * layer.wo.m);
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wo.dtype),
            layer.wo.buf,
            layer.wo.dtype,
            dn_wo_input,
            &scratch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        gpu.add_inplace_f32(&x_n, &scratch)?;
    } else if dn_wo_is_paro {
        // PARO wo residual: HFQ4G128 batched GEMM into scratch,
        // then add into x_batch. Reuse x_norm_batch (free at
        // this point — used earlier for the QKVZA stage; not
        // needed for the rest of this layer) as the scratch.
        let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq4G128,
            layer.wo.buf,
            layer.wo.dtype,
            dn_wo_input,
            &scratch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        gpu.add_inplace_f32(&x_n, &scratch)?;
    } else if let Some(prep) = gdn_fused_prep.as_ref().or(iu4_wo_prep.as_ref()) {
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(
            layer.wo.buf,
            prep,
            pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else {
        run_residual_gemm_key(
            gpu,
            crate::families::gemm::residual_gemm_key_for(layer.wo.dtype),
            layer.wo.buf,
            layer.wo.dtype,
            dn_wo_input,
            pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    }
    // Batched MoE FFN replaces the dense (rmsnorm + gate+up +
    // silu_mul + w_down) block. Takes pbs.x_batch as input AND
    // accumulates the FFN output residual back into it via the
    // batched indexed down kernel's atomicAdd path.
    // Attn half ends here; MoE FFN body (with its own window span) below.
    let attn_ns = t_attn.elapsed().as_nanos() as u64;
    let t_moe = std::time::Instant::now();
    moe_ffn(gpu)?;
    let moe_ns = t_moe.elapsed().as_nanos() as u64;
    let total_ns = t_layer.elapsed().as_nanos() as u64;
    if crate::moe_trace_enabled() {
        crate::moe_layer_trace(
            layer_idx,
            false,
            crate::MoeLayerTiming {
                total_ns,
                attn_ns,
                moe_ns,
                rows: n,
            },
        );
    }

    Ok(())
}

/// One batched MoE attention layer: [`attention_moe_prep_batched`], the
/// attend and output projection of [`attention_moe_finish_batched`], then the
/// layer's MoE FFN (`moe_ffn`). Admits the A3B F5 PM Q-prep for this chunk.
pub fn execute_attention_moe_layer_batched(
    gpu: &mut Gpu,
    fa_attn_multirow: bool,
    layer: &AttentionMoeLayerView<'_>,
    config: &HybridDims,
    pbs: &BatchScratch<'_>,
    s: &FlashScratch<'_>,
    kv_cache: &KvView<'_>,
    n: usize,
    dim: usize,
    start_pos: usize,
    max_ctx_len: usize,
    ctx: &DispatchCtx,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    q8_wmma_arch: bool,
    layer_idx: usize,
    tap: Option<BatchedAttentionTap<'_>>,
    moe_ffn: impl FnOnce(&mut Gpu) -> HipResult<()>,
) -> HipResult<()> {
    // Layer span (opt-in trace only): attention half vs MoE FFN body. Host
    // wall around work this chunk already runs; no new sync, no order change.
    // `t_attn` starts here so it covers the prep half (QKV projection +
    // norms + RoPE, the largest GPU piece) plus the finish's attend tail;
    // `attn_ns` closes just before the finish's MoE FFN body.
    let t_layer = std::time::Instant::now();
    let t_attn = std::time::Instant::now();
    // F2 split: QKV projection + norms + RoPE (everything above the old
    // attend call) live in `attention_moe_prep_batched`; called here
    // for the single-chunk path in the original position.
    let gfx12_fa_prep_fp8q = (config.n_heads, config.n_kv_heads) == (16, 2)
        && !ctx.verify_tile_attend
        && gfx12_fa_prep_fp8q_admitted(
            gpu,
            config,
            kv_cache,
            s,
            n,
            start_pos,
            max_ctx_len,
            batch_semantics,
            tree_verify,
            fa_attn_multirow,
            DflashFusionCtx::Off,
            None,
            tap.is_some(),
        );
    let fa_pm_qprep = !gfx12_fa_prep_fp8q
        && fa_pm_qprep_admitted(
            gpu,
            config,
            kv_cache,
            s,
            n,
            start_pos,
            max_ctx_len,
            batch_semantics,
            tree_verify,
            tap.is_some(),
        );
    attention_moe_prep_batched(
        gpu,
        layer,
        config,
        pbs,
        n,
        dim,
        kv_cache,
        tree_verify,
        layer_idx,
        fa_pm_qprep,
        gfx12_fa_prep_fp8q,
        tap,
    )?;
    let attn_head_ns = t_attn.elapsed().as_nanos() as u64;
    // F2 split: KV-write + flash attention, sigmoid/wo and the MoE FFN now
    // live in `attention_moe_finish_batched` so a chunk pair can share
    // one merged attend step; called here for the single-chunk path in the
    // original position.
    attention_moe_finish_batched(
        gpu,
        fa_attn_multirow,
        layer,
        config,
        pbs,
        s,
        kv_cache,
        n,
        start_pos,
        max_ctx_len,
        ctx,
        batch_semantics,
        tree_verify,
        q8_wmma_arch,
        layer_idx,
        fa_pm_qprep,
        gfx12_fa_prep_fp8q,
        moe_ffn,
    )?;
    // Single-chunk path only: the merged-pair path calls prep/finish directly
    // and stays untimed. `attn` = prep head (QKV/norms/RoPE); the finish's
    // attend-tail + MoE body are inside `finish` and not separately timed
    // here, so `moe` = total − head (reconciles standalone; precise per-layer
    // MoE comes from the `moe window` lines).
    let total_ns = t_layer.elapsed().as_nanos() as u64;
    if crate::moe_trace_enabled() {
        crate::moe_layer_trace(
            layer_idx,
            true,
            crate::MoeLayerTiming {
                total_ns,
                attn_ns: attn_head_ns,
                moe_ns: total_ns.saturating_sub(attn_head_ns),
                rows: n,
            },
        );
    }
    Ok(())
}

/// F2 split of `execute_attention_moe_layer_batched`: QKV projection (all
/// dtype arms) + deinterleave/Q-norm/K-norm + triattn tap + RoPE. Same
/// statements, same order, same launches as the inlined head. The F2 chunk
/// pair calls this per half before its merged attend step.
pub fn attention_moe_prep_batched(
    gpu: &mut Gpu,
    layer: &AttentionMoeLayerView<'_>,
    config: &HybridDims,
    pbs: &BatchScratch<'_>,
    n: usize,
    dim: usize,
    kv_cache: &KvView<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    layer_idx: usize,
    fa_pm_qprep: bool,
    gfx12_fa_prep_fp8q: bool,
    tap: Option<BatchedAttentionTap<'_>>,
) -> HipResult<()> {
    // Batched MoE FA layer. FA body is the same as FullAttn
    // (rmsnorm + qkv + deinterleave + q/k norm + RoPE +
    // kv_write + attention + sigmoid_mul + wo+residual);
    // only the FFN differs. Duplicated inline — will be
    // consolidated with the dense FA batched body once the
    // MoE path is proven byte-exact.
    // This body is unreachable for MQ3 / MQ3-Lloyd weights —
    // the upstream `mq3_in_moe` guard at the top of
    // `forward_prefill_batch_with_pbs` rejects any MoE layer
    // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
    // mirroring the captured-path guard at line 3367+. So
    // `layer.wq.dtype` is restricted to MQ4G256 / HFQ4G256
    // / MQ6G256 / HFQ6G256 here. Adding MQ3 to the matcher AND
    // the QKV dispatch is insufficient — the wo path below
    // (line 5320) is hardcoded MQ4 too — so the all-or-nothing
    // wiring lives in a separate PR (see followup issue).
    let qkv_is_mq = matches!(
        layer.wq.dtype,
        DType::MQ4G256
            | DType::MQ4G256V2
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256V2
            | DType::MQ2G256V2
    );
    let qkv_is_6bit = matches!(layer.wq.dtype, DType::MQ6G256 | DType::HFQ6G256);
    let qkv_is_q8 = matches!(layer.wq.dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let qkv_is_lowbit = matches!(layer.wq.dtype, DType::TQ2G128 | DType::BQ1G128);
    // qt=52 has no MoE-batched kernels (grouped/indexed LUT variants don't
    // exist): refuse loudly here rather than falling through to a uniform
    // fused key. Per-token decode serves Lloyd-MoE via generic dispatch paths.
    if matches!(layer.wq.dtype, DType::MQ4G256V2Lloyd) {
        return Err(HipError::new(
            0,
            "attention_moe_prep_batched: MQ4G256V2Lloyd attention weights have no MoE-batched prefill kernel — refusing (per-token fallback serves this layer)",
        ));
    }
    refuse_mq3_projections(
        "attention_moe_prep_batched",
        &[
            layer.wq.dtype,
            layer.wk.dtype,
            layer.wv.dtype,
            layer.wo.dtype,
        ],
    )?;
    // Phase 1.6 (PARO FullAttnMoe): wq/wk/wv are ParoQ4G128
    // (each with its own Givens rotation tables). The fused-QKV
    // kernels can't handle this — they assume one shared
    // rotation. Unfused 3-way dispatch (rotate + gemm_hfq4g128
    // per projection) matches the LA QKVZA Phase 1.5 pattern.
    let qkv_is_paro = matches!(layer.wq.dtype, DType::ParoQ4G128);
    // Fused QKV requires uniform dtype — see issue #249 for
    // the dense FA variant. Gate the same way here.
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);
    let qkv_same_dtype = layer.wk.dtype == layer.wq.dtype && layer.wv.dtype == layer.wq.dtype;

    let mut iu4_prep: Option<rdna_compute::Int4MmqPrepared> = None;
    if qkv_is_mq {
        if qkv_same_dtype {
            iu4_prep = try_gfx12_rmsnorm_exact_prepared(
                gpu,
                pbs.x_batch,
                layer.attn_norm,
                &layer.wq,
                pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
                true,
                crate::types::KernelKey::FusedQkvMq4G256V2,
            )?;
        }
        if iu4_prep.is_none() {
            // AWQ-aware: next linear is wq (Q/K/V share input → same AWQ scale).
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                pbs.x_batch,
                layer.attn_norm,
                &layer.wq,
                pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        }
    } else if qkv_is_paro {
        // PARO: rmsnorm into x_norm_batch (un-rotated). x_rot_batch
        // is reused as the per-weight rotation scratch.
        gpu.rmsnorm_batched(
            pbs.x_batch,
            layer.attn_norm,
            pbs.x_norm_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    } else {
        gpu.rmsnorm_batched(
            pbs.x_batch,
            layer.attn_norm,
            pbs.x_rot_batch,
            n,
            dim,
            config.norm_eps,
        )?;
    }
    if let Some(prep) = &iu4_prep {
        // The family taps every constituent weight on the F32 rotated X and
        // notes the F16 receipt once before the prepared consumer.
        use crate::families::fused_qkv::FusedQkvIu4PreparedParams;
        let ctx = DispatchCtx::new(gpu);
        let params = FusedQkvIu4PreparedParams {
            kind: crate::types::KernelKey::FusedQkvMq4G256V2,
            weights: &[layer.wq.buf, layer.wk.buf, layer.wv.buf],
            x_rot: pbs.x_rot_batch,
            prepared: prep,
            outputs: &[pbs.fa_q_full_batch, pbs.fa_k_batch, pbs.fa_v_batch],
            m: &[layer.wq.m, layer.wk.m, layer.wv.m],
            k: layer.wq.k,
            batch_size: n,
        };
        FUSED_QKV
            .run_iu4_prepared(&ctx, gpu, &params)
            .map_err(hip_err)?;
    } else if qkv_is_paro {
        // PARO 3-way unfused dispatch (wq, wk, wv each with own
        // Givens rotation). Same shape outputs as the fused
        // paths: fa_q_full_batch, fa_k_batch, fa_v_batch.
        let paro_wq = layer.wq.rotation.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wq missing paro metadata at FA layer {layer_idx}")
        });
        let paro_wk = layer.wk.rotation.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wk missing paro metadata at FA layer {layer_idx}")
        });
        let paro_wv = layer.wv.rotation.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wv missing paro metadata at FA layer {layer_idx}")
        });
        // wq
        gpu.givens_rotate_to(
            pbs.x_norm_batch,
            pbs.x_rot_batch,
            &paro_wq.pairs,
            &paro_wq.theta,
            &paro_wq.scales,
            n,
            dim,
            paro_wq.krot,
            128,
        )?;
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq4G128,
            layer.wq.buf,
            layer.wq.dtype,
            pbs.x_rot_batch,
            pbs.fa_q_full_batch,
            layer.wq.m,
            layer.wq.k,
            n,
        )?;
        // wk
        gpu.givens_rotate_to(
            pbs.x_norm_batch,
            pbs.x_rot_batch,
            &paro_wk.pairs,
            &paro_wk.theta,
            &paro_wk.scales,
            n,
            dim,
            paro_wk.krot,
            128,
        )?;
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq4G128,
            layer.wk.buf,
            layer.wk.dtype,
            pbs.x_rot_batch,
            pbs.fa_k_batch,
            layer.wk.m,
            layer.wk.k,
            n,
        )?;
        // wv
        gpu.givens_rotate_to(
            pbs.x_norm_batch,
            pbs.x_rot_batch,
            &paro_wv.pairs,
            &paro_wv.theta,
            &paro_wv.scales,
            n,
            dim,
            paro_wv.krot,
            128,
        )?;
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq4G128,
            layer.wv.buf,
            layer.wv.dtype,
            pbs.x_rot_batch,
            pbs.fa_v_batch,
            layer.wv.m,
            layer.wv.k,
            n,
        )?;
    } else if qkv_is_6bit && qkv_same_dtype {
        run_fused_qkv_key(
            gpu,
            crate::types::KernelKey::FusedQkvHfq6G256,
            layer.wq.buf,
            layer.wk.buf,
            layer.wv.buf,
            pbs.x_rot_batch,
            pbs.fa_q_full_batch,
            pbs.fa_k_batch,
            pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else if qkv_is_q8 && q8_wmma_arch && qkv_same_dtype {
        debug_assert!(
            matches!(layer.wk.dtype, DType::Q8_0) && matches!(layer.wv.dtype, DType::Q8_0),
            "FAMoe qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
        );
        run_fused_qkv_key(
            gpu,
            crate::types::KernelKey::FusedQkvQ8_0,
            layer.wq.buf,
            layer.wk.buf,
            layer.wv.buf,
            pbs.x_rot_batch,
            pbs.fa_q_full_batch,
            pbs.fa_k_batch,
            pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else if (qkv_is_q8 || qkv_is_lowbit) && qkv_same_dtype {
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wq.dtype),
            layer.wq.buf,
            layer.wq.dtype,
            pbs.x_rot_batch,
            pbs.fa_q_full_batch,
            layer.wq.m,
            layer.wq.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wk.dtype),
            layer.wk.buf,
            layer.wk.dtype,
            pbs.x_rot_batch,
            pbs.fa_k_batch,
            layer.wk.m,
            layer.wk.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wv.dtype),
            layer.wv.buf,
            layer.wv.dtype,
            pbs.x_rot_batch,
            pbs.fa_v_batch,
            layer.wv.m,
            layer.wv.k,
            n,
        )?;
    } else if qkv_same_dtype {
        run_fused_qkv_key(
            gpu,
            crate::families::fused_qkv::fused_qkv_key_for(layer.wq.dtype),
            layer.wq.buf,
            layer.wk.buf,
            layer.wv.buf,
            pbs.x_rot_batch,
            pbs.fa_q_full_batch,
            pbs.fa_k_batch,
            pbs.fa_v_batch,
            layer.wq.m,
            layer.wk.m,
            layer.wv.m,
            layer.wq.k,
            n,
        )?;
    } else {
        // Mixed-format fallback (issue #249). batched_gemm_single_weight
        // covers MQ4/HFQ4 + MQ6/HFQ6 + Q8_0; mixed-Q8/MQ4 within FAMoe
        // routes here.
        batched_gemm_single_weight(gpu, &layer.wq, pbs.x_rot_batch, pbs.fa_q_full_batch, n)?;
        batched_gemm_single_weight(gpu, &layer.wk, pbs.x_rot_batch, pbs.fa_k_batch, n)?;
        batched_gemm_single_weight(gpu, &layer.wv, pbs.x_rot_batch, pbs.fa_v_batch, n)?;
    }
    if gfx12_fa_prep_fp8q {
        // Both planes belong to this PBS until its attend completes; no
        // shared GPU scratch, Q16 rounding, or post-prep F32 Q reads.
        let bytes = n * config.n_heads * (config.head_dim + 4);
        let q_codes = GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(pbs.fa_q_batch.buf.as_ptr(), bytes) },
            shape: vec![bytes],
            dtype: DType::Raw,
        };
        gpu.qwen35_fa_prep_batched_gfx1201(
            pbs.fa_q_full_batch,
            rdna_compute::qwen35_fa_batch::FaPrepQOut::Fp8Codes(&q_codes),
            Some(pbs.fa_gate_batch),
            pbs.fa_k_batch,
            layer.q_norm,
            layer.k_norm,
            pbs.positions,
            config.norm_eps,
            config.rope_theta,
            kv_cache.compact_offset as i32,
            config.n_heads,
            config.n_kv_heads,
            n,
        )?;
        return Ok(());
    }
    // Fused deinterleave + Q-rmsnorm — one launch instead of two, no Q
    // global-memory round trip. K-norm, triattn tap, and RoPE below are
    // unchanged.
    gpu.deinterleave_q_rmsnorm_f32_batched(
        pbs.fa_q_full_batch,
        pbs.fa_q_batch,
        pbs.fa_gate_batch,
        layer.q_norm,
        config.n_heads,
        config.head_dim,
        n,
        config.norm_eps,
    )?;
    gpu.rmsnorm_batched(
        pbs.fa_k_batch,
        layer.k_norm,
        pbs.fa_k_batch,
        n * config.n_kv_heads,
        config.head_dim,
        config.norm_eps,
    )?;
    if let Some(tap) = tap {
        tap(gpu, layer_idx, pbs.fa_q_batch, pbs.fa_k_batch, n)?;
    }
    if fa_pm_qprep {
        // A3B F5 (admitted by the single-chunk wrapper only): K rmsnorm, K-only RoPE with
        // angle capture and the PM Q prep replace the deinterleave/Q-norm/K-norm/RoPE below;
        // the triattn tap is excluded by admission.
        batch_chunk_fa_pm_qprep_prep(gpu, layer.q_norm, layer.k_norm, config, pbs, kv_cache, n)?;
        return Ok(());
    }
    let n_rot = config.n_rot;
    // pos_offset = compact_offset (absolute RoPE phase post-eviction);
    // pbs.positions stays physical for the KV-write. 0 when no compaction.
    // 39aa358: in DDTree verify, rotate at DEPTH positions instead
    // (correct sibling phases); KV write below stays physical.
    let rope_pos_buf = if tree_verify.is_some() {
        pbs.rope_positions
    } else {
        pbs.positions
    };
    gpu.rope_partial_interleaved_f32_batched(
        pbs.fa_q_batch,
        pbs.fa_k_batch,
        rope_pos_buf,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        n_rot,
        config.rope_theta,
        n,
        kv_cache.compact_offset as i32,
    )?;
    Ok(())
}

/// F2 split of `execute_attention_moe_layer_batched`: KV-write + flash
/// attention, sigmoid/wo residual and the batched MoE FFN (`moe_ffn`). Same
/// statements, same order, same launches as the inlined tail. The F2 chunk
/// pair calls the prep for both halves, its merged attend step, then this
/// finish per half; the finish runs its own attend, so it rewrites the
/// merged attend's output (as on beta).
pub fn attention_moe_finish_batched(
    gpu: &mut Gpu,
    fa_attn_multirow: bool,
    layer: &AttentionMoeLayerView<'_>,
    config: &HybridDims,
    pbs: &BatchScratch<'_>,
    s: &FlashScratch<'_>,
    kv_cache: &KvView<'_>,
    n: usize,
    start_pos: usize,
    max_ctx_len: usize,
    ctx: &DispatchCtx,
    batch_semantics: BatchSemantics<'_>,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    q8_wmma_arch: bool,
    layer_idx: usize,
    fa_pm_qprep: bool,
    gfx12_fa_prep_fp8q: bool,
    moe_ffn: impl FnOnce(&mut Gpu) -> HipResult<()>,
) -> HipResult<()> {
    // (Layer timers live in the parent `execute_attention_moe_layer_batched`: `t_attn`
    // covers the prep half plus this attend tail; `attn_ns` closes below.)
    // Batched KV write + flash attention (via dispatch).
    if fa_pm_qprep {
        // A3B F5: the prep half already produced the final f16 Q, the gate and the angle
        // table; write the Q8 K/V rows and run the PM direct attention (admitted for this
        // exact chunk by the single-chunk wrapper; never set by the paired-chunk path).
        batch_chunk_fa_pm_qprep_direct(gpu, config, pbs, kv_cache, n, max_ctx_len, layer_idx)?;
    } else {
        batch_chunk_fa_attend(
            gpu,
            config,
            pbs,
            s,
            kv_cache,
            n,
            start_pos,
            max_ctx_len,
            ctx,
            batch_semantics,
            tree_verify,
            layer_idx,
            fa_attn_multirow,
            None, // commit_stride: MoE finish keeps legacy cadence
            gfx12_fa_prep_fp8q,
            false,
        )?;
    }
    // L4: the C8 PM route folds the gate into the wo rotate producer below; otherwise the standalone gate.
    let folded_wo_prep = try_gfx12_sigmoid_rotate_exact_prepared_no_epilogue(
        gpu,
        &layer.wo,
        pbs.fa_attn_out_batch,
        pbs.fa_gate_batch,
        pbs.fa_attn_out_rot_batch,
        layer.wo.k,
        n,
    )?;
    if folded_wo_prep.is_none() {
        gpu.sigmoid_mul_f32(pbs.fa_attn_out_batch, pbs.fa_gate_batch)?;
    }
    // wo + residual. Mirrors the dense FA wo dispatch at
    // qwen35.rs:5591-5623 — Q8 wo skips rotation (un-rotated
    // input expected); MQ4/MQ6 wo apply FWHT(awq_scale-adjusted).
    // MQ6 branch added alongside MQ6_ADMIT (without it, MQ6 wo
    // bytes get fed to gemm_hfq4g256_residual which reads them
    // as 136 B/group HFQ4 layout vs the actual 200 B/group MQ6
    // — catastrophic stride mismatch produces a single-token
    // attractor on AWQ A3B's 4/40 FA layers with MQ6 wo).
    let fa_wo_is_q8 = matches!(layer.wo.dtype, DType::Q8_0);
    // TQ2G128/BQ1G128 have no fused qkvza/gate_up/qkv kernel, so they take
    // the same UNFUSED plain-GEMM strategy as Q8 rather than falling through
    // to the HFQ4 arm, which would read these packed blocks at the wrong
    // stride and produce fluent-but-wrong tokens.
    let fa_wo_is_lowbit = matches!(layer.wo.dtype, DType::TQ2G128 | DType::BQ1G128);
    let fa_wo_is_6bit = matches!(layer.wo.dtype, DType::MQ6G256 | DType::HFQ6G256);
    // Phase 1.6 (PARO FullAttnMoe wo): own Givens rotation table,
    // 72 B/group HFQ4G128 layout. Rotate fa_attn_out_batch by wo's
    // paro into fa_attn_out_rot_batch, then HFQ4G128 GEMM into a
    // scratch, then add into x_batch.
    let fa_wo_is_paro = matches!(layer.wo.dtype, DType::ParoQ4G128);
    // T-B: fused rotate+quantize (always-Residual here — no epilogue param).
    let mut iu4_wo_prep = match folded_wo_prep {
        Some(prep) => Some(prep),
        None => try_iu4_rotate_prepared_no_epilogue(
            gpu,
            &layer.wo,
            pbs.fa_attn_out_batch,
            layer.wo.k,
            n,
        )?,
    };
    if iu4_wo_prep.is_none() {
        iu4_wo_prep = try_gfx12_rotate_exact_prepared_no_epilogue(
            gpu,
            &layer.wo,
            pbs.fa_attn_out_batch,
            pbs.fa_attn_out_rot_batch,
            layer.wo.k,
            n,
        )?;
    }
    let fa_wo_input = if iu4_wo_prep.is_some() {
        pbs.fa_attn_out_rot_batch
    } else if fa_wo_is_q8 {
        pbs.fa_attn_out_batch
    } else if fa_wo_is_paro {
        let paro_wo = layer.wo.rotation.as_ref().unwrap_or_else(|| {
            panic!("ParoQ4G128 wo missing paro metadata at FA layer {layer_idx}")
        });
        gpu.givens_rotate_to(
            pbs.fa_attn_out_batch,
            pbs.fa_attn_out_rot_batch,
            &paro_wo.pairs,
            &paro_wo.theta,
            &paro_wo.scales,
            n,
            layer.wo.k,
            paro_wo.krot,
            128,
        )?;
        pbs.fa_attn_out_rot_batch
    } else {
        // F2: AWQ-aware rotate for FullAttention wo (o_proj) input.
        rotate_x_mq_batched_for(
            gpu,
            &layer.wo,
            pbs.fa_attn_out_batch,
            pbs.fa_attn_out_rot_batch,
            layer.wo.k,
            n,
        )?;
        pbs.fa_attn_out_rot_batch
    };
    if fa_wo_is_6bit {
        run_residual_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq6G256Residual,
            layer.wo.buf,
            layer.wo.dtype,
            fa_wo_input,
            pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else if fa_wo_is_q8 && q8_wmma_arch {
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        run_residual_gemm_key(
            gpu,
            crate::types::KernelKey::GemmQ8_0ResidualWmma,
            layer.wo.buf,
            layer.wo.dtype,
            fa_wo_input,
            &x_n,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else if fa_wo_is_q8 || fa_wo_is_lowbit {
        // Non-WMMA Q8: GEMM into a scratch then add into x_batch.
        // Reuse `fa_attn_out_rot_batch` (free since MQ4 rotate
        // didn't run here) as scratch.
        let scratch = pbs.fa_attn_out_rot_batch.sub_offset(0, n * layer.wo.m);
        run_plain_gemm_key(
            gpu,
            plain_gemm_key_for(layer.wo.dtype),
            layer.wo.buf,
            layer.wo.dtype,
            fa_wo_input,
            &scratch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        gpu.add_inplace_f32(&x_n, &scratch)?;
    } else if fa_wo_is_paro {
        // PARO wo residual: HFQ4G128 batched GEMM into scratch,
        // then add into x_batch. Reuse x_norm_batch (free since
        // QKVZA is done — the MoE FFN body below rewrites it
        // as its first action) as the gemm output scratch.
        let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
        run_plain_gemm_key(
            gpu,
            crate::types::KernelKey::GemmHfq4G128,
            layer.wo.buf,
            layer.wo.dtype,
            fa_wo_input,
            &scratch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
        let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
        gpu.add_inplace_f32(&x_n, &scratch)?;
    } else if let Some(prep) = &iu4_wo_prep {
        gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(
            layer.wo.buf,
            prep,
            pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    } else {
        run_residual_gemm_key(
            gpu,
            crate::families::gemm::residual_gemm_key_for(layer.wo.dtype),
            layer.wo.buf,
            layer.wo.dtype,
            fa_wo_input,
            pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            n,
        )?;
    }

    // Batched MoE FFN (parent `execute_attention_moe_layer_batched` owns the layer
    // span on the single-chunk path; the pair path calls this untimed).
    moe_ffn(gpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MoE executors refuse MQ3G256 / MQ3-Lloyd anywhere in a mixer's
    /// projections and admit every other projection dtype.
    #[test]
    fn moe_mixers_refuse_mq3_projections() {
        for mq3 in [DType::MQ3G256, DType::MQ3G256Lloyd] {
            for slot in 0..4 {
                let mut dtypes = [DType::MQ4G256V2; 4];
                dtypes[slot] = mq3;
                assert!(
                    refuse_mq3_projections("t", &dtypes).is_err(),
                    "{mq3:?} at {slot}"
                );
            }
        }
        for ok in [
            DType::MQ4G256,
            DType::MQ3G256V2,
            DType::Q8_0,
            DType::ParoQ4G128,
        ] {
            assert!(refuse_mq3_projections("t", &[ok; 5]).is_ok(), "{ok:?}");
        }
    }

    #[test]
    fn gdn_moe_pre_fused_admission_truth_table() {
        let admit = |enabled, dense, capable, sequential, tree, tape, kh, vh, hd, vhd, rows| {
            gdn_moe_pre_fused_ratio(
                enabled, dense, capable, sequential, tree, tape, kh, vh, hd, vhd, rows,
            )
        };
        // Every combination of opt-in, dense kill switch, capability, batch
        // semantics, tree and tape. Only the all-admitted row may fuse.
        for mask in 0u8..64 {
            let enabled = mask & 1 != 0;
            let dense = mask & 2 != 0;
            let capable = mask & 4 != 0;
            let sequential = mask & 8 != 0;
            let tree = mask & 16 != 0;
            let tape = mask & 32 != 0;
            let fused = admit(
                enabled, dense, capable, sequential, tree, tape, 16, 32, 128, 128, 64,
            );
            assert_eq!(
                fused,
                if enabled && dense && capable && sequential && !tree && !tape {
                    Some(2)
                } else {
                    None
                },
                "mask={mask:06b}"
            );
            for quant in [StateQuant::Q8, StateQuant::FP32, StateQuant::Q4] {
                for indexed in [false, true] {
                    let route = gdn_qk_indexed_ratio(
                        indexed,
                        fused.is_some(),
                        quant,
                        tree,
                        !sequential,
                        16,
                        32,
                        128,
                        128,
                    );
                    let want = indexed
                        && fused.is_none()
                        && quant == StateQuant::Q8
                        && !tree
                        && sequential;
                    assert_eq!(route, if want { Some(2) } else { None });
                }
            }
        }
        for (kh, vh, hd, vhd, rows, want) in [
            (16, 32, 128, 128, 1, Some(2)),
            (16, 32, 128, 128, 512, Some(2)),
            (16, 32, 128, 128, 4096, Some(2)),
            (16, 48, 128, 128, 64, Some(3)),
            (0, 32, 128, 128, 64, None),
            (16, 16, 128, 128, 64, None),
            (32, 16, 128, 128, 64, None),
            (12, 32, 128, 128, 64, None),
            (16, 32, 64, 128, 64, None),
            (16, 32, 128, 64, 64, None),
            (16, 32, 256, 256, 64, None),
            (16, 32, 128, 128, 0, None),
        ] {
            assert_eq!(
                admit(true, true, true, true, false, false, kh, vh, hd, vhd, rows),
                want,
                "kh={kh} vh={vh} hd={hd} vhd={vhd} rows={rows}"
            );
        }
    }

    /// `HIPFIRE_GDN_QK_INDEXED` admission: `enabled` is the default-ON flag
    /// (`false`/`0` is the kill switch); when on only the sequential, non-tree,
    /// Q8, hd128, divisible-head shape takes the indexed launch; everything
    /// else keeps the repeat/copy.
    #[test]
    fn gdn_qk_indexed_admission_truth_table() {
        // A3B-class shape: 16 key heads, 32 value heads, hd 128.
        let admit = |enabled, quant, tree, independent, kh, vh, hd, vhd| {
            gdn_qk_indexed_ratio(enabled, false, quant, tree, independent, kh, vh, hd, vhd)
        };
        let q8 = StateQuant::Q8;
        assert_eq!(admit(true, q8, false, false, 16, 32, 128, 128), Some(2));
        assert_eq!(admit(true, q8, false, false, 16, 48, 128, 128), Some(3));
        // Equal head counts: ratio 1 still skips the copy.
        assert_eq!(admit(true, q8, false, false, 32, 32, 128, 128), Some(1));
        // Default OFF.
        assert_eq!(admit(false, q8, false, false, 16, 32, 128, 128), None);
        // Other state formats keep their launches.
        assert_eq!(
            admit(true, StateQuant::FP32, false, false, 16, 32, 128, 128),
            None
        );
        assert_eq!(
            admit(true, StateQuant::Q4, false, false, 16, 32, 128, 128),
            None
        );
        // Tree verify and Independent batches keep the repeat.
        assert_eq!(admit(true, q8, true, false, 16, 32, 128, 128), None);
        assert_eq!(admit(true, q8, false, true, 16, 32, 128, 128), None);
        // Head-shape guards: zero, more key than value heads, non-divisible.
        assert_eq!(admit(true, q8, false, false, 0, 32, 128, 128), None);
        assert_eq!(admit(true, q8, false, false, 32, 16, 128, 128), None);
        assert_eq!(admit(true, q8, false, false, 12, 32, 128, 128), None);
        // hd / value_hd must both be 128.
        assert_eq!(admit(true, q8, false, false, 16, 32, 64, 128), None);
        assert_eq!(admit(true, q8, false, false, 16, 32, 128, 64), None);
        assert_eq!(admit(true, q8, false, false, 16, 32, 256, 256), None);
    }

    /// `HIPFIRE_PREFILL_IU4_PRODUCER_FUSED` admission truth table: unset follows the
    /// exact-gfx1201 qualification and `0` is always off; `1` needs the capability
    /// inventory of the arch; both need an MQ4G256V2 weight and the producer shape/eager
    /// prerequisites.
    #[test]
    fn prefill_iu4_producer_fused_admission_truth_table() {
        for arch in ["gfx1201", "gfx1200", "gfx1100", "gfx1151", "gfx942", ""] {
            for explicit in [None, Some(false), Some(true)] {
                for mask in 0u8..4 {
                    let mq4v2 = mask & 1 != 0;
                    let shape = mask & 2 != 0;
                    let got = prefill_iu4_producer_fused_admitted(explicit, arch, mq4v2, shape);
                    let on = match explicit {
                        None => arch == "gfx1201",
                        Some(explicit) => explicit,
                    };
                    let want = on && arch == "gfx1201" && mq4v2 && shape;
                    assert_eq!(got, want, "arch={arch} explicit={explicit:?} mask={mask}");
                }
            }
        }
    }

    /// Chunk scan supersedes the fused preamble, which supersedes the indexed
    /// Q8 launch.
    #[test]
    fn gdn_moe_route_chunk_scan_supersedes_pre_fused_and_qk_indexed() {
        // A3B-class shape with the fused preamble admitted: it already wins
        // over the indexed launch inside the pure indexed predicate.
        let pre_fused =
            gdn_moe_pre_fused_ratio(true, true, true, true, false, false, 16, 32, 128, 128, 64);
        assert_eq!(pre_fused, Some(2));
        let q8 = StateQuant::Q8;
        let indexed_behind_fused =
            gdn_qk_indexed_ratio(true, true, q8, false, false, 16, 32, 128, 128);
        assert_eq!(indexed_behind_fused, None);
        let indexed = gdn_qk_indexed_ratio(true, false, q8, false, false, 16, 32, 128, 128);
        assert_eq!(indexed, Some(2));
        for pre_fused in [None, Some(2)] {
            for indexed in [None, Some(2)] {
                assert_eq!(
                    gdn_moe_route(true, pre_fused, indexed),
                    GdnMoeRoute::ChunkScan
                );
            }
        }
        assert_eq!(
            gdn_moe_route(false, Some(2), Some(2)),
            GdnMoeRoute::PreFused
        );
        assert_eq!(gdn_moe_route(false, Some(2), None), GdnMoeRoute::PreFused);
        assert_eq!(gdn_moe_route(false, None, Some(3)), GdnMoeRoute::QkIndexed);
        assert_eq!(gdn_moe_route(false, None, None), GdnMoeRoute::Legacy);
    }
}
