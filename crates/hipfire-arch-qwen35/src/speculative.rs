// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative decoding infrastructure for hipfire.
//!
//! Phase 1: holds target + draft model slots side-by-side on a single shared
//! `Gpu`. The actual speculative decode loop (draft → verify → accept) lives
//! in `spec_loop` once Phase 2 lands. For now, each slot just supports
//! independent forward passes so we can validate that loading two models at
//! once works and that both produce coherent output.
//!
//! Both slots share the same `Gpu` instance — HIP kernels run serialized on
//! the default stream, and the MQ rotation scratch buffers on `Gpu` are reused
//! across calls. This is correct as long as we never have two in-flight GEMVs
//! on different models sharing the same MQ scratch (which we won't, since
//! speculative decode serializes draft-generate then target-verify).

use crate::carrier::Qwen35Bundle;
use crate::dflash_spec::{DenseTpDflashRankState, DflashState};
use crate::dflash_verify_pm4::{
    fingerprint_u64, DflashVerifyBinding, DflashVerifyPm4, DflashVerifyPm4Phase, DflashVerifyRoute,
    DflashVerifyWindow,
};
use crate::qwen35::forward::{
    forward_prefill_dense_tp, forward_prefill_dense_tp_verify_capture, DenseTpSpecCapture,
};
use crate::qwen35::prefill::fp8_verify_target;
use crate::qwen35::{self, DeltaNetState, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use hip_bridge::{DeviceBuffer, HipError, HipResult, Stream};
use hipfire_dispatch::families::kv_tier::KTier;
use hipfire_dispatch::pipeline::batched::{verify_dense_math, DenseBatchMath};
use hipfire_runtime::dflash::{self, DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{self, KvCache};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tokenizer::{Tokenizer, TokenizerError};
use hipfire_runtime::tp_shard::ShardConfig;
use rdna_compute::dflash_sampled_accept::DflashAcceptScratch;
use rdna_compute::dflash_state_copy::{DflashStateCopyDesc, DFLASH_STATE_BULK_COPY_MAX_ITEMS};
use rdna_compute::{DType, Fp8VerifyTarget, Gpu, GpuTensor};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// #397 Ship 5.3: route a single spec-decode (DFlash) batched GEMM through
/// [`GemmFamily::run_key`](hipfire_dispatch::families::gemm::GemmFamily::run_key)
/// against an *explicit* dispatcher-entry [`KernelKey`].
///
/// Behavior-preserving migration primitive for the draft/verify lm_head GEMM
/// call sites in this module — the spec-decode analogue of qwen35.rs's
/// `run_plain_gemm_key`. Passing the dispatcher-entry key
/// (`GemmQ8_0BatchedChunked`, `GemmQ8_0Batched`, `GemmHfq4G256`,
/// `GemmHfq4G256BatchedLmhead`, `GemmHfq3G256BatchedLmhead`,
/// `GemmHfq6G256BatchedLmhead`) makes `run_key` dispatch to the IDENTICAL
/// `gpu.gemm_*` method the direct call used, so each method's own internal arch
/// routing (WMMA for batch>1 on gfx11/gfx12, dp4a on gfx906, fp16/scalar
/// fallback otherwise) is preserved byte-for-byte on every (dtype × arch ×
/// shape). All keys used here are registered `ArchPredicate::Always`, so
/// `run_key`'s registry check never rejects on a supported build. `resolve()`
/// is deliberately NOT used (it would front-run the kernel's internal dispatch
/// with a dtype-keyed WMMA preference and could diverge on some arches). The
/// weight buffer, `x`, output `y`, and m/k/n are passed in the IDENTICAL order
/// the prior direct call used.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_spec_gemm_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: rdna_compute::DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemm::GemmParams;
    use hipfire_dispatch::families::gemv::WeightRef;
    use std::cell::RefCell;
    // Cache the DispatchCtx per thread instead of rebuilding it on every call.
    // `DispatchCtx::new` lowers the immutable process snapshot into FeatureFlags
    // plus ArchCaps and ResourceManager; its own doc says it is "resolved once
    // ... and shared immutably across all dispatch calls". This helper is hit
    // ~170×/generation by the DFlash verify+draft
    // lm_head loop, so reconstructing the ctx per call cost ~9 tok/s at constant
    // τ (gfx1201, 27B AWQ DFlash). `gemm_family()` is already a OnceLock
    // singleton; this brings the ctx in line. Keyed on `gpu.arch` so a thread
    // that drives a different arch (multi-GPU) rebuilds rather than reusing stale.
    thread_local! {
        static SPEC_CTX: RefCell<Option<(String, DispatchCtx)>> = const { RefCell::new(None) };
    }
    let w = WeightRef {
        buf: w_buf,
        dtype: w_dtype,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let params = GemmParams {
        w: &w,
        x,
        y,
        batch_size: n,
    };
    SPEC_CTX.with(|cell| {
        let needs_rebuild = {
            let slot = cell.borrow();
            slot.as_ref().map_or(true, |(arch, _)| arch != &gpu.arch)
        };
        if needs_rebuild {
            let fresh = DispatchCtx::new(gpu);
            *cell.borrow_mut() = Some((gpu.arch.clone(), fresh));
        }
        let slot = cell.borrow();
        let ctx = &slot.as_ref().unwrap().1;
        hipfire_runtime::llama::gemm_family()
            .run_key(key, ctx, gpu, &params)
            .map_err(hip_bridge::HipError::from)
    })
}

fn dflash_gemm_q8_lmhead(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    if gpu.flags.dflash_q8_lmhead_wmma {
        // #397 Ship 5.3: GemmQ8_0BatchedChunked routes to the identical
        // gpu.gemm_q8_0_batched_chunked the prior direct call used.
        return run_spec_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
            &w_out.buf,
            w_out.gpu_dtype,
            x,
            y,
            w_out.m,
            w_out.k,
            n,
        );
    }

    // The dispatch key is the scalar Q8 kernel chunked at its 64-row cap
    // (`rdna_compute::gemm::q8_0_batched_chunks`); pass all `n` rows.
    run_spec_gemm_key(
        gpu,
        hipfire_dispatch::types::KernelKey::GemmQ8_0Batched,
        &w_out.buf,
        w_out.gpu_dtype,
        x,
        y,
        w_out.m,
        w_out.k,
        n,
    )
}

/// Shared MQ{2,3,4,5,6}G256V2 batched lm_head launcher used by the five
/// draft/verify call sites below. Keeps a single 5-way match so each new
/// speculative path does not re-list the five direct MQ-v2 batched-lmhead
/// Gpu entry points.
#[inline]
fn gemm_mq_batched_lmhead(
    gpu: &mut Gpu,
    dtype: rdna_compute::DType,
    w_buf: &GpuTensor,
    rotated: &GpuTensor,
    logits: &GpuTensor,
    m: usize,
    k: usize,
    batch: usize,
) -> HipResult<()> {
    match dtype {
        rdna_compute::DType::MQ4G256V2 => {
            gpu.gemm_mq4g256v2_batched_lmhead(w_buf, rotated, logits, m, k, batch)
        }
        rdna_compute::DType::MQ6G256V2 => {
            gpu.gemm_mq6g256v2_batched_lmhead(w_buf, rotated, logits, m, k, batch)
        }
        rdna_compute::DType::MQ5G256V2 => {
            gpu.gemm_mq5g256v2_batched_lmhead(w_buf, rotated, logits, m, k, batch)
        }
        rdna_compute::DType::MQ3G256V2 => {
            gpu.gemm_mq3g256v2_batched_lmhead(w_buf, rotated, logits, m, k, batch)
        }
        rdna_compute::DType::MQ2G256V2 => {
            gpu.gemm_mq2g256v2_batched_lmhead(w_buf, rotated, logits, m, k, batch)
        }
        other => unreachable!("gemm_mq_batched_lmhead: unexpected dtype {other:?}"),
    }
}

/// [`gemm_mq_batched_lmhead`] for the DFlash **draft** head: first the draft
/// GEMM layer's overwrite route ([`dflash::draft_lm_head_overwrite`], exact
/// gfx1201 MQ4G256V2, byte-identical to the memset + residual pair it
/// replaces; `HIPFIRE_DRAFT_COLLAPSE_OFF=1` vetoes it), else the shared
/// launcher.
#[inline]
#[allow(clippy::too_many_arguments)]
fn gemm_mq_batched_lmhead_draft(
    gpu: &mut Gpu,
    dtype: rdna_compute::DType,
    w_buf: &GpuTensor,
    rotated: &GpuTensor,
    logits: &GpuTensor,
    m: usize,
    k: usize,
    batch: usize,
) -> HipResult<()> {
    if dflash::draft_lm_head_overwrite(gpu, dtype, w_buf, rotated, logits, m, k, batch)? {
        return Ok(());
    }
    gemm_mq_batched_lmhead(gpu, dtype, w_buf, rotated, logits, m, k, batch)
}

fn dflash_moe_verify_graph_lmhead_enabled_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        }
        None => true,
    }
}

pub(crate) fn dflash_moe_verify_graph_lmhead_eligible(
    num_experts: usize,
    want_full_logits: bool,
    tree_verify_present: bool,
    env_value: Option<&str>,
) -> bool {
    // MoE-only: dense DFlash keeps the old forward-only verify graph. The
    // extended graph is also greedy-only because sampling needs full logits.
    num_experts > 0
        && !want_full_logits
        && !tree_verify_present
        && dflash_moe_verify_graph_lmhead_enabled_from_env_value(env_value)
}

fn dflash_moe_draft_ffn_graph_eligible(
    num_experts: usize,
    ctx_slice_present: bool,
    pld_present: bool,
    use_temp_sampling: bool,
    env_value: Option<&str>,
) -> bool {
    // The draft FFN graph captures fixed-B per-layer work. Keep it off for
    // dense, PLD, ctx-slice diagnostics, and sampling until each path is
    // separately benched/coherence-gated.
    num_experts > 0
        && !ctx_slice_present
        && !pld_present
        && !use_temp_sampling
        && dflash_moe_verify_graph_lmhead_enabled_from_env_value(env_value)
}

/// Whether `HIPFIRE_VERIFY_GRAPH` admits the HipGraph verify route for this
/// (arch, target lm_head dtype) pair.
///
/// `dc105ea64` newly admitted MQ2/3/4/5/6G256V2 to gfx1100 batched WMMA, and
/// `verify_graph_ok` shares that eligibility via `prefill_batch_pbs_eligible`.
/// Graph-off direct and forced-blob direct full-model V2 fixtures pass on
/// gfx1100, but two graph-on V2 campaigns lost the endpoint. Default-off graph
/// quarantines (direct batched HIP/WMMA remains on):
/// - exact gfx1100 + MQ*V2
/// - exact gfx1100 + legacy MQ4G256 (measured direct HIP faster)
/// - exact gfx1201 + legacy MQ4G256 (measured graph replay slower than direct)
///
/// `HIPFIRE_VERIFY_GRAPH=1` opts back in diagnostically; `=0` force-offs
/// everywhere; all other arch/dtype pairs stay default-on.
pub(crate) fn dflash_verify_graph_env_eligible(
    arch: &str,
    output_dtype: rdna_compute::DType,
    env_value: Option<&str>,
) -> bool {
    if env_value == Some("0") {
        return false;
    }
    let is_mq_v2 = matches!(
        output_dtype,
        rdna_compute::DType::MQ4G256V2
            | rdna_compute::DType::MQ6G256V2
            | rdna_compute::DType::MQ5G256V2
            | rdna_compute::DType::MQ3G256V2
            | rdna_compute::DType::MQ2G256V2
    );
    if arch == "gfx1100" && is_mq_v2 {
        return env_value == Some("1");
    }
    if arch == "gfx1100" && output_dtype == rdna_compute::DType::MQ4G256 {
        // Direct batched HIP is faster for the measured Qwen MQ4 workload;
        // keep graph available as an explicit diagnostic opt-in.
        return env_value == Some("1");
    }
    if arch == "gfx1201" && output_dtype == rdna_compute::DType::MQ4G256 {
        // Paired 11.5K DFlash runs on R9700 show graph replay slower than the
        // direct tiled-flash path; keep it available as an explicit diagnostic.
        return env_value == Some("1");
    }
    true
}

pub(crate) fn dflash_batched_lm_head_supported(dtype: rdna_compute::DType) -> bool {
    matches!(
        dtype,
        rdna_compute::DType::Q8_0
            | rdna_compute::DType::HFQ4G256
            | rdna_compute::DType::MQ4G256
            | rdna_compute::DType::MQ4G256V2
            | rdna_compute::DType::MQ6G256V2
            | rdna_compute::DType::MQ5G256V2
            | rdna_compute::DType::MQ3G256V2
            | rdna_compute::DType::MQ2G256V2
            | rdna_compute::DType::MQ3G256
            | rdna_compute::DType::HFQ6G256
            | rdna_compute::DType::MQ6G256
    )
}

pub(crate) fn dflash_enqueue_verify_lm_head(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    final_hidden: &GpuTensor,
    verify_scratch: &VerifyScratch,
    b: usize,
    vocab: usize,
    math: DenseBatchMath,
) -> HipResult<()> {
    if math == DenseBatchMath::SingletonWmma && w_out.gpu_dtype != rdna_compute::DType::MQ4G256V2 {
        return Err(HipError::new(
            0,
            &format!(
                "DFlash verify lm_head: SingletonWmma verify math requires a dense MQ4G256V2 head, got {:?}",
                w_out.gpu_dtype
            ),
        ));
    }
    let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
    match w_out.gpu_dtype {
        rdna_compute::DType::Q8_0 => {
            dflash_gemm_q8_lmhead(gpu, w_out, final_hidden, &logits_batch, b)?;
        }
        rdna_compute::DType::HFQ4G256 => {
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                final_hidden,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::MQ4G256 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rot,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::MQ4G256V2
        | rdna_compute::DType::MQ6G256V2
        | rdna_compute::DType::MQ5G256V2
        | rdna_compute::DType::MQ3G256V2
        | rdna_compute::DType::MQ2G256V2 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ v2 lm_head: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            if math == DenseBatchMath::SingletonWmma {
                run_spec_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmMq4G256V2LmheadVerifyExact,
                    &w_out.buf,
                    w_out.gpu_dtype,
                    &rot,
                    &logits_batch,
                    w_out.m,
                    w_out.k,
                    b,
                )?;
            } else {
                gemm_mq_batched_lmhead(
                    gpu,
                    w_out.gpu_dtype,
                    &w_out.buf,
                    &rot,
                    &logits_batch,
                    w_out.m,
                    w_out.k,
                    b,
                )?;
            }
        }
        rdna_compute::DType::MQ3G256 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ3 lm_head: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rot,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::HFQ6G256 => {
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                final_hidden,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::MQ6G256 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ6 lm_head: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rot,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        other => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("DFlash verify graph lm_head unsupported dtype {other:?}"),
            ));
        }
    }

    Ok(())
}

pub(crate) fn dflash_enqueue_verify_lm_head_argmax(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    final_hidden: &GpuTensor,
    verify_scratch: &VerifyScratch,
    b: usize,
    vocab: usize,
    math: DenseBatchMath,
) -> HipResult<()> {
    dflash_enqueue_verify_lm_head(gpu, w_out, final_hidden, verify_scratch, b, vocab, math)?;
    let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
    let argmax_buf = verify_scratch.argmax.sub_offset(0, b);
    gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, b)
}

pub(crate) fn dflash_download_verify_argmax(
    gpu: &Gpu,
    verify_scratch: &VerifyScratch,
    b: usize,
) -> HipResult<Vec<u32>> {
    let argmax_buf = verify_scratch.argmax.sub_offset(0, b);
    let mut host_idx = vec![0i32; b];
    {
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, b * 4) };
        gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf)?;
    }
    Ok(host_idx.into_iter().map(|idx| idx as u32).collect())
}

/// Finish a retained/recorded verify forward through the same lm-head and
/// argmax route as the ordinary greedy DFlash path.
///
/// Redline records only the target forward. The head intentionally remains
/// outside the retained tape, so its recorded-HIP oracle must call this helper
/// instead of substituting per-row GEMV (whose reduction order can differ from
/// the product batched head even when `final_hidden` is byte-identical).
pub fn dflash_finish_retained_lm_head_argmax(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    final_hidden: &GpuTensor,
    verify_scratch: &VerifyScratch,
    b: usize,
    vocab: usize,
) -> HipResult<Vec<u32>> {
    if dflash_batched_lm_head_supported(w_out.gpu_dtype) {
        let math = verify_dense_math(gpu);
        dflash_enqueue_verify_lm_head_argmax(
            gpu,
            w_out,
            final_hidden,
            verify_scratch,
            b,
            vocab,
            math,
        )?;
        return dflash_download_verify_argmax(gpu, verify_scratch, b);
    }

    let dim = w_out.k;
    let mut argmax = Vec::with_capacity(b);
    for i in 0..b {
        let hidden_row = final_hidden.sub_offset(i * dim, dim);
        let logits_row = verify_scratch.logits.sub_offset(i * vocab, vocab);
        llama::weight_gemv(gpu, w_out, &hidden_row, &logits_row)?;
        let row = gpu.download_f32(&logits_row)?;
        argmax.push(argmax_u32(&row));
    }
    Ok(argmax)
}

/// Shape guard shared by the dense and sparse selector-q materializers: the
/// flattened candidate ids must cover every row's top-K, otherwise a sparse
/// residual would subtract an incomplete q.
fn selector_candidates_shape(proposal: &dflash::DflashCandidateProposal) -> HipResult<()> {
    let rows = proposal.tokens.len();
    let top_k = proposal.top_k;
    if proposal.candidates.len() != rows * top_k {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "selector proposal candidates len {} != rows {} * top_k {}",
                proposal.candidates.len(),
                rows,
                top_k
            ),
        ));
    }
    Ok(())
}

/// The temperature>0 selector inputs: flattened normalized q (`[rows*top_k]`)
/// and the per-row selected probability (`[rows]`), shape-checked.
fn selector_temp_inputs(proposal: &dflash::DflashCandidateProposal) -> HipResult<(&[f32], &[f32])> {
    let rows = proposal.tokens.len();
    let top_k = proposal.top_k;
    let probs = proposal.probabilities.as_ref().ok_or_else(|| {
        hip_bridge::HipError::new(0, "selector proposal missing probabilities for temp>0")
    })?;
    if probs.len() != rows * top_k {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "selector proposal probabilities len {} != rows {} * top_k {}",
                probs.len(),
                rows,
                top_k
            ),
        ));
    }
    let sel = proposal.selected_probabilities.as_ref().ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            "selector proposal missing selected_probabilities for temp>0",
        )
    })?;
    if sel.len() != rows {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "selector proposal selected_probabilities len {} != rows {}",
                sel.len(),
                rows
            ),
        ));
    }
    Ok((probs.as_slice(), sel.as_slice()))
}

/// Sparse twin of [`apply_dflash2_selector_proposal`] for temperature>0: the
/// selector's q rows are kept as (ascending id, value) pairs of their top-K
/// support — no vocab-length allocation or fill. Duplicate candidate ids
/// reproduce the dense row's last-write rule (`row[cand] = q` in candidate
/// order), see [`SparseQRows::push_row`]. `q_rows` must have been
/// [`SparseQRows::clear`]ed by the caller.
fn apply_dflash2_selector_proposal_sparse(
    proposal: &dflash::DflashCandidateProposal,
    vocab: usize,
    drafted: &mut Vec<u32>,
    q_rows: &mut SparseQRows,
    draft_probs_at_drafted: &mut Vec<f32>,
) -> HipResult<()> {
    let rows = proposal.tokens.len();
    let top_k = proposal.top_k;
    selector_candidates_shape(proposal)?;
    let (probs, sel) = selector_temp_inputs(proposal)?;
    drafted.extend_from_slice(&proposal.tokens);
    draft_probs_at_drafted.clear();
    draft_probs_at_drafted.reserve(rows);
    for i in 0..rows {
        let off = i * top_k;
        q_rows.push_row(
            &proposal.candidates[off..off + top_k],
            &probs[off..off + top_k],
            vocab,
        )?;
        draft_probs_at_drafted.push(sel[i]);
    }
    Ok(())
}

/// Fold a DFlash2 candidate-selector proposal into the chain draft buffers.
///
/// Greedy: tokens only — no full-vocab D2H. Temperature: each sparse q row
/// is materialized into a dense `draft_softmaxes` vector (zeros outside the
/// candidate set) and `draft_probs_at_drafted` is taken from
/// `selected_probabilities`, so the existing host rejection/residual path
/// stays mathematically exact. Request top_p/top_k is NOT applied to this q.
/// `proposal.probabilities` is the flattened normalized selector q over every
/// top-K candidate row when temperature>0 (not raw unary logits); greedy may
/// return `None` probabilities.
fn apply_dflash2_selector_proposal(
    proposal: &dflash::DflashCandidateProposal,
    vocab: usize,
    use_temp_sampling: bool,
    drafted: &mut Vec<u32>,
    draft_softmaxes: &mut Vec<Vec<f32>>,
    draft_probs_at_drafted: &mut Vec<f32>,
) -> HipResult<()> {
    let rows = proposal.tokens.len();
    let top_k = proposal.top_k;
    // Flattened length guards — silently dropping tail masses would make the
    // sparse residual subtract an incomplete q and corrupt the bonus distribution.
    selector_candidates_shape(proposal)?;
    if use_temp_sampling {
        let (probs, sel) = selector_temp_inputs(proposal)?;
        drafted.extend_from_slice(&proposal.tokens);
        if draft_softmaxes.len() < rows {
            draft_softmaxes.resize_with(rows, Vec::new);
        }
        draft_probs_at_drafted.clear();
        draft_probs_at_drafted.reserve(rows);
        for i in 0..rows {
            let row = &mut draft_softmaxes[i];
            row.clear();
            row.resize(vocab, 0.0);
            let off = i * top_k;
            for j in 0..top_k {
                let cand = proposal.candidates[off + j] as usize;
                let q = probs[off + j];
                if cand < vocab {
                    row[cand] = q;
                } else {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!("selector candidate id {} out of vocab {}", cand, vocab),
                    ));
                }
            }
            // Retain extra reserved rows for a later, wider window.
            draft_probs_at_drafted.push(sel[i]);
        }
        Ok(())
    } else {
        // Greedy: no q materialization, but still validate candidate shape and
        // that probabilities is None (or absent); selected may be present as 1.0.
        if proposal.probabilities.is_some() {
            // Greedy may return None probabilities per contract; if Some, still
            // require correct shape but do not materialize.
            let probs = proposal.probabilities.as_ref().unwrap();
            if probs.len() != rows * top_k {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "selector proposal greedy probabilities len {} != rows {} * top_k {}",
                        probs.len(),
                        rows,
                        top_k
                    ),
                ));
            }
        }
        if let Some(sel) = &proposal.selected_probabilities {
            if sel.len() != rows {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "selector proposal greedy selected_probabilities len {} != rows {}",
                        sel.len(),
                        rows
                    ),
                ));
            }
        }
        drafted.extend_from_slice(&proposal.tokens);
        Ok(())
    }
}

/// Task #93 Phase B seed-prediction oracle counters.
///
/// Three proxies, all derived from data the draft already computes (zero
/// extra device work). For each cycle:
///   - REJ_BOUNDARY: `drafted[accept_len + 1] == bonus_token` (PRD's "naive"
///     proxy — argmax at rejection position). Zero-by-construction when
///     `accept_len < b - 1` because the accept loop broke precisely because
///     those didn't match. Reported anyway to document the dead-end.
///   - TAIL: `drafted[b - 1] == bonus_token`. Draft's final-position argmax.
///     Gives a non-zero signal. If the usual case is "target's bonus happens
///     at position b-1 because accept_len = b-2", this proxy catches those.
///   - ANYPOS: `bonus_token ∈ drafted[1..b]`. Upper bound of any position-
///     based single-guess proxy. Useful as a ceiling.
///
/// FULLACCEPT counts cycles where `accept_len == b - 1` (full acceptance —
/// draft has no native prediction at position `b`, so REJ_BOUNDARY is
/// undefined and TAIL/ANYPOS are the only candidates there).
///
static SEED_ORACLE_TOTAL: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_REJ_MATCH: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_TAIL_MATCH: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_ANYPOS_MATCH: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_FULLACCEPT: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_ACCEPT_LEN_SUM: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default)]
pub struct SeedOracleStats {
    pub total: u64,
    pub rej_match: u64,
    pub tail_match: u64,
    pub anypos_match: u64,
    pub full_accept: u64,
    pub accept_len_sum: u64,
}

/// Snapshot the process-global seed-oracle counters.
pub fn read_seed_oracle_stats() -> SeedOracleStats {
    SeedOracleStats {
        total: SEED_ORACLE_TOTAL.load(Ordering::Relaxed),
        rej_match: SEED_ORACLE_REJ_MATCH.load(Ordering::Relaxed),
        tail_match: SEED_ORACLE_TAIL_MATCH.load(Ordering::Relaxed),
        anypos_match: SEED_ORACLE_ANYPOS_MATCH.load(Ordering::Relaxed),
        full_accept: SEED_ORACLE_FULLACCEPT.load(Ordering::Relaxed),
        accept_len_sum: SEED_ORACLE_ACCEPT_LEN_SUM.load(Ordering::Relaxed),
    }
}

/// Zero all seed-oracle counters. Call before a fresh generation run.
pub fn reset_seed_oracle_stats() {
    SEED_ORACLE_TOTAL.store(0, Ordering::Relaxed);
    SEED_ORACLE_REJ_MATCH.store(0, Ordering::Relaxed);
    SEED_ORACLE_TAIL_MATCH.store(0, Ordering::Relaxed);
    SEED_ORACLE_ANYPOS_MATCH.store(0, Ordering::Relaxed);
    SEED_ORACLE_FULLACCEPT.store(0, Ordering::Relaxed);
    SEED_ORACLE_ACCEPT_LEN_SUM.store(0, Ordering::Relaxed);
}

/// DDTree meta-verifier pruner telemetry: per-cycle tree-size histogram.
/// `cycle_count` = cycles observed; `total_nodes` = sum of tree.num_nodes()
/// across cycles; `max_nodes` / `min_nodes` = range observed.
static DDTREE_META_CYCLES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_TOTAL_NODES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_MAX_NODES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_MIN_NODES: AtomicU64 = AtomicU64::new(u64::MAX);

pub fn record_ddtree_meta_nodes(n: usize) {
    let n64 = n as u64;
    DDTREE_META_CYCLES.fetch_add(1, Ordering::Relaxed);
    DDTREE_META_TOTAL_NODES.fetch_add(n64, Ordering::Relaxed);
    DDTREE_META_MAX_NODES.fetch_max(n64, Ordering::Relaxed);
    DDTREE_META_MIN_NODES.fetch_min(n64, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DdtreeMetaStats {
    pub cycles: u64,
    pub total_nodes: u64,
    pub max_nodes: u64,
    pub min_nodes: u64,
}

pub fn read_ddtree_meta_stats() -> DdtreeMetaStats {
    let c = DDTREE_META_CYCLES.load(Ordering::Relaxed);
    DdtreeMetaStats {
        cycles: c,
        total_nodes: DDTREE_META_TOTAL_NODES.load(Ordering::Relaxed),
        max_nodes: DDTREE_META_MAX_NODES.load(Ordering::Relaxed),
        min_nodes: if c == 0 {
            0
        } else {
            DDTREE_META_MIN_NODES.load(Ordering::Relaxed)
        },
    }
}

pub fn reset_ddtree_meta_stats() {
    DDTREE_META_CYCLES.store(0, Ordering::Relaxed);
    DDTREE_META_TOTAL_NODES.store(0, Ordering::Relaxed);
    DDTREE_META_MAX_NODES.store(0, Ordering::Relaxed);
    DDTREE_META_MIN_NODES.store(u64::MAX, Ordering::Relaxed);
}

/// Which KV cache layout to use when allocating a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    /// INT8 co-located K and V (default).
    Q8,
    /// Asym4: rotated 4-bit K + Q8 V (smaller than Q8, higher-fidelity than asym3).
    Asym4,
    /// Asym3: rotated 3-bit K + Q8 V. ~2.7× less KV BW than Q8, tightly-tuned
    /// kernel for the hot FA attention path. Good choice for long-context verify.
    Asym3,
    /// Asym2: rotated 2-bit K + Q8 V. Smallest but most lossy.
    Asym2,
    /// Fwht4: signed-FWHT-rotated 4-bit K + Q8 V. Byte-identical storage to
    /// Asym4 but with a Hadamard rotation (matches MQ4's weight-quant trick).
    /// Centroid LUTs were always Lloyd-Max-fit for post-FWHT N(0, 1/128) per
    /// turbo_common.h:13 — Fwht4 finally uses them on the distribution they
    /// were calibrated for. Opt-in via `--kv-mode fwht4`.
    Fwht4,
    /// Fwht3: signed-FWHT-256 rotated 3-bit K + Q8 V. Byte-identical storage to
    /// Asym3 (the canonical default). Single-pass 256-element FWHT — the
    /// natural fit for asym3's existing layout (8 dims/thread). Empirical
    /// prose-τ win on 3.5-27b at the 4-bit tier suggests the 3-bit tier
    /// should benefit even more from rotation. Opt-in via `--kv-mode fwht3`.
    Fwht3,
    /// Fwht2: signed-FWHT-128 rotated 2-bit K + Q8 V. Byte-identical storage
    /// to Asym2. 2-pass-over-128 structure matches fwht4. Highest theoretical
    /// leverage tier — Asym2 is doc'd "most lossy" and 2-bit centroid quant
    /// suffers most from outliers. Opt-in via `--kv-mode fwht2`.
    Fwht2,
    /// Native fp8 E4M3 KV (gfx1201 Qwen dense only). ModelSlot::load rejects
    /// this mode: single-slot speculative assembly goes through the carrier
    /// path, never the slot constructors below. Present so `--kv-mode fp8`
    /// fails closed here instead of parsing as a neighboring mode.
    Fp8,
    /// Flat native bf16 KV quality-control arm. Same rejection contract as
    /// Fp8 above: ModelSlot::load refuses, carrier path owns admission.
    Bf16,
}
impl Default for KvMode {
    fn default() -> Self {
        KvMode::Q8
    }
}

/// Configuration for loading a single model slot.
#[derive(Debug, Clone)]
pub struct ModelSlotConfig {
    pub max_seq: usize,
    pub kv_mode: KvMode,
    pub repeat_window: usize,
    pub state_quant: qwen35::StateQuant,
}

impl Default for ModelSlotConfig {
    fn default() -> Self {
        Self {
            max_seq: 2048,
            kv_mode: KvMode::Q8,
            repeat_window: 128,
            state_quant: qwen35::StateQuant::Q8,
        }
    }
}

/// A single loaded Qwen3.5 model with its own KV cache, DeltaNet state, and
/// forward-pass scratch. The `Gpu` is borrowed, not owned — multiple slots
/// share one `Gpu` instance.
pub struct ModelSlot {
    pub name: String,
    pub hfq: HfqFile,
    pub config: Qwen35Config,
    pub weights: Qwen35Weights,
    pub kv_cache: KvCache,
    pub dn_state: DeltaNetState,
    pub scratch: Qwen35Scratch,
    pub slot_config: ModelSlotConfig,
    /// DSpark (EAGLE-3) residual-hidden extract-layer ids. Empty for every
    /// non-DSpark path (AR / n-gram / DFlash / MTP), which keeps `new_spec_scratch`
    /// building a `num_extract = 0` no-op hidden ring — byte-identical to the
    /// pre-DSpark behaviour. A DSpark drafter populates this via
    /// `set_dflash_extract_layers` / `capture_seed_main_hidden` so the per-window
    /// verify captures hidden at exactly the sidecar's layer ids.
    pub dspark_extract_layers: Vec<usize>,
    /// Vision tower carried through the slot guard so `Qwen35Bundle` round-trips
    /// without loss. The slot never drives `vision_forward`; it just parks the
    /// tower while the bundle is out of `ModelState`.
    pub vision_config: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionConfig>,
    pub vision_weights: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionWeights>,
}

impl ModelSlot {
    /// Assemble the spec-decode target slot from a live [`Qwen35Bundle`] that was
    /// parked out of `ModelState`, opening the model's `HfqFile` (the mmap handle
    /// `ModelSlot` carries but the spec kernels never read).
    ///
    /// **Fallible WITHOUT loss:** on a reopen failure the bundle is handed BACK in
    /// the `Err` so the caller's RAII guard can re-park it and restore it into
    /// `ModelState` on `Drop` — consuming the bundle on the error path would drop
    /// it and leave `m.state == None`, reintroducing the #462 cross-request
    /// state-bleed class this seam exists to prevent. This is the only piece of
    /// `Qwen35Bundle` field knowledge the loader's guard previously inlined; it
    /// lives here so the loader never names the bundle's fields.
    pub fn from_bundle(bundle: Qwen35Bundle, path: &Path) -> Result<Self, (Qwen35Bundle, String)> {
        // Batch owners (fixed-lane staging or the per-request VMM store) are
        // outside the spec slot. Refuse WITHOUT loss rather than dropping
        // them here: a dropped store would leak its request VMM leases and
        // strand those requests' state. A request leaving the batch for the
        // singleton spec route first swaps its owners into the bundle
        // (`Qwen35RequestState::swap_with_bundle`) and the driver parks the
        // store before entering this seam.
        if bundle.qwen35_decode_batch.is_some() || bundle.vmm_store.is_some() {
            return Err((
                bundle,
                "ModelSlot::from_bundle: batch state is staged on the bundle; park it before speculative takeover"
                    .into(),
            ));
        }
        if bundle.weight_lease.is_some() {
            return Err((
                bundle,
                "ModelSlot::from_bundle: weight-host imported weights are not admitted to speculative decode".into(),
            ));
        }
        let hfq = match HfqFile::open(path) {
            Ok(h) => h,
            Err(e) => return Err((bundle, format!("reopen model: {e}"))),
        };
        let Qwen35Bundle {
            config,
            weights,
            scratch,
            kv_cache,
            dn_state,
            kv_adaptive: _,
            pp_scratch_set,
            vision_config,
            vision_weights,
            qwen35_decode_batch: _,
            vmm_store: _,
            weight_lease: _,
        } = bundle;
        debug_assert!(
            pp_scratch_set.is_none(),
            "ModelSlot::from_bundle: pp_scratch_set must be None (pp>1 never enters spec slot)"
        );
        let _ = pp_scratch_set;
        Ok(Self {
            name: String::from("target"),
            hfq,
            config,
            weights,
            kv_cache,
            dn_state,
            scratch,
            slot_config: ModelSlotConfig::default(),
            dspark_extract_layers: Vec::new(),
            vision_config,
            vision_weights,
        })
    }

    /// Disassemble the slot back into a [`Qwen35Bundle`] — the `HfqFile` mmap,
    /// `name`, and `slot_config` drop here; the five live pieces return to the
    /// bundle. The inverse of [`from_bundle`](Self::from_bundle); the loader's
    /// guard calls this on `Drop` to restore `ModelState`.
    pub fn into_bundle(self) -> Qwen35Bundle {
        Qwen35Bundle {
            config: self.config,
            weights: self.weights,
            scratch: self.scratch,
            kv_cache: self.kv_cache,
            dn_state: self.dn_state,
            // Controller lives on LoadedModel, not ModelSlot.
            kv_adaptive: None,
            pp_scratch_set: None,
            vision_config: self.vision_config,
            vision_weights: self.vision_weights,
            qwen35_decode_batch: None,
            vmm_store: None,
            weight_lease: None,
        }
    }
}

impl ModelSlot {
    /// for all allocations. `name` is a human-readable label used in logs.
    pub fn load(
        gpu: &mut Gpu,
        path: &Path,
        name: impl Into<String>,
        slot_config: ModelSlotConfig,
    ) -> HipResult<Self> {
        let name = name.into();
        let mut hfq = HfqFile::open(path).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("open {} ({}): {}", path.display(), name, e))
        })?;
        let config = qwen35::config_from_hfq(&hfq).map_err(|e| {
            hip_bridge::HipError::new(
                0,
                &format!(
                    "invalid Qwen3.5 config in {} ({}): {e}",
                    path.display(),
                    name
                ),
            )
        })?;
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        let weights = qwen35::load_weights(&mut src, std::slice::from_mut(gpu), &layout)?;

        // For hybrid arches (Qwen 3.5 = 48 DeltaNet LinearAttention + 16
        // FullAttention out of 64 total), only the FullAttention layers need
        // a KV cache slot. The LinearAttention layers carry their own state
        // via DeltaNetState (`new_with_quant` below) and never write to
        // kv_cache.k_gpu / .v_gpu. Pre-2026-05-15 the KV constructor
        // allocated full K/V slots for ALL layers regardless of type — at
        // ctx=64K that's ~5 GB of dead allocation on 27B. The `_filtered`
        // constructors take a `is_kv_layer` slice and substitute a
        // 1-element placeholder for non-KV layers. Indexing by absolute
        // layer_idx is preserved.
        let is_kv_layer: Vec<bool> = config
            .layer_types
            .iter()
            .map(|t| *t == qwen35::LayerType::FullAttention)
            .collect();

        // Honor the caller's requested KV cache mode. Default is Q8 for
        // backwards-compat, but DFlash verify is KV-bandwidth sensitive at
        // longer contexts — asym3/asym4 cut the verify attention cost.
        let kv_cache = match slot_config.kv_mode {
            KvMode::Q8 => KvCache::new_gpu_q8_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Asym4 => KvCache::new_gpu_asym4_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Asym3 => KvCache::new_gpu_asym3_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Asym2 => KvCache::new_gpu_asym2_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Fwht4 => KvCache::new_gpu_fwht4_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Fwht3 => KvCache::new_gpu_fwht3_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Fwht2 => KvCache::new_gpu_fwht2_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Fp8 | KvMode::Bf16 => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "ModelSlot::load rejects fp8/bf16 KV: single-slot speculative assembly \
                     has no native-format slot constructors; load through the carrier path",
                ));
            }
        };

        let dn_state = DeltaNetState::new_with_quant(gpu, &config, slot_config.state_quant)?;
        // Size flash partials for the slot's real KV capacity (as the carrier
        // path does). `Qwen35Scratch::new` assumes 8192, which at long context
        // leaves room for one verify row, so VerifyAttn re-scans the whole KV
        // once per verify row.
        let scratch = Qwen35Scratch::new_with_kv_max(
            gpu,
            &config,
            slot_config.repeat_window,
            slot_config.max_seq.max(8192),
        )?;

        Ok(Self {
            name,
            hfq,
            config,
            weights,
            kv_cache,
            dn_state,
            scratch,
            slot_config,
            dspark_extract_layers: Vec::new(),
            vision_config: None,
            vision_weights: None,
        })
    }

    /// Load the tokenizer from this slot's HFQ metadata. Each slot technically
    /// carries its own tokenizer; callers should validate that two slots'
    /// tokenizers are compatible via `Tokenizer::is_compatible_with` before
    /// sharing. Returns the underlying `TokenizerError` on failure so callers
    /// can surface specific diagnostics (e.g. `MissingMergeResult` from a
    /// truncated quantizer output) rather than a generic "no tokenizer"
    /// message — see #203.
    pub fn load_tokenizer(&self) -> Result<Tokenizer, TokenizerError> {
        Tokenizer::from_hfq_metadata(&self.hfq.metadata_json)
    }

    /// Single-token forward pass. Writes logits into `self.scratch.logits`.
    pub fn forward(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> HipResult<()> {
        gpu.graphs.ar_graph_eligible = false; // spec re-seed: never the plain-AR graph
        qwen35::forward_scratch(
            gpu,
            &self.weights,
            &self.config,
            token,
            pos,
            &mut self.kv_cache,
            &mut self.dn_state,
            &self.scratch,
        )
    }

    /// Reset the DeltaNet recurrent state (S/scale/conv + default-on EF residual)
    /// and zero the KV write head. Does NOT shrink the KV allocation — callers
    /// track `seq_pos` separately.
    pub fn reset_state(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        // Canonical path: `DeltaNetState::reset` is stream-aware and zeroes
        // s_ef_residual alongside S/scale/conv. Hand-rolled memset loops here
        // previously omitted EF and left residual noise across cold resets.
        self.dn_state.reset(gpu)?;
        self.kv_cache.compact_offset = 0;
        Ok(())
    }
}

/// Borrow-only trunk owners a DFlash prompt seed / verify commit / terminal
/// repair touches: a [`ModelSlot`]'s, or a continuous-batching request's own
/// KV/DeltaNet over the shared weights. DFlash counterpart of
/// [`crate::mtp_spec::MtpPrefillTarget`].
pub struct DflashTargetParts<'a> {
    pub weights: &'a Qwen35Weights,
    pub config: &'a Qwen35Config,
    pub kv_cache: &'a mut KvCache,
    pub dn_state: &'a mut DeltaNetState,
    pub scratch: &'a Qwen35Scratch,
}

impl<'a> DflashTargetParts<'a> {
    /// Borrow a [`ModelSlot`]'s trunk owners.
    pub fn from_slot(slot: &'a mut ModelSlot) -> Self {
        Self {
            weights: &slot.weights,
            config: &slot.config,
            kv_cache: &mut slot.kv_cache,
            dn_state: &mut slot.dn_state,
            scratch: &slot.scratch,
        }
    }

    /// Same as [`ModelSlot::reset_state`]: zero the DeltaNet recurrent state
    /// (incl. EF residual) and the KV compact offset.
    pub fn reset_state(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        self.dn_state.reset(gpu)?;
        self.kv_cache.compact_offset = 0;
        Ok(())
    }
}

/// One planned DFlash verify window handed to the batched verify/accept path.
/// `verify_tokens = [seed, B-1 draft candidates]`; `position` is the absolute
/// position of `verify_tokens[0]`; `max_accept` is the pre-commit clamp
/// (`remaining_budget - 1`); `thlog_mark` is the draft target-hidden log mark
/// captured before drafting (the singleton's window mark).
#[derive(Clone, Debug)]
pub struct DflashCbDraft {
    pub position: usize,
    pub seed: u32,
    pub verify_tokens: Vec<u32>,
    pub max_accept: usize,
    pub thlog_mark: dflash::TargetHiddenLogMark,
}

/// A pair of target + draft slots sharing one `Gpu` and one tokenizer.
///
/// Phase 1 just carries both slots. Phase 2+ adds the `spec_decode_step`
/// method for the verify-and-accept loop.
pub struct SpecPair {
    pub target: ModelSlot,
    pub draft: ModelSlot,
    pub tokenizer: Tokenizer,
}

impl SpecPair {
    /// Load target and draft from separate HFQ files on the same `Gpu`.
    /// Validates that the two models share a compatible tokenizer before
    /// returning — speculative decode requires identical vocab + token IDs.
    pub fn load(
        gpu: &mut Gpu,
        target_path: &Path,
        draft_path: &Path,
        target_cfg: ModelSlotConfig,
        draft_cfg: ModelSlotConfig,
    ) -> HipResult<Self> {
        let target = ModelSlot::load(gpu, target_path, "target", target_cfg)?;
        let draft = ModelSlot::load(gpu, draft_path, "draft", draft_cfg)?;

        let target_tok = target.load_tokenizer().map_err(|e| {
            hip_bridge::HipError::new(0, &format!("target tokenizer load failed: {e}"))
        })?;
        let draft_tok = draft.load_tokenizer().map_err(|e| {
            hip_bridge::HipError::new(0, &format!("draft tokenizer load failed: {e}"))
        })?;

        if target_tok.vocab_size() != draft_tok.vocab_size() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "tokenizer mismatch: target vocab={}, draft vocab={}. \
                     Speculative decode requires identical vocabularies.",
                    target_tok.vocab_size(),
                    draft_tok.vocab_size()
                ),
            ));
        }

        // Sanity-check a round-trip on a common string — catches vocab-size
        // match but token-ID mismatch (different BPE merges producing same
        // vocab count).
        let probe = "<|im_start|>user\nHello world\n<|im_end|>";
        let a = target_tok.encode(probe);
        let b = draft_tok.encode(probe);
        if a != b {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "tokenizer merge rules diverge: target={:?}, draft={:?}. \
                     Speculative decode requires identical tokenization.",
                    &a, &b
                ),
            ));
        }

        Ok(Self {
            target,
            draft,
            tokenizer: target_tok,
        })
    }

    /// Run a minimal smoke test: 8 forward passes on each slot with a dummy
    /// token sequence, ensuring neither model crashes and the logits buffers
    /// contain finite values. Returns `(target_ok, draft_ok)`.
    pub fn smoke_test(&mut self, gpu: &mut Gpu) -> HipResult<(bool, bool)> {
        // Token ID 1 is a safe placeholder for both Qwen3 and Qwen3.5; the
        // smoke test only checks that the forward pass runs without crashing
        // and produces finite logits.
        let probe_token: u32 = 1;
        for pos in 0..8 {
            self.target.forward(gpu, probe_token, pos)?;
        }
        for pos in 0..8 {
            self.draft.forward(gpu, probe_token, pos)?;
        }
        let target_logits = gpu.download_f32(&self.target.scratch.logits)?;
        let draft_logits = gpu.download_f32(&self.draft.scratch.logits)?;
        let target_ok = target_logits.iter().take(1024).all(|x| x.is_finite());
        let draft_ok = draft_logits.iter().take(1024).all(|x| x.is_finite());

        // Reset both after the smoke test so the caller starts from a clean
        // state at seq_pos=0.
        self.target.reset_state(gpu)?;
        self.draft.reset_state(gpu)?;

        Ok((target_ok, draft_ok))
    }
}

/// Result of one speculative decode step.
#[derive(Debug, Clone)]
pub struct SpecStepResult {
    /// Number of draft tokens accepted (0..=k).
    pub accepted: usize,
    /// Target's next-token prediction at the first rejection point (or after
    /// all drafted tokens if accepted == k). Appended to `committed`.
    pub bonus_token: u32,
    /// The full sequence of tokens the draft proposed this cycle.
    pub drafted: Vec<u32>,
    /// The tokens actually committed to both models: `drafted[..accepted]`
    /// followed by `bonus_token`. Always non-empty (length = accepted + 1).
    pub committed: Vec<u32>,
}

/// Exact LOGICAL per-layer byte extents of the four DeltaNet state families
/// for one config's Q8 + error-feedback state: what the kernels write (see
/// [`DeltaNetSnapshot::cb_verify_owned_exact`], which allows pool-rounded
/// scales / conv rings of equal size on both banks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DnCbExtents {
    n_layers: usize,
    s: usize,
    scales: usize,
    conv: usize,
    ef: usize,
}

impl DnCbExtents {
    /// `None` unless the geometry is the one the exact save-free kernels
    /// cover: 128-wide heads and conv kernel dimension 4 (3 ring slots).
    fn for_config(config: &Qwen35Config) -> Option<Self> {
        if config.conv_kernel_dim != 4
            || config.linear_key_head_dim != 128
            || config.linear_value_head_dim != 128
        {
            return None;
        }
        let hv = config.linear_num_value_heads;
        let hk = config.linear_num_key_heads;
        let n_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == qwen35::LayerType::LinearAttention)
            .count();
        let conv_channels = 2 * hk * 128 + hv * 128;
        Some(Self {
            n_layers,
            s: hv * 128 * 128,
            scales: hv * 128 * 4,
            conv: conv_channels * 3 * 4,
            ef: hv * 128 * 128 * 2,
        })
    }
}

/// Backing storage for a DeltaNetState snapshot. Holds device buffers sized
/// to match the source state's tensors. Allocate once per slot, reuse across
/// all speculative cycles.
///
/// Includes the default-on Q8 error-feedback residual (`s_ef_residual`) when
pub struct DeltaNetSnapshot {
    s_matrix_bufs: Vec<DeviceBuffer>,
    s_scale_bufs: Vec<DeviceBuffer>,
    conv_state_bufs: Vec<DeviceBuffer>,
    /// F16 per-element EF residual backups; `len == state.s_ef_residual.len()`.
    s_ef_residual_bufs: Vec<DeviceBuffer>,
    /// S1: persistent forward (live -> backup) descriptor table, device
    /// resident, built once at `new_for`. `None` unless the bulk route armed
    /// (arch outside `bulk_arch`, kill switch, JIT failure, or bad alignment
    /// all leave this `None` and every op uses the memcpy loops).
    bulk_fwd: Option<DeviceBuffer>,
    /// S1: persistent reverse (backup -> live) descriptor table. Same
    /// arming rule as `bulk_fwd`; both are always armed together.
    bulk_rev: Option<DeviceBuffer>,
    /// S1: descriptor count shared by both tables (fixed per snapshot).
    bulk_n_items: u32,
    /// S1: live-state pointer/size fingerprint the tables were built
    /// against. Save/restore re-fingerprint the passed state and fall back
    /// to memcpy on any mismatch (never copy through stale descriptors).
    bulk_fingerprint: u64,
}

impl DeltaNetSnapshot {
    /// S1: chunk size for bulk-copy descriptor splitting. Chunk offsets stay
    /// multiples of this, keeping every 16 B vector lane aligned.
    const BULK_CHUNK: usize = 64 * 1024;

    /// S1: FNV-1a fingerprint over the live state's family lengths plus every
    /// tensor's (pointer, size) pair in family order. Tables built at
    /// `new_for` are valid only while this matches; any mismatch routes to
    /// the memcpy loops (never copy through stale descriptors).
    fn bulk_fingerprint(state: &DeltaNetState) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x0100_0000_01b3);
        };
        mix(state.s_matrices.len() as u64);
        mix(state.s_scales.len() as u64);
        mix(state.conv_states.len() as u64);
        mix(state.s_ef_residual.len() as u64);
        for t in state
            .s_matrices
            .iter()
            .chain(state.s_scales.iter())
            .chain(state.conv_states.iter())
            .chain(state.s_ef_residual.iter())
        {
            mix(t.buf.as_ptr() as u64);
            mix(t.buf.size() as u64);
        }
        h
    }

    /// S1: build + upload the forward/reverse descriptor tables. Returns
    /// `None` — leaving the snapshot on the memcpy path — when the kernel
    /// cannot be ensured, any live/backup pair is size-mismatched or
    /// misaligned, or the item count does not fit the fixed grid. Never
    /// fails the allocation. EF-off is an empty fourth family, not a fake
    /// allocation: it contributes zero items.
    fn build_bulk_tables(
        gpu: &mut Gpu,
        state: &DeltaNetState,
        backs: [&[DeviceBuffer]; 4],
    ) -> Option<(DeviceBuffer, DeviceBuffer, u32)> {
        if gpu.ensure_dflash_state_bulk_copy_gfx1100().is_err() {
            return None;
        }
        let lives: [&[GpuTensor]; 4] = [
            &state.s_matrices,
            &state.s_scales,
            &state.conv_states,
            &state.s_ef_residual,
        ];
        let mut fwd: Vec<DflashStateCopyDesc> = Vec::new();
        let mut rev: Vec<DflashStateCopyDesc> = Vec::new();
        for (live_fam, back_fam) in lives.iter().zip(backs.iter()) {
            if live_fam.len() != back_fam.len() {
                return None;
            }
            for (live, back) in live_fam.iter().zip(back_fam.iter()) {
                let n = live.buf.size();
                if n != back.size() || n == 0 {
                    if n != back.size() {
                        return None;
                    }
                    continue;
                }
                let s = live.buf.as_ptr() as u64;
                let d = back.as_ptr() as u64;
                // The vector body needs 16 B aligned bases; chunk offsets are
                // 64-KiB multiples by construction.
                if s % 16 != 0 || d % 16 != 0 {
                    return None;
                }
                let mut off: usize = 0;
                while off < n {
                    let cnt = (n - off).min(Self::BULK_CHUNK);
                    fwd.push(DflashStateCopyDesc {
                        src: s,
                        dst: d,
                        off: off as u64,
                        cnt: cnt as u64,
                    });
                    rev.push(DflashStateCopyDesc {
                        src: d,
                        dst: s,
                        off: off as u64,
                        cnt: cnt as u64,
                    });
                    off += cnt;
                }
            }
        }
        if fwd.is_empty() || fwd.len() > DFLASH_STATE_BULK_COPY_MAX_ITEMS as usize {
            return None;
        }
        let n_items = fwd.len() as u32;
        let bytes = std::mem::size_of::<DflashStateCopyDesc>();
        let fwd_buf = gpu.hip.malloc(fwd.len() * bytes).ok()?;
        if gpu
            .hip
            .memcpy_htod(&fwd_buf, DflashStateCopyDesc::as_bytes(&fwd))
            .is_err()
        {
            let _ = gpu.hip.free(fwd_buf);
            return None;
        }
        let rev_buf = gpu.hip.malloc(rev.len() * bytes).ok()?;
        if gpu
            .hip
            .memcpy_htod(&rev_buf, DflashStateCopyDesc::as_bytes(&rev))
            .is_err()
        {
            let _ = gpu.hip.free(fwd_buf);
            let _ = gpu.hip.free(rev_buf);
            return None;
        }
        Some((fwd_buf, rev_buf, n_items))
    }

    /// S1 arch gate. gfx1100 (S1), exact gfx1201 (railgun E0 port) and exact
    /// gfx1151. The kernel is a pure byte copy with no arch-specific code
    /// (plain `float4` loads/stores, no intrinsics); the `_gfx1100` suffix on
    /// its symbol is historical. `HIPFIRE_DN_SNAPSHOT_BULK_OFF=1` opts out on
    /// all three.
    fn bulk_arch(gpu: &Gpu) -> bool {
        !gpu.flags.dn_snapshot_bulk_off
            && (gpu.arch_caps.is_gfx1100()
                || gpu.arch_caps.is_gfx1151()
                || gpu.arch_caps.is_gfx1201())
    }

    /// S1: shared fast-path gate. Returns the table + count to launch, or
    /// `None` when the call must use the memcpy loops (kill switch, arch,
    /// disarmed tables, or stale fingerprint).
    fn bulk_table(
        &self,
        state: &DeltaNetState,
        gpu: &Gpu,
        forward: bool,
    ) -> Option<(&DeviceBuffer, u32)> {
        if !Self::bulk_arch(gpu) {
            return None;
        }
        if self.bulk_n_items == 0 || Self::bulk_fingerprint(state) != self.bulk_fingerprint {
            return None;
        }
        match (forward, &self.bulk_fwd, &self.bulk_rev) {
            (true, Some(t), _) => Some((t, self.bulk_n_items)),
            (false, _, Some(t)) => Some((t, self.bulk_n_items)),
            _ => None,
        }
    }

    /// S1: host-visible completion barrier. `memcpy_dtod` blocks the host;
    /// the kernel launch does not, so sync the launch stream to preserve the
    /// exact synchronous contract (backup==L on save return, live==L on
    /// restore return). The optimized route still never allocates, uploads
    /// descriptors, reads host state, or JITs in a decode cycle.
    fn bulk_sync(gpu: &Gpu) -> HipResult<()> {
        match &gpu.active_stream {
            Some(s) => gpu.hip.stream_synchronize(s),
            None => gpu.hip.device_synchronize(),
        }
    }

    /// Allocate backup buffers matching `state`'s shapes (incl. EF residual).
    /// A mid-way allocation failure frees the buffers already allocated.
    pub fn new_for(gpu: &mut Gpu, state: &DeltaNetState) -> HipResult<Self> {
        let families = [
            &state.s_matrices,
            &state.s_scales,
            &state.conv_states,
            &state.s_ef_residual,
        ];
        let mut bufs: [Vec<DeviceBuffer>; 4] = Default::default();
        for (i, src) in families.into_iter().enumerate() {
            bufs[i].reserve_exact(src.len());
            for t in src {
                match gpu.hip.malloc(t.buf.size()) {
                    Ok(b) => bufs[i].push(b),
                    Err(e) => {
                        for b in bufs.into_iter().flatten() {
                            let _ = gpu.hip.free(b);
                        }
                        return Err(e);
                    }
                }
            }
        }
        let [s_matrix_bufs, s_scale_bufs, conv_state_bufs, s_ef_residual_bufs] = bufs;
        let mut snap = Self {
            s_matrix_bufs,
            s_scale_bufs,
            conv_state_bufs,
            s_ef_residual_bufs,
            bulk_fwd: None,
            bulk_rev: None,
            bulk_n_items: 0,
            bulk_fingerprint: Self::bulk_fingerprint(state),
        };
        // Arm the bulk route (gfx1100 / gfx1151 / gfx1201): JIT + table upload happen
        // here at setup, never in a decode cycle. Any failure leaves the
        // snapshot on the legacy memcpy path.
        if Self::bulk_arch(gpu) {
            let backs = [
                &snap.s_matrix_bufs[..],
                &snap.s_scale_bufs[..],
                &snap.conv_state_bufs[..],
                &snap.s_ef_residual_bufs[..],
            ];
            if let Some((f, r, n)) = Self::build_bulk_tables(gpu, state, backs) {
                snap.bulk_fwd = Some(f);
                snap.bulk_rev = Some(r);
                snap.bulk_n_items = n;
            }
        }
        Ok(snap)
    }

    /// Total device bytes a snapshot of `state` would occupy, computed
    /// WITHOUT allocating — lets `capture_checkpoint` pre-check the pool
    /// ceiling before paying for `hipMalloc` + device-to-device copy.
    pub fn bytes_for(state: &DeltaNetState) -> u64 {
        let mut total: u64 = 0;
        for t in &state.s_matrices {
            total += t.buf.size() as u64;
        }
        for t in &state.s_scales {
            total += t.buf.size() as u64;
        }
        for t in &state.conv_states {
            total += t.buf.size() as u64;
        }
        for t in &state.s_ef_residual {
            total += t.buf.size() as u64;
        }
        total
    }

    /// Number of EF residual backup buffers (0 when EF is off).
    #[inline]
    pub fn s_ef_len(&self) -> usize {
        self.s_ef_residual_bufs.len()
    }

    /// S1: armed descriptor count, or `None` when the snapshot rides the
    /// legacy memcpy loops. Diagnostics only (the launch-count gate proves
    /// engagement); always `None` outside `bulk_arch` or under the kill switch.
    #[inline]
    pub fn bulk_n_items(&self) -> Option<u32> {
        self.bulk_fwd
            .as_ref()
            .and(self.bulk_rev.as_ref())
            .map(|_| self.bulk_n_items)
    }

    /// Railgun D8: one same-size backup per live tensor in every family (true
    /// for a snapshot `new_for` made from this state), so a kernel can read
    /// backup `i` in place of live tensor `i`.
    pub fn mirrors(&self, state: &DeltaNetState) -> bool {
        fn same(live: &[GpuTensor], backs: &[DeviceBuffer]) -> bool {
            live.len() == backs.len()
                && live
                    .iter()
                    .zip(backs)
                    .all(|(t, b)| t.buf.size() == b.size())
        }
        same(&state.s_matrices, &self.s_matrix_bufs)
            && same(&state.s_scales, &self.s_scale_bufs)
            && same(&state.conv_states, &self.conv_state_bufs)
            && same(&state.s_ef_residual, &self.s_ef_residual_bufs)
    }

    /// Slice S (save-free CB verify): is this snapshot + `state` an exact,
    /// exchangeable pair of owned allocations for `config`? Whole-pair
    /// preflight, host-only, never touches the device.
    ///
    /// * `config` has a Q8 + error-feedback DeltaNet geometry the exact
    ///   kernels cover (conv kernel dimension 4, 128-wide heads) and `state`
    ///   is Q8;
    /// * all four families of both sides hold exactly the config's DeltaNet
    ///   layer count (EF complete, so a snapshot mirrors the state);
    /// * per layer and family the live buffer and its snapshot buffer have
    ///   the SAME allocated size, and it is at least the exact logical extent
    ///   the kernels write (S = Hv*128^2, scales = Hv*128*4, conv =
    ///   (2*Hk+Hv)*128*3*4, EF = Hv*128^2*2). S is exact (a direct malloc);
    ///   scales, conv rings and EF are pooled (`gpu.zeros`) and may be
    ///   rounded up (the kernels touch only the logical extent). Equal sizes
    ///   mean an exchange keeps every owner's capacity unchanged (a reused
    ///   EF bucket of 1,986,560 B for a 1,572,864 B extent refused most
    ///   sampled C8 chunks before EF was relaxed). `GpuTensor::byte_size` is
    ///   NOT used: the Q8 S
    ///   container's metadata is F32;
    /// * every buffer on both sides is a HIP-owned allocation (no borrowed
    ///   lane view, VMM arena or host-mapped buffer), so exchanging the
    ///   owning handles keeps "freed exactly once through its own owner".
    ///
    /// Cross-lane pointer distinctness is the caller's whole-chunk check
    /// ([`Self::push_owner_ptrs`]).
    pub(crate) fn cb_verify_owned_exact(
        &self,
        state: &DeltaNetState,
        config: &Qwen35Config,
    ) -> bool {
        let Some(ext) = DnCbExtents::for_config(config) else {
            return false;
        };
        self.exact_extents(state, &ext) && self.all_hip_owned(state)
    }

    /// Family lengths and per-layer extents of both sides (see
    /// [`Self::cb_verify_owned_exact`]); pure metadata.
    fn exact_extents(&self, state: &DeltaNetState, ext: &DnCbExtents) -> bool {
        /// Equal live/backup size per layer, each at least `logical`
        /// (`exact`: exactly `logical`).
        fn fam_ok(
            live: &[GpuTensor],
            back: &[DeviceBuffer],
            n: usize,
            logical: usize,
            exact: bool,
        ) -> bool {
            live.len() == n
                && back.len() == n
                && live.iter().zip(back).all(|(l, b)| {
                    let sz = l.buf.size();
                    sz == b.size() && if exact { sz == logical } else { sz >= logical }
                })
        }
        let n = ext.n_layers;
        n != 0
            && state.quant == qwen35::StateQuant::Q8
            && fam_ok(&state.s_matrices, &self.s_matrix_bufs, n, ext.s, true)
            && fam_ok(&state.s_scales, &self.s_scale_bufs, n, ext.scales, false)
            && fam_ok(
                &state.conv_states,
                &self.conv_state_bufs,
                n,
                ext.conv,
                false,
            )
            && fam_ok(
                &state.s_ef_residual,
                &self.s_ef_residual_bufs,
                n,
                ext.ef,
                false,
            )
    }

    /// Every buffer of `state` and of this snapshot is a HIP-owned allocation.
    fn all_hip_owned(&self, state: &DeltaNetState) -> bool {
        state
            .s_matrices
            .iter()
            .chain(&state.s_scales)
            .chain(&state.conv_states)
            .chain(&state.s_ef_residual)
            .all(|t| t.buf.is_hip_allocation())
            && self.backup_buffers().all(|b| b.is_hip_allocation())
    }

    /// Append every device pointer of `state` and of this snapshot (all four
    /// families, both sides) to `out`: the caller sorts the whole chunk's
    /// pointers to prove no allocation is shared between lanes or between a
    /// lane's two banks before it exchanges owners.
    pub(crate) fn push_owner_ptrs(&self, state: &DeltaNetState, out: &mut Vec<u64>) {
        out.extend(
            state
                .s_matrices
                .iter()
                .chain(&state.s_scales)
                .chain(&state.conv_states)
                .chain(&state.s_ef_residual)
                .map(|t| t.buf.as_ptr() as u64),
        );
        out.extend(self.backup_buffers().map(|b| b.as_ptr() as u64));
    }

    /// Snapshot-side device pointers of DeltaNet layer `la`, ordered
    /// `[S, scales, conv, EF]`; `0` for a family without that layer (EF off).
    pub(crate) fn layer_state_ptrs(&self, la: usize) -> [u64; 4] {
        let p = |v: &[DeviceBuffer]| v.get(la).map_or(0, |b| b.as_ptr() as u64);
        [
            p(&self.s_matrix_bufs),
            p(&self.s_scale_bufs),
            p(&self.conv_state_bufs),
            p(&self.s_ef_residual_bufs),
        ]
    }

    /// Slice S: exchange the owned device-buffer handles of the four state
    /// families between `state` (A, the live allocations) and this snapshot
    /// (B). Afterwards the canonical `state` owns B and this snapshot owns A;
    /// `GpuTensor` shape/dtype and `DeltaNetState::quant` are untouched, and
    /// no byte moves. Host-only, allocation-free and infallible: the caller
    /// has already run the whole-chunk preflight ([`Self::cb_verify_owned_exact`]),
    /// so every family mirrors and each handle is exchanged exactly once.
    ///
    /// The bulk save/restore descriptor tables are bound to (live, backup)
    /// pointer pairs. If they are valid for `state` right now they are
    /// inverted rather than rebuilt: the old reverse table (backup B -> live
    /// A) is exactly the new forward SAVE (new live B -> new backup A), and
    /// the old forward table (A -> B) is exactly the new RESTORE (new backup
    /// A -> new live B); the fingerprint is recomputed for the new live
    /// owners. If they are stale (they were built against another live
    /// state) the route is disarmed (`bulk_n_items = 0`): the memcpy loops
    /// then serve every save/restore, never a descriptor bound to the wrong
    /// allocations. Allocations behind the descriptor buffers stay owned
    /// until [`Self::free_gpu`].
    pub(crate) fn exchange_buffers(&mut self, state: &mut DeltaNetState) {
        let bulk_valid = self.bulk_n_items != 0
            && self.bulk_fwd.is_some()
            && self.bulk_rev.is_some()
            && Self::bulk_fingerprint(state) == self.bulk_fingerprint;
        fn swap_family(live: &mut [GpuTensor], back: &mut [DeviceBuffer]) {
            for (l, b) in live.iter_mut().zip(back.iter_mut()) {
                std::mem::swap(&mut l.buf, b);
            }
        }
        swap_family(&mut state.s_matrices, &mut self.s_matrix_bufs);
        swap_family(&mut state.s_scales, &mut self.s_scale_bufs);
        swap_family(&mut state.conv_states, &mut self.conv_state_bufs);
        swap_family(&mut state.s_ef_residual, &mut self.s_ef_residual_bufs);
        if bulk_valid {
            std::mem::swap(&mut self.bulk_fwd, &mut self.bulk_rev);
            self.bulk_fingerprint = Self::bulk_fingerprint(state);
        } else {
            self.bulk_n_items = 0;
        }
    }

    /// Railgun D8: every backup buffer, family by family (fingerprinting).
    fn backup_buffers(&self) -> impl Iterator<Item = &DeviceBuffer> {
        self.s_matrix_bufs
            .iter()
            .chain(&self.s_scale_bufs)
            .chain(&self.conv_state_bufs)
            .chain(&self.s_ef_residual_bufs)
    }

    /// Copy live state → backup (S/scale/conv + EF residual).
    ///
    /// S1: on gfx1100/gfx1151/gfx1201 with armed tables and a matching fingerprint this is a
    /// single descriptor-driven `dflash_state_bulk_copy_gfx1100` launch over
    /// the forward table (plus a stream sync preserving the synchronous
    /// contract); otherwise the legacy per-tensor memcpy loop below runs.
    pub fn save_from(&mut self, state: &DeltaNetState, gpu: &mut Gpu) -> HipResult<()> {
        if let Some((table, n)) = self.bulk_table(state, gpu, true) {
            gpu.dflash_state_bulk_copy_gfx1100(table.as_ptr() as *const _, n)?;
            Self::bulk_sync(gpu)?;
            return Ok(());
        }
        // Copy the common buffer extent: a snapshot may be restored into (or
        // saved from) a DeltaNet state other than the one it was sized from
        // (VMM prefix reuse moves conversations between owners); shapes match
        // and pool rounding only pads, so the common extent covers every
        // element and never overruns either side.
        for (dst, src) in self.s_matrix_bufs.iter().zip(state.s_matrices.iter()) {
            gpu.hip
                .memcpy_dtod(dst, &src.buf, src.buf.size().min(dst.size()))?;
        }
        for (dst, src) in self.s_scale_bufs.iter().zip(state.s_scales.iter()) {
            gpu.hip
                .memcpy_dtod(dst, &src.buf, src.buf.size().min(dst.size()))?;
        }
        for (dst, src) in self.conv_state_bufs.iter().zip(state.conv_states.iter()) {
            gpu.hip
                .memcpy_dtod(dst, &src.buf, src.buf.size().min(dst.size()))?;
        }
        for (dst, src) in self
            .s_ef_residual_bufs
            .iter()
            .zip(state.s_ef_residual.iter())
        {
            gpu.hip
                .memcpy_dtod(dst, &src.buf, src.buf.size().min(dst.size()))?;
        }
        Ok(())
    }

    /// Async copy live state → backup on `stream`.
    ///
    /// Caller owns cross-stream ordering. MTP trunk-spine uses this as an
    /// opt-in experiment to overlap DN snapshot copy with proposal work.
    /// S1: with armed tables this launches the same forward table on the
    /// supplied `stream` (no sync — caller owns ordering, exactly like the
    /// async memcpy loop it replaces on launch failure or fallback).
    pub fn save_from_async_on(
        &mut self,
        state: &DeltaNetState,
        gpu: &Gpu,
        stream: &Stream,
    ) -> HipResult<()> {
        if let Some((table, n)) = self.bulk_table(state, gpu, true) {
            if gpu
                .dflash_state_bulk_copy_gfx1100_on_stream(table.as_ptr() as *const _, n, stream)
                .is_ok()
            {
                return Ok(());
            }
        }
        for (dst, src) in self.s_matrix_bufs.iter().zip(state.s_matrices.iter()) {
            gpu.hip
                .memcpy_dtod_async_at(dst, 0, &src.buf, 0, src.buf.size(), stream)?;
        }
        for (dst, src) in self.s_scale_bufs.iter().zip(state.s_scales.iter()) {
            gpu.hip
                .memcpy_dtod_async_at(dst, 0, &src.buf, 0, src.buf.size(), stream)?;
        }
        for (dst, src) in self.conv_state_bufs.iter().zip(state.conv_states.iter()) {
            gpu.hip
                .memcpy_dtod_async_at(dst, 0, &src.buf, 0, src.buf.size(), stream)?;
        }
        for (dst, src) in self
            .s_ef_residual_bufs
            .iter()
            .zip(state.s_ef_residual.iter())
        {
            gpu.hip
                .memcpy_dtod_async_at(dst, 0, &src.buf, 0, src.buf.size(), stream)?;
        }
        Ok(())
    }

    /// Copy backup → live state (rewinds recurrent + EF residual to the snapshot).
    ///
    /// S1: on gfx1100/gfx1151/gfx1201 with armed tables and a matching fingerprint this is a
    /// single descriptor-driven `dflash_state_bulk_copy_gfx1100` launch over
    /// the reverse table (plus a stream sync preserving the synchronous
    /// contract); otherwise the legacy per-tensor memcpy loop below runs.
    pub fn restore_to(&self, state: &mut DeltaNetState, gpu: &mut Gpu) -> HipResult<()> {
        if let Some((table, n)) = self.bulk_table(state, gpu, false) {
            gpu.dflash_state_bulk_copy_gfx1100(table.as_ptr() as *const _, n)?;
            Self::bulk_sync(gpu)?;
            return Ok(());
        }
        // Common extent (see `save_from`).
        for (src, dst) in self.s_matrix_bufs.iter().zip(state.s_matrices.iter()) {
            gpu.hip
                .memcpy_dtod(&dst.buf, src, src.size().min(dst.buf.size()))?;
        }
        for (src, dst) in self.s_scale_bufs.iter().zip(state.s_scales.iter()) {
            gpu.hip
                .memcpy_dtod(&dst.buf, src, src.size().min(dst.buf.size()))?;
        }
        for (src, dst) in self.conv_state_bufs.iter().zip(state.conv_states.iter()) {
            gpu.hip
                .memcpy_dtod(&dst.buf, src, src.size().min(dst.buf.size()))?;
        }
        for (src, dst) in self
            .s_ef_residual_bufs
            .iter()
            .zip(state.s_ef_residual.iter())
        {
            gpu.hip
                .memcpy_dtod(&dst.buf, src, src.size().min(dst.buf.size()))?;
        }
        Ok(())
    }

    /// Free the backup GPU buffers, consuming the snapshot. `DeviceBuffer` has
    /// no `Drop`, so a bare `Vec::clear()`/`truncate()` on a checkpoint ring
    /// orphans this device memory — the source of the per-reset GPU-memory leak
    /// that OOMs long-lived serves (a fresh `hipMalloc` per reset, never freed).
    /// Every site that drops a snapshot must route through here.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for b in self.s_matrix_bufs {
            let _ = gpu.hip.free(b);
        }
        for b in self.s_scale_bufs {
            let _ = gpu.hip.free(b);
        }
        for b in self.conv_state_bufs {
            let _ = gpu.hip.free(b);
        }
        for b in self.s_ef_residual_bufs {
            let _ = gpu.hip.free(b);
        }
        // S1: descriptor tables are device allocations too — freeing them
        // here keeps the checkpoint-ring accounting leak-free.
        if let Some(t) = self.bulk_fwd {
            let _ = gpu.hip.free(t);
        }
        if let Some(t) = self.bulk_rev {
            let _ = gpu.hip.free(t);
        }
    }

    /// Total device bytes across all backup buffers (S matrices + scales +
    /// conv rings + EF residuals). Used by the checkpoint pool for
    /// byte-bounded LRU accounting (spec §4.5 C5).
    pub fn bytes_len(&self) -> u64 {
        let mut total: u64 = 0;
        for b in &self.s_matrix_bufs {
            total += b.size() as u64;
        }
        for b in &self.s_scale_bufs {
            total += b.size() as u64;
        }
        for b in &self.conv_state_bufs {
            total += b.size() as u64;
        }
        for b in &self.s_ef_residual_bufs {
            total += b.size() as u64;
        }
        total
    }

    /// Copy this snapshot's device buffers into `dst` (device-to-device).
    /// `dst` must have been allocated with matching shapes (e.g. via
    /// [`DeltaNetSnapshot::new_for`] against the same state). Used by the
    /// checkpoint pool's `restore_private` to copy an immutable cached
    /// snapshot into a caller-owned private snapshot (spec §4.5 C5).
    pub fn copy_to(&self, dst: &mut DeltaNetSnapshot, gpu: &mut Gpu) -> HipResult<()> {
        for (src, d) in self.s_matrix_bufs.iter().zip(dst.s_matrix_bufs.iter()) {
            gpu.hip.memcpy_dtod(d, src, src.size())?;
        }
        for (src, d) in self.s_scale_bufs.iter().zip(dst.s_scale_bufs.iter()) {
            gpu.hip.memcpy_dtod(d, src, src.size())?;
        }
        for (src, d) in self.conv_state_bufs.iter().zip(dst.conv_state_bufs.iter()) {
            gpu.hip.memcpy_dtod(d, src, src.size())?;
        }
        for (src, d) in self
            .s_ef_residual_bufs
            .iter()
            .zip(dst.s_ef_residual_bufs.iter())
        {
            gpu.hip.memcpy_dtod(d, src, src.size())?;
        }
        Ok(())
    }
}

/// A series of `n_slots` `DeltaNetSnapshot` slots, used by the tape-replay
/// rollback path. After each verify forward step writes its post-state into
/// the next slot, `restore_from(accept_len + 1)` jumps the live DN state
/// to exactly `start + accept_len + 1` positions of advance — no replay
/// loop needed.
///
/// VRAM cost: `n_slots × (one DeltaNetSnapshot)`. For Qwen3.5-4B and
/// `n_slots = B + 1 = 17`, that's roughly 100 MB; for 9B it scales with
/// the hybrid layer count.
pub struct DeltaNetTape {
    pub slots: Vec<DeltaNetSnapshot>,
}

/// Innovation tape for the GatedDeltaNet recurrence. During a batched verify
/// forward we capture the per-LA-layer pre-conv1d `qkv` projection and the
/// post-sigmoid `(α, β)` for every block position. On rollback we replay
/// conv1d + QK-norm + repeat-interleave + GDN for `accept_len + 1` steps
/// against the pre-verify DN snapshot — advancing both S-state AND
/// conv_state correctly, no full target re-run needed.
///
/// Why pre-conv1d qkv instead of post-conv1d (q, k, v): conv_state is a
/// recurrent buffer advanced by conv1d_silu_split. If we skipped conv1d on
/// replay the next verify would see a stale conv_state reflecting the
/// previous full-B aborted trajectory rather than the accepted prefix —
/// small numerical drift that empirically halves τ on our 4B hybrid target.
/// Running conv1d from the captured qkv advances conv_state to the right
/// place.
pub struct GdnTape {
    pub max_n: usize,
    pub qkv_dim: usize,
    pub v_dim: usize,
    pub k_dim: usize,
    pub n_v_heads: usize,
    pub n_key_heads: usize,
    pub value_head_dim: usize,
    pub key_head_dim: usize,
    /// Per-LA-layer [max_n × qkv_dim] F32 — raw qkvza projection output.
    pub qkv_bufs: Vec<GpuTensor>,
    /// Per-LA-layer [max_n × n_v_heads] F32 — post-sigmoid_alpha_gate.
    pub alpha_bufs: Vec<GpuTensor>,
    pub beta_bufs: Vec<GpuTensor>,
    /// Replay scratch (shared across layers — serial replay is fine).
    pub q_raw_scratch: GpuTensor, // [max_n × k_dim]
    pub k_raw_scratch: GpuTensor, // [max_n × k_dim]
    pub v_scratch: GpuTensor,     // [max_n × v_dim]
    pub q_scratch: GpuTensor,     // [max_n × v_dim] (post repeat-interleave)
    pub k_scratch: GpuTensor,     // [max_n × v_dim]
    pub attn_scratch: GpuTensor,  // [max_n × v_dim]
    /// Railgun E0 / L6c multi-layer replay state (gfx1201). Armed by
    /// [`Self::new_for_config`] whenever the process supports the route
    /// (`Gpu::gdn_replay_ml_supported`); `None` for a capture-only tape or an
    /// unsupported process. An admitted replay on an unarmed tape is an
    /// error, never a silent switch to the per-layer route.
    replay_ml: std::sync::Mutex<Option<GdnReplayMl>>,
}

impl GdnTape {
    /// Replay tape: [`Self::new_capture_only`] plus the multi-layer replay
    /// tables, armed here so an allocation failure refuses the tape at
    /// provision/admission instead of changing the committed DeltaNet
    /// numerics at a later accept. Any failure frees everything allocated
    /// and returns the error.
    pub fn new_for_config(
        gpu: &mut Gpu,
        config: &qwen35::Qwen35Config,
        max_n: usize,
    ) -> HipResult<Self> {
        Self::new_armed(gpu, config, max_n, false)
    }

    /// Device bytes [`Self::new_for_config`] allocates for `max_n` rows, computed
    /// from the config alone (no allocation): the per-LA-layer qkv/alpha/beta
    /// capture buffers plus the six replay scratches (all F32). With
    /// `replay_ml` the multi-layer replay arming is charged too: the shared
    /// q/k/v/out scratch (4 regions of `n_la * rows * v_dim` F32, `rows` capped
    /// at the replay step ceiling) and the two per-layer pointer tables. The
    /// scratch is shared per shape on a `Gpu`, so this is an upper bound when
    /// another tape of the same shape is already armed.
    pub fn requested_bytes(config: &qwen35::Qwen35Config, max_n: usize, replay_ml: bool) -> usize {
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let n_v_heads = config.linear_num_value_heads;
        let n_la = config
            .layer_types
            .iter()
            .filter(|t| **t == qwen35::LayerType::LinearAttention)
            .count();
        let f32_elems = n_la * max_n * (qkv_dim + 2 * n_v_heads) + max_n * (2 * k_dim + 4 * v_dim);
        let mut bytes = f32_elems * 4;
        if replay_ml && n_la != 0 {
            let rows = max_n.min(rdna_compute::dflash_gdn_replay::DFLASH_GDN_REPLAY_MAX_STEPS);
            let shape = rdna_compute::dflash_gdn_replay::GdnReplayScratchShape {
                n_layers: n_la,
                rows,
                v_dim,
            };
            bytes += 4 * shape.region_bytes()
                + n_la
                    * (std::mem::size_of::<rdna_compute::dflash_gdn_replay::DflashReplayPreLayer>(
                    ) + std::mem::size_of::<rdna_compute::dflash_gdn_replay::GdnLayerTable>());
        }
        bytes
    }

    /// DFlash replay tape: [`Self::new_for_config`] plus, when
    /// `HIPFIRE_DN_SNAPSHOT_FLIP` is on, the D8 snapshot-source tables that
    /// [`Self::replay_gdn_from_snapshot`] needs.
    pub fn new_for_dflash(
        gpu: &mut Gpu,
        config: &qwen35::Qwen35Config,
        max_n: usize,
    ) -> HipResult<Self> {
        Self::new_armed(gpu, config, max_n, gpu.flags.dn_snapshot_flip)
    }

    fn new_armed(
        gpu: &mut Gpu,
        config: &qwen35::Qwen35Config,
        max_n: usize,
        with_from: bool,
    ) -> HipResult<Self> {
        let mut tape = Self::new_capture_only(gpu, config, max_n)?;
        let n_la = tape.qkv_bufs.len();
        if n_la != 0
            && gpu.gdn_replay_ml_supported(
                tape.n_v_heads,
                tape.n_key_heads,
                tape.key_head_dim,
                tape.value_head_dim,
            )
        {
            match GdnReplayMl::arm(gpu, n_la, max_n, tape.v_dim, with_from) {
                Ok(ml) => *tape.replay_ml.get_mut().unwrap_or_else(|e| e.into_inner()) = Some(ml),
                Err(e) => {
                    eprintln!(
                        "GdnTape: arming the multi-layer GDN replay failed ({e}); \
                         refusing the tape (n_la={n_la} max_n={max_n} d8={with_from})"
                    );
                    tape.free_gpu(gpu);
                    return Err(e);
                }
            }
        }
        Ok(tape)
    }

    /// Capture-only tape: the per-layer qkv/α/β capture buffers and the
    /// per-layer replay scratch, without the multi-layer replay tables. For
    /// tapes that are only written by a verify forward and read by an
    /// explicit per-layer repair (the serve engine's batched verify tape);
    /// [`Self::replay_gdn`] on it fails wherever the multi-layer route
    /// would be admitted. A mid-way allocation failure frees what was
    /// allocated.
    pub fn new_capture_only(
        gpu: &mut Gpu,
        config: &qwen35::Qwen35Config,
        max_n: usize,
    ) -> HipResult<Self> {
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let n_v_heads = config.linear_num_value_heads;
        let n_key_heads = config.linear_num_key_heads;
        let n_la_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == qwen35::LayerType::LinearAttention)
            .count();

        // Per layer (qkv, α, β), then the six replay scratches, in the
        // historical allocation order.
        let mut sizes = Vec::with_capacity(3 * n_la_layers + 6);
        for _ in 0..n_la_layers {
            sizes.extend([max_n * qkv_dim, max_n * n_v_heads, max_n * n_v_heads]);
        }
        sizes.extend([
            max_n * k_dim,
            max_n * k_dim,
            max_n * v_dim,
            max_n * v_dim,
            max_n * v_dim,
            max_n * v_dim,
        ]);
        let mut bufs = Vec::with_capacity(sizes.len());
        for n in sizes {
            match gpu.alloc_tensor(&[n], rdna_compute::DType::F32) {
                Ok(t) => bufs.push(t),
                Err(e) => {
                    for t in bufs {
                        let _ = gpu.free_tensor(t);
                    }
                    return Err(e);
                }
            }
        }
        let mut bufs = bufs.into_iter();
        let mut next = || bufs.next().expect("allocated above");
        let mut qkv_bufs = Vec::with_capacity(n_la_layers);
        let mut alpha_bufs = Vec::with_capacity(n_la_layers);
        let mut beta_bufs = Vec::with_capacity(n_la_layers);
        for _ in 0..n_la_layers {
            qkv_bufs.push(next());
            alpha_bufs.push(next());
            beta_bufs.push(next());
        }

        Ok(Self {
            max_n,
            qkv_dim,
            v_dim,
            k_dim,
            n_v_heads,
            n_key_heads,
            value_head_dim: config.linear_value_head_dim,
            key_head_dim: config.linear_key_head_dim,
            qkv_bufs,
            alpha_bufs,
            beta_bufs,
            q_raw_scratch: next(),
            k_raw_scratch: next(),
            v_scratch: next(),
            q_scratch: next(),
            k_scratch: next(),
            attn_scratch: next(),
            replay_ml: std::sync::Mutex::new(None),
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self
            .qkv_bufs
            .into_iter()
            .chain(self.alpha_bufs.into_iter())
            .chain(self.beta_bufs.into_iter())
        {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.q_raw_scratch);
        let _ = gpu.free_tensor(self.k_raw_scratch);
        let _ = gpu.free_tensor(self.v_scratch);
        let _ = gpu.free_tensor(self.q_scratch);
        let _ = gpu.free_tensor(self.k_scratch);
        let _ = gpu.free_tensor(self.attn_scratch);
        if let Some(ml) = self
            .replay_ml
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
        {
            ml.free_gpu(gpu);
        }
    }

    /// Pointer fingerprint of everything the multi-layer tables reference
    /// outside the tape itself (conv weights and live DN state). A mismatch
    /// rebuilds the tables before launch; the tape's own buffers are fixed
    /// for its lifetime.
    fn replay_ml_fingerprint(
        weights: &qwen35::Qwen35Weights,
        config: &qwen35::Qwen35Config,
        dn_state: &qwen35::DeltaNetState,
    ) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x0100_0000_01b3);
        };
        let mut la_idx = 0usize;
        for (layer_idx, lt) in config.layer_types.iter().enumerate() {
            if *lt != qwen35::LayerType::LinearAttention {
                continue;
            }
            let conv_weight = match &weights.layers[layer_idx] {
                qwen35::LayerWeights::DeltaNet(l) => &l.conv_weight,
                qwen35::LayerWeights::DeltaNetMoe(l) => &l.conv_weight,
                _ => unreachable!("LA layer type mismatch in replay_gdn"),
            };
            mix(conv_weight.buf.as_ptr() as u64);
            mix(dn_state.conv_states[la_idx].buf.as_ptr() as u64);
            mix(dn_state.s_matrices[la_idx].buf.as_ptr() as u64);
            mix(dn_state.s_scales[la_idx].buf.as_ptr() as u64);
            mix(dn_state
                .ef_residual(la_idx)
                .map_or(0, |t| t.buf.as_ptr() as u64));
            la_idx += 1;
        }
        mix(la_idx as u64);
        h
    }

    /// Railgun E0 / L6c: host-side admission of the two-launch replay
    /// (arch, opt-out, open Redline recording, quant/requant mode, dims, step
    /// count). Shared by the in-place and the D8 snapshot-source routes.
    fn replay_ml_admits(
        &self,
        gpu: &Gpu,
        dn_state: &qwen35::DeltaNetState,
        n_steps: usize,
    ) -> bool {
        let n_la = self.qkv_bufs.len();
        dn_state.quant == qwen35::StateQuant::Q8
            && n_la != 0
            && n_steps <= self.max_n
            && dn_state.s_matrices.len() == n_la
            && dn_state.conv_states.len() == n_la
            && gpu.gdn_replay_ml_eligible(
                self.n_v_heads,
                self.n_key_heads,
                self.key_head_dim,
                self.value_head_dim,
                n_steps,
            )
    }

    /// Railgun E0 / L6c: one preamble row and one recurrence row per LA layer,
    /// with the live DeltaNet state as both source and destination.
    fn replay_ml_rows(
        &self,
        ml: &GdnReplayMl,
        weights: &qwen35::Qwen35Weights,
        config: &qwen35::Qwen35Config,
        dn_state: &qwen35::DeltaNetState,
    ) -> (
        Vec<rdna_compute::dflash_gdn_replay::DflashReplayPreLayer>,
        Vec<rdna_compute::dflash_gdn_replay::GdnLayerTable>,
    ) {
        let n_la = self.qkv_bufs.len();
        let row = ml.shape.rows * self.v_dim * 4;
        let mut pre = Vec::with_capacity(n_la);
        let mut gdn = Vec::with_capacity(n_la);
        let mut la_idx = 0usize;
        for (layer_idx, lt) in config.layer_types.iter().enumerate() {
            if *lt != qwen35::LayerType::LinearAttention {
                continue;
            }
            let conv_weight = match &weights.layers[layer_idx] {
                qwen35::LayerWeights::DeltaNet(l) => &l.conv_weight,
                qwen35::LayerWeights::DeltaNetMoe(l) => &l.conv_weight,
                _ => unreachable!("LA layer type mismatch in replay_gdn"),
            };
            let off = (la_idx * row) as u64;
            let region = ml.shape.region_bytes() as u64;
            let q = ml.base + off;
            let k = ml.base + region + off;
            let v = ml.base + 2 * region + off;
            pre.push(rdna_compute::dflash_gdn_replay::DflashReplayPreLayer {
                qkv_tape: self.qkv_bufs[la_idx].buf.as_ptr() as u64,
                conv_w: conv_weight.buf.as_ptr() as u64,
                conv_state: dn_state.conv_states[la_idx].buf.as_ptr() as u64,
                v_out: v,
                q_dst: q,
                k_dst: k,
            });
            gdn.push(rdna_compute::dflash_gdn_replay::GdnLayerTable {
                q,
                k,
                v,
                gate: self.alpha_bufs[la_idx].buf.as_ptr() as u64,
                beta: self.beta_bufs[la_idx].buf.as_ptr() as u64,
                s_q8: dn_state.s_matrices[la_idx].buf.as_ptr() as u64,
                s_scales: dn_state.s_scales[la_idx].buf.as_ptr() as u64,
                output: ml.base + 3 * region + off,
                ef: dn_state
                    .ef_residual(la_idx)
                    .map_or(0, |t| t.buf.as_ptr() as u64),
            });
            la_idx += 1;
        }
        (pre, gdn)
    }

    /// Railgun E0 / L6c: two-launch replay of every LA layer (preamble, then
    /// recurrence) over device pointer tables. Returns `Ok(false)` — the
    /// caller then runs the per-layer launches — only when the route is
    /// ineligible (arch, opt-out, open Redline recording, quant/requant mode,
    /// dims, step count). An admitted replay on a tape whose tables were not
    /// armed at construction (a capture-only tape) is an error: the
    /// per-layer route is not byte-identical to this one on the production
    /// model, so it is never a silent fallback.
    fn replay_gdn_ml(
        &self,
        gpu: &mut Gpu,
        weights: &qwen35::Qwen35Weights,
        config: &qwen35::Qwen35Config,
        dn_state: &qwen35::DeltaNetState,
        n_steps: usize,
    ) -> HipResult<bool> {
        if !self.replay_ml_admits(gpu, dn_state, n_steps) {
            return Ok(false);
        }
        let n_la = self.qkv_bufs.len();
        let mut guard = self.replay_ml.lock().unwrap_or_else(|e| e.into_inner());
        let Some(ml) = guard.as_mut() else {
            return Err(Self::unarmed_error("multi-layer"));
        };
        let fp = Self::replay_ml_fingerprint(weights, config, dn_state);
        if ml.fingerprint != Some(fp) {
            // Tables may still be read by an in-flight replay: drain first.
            if ml.fingerprint.is_some() {
                gpu.hip.device_synchronize()?;
            }
            let (pre, gdn) = self.replay_ml_rows(ml, weights, config, dn_state);
            gpu.hip.memcpy_htod(
                &ml.pre_table,
                rdna_compute::dflash_gdn_replay::table_bytes(&pre),
            )?;
            gpu.hip.memcpy_htod(
                &ml.gdn_table,
                rdna_compute::dflash_gdn_replay::table_bytes(&gdn),
            )?;
            ml.fingerprint = Some(fp);
        }
        ml.order(gpu)?;
        let hd = self.key_head_dim;
        gpu.dflash_gdn_replay_pre_ml(
            ml.pre_table.as_ptr() as *const _,
            n_la,
            self.n_v_heads,
            self.n_key_heads,
            self.k_dim,
            self.v_dim,
            self.qkv_dim,
            n_steps,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
        )?;
        gpu.gated_delta_net_q8_fast_ml(
            ml.gdn_table.as_ptr() as *const _,
            n_la,
            n_steps,
            self.n_v_heads,
            self.value_head_dim,
        )?;
        Ok(true)
    }

    /// Railgun D8: whether [`Self::replay_gdn_from_snapshot`] can run —
    /// `HIPFIRE_DN_SNAPSHOT_FLIP` not `0`, the two-launch replay admits this call,
    /// and `snap` mirrors `dn_state` (one same-size backup per live tensor in
    /// every family). Host-only; no allocation.
    pub fn replay_from_snapshot_admits(
        &self,
        gpu: &Gpu,
        dn_state: &qwen35::DeltaNetState,
        snap: &DeltaNetSnapshot,
        n_steps: usize,
    ) -> bool {
        gpu.flags.dn_snapshot_flip
            && self.replay_ml_admits(gpu, dn_state, n_steps)
            && snap.mirrors(dn_state)
    }

    /// Are this tape's D8 snapshot-source tables armed (`ml.from` exists)?
    /// [`Self::replay_from_snapshot_admits`] alone does not prove it: a
    /// capture-only tape, or a DFlash tape built with
    /// `HIPFIRE_DN_SNAPSHOT_FLIP=0`, has no `from` tables, and an admitted
    /// snapshot-source replay on it is an error. The save-free CB verify
    /// (which commits through that replay) requires this before it exchanges
    /// any owner. Reads the existing `replay_ml` under its existing lock.
    pub fn snapshot_source_armed(&self) -> bool {
        self.replay_ml
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|ml| ml.from.is_some())
    }

    /// Railgun D8: [`Self::replay_gdn`] for `n_steps` that reads the
    /// pre-verify state (conv ring, S, scales, EF residual) from `snap`'s
    /// buffers and writes the advanced state to `dn_state`, in the same two
    /// launches as the E0 multi-layer replay. It is the byte-for-byte result
    /// of `snap.restore_to(dn_state)` followed by the in-place replay (every
    /// live element is written; the kernels are the E0 ones with only the
    /// load addresses changed), without the restore copy, and `snap` is not
    /// written, so it stays the pre-window state that terminal repair
    /// restores. Returns `Ok(false)` without touching any state when
    /// [`Self::replay_from_snapshot_admits`] is false; the caller then
    /// restores and replays in place. An admitted call on a tape without
    /// armed snapshot-source tables is an error and touches no state.
    #[allow(clippy::too_many_arguments)]
    pub fn replay_gdn_from_snapshot(
        &self,
        gpu: &mut Gpu,
        weights: &qwen35::Qwen35Weights,
        config: &qwen35::Qwen35Config,
        dn_state: &mut qwen35::DeltaNetState,
        snap: &DeltaNetSnapshot,
        n_steps: usize,
    ) -> HipResult<bool> {
        if !self.replay_from_snapshot_admits(gpu, dn_state, snap, n_steps) {
            return Ok(false);
        }
        let n_la = self.qkv_bufs.len();
        let mut guard = self.replay_ml.lock().unwrap_or_else(|e| e.into_inner());
        let Some(ml) = guard.as_mut().filter(|ml| ml.from.is_some()) else {
            return Err(Self::unarmed_error("snapshot-source (D8)"));
        };
        let fp = {
            let mut h = Self::replay_ml_fingerprint(weights, config, dn_state);
            for buf in snap.backup_buffers() {
                h ^= buf.as_ptr() as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            h
        };
        // The owner exchange alternates two pointer configurations per lane, so
        // `GdnReplayMlFrom` keeps a table pair per configuration: a step whose
        // fingerprint is in either slot launches that slot's tables with no
        // drain and no upload; only a configuration seen for the first time
        // (or evicted) rewrites the least recently used slot.
        let lookup = match ml.from.as_ref() {
            Some(from) => from.slots.lookup(&fp),
            None => unreachable!("armed at construction"),
        };
        let slot = match lookup {
            Ok(slot) => {
                if let Some(from) = ml.from.as_mut() {
                    from.slots.mark_used(slot);
                }
                slot
            }
            Err(victim) => {
                let (pre, gdn) = self.replay_ml_rows(ml, weights, config, dn_state);
                let pre: Vec<_> = pre
                    .into_iter()
                    .enumerate()
                    .map(
                        |(la, base)| rdna_compute::dflash_gdn_replay::DflashReplayPreLayerFrom {
                            base,
                            conv_state_src: snap.conv_state_bufs[la].as_ptr() as u64,
                        },
                    )
                    .collect();
                let gdn: Vec<_> = gdn
                    .into_iter()
                    .enumerate()
                    .map(
                        |(la, base)| rdna_compute::dflash_gdn_replay::GdnLayerTableFrom {
                            base,
                            s_q8_src: snap.s_matrix_bufs[la].as_ptr() as u64,
                            s_scales_src: snap.s_scale_bufs[la].as_ptr() as u64,
                            ef_src: snap
                                .s_ef_residual_bufs
                                .get(la)
                                .map_or(0, |b| b.as_ptr() as u64),
                        },
                    )
                    .collect();
                let Some(from) = ml.from.as_mut() else {
                    unreachable!("armed at construction")
                };
                // Tables may still be read by an in-flight replay: drain first.
                if from.slots.written(victim) {
                    gpu.hip.device_synchronize()?;
                }
                // Unkeyed while its tables are rewritten: a failed upload never
                // leaves a fingerprint over half-written tables.
                from.slots.take(victim);
                gpu.hip.memcpy_htod(
                    &from.pre_tables[victim],
                    rdna_compute::dflash_gdn_replay::table_bytes(&pre),
                )?;
                gpu.hip.memcpy_htod(
                    &from.gdn_tables[victim],
                    rdna_compute::dflash_gdn_replay::table_bytes(&gdn),
                )?;
                from.slots.set(victim, fp);
                victim
            }
        };
        ml.order(gpu)?;
        let Some(from) = ml.from.as_ref() else {
            unreachable!("armed at construction")
        };
        let hd = self.key_head_dim;
        gpu.dflash_gdn_replay_pre_ml_from(
            from.pre_tables[slot].as_ptr() as *const _,
            n_la,
            self.n_v_heads,
            self.n_key_heads,
            self.k_dim,
            self.v_dim,
            self.qkv_dim,
            n_steps,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
        )?;
        gpu.gated_delta_net_q8_fast_ml_from(
            from.gdn_tables[slot].as_ptr() as *const _,
            n_la,
            n_steps,
            self.n_v_heads,
            self.value_head_dim,
        )?;
        Ok(true)
    }

    /// Replay the full LA sub-pipeline (conv1d + qk-l2norm + repeat-interleave +
    /// GDN recurrence) for `n_steps` across all LinearAttention layers. Advances
    /// both `dn_state.s_matrices`/`s_scales` AND `dn_state.conv_states` by
    /// exactly `n_steps` single-token updates. Caller must have restored the
    /// DN snapshot to the pre-verify point before calling this.
    ///
    /// Graph-capture path (OPT-IN with HIPFIRE_REPLAY_GRAPH=1): per distinct
    /// n_steps, the first call runs direct as a warmup, the second captures a
    /// hipGraph, and subsequent calls replay the graph. Eligibility:
    /// gpu.active_stream must be Some (so a verify-graph path that already
    /// created one has run first in this cycle).
    ///
    /// MEASURED NULL RESULT (2026-04-21, 27B HumanEval @ accept≈10):
    /// 77.27 → 77.41 tok/s (+0.18 %, noise). τ and mean_committed byte-exact
    /// across A/B. Replay's ~192 kernel launches per cycle add ~0.3 ms of
    /// dispatch API time out of a 130 ms cycle — graphing them saves that
    /// 0.3 ms but cycle cost lives in GDN kernel execution time (scales
    /// linearly with n_steps). Kept as opt-in infrastructure for future
    /// launch-overhead-dominated workloads (smaller models, finer kernels).
    pub fn replay_gdn(
        &self,
        gpu: &mut Gpu,
        weights: &qwen35::Qwen35Weights,
        config: &qwen35::Qwen35Config,
        dn_state: &mut qwen35::DeltaNetState,
        n_steps: usize,
    ) -> HipResult<()> {
        let graph_enabled = hipfire_config::developer_var("HIPFIRE_REPLAY_GRAPH")
            .ok()
            .as_deref()
            == Some("1");
        let can_graph = graph_enabled && gpu.active_stream.is_some();

        if can_graph && gpu.graphs.replay_has_graph(n_steps) {
            if let Some(ml) = self
                .replay_ml
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
            {
                ml.order(gpu)?;
            }
            return gpu.graphs.replay_graph_launch(
                &gpu.hip,
                gpu.device_id,
                gpu.active_stream.as_ref().unwrap(),
                n_steps,
            );
        }

        if can_graph && gpu.graphs.replay_needs_warmup(n_steps) {
            self.replay_gdn_inner(gpu, weights, config, dn_state, n_steps)?;
            gpu.graphs.replay_mark_warmup_done(n_steps);
            return Ok(());
        }

        if can_graph {
            gpu.graphs.begin_replay_graph_capture(
                &gpu.hip,
                gpu.device_id,
                gpu.active_stream.as_ref().unwrap(),
                n_steps,
            )?;
            let r = self.replay_gdn_inner(gpu, weights, config, dn_state, n_steps);
            if r.is_ok() {
                gpu.graphs.end_replay_graph_capture(
                    &gpu.hip,
                    gpu.device_id,
                    gpu.active_stream.as_ref().unwrap(),
                )?;
                // Same pattern as verify_graph: hipStreamBeginCapture records
                // without executing, so launch once here to apply this cycle's
                // state updates.
                gpu.graphs.replay_graph_launch(
                    &gpu.hip,
                    gpu.device_id,
                    gpu.active_stream.as_ref().unwrap(),
                    n_steps,
                )?;
                return Ok(());
            } else {
                let _ = gpu
                    .hip
                    .stream_end_capture(gpu.active_stream.as_ref().unwrap());
                gpu.graphs.capture_mode = false;
                gpu.graphs.capture_blobs.clear();
                return r;
            }
        }

        self.replay_gdn_inner(gpu, weights, config, dn_state, n_steps)
    }

    /// Direct kernel path — the original `replay_gdn` body, retained as a
    /// helper so both the graph-warmup first call and the non-graph fallback
    /// share one implementation.
    fn replay_gdn_inner(
        &self,
        gpu: &mut Gpu,
        weights: &qwen35::Qwen35Weights,
        config: &qwen35::Qwen35Config,
        dn_state: &mut qwen35::DeltaNetState,
        n_steps: usize,
    ) -> HipResult<()> {
        assert!(
            n_steps <= self.max_n,
            "replay_gdn: n_steps {n_steps} > max_n"
        );
        if self.replay_gdn_ml(gpu, weights, config, dn_state, n_steps)? {
            return Ok(());
        }
        let n_v_heads = self.n_v_heads;
        let n_key_heads = self.n_key_heads;
        let hd = self.key_head_dim;
        let v_dim = self.v_dim;
        let k_dim = self.k_dim;
        let value_head_dim = self.value_head_dim;
        let mut la_idx = 0usize;

        for (layer_idx, lt) in config.layer_types.iter().enumerate() {
            if *lt != qwen35::LayerType::LinearAttention {
                continue;
            }
            let conv_weight = match &weights.layers[layer_idx] {
                qwen35::LayerWeights::DeltaNet(l) => &l.conv_weight,
                qwen35::LayerWeights::DeltaNetMoe(l) => &l.conv_weight,
                _ => unreachable!("LA layer type mismatch in replay_gdn"),
            };

            // S5-gdn-pre-tape-fusion fast path: one launch for conv1d + QK
            // norm/interleave from the taped raw qkv. The launcher enforces
            // the exact route (gfx1100, hd == 128, consistent dims,
            // 1 <= n_steps <= 16); any decline runs the pre-change steps
            // 1-3 below launch-for-launch. q_raw/k_raw keep the old
            // in-place-norm postcondition (normed values), so step 4 and
            // every later consumer observe identical bytes.
            let fused = if gpu.flags.gdn_pre_fuse_off {
                false
            } else {
                gpu.dflash_gdn_pre_replay_gfx1100(
                    &self.qkv_bufs[la_idx],
                    conv_weight,
                    &dn_state.conv_states[la_idx],
                    &self.q_raw_scratch,
                    &self.k_raw_scratch,
                    &self.v_scratch,
                    &self.q_scratch,
                    &self.k_scratch,
                    n_v_heads,
                    n_key_heads,
                    hd,
                    k_dim,
                    v_dim,
                    self.qkv_dim,
                    n_steps,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                )?
            };
            if !fused {
                // 1. conv1d + SiLU + split — advances conv_state, writes
                //    (q_raw, k_raw, v) into scratch.
                gpu.conv1d_silu_split_f32_n(
                    &self.q_raw_scratch,
                    &self.k_raw_scratch,
                    &self.v_scratch,
                    &self.qkv_bufs[la_idx],
                    conv_weight,
                    &dn_state.conv_states[la_idx],
                    k_dim,
                    v_dim,
                    n_steps,
                )?;

                // 2. L2 norm(Q) + L2 norm(K) + scale(Q).
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    &self.q_raw_scratch,
                    &self.k_raw_scratch,
                    n_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                    n_steps,
                )?;

                // 3. Repeat-interleave if GQA.
                if n_key_heads < n_v_heads {
                    let ratio = n_v_heads / n_key_heads;
                    gpu.repeat_interleave_qk_f32_batched(
                        &self.q_raw_scratch,
                        &self.k_raw_scratch,
                        &self.q_scratch,
                        &self.k_scratch,
                        n_key_heads,
                        ratio,
                        hd,
                        n_steps,
                    )?;
                } else {
                    let bytes = n_steps * k_dim * 4;
                    gpu.hip.memcpy_dtod_at(
                        &self.q_scratch.buf,
                        0,
                        &self.q_raw_scratch.buf,
                        0,
                        bytes,
                    )?;
                    gpu.hip.memcpy_dtod_at(
                        &self.k_scratch.buf,
                        0,
                        &self.k_raw_scratch.buf,
                        0,
                        bytes,
                    )?;
                }
            }

            // 4. GDN recurrence — advances S_state.
            match dn_state.quant {
                qwen35::StateQuant::FP32 => gpu.gated_delta_net_f32_batch_seq(
                    &self.q_scratch,
                    &self.k_scratch,
                    &self.v_scratch,
                    &self.alpha_bufs[la_idx],
                    &self.beta_bufs[la_idx],
                    &dn_state.s_matrices[la_idx],
                    &self.attn_scratch,
                    n_steps,
                    n_v_heads,
                    value_head_dim,
                )?,
                qwen35::StateQuant::Q8 => gpu.gated_delta_net_q8_batch_seq(
                    &self.q_scratch,
                    &self.k_scratch,
                    &self.v_scratch,
                    &self.alpha_bufs[la_idx],
                    &self.beta_bufs[la_idx],
                    &dn_state.s_matrices[la_idx],
                    &dn_state.s_scales[la_idx],
                    &self.attn_scratch,
                    n_steps,
                    n_v_heads,
                    value_head_dim,
                    dn_state.ef_residual(la_idx),
                )?,
                qwen35::StateQuant::Q4 => gpu.gated_delta_net_q4(
                    &self.q_scratch,
                    &self.k_scratch,
                    &self.v_scratch,
                    &self.alpha_bufs[la_idx],
                    &self.beta_bufs[la_idx],
                    &dn_state.s_matrices[la_idx],
                    &dn_state.s_scales[la_idx],
                    &self.attn_scratch,
                    n_steps,
                    n_v_heads,
                    value_head_dim,
                )?,
            }

            la_idx += 1;
        }
        Ok(())
    }

    fn unarmed_error(route: &str) -> HipError {
        HipError::new(
            0,
            &format!(
                "GdnTape: the {route} GDN replay is admitted but its tables were not armed at \
                 construction (capture-only tape); refusing to commit DeltaNet state through \
                 the per-layer route"
            ),
        )
    }
}

/// Railgun E0 / L6c: device state of the multi-layer replay. The q, k, v and
/// (dead) recurrence-output scratch are the four regions of the `Gpu`'s
/// shared replay scratch for `shape`
/// (`rdna_compute::dflash_gdn_replay::GdnReplaySharedScratch`, which states
/// why sharing is sound), based at `base`. Within a region each LA layer owns
/// rows `[la * rows * v_dim, (la + 1) * rows * v_dim)`, so all layers run
/// concurrently; `rows` is the tape's `max_n` capped at the route's step
/// ceiling. `fingerprint` is `None` until the tables are first written.
struct GdnReplayMl {
    shape: rdna_compute::dflash_gdn_replay::GdnReplayScratchShape,
    base: u64,
    pre_table: DeviceBuffer,
    gdn_table: DeviceBuffer,
    fingerprint: Option<u64>,
    /// Railgun D8 snapshot-source tables; armed by
    /// [`GdnTape::new_for_dflash`] when `HIPFIRE_DN_SNAPSHOT_FLIP` is on.
    from: Option<GdnReplayMlFrom>,
}

/// Two-slot cache bookkeeping with least-recently-used replacement: each slot
/// holds the key its device table currently matches (`None` = never written, or
/// invalidated). It owns no device memory; the caller keeps the per-slot
/// buffers and writes slot `victim` only on a miss.
///
/// The save-free CB verify exchanges the live and snapshot owners of every
/// lane before each step, so the pointer-keyed tables of one lane alternate
/// between exactly two configurations. One slot would miss (drain + upload)
/// every step; two hit after the first two steps.
pub(crate) struct TwoSlot<K> {
    keys: [Option<K>; 2],
    /// The slot used most recently.
    mru: usize,
}

impl<K> TwoSlot<K> {
    pub(crate) fn new() -> Self {
        Self {
            keys: [None, None],
            mru: 0,
        }
    }

    /// `Ok(slot)` when `hit` accepts that slot's key; `Err(victim)` otherwise:
    /// the slot to overwrite (an unwritten slot first, else the one not used
    /// most recently). Pure: neither outcome changes the recency.
    pub(crate) fn lookup_by(&self, mut hit: impl FnMut(&K) -> bool) -> Result<usize, usize> {
        for slot in [self.mru, 1 - self.mru] {
            if self.keys[slot].as_ref().is_some_and(|k| hit(k)) {
                return Ok(slot);
            }
        }
        Err(if self.keys[0].is_none() {
            0
        } else if self.keys[1].is_none() {
            1
        } else {
            1 - self.mru
        })
    }

    /// Record a hit on `slot`.
    pub(crate) fn mark_used(&mut self, slot: usize) {
        self.mru = slot;
    }

    /// Key `slot` to its freshly written table and mark it used.
    pub(crate) fn set(&mut self, slot: usize, key: K) {
        self.keys[slot] = Some(key);
        self.mru = slot;
    }

    /// Unkey `slot` (its table is about to be rewritten, or is unusable);
    /// returns the old key.
    pub(crate) fn take(&mut self, slot: usize) -> Option<K> {
        self.keys[slot].take()
    }

    /// Has `slot` been written (so in-flight work may still read its table)?
    pub(crate) fn written(&self, slot: usize) -> bool {
        self.keys[slot].is_some()
    }
}

impl<K: PartialEq> TwoSlot<K> {
    /// [`Self::lookup_by`] for a slot whose key equals `key`.
    pub(crate) fn lookup(&self, key: &K) -> Result<usize, usize> {
        self.lookup_by(|k| k == key)
    }
}

/// Railgun D8: device tables of [`GdnTape::replay_gdn_from_snapshot`]
/// (`DflashReplayPreLayerFrom` / `GdnLayerTableFrom` rows). They reuse
/// [`GdnReplayMl`]'s q/k/v/out scratch. Two table pairs, one per slot of
/// `slots` (see [`TwoSlot`]): the save-free owner exchange alternates the
/// live/snapshot pointer sets, so each configuration keeps its own pair. A
/// slot's key is the fingerprint of the live state, the conv weights and the
/// snapshot buffers its pair was written for; unkeyed until first written.
struct GdnReplayMlFrom {
    pre_tables: [DeviceBuffer; 2],
    gdn_tables: [DeviceBuffer; 2],
    slots: TwoSlot<u64>,
}

impl GdnReplayMlFrom {
    /// JIT both snapshot-source kernels and allocate the two table pairs. Any
    /// failure frees what was allocated and returns the error.
    fn arm(gpu: &mut Gpu, n_la: usize) -> HipResult<Self> {
        gpu.ensure_dflash_gdn_replay_ml_from()?;
        let pre_bytes =
            n_la * std::mem::size_of::<rdna_compute::dflash_gdn_replay::DflashReplayPreLayerFrom>();
        let gdn_bytes =
            n_la * std::mem::size_of::<rdna_compute::dflash_gdn_replay::GdnLayerTableFrom>();
        let alloc_pair = |gpu: &mut Gpu| -> HipResult<(DeviceBuffer, DeviceBuffer)> {
            let pre_table = gpu.hip.malloc(pre_bytes)?;
            match gpu.hip.malloc(gdn_bytes) {
                Ok(gdn_table) => Ok((pre_table, gdn_table)),
                Err(e) => {
                    let _ = gpu.hip.free(pre_table);
                    Err(e)
                }
            }
        };
        let (pre0, gdn0) = alloc_pair(gpu)?;
        let (pre1, gdn1) = match alloc_pair(gpu) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.hip.free(pre0);
                let _ = gpu.hip.free(gdn0);
                return Err(e);
            }
        };
        Ok(Self {
            pre_tables: [pre0, pre1],
            gdn_tables: [gdn0, gdn1],
            slots: TwoSlot::new(),
        })
    }

    fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.pre_tables.into_iter().chain(self.gdn_tables) {
            let _ = gpu.hip.free(t);
        }
    }
}

impl GdnReplayMl {
    /// JIT both kernels, acquire the shared replay scratch and allocate the
    /// tables, plus the D8 snapshot-source tables when `with_from`. Any
    /// failure releases what was acquired and returns the error.
    fn arm(
        gpu: &mut Gpu,
        n_la: usize,
        max_n: usize,
        v_dim: usize,
        with_from: bool,
    ) -> HipResult<Self> {
        gpu.ensure_dflash_gdn_replay_ml()?;
        let shape = rdna_compute::dflash_gdn_replay::GdnReplayScratchShape {
            n_layers: n_la,
            rows: max_n.min(rdna_compute::dflash_gdn_replay::DFLASH_GDN_REPLAY_MAX_STEPS),
            v_dim,
        };
        let base = gpu.dflash_gdn_replay_scratch_acquire(shape)?;
        let pre_bytes =
            n_la * std::mem::size_of::<rdna_compute::dflash_gdn_replay::DflashReplayPreLayer>();
        let gdn_bytes =
            n_la * std::mem::size_of::<rdna_compute::dflash_gdn_replay::GdnLayerTable>();
        let tables = gpu
            .hip
            .malloc(pre_bytes)
            .and_then(|p| match gpu.hip.malloc(gdn_bytes) {
                Ok(g) => Ok((p, g)),
                Err(e) => {
                    let _ = gpu.hip.free(p);
                    Err(e)
                }
            });
        let (pre_table, gdn_table) = match tables {
            Ok(t) => t,
            Err(e) => {
                gpu.dflash_gdn_replay_scratch_release(shape, base);
                return Err(e);
            }
        };
        let mut ml = Self {
            shape,
            base: base as u64,
            pre_table,
            gdn_table,
            fingerprint: None,
            from: None,
        };
        if with_from {
            match GdnReplayMlFrom::arm(gpu, n_la) {
                Ok(from) => ml.from = Some(from),
                Err(e) => {
                    ml.free_gpu(gpu);
                    return Err(e);
                }
            }
        }
        Ok(ml)
    }

    /// Check and stream-order the shared scratch before a replay.
    fn order(&self, gpu: &mut Gpu) -> HipResult<()> {
        gpu.dflash_gdn_replay_scratch_order(self.shape, self.base as *mut _)
    }

    fn free_gpu(self, gpu: &mut Gpu) {
        gpu.dflash_gdn_replay_scratch_release(self.shape, self.base as *mut _);
        let _ = gpu.hip.free(self.pre_table);
        let _ = gpu.hip.free(self.gdn_table);
        if let Some(from) = self.from {
            from.free_gpu(gpu);
        }
    }
}

impl DeltaNetTape {
    pub fn new_for(gpu: &mut Gpu, state: &DeltaNetState, n_slots: usize) -> HipResult<Self> {
        let mut slots = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            slots.push(DeltaNetSnapshot::new_for(gpu, state)?);
        }
        Ok(Self { slots })
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn save_at(&mut self, slot: usize, state: &DeltaNetState, gpu: &mut Gpu) -> HipResult<()> {
        self.slots[slot].save_from(state, gpu)
    }

    pub fn restore_from(
        &self,
        slot: usize,
        state: &mut DeltaNetState,
        gpu: &mut Gpu,
    ) -> HipResult<()> {
        self.slots[slot].restore_to(state, gpu)
    }
}

/// Compute the DFlash target-layer extraction indices for a model of
/// `num_target_layers` layers. Matches the `build_target_layer_ids` function in
/// the DFlash reference implementation:
///
/// ```text
/// start = 1
/// end   = num_target_layers - 3        # 29 for num_target_layers=32
/// step  = (end - start) / (num_extract - 1)
/// layers[i] = round(start + i * step)  # for i in 0..num_extract
/// ```
///
/// For Qwen3.5-9B (32 layers) and 5 extraction layers this returns
/// `[1, 8, 15, 22, 29]`, matching the hard-coded indices in the HuggingFace
/// `z-lab/Qwen3.5-9B-DFlash` config.
pub fn dflash_extract_layer_ids(num_target_layers: usize, num_extract: usize) -> Vec<usize> {
    if num_extract == 0 {
        return Vec::new();
    }
    if num_extract == 1 {
        return vec![1];
    }
    let start: f32 = 1.0;
    let end: f32 = (num_target_layers as i32 - 3).max(1) as f32;
    let step = (end - start) / (num_extract as f32 - 1.0);
    (0..num_extract)
        .map(|i| (start + i as f32 * step).round() as usize)
        .collect()
}

/// Ring buffer holding the most recent `max_positions` of hidden state
/// extractions from the target model's forward pass. Each of the `extract_layers`
/// entries is a `[max_positions, hidden_dim]` f32 GPU tensor. `head` is the
/// position that the NEXT write will land at (0..max_positions). `written` is
/// the total cumulative number of writes, used to tell full vs partial buffer.
///
/// For DFlash, the draft model pulls a contiguous slice ending at the most
/// recent position to use as context KV input.
/// Persistent scratch for `spec_step_ddtree_batched` — eliminates the
/// per-cycle alloc/free churn that dominated early-benchmark wall-clock
/// time. Allocated once at session start, sized for the maximum tree we
/// may see.
///
/// Contents:
/// - `attn_bias`: `[max_n × max_n]` f32 additive bias buffer. `max_n =
///   1 + tree_budget`. Per cycle the caller uploads `big_n × big_n`
///   floats into its head — unused tail space is irrelevant because the
///   FA kernel reads at `global_bid * block_cols + col` with `block_cols
///   = big_n`.
///
/// Callers pass this by `&mut` to `spec_step_ddtree_batched`. It's OK
/// to over-allocate (max_n larger than any cycle's actual tree) — the
/// per-cycle cost is only the htod of the current cycle's mask bytes.
pub struct DdtreeScratch {
    pub max_n: usize,
    pub attn_bias: GpuTensor,
    /// Per-slot parent index consumed by tree-aware LA kernels when
    /// `HIPFIRE_DDTREE_TREE_LA=1`. `[max_n]` i32, uploaded fresh each
    /// cycle via `memcpy_htod` before calling `verify_dflash_block_tree`.
    /// Allocated as Raw bytes (4 × max_n) since there's no i32 DType.
    pub parent_indices: GpuTensor,
}

impl DdtreeScratch {
    /// Allocate for a worst-case tree of `max_budget` non-root nodes.
    pub fn new(gpu: &mut Gpu, max_budget: usize) -> HipResult<Self> {
        let max_n = 1 + max_budget;
        let attn_bias = gpu.alloc_tensor(&[max_n * max_n], rdna_compute::DType::F32)?;
        let parent_indices = gpu.alloc_tensor(&[max_n * 4], rdna_compute::DType::Raw)?;

        Ok(Self {
            max_n,
            attn_bias,
            parent_indices,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.attn_bias);
        let _ = gpu.free_tensor(self.parent_indices);
    }
}

/// Persistent per-decode-cycle scratch for the target verify pass and the
/// draft lm_head. Prior to 2026-04-16 these were allocated fresh every cycle
/// inside `verify_dflash_block_inner` and `spec_step_dflash` — ~8 hipMalloc
/// + hipFree pairs per cycle (biggest are 16 MB logits buffers). The HIP
/// allocator is 50–200 µs per call, so per-cycle overhead was 0.5–1.5 ms
/// just in allocator churn. Preallocating once at session start removes
/// the churn with no correctness impact.
///
/// `max_n` must be ≥ max of (verify block size, tree-verify node count).
/// The demo sizes it to `max(block_size, 1 + tree_budget)` to cover both
/// the vanilla DFlash and DDTree paths.
pub struct VerifyScratch {
    pub max_n: usize,
    pub dim: usize,
    pub vocab: usize,
    pub hidden_k: usize,
    /// Post-output-norm hidden from the target forward, [max_n × dim] F32.
    /// Drives the per-position lm_head GEMM.
    pub final_hidden: GpuTensor,
    /// Scratch logits from target + draft lm_head, [max_n × vocab] F32.
    /// Reused across target verify (n=B) and draft lm_head (n=B-1).
    pub logits: GpuTensor,
    /// FWHT-rotated hidden for MQ4 lm_head path, [max_n × hidden_k] F32.
    /// Allocated unconditionally; unused on non-MQ4 targets.
    pub rot: GpuTensor,
    /// Argmax output for greedy path, [max_n] f32 (treated as i32 host-side).
    pub argmax: GpuTensor,
    /// Persistent per-layer batch scratch for `qwen35::forward_prefill_batch`.
    /// Sized to `max_n`, so `verify_dflash_block` processes each block in a
    /// single chunk without the ~25 hipMalloc/hipFree pairs the in-function
    /// allocation would incur. Present whenever the caller passes a config
    /// to `VerifyScratch::with_prefill`. Absent (None) for the legacy
    /// constructor — `forward_prefill_batch` then falls back to allocating
    /// its own scratch.
    pub prefill_batch: Option<qwen35::PrefillBatchScratch>,
    /// Reusable buffers of the singleton device sampled accept
    /// (`HIPFIRE_DFLASH_DEVICE_ACCEPT`), grown on first sampled window. A
    /// `Mutex` because the singleton step borrows `VerifyScratch` shared.
    pub device_accept: std::sync::Mutex<Option<DflashAcceptScratch>>,
    /// Reusable verbatim-lookup tail buffer of the singleton DFlash step
    /// (`spec_step_dflash`'s `lookup` seam), reserved at its 15-candidate
    /// maximum so a spliced window allocates nothing. A `Mutex` because the
    /// singleton step borrows `VerifyScratch` shared; only locked when a
    /// lookup is supplied.
    pub lookup_tail: std::sync::Mutex<Vec<u32>>,
}

impl VerifyScratch {
    pub fn new(
        gpu: &mut Gpu,
        max_n: usize,
        dim: usize,
        vocab: usize,
        hidden_k: usize,
    ) -> HipResult<Self> {
        Ok(Self {
            max_n,
            dim,
            vocab,
            hidden_k,
            final_hidden: gpu.alloc_tensor(&[max_n * dim], rdna_compute::DType::F32)?,
            logits: gpu.alloc_tensor(&[max_n * vocab], rdna_compute::DType::F32)?,
            rot: gpu.alloc_tensor(&[max_n * hidden_k], rdna_compute::DType::F32)?,
            argmax: gpu.alloc_tensor(&[max_n], rdna_compute::DType::F32)?,
            prefill_batch: None,
            device_accept: std::sync::Mutex::new(None),
            lookup_tail: std::sync::Mutex::new(Vec::with_capacity(DFLASH_LOOKUP_MAX_CANDIDATES)),
        })
    }

    /// Like `new`, but also allocates a persistent `PrefillBatchScratch`
    /// sized to `max_n`. Use this for DFlash verify where the same block
    /// scratch is reused every cycle — drops ~25 tensor alloc/free pairs
    /// per cycle (measured ~3-5 ms/cycle on 27B Qwen3.5 where the per-call
    /// allocation dominated verify wall-time).
    pub fn with_prefill(
        gpu: &mut Gpu,
        max_n: usize,
        dim: usize,
        vocab: usize,
        hidden_k: usize,
        config: &qwen35::Qwen35Config,
    ) -> HipResult<Self> {
        let mut s = Self::new(gpu, max_n, dim, vocab, hidden_k)?;
        s.prefill_batch = Some(qwen35::PrefillBatchScratch::new(gpu, config, max_n)?);
        Ok(s)
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.final_hidden);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.free_tensor(self.rot);
        let _ = gpu.free_tensor(self.argmax);
        if let Some(pbs) = self.prefill_batch {
            pbs.free_gpu(gpu);
        }
        if let Some(acc) = self
            .device_accept
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
        {
            acc.free(gpu);
        }
    }
}

/// Host-state change of committing staged rows to the hidden ring (design A
/// `WindowHostDelta` ring transitions): head `from -> to`, written
/// `from -> to`. Computed by [`HiddenStateRingBuffer::commit_delta`], published
/// once by [`HiddenStateRingBuffer::apply_commit_delta`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HiddenCommitDelta {
    pub head_from: usize,
    pub head_to: usize,
    pub written_from: usize,
    pub written_to: usize,
}

pub struct HiddenStateRingBuffer {
    pub layer_bufs: Vec<GpuTensor>,
    pub extract_layers: Vec<usize>,
    pub max_positions: usize,
    pub hidden_dim: usize,
    pub head: usize,
    pub written: usize,
    /// Per-extract-layer staging buffer, shape `[max_batch × hidden_dim]`.
    /// Captured kernels (verify forward) write here with FIXED offsets so
    /// their captured pointers don't bake in a per-cycle `head`. After the
    /// graph returns, `commit_staging_to_ring` scatters staging → `layer_bufs`
    /// at the current head (outside the captured region, head-aware).
    pub staging_bufs: Vec<GpuTensor>,
    /// Max rows a single staging write can hold — sized to the maximum batch
    /// the caller ever passes to `write_rows_to_staging`. For DFlash verify
    /// this is `budget + 1`.
    pub max_batch: usize,
}

impl HiddenStateRingBuffer {
    /// Physical ring slot holding block-local row `r` of the most recent
    /// `block_size`-row block — the same mapping
    /// [`scatter_hidden_block_to_interleaved`] walks, exposed so callers that
    /// need to address block rows directly (DDTree post-accept gather) do not
    /// re-derive the ring arithmetic.
    pub fn block_row_slot(&self, block_size: usize, r: usize) -> usize {
        let r_skip = block_size.saturating_sub(self.max_positions);
        let start_slot =
            (self.head + self.max_positions - (block_size - r_skip)) % self.max_positions;
        (start_slot + r.saturating_sub(r_skip)) % self.max_positions
    }

    /// Allocate GPU ring buffer for explicit extraction layer list.
    ///
    /// `extract_layers` must be in ascending order; validation of range is the
    /// caller's responsibility ( `load_dflash_state` checks against
    /// `num_target_layers` ).
    pub fn new_for_layers(
        gpu: &mut Gpu,
        extract_layers: &[usize],
        hidden_dim: usize,
        max_positions: usize,
        max_batch: usize,
    ) -> HipResult<Self> {
        let num_extract = extract_layers.len();
        let mut layer_bufs = Vec::with_capacity(num_extract);
        let mut staging_bufs = Vec::with_capacity(num_extract);
        for _ in 0..num_extract {
            layer_bufs
                .push(gpu.alloc_tensor(&[max_positions * hidden_dim], rdna_compute::DType::F32)?);
            staging_bufs
                .push(gpu.alloc_tensor(&[max_batch * hidden_dim], rdna_compute::DType::F32)?);
        }
        Ok(Self {
            layer_bufs,
            extract_layers: extract_layers.to_vec(),
            max_positions,
            hidden_dim,
            head: 0,
            written: 0,
            staging_bufs,
            max_batch,
        })
    }

    /// Allocate GPU ring buffer for `num_extract` target layers.
    ///
    /// `max_batch` sizes the staging buffers used by the graph-capture path.
    /// Typical value for DFlash verify is `budget + 1`.
    pub fn new(
        gpu: &mut Gpu,
        num_target_layers: usize,
        num_extract: usize,
        hidden_dim: usize,
        max_positions: usize,
        max_batch: usize,
    ) -> HipResult<Self> {
        let extract_layers = dflash_extract_layer_ids(num_target_layers, num_extract);
        Self::new_for_layers(gpu, &extract_layers, hidden_dim, max_positions, max_batch)
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.layer_bufs {
            let _ = gpu.free_tensor(t);
        }
        for t in self.staging_bufs {
            let _ = gpu.free_tensor(t);
        }
    }

    /// If `target_layer_idx` matches one of the extraction layers, return the
    /// index into `layer_bufs`/`extract_layers` for that layer. Otherwise None.
    #[inline]
    pub fn extract_slot(&self, target_layer_idx: usize) -> Option<usize> {
        self.extract_layers
            .iter()
            .position(|&l| l == target_layer_idx)
    }

    /// Copy `x` (shape `[hidden_dim]`) into the ring buffer slot for the given
    /// extraction layer at the CURRENT head position. Call once per extracted
    /// layer per forward pass, then `advance_head()` at the end of the forward
    /// to move to the next slot.
    pub fn write_at_head(&self, gpu: &mut Gpu, extract_idx: usize, x: &GpuTensor) -> HipResult<()> {
        let offset = self.head * self.hidden_dim * 4;
        gpu.hip.memcpy_dtod_at(
            &self.layer_bufs[extract_idx].buf,
            offset,
            &x.buf,
            0,
            self.hidden_dim * 4,
        )
    }

    /// Advance the write head. Call once per forward pass, AFTER all layer
    /// extractions for this position have been written.
    #[inline]
    pub fn advance_head(&mut self) {
        self.head = (self.head + 1) % self.max_positions;
        self.written += 1;
    }

    /// Advance the write head by `n`. Used by the batched prefill path after
    /// writing N rows per extract layer in a single dispatch.
    #[inline]
    pub fn advance_head_by(&mut self, n: usize) {
        self.head = (self.head + n) % self.max_positions;
        self.written += n;
    }

    /// Copy `n` contiguous rows from `src` (shape `[n × hidden_dim]` row-major)
    /// into the ring buffer slot for the given extraction layer, starting at
    /// the CURRENT head position. Handles the ring-buffer wrap: if head + n
    /// exceeds max_positions, the write splits into a head→end + 0→tail pair.
    /// Call this once per extracted layer per batched forward, then advance
    /// the head by `n` via `advance_head_by(n)` at the end.
    pub fn write_rows_at_head(
        &self,
        gpu: &mut Gpu,
        extract_idx: usize,
        src: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        let row_bytes = self.hidden_dim * 4;
        let head = self.head;
        let max_pos = self.max_positions;
        if head + n <= max_pos {
            gpu.hip.memcpy_dtod_at(
                &self.layer_bufs[extract_idx].buf,
                head * row_bytes,
                &src.buf,
                0,
                n * row_bytes,
            )?;
        } else {
            let first = max_pos - head;
            gpu.hip.memcpy_dtod_at(
                &self.layer_bufs[extract_idx].buf,
                head * row_bytes,
                &src.buf,
                0,
                first * row_bytes,
            )?;
            gpu.hip.memcpy_dtod_at(
                &self.layer_bufs[extract_idx].buf,
                0,
                &src.buf,
                first * row_bytes,
                (n - first) * row_bytes,
            )?;
        }
        Ok(())
    }

    /// Write `n` contiguous rows from `src` into the staging buffer for the
    /// given extraction layer at FIXED offset 0. Safe to call inside a
    /// hipGraph stream capture: the captured memcpy node bakes in the
    /// staging pointer (which is stable across cycles), not a per-cycle head.
    ///
    /// Callers must call `commit_staging_to_ring(n)` after the forward
    /// returns (outside the captured region) to scatter staging → `layer_bufs`
    /// at the current head, then advance the head.
    pub fn write_rows_to_staging(
        &self,
        gpu: &mut Gpu,
        extract_idx: usize,
        src: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        // Hard assert (not debug_assert): a violation silently overran the
        // staging buffer in release and surfaced as a cryptic d2d-bounds panic
        // deep in ffi.rs. Fail loud and named instead.
        assert!(
            n <= self.max_batch,
            "write_rows_to_staging: n {} > staging max_batch {} — staging buffer \
             too small for this prefill chunk",
            n,
            self.max_batch
        );
        let row_bytes = self.hidden_dim * 4;
        let bytes = n * row_bytes;
        // The software Redline recorder retains typed kernel launches, not HIP
        // memcpy commands. During recording, express this byte-identical F32
        // copy as a blob-backed kernel so recorded-HIP and PM4 refresh staging
        // on every replay instead of committing stale rows to the hidden ring.
        // HipGraph capture keeps the native async memcpy node below.
        if gpu.replay.is_recording() {
            return gpu.copy_f32_buffer(&self.staging_bufs[extract_idx], src, n * self.hidden_dim);
        }
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.memcpy_dtod_async_at(
                &self.staging_bufs[extract_idx].buf,
                0,
                &src.buf,
                0,
                bytes,
                stream,
            )
        } else {
            gpu.hip
                .memcpy_dtod_at(&self.staging_bufs[extract_idx].buf, 0, &src.buf, 0, bytes)
        }
    }

    /// Scatter staging buffers into `layer_bufs` at the current head, handling
    /// ring wrap, then advance the head by `n`. Must be called AFTER the
    /// forward (outside any captured region) — uses the current `head` to
    /// compute destination offsets, which would be baked wrong in a replayed
    /// graph.
    ///
    /// When `gpu.active_stream` is Some, we first sync the stream (so the
    /// captured forward's staging writes are complete) then use sync D2D
    /// for the scatter. This matches the existing sync-memcpy semantics the
    /// rest of the engine relies on for ordering with null-stream consumers
    /// (e.g. the draft forward's D2H of hidden rows after this commit).
    pub fn commit_staging_to_ring(&mut self, gpu: &mut Gpu, n: usize) -> HipResult<()> {
        // If running under an explicit stream (graph capture path), wait
        // for the captured writes to complete before the scatter so we
        // don't read uninitialized staging.
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(stream)?;
        }
        self.emit_commit_staging(gpu, n)?;
        // Head/written advance only after successful enqueue, preserving the
        // existing stream synchronization boundary above.
        let delta = self.commit_delta(n);
        self.apply_commit_delta(delta);
        Ok(())
    }

    /// The host-state change committing `n` staged rows makes: head
    /// `(head + n) % max_positions`, written `+ n`. Pure (design A
    /// `WindowHostDelta` ring transitions); nothing is published until
    /// [`Self::apply_commit_delta`].
    pub fn commit_delta(&self, n: usize) -> HiddenCommitDelta {
        HiddenCommitDelta {
            head_from: self.head,
            head_to: (self.head + n) % self.max_positions,
            written_from: self.written,
            written_to: self.written + n,
        }
    }

    /// Publish a [`HiddenCommitDelta`] computed against the current head.
    pub fn apply_commit_delta(&mut self, delta: HiddenCommitDelta) {
        debug_assert_eq!(
            delta.head_from, self.head,
            "stale hidden commit delta (head)"
        );
        debug_assert_eq!(
            delta.written_from, self.written,
            "stale hidden commit delta (written)"
        );
        self.head = delta.head_to;
        self.written = delta.written_to;
    }

    /// GPU half of [`Self::commit_staging_to_ring`]: enqueue the staging ->
    /// ring scatter at the current head. No stream synchronization and no
    /// host-state mutation (head/written stay put; apply
    /// [`Self::commit_delta`] separately). Returns `true` when the scatter is
    /// the fused `commit5` kernel, whose `head` kernarg (byte 80) is the one
    /// per-window scalar; `false` when the per-extract copy loop ran (its
    /// copy offsets bake the head).
    pub fn emit_commit_staging(&self, gpu: &mut Gpu, n: usize) -> HipResult<bool> {
        assert!(
            n <= self.max_batch,
            "commit_staging_to_ring: n {} > staging max_batch {} — a chunk wider than \
             staging wrote straight to the ring; finish it with finish_prefill_chunk",
            n,
            self.max_batch
        );
        let row_bytes = self.hidden_dim * 4;
        let head = self.head;
        let max_pos = self.max_positions;

        // S2 launch fusion: pure-copy commit5 kernel (exact gfx1100/gfx1151/gfx1201). Copies
        // staging[ext][r, :] -> layer_bufs[ext][(head + r) % max_pos, :] for
        // all five extracts in one launch (bit-identical: one writer per
        // destination element, no FP arithmetic on the data). The launch
        // also ensures the scatter5 symbol, which the same-cycle scatter
        // reuses without its own `&mut` ensure. Any failed predicate
        // (non-admitted arch, kill switch, capture/recording, non-5-extract or
        // non-F32 shapes, n > max_pos) falls through to today's loop.
        if gpu.dflash_hidden_commit5_applicable(
            &self.staging_bufs,
            &self.layer_bufs,
            n,
            self.hidden_dim,
            max_pos,
        ) {
            gpu.dflash_hidden_commit5_launch(
                &self.staging_bufs,
                &self.layer_bufs,
                head,
                n,
                self.hidden_dim,
                max_pos,
            )?;
            return Ok(true);
        }

        for ei in 0..self.layer_bufs.len() {
            if head + n <= max_pos {
                gpu.hip.memcpy_dtod_at(
                    &self.layer_bufs[ei].buf,
                    head * row_bytes,
                    &self.staging_bufs[ei].buf,
                    0,
                    n * row_bytes,
                )?;
            } else {
                let first = max_pos - head;
                gpu.hip.memcpy_dtod_at(
                    &self.layer_bufs[ei].buf,
                    head * row_bytes,
                    &self.staging_bufs[ei].buf,
                    0,
                    first * row_bytes,
                )?;
                gpu.hip.memcpy_dtod_at(
                    &self.layer_bufs[ei].buf,
                    0,
                    &self.staging_bufs[ei].buf,
                    first * row_bytes,
                    (n - first) * row_bytes,
                )?;
            }
        }
        Ok(false)
    }

    /// Whether a ring-carrying prompt prefill may take the ordinary (AR)
    /// prefill route: widened chunks, the GDN chunk scan, and chunks wider than
    /// staging, whose rows [`write_chunk_rows`](Self::write_chunk_rows) writes
    /// straight to the ring at `head`. Eager only: under graph capture or
    /// Redline recording the rows must go through fixed-offset staging (a baked
    /// `head` is wrong on replay), so chunks stay capped at `max_batch`.
    /// `HIPFIRE_DFLASH_LEGACY_PREFILL=1` keeps the staged route everywhere.
    pub fn rides_ordinary_prefill(&self, gpu: &Gpu) -> bool {
        !gpu.graphs.capture_mode && !gpu.replay.is_recording() && !dflash_legacy_prefill()
    }

    /// Whether an `n`-row prefill chunk writes its rows straight to the ring
    /// instead of staging. Only chunks wider than staging do, so every chunk
    /// that fits (verify blocks, short prompts) keeps the staged commit.
    pub fn writes_direct(&self, gpu: &Gpu, n: usize) -> bool {
        n > self.max_batch && self.rides_ordinary_prefill(gpu)
    }

    /// Per-extract-layer write of an `n`-row prefill chunk: staging (see
    /// [`write_rows_to_staging`](Self::write_rows_to_staging)), or for a
    /// [`writes_direct`](Self::writes_direct) chunk the ring slots
    /// `head..head + n`, wrapping. Pair with
    /// [`finish_prefill_chunk`](Self::finish_prefill_chunk) after the chunk.
    pub fn write_chunk_rows(
        &self,
        gpu: &mut Gpu,
        extract_idx: usize,
        src: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        if !self.writes_direct(gpu, n) {
            return self.write_rows_to_staging(gpu, extract_idx, src, n);
        }
        assert!(
            n <= self.max_positions,
            "write_chunk_rows: n {} > ring max_positions {}",
            n,
            self.max_positions
        );
        let first = n.min(self.max_positions - self.head);
        self.copy_rows_into_ring(gpu, extract_idx, self.head, src, 0, first)?;
        if first < n {
            self.copy_rows_into_ring(gpu, extract_idx, 0, src, first, n - first)?;
        }
        Ok(())
    }

    /// Place an `n`-row prefill chunk written by
    /// [`write_chunk_rows`](Self::write_chunk_rows) and advance `head` by `n`:
    /// commit staging, or for a direct chunk wait for its copies when a stream
    /// is active (null-stream readers such as `download_hidden_block` follow).
    pub fn finish_prefill_chunk(&mut self, gpu: &mut Gpu, n: usize) -> HipResult<()> {
        if !self.writes_direct(gpu, n) {
            return self.commit_staging_to_ring(gpu, n);
        }
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(stream)?;
        }
        self.advance_head_by(n);
        Ok(())
    }

    /// Copy `rows` rows of `src` from row `src_row` to ring slot `dst_slot` of
    /// one extract layer, on the active stream when one is set so the copy is
    /// ordered after the forward kernels that produced `src`.
    fn copy_rows_into_ring(
        &self,
        gpu: &Gpu,
        extract_idx: usize,
        dst_slot: usize,
        src: &GpuTensor,
        src_row: usize,
        rows: usize,
    ) -> HipResult<()> {
        let row_bytes = self.hidden_dim * 4;
        let dst = &self.layer_bufs[extract_idx].buf;
        match gpu.active_stream.as_ref() {
            Some(stream) => gpu.hip.memcpy_dtod_async_at(
                dst,
                dst_slot * row_bytes,
                &src.buf,
                src_row * row_bytes,
                rows * row_bytes,
                stream,
            ),
            None => gpu.hip.memcpy_dtod_at(
                dst,
                dst_slot * row_bytes,
                &src.buf,
                src_row * row_bytes,
                rows * row_bytes,
            ),
        }
    }

    /// Reset to empty (head=0, written=0). GPU buffers are not zeroed; stale
    /// data is simply unreadable because `written < max_positions`.
    pub fn reset(&mut self) {
        self.head = 0;
        self.written = 0;
    }
}

/// Single-pass argmax for token sampling. Not SIMD-optimized — the logit
/// vector is downloaded once per verify step so the CPU scan cost is
/// negligible relative to GEMV work.
#[inline]
fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Temperature-scaled softmax. Writes into `out` (reused across calls to
/// avoid per-position allocation in the rejection-sampling hot loop).
#[inline]
pub(crate) fn softmax_temp_into(logits: &[f32], temp: f32, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(logits.len());
    let inv_t = 1.0 / temp;
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        let s = v * inv_t;
        if s > max {
            max = s;
        }
    }
    let mut sum = 0.0f32;
    for &v in logits {
        let e = (v * inv_t - max).exp();
        out.push(e);
        sum += e;
    }
    let inv_sum = 1.0 / sum;
    for p in out.iter_mut() {
        *p *= inv_sum;
    }
}

/// Apply a nucleus (top_p) truncation to a normalized probability row IN PLACE,
/// given the per-row threshold `tau_cut` and kept mass `Z` produced by the GPU
/// `softmax_temp_topp_batched_f32` kernel:
///
///   p'(x) = if p(x) >= tau_cut { p(x) / Z } else { 0 }
///
/// After this the row sums to 1 (it was divided by the kept mass Z), so
/// `sample_categorical` / `sample_residual` operate on a proper distribution.
///
/// When `tau_cut == 0.0` (the kernel's top_p>=1.0 guard) this is the IDENTITY
/// (every p >= 0 is kept, Z == 1) → byte-equivalent to no truncation.
///
/// This reproduces the AR nucleus cut in `mtp_spec.rs` (sort desc, accumulate,
/// cut at the FIRST prefix whose cumulative >= top_p INCLUSIVE, renorm by kept
/// mass) in threshold form. Exact float ties at the boundary are kept together
/// here (`>=`), whereas AR cuts by sort position — a measure-zero divergence for
/// real logits.
#[inline]
pub(crate) fn apply_topp_trunc(row: &mut [f32], tau_cut: f32, z: f32) {
    if tau_cut <= 0.0 {
        // top_p disabled (or single-token row already at full mass): identity.
        return;
    }
    let inv_z = 1.0f32 / z.max(f32::MIN_POSITIVE);
    for p in row.iter_mut() {
        if *p >= tau_cut {
            *p *= inv_z;
        } else {
            *p = 0.0;
        }
    }
}

/// Host-side nucleus (top_p) truncation of an ALREADY-normalized softmax row,
/// IN PLACE, matching the AR rule in `mtp_spec.rs:307-325`: sort descending,
/// accumulate, cut at the FIRST token whose cumulative prob >= top_p (inclusive
/// crossing token), renormalize the kept set by its mass. Used on the non-fast
/// DFlash temp arms (which compute the softmax on host, not GPU) so top_p is
/// honored on the WHOLE temp path, not only under HIPFIRE_DFLASH_FAST_SAMPLE.
///
/// `top_p >= 1.0` is a no-op (identity). Produces the same kept-set and the same
/// renormalized values as `apply_topp_trunc` would given the GPU `tau_cut/Z`,
/// modulo the boundary-tie convention (this version keeps AR's sort-position
/// tie-break exactly; `apply_topp_trunc` keeps all boundary-equal tokens).
pub(crate) fn apply_host_nucleus(row: &mut [f32], top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    // Indices sorted by probability descending (stable for tie reproducibility).
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0f64;
    let mut kept_mass = 0.0f64;
    let mut cutoff = order.len();
    for (rank, &idx) in order.iter().enumerate() {
        cum += row[idx] as f64;
        kept_mass += row[idx] as f64;
        if cum >= top_p as f64 {
            cutoff = rank + 1;
            break;
        }
    }
    let inv = if kept_mass > 0.0 {
        1.0f64 / kept_mass
    } else {
        0.0
    };
    // Zero out the dropped tail, renorm the kept head.
    for (rank, &idx) in order.iter().enumerate() {
        if rank < cutoff {
            row[idx] = (row[idx] as f64 * inv) as f32;
        } else {
            row[idx] = 0.0;
        }
    }
}

/// Host-side top-k truncation of an ALREADY-normalized softmax row, IN PLACE.
/// Keeps the `k` highest-probability tokens, zeros the rest, and renormalizes
/// the kept mass to 1. `k == 0` or `k >= row.len()` is a no-op (identity).
#[inline]
pub(crate) fn apply_host_topk(row: &mut [f32], k: usize) {
    if k == 0 || k >= row.len() {
        return;
    }
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept_mass = 0.0f64;
    for &idx in order.iter().take(k) {
        kept_mass += row[idx] as f64;
    }
    if kept_mass <= 0.0 {
        return;
    }
    let inv = 1.0f64 / kept_mass;
    for (rank, &idx) in order.iter().enumerate() {
        if rank < k {
            row[idx] = (row[idx] as f64 * inv) as f32;
        } else {
            row[idx] = 0.0;
        }
    }
}

/// Draw a categorical sample from `probs` given uniform u ∈ [0, 1).
#[inline]
pub(crate) fn sample_categorical(probs: &[f32], u: f32) -> u32 {
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// The AR sampler's truncation (`llama::sample_top_k_p`, min_p off) of a
/// normalized softmax row, IN PLACE: keep the `cap` most probable tokens
/// (`cap` from `llama::ar_candidate_cap`; like the kernel, 0 or > 64 → 64),
/// then the nucleus over their own mass, renormalized. The host twin of
/// `softmax_temp_topp_batched_f32`'s `tau`/`Z`.
pub(crate) fn apply_host_ar_trunc(row: &mut [f32], cap: usize, top_p: f32) {
    const MAX_CAP: usize = 64;
    let cap = if cap == 0 { MAX_CAP } else { cap.min(MAX_CAP) };
    apply_host_topk(row, cap);
    apply_host_nucleus(row, top_p);
}

/// Request controls used by the singleton sampled DFlash chain (formerly
/// `spec_step_dflash` lines 5905–5921 and prefill lines 890–913).
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct DflashSampleCfg {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub cactus_delta: f32,
}

/// Sparse selector q rows of one sampled window: row `i` is the ascending
/// (id, value) support of the dense vocab row the dense path would build
/// (zeros outside the support, last write wins on duplicate candidate ids).
/// Flat storage, reused across windows: no per-row or vocab-length allocation.
#[doc(hidden)]
#[derive(Default)]
pub struct SparseQRows {
    off: Vec<u32>,
    pub idx: Vec<u32>,
    pub val: Vec<f32>,
    stage: Vec<(u32, f32)>,
}

impl SparseQRows {
    pub fn with_capacity(rows: usize, per_row: usize) -> Self {
        let mut off = Vec::with_capacity(rows + 1);
        off.push(0);
        Self {
            off,
            idx: Vec::with_capacity(rows * per_row),
            val: Vec::with_capacity(rows * per_row),
            stage: Vec::with_capacity(per_row),
        }
    }

    pub fn clear(&mut self) {
        self.off.clear();
        self.off.push(0);
        self.idx.clear();
        self.val.clear();
    }

    /// Keep the first `rows` rows (no-op when fewer exist).
    pub fn truncate_rows(&mut self, rows: usize) {
        if rows < self.rows() {
            let end = self.off[rows] as usize;
            self.off.truncate(rows + 1);
            self.idx.truncate(end);
            self.val.truncate(end);
        }
    }

    pub fn rows(&self) -> usize {
        self.off.len().saturating_sub(1)
    }

    /// Row offsets into `idx` / `val` (`rows() + 1` entries, `[0] == 0`).
    pub fn offsets(&self) -> &[u32] {
        &self.off
    }

    /// Row `i`'s strictly ascending ids and their values.
    pub fn row(&self, i: usize) -> (&[u32], &[f32]) {
        let (a, b) = (self.off[i] as usize, self.off[i + 1] as usize);
        (&self.idx[a..b], &self.val[a..b])
    }

    /// Append the row of candidate `cands[j]` ↦ `probs[j]`. The dense row is
    /// `row[cands[j]] = probs[j]` for `j = 0..` over a zero row, so for a
    /// repeated id the LAST candidate's value survives. A stable sort by id
    /// keeps candidate order inside each id run; keeping the run's last
    /// element is exactly that last write. A candidate with value 0.0 stays
    /// an explicit entry (the dense row holds 0.0 there too).
    pub fn push_row(&mut self, cands: &[u32], probs: &[f32], vocab: usize) -> HipResult<()> {
        debug_assert_eq!(cands.len(), probs.len());
        if self.off.is_empty() {
            self.off.push(0);
        }
        self.stage.clear();
        for (&c, &p) in cands.iter().zip(probs) {
            if c as usize >= vocab {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!("selector candidate id {} out of vocab {}", c, vocab),
                ));
            }
            self.stage.push((c, p));
        }
        self.stage.sort_by_key(|e| e.0);
        let n = self.stage.len();
        let mut k = 0;
        while k < n {
            let mut e = k;
            while e + 1 < n && self.stage[e + 1].0 == self.stage[k].0 {
                e += 1;
            }
            self.idx.push(self.stage[e].0);
            self.val.push(self.stage[e].1);
            k = e + 1;
        }
        self.off.push(self.idx.len() as u32);
        Ok(())
    }
}

/// Reusable selector receipt from singleton lines 6136–6346 / 6460–6500.
/// `block` includes the seed; q row i proposes `block[i + 1]`. Extra dense
/// rows may be retained across short windows but are not part of this receipt.
///
/// q rows live in `sparse_q` when `sparse` (the FAST_SAMPLE default,
/// [`dflash_sparse_target_enabled`]) and in `dense_q` otherwise.
#[doc(hidden)]
#[derive(Default)]
pub struct DflashSampledDraft {
    pub block: Vec<u32>,
    pub selected_q: Vec<f32>,
    pub dense_q: Vec<Vec<f32>>,
    pub sparse_q: SparseQRows,
    pub sparse: bool,
    pub proposal: Option<dflash::DflashCandidateProposal>,
    pub uniforms: Vec<f32>,
}

impl DflashSampledDraft {
    /// Reserve a request's maximum window without changing the selector law.
    /// Sparse mode reserves only the top-K supports; dense rows (cactus or
    /// `HIPFIRE_DFLASH_SPARSE_TARGET=0`) are allocated on first use.
    pub fn new(b: usize, vocab: usize) -> Self {
        let rows = b.saturating_sub(1);
        let dense_q = if *DFLASH_SPARSE_TARGET_ON {
            Vec::new()
        } else {
            (0..rows).map(|_| Vec::with_capacity(vocab)).collect()
        };
        Self {
            block: Vec::with_capacity(b),
            selected_q: Vec::with_capacity(rows),
            dense_q,
            sparse_q: SparseQRows::with_capacity(rows, 64),
            sparse: false,
            proposal: None,
            uniforms: Vec::with_capacity(rows),
        }
    }
}

/// Host acceptance result from singleton lines 6832–6907. The committed
/// sequence includes the already-emitted seed and the new pending bonus.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct SampledWindowDecision {
    pub accepted_drafts: usize,
    pub bonus_token: u32,
    pub committed_len: usize,
}

/// Per-row capacity of the FAST_SAMPLE sparse target compaction (CAP256). A
/// row whose kept set (`p >= tau`) is larger falls back to its dense row.
pub const DFLASH_SPARSE_CAP: usize = 256;
/// `dense_slot` marker: the row is held sparse.
const SPARSE_NONE: u32 = u32::MAX;

/// `HIPFIRE_DFLASH_SPARSE_TARGET=0` (developer kill switch): download the
/// full `[rows * vocab]` target probabilities and keep selector q dense.
/// Default on. Read through [`dflash_sparse_target_enabled`].
static DFLASH_SPARSE_TARGET_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_DFLASH_SPARSE_TARGET").map_or(true, |v| v.trim() != "0")
});

/// The one switch of the sparse sampled-DFlash representation, shared by the
/// singleton route and the continuous-batching lanes (target rows AND selector
/// q): the kill switch is on and CACTUS is off. CACTUS rewrites the whole dense
/// target row (`scale * v` everywhere) and so stays on the dense arithmetic.
#[doc(hidden)]
pub fn dflash_sparse_target_enabled(cfg: &DflashSampleCfg) -> bool {
    *DFLASH_SPARSE_TARGET_ON && !(cfg.cactus_delta > 0.0)
}

/// One verify row of [`FastTargetRows`].
#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum TargetRowRef<'a> {
    /// The RAW temperature softmax row plus the kernel's `tau`/`z`; the
    /// nucleus truncation [`apply_topp_trunc`] is still to be applied.
    Dense { raw: &'a [f32], tau: f32, z: f32 },
    /// The kept set `p >= tau` (`tau > 0`): ascending ids and RAW probs. The
    /// truncated value of id `idx[k]` is `val[k] * inv_z`, every other id 0.
    Sparse {
        idx: &'a [u32],
        val: &'a [f32],
        inv_z: f32,
    },
}

/// FAST_SAMPLE target rows of a sampled DFlash window, downloaded from the
/// GPU. Every row is the temperature softmax with the per-row AR cutoff `tau`
/// and kept mass `z` from `softmax_temp_topp_batched_into_f32`.
///
/// Dense mode (`capacity == 0`): `probs` is row-major `[rows * vocab]`.
/// Sparse mode (`capacity > 0`): per row `counts[r]` kept entries in
/// `idx/val[r * capacity ..]` (ascending ids, raw probs, from
/// `sampled_rows_compact_into_f32`); rows that cannot be sparse — `tau <= 0`
/// (identity truncation, kept set = everything) or `counts > capacity`
/// (overflow) — are held dense and raw in `dense_rows` at `dense_slot[r]`.
#[doc(hidden)]
#[derive(Default)]
pub struct FastTargetRows {
    pub probs: Vec<f32>,
    pub tau: Vec<f32>,
    pub z: Vec<f32>,
    pub capacity: usize,
    pub counts: Vec<u32>,
    pub idx: Vec<u32>,
    pub val: Vec<f32>,
    pub dense_slot: Vec<u32>,
    pub dense_rows: Vec<f32>,
}

impl FastTargetRows {
    /// Dense-mode rows.
    pub fn dense(probs: Vec<f32>, tau: Vec<f32>, z: Vec<f32>) -> Self {
        Self {
            probs,
            tau,
            z,
            ..Self::default()
        }
    }

    pub fn rows(&self) -> usize {
        self.tau.len()
    }

    pub fn is_sparse(&self) -> bool {
        self.capacity > 0
    }

    /// Reserve a lane's maximum window in the mode `cfg` selects.
    pub fn reserve_rows(&mut self, rows: usize, vocab: usize, cfg: &DflashSampleCfg) {
        self.tau.reserve(rows);
        self.z.reserve(rows);
        if dflash_sparse_target_enabled(cfg) {
            self.counts.reserve(rows);
            self.dense_slot.reserve(rows);
            self.idx.reserve(rows * DFLASH_SPARSE_CAP);
            self.val.reserve(rows * DFLASH_SPARSE_CAP);
        } else {
            self.probs.reserve(rows * vocab);
        }
    }

    pub fn row(&self, row: usize, vocab: usize) -> TargetRowRef<'_> {
        if self.capacity == 0 {
            return TargetRowRef::Dense {
                raw: &self.probs[row * vocab..(row + 1) * vocab],
                tau: self.tau[row],
                z: self.z[row],
            };
        }
        let slot = self.dense_slot[row];
        if slot != SPARSE_NONE {
            let s = slot as usize;
            return TargetRowRef::Dense {
                raw: &self.dense_rows[s * vocab..(s + 1) * vocab],
                tau: self.tau[row],
                z: self.z[row],
            };
        }
        let n = self.counts[row] as usize;
        let o = row * self.capacity;
        TargetRowRef::Sparse {
            idx: &self.idx[o..o + n],
            val: &self.val[o..o + n],
            // The exact scale `apply_topp_trunc` multiplies by.
            inv_z: 1.0f32 / self.z[row].max(f32::MIN_POSITIVE),
        }
    }

    /// Sparse mode: classify rows after `tau`, `z`, `counts`, `idx`, `val`
    /// are downloaded. Fills `dense_slot` (slots in ascending row order),
    /// returns the number of dense rows, and validates every sparse row's ids
    /// (ascending, `< vocab`) — the sequential-sum exactness of the sparse
    /// arithmetic depends on that order.
    pub fn plan_dense_rows(&mut self, vocab: usize) -> HipResult<usize> {
        let rows = self.tau.len();
        let cap = self.capacity;
        if cap == 0
            || self.z.len() != rows
            || self.counts.len() != rows
            || self.idx.len() != rows * cap
            || self.val.len() != rows * cap
        {
            return Err(HipError::new(
                0,
                "sparse target rows: inconsistent compact arrays",
            ));
        }
        self.dense_slot.clear();
        let mut n_dense = 0usize;
        for r in 0..rows {
            let c = self.counts[r] as usize;
            // `!(tau > 0)` also routes a NaN tau to the dense row, where
            // `apply_topp_trunc` keeps its exact (all-compare-false) behavior.
            if !(self.tau[r] > 0.0) || c > cap {
                self.dense_slot.push(n_dense as u32);
                n_dense += 1;
                continue;
            }
            let ids = &self.idx[r * cap..r * cap + c];
            let ok =
                ids.iter().all(|&i| (i as usize) < vocab) && ids.windows(2).all(|w| w[0] < w[1]);
            if !ok {
                return Err(HipError::new(
                    0,
                    "sparse target rows: compact ids not ascending/in range",
                ));
            }
            self.dense_slot.push(SPARSE_NONE);
        }
        Ok(n_dense)
    }
}

fn f32_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: any byte pattern is a valid f32; the slice covers exactly `v`.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
}

fn u32_bytes_mut(v: &mut [u32]) -> &mut [u8] {
    // SAFETY: any byte pattern is a valid u32; the slice covers exactly `v`.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
}

fn download_tensor_bytes(gpu: &Gpu, t: &GpuTensor, dst: &mut [u8]) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.hip.memcpy_dtoh(dst, &t.buf)
}

/// Device buffers of one target-rows pass over `rows` verify rows:
/// `[probs (rows*vocab), tau (rows), z (rows)]`, plus
/// `[counts (rows), idx (rows*cap), val (rows*cap)]` in sparse mode (six
/// tensors). Each may be larger than the pass needs (a persistent scratch).
fn alloc_target_bufs(
    gpu: &mut Gpu,
    rows: usize,
    vocab: usize,
    sparse: bool,
) -> HipResult<Vec<GpuTensor>> {
    let cap = DFLASH_SPARSE_CAP;
    let mut shapes: Vec<usize> = vec![rows * vocab, rows, rows];
    if sparse {
        shapes.extend_from_slice(&[rows, rows * cap, rows * cap]);
    }
    let mut bufs: Vec<GpuTensor> = Vec::with_capacity(shapes.len());
    for n in shapes {
        match gpu.alloc_tensor(&[n], rdna_compute::DType::F32) {
            Ok(t) => bufs.push(t),
            Err(e) => {
                free_target_bufs(gpu, &mut bufs);
                return Err(e);
            }
        }
    }
    Ok(bufs)
}

fn free_target_bufs(gpu: &mut Gpu, bufs: &mut Vec<GpuTensor>) {
    for t in bufs.drain(..) {
        let _ = gpu.free_tensor(t);
    }
}

/// Persistent, grow-on-demand device buffers plus the run-level host rows of
/// the batched sampled target path (`dflash_cb_verify`). Grown only when a
/// run needs more rows than any before (or the vocab changes); a steady-state
/// step allocates and frees no device memory.
#[doc(hidden)]
#[derive(Default)]
pub struct TargetRowsScratch {
    rows_cap: usize,
    vocab: usize,
    bufs: Vec<GpuTensor>,
    host: FastTargetRows,
}

impl TargetRowsScratch {
    /// Make the sparse-mode buffers hold at least `rows` rows of `vocab`.
    fn ensure_sparse(&mut self, gpu: &mut Gpu, rows: usize, vocab: usize) -> HipResult<()> {
        if self.bufs.len() == 6 && self.vocab == vocab && self.rows_cap >= rows {
            return Ok(());
        }
        free_target_bufs(gpu, &mut self.bufs);
        self.rows_cap = 0;
        self.bufs = alloc_target_bufs(gpu, rows, vocab, true)?;
        self.rows_cap = rows;
        self.vocab = vocab;
        Ok(())
    }

    pub fn free_gpu(mut self, gpu: &mut Gpu) {
        free_target_bufs(gpu, &mut self.bufs);
    }

    /// Sparse target rows of `run_rows` consecutive verify rows of `logits`
    /// (`[run_rows * vocab]`): ONE softmax, ONE compaction and one download of
    /// tau/z/counts/idx/val (dense rows only for identity/overflow rows) into
    /// the run-level host rows. Each lane's rows are then taken with
    /// [`Self::split_lane_into`], exactly as the per-lane
    /// [`dflash_target_rows_into`] would have left them. Every row is computed
    /// by its own block with row-local state, so each lane's rows are
    /// bit-identical to a per-lane call. Requires
    /// `dflash_sparse_target_enabled(&cfg)`.
    pub fn run_sparse(
        &mut self,
        gpu: &mut Gpu,
        logits: &GpuTensor,
        run_rows: usize,
        vocab: usize,
        cfg: DflashSampleCfg,
    ) -> HipResult<()> {
        if !dflash_sparse_target_enabled(&cfg) || run_rows == 0 {
            return Err(HipError::new(0, "sparse target run: not a sparse run"));
        }
        self.ensure_sparse(gpu, run_rows, vocab)?;
        dflash_target_rows_core(
            gpu,
            logits,
            &self.bufs,
            run_rows,
            vocab,
            cfg,
            &mut self.host,
        )
    }

    /// Lane-owned copy of rows `r0..r0 + b` of the last [`Self::run_sparse`].
    pub fn split_lane_into(
        &self,
        r0: usize,
        b: usize,
        vocab: usize,
        out: &mut FastTargetRows,
    ) -> HipResult<()> {
        self.host.split_lane_into(r0, b, vocab, out)
    }
}

impl FastTargetRows {
    /// Sparse mode: the `b` rows `r0..r0 + b` of this run as a lane-owned
    /// `FastTargetRows`, identical to what the per-lane download of those
    /// rows produces (including its validation and dense-row slots).
    fn split_lane_into(
        &self,
        r0: usize,
        b: usize,
        vocab: usize,
        out: &mut FastTargetRows,
    ) -> HipResult<()> {
        let cap = self.capacity;
        if cap == 0 || r0 + b > self.tau.len() || self.dense_slot.len() != self.tau.len() {
            return Err(HipError::new(0, "sparse target run: inconsistent run rows"));
        }
        out.capacity = cap;
        out.probs.clear();
        out.tau.clear();
        out.tau.extend_from_slice(&self.tau[r0..r0 + b]);
        out.z.clear();
        out.z.extend_from_slice(&self.z[r0..r0 + b]);
        out.counts.clear();
        out.counts.extend_from_slice(&self.counts[r0..r0 + b]);
        out.idx.clear();
        out.idx
            .extend_from_slice(&self.idx[r0 * cap..(r0 + b) * cap]);
        out.val.clear();
        out.val
            .extend_from_slice(&self.val[r0 * cap..(r0 + b) * cap]);
        let n_dense = out.plan_dense_rows(vocab)?;
        out.dense_rows.clear();
        if n_dense == 0 {
            return Ok(());
        }
        out.dense_rows.resize(n_dense * vocab, 0.0);
        for i in 0..b {
            let slot = out.dense_slot[i];
            if slot == SPARSE_NONE {
                continue;
            }
            let src = self.dense_slot[r0 + i];
            if src == SPARSE_NONE {
                return Err(HipError::new(
                    0,
                    "sparse target run: dense row classification mismatch",
                ));
            }
            let (s, src) = (slot as usize, src as usize);
            out.dense_rows[s * vocab..(s + 1) * vocab]
                .copy_from_slice(&self.dense_rows[src * vocab..(src + 1) * vocab]);
        }
        Ok(())
    }
}

/// FAST_SAMPLE materialization lifted from singleton lines 6802–6826:
/// normalize all verify rows, then download tau, Z and the probabilities.
/// `logits` may be a lane-owned view but must contain exactly its rows.
///
/// With [`dflash_sparse_target_enabled`], the probabilities are compacted on
/// the GPU (`sampled_rows_compact_into_f32`, per-row `tau`, CAP256) and only the
/// small count/id/value arrays plus the identity (`tau <= 0`) and overflow
/// rows' dense rows are downloaded; otherwise the whole `[b * vocab]` buffer.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn dflash_target_rows_into(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    b: usize,
    vocab: usize,
    cfg: DflashSampleCfg,
    out: &mut FastTargetRows,
) -> HipResult<()> {
    let mut bufs = alloc_target_bufs(gpu, b, vocab, dflash_sparse_target_enabled(&cfg))?;
    let res = dflash_target_rows_core(gpu, logits, &bufs, b, vocab, cfg, out);
    free_target_bufs(gpu, &mut bufs);
    res
}

/// Rows-generic core of the target rows: `bufs` is
/// [`alloc_target_bufs`]' layout (six tensors = sparse mode), each at least
/// `b` rows. Writes `out` for the `b` rows of `logits`.
#[allow(clippy::too_many_arguments)]
fn dflash_target_rows_core(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    bufs: &[GpuTensor],
    b: usize,
    vocab: usize,
    cfg: DflashSampleCfg,
    out: &mut FastTargetRows,
) -> HipResult<()> {
    let sparse = bufs.len() == 6;
    let (probs_gpu, tau_gpu, z_gpu) = (&bufs[0], &bufs[1], &bufs[2]);
    gpu.softmax_temp_topp_batched_into_f32(
        logits, probs_gpu, tau_gpu, z_gpu, vocab, b, cfg.temp, cfg.top_p, cfg.top_k, 0.0,
    )?;
    // Exact-row views: a persistent scratch may be larger than this pass.
    gpu.download_f32_into(&tau_gpu.sub_offset(0, b), &mut out.tau)?;
    gpu.download_f32_into(&z_gpu.sub_offset(0, b), &mut out.z)?;
    if !sparse {
        out.capacity = 0;
        out.counts.clear();
        out.idx.clear();
        out.val.clear();
        out.dense_slot.clear();
        out.dense_rows.clear();
        return gpu.download_f32_into(&probs_gpu.sub_offset(0, b * vocab), &mut out.probs);
    }
    let cap = DFLASH_SPARSE_CAP;
    let (counts_gpu, idx_gpu, val_gpu) = (&bufs[3], &bufs[4], &bufs[5]);
    gpu.sampled_rows_compact_into_f32(
        probs_gpu, tau_gpu, counts_gpu, idx_gpu, val_gpu, vocab, b, cap,
    )?;
    out.capacity = cap;
    out.counts.clear();
    out.counts.resize(b, 0);
    out.idx.clear();
    out.idx.resize(b * cap, 0);
    out.val.clear();
    out.val.resize(b * cap, 0.0);
    // counts/idx are i32 buffers behind F32-typed tensors: raw byte copies
    // (the copy length is the host slice, a prefix of the device buffer).
    download_tensor_bytes(gpu, counts_gpu, u32_bytes_mut(&mut out.counts))?;
    download_tensor_bytes(gpu, idx_gpu, u32_bytes_mut(&mut out.idx))?;
    download_tensor_bytes(gpu, val_gpu, f32_bytes_mut(&mut out.val))?;
    out.probs.clear();
    let n_dense = out.plan_dense_rows(vocab)?;
    out.dense_rows.clear();
    if n_dense == 0 {
        return Ok(());
    }
    out.dense_rows.resize(n_dense * vocab, 0.0);
    if n_dense == b {
        return download_tensor_bytes(gpu, probs_gpu, f32_bytes_mut(&mut out.dense_rows));
    }
    for r in 0..b {
        let slot = out.dense_slot[r];
        if slot == SPARSE_NONE {
            continue;
        }
        let s = slot as usize;
        let view = probs_gpu.sub_offset(r * vocab, vocab);
        download_tensor_bytes(
            gpu,
            &view,
            f32_bytes_mut(&mut out.dense_rows[s * vocab..(s + 1) * vocab]),
        )?;
    }
    Ok(())
}

/// `HIPFIRE_DFLASH_DEVICE_ACCEPT=0` (developer kill switch): decide the
/// singleton sampled window on the host again (the FAST_SAMPLE target rows come
/// back as sparse/dense host rows and the host accept loop runs). Default on.
static DFLASH_DEVICE_ACCEPT_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_DFLASH_DEVICE_ACCEPT").map_or(true, |v| v.trim() != "0")
});

/// The kill switch of the device sampled accept ([`dflash_sampled_accept_device`]).
/// The kernel is the certified builder module (gfx1201 only, no hipcc production
/// fallback), so callers also require `Gpu::dflash_device_accept_supported()`.
#[doc(hidden)]
pub fn dflash_device_accept_enabled() -> bool {
    *DFLASH_DEVICE_ACCEPT_ON
}

/// Whether a sampled window is decided by [`dflash_sampled_accept_device`]:
/// the switch is on, CACTUS is off (it rewrites the whole dense target row and
/// keeps the host arithmetic), the selector q is the sparse representation
/// (`HIPFIRE_DFLASH_SPARSE_TARGET=0` keeps dense q and the host loop) and the
/// receipt covers the `block.len() - 1` proposed rows. Repeat / n-gram
/// penalties never reach the sampled arm (they are greedy-only host paths).
#[doc(hidden)]
pub fn dflash_device_accept_eligible(
    enabled: bool,
    cfg: &DflashSampleCfg,
    q: &DflashSampledDraft,
    block_len: usize,
) -> bool {
    enabled
        && !(cfg.cactus_delta > 0.0)
        && q.sparse
        && block_len >= 2
        && q.selected_q.len() == block_len - 1
        && q.sparse_q.rows() == block_len - 1
}

/// Singleton sampled window decided entirely on the device. `logits` is the
/// resident verify logits (at least `block.len() * vocab` elements, one row per
/// verify position). The temperature softmax / nucleus normalization is
/// enqueued, then ONE kernel (`dflash_sampled_accept_sparse_q_f32`) takes the
/// accept tests, the residual and the bonus over those device rows: no
/// full-vocab D2H, no compaction, no dense q. The host uploads the proposed
/// tokens, the sparse selector q and `block.len()` uniforms pre-drawn from
/// `rng` (one per proposed row plus ONE for the residual-or-bonus draw: a
/// fixed count per window, whatever the outcome) and reads back
/// `{accept_len, bonus}`. `max_accept` bounds the accepted prefix exactly as
/// the host loop does (the bonus is then drawn from the boundary row).
/// `scratch` is the session's reusable buffers (grown on demand): a
/// steady-state window allocates nothing.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn dflash_sampled_accept_device(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    block: &[u32],
    q: &DflashSampledDraft,
    vocab: usize,
    cfg: DflashSampleCfg,
    max_accept: usize,
    rng: &mut u64,
    scratch: &mut Option<DflashAcceptScratch>,
) -> HipResult<SampledWindowDecision> {
    let b = block.len();
    if !dflash_device_accept_eligible(true, &cfg, q, b) {
        return Err(HipError::new(
            0,
            "device sampled accept: ineligible window (cactus / dense q / receipt shape)",
        ));
    }
    let n = b - 1;
    let nnz = q.sparse_q.offsets()[n] as usize;
    let payload_words = rdna_compute::dflash_sampled_accept::dflash_accept_payload_words(n, nnz);
    DflashAcceptScratch::ensure(scratch, gpu, b, vocab, payload_words)?;
    let s = scratch.as_mut().expect("scratch ensured above");
    s.host_uniforms.clear();
    s.host_uniforms
        .extend((0..n).map(|_| xorshift_next_unit(rng)));
    let u_fin = xorshift_next_unit(rng);
    rdna_compute::dflash_sampled_accept::dflash_accept_pack_payload(
        &mut s.host_payload,
        &block[1..],
        &s.host_uniforms,
        u_fin,
        max_accept,
        &q.selected_q,
        q.sparse_q.offsets(),
        &q.sparse_q.idx,
        &q.sparse_q.val,
    )
    .map_err(|e| HipError::new(0, &format!("device sampled accept: {e}")))?;

    let logits_batch = logits.sub_offset(0, b * vocab);
    // The scratch tensors may be larger than this window: kernels size by the
    // explicit (rows, vocab) arguments, never by the tensor.
    gpu.softmax_temp_topp_batched_into_f32(
        &logits_batch,
        &s.probs,
        &s.tau,
        &s.z,
        vocab,
        b,
        cfg.temp,
        cfg.top_p,
        cfg.top_k,
        0.0,
    )?;
    let d = gpu.dflash_sampled_accept_sparse_q_f32(
        s,
        &s.probs,
        &s.tau,
        &s.z,
        &s.host_payload,
        n,
        vocab,
    )?;
    Ok(SampledWindowDecision {
        accepted_drafts: d.accept_len,
        bonus_token: d.bonus,
        committed_len: d.accept_len + 2,
    })
}

/// Selector q rows as the accept loop reads them.
#[derive(Clone, Copy)]
enum QRows<'a> {
    Dense(&'a [Vec<f32>]),
    Sparse(&'a SparseQRows),
}

/// Selector host rejection/bonus loop from singleton lines 6832–6907.
/// Proposal uniforms have already been drawn; budget checks precede each
/// acceptance draw, rejection draws one residual, and bonus draws only when
/// there was no rejection. Dense q retains vocab-order residual arithmetic.
///
/// # Sparse exactness
/// When the target row is sparse ([`TargetRowRef::Sparse`]), the q row is
/// sparse and CACTUS is off, no vocab row is materialized. Every quantity of
/// the dense loop is reproduced bit-for-bit, because the dense loop only
/// ever adds, compares and returns at positions where its summand is
/// non-zero:
///  * target value `p(t)`: kept ids hold `val * inv_z` — the very product
///    `apply_topp_trunc` computes — every other id is the literal `0.0`;
///  * residual `d_i = p_i - q_i` is `0.0 - 0.0 = 0.0` (never `> 0`, never
///    added) off the union of the two supports, so both sums walk the union
///    in ascending id, i.e. the dense order, with identical f32 operands;
///  * the final "last index" fallback returns `vocab - 1` like the dense one;
///  * the degenerate fallback (`sum <= 0`) is the dense `argmax_u32` scan
///    replayed over the implicit zeros ([`argmax_sparse`]).
/// Mixed rows (dense target with sparse q, sparse target with dense q or
/// CACTUS) rebuild the dense target row in `scratch` and use the dense or
/// dense-p/sparse-q residual, which share the same operand sequence.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn dflash_sampled_accept_rows(
    block: &[u32],
    q: &DflashSampledDraft,
    fast: Option<&FastTargetRows>,
    logits: &[f32],
    vocab: usize,
    cfg: DflashSampleCfg,
    max_accept: usize,
    rng: &mut u64,
    scratch: &mut Vec<f32>,
) -> SampledWindowDecision {
    let q_rows = if q.sparse {
        QRows::Sparse(&q.sparse_q)
    } else {
        QRows::Dense(&q.dense_q)
    };
    dflash_sampled_accept_q_rows(
        block,
        &q.selected_q,
        q_rows,
        fast,
        logits,
        vocab,
        cfg,
        max_accept,
        rng,
        scratch,
    )
}

#[allow(clippy::too_many_arguments)]
fn dflash_sampled_accept_q_rows(
    block: &[u32],
    selected_q: &[f32],
    q: QRows<'_>,
    fast: Option<&FastTargetRows>,
    logits: &[f32],
    vocab: usize,
    cfg: DflashSampleCfg,
    max_accept: usize,
    rng: &mut u64,
    scratch: &mut Vec<f32>,
) -> SampledWindowDecision {
    dflash_sampled_accept_q_rows_with(
        block,
        selected_q,
        q,
        fast,
        logits,
        vocab,
        cfg,
        max_accept,
        &mut || xorshift_next_unit(rng),
        scratch,
    )
}

/// [`dflash_sampled_accept_q_rows`] over an explicit uniform source: the body
/// is the single accept/residual/bonus implementation, `next_u` replaces the
/// xorshift draw (one call per accept test, per residual draw and per bonus
/// draw, in the same order). The law tests drive it with exact uniform grids.
#[allow(clippy::too_many_arguments)]
#[inline]
fn dflash_sampled_accept_q_rows_with(
    block: &[u32],
    selected_q: &[f32],
    q: QRows<'_>,
    fast: Option<&FastTargetRows>,
    logits: &[f32],
    vocab: usize,
    cfg: DflashSampleCfg,
    max_accept: usize,
    next_u: &mut impl FnMut() -> f32,
    scratch: &mut Vec<f32>,
) -> SampledWindowDecision {
    let b = block.len();
    let mut accept_len = 0;
    let mut rejected_bonus = None;
    let use_cactus = cfg.cactus_delta > 0.0;
    for i in 0..b - 1 {
        if accept_len >= max_accept {
            break;
        }
        let t = block[i + 1] as usize;
        let p_d = selected_q[i].max(f32::MIN_POSITIVE);
        if !use_cactus {
            if let (
                Some(TargetRowRef::Sparse {
                    idx: tidx,
                    val: tval,
                    inv_z,
                }),
                QRows::Sparse(qs),
            ) = (fast.map(|f| f.row(i, vocab)), q)
            {
                let p_t = sparse_row_prob(tidx, tval, inv_z, block[i + 1]);
                let u = next_u();
                if u * p_d <= p_t {
                    accept_len += 1;
                } else {
                    let u2 = next_u();
                    let (qidx, qval) = qs.row(i);
                    rejected_bonus = Some(sample_residual_sparse(
                        tidx, tval, inv_z, qidx, qval, vocab, u2,
                    ));
                    break;
                }
                continue;
            }
        }
        sampled_target_row_into(
            scratch, i, vocab, fast, logits, cfg.temp, cfg.top_k, cfg.top_p,
        );
        let p_t = scratch[t];
        let accept_prob = if use_cactus {
            let bump = (2.0 * cfg.cactus_delta * p_t * (1.0 - p_t)).max(0.0).sqrt();
            (p_t + bump).min(1.0)
        } else {
            p_t
        };
        let u = next_u();
        if u * p_d <= accept_prob {
            accept_len += 1;
        } else {
            if use_cactus {
                let qn = p_t.clamp(0.0, 1.0);
                let gamma_star = accept_prob;
                if qn >= 1.0 - 1e-6 {
                    for v in scratch.iter_mut() {
                        *v = 0.0;
                    }
                    scratch[t] = 1.0;
                } else {
                    let scale = (1.0 - gamma_star) / (1.0 - qn);
                    for (j, v) in scratch.iter_mut().enumerate() {
                        *v = if j == t { gamma_star } else { scale * *v };
                    }
                }
            }
            let u2 = next_u();
            rejected_bonus = Some(match q {
                QRows::Dense(d) => sample_residual(scratch, &d[i], u2),
                QRows::Sparse(s) => {
                    let (qidx, qval) = s.row(i);
                    sample_residual_dense_p_sparse_q(scratch, qidx, qval, u2)
                }
            });
            break;
        }
    }
    let bonus_token = if let Some(b) = rejected_bonus {
        b
    } else {
        let i = accept_len.min(b - 1);
        let u = next_u();
        sampled_target_bonus(
            scratch, i, vocab, fast, logits, cfg.temp, cfg.top_k, cfg.top_p, u,
        )
    };
    SampledWindowDecision {
        accepted_drafts: accept_len,
        bonus_token,
        committed_len: accept_len + 2,
    }
}

/// Target distribution of verify row `row` in a sampled DFlash window, written
/// into `out`: AR's law (`llama::sample_top_k_p`) — temperature softmax, the
/// `top_k` most probable tokens (`top_k` is the AR candidate cap,
/// `llama::ar_candidate_cap`), then the nucleus over their mass — from the GPU
/// rows (`fast`, truncated by their `tau`/`z`) or, without FAST_SAMPLE, from
/// the host `logits` (`[rows * vocab]`). Every sampled target read of the host
/// accept loop goes through here — the accept test, the rejection residual
/// and the full-accept / budget bonus — so they share one law.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn sampled_target_row_into(
    out: &mut Vec<f32>,
    row: usize,
    vocab: usize,
    fast: Option<&FastTargetRows>,
    logits: &[f32],
    temp: f32,
    top_k: usize,
    top_p: f32,
) {
    match fast {
        Some(fast) => match fast.row(row, vocab) {
            TargetRowRef::Dense { raw, tau, z } => {
                out.clear();
                out.extend_from_slice(raw);
                apply_topp_trunc(out, tau, z);
            }
            // Legacy dense consumers (CACTUS, dense q, tests): rebuild the
            // truncated row — `val * inv_z` on the kept ids, 0.0 elsewhere.
            TargetRowRef::Sparse { idx, val, inv_z } => {
                out.clear();
                out.resize(vocab, 0.0);
                for (&i, &v) in idx.iter().zip(val) {
                    out[i as usize] = v * inv_z;
                }
            }
        },
        None => {
            softmax_temp_into(&logits[row * vocab..(row + 1) * vocab], temp, out);
            apply_host_ar_trunc(out, top_k, top_p);
        }
    }
}

/// Full-accept / budget-bound bonus of a sampled DFlash window: one draw with
/// uniform `u` from [`sampled_target_row_into`]'s row `row`. `scratch` is
/// reused as the row buffer.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn sampled_target_bonus(
    scratch: &mut Vec<f32>,
    row: usize,
    vocab: usize,
    fast: Option<&FastTargetRows>,
    logits: &[f32],
    temp: f32,
    top_k: usize,
    top_p: f32,
    u: f32,
) -> u32 {
    // A sparse FAST_SAMPLE row is drawn directly from its kept set (no vocab row).
    if let Some(TargetRowRef::Sparse { idx, val, inv_z }) = fast.map(|f| f.row(row, vocab)) {
        return sample_categorical_sparse(idx, val, inv_z, vocab, u);
    }
    sampled_target_row_into(scratch, row, vocab, fast, logits, temp, top_k, top_p);
    sample_categorical(scratch, u)
}

/// Draw from (p_target − p_draft)₊, renormalized. Used on rejection to
/// sample the "corrective" bonus token in speculative rejection sampling
/// (Chen & Leviathan 2023, algorithm 1).
#[inline]
pub(crate) fn sample_residual(p_target: &[f32], p_draft: &[f32], u: f32) -> u32 {
    let mut sum = 0.0f32;
    for i in 0..p_target.len() {
        let d = p_target[i] - p_draft[i];
        if d > 0.0 {
            sum += d;
        }
    }
    if sum <= 0.0 {
        // Degenerate case (p_draft >= p_target everywhere). Should not
        // happen in practice if a rejection was just drawn. Fall back to
        // argmax of p_target.
        return argmax_u32(p_target);
    }
    let u_scaled = u * sum;
    let mut acc = 0.0f32;
    for i in 0..p_target.len() {
        let d = p_target[i] - p_draft[i];
        if d > 0.0 {
            acc += d;
            if u_scaled < acc {
                return i as u32;
            }
        }
    }
    (p_target.len() - 1) as u32
}

/// [`sample_residual`] against a deterministic (point-mass) draft
/// `q = δ(excluded)`: draw from `p_target` with `excluded` removed,
/// renormalized, WITHOUT materializing a one-hot vocab row. Bit-identical to
/// `sample_residual(p_target, onehot(excluded), u)`: the excluded slot
/// contributes `p_target[excluded] - 1.0` (≤ 0 for a valid distribution, so
/// skipped), every other slot contributes `p_target[i]` itself, with the same
/// summation order and the same degenerate (`argmax`) / last-index fallbacks.
#[inline]
pub(crate) fn sample_residual_point_mass(p_target: &[f32], excluded: u32, u: f32) -> u32 {
    let ex = excluded as usize;
    let mut sum = 0.0f32;
    for i in 0..p_target.len() {
        let d = if i == ex {
            p_target[i] - 1.0
        } else {
            p_target[i]
        };
        if d > 0.0 {
            sum += d;
        }
    }
    if sum <= 0.0 {
        return argmax_u32(p_target);
    }
    let u_scaled = u * sum;
    let mut acc = 0.0f32;
    for i in 0..p_target.len() {
        let d = if i == ex {
            p_target[i] - 1.0
        } else {
            p_target[i]
        };
        if d > 0.0 {
            acc += d;
            if u_scaled < acc {
                return i as u32;
            }
        }
    }
    (p_target.len() - 1) as u32
}

/// Truncated target probability of token `t` in a sparse row: `val * inv_z`
/// when kept, the literal `0.0` of the dense row otherwise. `idx` ascending.
#[inline]
fn sparse_row_prob(idx: &[u32], val: &[f32], inv_z: f32, t: u32) -> f32 {
    match idx.binary_search(&t) {
        Ok(k) => val[k] * inv_z,
        Err(_) => 0.0,
    }
}

/// [`sample_categorical`] over a sparse truncated row without materializing it.
///
/// The dense scan adds `p_i` in id order and returns the first `i` with
/// `u < acc`. Off the support `p_i = 0.0`, so `acc` is unchanged (`x + 0.0 == x`
/// exactly) and a test that failed at the previous id fails again — the first
/// success is therefore always a support id, found by walking the support in
/// the same order with the same partial sums. The single exception is the
/// very first id (`acc = 0.0 + p_0`): for `u < 0` it succeeds at id 0 even when
/// `p_0 = 0`, so that case is returned explicitly (`u` from the xorshift
/// stream is always `>= 0`; NaN compares false in both). Falling off the end
/// returns the dense `len - 1` = `vocab - 1`.
#[inline]
fn sample_categorical_sparse(idx: &[u32], val: &[f32], inv_z: f32, vocab: usize, u: f32) -> u32 {
    if u < 0.0 {
        return 0;
    }
    let mut acc = 0.0f32;
    for (&i, &v) in idx.iter().zip(val) {
        acc += v * inv_z;
        if u < acc {
            return i;
        }
    }
    (vocab - 1) as u32
}

/// [`argmax_u32`] of a sparse truncated row: the scan replayed over the
/// virtual dense row. A run of implicit zeros between support ids is a run of
/// equal values, of which only the first can win the strict `>` — so one
/// candidate `(first id of the gap, 0.0)` per gap, the support entries in
/// between, and a trailing gap, visited in ascending id.
fn argmax_sparse(idx: &[u32], val: &[f32], inv_z: f32, vocab: usize) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut pos = 0usize;
    for (&i, &v) in idx.iter().zip(val) {
        let i = i as usize;
        if i > pos && 0.0 > best_v {
            best_v = 0.0;
            best = pos;
        }
        let x = v * inv_z;
        if x > best_v {
            best_v = x;
            best = i;
        }
        pos = i + 1;
    }
    if pos < vocab && 0.0 > best_v {
        best = pos;
    }
    best as u32
}

/// Visit the ascending union of a sparse target row (`val * inv_z`, 0.0 off
/// support) and a sparse q row (0.0 off support) as `(p, q)`; stops and
/// returns the id when `f` returns true.
#[inline]
fn walk_union(
    tidx: &[u32],
    tval: &[f32],
    inv_z: f32,
    qidx: &[u32],
    qval: &[f32],
    mut f: impl FnMut(f32, f32) -> bool,
) -> Option<u32> {
    let (mut a, mut b) = (0usize, 0usize);
    while a < tidx.len() || b < qidx.len() {
        let ti = tidx.get(a).copied().unwrap_or(u32::MAX);
        let qi = qidx.get(b).copied().unwrap_or(u32::MAX);
        let i = ti.min(qi);
        let p = if ti == i {
            let v = tval[a] * inv_z;
            a += 1;
            v
        } else {
            0.0
        };
        let q = if qi == i {
            let v = qval[b];
            b += 1;
            v
        } else {
            0.0
        };
        if f(p, q) {
            return Some(i);
        }
    }
    None
}

/// [`sample_residual`] of a sparse target row against a sparse q row, with no
/// vocab row. The dense function skips every `d = p - q <= 0` (including
/// `0.0 - 0.0` off both supports), adds the positive `d` in ascending id into
/// `sum`, rescales `u`, and returns the first id whose running `acc` exceeds
/// `u * sum`; `d` is positive only on the union of the supports, so walking
/// the union ascending performs the identical f32 operations in the identical
/// order (q needs no sign assumption: an off-target negative q entry gives
/// `0.0 - q > 0` and is on the union). `sum <= 0` takes the dense degenerate
/// `argmax` ([`argmax_sparse`]); running off the end returns `vocab - 1`.
fn sample_residual_sparse(
    tidx: &[u32],
    tval: &[f32],
    inv_z: f32,
    qidx: &[u32],
    qval: &[f32],
    vocab: usize,
    u: f32,
) -> u32 {
    let mut sum = 0.0f32;
    walk_union(tidx, tval, inv_z, qidx, qval, |p, q| {
        let d = p - q;
        if d > 0.0 {
            sum += d;
        }
        false
    });
    if sum <= 0.0 {
        return argmax_sparse(tidx, tval, inv_z, vocab);
    }
    let u_scaled = u * sum;
    let mut acc = 0.0f32;
    let hit = walk_union(tidx, tval, inv_z, qidx, qval, |p, q| {
        let d = p - q;
        if d > 0.0 {
            acc += d;
            if u_scaled < acc {
                return true;
            }
        }
        false
    });
    match hit {
        Some(i) => i,
        None => (vocab - 1) as u32,
    }
}

/// [`sample_residual`] against a sparse q row for a dense target row: the
/// dense q is `0.0` off its support, so `p_i - q_i` is computed with the same
/// operands at every id, without a q vocab row.
fn sample_residual_dense_p_sparse_q(p_target: &[f32], qidx: &[u32], qval: &[f32], u: f32) -> u32 {
    let q_at = |i: usize, k: &mut usize| -> f32 {
        if *k < qidx.len() && qidx[*k] as usize == i {
            let v = qval[*k];
            *k += 1;
            v
        } else {
            0.0
        }
    };
    let mut sum = 0.0f32;
    let mut k = 0usize;
    for i in 0..p_target.len() {
        let d = p_target[i] - q_at(i, &mut k);
        if d > 0.0 {
            sum += d;
        }
    }
    if sum <= 0.0 {
        return argmax_u32(p_target);
    }
    let u_scaled = u * sum;
    let mut acc = 0.0f32;
    let mut k = 0usize;
    for i in 0..p_target.len() {
        let d = p_target[i] - q_at(i, &mut k);
        if d > 0.0 {
            acc += d;
            if u_scaled < acc {
                return i as u32;
            }
        }
    }
    (p_target.len() - 1) as u32
}

// `NgramCache` / `PldMatcher` / `PldMatch` moved to `hipfire_runtime::spec` so
// the arch-generic `NgramSpeculator` can use them without an arch-crate
// dependency. Re-exported here so `spec_step_dflash` (and the dflash_spec_demo)
// keep referring to them unqualified.
pub use hipfire_runtime::spec::{NgramCache, PldMatch, PldMatcher};

/// Small, fast RNG for per-cycle sampling u ∈ [0, 1). Xorshift64*; deterministic
/// given the seed, cheap enough to inline into the B-rejection loop.
#[inline]
pub(crate) fn xorshift_next_unit(state: &mut u64) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    // Top 24 bits for a reasonable float mantissa; divide by 2^24.
    ((s >> 40) as f32) * (1.0 / 16_777_216.0)
}

/// Aggregated metrics for a sequence of speculative decode steps.
#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    /// Total number of speculative cycles run.
    pub cycles: usize,
    /// Total number of tokens committed (sum of committed.len() across cycles).
    pub committed_tokens: usize,
    /// Total number of draft tokens accepted (sum of `accepted`).
    pub accepted_tokens: usize,
    /// Per-cycle acceptance count histogram, indexed by accepted count
    /// (0..=k). `acceptance_hist[i]` = number of cycles where exactly `i`
    /// draft tokens were accepted.
    pub acceptance_hist: Vec<usize>,
}

impl SpecStats {
    pub fn new(k: usize) -> Self {
        Self {
            cycles: 0,
            committed_tokens: 0,
            accepted_tokens: 0,
            acceptance_hist: vec![0; k + 1],
        }
    }

    pub fn record(&mut self, step: &SpecStepResult) {
        self.cycles += 1;
        self.committed_tokens += step.committed.len();
        self.accepted_tokens += step.accepted;
        if step.accepted < self.acceptance_hist.len() {
            self.acceptance_hist[step.accepted] += 1;
        }
    }

    /// Mean accepted draft tokens per cycle. This is τ from the Leviathan paper.
    pub fn tau(&self) -> f32 {
        if self.cycles == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.cycles as f32
        }
    }

    /// Mean committed tokens per cycle (tau + 1 on average, since each
    /// cycle always commits one bonus token).
    pub fn mean_committed(&self) -> f32 {
        if self.cycles == 0 {
            0.0
        } else {
            self.committed_tokens as f32 / self.cycles as f32
        }
    }
}

/// One speculative decode step (greedy, Leviathan verify-and-accept).
/// Operates on separate `target` and `draft` `ModelSlot` handles so the
/// caller can keep them owned in top-level variables.
///
/// Preconditions:
/// - Both `target.scratch.logits` and `draft.scratch.logits` contain the
///   logits for position `pos` (from the previous commit or prompt prefill).
/// - `target_snap` / `draft_snap` are preallocated via `DeltaNetSnapshot::new_for`.
/// - `k >= 1` is the speculation count.
///
/// Postconditions:
/// - Both slots' state advances to `pos + committed.len()`, and their
///   `scratch.logits` contain logits at the new position.
/// - Returns a `SpecStepResult` describing how many draft tokens were
///   accepted, the bonus token, and the full committed sequence.
///
/// Naive sequential verification: runs the target on each drafted token one
/// at a time. Phase 5 replaces the inner loop with a single batched prefill.
pub fn spec_step_greedy(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft: &mut ModelSlot,
    pos: usize,
    k: usize,
    target_snap: &mut DeltaNetSnapshot,
    draft_snap: &mut DeltaNetSnapshot,
) -> HipResult<SpecStepResult> {
    assert!(k >= 1, "speculation count k must be ≥ 1");

    // Snapshot both models' recurrent state at position `pos` so we can
    // rewind after verification and commit the final accepted prefix.
    target_snap.save_from(&target.dn_state, gpu)?;
    draft_snap.save_from(&draft.dn_state, gpu)?;

    // Target's current logits (at position `pos`) are used to verify
    // drafted[0]. Capture before anything trashes them.
    let target_logits_at_pos: Vec<f32> = gpu.download_f32(&target.scratch.logits)?;

    // Draft k tokens. drafted[0] samples from draft's current logits (which
    // are also for position `pos`). drafted[i] samples from the logits
    // produced by draft.forward(drafted[i-1], pos+i-1).
    let mut drafted: Vec<u32> = Vec::with_capacity(k);
    {
        let first_logits = gpu.download_f32(&draft.scratch.logits)?;
        drafted.push(argmax_u32(&first_logits));
    }
    for i in 0..k {
        draft.forward(gpu, drafted[i], pos + i)?;
        if i + 1 < k {
            let logits = gpu.download_f32(&draft.scratch.logits)?;
            drafted.push(argmax_u32(&logits));
        }
    }

    // Verification: run the target on each drafted token, collect logits.
    // target_mid_logits[i] = target's prediction at position pos+i+1.
    let mut target_mid_logits: Vec<Vec<f32>> = Vec::with_capacity(k);
    for i in 0..k {
        target.forward(gpu, drafted[i], pos + i)?;
        target_mid_logits.push(gpu.download_f32(&target.scratch.logits)?);
    }
    // Acceptance:
    //   drafted[0] verified by target_logits_at_pos  (logits at pos)
    //   drafted[i] (i >= 1) verified by target_mid_logits[i-1] (logits at pos+i)
    let mut accepted: usize = 0;
    if !target_logits_at_pos.is_empty() && argmax_u32(&target_logits_at_pos) == drafted[0] {
        accepted = 1;
        for i in 1..k {
            if argmax_u32(&target_mid_logits[i - 1]) == drafted[i] {
                accepted += 1;
            } else {
                break;
            }
        }
    }

    // Bonus token = target's prediction at position pos+accepted.
    let bonus_logits: &[f32] = if accepted == 0 {
        &target_logits_at_pos
    } else {
        &target_mid_logits[accepted - 1]
    };
    let bonus_token = argmax_u32(bonus_logits);

    // Commit = accepted draft prefix + bonus.
    let mut committed: Vec<u32> = Vec::with_capacity(accepted + 1);
    committed.extend_from_slice(&drafted[..accepted]);
    committed.push(bonus_token);

    // Restore both models' state and replay the committed sequence so both
    // slots end at `pos + committed.len()` with correct logits.
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    draft_snap.restore_to(&mut draft.dn_state, gpu)?;
    for (i, &tok) in committed.iter().enumerate() {
        target.forward(gpu, tok, pos + i)?;
        draft.forward(gpu, tok, pos + i)?;
    }

    Ok(SpecStepResult {
        accepted,
        bonus_token,
        drafted,
        committed,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// DFlash-specific target-side verify
// ═══════════════════════════════════════════════════════════════════════════

/// Output of a DFlash target verify step.
pub struct DflashVerifyOutput {
    /// Target argmax token at each of the B positions. argmax_per_pos[i]
    /// is what the target would greedy-decode at absolute position
    /// `start_pos + i` given the preceding context plus `draft_tokens[0..i]`.
    pub argmax_per_pos: Vec<u32>,
    /// Full logits downloaded for every position, concatenated row-major
    /// as `[B * vocab_size]`. Only populated when `want_full_logits=true`
    /// (i.e. temperature sampling). Empty otherwise — greedy decode
    /// uses GPU argmax and ships just B × 4 bytes to the host.
    pub logits_per_pos: Vec<f32>,
}

fn dflash_use_gdn_tape_replay(caller_supplied_tape: bool, verify_populates_tape: bool) -> bool {
    caller_supplied_tape && verify_populates_tape
}

/// Actual verify-route facts the decision→commit body needs. These are
/// measured by whoever ran the verify forward, never assumed by the commit.
///
/// * `same_as_hip_graph`: certified state parity
///   (`railgun::window::WindowVerifyReceipt::same_as_hip_graph`): the verify
///   left exactly the state the ordinary HIP/HipGraph body leaves (live
///   DeltaNet, tape, KV, hidden staging, frames). True for the ordinary
///   HIP/HipGraph route (or a PM4 route that planned this window as an
///   ordinary HIP window) and for a railgun window whose plan was certified;
///   false for legacy retained-PM4 windows. Only then may a full accept keep
///   the verify-advanced DeltaNet state (pre-factor singleton
///   `spec_step_dflash` lines 7174-7180).
/// * `tape_populated`: the verify actually captured the GDN tape into the
///   state's `gdn_tape` at offset 0 (caller supplied a tape AND
///   `prefill_batch_pbs_eligible` held; the singleton's
///   `gdn_tape_opt.is_some()`). `false` selects the committed-token batched
///   forward fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflashVerifyReceipt {
    pub same_as_hip_graph: bool,
    pub tape_populated: bool,
}

/// Receipt for a window whose verify ran the shared eager HIP route
/// (`forward_prefill_batch` + tape capture) over `b` rows on explicit trunk
/// owners: `same_as_hip_graph = true`, `tape_populated` = the actual
/// `prefill_batch_pbs_eligible` decision the verify used (pre-factor greedy
/// lines 3839-3853).
pub fn dflash_eager_hip_verify_receipt(
    gpu: &Gpu,
    target: &DflashTargetParts<'_>,
    df: &crate::dflash_spec::DflashState,
    b: usize,
) -> DflashVerifyReceipt {
    let moe_router_logits_present = df
        .verify_scratch
        .prefill_batch
        .as_ref()
        .map(|pbs| pbs.moe_router_logits_batch.is_some())
        .unwrap_or(true);
    let verify_populates_tape = qwen35::prefill_batch_pbs_eligible(
        target.weights,
        target.config,
        &*target.dn_state,
        b,
        gpu.arch.as_str(),
        moe_router_logits_present,
    );
    DflashVerifyReceipt {
        same_as_hip_graph: true,
        tape_populated: dflash_use_gdn_tape_replay(true, verify_populates_tape),
    }
}

/// CPU-shadow (`ctx_slice = Some`) hidden mirror the commit must extend:
/// `host` receives the first `accept+1` rows of the verify's `b` hidden rows,
/// `ne * h` f32 per row.
struct DflashCpuShadow<'a> {
    host: &'a mut Vec<f32>,
    ne: usize,
    h: usize,
}

/// Borrowed DFlash state the decision→commit body touches. The singleton owns
/// these as separate parameters (not a `DflashState`), the lane as
/// `DflashState` fields; both build this view so the repair logic exists once.
/// `gdn_tape` is `Some` iff the verify populated the tape.
struct DflashBorrowedCommit<'a> {
    hidden_rb: &'a HiddenStateRingBuffer,
    draft_scratch: &'a mut DflashScratch,
    target_snap: &'a DeltaNetSnapshot,
    gdn_tape: Option<&'a GdnTape>,
    cpu_shadow: Option<DflashCpuShadow<'a>>,
}

/// Window facts for the shared commit body. `decision` is already budgeted
/// (greedy: clamped to `max_accept`; sampled: drawn under the budget).
struct DflashCommitWindow<'a> {
    position: usize,
    seed_token: u32,
    block: &'a [u32],
    decision: SampledWindowDecision,
    same_as_hip_graph: bool,
    phase_on: bool,
}

/// Phase-diagnostic instants captured by the commit body (the singleton's
/// `[phase]` line consumes them; every other caller ignores them). When
/// `phase_on`, each is preceded by a `device_synchronize`, exactly as in the
/// pre-factor singleton.
struct DflashCommitPhaseTimes {
    accept_end: std::time::Instant,
    scatter_end: std::time::Instant,
    restore_end: std::time::Instant,
}

/// The single decision→commit body shared by the greedy lane commit, the
/// public sampled-decision commit and the singleton `spec_step_dflash`:
/// committed assembly → hidden ring → interleaved draft context scatter +
/// thlog append (or CPU-shadow extend) → DeltaNet rewind/replay (snapshot-source
/// tape replay, full-accept keep-verified skip, restore + tape replay, or the
/// batched-forward fallback). Statement order is the pre-factor singleton's
/// (`spec_step_dflash` lines 7057-7240) and greedy's (lines 3813-3908).
/// Returns `committed = [seed, accepted drafts.., bonus]`.
fn dflash_commit_decision_borrowed(
    gpu: &mut Gpu,
    target: &mut DflashTargetParts<'_>,
    parts: DflashBorrowedCommit<'_>,
    window: DflashCommitWindow<'_>,
) -> HipResult<(Vec<u32>, DflashCommitPhaseTimes)> {
    let DflashBorrowedCommit {
        hidden_rb,
        draft_scratch,
        target_snap,
        gdn_tape,
        cpu_shadow,
    } = parts;
    let DflashCommitWindow {
        position,
        seed_token,
        block,
        decision,
        same_as_hip_graph,
        phase_on,
    } = window;
    let b = block.len();
    let accept_len = decision.accepted_drafts;
    if b < 2 || accept_len + 1 > b || decision.committed_len != accept_len + 2 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dflash commit: inconsistent decision (b={b}, accepted={accept_len}, committed_len={})",
                decision.committed_len
            ),
        ));
    }

    // ── 8. Committed sequence ───────────────────────────────────────────
    // committed[0] is the seed_token (already emitted by prev iter). The
    // caller's output stream appends committed[1..].
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 2);
    committed.push(seed_token);
    for i in 0..accept_len {
        committed.push(block[i + 1]);
    }
    committed.push(decision.bonus_token);
    debug_assert_eq!(committed.len(), accept_len + 2);

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_accept_end = std::time::Instant::now();

    // ── 9. Append accepted target hidden rows (accept_len + 1) ──────────
    // The bonus token's hidden is NOT appended: its slot holds the rejected
    // draft's forward; the next verify forwards the bonus as block[0].
    let rows_to_keep = accept_len + 1;
    if let Some(shadow) = cpu_shadow {
        // ctx_slice path: CPU shadow still required for the window slice.
        let hidden_block = download_hidden_block(gpu, hidden_rb, b)?;
        shadow
            .host
            .extend_from_slice(&hidden_block[..rows_to_keep * shadow.ne * shadow.h]);
    } else {
        // Fast path: scatter straight from hidden_rb into draft scratch on GPU
        // (block_size = b aligns to the verify-block origin, not the ring
        // tail), then track the appended absolute positions for the next
        // cycle's `positions_k` construction.
        scatter_hidden_block_to_interleaved(
            gpu,
            hidden_rb,
            &draft_scratch.target_hidden,
            position,
            b,
            rows_to_keep,
            draft_scratch.ctx_modulus(),
        )?;
        let co = target.kv_cache.compact_offset as i32;
        draft_scratch
            .thlog
            .append_committed(position, rows_to_keep, co);
    }

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_scatter_end = std::time::Instant::now();

    // ── 10. Rewind DeltaNet + replay committed tokens ───────────────────
    // Railgun D8: `target_snap` stays the pre-window state; a full accept on
    // the HIP/HipGraph verify with EF on keeps the verify-advanced live
    // buffers, any shorter accept replays from the snapshot source, any other
    // decline restores then replays (tape) or re-forwards (no tape).
    let replay_from_snapshot = gdn_tape.is_some_and(|tape| {
        tape.replay_from_snapshot_admits(gpu, &*target.dn_state, target_snap, accept_len + 1)
    });
    let keep_verified = replay_from_snapshot
        && same_as_hip_graph
        && accept_len + 1 == b
        && !target.dn_state.s_ef_residual.is_empty();
    if !replay_from_snapshot {
        target_snap.restore_to(target.dn_state, gpu)?;
    }

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_restore_end = std::time::Instant::now();
    if let Some(tape) = gdn_tape {
        // The tables are armed with the tape, so an admitted snapshot-source
        // replay runs or errors; it never falls back to another route.
        if replay_from_snapshot {
            if !keep_verified {
                let replayed = tape.replay_gdn_from_snapshot(
                    gpu,
                    target.weights,
                    target.config,
                    target.dn_state,
                    target_snap,
                    accept_len + 1,
                )?;
                debug_assert!(replayed, "admitted above");
            }
        } else {
            tape.replay_gdn(
                gpu,
                target.weights,
                target.config,
                target.dn_state,
                accept_len + 1,
            )?;
        }
    } else {
        let replay_tokens = &committed[..accept_len + 1];
        // Batched-forward rollback replay: inside the PM FP8 / IU4 verify scope
        // so the rewritten KV/DeltaNet rows are the verify's FP8 / IU4 rows
        // (n >= 2; the per-token n == 1 fallback inside is the decode GEMV).
        let fp8 = fp8_verify_target(target.weights, target.config, target.kv_cache);
        let iu4 =
            qwen35::prefill::multi::dflash_iu4_verify_admitted(gpu, target.weights, target.config);
        fp8_verify_scoped(gpu, fp8, |gpu| {
            iu4_verify_scoped(gpu, iu4, |gpu| {
                qwen35::forward_prefill_batch(
                    gpu,
                    target.weights,
                    target.config,
                    replay_tokens,
                    position,
                    target.kv_cache,
                    target.dn_state,
                    target.scratch,
                    None,
                    None,
                    None,
                    None,
                )
            })
        })?;
    }
    Ok((
        committed,
        DflashCommitPhaseTimes {
            accept_end: t_accept_end,
            scatter_end: t_scatter_end,
            restore_end: t_restore_end,
        },
    ))
}

/// Decision→commit consumer for one already-decided DFlash window over
/// explicit trunk owners (continuous-batching lanes). `decision` MUST already
/// respect `draft.max_accept` (greedy clamps before calling; sampled draws
/// under the budget): no argmax, clamp or re-draw happens here, and a decision
/// over budget is an error rather than silently clamped.
///
/// Caller contract: the pre-window `df.target_snap` was saved from
/// `target.dn_state` before the verify trunk ran, the verify captured its tape
/// (iff `verify.tape_populated`) into `df.gdn_tape` at offset 0 and its
/// `draft.verify_tokens.len()` hidden rows into `df.hidden_rb`, and `verify`
/// reports the actual route. Lane windows use ctx_slice=None (no CPU shadow);
/// the singleton's CPU-shadow variant goes through the same private body.
pub fn dflash_commit_decision_parts(
    gpu: &mut Gpu,
    target: &mut DflashTargetParts<'_>,
    df: &mut crate::dflash_spec::DflashState,
    draft: &DflashCbDraft,
    decision: SampledWindowDecision,
    verify: DflashVerifyReceipt,
) -> HipResult<SpecStepResult> {
    let block: &[u32] = &draft.verify_tokens;
    if block.len() < 2 {
        return Err(hip_bridge::HipError::new(
            0,
            "dflash commit: block size must be >= 2",
        ));
    }
    if decision.accepted_drafts > draft.max_accept {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dflash commit: decision accepts {} drafts over the pre-commit budget {}",
                decision.accepted_drafts, draft.max_accept
            ),
        ));
    }
    let tape = verify.tape_populated.then_some(&df.gdn_tape);
    let (committed, _times) = dflash_commit_decision_borrowed(
        gpu,
        target,
        DflashBorrowedCommit {
            hidden_rb: &df.hidden_rb,
            draft_scratch: &mut df.draft_scratch,
            target_snap: &df.target_snap,
            gdn_tape: tape,
            cpu_shadow: None,
        },
        DflashCommitWindow {
            position: draft.position,
            seed_token: draft.seed,
            block,
            decision,
            same_as_hip_graph: verify.same_as_hip_graph,
            phase_on: false,
        },
    )?;
    Ok(SpecStepResult {
        accepted: decision.accepted_drafts,
        bonus_token: decision.bonus_token,
        drafted: block.to_vec(),
        committed,
    })
}

/// Greedy accept + committed-state consumer for one precomputed DFlash verify
/// window over explicit trunk owners (continuous-batching lanes).
///
/// This is the fast greedy chain arm of [`spec_step_dflash`] (ctx_slice=None,
/// no host-shaped penalties, no PLD/ngram/CACTUS) lifted over a precomputed
/// [`DflashVerifyOutput`]: `eos=None` greedy accept prefix, pre-commit
/// `max_accept` clamp with bonus re-pick (pre-factor lines 3802-3811), then the
/// shared [`dflash_commit_decision_parts`] body (pre-factor lines 3813-3908).
///
/// Caller contract: the pre-window `df.target_snap` was saved from
/// `target.dn_state` before the verify trunk ran, the verify captured its tape
/// into `df.gdn_tape` at offset 0 and its `draft.verify_tokens.len()` hidden
/// rows into `df.hidden_rb`, and the verify ran the eager HIP route (so the
/// full-accept keep-verified skip is admissible exactly as on the singleton
/// HIP/HipGraph route). Seed-oracle statistics and phase timing diagnostics
/// stay on the singleton path.
pub fn dflash_greedy_accept_commit_parts(
    gpu: &mut Gpu,
    target: &mut DflashTargetParts<'_>,
    df: &mut crate::dflash_spec::DflashState,
    draft: &DflashCbDraft,
    verified: &DflashVerifyOutput,
) -> HipResult<SpecStepResult> {
    let block: &[u32] = &draft.verify_tokens;
    let b = block.len();
    if b < 2 {
        return Err(hip_bridge::HipError::new(
            0,
            "dflash greedy accept: block size must be >= 2",
        ));
    }
    if verified.argmax_per_pos.len() != b {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "dflash greedy accept: {} target picks for a {b}-row window",
                verified.argmax_per_pos.len()
            ),
        ));
    }
    let argmax_per_pos: &[u32] = &verified.argmax_per_pos;

    // Shared greedy accept-prefix (eos=None: DFlash never early-stops on EOS).
    let acc = hipfire_runtime::spec::accept_greedy_prefix(&block[1..b], argmax_per_pos, None);
    let mut accept_len = acc.accepted;
    let mut bonus_token = *acc.committed.last().expect("eos=None yields a bonus");
    // Pre-commit budget: clamp drafts so emit (= accept + 1) <= max_emit, then
    // re-pick bonus = argmax at the clamped boundary.
    if accept_len > draft.max_accept {
        accept_len = draft.max_accept;
        bonus_token = argmax_per_pos[accept_len];
    }

    let receipt = dflash_eager_hip_verify_receipt(gpu, target, df, b);
    dflash_commit_decision_parts(
        gpu,
        target,
        df,
        draft,
        SampledWindowDecision {
            accepted_drafts: accept_len,
            bonus_token,
            committed_len: accept_len + 2,
        },
        receipt,
    )
}

/// Run the target on `draft_tokens` (length B) positions starting at
/// `start_pos`. Advances `target.kv_cache` and `target.dn_state` by B
/// positions. Writes B hidden-state rows into `hidden_rb` (ring head
/// advances B times). Returns downloaded logits + argmax per position.
///
/// Fast path (0.1.7 batched verify): one `forward_prefill_batch` call
/// over all B tokens with hidden extraction + per-token post-output-norm
/// hidden capture. Then B sequential `weight_gemv`s against the target's
/// lm_head to get per-position logits. The batched layer-level kernels
/// amortize launch overhead across all B tokens; the lm_head still loops
/// because a batched Q8/MQ4 lm_head GEMM isn't wired yet (task #13).
///
/// Fallback: when the batched path is ineligible (non-MQ weights,
/// non-Q8/asym KV cache, N < MIN_BATCH), `forward_prefill_batch` routes
/// to the per-token loop using `forward_scratch_with_hidden`, so hidden
/// extraction still works.
pub fn verify_dflash_block(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    verify_scratch: &VerifyScratch,
) -> HipResult<DflashVerifyOutput> {
    verify_dflash_block_inner(
        gpu,
        target,
        draft_tokens,
        start_pos,
        hidden_rb,
        gdn_tape,
        want_full_logits,
        None,
        verify_scratch,
        false, // chain path always needs argmax
        None,  // no retained-PM4 route on the generic wrapper
    )
}

/// Tree-verify variant of `verify_dflash_block`. Pass the linearized
/// `(positions, attn_bias)` built from a `DdTree` and this runs the whole
/// tree through a single batched forward — per-position argmax at slot i
/// corresponds to target's prediction after tree node i (slot 0 = after
/// seed / tree root).
///
/// Note: `gdn_tape` captured from a tree verify records innovations in
/// linearization order, NOT commit order. Callers that need to advance
/// GDN state to a specific committed path should either (a) re-verify
/// the committed linear prefix with `verify_dflash_block` (no tree) and
/// capture tape on that, matching `spec_step_ddtree`'s pattern, or (b)
/// implement a slot-reordering replay (not currently available).
pub fn verify_dflash_block_tree(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    tree_verify: qwen35::TreeVerifyCtx<'_>,
    verify_scratch: &VerifyScratch,
    // D9: when the caller will use the SWOR walk (which derives accepted
    // indices from the 68-byte walk-result D2H, not from argmax_per_pos),
    // skip the big_n × 4 byte argmax download entirely — the returned
    // argmax_per_pos will be empty. Pass `false` for the greedy path.
    skip_argmax_d2h: bool,
) -> HipResult<DflashVerifyOutput> {
    verify_dflash_block_inner(
        gpu,
        target,
        draft_tokens,
        start_pos,
        hidden_rb,
        gdn_tape,
        want_full_logits,
        Some(tree_verify),
        verify_scratch,
        skip_argmax_d2h,
        None, // tree verify never takes the retained-PM4 route
    )
}

/// Run `f` inside the PM FP8 verify scope opened for `target`
/// (`Gpu::fp8_verify_enter`; closed unless the forward qualifies), closing it
/// on every path. For target forwards that re-commit state after a verify, so
/// they reproduce the verify's FP8 rows.
pub(crate) fn fp8_verify_scoped<T>(
    gpu: &mut Gpu,
    target: Fp8VerifyTarget,
    f: impl FnOnce(&mut Gpu) -> HipResult<T>,
) -> HipResult<T> {
    let prev = gpu.fp8_verify_enter(target)?;
    let r = f(gpu);
    gpu.fp8_verify_exit(prev);
    r
}

/// Run `f` inside the IU4 verify scope when `admitted`
/// ([`qwen35::prefill::multi::dflash_iu4_verify_admitted`]), closing it on
/// every path. For DFlash target forwards that re-commit state after a verify,
/// so they reproduce the verify's IU4 rows.
pub(crate) fn iu4_verify_scoped<T>(
    gpu: &mut Gpu,
    admitted: bool,
    f: impl FnOnce(&mut Gpu) -> HipResult<T>,
) -> HipResult<T> {
    let prev = gpu.iu4_verify_enter(admitted);
    let r = f(gpu);
    gpu.iu4_verify_exit(prev);
    r
}

/// Target verify trunk + verify lm_head/argmax inside the PM FP8 verify GEMM
/// scope (`HIPFIRE_CB_VERIFY_FP8`, see `qwen35::prefill::fp8_verify_target`):
/// one MQ4G256V2 -> E4M3 arithmetic for every row count, so this singleton ==
/// the batched continuous-batching verify byte-for-byte. The scope covers the
/// HIP-graph capture/replay and eager arms and the lm_head enqueue, and is
/// closed on every path. A tree verify never opens it.
///
/// DFlash IU4 verify (the load's decision: default on at `dflash_block` 8,
/// `HIPFIRE_CB_VERIFY_IU4` forces/kills; never with `HIPFIRE_CB_VERIFY_FP8`): a chain verify
/// of an admitted target runs inside the IU4 verify scope (see
/// `Gpu::iu4_verify_enter`), the same A4 producers + IU4 GEMMs as the
/// multi-request verify trunk; lm_head stays F16. The scope is eager-only:
/// the body takes no HipGraph route while it is set, and the retained PM4
/// route is never armed with it (`dflash_spec`).
#[allow(clippy::too_many_arguments)]
fn verify_dflash_block_inner(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    tree_verify: Option<qwen35::TreeVerifyCtx<'_>>,
    verify_scratch: &VerifyScratch,
    skip_argmax_d2h: bool,
    retained: Option<&mut RetainedCtx<'_>>,
) -> HipResult<DflashVerifyOutput> {
    let fp8 = if tree_verify.is_some() {
        Fp8VerifyTarget::Ineligible
    } else {
        fp8_verify_target(&target.weights, &target.config, &target.kv_cache)
    };
    // A retained route that stands down (`HipAuto`, e.g. PM4 disabled) runs
    // the ordinary HIP body; prime/capture/PM4 windows are never IU4-scoped.
    let iu4 = tree_verify.is_none()
        && retained
            .as_ref()
            .is_none_or(|c| c.selected == DflashVerifyRoute::HipAuto)
        && qwen35::prefill::multi::dflash_iu4_verify_admitted(gpu, &target.weights, &target.config);
    let prev = gpu.fp8_verify_enter(fp8)?;
    let prev_iu4 = gpu.iu4_verify_enter(iu4);
    let r = verify_dflash_block_inner_body(
        gpu,
        target,
        draft_tokens,
        start_pos,
        hidden_rb,
        gdn_tape,
        want_full_logits,
        tree_verify,
        verify_scratch,
        skip_argmax_d2h,
        retained,
    );
    gpu.iu4_verify_exit(prev_iu4);
    gpu.fp8_verify_exit(prev);
    r
}

/// Launch-fusion context of a verify window: the frozen AR/verify
/// discriminator. The split variant is selected only when this exact live
/// window also satisfies the dense gfx1100 Q8 attention route. That prevents
/// the split flag from widening capture-time F16/LDS auxiliary kernels below
/// the crossover or on unsupported model shapes. Shared by the verify body
/// and the railgun plan emitter so both read one decision.
pub(crate) fn dflash_verify_fusion_ctx(
    gpu: &Gpu,
    target: &ModelSlot,
    tree_verify_present: bool,
    b: usize,
    required_tokens: usize,
) -> qwen35::DflashFusionCtx {
    if !tree_verify_present {
        if qwen35::prefill::gfx1100_split_verify_admitted(
            gpu,
            &target.config,
            target.kv_cache.quant_q8 && matches!(target.kv_cache.v_mode, llama::VMode::Q8),
            target.kv_cache.uses_vmm_backend(),
            b,
            required_tokens,
            target.scratch.flash_partials.numel(),
        ) {
            qwen35::DflashFusionCtx::ChainVerifySplit
        } else {
            qwen35::DflashFusionCtx::ChainVerify
        }
    } else {
        qwen35::DflashFusionCtx::Off
    }
}

/// Route class of one DFlash verify window for the HipGraph cache: does the
/// gfx1201 native-fp8 split-KV verifier (`Gpu::fp8_fa2_split_verify_admitted`)
/// serve it at its LIVE context (`start_pos + b`)? A captured graph replays
/// at any context, so it is re-captured whenever this class changes (the
/// graph's attend is sized by `physical_cap`, which says nothing about the
/// window). Plain chain verify only: tree windows and non-fp8 KV never take
/// the route.
fn dflash_fp8_split_verify_class(
    gpu: &Gpu,
    target: &ModelSlot,
    start_pos: usize,
    b: usize,
    tree: bool,
) -> bool {
    !tree
        && target.kv_cache.quant_fp8
        && gpu
            .fp8_fa2_split_verify_admitted(
                start_pos + b,
                target.config.n_heads,
                target.config.n_kv_heads,
                target.config.head_dim,
                b,
                target.scratch.flash_partials.numel(),
            )
            .is_some()
}

/// Whether this verify window takes the HIP-graph body (warmup / capture /
/// replay of `forward_prefill_batch_single_chunk_captured_opts`). Shared by
/// the verify body and the railgun plan emitter so both read one decision.
pub(crate) fn dflash_verify_hip_graph_route_ok(
    gpu: &Gpu,
    target: &ModelSlot,
    verify_scratch: &VerifyScratch,
    b: usize,
    tree_ok_for_graph: bool,
) -> bool {
    // NB: NO Q8-long-context skip-gate here. The captured single-chunk verify
    // forward is capture-safe at any physical_cap for Q8 (non-asym) KV:
    // forward_prefill_chunk routes max_ctx_len > 8192 to the tiled
    // attention_flash_q8_0_tile_batched (O(1) LDS, no per-position malloc), so there
    // is no malloc-in-capture and no per-ctx LDS scaling — see the comment in
    // forward_prefill_batch_single_chunk_captured in qwen35.rs. The former
    // `VERIFY_GRAPH_Q8_CTX_LIMIT = 15000` skip (re-introduced by #481, obsolete once
    // the tiled crossover landed 2026-06-09) is intentionally NOT reinstated: it
    // would skip a verify-graph that now replays fine and forfeit the long-context
    // spec speedup.
    let moe_router_logits_present = verify_scratch
        .prefill_batch
        .as_ref()
        .map(|pbs| pbs.moe_router_logits_batch.is_some())
        .unwrap_or(true);
    let pbs_eligible = qwen35::prefill_batch_pbs_eligible(
        &target.weights,
        &target.config,
        &target.dn_state,
        b,
        gpu.arch.as_str(),
        moe_router_logits_present,
    );
    // See `dflash_verify_graph_env_eligible`: exact gfx1100 MQ*V2 / MQ4G256 and
    // exact gfx1201 MQ4G256 are default-off for HipGraph only.
    let verify_graph_env = hipfire_config::developer_var("HIPFIRE_VERIFY_GRAPH").ok();
    dflash_verify_graph_env_eligible(
        gpu.arch.as_str(),
        target.weights.output.gpu_dtype,
        verify_graph_env.as_deref(),
    ) && tree_ok_for_graph
        && matches!(
            target.weights.embd_format,
            hipfire_runtime::llama::EmbeddingFormat::HFQ4G256
                | hipfire_runtime::llama::EmbeddingFormat::Q8_0,
        )
        && verify_scratch.prefill_batch.is_some()
        && pbs_eligible
        // IU4 verify: the gfx1201 A4 producers are eager-only (their gates
        // refuse capture), so the scoped verify always runs eager.
        && !gpu.iu4_verify_scope
}

#[allow(clippy::too_many_arguments)]
fn verify_dflash_block_inner_body(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    tree_verify: Option<qwen35::TreeVerifyCtx<'_>>,
    verify_scratch: &VerifyScratch,
    // D9: skip the big_n × 4 argmax D2H when the caller will use SWOR walk
    // (which gets accepted indices from the 68-byte walk-result D2H instead).
    skip_argmax_d2h: bool,
    // Retained-PM4 route for this window, or `None` for the shipping path.
    // Only `verify_dflash_block_retained` ever passes `Some`, and only for the
    // fixed B=16 chain verify.
    mut retained: Option<&mut RetainedCtx<'_>>,
) -> HipResult<DflashVerifyOutput> {
    let b = draft_tokens.len();
    let vocab = target.config.vocab_size;
    let dim = target.config.dim;
    let required_tokens = start_pos.checked_add(b).ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!("DFlash verify KV token range overflow ({start_pos} + {b})"),
        )
    })?;
    qwen35::prefill::release_widened_pbs_for_kv_growth(
        gpu,
        &target.kv_cache,
        &target.config,
        &target.scratch,
        required_tokens,
    )?;
    // Verify replay bypasses the regular qwen35 forward wrappers.
    target
        .kv_cache
        .ensure_mapped_capacity(gpu, required_tokens)?;

    assert!(
        b <= verify_scratch.max_n,
        "verify_scratch max_n {} < b {}",
        verify_scratch.max_n,
        b
    );
    assert_eq!(verify_scratch.dim, dim, "verify_scratch dim mismatch");
    assert_eq!(verify_scratch.vocab, vocab, "verify_scratch vocab mismatch");

    // Views into the persistent scratch — no per-cycle allocation. Sized to
    // the actual current `b` (≤ max_n) so downstream kernels see the right
    // shapes. sub_offset returns a non-owning view; do NOT free these.
    let final_hidden = verify_scratch.final_hidden.sub_offset(0, b * dim);
    let tree_verify_present = tree_verify.is_some();
    // Launch-fusion prescaffold: frozen AR/verify discriminator. The split
    // variant is selected only when this exact live window also satisfies the
    // dense gfx1100 Q8 attention route. That prevents the split flag from
    // widening capture-time F16/LDS auxiliary kernels below the crossover or
    // on unsupported model shapes.
    let fusion = dflash_verify_fusion_ctx(gpu, target, tree_verify_present, b, required_tokens);
    let moe_lmhead_graph_env =
        hipfire_config::developer_var("HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD").ok();
    let moe_lmhead_graph_ok =
        dflash_moe_verify_graph_lmhead_eligible(
            target.config.num_experts,
            want_full_logits,
            tree_verify_present,
            moe_lmhead_graph_env.as_deref(),
        ) && dflash_batched_lm_head_supported(target.weights.output.gpu_dtype);

    // Graph-capture path eligibility. The captured forward bakes in:
    //   - N (the batch size) — via kernel grid dims
    //   - kernel selection + layer-type branches (dispatched once at capture)
    //   - weight/bias/buffer pointers (stable across cycles)
    // Per-cycle inputs (tokens, positions, kv_cache contents, dn_state contents,
    // hidden_rb staging dest) are read from device buffers whose *contents*
    // change between replays — the captured graph reads the current bytes.
    //
    // Eligibility is narrow: HFQ4G256/Q8_0 embedding (uploads via pbs.tokens),
    // tree_ok_for_graph, pbs is Some, and `qwen35::prefill_batch_pbs_eligible`
    // (shared with the forward / spec_step tape gate). When that shared
    // predicate is false — e.g. qt44/qt45 on non-gfx12 — skip capture entirely
    // and take the direct `forward_prefill_batch_with_pbs_opts` route, which
    // performs the correct per-token fallback instead of erroring inside
    // `forward_prefill_batch_single_chunk_captured_opts`.
    // `gdn_tape` is safe because verify is single-chunk → tape_offset=0 always
    // → captured node's dst offset is correct across cycles.
    //
    // Default-on for eligible models (2026-04-21 smoke on 27B MQ4 Qwen3.5
    // showed +14 % tok/s 25.6→29.2, wall-per-cycle 89→80 ms via coalescing
    // verify kernels into one graph replay and saving ~1.3 ms of per-cycle
    // launch overhead). Opt out with HIPFIRE_VERIFY_GRAPH=0.
    // Tree-verify was historically excluded (tree_verify.is_none()) because
    // the tree-attention mask varies per cycle. In theory mask +
    // parent_indices live in fixed `ddtree_scratch` buffers that the caller
    // repopulates via uncaptured memcpy_htod before each graph replay, so
    // the graph's kernels would read fresh data every cycle.
    //
    // DIAGNOSTIC ONLY — known broken 2026-04-24 (commit 480e51e +
    // A/B bench ee0bedf-followup). 3-run median on 27B MQ4 asym3 b12-k2:
    //   code     τ 7.08 → 4.51 (-36 %)   tok/s 110 → 80.1 (-27 %)
    //   prose    τ 2.50 → 3.58 (+43 %)   tok/s 45.8 → 60.2 (+31 %, noisy)
    //   instr    τ 2.19 → 1.77 (-19 %)   tok/s 47.6 → 35.6 (-25 %)
    // Coherence-gate-dflash passes (no attractors), so it's a τ bug, not a
    // correctness bug. Suspect: a scalar kernarg or intra-forward memcpy
    // inside captured region bakes in first-cycle state; when tree shape
    // varies, acceptance collapses on code (high-variance trees) but
    // coincidentally holds on prose (more uniform trees). Needs root-cause
    // dive: most likely candidates are GDN tape-offset scalar kernargs or
    // the parent_indices-driven conv1d path. DO NOT ENABLE in production.
    //
    // Gate kept live so the next session can bisect without re-plumbing.
    let tree_graph_enabled = hipfire_config::developer_var("HIPFIRE_VERIFY_GRAPH_TREE")
        .ok()
        .as_deref()
        == Some("1");
    if tree_graph_enabled && tree_verify_present {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[verify-graph-tree] WARN: HIPFIRE_VERIFY_GRAPH_TREE=1 is DIAGNOSTIC ONLY — known τ regression on code/instruct. Do not use for production benchmarks."
            );
        }
    }
    let tree_ok_for_graph = !tree_verify_present || tree_graph_enabled;
    // NB: NO Q8-long-context skip-gate here. The captured single-chunk verify
    // forward is capture-safe at any physical_cap for Q8 (non-asym) KV:
    // forward_prefill_chunk routes max_ctx_len > 8192 to the tiled
    // attention_flash_q8_0_tile_batched (O(1) LDS, no per-position malloc), so there
    // is no malloc-in-capture and no per-ctx LDS scaling — see the comment in
    // forward_prefill_batch_single_chunk_captured in qwen35.rs. The former
    // `VERIFY_GRAPH_Q8_CTX_LIMIT = 15000` skip (re-introduced by #481, obsolete once
    // the tiled crossover landed 2026-06-09) is intentionally NOT reinstated: it
    // would skip a verify-graph that now replays fine and forfeit the long-context
    // spec speedup.
    let verify_graph_ok =
        dflash_verify_hip_graph_route_ok(gpu, target, verify_scratch, b, tree_ok_for_graph);

    // Per-cycle timing for verify-graph A/B diagnostic
    // (HIPFIRE_VERIFY_GRAPH_TIMING=1). Two device-sync points bracket the
    // forward + lm_head; the recorded mode tag distinguishes replay vs
    // warmup-direct vs first-capture vs no-graph-eligible.
    let vg_timing = hipfire_config::developer_var("HIPFIRE_VERIFY_GRAPH_TIMING")
        .ok()
        .as_deref()
        == Some("1");
    let mut vg_mode = "direct";
    let vg_t0 = if vg_timing {
        gpu.hip.device_synchronize()?;
        Some(std::time::Instant::now())
    } else {
        None
    };
    let mut graph_includes_lmhead_argmax = false;

    // Retained-PM4 route. When the caller handed us a live route for this
    // window it owns the forward outright, and the HipGraph selection below is
    // skipped so the tape never records a graph launch.
    let retained_selected = retained
        .as_ref()
        .map(|ctx| ctx.selected)
        .unwrap_or(DflashVerifyRoute::HipAuto);
    let retained_active = !matches!(retained_selected, DflashVerifyRoute::HipAuto);
    let batch_result = if retained_active {
        vg_mode = "retained";
        let ctx = retained
            .as_mut()
            .expect("retained_active implies a retained context");
        run_retained_verify_forward(
            gpu,
            target,
            draft_tokens,
            start_pos,
            hidden_rb,
            &final_hidden,
            gdn_tape,
            verify_scratch,
            ctx,
            fusion,
        )
    } else if verify_graph_ok {
        let pbs = verify_scratch.prefill_batch.as_ref().unwrap();
        debug_assert!(b <= pbs.max_batch);
        // Pre-capture: pre-upload inputs and ensure a stream exists. memcpy_htod
        // runs on the host/null-stream side and is NOT captured.
        qwen35::upload_prefill_batch_inputs(gpu, pbs, draft_tokens, start_pos)?;
        // 39aa358: tree verify also needs the depth-based RoPE angles on
        // device before replay (the captured RoPE kernel reads
        // pbs.rope_positions; pbs.positions stays the flat KV slot array).
        if let Some(tv) = tree_verify.as_ref() {
            debug_assert_eq!(tv.positions.len(), b, "tree RoPE positions length");
            let rope_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tv.positions.as_ptr() as *const u8, b * 4) };
            gpu.hip.memcpy_htod(&pbs.rope_positions.buf, rope_bytes)?;
        }
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create()?);
        }
        // gfx1201 native-fp8 split-KV verifier: the route depends on this
        // window's LIVE context (`start_pos + b`), which a replay cannot
        // change, so the graph carries the route class it was captured in.
        // Load the launcher's kernels and grow its Q scratch HERE, outside
        // any capture (growth may drop graph state, so before the cache is
        // consulted); then a graph whose class differs from this window's is
        // dropped and re-warmed/captured at the new class.
        let split_class =
            dflash_fp8_split_verify_class(gpu, target, start_pos, b, tree_verify.is_some());
        if split_class && !gpu.fp8_fa2_split_verify_capture_ready(b) {
            gpu.fp8_fa2_split_verify_prepare_capture(b)?;
        }
        if gpu.graphs.verify_has_graph(b) && gpu.graphs.verify_graph_has_fp8_split(b) != split_class
        {
            gpu.hip
                .stream_synchronize(gpu.active_stream.as_ref().unwrap())?;
            gpu.graphs.verify_graph_destroy(&gpu.hip, gpu.device_id, b);
            eprintln!(
                "[verify-graph] B={} route class changed at ctx {} (fp8 split-KV verify {}) — re-warming",
                b,
                start_pos + b,
                if split_class { "on" } else { "off" },
            );
        }
        if gpu.graphs.verify_has_graph(b) {
            vg_mode = "replay";
            graph_includes_lmhead_argmax =
                moe_lmhead_graph_ok && gpu.graphs.verify_graph_has_lmhead_argmax(b);
            // Replay path: kernels read pbs.tokens/pbs.positions/dn_state/
            // kv_cache contents that were freshly updated above + upstream.
            gpu.graphs.verify_graph_launch(
                &gpu.hip,
                gpu.device_id,
                gpu.active_stream.as_ref().unwrap(),
                b,
            )?;
            Ok(())
        } else if gpu.graphs.verify_needs_warmup(b) {
            vg_mode = "warmup";
            // Warmup for this b: run direct so kernel JIT and any lazy scratch
            // allocations (e.g., MQ signs/x_rot/x_q8, FP16 shadow) happen
            // outside any captured region. Capturing a JIT + scratch-malloc
            // hits "hipMalloc not permitted under stream capture" the first
            // time any kernel is compiled inline. One warmup per distinct b.
            //
            // A successful launch sequence is not yet a successful warmup:
            // launches are asynchronous. Synchronize the active stream before
            // admitting this B into the capture branch. Otherwise an async
            // kernel fault can leave `warmed_up` set and make the next call
            // capture a path whose warmup never completed.
            let r = qwen35::forward_prefill_batch_single_chunk_captured_opts(
                gpu,
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
                tree_verify,
                false, // DFlash computes all verify logits from final_hidden below
                fusion,
            );
            r.and_then(|_| {
                gpu.hip.stream_synchronize(
                    gpu.active_stream
                        .as_ref()
                        .expect("verify warmup requires an active stream"),
                )?;
                gpu.graphs.verify_mark_warmup_done(b);
                eprintln!(
                    "[verify-graph] warmup for B={} complete — capture next cycle at this B",
                    b
                );
                Ok(())
            })
        } else {
            vg_mode = "capture";
            // Capture path: first call at this B after warmup.
            let capture_lmhead_argmax = moe_lmhead_graph_ok;
            gpu.graphs.begin_verify_graph_capture(
                &gpu.hip,
                gpu.device_id,
                gpu.active_stream.as_ref().unwrap(),
                b,
            )?;
            let r = qwen35::forward_prefill_batch_single_chunk_captured_opts(
                gpu,
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
                tree_verify,
                false, // DFlash computes all verify logits from final_hidden below
                fusion,
            );
            let r = if r.is_ok() && capture_lmhead_argmax {
                r.and_then(|_| {
                    let math = verify_dense_math(gpu);
                    dflash_enqueue_verify_lm_head_argmax(
                        gpu,
                        &target.weights.output,
                        &final_hidden,
                        verify_scratch,
                        b,
                        vocab,
                        math,
                    )
                })
            } else {
                r
            };
            if r.is_ok() {
                let blob_count = gpu.graphs.capture_blobs.len();
                gpu.graphs.end_verify_graph_capture(
                    &gpu.hip,
                    gpu.device_id,
                    gpu.active_stream.as_ref().unwrap(),
                )?;
                // Under `hipStreamBeginCapture`, kernels + memcpys on the
                // captured stream are RECORDED, not executed. final_hidden
                // and hidden_rb staging are left stale. Launching the graph
                // once here makes this cycle's forward actually run so lm_head
                // reads fresh data. DN state double-advance (if any future
                // HIP version does execute during capture) is washed out by
                // target_snap.restore_to after verify returns. KV cache
                // double-write writes the same data to the same positions.
                //
                // The launch is asynchronous. Complete this one-time first
                // execution before admitting the graph into the replay cache:
                // if execution fails, destroy every per-B entry so a retry
                // cannot enqueue the failed graph (or another graph recorded
                // against the now-faulted stream state).
                let first_launch = gpu
                    .graphs
                    .verify_graph_launch(
                        &gpu.hip,
                        gpu.device_id,
                        gpu.active_stream.as_ref().unwrap(),
                        b,
                    )
                    .and_then(|_| {
                        gpu.hip
                            .stream_synchronize(gpu.active_stream.as_ref().unwrap())
                    });
                if let Err(err) = first_launch {
                    gpu.graphs.verify_graph_destroy_all(&gpu.hip, gpu.device_id);
                    return Err(err);
                }
                if capture_lmhead_argmax {
                    gpu.graphs.verify_mark_graph_lmhead_argmax(b);
                    graph_includes_lmhead_argmax = true;
                }
                if split_class {
                    gpu.graphs.verify_mark_graph_fp8_split(b);
                }
                eprintln!(
                    "[verify-graph] captured for B={} with {} blobs (cache size: {})",
                    b,
                    blob_count,
                    gpu.graphs.verify_graph_count(),
                );
            } else {
                // If capture failed, tear down the partial capture so we fall
                // back to the direct path next cycle cleanly.
                let _ = gpu
                    .hip
                    .stream_end_capture(gpu.active_stream.as_ref().unwrap());
                gpu.graphs.capture_mode = false;
                gpu.graphs.capture_blobs.clear();
            }
            r
        }
    } else {
        qwen35::forward_prefill_batch_with_pbs_opts(
            gpu,
            &target.weights,
            &target.config,
            draft_tokens,
            start_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            Some(hidden_rb),
            Some(&final_hidden),
            gdn_tape,
            tree_verify,
            verify_scratch.prefill_batch.as_ref(),
            None,  // mask_override: speculative verify path doesn't use the MTP probe hook
            None,  // max_layer: DFlash verify always runs the full stack
            false, // DFlash computes all verify logits from final_hidden below
            fusion,
        )
    };

    // Commit hidden_rb staging to the ring (outside any captured region).
    // The captured forward wrote to staging[0..b*h]; this scatter places
    // those rows at the current head and advances head by b. Under the
    // graph path we manually drive this because the non-graph chunk loop
    // (forward_prefill_batch_with_pbs) that usually calls it was bypassed.
    // The retained route runs the same capture-safe single-chunk body as the
    // graph path, so it owes the same one external staging commit.
    if (retained_active || verify_graph_ok) && batch_result.is_ok() {
        hidden_rb.commit_staging_to_ring(gpu, b)?;
    }
    // Tree mode at topk>1 REQUIRES this sync. Without it τ degrades badly
    // (e.g. budget=60 topk=8 drops 7.0 → 3.3; 9B asym3 2026-04-14). topk=1
    // is fine without the sync (byte-exact with baseline DFlash either way).
    // Root cause suspected: siblings at the same tree depth produce
    // duplicate entries in `positions[]`, so `kv_cache_write` dispatches
    // multiple batch rows targeting the same cache slot — the async write
    // order lets a subsequent attention kernel read a partially-committed
    // slot. Fix TODO: either serialize within-kernel per-slot, or ensure
    // the "winning" sibling's write happens last.
    //
    // D16: narrowed from device_synchronize() (full device drain, ~3–5 ms
    // on PCIe, ~50–200 µs on UMA) to stream_synchronize (scoped to the
    // active stream only). Semantically identical on hipfire's single-stream
    // setup, but does not stall work enqueued on other streams. The ordering
    // guarantee — all prior KV writes complete before the next attention
    // kernel reads the slot — is preserved because all ops ride the same
    // stream. Caller (spec_step_ddtree_batched) ensures active_stream is
    // Some before reaching here; the fallback keeps the old full drain for
    // the rare path where it isn't set.
    if batch_result.is_ok() && tree_verify.is_some() {
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(stream)?;
        } else {
            gpu.hip.device_synchronize()?;
        }
    }
    batch_result?;

    // Per-position lm_head. Fast paths in priority order:
    //   Q8_0      → DFlash-scoped Q8 lm_head dispatcher (gfx12 WMMA by default).
    //   MQ4G256   → batched rotate + gemm_hfq4g256 (one launch + one D2H).
    //   HFQ4G256  → batched gemm_hfq4g256 directly.
    //   else      → B sequential weight_gemv calls + B downloads (legacy).
    let DflashVerifyOutput {
        argmax_per_pos,
        logits_per_pos,
    } = dflash_verify_head(
        gpu,
        &target.weights,
        &target.scratch,
        &final_hidden,
        verify_scratch,
        b,
        want_full_logits,
        skip_argmax_d2h,
        graph_includes_lmhead_argmax,
    )?;

    if let Some(t0) = vg_t0 {
        gpu.hip.device_synchronize()?;
        eprintln!(
            "[vg-time] B={} mode={} elapsed_us={}",
            b,
            vg_mode,
            t0.elapsed().as_micros()
        );
    }

    Ok(DflashVerifyOutput {
        argmax_per_pos,
        logits_per_pos,
    })
}

/// Head / materialization half of a DFlash verify window: the target lm_head
/// over the `b` post-norm rows in `final_hidden`, GPU argmax, and the small
/// argmax (or full-logit) download. Shared verbatim by the singleton verify
/// ([`verify_dflash_block_inner`]) and the batched continuous-batching path.
///
/// Per-position lm_head. Fast paths in priority order:
///   Q8_0      → DFlash-scoped Q8 lm_head dispatcher (gfx12 WMMA by default).
///   MQ4G256   → batched rotate + gemm_hfq4g256 (one launch + one D2H).
///   HFQ4G256  → batched gemm_hfq4g256 directly.
///   else      → B sequential weight_gemv calls + B downloads (legacy).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dflash_verify_head(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    scratch: &Qwen35Scratch,
    final_hidden: &GpuTensor,
    verify_scratch: &VerifyScratch,
    b: usize,
    want_full_logits: bool,
    skip_argmax_d2h: bool,
    // The MoE-only extended verify graph already enqueued lm_head+argmax.
    graph_includes_lmhead_argmax: bool,
) -> HipResult<DflashVerifyOutput> {
    let vocab = verify_scratch.vocab;
    let dim = verify_scratch.dim;
    let w_out = &weights.output;
    let mut logits_per_pos: Vec<f32> = Vec::with_capacity(b * vocab);
    let mut argmax_per_pos: Vec<u32> = Vec::with_capacity(b);

    let try_batched = dflash_batched_lm_head_supported(w_out.gpu_dtype);
    if gpu.fp8_verify_scope && !try_batched && !graph_includes_lmhead_argmax {
        // Fail closed: the per-row GEMV fallback is not an FP8-verify arm.
        return Err(HipError::new(
            0,
            "dflash_verify_head: fp8 verify scope needs a batched MQ4G256V2 lm_head",
        ));
    }

    if graph_includes_lmhead_argmax {
        debug_assert!(!want_full_logits);
        // The MoE-only extended verify graph already enqueued lm_head+argmax.
        // Mirror MTP's graph shape: keep device work captured, then perform
        // the single small D2H read after graph launch.
        argmax_per_pos = dflash_download_verify_argmax(gpu, verify_scratch, b)?;
    } else if try_batched {
        let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
        // Q8_0 routes through the DFlash lm_head helper so the gfx12 WMMA
        // arm can be coherence-gated independently of MTP. Set
        // HIPFIRE_DFLASH_Q8_LMHEAD_WMMA=0 to force the legacy scalar chunks.
        // MQ4/HFQ4/HFQ6/MQ6 kernels have no 64-row cap and take the
        // single-shot path.
        let math = verify_dense_math(gpu);
        dflash_enqueue_verify_lm_head(gpu, w_out, final_hidden, verify_scratch, b, vocab, math)?;
        if want_full_logits {
            // Rejection-sampling path needs full target distribution.
            // Cost: B × vocab × 4 bytes D2H per verify (~15 MB at B=16 × 248K).
            let host_logits = gpu.download_f32(&logits_batch)?;
            for i in 0..b {
                let row = &host_logits[i * vocab..(i + 1) * vocab];
                argmax_per_pos.push(argmax_u32(row));
            }
            logits_per_pos = host_logits;
        } else if skip_argmax_d2h {
            // D9: SWOR verify path — the accept walk uses the 68-byte walk-result
            // D2H (D13), NOT argmax_per_pos. Skip this D2H entirely; argmax_per_pos
            // stays empty. Still enqueue the GPU argmax kernel so verify_scratch.argmax
            // is populated if some other consumer ever needs it (none currently).
            let argmax_buf = verify_scratch.argmax.sub_offset(0, b);
            gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, b)?;
            // argmax_per_pos intentionally left empty — caller must not read it.
        } else {
            // GPU-side batched argmax. Writes B i32 indices; we download just
            // 4*B bytes instead of the full B×vocab logits. Saves ~15 MB of
            // PCIe D2H per verify on the 4B Q8 lm_head (~3-5 ms/iter).
            let argmax_buf = verify_scratch.argmax.sub_offset(0, b);
            gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, b)?;
            argmax_per_pos = dflash_download_verify_argmax(gpu, verify_scratch, b)?;
        }
        // Greedy path doesn't need `logits_per_pos`; leave empty to avoid
        // the 15 MB D2H. If temp>0 sampling is added later, reinstate the
        // download or sample on-GPU.
    } else {
        // Fallback: B sequential GEMVs.
        for i in 0..b {
            let hidden_row = final_hidden.sub_offset(i * dim, dim);
            llama::weight_gemv(gpu, &weights.output, &hidden_row, &scratch.logits)?;
            let row = gpu.download_f32(&scratch.logits)?;
            debug_assert_eq!(row.len(), vocab);
            argmax_per_pos.push(argmax_u32(&row));
            logits_per_pos.extend_from_slice(&row);
        }
    }

    Ok(DflashVerifyOutput {
        argmax_per_pos,
        logits_per_pos,
    })
}

/// Retained-PM4 execution context for one DFlash verify window.
///
/// Built by [`verify_dflash_block_retained`] and threaded into
/// [`verify_dflash_block_inner`], which owns the forward. `selected` is the
/// route [`DflashVerifyPm4::plan_route`] chose for this window;
/// [`DflashVerifyRoute::HipAuto`] means the retained machinery stands down and
/// the shipping path runs unchanged.
struct RetainedCtx<'a> {
    state: &'a mut DflashVerifyPm4,
    binding: DflashVerifyBinding,
    selected: DflashVerifyRoute,
}

/// Why a recording capture stopped short.
///
/// The distinction is the whole safety argument: `Forward` means the model body
/// itself failed and there is no valid result for this cycle, while the other
/// three happen after a synchronized successful body and therefore only cost us
/// the route.
enum CaptureFailure {
    Forward(hip_bridge::HipError),
    Record(String),
    Contract(String),
    Prepare(String),
}

/// How far a successful recording window got.
///
/// The first recording only calibrates: a prepared route may retain a scalar
/// kernarg solely when a second recording at a different position proves the
/// scalar tracks the position.
enum CaptureStep {
    Calibrated(rdna_compute::replay::RecordedKernargSnapshot),
    Ready {
        identity: rdna_compute::replay::PreparedReplayIdentity,
        position_bindings: usize,
    },
}

fn retained_hip_error(reason: &str) -> hip_bridge::HipError {
    hip_bridge::HipError::new(0, reason)
}

/// Fingerprint every shape, dtype, extraction id, and captured allocation base
/// the retained tape depends on. Bases and shapes only — never buffer contents,
/// which change every cycle by design.
fn dflash_verify_fingerprint(
    target: &ModelSlot,
    hidden_rb: &HiddenStateRingBuffer,
    verify_scratch: &VerifyScratch,
    gdn_tape: Option<&GdnTape>,
) -> u64 {
    let mut values: Vec<u64> = Vec::new();
    values.push(target.config.dim as u64);
    values.push(target.config.n_layers as u64);
    values.push(target.config.vocab_size as u64);
    values.push(target.weights.output.gpu_dtype as u64);
    values.push(verify_scratch.max_n as u64);
    values.push(verify_scratch.dim as u64);
    values.push(verify_scratch.vocab as u64);
    values.push(verify_scratch.hidden_k as u64);
    values.push(verify_scratch.final_hidden.buf.as_ptr() as u64);
    values.push(verify_scratch.logits.buf.as_ptr() as u64);
    values.push(verify_scratch.rot.buf.as_ptr() as u64);
    values.push(verify_scratch.argmax.buf.as_ptr() as u64);
    if let Some(pbs) = verify_scratch.prefill_batch.as_ref() {
        values.push(pbs.max_batch as u64);
        values.push(pbs.x_batch.buf.as_ptr() as u64);
        values.push(pbs.rope_positions.buf.as_ptr() as u64);
    }
    values.push(hidden_rb.max_positions as u64);
    values.push(hidden_rb.hidden_dim as u64);
    values.push(hidden_rb.max_batch as u64);
    for &layer_id in &hidden_rb.extract_layers {
        values.push(layer_id as u64);
    }
    // Only the staging buffers are captured. `layer_bufs` are written by the
    // head-dependent commit, which stays outside the tape.
    for staging in &hidden_rb.staging_bufs {
        values.push(staging.buf.as_ptr() as u64);
    }
    for s_matrix in &target.dn_state.s_matrices {
        values.push(s_matrix.buf.as_ptr() as u64);
    }
    if let Some(tape) = gdn_tape {
        values.push(tape.max_n as u64);
        values.push(tape.qkv_dim as u64);
        values.push(tape.v_dim as u64);
        for buf in tape
            .qkv_bufs
            .iter()
            .chain(tape.alpha_bufs.iter())
            .chain(tape.beta_bufs.iter())
        {
            values.push(buf.buf.as_ptr() as u64);
        }
    }
    fingerprint_u64(&values)
}

/// Build the admission binding for the current window.
///
/// `max_position` is the smaller of the hidden ring capacity and the KV
/// physical capacity: a window past it re-primes rather than replaying dynamic
/// geometry that was prepared too small.
fn build_dflash_verify_binding(
    b: usize,
    arch: &str,
    target: &ModelSlot,
    hidden_rb: &HiddenStateRingBuffer,
    verify_scratch: &VerifyScratch,
    gdn_tape: Option<&GdnTape>,
) -> DflashVerifyBinding {
    let fingerprint = dflash_verify_fingerprint(target, hidden_rb, verify_scratch, gdn_tape);
    // The layout generation moves whenever the KV slab is remapped or resized.
    // Expected VMM growth may keep the same reserved base, so capacity is
    // folded in alongside every per-layer allocation base.
    let kv = &target.kv_cache;
    let mut layout: Vec<u64> = Vec::with_capacity(2 + kv.k_gpu.len() * 2);
    layout.push(kv.physical_cap as u64);
    layout.push(kv.max_seq as u64);
    for (k, v) in kv.k_gpu.iter().zip(kv.v_gpu.iter()) {
        layout.push(k.buf.as_ptr() as u64);
        layout.push(v.buf.as_ptr() as u64);
    }
    let layout_generation = fingerprint_u64(&layout);
    DflashVerifyBinding::new(
        b,
        arch,
        fingerprint,
        layout_generation,
        hidden_rb.max_positions.min(kv.physical_cap),
    )
}

/// The capture-safe single-chunk body, run directly with no graph and no
/// recording. Shared by every retained route so the recorded body and the
/// fallback body are literally the same call.
#[allow(clippy::too_many_arguments)]
fn dflash_direct_verify_forward(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    final_hidden: &GpuTensor,
    gdn_tape: Option<&mut GdnTape>,
    pbs: &qwen35::PrefillBatchScratch,
    fusion: qwen35::DflashFusionCtx,
) -> HipResult<()> {
    qwen35::forward_prefill_batch_single_chunk_captured_opts(
        gpu,
        &target.weights,
        &target.config,
        draft_tokens,
        start_pos,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        pbs,
        Some(hidden_rb),
        Some(final_hidden),
        gdn_tape,
        None,
        false, // DFlash computes all verify logits from final_hidden
        fusion,
    )
}

/// Execute one retained-route verify window and report the outcome to the
/// route state machine.
///
/// Host token/position upload happens here, before any recording, and never
/// enters the indirect buffer. lm-head, argmax, accept, and the staging commit
/// all stay with the caller.
#[allow(clippy::too_many_arguments)]
fn run_retained_verify_forward(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    final_hidden: &GpuTensor,
    gdn_tape: Option<&mut GdnTape>,
    verify_scratch: &VerifyScratch,
    ctx: &mut RetainedCtx<'_>,
    fusion: qwen35::DflashFusionCtx,
) -> HipResult<()> {
    let pbs = verify_scratch.prefill_batch.as_ref().ok_or_else(|| {
        retained_hip_error("retained DFlash verify requires a persistent PrefillBatchScratch")
    })?;
    qwen35::upload_prefill_batch_inputs(gpu, pbs, draft_tokens, start_pos)?;
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    match ctx.selected {
        DflashVerifyRoute::HipAuto => Err(retained_hip_error(
            "run_retained_verify_forward called for a HipAuto window",
        )),
        DflashVerifyRoute::PrimeDirect => {
            let result = dflash_direct_verify_forward(
                gpu,
                target,
                draft_tokens,
                start_pos,
                hidden_rb,
                final_hidden,
                gdn_tape,
                pbs,
                fusion,
            );
            if result.is_ok() {
                ctx.state.note_prime_success(ctx.binding.clone());
            }
            result
        }
        DflashVerifyRoute::CaptureRecord => {
            let max_position = ctx.binding.max_position;
            swap_retained_controller(gpu, ctx.state)?;
            let began = gpu.replay.begin_capture();
            if let Err(reason) = began {
                // Nothing was recorded and nothing ran: restore the ordinary
                // controller and produce this cycle's result on direct HIP.
                swap_retained_controller(gpu, ctx.state)?;
                ctx.state
                    .poison(format!("DFlash retained begin capture: {reason}"));
                return dflash_direct_verify_forward(
                    gpu,
                    target,
                    draft_tokens,
                    start_pos,
                    hidden_rb,
                    final_hidden,
                    gdn_tape,
                    pbs,
                    fusion,
                );
            }
            // A prepared route may only retain a kernarg scalar that provably
            // tracks the decode position, and one recording cannot prove that.
            // The first eligible window records a calibration tape; the next one
            // at a different position differences against it.
            let earlier = ctx
                .state
                .calibration()
                .map(|(snapshot, at)| (snapshot.clone(), at));
            let outcome = (|| -> Result<CaptureStep, CaptureFailure> {
                dflash_direct_verify_forward(
                    gpu,
                    target,
                    draft_tokens,
                    start_pos,
                    hidden_rb,
                    final_hidden,
                    gdn_tape,
                    pbs,
                    fusion,
                )
                .map_err(CaptureFailure::Forward)?;
                gpu.hip
                    .device_synchronize()
                    .map_err(CaptureFailure::Forward)?;
                gpu.replay.finish_capture().map_err(|reason| {
                    CaptureFailure::Record(format!("finish capture: {reason}"))
                })?;
                let Some((earlier_snapshot, earlier_position)) = earlier else {
                    return Ok(CaptureStep::Calibrated(
                        gpu.replay.snapshot_recorded_kernargs(),
                    ));
                };
                if start_pos <= earlier_position {
                    // Differencing needs a strictly later position; keep the
                    // fresher recording and wait for generation to advance.
                    return Ok(CaptureStep::Calibrated(
                        gpu.replay.snapshot_recorded_kernargs(),
                    ));
                }
                let position_bindings = gpu
                    .replay
                    .synthesize_position_bindings(&earlier_snapshot, earlier_position, start_pos)
                    .map_err(|reason| {
                        CaptureFailure::Contract(format!("position binding synthesis: {reason}"))
                    })?;
                let launches = gpu.replay.recorded_launches().len();
                gpu.replay
                    .probe_aql_contracts(gpu.device_id as usize)
                    .map_err(|reason| {
                        CaptureFailure::Contract(format!("AQL contract probe: {reason}"))
                    })?;
                gpu.replay.set_prepared_max_position(max_position);
                gpu.replay
                    .prepare_pm4_prefix_calibrated(gpu.device_id as usize, launches)
                    .map_err(|reason| CaptureFailure::Prepare(format!("PM4 prepare: {reason}")))?;
                gpu.replay
                    .prepared_route_identity()
                    .map(|identity| CaptureStep::Ready {
                        identity,
                        position_bindings,
                    })
                    .ok_or_else(|| {
                        CaptureFailure::Prepare(
                            "prepared PM4 route reported no identity".to_string(),
                        )
                    })
            })();
            swap_retained_controller(gpu, ctx.state)?;
            match outcome {
                Ok(CaptureStep::Calibrated(snapshot)) => {
                    ctx.state.note_calibration_capture(snapshot, start_pos);
                    Ok(())
                }
                Ok(CaptureStep::Ready {
                    identity,
                    position_bindings,
                }) => {
                    ctx.state.note_capture();
                    ctx.state.note_position_bindings(position_bindings);
                    ctx.state.note_ready(ctx.binding.clone(), identity);
                    Ok(())
                }
                Err(CaptureFailure::Forward(error)) => {
                    // The body failed mid-capture: KV/DeltaNet/staging may be
                    // half-written, so there is nothing to salvage here.
                    ctx.state
                        .poison(format!("DFlash retained capture body failed: {error:?}"));
                    Err(error)
                }
                Err(CaptureFailure::Record(reason)) => {
                    ctx.state.poison(reason);
                    Ok(())
                }
                Err(CaptureFailure::Contract(reason)) => {
                    ctx.state.note_capture();
                    ctx.state.note_contract_failure(reason);
                    Ok(())
                }
                Err(CaptureFailure::Prepare(reason)) => {
                    ctx.state.note_capture();
                    ctx.state.note_prepare_failure(reason);
                    Ok(())
                }
            }
        }
        DflashVerifyRoute::Pm4 => {
            swap_retained_controller(gpu, ctx.state)?;
            let replayed = unsafe { gpu.replay.replay_pm4_checked(start_pos) };
            swap_retained_controller(gpu, ctx.state)?;
            match replayed {
                Ok(_) => {
                    ctx.state.note_replay_success(start_pos);
                    Ok(())
                }
                Err(failure) => {
                    let quiescence = failure.quiescence;
                    ctx.state
                        .note_replay_failure(start_pos, quiescence, failure.error.clone());
                    Err(retained_hip_error(&format!(
                        "DFlash retained PM4 replay failed at position {start_pos} \
                         (quiescence {quiescence:?}): {}",
                        failure.error
                    )))
                }
            }
        }
    }
}

/// Swap the dedicated retained controller in or out of `gpu.replay`.
///
/// The ordinary-AR controller must never record or replay the DFlash tape, and
/// the DFlash controller must never own the AR route, so the swap is always
/// paired around exactly the recorded or replayed body.
fn swap_retained_controller(gpu: &mut Gpu, state: &mut DflashVerifyPm4) -> HipResult<()> {
    let controller = state.controller_mut().ok_or_else(|| {
        retained_hip_error("retained DFlash verify route has no dedicated controller")
    })?;
    std::mem::swap(&mut gpu.replay, controller);
    Ok(())
}

/// Retained-route entry point for the fixed B=16 DFlash2 chain verify.
///
/// Plans the route for this window, then defers to the ordinary
/// [`verify_dflash_block_inner`] so lm-head, argmax, and the staging commit run
/// exactly once and in their existing order regardless of which forward ran.
#[allow(clippy::too_many_arguments)]
pub fn verify_dflash_block_retained(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    verify_scratch: &VerifyScratch,
    route: &mut DflashVerifyPm4,
) -> HipResult<DflashVerifyOutput> {
    let b = draft_tokens.len();
    let binding = build_dflash_verify_binding(
        b,
        gpu.arch.as_str(),
        target,
        hidden_rb,
        verify_scratch,
        gdn_tape.as_deref(),
    );
    let selected = {
        let window = DflashVerifyWindow {
            batch: b,
            tree: false,
            want_full_logits,
            position: start_pos,
            binding: &binding,
        };
        route.plan_route(&window)
    };
    let mut ctx = RetainedCtx {
        state: route,
        binding,
        selected,
    };
    verify_dflash_block_inner(
        gpu,
        target,
        draft_tokens,
        start_pos,
        hidden_rb,
        gdn_tape,
        want_full_logits,
        None,
        verify_scratch,
        false,
        Some(&mut ctx),
    )
}

/// Prompt-only hidden scatter. Each pitched copy preserves the legacy byte
/// mapping, splitting at both source and destination ring boundaries. Producers
/// and consumers run on the same execution stream; no arithmetic or ring
/// bookkeeping changes. Decode/terminal-repair scatters keep their old route.
#[allow(clippy::too_many_arguments)]
pub fn scatter_hidden_prefill_to_interleaved(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    dst: &GpuTensor,
    dst_row_offset: usize,
    block_size: usize,
    n_rows: usize,
    dst_modulus: usize,
) -> HipResult<()> {
    static FAST: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_bool("HIPFIRE_DFLASH_PREFILL_SCATTER_FAST", true)
    });
    if !*FAST {
        return scatter_hidden_block_to_interleaved(
            gpu,
            hidden_rb,
            dst,
            dst_row_offset,
            block_size,
            n_rows,
            dst_modulus,
        );
    }
    assert!(n_rows <= block_size, "scatter: n_rows > block_size");
    let max_pos = hidden_rb.max_positions;
    assert!(
        block_size <= hidden_rb.written.max(max_pos),
        "scatter: block_size > written"
    );
    let num_extract = hidden_rb.extract_layers.len();
    let row_bytes = hidden_rb.hidden_dim * 4;
    let r_skip = block_size.saturating_sub(max_pos);
    let start_slot = (hidden_rb.head + max_pos - (block_size - r_skip)) % max_pos;
    let mut r = r_skip;
    while r < n_rows {
        let slot = (start_slot + r - r_skip) % max_pos;
        let absolute_dst = dst_row_offset + r;
        let (dst_row, dst_remaining) = if dst_modulus == usize::MAX {
            (absolute_dst, n_rows - r)
        } else {
            let row = absolute_dst % dst_modulus;
            (row, dst_modulus - row)
        };
        let rows = (n_rows - r).min(max_pos - slot).min(dst_remaining);
        for ext in 0..num_extract {
            gpu.memcpy_dtod_2d_at_ordered_async(
                &dst.buf,
                (dst_row * num_extract + ext) * row_bytes,
                num_extract * row_bytes,
                &hidden_rb.layer_bufs[ext].buf,
                slot * row_bytes,
                row_bytes,
                row_bytes,
                rows,
            )?;
        }
        r += rows;
    }
    Ok(())
}

/// Download extracted target hidden states for the most recent B positions
/// from `hidden_rb` and concat them into a flat `[B × num_extract × hidden]`
/// host vector in the order expected by `dflash::draft_forward` (per-position,
/// then per-extract-layer).
///
/// Caller typically slices this by `[0..accept_len+1]` of the position
/// dimension when appending to the cumulative target_hidden buffer used
/// by subsequent draft forwards.
///
/// Partial-download path (2026-04-16): downloads only the B most recent
/// rows per layer via `memcpy_dtoh` of the exact slice needed, handling
/// the ring-buffer wrap as two segments when necessary. Prior version
/// downloaded the full `max_pos × hidden` per layer (~170 MB at ctx=2048
/// × hidden=4096 × 5 layers); this cuts per-cycle D2H to the useful
/// `B × hidden × 5 × 4` bytes (~1.3 MB). For a math prompt at ctx=1024
/// this saves ~7 ms/cycle of PCIe + sync overhead.
/// GPU-side scatter of the FIRST `n_rows` rows of the most recently written
/// `block_size` slots of `hidden_rb` into a flat dst tensor laid out as
/// `[max_ctx × num_extract × hidden]` (interleaved per-position).
///
/// Semantics mirror `download_hidden_block(b)` followed by a slice of the
/// first `rows_to_keep * num_extract * hidden` f32s — the spec_step caller
/// pattern. The ring has just been advanced by `block_size` (by a verify
/// forward); the slots written in THAT verify occupy
/// `[head − block_size, head)` (mod max_pos). We copy the first `n_rows`
/// of those to rows `[dst_row_offset, dst_row_offset + n_rows)` of dst.
///
/// For each row r in 0..n_rows:
/// - ring slot = (head − block_size + r) mod max_pos
/// - dst row   = (dst_row_offset + r)
/// - For each extract layer `ext`: D2D copy hidden×4 bytes from
///   `hidden_rb.layer_bufs[ext][slot × hidden ..]` to
///   `dst[(dst_row × num_extract + ext) × hidden ..]`.
///
/// When called after `seed_target_hidden_from_prompt` (no prior block), the
/// caller passes `block_size = n_rows = prompt_len` and `dst_row_offset = 0`.
///
/// Replaces the previous D2H-then-H2D roundtrip via `target_hidden_host`
/// Vec<f32> + `draft_forward`'s upload for the common ctx_slice=None path.
/// Eliminates 5 blocking D2H sync points per spec step (one per extract
/// layer) and the follow-on per-cycle H2D upload; remains about 80 small
/// async D2D enqueues per cycle (ne × n_rows ≤ 5 × 16), which the stream
/// dispatcher handles in ~200 µs of CPU time with zero cross-device waits.
pub fn scatter_hidden_block_to_interleaved(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    dst: &GpuTensor,
    dst_row_offset: usize,
    block_size: usize,
    n_rows: usize,
    dst_modulus: usize,
) -> HipResult<()> {
    assert!(
        n_rows <= block_size,
        "scatter: n_rows {n_rows} > block_size {block_size}"
    );
    let num_extract = hidden_rb.extract_layers.len();
    let hidden = hidden_rb.hidden_dim;
    let max_pos = hidden_rb.max_positions;
    let head = hidden_rb.head;
    let written = hidden_rb.written;
    assert!(
        block_size <= written.max(max_pos),
        "scatter: block_size {block_size} > written {written}"
    );
    let row_bytes = hidden * 4;
    // The ring physically holds only the last `max_pos` rows of a logical
    // block longer than that — skip the fallen-off prefix (windowed draft
    // mode: SWA layers never read out-of-window rows; the last layer's
    // long-reach fill is the post-seed backfill's job). Legacy callers pass
    // block_size <= max_pos ⇒ r_skip = 0, identical behaviour.
    let r_skip = block_size.saturating_sub(max_pos);
    let start_slot = (head + max_pos - (block_size - r_skip)) % max_pos;
    // S2 launch fusion: pure-copy scatter5 kernel (exact gfx1100/gfx1151/gfx1201). Copies the retained
    // block rows into dst[((dst_row_offset + r) % dst_modulus), ext, :] in
    // one launch (bit-identical: one writer per destination element, no FP
    // arithmetic on the data; usize::MAX keeps absolute addressing). The
    // symbol is ensured by the same-cycle fused commit, which strictly
    // precedes every fused scatter; without it (seed paths, non-admitted arch,
    // kill switch, capture/recording, funny shapes) the launcher reports
    // false and the loop below runs byte-for-byte as before. Never mutates
    // head/written or the source ring.
    if gpu.dflash_hidden_scatter5_try(
        &hidden_rb.layer_bufs,
        dst,
        start_slot,
        n_rows,
        r_skip,
        hidden,
        max_pos,
        dst_row_offset,
        dst_modulus,
        num_extract,
    )? {
        return Ok(());
    }

    for r in r_skip..n_rows {
        let slot = (start_slot + (r - r_skip)) % max_pos;
        // dst_row_offset is an absolute row index; windowed-mode dst rings
        // address it by slot (row % modulus). Legacy passes usize::MAX.
        let dst_row = if dst_modulus == usize::MAX {
            dst_row_offset + r
        } else {
            (dst_row_offset + r) % dst_modulus
        };
        let dst_row_base_bytes = dst_row * num_extract * row_bytes;
        for ext in 0..num_extract {
            let src_offset_bytes = slot * row_bytes;
            let dst_offset_bytes = dst_row_base_bytes + ext * row_bytes;
            gpu.hip.memcpy_dtod_at(
                &dst.buf,
                dst_offset_bytes,
                &hidden_rb.layer_bufs[ext].buf,
                src_offset_bytes,
                row_bytes,
            )?;
        }
    }
    Ok(())
}

/// Append a cold prefill piece only when a caller actually needs its host
/// shadow. The paired all-sliding speculator has no host reader: it retains
/// these same rows in `hidden_rb` and scatters the final ring on device.
pub fn append_prefill_hidden_host(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    rows: usize,
    host: Option<&mut Vec<f32>>,
) -> HipResult<()> {
    if let Some(host) = host {
        let block = download_hidden_block(gpu, hidden_rb, rows)?;
        host.extend_from_slice(&block);
    }
    Ok(())
}

pub fn download_hidden_block(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    b: usize,
) -> HipResult<Vec<f32>> {
    let num_extract = hidden_rb.extract_layers.len();
    let hidden = hidden_rb.hidden_dim;
    let max_pos = hidden_rb.max_positions;
    let written = hidden_rb.written;

    // Figure out which ring positions hold the most recent B writes.
    // `head` points to where the NEXT write will land. After B advances,
    // the most recent B sit at ring slots (head - B) mod max_pos ..
    // (head - 1) mod max_pos.
    assert!(
        b <= written,
        "verify must have written at least B rows to ring buffer"
    );
    let head = hidden_rb.head;
    let start_slot = (head + max_pos - b) % max_pos;
    let row_bytes = hidden * 4;

    // Per layer, download only the B needed rows (not the full ring).
    // If start_slot + b <= max_pos: one contiguous segment.
    // Otherwise: two segments (head→end + 0→tail).
    //
    // Each layer's B-row slice lands at [ext × b × hidden] in layer_data_flat.
    let mut layer_data_flat = vec![0f32; num_extract * b * hidden];
    for ext in 0..num_extract {
        let src_buf = &hidden_rb.layer_bufs[ext].buf;
        let dst_offset_floats = ext * b * hidden;
        if start_slot + b <= max_pos {
            // Single contiguous copy.
            let dst_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    layer_data_flat.as_mut_ptr().add(dst_offset_floats) as *mut u8,
                    b * row_bytes,
                )
            };
            gpu.hip
                .memcpy_dtoh_at(dst_bytes, src_buf, start_slot * row_bytes)?;
        } else {
            // Two-segment ring wrap: tail of buffer, then head.
            let first_rows = max_pos - start_slot;
            let second_rows = b - first_rows;
            let dst_first_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    layer_data_flat.as_mut_ptr().add(dst_offset_floats) as *mut u8,
                    first_rows * row_bytes,
                )
            };
            gpu.hip
                .memcpy_dtoh_at(dst_first_bytes, src_buf, start_slot * row_bytes)?;
            let dst_second_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    layer_data_flat
                        .as_mut_ptr()
                        .add(dst_offset_floats + first_rows * hidden)
                        as *mut u8,
                    second_rows * row_bytes,
                )
            };
            gpu.hip.memcpy_dtoh_at(dst_second_bytes, src_buf, 0)?;
        }
    }

    // Rearrange into per-position-then-per-extract-layer order.
    // layer_data_flat is [ext × b × hidden]; we want [b × ext × hidden].
    let mut out: Vec<f32> = Vec::with_capacity(b * num_extract * hidden);
    for pi in 0..b {
        for ext in 0..num_extract {
            let src_off = (ext * b + pi) * hidden;
            out.extend_from_slice(&layer_data_flat[src_off..src_off + hidden]);
        }
    }

    debug_assert_eq!(out.len(), b * num_extract * hidden);
    Ok(out)
}

/// S7: batch the draft noise embeddings into a single launch.
///
/// Uploads the `block` token IDs once into the persistent `noise_tokens`
/// plane (i32 IDs stored as F32 bits, same cosmetic pattern as the
/// `positions_*` planes) and runs one `embedding_lookup_q8_batched` over
/// all `b` rows directly into `draft_scratch.x` ([b*h]).
///
/// Returns `true` when the fast path ran. Returns `false` — leaving every
/// buffer untouched — when the route predicates fail, in which case the
/// caller runs the legacy per-token loop. Route: Q8_0 target embedding,
/// `1 <= b <= max_block_size`, and either exact gfx1100 with
/// `HIPFIRE_DRAFT_COLLAPSE_OFF` unset or the gfx1201 batched noise lookup
/// ([`noise_embed_batch_admitted`], kill switch
/// `HIPFIRE_DRAFT_NOISE_EMBED_BATCH=0`) that the lane-batched draft already
/// uses. The batched kernel dequantizes each row with the same per-element
/// math as the scalar loop, so the plane is bit-identical.
///
/// Weights-based core shared with the dense-TP mesh (which owns rank shards,
/// not a `ModelSlot`).
pub fn build_dflash_noise_embeddings_weights(
    gpu: &mut Gpu,
    target_weights: &Qwen35Weights,
    block: &[u32],
    h: usize,
    draft_scratch: &mut DflashScratch,
) -> HipResult<bool> {
    if !matches!(
        target_weights.embd_format,
        hipfire_runtime::llama::EmbeddingFormat::Q8_0
    ) {
        return Ok(false);
    }
    let b = block.len();
    if b == 0 || b > draft_scratch.max_block_size {
        return Ok(false);
    }
    if !gpu.draft_collapse_fused_enabled()
        && !noise_embed_batch_admitted(gpu, target_weights, draft_scratch, b)
    {
        return Ok(false);
    }
    let ids: Vec<i32> = block.iter().map(|&t| t as i32).collect();
    let id_view = draft_scratch.noise_tokens.sub_offset(0, b);
    let id_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, ids.len() * 4) };
    gpu.hip.memcpy_htod(&id_view.buf, id_bytes)?;
    let out_view = draft_scratch.x.sub_offset(0, b * h);
    gpu.embedding_lookup_q8_batched(&target_weights.token_embd, &out_view, &id_view, b, h)?;
    Ok(true)
}

pub fn build_dflash_noise_embeddings(
    gpu: &mut Gpu,
    target: &ModelSlot,
    block: &[u32],
    h: usize,
    draft_scratch: &mut DflashScratch,
) -> HipResult<bool> {
    build_dflash_noise_embeddings_weights(gpu, &target.weights, block, h, draft_scratch)
}

/// Sampled selector draft from singleton lines 6136–6346 / 6460–6500.
/// The shared parts body runs the identical forward and Product head, draws
/// all b-1 host uniforms before scoring, then materializes actual selector q.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn dflash_sampled_selector_draft_into(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    df: &mut DflashState,
    position: usize,
    compact_offset: i32,
    seed: u32,
    b: usize,
    vocab: usize,
    cfg: DflashSampleCfg,
    rng: &mut u64,
    out: &mut DflashSampledDraft,
) -> HipResult<()> {
    dflash_sampled_selector_draft_parts_into(
        gpu,
        weights,
        &df.draft_weights,
        &df.draft_config,
        &mut df.draft_scratch,
        &df.verify_scratch,
        &[],
        &df.verify_scratch.logits.sub_offset(0, vocab),
        position,
        compact_offset,
        seed,
        b,
        vocab,
        None,
        false,
        cfg,
        rng,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
fn dflash_sampled_selector_draft_parts_into(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    verify_scratch: &VerifyScratch,
    target_hidden_host: &[f32],
    single_logits: &GpuTensor,
    position: usize,
    compact_offset: i32,
    seed: u32,
    b: usize,
    vocab: usize,
    ctx_slice: Option<usize>,
    draft_ffn_graph: bool,
    cfg: DflashSampleCfg,
    rng: &mut u64,
    out: &mut DflashSampledDraft,
) -> HipResult<()> {
    if b < 2 || cfg.temp <= 0.0 || !draft_weights.has_candidate_selector() {
        return Err(HipError::new(
            0,
            "sampled DFlash draft requires b>=2, temp>0 and selector",
        ));
    }
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    out.block.clear();
    out.block.resize(b, draft_cfg.mask_token_id);
    out.block[0] = seed;
    draft_dflash_forward_rank(
        gpu,
        weights,
        draft_weights,
        draft_cfg,
        draft_scratch,
        target_hidden_host,
        &out.block,
        position,
        compact_offset,
        ctx_slice,
        draft_ffn_graph,
    )?;
    let batch = b - 1;
    let h = draft_cfg.hidden;
    let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
    out.uniforms.clear();
    let proposal = if dflash_batched_lm_head_supported(weights.output.gpu_dtype) {
        assert!(
            batch <= verify_scratch.max_n,
            "verify_scratch max_n {} < draft batch {}",
            verify_scratch.max_n,
            batch,
        );
        dflash_enqueue_verify_lm_head(
            gpu,
            &weights.output,
            &hidden_rows,
            verify_scratch,
            batch,
            vocab,
            DenseBatchMath::Product,
        )?;
        let logits_batch = verify_scratch.logits.sub_offset(0, batch * vocab);
        return dflash_sampled_selector_rows_into(
            gpu,
            draft_weights,
            draft_scratch,
            &hidden_rows,
            &logits_batch,
            seed,
            batch,
            vocab,
            cfg,
            rng,
            out,
        );
    } else {
        let mut host_logits = Vec::with_capacity(batch * vocab);
        for i in 1..b {
            let hidden_row = draft_scratch.x.sub_offset(i * h, h);
            llama::weight_gemv(gpu, &weights.output, &hidden_row, single_logits)?;
            let logits = gpu.download_f32(single_logits)?;
            debug_assert_eq!(logits.len(), vocab);
            host_logits.extend_from_slice(&logits);
        }
        out.uniforms
            .extend((0..batch).map(|_| xorshift_next_unit(rng)));
        dflash::propose_candidates_host(
            gpu,
            draft_weights,
            draft_scratch,
            &hidden_rows,
            &host_logits,
            batch,
            seed,
            cfg.temp,
            Some(&out.uniforms),
        )?
    };
    dflash_sampled_apply_proposal_into(proposal, vocab, cfg, out)
}

#[allow(clippy::too_many_arguments)]
fn dflash_sampled_selector_rows_into(
    gpu: &mut Gpu,
    draft_weights: &DflashWeights,
    draft_scratch: &mut DflashScratch,
    hidden_rows: &GpuTensor,
    logits_rows: &GpuTensor,
    seed: u32,
    rows: usize,
    vocab: usize,
    cfg: DflashSampleCfg,
    rng: &mut u64,
    out: &mut DflashSampledDraft,
) -> HipResult<()> {
    out.uniforms.clear();
    out.uniforms
        .extend((0..rows).map(|_| xorshift_next_unit(rng)));
    let proposal = dflash::propose_candidates_device(
        gpu,
        draft_weights,
        draft_scratch,
        hidden_rows,
        logits_rows,
        rows,
        seed,
        cfg.temp,
        Some(&out.uniforms),
    )?;
    dflash_sampled_apply_proposal_into(proposal, vocab, cfg, out)
}

fn dflash_sampled_apply_proposal_into(
    proposal: dflash::DflashCandidateProposal,
    vocab: usize,
    cfg: DflashSampleCfg,
    out: &mut DflashSampledDraft,
) -> HipResult<()> {
    out.block.truncate(1);
    // Same switch as the target rows (`dflash_target_rows_into`): sparse q is
    // the exact (id, value) support of the dense row below, minus its vocab fill.
    out.sparse = dflash_sparse_target_enabled(&cfg);
    if out.sparse {
        out.sparse_q.clear();
        apply_dflash2_selector_proposal_sparse(
            &proposal,
            vocab,
            &mut out.block,
            &mut out.sparse_q,
            &mut out.selected_q,
        )?;
    } else {
        apply_dflash2_selector_proposal(
            &proposal,
            vocab,
            true,
            &mut out.block,
            &mut out.dense_q,
            &mut out.selected_q,
        )?;
    }
    out.proposal = Some(proposal);
    Ok(())
}

/// Rank-local DFlash draft forward prefix shared by the single-GPU step and
/// the dense-TP mesh: embeds `block` into `draft_scratch.x` (S7 batched Q8
/// path or the scalar per-token loop, bit-identical), builds absolute Q/K
/// positions from the draft's target-hidden log, and runs `draft_forward_opts`
/// (forward `None` target_hidden = GPU-resident fast path unless `ctx_slice`
/// selects the CPU-shadow window). Leaves draft hidden rows in
/// `draft_scratch.x`; the caller applies its LM head. Advances no target KV
/// or DeltaNet state — only the draft scratch.
#[allow(clippy::too_many_arguments)]
fn draft_dflash_forward_rank(
    gpu: &mut Gpu,
    target_weights: &Qwen35Weights,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    target_hidden_host: &[f32],
    block: &[u32],
    position: usize,
    compact_offset: i32,
    ctx_slice: Option<usize>,
    draft_ffn_graph: bool,
) -> HipResult<()> {
    let b = block.len();
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    if !build_dflash_noise_embeddings_weights(gpu, target_weights, block, h, draft_scratch)? {
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
                llama::EmbeddingFormat::F32 => {
                    gpu.embedding_lookup(&target_weights.token_embd, &dst, tok, h)?
                }
                _ => panic!("dflash: unsupported target embedding format for noise lookup"),
            }
        }
    }
    let effective_ctx_len = match ctx_slice {
        Some(n) => n.min(position),
        None => draft_scratch.thlog.abs_positions().len().min(position),
    };
    let ctx_start = position - effective_ctx_len;
    let co = compact_offset;
    let positions_q: Vec<i32> =
        ((position as i32 + co)..(position as i32 + b as i32 + co)).collect();
    let positions_k: Vec<i32> = if ctx_slice.is_some() {
        (ctx_start as i32..(position + b) as i32).collect()
    } else {
        let mut v = Vec::with_capacity(effective_ctx_len + b);
        let th_abs = draft_scratch.thlog.abs_positions();
        let start_idx = th_abs.len().saturating_sub(effective_ctx_len);
        v.extend_from_slice(&th_abs[start_idx..]);
        for p in 0..b {
            v.push(position as i32 + p as i32 + co);
        }
        v
    };
    let (th_arg, _th_offset): (Option<&[f32]>, usize) = if ctx_slice.is_some() {
        let th_offset = ctx_start * ne * h;
        (Some(&target_hidden_host[th_offset..]), th_offset)
    } else {
        (None, 0)
    };
    dflash::draft_forward_opts(
        gpu,
        draft_weights,
        draft_cfg,
        None,
        th_arg,
        &positions_q,
        &positions_k,
        b,
        effective_ctx_len,
        draft_scratch,
        draft_ffn_graph,
    )?;
    Ok(())
}

/// The batched LM-head GEMM of the draft (rows `1..B` of `draft_scratch.x`
/// into `logits`) for every `use_batched_gemm` weight dtype: the one launch
/// sequence shared by [`draft_dflash_block_rank`] and the planned draft
/// emitter (`dflash_draft_emit`). `hidden_rows` is `x[h..]`, `logits_batch`
/// the `[batch × vocab]` view of `verify_scratch.logits`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draft_lmhead_batched(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    verify_scratch: &VerifyScratch,
    hidden_rows: &GpuTensor,
    logits_batch: &GpuTensor,
    batch: usize,
    h: usize,
) -> HipResult<()> {
    match w_out.gpu_dtype {
        rdna_compute::DType::Q8_0 => {
            dflash_gemm_q8_lmhead(gpu, w_out, hidden_rows, logits_batch, batch)?;
        }
        rdna_compute::DType::HFQ4G256 => {
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                hidden_rows,
                logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )?;
        }
        rdna_compute::DType::MQ4G256 => {
            assert!(
                batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for draft lm_head"
            );
            let rotated = verify_scratch.rot.sub_offset(0, batch * h);
            llama::rotate_x_mq_batched_for(gpu, w_out, hidden_rows, &rotated, h, batch)?;
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )?;
        }
        rdna_compute::DType::MQ4G256V2
        | rdna_compute::DType::MQ6G256V2
        | rdna_compute::DType::MQ5G256V2
        | rdna_compute::DType::MQ3G256V2
        | rdna_compute::DType::MQ2G256V2 => {
            assert!(
                batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ v2 draft lm_head"
            );
            let rotated = verify_scratch.rot.sub_offset(0, batch * h);
            llama::rotate_x_mq_batched_for(gpu, w_out, hidden_rows, &rotated, h, batch)?;
            gemm_mq_batched_lmhead_draft(
                gpu,
                w_out.gpu_dtype,
                &w_out.buf,
                &rotated,
                logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )?;
        }
        rdna_compute::DType::MQ3G256 => {
            assert!(
                batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ3 draft lm_head"
            );
            let rotated = verify_scratch.rot.sub_offset(0, batch * h);
            llama::rotate_x_mq_batched_for(gpu, w_out, hidden_rows, &rotated, h, batch)?;
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )?;
        }
        rdna_compute::DType::HFQ6G256 => {
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                hidden_rows,
                logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )?;
        }
        rdna_compute::DType::MQ6G256 => {
            assert!(
                batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ6 draft lm_head"
            );
            let rotated = verify_scratch.rot.sub_offset(0, batch * h);
            llama::rotate_x_mq_batched_for(gpu, w_out, hidden_rows, &rotated, h, batch)?;
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Greedy rank-local draft production: the forward prefix above plus the
/// temp==0 LM-head leaves (batched GEMM + selector-device / GPU argmax, or
/// the non-batched selector-host / per-row argmax fallbacks). Returns
/// `drafted` with the seed at index 0. Greedy-only by contract (no `temp`,
/// repeat-penalty, n-gram, or logit-dump knobs); sampled/host-shaped paths
/// stay inline in `spec_step_dflash`. Called by the single-GPU greedy fast
/// path and by every dense-TP mesh rank.
#[allow(clippy::too_many_arguments)]
pub fn draft_dflash_block_rank(
    gpu: &mut Gpu,
    target_weights: &Qwen35Weights,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    verify_scratch: &VerifyScratch,
    target_hidden_host: &[f32],
    block: &[u32],
    position: usize,
    compact_offset: i32,
    ctx_slice: Option<usize>,
    draft_ffn_graph: bool,
    seed_token: u32,
    vocab: usize,
) -> HipResult<Vec<u32>> {
    let b = block.len();
    let h = draft_cfg.hidden;
    debug_assert_eq!(vocab, draft_cfg.vocab_size);
    draft_dflash_forward_rank(
        gpu,
        target_weights,
        draft_weights,
        draft_cfg,
        draft_scratch,
        target_hidden_host,
        block,
        position,
        compact_offset,
        ctx_slice,
        draft_ffn_graph,
    )?;
    let mut drafted: Vec<u32> = vec![seed_token];
    let w_out = &target_weights.output;
    let use_batched_gemm = matches!(
        w_out.gpu_dtype,
        rdna_compute::DType::Q8_0
            | rdna_compute::DType::HFQ4G256
            | rdna_compute::DType::MQ4G256
            | rdna_compute::DType::MQ4G256V2
            | rdna_compute::DType::MQ6G256V2
            | rdna_compute::DType::MQ5G256V2
            | rdna_compute::DType::MQ3G256V2
            | rdna_compute::DType::MQ2G256V2
            | rdna_compute::DType::MQ3G256
            | rdna_compute::DType::HFQ6G256
            | rdna_compute::DType::MQ6G256,
    );
    let selector_mode = draft_weights.has_candidate_selector();
    let batch = b - 1;
    if use_batched_gemm {
        assert!(
            batch <= verify_scratch.max_n,
            "verify_scratch max_n {} < draft batch {}",
            verify_scratch.max_n,
            batch
        );
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
        if selector_mode {
            let proposal = dflash::propose_candidates_device(
                gpu,
                draft_weights,
                draft_scratch,
                &hidden_rows,
                &logits_batch,
                batch,
                seed_token,
                0.0,
                None,
            )?;
            apply_dflash2_selector_proposal(
                &proposal,
                vocab,
                false,
                &mut drafted,
                &mut Vec::new(),
                &mut Vec::new(),
            )?;
        } else if let Some(tuner) = draft_scratch.online.as_mut() {
            drafted.extend(tuner.propose_from_logits(
                gpu,
                &logits_batch,
                vocab,
                batch,
                position,
                seed_token,
            )?);
            if tuner.mode == hipfire_runtime::dflash_online::Mode::Sweep {
                // Always rejected: every cycle commits exactly one target token.
                drafted[1..].fill(vocab as u32 - 1);
            }
        } else {
            // GPU argmax over (B-1) rows — one kernel, small D2H.
            let argmax_buf = verify_scratch.argmax.sub_offset(0, batch);
            gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, batch)?;
            let mut host_idx = vec![0i32; batch];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, batch * 4)
                };
                gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf)?;
            }
            for &idx in &host_idx {
                drafted.push(idx as u32);
            }
        }
    } else if selector_mode {
        // lm-head fallback: still go through propose_candidates_host so
        // greedy is not an independent argmax and sampled q stays sparse.
        let mut host_logits = Vec::with_capacity(batch * vocab);
        let row_logits = verify_scratch.logits.sub_offset(0, vocab);
        for i in 1..b {
            let hidden_row = draft_scratch.x.sub_offset(i * h, h);
            llama::weight_gemv(gpu, w_out, &hidden_row, &row_logits)?;
            let logits = gpu.download_f32(&row_logits)?;
            debug_assert_eq!(logits.len(), vocab);
            host_logits.extend_from_slice(&logits);
        }
        let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
        let proposal = dflash::propose_candidates_host(
            gpu,
            draft_weights,
            draft_scratch,
            &hidden_rows,
            &host_logits,
            batch,
            seed_token,
            0.0,
            None,
        )?;
        apply_dflash2_selector_proposal(
            &proposal,
            vocab,
            false,
            &mut drafted,
            &mut Vec::new(),
            &mut Vec::new(),
        )?;
    } else {
        // Fallback: per-row weight_gemv loop, greedy argmax.
        let row_logits = verify_scratch.logits.sub_offset(0, vocab);
        for i in 1..b {
            let hidden_row = draft_scratch.x.sub_offset(i * h, h);
            llama::weight_gemv(gpu, w_out, &hidden_row, &row_logits)?;
            let logits = gpu.download_f32(&row_logits)?;
            debug_assert_eq!(logits.len(), vocab);
            drafted.push(argmax_u32(&logits));
        }
    }
    Ok(drafted)
}

/// Rows one batched draft forward holds by default. The draft projections /
/// FFN / head GEMMs are row-count independent on gfx1201 below the 64-row
/// route boundary (the exact-trunk ceiling is 63); 48 keeps the draft on the
/// rungs the exact trunk and the oracle already cover (three full 16-row
/// lanes). [`dflash_draft_batch_max_rows`] widens it to
/// [`DFLASH_DRAFT_BATCH_ROWS64`] where that stays exact.
pub const DFLASH_DRAFT_BATCH_MAX_ROWS: usize = 48;

/// One full 64-row draft chunk (C8 at 8-row lanes in a single forward).
pub const DFLASH_DRAFT_BATCH_ROWS64: usize = 64;

/// `HIPFIRE_CB_DRAFT_ROWS64=0` (developer kill switch): keep the batched
/// draft at [`DFLASH_DRAFT_BATCH_MAX_ROWS`] rows per forward. Default on.
static DRAFT_ROWS64_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_CB_DRAFT_ROWS64").map_or(true, |v| v.trim() != "0")
});

/// `HIPFIRE_DFLASH_SELECTOR_LANES=0` (developer kill switch): score the
/// batched draft's candidate selector per lane (one projection / top-K /
/// three host syncs each) instead of once per chunk through
/// [`dflash::propose_candidates_lanes_device`]. Default on.
static SELECTOR_LANES_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_DFLASH_SELECTOR_LANES").map_or(true, |v| v.trim() != "0")
});

/// Whether the lane-batched selector applies: switch on, exact gfx1201 with
/// the draft-collapse overwrite route enabled (the only route whose
/// projection is per-row identical across row counts), and the shared draft
/// scratch owns the FWHT rotation buffer the projection rotates through.
fn selector_lanes_batched(gpu: &Gpu, shared: &DflashScratch) -> bool {
    *SELECTOR_LANES_ON
        && gpu.arch == "gfx1201"
        && !gpu.flags.draft_collapse_off
        && shared.mq_x_rot.is_some()
        && shared.mq_x_rot_f16.is_some()
}

/// `HIPFIRE_DRAFT_NOISE_EMBED_BATCH=0` (developer kill switch): keep the
/// per-row `embedding_lookup_q8` loop for the batched draft's noise rows
/// (64 launches at C8) instead of one `embedding_lookup_q8_batched` launch.
/// Default on; every row byte is identical either way.
static DRAFT_NOISE_EMBED_BATCH_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_DRAFT_NOISE_EMBED_BATCH")
        .map_or(true, |v| v.trim() != "0")
});

/// Whether the batched draft's noise rows come from one batched Q8_0
/// lookup: switch on, exact gfx1201, Q8_0 target embedding, and `rows` fit
/// the shared `noise_tokens` plane (`max_block_size` ids). Independent of
/// `draft_collapse_fused_enabled` (gfx1100 only). Exact: `embedding_q8` and
/// `embedding_q8_batched` run the same per-element expression
/// `(float)half(d) * (float)int8(q)` with the same 256-thread element
/// striding — one block per row instead of one launch per row — so each
/// row's bytes cannot differ.
pub(crate) fn noise_embed_batch_admitted(
    gpu: &Gpu,
    target_weights: &Qwen35Weights,
    shared: &DflashScratch,
    rows: usize,
) -> bool {
    *DRAFT_NOISE_EMBED_BATCH_ON
        && gpu.arch == "gfx1201"
        && matches!(target_weights.embd_format, llama::EmbeddingFormat::Q8_0)
        && rows >= 1
        && rows <= shared.max_block_size
}

/// Rows one batched draft forward holds on this GPU: 64 when the 64-row chunk
/// is exact, else [`DFLASH_DRAFT_BATCH_MAX_ROWS`]. The 64-row chunk is exact
/// only through the scoped overwrite route
/// (`Gpu::gemm_mq4g256v2_overwrite_gfx12_dflash` inside
/// `Gpu::draft_rows64_enter`), whose `_vt4` kernel is per-row identical to the
/// one-tile kernel; the legacy `>= 64` arms (A8 / IU4 / bt-tile / fp8) are
/// not. It therefore needs: kill switch unset, exact gfx1201, the draft
/// collapse routes enabled, the batch-tile kernels, the F16 / WMMA lm_head
/// chain with no LDS-staged arm (`mq4_verify_singleton_chain`) and no A8
/// prefill route at 64 rows.
pub fn dflash_draft_batch_max_rows(gpu: &Gpu) -> usize {
    if *DRAFT_ROWS64_ON
        && gpu.arch == "gfx1201"
        && !gpu.flags.draft_collapse_off
        && gpu.flags.wmma_batch_tiles
        && !gpu.flags.fp16_disabled
        && !gpu.flags.lm_head_wmma_disabled
        && !gpu.flags.hfq4g256_ldsstage_wmma
        && !gpu.a8_prefill_active(DFLASH_DRAFT_BATCH_ROWS64, 256)
    {
        DFLASH_DRAFT_BATCH_ROWS64
    } else {
        DFLASH_DRAFT_BATCH_MAX_ROWS
    }
}

/// One lane of [`draft_dflash_blocks_batched`].
pub struct DflashDraftLane<'a> {
    pub df: &'a mut DflashState,
    /// Committed target position (`position` of the singleton step).
    pub position: usize,
    /// Pending seed (`block[0]`).
    pub seed: u32,
    /// Block rows `b` (`3..`: one-row heads are GEMVs, a different kernel).
    pub b: usize,
    /// The lane's `kv_cache.compact_offset`.
    pub compact_offset: i32,
}

/// One sampled lane of [`draft_dflash_sampled_blocks_batched`].
pub struct DflashSampledDraftLane<'a> {
    pub draft: DflashDraftLane<'a>,
    pub cfg: DflashSampleCfg,
    pub rng: &'a mut u64,
    pub out: &'a mut DflashSampledDraft,
}

/// Share the admitted forward/head while retaining per-lane selector and RNG.
///
/// The caller must admit the normal F16 gfx1201 route before calling, with
/// no capture/recording/G0 or FP8/IU4/A8/LDS scope, and respect the existing
/// batch capacity and shared-asset/layout requirements. Every lane needs
/// positive temperature and a selector. `b=2` must remain singleton: its
/// one-candidate head and rank projection are GEMVs, not the batched WMMA.
/// Receipts and returned blocks follow the original input-lane order.
#[allow(clippy::too_many_arguments)]
pub fn draft_dflash_sampled_blocks_batched(
    gpu: &mut Gpu,
    target_weights: &Qwen35Weights,
    vocab: usize,
    shared: &mut DflashScratch,
    head: &VerifyScratch,
    lanes: &mut [DflashSampledDraftLane<'_>],
) -> HipResult<Vec<Vec<u32>>> {
    for (i, lane) in lanes.iter().enumerate() {
        if lane.draft.b < 3
            || lane.cfg.temp <= 0.0
            || !lane.draft.df.draft_weights.has_candidate_selector()
        {
            return Err(HipError::new(
                0,
                &format!(
                "draft_dflash_sampled_blocks_batched: lane {i} requires b>=3, temp>0 and selector"
            ),
            ));
        }
    }
    {
        let mut drafts: Vec<DflashDraftLane<'_>> = lanes
            .iter_mut()
            .map(|lane| DflashDraftLane {
                df: &mut *lane.draft.df,
                position: lane.draft.position,
                seed: lane.draft.seed,
                b: lane.draft.b,
                compact_offset: lane.draft.compact_offset,
            })
            .collect();
        draft_dflash_forward_head_batched(gpu, target_weights, vocab, shared, head, &mut drafts)?;
    }
    let mut blocks = Vec::with_capacity(lanes.len());
    let mut offset = 0usize;
    for lane in lanes.iter_mut() {
        let draft = &mut lane.draft;
        let rows = draft.b - 1;
        let h = draft.df.draft_config.hidden;
        let hidden_rows = shared.x.sub_offset((offset + 1) * h, rows * h);
        let logits_rows = head.logits.sub_offset((offset + 1) * vocab, rows * vocab);
        let scratch = &mut draft.df.draft_scratch;
        let lent = scratch.mq_x_rot.is_none();
        if lent {
            if shared.mq_x_rot.is_none() || shared.mq_x_rot_f16.is_none() {
                return Err(HipError::new(
                    0,
                    "the shared draft scratch has no FWHT rotation buffers for the lane selector",
                ));
            }
            scratch.mq_x_rot = shared.mq_x_rot.take();
            scratch.mq_x_rot_f16 = shared.mq_x_rot_f16.take();
        }
        lane.out.block.clear();
        lane.out.block.push(draft.seed);
        let result = dflash_sampled_selector_rows_into(
            gpu,
            &draft.df.draft_weights,
            scratch,
            &hidden_rows,
            &logits_rows,
            draft.seed,
            rows,
            vocab,
            lane.cfg,
            lane.rng,
            lane.out,
        );
        // Restore the shared buffers even when projection/scoring failed.
        if lent {
            shared.mq_x_rot = scratch.mq_x_rot.take();
            shared.mq_x_rot_f16 = scratch.mq_x_rot_f16.take();
        }
        result?;
        blocks.push(lane.out.block.clone());
        offset += draft.b;
    }
    Ok(blocks)
}

/// Whether the batched draft is exact for this machine / draft / target head:
/// the non-fused (non-gfx1100) draft route, MoE-free target, every draft
/// matmul an MQ4 v2 (or, for the DFlash2 convolution kernel projections,
/// F16/F32) weight and the target head MQ4 v2 — the families whose WMMA GEMMs
/// are row-independent below 64 rows on gfx1201 (verified per lane against
/// the singleton by the CB state oracle).
pub fn dflash_draft_batch_eligible(
    gpu: &Gpu,
    target_weights: &Qwen35Weights,
    config: &Qwen35Config,
    draft_weights: &DflashWeights,
) -> bool {
    use rdna_compute::DType::{F16, F32, MQ4G256V2};
    if config.num_experts > 0
        || gpu.draft_collapse_fused_enabled()
        || !gpu.arch.starts_with("gfx12")
    {
        return false;
    }
    if target_weights.output.gpu_dtype != MQ4G256V2 {
        return false;
    }
    let main_ok = |w: &llama::WeightTensor| w.gpu_dtype == MQ4G256V2;
    let conv_ok = |w: &llama::WeightTensor| matches!(w.gpu_dtype, MQ4G256V2 | F16 | F32);
    main_ok(&draft_weights.fc)
        && draft_weights.layers.iter().all(|l| {
            main_ok(&l.wq)
                && main_ok(&l.wk)
                && main_ok(&l.wv)
                && main_ok(&l.wo)
                && main_ok(&l.w_gate)
                && main_ok(&l.w_up)
                && main_ok(&l.w_down)
                && l.attn_conv_proj.as_ref().is_none_or(conv_ok)
                && l.mlp_conv_proj.as_ref().is_none_or(conv_ok)
        })
}

#[allow(clippy::too_many_arguments)]
fn draft_dflash_forward_head_batched(
    gpu: &mut Gpu,
    target_weights: &Qwen35Weights,
    vocab: usize,
    shared: &mut DflashScratch,
    head: &VerifyScratch,
    lanes: &mut [DflashDraftLane<'_>],
) -> HipResult<bool> {
    let err = |m: String| {
        Err(HipError::new(
            0,
            &format!("draft_dflash_blocks_batched: {m}"),
        ))
    };
    if lanes.is_empty() {
        return Ok(false);
    }
    let rows: usize = lanes.iter().map(|l| l.b).sum();
    let cap = dflash_draft_batch_max_rows(gpu);
    if rows > cap || rows > shared.max_block_size || rows > head.max_n {
        return err(format!(
            "{rows} rows exceed the batch capacity (cap {cap}, scratch {}, head {})",
            shared.max_block_size, head.max_n
        ));
    }
    let draft_weights = std::sync::Arc::clone(&lanes[0].df.draft_weights);
    let draft_cfg = lanes[0].df.draft_config.clone();
    let h = draft_cfg.hidden;
    debug_assert_eq!(vocab, draft_cfg.vocab_size);
    for (i, l) in lanes.iter().enumerate() {
        if l.b < 3 {
            return err(format!("lane {i}: block {} < 3 rows", l.b));
        }
        if !std::sync::Arc::ptr_eq(&l.df.draft_weights, &draft_weights) {
            return err(format!("lane {i}: draft weights are not the shared asset"));
        }
        if l.b > l.df.draft_scratch.max_block_size {
            return err(format!("lane {i}: block {} > its draft scratch", l.b));
        }
    }
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // Noise embeddings of every lane's block into the shared rows (the
    // singleton's non-fused per-token lookup loop, `draft_dflash_forward_rank`).
    if noise_embed_batch_admitted(gpu, target_weights, shared, rows) {
        // One ids upload (the only host-sync H2D this adds, replacing 64
        // launches) into the shared `noise_tokens` plane, then one batched
        // lookup over all rows; row r = seed for each lane's row 0, else
        // the mask token — the same ids the per-row loop below uses.
        let mut ids: Vec<i32> = Vec::with_capacity(rows);
        for l in lanes.iter() {
            for i in 0..l.b {
                ids.push(
                    (if i == 0 {
                        l.seed
                    } else {
                        draft_cfg.mask_token_id
                    }) as i32,
                );
            }
        }
        let id_view = shared.noise_tokens.sub_offset(0, rows);
        let id_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, rows * 4) };
        gpu.hip.memcpy_htod(&id_view.buf, id_bytes)?;
        let out_view = shared.x.sub_offset(0, rows * h);
        gpu.embedding_lookup_q8_batched(&target_weights.token_embd, &out_view, &id_view, rows, h)?;
    } else {
        let mut off = 0usize;
        for l in lanes.iter() {
            for i in 0..l.b {
                let tok = if i == 0 {
                    l.seed
                } else {
                    draft_cfg.mask_token_id
                };
                let dst = shared.x.sub_offset((off + i) * h, h);
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
                    llama::EmbeddingFormat::F32 => {
                        gpu.embedding_lookup(&target_weights.token_embd, &dst, tok, h)?
                    }
                    _ => return err("unsupported target embedding format for noise lookup".into()),
                }
            }
            off += l.b;
        }
    }

    // Per-lane positions (ctx_slice = None geometry of the singleton).
    struct Geo {
        ctx: usize,
        pos_q: Vec<i32>,
        pos_k: Vec<i32>,
    }
    let geos: Vec<Geo> = lanes
        .iter()
        .map(|l| {
            let ctx =
                l.df.draft_scratch
                    .thlog
                    .abs_positions()
                    .len()
                    .min(l.position);
            let co = l.compact_offset;
            let pos_q: Vec<i32> =
                ((l.position as i32 + co)..(l.position as i32 + l.b as i32 + co)).collect();
            let mut pos_k = Vec::with_capacity(ctx + l.b);
            let th_abs = l.df.draft_scratch.thlog.abs_positions();
            pos_k.extend_from_slice(&th_abs[th_abs.len().saturating_sub(ctx)..]);
            for p in 0..l.b {
                pos_k.push(l.position as i32 + p as i32 + co);
            }
            Geo { ctx, pos_q, pos_k }
        })
        .collect();
    // One shared target lm-head over every lane's rows (seed rows included:
    // they are discarded, the head is row-independent).
    let w_out = &target_weights.output;
    assert!(
        rows * h <= head.max_n * head.hidden_k,
        "head.rot undersized for the batched draft lm_head"
    );
    let hidden_all = shared.x.sub_offset(0, rows * h);
    let logits_all = head.logits.sub_offset(0, rows * vocab);
    let rotated = head.rot.sub_offset(0, rows * h);
    // The forward and the head GEMM (and only they) run inside the 64-row
    // scope: a full 64-row chunk takes the scoped `_vt4` overwrite route,
    // per-row identical to the 48/16-row chunks. The scope is closed on every
    // path before the result propagates.
    let rows64 = rows == DFLASH_DRAFT_BATCH_ROWS64 && cap == DFLASH_DRAFT_BATCH_ROWS64;
    let prev = gpu.draft_rows64_enter(rows64);
    let scoped: HipResult<()> = (|| {
        {
            let mut dl: Vec<dflash::DraftLane<'_>> = lanes
                .iter_mut()
                .zip(&geos)
                .map(|(l, g)| dflash::DraftLane {
                    b: l.b,
                    scratch: &mut l.df.draft_scratch,
                    ctx_len: g.ctx,
                    positions_q: &g.pos_q,
                    positions_k: &g.pos_k,
                })
                .collect();
            dflash::draft_forward_lanes(gpu, &draft_weights, &draft_cfg, shared, &mut dl)?;
        }
        llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_all, &rotated, h, rows)?;
        gemm_mq_batched_lmhead_draft(
            gpu,
            w_out.gpu_dtype,
            &w_out.buf,
            &rotated,
            &logits_all,
            w_out.m,
            w_out.k,
            rows,
        )
    })();
    gpu.draft_rows64_exit(prev);
    scoped?;
    Ok(rows64)
}

/// Greedy chain drafts of several lanes through ONE draft-model forward
/// (`dflash::draft_forward_lanes`) and ONE shared target lm-head GEMM, each
/// lane keeping its own draft context rings, positions and hidden context:
/// the batched counterpart of per-lane `draft_dflash_block_rank` calls (the
/// singleton path is untouched). `shared` is the row scratch of the batched
/// forward (`>= sum b` rows), `head` the head scratch (`max_n >= sum b`).
/// Returns each lane's `[seed, candidates..]`. Preconditions:
/// [`dflash_draft_batch_eligible`], `3 <= b` per lane, `sum b <=
/// dflash_draft_batch_max_rows(gpu)`, one shared draft-weight asset.
#[allow(clippy::too_many_arguments)]
pub fn draft_dflash_blocks_batched(
    gpu: &mut Gpu,
    target_weights: &Qwen35Weights,
    vocab: usize,
    shared: &mut DflashScratch,
    head: &VerifyScratch,
    lanes: &mut [DflashDraftLane<'_>],
) -> HipResult<Vec<Vec<u32>>> {
    if lanes.is_empty() {
        return Ok(Vec::new());
    }
    let rows64 =
        draft_dflash_forward_head_batched(gpu, target_weights, vocab, shared, head, lanes)?;
    let err = |m: String| {
        Err(HipError::new(
            0,
            &format!("draft_dflash_blocks_batched: {m}"),
        ))
    };
    let rows: usize = lanes.iter().map(|l| l.b).sum();
    let draft_weights = std::sync::Arc::clone(&lanes[0].df.draft_weights);
    let h = lanes[0].df.draft_config.hidden;
    let hidden_all = shared.x.sub_offset(0, rows * h);
    let logits_all = head.logits.sub_offset(0, rows * vocab);

    let mut out: Vec<Vec<u32>> = Vec::with_capacity(lanes.len());
    if draft_weights.has_candidate_selector() && selector_lanes_batched(gpu, shared) {
        // ONE selector projection GEMM, top-K and download for every lane
        // (3 host syncs per chunk instead of 3 per lane), then per-lane host
        // scoring of each lane's rows `off+1 .. off+b` (seed row excluded) —
        // exactly the slices the per-lane loop below hands
        // `propose_candidates_device`. Byte-exact: the projection is the MQ4
        // v2 draft GEMM, whose gfx1201 overwrite route is per-row identical
        // for 2..=63 rows and, inside the 64-row draft scope (`rows64`, the
        // same predicate as the forward above), for exactly 64; the top-K is
        // one block per row. Both are row-local, so running them over every
        // row (seed rows included, ignored) yields each candidate row's
        // bytes from the lane-sized launches. Rotates through the shared
        // scratch's own FWHT buffers, so no lending.
        let mut sel: Vec<dflash::SelectorLane> = Vec::with_capacity(lanes.len());
        let mut off = 0usize;
        for l in lanes.iter() {
            sel.push(dflash::SelectorLane {
                first_row: off + 1,
                rows: l.b - 1,
                anchor: l.seed,
            });
            off += l.b;
        }
        let proposals = dflash::propose_candidates_lanes_device(
            gpu,
            &draft_weights,
            shared,
            &hidden_all,
            &logits_all,
            rows,
            &sel,
            rows64,
        )?;
        for (l, proposal) in lanes.iter().zip(&proposals) {
            let mut drafted = vec![l.seed];
            apply_dflash2_selector_proposal(
                proposal,
                vocab,
                false,
                &mut drafted,
                &mut Vec::new(),
                &mut Vec::new(),
            )?;
            out.push(drafted);
        }
    } else if draft_weights.has_candidate_selector() {
        let mut off = 0usize;
        for l in lanes.iter_mut() {
            let batch = l.b - 1;
            let hidden_rows = shared.x.sub_offset((off + 1) * h, batch * h);
            let logits_batch = head.logits.sub_offset((off + 1) * vocab, batch * vocab);
            // The selector's hidden projection is a lane-scratch GEMM
            // (`propose_candidates_device` rotates through the scratch's FWHT
            // buffers). A VMM lane owns none: lend the shared scratch's for
            // this call and hand them back before looking at the result
            // (per-call temporary, identical element count, single stream).
            let lent = l.df.draft_scratch.mq_x_rot.is_none();
            if lent {
                if shared.mq_x_rot.is_none() || shared.mq_x_rot_f16.is_none() {
                    return err("the shared draft scratch has no FWHT rotation buffers for the lane selector".into());
                }
                l.df.draft_scratch.mq_x_rot = shared.mq_x_rot.take();
                l.df.draft_scratch.mq_x_rot_f16 = shared.mq_x_rot_f16.take();
            }
            let proposal = dflash::propose_candidates_device(
                gpu,
                &draft_weights,
                &l.df.draft_scratch,
                &hidden_rows,
                &logits_batch,
                batch,
                l.seed,
                0.0,
                None,
            );
            if lent {
                shared.mq_x_rot = l.df.draft_scratch.mq_x_rot.take();
                shared.mq_x_rot_f16 = l.df.draft_scratch.mq_x_rot_f16.take();
            }
            let proposal = proposal?;
            let mut drafted = vec![l.seed];
            apply_dflash2_selector_proposal(
                &proposal,
                vocab,
                false,
                &mut drafted,
                &mut Vec::new(),
                &mut Vec::new(),
            )?;
            out.push(drafted);
            off += l.b;
        }
    } else {
        let argmax_buf = head.argmax.sub_offset(0, rows);
        gpu.argmax_f32_batched(&logits_all, &argmax_buf, vocab, rows)?;
        let mut host_idx = vec![0i32; rows];
        {
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, rows * 4)
            };
            gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf)?;
        }
        let mut off = 0usize;
        for l in lanes.iter() {
            let mut drafted = vec![l.seed];
            drafted.extend(host_idx[off + 1..off + l.b].iter().map(|&i| i as u32));
            out.push(drafted);
            off += l.b;
        }
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// DFlash verbatim-lookup splice seam
// ═══════════════════════════════════════════════════════════════════════════

/// Largest lookup candidate count the singleton DFlash2 window can verify:
/// 16 verify rows incl. the seed. Mirrors `spec_lookup::LOOKUP_MAX_CANDIDATES`
/// (this crate does not depend on the policy module).
pub const DFLASH_LOOKUP_MAX_CANDIDATES: usize = 15;

/// Verbatim-lookup candidates composed into one DFlash2 verify window
/// (selector mode only). The window's effective verify rows are
/// `b_eff = 1 + K + L <= b`; every downstream verify / accept / commit /
/// rollback step runs at `b_eff`. The learned draft always runs at the
/// original `b` (no draft-work saving), so RNG draws of the learned draft are
/// identical with and without a lookup.
pub enum DflashLookup<'a> {
    /// Copy window: `tokens` (1..=b-1) replace every learned candidate
    /// (K = 0). The learned draft forward still runs; its candidates are
    /// discarded.
    Copy(&'a [u32]),
    /// Hybrid: after the learned draft, the source may keep the first K
    /// learned candidates and append a lookup tail.
    Tail(&'a mut dyn DflashTailSource),
}

pub trait DflashTailSource {
    /// `drafted` = learned candidates d1..d(b-1) (seed excluded). Sequential
    /// stopping rule REQUIRED: the decision to splice at K may read only
    /// `drafted[..K]` (sampled-law validity).
    /// `Some(k)` with `1 <= k` and the tail (`1..=b-1-k` tokens) written into
    /// `tail` (cleared first); `None` keeps the incumbent window unchanged.
    fn splice(&mut self, drafted: &[u32], tail: &mut Vec<u32>) -> Option<usize>;

    /// `NoTailAction` (railgun design A "Whole16 vs HostSplice"): a pure,
    /// conservative predicate over the source's current (immutable) decision
    /// state. `true` only when **no** `splice` over a `candidates`-candidate
    /// learned block can return `Some`, whatever the learned tokens and
    /// whatever the lookup context offers. The default `false` never proves
    /// it, so an unknown source always keeps the HostSplice shape.
    fn declines_every_tail(&self, _candidates: usize) -> bool {
        false
    }
}

fn lookup_err(msg: &str) -> hip_bridge::HipError {
    hip_bridge::HipError::new(0, msg)
}

/// Copy-window shape: `1 <= tokens.len() <= b - 1`, every id `< vocab`.
fn dflash_lookup_check_copy(tokens: &[u32], b: usize, vocab: usize) -> HipResult<()> {
    if tokens.is_empty() {
        return Err(lookup_err(
            "dflash lookup: Copy window needs at least one token",
        ));
    }
    if tokens.len() + 1 > b {
        return Err(lookup_err(&format!(
            "dflash lookup: Copy window of {} tokens exceeds the window's {} verify rows",
            tokens.len(),
            b
        )));
    }
    dflash_lookup_check_ids(tokens, vocab)
}

fn dflash_lookup_check_ids(tokens: &[u32], vocab: usize) -> HipResult<()> {
    if let Some(&t) = tokens.iter().find(|&&t| t as usize >= vocab) {
        return Err(lookup_err(&format!(
            "dflash lookup: token id {t} out of vocab {vocab}"
        )));
    }
    Ok(())
}

/// Pre-draft refusal: nothing has run yet, so an `Err` leaves every piece of
/// state (RNG included) untouched.
fn dflash_lookup_preflight(
    lookup: &Option<DflashLookup<'_>>,
    selector_mode: bool,
    pld_spine: bool,
    b: usize,
    vocab: usize,
) -> HipResult<()> {
    let Some(lookup) = lookup else {
        return Ok(());
    };
    if !selector_mode {
        return Err(lookup_err(
            "dflash lookup: requires the DFlash2 candidate selector (learned q receipts)",
        ));
    }
    if pld_spine {
        return Err(lookup_err("dflash lookup: not supported with a PLD spine"));
    }
    if b < 2 {
        return Err(lookup_err("dflash lookup: window has no candidate rows"));
    }
    if let DflashLookup::Copy(tokens) = lookup {
        dflash_lookup_check_copy(tokens, b, vocab)?;
    }
    Ok(())
}

/// Resolve the splice against the learned block `drafted` (`[seed, d1..d(b-1)]`).
/// On `Ok(Some(keep))` the lookup tokens (Copy: the caller's tokens; Tail: the
/// source's tail) are in `tail` and `1 + keep + tail.len() <= b`. `Ok(None)`:
/// the source declined, keep the incumbent window. Validation precedes every
/// verify/commit write; `drafted` is never mutated here.
fn dflash_lookup_resolve(
    lookup: DflashLookup<'_>,
    drafted: &[u32],
    b: usize,
    vocab: usize,
    tail: &mut Vec<u32>,
) -> HipResult<Option<usize>> {
    match lookup {
        DflashLookup::Copy(tokens) => {
            dflash_lookup_check_copy(tokens, b, vocab)?;
            tail.clear();
            tail.extend_from_slice(tokens);
            Ok(Some(0))
        }
        DflashLookup::Tail(source) => {
            if drafted.len() != b {
                return Err(lookup_err(&format!(
                    "dflash lookup: learned block has {} rows, window has {}",
                    drafted.len(),
                    b
                )));
            }
            tail.clear();
            let Some(k) = source.splice(&drafted[1..], tail) else {
                tail.clear();
                return Ok(None);
            };
            if k < 1 || k > b - 1 {
                return Err(lookup_err(&format!(
                    "dflash lookup: Tail keep {k} outside 1..={}",
                    b - 1
                )));
            }
            if tail.is_empty() {
                return Err(lookup_err("dflash lookup: Tail spliced with an empty tail"));
            }
            if k + tail.len() > b - 1 {
                return Err(lookup_err(&format!(
                    "dflash lookup: Tail keep {k} + tail {} exceeds {} candidate rows",
                    tail.len(),
                    b - 1
                )));
            }
            dflash_lookup_check_ids(tail, vocab)?;
            Ok(Some(k))
        }
    }
}

/// A sampled window needs the learned q receipt of at least `keep` rows to
/// splice onto; checked before any receipt mutation.
fn dflash_lookup_check_receipt(receipt: Option<&DflashSampledDraft>, keep: usize) -> HipResult<()> {
    let Some(r) = receipt else {
        return Err(lookup_err(
            "dflash lookup: sampled window without a q receipt",
        ));
    };
    let rows = if r.sparse {
        r.sparse_q.rows()
    } else {
        r.dense_q.len()
    };
    if r.selected_q.len() < keep || rows < keep {
        return Err(lookup_err(&format!(
            "dflash lookup: q receipt holds {} rows, Tail keeps {keep}",
            r.selected_q.len().min(rows)
        )));
    }
    Ok(())
}

/// Rebuild the q receipt of a spliced sampled window: learned rows `..keep`
/// keep their real q (selected probability and support), every lookup row is a
/// POINT-MASS q (`selected_q = 1.0`, support `{c: 1.0}` — a one-entry sparse
/// row, or a one-hot dense row). The learned selector `proposal` is dropped:
/// it describes the discarded learned rows and must not feed lookup rows.
/// Dense mode reuses the discarded learned rows' buffers.
fn dflash_lookup_splice_receipt(
    receipt: &mut DflashSampledDraft,
    keep: usize,
    tokens: &[u32],
    vocab: usize,
) -> HipResult<()> {
    receipt.selected_q.truncate(keep);
    receipt.uniforms.truncate(keep);
    receipt.proposal = None;
    if receipt.sparse {
        receipt.sparse_q.truncate_rows(keep);
        for &c in tokens {
            receipt.sparse_q.push_row(&[c], &[1.0], vocab)?;
            receipt.selected_q.push(1.0);
        }
    } else {
        for (j, &c) in tokens.iter().enumerate() {
            let row = keep + j;
            if row < receipt.dense_q.len() {
                let r = &mut receipt.dense_q[row];
                r.clear();
                r.resize(vocab, 0.0);
            } else {
                receipt.dense_q.push(vec![0.0; vocab]);
            }
            receipt.dense_q[row][c as usize] = 1.0;
            receipt.selected_q.push(1.0);
        }
        receipt.dense_q.truncate(keep + tokens.len());
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// DFlash spec step — one speculative decode iteration
// ═══════════════════════════════════════════════════════════════════════════

/// One DFlash speculative iteration. Given a previously-accepted token at
/// `position - 1` (the "seed" for block_output_ids[0]) and a cumulative
/// `target_hidden_host` buffer of shape `[position × num_extract × hidden]`,
/// runs the draft to fill B-1 mask slots, verifies against the target,
/// commits the accepted prefix plus a bonus target token, and rewinds the
/// target's DeltaNet state so only `accept_len + 1` forwards are reflected.
///
/// Returns `SpecStepResult` describing accepted draft count, bonus token,
/// drafted proposals, and the full committed sequence (length accept+2:
/// `[seed_token, draft[..accept_len], posterior[accept_len]]` — note the
/// seed_token is ALSO committed here because it was the bonus token from
/// the PREVIOUS iteration and still needs the target forward at its
/// position). Callers append `committed[1..]` to the output token stream
/// (the seed was already emitted).
///
/// Side effects:
/// - With `ctx_slice=Some`, appends `accept_len + 1` positions ×
///   `num_extract × hidden` floats to `target_hidden_host`; otherwise the
///   appended state stays on device and this argument is not read.
/// - Advances target's KV cache and DeltaNet state by `accept_len + 1`
///   positions. Draft has no persistent state.
///
/// Preconditions:
/// - With `ctx_slice=Some`, `target_hidden_host.len() == position ×
///   num_extract × hidden` (set up by `seed_target_hidden_from_prompt`).
/// - `position ≤ draft_scratch.max_ctx_len`.
/// - `draft_cfg.block_size ≤ draft_scratch.max_block_size`.
///
/// `ctx_slice`: if `Some(N)`, the draft only sees the most recent `N` rows
/// of `target_hidden_host` (with RoPE positions `[position-N..position+B)`).
/// Use this for accept-rate bisect experiments — if training-time context
/// was shorter than inference-time, truncation may help. `None` uses the
/// full cumulative context (the default, distribution-preserving path).
///
/// `lookup`: `None` is the incumbent window. `Some(Copy(tokens))` verifies
/// `[seed, tokens..]` in place of the learned candidates; `Some(Tail(src))`
/// calls `src.splice` after the learned draft and, on `Some(k)`, verifies
/// `[seed, d1..dk, tail..]`. A declining `Tail` source is byte-identical to
/// `None`. DFlash2 selector mode only. The learned draft still runs at the
/// original `b`; all later stages run at `b_eff = 1 + K + L <= b`.
/// Sampled windows give lookup rows point-mass q receipts (see
/// `dflash_lookup_splice_receipt`). Invalid splices return `Err` before any
/// verify/commit write, with the sampling RNG restored to its entry state.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_dflash(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut DeltaNetSnapshot,
    verify_scratch: &VerifyScratch,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    gdn_tape: Option<&mut GdnTape>,
    temp: f32,
    // Nucleus (top_p) mass on the temp sampling path, applied with `top_k`
    // to both the draft and target softmax rows as AR does
    // (`llama::sample_top_k_p`): the nucleus is taken over the top_k pool's
    // own mass. Lossless == AR via the TARGET truncation (Leviathan marginal
    // preservation); the draft is truncated too for acceptance efficiency.
    // See apply_topp_trunc / apply_host_ar_trunc.
    top_p: f32,
    // AR candidate cap (`llama::ar_candidate_cap`: absent top_k → 20, 0 → 64,
    // k → min(k, 64)). The sampled path keeps the `top_k` most probable
    // tokens before the nucleus, like AR's pool.
    top_k: usize,
    rng_state: &mut u64,
    block_size_override: Option<usize>,
    ngram_cache: Option<&NgramCache>,
    prev_committed: &[u32],
    cactus_delta: f32,
    pld_spine: Option<&[u32]>,
    repeat_penalty: f32,
    repeat_window: usize,
    // Max accepted drafts this window (`emit.len() == accepted + 1` after the
    // bonus). `None` = uncapped (bench/demo). Clamped **before** hidden/KV/
    // DeltaNet/drafter commit so post-hoc `cap_emit` is defense only.
    max_accept: Option<usize>,
    // Retained-PM4 verify route owned by the speculator. `None` keeps the
    // shipping HIP/HipGraph path byte-for-byte.
    verify_pm4: Option<&mut DflashVerifyPm4>,
    lookup: Option<DflashLookup<'_>>,
) -> HipResult<SpecStepResult> {
    spec_step_dflash_with_railgun(
        gpu,
        target,
        draft_weights,
        draft_cfg,
        draft_scratch,
        hidden_rb,
        target_hidden_host,
        target_snap,
        verify_scratch,
        position,
        seed_token,
        ctx_slice,
        gdn_tape,
        temp,
        top_p,
        top_k,
        rng_state,
        block_size_override,
        ngram_cache,
        prev_committed,
        cactus_delta,
        pld_spine,
        repeat_penalty,
        repeat_window,
        max_accept,
        verify_pm4,
        lookup,
        None,
    )
}

/// The railgun window owner a `DflashSpeculator` lends one DFlash window
/// (`None` everywhere else: [`spec_step_dflash`] passes `None`, so every
/// other caller keeps the incumbent route byte for byte).
pub struct RailgunStep<'a> {
    pub owner: &'a mut crate::dflash_railgun::RailgunWindowOwner,
    /// The lookup layer planned this window as a price probe (it keeps its
    /// HIP owner).
    pub probe: bool,
}

/// [`spec_step_dflash`] with an optional certified-window owner (design A
/// slice A6). With `railgun = None` this is `spec_step_dflash` exactly. With
/// `Some`, an admitted window runs `Whole16` (draft + selector + verify +
/// hidden commit in one lowered segment, then the unchanged CPU accept /
/// clamp / scatter / GDN repair) or `HostSplice` (draft segment, then the
/// existing host splice and the incumbent verify at its actual B); any
/// refusal before the queue is touched runs the window on the ordinary HIP
/// route with nothing changed.
#[allow(clippy::too_many_arguments)]
pub fn spec_step_dflash_with_railgun(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut DeltaNetSnapshot,
    verify_scratch: &VerifyScratch,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    mut gdn_tape: Option<&mut GdnTape>,
    temp: f32,
    top_p: f32,
    top_k: usize,
    rng_state: &mut u64,
    block_size_override: Option<usize>,
    ngram_cache: Option<&NgramCache>,
    prev_committed: &[u32],
    cactus_delta: f32,
    pld_spine: Option<&[u32]>,
    repeat_penalty: f32,
    repeat_window: usize,
    max_accept: Option<usize>,
    verify_pm4: Option<&mut DflashVerifyPm4>,
    lookup: Option<DflashLookup<'_>>,
    mut railgun: Option<RailgunStep<'_>>,
) -> HipResult<SpecStepResult> {
    let selector_mode = draft_weights.has_candidate_selector();
    if selector_mode {
        if pld_spine.is_some() {
            return Err(hip_bridge::HipError::new(
                0,
                "selector mode: PLD spine rewrite is not supported with DFlash2 candidate selector",
            ));
        }
        if ngram_cache.is_some() {
            return Err(hip_bridge::HipError::new(
                0,
                "selector mode: ngram-cache rewrite is not supported with DFlash2 candidate selector",
            ));
        }
        if cactus_delta != 0.0 {
            return Err(hip_bridge::HipError::new(
                0,
                "selector mode: CACTUS is not supported with DFlash2 candidate selector",
            ));
        }
        if repeat_penalty != 1.0 {
            return Err(hip_bridge::HipError::new(
                0,
                "selector mode: repeat penalty is not supported with DFlash2 candidate selector",
            ));
        }
        if hipfire_runtime::config::get()
            .dflash_ngram_block
            .unwrap_or(false)
        {
            return Err(hip_bridge::HipError::new(
                0,
                "selector mode: ngram-block rewriting is not supported with DFlash2 candidate selector",
            ));
        }
    }
    // Effective block size for THIS step. Usually `draft_cfg.block_size`
    // (what the draft was trained at, 16 for Qwen3.5-*-DFlash) but a caller
    // doing adaptive-B based on rolling τ can shrink to save per-iter cost.
    //
    // When `pld_spine` is Some, shrink b to 1+pld.len() (capped at requested)
    // so we don't run off the end of the PLD continuation. PLD-supplied
    // spines are often shorter than the trained B; the paper caps at 8.
    let requested_b = block_size_override.unwrap_or(draft_cfg.block_size);
    let b = match pld_spine {
        Some(pld) => (1 + pld.len()).min(requested_b).max(2),
        None => requested_b,
    };
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let vocab = target.config.vocab_size;
    let mask_token = draft_cfg.mask_token_id;
    dflash_lookup_preflight(&lookup, selector_mode, pld_spine.is_some(), b, vocab)?;
    // Sampled windows consume RNG in the learned draft before the splice can
    // be refused; an abandoned window restores it.
    let rng_at_entry = *rng_state;

    // Ensure active_stream is set before any draft/verify work so memset_async
    // and stream-ordered launches have a non-null stream to ride on. Without
    // this, the lm_head pre-zero memsets in dispatch.rs:4475/4545 fall through
    // to the sync hipMemset path (~46 hot calls/cycle on 27B).
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    assert!(b >= 2, "dflash block size must be ≥ 2");
    // `target_hidden_host` is only authoritative on the ctx_slice=Some path,
    // where it backs the CPU slice handed to draft_forward. On the default
    // ctx_slice=None path the data lives on GPU in draft_scratch.target_hidden
    // (populated by D2D scatter, no CPU shadow). Only enforce the length
    // invariant when we actually read it.
    if ctx_slice.is_some() {
        assert_eq!(
            target_hidden_host.len(),
            position * ne * h,
            "target_hidden_host size mismatches position"
        );
    }

    // HIPFIRE_SPEC_PHASES=1: per-cycle phase breakdown. Inserts a
    // device_synchronize at each phase boundary so the wall-clock reflects
    // ACTUAL GPU completion (not CPU enqueue of async work). Perf-heavy —
    // use only for diagnostics. When disabled, zero cost beyond a handful
    // of Instant::now() calls.
    static PHASE_ON_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let phase_on = *PHASE_ON_ENV.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_SPEC_PHASES")
            .ok()
            .as_deref()
            == Some("1")
    });
    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_spec_start = std::time::Instant::now();
    let mut t_phase = t_spec_start;

    // ── 1. block_output_ids seeded with prev bonus at [0], masks at [1..B] ──
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;

    // Draft state: either synthesized from a PLD spine (Goose §4.3 bypass
    // mode — deterministic, skips the DFlash forward) or produced by the
    // DFlash draft forward pass below. Declared out here so the post-draft
    // common code (ngram gating, target verify, rejection) sees the same
    // `drafted` / `draft_softmaxes` / `draft_probs_at_drafted` regardless
    // of draft source.
    let mut drafted: Vec<u32> = vec![seed_token];
    let mut draft_probs_at_drafted: Vec<f32> = Vec::new();
    let mut draft_softmaxes: Vec<Vec<f32>> = Vec::new();
    // Selector / PLD q rows in the sparse FAST_SAMPLE representation; `None`
    // keeps the dense `draft_softmaxes` rows.
    let mut q_receipt: Option<DflashSampledDraft> = None;
    let use_temp_sampling = temp > 0.0;
    let rp_active = repeat_penalty > 1.0 && !use_temp_sampling;
    // HIPFIRE_DFLASH_NGRAM_BLOCK=1: apply llama::apply_ngram_block to every
    // host-path row in BOTH draft and target argmax paths. Bans the next
    // token after any 3/4/5/6-gram repeat (NEG_INFINITY logit). Matches the
    // production-path defense in daemon/run/infer for the AR sampler.
    // Forces the per-row host download even when RP is off (extra D2H per
    // cycle); off-by-default for that reason.
    let ngram_block_active = !use_temp_sampling
        && hipfire_runtime::config::get()
            .dflash_ngram_block
            .unwrap_or(false);
    // HIPFIRE_DFLASH_LOGIT_DUMP=1: per-cycle diagnostic. Forces the host-logits
    // path and prints, at the acceptance boundary, the target top1/top2 margin
    // and the logit gap to the drafted (rejected) token. A tiny gap at the
    // first divergence position is the signature of a sub-ULP numerics
    // tie-break (noise), vs a large gap = a real forward divergence. Greedy
    // only; zero cost when unset.
    static LOGIT_DUMP_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let logit_dump_active = !use_temp_sampling
        && *LOGIT_DUMP_ENV.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DFLASH_LOGIT_DUMP")
                .ok()
                .as_deref()
                == Some("1")
        });
    let host_path_active = rp_active || ngram_block_active || logit_dump_active;
    // HIPFIRE_DFLASH_FAST_SAMPLE (default ON, opt out with =0): in the temp>0
    // sampled path, compute the per-row softmax (and the top_p nucleus cutoff) on
    // the GPU instead of the host `exp` loop over the full vocab. The host RNG,
    // accept test, residual sampler, and CACTUS math are UNCHANGED — they just
    // read the GPU-produced probabilities instead of host-recomputed ones. The
    // D2H transfer size is identical (full-vocab probs vs full-vocab logits);
    // what moves to the GPU is the ~31×vocab `exp`/max/normalize work per
    // cycle. PARITY: distribution-only, NOT byte-identical (GPU tree-reduction
    // vs host sequential sum can differ at the last ULP and rarely flip a
    // borderline `u*p_d <= p_t` accept) — validated coherent across genres
    // (no attractors), so default-on. Greedy path (temp==0) is never affected.
    let fast_sample_active = use_temp_sampling && gpu.flags.dflash_fast_sample;
    let draft_ffn_graph_env =
        hipfire_config::developer_var("HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH").ok();
    let draft_ffn_graph = dflash_moe_draft_ffn_graph_eligible(
        target.config.num_experts,
        ctx_slice.is_some(),
        pld_spine.is_some(),
        use_temp_sampling,
        draft_ffn_graph_env.as_deref(),
    );

    // C8: device tensors kept alive from draft-sample through verify-accept.
    // Only populated when fast_sample_active AND the batched GEMM path runs.
    // Otherwise stays None and the host accept loop falls back to draft_softmaxes.
    let mut c8_draft_probs_dev: Option<GpuTensor> = None;
    let mut c8_draft_tau_dev: Option<GpuTensor> = None;
    let mut c8_draft_z_dev: Option<GpuTensor> = None;
    let mut c8_draft_tokens_dev: Option<GpuTensor> = None;
    let mut c8_draft_p_at_token_dev: Option<GpuTensor> = None;

    // ── Railgun certified window (design A slice A6) ────────────────────
    // Inert unless the speculator lends an owner (`HIPFIRE_DFLASH_RAILGUN_WINDOW=1`
    // at build): with `railgun = None` nothing below this comment block
    // executes and every later stage is the incumbent code unchanged.
    // `Whole16` supplies the proposals AND the verify picks; `HostSplice`
    // supplies the proposals only (splice and verify stay incumbent).
    // Every refusal leaves all state untouched and falls through to HIP.
    let mut railgun_draft: Option<Vec<u32>> = None;
    let mut railgun_verify: Option<crate::dflash_railgun::RailgunVerify> = None;
    let mut railgun_shape: Option<railgun::window::WindowShape> = None;
    if let Some(rg) = railgun.as_mut() {
        use crate::dflash_railgun as drg;
        railgun_shape = drg::RailgunWindowOwner::shape_hint(
            position,
            draft_scratch,
            hidden_rb,
            target.kv_cache.compact_offset,
            target.kv_cache.physical_cap,
        );
        let static_verdict = rg.owner.static_verdict(|| {
            let kv = &target.kv_cache;
            drg::StaticFacts {
                arch: gpu.arch.clone(),
                num_experts: target.config.num_experts,
                dim: target.config.dim,
                ffn_dim: target.config.hidden_dim,
                n_heads: target.config.n_heads,
                n_kv_heads: target.config.n_kv_heads,
                kv_fp8: kv.quant_fp8,
                kv_other_quant: kv.quant_q8
                    || kv.quant_int8
                    || kv.quant_hfq4
                    || kv.quant_asym4
                    || kv.quant_asym3
                    || kv.quant_asym2
                    || kv.quant_fwht
                    || kv.quant_bf16,
                // The runtime proposal width (16 on the 27B pair, whose draft
                // was trained at block 8), not the draft's trained block.
                draft_block: requested_b,
                ctx_mode: draft_scratch.ctx_mode,
                retained_pm4_admitted: verify_pm4.as_deref().map_or(false, |r| r.admitted()),
                online_tuner: draft_scratch.online.is_some(),
            }
        });
        match static_verdict {
            Err(why) => rg.owner.refuse(why, ""),
            Ok(()) => {
                // `NoTailAction`: an armed Tail source is Whole16 only when
                // it proves every legal tail decision declines; Copy and
                // price-probe windows planned before the draft keep HIP.
                let lookup_facts = if rg.probe {
                    drg::LookupFacts::Probe
                } else {
                    match &lookup {
                        None => drg::LookupFacts::None,
                        Some(DflashLookup::Copy(_)) => drg::LookupFacts::Copy,
                        Some(DflashLookup::Tail(src)) => {
                            if src.declines_every_tail(b.saturating_sub(1)) {
                                drg::LookupFacts::TailDeclinesAll
                            } else {
                                drg::LookupFacts::TailMayAct
                            }
                        }
                    }
                };
                let seed_oracle = hipfire_config::developer_var("HIPFIRE_DFLASH_SEED_ORACLE")
                    .ok()
                    .as_deref()
                    == Some("1");
                let facts = drg::WindowFacts {
                    b,
                    greedy: !use_temp_sampling,
                    cactus: cactus_delta != 0.0,
                    pld: pld_spine.is_some(),
                    ngram_cache: ngram_cache.is_some(),
                    repeat_penalty: repeat_penalty != 1.0,
                    host_path: host_path_active,
                    diagnostics: phase_on || seed_oracle,
                    ctx_slice: ctx_slice.is_some(),
                    gdn_tape: gdn_tape.is_some(),
                    gdn_frame_independent: target.dn_state.quant != qwen35::StateQuant::Q8
                        || !target.dn_state.s_ef_residual.is_empty(),
                    lookup: lookup_facts,
                };
                match drg::admit_window(&facts) {
                    Err(why) => rg.owner.refuse(why, ""),
                    Ok(execution) => {
                        let attempt = rg.owner.run(
                            drg::WindowOwners {
                                gpu: &mut *gpu,
                                target: &mut *target,
                                draft_weights,
                                draft_cfg,
                                draft_scratch: &mut *draft_scratch,
                                hidden_rb: &mut *hidden_rb,
                                verify_scratch,
                                gdn_tape: gdn_tape.as_deref_mut(),
                                target_snap: &mut *target_snap,
                                position,
                                seed: seed_token,
                            },
                            execution,
                        )?;
                        if let drg::RailgunAttempt::Ran(run) = attempt {
                            railgun_draft = Some(run.block);
                            railgun_verify = run.verify;
                        }
                    }
                }
            }
        }
    }
    let railgun_ran = railgun_draft.is_some();

    if let Some(rail_block) = railgun_draft.take() {
        // The certified segment already produced the proposal block (the
        // selector-greedy block with the seed at [0]).
        drafted = rail_block;
    } else if let Some(pld) = pld_spine {
        // PLD spine path: drafted tokens come from context-suffix match.
        // At temp>0, draft "probability" at each PLD token is 1.0 — PLD is
        // context-deterministic, not a softmax. The rejection math below
        // computes residual from (target_probs − draft_probs)+ normalized,
        // and with draft one-hot at tok, the residual pulls correctly from
        // target minus just that single-position overclaim.
        for i in 0..b - 1 {
            drafted.push(pld[i]);
        }
        if use_temp_sampling {
            let sample_cfg = DflashSampleCfg {
                temp,
                top_p,
                top_k,
                cactus_delta,
            };
            if dflash_sparse_target_enabled(&sample_cfg) {
                // q = point mass at the PLD token: a one-entry sparse row,
                // identical arithmetic to the one-hot vocab row below.
                let mut r = DflashSampledDraft::default();
                r.sparse = true;
                r.sparse_q.clear();
                r.selected_q.reserve(b - 1);
                for i in 0..b - 1 {
                    r.sparse_q.push_row(&[pld[i]], &[1.0], vocab)?;
                    r.selected_q.push(1.0);
                }
                q_receipt = Some(r);
            } else {
                draft_probs_at_drafted.reserve(b - 1);
                draft_softmaxes.reserve(b - 1);
                for i in 0..b - 1 {
                    let mut probs = vec![0f32; vocab];
                    probs[pld[i] as usize] = 1.0;
                    draft_softmaxes.push(probs);
                    draft_probs_at_drafted.push(1.0);
                }
            }
        }
    } else if use_temp_sampling && selector_mode {
        let mut receipt = DflashSampledDraft::default();
        dflash_sampled_selector_draft_parts_into(
            gpu,
            &target.weights,
            draft_weights,
            draft_cfg,
            draft_scratch,
            verify_scratch,
            target_hidden_host,
            &target.scratch.logits,
            position,
            target.kv_cache.compact_offset as i32,
            seed_token,
            b,
            vocab,
            ctx_slice,
            draft_ffn_graph,
            DflashSampleCfg {
                temp,
                top_p,
                top_k,
                cactus_delta,
            },
            rng_state,
            &mut receipt,
        )?;
        drafted = std::mem::take(&mut receipt.block);
        q_receipt = Some(receipt);
    } else if !use_temp_sampling && !host_path_active {
        // Greedy fast path shared verbatim with the dense-TP mesh ranks
        // (byte-identical draft production through draft_dflash_block_rank).
        drafted = draft_dflash_block_rank(
            gpu,
            &target.weights,
            draft_weights,
            draft_cfg,
            draft_scratch,
            verify_scratch,
            target_hidden_host,
            &block,
            position,
            target.kv_cache.compact_offset as i32,
            ctx_slice,
            draft_ffn_graph,
            seed_token,
            vocab,
        )?;
    } else {
        // Sampled / host-shaped (RP, n-gram-block, logit-dump) path: shared
        // forward prefix, per-row head selection below.
        let co = target.kv_cache.compact_offset as i32;
        draft_dflash_forward_rank(
            gpu,
            &target.weights,
            draft_weights,
            draft_cfg,
            draft_scratch,
            target_hidden_host,
            &block,
            position,
            co,
            ctx_slice,
            draft_ffn_graph,
        )?;

        // ── 5. Apply target.lm_head to draft hidden positions 1..B ──────────
        // Fast path: a single batched GEMM against target.weights.output over
        // (B-1) hidden rows at once. Drops lm_head from ~40 ms (B-1 serial
        // weight_gemv + downloads) to ~8 ms (one batched GEMM + one download)
        // for MQ4/HFQ4 lm_heads. Falls back to the per-row loop when the
        // output weight dtype isn't covered by the batched gemm dispatch.
        //
        // Temperature-sampling mode (temp > 0): we must DOWNLOAD the full
        // (B-1, vocab) draft logits, softmax + sample + record p_draft[token]
        // for later rejection acceptance. The greedy GPU-argmax path is kept
        // intact for temp == 0 so we don't regress that case.
        let w_out = &target.weights.output;
        let use_batched_gemm = matches!(
            w_out.gpu_dtype,
            rdna_compute::DType::Q8_0
                | rdna_compute::DType::HFQ4G256
                | rdna_compute::DType::MQ4G256
                | rdna_compute::DType::MQ4G256V2
                | rdna_compute::DType::MQ6G256V2
                | rdna_compute::DType::MQ5G256V2
                | rdna_compute::DType::MQ3G256V2
                | rdna_compute::DType::MQ2G256V2
                | rdna_compute::DType::MQ3G256
                | rdna_compute::DType::HFQ6G256
                | rdna_compute::DType::MQ6G256,
        );
        if use_batched_gemm {
            // Unified batched path: one GEMM over B-1 rows, GPU-side argmax,
            // download just (B-1) × 4 bytes of indices.
            //
            // Reuses `verify_scratch.logits` and `.rot` — same buffers the target
            // verify uses. Draft calls this BEFORE verify in the cycle, so
            // there's no aliasing. The verify call overwrites these buffers
            // afterward. Avoids 2-3 hipMalloc/Free pairs per cycle.
            let batch = b - 1;
            assert!(
                batch <= verify_scratch.max_n,
                "verify_scratch max_n {} < draft batch {}",
                verify_scratch.max_n,
                batch
            );
            let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
            let logits_batch = verify_scratch.logits.sub_offset(0, batch * vocab);

            match w_out.gpu_dtype {
                rdna_compute::DType::Q8_0 => {
                    dflash_gemm_q8_lmhead(gpu, w_out, &hidden_rows, &logits_batch, batch)?;
                }
                rdna_compute::DType::HFQ4G256 => {
                    run_spec_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
                        &w_out.buf,
                        w_out.gpu_dtype,
                        &hidden_rows,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ4G256 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    // AWQ-aware rotation; same rationale as the target-verify
                    // arms above.
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    run_spec_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
                        &w_out.buf,
                        w_out.gpu_dtype,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ4G256V2
                | rdna_compute::DType::MQ6G256V2
                | rdna_compute::DType::MQ5G256V2
                | rdna_compute::DType::MQ3G256V2
                | rdna_compute::DType::MQ2G256V2 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for MQ v2 draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    gemm_mq_batched_lmhead_draft(
                        gpu,
                        w_out.gpu_dtype,
                        &w_out.buf,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ3G256 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for MQ3 draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    run_spec_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq3G256BatchedLmhead,
                        &w_out.buf,
                        w_out.gpu_dtype,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::HFQ6G256 => {
                    run_spec_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                        &w_out.buf,
                        w_out.gpu_dtype,
                        &hidden_rows,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ6G256 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for MQ6 draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    run_spec_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                        &w_out.buf,
                        w_out.gpu_dtype,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                _ => unreachable!(),
            }

            if selector_mode {
                // Selector q is proposal-local: seed anchor this cycle, then
                // sequential tokens. Greedy stays on-device (small top-K D2H
                // inside propose_candidates_device). Temperature materializes
                // sparse q into draft_softmaxes and does NOT take the GPU
                // full-vocab fast-accept path. Request top_p/top_k is not
                // applied to selector q; target p still uses them at verify.
                let uniforms = if use_temp_sampling {
                    Some(
                        (0..batch)
                            .map(|_| xorshift_next_unit(rng_state))
                            .collect::<Vec<f32>>(),
                    )
                } else {
                    None
                };
                let proposal = dflash::propose_candidates_device(
                    gpu,
                    draft_weights,
                    draft_scratch,
                    &hidden_rows,
                    &logits_batch,
                    batch,
                    seed_token,
                    temp,
                    uniforms.as_deref(),
                )?;
                apply_dflash2_selector_proposal(
                    &proposal,
                    vocab,
                    use_temp_sampling,
                    &mut drafted,
                    &mut draft_softmaxes,
                    &mut draft_probs_at_drafted,
                )?;
            } else if use_temp_sampling && fast_sample_active {
                // C8 GPU-sample path: softmax stays device-resident; only
                // draft_tokens + draft_p_at_token (batch×8 bytes) come back.
                // draft_probs_dev is kept alive in c8_draft_probs_dev until
                // chain_accept_spec_f32 consumes it after verify.
                let probs_dev = gpu.alloc_tensor(&[batch * vocab], rdna_compute::DType::F32)?;
                let tau_dev = gpu.alloc_tensor(&[batch], rdna_compute::DType::F32)?;
                let z_dev = gpu.alloc_tensor(&[batch], rdna_compute::DType::F32)?;
                gpu.softmax_temp_topp_batched_into_f32(
                    &logits_batch,
                    &probs_dev,
                    &tau_dev,
                    &z_dev,
                    vocab,
                    batch,
                    temp,
                    top_p,
                    top_k,
                    0.0, // min_p: DFlash min_p parity is the follow-up; off here
                )?;
                // GPU categorical sample per row: writes draft_tokens + draft_p_at_token.
                let tok_dev = gpu.alloc_tensor(&[batch], rdna_compute::DType::F32)?; // i32 via f32 slot
                let pat_dev = gpu.alloc_tensor(&[batch], rdna_compute::DType::F32)?;
                // Per-row seeds: hash of (stream state at window start,
                // absolute position, row) — `spec_draft_row_seed`.
                let seeds: Vec<u8> = (0..batch)
                    .flat_map(|r| {
                        rdna_compute::sampling::spec_draft_row_seed(
                            *rng_state,
                            position as u64,
                            r as u32,
                        )
                        .to_ne_bytes()
                    })
                    .collect();
                let seeds_dev = gpu.upload_raw(&seeds, &[seeds.len()])?;
                gpu.batched_categorical_sample_f32(
                    &probs_dev, &tau_dev, &z_dev, &seeds_dev, &tok_dev, &pat_dev, vocab, batch,
                )?;
                // Download only tokens + probs: batch×8 bytes total.
                let mut raw_tok = vec![0i32; batch];
                {
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(raw_tok.as_mut_ptr() as *mut u8, batch * 4)
                    };
                    gpu.hip.memcpy_dtoh(bytes, &tok_dev.buf)?;
                }
                let raw_pat = gpu.download_f32(&pat_dev)?;
                let _ = gpu.free_tensor(seeds_dev);
                // Keep pat_dev alive on device for chain_accept_spec_f32.
                for i in 0..batch {
                    drafted.push(raw_tok[i] as u32);
                    draft_probs_at_drafted.push(raw_pat[i]);
                }
                // Advance host rng_state by batch steps to maintain entropy across
                // cycles (GPU uses its own LCG; host state must change per cycle).
                for _ in 0..batch {
                    xorshift_next_unit(rng_state);
                }
                // Stash device tensors for chain_accept_spec_f32 in verify step.
                c8_draft_probs_dev = Some(probs_dev);
                c8_draft_tau_dev = Some(tau_dev);
                c8_draft_z_dev = Some(z_dev);
                c8_draft_tokens_dev = Some(tok_dev);
                c8_draft_p_at_token_dev = Some(pat_dev);
                // draft_probs_at_drafted populated above; draft_softmaxes NOT
                // populated (not needed when GPU accept kernel runs).
            } else if use_temp_sampling {
                // Full D2H of (B-1)×vocab logits, CPU softmax+sample.
                let host_logits = gpu.download_f32(&logits_batch)?;
                debug_assert_eq!(host_logits.len(), batch * vocab);
                draft_softmaxes.reserve(batch);
                for i in 0..batch {
                    let row = &host_logits[i * vocab..(i + 1) * vocab];
                    let mut probs = Vec::with_capacity(vocab);
                    softmax_temp_into(row, temp, &mut probs);
                    // Host truncation: the same AR law as the GPU tau/Z cut.
                    apply_host_ar_trunc(&mut probs, top_k, top_p);
                    let u = xorshift_next_unit(rng_state);
                    let t = sample_categorical(&probs, u);
                    draft_probs_at_drafted.push(probs[t as usize]);
                    drafted.push(t);
                    draft_softmaxes.push(probs);
                }
            } else if host_path_active {
                // RP / n-gram-block path: apply per-row penalties before argmax
                // so draft and target pick from the same reshaped distribution.
                // Keeps spec-decode aligned (τ doesn't collapse from mismatched
                // argmaxes — both sides see identical -inf logits on banned toks).
                let host_logits = gpu.download_f32(&logits_batch)?;
                debug_assert_eq!(host_logits.len(), batch * vocab);
                let mut row = vec![0f32; vocab];
                for i in 0..batch {
                    row.copy_from_slice(&host_logits[i * vocab..(i + 1) * vocab]);
                    if rp_active {
                        llama::apply_repeat_penalty(
                            &mut row,
                            prev_committed,
                            repeat_window,
                            repeat_penalty,
                        );
                    }
                    if ngram_block_active {
                        llama::apply_ngram_block(&mut row, prev_committed);
                    }
                    drafted.push(argmax_u32(&row));
                }
            } else {
                // Unreachable: greedy non-host production runs through
                // draft_dflash_block_rank (the `else if` above), so reaching
                // this arm would mean a predicate inversion — fail loud.
                unreachable!("greedy draft head must run in draft_dflash_block_rank");
            }
        } else if selector_mode {
            // lm-head fallback: still go through propose_candidates_host so
            // greedy is not an independent argmax and sampled q stays sparse.
            let batch = b - 1;
            let mut host_logits = Vec::with_capacity(batch * vocab);
            for i in 1..b {
                let hidden_row = draft_scratch.x.sub_offset(i * h, h);
                llama::weight_gemv(gpu, w_out, &hidden_row, &target.scratch.logits)?;
                let logits = gpu.download_f32(&target.scratch.logits)?;
                debug_assert_eq!(logits.len(), vocab);
                host_logits.extend_from_slice(&logits);
            }
            let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
            let uniforms = if use_temp_sampling {
                Some(
                    (0..batch)
                        .map(|_| xorshift_next_unit(rng_state))
                        .collect::<Vec<f32>>(),
                )
            } else {
                None
            };
            let proposal = dflash::propose_candidates_host(
                gpu,
                draft_weights,
                draft_scratch,
                &hidden_rows,
                &host_logits,
                batch,
                seed_token,
                temp,
                uniforms.as_deref(),
            )?;
            apply_dflash2_selector_proposal(
                &proposal,
                vocab,
                use_temp_sampling,
                &mut drafted,
                &mut draft_softmaxes,
                &mut draft_probs_at_drafted,
            )?;
        } else {
            // Fallback: per-row weight_gemv loop.
            for i in 1..b {
                let hidden_row = draft_scratch.x.sub_offset(i * h, h);
                llama::weight_gemv(gpu, w_out, &hidden_row, &target.scratch.logits)?;
                let logits = gpu.download_f32(&target.scratch.logits)?;
                debug_assert_eq!(logits.len(), vocab);
                if use_temp_sampling {
                    let mut probs = Vec::with_capacity(vocab);
                    softmax_temp_into(&logits, temp, &mut probs);
                    apply_host_ar_trunc(&mut probs, top_k, top_p);
                    let u = xorshift_next_unit(rng_state);
                    let t = sample_categorical(&probs, u);
                    draft_probs_at_drafted.push(probs[t as usize]);
                    drafted.push(t);
                    draft_softmaxes.push(probs);
                } else if host_path_active {
                    let mut row = logits.clone();
                    if rp_active {
                        llama::apply_repeat_penalty(
                            &mut row,
                            prev_committed,
                            repeat_window,
                            repeat_penalty,
                        );
                    }
                    if ngram_block_active {
                        llama::apply_ngram_block(&mut row, prev_committed);
                    }
                    drafted.push(argmax_u32(&row));
                } else {
                    drafted.push(argmax_u32(&logits));
                }
            }
        }
    } // close else (DFlash draft path)

    // ── 5a. Verbatim-lookup splice ───────────────────────────────────────
    // From here every stage runs at the effective verify rows `b_eff`; the
    // learned draft above ran at the original `b`. `block` is truncated to
    // `b_eff` before it is filled, so verify/graph B/logit slices/accept/
    // hidden ring/GDN tape+rollback/thlog all see the spliced window.
    let b = match lookup {
        None => b,
        Some(lk) => {
            let mut tail = verify_scratch
                .lookup_tail
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let plan = dflash_lookup_resolve(lk, &drafted, b, vocab, &mut tail).and_then(|keep| {
                if let Some(keep) = keep {
                    if use_temp_sampling {
                        dflash_lookup_check_receipt(q_receipt.as_ref(), keep)?;
                    }
                }
                Ok(keep)
            });
            match plan {
                Err(e) => {
                    *rng_state = rng_at_entry;
                    return Err(e);
                }
                Ok(None) => b,
                Ok(Some(keep)) => {
                    if use_temp_sampling {
                        let receipt = q_receipt.as_mut().expect("receipt checked above");
                        if let Err(e) = dflash_lookup_splice_receipt(receipt, keep, &tail, vocab) {
                            *rng_state = rng_at_entry;
                            return Err(e);
                        }
                    }
                    drafted.truncate(1 + keep);
                    drafted.extend_from_slice(&tail);
                    drafted.len()
                }
            }
        }
    };
    if railgun_verify.is_some() && b != 16 {
        // `NoTailAction` proved the splice declines; a splice here means the
        // proof was wrong after the window already ran. Fail loud.
        return Err(hip_bridge::HipError::new(
            0,
            "railgun Whole16 window: the host splice changed the verify rows (NoTailAction violated)",
        ));
    }
    block.truncate(b);
    for i in 1..b {
        block[i] = drafted[i];
    }

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_draft_end = std::time::Instant::now();

    // ── 5b. N-gram override (DFlash path only) ───────────────────────────
    // When an n-gram cache is supplied, walk the block left-to-right. For
    // each position i, look up the bigram (block[i-2], block[i-1]) → t. If
    // the cache has a high-enough count for t, override block[i] with t.
    // Chained: subsequent lookups use the (possibly-overridden) prior
    // tokens. Chained overrides only "compound" when the cache captures
    // multi-step patterns (e.g. boilerplate phrases, code indentation).
    //
    // Cost: two HashMap lookups per block position = microseconds.
    //
    // Limitation: dflash's draft_forward already ran against the ORIGINAL
    // draft argmax block; overrides don't feed back into the draft. So
    // downstream positions' target-hidden cross-attention was computed
    // against the un-overridden block. In practice this doesn't matter
    // because the per-position target attention at verify time reruns
    // anyway — what matters is target's argmax at position i versus
    // block[i+1] (the override).
    // Skip bigram override when PLD is the draft source: per Goose §3,
    // PLD tokens have 2–18× higher acceptance than bigram (TR) tokens
    // (median 6×). Overriding PLD with a bigram guess strictly lowers τ.
    if pld_spine.is_none() {
        if let Some(ng) = ngram_cache {
            if prev_committed.len() >= 2 {
                let mut a = prev_committed[prev_committed.len() - 2];
                let mut bb = seed_token;
                for i in 1..b {
                    if let Some((tok, _cnt)) = ng.predict(a, bb) {
                        block[i] = tok;
                        // Also reflect the override in `drafted` so the committed
                        // sequence reported back to the caller matches what was
                        // actually verified against the target.
                        drafted[i] = tok;
                    }
                    a = bb;
                    bb = block[i];
                }
            }
        }
    }

    // ── 6. Snapshot DeltaNet pre-verify, run verify (advances state by B) ─
    //
    // If a GdnTape is supplied, the verify forward also records the
    // per-LA-layer (q, k, v, α, β) innovation tape so the rollback can
    // replay just the GDN recurrence for `accept+1` steps without
    // re-running the target.
    if railgun_verify.is_none() {
        target_snap.save_from(&target.dn_state, gpu)?;
    }
    // Mutable variable to allow both verify capture + rollback replay usage.
    let moe_router_logits_present = verify_scratch
        .prefill_batch
        .as_ref()
        .map(|pbs| pbs.moe_router_logits_batch.is_some())
        .unwrap_or(true);
    let verify_populates_tape = qwen35::prefill_batch_pbs_eligible(
        &target.weights,
        &target.config,
        &target.dn_state,
        b,
        gpu.arch.as_str(),
        moe_router_logits_present,
    );
    let use_tape_replay = match &railgun_verify {
        Some(rv) => rv.tape_populated,
        None => dflash_use_gdn_tape_replay(gdn_tape.is_some(), verify_populates_tape),
    };
    let mut gdn_tape_opt = if use_tape_replay { gdn_tape } else { None };

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_verify_start = std::time::Instant::now();
    // Full target logits are D2H'd to the host for rejection sampling, RP, or
    // n-gram block. Under FAST_SAMPLE the rejection sampler reads the
    // GPU-softmaxed target probs directly from the resident
    // `verify_scratch.logits` buffer, so the host full-logit download + host
    // argmax are skipped — that's the target-side half of the cost cut.
    let want_full_logits = (use_temp_sampling && !fast_sample_active) || host_path_active;
    // Railgun D8: a kept full accept relies on the HIP/HipGraph verify body —
    // no route, or a route that planned this window as its ordinary HIP window
    // (`full_hip`/`partial_hip` advanced; prime/capture/PM4 windows do not).
    let mut same_as_hip_graph = verify_pm4.is_none();
    let verify_out = if let Some(rv) = railgun_verify.take() {
        // `Whole16`: the lowered segment ran the verify, the hidden commit and
        // the head; the state is exactly the HIP graph body's iff certified.
        same_as_hip_graph = rv.same_as_hip_graph;
        DflashVerifyOutput {
            argmax_per_pos: rv.argmax_per_pos,
            logits_per_pos: Vec::new(),
        }
    } else {
        match verify_pm4 {
            Some(route) => {
                let replay_failures_before = route.counters().replay_failures;
                let hip_windows =
                    |r: &DflashVerifyPm4| r.counters().full_hip + r.counters().partial_hip;
                let hip_windows_before = hip_windows(route);
                match verify_dflash_block_retained(
                    gpu,
                    target,
                    &block,
                    position,
                    hidden_rb,
                    gdn_tape_opt.as_deref_mut(),
                    want_full_logits,
                    verify_scratch,
                    route,
                ) {
                    Ok(out) => {
                        same_as_hip_graph = hip_windows(route) == hip_windows_before + 1;
                        out
                    }
                    Err(error) => {
                        // Fail closed when quiescence was never proven: a PM4 body
                        // may still be executing against KV, DeltaNet state, and
                        // hidden staging, so restoring or re-running would race it.
                        if matches!(route.phase(), DflashVerifyPm4Phase::Quarantined { .. }) {
                            return Err(error);
                        }
                        // Not a retained-execution failure — an ordinary model
                        // error propagates exactly as it does on the HIP path.
                        if route.counters().replay_failures == replay_failures_before {
                            return Err(error);
                        }
                        // Proven-quiescent replay failure. Rewind the recurrent
                        // state to the pre-window snapshot and redo the window on
                        // the ordinary route; KV slots, hidden staging,
                        // final_hidden, and the GDN tape are all overwritten by the
                        // retry, and the ring is committed exactly once because the
                        // failed attempt never committed.
                        target_snap.restore_to(&mut target.dn_state, gpu)?;
                        route.note_safe_hip_retry();
                        verify_dflash_block(
                            gpu,
                            target,
                            &block,
                            position,
                            hidden_rb,
                            gdn_tape_opt.as_deref_mut(),
                            want_full_logits,
                            verify_scratch,
                        )?
                    }
                }
            }
            None => verify_dflash_block(
                gpu,
                target,
                &block,
                position,
                hidden_rb,
                gdn_tape_opt.as_deref_mut(),
                want_full_logits,
                verify_scratch,
            )?,
        }
    };

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_verify_end = std::time::Instant::now();

    // ── 7. Acceptance ──────────────────────────────────────────────────
    //
    // Greedy path: longest prefix where block[i+1] == argmax_per_pos[i].
    //   bonus = argmax_per_pos[accept_len].
    //
    // Rejection-sampling path (temp > 0):
    //   For each i in 0..B-1:
    //     t = block[i+1] (draft sampled this at position start+i+1)
    //     p_d = draft_softmax[i][t]
    //     p_t = target_softmax[i][t]  (softmax of verify logits row i, same temp)
    //     u = rng
    //     accept if u * p_d < p_t
    //     else: rejected → bonus = sample from (p_target - p_draft)+
    //   If all accepted → bonus = sample from target_softmax[B-1].
    let mut accept_len = 0usize;
    let mut bonus_token;
    if use_temp_sampling {
        // When the GPU sampling path ran on the draft side AND we have the
        // required device tensors, run chain_accept_spec_f32 for the full
        // accept loop (C8b: eliminates the ~9 MB target probs D2H + host loop).
        // Otherwise fall through to the host loop (PLD, fallback per-row path).
        // GPU chain_accept only returns the final boundary draw — when the
        // remaining budget can bind mid-window, force the host path so we can
        // clamp accept_len and re-draw the boundary sample from the target row.
        let budget_binds = matches!(max_accept, Some(m) if m < b - 1);
        let gpu_accept = !selector_mode
            && fast_sample_active
            && !budget_binds
            && c8_draft_probs_dev.is_some()
            && c8_draft_tokens_dev.is_some()
            && c8_draft_p_at_token_dev.is_some();

        // Singleton sampled window on the device (sparse selector / PLD q, no
        // CACTUS, FAST_SAMPLE rows): handles the budget bound itself, so it
        // runs where the C8 chain accept cannot.
        let device_decision: Option<SampledWindowDecision> = match q_receipt.as_ref() {
            Some(r)
                if !gpu_accept
                    && fast_sample_active
                    && dflash_device_accept_eligible(
                        dflash_device_accept_enabled() && gpu.dflash_device_accept_supported(),
                        &DflashSampleCfg {
                            temp,
                            top_p,
                            top_k,
                            cactus_delta,
                        },
                        r,
                        b,
                    ) =>
            {
                Some(dflash_sampled_accept_device(
                    gpu,
                    &verify_scratch.logits,
                    &block,
                    r,
                    vocab,
                    DflashSampleCfg {
                        temp,
                        top_p,
                        top_k,
                        cactus_delta,
                    },
                    max_accept.unwrap_or(usize::MAX),
                    rng_state,
                    &mut *verify_scratch
                        .device_accept
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()),
                )?)
            }
            _ => None,
        };

        if gpu_accept {
            // ── C8 GPU accept path ──────────────────────────────────────────
            // verify_scratch.logits[0..b*vocab] contains the target logits for
            // all b positions (b = draft_batch + 1; last row = bonus position).
            // Softmax them into tgt_probs_dev (kept device-resident).
            let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
            let tgt_probs_dev = gpu.alloc_tensor(&[b * vocab], rdna_compute::DType::F32)?;
            let tau_t_dev = gpu.alloc_tensor(&[b], rdna_compute::DType::F32)?;
            let z_t_dev = gpu.alloc_tensor(&[b], rdna_compute::DType::F32)?;
            gpu.softmax_temp_topp_batched_into_f32(
                &logits_batch,
                &tgt_probs_dev,
                &tau_t_dev,
                &z_t_dev,
                vocab,
                b,
                temp,
                top_p,
                top_k,
                0.0,
            )?;

            let dft_probs_dev = c8_draft_probs_dev.as_ref().unwrap();
            let dft_tok_dev = c8_draft_tokens_dev.as_ref().unwrap();
            let dft_pat_dev = c8_draft_p_at_token_dev.as_ref().unwrap();
            let tau_d_dev = c8_draft_tau_dev.as_ref().unwrap();
            let z_d_dev = c8_draft_z_dev.as_ref().unwrap();

            let out_dev = gpu.alloc_tensor(&[4], rdna_compute::DType::F32)?; // 4×i32 via f32 slot
                                                                             // kernel's b parameter = number of draft comparison positions = b-1
            let draft_b = b - 1;
            let rng_seed = (*rng_state >> 32) as u32 ^ (*rng_state as u32);
            gpu.chain_accept_spec_f32(
                &tgt_probs_dev,
                dft_probs_dev,
                dft_tok_dev,
                dft_pat_dev,
                &tau_t_dev,
                &z_t_dev,
                tau_d_dev,
                z_d_dev,
                &out_dev,
                draft_b,
                vocab,
                rng_seed,
                cactus_delta,
            )?;

            // Download 16 bytes: {accept_len, bonus_token, rejected_at, new_rng}
            let mut out_raw = [0i32; 4];
            {
                let bytes: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(out_raw.as_mut_ptr() as *mut u8, 16) };
                gpu.hip.memcpy_dtoh(bytes, &out_dev.buf)?;
            }
            accept_len = out_raw[0] as usize;
            bonus_token = out_raw[1] as u32;
            // Advance host rng_state once so future cycles seed differently.
            // GPU consumed its own LCG draws; host state just needs to change.
            xorshift_next_unit(rng_state);

            let _ = gpu.free_tensor(out_dev);
            let _ = gpu.free_tensor(tgt_probs_dev);
            let _ = gpu.free_tensor(tau_t_dev);
            let _ = gpu.free_tensor(z_t_dev);
        } else if let Some(decision) = device_decision {
            // ── Device accept path (HIPFIRE_DFLASH_DEVICE_ACCEPT) ───────────
            // Selector / PLD sparse q + resident verify logits: softmax, accept
            // tests, residual and bonus all on the device; the host only reads
            // {accept_len, bonus}. See `dflash_sampled_accept_device`.
            accept_len = decision.accepted_drafts;
            bonus_token = decision.bonus_token;
        } else {
            // ── Host accept path (PLD / fallback per-row) ──────────────────
            debug_assert!(q_receipt.is_some() || draft_softmaxes.len() == b - 1);
            // FAST_SAMPLE: compute the per-row target softmax on the GPU once for
            // all B rows and download the probs. The accept loop below then reads
            // `target_probs` from this buffer instead of calling the host
            // `softmax_temp_into` — identical RNG/accept/residual/CACTUS math,
            // distribution-parity probs. The verify logits are still resident in
            // `verify_scratch.logits[0..b*vocab]` (verify enqueued lm_head into it
            // and only ran GPU-argmax afterwards, which does not overwrite it).
            let fast_tgt: Option<FastTargetRows> = if fast_sample_active {
                let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
                let mut rows = FastTargetRows::default();
                dflash_target_rows_into(
                    gpu,
                    &logits_batch,
                    b,
                    vocab,
                    DflashSampleCfg {
                        temp,
                        top_p,
                        top_k,
                        cactus_delta,
                    },
                    &mut rows,
                )?;
                Some(rows)
            } else {
                debug_assert_eq!(verify_out.logits_per_pos.len(), b * vocab);
                None
            };
            let q = match q_receipt.take() {
                Some(r) => r,
                None => DflashSampledDraft {
                    selected_q: draft_probs_at_drafted,
                    dense_q: draft_softmaxes,
                    ..DflashSampledDraft::default()
                },
            };
            let mut target_probs = Vec::with_capacity(vocab);
            let decision = dflash_sampled_accept_rows(
                &block,
                &q,
                fast_tgt.as_ref(),
                &verify_out.logits_per_pos,
                vocab,
                DflashSampleCfg {
                    temp,
                    top_p,
                    top_k,
                    cactus_delta,
                },
                max_accept.unwrap_or(usize::MAX),
                rng_state,
                &mut target_probs,
            );
            accept_len = decision.accepted_drafts;
            bonus_token = decision.bonus_token;
        }
    } else {
        // Greedy path. If RP or n-gram-block is active, re-derive argmax per
        // row after applying penalties to the full target logits (requires
        // want_full_logits). `prev_committed` carries the emitted history
        // used as the penalty / block window.
        let argmax_per_pos: std::borrow::Cow<'_, [u32]> = if host_path_active {
            let tgt_logits = &verify_out.logits_per_pos;
            debug_assert_eq!(tgt_logits.len(), b * vocab);
            let mut out: Vec<u32> = Vec::with_capacity(b);
            let mut row = vec![0f32; vocab];
            for i in 0..b {
                row.copy_from_slice(&tgt_logits[i * vocab..(i + 1) * vocab]);
                if rp_active {
                    llama::apply_repeat_penalty(
                        &mut row,
                        prev_committed,
                        repeat_window,
                        repeat_penalty,
                    );
                }
                if ngram_block_active {
                    llama::apply_ngram_block(&mut row, prev_committed);
                }
                out.push(argmax_u32(&row));
            }
            std::borrow::Cow::Owned(out)
        } else {
            std::borrow::Cow::Borrowed(verify_out.argmax_per_pos.as_slice())
        };
        // Shared greedy accept-prefix (eos=None: DFlash never early-stops on EOS
        // here — the daemon handles EOS downstream). drafts = block[1..b],
        // target_pick = the (repeat-penalty / n-gram-adjusted) argmax;
        // bonus = target_pick[accept_len].
        let acc = hipfire_runtime::spec::accept_greedy_prefix(&block[1..b], &argmax_per_pos, None);
        accept_len = acc.accepted;
        bonus_token = *acc.committed.last().expect("eos=None yields a bonus");
        // Pre-commit budget: clamp drafts so emit (= accept + 1) ≤ max_emit,
        // then re-pick bonus = argmax at the clamped boundary (temp-0 correct).
        if let Some(m) = max_accept {
            if accept_len > m {
                accept_len = m;
                bonus_token = argmax_per_pos[accept_len];
            }
        }

        if logit_dump_active {
            // Inspect the acceptance boundary. If accept_len < b-1 the block
            // was rejected at position accept_len (drafted loser = block[accept_len+1]);
            // otherwise the whole block accepted and we report the bonus slot.
            let tgt = &verify_out.logits_per_pos;
            let rej_i = accept_len.min(b - 1);
            let row = &tgt[rej_i * vocab..(rej_i + 1) * vocab];
            let (mut top1_idx, mut top1, mut top2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
            for (j, &v) in row.iter().enumerate() {
                if v > top1 {
                    top2 = top1;
                    top1 = v;
                    top1_idx = j;
                } else if v > top2 {
                    top2 = v;
                }
            }
            let rejected = accept_len < b - 1;
            let rej_tok: i64 = if rejected {
                block[accept_len + 1] as i64
            } else {
                -1
            };
            let (rej_logit, gap, rej_rank) = if rejected {
                let rv = row[rej_tok as usize];
                let rank = row.iter().filter(|&&v| v > rv).count() + 1;
                (rv, top1 - rv, rank as i64)
            } else {
                (f32::NAN, f32::NAN, -1)
            };
            eprintln!(
                "DFLDUMP pos={} b={} accept={} rejected={} top1_tok={} top1={:.5} top2={:.5} margin={:.5} rej_tok={} rej_logit={:.5} gap={:.5} rej_rank={} bonus={}",
                position, b, accept_len, rejected as u8, top1_idx, top1, top2,
                top1 - top2, rej_tok, rej_logit, gap, rej_rank, bonus_token
            );
        }
    }

    // Free C8 device tensors now that accept is resolved.
    if let Some(t) = c8_draft_probs_dev {
        let _ = gpu.free_tensor(t);
    }
    if let Some(t) = c8_draft_tau_dev {
        let _ = gpu.free_tensor(t);
    }
    if let Some(t) = c8_draft_z_dev {
        let _ = gpu.free_tensor(t);
    }
    if let Some(t) = c8_draft_tokens_dev {
        let _ = gpu.free_tensor(t);
    }
    if let Some(t) = c8_draft_p_at_token_dev {
        let _ = gpu.free_tensor(t);
    }

    // ── 7b. Seed-prediction oracle (Task #93 Phase B) ───────────────────
    // Three position-based proxies for the next cycle's `seed_token`
    // (= this cycle's `bonus_token`). See comment at top of file for the
    // reasoning — the PRD's "naive argmax at rejection boundary" proxy is
    // 0 % by construction (the accept loop broke precisely there), which
    // we measure as REJ_MATCH to document the dead-end. TAIL_MATCH and
    // ANYPOS_MATCH are the actually-usable ceilings.
    let rej_proxy: Option<u32> = if accept_len + 1 < b {
        Some(drafted[accept_len + 1])
    } else {
        None
    };
    let tail_proxy: u32 = drafted[b - 1];
    let anypos_hit: bool = drafted[1..b].iter().any(|&t| t == bonus_token);
    let rej_hit: bool = rej_proxy == Some(bonus_token);
    let tail_hit: bool = tail_proxy == bonus_token;
    SEED_ORACLE_TOTAL.fetch_add(1, Ordering::Relaxed);
    SEED_ORACLE_ACCEPT_LEN_SUM.fetch_add(accept_len as u64, Ordering::Relaxed);
    if rej_hit {
        SEED_ORACLE_REJ_MATCH.fetch_add(1, Ordering::Relaxed);
    }
    if tail_hit {
        SEED_ORACLE_TAIL_MATCH.fetch_add(1, Ordering::Relaxed);
    }
    if anypos_hit {
        SEED_ORACLE_ANYPOS_MATCH.fetch_add(1, Ordering::Relaxed);
    }
    if rej_proxy.is_none() {
        SEED_ORACLE_FULLACCEPT.fetch_add(1, Ordering::Relaxed);
    }
    if hipfire_config::developer_var("HIPFIRE_DFLASH_SEED_ORACLE")
        .ok()
        .as_deref()
        == Some("1")
    {
        let s = read_seed_oracle_stats();
        let denom = s.total.max(1) as f32;
        eprintln!(
            "[seed-oracle] cycle: accept_len={} b={} bonus={} rej={:?}/{} tail={}/{} anypos={} fullacc={} | cum rej={:.3} tail={:.3} anypos={:.3} mean_accept={:.2}",
            accept_len, b, bonus_token, rej_proxy, rej_hit,
            tail_proxy, tail_hit, anypos_hit, rej_proxy.is_none(),
            s.rej_match as f32 / denom,
            s.tail_match as f32 / denom,
            s.anypos_match as f32 / denom,
            s.accept_len_sum as f32 / denom,
        );
    }

    // ── 8-10. Committed sequence → hidden scatter / CPU shadow → DeltaNet
    // rewind + replay ────────────────────────────────────────────────────
    // One shared body (`dflash_commit_decision_borrowed`, lifted verbatim from
    // the pre-factor lines 7057-7240 of this function) serves greedy and
    // sampled decisions here and the lane commits. `accept_len`/`bonus_token`
    // are already budgeted by the arm above; nothing re-clamps or re-draws.
    // The bonus token's hidden is deliberately NOT appended (its ring slot
    // holds a rejected draft's forward); the next verify forwards it as
    // block[0]. A CPU shadow (`ctx_slice`) extends `target_hidden_host`
    // instead of scattering + thlog append.
    let decision = SampledWindowDecision {
        accepted_drafts: accept_len,
        bonus_token,
        committed_len: accept_len + 2,
    };
    let (committed, commit_times) = {
        let mut target_parts = DflashTargetParts::from_slot(target);
        dflash_commit_decision_borrowed(
            gpu,
            &mut target_parts,
            DflashBorrowedCommit {
                hidden_rb: &*hidden_rb,
                draft_scratch: &mut *draft_scratch,
                target_snap: &*target_snap,
                gdn_tape: gdn_tape_opt.as_deref(),
                cpu_shadow: if ctx_slice.is_some() {
                    Some(DflashCpuShadow {
                        host: &mut *target_hidden_host,
                        ne,
                        h,
                    })
                } else {
                    None
                },
            },
            DflashCommitWindow {
                position,
                seed_token,
                block: &block,
                decision,
                same_as_hip_graph,
                phase_on,
            },
        )?
    };
    // Target state is now at position + accept_len + 1. KV cache has
    // written K/V at positions [position..position+accept_len]. The bonus
    // token's K/V will be written on the next iter's verify (at position
    // `position + accept_len + 1`) as part of that iter's block[0] forward.

    if phase_on {
        gpu.hip.device_synchronize()?;
        let t_end = std::time::Instant::now();
        let us_draft = t_draft_end.duration_since(t_spec_start).as_micros();
        let us_ngram = t_verify_start.duration_since(t_draft_end).as_micros();
        let us_verify = t_verify_end.duration_since(t_verify_start).as_micros();
        let us_accept = commit_times
            .accept_end
            .duration_since(t_verify_end)
            .as_micros();
        let us_scatter = commit_times
            .scatter_end
            .duration_since(commit_times.accept_end)
            .as_micros();
        let us_restore = commit_times
            .restore_end
            .duration_since(commit_times.scatter_end)
            .as_micros();
        let us_replay = t_end.duration_since(commit_times.restore_end).as_micros();
        let us_total = t_end.duration_since(t_spec_start).as_micros();
        eprintln!(
            "[phase] B={} accept={} draft={}µs ngram={}µs verify={}µs \
             cmpr={}µs scatter={}µs restore={}µs replay={}µs | total={}µs",
            b,
            accept_len,
            us_draft,
            us_ngram,
            us_verify,
            us_accept,
            us_scatter,
            us_restore,
            us_replay,
            us_total,
        );
    }
    let _ = (t_phase, t_draft_end, t_verify_start, t_verify_end);
    // A chain window that ran on the ordinary HIP route has materialized the
    // lazy scratch/JIT of its shape family: later windows of the family may
    // be planned (never before: planning refuses a `hipModuleLoad`/`hipMalloc`).
    if let (Some(rg), Some(shape), false, 16) =
        (railgun.as_mut(), railgun_shape.as_ref(), railgun_ran, b)
    {
        rg.owner.note_hip_window(shape);
    }

    Ok(SpecStepResult {
        accepted: accept_len,
        bonus_token,
        drafted,
        committed,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense-TP (TP=2) DFlash2 mesh transaction — eager, greedy, B=16 only
// ═══════════════════════════════════════════════════════════════════════════
//
// Mirror of the single-GPU `spec_step_dflash` transaction across two target
// ranks: duplicated greedy draft on both GPUs (exact ID agreement gate, no
// draft collective), dense-TP verify with two rooted reductions per layer,
// rank-0-authoritative accept, per-rank snapshot restore + tape replay, and
// matched commit receipts before anything is published. Rank 1 never runs
// target final norm / LM head and never advances a host cursor.

/// Borrowed dense target mesh for one DFlash window. All rank slices must
/// have exactly two entries; every rank-local config/weight/KV/DN/scratch
/// stays on its own device (never concatenated or transferred).
pub struct DenseTpTargetView<'a> {
    pub shard: &'a ShardConfig,
    pub weights: &'a [Qwen35Weights],
    pub configs: &'a [Qwen35Config],
    pub kv_caches: &'a mut [KvCache],
    pub dn_states: &'a mut [DeltaNetState],
    pub scratches: &'a [Qwen35Scratch],
}

/// Frozen mesh admission: exactly two ranks everywhere, dense configs that
/// agree on global shape with `num_experts == 0`, DFlash2 all-sliding draft
/// with candidate selector at fixed B=16 on both ranks, and identical
/// capture-layer lists. CPU-only — runs before any GPU mutation. Returns the
/// agreed block size.
fn dense_tp_dflash_gates(
    gpus: &Gpus,
    target: &DenseTpTargetView<'_>,
    states: &[DenseTpDflashRankState],
) -> Result<usize, String> {
    let tp = 2usize;
    if gpus.devices.len() != tp
        || target.weights.len() != tp
        || target.configs.len() != tp
        || target.kv_caches.len() != tp
        || target.dn_states.len() != tp
        || target.scratches.len() != tp
        || states.len() != tp
        || target.shard.tp_size != tp
    {
        return Err("dense TP DFlash requires exactly 2 ranks in every mesh slot".into());
    }
    let dim = target.configs[0].dim;
    let n_layers = target.configs[0].n_layers;
    for cfg in target.configs.iter() {
        if cfg.dim != dim
            || cfg.n_layers != n_layers
            || cfg.layer_types != target.configs[0].layer_types
        {
            return Err("dense TP DFlash configs diverge on global shape".into());
        }
        if cfg.num_experts != 0 {
            return Err("dense TP DFlash requires num_experts == 0 (dense target)".into());
        }
    }
    let b0 = states[0].dflash.block_size;
    for s in states.iter() {
        if s.dflash.block_size != b0 {
            return Err("dense TP DFlash rank draft block sizes diverge".into());
        }
        if !s.dflash.draft_config.all_layers_sliding
            || !s.dflash.draft_weights.has_candidate_selector()
        {
            return Err("dense TP DFlash requires a DFlash2 all-sliding selector draft".into());
        }
        if s.dflash.draft_config.target_layer_ids.len() != s.dflash.hidden_rb.extract_layers.len() {
            return Err("dense TP DFlash rank hidden ring disagrees with draft layers".into());
        }
    }
    if b0 != 16 {
        return Err(format!(
            "dense TP DFlash requires fixed B=16 (rank block_size={b0})"
        ));
    }
    if states[0].dflash.hidden_rb.extract_layers != states[1].dflash.hidden_rb.extract_layers {
        return Err("dense TP DFlash rank capture layer lists diverge".into());
    }
    if states[0].dflash.draft_config.target_layer_ids
        != states[1].dflash.draft_config.target_layer_ids
    {
        return Err("dense TP DFlash rank draft target_layer_ids diverge".into());
    }
    // Draft log sync: both replicas must cover the same committed prefix.
    if states[0].dflash.draft_scratch.thlog.abs_positions().len()
        != states[1].dflash.draft_scratch.thlog.abs_positions().len()
    {
        return Err("dense TP DFlash rank draft logs cover different prefixes".into());
    }
    if target.kv_caches[0].compact_offset != target.kv_caches[1].compact_offset {
        return Err("dense TP DFlash rank KV compact_offsets diverge".into());
    }
    Ok(b0)
}

/// Reset every rank to cold state: zero DeltaNet recurrence, rewind KV
/// compaction, clear hidden rings / draft upload tracking, and
/// invalidate window receipts. Drains each rank stream so a poisoned window
/// cannot leak kernels past the reset.
fn reset_dense_tp_dflash_ranks(
    gpus: &mut Gpus,
    kv_caches: &mut [KvCache],
    dn_states: &mut [DeltaNetState],
    states: &mut [DenseTpDflashRankState],
) -> HipResult<()> {
    for rank in 0..gpus.devices.len() {
        gpus.devices[rank].bind_thread()?;
        dn_states[rank].reset(&mut gpus.devices[rank])?;
        kv_caches[rank].compact_offset = 0;
        states[rank].dflash.hidden_rb.reset();
        states[rank].dflash.draft_scratch.reset_upload_tracking();
        states[rank].logical_pos = 0;
        states[rank].receipt_generation = u64::MAX;
    }
    for gpu in gpus.devices.iter() {
        gpu.bind_thread()?;
        gpu.hip.device_synchronize()?;
    }
    Ok(())
}

/// Cold prompt prefill across both target ranks with hidden capture into each
/// rank's ring, prompt-row scatter into each rank's draft cache, and greedy
/// first-token selection from rank-0 target logits. Chunks the prompt at the
/// persistent rank PBS width. Returns the first emitted token. On abort,
/// ranks are left fully reset and an `"aborted"` error is returned.
pub fn seed_target_hidden_dense_tp2_abortable(
    gpus: &mut Gpus,
    mut target: DenseTpTargetView<'_>,
    states: &mut [DenseTpDflashRankState],
    prompt: &[u32],
    abort: &dyn Fn() -> bool,
) -> Result<u32, String> {
    let b = dense_tp_dflash_gates(gpus, &target, states)?;
    if prompt.is_empty() {
        return Err("dense TP DFlash seed requires a non-empty prompt".into());
    }
    let dim = target.configs[0].dim;
    let vocab = target.configs[0].vocab_size;
    if target.configs[1].vocab_size != vocab {
        return Err("dense TP DFlash rank vocabs diverge".into());
    }
    let cap = states[0]
        .dflash
        .verify_scratch
        .prefill_batch
        .as_ref()
        .map(|p| p.max_batch)
        .unwrap_or(0);
    if cap < b {
        return Err("dense TP DFlash rank PBS smaller than the draft block".into());
    }
    if states[1]
        .dflash
        .verify_scratch
        .prefill_batch
        .as_ref()
        .map(|p| p.max_batch)
        .unwrap_or(0)
        != cap
    {
        return Err("dense TP DFlash rank PBS widths diverge".into());
    }
    let ring_room = states[0].dflash.hidden_rb.max_positions;
    if prompt.len() + b > ring_room {
        return Err(format!(
            "dense TP DFlash prompt ({} tokens) exceeds hidden ring room {}",
            prompt.len(),
            ring_room
        ));
    }
    for kv in target.kv_caches.iter() {
        if prompt.len() + b > kv.max_seq {
            return Err("dense TP DFlash prompt exceeds rank KV max_seq".into());
        }
    }
    // All CPU gates passed — now mutate device state.
    reset_dense_tp_dflash_ranks(gpus, &mut *target.kv_caches, &mut *target.dn_states, states)
        .map_err(|e| e.to_string())?;
    let mut off = 0usize;
    let mut last_n = 0usize;
    while off < prompt.len() {
        if abort() {
            let _ = reset_dense_tp_dflash_ranks(
                gpus,
                &mut *target.kv_caches,
                &mut *target.dn_states,
                states,
            );
            return Err("dense TP DFlash seed aborted".into());
        }
        let n = (prompt.len() - off).min(cap);
        last_n = n;
        {
            let [s0, s1] = states else {
                return Err("dense TP DFlash seed lost a rank".into());
            };
            let pbs0 = s0
                .dflash
                .verify_scratch
                .prefill_batch
                .as_ref()
                .ok_or("dense TP DFlash rank 0 lost its persistent PBS")?;
            let pbs1 = s1
                .dflash
                .verify_scratch
                .prefill_batch
                .as_ref()
                .ok_or("dense TP DFlash rank 1 lost its persistent PBS")?;
            let fh = s0.dflash.verify_scratch.final_hidden.sub_offset(0, n * dim);
            let mut caps = [
                DenseTpSpecCapture {
                    hidden: &mut s0.dflash.hidden_rb,
                    tape: Some(&mut s0.dflash.gdn_tape),
                },
                DenseTpSpecCapture {
                    hidden: &mut s1.dflash.hidden_rb,
                    tape: Some(&mut s1.dflash.gdn_tape),
                },
            ];
            forward_prefill_dense_tp_verify_capture(
                gpus,
                target.shard,
                target.weights,
                target.configs,
                &prompt[off..off + n],
                off,
                &mut *target.kv_caches,
                &mut *target.dn_states,
                target.scratches,
                &[pbs0, pbs1],
                &[&s0.tp_partial, &s1.tp_partial],
                &mut caps,
                &fh,
                None,
            )
            .map_err(|e| e.to_string())?;
        }
        off += n;
    }
    // Prime each rank's draft target-hidden cache + upload log, and keep a
    // host shadow for the (all-sliding no-op) backfill call below.
    for rank in 0..2 {
        gpus.devices[rank]
            .bind_thread()
            .map_err(|e| e.to_string())?;
        let modulus = states[rank].dflash.draft_scratch.ctx_modulus();
        scatter_hidden_block_to_interleaved(
            &gpus.devices[rank],
            &states[rank].dflash.hidden_rb,
            &states[rank].dflash.draft_scratch.target_hidden,
            0,
            prompt.len(),
            prompt.len(),
            modulus,
        )
        .map_err(|e| e.to_string())?;
        let host = download_hidden_block(
            &gpus.devices[rank],
            &states[rank].dflash.hidden_rb,
            prompt.len(),
        )
        .map_err(|e| e.to_string())?;
        hipfire_runtime::dflash::draft_seed_backfill(
            &mut gpus.devices[rank],
            &states[rank].dflash.draft_weights,
            &states[rank].dflash.draft_config,
            &mut states[rank].dflash.draft_scratch,
            &host,
            prompt.len(),
        )
        .map_err(|e| e.to_string())?;
        states[rank]
            .dflash
            .draft_scratch
            .thlog
            .seed_prompt(prompt.len());
        states[rank].logical_pos = prompt.len();
        states[rank].receipt_generation = u64::MAX;
    }
    // Greedy first token from rank-0 target logits over the last prompt row:
    // same norm + LM-head epilogue as the dense-TP AR prefill tail.
    let first_token = {
        gpus.devices[0].bind_thread().map_err(|e| e.to_string())?;
        let last_off = (last_n - 1) * dim;
        let x_last = states[0]
            .dflash
            .verify_scratch
            .final_hidden
            .sub_offset(last_off, dim);
        gpus.devices[0]
            .rmsnorm_f32(
                &x_last,
                &target.weights[0].output_norm,
                &target.scratches[0].tmp,
                target.configs[0].norm_eps,
            )
            .map_err(|e| e.to_string())?;
        let logits_row = states[0].dflash.verify_scratch.logits.sub_offset(0, vocab);
        llama::weight_gemv(
            &mut gpus.devices[0],
            &target.weights[0].output,
            &target.scratches[0].tmp,
            &logits_row,
        )
        .map_err(|e| e.to_string())?;
        let row = gpus.devices[0]
            .download_f32(&logits_row)
            .map_err(|e| e.to_string())?;
        row.iter()
            .enumerate()
            .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
                if v > bv {
                    (i as u32, v)
                } else {
                    (best, bv)
                }
            })
            .0
    };
    Ok(first_token)
}

/// Dense-TP target verify for one `[position, position+B)` window: two rooted
/// reductions per layer on both ranks, per-rank tape + hidden capture, and
/// rank-0 greedy predictions per row. Requires both ranks' logical cursors at
/// `position`. Returns the B rank-0 target picks (bonus slot included).
pub fn verify_dflash_block_dense_tp2(
    gpus: &mut Gpus,
    mut target: DenseTpTargetView<'_>,
    states: &mut [DenseTpDflashRankState],
    block: &[u32],
    position: usize,
    _generation: u64,
) -> HipResult<Vec<u32>> {
    let b = dense_tp_dflash_gates(gpus, &target, states).map_err(|e| HipError::new(0, &e))?;
    if block.len() != b {
        return Err(HipError::new(
            0,
            "dense TP DFlash verify block must be B rows",
        ));
    }
    for s in states.iter() {
        if s.logical_pos != position {
            return Err(HipError::new(
                0,
                "dense TP DFlash rank logical cursor != verify position",
            ));
        }
    }
    let dim = target.configs[0].dim;
    let vocab = target.configs[0].vocab_size;
    // Peer-direct reduction needs a live stream on both ranks; otherwise the
    // rooted fallback still runs (slower, same arithmetic order).
    for gpu in gpus.devices.iter_mut() {
        gpu.bind_thread()?;
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create()?);
        }
    }
    // Two-phase VMM preflight: map both rank KVs before any target mutation.
    let required = position
        .checked_add(b)
        .ok_or_else(|| HipError::new(0, "dense TP DFlash verify KV range overflow"))?;
    for rank in 0..2 {
        gpus.devices[rank].bind_thread()?;
        target.kv_caches[rank].ensure_mapped_capacity(&mut gpus.devices[rank], required)?;
    }
    {
        let [s0, s1] = states else {
            return Err(HipError::new(0, "dense TP DFlash verify lost a rank"));
        };
        let pbs0 = s0
            .dflash
            .verify_scratch
            .prefill_batch
            .as_ref()
            .ok_or_else(|| HipError::new(0, "dense TP DFlash rank 0 lost its PBS"))?;
        let pbs1 = s1
            .dflash
            .verify_scratch
            .prefill_batch
            .as_ref()
            .ok_or_else(|| HipError::new(0, "dense TP DFlash rank 1 lost its PBS"))?;
        let fh = s0.dflash.verify_scratch.final_hidden.sub_offset(0, b * dim);
        let mut caps = [
            DenseTpSpecCapture {
                hidden: &mut s0.dflash.hidden_rb,
                tape: Some(&mut s0.dflash.gdn_tape),
            },
            DenseTpSpecCapture {
                hidden: &mut s1.dflash.hidden_rb,
                tape: Some(&mut s1.dflash.gdn_tape),
            },
        ];
        forward_prefill_dense_tp_verify_capture(
            gpus,
            target.shard,
            target.weights,
            target.configs,
            block,
            position,
            &mut *target.kv_caches,
            &mut *target.dn_states,
            target.scratches,
            &[pbs0, pbs1],
            &[&s0.tp_partial, &s1.tp_partial],
            &mut caps,
            &fh,
            None,
        )?;
    }
    // Rank-0 final norm (per row; the single-GPU batch path norms inside its
    // forward) then the batched LM head + GPU argmax — same helper the
    // single-GPU verify tail uses.
    gpus.devices[0].bind_thread()?;
    // Rank-0 final norm, one row at a time: source rows come straight from
    // the rank-0 PBS residual (distinct from the norm destination, so no
    // in-place hazard), landing normed into final_hidden for the head below.
    // This mirrors the single-GPU verify, which norms inside its forward.
    let pbs0 = states[0]
        .dflash
        .verify_scratch
        .prefill_batch
        .as_ref()
        .ok_or_else(|| HipError::new(0, "dense TP DFlash rank 0 lost its PBS"))?;
    for i in 0..b {
        let src = pbs0.x_batch.sub_offset(i * dim, dim);
        let dst = states[0]
            .dflash
            .verify_scratch
            .final_hidden
            .sub_offset(i * dim, dim);
        gpus.devices[0].rmsnorm_f32(
            &src,
            &target.weights[0].output_norm,
            &dst,
            target.configs[0].norm_eps,
        )?;
    }
    // final_hidden now holds post-norm rows; the head reads them below.
    let w_out = &target.weights[0].output;
    let mut argmax_per_pos: Vec<u32> = Vec::with_capacity(b);
    if dflash_batched_lm_head_supported(w_out.gpu_dtype) {
        let fh = states[0]
            .dflash
            .verify_scratch
            .final_hidden
            .sub_offset(0, b * dim);
        dflash_enqueue_verify_lm_head(
            &mut gpus.devices[0],
            w_out,
            &fh,
            &states[0].dflash.verify_scratch,
            b,
            vocab,
            DenseBatchMath::Product,
        )?;
        let argmax_buf = states[0].dflash.verify_scratch.argmax.sub_offset(0, b);
        gpus.devices[0].argmax_f32_batched(
            &states[0]
                .dflash
                .verify_scratch
                .logits
                .sub_offset(0, b * vocab),
            &argmax_buf,
            vocab,
            b,
        )?;
        argmax_per_pos =
            dflash_download_verify_argmax(&gpus.devices[0], &states[0].dflash.verify_scratch, b)?;
    } else {
        for i in 0..b {
            let hidden_row = states[0]
                .dflash
                .verify_scratch
                .final_hidden
                .sub_offset(i * dim, dim);
            let logits_row = states[0]
                .dflash
                .verify_scratch
                .logits
                .sub_offset(i * vocab, vocab);
            llama::weight_gemv(
                &mut gpus.devices[0],
                &target.weights[0].output,
                &hidden_row,
                &logits_row,
            )?;
            let row = gpus.devices[0].download_f32(&logits_row)?;
            argmax_per_pos.push(argmax_u32(&row));
        }
    }
    Ok(argmax_per_pos)
}

/// One dense-TP DFlash2 cycle. Greedy-only by construction (no sampling
/// knobs exist on this path). Drafts B-1 tokens independently on both ranks
/// and requires byte-identical candidate vectors before the target may
/// mutate; on mismatch both ranks reset and the window fails with no output.
/// Accept/replay/commit mirror `spec_step_dflash` with the pending-bonus
/// convention: target state ends at `position + accept + 1`, and only
/// `committed[1..]` is new output. Returns after both ranks report the
/// matching `(generation, position + accept + 1)` commit receipt.
pub fn spec_step_dflash_dense_tp2(
    gpus: &mut Gpus,
    mut target: DenseTpTargetView<'_>,
    states: &mut [DenseTpDflashRankState],
    position: usize,
    seed_token: u32,
    generation: u64,
) -> HipResult<SpecStepResult> {
    let b = dense_tp_dflash_gates(gpus, &target, states).map_err(|e| HipError::new(0, &e))?;
    for s in states.iter() {
        if s.logical_pos != position {
            return Err(HipError::new(
                0,
                "dense TP DFlash step cursor != rank logical position",
            ));
        }
    }
    let vocab = target.configs[0].vocab_size;
    let mask_token = states[0].dflash.draft_config.mask_token_id;
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;
    // HIPFIRE_SPEC_PHASES=1: mesh phase breakdown (diagnostic; device-sync per
    // boundary so wall time reflects GPU completion, like the single-GPU path).
    static MESH_PHASE_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let mesh_phase_on = *MESH_PHASE_ON.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_SPEC_PHASES")
            .ok()
            .as_deref()
            == Some("1")
    });
    macro_rules! mesh_sync {
        () => {
            if mesh_phase_on {
                for gpu in gpus.devices.iter() {
                    gpu.bind_thread()?;
                    gpu.hip.device_synchronize()?;
                }
            }
        };
    }
    mesh_sync!();
    let t_mesh_start = std::time::Instant::now();
    // ── Draft on both ranks (no target mutation yet) ──
    let mut drafted_per_rank: Vec<Vec<u32>> = Vec::with_capacity(2);
    for rank in 0..2 {
        let co = target.kv_caches[rank].compact_offset as i32;
        let drafted = draft_dflash_block_rank(
            &mut gpus.devices[rank],
            &target.weights[rank],
            &states[rank].dflash.draft_weights,
            &states[rank].dflash.draft_config,
            &mut states[rank].dflash.draft_scratch,
            &states[rank].dflash.verify_scratch,
            &[],
            &block,
            position,
            co,
            None,
            false,
            seed_token,
            vocab,
        )?;
        drafted_per_rank.push(drafted);
    }
    if drafted_per_rank[0] != drafted_per_rank[1] {
        // Fail before any target mutation; all-rank reset, no output.
        let _ = reset_dense_tp_dflash_ranks(
            gpus,
            &mut *target.kv_caches,
            &mut *target.dn_states,
            states,
        );
        return Err(HipError::new(
            0,
            "dense TP DFlash rank draft candidates diverge — mesh reset, no tokens published",
        ));
    }
    let drafted = drafted_per_rank
        .into_iter()
        .next()
        .expect("two rank drafts");
    for i in 1..b {
        block[i] = drafted[i];
    }
    mesh_sync!();
    let t_mesh_draft = std::time::Instant::now();
    // ── VMM preflight on both ranks before snapshot/verify ──
    let required = position
        .checked_add(b)
        .ok_or_else(|| HipError::new(0, "dense TP DFlash step KV range overflow"))?;
    for rank in 0..2 {
        gpus.devices[rank].bind_thread()?;
        target.kv_caches[rank].ensure_mapped_capacity(&mut gpus.devices[rank], required)?;
    }
    // ── Snapshot both ranks, tagged with this window generation ──
    for rank in 0..2 {
        gpus.devices[rank].bind_thread()?;
        states[rank]
            .dflash
            .target_snap
            .save_from(&target.dn_states[rank], &mut gpus.devices[rank])?;
        states[rank].receipt_generation = generation;
    }
    // ── Verify (tentatively advances both ranks by B) ──
    let argmax_per_pos = verify_dflash_block_dense_tp2(
        gpus,
        DenseTpTargetView {
            shard: target.shard,
            weights: target.weights,
            configs: target.configs,
            kv_caches: &mut *target.kv_caches,
            dn_states: &mut *target.dn_states,
            scratches: target.scratches,
        },
        states,
        &block,
        position,
        generation,
    )?;
    mesh_sync!();
    let t_mesh_verify = std::time::Instant::now();
    // ── Accept on rank-0 picks (shared greedy math, no fork) ──
    let acc = hipfire_runtime::spec::accept_greedy_prefix(&block[1..b], &argmax_per_pos, None);
    let accept_len = acc.accepted;
    let bonus_token = *acc.committed.last().expect("eos=None yields a bonus");
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 2);
    committed.push(seed_token);
    for i in 0..accept_len {
        committed.push(drafted[i + 1]);
    }
    committed.push(bonus_token);
    let t_mesh_accept = std::time::Instant::now();
    // ── Commit a+1 hidden rows into each rank's own draft cache ──
    let rows_to_keep = accept_len + 1;
    for rank in 0..2 {
        let modulus = states[rank].dflash.draft_scratch.ctx_modulus();
        scatter_hidden_block_to_interleaved(
            &gpus.devices[rank],
            &states[rank].dflash.hidden_rb,
            &states[rank].dflash.draft_scratch.target_hidden,
            position,
            b,
            rows_to_keep,
            modulus,
        )?;
        let co = target.kv_caches[rank].compact_offset as i32;
        states[rank]
            .dflash
            .draft_scratch
            .thlog
            .append_committed(position, rows_to_keep, co);
    }
    mesh_sync!();
    let t_mesh_scatter = std::time::Instant::now();
    // ── Restore + replay both ranks to P+a+1 ──
    // Tape replay needs the same PBS-eligibility the single-GPU path gates
    // on; without it, fall back to the shared dense-TP re-forward over the
    // committed prefix (slower, exact).
    let mut use_tape = true;
    for rank in 0..2 {
        let moe_router = states[rank]
            .dflash
            .verify_scratch
            .prefill_batch
            .as_ref()
            .map(|p| p.moe_router_logits_batch.is_some())
            .unwrap_or(true);
        if !qwen35::prefill_batch_pbs_eligible(
            &target.weights[rank],
            &target.configs[rank],
            &target.dn_states[rank],
            b,
            gpus.devices[rank].arch.as_str(),
            moe_router,
        ) {
            use_tape = false;
        }
    }
    for rank in 0..2 {
        gpus.devices[rank].bind_thread()?;
        states[rank]
            .dflash
            .target_snap
            .restore_to(&mut target.dn_states[rank], &mut gpus.devices[rank])?;
        if use_tape {
            states[rank].dflash.gdn_tape.replay_gdn(
                &mut gpus.devices[rank],
                &target.weights[rank],
                &target.configs[rank],
                &mut target.dn_states[rank],
                accept_len + 1,
            )?;
        }
    }
    mesh_sync!();
    let t_mesh_restore = std::time::Instant::now();
    if !use_tape {
        let replay_tokens = committed[..accept_len + 1].to_vec();
        forward_prefill_dense_tp(
            gpus,
            target.shard,
            target.weights,
            target.configs,
            &replay_tokens,
            position,
            &mut *target.kv_caches,
            &mut *target.dn_states,
            target.scratches,
        )?;
    }
    mesh_sync!();
    let t_mesh_replay = std::time::Instant::now();
    if mesh_phase_on {
        for gpu in gpus.devices.iter() {
            gpu.bind_thread()?;
            gpu.hip.device_synchronize()?;
        }
        let t_mesh_end = std::time::Instant::now();
        eprintln!(
            "[phase-tp2] B={} accept={} draft={}µs verify={}µs accept={}µs scatter={}µs restore={}µs replay={}µs | total={}µs",
            b,
            accept_len,
            t_mesh_draft.duration_since(t_mesh_start).as_micros(),
            t_mesh_verify.duration_since(t_mesh_draft).as_micros(),
            t_mesh_accept.duration_since(t_mesh_verify).as_micros(),
            t_mesh_scatter.duration_since(t_mesh_accept).as_micros(),
            t_mesh_restore.duration_since(t_mesh_scatter).as_micros(),
            t_mesh_replay.duration_since(t_mesh_restore).as_micros(),
            t_mesh_end.duration_since(t_mesh_start).as_micros(),
        );
    }
    // ── Matched commit receipts gate publication ──
    for rank in 0..2 {
        states[rank].logical_pos = position + accept_len + 1;
    }
    if states[0].logical_pos != states[1].logical_pos
        || states[0].receipt_generation != generation
        || states[1].receipt_generation != generation
    {
        return Err(HipError::new(0, "dense TP DFlash commit receipts mismatch"));
    }
    // Both ranks acknowledged generation g at P+a+1 on their own streams;
    // drain them so generation g+1 cannot reuse scratch/tape/ring early.
    for gpu in gpus.devices.iter() {
        gpu.bind_thread()?;
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(stream)?;
        } else {
            gpu.hip.device_synchronize()?;
        }
    }
    Ok(SpecStepResult {
        accepted: accept_len,
        bonus_token,
        drafted,
        committed,
    })
}

/// Run the DFlash draft forward + lm_head, return the raw per-position draft
/// logits as a host `Vec<f32>` of length `(b - 1) * vocab`.
///
/// Shared factor-out of the draft-producing half of spec_step_dflash — used by
/// spec_step_ddtree to feed Algorithm 1 with per-position top-K. The vanilla
/// DFlash path doesn't call this because it takes the argmax/softmax directly
/// on GPU (smaller D2H); the tree path needs raw logits for top-K + log-norm.
///
/// Leaves `draft_scratch.x` populated with draft hidden rows, so callers that
/// also want argmax for diagnostics can walk those rows afterward (not used
/// here). Does NOT advance the target KV cache or DeltaNet state — only the
/// draft forward runs.
#[cfg(feature = "deltanet")]
fn run_dflash_draft_for_logits(
    gpu: &mut Gpu,
    target: &ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    target_hidden_host: &[f32],
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    b: usize,
) -> HipResult<Vec<f32>> {
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let vocab = target.config.vocab_size;
    let mask_token = draft_cfg.mask_token_id;
    assert!(b >= 2, "dflash draft: b must be ≥ 2");

    // Block: [seed, mask, mask, ...].
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;

    // Step 1: D2D embedding lookup per block slot (parallels spec_step_dflash).
    for (i, &tok) in block.iter().enumerate() {
        let dst = draft_scratch.x.sub_offset(i * h, h);
        match target.weights.embd_format {
            hipfire_runtime::llama::EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&target.weights.token_embd, &dst, tok, h)?
            }
            _ => panic!("ddtree draft: unsupported target embedding format"),
        }
    }

    // Step 2: Positions + optional ctx_slice (identical to spec_step_dflash).
    let effective_ctx_len = match ctx_slice {
        Some(n) => n.min(position),
        None => position,
    };
    let ctx_start = position - effective_ctx_len;
    let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
    let positions_k: Vec<i32> = (ctx_start as i32..(position + b) as i32).collect();
    let th_offset = ctx_start * ne * h;
    let th_slice: &[f32] = &target_hidden_host[th_offset..];

    // Step 3: Draft forward (fills draft_scratch.x with per-position draft
    // hidden rows).
    dflash::draft_forward(
        gpu,
        draft_weights,
        draft_cfg,
        None,
        Some(th_slice),
        &positions_q,
        &positions_k,
        b,
        effective_ctx_len,
        draft_scratch,
    )?;

    // Step 4: Apply target.lm_head to draft hidden rows [1..B). Same batched
    // GEMM paths as spec_step_dflash. Unlike the vanilla path we download
    // the full (B-1) × vocab logits so the tree builder can compute top-K.
    let batch = b - 1;
    let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
    let logits_batch = gpu.alloc_tensor(&[batch * vocab], rdna_compute::DType::F32)?;
    let w_out = &target.weights.output;

    let gemm_result = match w_out.gpu_dtype {
        rdna_compute::DType::Q8_0 => {
            dflash_gemm_q8_lmhead(gpu, w_out, &hidden_rows, &logits_batch, batch)
        }
        rdna_compute::DType::HFQ4G256 => {
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                &w_out.buf,
                w_out.gpu_dtype,
                &hidden_rows,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )
        }
        rdna_compute::DType::MQ4G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            // AWQ-aware rotation; see the target-verify dispatch above for the
            // rationale (numerically identical when `w_out.awq_scale` is None).
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ4G256V2
        | rdna_compute::DType::MQ6G256V2
        | rdna_compute::DType::MQ5G256V2
        | rdna_compute::DType::MQ3G256V2
        | rdna_compute::DType::MQ2G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gemm_mq_batched_lmhead_draft(
                gpu,
                w_out.gpu_dtype,
                &w_out.buf,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ3G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::HFQ6G256 => {
            // Phase A.4: HFQ6 lm_head batched via gemm_hfq6g256_batched_lmhead
            // (which zeros Y then dispatches the dp4a residual on gfx906 or
            // WMMA / FP16 fallbacks elsewhere).
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &hidden_rows,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )
        }
        rdna_compute::DType::MQ6G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        _ => Err(hip_bridge::HipError::new(
            0,
            "ddtree: unsupported target.output dtype (need Q8/HFQ4G256/MQ4G256/MQ4G256V2/MQ6G256V2/MQ5G256V2/MQ3G256V2/MQ2G256V2/MQ3G256/HFQ6G256/MQ6G256)",
        )),
    };
    if let Err(e) = gemm_result {
        let _ = gpu.free_tensor(logits_batch);
        return Err(e);
    }

    let host_logits = match gpu.download_f32(&logits_batch) {
        Ok(v) => v,
        Err(e) => {
            let _ = gpu.free_tensor(logits_batch);
            return Err(e);
        }
    };
    let _ = gpu.free_tensor(logits_batch);
    debug_assert_eq!(host_logits.len(), batch * vocab);
    Ok(host_logits)
}

/// Like `run_dflash_draft_for_logits` but does the top-K + log-sum-exp
/// ON GPU via `topk_logsumexp_batched_f32`, returning only the top-K
/// tokens and log-probs per row. Used by `spec_step_ddtree_batched` to
/// skip the ~20 ms CPU sort and the 15 MB logits D2H.
///
/// Returns `(top_tokens, top_log_probs)` each of size `(b-1) * k` in
/// row-major order (same convention as `ddtree::topk_from_logits`).
#[allow(clippy::too_many_arguments)]
/// Upload an `i32` slice as a device buffer (no I32 DType — raw bytes; kernels
/// read it as `int*`).
fn upload_i32(gpu: &Gpu, data: &[i32]) -> HipResult<GpuTensor> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    gpu.upload_raw(bytes, &[data.len()])
}

/// Run the fused on-device SWOR tree-verify walk and return the CPU-shaped
/// `(accepted_node_indices, bonus_token)`. Builds the device metadata
/// (slot→depth, draw-ordered candidates, child-of-candidate adjacency), launches
/// `ddtree_swor_walk_f32` against the device-resident target + draft logits, and
/// reads back only the tiny result. No full-vocab work on the host.
#[allow(clippy::too_many_arguments)]
fn swor_walk_gpu(
    gpu: &mut Gpu,
    tree: &hipfire_runtime::ddtree::DdTree,
    target_logits: &GpuTensor,
    draft_logits: &GpuTensor,
    pos_cands: &[u32],
    num_pos: usize,
    k: usize,
    vocab: usize,
    temp: f32,
    seed: u64,
) -> HipResult<(Vec<usize>, u32)> {
    let n_slots = 1 + tree.nodes.len();
    let mut slot_depth = vec![0i32; n_slots];
    for s in 1..n_slots {
        slot_depth[s] = tree.nodes[s - 1].depth as i32;
    }
    let mut child_of_cand = vec![-1i32; n_slots * k];
    for s in 0..n_slots {
        let depth = slot_depth[s] as usize;
        if depth >= num_pos {
            continue;
        }
        for r in 0..k {
            let token = pos_cands[depth * k + r];
            if let Some(&ci) = tree.child_maps[s].get(&token) {
                child_of_cand[s * k + r] = ci as i32;
            }
        }
    }
    let pos_cands_i32: Vec<i32> = pos_cands.iter().map(|&t| t as i32).collect();
    let t_pcand = upload_i32(gpu, &pos_cands_i32)?;
    let t_depth = upload_i32(gpu, &slot_depth)?;
    let t_child = upload_i32(gpu, &child_of_cand)?;
    let t_pres = gpu.alloc_tensor(&[vocab], rdna_compute::DType::F32)?;
    let t_qpos = gpu.alloc_tensor(&[vocab], rdna_compute::DType::F32)?;
    let t_out = gpu.alloc_tensor(&[2 + num_pos], rdna_compute::DType::F32)?;
    gpu.ddtree_swor_walk_f32(
        target_logits,
        draft_logits,
        &t_pcand,
        &t_depth,
        &t_child,
        &t_pres,
        &t_qpos,
        &t_out,
        temp,
        k,
        vocab,
        n_slots,
        num_pos,
        seed,
    )?;
    let raw = gpu.download_f32(&t_out)?;
    let _ = gpu.free_tensor(t_pcand);
    let _ = gpu.free_tensor(t_depth);
    let _ = gpu.free_tensor(t_child);
    let _ = gpu.free_tensor(t_pres);
    let _ = gpu.free_tensor(t_qpos);
    let _ = gpu.free_tensor(t_out);
    let accept_len = (raw[0].to_bits() as i32).max(0) as usize;
    let bonus = raw[1].to_bits() as u32;
    let mut accepted = Vec::with_capacity(accept_len);
    for i in 0..accept_len {
        accepted.push((raw[2 + i].to_bits() as i32) as usize);
    }
    Ok((accepted, bonus))
}

fn run_dflash_draft_for_topk_gpu(
    gpu: &mut Gpu,
    target: &ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    // Stage 1b: `None` = GPU-resident path (scratch.target_hidden already
    // populated via D2D scatter; thlog tracks the live rows). `Some(slice)`
    // = ctx_slice diagnostic path (host shadow, always full H2D upload).
    target_hidden_host: Option<&[f32]>,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    b: usize,
    k: usize,
    // `Some((temp, rng))` ⇒ draw the `k` children per position WITHOUT replacement
    // from the draft softmax at `temp` (Gumbel-top-k = exact SWOR sampling), and
    // return their TRUE per-position log-probs. This makes the tree's candidates
    // genuine draft samples, which is what the q-exploiting SWOR verify needs to
    // be distribution-exact. `None` ⇒ the default deterministic GPU top-k.
    sample: Option<(f32, &mut u64)>,
) -> HipResult<(Vec<u32>, Vec<f32>, Option<GpuTensor>)> {
    // The 3rd return is the draft logits kept ON DEVICE (`[batch·vocab]`),
    // populated only in `sample` mode — the fused GPU SWOR walk softmaxes them
    // for its residual, so they must NOT be freed here. `None` on the top-k path;
    // the SWOR caller frees the tensor after the walk.
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let vocab = target.config.vocab_size;
    let mask_token = draft_cfg.mask_token_id;
    assert!(b >= 2, "dflash draft: b must be ≥ 2");
    assert!(k >= 1 && k <= 8, "topk k={} must be in [1, 8]", k);

    // Step 1-3: identical to run_dflash_draft_for_logits — embed, positions,
    // draft forward. Duplicating the small glue to avoid a refactor risk;
    // this path is shipped after. (Could factor out, but the savings is <50
    // lines and the call site is stable.)
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;
    for (i, &tok) in block.iter().enumerate() {
        let dst = draft_scratch.x.sub_offset(i * h, h);
        match target.weights.embd_format {
            hipfire_runtime::llama::EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&target.weights.token_embd, &dst, tok, h)?
            }
            _ => panic!("ddtree draft: unsupported target embedding format"),
        }
    }
    // Stage 1b: positions and hidden-source depend on whether we are using the
    // GPU-resident path (target_hidden_host=None) or the ctx_slice host path.
    let (positions_q, positions_k, th_arg, effective_ctx_len) =
        if let Some(host_slice) = target_hidden_host {
            // ctx_slice diagnostic path: host shadow, simple contiguous positions.
            let effective_ctx_len = match ctx_slice {
                Some(n) => n.min(position),
                None => position,
            };
            let ctx_start = position - effective_ctx_len;
            let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
            let positions_k: Vec<i32> = (ctx_start as i32..(position + b) as i32).collect();
            let th_offset = ctx_start * ne * h;
            (
                positions_q,
                positions_k,
                Some(&host_slice[th_offset..]),
                effective_ctx_len,
            )
        } else {
            // GPU-resident path: scratch.target_hidden already contains all rows
            // via D2D scatter (Stage 1); thlog tracks uploaded_rows + abs_positions.
            // Mirror the chain path (spec_step_dflash ~line 3157): use thlog's
            // eviction-aware abs_positions for k-positions, pass None to skip H2D.
            let effective_ctx_len = draft_scratch.thlog.abs_positions().len().min(position);
            let ctx_start = position - effective_ctx_len;
            let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
            let th_abs = draft_scratch.thlog.abs_positions();
            let start_idx = th_abs.len().saturating_sub(effective_ctx_len);
            let mut positions_k: Vec<i32> = Vec::with_capacity(effective_ctx_len + b);
            positions_k.extend_from_slice(&th_abs[start_idx..]);
            for p in 0..b {
                positions_k.push(position as i32 + p as i32);
            }
            let _ = ctx_start; // not used in this branch
            (positions_q, positions_k, None, effective_ctx_len)
        };

    dflash::draft_forward(
        gpu,
        draft_weights,
        draft_cfg,
        None,
        th_arg,
        &positions_q,
        &positions_k,
        b,
        effective_ctx_len,
        draft_scratch,
    )?;

    // Step 4: lm_head → [batch × vocab] logits (GPU-resident).
    let batch = b - 1;
    let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
    let logits_batch = gpu.alloc_tensor(&[batch * vocab], rdna_compute::DType::F32)?;
    let w_out = &target.weights.output;
    let gemm_result = match w_out.gpu_dtype {
        rdna_compute::DType::Q8_0 => {
            dflash_gemm_q8_lmhead(gpu, w_out, &hidden_rows, &logits_batch, batch)
        }
        rdna_compute::DType::HFQ4G256 => {
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                &w_out.buf,
                w_out.gpu_dtype,
                &hidden_rows,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )
        }
        rdna_compute::DType::MQ4G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            // AWQ-aware rotation for the target lm_head when an AWQ
            // sidecar is attached. Sister of the spec-verify dispatch above.
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ4G256V2
        | rdna_compute::DType::MQ6G256V2
        | rdna_compute::DType::MQ5G256V2
        | rdna_compute::DType::MQ3G256V2
        | rdna_compute::DType::MQ2G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gemm_mq_batched_lmhead_draft(
                gpu,
                w_out.gpu_dtype,
                &w_out.buf,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ3G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::HFQ6G256 => {
            // Phase A.4: HFQ6 lm_head batched.
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &hidden_rows,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )
        }
        rdna_compute::DType::MQ6G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        _ => Err(hip_bridge::HipError::new(
            0,
            "ddtree: unsupported target.output dtype (need Q8/HFQ4G256/MQ4G256/MQ4G256V2/MQ6G256V2/MQ5G256V2/MQ3G256V2/MQ2G256V2/MQ3G256/HFQ6G256/MQ6G256)",
        )),
    };
    if let Err(e) = gemm_result {
        let _ = gpu.free_tensor(logits_batch);
        return Err(e);
    }

    // Step 5a: Gumbel-top-k SWOR sampling (q-exploiting verify). Download the
    // draft logits and, per position, draw k tokens WITHOUT replacement from
    // softmax(logits/temp) via the Gumbel-top-k trick (top-k by `logit/temp +
    // Gumbel(0,1)` is an exact size-k SWOR sample), returning each token's TRUE
    // per-position log-prob `log_softmax(logits/temp)[token]`. The candidates are
    // then genuine draft samples — the precondition the SWOR verify needs.
    if let Some((temp, rng)) = sample {
        // Device-side Gumbel-top-k SWOR sampling: NO [B×vocab] D2H. Draw the k
        // draw-ordered candidates per position + their true log-q on the GPU; only
        // B×k come back for the CPU tree build. `logits_batch` stays resident for
        // the fused SWOR walk (returned below).
        let seed = *rng;
        xorshift_next_unit(rng);
        let idx_gpu = gpu.alloc_tensor(&[batch * k], rdna_compute::DType::F32)?;
        let logp_gpu = gpu.alloc_tensor(&[batch * k], rdna_compute::DType::F32)?;
        gpu.ddtree_gumbel_topk_batched_f32(
            &logits_batch,
            &idx_gpu,
            &logp_gpu,
            vocab,
            k,
            batch,
            temp,
            seed,
        )?;
        let idx_host = gpu.download_f32(&idx_gpu)?;
        let top_log_probs = gpu.download_f32(&logp_gpu)?;
        let _ = gpu.free_tensor(idx_gpu);
        let _ = gpu.free_tensor(logp_gpu);
        let top_tokens: Vec<u32> = idx_host.iter().map(|f| f.to_bits()).collect();
        return Ok((top_tokens, top_log_probs, Some(logits_batch)));
    }

    // Step 5: GPU top-K + log-sum-exp. Writes [batch × k] indices + log-probs.
    let topk_idx_gpu = gpu.alloc_tensor(&[batch * k], rdna_compute::DType::F32)?;
    let topk_val_gpu = gpu.alloc_tensor(&[batch * k], rdna_compute::DType::F32)?;
    let topk_result = gpu.topk_logsumexp_batched_f32(
        &logits_batch,
        &topk_idx_gpu,
        &topk_val_gpu,
        vocab,
        k,
        batch,
    );
    let _ = gpu.free_tensor(logits_batch);
    if let Err(e) = topk_result {
        let _ = gpu.free_tensor(topk_idx_gpu);
        let _ = gpu.free_tensor(topk_val_gpu);
        return Err(e);
    }

    // Step 6: D2H just the top-K outputs (tiny — 8 × 15 × 4 = 480 bytes for k=8).
    let mut idx_host: Vec<i32> = vec![0i32; batch * k];
    let mut val_host: Vec<f32> = vec![0f32; batch * k];
    let idx_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(idx_host.as_mut_ptr() as *mut u8, batch * k * 4) };
    let val_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(val_host.as_mut_ptr() as *mut u8, batch * k * 4) };
    gpu.hip.memcpy_dtoh(idx_bytes, &topk_idx_gpu.buf)?;
    gpu.hip.memcpy_dtoh(val_bytes, &topk_val_gpu.buf)?;
    let _ = gpu.free_tensor(topk_idx_gpu);
    let _ = gpu.free_tensor(topk_val_gpu);

    let top_tokens: Vec<u32> = idx_host.into_iter().map(|x| x as u32).collect();
    Ok((top_tokens, val_host, None))
}

/// Enumerate all root-to-leaf paths in a DdTree. Returns paths as Vec<Vec<usize>>
/// where each inner Vec is the sequence of node indices from the first
/// child-of-root (depth 1) down to a leaf. Leaves are nodes with no children
/// in the tree; if the tree is empty (N=0) this returns a single empty path.
fn enumerate_paths(tree: &hipfire_runtime::ddtree::DdTree) -> Vec<Vec<usize>> {
    if tree.nodes.is_empty() {
        return vec![Vec::new()];
    }
    let mut leaves: Vec<usize> = Vec::new();
    for i in 0..tree.nodes.len() {
        let slot = i + 1;
        if tree.child_maps[slot].is_empty() {
            leaves.push(i);
        }
    }
    let mut paths: Vec<Vec<usize>> = Vec::with_capacity(leaves.len());
    for &leaf_idx in &leaves {
        let mut path: Vec<usize> = Vec::new();
        let mut cur: i32 = leaf_idx as i32;
        while cur >= 0 {
            path.push(cur as usize);
            cur = tree.nodes[cur as usize].parent_index;
        }
        path.reverse();
        paths.push(path);
    }
    paths
}

/// DDTree speculative step (Ringel & Romano 2026, our hybrid-arch port).
///
/// Flow per cycle:
///   1. Run DFlash draft, download raw (B-1) × vocab logits.
///   2. CPU top-K + log-norm per row → per-position (tokens, log-probs).
///   3. Algorithm 1: best-first heap builds up to `tree_budget` tree nodes.
///   4. Snapshot target state (pre-seed). Forward seed once to get posterior[0]
///      and the post-seed branch point; snapshot post-seed state.
///   5. For each root-to-leaf path in the tree, forward each node sequentially
///      through `forward_scratch`; on first visit of a node slot, record its
///      target argmax as `posterior[slot]`. Restore post-seed state between paths.
///   6. Greedy walk: follow target's argmax down the tree to the longest
///      accepted path + bonus token.
///   7. Restore to pre-seed, re-forward (seed + accepted path) with hidden
///      capture so the next cycle's DFlash draft has valid target_hidden_host.
///
/// Cost per cycle: O(N) target forwards where N is the node budget (paper
/// uses 60; we default to `draft_cfg.block_size` = 16 for a cheaper spike).
/// That's ~5× the batched-verify cost of spec_step_dflash; no batched tree
/// attention on hybrid arch would change that, but per-path verify is the
/// correctness-first path (LA state is not polluted across branches).
///
/// Temp=0 only for now — rejection-sampling / CACTUS integration is deferred
/// until the greedy signal looks promising. Paper's DDTree numbers are
/// temp=0 too, so this matches the reference setup.
#[cfg(feature = "deltanet")]
pub fn spec_step_ddtree(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut DeltaNetSnapshot,
    post_seed_snap: &mut DeltaNetSnapshot,
    gdn_tape: &mut GdnTape,
    verify_scratch: &VerifyScratch,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    tree_budget: usize,
    tree_topk: usize,
) -> HipResult<SpecStepResult> {
    let b = draft_cfg.block_size;
    let vocab = target.config.vocab_size;
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    assert!(b >= 2, "spec_step_ddtree: block_size must be ≥ 2");
    assert_eq!(
        target_hidden_host.len(),
        position * ne * h,
        "target_hidden_host size mismatches position"
    );
    assert!(
        tree_topk >= 1 && tree_topk <= vocab,
        "tree_topk must be in [1, vocab]"
    );
    if draft_weights.has_candidate_selector() {
        return Err(hip_bridge::HipError::new(
            0,
            "DFlash2 candidate selector is chain-only; DDTree independent-marginal verify is disabled",
        ));
    }

    // ── 1. Run DFlash draft, download raw logits ─────────────────────────
    let draft_logits = run_dflash_draft_for_logits(
        gpu,
        target,
        draft_weights,
        draft_cfg,
        draft_scratch,
        target_hidden_host,
        position,
        seed_token,
        ctx_slice,
        b,
    )?;

    // ── 2. Per-position top-K + log-normalize (CPU) ───────────────────────
    let (top_tokens, top_log_probs) =
        hipfire_runtime::ddtree::topk_from_logits(&draft_logits, b - 1, vocab, tree_topk);

    // ── 3. Build the DDTree ───────────────────────────────────────────────
    // HIPFIRE_DDTREE_LOGW_CUTOFF=<f32> enables the meta-verifier pruner: stop
    // heap expansion when the next candidate's cumulative log-probability
    // drops below -cutoff. Per-cycle dynamic budget. Disabled (= 0.0 or
    // unset) preserves the fixed-budget behaviour.
    let tree = hipfire_runtime::ddtree::build_ddtree_tree_bounded(
        &top_tokens,
        &top_log_probs,
        b - 1,
        tree_topk,
        0, // min_nodes=0: pure-cutoff DDTree build
        tree_budget,
        gpu.flags.ddtree_logw_cutoff_value(),
    );
    record_ddtree_meta_nodes(tree.num_nodes());

    // Edge case: empty tree (shouldn't happen if budget≥1 and b≥2, but guard).
    // With zero nodes there's nothing to verify — just forward seed, sample,
    // commit. Mirrors the behavior of a B=2 DFlash cycle.
    // Note: `forward_scratch_with_hidden` runs the final rmsnorm + lm_head
    // internally and leaves the next-token logits in `scratch.logits` — do
    // NOT call weight_gemv again on scratch.x (that's pre-rmsnorm hidden
    // and produces incorrect logits).
    if tree.nodes.is_empty() {
        target_snap.save_from(&target.dn_state, gpu)?;
        qwen35::forward_scratch_with_hidden(
            gpu,
            &target.weights,
            &target.config,
            seed_token,
            position,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            hidden_rb,
        )?;
        let logits0 = gpu.download_f32(&target.scratch.logits)?;
        let bonus = argmax_u32(&logits0);
        let hidden_block = download_hidden_block(gpu, hidden_rb, 1)?;
        target_hidden_host.extend_from_slice(&hidden_block[..1 * ne * h]);
        return Ok(SpecStepResult {
            accepted: 0,
            bonus_token: bonus,
            drafted: vec![seed_token],
            committed: vec![seed_token, bonus],
        });
    }

    // ── 4. Snapshot pre-seed target state ─────────────────────────────────
    //
    // We verify each root-to-leaf path via `verify_dflash_block` starting
    // from the pre-seed state — this is the same batched target forward
    // DFlash uses for its verify, so we stay byte-exact with the non-tree
    // path. Between paths we restore pre-seed (both DN and KV cache; KV
    // overwrites happen naturally because each verify writes to the same
    // position range starting at `position`).
    target_snap.save_from(&target.dn_state, gpu)?;
    // post_seed_snap is allocated by the caller but unused in this path —
    // kept in the signature so the API stays compatible with potentially
    // sharing-the-seed-forward optimizations in a later rev. Suppress the
    // unused warning without asking the caller to annotate.
    let _ = &post_seed_snap;

    let mut posterior: Vec<u32> = vec![0; 1 + tree.num_nodes()];
    let mut posterior_set: Vec<bool> = vec![false; 1 + tree.num_nodes()];

    // ── 5. Per-path verify via verify_dflash_block ───────────────────────
    //
    // For each root-to-leaf path, run the batched target verify on
    // [seed_token, path_tokens...]. verify_dflash_block gives us argmax
    // per position via the same code path as spec_step_dflash, which
    // guarantees no numerical drift vs baseline at temp=0. Per-node
    // posterior records are first-visit-wins — all paths traversing the
    // same ancestor produce the same argmax at that ancestor's slot.
    let paths = enumerate_paths(&tree);
    for path in &paths {
        // Build verify block: [seed] + path_tokens.
        let mut verify_block: Vec<u32> = Vec::with_capacity(1 + path.len());
        verify_block.push(seed_token);
        for &ni in path {
            verify_block.push(tree.nodes[ni].token);
        }

        // Restore pre-seed state before each verify. DN state via snapshot;
        // KV cache self-overwrites at positions [position, position+N).
        target_snap.restore_to(&mut target.dn_state, gpu)?;

        // NOTE: verify_dflash_block takes &mut HiddenStateRingBuffer (not
        // Option); we pass our buffer but its writes get clobbered by the
        // step-8 replay. That's fine — we only read hidden_rb in step 9
        // after the replay. Path verifies DO advance the ring buffer head
        // but the final replay brings it right back.
        let verify_out = verify_dflash_block(
            gpu,
            target,
            &verify_block,
            position,
            hidden_rb,
            None,
            false, // want_full_logits=false — greedy only for now
            verify_scratch,
        )?;

        // verify_out.argmax_per_pos has length N = verify_block.len().
        // argmax_per_pos[i] = target's predicted NEXT token at position
        // `position + i`. That's:
        //   i=0          → prediction after seed = what should match block[1]
        //                  = posterior at root slot
        //   i=1..N-1     → prediction after node at path-position i-1
        //                  = posterior at path[i-1]'s slot
        // (We don't use argmax_per_pos[N-1] because we'd need a child of
        // the leaf, which the tree doesn't have — greedy walk stops there.)
        if !posterior_set[0] {
            posterior[0] = verify_out.argmax_per_pos[0];
            posterior_set[0] = true;
        }
        for (i, &ni) in path.iter().enumerate() {
            let slot = ni + 1;
            if !posterior_set[slot] && i + 1 < verify_out.argmax_per_pos.len() {
                posterior[slot] = verify_out.argmax_per_pos[i + 1];
                posterior_set[slot] = true;
            }
        }
    }

    // ── 6. Greedy walk: longest accepted path + bonus ─────────────────────
    let (accepted_node_indices, bonus_token) =
        hipfire_runtime::ddtree::follow_verified_tree(&tree, &posterior);
    let accept_len = accepted_node_indices.len();

    // ── 7. Build committed + drafted sequences ────────────────────────────
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 2);
    committed.push(seed_token);
    for &ni in &accepted_node_indices {
        committed.push(tree.nodes[ni].token);
    }
    committed.push(bonus_token);

    let mut drafted: Vec<u32> = Vec::with_capacity(accept_len + 1);
    drafted.push(seed_token);
    for &ni in &accepted_node_indices {
        drafted.push(tree.nodes[ni].token);
    }

    // ── 8. Tape-capturing verify on the committed path, then tape replay ─
    //
    // The tape records per-LA-layer (q, k, v, α, β) innovations for the
    // tokens it processes. Replaying the tape then advances DN state
    // through THOSE tokens. So the tape MUST be captured from a verify
    // whose block contains the actual committed tokens — any divergence
    // (e.g., capturing from the top-1 chain when the tree accepted a
    // rank>0 branch) feeds wrong LA updates into the next cycle's state.
    //
    // For topk=1 the tree's only path IS the top-1 chain, so committed
    // (length accept_len+1) is a prefix of the full-B DFlash block; we
    // still verify at full B here to stay batch-size-identical with the
    // DFlash baseline, then replay just the first accept_len+1 tape
    // steps. That path is byte-exact with baseline.
    //
    // For topk>1 the committed path may contain branch tokens that don't
    // appear in dflash_block's top-1 chain. In that case we fall back to
    // running the tape capture over the committed path directly — not
    // batch-size-equal to DFlash but tokens-correct. Some cross-cycle
    // numerical drift vs baseline is the tradeoff; output should remain
    // a valid target-greedy sequence.
    let topk1_is_committed_prefix = accept_len > 0
        && committed[1..=accept_len]
            .iter()
            .enumerate()
            .all(|(d, &tok)| tok == top_tokens[d * tree_topk]);
    let tape_block: Vec<u32> = if topk1_is_committed_prefix || accept_len == 0 {
        // Safe to use full-B top-1 block (byte-exact with DFlash path).
        let mut vb: Vec<u32> = Vec::with_capacity(b);
        vb.push(seed_token);
        for d in 0..(b - 1) {
            vb.push(top_tokens[d * tree_topk]);
        }
        vb
    } else {
        // Accepted a branch — verify over the committed tokens to get
        // correct LA innovations.
        committed[..accept_len + 1].to_vec()
    };
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    let _tape_verify = verify_dflash_block(
        gpu,
        target,
        &tape_block,
        position,
        hidden_rb,
        Some(gdn_tape),
        false,
        verify_scratch,
    )?;
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    gdn_tape.replay_gdn(
        gpu,
        &target.weights,
        &target.config,
        &mut target.dn_state,
        accept_len + 1,
    )?;
    // Target state is now at position + accept_len + 1. Bonus token's state
    // is deferred to next cycle's block[0], matching spec_step_dflash.

    // ── 9. Append (1 + accept_len) hidden rows to target_hidden_host ─────
    //
    // The tape-capturing verify wrote `tape_block.len()` rows to hidden_rb.
    // We want the FIRST (accept_len + 1) — positions [position, position +
    // accept_len] of the verified block. download_hidden_block returns the
    // most-recent N rows in order, so pulling tape_block.len() rows and
    // slicing to accept_len+1 grabs the right prefix.
    let hidden_rows_written = tape_block.len();
    let hidden_block = download_hidden_block(gpu, hidden_rb, hidden_rows_written)?;
    let rows_to_keep = accept_len + 1;
    target_hidden_host.extend_from_slice(&hidden_block[..rows_to_keep * ne * h]);

    Ok(SpecStepResult {
        accepted: accept_len,
        bonus_token,
        drafted,
        committed,
    })
}

/// Committed-order row moves for a DDTree post-accept gather, in BLOCK-LOCAL
/// row space (row 0 is the seed slot and never moves).
///
/// Committed row `i + 1` must end up holding linearization slot
/// `accepted_node_indices[i] + 1`. Consecutive moves with the same
/// source→destination delta are coalesced into one run. A spine accept yields
/// an empty list, which is exactly the existing fast path.
fn ddtree_commit_row_runs(accepted_node_indices: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for (i, &ni) in accepted_node_indices.iter().enumerate() {
        let (dst, src) = (i + 1, ni + 1);
        if dst == src {
            continue;
        }
        match runs.last_mut() {
            Some((d, s, n)) if *d + *n == dst && *s + *n == src => *n += 1,
            _ => runs.push((dst, src, 1)),
        }
    }
    runs
}

/// Move rows within a single position-addressed buffer, in place.
///
/// `moves` are `(dst_row, src_row, n_rows)` in PHYSICAL row space, ordered by
/// ascending `dst_row`.
///
/// No scratch buffer is required, and that is a property of the linearization
/// rather than luck: slots are assigned in depth order, so the i-th accepted
/// node can never occupy a slot below `i` (`src >= dst` for every move). Writing
/// row `d` can therefore only alias the source of a LATER move, whose source
/// `s' >= d' > d` — every source is read before anything overwrites it. Distinct
/// block rows map to distinct physical rows, so this also holds when the caller
/// maps through a ring modulus.
///
/// That argument is about separate moves. Within one coalesced run the ranges
/// overlap whenever `src - dst < n` (e.g. `(1, 2, 3)`), and a D2D memcpy over
/// overlapping ranges is undefined: the blit copies chunks in parallel, so which
/// bytes win varies run to run (HIPF-136, greedy DDTree nondeterminism). Each run
/// is therefore issued as ascending chunks of at most `src - dst` rows; chunk k
/// overwrites only rows chunk k-1 already read.
fn move_rows_in_place(
    gpu: &Gpu,
    buf: &GpuTensor,
    row_bytes: usize,
    moves: &[(usize, usize, usize)],
) -> HipResult<()> {
    for (dst, src, n) in moves.iter().flat_map(|&m| non_overlapping_chunks(m)) {
        let (dst_off, src_off, size) = (dst * row_bytes, src * row_bytes, n * row_bytes);
        match gpu.active_stream.as_ref() {
            // Stream-ordered: the moves serialize against each other and against
            // the verify that produced the rows, with no host-side drain.
            Some(stream) => gpu
                .hip
                .memcpy_dtod_async_at(&buf.buf, dst_off, &buf.buf, src_off, size, stream)?,
            None => gpu
                .hip
                .memcpy_dtod_at(&buf.buf, dst_off, &buf.buf, src_off, size)?,
        }
    }
    Ok(())
}

/// Split a `(dst, src, n)` move with `src > dst` into ascending sub-moves whose
/// source and destination ranges are disjoint.
fn non_overlapping_chunks(
    (dst, src, n): (usize, usize, usize),
) -> impl Iterator<Item = (usize, usize, usize)> {
    debug_assert!(src > dst, "in-place row moves only shift rows down");
    let step = src - dst;
    (0..n)
        .step_by(step)
        .map(move |k| (dst + k, src + k, step.min(n - k)))
}

/// DDTree post-accept gather: relocate the accepted chain's rows from their
/// linearization slots into the committed linear slots.
///
/// The tree verify already wrote every node's K/V at its OWN slot (commit
/// 39aa358 decoupled the two: RoPE rotates at DEPTH positions while KV writes
/// use flat physical slots), and the i-th accepted node sits at depth `i + 1`,
/// so its baked-in RoPE phase already equals its committed linear position.
/// The bytes are correct where they sit — only the row INDEX is wrong when the
/// greedy walk detoured off the spine. Moving them is therefore equivalent to
/// (and cheaper than) re-running the committed prefix through a second verify.
///
/// Covers all three row-addressed structures the rest of the cycle consumes:
/// the target KV cache (per full-attention layer, plus the separate scale planes
/// in `quant_int8` mode), the GDN tape (per linear-attention layer), and the
/// extracted-hidden ring. Afterwards the cycle is indistinguishable from a spine
/// accept, so every downstream consumer takes its fast path unchanged.
fn ddtree_gather_committed_rows(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    gdn_tape: &GdnTape,
    hidden_rb: &HiddenStateRingBuffer,
    position: usize,
    block_size: usize,
    accepted_node_indices: &[usize],
) -> HipResult<()> {
    let runs = ddtree_commit_row_runs(accepted_node_indices);
    if runs.is_empty() {
        return Ok(());
    }

    // ── target KV cache ───────────────────────────────────────────────────
    // Row stride is derived from the allocation rather than recomputed per
    // quant mode: every K/V layout here is `physical_cap` rows of equal size,
    // and re-deriving it would duplicate five mode-specific formulas.
    let cap = target.kv_cache.physical_cap.max(1);
    let kv_moves: Vec<(usize, usize, usize)> = runs
        .iter()
        .map(|&(d, s, n)| (position + d, position + s, n))
        .collect();
    for li in 0..target.config.layer_types.len() {
        if target.config.layer_types[li] != qwen35::LayerType::FullAttention {
            continue;
        }
        for plane in [
            target.kv_cache.k_gpu.get(li),
            target.kv_cache.v_gpu.get(li),
            target.kv_cache.k_scales.get(li),
            target.kv_cache.v_scales.get(li),
        ]
        .into_iter()
        .flatten()
        {
            let bytes = plane.byte_size();
            if bytes == 0 {
                continue;
            }
            move_rows_in_place(gpu, plane, bytes / cap, &kv_moves)?;
        }
    }

    // ── GDN tape (block-local rows) ───────────────────────────────────────
    let gate_row_bytes = gdn_tape.n_v_heads * 4;
    for la in 0..gdn_tape.qkv_bufs.len() {
        move_rows_in_place(gpu, &gdn_tape.qkv_bufs[la], gdn_tape.qkv_dim * 4, &runs)?;
        move_rows_in_place(gpu, &gdn_tape.alpha_bufs[la], gate_row_bytes, &runs)?;
        move_rows_in_place(gpu, &gdn_tape.beta_bufs[la], gate_row_bytes, &runs)?;
    }

    // ── extracted-hidden ring ─────────────────────────────────────────────
    // Ring slots wrap, so these go row-by-row through the ring mapping instead
    // of as coalesced runs (the no-scratch argument above survives the wrap
    // because distinct block rows map to distinct slots).
    let ring_moves: Vec<(usize, usize, usize)> = runs
        .iter()
        .flat_map(|&(d, s, n)| (0..n).map(move |k| (d + k, s + k, 1)))
        .map(|(d, s, n)| {
            (
                hidden_rb.block_row_slot(block_size, d),
                hidden_rb.block_row_slot(block_size, s),
                n,
            )
        })
        .collect();
    let hidden_row_bytes = hidden_rb.hidden_dim * 4;
    for buf in &hidden_rb.layer_bufs {
        move_rows_in_place(gpu, buf, hidden_row_bytes, &ring_moves)?;
    }
    Ok(())
}

/// Batched tree-verify counterpart of `spec_step_ddtree`. Replaces the
/// per-path DFS with a single `verify_dflash_block_tree` call using the
/// FA tree-attention mask infrastructure (commits 835aa46 / f0ee980 /
/// 704bf11). Same return value and side-effect semantics as the per-path
/// version — callers swap the two transparently.
///
/// Correctness notes:
///
/// - **FA side (tree-exact):** each tree node's Q attends only to its
///   ancestors + prompt. The mask is -inf on non-ancestor in-block keys
///   so exp-sum collapses, matching per-path DFS argmaxes exactly at
///   temp=0.
/// - **GDN side (linear-replay approximation):** in the tree forward
///   the recurrent GDN kernel advances state sequentially through the
///   linearized token order `[seed, n0, n1, ...]`. For siblings at the
///   same tree depth this cross-contaminates state — node b's S-state
///   update sees node a's innovations even though they're alternatives,
///   not sequential. At `topk=1` the tree is a pure chain so state
///   advance is identical to DFlash (byte-exact). At `topk>1` the FA
///   posteriors are still correct (ancestor attention), but the GDN
///   contribution to each node's hidden has small drift vs per-path DFS.
/// - **Tape/commit path (correct):** we do a SECOND verify on the
///   committed prefix (no tree) for tape capture. That tape is byte-
///   exact with per-path DFS's committed-prefix verify, so LA state
///   advances correctly after the cycle completes.
///
#[allow(clippy::too_many_arguments)]
pub fn spec_step_ddtree_batched(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut DeltaNetSnapshot,
    post_seed_snap: &mut DeltaNetSnapshot,
    gdn_tape: &mut GdnTape,
    scratch: &DdtreeScratch,
    verify_scratch: &VerifyScratch,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    tree_budget: usize,
    tree_topk: usize,
    // Temperature for the verify-side acceptance. temp == 0 → greedy argmax
    // walk (follow_verified_tree); temp > 0 → distribution-preserving naive
    // tree sampling (sample_verified_tree), which needs the full per-slot
    // target logits and consumes `rng_state` (xorshift, shared with the linear
    // DFlash sampler convention).
    temp: f32,
    rng_state: &mut u64,
    // Max accepted tree nodes this window (`emit.len() == accepted + 1`).
    // `None` = uncapped. Clamped before tape replay / hidden / KV commit.
    max_accept: Option<usize>,
) -> HipResult<SpecStepResult> {
    let b = draft_cfg.block_size;
    let vocab = target.config.vocab_size;
    assert!(b >= 2, "spec_step_ddtree_batched: block_size must be ≥ 2");
    // Stage 1b: target_hidden_host is no longer maintained in the GPU-resident
    // default path (ctx_slice=None). The length-invariant assert is removed
    // to avoid false failures. The ctx_slice=Some path still uses the host Vec.
    assert!(
        tree_topk >= 1 && tree_topk <= vocab,
        "tree_topk must be in [1, vocab]"
    );
    if draft_weights.has_candidate_selector() {
        return Err(hip_bridge::HipError::new(
            0,
            "DFlash2 candidate selector is chain-only; DDTree independent-marginal verify is disabled",
        ));
    }

    // D16: ensure active_stream is set before any work so memset_async and
    // the stream-scoped sync in verify_dflash_block_inner have a non-null
    // stream to ride on. Mirrors the identical setup in spec_step_dflash
    // (~line 2968) — da2753e pattern.
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // Unused in the batched path (no per-path DFS), kept in signature for
    // API compatibility with `spec_step_ddtree` so callers can switch by
    // flipping a single fn pointer.
    let _ = &post_seed_snap;

    // `DDTREE_TIMING=1` prints per-cycle breakdown: draft / topk / build /
    // pre_verify / verify. Used to diagnose where the wall-clock goes.
    // `DDTREE_TIMING=1` prints per-cycle breakdown: draft+topk / build /
    // pre_verify / verify. The draft and top-K are fused into one GPU-
    // resident path now — no separate timer.
    let debug_tm = std::env::var("DDTREE_TIMING").is_ok();
    let t_all = std::time::Instant::now();

    // Verify-scheme selector. SWOR (q-exploiting Sequoia/SpecTr, distribution-
    // exact) is the ONLY temp>0 verify path — the naive-sampling fallback
    // (HIPFIRE_DDTREE_VERIFY=naive) was removed in D8 because it required
    // a ~37 MB/cycle full-logits D2H and SWOR is strictly superior.
    // temp=0 → greedy argmax walk; temp>0 → on-GPU SWOR.
    let use_swor = temp > 0.0;

    // ── 1+2. GPU-resident draft + per-row top-K + log-sum-exp ────────────
    // Keeps logits on device; returns only (b-1) × k indices + log-probs
    // to the host. Replaces the prior 15 MB D2H + CPU sort pair (~34 ms)
    // with an on-device top-K (~µs) plus a ~480 byte D2H. SWOR mode instead
    // Gumbel-top-k samples the children (download + CPU sample).
    //
    // Stage 1b: Pass `None` as target_hidden_host in the default production
    // path (ctx_slice=None) so draft_forward skips the H2D upload — the
    // GPU-resident scratch.target_hidden is already populated by the previous
    // cycle's D2D scatter (or the prefill scatter). The ctx_slice diagnostic
    // path (Some(n)) still uses the CPU host shadow.
    let th_host_arg: Option<&[f32]> = if ctx_slice.is_none() {
        None // GPU-resident: scratch.target_hidden populated via D2D scatter
    } else {
        Some(target_hidden_host.as_slice())
    };
    let (top_tokens, top_log_probs, draft_logits_dev) = run_dflash_draft_for_topk_gpu(
        gpu,
        target,
        draft_weights,
        draft_cfg,
        draft_scratch,
        th_host_arg,
        position,
        seed_token,
        ctx_slice,
        b,
        tree_topk,
        if use_swor {
            Some((temp, &mut *rng_state))
        } else {
            None
        },
    )?;
    // SWOR verify needs the draw-ordered candidates per position; `top_tokens` is
    // already in Gumbel-draw order (rank 0 = first drawn) in sample mode.
    let swor_pos_cands = top_tokens.clone();

    let t_draft = t_all.elapsed();
    let t_topk = t_draft; // fused with draft now

    // ── 3. Build the DDTree ───────────────────────────────────────────────
    // HIPFIRE_DDTREE_LOGW_CUTOFF=<f32> enables the meta-verifier pruner: stop
    // heap expansion when the next candidate's cumulative log-probability
    // drops below -cutoff. Per-cycle dynamic budget. Disabled (= 0.0 or
    // unset) preserves the fixed-budget behaviour.
    let tree = hipfire_runtime::ddtree::build_ddtree_tree_bounded(
        &top_tokens,
        &top_log_probs,
        b - 1,
        tree_topk,
        0, // min_nodes=0: pure-cutoff DDTree build
        tree_budget,
        gpu.flags.ddtree_logw_cutoff_value(),
    );
    record_ddtree_meta_nodes(tree.num_nodes());

    let t_build = t_all.elapsed();

    // Empty-tree shortcut (identical to spec_step_ddtree's path).
    if tree.nodes.is_empty() {
        target_snap.save_from(&target.dn_state, gpu)?;
        qwen35::forward_scratch_with_hidden(
            gpu,
            &target.weights,
            &target.config,
            seed_token,
            position,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            hidden_rb,
        )?;
        let logits0 = gpu.download_f32(&target.scratch.logits)?;
        let bonus = argmax_u32(&logits0);
        // D2D scatter 1 row into draft_scratch.target_hidden, then keep
        // target_hidden_host length-consistent for the next cycle's assert.
        scatter_hidden_block_to_interleaved(
            gpu,
            hidden_rb,
            &draft_scratch.target_hidden,
            position,
            1,
            1,
            draft_scratch.ctx_modulus(),
        )?;
        let co = target.kv_cache.compact_offset as i32;
        draft_scratch.thlog.append_committed(position, 1, co);
        // Stage 1b: no CPU download needed — GPU buffer is authoritative.
        return Ok(SpecStepResult {
            accepted: 0,
            bonus_token: bonus,
            drafted: vec![seed_token],
            committed: vec![seed_token, bonus],
        });
    }

    // ── 4. Linearize the tree into (tokens, positions, mask_host, parents) ─
    //
    // mask_host is computed here for the HIPFIRE_DDTREE_ASSERT_MASK=1 dual-
    // path check (byte-equality proof vs GPU build). In normal operation it
    // is not uploaded — the GPU kernel (step 5) rebuilds it on-device.
    let (verify_tokens, verify_positions, mask_host, parent_host) =
        hipfire_runtime::ddtree::linearize_tree_with_parents(&tree, seed_token, position as u32);
    let big_n = verify_tokens.len();
    debug_assert_eq!(big_n, 1 + tree.num_nodes());
    debug_assert_eq!(parent_host.len(), big_n);

    assert!(
        big_n <= scratch.max_n,
        "tree big_n {} exceeds scratch.max_n {} (increase DdtreeScratch size)",
        big_n,
        scratch.max_n,
    );

    // ── 5. Upload parent_indices; build attn_bias mask on-GPU (Stage 3a) ─
    //
    // D5: parent_indices H2D (244 B, stays — needed for tree-aware LA and
    // for the mask kernel itself).  D4 (attn_bias H2D, ~15 KB/cycle) is
    // eliminated: instead of uploading mask_host we launch
    // ddtree_build_attn_mask_f32 which walks the parent chain per-thread
    // and writes scratch.attn_bias directly on-device.
    //
    // parent_indices is always uploaded (needed both by the mask kernel and,
    // when ddtree_tree_la=true, by the tree-aware GDN kernels).
    let use_tree_la = gpu.flags.ddtree_tree_la;
    {
        let parent_bytes = unsafe {
            std::slice::from_raw_parts(parent_host.as_ptr() as *const u8, parent_host.len() * 4)
        };
        gpu.hip
            .memcpy_htod(&scratch.parent_indices.buf, parent_bytes)?;
    }

    // Build attn_bias on GPU from the now-resident parent_indices. The sub-
    // offset view is used by the FA kernel (step 7) to bound-check big_n²
    // reads.  The kernel writes exactly big_n² floats at the head of
    // scratch.attn_bias; tail is never read by the FA kernel.
    gpu.ddtree_build_attn_mask_f32(&scratch.parent_indices, &scratch.attn_bias, big_n)?;

    // ── 5b. Byte-equality assert (HIPFIRE_DDTREE_ASSERT_MASK=1) ──────────
    //
    // Dual-path proof: download the GPU mask and compare byte-for-byte with
    // the host mask_host computed by linearize_tree_with_parents above.
    // Off by default (costs one D2H per cycle).
    if hipfire_config::developer_var("HIPFIRE_DDTREE_ASSERT_MASK")
        .ok()
        .as_deref()
        == Some("1")
    {
        // Synchronize so the kernel has finished writing before the D2H.
        gpu.hip.device_synchronize()?;
        let n_floats = big_n * big_n;
        let mut gpu_mask = vec![0.0f32; n_floats];
        let gpu_mask_bytes = unsafe {
            std::slice::from_raw_parts_mut(gpu_mask.as_mut_ptr() as *mut u8, n_floats * 4)
        };
        gpu.hip
            .memcpy_dtoh(gpu_mask_bytes, &scratch.attn_bias.buf)?;
        // mask_host is big_n*big_n; gpu_mask is big_n*big_n.
        assert_eq!(
            mask_host.len(),
            n_floats,
            "ASSERT_MASK: mask_host.len() mismatch"
        );
        for idx in 0..n_floats {
            let h = mask_host[idx].to_bits();
            let g = gpu_mask[idx].to_bits();
            assert_eq!(
                h, g,
                "ASSERT_MASK: mismatch at flat index {idx} (row={}, col={}): host={:08x} gpu={:08x}",
                idx / big_n,
                idx % big_n,
                h,
                g,
            );
        }
        eprintln!(
            "[DDTREE_ASSERT_MASK] big_n={big_n} mask byte-identical ({} floats)",
            n_floats
        );
    }

    // ── 6. Snapshot pre-seed target state ─────────────────────────────────
    target_snap.save_from(&target.dn_state, gpu)?;

    // ── 7. Tree verify: single batched forward with tree-attention mask ──
    //
    // Key optimization: pass `gdn_tape` INTO the tree verify so GDN
    // innovations get captured in the linear tree-traversal order. For the
    // topk=1 (or topk>1 where the accepted path coincides with the top-1
    // linear chain) case, the committed path is a contiguous prefix of the
    // linear order — so replaying `tape[0..accept_len+1]` advances LA state
    // correctly and we save an entire forward pass. For topk>1 paths that
    // diverge from the linear prefix, we fall back to a second verify over
    // the committed tokens (step 10 below).
    //
    // argmax_per_pos[i] = target's argmax prediction at slot i in the
    // linearization, i.e. what comes AFTER the token at that slot.
    // Sub-offset view sized to the exact big_n × big_n the current tree needs.
    // scratch.attn_bias is sized for the worst case (max_n² = (1+max_budget)²),
    // but when the actual tree is smaller (e.g. topk=1 linear-chain trees
    // don't fill max_budget), forward_prefill_batch's assert rejects the
    // oversized buffer. The kernel only ever reads up to big_n² floats via
    // `tree_bias[row × block_cols + col]`, so a view is equivalent and keeps
    // the assert semantics meaningful.
    let attn_bias_view = scratch.attn_bias.sub_offset(0, big_n * big_n);
    // Parent-indices sub-view sized to big_n (one i32 per slot; stored as
    // 4 × big_n raw bytes). Only populated when HIPFIRE_DDTREE_TREE_LA=1.
    let parent_view = scratch.parent_indices.sub_offset(0, big_n * 4);
    let ctx = qwen35::TreeVerifyCtx {
        positions: &verify_positions,
        attn_bias: &attn_bias_view,
        parent_indices: if use_tree_la {
            Some(&parent_view)
        } else {
            None
        },
    };
    let t_pre_verify = t_all.elapsed();
    // D8: `want_full_logits` was `temp > 0.0 && !use_swor` (the naive path,
    // gated by HIPFIRE_DDTREE_VERIFY=naive). Since use_swor is now the only
    // temp>0 path (the naive flag and its ~37 MB/cycle D2H have been removed),
    // want_full_logits is permanently false. Greedy: GPU-argmax + tiny D2H.
    // SWOR: no host logits needed; verify_scratch.logits stays device-resident.
    let verify_out = verify_dflash_block_tree(
        gpu,
        target,
        &verify_tokens,
        position,
        hidden_rb,
        Some(gdn_tape),
        false, // want_full_logits: always false; D8 naive path removed
        ctx,
        verify_scratch,
        use_swor, // D9: SWOR uses 68-byte walk result, not argmax_per_pos
    )?;
    let t_post_verify = t_all.elapsed();

    // ── 8. Accept walk: longest accepted path + bonus ─────────────────────
    // temp=0 → greedy argmax walk (follow_verified_tree).
    // temp>0 → q-exploiting Sequoia/SpecTr SWOR walk (use_swor, distribution-
    //   exact). The naive-sampling fallback (HIPFIRE_DDTREE_VERIFY=naive) has
    //   been removed (D8): it required a ~37 MB/cycle full-logits D2H and is
    //   superseded by SWOR which achieves the same distribution-preservation
    //   on-device. All paths return the same (accepted_node_indices, bonus_token)
    //   shape; step 10's divergent-path commit handles non-linear accepted paths.
    let (mut accepted_node_indices, mut bonus_token) = if use_swor {
        // Fully on-device fused SWOR walk: target logits from verify scratch,
        // draft logits kept on device — no full-vocab host work, no q D2H.
        let target_dev = verify_scratch.logits.sub_offset(0, big_n * vocab);
        let draft_dev = draft_logits_dev
            .as_ref()
            .expect("use_swor ⇒ draft kept its device logits");
        let seed = *rng_state;
        xorshift_next_unit(rng_state);
        swor_walk_gpu(
            gpu,
            &tree,
            &target_dev,
            draft_dev,
            &swor_pos_cands,
            b - 1,
            tree_topk,
            vocab,
            temp,
            seed,
        )?
    } else {
        // Greedy (temp=0): follow argmax at each tree node.
        hipfire_runtime::ddtree::follow_verified_tree(&tree, &verify_out.argmax_per_pos)
    };
    // The kept draft logits are no longer needed once the walk has run.
    if let Some(t) = draft_logits_dev {
        let _ = gpu.free_tensor(t);
    }
    // Pre-commit budget: emit = accept_len + 1 ≤ max_emit. Truncate the walk
    // BEFORE tape replay / hidden scatter / DN advance. When the walk went
    // past the budget, the next accepted node token is itself a genuine
    // target draw (greedy match or SWOR sample) — promote it to bonus.
    // Greedy boundary can equivalently re-read argmax_per_pos[slot].
    if let Some(m) = max_accept {
        if accepted_node_indices.len() > m {
            if use_swor {
                bonus_token = tree.nodes[accepted_node_indices[m]].token;
            } else {
                let bonus_slot = if m == 0 {
                    0
                } else {
                    accepted_node_indices[m - 1] + 1
                };
                bonus_token = verify_out.argmax_per_pos[bonus_slot];
            }
            accepted_node_indices.truncate(m);
        }
    }
    let accept_len = accepted_node_indices.len();

    // ── 9. Build committed + drafted sequences ────────────────────────────
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 2);
    committed.push(seed_token);
    for &ni in &accepted_node_indices {
        committed.push(tree.nodes[ni].token);
    }
    committed.push(bonus_token);

    let mut drafted: Vec<u32> = Vec::with_capacity(accept_len + 1);
    drafted.push(seed_token);
    for &ni in &accepted_node_indices {
        drafted.push(tree.nodes[ni].token);
    }

    // ── 10. Tape/hidden path selection ────────────────────────────────────
    //
    // Fast path: accepted tree nodes occupy linear slots [0, 1, 2, ...,
    // accept_len - 1] in the tree. Their tokens are in linear-order
    // positions [1, 2, ..., accept_len] of the tape (slot 0 = seed). The
    // tree tape captures innovations at those same linear slots, so
    // `replay_gdn(accept_len + 1)` is exact with DFlash.
    //
    // Fast path is ALWAYS the case at topk=1 (tree is a chain, accepted
    // indices are [0, 1, 2, ...]). At topk>1 it holds iff the greedy walk
    // picked the rank-0 child at every accepted step (no sibling detour).
    //
    // Slow path (topk>1 detour): re-capture tape on the committed prefix
    // with a second verify, then replay as before. Costs +1 forward but
    // keeps LA state byte-correct.
    // HIPFIRE_DDTREE_FORCE_SLOW=1: force the slow (re-verify) path even when
    // committed path == linearization prefix. Diagnostic — quantifies the
    // cost of always re-running the committed tokens through a non-tree
    // verify to fix KV cache entries at committed slots (topk>1 siblings
    // at same depth otherwise race and the LAST write wins regardless of
    // which sibling was committed).
    let force_slow = hipfire_config::developer_var("HIPFIRE_DDTREE_FORCE_SLOW")
        .ok()
        .as_deref()
        == Some("1");
    let spine_accept = accepted_node_indices
        .iter()
        .enumerate()
        .all(|(i, &ni)| ni == i);
    let fast_tape_ok = !force_slow && spine_accept;
    // Per-cycle fast/slow accounting. HIPFIRE_DDTREE_TAPE_DUMP=1 emits a
    // per-cycle line to stderr; useful to quantify how often the slow-path
    // 2nd verify fires at a given topk / workload. Aggregate stats are
    // printed by dflash_spec_demo at end-of-generation via this thread-local.
    thread_local! {
        static DDTREE_FAST_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static DDTREE_SLOW_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    if fast_tape_ok {
        DDTREE_FAST_COUNT.with(|c| c.set(c.get() + 1));
    } else if !force_slow {
        DDTREE_SLOW_COUNT.with(|c| c.set(c.get() + 1));
    }
    if hipfire_config::developer_var("HIPFIRE_DDTREE_TAPE_DUMP")
        .ok()
        .as_deref()
        == Some("1")
    {
        let fast = DDTREE_FAST_COUNT.with(|c| c.get());
        let slow = DDTREE_SLOW_COUNT.with(|c| c.get());
        eprintln!(
            "[ddtree-tape] cycle: fast_tape_ok={} accept_len={} spine_accept={} tree_la={} (cumulative fast={}/slow={})",
            fast_tape_ok, accept_len, spine_accept, use_tree_la, fast, slow,
        );
    }
    // True when the block's rows sit in committed order (either the accept
    // followed the spine, or the gather below put them there), which is what
    // lets the hidden commit take its GPU-resident fast path.
    let committed_row_order;
    let hidden_rows_written;
    if fast_tape_ok {
        // Tape already captured in tree verify. Restore + replay directly.
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
        hidden_rows_written = big_n;
        committed_row_order = true;
    } else if !force_slow {
        // Post-accept gather instead of a second forward: the tree verify
        // already produced every accepted row at its own linearization slot,
        // with the correct baked-in RoPE phase (the i-th accepted node has
        // depth i+1, so its phase IS its committed position). Relocating those
        // rows into the committed slots reproduces what the re-verify below
        // would compute, without the ~40-50 ms forward that fired on ~31% of
        // cycles at topk=2.
        ddtree_gather_committed_rows(
            gpu,
            target,
            gdn_tape,
            hidden_rb,
            position,
            big_n,
            &accepted_node_indices,
        )?;
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
        hidden_rows_written = big_n;
        committed_row_order = true;
    } else {
        // Slow path (non-spine accept): re-verify the committed prefix to
        // get a linear-order tape AND correctly RoPE'd K written to committed
        // slots. ~40-50 ms cost on 27B.
        let tape_block: Vec<u32> = committed[..accept_len + 1].to_vec();
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        let _tape_verify = verify_dflash_block(
            gpu,
            target,
            &tape_block,
            position,
            hidden_rb,
            Some(gdn_tape),
            false,
            verify_scratch,
        )?;
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
        hidden_rows_written = tape_block.len();
        // The re-verify wrote committed-order rows itself.
        committed_row_order = true;
    }

    // ── 11. D2D-scatter committed rows into draft_scratch.target_hidden ─
    // Mirror the chain path (spec_step_dflash ~line 3911): scatter only the
    // accept_len+1 committed rows directly from hidden_rb into the GPU-resident
    // draft_scratch.target_hidden buffer, then call thlog.append_committed so
    // the next cycle's draft_forward sees prev==l and skips the H2D upload.
    // We still download the same rows to CPU to maintain the target_hidden_host
    // length invariant (checked at cycle entry, line 4992-4995).
    //
    // Fast-tape: hidden_rb holds big_n rows in linearization order; the first
    // rows_to_keep are the committed prefix (spine_accept == true guarantees
    // linear order). Pass block_size=big_n so scatter aligns to ring origin.
    //
    // Slow-tape: the 2nd verify (above) wrote exactly accept_len+1 rows in
    // committed order to hidden_rb. Pass block_size=rows_to_keep.
    //
    // Every branch above leaves hidden_rb in committed row order — the accept
    // followed the spine, the post-accept gather relocated the rows, or the
    // force-slow re-verify wrote them directly — so this is unconditionally the
    // GPU-resident scatter. (The old host-gather arm that read
    // `accepted_node_indices` on the CPU is gone: it maintained only
    // `target_hidden_host`, skipping both the GPU scatter and the thlog append,
    // which would leave the drafter with an unwritten row hole.)
    let rows_to_keep = accept_len + 1;
    debug_assert!(
        committed_row_order,
        "hidden_rb rows must be in committed order before the scatter"
    );
    // Fast/gathered: block_size=big_n, n_rows=rows_to_keep (committed prefix).
    // Force-slow: block_size=rows_to_keep=hidden_rows_written (linear order).
    let block_size = hidden_rows_written;
    scatter_hidden_block_to_interleaved(
        gpu,
        hidden_rb,
        &draft_scratch.target_hidden,
        position,
        block_size,
        rows_to_keep,
        draft_scratch.ctx_modulus(),
    )?;
    let co = target.kv_cache.compact_offset as i32;
    draft_scratch
        .thlog
        .append_committed(position, rows_to_keep, co);
    // Stage 1b: GPU buffer (scratch.target_hidden) is now authoritative — the
    // D2D scatter above populated it. No CPU download needed; the next cycle's
    // draft_forward receives None and skips H2D entirely. target_hidden_host is
    // intentionally NOT updated (it's unused in the GPU-resident path).

    if debug_tm {
        let total = t_all.elapsed();
        eprintln!(
            "[ddtree-tm] draft={:.2}ms topk={:.2}ms build={:.2}ms pre_verify={:.2}ms verify={:.2}ms total={:.2}ms  (N={} accept={})",
            t_draft.as_secs_f64() * 1000.0,
            (t_topk - t_draft).as_secs_f64() * 1000.0,
            (t_build - t_topk).as_secs_f64() * 1000.0,
            (t_pre_verify - t_build).as_secs_f64() * 1000.0,
            (t_post_verify - t_pre_verify).as_secs_f64() * 1000.0,
            total.as_secs_f64() * 1000.0,
            big_n, accept_len,
        );
    }

    Ok(SpecStepResult {
        accepted: accept_len,
        bonus_token,
        drafted,
        committed,
    })
}

/// Seed `target_hidden_host` from the prompt by running the target over
/// each prompt token one at a time with hidden-state extraction enabled.
/// This is a slow but correct MVP path — the target already ran a fast
/// prefill earlier; this exists only to populate `hidden_rb` + host vec
/// with the prompt's layer-selected hidden states.
///
/// Callers with a fast-path prefill that already populates `hidden_rb`
/// should skip this and just call `download_hidden_block(hidden_rb, len)`
/// instead. For MVP we eat the redundant work because it's a one-shot
/// cost at session start.
/// Snapshot the DeltaNet recurrent state into a bounded ring `cks` (pairs of
/// `(seq_pos, snapshot)`) when `interval` tokens have elapsed since the last
/// one. Shared by BOTH the AR `generate` and the DFlash prompt-cache paths to
/// enable resume-from-checkpoint on a divergent client render (see the daemon's
/// `generate` divergence branch + `generate_dflash`). Oldest evicted at `cap`
/// (buffers reused — no realloc churn after warmup). Cheap: one device-to-device
/// memcpy of the recurrent S/scale/conv buffers; no KV copy (FullAttention KV is
/// positional and stays resident, so resume only restores the recurrent state).
/// Gating (resume enabled / no eviction) is the caller's responsibility.
pub fn take_dn_checkpoint(
    cks: &mut Vec<(usize, DeltaNetSnapshot)>,
    dn: &DeltaNetState,
    kv: &KvCache,
    gpu: &mut Gpu,
    pos: usize,
    interval: usize,
    cap: usize,
) {
    if pos == 0 || cap == 0 {
        return;
    }
    match cks.last().map(|(p, _)| *p) {
        Some(p) if pos < p + interval => return,
        Some(p) if p == pos => return,
        _ => {}
    }
    // A new snapshot beyond the ring's first (the charged one) is allowed only
    // from memory the admitted context does not need; otherwise the oldest is
    // recycled, shrinking the ring to what fits.
    let recycle =
        cks.len() >= cap || (!cks.is_empty() && !checkpoint_fits_admitted_context(gpu, kv, dn));
    let mut snap = if recycle {
        cks.remove(0).1
    } else {
        match DeltaNetSnapshot::new_for(gpu, dn) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("take_dn_checkpoint: pos={pos} skipped: snapshot allocation failed: {e}");
                return;
            }
        }
    };
    if let Err(e) = snap.save_from(dn, gpu) {
        // A recycled slot is partially overwritten, so it cannot rejoin the
        // ring either way: free it.
        eprintln!("take_dn_checkpoint: pos={pos} skipped: snapshot save failed: {e}");
        snap.free_gpu(gpu);
        return;
    }
    cks.push((pos, snap));
}

/// Whether one more DeltaNet snapshot can be allocated without eating what the
/// admitted context still needs: `free_now >= snapshot + KV mapping growth to
/// the admitted tokens + the fit's reserve`. No floor set => always true; a
/// memory query or growth-plan error => false (do not allocate).
fn checkpoint_fits_admitted_context(gpu: &Gpu, kv: &KvCache, dn: &DeltaNetState) -> bool {
    let Some(floor) = gpu.kv_fit_floor else {
        return true;
    };
    let Ok((free_now, _)) = gpu.device_mem_info() else {
        return false;
    };
    let Ok(growth) = kv.planned_mapped_growth_bytes(gpu, floor.admitted_tokens) else {
        return false;
    };
    let Ok(snapshot) = usize::try_from(DeltaNetSnapshot::bytes_for(dn)) else {
        return false;
    };
    free_now
        >= snapshot
            .saturating_add(growth)
            .saturating_add(floor.reserve_bytes)
}

/// Outer chunk ceiling of the DFlash prompt seed ([`SeedPrefill::plan`]):
/// the legacy 256 rows under `HIPFIRE_DFLASH_LEGACY_PREFILL=1`, else AR's
/// memory-admitted `ordinary_prefill_chunk_limit` evaluated as if
/// `free_deduction` more device bytes were already allocated (`0` = now; see
/// `ordinary_prefill_chunk_limit_deducting`). Taken once per prompt. Shared
/// by the seed itself and by VMM lane admission, which predicts the seed's
/// chunk plan (and so its checkpoint count) before the lane exists.
pub fn dflash_seed_ceiling(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
    kv_cache: &KvCache,
    free_deduction: usize,
) -> HipResult<usize> {
    if dflash_legacy_prefill() {
        return Ok(qwen35::PREFILL_MAX_BATCH);
    }
    qwen35::prefill::ordinary_prefill_chunk_limit_deducting(
        gpu,
        weights,
        config,
        dn_state,
        kv_cache,
        free_deduction,
    )
}

/// Rows of the next outer seed chunk with `remaining` prompt rows left under
/// `ceiling`: AR's `ordinary_serve_prefill_chunk_len`.
fn dflash_seed_outer_chunk_len(remaining: usize, ceiling: usize) -> usize {
    qwen35::prefill::ordinary_serve_prefill_chunk_len(remaining, ceiling)
        .unwrap_or(remaining.min(ceiling).max(1))
}

/// DeltaNet checkpoints a cold DFlash prompt seed of `prompt_len` rows takes
/// under outer-chunk `ceiling` (the ring's final length): mirrors
/// `seed_target_hidden_from_prompt_abortable_parts`, which runs outer chunks of
/// [`dflash_seed_outer_chunk_len`] and calls [`take_dn_checkpoint`] at the end
/// of each. That call skips `cap == 0`, skips a position below the last
/// checkpoint plus `interval`, and at `cap` recycles the oldest, so the ring
/// length never exceeds `cap`. (A failed snapshot allocation only lowers the
/// real count.)
pub fn dflash_seed_checkpoint_count(
    prompt_len: usize,
    ceiling: usize,
    interval: usize,
    cap: usize,
) -> usize {
    dflash_seed_checkpoint_positions(prompt_len, ceiling, interval, cap).len()
}

/// Positions (the ring's final contents, oldest first) of the DeltaNet
/// checkpoints a cold DFlash prompt seed of `prompt_len` rows takes under
/// outer-chunk `ceiling`; same simulation of [`take_dn_checkpoint`] as
/// [`dflash_seed_checkpoint_count`], which is its length.
pub fn dflash_seed_checkpoint_positions(
    prompt_len: usize,
    ceiling: usize,
    interval: usize,
    cap: usize,
) -> Vec<usize> {
    let mut ring = std::collections::VecDeque::new();
    if cap == 0 {
        return Vec::new();
    }
    let (mut pos, mut last) = (0usize, None::<usize>);
    while pos < prompt_len {
        pos += dflash_seed_outer_chunk_len(prompt_len - pos, ceiling);
        if last.is_some_and(|p| pos < p.saturating_add(interval)) {
            continue;
        }
        if ring.len() >= cap {
            ring.pop_front();
        }
        ring.push_back(pos);
        last = Some(pos);
    }
    ring.into()
}

/// Outer chunk ceiling of the most recent cold DFlash prompt seed
/// (`seed_target_hidden_from_prompt_abortable_parts`) in this process; `0`
/// before any. Read by the state oracle to compare checkpoint positions
/// between walks that planned the same ceiling.
pub fn last_dflash_seed_ceiling() -> usize {
    LAST_DFLASH_SEED_CEILING.load(std::sync::atomic::Ordering::Relaxed)
}

static LAST_DFLASH_SEED_CEILING: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Upper bound on [`dflash_seed_checkpoint_count`] for every outer-chunk
/// ceiling the seed can take at least `min_ceiling`: the maximum over
/// `min_ceiling` and each staged rung `>= min_ceiling` (the count is not
/// monotone in the ceiling, so every larger candidate is evaluated). Valid
/// whenever the seed's actual ceiling is `>= min_ceiling` and a staged rung or
/// `min_ceiling` itself, which holds for [`dflash_seed_ceiling`] evaluated
/// with a free-byte deduction that covers everything allocated before the
/// seed's own query (admission is monotone in free bytes, and a ceiling that
/// does not depend on memory equals `min_ceiling`).
pub fn dflash_seed_checkpoint_bound(
    prompt_len: usize,
    min_ceiling: usize,
    interval: usize,
    cap: usize,
) -> usize {
    qwen35::prefill::WIDENED_RUNGS
        .iter()
        .copied()
        .filter(|&r| r >= min_ceiling)
        .chain(std::iter::once(min_ceiling))
        .map(|ceiling| dflash_seed_checkpoint_count(prompt_len, ceiling, interval, cap))
        .max()
        .unwrap_or(0)
        .min(cap)
}

pub fn seed_target_hidden_from_prompt(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    prompt_tokens: &[u32],
) -> HipResult<()> {
    // Reset target state to avoid double-prefill of the same context.
    target.reset_state(gpu)?;
    // Fast path: one batched prefill populates hidden_rb + KV + dn_state in a
    // single forward, instead of N per-token forwards. On 9B MQ4 with a
    // 6.2k-token prompt this drops prompt ingest from ~51s (121 tok/s) to
    // a few seconds, which is the primary cost of an agent's first turn.
    // `forward_prefill_batch` itself falls back to per-token internally if
    // the KV quant mode / batch size aren't on the fast path, so the call
    // is always safe — the effective cadence just varies.
    qwen35::forward_prefill_batch(
        gpu,
        &target.weights,
        &target.config,
        prompt_tokens,
        0,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        Some(hidden_rb),
        None,
        None,
        None,
    )?;
    // Gather the just-written rows from the ring buffer.
    let block = download_hidden_block(gpu, hidden_rb, prompt_tokens.len())?;
    target_hidden_host.extend_from_slice(&block);
    Ok(())
}

/// `HIPFIRE_DFLASH_LEGACY_PREFILL=1`: the DFlash target prompt prefill keeps
/// its previous route, 256-row seed chunks whose hidden rows go through ring
/// staging, outside the widened chunk and the GDN chunk scan. Default off: the
/// seed prefills on AR's ordinary route (see [`SeedPrefill`]).
pub fn dflash_legacy_prefill() -> bool {
    hipfire_config::developer_bool("HIPFIRE_DFLASH_LEGACY_PREFILL", false)
}

/// Outer chunk plan and scratch shared by the DFlash prompt seeds.
///
/// Default: the plan of AR's ordinary prefill (`hipfire-generate` `ar.rs`,
/// no-eviction branch): the ceiling is `ordinary_prefill_chunk_limit`, taken
/// once per prompt, and each outer chunk is `ordinary_serve_prefill_chunk_len`.
/// Each chunk runs the same forward AR runs, plus the hidden ring, which
/// rides AR's route ([`HiddenStateRingBuffer::rides_ordinary_prefill`]). So
/// after the prompt the target's KV, DeltaNet state and last-token logits are
/// byte-identical to AR's prefill of the same prompt.
///
/// The ring holds only `max_positions` rows (a windowed draft's ring is W
/// rows), so an outer chunk wider than the ring runs as ring-sized pieces
/// ([`SeedPrefill::piece_len`]) whose hidden rows are read back one piece at
/// a time.
///
/// Legacy (`HIPFIRE_DFLASH_LEGACY_PREFILL=1`): 256-row outer chunks.
struct SeedPrefill {
    ceiling: usize,
    /// One scratch for every chunk that does not reuse the retained widened
    /// scratch. Allocated on the first such chunk and freed by
    /// [`SeedPrefill::free`]. Every PBS tensor is overwritten before it is
    /// read, so reuse is bit-identical to a fresh scratch per chunk.
    pbs: Option<qwen35::PrefillBatchScratch>,
}

impl SeedPrefill {
    fn plan(gpu: &Gpu, target: &DflashTargetParts<'_>, end_tokens: usize) -> Self {
        let ceiling = match dflash_seed_ceiling(
            gpu,
            target.weights,
            target.config,
            target.dn_state,
            target.kv_cache,
            qwen35::prefill::request_kv_growth_deduction(gpu, target.kv_cache, end_tokens),
        ) {
            Ok(limit) => limit,
            Err(e) => {
                eprintln!("dflash seed: chunk-limit query failed ({e}); keeping legacy ceiling");
                qwen35::prefill_max_batch(gpu)
            }
        };
        Self { ceiling, pbs: None }
    }

    fn next_len(&self, remaining: usize) -> usize {
        dflash_seed_outer_chunk_len(remaining, self.ceiling)
    }

    /// Rows of the next forward call inside an outer chunk with `remaining`
    /// rows left. A chunk the ring can hold runs whole. A wider one runs as
    /// pieces planned by the widened chunk rule at the ring size rounded down
    /// to the 512-row commit stride: the pieces keep every boundary AR's own
    /// chunk planner makes, and each added boundary falls on one of the 512-row
    /// commits the widened chunk already makes there.
    fn piece_len(hidden_rb: &HiddenStateRingBuffer, remaining: usize) -> usize {
        let ring = hidden_rb.max_positions;
        if remaining <= ring {
            return remaining;
        }
        let stride = qwen35::prefill::WIDENED_COMMIT_ROWS;
        let cap = if ring >= stride {
            ring / stride * stride
        } else {
            ring
        };
        qwen35::prefill::next_exact_prefill_chunk_len(remaining, cap).unwrap_or(remaining.min(cap))
    }

    /// Prefill one piece at `pos`, extracting hidden rows into the ring.
    /// `total` is the seed's token count, which bounds the scratch.
    fn forward(
        &mut self,
        gpu: &mut Gpu,
        target: &mut DflashTargetParts<'_>,
        hidden_rb: &mut HiddenStateRingBuffer,
        chunk: &[u32],
        pos: usize,
        total: usize,
    ) -> HipResult<()> {
        // A widened ceiling reuses the retained widened scratch, and a
        // configured legacy cache (`HIPFIRE_PREFILL_REUSE_PBS=1`) is its own
        // reuse: both run exactly the forward AR runs.
        if self.ceiling > qwen35::prefill::WIDENED_COMMIT_ROWS
            || target.scratch.prefill_batch.is_some()
        {
            return qwen35::forward_prefill_batch(
                gpu,
                target.weights,
                target.config,
                chunk,
                pos,
                target.kv_cache,
                target.dn_state,
                target.scratch,
                Some(hidden_rb),
                None,
                None,
                None,
            );
        }
        if self.pbs.is_none() {
            let rows = self.ceiling.min(total).max(2);
            self.pbs = Some(qwen35::PrefillBatchScratch::new_opt(
                gpu,
                target.config,
                rows,
                false,
            )?);
        }
        qwen35::forward_prefill_batch_with_pbs(
            gpu,
            target.weights,
            target.config,
            chunk,
            pos,
            target.kv_cache,
            target.dn_state,
            target.scratch,
            Some(hidden_rb),
            None,
            None,
            None,
            self.pbs.as_ref(),
            None,
            None,
        )
    }

    fn free(self, gpu: &mut Gpu) {
        if let Some(pbs) = self.pbs {
            let _ = pbs.free_gpu(gpu);
        }
    }
}

/// Abortable variant of `seed_target_hidden_from_prompt`. Chunks the prefill
/// with [`SeedPrefill`] (AR's plan by default) and calls `abort_check`
/// between chunks. Returns `Ok(true)` if aborted (state has been fully reset,
/// so the caller should NOT continue with decode), `Ok(false)` on normal
/// completion. `Some(target_hidden_host)` captures each piece through
/// `download_hidden_block` (a host sync per piece); `None` leaves the same
/// hidden rows on device, so the pieces enqueue back to back. Callers pass
/// `Some` only when the long-reach backfill reads the shadow
/// (`dflash_prefill_host_required`).
///
/// Used by the daemon's `generate_dflash` to honor client-side
/// cancellation on long-context retries (cache-miss scenarios where
/// the full conversation must be re-prefilled from scratch).
#[allow(clippy::too_many_arguments)]
pub fn seed_target_hidden_from_prompt_abortable(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: Option<&mut Vec<f32>>,
    prompt_tokens: &[u32],
    abort_check: &dyn Fn() -> bool,
    // Optional DeltaNet checkpoint ring for divergent-render resume. When
    // `Some`, the recurrent state is snapshotted every `ckpt_interval` tokens
    // (bounded at `ckpt_cap`). `None` ⇒ no checkpointing (zero overhead).
    checkpoints: Option<&mut Vec<(usize, DeltaNetSnapshot)>>,
    ckpt_interval: usize,
    ckpt_cap: usize,
) -> HipResult<bool> {
    let mut parts = DflashTargetParts::from_slot(target);
    seed_target_hidden_from_prompt_abortable_parts(
        gpu,
        &mut parts,
        hidden_rb,
        target_hidden_host,
        prompt_tokens,
        abort_check,
        checkpoints,
        ckpt_interval,
        ckpt_cap,
    )
}

/// [`seed_target_hidden_from_prompt_abortable`] over explicit trunk owners.
#[allow(clippy::too_many_arguments)]
pub fn seed_target_hidden_from_prompt_abortable_parts(
    gpu: &mut Gpu,
    target: &mut DflashTargetParts<'_>,
    hidden_rb: &mut HiddenStateRingBuffer,
    mut target_hidden_host: Option<&mut Vec<f32>>,
    prompt_tokens: &[u32],
    abort_check: &dyn Fn() -> bool,
    mut checkpoints: Option<&mut Vec<(usize, DeltaNetSnapshot)>>,
    ckpt_interval: usize,
    ckpt_cap: usize,
) -> HipResult<bool> {
    target.reset_state(gpu)?;
    if let Some(host) = target_hidden_host.as_deref_mut() {
        host.clear();
    }
    if let Some(cks) = checkpoints.as_deref_mut() {
        // fresh cold prefill ⇒ stale checkpoints no longer valid; free their GPU buffers
        for (_, snap) in cks.drain(..) {
            snap.free_gpu(gpu);
        }
    }
    let mut seed = SeedPrefill::plan(gpu, target, prompt_tokens.len());
    LAST_DFLASH_SEED_CEILING.store(seed.ceiling, std::sync::atomic::Ordering::Relaxed);
    let result = (|| -> HipResult<bool> {
        let mut seq_pos: usize = 0;
        while seq_pos < prompt_tokens.len() {
            if abort_check() {
                let _ = target.reset_state(gpu);
                if let Some(host) = target_hidden_host.as_deref_mut() {
                    host.clear();
                }
                if let Some(cks) = checkpoints.as_deref_mut() {
                    for (_, snap) in cks.drain(..) {
                        snap.free_gpu(gpu);
                    }
                }
                return Ok(true);
            }
            let end = seq_pos + seed.next_len(prompt_tokens.len() - seq_pos);
            while seq_pos < end {
                let piece = SeedPrefill::piece_len(hidden_rb, end - seq_pos);
                let chunk = &prompt_tokens[seq_pos..seq_pos + piece];
                seed.forward(gpu, target, hidden_rb, chunk, seq_pos, prompt_tokens.len())?;
                append_prefill_hidden_host(
                    gpu,
                    hidden_rb,
                    piece,
                    target_hidden_host.as_deref_mut(),
                )?;
                seq_pos += piece;
            }
            // Prompt-prefill progress (silent outside a `prompt_phase`).
            hipfire_runtime::prefill_progress::report(seq_pos, prompt_tokens.len());
            if let Some(cks) = checkpoints.as_deref_mut() {
                take_dn_checkpoint(
                    cks,
                    target.dn_state,
                    target.kv_cache,
                    gpu,
                    seq_pos,
                    ckpt_interval,
                    ckpt_cap,
                );
            }
        }
        Ok(false)
    })();
    seed.free(gpu);
    result
}

/// Incremental prompt seed for the DFlash prompt cache: prefill ONLY the
/// `suffix` tokens starting at absolute position `start_pos`, WITHOUT resetting
/// target KV / DeltaNet state. Used when a turn is a pure extension of the
/// cached conversation (LCP == prior length) — the target KV[0..start_pos] and
/// the recurrent DeltaNet state are already correct from the prior turn, so we
/// only advance them through the new suffix. `hidden_rb` is left holding the
/// suffix's extracted hidden rows so the caller can scatter them into the
/// draft's cumulative `target_hidden` at row offset `start_pos` (the draft's
/// projection cache, keyed on `draft_ctx_cached_rows`, then projects only the
/// new rows — same delta path decode already uses).
///
/// Correctness rests on the same invariant the AR `generate` cache relies on:
/// `forward_prefill_batch` at a nonzero `seq_pos` continues the hybrid
/// (FullAttention KV + DeltaNet recurrent) forward exactly as if the prefix had
/// just been prefilled, because the recurrent state is naturally at the end of
/// the prior conversation (pure extension — no rewind). Returns `Ok(true)` if
/// aborted mid-prefill (state left as-is; caller must full-reset & retry),
/// `Ok(false)` on completion.
#[allow(clippy::too_many_arguments)]
pub fn seed_target_hidden_suffix_abortable(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    hidden_rb: &mut HiddenStateRingBuffer,
    suffix: &[u32],
    start_pos: usize,
    abort_check: &dyn Fn() -> bool,
    // Optional DeltaNet checkpoint ring (see from_prompt variant). Lets a HIT
    // or a resume keep adding checkpoints as the conversation grows, so a later
    // divergence resumes from a recent point rather than the initial prefill.
    checkpoints: Option<&mut Vec<(usize, DeltaNetSnapshot)>>,
    ckpt_interval: usize,
    ckpt_cap: usize,
) -> HipResult<bool> {
    let mut parts = DflashTargetParts::from_slot(target);
    seed_target_hidden_suffix_abortable_parts(
        gpu,
        &mut parts,
        hidden_rb,
        suffix,
        start_pos,
        abort_check,
        checkpoints,
        ckpt_interval,
        ckpt_cap,
    )
}

/// [`seed_target_hidden_suffix_abortable`] over explicit trunk owners.
#[allow(clippy::too_many_arguments)]
pub fn seed_target_hidden_suffix_abortable_parts(
    gpu: &mut Gpu,
    target: &mut DflashTargetParts<'_>,
    hidden_rb: &mut HiddenStateRingBuffer,
    suffix: &[u32],
    start_pos: usize,
    abort_check: &dyn Fn() -> bool,
    mut checkpoints: Option<&mut Vec<(usize, DeltaNetSnapshot)>>,
    ckpt_interval: usize,
    ckpt_cap: usize,
) -> HipResult<bool> {
    let mut seed = SeedPrefill::plan(gpu, target, start_pos + suffix.len());
    let result = (|| -> HipResult<bool> {
        let mut off: usize = 0;
        let mut pos = start_pos;
        while off < suffix.len() {
            if abort_check() {
                return Ok(true);
            }
            let end = off + seed.next_len(suffix.len() - off);
            while off < end {
                let piece = SeedPrefill::piece_len(hidden_rb, end - off);
                seed.forward(
                    gpu,
                    target,
                    hidden_rb,
                    &suffix[off..off + piece],
                    pos,
                    suffix.len(),
                )?;
                pos += piece;
                off += piece;
            }
            // Prompt-prefill progress (silent outside a `prompt_phase`, so
            // forced advances and terminal repair stay quiet).
            hipfire_runtime::prefill_progress::report(off, suffix.len());
            if let Some(cks) = checkpoints.as_deref_mut() {
                take_dn_checkpoint(
                    cks,
                    target.dn_state,
                    target.kv_cache,
                    gpu,
                    pos,
                    ckpt_interval,
                    ckpt_cap,
                );
            }
        }
        Ok(false)
    })();
    seed.free(gpu);
    result
}

/// Mirror a TriAttention KV eviction into the DFlash draft's GPU-resident
/// `target_hidden` and `target_hidden_abs_positions`, so the draft's cross-
/// attention sees the same subset of context target now has.
///
/// `retain_mask` is the source-position retain selection returned by
/// `EvictionCtx::maybe_evict` (ascending, length == budget). An empty
/// `retain_mask` is a no-op — the caller should have skipped calling this
/// (CASK m-fold path returns empty because merged slots don't map cleanly
/// to a single source position).
///
/// Implementation: download the relevant `physical` rows of `target_hidden`,
/// reorder to `budget` rows on the host per `retain_mask`, upload back. Runs
/// at eviction cadence (~once per β decoded tokens) so the PCIe round-trip
/// is amortized — perf impact is small relative to the τ recovery.
///
/// Post-conditions (all via `draft_scratch.thlog.rebuild_after_eviction`):
/// - `thlog.abs_positions()` has exactly `budget` entries, each pulled from
///   `retain_mask[i]` of the pre-eviction abs_positions.
/// - `draft_scratch.target_hidden` GPU slots [0..budget) hold the retained
///   rows in ascending source order.
/// - `thlog.uploaded_rows() == budget` so the next draft_forward sees the
///   compacted layout as already-uploaded (and the projection cache is
///   invalidated).
pub fn apply_eviction_retain_to_draft(
    gpu: &mut rdna_compute::Gpu,
    draft_scratch: &mut dflash::DflashScratch,
    retain_mask: &[u32],
    ne: usize,
    h: usize,
    physical: usize,
) -> HipResult<()> {
    if retain_mask.is_empty() {
        return Ok(());
    }
    let row_floats = ne * h;
    // Download only the populated prefix of target_hidden. `alloc_tensor`
    // is sized to max_ctx_len — we just need `physical` rows.
    let mut host = vec![0f32; physical * row_floats];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                host.as_mut_ptr() as *mut u8,
                host.len() * std::mem::size_of::<f32>(),
            )
        };
        gpu.hip
            .memcpy_dtoh(bytes, &draft_scratch.target_hidden.buf)?;
    }
    let budget = retain_mask.len();
    let mut compacted = Vec::with_capacity(budget * row_floats);
    let mut new_abs = Vec::with_capacity(budget);
    for &src_idx in retain_mask {
        let s = src_idx as usize;
        let row = &host[s * row_floats..(s + 1) * row_floats];
        compacted.extend_from_slice(row);
        new_abs.push(
            *draft_scratch
                .thlog
                .abs_positions()
                .get(s)
                .expect("retain_mask index out of range for abs_positions"),
        );
    }
    let dst_bytes = budget * row_floats * std::mem::size_of::<f32>();
    let compacted_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(compacted.as_ptr() as *const u8, dst_bytes) };
    gpu.hip
        .memcpy_htod(&draft_scratch.target_hidden.buf, compacted_bytes)?;
    // Replace the row layout with the compacted positions and invalidate the
    // per-layer k_ctx/v_ctx projection cache (indexed by the pre-eviction row
    // layout, now stale — rebuilt on the next draft_forward; one slow cycle per
    // eviction is fine). `rebuild_after_eviction` sets the upload watermark to
    // `new_abs.len()` (== budget) so the two cursors can't desync.
    draft_scratch.thlog.rebuild_after_eviction(new_abs);
    Ok(())
}

#[cfg(test)]
mod sparse_target_tests {
    //! Host-only exactness of the sparse FAST_SAMPLE target rows and sparse
    //! selector q against the dense path (same uniforms ⇒ same decision, same
    //! RNG state).
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 24
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
        fn unit(&mut self) -> f32 {
            (self.next_u64() & 0xFF_FFFF) as f32 / 16_777_216.0
        }
    }

    /// Host twin of `sampled_rows_compact_f32` + the host assembly of
    /// `dflash_target_rows_sparse`: kept = `p >= tau[r]` ascending, true
    /// count, first `cap` stored, garbage past the count.
    fn host_compact(
        probs: &[f32],
        tau: Vec<f32>,
        z: Vec<f32>,
        vocab: usize,
        cap: usize,
    ) -> FastTargetRows {
        let rows = tau.len();
        let mut counts = vec![0u32; rows];
        let mut idx = vec![0xDEAD_BEEFu32; rows * cap];
        let mut val = vec![f32::NAN; rows * cap];
        for r in 0..rows {
            let mut c = 0usize;
            for j in 0..vocab {
                let p = probs[r * vocab + j];
                if p >= tau[r] {
                    if c < cap {
                        idx[r * cap + c] = j as u32;
                        val[r * cap + c] = p;
                    }
                    c += 1;
                }
            }
            counts[r] = c as u32;
        }
        let mut f = FastTargetRows {
            tau,
            z,
            capacity: cap,
            counts,
            idx,
            val,
            ..FastTargetRows::default()
        };
        let n = f.plan_dense_rows(vocab).unwrap();
        f.dense_rows = vec![0.0; n * vocab];
        for r in 0..rows {
            let s = f.dense_slot[r];
            if s != SPARSE_NONE {
                let s = s as usize;
                f.dense_rows[s * vocab..(s + 1) * vocab]
                    .copy_from_slice(&probs[r * vocab..(r + 1) * vocab]);
            }
        }
        f
    }

    /// The kernel's `tau`/`Z` for a row, with identity rows, exact-tie
    /// boundaries (tau = an existing probability) and AR-truncated rows.
    fn row_tau_z(rng: &mut Lcg, raw: &[f32], top_p: f32, top_k: usize) -> (f32, f32) {
        let vocab = raw.len();
        match rng.below(10) {
            0 => (0.0, 1.0),
            1 => {
                let t = raw[rng.below(vocab)];
                (t, raw.iter().filter(|&&p| p >= t).sum())
            }
            _ => {
                let mut ar = raw.to_vec();
                apply_host_ar_trunc(&mut ar, top_k, top_p);
                let t = (0..vocab)
                    .filter(|&i| ar[i] > 0.0)
                    .map(|i| raw[i])
                    .fold(f32::INFINITY, f32::min);
                if t.is_finite() {
                    (t, raw.iter().filter(|&&p| p >= t).sum())
                } else {
                    (0.0, 1.0)
                }
            }
        }
    }

    #[test]
    fn sparse_accept_matches_dense_10k_cases() {
        let mut rng = Lcg(0x5EED_0001);
        let (mut full, mut zero, mut budget, mut rejected) = (0usize, 0usize, 0usize, 0usize);
        let (mut dense_rows_seen, mut overflow_seen, mut sparse_rows_seen) =
            (0usize, 0usize, 0usize);
        let mut empty_q_rows = 0usize;
        for case in 0..10_000usize {
            let vocab = 2 + rng.below(if case % 50 == 0 { 400 } else { 40 });
            let b = 2 + rng.below(5);
            let temp = [0.5f32, 0.8, 1.0][rng.below(3)];
            let top_p = [1.0f32, 0.9, 0.5, 0.3][rng.below(4)];
            let top_k = [0usize, 1, 3, 20, 64][rng.below(5)];
            let cap = [DFLASH_SPARSE_CAP, DFLASH_SPARSE_CAP, 1, 2, 4, 8][rng.below(6)];
            let quantized = rng.below(3) == 0;
            let (mut probs, mut tau, mut z) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..b {
                let logits: Vec<f32> = (0..vocab)
                    .map(|_| {
                        if quantized {
                            rng.below(5) as f32 * 0.7
                        } else {
                            rng.unit() * 8.0 - 4.0
                        }
                    })
                    .collect();
                let mut row = Vec::new();
                softmax_temp_into(&logits, temp, &mut row);
                let (t, zz) = row_tau_z(&mut rng, &row, top_p, top_k);
                tau.push(t);
                z.push(zz);
                probs.extend_from_slice(&row);
            }
            let dense_fast = FastTargetRows::dense(probs.clone(), tau.clone(), z.clone());
            let sparse_fast = host_compact(&probs, tau.clone(), z.clone(), vocab, cap);
            for r in 0..b {
                if sparse_fast.dense_slot[r] == SPARSE_NONE {
                    sparse_rows_seen += 1;
                } else {
                    dense_rows_seen += 1;
                    if tau[r] > 0.0 && sparse_fast.counts[r] as usize > cap {
                        overflow_seen += 1;
                    }
                }
                // Rebuilt legacy rows are bit-identical to the dense rows.
                let (mut a, mut c) = (Vec::new(), Vec::new());
                sampled_target_row_into(
                    &mut a,
                    r,
                    vocab,
                    Some(&dense_fast),
                    &[],
                    temp,
                    top_k,
                    top_p,
                );
                sampled_target_row_into(
                    &mut c,
                    r,
                    vocab,
                    Some(&sparse_fast),
                    &[],
                    temp,
                    top_k,
                    top_p,
                );
                assert_eq!(a, c, "case {case} row {r}");
            }

            // Selector q: candidates with duplicates, zeros, overlap with the
            // target support, and empty rows.
            let (mut cands, mut cprobs, mut dq) = (Vec::new(), Vec::new(), Vec::new());
            let mut sq = SparseQRows::default();
            sq.clear();
            for i in 0..b - 1 {
                let k = rng.below(7);
                let (mut c, mut p) = (Vec::new(), Vec::new());
                for _ in 0..k {
                    let mut id = rng.below(vocab);
                    if rng.below(10) < 6 && tau[i] > 0.0 {
                        let start = rng.below(vocab);
                        for j in 0..vocab {
                            let cand = (start + j) % vocab;
                            if probs[i * vocab + cand] >= tau[i] {
                                id = cand;
                                break;
                            }
                        }
                    }
                    c.push(id as u32);
                    p.push(match rng.below(10) {
                        0 => 0.0,
                        1 => 1.0,
                        2 => rng.unit() * 3.0,
                        _ => rng.unit() * 0.5,
                    });
                }
                if k == 0 {
                    empty_q_rows += 1;
                }
                // The dense row's last-write rule.
                let mut row = vec![0.0f32; vocab];
                for (&id, &v) in c.iter().zip(&p) {
                    row[id as usize] = v;
                }
                sq.push_row(&c, &p, vocab).unwrap();
                dq.push(row);
                cands.push(c);
                cprobs.push(p);
            }
            let mut block = vec![rng.below(vocab) as u32];
            let mut selected = Vec::new();
            for i in 0..b - 1 {
                let tok = if !cands[i].is_empty() && rng.below(2) == 0 {
                    cands[i][rng.below(cands[i].len())]
                } else {
                    rng.below(vocab) as u32
                };
                block.push(tok);
                selected.push(if rng.below(10) == 0 { 0.0 } else { rng.unit() });
            }
            let cfg = DflashSampleCfg {
                temp,
                top_p,
                top_k,
                cactus_delta: if rng.below(8) == 0 { 0.1 } else { 0.0 },
            };
            let max_accept = if rng.below(10) < 7 {
                usize::MAX
            } else {
                rng.below(b + 1)
            };
            let seed = rng.next_u64() | 1;
            let q_dense = DflashSampledDraft {
                selected_q: selected.clone(),
                dense_q: dq,
                ..DflashSampledDraft::default()
            };
            let q_sparse = DflashSampledDraft {
                selected_q: selected,
                sparse_q: sq,
                sparse: true,
                ..DflashSampledDraft::default()
            };
            let run = |fast: &FastTargetRows, q: &DflashSampledDraft, scratch: &mut Vec<f32>| {
                let mut s = seed;
                let d = dflash_sampled_accept_rows(
                    &block,
                    q,
                    Some(fast),
                    &[],
                    vocab,
                    cfg,
                    max_accept,
                    &mut s,
                    scratch,
                );
                (d.accepted_drafts, d.bonus_token, d.committed_len, s)
            };
            let mut scr = Vec::new();
            let reference = run(&dense_fast, &q_dense, &mut scr);
            if reference.0 == b - 1 {
                full += 1;
            } else if reference.0 == 0 {
                zero += 1;
            }
            if reference.0 == max_accept && max_accept < b - 1 {
                budget += 1;
            }
            if reference.0 < b - 1 && reference.0 < max_accept {
                rejected += 1;
            }
            let combos: [(&str, &FastTargetRows, &DflashSampledDraft); 3] = [
                ("sparse/sparse", &sparse_fast, &q_sparse),
                ("sparse/dense", &sparse_fast, &q_dense),
                ("dense/sparse", &dense_fast, &q_sparse),
            ];
            for (name, f, q) in combos {
                let mut s = Vec::new();
                let got = run(f, q, &mut s);
                assert_eq!(got, reference, "case {case} {name}");
                let all_sparse = sparse_fast.dense_slot.iter().all(|&s| s == SPARSE_NONE);
                if name == "sparse/sparse" && all_sparse && cfg.cactus_delta <= 0.0 {
                    assert_eq!(
                        s.capacity(),
                        0,
                        "case {case}: sparse path built a vocab row"
                    );
                }
            }
        }
        assert!(
            full > 0 && zero > 0 && budget > 0 && rejected > 0,
            "coverage {full} {zero} {budget} {rejected}"
        );
        assert!(
            dense_rows_seen > 0 && overflow_seen > 0 && sparse_rows_seen > 0,
            "rows {dense_rows_seen} {overflow_seen} {sparse_rows_seen}"
        );
        assert!(empty_q_rows > 0);
    }

    #[test]
    fn sparse_residual_categorical_match_dense_10k_cases() {
        let mut rng = Lcg(0xC0FF_EE01);
        for case in 0..10_000usize {
            let vocab = 1 + rng.below(if case % 40 == 0 { 300 } else { 24 });
            let inv_z = 1.0f32 / (0.2 + rng.unit());
            let tied = rng.below(3) == 0;
            let dens = 1 + rng.below(5);
            let (mut tidx, mut tval) = (Vec::new(), Vec::new());
            for j in 0..vocab {
                if rng.below(dens) == 0 && rng.below(8) != 0 {
                    tidx.push(j as u32);
                    tval.push(if tied {
                        (1 + rng.below(3)) as f32 * 0.07
                    } else {
                        0.01 + rng.unit() * 0.4
                    });
                }
            }
            let (mut qidx, mut qval) = (Vec::new(), Vec::new());
            for j in 0..vocab {
                if rng.below(4) == 0 {
                    qidx.push(j as u32);
                    qval.push(match rng.below(6) {
                        0 => 0.0,
                        1 => rng.unit() * 3.0,
                        2 => -rng.unit() * 0.2,
                        3 if tied => (1 + rng.below(3)) as f32 * 0.07,
                        _ => rng.unit() * 0.4,
                    });
                }
            }
            let mut p = vec![0.0f32; vocab];
            for (&i, &v) in tidx.iter().zip(&tval) {
                p[i as usize] = v * inv_z;
            }
            let mut qd = vec![0.0f32; vocab];
            for (&i, &v) in qidx.iter().zip(&qval) {
                qd[i as usize] = v;
            }
            let mut us = vec![
                0.0f32,
                -0.5,
                f32::NAN,
                f32::MIN_POSITIVE,
                0.25,
                0.5,
                0.75,
                1.0 - 1.0 / 16_777_216.0,
                rng.unit(),
                rng.unit(),
            ];
            let nudge = |us: &mut Vec<f32>, x: f32| {
                if x > 0.0 && x.is_finite() {
                    us.push(x);
                    us.push(f32::from_bits(x.to_bits() - 1));
                    us.push(f32::from_bits(x.to_bits() + 1));
                }
            };
            // Categorical boundaries: u equal to each running sum.
            let mut acc = 0.0f32;
            for &x in &p {
                acc += x;
                nudge(&mut us, acc);
            }
            // Residual boundaries: u * sum equal to each running residual sum.
            let mut rs = 0.0f32;
            for i in 0..vocab {
                let d = p[i] - qd[i];
                if d > 0.0 {
                    rs += d;
                }
            }
            if rs > 0.0 {
                let mut a = 0.0f32;
                for i in 0..vocab {
                    let d = p[i] - qd[i];
                    if d > 0.0 {
                        a += d;
                        nudge(&mut us, a / rs);
                    }
                }
            }
            for &u in &us {
                let dense = sample_residual(&p, &qd, u);
                assert_eq!(
                    dense,
                    sample_residual_sparse(&tidx, &tval, inv_z, &qidx, &qval, vocab, u),
                    "case {case} u {u} residual"
                );
                assert_eq!(
                    dense,
                    sample_residual_dense_p_sparse_q(&p, &qidx, &qval, u),
                    "case {case} u {u} residual dense-p"
                );
                assert_eq!(
                    sample_categorical(&p, u),
                    sample_categorical_sparse(&tidx, &tval, inv_z, vocab, u),
                    "case {case} u {u} categorical"
                );
            }
            assert_eq!(
                argmax_u32(&p),
                argmax_sparse(&tidx, &tval, inv_z, vocab),
                "case {case} argmax"
            );
        }
    }

    #[test]
    fn sparse_point_mass_residual_matches_dense_10k_cases() {
        let mut rng = Lcg(0x90_1A7);
        for case in 0..10_000usize {
            let vocab = 1 + rng.below(30);
            let (mut idx, mut val) = (Vec::new(), Vec::new());
            let mut p = vec![0.0f32; vocab];
            for j in 0..vocab {
                if rng.below(3) == 0 {
                    let v = if rng.below(4) == 0 {
                        1.0
                    } else {
                        rng.unit() * 0.6
                    };
                    idx.push(j as u32);
                    val.push(v);
                    p[j] = v;
                }
            }
            let ex = rng.below(vocab) as u32;
            for u in [0.0f32, 0.3, 0.999, rng.unit(), rng.unit()] {
                assert_eq!(
                    sample_residual_point_mass(&p, ex, u),
                    sample_residual_sparse(&idx, &val, 1.0, &[ex], &[1.0], vocab, u),
                    "case {case} u {u}"
                );
            }
        }
    }

    #[test]
    fn sparse_q_rows_reproduce_last_write_and_reject_bad_ids() {
        let mut q = SparseQRows::default();
        q.clear();
        q.push_row(&[5, 3, 5, 9, 3], &[0.1, 0.2, 0.3, 0.4, 0.5], 16)
            .unwrap();
        q.push_row(&[], &[], 16).unwrap();
        q.push_row(&[7, 7], &[0.25, 0.0], 16).unwrap();
        assert_eq!(q.rows(), 3);
        assert_eq!(q.row(0), (&[3u32, 5, 9][..], &[0.5f32, 0.3, 0.4][..]));
        assert_eq!(q.row(1), (&[0u32; 0][..], &[0f32; 0][..]));
        assert_eq!(q.row(2), (&[7u32][..], &[0.0f32][..]));
        assert!(q.push_row(&[16], &[0.5], 16).is_err());
    }

    #[test]
    fn plan_dense_rows_routes_identity_overflow_and_rejects_unsorted() {
        let vocab = 6usize;
        // row0: tau 0 (identity), row1: 3 kept > cap 2 (overflow), row2: 1 kept.
        let probs = [
            0.2f32, 0.2, 0.2, 0.2, 0.1, 0.1, //
            0.3, 0.3, 0.3, 0.05, 0.03, 0.02, //
            0.9, 0.02, 0.02, 0.02, 0.02, 0.02,
        ];
        let f = host_compact(&probs, vec![0.0, 0.3, 0.5], vec![1.0, 0.9, 0.9], vocab, 2);
        assert_eq!(f.dense_slot, vec![0, 1, SPARSE_NONE]);
        match f.row(2, vocab) {
            TargetRowRef::Sparse { idx, val, .. } => {
                assert_eq!(idx, &[0u32][..]);
                assert_eq!(val, &[0.9f32][..]);
            }
            _ => panic!("row 2 must be sparse"),
        }
        match f.row(1, vocab) {
            TargetRowRef::Dense { raw, .. } => assert_eq!(raw, &probs[vocab..2 * vocab]),
            _ => panic!("row 1 must be dense"),
        }
        let mut bad = FastTargetRows {
            tau: vec![0.1],
            z: vec![1.0],
            capacity: 2,
            counts: vec![2],
            idx: vec![5, 3],
            val: vec![0.5, 0.5],
            ..FastTargetRows::default()
        };
        assert!(bad.plan_dense_rows(vocab).is_err());
    }

    #[test]
    fn cactus_disables_the_sparse_representation() {
        let cfg = DflashSampleCfg {
            temp: 0.8,
            top_p: 0.9,
            top_k: 20,
            cactus_delta: 0.1,
        };
        assert!(!dflash_sparse_target_enabled(&cfg));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_slot_alternating_keys_hit_after_two_misses() {
        let mut s = TwoSlot::<u64>::new();
        // Cold: slot 0, then slot 1 (an unwritten slot is always the victim).
        assert_eq!(s.lookup(&10), Err(0));
        assert!(!s.written(0));
        s.set(0, 10);
        assert_eq!(s.lookup(&20), Err(1));
        s.set(1, 20);
        // A/B alternation now hits in either slot, forever.
        for _ in 0..4 {
            assert_eq!(s.lookup(&10), Ok(0));
            s.mark_used(0);
            assert_eq!(s.lookup(&20), Ok(1));
            s.mark_used(1);
        }
        assert!(s.written(0) && s.written(1));
    }

    #[test]
    fn two_slot_miss_evicts_the_least_recently_used_slot() {
        let mut s = TwoSlot::<u64>::new();
        s.set(0, 10);
        s.set(1, 20);
        // Slot 1 is most recent: a third key evicts slot 0.
        assert_eq!(s.lookup(&30), Err(0));
        s.set(0, 30);
        // Now slot 0 is most recent: the next new key evicts slot 1.
        assert_eq!(s.lookup(&40), Err(1));
        // A hit refreshes recency: touching slot 1 makes slot 0 the victim.
        assert_eq!(s.lookup(&20), Ok(1));
        s.mark_used(1);
        assert_eq!(s.lookup(&40), Err(0));
        // The evicted key no longer hits.
        s.set(0, 40);
        assert_eq!(s.lookup(&30), Err(1));
        assert_eq!(s.lookup(&40), Ok(0));
    }

    #[test]
    fn two_slot_unkeyed_slot_is_the_next_victim_and_not_written() {
        let mut s = TwoSlot::<u64>::new();
        s.set(0, 10);
        s.set(1, 20);
        // An unkeyed slot (failed upload) is the next victim and no longer hits.
        assert_eq!(s.take(1), Some(20));
        assert!(!s.written(1));
        assert_eq!(s.lookup(&20), Err(1));
        assert_eq!(s.lookup(&10), Ok(0));
        // `lookup_by` sees only written slots.
        assert_eq!(s.lookup_by(|&k| k == 10), Ok(0));
        assert_eq!(s.lookup_by(|_| true), Ok(0));
    }

    #[test]
    fn seed_checkpoint_count_follows_the_outer_chunk_plan() {
        // interval 2048, cap 8: a checkpoint at each outer chunk end at least
        // `interval` past the previous one.
        let count = |p, ceiling| dflash_seed_checkpoint_count(p, ceiling, 2048, 8);
        assert_eq!(count(8316, 8192), 1);
        assert_eq!(count(8316, 4096), 2);
        assert_eq!(count(8316, 2048), 4);
        assert_eq!(count(8316, 512), 4);
        assert_eq!(count(4300, 2048), 2);
        assert_eq!(count(4300, 4096), 1);
        // A prompt inside one chunk checkpoints once, at its end.
        assert_eq!(count(100, 8192), 1);
        assert_eq!(count(0, 8192), 0);
    }

    #[test]
    fn seed_checkpoint_count_respects_cap_and_policy() {
        // Ring recycling never exceeds the cap.
        assert_eq!(dflash_seed_checkpoint_count(8316, 512, 512, 3), 3);
        assert_eq!(dflash_seed_checkpoint_count(8316, 512, 512, 100), 16);
        // Resume off (cap 0) takes none.
        assert_eq!(dflash_seed_checkpoint_count(8316, 512, 512, 0), 0);
        // A tail chunk shorter than the interval does not add one.
        assert_eq!(dflash_seed_checkpoint_count(4100, 2048, 2048, 8), 2);
    }

    #[test]
    fn seed_checkpoint_bound_is_the_max_over_larger_ceilings() {
        // Not monotone in the ceiling: P=4100, interval 2048. 2048-row chunks
        // end at 2048 and 4096 (both taken): 2; a 4096-row ceiling ends at
        // 4096 only: 1.
        assert_eq!(dflash_seed_checkpoint_count(4100, 4096, 2048, 8), 1);
        // Bound from min_ceiling 4096 covers 4096 and 8192 only.
        assert_eq!(dflash_seed_checkpoint_bound(4100, 4096, 2048, 8), 1);
        // From 512 every rung is a candidate: the worst is 2.
        assert_eq!(dflash_seed_checkpoint_bound(4100, 512, 2048, 8), 2);
        // A bound is never below the count at its own minimum and respects cap.
        for p in [1, 511, 512, 2047, 2049, 4300, 8316, 20000] {
            for min in [512, 1024, 2048, 4096, 8192] {
                let b = dflash_seed_checkpoint_bound(p, min, 1024, 4);
                assert!(b >= dflash_seed_checkpoint_count(p, min, 1024, 4));
                assert!(b <= 4);
            }
        }
        assert_eq!(dflash_seed_checkpoint_bound(8316, 512, 2048, 0), 0);
    }

    /// `sample_residual_point_mass` must equal `sample_residual` against an
    /// explicit one-hot row, bit for bit, and never return the excluded token
    /// when the residual has mass.
    #[test]
    fn residual_point_mass_matches_onehot() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let us = [0.0f32, 1e-6, 0.1, 0.25, 0.5, 0.75, 0.999, 0.999_999];
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for &n in &[2usize, 5, 64, 257] {
            for _ in 0..8 {
                let mut r: Vec<f32> = (0..n)
                    .map(|_| {
                        let x = xorshift_next_unit(&mut rng);
                        // Sparse-ish rows: many exact zeros like a nucleus.
                        if x < 0.5 {
                            0.0
                        } else {
                            x
                        }
                    })
                    .collect();
                let s: f32 = r.iter().sum();
                if s > 0.0 {
                    r.iter_mut().for_each(|v| *v /= s);
                } else {
                    r[0] = 1.0;
                }
                rows.push(r);
            }
        }
        // p(excluded) = 1 (degenerate), ties, p(excluded) = 0.
        rows.push(vec![0.0, 1.0, 0.0]);
        rows.push(vec![0.25, 0.25, 0.25, 0.25]);
        rows.push(vec![0.5, 0.5, 0.0]);
        rows.push(vec![1.0]);
        for row in &rows {
            for ex in 0..row.len() {
                let mut onehot = vec![0.0f32; row.len()];
                onehot[ex] = 1.0;
                let sum_without: f32 = row
                    .iter()
                    .enumerate()
                    .filter(|&(i, &p)| i != ex && p > 0.0)
                    .map(|(_, &p)| p)
                    .sum();
                for &u in &us {
                    let a = sample_residual(row, &onehot, u);
                    let b = sample_residual_point_mass(row, ex as u32, u);
                    assert_eq!(a, b, "row={row:?} ex={ex} u={u}");
                    if sum_without > 0.0 {
                        assert_ne!(b as usize, ex, "row={row:?} ex={ex} u={u}");
                    }
                }
            }
        }
    }

    /// A spine accept needs no row motion — this is what keeps the existing
    /// fast path free of any gather work.
    #[test]
    fn commit_runs_empty_on_spine_accept() {
        assert!(ddtree_commit_row_runs(&[]).is_empty());
        assert!(ddtree_commit_row_runs(&[0]).is_empty());
        assert!(ddtree_commit_row_runs(&[0, 1, 2, 3]).is_empty());
    }

    /// Committed row `i+1` must take linearization slot `accepted[i]+1`, and a
    /// contiguous stretch sharing one delta must coalesce into a single copy —
    /// per-row copies would still be correct but cost one D2D launch each,
    /// across every KV plane and tape buffer.
    #[test]
    fn commit_runs_coalesce_uniform_delta() {
        // Detour at depth 1: slots 2,3,4 carry committed rows 1,2,3 (delta +1).
        assert_eq!(ddtree_commit_row_runs(&[1, 2, 3]), vec![(1, 2, 3)]);
        // Spine prefix stays put; only the tail shifts.
        assert_eq!(ddtree_commit_row_runs(&[0, 1, 3, 4]), vec![(3, 4, 2)]);
    }

    /// Two detours with different deltas cannot share a run.
    #[test]
    fn commit_runs_split_on_delta_change() {
        // rows 1,2 <- slots 2,3 (delta +1); row 3 <- slot 5 (delta +2).
        assert_eq!(
            ddtree_commit_row_runs(&[1, 2, 4]),
            vec![(1, 2, 2), (3, 5, 1)]
        );
    }

    /// A coalesced run overlaps itself when its shift is shorter than the run
    /// (HIPF-136). The issued copies must be pairwise source/destination
    /// disjoint and, executed in order, must equal the per-row move.
    #[test]
    fn move_chunks_never_overlap_and_match_per_row_moves() {
        assert_eq!(
            non_overlapping_chunks((1, 2, 3)).collect::<Vec<_>>(),
            vec![(1, 2, 1), (2, 3, 1), (3, 4, 1)]
        );
        assert_eq!(
            non_overlapping_chunks((1, 3, 5)).collect::<Vec<_>>(),
            vec![(1, 3, 2), (3, 5, 2), (5, 7, 1)]
        );
        assert_eq!(
            non_overlapping_chunks((1, 9, 4)).collect::<Vec<_>>(),
            vec![(1, 9, 4)]
        );
        for n in 1..12 {
            for step in 1..12 {
                let (dst, src) = (1, 1 + step);
                let mut rows: Vec<usize> = (0..dst + step + n).collect();
                for (d, s, m) in non_overlapping_chunks((dst, src, n)) {
                    assert!(s >= d + m, "chunk ({d}, {s}, {m}) overlaps itself");
                    rows.copy_within(s..s + m, d);
                }
                let want: Vec<usize> = (src..src + n).collect();
                assert_eq!(rows[dst..dst + n], want[..], "n={n} step={step}");
            }
        }
    }

    /// The in-place, scratch-free move is only sound because every move reads a
    /// row at or after the row it writes, with runs ordered by ascending
    /// destination. Assert that invariant over every monotonically increasing
    /// accept pattern up to a small bound.
    #[test]
    fn commit_runs_never_write_below_a_later_source() {
        fn check(accepted: &[usize]) {
            let runs = ddtree_commit_row_runs(accepted);
            let mut prev_dst_end = 0usize;
            for &(dst, src, n) in &runs {
                assert!(src >= dst, "src {src} < dst {dst} would need scratch");
                assert!(dst >= prev_dst_end, "runs must ascend by destination");
                prev_dst_end = dst + n;
            }
        }
        // Accepted node indices are strictly increasing (root-to-leaf walk over
        // a depth-ordered linearization), so enumerate those shapes.
        for a in 0..6 {
            for b in (a + 1)..7 {
                for c in (b + 1)..8 {
                    check(&[a, b, c]);
                }
            }
        }
    }

    /// CPU mirror of the GPU `softmax_temp_topp_batched_f32` nucleus phase:
    /// bisect tau over [0, p_max] for the inclusive crossing threshold and
    /// recompute Z = mass(tau) exactly. Used only by the parity test below.
    fn cpu_tau_cut_z(probs: &[f32], top_p: f32) -> (f32, f32) {
        if top_p >= 1.0 {
            return (0.0, 1.0);
        }
        let pmax = probs.iter().cloned().fold(0.0f32, f32::max);
        let mass = |tau: f32| -> f32 { probs.iter().filter(|&&p| p >= tau).sum() };
        let (mut lo, mut hi) = (0.0f32, pmax);
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if mass(mid) >= top_p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let tau = lo;
        (tau, mass(tau))
    }

    #[test]
    fn apply_topp_trunc_matches_ar_nucleus_cut() {
        // A row with a clear nucleus boundary (no exact ties): the GPU-style
        // threshold cut must reproduce the AR sort-and-cut nucleus exactly
        // (same kept set, same renormalized values).
        let base = [0.40f32, 0.25, 0.20, 0.10, 0.05];
        for &top_p in &[0.5f32, 0.6, 0.85, 0.95, 0.99] {
            // AR reference (sort desc, cumulative >= top_p inclusive, renorm).
            let mut ar = base.to_vec();
            apply_host_nucleus(&mut ar, top_p);

            // GPU-style: compute tau_cut/Z by bisection, then host truncate.
            let (tau, z) = cpu_tau_cut_z(&base, top_p);
            let mut gpu_style = base.to_vec();
            apply_topp_trunc(&mut gpu_style, tau, z);

            // Same kept set.
            for i in 0..base.len() {
                assert_eq!(
                    ar[i] == 0.0,
                    gpu_style[i] == 0.0,
                    "kept-set mismatch at idx {i}, top_p {top_p}: ar={:?} gpu={:?}",
                    ar,
                    gpu_style
                );
            }
            // Same renormalized values for kept tokens.
            for i in 0..base.len() {
                assert!(
                    (ar[i] - gpu_style[i]).abs() < 1e-5,
                    "value mismatch at idx {i}, top_p {top_p}: ar={} gpu={}",
                    ar[i],
                    gpu_style[i]
                );
            }
            // Renormalized row sums to 1.
            let s: f32 = gpu_style.iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "row should renorm to 1, got {s}");
        }
    }

    #[test]
    fn apply_topp_trunc_identity_when_disabled() {
        let base = [0.4f32, 0.3, 0.2, 0.1];
        let mut row = base.to_vec();
        // tau_cut == 0.0 (the kernel's top_p>=1.0 guard) → identity.
        apply_topp_trunc(&mut row, 0.0, 1.0);
        assert_eq!(row, base.to_vec());
        // apply_host_nucleus with top_p>=1.0 → identity too.
        let mut row2 = base.to_vec();
        apply_host_nucleus(&mut row2, 1.0);
        assert_eq!(row2, base.to_vec());
    }

    #[test]
    fn apply_topp_trunc_single_token_nucleus() {
        // Very tight top_p keeps only the top token; Z == p_max.
        let base = [0.7f32, 0.2, 0.07, 0.03];
        let (tau, z) = cpu_tau_cut_z(&base, 0.5);
        let mut row = base.to_vec();
        apply_topp_trunc(&mut row, tau, z);
        assert!((row[0] - 1.0).abs() < 1e-5, "single-token nucleus → 1.0");
        assert_eq!(&row[1..], &[0.0, 0.0, 0.0]);
    }

    /// Bucket frequencies of `sampled_target_bonus` over a uniform `u` grid
    /// (each bucket exact to one grid step per kept token).
    #[allow(clippy::too_many_arguments)]
    fn bonus_grid_freq(
        row: usize,
        vocab: usize,
        fast: Option<&FastTargetRows>,
        logits: &[f32],
        temp: f32,
        top_k: usize,
        top_p: f32,
        grid: usize,
    ) -> Vec<f64> {
        let mut scratch = Vec::new();
        let mut freq = vec![0.0f64; vocab];
        for g in 0..grid {
            let u = (g as f32 + 0.5) / grid as f32;
            let t = sampled_target_bonus(
                &mut scratch,
                row,
                vocab,
                fast,
                logits,
                temp,
                top_k,
                top_p,
                u,
            );
            freq[t as usize] += 1.0 / grid as f64;
        }
        freq
    }

    #[test]
    fn sampled_bonus_draws_from_the_acceptance_row_law() {
        // Two verify rows; the bonus sits at row 1. Logits are distinct, so
        // nucleus boundaries carry no ties.
        let vocab = 64usize;
        let rows = 2usize;
        let logits: Vec<f32> = (0..rows * vocab)
            .map(|j| {
                let (r, i) = (j / vocab, j % vocab);
                let rank = (i * 37 + r * 11) % vocab;
                6.0 - 1.1 * ((rank + 1) as f32).ln()
            })
            .collect();
        let grid = 200_000usize;
        for &(temp, top_p, top_k) in &[(0.7f32, 0.95f32, 20usize), (1.0, 0.8, 64), (0.6, 0.9, 5)] {
            // FAST_SAMPLE rows: the per-row softmax plus the kernel's AR
            // tau/Z (prob of the last kept token, kept mass).
            let mut probs = Vec::with_capacity(rows * vocab);
            let (mut tau, mut z) = (Vec::new(), Vec::new());
            for r in 0..rows {
                let mut row = Vec::new();
                softmax_temp_into(&logits[r * vocab..(r + 1) * vocab], temp, &mut row);
                let mut ar = row.clone();
                apply_host_ar_trunc(&mut ar, top_k, top_p);
                let t = (0..vocab)
                    .filter(|&i| ar[i] > 0.0)
                    .map(|i| row[i])
                    .fold(f32::INFINITY, f32::min);
                tau.push(t);
                z.push(row.iter().filter(|&&p| p >= t).sum());
                probs.extend_from_slice(&row);
            }
            let fast = FastTargetRows::dense(probs, tau, z);
            for (fast, label) in [(Some(&fast), "fast"), (None, "host")] {
                let mut law = Vec::new();
                sampled_target_row_into(&mut law, 1, vocab, fast, &logits, temp, top_k, top_p);
                let freq = bonus_grid_freq(1, vocab, fast, &logits, temp, top_k, top_p, grid);
                for i in 0..vocab {
                    if law[i] == 0.0 {
                        assert_eq!(
                            freq[i], 0.0,
                            "{label} T{temp}: bonus {i} outside the row law"
                        );
                    } else {
                        assert!(
                            (freq[i] - law[i] as f64).abs() < 2e-4,
                            "{label} T{temp}: token {i} freq {} vs law {}",
                            freq[i],
                            law[i]
                        );
                    }
                }
                if top_k > 0 {
                    let kept = law.iter().filter(|&&p| p > 0.0).count();
                    assert!(kept <= top_k, "{label}: {kept} kept > top_k {top_k}");
                }
            }
            // Negative control: the pre-fix bonus read the raw FAST_SAMPLE
            // row and lands outside the truncated support.
            let raw = &fast.probs[vocab..2 * vocab];
            let mut truncated = Vec::new();
            sampled_target_row_into(
                &mut truncated,
                1,
                vocab,
                Some(&fast),
                &logits,
                temp,
                top_k,
                top_p,
            );
            let outside = (0..grid)
                .map(|g| sample_categorical(raw, (g as f32 + 0.5) / grid as f32) as usize)
                .filter(|&t| truncated[t] == 0.0)
                .count();
            assert!(
                outside > 0,
                "T{temp}: legacy raw-row bonus stayed inside the nucleus"
            );
        }
    }

    #[test]
    fn dflash_gdn_tape_replay_uses_actual_verify_eligibility() {
        assert!(dflash_use_gdn_tape_replay(true, true));
        assert!(!dflash_use_gdn_tape_replay(false, true));
        assert!(!dflash_use_gdn_tape_replay(true, false));
    }

    #[test]
    fn dflash_extended_verify_graph_is_moe_greedy_only() {
        assert!(dflash_moe_verify_graph_lmhead_eligible(
            256, false, false, None
        ));
        assert!(!dflash_moe_verify_graph_lmhead_eligible(
            0, false, false, None
        ));
        assert!(!dflash_moe_verify_graph_lmhead_eligible(
            256, true, false, None
        ));
        assert!(!dflash_moe_verify_graph_lmhead_eligible(
            256, false, true, None
        ));
        assert!(!dflash_moe_verify_graph_lmhead_eligible(
            256,
            false,
            false,
            Some("0")
        ));
    }

    #[test]
    fn dflash_draft_ffn_graph_is_moe_plain_dflash_only() {
        assert!(dflash_moe_draft_ffn_graph_eligible(
            256, false, false, false, None
        ));
        assert!(!dflash_moe_draft_ffn_graph_eligible(
            0, false, false, false, None
        ));
        assert!(!dflash_moe_draft_ffn_graph_eligible(
            256, true, false, false, None
        ));
        assert!(!dflash_moe_draft_ffn_graph_eligible(
            256, false, true, false, None
        ));
        assert!(!dflash_moe_draft_ffn_graph_eligible(
            256, false, false, true, None
        ));
        assert!(!dflash_moe_draft_ffn_graph_eligible(
            256,
            false,
            false,
            false,
            Some("0")
        ));
    }

    #[test]
    fn dflash_verify_graph_env_quarantines_gfx1100_mq_v2() {
        use rdna_compute::DType;

        // gfx1100 + every V2 width: default-off; =0 force-off; =1 diagnostic opt-in.
        for dtype in [
            DType::MQ2G256V2,
            DType::MQ3G256V2,
            DType::MQ4G256V2,
            DType::MQ5G256V2,
            DType::MQ6G256V2,
        ] {
            assert!(!dflash_verify_graph_env_eligible("gfx1100", dtype, None));
            assert!(!dflash_verify_graph_env_eligible(
                "gfx1100",
                dtype,
                Some("0")
            ));
            assert!(dflash_verify_graph_env_eligible(
                "gfx1100",
                dtype,
                Some("1")
            ));
        }

        // gfx1100 + non-MQ4 legacy quant remains default-on.
        assert!(dflash_verify_graph_env_eligible(
            "gfx1100",
            DType::MQ3G256,
            None
        ));

        // Other arches keep default-on for V2; =0 still force-off.
        assert!(dflash_verify_graph_env_eligible(
            "gfx1151",
            DType::MQ3G256V2,
            None
        ));
        assert!(!dflash_verify_graph_env_eligible(
            "gfx1151",
            DType::MQ3G256V2,
            Some("0")
        ));
        assert!(dflash_verify_graph_env_eligible(
            "gfx1151",
            DType::MQ3G256V2,
            Some("1")
        ));
        assert!(dflash_verify_graph_env_eligible(
            "gfx1201",
            DType::MQ3G256V2,
            None
        ));
    }

    #[test]
    fn dflash_verify_graph_env_quarantines_measured_legacy_mq4() {
        use rdna_compute::DType;

        // Exact gfx1100 + legacy MQ4G256: default-off; =0 force-off; =1 opt-in.
        assert!(!dflash_verify_graph_env_eligible(
            "gfx1100",
            DType::MQ4G256,
            None
        ));
        assert!(!dflash_verify_graph_env_eligible(
            "gfx1100",
            DType::MQ4G256,
            Some("0")
        ));
        assert!(dflash_verify_graph_env_eligible(
            "gfx1100",
            DType::MQ4G256,
            Some("1")
        ));

        // Exact gfx1201 + legacy MQ4G256: default-off; =0 force-off; =1 opt-in.
        assert!(!dflash_verify_graph_env_eligible(
            "gfx1201",
            DType::MQ4G256,
            None
        ));
        assert!(!dflash_verify_graph_env_eligible(
            "gfx1201",
            DType::MQ4G256,
            Some("0")
        ));
        assert!(dflash_verify_graph_env_eligible(
            "gfx1201",
            DType::MQ4G256,
            Some("1")
        ));

        // Sibling arches keep default-on for legacy MQ4; =0 still force-off.
        assert!(dflash_verify_graph_env_eligible(
            "gfx1151",
            DType::MQ4G256,
            None
        ));
        assert!(dflash_verify_graph_env_eligible(
            "gfx1200",
            DType::MQ4G256,
            None
        ));
        assert!(!dflash_verify_graph_env_eligible(
            "gfx1200",
            DType::MQ4G256,
            Some("0")
        ));
        assert!(dflash_verify_graph_env_eligible(
            "gfx1201",
            DType::MQ3G256,
            None
        ));
    }

    fn try_gpu() -> Option<Gpu> {
        Gpu::init().ok()
    }

    /// Tiny hand-built DeltaNetState with one EF residual buffer.
    fn tiny_dn_with_ef(gpu: &mut Gpu) -> DeltaNetState {
        DeltaNetState {
            s_matrices: vec![gpu.zeros(&[8], DType::F32).expect("s")],
            s_scales: vec![gpu.zeros(&[1], DType::F32).expect("scale")],
            conv_states: vec![gpu.zeros(&[4], DType::F32).expect("conv")],
            s_ef_residual: vec![gpu.zeros(&[8], DType::F16).expect("ef")],
            quant: qwen35::StateQuant::Q8,
        }
    }

    fn read_f16_as_f32(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
        // Use logical element count from shape; device buf may be padded.
        let n: usize = t.shape.iter().product();
        let mut bytes = vec![0u8; t.buf.size()];
        gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).expect("dtoh f16");
        (0..n)
            .map(|i| {
                let h = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                hipfire_runtime::llama::f16_to_f32(h)
            })
            .collect()
    }

    fn write_f16_from_f32(gpu: &Gpu, t: &GpuTensor, vals: &[f32]) {
        let n: usize = t.shape.iter().product();
        assert_eq!(vals.len(), n, "logical F16 element count");
        let mut bytes = vec![0u8; t.buf.size()];
        for (i, &v) in vals.iter().enumerate() {
            let h = hipfire_runtime::llama::f32_to_f16(v);
            bytes[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
        }
        gpu.hip.memcpy_htod(&t.buf, &bytes).expect("htod f16");
    }

    /// Production behavior: dirty EF residual, save/restore via DeltaNetSnapshot,
    /// and confirm restore recovers the saved values (not the dirtied ones).
    #[test]
    fn deltanet_snapshot_save_restore_includes_ef_residual() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let mut state = tiny_dn_with_ef(&mut gpu);
        assert_eq!(state.s_ef_residual.len(), 1);

        let saved_vals = [1.0f32, -0.5, 0.25, 2.0, -1.0, 0.125, 0.75, -0.25];
        write_f16_from_f32(&gpu, &state.s_ef_residual[0], &saved_vals);

        let mut snap = DeltaNetSnapshot::new_for(&mut gpu, &state).expect("snap alloc");
        assert_eq!(
            snap.s_ef_len(),
            state.s_ef_residual.len(),
            "snapshot EF count must track live residual"
        );
        snap.save_from(&state, &mut gpu).expect("save");

        // Dirty live EF after save.
        let dirty = [9.0f32; 8];
        write_f16_from_f32(&gpu, &state.s_ef_residual[0], &dirty);
        let after_dirty = read_f16_as_f32(&gpu, &state.s_ef_residual[0]);
        assert!(
            after_dirty.iter().all(|&v| (v - 9.0).abs() < 1e-2),
            "precondition: live EF dirtied"
        );

        snap.restore_to(&mut state, &mut gpu).expect("restore");
        let restored = read_f16_as_f32(&gpu, &state.s_ef_residual[0]);
        for (i, (&got, &want)) in restored.iter().zip(saved_vals.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "EF[{i}] restore mismatch: got {got} want {want}"
            );
        }

        snap.free_gpu(&mut gpu);
        state.free_gpu(&mut gpu);
    }

    /// Production behavior: DeltaNetState::reset zeroes a dirtied EF residual
    /// (the path ModelSlot::reset_state / reset_recurrent must use).
    #[test]
    fn deltanet_reset_clears_dirtied_ef_residual() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let mut state = tiny_dn_with_ef(&mut gpu);
        write_f16_from_f32(&gpu, &state.s_ef_residual[0], &[3.5f32; 8]);
        let before = read_f16_as_f32(&gpu, &state.s_ef_residual[0]);
        assert!(
            before.iter().any(|&v| v.abs() > 1.0),
            "precondition: EF dirty"
        );

        state.reset(&mut gpu);
        let after = read_f16_as_f32(&gpu, &state.s_ef_residual[0]);
        for (i, &v) in after.iter().enumerate() {
            assert!(v.abs() < 1e-3, "EF[{i}] must be zero after reset, got {v}");
        }
        state.free_gpu(&mut gpu);
    }
}

/// Host-only tests of the save-free CB verify ownership API
/// ([`DeltaNetSnapshot::exchange_buffers`] and friends): fake device
/// addresses, never dereferenced or freed.
#[cfg(test)]
mod dn_exchange_tests {
    use super::*;

    const EXT: DnCbExtents = DnCbExtents {
        n_layers: 2,
        s: 64,
        scales: 16,
        conv: 24,
        ef: 128,
    };
    const LIVE: usize = 0x10_0000;
    const SNAP: usize = 0x20_0000;
    const FWD0: usize = 0xF0_0000;
    const REV0: usize = 0xE0_0000;

    fn buf(addr: usize, size: usize) -> DeviceBuffer {
        // SAFETY: a borrowed handle to a fake address; host-only tests never
        // dereference or free it.
        unsafe { DeviceBuffer::from_raw(addr as *mut std::ffi::c_void, size) }
    }

    fn addr(base: usize, fam: usize, layer: usize) -> usize {
        base + fam * 0x1000 + layer * 0x100
    }

    fn fam_bytes(fam: usize) -> usize {
        [EXT.s, EXT.scales, EXT.conv, EXT.ef][fam]
    }

    fn fam_dtype(fam: usize) -> DType {
        if fam == 3 {
            DType::F16
        } else {
            DType::F32
        }
    }

    fn state(base: usize) -> DeltaNetState {
        let fam = |f: usize| -> Vec<GpuTensor> {
            (0..EXT.n_layers)
                .map(|l| GpuTensor {
                    buf: buf(addr(base, f, l), fam_bytes(f)),
                    shape: vec![fam_bytes(f)],
                    dtype: fam_dtype(f),
                })
                .collect()
        };
        DeltaNetState {
            s_matrices: fam(0),
            s_scales: fam(1),
            conv_states: fam(2),
            s_ef_residual: fam(3),
            quant: qwen35::StateQuant::Q8,
        }
    }

    fn snapshot(base: usize, live: &DeltaNetState) -> DeltaNetSnapshot {
        let fam = |f: usize| -> Vec<DeviceBuffer> {
            (0..EXT.n_layers)
                .map(|l| buf(addr(base, f, l), fam_bytes(f)))
                .collect()
        };
        DeltaNetSnapshot {
            s_matrix_bufs: fam(0),
            s_scale_bufs: fam(1),
            conv_state_bufs: fam(2),
            s_ef_residual_bufs: fam(3),
            bulk_fwd: Some(buf(FWD0, 8)),
            bulk_rev: Some(buf(REV0, 8)),
            bulk_n_items: 3,
            bulk_fingerprint: DeltaNetSnapshot::bulk_fingerprint(live),
        }
    }

    fn live_ptrs(s: &DeltaNetState, layer: usize) -> [u64; 4] {
        [
            s.s_matrices[layer].buf.as_ptr() as u64,
            s.s_scales[layer].buf.as_ptr() as u64,
            s.conv_states[layer].buf.as_ptr() as u64,
            s.s_ef_residual[layer].buf.as_ptr() as u64,
        ]
    }

    fn expect(base: usize, layer: usize) -> [u64; 4] {
        [0, 1, 2, 3].map(|f| addr(base, f, layer) as u64)
    }

    #[test]
    fn exchange_swaps_every_family_and_keeps_metadata() {
        let mut st = state(LIVE);
        let mut sn = snapshot(SNAP, &st);
        sn.exchange_buffers(&mut st);
        for l in 0..EXT.n_layers {
            assert_eq!(
                live_ptrs(&st, l),
                expect(SNAP, l),
                "live now owns the old snapshot bank"
            );
            assert_eq!(
                sn.layer_state_ptrs(l),
                expect(LIVE, l),
                "snapshot now owns the old live bank"
            );
        }
        // Handle sizes, tensor shape/dtype and quant mode are untouched.
        for f in 0..4 {
            let fam = [
                &st.s_matrices,
                &st.s_scales,
                &st.conv_states,
                &st.s_ef_residual,
            ][f];
            for t in fam.iter() {
                assert_eq!(t.buf.size(), fam_bytes(f));
                assert_eq!(t.shape, vec![fam_bytes(f)]);
                assert_eq!(t.dtype, fam_dtype(f));
            }
        }
        assert_eq!(st.quant, qwen35::StateQuant::Q8);
        assert!(
            sn.exact_extents(&st, &EXT),
            "still an exact pair after the exchange"
        );
    }

    #[test]
    fn two_exchanges_restore_the_original_owners() {
        let mut st = state(LIVE);
        let mut sn = snapshot(SNAP, &st);
        let fp0 = sn.bulk_fingerprint;
        sn.exchange_buffers(&mut st);
        sn.exchange_buffers(&mut st);
        for l in 0..EXT.n_layers {
            assert_eq!(live_ptrs(&st, l), expect(LIVE, l));
            assert_eq!(sn.layer_state_ptrs(l), expect(SNAP, l));
        }
        assert_eq!(sn.bulk_fingerprint, fp0);
        assert_eq!(
            sn.bulk_fwd.as_ref().map(|b| b.as_ptr() as usize),
            Some(FWD0)
        );
        assert_eq!(
            sn.bulk_rev.as_ref().map(|b| b.as_ptr() as usize),
            Some(REV0)
        );
        assert_eq!(sn.bulk_n_items, 3);
    }

    #[test]
    fn bulk_descriptors_invert_and_refingerprint() {
        let mut st = state(LIVE);
        let mut sn = snapshot(SNAP, &st);
        sn.exchange_buffers(&mut st);
        // Old reverse (backup -> live) is the new forward SAVE; old forward is
        // the new RESTORE.
        assert_eq!(
            sn.bulk_fwd.as_ref().map(|b| b.as_ptr() as usize),
            Some(REV0)
        );
        assert_eq!(
            sn.bulk_rev.as_ref().map(|b| b.as_ptr() as usize),
            Some(FWD0)
        );
        assert_eq!(sn.bulk_n_items, 3);
        assert_eq!(
            sn.bulk_fingerprint,
            DeltaNetSnapshot::bulk_fingerprint(&st),
            "valid for the new live owners"
        );
    }

    #[test]
    fn stale_descriptors_are_disarmed_not_blessed() {
        let mut st = state(LIVE);
        let mut sn = snapshot(SNAP, &st);
        sn.bulk_fingerprint ^= 1; // built against some other live state
        sn.exchange_buffers(&mut st);
        assert_eq!(
            sn.bulk_n_items, 0,
            "stale tables must not be re-fingerprinted into validity"
        );
        // Allocations stay owned (freed by `free_gpu`), directions not swapped.
        assert_eq!(
            sn.bulk_fwd.as_ref().map(|b| b.as_ptr() as usize),
            Some(FWD0)
        );
        assert_eq!(
            sn.bulk_rev.as_ref().map(|b| b.as_ptr() as usize),
            Some(REV0)
        );
        // The exchange itself still happened.
        assert_eq!(live_ptrs(&st, 0), expect(SNAP, 0));
        // Disarmed stays disarmed across a second exchange.
        sn.exchange_buffers(&mut st);
        assert_eq!(sn.bulk_n_items, 0);
    }

    #[test]
    fn size_and_shape_refusals() {
        let st = state(LIVE);
        let sn = snapshot(SNAP, &st);
        assert!(sn.exact_extents(&st, &EXT));

        let mut bad = state(LIVE);
        bad.conv_states[1].buf = buf(addr(LIVE, 2, 1), EXT.conv + 4096);
        assert!(
            !sn.exact_extents(&bad, &EXT),
            "live conv ring rounded but the snapshot's is not (unequal banks)"
        );

        let mut bad = state(LIVE);
        bad.s_scales[0].buf = buf(addr(LIVE, 1, 0), EXT.scales - 4);
        let mut bad_snap = snapshot(SNAP, &st);
        bad_snap.s_scale_bufs[0] = buf(addr(SNAP, 1, 0), EXT.scales - 4);
        assert!(
            !bad_snap.exact_extents(&bad, &EXT),
            "equal banks below the logical scales extent"
        );

        let mut bad = state(LIVE);
        bad.s_matrices[0].buf = buf(addr(LIVE, 0, 0), EXT.s + 4096);
        let mut bad_snap = snapshot(SNAP, &st);
        bad_snap.s_matrix_bufs[0] = buf(addr(SNAP, 0, 0), EXT.s + 4096);
        assert!(!bad_snap.exact_extents(&bad, &EXT), "S must stay exact");

        // EF is pooled (`gpu.zeros`), so a reused bucket can be rounded up:
        // equal rounded banks are admitted (kernels touch only the logical
        // extent, the exchange keeps each owner's capacity).
        let mut ok = state(LIVE);
        ok.s_ef_residual[0].buf = buf(addr(LIVE, 3, 0), EXT.ef + 4096);
        let mut ok_snap = snapshot(SNAP, &st);
        ok_snap.s_ef_residual_bufs[0] = buf(addr(SNAP, 3, 0), EXT.ef + 4096);
        assert!(
            ok_snap.exact_extents(&ok, &EXT),
            "equal pool-rounded EF banks are admitted"
        );
        let mut bad = state(LIVE);
        bad.s_ef_residual[0].buf = buf(addr(LIVE, 3, 0), EXT.ef + 4096);
        assert!(!sn.exact_extents(&bad, &EXT), "unequal EF banks");

        let mut bad = state(LIVE);
        bad.s_ef_residual.pop();
        assert!(!sn.exact_extents(&bad, &EXT), "EF family incomplete");

        let mut bad = state(LIVE);
        bad.s_ef_residual.clear();
        assert!(!sn.exact_extents(&bad, &EXT), "EF off");

        let mut bad = state(LIVE);
        bad.quant = qwen35::StateQuant::FP32;
        assert!(!sn.exact_extents(&bad, &EXT), "non-Q8 state");

        let mut bad_snap = snapshot(SNAP, &st);
        bad_snap.s_matrix_bufs[0] = buf(addr(SNAP, 0, 0), EXT.s * 2);
        assert!(!bad_snap.exact_extents(&st, &EXT), "oversized snapshot S");

        let mut bad_snap = snapshot(SNAP, &st);
        bad_snap.s_scale_bufs.pop();
        assert!(
            !bad_snap.exact_extents(&st, &EXT),
            "snapshot scales family short"
        );

        let other = DnCbExtents { n_layers: 3, ..EXT };
        assert!(
            !sn.exact_extents(&st, &other),
            "layer count differs from the config's"
        );
        let none = DnCbExtents { n_layers: 0, ..EXT };
        assert!(!sn.exact_extents(&st, &none));
    }

    /// Pool-rounded scales / conv rings (both banks the same rounded size) are
    /// admitted: the kernels write only the logical extent.
    #[test]
    fn equal_pool_rounded_scales_and_conv_are_admitted() {
        let mut st = state(LIVE);
        let mut sn = snapshot(SNAP, &st);
        for l in 0..EXT.n_layers {
            st.s_scales[l].buf = buf(addr(LIVE, 1, l), EXT.scales + 8192);
            sn.s_scale_bufs[l] = buf(addr(SNAP, 1, l), EXT.scales + 8192);
            st.conv_states[l].buf = buf(addr(LIVE, 2, l), EXT.conv + 8192);
            sn.conv_state_bufs[l] = buf(addr(SNAP, 2, l), EXT.conv + 8192);
        }
        assert!(sn.exact_extents(&st, &EXT));
        sn.exchange_buffers(&mut st);
        assert!(
            sn.exact_extents(&st, &EXT),
            "equal sizes: an exchange keeps every owner's capacity"
        );
        assert_eq!(st.conv_states[0].buf.size(), EXT.conv + 8192);
        assert_eq!(sn.conv_state_bufs[0].size(), EXT.conv + 8192);
    }

    #[test]
    fn borrowed_buffers_are_never_owned_exact() {
        // Every fake handle above is a `from_raw` borrowed view: none may be
        // exchanged as an owner.
        let st = state(LIVE);
        let sn = snapshot(SNAP, &st);
        assert!(!sn.all_hip_owned(&st));
    }

    #[test]
    fn owner_pointers_cover_both_banks_distinctly() {
        let st = state(LIVE);
        let sn = snapshot(SNAP, &st);
        let mut ptrs = Vec::new();
        sn.push_owner_ptrs(&st, &mut ptrs);
        assert_eq!(ptrs.len(), 2 * 4 * EXT.n_layers);
        ptrs.sort_unstable();
        assert!(ptrs.windows(2).all(|w| w[0] != w[1]));
    }
}

#[cfg(test)]
mod device_accept_tests {
    //! Host-only checks of the singleton device sampled accept
    //! (`HIPFIRE_DFLASH_DEVICE_ACCEPT`): admission, the payload built from the
    //! real `SparseQRows` (last-write duplicate ids), and the kernel's bit-exact
    //! host twin against the legacy sparse host loop. The kernel itself is
    //! compared with the twin by `tests/sampled_dflash_device_accept_hw.rs`.
    use super::*;
    use rdna_compute::dflash_sampled_accept::{
        dflash_accept_pack_payload, dflash_sampled_accept_reference,
    };

    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 24
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
        fn unit(&mut self) -> f32 {
            (self.next_u64() & 0xFF_FFFF) as f32 / 16_777_216.0
        }
    }

    fn sparse_draft(b: usize) -> DflashSampledDraft {
        let mut d = DflashSampledDraft::default();
        d.sparse = true;
        d.block = vec![0; b];
        d.sparse_q.clear();
        for i in 0..b - 1 {
            d.sparse_q.push_row(&[i as u32 + 1], &[1.0], 64).unwrap();
            d.selected_q.push(1.0);
        }
        d
    }

    #[test]
    fn admission_follows_switch_cactus_and_q_representation() {
        let cfg = DflashSampleCfg {
            temp: 0.7,
            top_p: 0.9,
            top_k: 20,
            cactus_delta: 0.0,
        };
        let d = sparse_draft(5);
        assert!(dflash_device_accept_eligible(true, &cfg, &d, 5));
        // Kill switch.
        assert!(!dflash_device_accept_eligible(false, &cfg, &d, 5));
        // CACTUS keeps the dense host arithmetic.
        let cactus = DflashSampleCfg {
            cactus_delta: 0.05,
            ..cfg
        };
        assert!(!dflash_device_accept_eligible(true, &cactus, &d, 5));
        // Dense q (HIPFIRE_DFLASH_SPARSE_TARGET=0) keeps the host loop.
        let mut dense = sparse_draft(5);
        dense.sparse = false;
        assert!(!dflash_device_accept_eligible(true, &cfg, &dense, 5));
        // Receipt not covering the window.
        assert!(!dflash_device_accept_eligible(true, &cfg, &d, 6));
        assert!(!dflash_device_accept_eligible(true, &cfg, &d, 1));
        assert!(!dflash_device_accept_eligible(
            true,
            &cfg,
            &DflashSampledDraft::default(),
            5
        ));
    }

    /// Duplicate selector candidate ids: the payload carries the last write.
    #[test]
    fn payload_carries_the_last_write_sparse_rows() {
        let mut q = SparseQRows::with_capacity(1, 8);
        q.push_row(&[9, 4, 9, 2], &[0.1, 0.2, 0.3, 0.4], 16)
            .unwrap();
        let mut payload = Vec::new();
        dflash_accept_pack_payload(
            &mut payload,
            &[4],
            &[0.5],
            0.25,
            1,
            &[0.2],
            q.offsets(),
            &q.idx,
            &q.val,
        )
        .unwrap();
        // 4n+3 header words, then ids [2, 4, 9] and values [0.4, 0.2, 0.3].
        let n = 1;
        let ids = &payload[4 * n + 3..4 * n + 6];
        let vals: Vec<f32> = payload[4 * n + 6..]
            .iter()
            .map(|&w| f32::from_bits(w))
            .collect();
        assert_eq!(ids, &[2, 4, 9]);
        assert_eq!(vals, vec![0.4, 0.2, 0.3]);
    }

    /// The device decision (host twin) against the legacy sparse host loop
    /// with the same uniforms: acceptance reads identical operands, so
    /// `accept_len` agrees exactly; the residual / bonus differ only by f32
    /// association (≤ 1 % of windows at a CDF boundary).
    #[test]
    fn twin_matches_sparse_host_loop() {
        let vocab = 512usize;
        let n = 4usize;
        let mut rng = Lcg(0xD1CE);
        let (mut checked, mut bonus_mismatch) = (0usize, 0usize);
        for round in 0..3000 {
            // Raw softmax-like rows + nucleus tau/z.
            let keep = [1usize, 5, 20, 64][round % 4];
            let mut probs = Vec::new();
            let (mut tau, mut z) = (Vec::new(), Vec::new());
            for _ in 0..=n {
                let mut r: Vec<f32> = (0..vocab).map(|_| rng.unit().powi(6) + 1e-7).collect();
                let s: f32 = r.iter().sum();
                r.iter_mut().for_each(|v| *v /= s);
                let mut sorted = r.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let t = sorted[keep - 1];
                z.push(r.iter().filter(|&&p| p >= t).sum());
                tau.push(t);
                probs.extend(r);
            }
            let mut draft = DflashSampledDraft::default();
            draft.sparse = true;
            draft.sparse_q.clear();
            draft.block = vec![0];
            for _ in 0..n {
                let cands: Vec<u32> = (0..6).map(|_| rng.below(vocab) as u32).collect();
                let w: Vec<f32> = (0..6).map(|_| rng.unit() * 0.3 + 0.01).collect();
                draft.sparse_q.push_row(&cands, &w, vocab).unwrap();
                // the proposed token is one of the candidates; its q is the row's value.
                let pick = cands[rng.below(6)];
                let (ids, vals) = draft.sparse_q.row(draft.sparse_q.rows() - 1);
                draft
                    .selected_q
                    .push(vals[ids.iter().position(|&i| i == pick).unwrap()]);
                draft.block.push(pick);
            }
            let max_accept = [n, n, n, 2][round % 4];
            let uniforms: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
            let u_fin = rng.unit();
            let mut payload = Vec::new();
            dflash_accept_pack_payload(
                &mut payload,
                &draft.block[1..],
                &uniforms,
                u_fin,
                max_accept,
                &draft.selected_q,
                draft.sparse_q.offsets(),
                &draft.sparse_q.idx,
                &draft.sparse_q.val,
            )
            .unwrap();
            let dev = dflash_sampled_accept_reference(&probs, &tau, &z, &payload, n, vocab);

            // Legacy sparse loop with the same uniforms.
            let fast = {
                let cap = DFLASH_SPARSE_CAP;
                let mut counts = vec![0u32; n + 1];
                let mut idx = vec![0u32; (n + 1) * cap];
                let mut val = vec![0f32; (n + 1) * cap];
                for r in 0..=n {
                    let mut c = 0;
                    for j in 0..vocab {
                        let p = probs[r * vocab + j];
                        if p >= tau[r] {
                            idx[r * cap + c] = j as u32;
                            val[r * cap + c] = p;
                            c += 1;
                        }
                    }
                    counts[r] = c as u32;
                }
                let mut f = FastTargetRows {
                    tau: tau.clone(),
                    z: z.clone(),
                    capacity: cap,
                    counts,
                    idx,
                    val,
                    ..FastTargetRows::default()
                };
                let nd = f.plan_dense_rows(vocab).unwrap();
                assert_eq!(nd, 0);
                f
            };
            let mut accept = 0usize;
            let mut host_bonus = None;
            for i in 0..n.min(max_accept) {
                let TargetRowRef::Sparse { idx, val, inv_z } = fast.row(i, vocab) else {
                    unreachable!()
                };
                let p_t = sparse_row_prob(idx, val, inv_z, draft.block[i + 1]);
                if uniforms[i] * draft.selected_q[i].max(f32::MIN_POSITIVE) <= p_t {
                    accept += 1;
                } else {
                    let (qi, qv) = draft.sparse_q.row(i);
                    host_bonus = Some(sample_residual_sparse(
                        idx, val, inv_z, qi, qv, vocab, u_fin,
                    ));
                    break;
                }
            }
            let host_bonus = host_bonus.unwrap_or_else(|| {
                let TargetRowRef::Sparse { idx, val, inv_z } = fast.row(accept, vocab) else {
                    unreachable!()
                };
                sample_categorical_sparse(idx, val, inv_z, vocab, u_fin)
            });
            assert_eq!(dev.accept_len, accept, "round {round}");
            checked += 1;
            if dev.bonus != host_bonus {
                bonus_mismatch += 1;
            }
        }
        assert!(
            bonus_mismatch * 100 <= checked,
            "{bonus_mismatch}/{checked} bonus draws differ"
        );
    }
}

#[cfg(test)]
mod lookup_tests {
    //! Verbatim-lookup splice seam, host only.
    //!
    //! The law tests drive the REAL host accept loop
    //! (`dflash_sampled_accept_q_rows_with`, the single implementation behind
    //! `dflash_sampled_accept_rows`) with an EXACT uniform source: every
    //! uniform the window consumes is enumerated over a midpoint grid of
    //! `grid` cells per draw (depth-first, only as many draws as the window
    //! takes). Every threshold of these toy windows (accept ratio `p/q`,
    //! residual / bonus CDF fractions) is a multiple of `1/grid`, so each cell
    //! lies wholly on one side of every threshold and the integral is exact
    //! up to f64 rounding (asserted to 1e-6). Candidates of model rows are
    //! enumerated with weight `q(c)`; point-mass rows are fixed. The q
    //! receipts are built by the production `dflash_lookup_splice_receipt`.
    //!
    //! Law checked at every window position `i` that the chain can reach:
    //! `P(emitted_i = x | positions < i all accepted) == p_i(x)` — covering
    //! the accept-or-residual token for candidate rows (q = 0 / p = 0 at the
    //! candidate included) and the bonus after a full or budget-bound accept —
    //! plus `P(reach i) == prod_{j<i} sum_c min(q_j, p_j)`.
    //!
    //! Pre-existing: `xorshift_next_unit` may return exactly `0.0` (2^-24), at
    //! which a point-mass candidate with `p = 0` is accepted (`0 <= 0`). That
    //! measure-2^-24 event predates the seam (PLD uses the same arithmetic)
    //! and is not representable on the midpoint grid.
    use super::*;

    fn integrate<T>(
        grid: usize,
        run: &mut impl FnMut(&mut dyn FnMut() -> f32) -> T,
        visit: &mut impl FnMut(T, f64),
    ) {
        fn go<T>(
            grid: usize,
            script: &mut Vec<usize>,
            run: &mut impl FnMut(&mut dyn FnMut() -> f32) -> T,
            visit: &mut impl FnMut(T, f64),
        ) {
            let mut pos = 0usize;
            let mut need_more = false;
            let out = {
                let s: &[usize] = &script[..];
                let mut src = || -> f32 {
                    if pos < s.len() {
                        let u = (s[pos] as f32 + 0.5) / (grid as f32);
                        pos += 1;
                        u
                    } else {
                        need_more = true;
                        0.5
                    }
                };
                run(&mut src)
            };
            if need_more {
                for cell in 0..grid {
                    script.push(cell);
                    go(grid, script, run, visit);
                    script.pop();
                }
            } else {
                debug_assert_eq!(pos, script.len());
                visit(out, (grid as f64).powi(-(script.len() as i32)));
            }
        }
        go(grid, &mut Vec::new(), run, visit)
    }

    const V: usize = 6;
    const T0: [f32; V] = [0.5, 0.25, 0.125, 0.125, 0.0, 0.0];
    const T1: [f32; V] = [0.25, 0.25, 0.25, 0.125, 0.125, 0.0];
    const T2: [f32; V] = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
    const T3: [f32; V] = [0.125, 0.125, 0.125, 0.125, 0.25, 0.25];
    const T4: [f32; V] = [0.25, 0.25, 0.5, 0.0, 0.0, 0.0];
    const T5: [f32; V] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5];
    const T6: [f32; V] = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0];
    const TONE: [f32; V] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    #[derive(Clone)]
    enum RowQ {
        Model(Vec<f32>),
        Point(u32),
    }

    /// Raw target row plus the nucleus/top-k cut that `tau`/`z` describe
    /// (`tau <= 0`: no cut). Logits mode derives the same cut from top_k/top_p.
    #[derive(Clone)]
    struct Target {
        raw: Vec<f32>,
        tau: f32,
        z: f32,
    }

    impl Target {
        fn plain(raw: &[f32]) -> Self {
            Self {
                raw: raw.to_vec(),
                tau: 0.0,
                z: 1.0,
            }
        }
        fn cut(raw: &[f32], tau: f32, z: f32) -> Self {
            Self {
                raw: raw.to_vec(),
                tau,
                z,
            }
        }
        /// The truncated distribution, computed independently of the code under test.
        fn expected(&self) -> Vec<f64> {
            self.raw
                .iter()
                .map(|&p| {
                    if self.tau <= 0.0 {
                        p as f64
                    } else if p >= self.tau {
                        p as f64 / self.z as f64
                    } else {
                        0.0
                    }
                })
                .collect()
        }
    }

    #[derive(Clone, Copy)]
    enum Mode {
        FastDense,
        FastSparse,
        Logits { top_k: usize, top_p: f32 },
    }

    struct Win {
        rows: Vec<RowQ>,
        targets: Vec<Target>,
        max_accept: usize,
        grid: usize,
    }

    fn build_target(win: &Win, mode: Mode) -> (DflashSampleCfg, Option<FastTargetRows>, Vec<f32>) {
        let rows = win.targets.len();
        let base = DflashSampleCfg {
            temp: 1.0,
            top_p: 1.0,
            top_k: 0,
            cactus_delta: 0.0,
        };
        match mode {
            Mode::FastDense => {
                let probs: Vec<f32> = win
                    .targets
                    .iter()
                    .flat_map(|t| t.raw.iter().copied())
                    .collect();
                let tau = win.targets.iter().map(|t| t.tau).collect();
                let z = win.targets.iter().map(|t| t.z).collect();
                (base, Some(FastTargetRows::dense(probs, tau, z)), Vec::new())
            }
            Mode::FastSparse => {
                let cap = V;
                let mut tau = Vec::new();
                let mut z = Vec::new();
                let mut counts = Vec::new();
                let mut idx = vec![0u32; rows * cap];
                let mut val = vec![0f32; rows * cap];
                for (r, t) in win.targets.iter().enumerate() {
                    // An uncut row keeps its non-zero entries (z = 1): same law as the dense row.
                    let tau_r = if t.tau <= 0.0 { 1e-9 } else { t.tau };
                    let mut c = 0;
                    for (j, &p) in t.raw.iter().enumerate() {
                        if p >= tau_r {
                            idx[r * cap + c] = j as u32;
                            val[r * cap + c] = p;
                            c += 1;
                        }
                    }
                    counts.push(c as u32);
                    tau.push(tau_r);
                    z.push(t.z);
                }
                let mut f = FastTargetRows {
                    tau,
                    z,
                    capacity: cap,
                    counts,
                    idx,
                    val,
                    ..FastTargetRows::default()
                };
                assert_eq!(f.plan_dense_rows(V).unwrap(), 0);
                (base, Some(f), Vec::new())
            }
            Mode::Logits { top_k, top_p } => {
                let logits: Vec<f32> = win
                    .targets
                    .iter()
                    .flat_map(|t| {
                        t.raw.iter().map(|&p| {
                            if p > 0.0 {
                                (p as f64).ln() as f32
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect();
                (
                    DflashSampleCfg {
                        top_p,
                        top_k,
                        ..base
                    },
                    None,
                    logits,
                )
            }
        }
    }

    /// The learned receipt of the window (`rows.len()` rows: real q for the
    /// leading model rows, garbage learned q for the rows a lookup replaces),
    /// spliced by the production receipt builder.
    fn build_receipt(win: &Win, cands: &[u32], sparse_q: bool) -> DflashSampledDraft {
        let l = win.rows.len();
        let keep = win
            .rows
            .iter()
            .take_while(|r| matches!(r, RowQ::Model(_)))
            .count();
        let mut r = DflashSampledDraft::default();
        r.sparse = sparse_q;
        for i in 0..l {
            let (qrow, tok) = match &win.rows[i] {
                RowQ::Model(q) => (q.clone(), cands[i]),
                RowQ::Point(_) => (vec![1.0 / V as f32; V], 0),
            };
            r.selected_q.push(qrow[tok as usize]);
            if sparse_q {
                let ids: Vec<u32> = (0..V as u32).filter(|&j| qrow[j as usize] > 0.0).collect();
                let vals: Vec<f32> = ids.iter().map(|&j| qrow[j as usize]).collect();
                r.sparse_q.push_row(&ids, &vals, V).unwrap();
            } else {
                r.dense_q.push(qrow);
            }
        }
        let toks: Vec<u32> = win.rows[keep..]
            .iter()
            .map(|row| match row {
                RowQ::Point(c) => *c,
                RowQ::Model(_) => panic!("model rows must precede point rows"),
            })
            .collect();
        dflash_lookup_check_receipt(Some(&r), keep).unwrap();
        dflash_lookup_splice_receipt(&mut r, keep, &toks, V).unwrap();
        r
    }

    fn check_law(name: &str, win: &Win, mode: Mode, sparse_q: bool) {
        const TOL: f64 = 1e-6;
        let l = win.rows.len();
        assert_eq!(win.targets.len(), l + 1);
        let expect: Vec<Vec<f64>> = win.targets.iter().map(Target::expected).collect();
        for e in &expect {
            let s: f64 = e.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-6,
                "{name}: target row does not sum to 1: {e:?}"
            );
        }
        let (cfg, fast, logits) = build_target(win, mode);
        let choices: Vec<Vec<(u32, f64)>> = win
            .rows
            .iter()
            .map(|r| match r {
                RowQ::Model(q) => (0..V as u32)
                    .filter(|&c| q[c as usize] > 0.0)
                    .map(|c| (c, q[c as usize] as f64))
                    .collect(),
                RowQ::Point(c) => vec![(*c, 1.0)],
            })
            .collect();

        let mut reach = vec![0f64; l + 1];
        let mut emit = vec![vec![0f64; V]; l + 1];
        let mut scratch: Vec<f32> = Vec::new();
        let mut odo = vec![0usize; l];
        'combos: loop {
            let cands: Vec<u32> = (0..l).map(|i| choices[i][odo[i]].0).collect();
            let wc: f64 = (0..l).map(|i| choices[i][odo[i]].1).product();
            let receipt = build_receipt(win, &cands, sparse_q);
            let block: Vec<u32> = std::iter::once(0u32).chain(cands.iter().copied()).collect();
            let q_rows = if receipt.sparse {
                QRows::Sparse(&receipt.sparse_q)
            } else {
                QRows::Dense(&receipt.dense_q)
            };
            let mut run = |src: &mut dyn FnMut() -> f32| {
                let mut next = || src();
                dflash_sampled_accept_q_rows_with(
                    &block,
                    &receipt.selected_q,
                    q_rows,
                    fast.as_ref(),
                    &logits,
                    V,
                    cfg,
                    win.max_accept,
                    &mut next,
                    &mut scratch,
                )
            };
            let mut visit = |d: SampledWindowDecision, wu: f64| {
                let w = wc * wu;
                let a = d.accepted_drafts;
                for i in 0..a {
                    reach[i] += w;
                    emit[i][block[i + 1] as usize] += w;
                }
                reach[a] += w;
                emit[a][d.bonus_token as usize] += w;
            };
            integrate(win.grid, &mut run, &mut visit);
            let mut i = l;
            loop {
                if i == 0 {
                    break 'combos;
                }
                i -= 1;
                odo[i] += 1;
                if odo[i] < choices[i].len() {
                    break;
                }
                odo[i] = 0;
            }
        }

        let mut analytic = 1.0f64;
        for i in 0..=l {
            let reachable = i <= win.max_accept;
            let want_reach = if reachable { analytic } else { 0.0 };
            assert!(
                (reach[i] - want_reach).abs() < TOL,
                "{name}: reach[{i}] = {} want {want_reach}",
                reach[i]
            );
            if reachable && want_reach > 1e-9 {
                for x in 0..V {
                    let got = emit[i][x] / reach[i];
                    assert!(
                        (got - expect[i][x]).abs() < TOL,
                        "{name}: position {i} token {x}: P = {got}, target p = {}",
                        expect[i][x]
                    );
                }
            }
            if i < l {
                let alpha: f64 = match &win.rows[i] {
                    RowQ::Model(q) => (0..V).map(|c| (q[c] as f64).min(expect[i][c])).sum(),
                    RowQ::Point(c) => expect[i][*c as usize].min(1.0),
                };
                analytic *= alpha;
            }
        }
    }

    fn model_uniform4() -> RowQ {
        RowQ::Model(vec![0.25, 0.25, 0.25, 0.25, 0.0, 0.0])
    }

    /// Model rows (real q, incl. q = 0 entries) then a point-mass row, then the
    /// bonus row: all four target-row shapes and both q representations.
    #[test]
    fn lookup_law_mixed_model_then_point_rows() {
        let win = Win {
            rows: vec![
                model_uniform4(),
                RowQ::Model(vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
                RowQ::Point(2),
            ],
            targets: vec![
                Target::plain(&T0),
                Target::plain(&T1),
                Target::plain(&T2),
                Target::plain(&T3),
            ],
            max_accept: usize::MAX,
            grid: 16,
        };
        check_law("mixed/fast-dense/sparse-q", &win, Mode::FastDense, true);
        check_law("mixed/fast-dense/dense-q", &win, Mode::FastDense, false);
        check_law("mixed/fast-sparse/sparse-q", &win, Mode::FastSparse, true);
        check_law("mixed/fast-sparse/dense-q", &win, Mode::FastSparse, false);
        check_law(
            "mixed/logits/sparse-q",
            &win,
            Mode::Logits {
                top_k: 0,
                top_p: 1.0,
            },
            true,
        );
    }

    /// Pure point-mass windows: p(c) in {1/2, 1/2, 1/2}, certain accept
    /// (p(c) = 1), zero target mass at the candidate (always rejected, residual
    /// is p itself), and the bonus after a full accept.
    #[test]
    fn lookup_law_point_rows_zero_and_full_target_mass() {
        let all_half = Win {
            rows: vec![RowQ::Point(0), RowQ::Point(1), RowQ::Point(5)],
            targets: vec![
                Target::plain(&T0),
                Target::plain(&T6),
                Target::plain(&T5),
                Target::plain(&T3),
            ],
            max_accept: usize::MAX,
            grid: 16,
        };
        let zero_mass = Win {
            rows: vec![RowQ::Point(0), RowQ::Point(4)],
            targets: vec![Target::plain(&T0), Target::plain(&T4), Target::plain(&T3)],
            max_accept: usize::MAX,
            grid: 16,
        };
        let certain = Win {
            rows: vec![RowQ::Point(1), RowQ::Point(0)],
            targets: vec![Target::plain(&TONE), Target::plain(&T0), Target::plain(&T3)],
            max_accept: usize::MAX,
            grid: 16,
        };
        for (name, w) in [
            ("all-half", &all_half),
            ("zero-mass", &zero_mass),
            ("certain", &certain),
        ] {
            check_law(
                &format!("{name}/fast-dense/sparse-q"),
                w,
                Mode::FastDense,
                true,
            );
            check_law(
                &format!("{name}/fast-sparse/sparse-q"),
                w,
                Mode::FastSparse,
                true,
            );
            check_law(
                &format!("{name}/fast-dense/dense-q"),
                w,
                Mode::FastDense,
                false,
            );
            check_law(
                &format!("{name}/logits/sparse-q"),
                w,
                Mode::Logits {
                    top_k: 0,
                    top_p: 1.0,
                },
                true,
            );
        }
    }

    /// Tail window (one learned row then two lookup rows) under every
    /// `max_accept` budget: the boundary bonus is drawn from the boundary row.
    #[test]
    fn lookup_law_tail_window_budget_clamp() {
        for max_accept in [0usize, 1, 2, 3, usize::MAX] {
            let win = Win {
                rows: vec![model_uniform4(), RowQ::Point(2), RowQ::Point(3)],
                targets: vec![
                    Target::plain(&T0),
                    Target::plain(&T2),
                    Target::plain(&T2),
                    Target::plain(&T3),
                ],
                max_accept,
                grid: 16,
            };
            check_law(
                &format!("tail/ma{max_accept}/fast-sparse"),
                &win,
                Mode::FastSparse,
                true,
            );
            check_law(
                &format!("tail/ma{max_accept}/fast-dense/dense-q"),
                &win,
                Mode::FastDense,
                false,
            );
        }
    }

    /// top_k / top_p truncation of p (the target of every row, including the
    /// point-mass rows' residual and the bonus) in all three target
    /// representations; point mass on a zero-mass (truncated-away) token.
    #[test]
    fn lookup_law_truncated_targets() {
        let q = RowQ::Model(vec![0.125, 0.125, 0.125, 0.125, 0.25, 0.25]);
        let mk = |raw: &[f32], tau: f32, z: f32| Win {
            rows: vec![q.clone(), RowQ::Point(1), RowQ::Point(3)],
            targets: (0..4).map(|_| Target::cut(raw, tau, z)).collect(),
            max_accept: usize::MAX,
            grid: 16,
        };
        // top_k = 2 of [.25, .25, .125 x4] -> [.5, .5, 0 ...].
        let topk = mk(&[0.25, 0.25, 0.125, 0.125, 0.125, 0.125], 0.2, 0.5);
        check_law("topk/fast-dense", &topk, Mode::FastDense, true);
        check_law("topk/fast-sparse", &topk, Mode::FastSparse, true);
        check_law("topk/fast-sparse/dense-q", &topk, Mode::FastSparse, false);
        check_law(
            "topk/logits",
            &topk,
            Mode::Logits {
                top_k: 2,
                top_p: 1.0,
            },
            false,
        );
        // top_p = 0.7 of [.375, .375, .125, .125, 0, 0] -> [.5, .5, 0 ...].
        let topp = mk(&[0.375, 0.375, 0.125, 0.125, 0.0, 0.0], 0.3, 0.75);
        check_law("topp/fast-dense", &topp, Mode::FastDense, true);
        check_law("topp/fast-sparse", &topp, Mode::FastSparse, true);
        check_law(
            "topp/logits",
            &topp,
            Mode::Logits {
                top_k: 0,
                top_p: 0.7,
            },
            true,
        );
    }

    /// One lookup (or learned) row, thresholds at arbitrary multiples of 1/8
    /// (thirds, fifths, sixths, sevenths of the renormalized mass): grid 840.
    #[test]
    fn lookup_law_single_row_fine_grid() {
        let point = |c: u32, t: Target| Win {
            rows: vec![RowQ::Point(c)],
            targets: vec![t.clone(), t],
            max_accept: usize::MAX,
            grid: 840,
        };
        // p(c) = 1/4: residual over thirds-of-six masses.
        let a = point(0, Target::plain(&[0.25, 0.375, 0.125, 0.125, 0.125, 0.0]));
        check_law("fine/point/plain", &a, Mode::FastDense, true);
        // top_p = 0.7 of [.5, .25, .125, .125, 0, 0] -> [2/3, 1/3, 0 ...].
        let b = point(
            1,
            Target::cut(&[0.5, 0.25, 0.125, 0.125, 0.0, 0.0], 0.2, 0.75),
        );
        check_law(
            "fine/point/topp-logits",
            &b,
            Mode::Logits {
                top_k: 0,
                top_p: 0.7,
            },
            true,
        );
        check_law("fine/point/topp-sparse", &b, Mode::FastSparse, true);
        // kept mass 5/8 -> fifths.
        let c = point(
            0,
            Target::cut(&[0.375, 0.25, 0.125, 0.125, 0.125, 0.0], 0.2, 0.625),
        );
        check_law("fine/point/fifths", &c, Mode::FastDense, false);
        // Learned row, q != p everywhere (ratios 1/3 and sub-1 residual).
        let d = Win {
            rows: vec![RowQ::Model(vec![0.375, 0.375, 0.25, 0.0, 0.0, 0.0])],
            targets: vec![
                Target::plain(&[0.125, 0.5, 0.25, 0.125, 0.0, 0.0]),
                Target::plain(&T3),
            ],
            max_accept: usize::MAX,
            grid: 840,
        };
        check_law("fine/model", &d, Mode::FastDense, true);
    }

    /// The accept loop draws exactly: one uniform per accept test, one more
    /// per rejection (residual) or one bonus draw — point-mass rows follow the
    /// same rule as learned rows.
    #[test]
    fn lookup_rng_draw_counts_follow_accept_rule() {
        let run = |rows: &[RowQ], targets: &[&[f32; V]], max_accept: usize| -> (usize, usize) {
            let win = Win {
                rows: rows.to_vec(),
                targets: targets.iter().map(|t| Target::plain(*t)).collect(),
                max_accept,
                grid: 1,
            };
            let (cfg, fast, logits) = build_target(&win, Mode::FastDense);
            let cands: Vec<u32> = rows
                .iter()
                .map(|r| match r {
                    RowQ::Point(c) => *c,
                    RowQ::Model(_) => unreachable!(),
                })
                .collect();
            let receipt = build_receipt(&win, &cands, true);
            let block: Vec<u32> = std::iter::once(0u32).chain(cands.iter().copied()).collect();
            let mut draws = 0usize;
            let mut next = || {
                draws += 1;
                0.5f32
            };
            let mut scratch = Vec::new();
            let d = dflash_sampled_accept_q_rows_with(
                &block,
                &receipt.selected_q,
                QRows::Sparse(&receipt.sparse_q),
                fast.as_ref(),
                &logits,
                V,
                cfg,
                max_accept,
                &mut next,
                &mut scratch,
            );
            (d.accepted_drafts, draws)
        };
        let two_ok = [RowQ::Point(0), RowQ::Point(1)];
        // u = 0.5 accepts p = 0.5 (0.5 <= 0.5): 2 accept tests + 1 bonus.
        assert_eq!(run(&two_ok, &[&T0, &T6, &T3], usize::MAX), (2, 3));
        // Budget 1: 1 accept test, loop stops, 1 bonus draw from the boundary row.
        assert_eq!(run(&two_ok, &[&T0, &T6, &T3], 1), (1, 2));
        // Budget 0: no accept test, only the bonus draw.
        assert_eq!(run(&two_ok, &[&T0, &T6, &T3], 0), (0, 1));
        // Row 1 rejects (p = 0): accept test + residual draw, no bonus.
        let rej = [RowQ::Point(0), RowQ::Point(4)];
        assert_eq!(run(&rej, &[&T0, &T4, &T3], usize::MAX), (1, 3));
    }

    /// The public RNG-driven wrapper is the generic body fed the xorshift
    /// stream: same decision, same RNG state afterwards.
    #[test]
    fn lookup_accept_wrapper_matches_uniform_source() {
        let win = Win {
            rows: vec![RowQ::Point(0), RowQ::Point(2), RowQ::Point(3)],
            targets: vec![
                Target::plain(&T0),
                Target::plain(&T1),
                Target::plain(&T2),
                Target::plain(&T3),
            ],
            max_accept: usize::MAX,
            grid: 1,
        };
        let (cfg, fast, logits) = build_target(&win, Mode::FastDense);
        let receipt = build_receipt(&win, &[0, 2, 3], true);
        let block = [0u32, 0, 2, 3];
        for seed in 1u64..200 {
            for max_accept in [0usize, 1, 2, usize::MAX] {
                let mut rng_a = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                let mut rng_b = rng_a;
                let mut scratch_a = Vec::new();
                let mut scratch_b = Vec::new();
                let a = dflash_sampled_accept_rows(
                    &block,
                    &receipt,
                    fast.as_ref(),
                    &logits,
                    V,
                    cfg,
                    max_accept,
                    &mut rng_a,
                    &mut scratch_a,
                );
                let b = dflash_sampled_accept_q_rows_with(
                    &block,
                    &receipt.selected_q,
                    QRows::Sparse(&receipt.sparse_q),
                    fast.as_ref(),
                    &logits,
                    V,
                    cfg,
                    max_accept,
                    &mut || xorshift_next_unit(&mut rng_b),
                    &mut scratch_b,
                );
                assert_eq!(a.accepted_drafts, b.accepted_drafts);
                assert_eq!(a.bonus_token, b.bonus_token);
                assert_eq!(rng_a, rng_b);
            }
        }
    }

    struct FakeSource {
        ret: Option<usize>,
        tail: Vec<u32>,
        seen: Vec<u32>,
        buffer_was_clear: bool,
    }

    impl FakeSource {
        fn new(ret: Option<usize>, tail: &[u32]) -> Self {
            Self {
                ret,
                tail: tail.to_vec(),
                seen: Vec::new(),
                buffer_was_clear: false,
            }
        }
    }

    impl DflashTailSource for FakeSource {
        fn splice(&mut self, drafted: &[u32], tail: &mut Vec<u32>) -> Option<usize> {
            self.seen = drafted.to_vec();
            self.buffer_was_clear = tail.is_empty();
            tail.extend_from_slice(&self.tail);
            self.ret
        }
    }

    const B: usize = 8;
    const VOCAB: usize = 100;

    fn learned_block() -> Vec<u32> {
        vec![7, 11, 12, 13, 14, 15, 16, 17]
    }

    /// Every refusal happens in the pure helpers: the learned block and the
    /// (caller-visible) tail scratch of a rejected Copy are never touched.
    #[test]
    fn lookup_copy_validation_errors_precede_mutation() {
        let drafted = learned_block();
        let mut tail = vec![9u32, 9, 9];
        let too_long: Vec<u32> = (20..28).collect(); // 8 tokens -> 9 rows > b = 8
        let oob = [20u32, VOCAB as u32];
        for (name, tokens) in [
            ("empty", &[] as &[u32]),
            ("too long", &too_long[..]),
            ("id out of vocab", &oob[..]),
        ] {
            let r =
                dflash_lookup_resolve(DflashLookup::Copy(tokens), &drafted, B, VOCAB, &mut tail);
            assert!(r.is_err(), "{name} must be refused");
            assert_eq!(tail, vec![9, 9, 9], "{name}: tail scratch mutated");
            assert_eq!(drafted, learned_block(), "{name}: learned block mutated");
            assert!(
                dflash_lookup_preflight(&Some(DflashLookup::Copy(tokens)), true, false, B, VOCAB)
                    .is_err(),
                "{name}: preflight must refuse before the draft runs"
            );
        }
        // b - 1 tokens fit exactly (b_eff == b) and replace every learned row.
        let full: Vec<u32> = (20..27).collect();
        let r = dflash_lookup_resolve(DflashLookup::Copy(&full), &drafted, B, VOCAB, &mut tail);
        assert_eq!(r.unwrap(), Some(0));
        assert_eq!(tail, full);
        assert_eq!(drafted, learned_block());
    }

    #[test]
    fn lookup_tail_validation_errors_precede_mutation() {
        let drafted = learned_block();
        let cases: Vec<(&str, Option<usize>, Vec<u32>, bool)> = vec![
            ("k = 0", Some(0), vec![50], false),
            ("k > b - 1", Some(8), vec![50], false),
            ("empty tail", Some(2), vec![], false),
            (
                "k + tail > b - 1",
                Some(2),
                vec![50, 51, 52, 53, 54, 55],
                false,
            ),
            ("k = b - 1 leaves no room", Some(7), vec![50], false),
            (
                "tail id out of vocab",
                Some(2),
                vec![50, VOCAB as u32],
                false,
            ),
            ("exact fit", Some(2), vec![50, 51, 52, 53, 54], true),
            ("k = 1", Some(1), vec![50], true),
        ];
        for (name, ret, tail_tokens, ok) in cases {
            let mut src = FakeSource::new(ret, &tail_tokens);
            let mut tail = vec![9u32, 9, 9];
            let r =
                dflash_lookup_resolve(DflashLookup::Tail(&mut src), &drafted, B, VOCAB, &mut tail);
            assert_eq!(r.is_ok(), ok, "{name}");
            assert_eq!(drafted, learned_block(), "{name}: learned block mutated");
            assert_eq!(src.seen, drafted[1..], "{name}: source must see d1..d(b-1)");
            assert!(
                src.buffer_was_clear,
                "{name}: tail buffer not cleared first"
            );
            if ok {
                assert_eq!(r.unwrap(), ret);
                assert_eq!(tail, tail_tokens);
                assert!(1 + ret.unwrap() + tail.len() <= B);
            }
        }
        // A declining source: incumbent window, tail left empty.
        let mut src = FakeSource::new(None, &[50, 51]);
        let mut tail = vec![9u32];
        let r = dflash_lookup_resolve(DflashLookup::Tail(&mut src), &drafted, B, VOCAB, &mut tail);
        assert_eq!(r.unwrap(), None);
        assert!(tail.is_empty());
        // The learned block must be the window's full block.
        let mut src = FakeSource::new(Some(1), &[50]);
        assert!(dflash_lookup_resolve(
            DflashLookup::Tail(&mut src),
            &drafted[..5],
            B,
            VOCAB,
            &mut tail
        )
        .is_err());
    }

    #[test]
    fn lookup_preflight_refuses_unsupported_modes_before_the_draft() {
        let copy = [1u32, 2, 3];
        assert!(dflash_lookup_preflight(&None, false, true, B, VOCAB).is_ok());
        assert!(
            dflash_lookup_preflight(&Some(DflashLookup::Copy(&copy)), true, false, B, VOCAB)
                .is_ok()
        );
        // Not selector mode: no learned q receipts to compose with.
        assert!(
            dflash_lookup_preflight(&Some(DflashLookup::Copy(&copy)), false, false, B, VOCAB)
                .is_err()
        );
        // PLD spine owns the block.
        assert!(
            dflash_lookup_preflight(&Some(DflashLookup::Copy(&copy)), true, true, B, VOCAB)
                .is_err()
        );
        let mut src = FakeSource::new(Some(1), &[1]);
        assert!(dflash_lookup_preflight(
            &Some(DflashLookup::Tail(&mut src)),
            false,
            false,
            B,
            VOCAB
        )
        .is_err());
        assert!(src.seen.is_empty(), "preflight must not call the source");
    }

    fn learned_receipt(sparse: bool, rows: usize) -> DflashSampledDraft {
        let mut r = DflashSampledDraft::default();
        r.sparse = sparse;
        for i in 0..rows {
            let ids = [i as u32 + 1, i as u32 + 40];
            let vals = [0.5f32 - 0.01 * i as f32, 0.25];
            r.selected_q.push(vals[0]);
            r.uniforms.push(0.1 * i as f32);
            if sparse {
                r.sparse_q.push_row(&ids, &vals, VOCAB).unwrap();
            } else {
                let mut d = vec![0f32; VOCAB];
                d[ids[0] as usize] = vals[0];
                d[ids[1] as usize] = vals[1];
                r.dense_q.push(d);
            }
        }
        r
    }

    /// Learned rows `..keep` keep their real receipts; lookup rows are point
    /// masses; the learned selector proposal never reaches a lookup row.
    #[test]
    fn lookup_receipt_keeps_learned_rows_and_point_masses_the_tail() {
        for sparse in [true, false] {
            for (keep, tokens) in [
                (0usize, vec![10u32, 11, 12]),
                (2, vec![10, 11, 12]),
                (3, vec![5]),
            ] {
                let orig = learned_receipt(sparse, 7);
                let mut r = learned_receipt(sparse, 7);
                dflash_lookup_check_receipt(Some(&r), keep).unwrap();
                dflash_lookup_splice_receipt(&mut r, keep, &tokens, VOCAB).unwrap();
                let rows = keep + tokens.len();
                assert_eq!(r.selected_q.len(), rows);
                assert!(r.proposal.is_none());
                assert_eq!(r.uniforms.len(), keep);
                assert_eq!(&r.selected_q[..keep], &orig.selected_q[..keep]);
                assert!(r.selected_q[keep..].iter().all(|&q| q == 1.0));
                if sparse {
                    assert_eq!(r.sparse_q.rows(), rows);
                    for i in 0..keep {
                        assert_eq!(r.sparse_q.row(i), orig.sparse_q.row(i), "learned row {i}");
                    }
                    for (j, &c) in tokens.iter().enumerate() {
                        assert_eq!(r.sparse_q.row(keep + j), (&[c][..], &[1.0f32][..]));
                    }
                } else {
                    assert_eq!(r.dense_q.len(), rows);
                    for i in 0..keep {
                        assert_eq!(r.dense_q[i], orig.dense_q[i], "learned row {i}");
                    }
                    for (j, &c) in tokens.iter().enumerate() {
                        let row = &r.dense_q[keep + j];
                        assert_eq!(row.len(), VOCAB);
                        assert_eq!(row[c as usize], 1.0);
                        assert_eq!(row.iter().filter(|&&x| x != 0.0).count(), 1);
                    }
                }
            }
        }
    }

    #[test]
    fn lookup_receipt_check_refuses_short_or_missing_receipts() {
        assert!(dflash_lookup_check_receipt(None, 1).is_err());
        let r = learned_receipt(true, 3);
        assert!(dflash_lookup_check_receipt(Some(&r), 3).is_ok());
        assert!(dflash_lookup_check_receipt(Some(&r), 4).is_err());
        let d = learned_receipt(false, 2);
        assert!(dflash_lookup_check_receipt(Some(&d), 2).is_ok());
        assert!(dflash_lookup_check_receipt(Some(&d), 3).is_err());
    }

    #[test]
    fn lookup_sparse_q_truncate_rows() {
        let mut q = SparseQRows::with_capacity(4, 4);
        q.push_row(&[3, 1], &[0.1, 0.2], 16).unwrap();
        q.push_row(&[5], &[1.0], 16).unwrap();
        q.push_row(&[2, 9], &[0.3, 0.4], 16).unwrap();
        q.truncate_rows(5); // no-op beyond the end
        assert_eq!(q.rows(), 3);
        q.truncate_rows(1);
        assert_eq!(q.rows(), 1);
        assert_eq!(q.row(0), (&[1u32, 3][..], &[0.2f32, 0.1][..]));
        q.push_row(&[7], &[1.0], 16).unwrap();
        assert_eq!(q.row(1), (&[7u32][..], &[1.0f32][..]));
        q.truncate_rows(0);
        assert_eq!(q.rows(), 0);
        assert_eq!(q.offsets(), &[0]);
    }
}
