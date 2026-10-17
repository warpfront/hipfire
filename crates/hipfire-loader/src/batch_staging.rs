// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Continuous-batch staging, moved out of the daemon load handler.
//!
//! This block constructs `Qwen35DecodeBatchState`, `Lfm2DecodeBatchState` and
//! the EP `Qwen35DecodeBatchEpState`, and it was the single largest remaining
//! reason the daemon named architecture types at all (17 of 87 uses).
//!
//! It belongs here rather than in the daemon because `LoadedModel` already
//! owns the typed fields these write (`qwen35_decode_batch`,
//! `lfm2_decode_batch`, `EpArch::Qwen35 { batch, .. }`) — only the
//! *construction* had leaked upward.
//!
//! `ContinuousBatchScheduler` lives in `hipfire-engine`, which is *above* this
//! crate, so staging returns the two numbers the scheduler needs and the
//! caller constructs it. That keeps the dependency edge pointing one way.
//!
//! The body is a verbatim move: identical branch order, identical fail-closed
//! EP invariants, identical operator-visible `[daemon]` / `[daemon][EP]`
//! strings. Only the local `staged_*` variables became [`BatchStaging`]
//! fields.

use crate::LoadedModel;
use hipfire_arch_qwen35::qwen35::StateQuant;
use rdna_compute::Gpu;

/// What the load handler needs to know after staging.
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchStaging {
    /// Continuous batching is live for this model.
    pub capable: bool,
    /// Scheduler slot count (0 when not staged).
    pub slots: usize,
    /// Per-lane KV capacity, clamped to what independent attention admits.
    pub lane_capacity: usize,
    /// The expert-parallel (TP=4) route was taken rather than single-GPU.
    pub ep: bool,
    /// EP slot count, mirrored for the receipt.
    pub ep_slots: usize,
    /// EP lane capacity, mirrored for the receipt.
    pub ep_lane_cap: usize,
    /// The VMM continuous-batching route was staged (per-request VMM KV in
    /// `Qwen35Bundle::vmm_store`, no fixed lanes).
    pub vmm: bool,
    /// Admitted VMM step row budget (scratch-fitted `serve.max_batch_tokens`).
    pub row_budget: usize,
}

/// Load-time request for the VMM continuous-batching route.
#[derive(Debug, Clone, Copy)]
pub struct VmmStagingRequest {
    /// Global trunk-row budget per step (`serve.max_batch_tokens`).
    pub row_budget: usize,
    /// `Exact` (default; byte-identical to the singleton route) or
    /// `Nonexact` (only under `HIPFIRE_SERVE_BATCH_NONEXACT=1`).
    pub route: hipfire_arch_qwen35::forward_slots::vmm::VmmRoute,
    /// Decode requests as cross-request batched MTP spec lanes when the
    /// model serves MTP (`serve_vmm_batch.spec`, default on).
    pub spec: bool,
    /// Admit sampled (temperature>0) requests to the batch route for lane
    /// modes proven byte-identical to the same-mode singleton
    /// (`serve_vmm_batch.sampled`, `serve.batch_sampled`, default off).
    pub sampled: bool,
}

/// True when embedding and lm_head formats admit the batched decode kernels.
pub fn qwen_batch_weight_formats_supported(
    weights: &hipfire_arch_qwen35::qwen35::Qwen35Weights,
) -> bool {
    use hipfire_runtime::llama::EmbeddingFormat;
    use rdna_compute::DType;
    let embd_ok = matches!(
        weights.embd_format,
        EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0
    );
    let lm_ok = matches!(
        weights.output.gpu_dtype,
        DType::Q8_0
            | DType::HFQ4G256
            | DType::MQ4G256
            | DType::HFQ6G256
            | DType::MQ6G256
            | DType::MQ3G256
    );
    embd_ok && lm_ok
}

/// Stage continuous batching for `m`, returning what the caller must publish.
///
/// Returns `Err` when staging leaves no usable pool behind: any device-sync
/// or GPU-free failure during batch publication/cleanup is fatal (the batch
/// is unpublished and the sequential PBS/partial pool is drained or
/// untrusted). The caller must roll back the whole staged model and never
/// emit a `loaded` ack. `Ok` preserves the safe sequential fallback: failed
/// optional allocation, or a failed peer-enable whose cleanup succeeds with
/// the sequential pool intact, simply reports `capable == false`.
pub fn stage_continuous_batch(
    m: &mut LoadedModel,
    gpu: &mut Gpu,
    requested: usize,
    vmm: Option<VmmStagingRequest>,
) -> Result<BatchStaging, String> {
    let mut out = BatchStaging::default();
    if let Some(req) = vmm {
        if requested > 1 && m.pp == 1 && m.ep.is_none() {
            match crate::continuous_batch_route(m.arch_id) {
                Some(crate::ContinuousBatchRoute::Qwen35) => {
                    return Ok(stage_qwen_vmm_batch(m, gpu, requested, req));
                }
                Some(crate::ContinuousBatchRoute::Qwen4) => {
                    return Ok(stage_qwen4_lanes(m, gpu, requested, req));
                }
                _ => {}
            }
        }
        eprintln!(
            "[daemon] VMM continuous batch requested but unsupported (arch_id={} pp={} ep={}) — existing route",
            m.arch_id,
            m.pp,
            m.ep.is_some()
        );
    }
    // ── Continuous batch staging (must be before `loaded` ack) ──
    // Stage Qwen35DecodeBatchState / hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState (single-GPU) or
    // Qwen35DecodeBatchEpState (EP TP=4 pure gfx1201) + host scheduler.
    // `continuous_batch_capable` reflects the newly staged state, not the previous.
    // EP is batch-only: TP must be 4 and exactly 4×gfx1201, else fail closed.
    // Allocation failure advertises false and preserves sequential/poison handling.
    if requested > 1 && m.pp == 1 && m.ep.is_none() {
        match crate::continuous_batch_route(m.arch_id) {
            Some(crate::ContinuousBatchRoute::Qwen35) => {
                // Immutable borrow of `m` ends after this extraction; mutable borrow for batch field later is disjoint.
                let qwen_info = m.qwen35().map(|b| {
                    (
                        qwen_batch_weight_formats_supported(&b.weights),
                        b.scratch.repeat_buf.buf.size(),
                        b.config.head_dim,
                        b.config.clone(),
                        b.weights.embd_format,
                        b.weights.output.gpu_dtype,
                    )
                });
                if let Some((
                    weight_ok,
                    scratch_size,
                    head_dim,
                    config_clone,
                    embd_fmt,
                    out_dtype,
                )) = qwen_info
                {
                    if !weight_ok {
                        eprintln!(
                                            "[daemon] continuous batch requested but weight formats unsupported (embd={:?} lm_head={:?}) — fallback to sequential",
                                            embd_fmt, out_dtype
                                        );
                    } else {
                        let repeat_cap = (scratch_size / 4).max(1);
                        let max_attention_lane =
                            gpu.attention_q8_0_kv_independent_max_lane_capacity(head_dim);
                        let batch_lane_capacity = m.max_seq.min(max_attention_lane);
                        if batch_lane_capacity == 0 {
                            eprintln!(
                                                "[daemon] continuous batch unavailable: independent attention admits no lanes — fallback to sequential"
                                            );
                        } else {
                            if batch_lane_capacity < m.max_seq {
                                eprintln!(
                                                    "[daemon] continuous batch lane capacity clamped: requested={} supported={}",
                                                    m.max_seq,
                                                    batch_lane_capacity
                                                );
                            }
                            match hipfire_arch_qwen35::qwen35::Qwen35DecodeBatchState::new(
                                gpu,
                                &config_clone,
                                requested,
                                batch_lane_capacity,
                                repeat_cap,
                            ) {
                                Ok(batch_state) => {
                                    m.qwen35_mut().unwrap().qwen35_decode_batch = Some(batch_state);
                                    out.slots = requested;
                                    out.lane_capacity = batch_lane_capacity;
                                    out.capable = true;
                                    eprintln!(
                                                        "[daemon] continuous batch staged: slots={} lane_cap={} repeat_cap={}",
                                                        requested,
                                                        batch_lane_capacity,
                                                        repeat_cap
                                                    );
                                }
                                Err(e) => {
                                    eprintln!(
                                                        "[daemon] continuous batch allocation failed: {e} — fallback to sequential"
                                                    );
                                }
                            }
                        }
                    }
                } else {
                    eprintln!("[daemon] continuous batch requested but model state not Qwen35 — fallback to sequential");
                }
            }
            Some(crate::ContinuousBatchRoute::Qwen4) => {
                // The only qwen4 batch route is the exact fn-lanes store,
                // staged through the VMM branch above; there is no
                // fixed-lane qwen4 batch state.
                eprintln!("[daemon] qwen4 lanes need serve.vmm_batch — fallback to sequential");
            }
            Some(crate::ContinuousBatchRoute::Lfm2Moe) => {
                if m.lfm2moe().is_none() {
                    eprintln!("[daemon] continuous batch requested but model state not Lfm2Moe — fallback to sequential");
                } else if !m.lfm2moe().unwrap().config.is_dense() {
                    eprintln!(
                                        "[daemon] continuous batch requested but LFM MoE not supported (dense only) — fallback to sequential"
                                    );
                } else if let Err(reason) = hipfire_arch_lfm2moe::batch_weight_formats_supported(
                    &m.lfm2moe().unwrap().weights,
                ) {
                    eprintln!(
                                        "[daemon] continuous batch requested but weight formats unsupported: {} — fallback to sequential",
                                        reason
                                    );
                } else {
                    let repeat_cap = 2048usize.max(1);
                    let max_attention_lane = {
                        let b = m.lfm2moe().unwrap();
                        gpu.attention_q8_0_kv_independent_max_lane_capacity(b.config.head_dim)
                    };
                    let batch_lane_capacity = m.max_seq.min(max_attention_lane);
                    if batch_lane_capacity == 0 {
                        eprintln!(
                                            "[daemon] continuous batch unavailable: independent attention admits no lanes — fallback to sequential"
                                        );
                    } else {
                        if batch_lane_capacity < m.max_seq {
                            eprintln!(
                                                "[daemon] continuous batch lane capacity clamped: requested={} supported={}",
                                                m.max_seq,
                                                batch_lane_capacity
                                            );
                        }
                        // Clone config for the call so the immutable borrow ends before the mutable one.
                        let cfg = m.lfm2moe().unwrap().config.clone();
                        match hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState::new(
                            gpu,
                            &cfg,
                            requested,
                            batch_lane_capacity,
                            repeat_cap,
                        ) {
                            Ok(batch_state) => {
                                if let Some(b) = m.lfm2moe_mut() {
                                    b.lfm2_decode_batch = Some(batch_state);
                                    out.slots = requested;
                                    out.lane_capacity = batch_lane_capacity;
                                    out.capable = true;
                                    eprintln!(
                                                        "[daemon] continuous batch staged: slots={} lane_cap={} repeat_cap={}",
                                                        requested,
                                                        batch_lane_capacity,
                                                        repeat_cap
                                                    );
                                } else {
                                    // Should be unreachable (we checked is_some above), but free to avoid leak.
                                    batch_state.free_gpu(gpu);
                                    eprintln!("[daemon] continuous batch requested but model state not Lfm2Moe — fallback to sequential");
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                                    "[daemon] continuous batch allocation failed: {e} — fallback to sequential"
                                                );
                            }
                        }
                    }
                }
            }
            None => {
                eprintln!("[daemon] continuous batch requested but not capable (arch_id={} pp={} ep={:?}) — fallback to sequential", m.arch_id, m.pp, m.ep.is_some());
            }
        }
    } else if requested > 1 && m.pp == 1 && m.ep.is_some() {
        // EP Qwen35 pure expert-parallel batch route: TP=4, 4×gfx1201, batch-only.
        let tp_ok =
            m.ep.as_ref()
                .map(|ep| ep.gpus.devices.len() == 4)
                .unwrap_or(false);
        let gfx_ok =
            m.ep.as_ref()
                .map(|ep| ep.gpus.devices.iter().all(|d| d.arch_caps.is_gfx1201()))
                .unwrap_or(false);
        let arch_ok = matches!(m.arch_id, 5 | 6);
        if !arch_ok || !tp_ok || !gfx_ok {
            eprintln!("[daemon][EP] continuous batch requires arch 5/6, TP=4, 4×gfx1201 (arch_ok={arch_ok} tp_ok={tp_ok} gfx_ok={gfx_ok}) — fail closed");
            out.capable = false;
        } else if let Some(ep) = m.ep.as_mut() {
            if let crate::EpArch::Qwen35 {
                config,
                weights,
                batch,
                prefill_pbs,
                prefill_partials,
                ..
            } = &mut ep.inner
            {
                // Format admission lives solely in
                // `validate_ep_batch_compatibility` below (single authority).
                // Derive capacities similar to single-GPU but via EP Gpus handle when possible.
                let max_attention_lane = ep.gpus.devices[0]
                    .attention_q8_0_kv_independent_max_lane_capacity(config.head_dim);
                let batch_lane_capacity = m.max_seq.min(max_attention_lane).max(1);
                let repeat_cap = 128usize.max(1);
                let prefill_chunk = hipfire_arch_qwen35::qwen35::prefill_max_batch_ep();
                if batch_lane_capacity == 0 || batch_lane_capacity >= m.max_seq + 1 {
                    eprintln!("[daemon][EP] continuous batch lane capacity invalid — fail closed");
                } else {
                    let load_cfg = hipfire_arch_qwen35::qwen35::Qwen35BatchLoadConfig::new(
                        requested,
                        batch_lane_capacity,
                        repeat_cap,
                        prefill_chunk,
                    );
                    // Fail-closed validation before allocation.
                    match hipfire_arch_qwen35::qwen35::validate_ep_batch_compatibility(
                        &ep.gpus, weights, config, &load_cfg,
                    ) {
                        Ok(compat) => {
                            // Enforce frozen invariants.
                            if compat.rank_count() != 4
                                || compat.rank_mask() != 0x0f
                                || compat.reduce()
                                    != hipfire_arch_qwen35::qwen35::Qwen35EpReduce::PeerRootedF32
                                || compat.topology()
                                    != hipfire_arch_qwen35::qwen35::Qwen35EpTopology::ExpertParallel
                            {
                                eprintln!("[daemon][EP] compat invariants violated — fail closed: rank_count={} mask={:#x} reduce={:?} topo={:?}", compat.rank_count(), compat.rank_mask(), compat.reduce(), compat.topology());
                            } else {
                                // Keep load-owned sequential PBS/partials intact through
                                // batch construction and peer enable so allocation or
                                // enable_peer_all failure can still fall back to them.
                                // Free them only at the publication boundary after both
                                // succeed (post-sync), retaining only the batch seed pool.
                                match hipfire_arch_qwen35::qwen35::Qwen35DecodeBatchEpState::new(
                                    &mut ep.gpus,
                                    weights,
                                    config,
                                    &load_cfg,
                                ) {
                                    Ok(ep_batch) => {
                                        // Attest receipt getters work before publishing.
                                        let _ = ep_batch.max_batch();
                                        let _ = ep_batch.lane_capacity();
                                        // Peer access MUST follow every peer-visible batch
                                        // allocation (partials + leased scratch); ROCm may
                                        // not retroactively map late allocs.
                                        match ep.gpus.enable_peer_all() {
                                            Ok(peer_access) => {
                                                // Publication transaction: sync, free unused
                                                // sequential owners, then publish batch. Any
                                                // cleanup failure must not leave batch=None
                                                // with empty PBS as a silent sequential path.
                                                let mut first_cleanup_err = None;
                                                for dev in ep.gpus.devices.iter_mut() {
                                                    if let Err(e) = dev.bind_thread() {
                                                        first_cleanup_err.get_or_insert(e);
                                                        continue;
                                                    }
                                                    if let Err(e) = dev.hip.device_synchronize() {
                                                        first_cleanup_err.get_or_insert(e);
                                                    }
                                                }
                                                if let Some(sync_err) = first_cleanup_err.take() {
                                                    // A failed device sync leaves neither pool
                                                    // trustworthy: free the batch best-effort
                                                    // and fail the whole load (never `loaded`
                                                    // ack on an unusable resident).
                                                    let cleanup_note = match ep_batch
                                                        .free_gpu(&mut ep.gpus)
                                                    {
                                                        Ok(()) => {
                                                            eprintln!("[daemon][EP] device sync failed after peer enable before sequential free: {sync_err:?} — fail closed (batch freed)");
                                                            String::new()
                                                        }
                                                        Err(cleanup_err) => {
                                                            eprintln!("[daemon][EP] device sync failed after peer enable before sequential free: {sync_err:?}; batch cleanup also failed: {cleanup_err:?} — fail closed");
                                                            format!("; batch cleanup also failed: {cleanup_err:?}")
                                                        }
                                                    };
                                                    return Err(format!("device sync failed after peer enable before sequential free: {sync_err:?}{cleanup_note}"));
                                                } else {
                                                    for (r, pbs) in
                                                        prefill_pbs.drain(..).enumerate()
                                                    {
                                                        if let Some(dev) =
                                                            ep.gpus.devices.get_mut(r)
                                                        {
                                                            if let Err(e) = dev.bind_thread() {
                                                                first_cleanup_err.get_or_insert(e);
                                                            }
                                                            if let Err(e) = pbs.free_gpu(dev) {
                                                                first_cleanup_err.get_or_insert(e);
                                                            }
                                                        }
                                                    }
                                                    for (r, p) in
                                                        prefill_partials.drain(..).enumerate()
                                                    {
                                                        if let Some(dev) =
                                                            ep.gpus.devices.get_mut(r)
                                                        {
                                                            if let Err(e) = dev.bind_thread() {
                                                                first_cleanup_err.get_or_insert(e);
                                                            }
                                                            if let Err(e) = dev.free_tensor(p) {
                                                                first_cleanup_err.get_or_insert(e);
                                                            }
                                                        }
                                                    }
                                                    if let Some(free_err) = first_cleanup_err {
                                                        // Sequential pool already drained — do not
                                                        // publish batch, free it, and fail the
                                                        // load as unusable (no silent empty-PBS
                                                        // sequential fallback).
                                                        let cleanup_note = match ep_batch
                                                            .free_gpu(&mut ep.gpus)
                                                        {
                                                            Ok(()) => {
                                                                eprintln!("[daemon][EP] sequential scratch free failed after peer enable: {free_err:?} — fail closed (batch freed; sequential pool unusable)");
                                                                String::new()
                                                            }
                                                            Err(cleanup_err) => {
                                                                eprintln!("[daemon][EP] sequential scratch free failed after peer enable: {free_err:?}; batch cleanup also failed: {cleanup_err:?} — fail closed (sequential pool unusable)");
                                                                format!("; batch cleanup also failed: {cleanup_err:?}")
                                                            }
                                                        };
                                                        return Err(format!("sequential scratch free failed after peer enable: {free_err:?}{cleanup_note}"));
                                                    } else {
                                                        *batch = Some(ep_batch);
                                                        out.slots = requested;
                                                        out.lane_capacity = batch_lane_capacity;
                                                        out.capable = true;
                                                        out.ep = true;
                                                        out.ep_slots = requested;
                                                        out.ep_lane_cap = batch_lane_capacity;
                                                        eprintln!("[daemon][EP] expert-parallel batch staged: slots={} lane_cap={} repeat_cap={} prefill_chunk={} reduce=peer_rooted_f32 rank_count=4 peer_access={}", requested, batch_lane_capacity, repeat_cap, prefill_chunk, peer_access);
                                                    }
                                                }
                                            }
                                            Err(enable_err) => {
                                                // Sequential owners untouched: a successful
                                                // batch free keeps the sequential fallback
                                                // usable. A failed free leaves neither pool
                                                // trustworthy — fail the whole load.
                                                match ep_batch.free_gpu(&mut ep.gpus) {
                                                    Ok(()) => {
                                                        eprintln!("[daemon][EP] enable_peer_all failed after batch alloc: {enable_err:?} — fail closed (batch freed; sequential pool preserved)");
                                                    }
                                                    Err(cleanup_err) => {
                                                        eprintln!("[daemon][EP] enable_peer_all failed after batch alloc: {enable_err:?}; cleanup also failed: {cleanup_err:?} — fail closed");
                                                        return Err(format!("enable_peer_all failed after batch alloc: {enable_err:?}; batch cleanup also failed: {cleanup_err:?}"));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        // Sequential owners untouched — sequential fallback remains usable.
                                        eprintln!("[daemon][EP] expert-parallel batch allocation failed: {e} — fail closed (sequential pool preserved)");
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[daemon][EP] expert-parallel batch compatibility failed: {e} — fail closed");
                        }
                    }
                }
            } else {
                eprintln!(
                    "[daemon][EP] continuous batch requested but EP arch not Qwen35 — fail closed"
                );
            }
        }
    } else if requested > 1 {
        eprintln!("[daemon] continuous batch requested but not capable (arch_id={} pp={} ep={:?}) — fallback to sequential", m.arch_id, m.pp, m.ep.is_some());
    }
    // Qwen EP peer finalization after every optional batch alloc/cleanup path.
    // Load defers enable_peer_all so late batch scratch is peer-mapped; sequential
    // (requested<=1) never hits the batch success enable, so without this the flag
    // stays false and EP falls through to RCCL / HIP host-staging allreduce.
    // Batch success already enabled → skip. Err/false: report truthfully; do not
    // claim peer-rooted when unavailable (sequential owners stay usable).
    if let Some(ep) = m.ep.as_mut() {
        if matches!(&ep.inner, crate::EpArch::Qwen35 { .. }) && !ep.gpus.peer_access_enabled {
            match ep.gpus.enable_peer_all() {
                Ok(true) => {
                    eprintln!(
                        "[daemon][EP] peer access enabled after staging (sequential/fallback path)"
                    );
                }
                Ok(false) => {
                    eprintln!("[daemon][EP] peer access incomplete after staging — peer-rooted path unavailable");
                }
                Err(e) => {
                    eprintln!("[daemon][EP] enable_peer_all failed after staging: {e:?} — peer-rooted path unavailable");
                }
            }
        }
    }
    Ok(out)
}

/// Stage the exact Flash-Next (qwen4) lane store beside the resident
/// bundle: native MTP lanes when `req.spec` (`serve.batch_spec`) is on and
/// the loaded model carries the MTP head, AR lanes otherwise. Qwen4 has only
/// the exact route: the non-exact `VmmRoute` and any
/// non-VMM QSA backend are refused, as is a lane store that does not fit the
/// free device memory. Every refusal logs the reason and keeps the singleton
/// route (`capable == false`); nothing is staged.
fn stage_qwen4_lanes(
    m: &mut LoadedModel,
    gpu: &mut Gpu,
    requested: usize,
    req: VmmStagingRequest,
) -> BatchStaging {
    use hipfire_arch_qwen4::lane::Qwen4LaneStore;
    use hipfire_arch_qwen4::{Qwen4ContextCommit, Qwen4KvBackend};
    let refuse = |reason: &str| {
        eprintln!("[daemon] qwen4 lanes refused: {reason} — fallback to sequential");
        BatchStaging::default()
    };
    if req.route == hipfire_arch_qwen35::forward_slots::vmm::VmmRoute::Nonexact {
        return refuse("qwen4 has only the exact lane route");
    }
    let ep = crate::admission::effective_ep(m.arch_id);
    if ep > 1 {
        return refuse(&format!(
            "expert parallelism (topology.ep = {ep}) serves one request at a time; lanes/continuous batching need topology.ep = 1"
        ));
    }
    let max_seq = m.max_seq;
    let spec_off_reason = if !req.spec {
        Some("serve.batch_spec off")
    } else if !m.mtp_weights_present {
        Some("no MTP head loaded")
    } else if m.mtp_mode == "off" {
        Some("MTP mode off")
    } else {
        None
    };
    // The lane ring is `K + 1` rows and a verify block is at most
    // `LANE_MAX_ROWS` rows, so the lane draft depth is bounded by it.
    let spec_k = m
        .mtp_k
        .min(hipfire_arch_qwen4::gpu_forward_lanes::LANE_MAX_ROWS - 1)
        .max(1);
    let end_of_turn = m
        .tokenizer
        .as_ref()
        .and_then(|t| t.special_token_id("<|im_end|>"));
    let Some(bundle) = m.qwen4_mut() else {
        return refuse("model state not Qwen4");
    };
    let spec = spec_off_reason
        .is_none()
        .then_some(hipfire_arch_qwen4::lane::LaneSpecConfig {
            k: spec_k,
            end_of_turn,
        });
    let spec_on_k = spec.as_ref().map(|s| s.k);
    if bundle.state.qsa_backend() != Qwen4KvBackend::Vmm {
        return refuse("QSA context backend is not VMM");
    }
    let row_budget = req.row_budget.max(requested);
    let chunk = hipfire_arch_qwen4::gpu_forward::qwen4_prefill_chunk_requested(&gpu.arch, max_seq);
    let context = Qwen4ContextCommit::new(
        Qwen4KvBackend::Vmm,
        max_seq,
        chunk,
        gpu.vmm_recommended_granularity().unwrap_or(1),
    );
    let Some(need) = Qwen4LaneStore::device_bytes(
        &bundle.config,
        bundle.state_format(),
        &context,
        requested,
        row_budget,
        spec_on_k,
    ) else {
        return refuse("lane store device bytes overflow");
    };
    let free = match gpu.device_mem_info() {
        Ok((free, _)) => free as u64,
        Err(e) => return refuse(&format!("device memory query failed: {e}")),
    };
    if need > free {
        return refuse(&format!(
            "lane store needs {} MiB, {} MiB free (lanes={requested} row_budget={row_budget})",
            need >> 20,
            free >> 20
        ));
    }
    // The fn-lanes route is outside Flash-Next's walked kernel plan: return the
    // load to lazy kernel loading before staging dispatches anything.
    if gpu.route_kernels_finished() {
        if let Err(e) = gpu.abort_route_kernel_load() {
            return refuse(&format!("kernel route reset failed: {e}"));
        }
        eprintln!(
            "  kernels: {} [left the planned route: qwen4 fn-lanes]",
            crate::LAZY_KERNELS_NOTE
        );
    }
    if let Err(e) = bundle.stage_lanes(gpu, requested, row_budget, spec) {
        return refuse(&format!("staging failed: {e}"));
    }
    let Some(store) = bundle.lanes() else {
        return refuse("lane store missing after staging");
    };
    let spec_log = match (spec_on_k, spec_off_reason) {
        (Some(k), _) => format!("mtp K={k}"),
        (None, Some(reason)) => format!("off ({reason})"),
        (None, None) => "off".to_string(),
    };
    eprintln!(
        "[daemon] qwen4 fn-lanes staged: lanes={requested} row_budget={} spec={spec_log} policy={}",
        store.row_budget(),
        store.stage_policy().describe()
    );
    BatchStaging {
        capable: true,
        slots: requested,
        vmm: true,
        row_budget: store.row_budget(),
        lane_capacity: max_seq,
        ..Default::default()
    }
}

fn stochastic_q8_state_without_ef(quant: StateQuant, has_state: bool, has_ef: bool) -> bool {
    quant == StateQuant::Q8 && has_state && !has_ef
}

/// Stage the VMM continuous-batching store beside the resident Qwen35
/// bundle. No fixed-lane KV is allocated: request owners are created lazily
/// at admission, shaped like the resident (VMM) singleton KV. Any failure
/// keeps the existing singleton route (`capable == false`).
fn stage_qwen_vmm_batch(
    m: &mut LoadedModel,
    gpu: &mut Gpu,
    requested: usize,
    req: VmmStagingRequest,
) -> BatchStaging {
    let mut out = BatchStaging::default();
    let max_seq = m.max_seq;
    let Some(b) = m.qwen35_mut() else {
        eprintln!(
            "[daemon] VMM continuous batch requested but model state not Qwen35 — existing route"
        );
        return out;
    };
    if gpu.arch_caps.is_gfx1100()
        && req.route == hipfire_arch_qwen35::forward_slots::vmm::VmmRoute::Exact
        && stochastic_q8_state_without_ef(
            b.dn_state.quant,
            !b.dn_state.s_scales.is_empty(),
            !b.dn_state.s_ef_residual.is_empty(),
        )
    {
        // Legacy stochastic Q8 state is not singleton-exact under hipGraph:
        // replay freezes the captured requant frame, unlike eager AR lanes.
        eprintln!("[daemon] gfx1100 Exact VMM requires Q8 error feedback (HIPFIRE_DN_STATE_EF=1); stochastic EF-off singleton graph parity is unproven — existing route");
        return out;
    }
    // The executor's own coverage predicate (embedding/lm_head dtypes,
    // slots-body layer coverage, VMM KV route and flash plan) — not the
    // fixed-lane `qwen_batch_weight_formats_supported` gate.
    if let Err(e) = hipfire_arch_qwen35::forward_slots::vmm::vmm_executor_supports(
        gpu,
        &b.weights,
        &b.config,
        &b.kv_cache,
        req.route,
    ) {
        eprintln!("[daemon] VMM continuous batch unsupported for this model: {e} — existing route");
        return out;
    }
    // Requested trunk-row budget; when its stable scratch does not fit,
    // halve (never below the width) and report the admitted value — the
    // planner budget is the store's actual `row_budget()`.
    let mut row_budget = req.row_budget.max(requested);
    // Shared physical KV budget for every request owner: free VRAM after the
    // resident model, minus fixed headroom for executor scratch and
    // transient allocations. Admission/provision refuse past it.
    const KV_HEADROOM: usize = 1 << 30;
    loop {
        let free = gpu.device_mem_info().map(|(f, _)| f).unwrap_or(0);
        let kv_budget_bytes = free.saturating_sub(KV_HEADROOM);
        match hipfire_arch_qwen35::forward_slots::vmm::Qwen35VmmStore::new(
            gpu,
            &b.config,
            &b.kv_cache,
            requested,
            row_budget,
            kv_budget_bytes,
            req.route,
        ) {
            Ok(mut store) => {
                store.set_sampled(req.sampled);
                b.vmm_store = Some(store);
                out.capable = true;
                out.vmm = true;
                out.slots = requested;
                out.lane_capacity = max_seq;
                out.row_budget = row_budget;
                eprintln!(
                    "[daemon] VMM continuous batch staged: width={requested} row_budget={row_budget} (requested {}) max_seq_bound={max_seq} free_vram_mb={}",
                    req.row_budget,
                    free >> 20
                );
                break;
            }
            Err(e) if e.contains("out of memory") && row_budget / 2 >= requested => {
                eprintln!(
                    "[daemon] VMM continuous batch scratch for row_budget={row_budget} does not fit ({e}); halving"
                );
                row_budget /= 2;
            }
            Err(e) => {
                eprintln!(
                    "[daemon] VMM continuous batch unavailable: {e} (row_budget={row_budget}, free_vram_mb={}) — existing route",
                    free >> 20
                );
                break;
            }
        }
    }
    if out.vmm && req.spec {
        if gpu.arch_caps.is_gfx1100()
            && req.route == hipfire_arch_qwen35::forward_slots::vmm::VmmRoute::Exact
        {
            // gfx1100 exact AR uses singleton layers per request. Shared
            // speculative verify has a separate numerical/state contract.
            eprintln!("[daemon] gfx1100 Exact VMM: AR lanes only; batched speculative verification is not admitted");
        } else {
            stage_qwen_vmm_spec(m, gpu, requested, out.row_budget);
        }
    }
    out
}

/// Per-lane context cap of VMM spec lanes: the MTP head KV is dense per
/// lane, so the engine's head is loaded with this cap; longer requests
/// decode as AR lanes.
const VMM_SPEC_LANE_CAP: usize = 32768;

/// Stage the VMM store's speculation engine from the **actually loaded**
/// speculator (singleton load precedence DFlash > MTP > n-gram), never from
/// weight presence alone: a loaded DFlash draft installs a `VmmDflashEngine`
/// (shared draft weights, exact block-16 chain lanes) and never an MTP head;
/// an MTP speculator (or none) keeps the MTP engine path. Any refusal logs a
/// concrete reason and leaves AR lanes only.
fn stage_qwen_vmm_spec(m: &mut LoadedModel, gpu: &mut Gpu, lanes: usize, row_budget: usize) {
    if m.speculator.as_ref().is_some_and(|s| s.name() == "dflash") {
        stage_qwen_vmm_dflash(m, gpu, row_budget);
        return;
    }
    if !m.mtp_weights_present || m.mtp_mode == "off" {
        return;
    }
    let cap = m.max_seq.min(VMM_SPEC_LANE_CAP);
    let k = m.mtp_k;
    let trunk_path = std::path::PathBuf::from(&m.model_path);
    let Some(b) = m.qwen35_mut() else {
        return;
    };
    let (head, errors) = crate::resolve_qwen35_mtp_head(
        &trunk_path,
        None,
        (b.config.dim, b.config.vocab_size),
        gpu,
        cap,
        None,
    );
    let Some(head) = head else {
        eprintln!(
            "[daemon] VMM spec lanes unavailable: MTP head not loaded ({}) — AR lanes",
            errors.join("; ")
        );
        return;
    };
    let route = hipfire_arch_qwen35::mtp_spec::MtpPromptRoute::from_own_prefill(
        hipfire_config::mtp_own_prefill(),
    );
    // AR's penalty window bound: the resident scratch's `repeat_buf` (what
    // `spec_prefill` passes every lane).
    let ar_cap = b.scratch.repeat_buf.buf.size() / 4;
    match hipfire_arch_qwen35::forward_slots::vmm::spec::VmmSpecEngine::new(
        gpu,
        &b.config,
        &b.dn_state,
        head,
        k,
        lanes,
        route,
        ar_cap,
    ) {
        Ok(engine) => {
            let lane_bytes = engine.lane_bytes();
            let store = b.vmm_store.as_mut().expect("staged above");
            match store.install_spec(engine) {
                Ok(()) => {
                    eprintln!("[daemon] VMM MTP lane footprint: lane_state={lane_bytes} B");
                    eprintln!("[daemon] VMM spec lanes staged: MTP K={k} lane_cap={cap}");
                }
                Err(engine) => engine.free_gpu(gpu),
            }
        }
        Err(e) => eprintln!("[daemon] VMM spec lanes unavailable: {e} — AR lanes"),
    }
}

/// Stage the VMM store's DFlash engine over the loaded singleton's resident
/// draft weights (shared by reference count, never reloaded). Eligibility is
/// derived from the resolved loaded speculator (`vmm_assets`: fixed-block
/// chain, no DDTree, no admitted retained PM4, adaptive block off) and the
/// store (Exact route, `max_rows >= block`), not from a second env parse.
fn stage_qwen_vmm_dflash(m: &mut LoadedModel, gpu: &mut Gpu, row_budget: usize) {
    let assets = match m
        .speculator
        .as_mut()
        .and_then(|s| s.drafter_any_mut())
        .and_then(|a| a.downcast_mut::<hipfire_arch_qwen35::dflash_spec::DflashSpeculator>())
    {
        Some(df) => df.vmm_assets(),
        None => Err("loaded speculator is not a Qwen3.5 DFlashSpeculator".to_string()),
    };
    let assets = match assets {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[daemon] VMM DFlash lanes unavailable: {e} — AR lanes (singleton DFlash unchanged)");
            return;
        }
    };
    let Some(b) = m.qwen35_mut() else {
        hipfire_arch_qwen35::dflash_spec::release_shared_dflash_weights(gpu, assets.weights);
        return;
    };
    let route = b.vmm_store.as_ref().map(|s| s.route());
    if route != Some(hipfire_arch_qwen35::forward_slots::vmm::VmmRoute::Exact) {
        eprintln!(
            "[daemon] VMM DFlash lanes unavailable: store route {route:?} is not Exact (exactness is proven only on the Exact route) — AR lanes"
        );
        hipfire_arch_qwen35::dflash_spec::release_shared_dflash_weights(gpu, assets.weights);
        return;
    }
    let engine = match hipfire_arch_qwen35::forward_slots::vmm::dflash::VmmDflashEngine::new(
        gpu,
        &b.weights,
        &b.config,
        &b.dn_state,
        assets,
        row_budget.min(hipfire_arch_qwen35::qwen35::prefill::multi::MULTI_CHUNK_MAX_ROWS),
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[daemon] VMM DFlash lanes unavailable: {e} — AR lanes");
            return;
        }
    };
    let receipt = engine.receipt();
    let (lane_bytes, snapshot_bytes) = (engine.lane_bytes(), engine.snapshot_bytes());
    let store = b.vmm_store.as_mut().expect("route checked above");
    match store.install_dflash(engine) {
        Ok(()) => {
            eprintln!(
                "[daemon] VMM DFlash lanes staged: block={} chunk_row_limit={} ctx_capacity={} (shared draft weights)",
                receipt.block_size, receipt.chunk_row_limit, receipt.ctx_capacity
            );
            eprintln!("[daemon] VMM DFlash lane footprint: lane_state={lane_bytes} B dn_snapshot={snapshot_bytes} B");
        }
        Err(engine) => {
            eprintln!(
                "[daemon] VMM DFlash lanes unavailable: store row budget {} cannot run a {}-row block — AR lanes",
                store.row_budget(),
                receipt.block_size
            );
            engine.free_gpu(gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{stochastic_q8_state_without_ef, StateQuant};

    #[test]
    fn exact_ef_guard_only_rejects_live_stochastic_q8_state() {
        assert!(stochastic_q8_state_without_ef(StateQuant::Q8, true, false));
        assert!(!stochastic_q8_state_without_ef(StateQuant::Q8, true, true));
        assert!(!stochastic_q8_state_without_ef(
            StateQuant::Q8,
            false,
            false
        ));
        for quant in [StateQuant::FP32, StateQuant::Q4] {
            for has_state in [false, true] {
                for has_ef in [false, true] {
                    assert!(!stochastic_q8_state_without_ef(quant, has_state, has_ef));
                }
            }
        }
    }
}
