// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Expert-parallel (EP) executor for the Ship 6 super-op substrate.
//!
//! Runs a lowered [`LayerProgram`] **replicated across N ranks** (every rank
//! runs every op on full, replicated attention/dense weights), special-casing
//! the `Moe` super-op with one of two EP combines, selected per binding via
//! [`ForwardBindings::ep_moe_combine_mode`] (all ranks must agree; mixed
//! modes refuse):
//!
//! - **Root-routed slot order** (`EpMoeCombineMode::RootRoutedPartial`, Qwen
//!   plan-bound compact EP): the root seals SoftmaxTopK route production and
//!   returns a sealer-issued proof; every rank writes owned expert outputs into
//!   the global slot layout while zero-dummy experts write zero. The driver
//!   broadcasts root IDs and weights, gathers the slot rows to root, invokes
//!   the ordinary single-device slot-order combine once, byte-copies the
//!   finished partial to every rank, then adds it to each residual.
//! - **Rank-partial all-reduce** (`EpMoeCombineMode::RankPartial`, the default
//!   for DeepSeek4/MiniMax):
//!
//!   1. zero each rank's routed partial,
//!   2. each rank computes ONLY its owned experts (+ the shared expert on rank 0)
//!      into its partial via [`ForwardBindings::run_moe_ep`] (non-owned experts
//!      read load-time zero-dummy weights → contribute 0),
//!   3. `all_reduce_sum_f32` the partials across ranks (canonical deterministic
//!      rooted peer reduce; RCCL/legacy-unrooted only via explicit opt-in),
//!   4. each rank adds the reduced partial into its residual stream via
//!      [`ForwardBindings::ep_add_into_residual`].
//!
//! Other super-ops ordinarily run **replicated** and unchanged. An architecture
//! may explicitly opt `Attend` into dense tensor parallelism through the
//! fail-closed `ForwardBindings` attention-TP hooks; the default remains false,
//! so Qwen, MiniMax, and every existing EP route retain replicated attention.
//!
//! Ordering: every op (zero, root/contrib or `run_moe_ep`, transport, combine,
//! residual add, and the next layer's ops) is enqueued on each device's
//! `active_stream`, which is FIFO — so the per-rank sequence is correctly
//! ordered without host syncs between ops or layers. The decode driver syncs
//! once at the end before reading logits.
//!
//! This executor drives ONE layer's program across all ranks; the per-arch EP
//! driver loops layers (advancing each rank's per-layer binding state) the same
//! way the single-GPU lowered driver loops `run_layer_program`.

use crate::multi_gpu::{Gpus, PeerReduceScratchLease};
use hip_bridge::{DeviceBuffer, HipError};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    dispatch_super_op, EpMoeCombineMode, ForwardBindings, LayerProgram, SuperOpKind,
};
use hipfire_dispatch::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};

fn hip_err(e: HipError) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

/// Ensure every device owns an `active_stream` (the stream the EP collectives
/// and per-rank work run on). Idempotent; safe to call before each layer.
pub fn ensure_rank_streams(gpus: &mut Gpus) -> Result<(), DispatchError> {
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().map_err(hip_err)?;
        if dev.active_stream.is_none() {
            dev.active_stream = Some(dev.hip.stream_create().map_err(hip_err)?);
        }
    }
    Ok(())
}

/// Decode all-reduce selection. The DEFAULT is the canonical deterministic
/// rooted peer reduce ([`crate::multi_gpu::Gpus::all_reduce_sum_f32_peer_rooted`]:
/// every rank observes the exact same left-associated sum `((p0+p1)+p2)+...`
/// over ranks in index order, regardless of N). Both non-canonical transports
/// stay reachable via explicit opt-in only:
/// - `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` → legacy unrooted peer diagnostic,
/// - `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=0` → RCCL.
/// Without peer access the canonical path cannot run, so it falls back to
/// RCCL (and the selection log line says so).
#[derive(Clone, Copy, PartialEq, Eq)]
enum DecodeArMode {
    Canonical,
    LegacyPeer,
    Rccl,
}

static DECODE_AR_MODE: std::sync::LazyLock<DecodeArMode> = std::sync::LazyLock::new(|| {
    match hipfire_config::developer_var("HIPFIRE_EP_PEER_ALLREDUCE_DECODE").as_deref() {
        Ok("1") => DecodeArMode::LegacyPeer,
        Ok("0") => DecodeArMode::Rccl,
        _ => DecodeArMode::Canonical,
    }
});

fn all_reduce_sum_f32_decode(
    gpus: &mut Gpus,
    refs: &[&DeviceBuffer],
    count: usize,
    peer_lease: Option<&PeerReduceScratchLease>,
) -> Result<(), DispatchError> {
    // Batch-owned lease: fixed PeerRootedF32 contract, same ascending-rank
    // arithmetic as prefill. Bypasses decode mode selection entirely.
    if let Some(lease) = peer_lease {
        return gpus
            .all_reduce_sum_f32_peer_rooted_leased(lease, refs, count)
            .map_err(hip_err);
    }
    let mode = *DECODE_AR_MODE;
    let use_rooted = mode == DecodeArMode::Canonical && gpus.peer_access_enabled;
    // Selection log line: names the active path (once per process).
    static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LOGGED.get_or_init(|| {
        let path = match mode {
            DecodeArMode::Canonical if gpus.peer_access_enabled => {
                "canonical rooted-peer (fixed left fold over ranks; \
                 HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1 selects legacy unrooted peer, \
                 =0 selects RCCL)"
            }
            DecodeArMode::Canonical => {
                "RCCL (canonical rooted-peer unavailable: peer access disabled)"
            }
            DecodeArMode::LegacyPeer => "legacy unrooted peer (HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1)",
            DecodeArMode::Rccl => "RCCL (HIPFIRE_EP_PEER_ALLREDUCE_DECODE=0)",
        };
        eprintln!("EP decode all-reduce: {path}");
    });
    if mode == DecodeArMode::LegacyPeer {
        gpus.all_reduce_sum_f32_peer(refs, count).map_err(hip_err)
    } else if use_rooted {
        gpus.all_reduce_sum_f32_peer_rooted(refs, count)
            .map_err(hip_err)
    } else {
        gpus.all_reduce_sum_f32(refs, count).map_err(hip_err)
    }
}

/// Reduction transport selected by a sealed root-routed EP schedule.
///
/// The two normal modes intentionally retain their existing environment and
/// lease policy. `PrefillSkipAllReduce` is the explicit diagnostic switch used
/// by the prefill caller; it is not a fallback and therefore omits both the
/// reduction and residual completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootRoutedEpReduction {
    Decode,
    Prefill,
    PrefillSkipAllReduce,
}

/// Borrowed per-rank activation operands consumed by the executable schedule.
///
/// The slice is owned by the caller (typically a load-owned scratch pool); the
/// schedule never allocates or caches mutable resource identity.
pub struct RootRoutedEpOperands<'a> {
    pub partials: &'a [GpuTensor],
    pub partial_bytes: usize,
    pub reduce_count: usize,
    pub route_count: usize,
    pub contribution_count: usize,
    pub contribution_chunk: usize,
}

/// Existing route buffers borrowed by one schedule stage.
#[derive(Clone, Copy)]
pub struct EpRouteBuffers<'a> {
    pub ids: &'a DeviceBuffer,
    pub weights: &'a DeviceBuffer,
    pub slot_outputs: &'a DeviceBuffer,
}

/// Checked root-routed execution metadata for one admitted root-routed MoE
/// layer.
///
/// `derive` binds the metadata to the dispatch-owned execution contract and
/// its actual `moe`/layer/EP all-reduce row. It never infers an axis from the
/// first row and rejects duplicate or missing rows before the first side
/// effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootRoutedEpSchedule {
    contract_id: u64,
    n_ranks: usize,
    layer: usize,
    route_count: usize,
    partial_bytes: usize,
    reduce_count: usize,
    contribution_count: usize,
    contribution_chunk: usize,
    reduction: RootRoutedEpReduction,
}

impl RootRoutedEpSchedule {
    /// Derive the checked root-routed execution metadata from one load-bound
    /// execution contract.
    /// `n_ranks`, route length, and reduction geometry are supplied by the
    /// architecture adapter, but are checked here and again by the executor.
    /// The contract remains the only authority for the collective name/axis.
    pub fn derive(
        contract: &hipfire_dispatch::pipeline::sealed_moe::ExpertExecutionContract,
        n_ranks: usize,
        layer: usize,
        route_count: usize,
        partial_bytes: usize,
        reduce_count: usize,
        contribution_count: usize,
        contribution_chunk: usize,
        reduction: RootRoutedEpReduction,
    ) -> Result<Self, DispatchError> {
        if n_ranks == 0 {
            return Err(DispatchError::Hip(
                "root-routed EP schedule requires at least one rank".into(),
            ));
        }
        if n_ranks > MAX_EP_STACK_RANKS {
            return Err(DispatchError::Hip(format!(
                "root-routed EP schedule rank count {n_ranks} exceeds host-stack bound {MAX_EP_STACK_RANKS}"
            )));
        }
        if contract.physical_devices().len() != n_ranks {
            return Err(DispatchError::Hip(format!(
                "root-routed EP schedule mesh has {} contract devices, expected {n_ranks}",
                contract.physical_devices().len()
            )));
        }
        if contract.layer() != Some(layer) {
            return Err(DispatchError::Hip(format!(
                "root-routed EP schedule layer {layer} disagrees with contract layer {:?}",
                contract.layer()
            )));
        }
        if !contract.is_root_routed_ep() {
            return Err(DispatchError::Hip(
                "root-routed EP schedule requires an admitted root-routed EP contract".into(),
            ));
        }
        let mut matched = false;
        for row in contract.collective_rows() {
            if row.name != "moe" || row.layer != layer {
                continue;
            }
            if matched {
                return Err(DispatchError::Hip(format!(
                    "root-routed EP schedule has duplicate moe collective rows for layer {layer}"
                )));
            }
            if !matches!(
                row.hint,
                hipfire_dispatch::pipeline::sealed_moe::ContractCollectiveHint::AllReduce {
                    kind: hipfire_dispatch::pipeline::sealed_moe::ContractAxis::Ep
                }
            ) {
                return Err(DispatchError::Hip(format!(
                    "root-routed EP schedule moe row for layer {layer} is not an EP all-reduce"
                )));
            }
            matched = true;
        }
        if !matched {
            return Err(DispatchError::Hip(format!(
                "root-routed EP schedule has no moe EP all-reduce row for layer {layer}"
            )));
        }
        if route_count == 0 {
            return Err(DispatchError::Hip(
                "root-routed EP schedule requires a non-zero route count".into(),
            ));
        }
        if partial_bytes == 0 || partial_bytes % std::mem::size_of::<f32>() != 0 {
            return Err(DispatchError::Hip(
                "root-routed EP schedule partial byte count is not a non-zero f32 span".into(),
            ));
        }
        if reduce_count == 0 {
            return Err(DispatchError::Hip(
                "root-routed EP schedule reduction geometry is invalid".into(),
            ));
        }
        if let Some(bytes) = reduce_count.checked_mul(std::mem::size_of::<f32>()) {
            if bytes > partial_bytes {
                return Err(DispatchError::Hip(
                    "root-routed EP schedule reduction geometry is invalid".into(),
                ));
            }
        } else {
            return Err(DispatchError::Hip(
                "root-routed EP schedule reduction geometry is invalid".into(),
            ));
        }
        if contribution_count == 0 || contribution_chunk == 0 {
            return Err(DispatchError::Hip(
                "root-routed EP contribution geometry is invalid".into(),
            ));
        }

        Ok(Self {
            contract_id: contract.contract_id(),
            n_ranks,
            layer,
            route_count,
            partial_bytes,
            reduce_count,
            contribution_count,
            contribution_chunk,
            reduction,
        })
    }

    pub fn contract_id(&self) -> u64 {
        self.contract_id
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn rank_count(&self) -> usize {
        self.n_ranks
    }

    pub fn route_count(&self) -> usize {
        self.route_count
    }

    pub fn partial_bytes(&self) -> usize {
        self.partial_bytes
    }

    pub fn reduce_count(&self) -> usize {
        self.reduce_count
    }

    pub fn contribution_count(&self) -> usize {
        self.contribution_count
    }

    pub fn contribution_chunk(&self) -> usize {
        self.contribution_chunk
    }
}

/// Execute one checked root-routed EP schedule.
///
/// The callbacks are deliberately operand/arithmetic adapters only. This
/// function owns the preflight barrier, fixed stage order, ascending rank
/// traversal, route transfer, reduction selection, and stop-on-error behavior.
#[allow(clippy::too_many_arguments)]
pub fn execute_root_routed_ep<C, Admission: Copy, Proof: Copy, PF, PR, RC, RB, CN, FC, PP, RF>(
    gpus: &mut Gpus,
    context: &mut C,
    schedule: RootRoutedEpSchedule,
    operands: RootRoutedEpOperands<'_>,
    peer_lease: Option<&PeerReduceScratchLease>,
    mut preflight_root: PF,
    mut preflight_rank: PR,
    mut root_compute: RC,
    mut route_buffers: RB,
    mut rank_contribute: CN,
    mut finish_combine: FC,
    mut prepare_reduce: PP,
    mut residual_finish: RF,
) -> Result<(), DispatchError>
where
    PF: FnMut(&C, &Gpu, &GpuTensor) -> Result<Admission, DispatchError>,
    PR: FnMut(&C, usize, &Gpu, &GpuTensor, &Admission) -> Result<(), DispatchError>,
    RC: FnMut(&mut C, &mut Gpu, &GpuTensor, &Admission) -> Result<Proof, DispatchError>,
    RB: for<'a> FnMut(&'a C, usize) -> Result<EpRouteBuffers<'a>, DispatchError>,
    CN: FnMut(&mut C, usize, &mut Gpu, &Proof, &GpuTensor) -> Result<(), DispatchError>,
    FC: FnMut(&mut C, &mut Gpu, &GpuTensor) -> Result<(), DispatchError>,
    PP: FnMut(&mut C, usize, &mut Gpu, &GpuTensor) -> Result<(), DispatchError>,
    RF: FnMut(&mut C, usize, &mut Gpu, &GpuTensor) -> Result<(), DispatchError>,
{
    let n = gpus.devices.len();
    if n != schedule.n_ranks {
        return Err(DispatchError::Hip(format!(
            "root-routed EP schedule mesh has {n} ranks, expected {}",
            schedule.n_ranks
        )));
    }
    if operands.partials.len() != n {
        return Err(DispatchError::Hip(format!(
            "root-routed EP schedule has {} partials, expected {n}",
            operands.partials.len()
        )));
    }
    if operands.partial_bytes != schedule.partial_bytes
        || operands.reduce_count != schedule.reduce_count
        || operands.route_count != schedule.route_count
        || operands.contribution_count != schedule.contribution_count
        || operands.contribution_chunk != schedule.contribution_chunk
    {
        return Err(DispatchError::Hip(
            "root-routed EP operands disagree with checked schedule geometry".into(),
        ));
    }
    let route_bytes = schedule
        .route_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| DispatchError::Hip("root-routed EP route byte count overflow".into()))?;
    let contribution_bytes = schedule
        .contribution_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DispatchError::Hip("root-routed EP contribution byte count overflow".into())
        })?;
    for rank in 0..n {
        if gpus.devices[rank].active_stream.is_none() {
            return Err(DispatchError::Hip(format!(
                "root-routed EP device {rank} has no active_stream"
            )));
        }
        if operands.partials[rank].buf.size() < schedule.partial_bytes {
            return Err(DispatchError::Hip(format!(
                "root-routed EP rank {rank} partial has {} bytes, needs {}",
                operands.partials[rank].buf.size(),
                schedule.partial_bytes
            )));
        }
    }

    // Every callback and route buffer is validated before the first partial
    // memset. This is the transaction boundary for the whole schedule.
    let admission = preflight_root(context, &gpus.devices[0], &operands.partials[0])?;
    for rank in 1..n {
        preflight_rank(
            context,
            rank,
            &gpus.devices[rank],
            &operands.partials[rank],
            &admission,
        )?;
    }
    for rank in 0..n {
        let route = route_buffers(&*context, rank)?;
        if route.ids.size() < route_bytes || route.weights.size() < route_bytes {
            return Err(DispatchError::Hip(format!(
                "root-routed EP rank {rank} route buffers are smaller than {route_bytes} bytes"
            )));
        }
        if route.slot_outputs.size() < contribution_bytes {
            return Err(DispatchError::Hip(format!(
                "root-routed EP rank {rank} slot outputs have {} bytes, need {contribution_bytes}",
                route.slot_outputs.size()
            )));
        }
    }
    if let Some(lease) = peer_lease {
        if schedule.reduction != RootRoutedEpReduction::PrefillSkipAllReduce {
            let partial_refs: Vec<&DeviceBuffer> = operands
                .partials
                .iter()
                .map(|partial| &partial.buf)
                .collect();
            gpus.validate_peer_reduce_scratch_lease(
                lease,
                &partial_refs,
                schedule.contribution_chunk.max(schedule.reduce_count),
            )
            .map_err(hip_err)?;
        }
    }

    // The fixed order below is intentionally procedural rather than a second
    // command interpreter: this function is the one root-routed EP sequencer.
    // The proof is produced exactly once after zeroing and consumed by every
    // non-root contribution.
    for rank in 0..n {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread().map_err(hip_err)?;
        let stream = gpu.active_stream.as_ref().ok_or_else(|| {
            DispatchError::Hip(format!(
                "root-routed EP device {rank} lost its active_stream"
            ))
        })?;
        gpu.hip
            .memset_async(
                &operands.partials[rank].buf,
                0,
                schedule.partial_bytes,
                stream,
            )
            .map_err(hip_err)?;
    }

    let proof = {
        let gpu = &mut gpus.devices[0];
        gpu.bind_thread().map_err(hip_err)?;
        root_compute(context, gpu, &operands.partials[0], &admission)?
    };

    {
        let mut staged_routes = [None; MAX_EP_STACK_RANKS];
        for rank in 0..n {
            staged_routes[rank] = Some(route_buffers(&*context, rank)?);
        }
        let root = staged_routes[0].ok_or_else(|| {
            DispatchError::Hip(
                "root-routed EP root route buffers disappeared after root compute".into(),
            )
        })?;
        gpus.broadcast_ep_route(
            root.ids,
            root.weights,
            |rank| {
                let route =
                    staged_routes[rank].expect("root-routed EP route preflight omitted a rank");
                (route.ids, route.weights)
            },
            schedule.route_count,
        )
        .map_err(hip_err)?;
    }

    for rank in 1..n {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread().map_err(hip_err)?;
        rank_contribute(context, rank, gpu, &proof, &operands.partials[rank])?;
    }

    if schedule.reduction == RootRoutedEpReduction::PrefillSkipAllReduce {
        return Ok(());
    }

    {
        let mut staged_slots: [&DeviceBuffer; MAX_EP_STACK_RANKS] =
            [route_buffers(&*context, 0)?.slot_outputs; MAX_EP_STACK_RANKS];
        for rank in 1..n {
            staged_slots[rank] = route_buffers(&*context, rank)?.slot_outputs;
        }
        gpus.gather_unique_f32_to_root(
            peer_lease,
            &staged_slots[..n],
            schedule.contribution_count,
            schedule.contribution_chunk,
        )
        .map_err(hip_err)?;
    }

    {
        let gpu = &mut gpus.devices[0];
        gpu.bind_thread().map_err(hip_err)?;
        finish_combine(context, gpu, &operands.partials[0])?;
    }

    for rank in 0..n {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread().map_err(hip_err)?;
        prepare_reduce(context, rank, gpu, &operands.partials[rank])?;
    }

    {
        let mut staged: [&DeviceBuffer; MAX_EP_STACK_RANKS] =
            [&operands.partials[0].buf; MAX_EP_STACK_RANKS];
        for rank in 1..n {
            staged[rank] = &operands.partials[rank].buf;
        }
        gpus.broadcast_f32_from_root(&staged[..n], schedule.reduce_count)
            .map_err(hip_err)?;
    }

    for rank in 0..n {
        let gpu = &mut gpus.devices[rank];
        gpu.bind_thread().map_err(hip_err)?;
        residual_finish(context, rank, gpu, &operands.partials[rank])?;
    }
    Ok(())
}

fn all_reduce_sum_f32_prefill(
    gpus: &mut Gpus,
    refs: &[&DeviceBuffer],
    count: usize,
    peer_lease: Option<&PeerReduceScratchLease>,
) -> Result<(), DispatchError> {
    let ep_peer_ar_var = hipfire_config::developer_var("HIPFIRE_EP_PEER_ALLREDUCE");
    let ep_peer_ar = ep_peer_ar_var.as_deref();
    let ep_ar_canonical = !matches!(ep_peer_ar, Ok("0") | Ok("1"));
    static PREFILL_AR_LOG: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let use_rooted = ep_ar_canonical && gpus.peer_access_enabled;
    PREFILL_AR_LOG.get_or_init(|| {
        let path = if peer_lease.is_some() {
            "leased rooted-peer under batch lease (fixed left fold over ranks)"
        } else if use_rooted {
            "canonical rooted-peer (fixed left fold over ranks)"
        } else if ep_ar_canonical {
            "RCCL (canonical rooted-peer unavailable: peer access disabled)"
        } else if ep_peer_ar == Ok("0") {
            "RCCL (HIPFIRE_EP_PEER_ALLREDUCE=0)"
        } else {
            "legacy unrooted peer (HIPFIRE_EP_PEER_ALLREDUCE=1)"
        };
        eprintln!("EP prefill all-reduce: {path}");
    });
    if let Some(lease) = peer_lease {
        gpus.all_reduce_sum_f32_peer_rooted_leased(lease, refs, count)
            .map_err(hip_err)
    } else if use_rooted {
        gpus.all_reduce_sum_f32_peer_rooted(refs, count)
            .map_err(hip_err)
    } else if ep_ar_canonical || ep_peer_ar == Ok("0") {
        gpus.all_reduce_sum_f32(refs, count).map_err(hip_err)
    } else {
        gpus.all_reduce_sum_f32_peer(refs, count).map_err(hip_err)
    }
}

fn tp_peer_hc4_admitted<B: ForwardBindings>(gpus: &Gpus, bindings: &[B]) -> bool {
    gpus.devices.len() == 4
        && gpus.peer_access_enabled
        && gpus
            .devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201())
        && bindings.iter().all(ForwardBindings::supports_tp_peer_hc4)
}

fn tp_peer_hc3_admitted<B: ForwardBindings>(gpus: &Gpus, bindings: &[B]) -> bool {
    gpus.devices.len() == 3
        && gpus.peer_access_enabled
        && gpus
            .devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201())
        && bindings.iter().all(ForwardBindings::supports_tp_peer_hc3)
}

/// Execute one lowered layer program across `gpus.devices.len()` EP ranks.
///
/// - `bindings[r]` drives rank `r`'s forward (it holds that rank's state /
///   weights / per-layer counters by reference, exactly like the single-GPU
///   `ForwardBindings` impl).
/// - `partials[r]` is rank `r`'s zeroed routed-output scratch, a contiguous f32
///   buffer of length `residual_dim` on `gpus.devices[r]`. The executor owns the
///   zero/all-reduce/add lifecycle; the binding only writes its owned-expert
///   contribution into it during `run_moe_ep`.
/// - `residual_dim` is the residual width (= hidden size) used for the partial
///   memset byte size and the all-reduce element count.
///
/// Every device must have an `active_stream` set ([`ensure_rank_streams`]).
pub fn run_layer_program_ep<B: ForwardBindings>(
    gpus: &mut Gpus,
    bindings: &mut [B],
    partials: &[GpuTensor],
    program: &LayerProgram,
    residual_dim: usize,
    peer_lease: Option<&PeerReduceScratchLease>,
) -> Result<(), DispatchError> {
    let n = gpus.devices.len();
    assert_eq!(
        bindings.len(),
        n,
        "run_layer_program_ep: bindings.len() != n_ranks"
    );
    assert_eq!(
        partials.len(),
        n,
        "run_layer_program_ep: partials.len() != n_ranks"
    );

    for op in program {
        if matches!(op.kind, SuperOpKind::Attend)
            && bindings.iter().any(ForwardBindings::attention_tp_enabled)
        {
            if !bindings.iter().all(ForwardBindings::attention_tp_enabled) {
                return Err(DispatchError::Hip(
                    "run_layer_program_ep: mixed attention-TP admission across ranks".into(),
                ));
            }

            // Each rank computes its local head/O-LoRA shard and stops before
            // the residual mix, leaving one hidden-width partial in the
            // architecture-owned attention output tensor.
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                let ctx = DispatchCtx::new(&gpus.devices[r]);
                bindings[r].run_attend_ep(&mut gpus.devices[r], &ctx, &op.binding)?;
            }

            if tp_peer_hc3_admitted(gpus, bindings) {
                let peer_partials = bindings
                    .iter()
                    .map(|binding| {
                        let partial = binding.ep_attention_partial().ok_or_else(|| {
                            DispatchError::Hip(
                                "run_layer_program_ep: attention TP partial missing".into(),
                            )
                        })?;
                        Ok(GpuTensor {
                            buf: unsafe { partial.buf.alias() },
                            shape: partial.shape.clone(),
                            dtype: partial.dtype,
                        })
                    })
                    .collect::<Result<Vec<_>, DispatchError>>()?;
                let peers = [&peer_partials[0], &peer_partials[1], &peer_partials[2]];
                gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    bindings[r].ep_finish_attend_peer_hc3(&mut gpus.devices[r], peers)?;
                }
            } else if tp_peer_hc4_admitted(gpus, bindings) {
                // Borrow-independent aliases let the architecture hooks
                // consume all four peer pointers while each binding is
                // mutably advanced through its own HC residual mix.
                let peer_partials = bindings
                    .iter()
                    .map(|binding| {
                        let partial = binding.ep_attention_partial().ok_or_else(|| {
                            DispatchError::Hip(
                                "run_layer_program_ep: attention TP partial missing".into(),
                            )
                        })?;
                        Ok(GpuTensor {
                            buf: unsafe { partial.buf.alias() },
                            shape: partial.shape.clone(),
                            dtype: partial.dtype,
                        })
                    })
                    .collect::<Result<Vec<_>, DispatchError>>()?;
                let peers = [
                    &peer_partials[0],
                    &peer_partials[1],
                    &peer_partials[2],
                    &peer_partials[3],
                ];
                gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    bindings[r].ep_finish_attend_peer_hc4(&mut gpus.devices[r], peers)?;
                }
            } else {
                // Sum the input-column-sharded output projection directly in
                // its destination tensor. No staging copy or extra scratch.
                let refs: Vec<&DeviceBuffer> = bindings
                    .iter()
                    .map(|binding| {
                        binding
                            .ep_attention_partial()
                            .map(|partial| &partial.buf)
                            .ok_or_else(|| {
                                DispatchError::Hip(
                                    "run_layer_program_ep: attention TP partial missing".into(),
                                )
                            })
                    })
                    .collect::<Result<_, _>>()?;
                all_reduce_sum_f32_decode(gpus, &refs, residual_dim, peer_lease)?;
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    bindings[r].ep_finish_attend(&mut gpus.devices[r])?;
                }
            }
        } else if matches!(op.kind, SuperOpKind::Moe) {
            // Root-routed vs rank-partial is reported per binding; every rank
            // must agree. Mixed modes return Err — automatic fallback from
            // root-routed to rank partials is a review veto.
            let all_root_routed = bindings
                .iter()
                .all(|b| b.ep_moe_combine_mode() == EpMoeCombineMode::RootRoutedPartial);
            let all_partial = bindings
                .iter()
                .all(|b| b.ep_moe_combine_mode() == EpMoeCombineMode::RankPartial);
            if all_root_routed {
                run_moe_ep_root_routed(
                    gpus,
                    bindings,
                    &op.binding,
                    partials,
                    residual_dim,
                    peer_lease,
                )?;
            } else if all_partial {
                // 1. Zero each rank's routed partial on its own stream.
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    let stream = gpus.devices[r]
                        .active_stream
                        .as_ref()
                        .ok_or_else(|| DispatchError::Hip(format!(
                            "run_layer_program_ep: device {r} has no active_stream (call ensure_rank_streams)"
                        )))?;
                    gpus.devices[r]
                        .hip
                        .memset_async(&partials[r].buf, 0, residual_dim * 4, stream)
                        .map_err(hip_err)?;
                }

                // 2. Each rank computes its owned-expert routed partial (+ shared on
                //    rank 0 via skip_shared=false; ranks>0 skip the shared down).
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    let ctx = DispatchCtx::new(&gpus.devices[r]);
                    bindings[r].run_moe_ep(
                        &mut gpus.devices[r],
                        &ctx,
                        &op.binding,
                        &partials[r],
                        /* skip_shared = */ r != 0,
                    )?;
                }

                if tp_peer_hc3_admitted(gpus, bindings) {
                    let peers = [&partials[0], &partials[1], &partials[2]];
                    gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].ep_finish_moe_peer_hc3(&mut gpus.devices[r], peers)?;
                    }
                } else if tp_peer_hc4_admitted(gpus, bindings) {
                    let peers = [&partials[0], &partials[1], &partials[2], &partials[3]];
                    gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].ep_finish_moe_peer_hc4(&mut gpus.devices[r], peers)?;
                    }
                } else {
                    // 3. All-reduce-sum the partials across ranks (in-place, RCCL).
                    let refs: Vec<&DeviceBuffer> = partials.iter().map(|p| &p.buf).collect();
                    all_reduce_sum_f32_decode(gpus, &refs, residual_dim, peer_lease)?;

                    // 4. Fold the reduced partial into each residual stream.
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].ep_add_into_residual(&mut gpus.devices[r], &partials[r])?;
                    }
                }
            } else {
                return Err(DispatchError::Hip(
                    "run_layer_program_ep: mixed EP MoE combine modes across ranks; refusing (no fallback from root-routed to rank partials)".into(),
                ));
            }
        } else {
            // Replicated op — every rank runs it unchanged on full weights.
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                let ctx = DispatchCtx::new(&gpus.devices[r]);
                dispatch_super_op(&mut gpus.devices[r], &ctx, op, &mut bindings[r])?;
            }
        }
    }
    Ok(())
}

/// Maximum EP ranks the root-routed driver stages in host-stack storage.
/// Existing architectural bound: rank identity is a `u8` (the qwen35 loader
/// refuses wider EP shard identity) and rank sets are `u64` masks
/// (`Qwen35EpBatchReceipt` / `Qwen35BatchCompatibility`), so meshes past 64
/// ranks are not admittable upstream. The driver still checks `n` against
/// this bound fail-closed before ANY work. Host stack only (512 B worst
/// case) — never a GPU allocation, never a 4-only mesh assumption.
const MAX_EP_STACK_RANKS: usize = 64;

/// Root-routed EP MoE for one layer program op. Only entered when every rank
/// reports [`EpMoeCombineMode::RootRoutedPartial`].
///
/// State machine (decode; k is the flat top-k width, typically 8):
/// 1. Preflight partial capacities, root/non-root sealed roles, and static
///    contract/layer/hidden/k/n_exp agreement across ranks — fail-closed
///    **before** any scratch mutation. Contract identity is the load-bound
///    `u64` (`contract_id`); no per-token fingerprint String render.
/// 1b. Pure binding/params/proof preflight on EVERY rank (root builds the
///    actual seal inputs and returns a validation proof; each non-root
///    validates its actual seal inputs against it; enqueues NOTHING) —
///    still before zeroing ANY partial.
/// 2. Zero every rank's routed partial on its own stream.
/// 3. Rank 0 runs [`ForwardBindings::ep_run_moe_root`] (router + owned slot
///    outputs + shared exactly once into its zeroed partial) and returns the
///    opaque route-producer proof.
/// 4. Re-borrow each rank's resident route buffers and broadcast root top-k
///    IDs and weights via [`Gpus::broadcast_ep_route`].
/// 5. Every non-root runs [`ForwardBindings::ep_run_moe_contrib`] against the
///    proof, writing its owned outputs into the same global slot layout.
/// 6. Gather slot outputs to root without folding slots, run the ordinary
///    single-device slot-order combine once, byte-copy the finished partial to
///    every rank, then call [`ForwardBindings::ep_add_into_residual`] once per
///    rank.
///
/// Any error is fail-stop for the token: no retry, no fallback to the
/// independent-router rank-partial path.
struct DecodeRootRoutedContext<'a, B: ForwardBindings> {
    bindings: &'a mut [B],
    op: &'a hipfire_dispatch::pipeline::superop::OpBinding,
}

fn run_moe_ep_root_routed<B: ForwardBindings>(
    gpus: &mut Gpus,
    bindings: &mut [B],
    op: &hipfire_dispatch::pipeline::superop::OpBinding,
    partials: &[GpuTensor],
    residual_dim: usize,
    peer_lease: Option<&PeerReduceScratchLease>,
) -> Result<(), DispatchError> {
    let n = gpus.devices.len();
    if n == 0 {
        return Err(DispatchError::Hip(
            "run_layer_program_ep: root-routed EP mesh has no ranks".into(),
        ));
    }
    if bindings.len() != n {
        return Err(DispatchError::Hip(
            "run_layer_program_ep: root-routed EP binding count disagrees with mesh".into(),
        ));
    }
    if partials.len() != n {
        return Err(DispatchError::Hip(
            "run_layer_program_ep: root-routed EP partial count disagrees with mesh".into(),
        ));
    }
    let need_bytes = residual_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DispatchError::Hip(
                "run_layer_program_ep: residual_dim*4 partial byte count overflow".into(),
            )
        })?;

    let (
        schedule,
        root_contract_id,
        root_layer,
        root_hidden,
        root_k,
        root_n_exp,
        contribution_count,
    ) = {
        let root_view = bindings[0].ep_moe_route_view().ok_or_else(|| {
            DispatchError::Hip(
                "run_layer_program_ep: root-routed EP rank 0 is missing its route view".into(),
            )
        })?;
        let contract = root_view.execution_contract().ok_or_else(|| {
            DispatchError::Hip(
                "run_layer_program_ep: root-routed EP root has no execution contract".into(),
            )
        })?;
        let contribution_count = root_view.k.checked_mul(root_view.hidden).ok_or_else(|| {
            DispatchError::Hip(
                "run_layer_program_ep: root-routed slot contribution count overflow".into(),
            )
        })?;
        let schedule = RootRoutedEpSchedule::derive(
            contract,
            n,
            root_view.layer,
            root_view.k,
            need_bytes,
            residual_dim,
            contribution_count,
            root_view.hidden,
            RootRoutedEpReduction::Decode,
        )?;
        if root_view.contract_id() != Some(schedule.contract_id()) {
            return Err(DispatchError::Hip(
                "run_layer_program_ep: root route view contract identity disagrees with execution contract".into(),
            ));
        }
        if root_view.experts.local_rank() != 0 {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank 0 binding reports local_rank {}",
                root_view.experts.local_rank()
            )));
        }
        if root_view.experts.rank_count() != n {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP root rank_count {} != mesh {n}",
                root_view.experts.rank_count()
            )));
        }
        if root_view.k == 0 {
            return Err(DispatchError::Hip(
                "run_layer_program_ep: root-routed EP requires k > 0".into(),
            ));
        }
        if root_view.hidden != residual_dim {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP root hidden {} != residual_dim {residual_dim}",
                root_view.hidden
            )));
        }
        (
            schedule,
            schedule.contract_id(),
            root_view.layer,
            root_view.hidden,
            root_view.k,
            root_view.n_exp,
            contribution_count,
        )
    };

    for r in 1..n {
        let view = bindings[r].ep_moe_route_view().ok_or_else(|| {
            DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank {r} is missing its route view"
            ))
        })?;
        if view.contract_id() != Some(root_contract_id)
            || view.layer != root_layer
            || view.hidden != root_hidden
            || view.k != root_k
            || view.n_exp != root_n_exp
        {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank {r} disagrees with the root plan/dimensions"
            )));
        }
        if view.experts.local_rank() != r {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank {r} binding reports local_rank {}",
                view.experts.local_rank()
            )));
        }
        if view.experts.rank_count() != n {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank {r} rank_count {} != mesh {n}",
                view.experts.rank_count()
            )));
        }
    }

    let mut context = DecodeRootRoutedContext { bindings, op };
    execute_root_routed_ep(
        gpus,
        &mut context,
        schedule,
        RootRoutedEpOperands {
            partials,
            partial_bytes: need_bytes,
            reduce_count: residual_dim,
            route_count: root_k,
            contribution_count,
            contribution_chunk: root_hidden,
        },
        peer_lease,
        |ctx, gpu, partial| {
            let dispatch_ctx = DispatchCtx::new(gpu);
            ctx.bindings[0].ep_preflight_moe_root(gpu, &dispatch_ctx, ctx.op, partial)
        },
        |ctx, rank, gpu, partial, proof| {
            let dispatch_ctx = DispatchCtx::new(gpu);
            ctx.bindings[rank].ep_preflight_moe_contrib(gpu, &dispatch_ctx, ctx.op, proof, partial)
        },
        |ctx, gpu, partial, _admission| {
            let dispatch_ctx = DispatchCtx::new(gpu);
            ctx.bindings[0].ep_run_moe_root(gpu, &dispatch_ctx, ctx.op, partial)
        },
        |ctx, rank| {
            let view = ctx.bindings[rank].ep_moe_route_view().ok_or_else(|| {
                DispatchError::Hip(format!(
                    "run_layer_program_ep: root-routed EP rank {rank} route view unavailable"
                ))
            })?;
            Ok(EpRouteBuffers {
                ids: &view.topk_ids.buf,
                weights: &view.topk_weights.buf,
                slot_outputs: &view.slot_outputs.buf,
            })
        },
        |ctx, rank, gpu, proof, partial| {
            let dispatch_ctx = DispatchCtx::new(gpu);
            ctx.bindings[rank].ep_run_moe_contrib(gpu, &dispatch_ctx, ctx.op, proof, partial)
        },
        |ctx, gpu, partial| {
            let dispatch_ctx = DispatchCtx::new(gpu);
            ctx.bindings[0].ep_finish_moe_slot_order(gpu, &dispatch_ctx, ctx.op, partial)
        },
        |_ctx, _rank, _gpu, _partial| Ok(()),
        |ctx, rank, gpu, partial| ctx.bindings[rank].ep_add_into_residual(gpu, partial),
    )
}
