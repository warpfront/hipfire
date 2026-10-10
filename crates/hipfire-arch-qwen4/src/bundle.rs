// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 model ownership and teardown boundary.
//!
//! A published bundle owns every model resource: the bounded SSD PLE reader,
//! mutable GPU state, finalized resident weights, and the attached canonical
//! load transaction/census (including external PLE descriptors).  No loader
//! local may outlive publication as a second owner.

use crate::config::Qwen4Config;
use crate::gpu_forward::{
    qwen4_forward_device_bytes, qwen4_prefill_chunk_requested, qwen4_prefill_chunk_rungs,
    qwen4_spec_logit_rows, Qwen4GpuForward, Qwen4OutputRows, QWEN4_FORWARD_HEADROOM_BYTES,
};
use crate::kv_backend::Qwen4KvBackend;
use crate::mtp_gpu::{MtpAppendScratch, MtpGpuStateSnapshot, MtpStep, Qwen4MtpGpu};
use crate::ple::PleHashMetadata;
use crate::state::Qwen4StateFormat;
use crate::state::{Qwen4State, Qwen4StateSnapshot, StateError};
use crate::weights::{
    ple_valid_rows_for_shard, Qwen4Manifest, Qwen4Placement, Qwen4Weights, WeightError,
    PLE_ROW_WIDTH, PLE_SHARD_COUNT, PLE_SHARD_ROWS,
};
use hipfire_runtime::external_rows::{RowEncoding, RowStore, RowStoreError};
use hipfire_runtime::model_source::{SourceFormat, SourceRangeDescriptor};
use hipfire_runtime::sampler::{
    apply_logit_policy_candidates_cpu, apply_logit_policy_cpu, PenaltyTable, SamplerConfig,
};
use hipfire_runtime::session_cache::{
    stride_boundaries, SessionCache, SessionDriver, SessionRoute, SessionState, SnapshotParts,
    StateLayout,
};
use hipfire_runtime::spec_sampling::{SampleSpec, SparseDist};
use hipfire_runtime::weight_manifest::{WeightEntry, WeightResidency};
use hipfire_runtime::weight_store::{WeightLoadTransaction, WeightStoreError};
use rdna_compute::{Gpu, GpuTensor};
use rdna_compute::sampling::PenaltyTableStage;
use std::fmt;
use std::time::Duration;

const PLE_RESET_TIMEOUT: Duration = Duration::from_secs(5);

/// Architecture-private owner for the fulfilled manifest census.
///
/// Assembly takes every resident handle into [`Qwen4Weights`], so rollback at
/// unload releases no duplicate device allocations.  The transaction still
/// owns the immutable projections, aliases, and external PLE descriptors and
/// is drained only after the reader, mutable state, and resident weights.
pub(crate) struct AttachedWeightStore {
    transaction: WeightLoadTransaction,
}

impl AttachedWeightStore {
    fn new(transaction: WeightLoadTransaction) -> Self {
        Self { transaction }
    }

    fn drain(self, gpu: &mut Gpu) -> hip_bridge::HipResult<()> {
        self.transaction.rollback(gpu)
    }

    fn external_descriptor(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<&SourceRangeDescriptor> {
        self.transaction.external_descriptor(name, layer, device)
    }

    pub(crate) fn external_rows_len(&self) -> usize {
        self.transaction.external_rows_len()
    }

    pub(crate) fn inventory_len(&self) -> usize {
        self.transaction.inventory_len()
    }

    pub(crate) fn origin(&self) -> Option<hipfire_runtime::weight_store::WeightOrigin> {
        self.transaction.origin()
    }
}

/// Where every owner of `route` sits when the live state can continue in
/// place: the target position, for native MTP only when the head sits there
/// too (a retired head lags, a missing one cannot continue), and never while
/// a verify row capture is armed. Plain data, so it runs without a GPU.
fn live_position(
    route: SessionRoute,
    target_position: usize,
    head_position: Option<usize>,
    row_capture_armed: bool,
) -> Option<usize> {
    (!row_capture_armed
        && (route == SessionRoute::Ar || head_position == Some(target_position)))
    .then_some(target_position)
}

/// Environment flag that adds message-end snapshot boundaries (P1b), read once
/// when the bundle is assembled. Off by default.
const TURN_SNAPSHOTS_ENV: &str = "HIPFIRE_QWEN4_TURN_SNAPSHOTS";

/// Published Qwen4 architecture owner.
pub struct Qwen4Bundle {
    pub config: Qwen4Config,
    pub weights: Qwen4Weights,
    pub state: Qwen4State,
    /// Bounded model-owned PLE reader/cache.  It must quiesce before source
    /// descriptors and the attached transaction are dropped.
    pub(crate) ple_rows: RowStore,
    pub(crate) ple_metadata: PleHashMetadata,
    /// Canonical load census and external descriptors.  This is deliberately
    /// not left in the loader or carrier after publication.
    pub(crate) weight_store: AttachedWeightStore,
    /// Reusable ordinary-HIP execution resources.  This remains attached to
    /// the published bundle so unload owns the scratch, expert pointer tables,
    /// and all per-layer dispatch state exactly once.
    pub(crate) execution: Option<Qwen4GpuForward>,
    /// Reusable native MTP execution resources, attached only when the
    /// admitted artifact carries the validated one-layer MTP head.
    pub(crate) mtp: Option<Qwen4MtpGpu>,
    /// Fixed target-side output buffers for the arch-generic speculative seam.
    /// These are allocated with the ordinary forward owner and reused by every
    /// verify/advance call.
    pub(crate) spec_logits: Option<GpuTensor>,
    pub(crate) spec_top1: Option<GpuTensor>,
    pub(crate) spec_hidden: Option<GpuTensor>,
    pub(crate) spec_host_top1: Vec<u8>,
    /// Staging of the GPU penalty prepass (`apply_penalty_table`), allocated
    /// by the first penalized row and kept until unload.
    penalty_stage: Option<PenaltyTableStage>,
    /// Prefill snapshot cache shared by every session plus this
    /// conversation's committed live state (`attach_session_cache`); `None` =
    /// off. Message-end snapshots of a live continuation live in their own
    /// `/turns` scope (`HIPFIRE_QWEN4_TURN_SNAPSHOTS`).
    session: Option<SessionDriver>,
    /// `HIPFIRE_QWEN4_TURN_SNAPSHOTS` (read once at assembly): also snapshot
    /// at message ends. Active only with `turn_end_token`.
    turn_snapshots: bool,
    /// The tokenizer's `<|im_end|>` id (`set_turn_end_token`).
    turn_end_token: Option<u32>,
    /// Request-state storage, part of every session snapshot's scope.
    state_format: Qwen4StateFormat,
    /// Exact AR lane store of the Flash-Next batch route (`stage_lanes`);
    /// `None` = not staged. Lane steps never touch `state` or `mtp`.
    pub(crate) lanes: Option<crate::lane::Qwen4LaneStore>,
}

/// `HIPFIRE_QWEN4_PENALTY_PREPASS=0` keeps a penalized request's repeat /
/// presence / frequency penalties on the host (download the logit row, then
/// `apply_logit_policy_cpu`) instead of the bit-identical GPU prepass
/// ([`Qwen4Bundle::apply_penalty_table`]) on the AR greedy row and the
/// sampled-MTP target rows. Diagnostic / A-B only; read once per request.
pub fn penalty_prepass_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_QWEN4_PENALTY_PREPASS").as_deref() != Ok("0")
}

/// Device bytes of one session-cache snapshot holding `p` rows per token
/// stream (each part rounded to the cache's 256-byte packing): GDN recurrent
/// and conv, `p` QSA rows plus selections, PLE conv and hyper feedback, and
/// with native MTP the head's rows, selection and wide hidden. A delta
/// snapshot at a chunk boundary over its parent holds `p` = one chunk.
pub fn session_snapshot_bytes(
    config: &Qwen4Config,
    format: Qwen4StateFormat,
    native_mtp: bool,
    p: usize,
) -> Option<u64> {
    let part = |bytes: usize| bytes.checked_next_multiple_of(256);
    let gdn_recurrent = format.gdn.state_units(
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    );
    let conv_channels = (2 * config.linear_num_key_heads)
        .checked_mul(config.linear_key_head_dim)?
        .checked_add(
            config
                .linear_num_value_heads
                .checked_mul(config.linear_value_head_dim)?,
        )?;
    let gdn = part(gdn_recurrent.checked_mul(format.gdn.dtype().size())?)?.checked_add(part(
        conv_channels
            .checked_mul(config.linear_conv_kernel_dim.saturating_sub(1))?
            .checked_mul(4)?,
    )?)?;
    let compress = config.indexer_compress_ratio;
    let raw_width = config
        .indexer_kv_heads
        .checked_mul(config.indexer_head_dim)?;
    let full_width = config.num_key_value_heads.checked_mul(config.head_dim)?;
    let kv_row = format
        .qsa
        .kv_row_bytes(config.num_key_value_heads, config.head_dim);
    let raw_row = raw_width.checked_mul(format.qsa.index_dtype().size())?;
    let selected = part(config.qsa_selected_capacity().checked_mul(4)?)?;
    let pooled_rows = p.div_ceil(compress);
    let qsa = 2 * part(p.checked_mul(kv_row)?)?
        + part(p.checked_mul(raw_row)?)?
        + part(pooled_rows.checked_mul(raw_row)?)?
        + selected;
    let ple = config
        .ple_conv_history_rows()
        .checked_mul(config.ple_embed_dim)?
        .checked_mul(config.hc_count)?
        .checked_mul(4)?;
    let feedback = config
        .hc_count
        .checked_mul(config.hidden_size)?
        .checked_mul(4)?;
    let mut total = gdn
        .checked_mul(config.n_linear_layers())?
        .checked_add(qsa.checked_mul(config.n_full_layers())?)?
        .checked_add(part(ple)? + part(feedback)?)?;
    if native_mtp {
        // The head keeps F32 rows.
        total = total
            .checked_add(2 * part(p.checked_mul(full_width * 4)?)?)?
            .checked_add(part(p.checked_mul(raw_width * 4)?)?)?
            .checked_add(part(pooled_rows.checked_mul(raw_width * 4)?)?)?
            .checked_add(selected + part(4)? + part(feedback)?)?;
    }
    u64::try_from(total).ok()
}

impl SessionState for Qwen4Bundle {
    fn snapshot_scope(&self, route: SessionRoute) -> Option<String> {
        let chunk = self.spec_chunk_rows()?;
        if route == SessionRoute::Mtp && self.mtp.is_none() {
            return None;
        }
        // Message-end snapshots split prefill off the plain chunk schedule, so
        // they never share a domain with chunk-only ones.
        let turns = if self.turn_snapshots_active() {
            "/turns"
        } else {
            ""
        };
        Some(format!(
            "qwen4/{route:?}/chunk{chunk}/{:?}{turns}",
            self.state_format
        ))
    }

    fn snapshot_boundaries(&self, prompt: &[u32], after: usize, up_to: usize) -> Vec<usize> {
        let Some(chunk) = self.spec_chunk_rows() else {
            return Vec::new();
        };
        stride_boundaries(chunk, prompt, self.turn_end_token_active(), after, up_to)
    }

    fn snapshot_parts(
        &mut self,
        route: SessionRoute,
        position: usize,
    ) -> Result<SnapshotParts<'_>, String> {
        self.quiesce_ple().map_err(|e| e.to_string())?;
        let mtp = match route {
            SessionRoute::Ar => None,
            SessionRoute::Mtp => Some(
                self.mtp
                    .as_ref()
                    .ok_or("Qwen4 MTP resources are not attached")?,
            ),
        };
        if self.state.position != position || mtp.is_some_and(|mtp| mtp.position() != position) {
            return Err(format!("qwen4 session capture: owners not at {position}"));
        }
        let (mut meta, mut layout) = self.state.session_parts().map_err(|e| e.to_string())?;
        if let Some(mtp) = mtp {
            let (mtp_meta, mtp_layout) = mtp.session_parts().map_err(|e| e.to_string())?;
            meta.extend(mtp_meta);
            layout.fixed.extend(mtp_layout.fixed);
            layout.rows.extend(mtp_layout.rows);
        }
        Ok(SnapshotParts { meta, layout })
    }

    fn restore_parts(
        &mut self,
        gpu: &mut Gpu,
        route: SessionRoute,
        meta: &[u8],
    ) -> Result<StateLayout<'_>, String> {
        self.invalidate_ple_epoch().map_err(|e| e.to_string())?;
        let (target_meta, mtp_meta) = meta
            .split_at_checked(self.state.session_meta_bytes())
            .ok_or("qwen4 session snapshot shape mismatch")?;
        let mut layout = self
            .state
            .prepare_session_restore(gpu, target_meta)
            .map_err(|e| e.to_string())?;
        if route == SessionRoute::Mtp {
            let mtp = self
                .mtp
                .as_mut()
                .ok_or("Qwen4 MTP resources are not attached")?;
            let mtp_layout = mtp
                .prepare_session_restore(gpu, mtp_meta)
                .map_err(|e| e.to_string())?;
            layout.fixed.extend(mtp_layout.fixed);
            layout.rows.extend(mtp_layout.rows);
        }
        Ok(layout)
    }

    fn finish_restore(
        &mut self,
        gpu: &mut Gpu,
        route: SessionRoute,
        meta: &[u8],
    ) -> Result<(), String> {
        let (target_meta, mtp_meta) = meta
            .split_at_checked(self.state.session_meta_bytes())
            .ok_or("qwen4 session snapshot shape mismatch")?;
        self.state
            .finish_session_restore(target_meta)
            .map_err(|e| e.to_string())?;
        match (route, self.mtp.as_mut()) {
            (SessionRoute::Mtp, Some(mtp)) => mtp.finish_session_restore(mtp_meta),
            // An AR prefill leaves the head cold, as a reset would.
            (SessionRoute::Ar, Some(mtp)) => mtp.reset(gpu),
            (_, None) => Ok(()),
        }
        .map_err(|e| e.to_string())
    }

    fn live_position(&self, route: SessionRoute) -> Option<usize> {
        live_position(
            route,
            self.state.position,
            self.mtp.as_ref().map(Qwen4MtpGpu::position),
            self.state.row_capture_armed,
        )
    }

    fn growth_reserve_bytes(&self, gpu: &Gpu, through: usize) -> u64 {
        let (context, gather) = self.context_growth_bytes(gpu, through);
        context.saturating_add(gather)
    }

    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        Qwen4Bundle::reset(self, gpu).map_err(|e| e.to_string())
    }
}

impl Qwen4Bundle {
    /// Assemble a complete Single bundle using metadata parsed from the
    /// artifact's canonical `qwen4_ple` object. `state_format` is the request
    /// state's storage ([`crate::resolve_state_format`]); `backend` the QSA
    /// context storage load admission resolved, shared by target and MTP.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        config: Qwen4Config,
        transaction: WeightLoadTransaction,
        placements: &[Qwen4Placement],
        gpu: &mut Gpu,
        max_seq_len: usize,
        metadata: PleHashMetadata,
        state_format: Qwen4StateFormat,
        backend: Qwen4KvBackend,
    ) -> Result<Self, BundleError> {
        Self::assemble_with_metadata(
            config,
            transaction,
            placements,
            gpu,
            max_seq_len,
            metadata,
            state_format,
            backend,
        )
    }

    /// Assemble with validated metadata read from the artifact's exact I64
    /// arrays.  The transaction is consumed so no load-side owner can remain
    /// live after this method publishes the bundle.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble_with_metadata(
        config: Qwen4Config,
        mut transaction: WeightLoadTransaction,
        placements: &[Qwen4Placement],
        gpu: &mut Gpu,
        max_seq_len: usize,
        metadata: PleHashMetadata,
        state_format: Qwen4StateFormat,
        backend: Qwen4KvBackend,
    ) -> Result<Self, BundleError> {
        let weights = match Qwen4Weights::assemble(&mut transaction, &config, placements) {
            Ok(weights) => weights,
            Err(error) => {
                return Err(cleanup_transaction(
                    BundleError::Weights(error),
                    transaction.rollback(gpu),
                ));
            }
        };
        let descriptors = match ple_descriptors(&transaction, &weights.manifest, &metadata) {
            Ok(descriptors) => descriptors,
            Err(error) => {
                let weight_result = weights.free_gpu(gpu);
                let cleanup = transaction.rollback(gpu);
                return Err(cleanup_bundle_failure(error, weight_result, cleanup));
            }
        };
        // One prefill chunk prefetches `rows * PLE_HEAD_COUNT` n-gram rows in
        // a single row-store request, so staging holds the requested chunk.
        let staging_rows =
            qwen4_prefill_chunk_requested(&gpu.arch, max_seq_len) * crate::ple::PLE_HEAD_COUNT;
        let ple_rows = match RowStore::with_staging_rows(
            "qwen4-ple-reader",
            descriptors,
            metadata.valid_rows(),
            staging_rows,
            // `HIPFIRE_QWEN4_PLE_WIDE_READERS=0` keeps 16 concurrent reads
            // for every PLE request, including a whole prefill chunk.
            hipfire_config::developer_bool("HIPFIRE_QWEN4_PLE_WIDE_READERS", true),
        ) {
            Ok(rows) => rows,
            Err(error) => {
                let weight_result = weights.free_gpu(gpu);
                let cleanup = transaction.rollback(gpu);
                return Err(cleanup_bundle_failure(
                    BundleError::PleRows(error),
                    weight_result,
                    cleanup,
                ));
            }
        };
        let mut state =
            match Qwen4State::new_with_backend(gpu, &config, max_seq_len, state_format, backend) {
                Ok(state) => state,
                Err(error) => {
                    // `unload` consumes the reader and joins its worker even on a
                    // quiesce error, so source descriptors cannot outlive failure.
                    let _ = ple_rows.unload();
                    let weight_result = weights.free_gpu(gpu);
                    let cleanup = transaction.rollback(gpu);
                    return Err(cleanup_bundle_failure(
                        BundleError::State(error),
                        weight_result,
                        cleanup,
                    ));
                }
            };
        state.bind_transaction_generation(transaction.inventory_len() as u64);
        Ok(Self {
            config,
            weights,
            state,
            ple_rows,
            ple_metadata: metadata,
            weight_store: AttachedWeightStore::new(transaction),
            execution: None,
            mtp: None,
            spec_logits: None,
            spec_top1: None,
            spec_hidden: None,
            spec_host_top1: Vec::new(),
            penalty_stage: None,
            session: None,
            turn_snapshots: hipfire_config::developer_bool(TURN_SNAPSHOTS_ENV, false),
            turn_end_token: None,
            state_format,
            lanes: None,
        })
    }

    pub fn manifest(&self) -> &Qwen4Manifest {
        &self.weights.manifest
    }

    pub fn external_descriptor(
        &self,
        placement: &Qwen4Placement,
    ) -> Option<&SourceRangeDescriptor> {
        self.weight_store
            .external_descriptor(&placement.name, placement.layer, placement.device)
    }

    /// Return all numerically ordered PLE shard descriptors from the
    /// attached canonical transaction census.
    ///
    /// This strict accessor is useful to admission and diagnostics callers;
    /// the bundle's own reader already owns the same sealed descriptors.
    pub fn ple_descriptors(&self) -> Result<Vec<SourceRangeDescriptor>, BundleError> {
        ple_descriptors(
            &self.weight_store.transaction,
            self.manifest(),
            &self.ple_metadata,
        )
    }

    pub fn ple_rows(&self) -> &RowStore {
        &self.ple_rows
    }

    pub fn ple_rows_mut(&mut self) -> &mut RowStore {
        &mut self.ple_rows
    }

    pub fn attached_origin(&self) -> Option<hipfire_runtime::weight_store::WeightOrigin> {
        self.weight_store.origin()
    }

    pub fn attached_inventory_len(&self) -> usize {
        self.weight_store.inventory_len()
    }

    pub fn attached_external_rows_len(&self) -> usize {
        self.weight_store.external_rows_len()
    }

    /// Attach reusable ordinary-HIP execution resources after the manifest
    /// transaction and model state have assembled successfully.
    pub fn attach_forward(&mut self, gpu: &mut Gpu, max_chunk: usize) -> Result<(), BundleError> {
        if self.execution.is_some() {
            return Err(BundleError::Forward(
                "Qwen4 forward resources are already attached".to_string(),
            ));
        }
        if max_chunk == 0 {
            return Err(BundleError::Forward(
                "Qwen4 forward chunk capacity is zero".to_string(),
            ));
        }
        // A chunk's PLE prefetch is one row-store request: the staging sized
        // at assembly bounds it.
        let ple_rows_cap = self.ple_rows.max_rows_per_prefetch() / crate::ple::PLE_HEAD_COUNT;
        let requested = qwen4_prefill_chunk_requested(&gpu.arch, max_chunk).min(ple_rows_cap);
        if requested == 0 {
            return Err(BundleError::Forward(
                "Qwen4 PLE row store cannot stage one chunk row".to_string(),
            ));
        }
        // The gathered QSA prefill attention (HIPFIRE_QWEN4_QSA_WMMA_GATHER)
        // converts the cache rows it reads into its own workspace. Reserve
        // its address for the whole context now, before any capture or
        // record, so a later longer prefill never moves it. With VMM QSA
        // state only the reservation is made and each forward maps the
        // prefix it reads (`Qwen4GpuForward`); legacy state commits all of
        // it here, as before. A no-op when the route is off for this arch
        // and state format.
        if let Some(qsa) = self.state.qsa.first() {
            let heads = self.config.num_key_value_heads;
            let reserved = match self.state.qsa_backend() {
                Qwen4KvBackend::Legacy => {
                    rdna_compute::tensor_ops::reserve_qsa_gathered_wmma_scratch(
                        gpu,
                        qsa.format,
                        heads,
                        qsa.full_capacity,
                    )
                }
                Qwen4KvBackend::Vmm => {
                    rdna_compute::tensor_ops::reserve_qsa_gathered_wmma_workspace(
                        gpu,
                        qsa.format,
                        heads,
                        qsa.full_capacity,
                    )
                }
            }
            .map_err(BundleError::Hip)?;
            if reserved > 0 {
                eprintln!(
                    "  qwen4 QSA gather workspace: {} MiB reserved for {} context tokens, {} MiB committed",
                    reserved >> 20,
                    qsa.full_capacity,
                    gpu.qsa_gather_scratch_bytes() >> 20
                );
            }
        }
        self.weights
            .requant_from_env(gpu)
            .map_err(BundleError::Forward)?;
        // The largest rung whose chunk-sized resources fit the free device
        // memory beside the kernels' lazily sized workspaces; the smallest
        // rung is attempted regardless and fails at allocation if it must.
        let (free, _) = gpu.device_mem_info().map_err(BundleError::Hip)?;
        let max_chunk = qwen4_prefill_chunk_rungs(requested)
            .find(|&rows| {
                qwen4_forward_device_bytes(&self.config, rows)
                    .and_then(|bytes| bytes.checked_add(QWEN4_FORWARD_HEADROOM_BYTES))
                    .is_some_and(|bytes| bytes <= free as u64)
            })
            .unwrap_or_else(|| requested.min(1536));
        eprintln!("  qwen4 prefill chunk: {max_chunk} rows (requested {requested})");
        let forward = Qwen4GpuForward::new(gpu, self, max_chunk)
            .map_err(|error| BundleError::Forward(error.to_string()))?;
        let spec_rows = qwen4_spec_logit_rows(max_chunk);
        let logits_len = spec_rows
            .checked_mul(self.config.vocab_size)
            .ok_or_else(|| BundleError::Forward("spec logit scratch overflow".to_string()))?;
        let spec_logits = match gpu.zeros(&[logits_len], rdna_compute::DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = forward.free_gpu(gpu);
                return Err(BundleError::Hip(error));
            }
        };
        let top1_len = match spec_rows.checked_mul(std::mem::size_of::<i32>()) {
            Some(len) => len,
            None => {
                let _ = gpu.free_tensor(spec_logits);
                let _ = forward.free_gpu(gpu);
                return Err(BundleError::Forward(
                    "spec argmax scratch overflow".to_string(),
                ));
            }
        };
        let spec_top1 = match gpu.zeros(&[top1_len], rdna_compute::DType::Raw) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(spec_logits);
                let _ = forward.free_gpu(gpu);
                return Err(BundleError::Hip(error));
            }
        };
        self.execution = Some(forward);
        self.spec_logits = Some(spec_logits);
        self.spec_top1 = Some(spec_top1);
        self.spec_host_top1 = vec![0; top1_len];
        Ok(())
    }
    /// Attach the reusable native MTP head and its bounded GPU state.
    pub fn attach_mtp(&mut self, gpu: &mut Gpu, max_seq: usize) -> Result<(), BundleError> {
        if self.mtp.is_some() {
            return Err(BundleError::Forward(
                "Qwen4 MTP resources are already attached".to_string(),
            ));
        }
        let mtp = Qwen4MtpGpu::new_with_backend(
            gpu,
            &self.weights,
            &self.config,
            max_seq,
            self.state.qsa_backend(),
        )
        .map_err(|error| BundleError::Forward(error.to_string()))?;
        self.mtp = Some(mtp);
        Ok(())
    }

    pub(crate) fn ensure_spec_hidden(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
    ) -> Result<(), BundleError> {
        if rows == 0 {
            return Err(BundleError::Forward(
                "Qwen4 spec hidden capacity is zero".to_string(),
            ));
        }
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| BundleError::Forward("spec hidden width overflow".to_string()))?;
        let elements = rows
            .checked_mul(width)
            .ok_or_else(|| BundleError::Forward("spec hidden capacity overflow".to_string()))?;
        if let Some(hidden) = self.spec_hidden.as_ref() {
            if hidden.dtype != rdna_compute::DType::F32 || hidden.numel() < elements {
                return Err(BundleError::Forward(
                    "Qwen4 spec hidden capacity is too small".to_string(),
                ));
            }
            return Ok(());
        }
        self.spec_hidden = Some(
            gpu.zeros(&[elements], rdna_compute::DType::F32)
                .map_err(BundleError::Hip)?,
        );
        Ok(())
    }

    /// Install (or clear) the QSA parity observer on the attached forward.
    #[cfg(feature = "reference-parity")]
    pub fn set_qsa_tap(
        &mut self,
        tap: Option<crate::gpu_forward::Qwen4QsaTap>,
    ) -> Result<(), BundleError> {
        let forward = self.execution.as_mut().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        forward.qsa_tap = tap;
        Ok(())
    }

    /// Install (or clear) the QSA projection hook of this thread's forward:
    /// `(gpu, qsa_slot, op)` right after each QSA step's projections, i.e. on
    /// the raw projected index row / query + gate / K / V rows before the
    /// prologue (the step's cache and pool are still pre-step). Same contract
    /// as [`Self::set_qsa_tap`]: a forward with a hook never records or replays a
    /// retained body, and with none installed nothing runs or allocates.
    #[cfg(feature = "reference-parity")]
    pub fn set_qsa_projection_hook(
        &mut self,
        hook: Option<hipfire_dispatch::pipeline::QsaProjectionHook>,
    ) -> Result<(), BundleError> {
        if self.execution.is_none() {
            return Err(BundleError::Forward(
                "Qwen4 forward resources are not attached".to_string(),
            ));
        }
        hipfire_dispatch::pipeline::set_qsa_projection_hook(hook);
        Ok(())
    }

    /// Rows the attached forward can process in one chunked call.  The MTP
    /// prefill uses this to batch a whole prompt chunk through the shared
    /// forward instead of one single-row forward per prompt token.
    pub fn spec_chunk_rows(&self) -> Option<usize> {
        self.execution
            .as_ref()
            .map(|forward| forward.scratch.max_chunk)
    }

    /// The `<|im_end|>` token message-end snapshots split after, when active.
    fn turn_end_token_active(&self) -> Option<u32> {
        self.turn_end_token.filter(|_| self.turn_snapshots)
    }

    /// Whether prefill also snapshots at message ends (`HIPFIRE_QWEN4_TURN_SNAPSHOTS`
    /// on and the `<|im_end|>` token known).
    pub fn turn_snapshots_active(&self) -> bool {
        self.turn_end_token_active().is_some()
    }

    /// Set the tokenizer's `<|im_end|>` id: with `HIPFIRE_QWEN4_TURN_SNAPSHOTS`
    /// on, every message end becomes a snapshot boundary (cold, restored and
    /// live prefills), in the session-cache scope `.../turns`. `None` keeps
    /// chunk multiples only.
    pub fn set_turn_end_token(&mut self, token: Option<u32>) {
        self.turn_end_token = token;
    }

    /// Force message-end snapshots on or off regardless of the environment
    /// (hardware tests; production reads `HIPFIRE_QWEN4_TURN_SNAPSHOTS` once
    /// at assembly). Changing it changes the snapshot scope, so call it before
    /// the first prefill that fills the cache.
    pub fn set_turn_snapshots(&mut self, on: bool) {
        self.turn_snapshots = on;
    }

    pub(crate) fn spec_forward_rows(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        capture_hidden: bool,
    ) -> Result<Vec<u32>, BundleError> {
        self.spec_forward_rows_with_output(gpu, tokens, capture_hidden, Qwen4OutputRows::All)
    }

    /// Final-row argmax of one forward over `tokens`: prompt fills and
    /// advances, which read no other row's logits.
    pub(crate) fn spec_prefill_rows(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        capture_hidden: bool,
    ) -> Result<u32, BundleError> {
        self.spec_forward_rows_with_output(gpu, tokens, capture_hidden, Qwen4OutputRows::Final)?
            .into_iter()
            .next()
            .ok_or_else(|| BundleError::Forward("Qwen4 prefill produced no argmax".into()))
    }

    /// [`Self::spec_prefill_rows`] for a prompt chunk whose pick nothing
    /// reads (every chunk but the prompt's last): the trunk commits state and
    /// still captures the full wide hidden rows for the head, but the final
    /// hyper, LM head, argmax and readback are skipped. Tiles and trunk route
    /// are `spec_prefill_rows`'s, so the committed state is byte-identical.
    pub(crate) fn spec_prefill_rows_silent(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
    ) -> Result<(), BundleError> {
        if tokens.is_empty() {
            return Err(BundleError::Forward(
                "Qwen4 spec forward cannot process an empty block".to_string(),
            ));
        }
        let max_chunk = self
            .execution
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 forward resources are not attached".to_string())
            })?
            .scratch
            .max_chunk;
        if tokens.len() > max_chunk {
            return Err(BundleError::Forward(format!(
                "Qwen4 spec block length {} exceeds capacity {max_chunk}",
                tokens.len()
            )));
        }
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| BundleError::Forward("spec hidden width overflow".to_string()))?;
        let hidden_len = tokens
            .len()
            .checked_mul(width)
            .ok_or_else(|| BundleError::Forward("spec hidden row overflow".to_string()))?;
        let hidden = self
            .spec_hidden
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 spec hidden is not allocated".to_string()))?
            .sub_offset(0, hidden_len);
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_chunk_silent(self, gpu, tokens, Some(&hidden))
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result
    }

    fn spec_forward_rows_with_output(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        capture_hidden: bool,
        output_rows: Qwen4OutputRows,
    ) -> Result<Vec<u32>, BundleError> {
        if tokens.is_empty() {
            return Err(BundleError::Forward(
                "Qwen4 spec forward cannot process an empty block".to_string(),
            ));
        }
        let max_chunk = self
            .execution
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 forward resources are not attached".to_string())
            })?
            .scratch
            .max_chunk;
        if tokens.len() > max_chunk {
            return Err(BundleError::Forward(format!(
                "Qwen4 spec block length {} exceeds capacity {max_chunk}",
                tokens.len()
            )));
        }
        let vocab = self.config.vocab_size;
        let output_count = output_rows.count(tokens.len());
        let spec_rows = qwen4_spec_logit_rows(max_chunk);
        if output_count > spec_rows {
            return Err(BundleError::Forward(format!(
                "Qwen4 spec block of {output_count} output rows exceeds the {spec_rows}-row verify capacity"
            )));
        }
        let logits_len = output_count
            .checked_mul(vocab)
            .ok_or_else(|| BundleError::Forward("Qwen4 spec logits overflow".to_string()))?;
        let logits = self
            .spec_logits
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 spec logits are not attached".to_string()))?
            .sub_offset(0, logits_len);
        let top1_len = output_count
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| BundleError::Forward("Qwen4 spec argmax overflow".to_string()))?;
        let top1 = self
            .spec_top1
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 spec argmax is not attached".to_string()))?
            .sub_offset(0, top1_len);
        let hidden = if capture_hidden {
            let width = self
                .config
                .hc_count
                .checked_mul(self.config.hidden_size)
                .ok_or_else(|| BundleError::Forward("spec hidden width overflow".to_string()))?;
            let hidden_len = tokens
                .len()
                .checked_mul(width)
                .ok_or_else(|| BundleError::Forward("spec hidden row overflow".to_string()))?;
            Some(
                self.spec_hidden
                    .as_ref()
                    .ok_or_else(|| {
                        BundleError::Forward("Qwen4 spec hidden is not allocated".to_string())
                    })?
                    .sub_offset(0, hidden_len),
            )
        } else {
            None
        };
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_chunk(
                self,
                gpu,
                tokens,
                &logits,
                Some(&top1),
                hidden.as_ref(),
                output_rows,
            )
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result?;
        let bytes_len = output_count * std::mem::size_of::<i32>();
        if self.spec_host_top1.len() < bytes_len {
            return Err(BundleError::Forward(
                "Qwen4 spec host argmax capacity is too small".to_string(),
            ));
        }
        gpu.hip
            .memcpy_dtoh(&mut self.spec_host_top1[..bytes_len], &top1.buf)
            .map_err(BundleError::Hip)?;
        let mut picks = Vec::with_capacity(output_count);
        for bytes in self.spec_host_top1[..bytes_len].chunks_exact(4) {
            picks.push(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        }
        Ok(picks)
    }

    pub(crate) fn copy_spec_hidden_row_to(
        &self,
        gpu: &mut Gpu,
        row: usize,
        destination: &GpuTensor,
    ) -> Result<(), BundleError> {
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| BundleError::Forward("spec hidden width overflow".to_string()))?;
        if destination.dtype != rdna_compute::DType::F32 || destination.numel() != width {
            return Err(BundleError::Forward(
                "Qwen4 spec hidden destination shape mismatch".to_string(),
            ));
        }
        let source = self.spec_hidden.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 spec hidden is not allocated".to_string())
        })?;
        let offset = row
            .checked_mul(width)
            .ok_or_else(|| BundleError::Forward("spec hidden row offset overflow".to_string()))?;
        if offset
            .checked_add(width)
            .is_none_or(|end| end > source.numel())
        {
            return Err(BundleError::Forward(
                "Qwen4 spec hidden row is outside capture".to_string(),
            ));
        }
        let source = source.sub_offset(offset, width);
        // Stream-ordered and host-asynchronous: the copy rides the stream the
        // kernels launch on (the active stream, else the default stream), so
        // every consumer sees it in order without the host waiting for the
        // work queued before it.
        gpu.memcpy_dtod_at_ordered_async(&destination.buf, 0, &source.buf, 0, destination.byte_size())
            .map_err(BundleError::Hip)
    }

    /// Keep the first `keep` rows of the armed `tokens.len()`-row verify the
    /// active snapshot ticket brackets, without re-running them; the ticket
    /// stays active for the caller's commit or restore.
    pub(crate) fn rollback_verify_rows_retain(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
        keep: usize,
        tokens: &[u32],
    ) -> Result<(), BundleError> {
        let ple_normed = &self
            .execution
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 forward resources are not attached".to_string())
            })?
            .scratch
            .ple_normed;
        self.state
            .rollback_rows_retain(
                gpu,
                snapshot,
                keep,
                tokens.len(),
                tokens,
                ple_normed,
                self.config.ple_conv_history_rows(),
                self.config.linear_conv_kernel_dim - 1,
                self.config.indexer_compress_ratio,
                self.config.indexer_budget,
            )
            .map_err(BundleError::State)
    }

    /// Start reading the PLE rows `tokens` (the next tokens after the
    /// committed history, in order) will need, so a forward over them later
    /// finds them cached. Best effort: a failure only loses the head start.
    pub(crate) fn warm_ple_rows(&self, tokens: &[u32]) {
        let ids = self.state.ple_history.row_ids(&self.ple_metadata, tokens);
        let _ = self.ple_rows.warm(ids);
    }

    pub(crate) fn spec_capture_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
    ) -> Result<u32, BundleError> {
        self.spec_forward_rows(gpu, std::slice::from_ref(&token), true)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 spec capture returned no argmax".to_string())
            })
    }

    /// One ordinary target-only forward of `token`: a single final row, no
    /// wide-hidden capture, so the replay graph route stays eligible. Returns
    /// the greedy argmax (read into the first 4 bytes of `spec_host_top1`, no
    /// allocation); the row's logits stay in row 0 of `spec_logits` for a
    /// sampled follow-up. Clears any stale PLE lookahead first.
    pub(crate) fn spec_ar_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
    ) -> Result<u32, BundleError> {
        let vocab = self.config.vocab_size;
        let logits = self
            .spec_logits
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 spec logits are not attached".to_string()))?
            .sub_offset(0, vocab);
        let top1_bytes = std::mem::size_of::<i32>();
        let top1 = self
            .spec_top1
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 spec argmax is not attached".to_string()))?
            .sub_offset(0, top1_bytes);
        if self.spec_host_top1.len() < top1_bytes {
            return Err(BundleError::Forward(
                "Qwen4 spec host argmax capacity is too small".to_string(),
            ));
        }
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        forward.set_ple_lookahead(&[]);
        let result = forward
            .forward_token(self, gpu, token, &logits, Some(&top1))
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result?;
        gpu.hip
            .memcpy_dtoh(&mut self.spec_host_top1[..top1_bytes], &top1.buf)
            .map_err(BundleError::Hip)?;
        let bytes = &self.spec_host_top1[..top1_bytes];
        Ok(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Copy the wide HC stream row the last [`Self::spec_ar_token`] (or any
    /// ordinary one-token forward) left in the forward scratch into
    /// `destination` (F32, exactly `hc_count * hidden_size` elements). Valid
    /// only until the next forward; for calibration, not the hot path.
    pub(crate) fn copy_ar_hidden_to(
        &self,
        gpu: &mut Gpu,
        destination: &GpuTensor,
    ) -> Result<(), BundleError> {
        self.execution
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 forward resources are not attached".to_string())
            })?
            .copy_last_wide_hidden_to(gpu, &self.config, destination)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_forward_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
        fresh_qsa_selection: bool,
    ) -> Result<u32, BundleError> {
        let mtp = mapped_mtp(self.mtp.as_mut(), gpu, position)?;
        mtp.forward_token(
            gpu,
            &self.weights,
            &self.config,
            token,
            backbone_hidden,
            position,
            fresh_qsa_selection,
            MtpStep::Predict,
        )
        .map_err(|error| BundleError::Forward(error.to_string()))?
        .ok_or_else(|| {
            BundleError::Forward("MTP prediction requested but no token produced".into())
        })
    }

    /// Exact logit margin of the last MTP draft over its runner-up.
    pub(crate) fn mtp_draft_margin(&self) -> f32 {
        self.mtp
            .as_ref()
            .map_or(f32::INFINITY, |mtp| mtp.draft.margin())
    }

    /// The draft head's request-local policy (full-vocabulary hold, last
    /// margin). Not part of the MTP state snapshot: a caller that drafts and
    /// then restores the head must save and restore it separately.
    pub(crate) fn mtp_draft_request_state(
        &self,
    ) -> Result<hipfire_dispatch::pipeline::DraftHeadRequestState, BundleError> {
        self.mtp
            .as_ref()
            .map(|mtp| mtp.draft.request_state())
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })
    }

    pub(crate) fn set_mtp_draft_request_state(
        &mut self,
        state: hipfire_dispatch::pipeline::DraftHeadRequestState,
    ) -> Result<(), BundleError> {
        self.mtp
            .as_mut()
            .map(|mtp| mtp.draft.set_request_state(state))
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })
    }

    /// GPU penalty prepass: apply `table` in place to its `table.rows()`
    /// consecutive vocabulary-wide rows of `logits`, stream-ordered after the
    /// work that produced them (`Gpu::apply_penalty_table`). Each row ends
    /// bit-identical to `apply_logit_policy_cpu` over the row's history.
    pub fn apply_penalty_table(
        &mut self,
        gpu: &mut Gpu,
        logits: &GpuTensor,
        table: &PenaltyTable,
    ) -> Result<(), BundleError> {
        if table.is_noop() {
            return Ok(());
        }
        gpu.apply_penalty_table(
            &mut self.penalty_stage,
            logits,
            self.config.vocab_size,
            table.flags(),
            table.row_ends(),
            table.entries(),
        )
        .map_err(BundleError::Hip)
    }

    /// [`Self::apply_penalty_table`] on the last speculative forward's logit
    /// rows `first_row..first_row + table.rows()`.
    pub(crate) fn apply_spec_penalty_table(
        &mut self,
        gpu: &mut Gpu,
        first_row: usize,
        table: &PenaltyTable,
    ) -> Result<(), BundleError> {
        let vocab = self.config.vocab_size;
        let logits = self.spec_logits.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 spec logits are not attached".to_string())
        })?;
        let rows = table.rows();
        let view = first_row
            .checked_add(rows)
            .and_then(|end| end.checked_mul(vocab))
            .filter(|&end| end <= logits.numel())
            .map(|_| logits.sub_offset(first_row * vocab, rows * vocab))
            .ok_or_else(|| {
                BundleError::Forward(format!(
                    "Qwen4 spec logit rows {first_row}..{} are outside capacity",
                    first_row + rows
                ))
            })?;
        self.apply_penalty_table(gpu, &view, table)
    }

    /// Host copy of row `row` of the last speculative forward's logits into
    /// the reused buffer `host`.
    pub(crate) fn spec_row_logits(
        &self,
        gpu: &Gpu,
        row: usize,
        host: &mut Vec<f32>,
    ) -> Result<(), BundleError> {
        let vocab = self.config.vocab_size;
        let logits = self.spec_logits.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 spec logits are not attached".to_string())
        })?;
        let offset = row
            .checked_mul(vocab)
            .filter(|offset| offset + vocab <= logits.numel())
            .ok_or_else(|| {
                BundleError::Forward(format!("Qwen4 spec logit row {row} is outside capacity"))
            })?;
        gpu.download_f32_into(&logits.sub_offset(offset, vocab), host)
            .map_err(BundleError::Hip)
    }

    /// Row `row` of the last speculative forward's logits as `spec`'s
    /// truncated distribution, after `policy` (repeat/presence/frequency
    /// penalties and blocked tokens) is applied over `history` — the row's AR
    /// history — to the whole downloaded row, before the pool gather. `host`
    /// and `scratch` are reused buffers.
    pub(crate) fn spec_row_dist(
        &self,
        gpu: &Gpu,
        row: usize,
        spec: SampleSpec,
        history: &[u32],
        policy: &SamplerConfig,
        host: &mut Vec<f32>,
        scratch: &mut Vec<(u32, f32)>,
        out: &mut SparseDist,
    ) -> Result<(), BundleError> {
        self.spec_row_logits(gpu, row, host)?;
        apply_logit_policy_cpu(host, history, policy);
        out.build_from_logits(host, spec, scratch)
            .map_err(BundleError::Forward)
    }

    /// The last MTP prediction's draft distribution under `spec`: its 8
    /// re-scored candidates' exact logits, or the whole draft logit row when
    /// the draft head does not re-score. `policy` is applied over `history`
    /// (the AR history of the row being drafted) to those exact logits, by
    /// token id, before truncation.
    pub(crate) fn mtp_draft_dist(
        &self,
        gpu: &Gpu,
        spec: SampleSpec,
        history: &[u32],
        policy: &SamplerConfig,
        host: &mut Vec<f32>,
        scratch: &mut Vec<(u32, f32)>,
        out: &mut SparseDist,
    ) -> Result<(), BundleError> {
        let draft = &self
            .mtp
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .draft;
        let vocab = self.config.vocab_size as u32;
        match draft.rescored_candidates() {
            Some(candidates) => {
                scratch.clear();
                scratch.extend(
                    candidates
                        .iter()
                        .copied()
                        .filter(|&(token, _)| token < vocab),
                );
                // `rescored_candidates` is a fixed `[_; 8]`, so the filtered
                // count is at most 8.
                let mut ids = [0u32; 8];
                let mut values = [0f32; 8];
                let n = scratch.len();
                for (slot, &(token, logit)) in scratch.iter().enumerate() {
                    ids[slot] = token;
                    values[slot] = logit;
                }
                apply_logit_policy_candidates_cpu(&ids[..n], &mut values[..n], history, policy);
                for (entry, &logit) in scratch.iter_mut().zip(&values[..n]) {
                    entry.1 = logit;
                }
                out.build_from_candidates(scratch, spec)
            }
            None => {
                gpu.download_f32_into(draft.logits(), host)
                    .map_err(BundleError::Hip)?;
                apply_logit_policy_cpu(host, history, policy);
                out.build_from_logits(host, spec, scratch)
            }
        }
        .map_err(BundleError::Forward)
    }

    pub(crate) fn mtp_advance_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
        fresh_qsa_selection: bool,
    ) -> Result<(), BundleError> {
        mapped_mtp(self.mtp.as_mut(), gpu, position)?
            .forward_token(
                gpu,
                &self.weights,
                &self.config,
                token,
                backbone_hidden,
                position,
                fresh_qsa_selection,
                MtpStep::Advance,
            )
            .map(|_| ())
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    /// The last MTP step of a chain: append only its K/V and index-key cache
    /// rows (see [`MtpStep::Append`]); the next step must bring its own
    /// backbone hidden and a fresh selection.
    pub(crate) fn mtp_append_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
    ) -> Result<(), BundleError> {
        mapped_mtp(self.mtp.as_mut(), gpu, position)?
            .forward_token(
                gpu,
                &self.weights,
                &self.config,
                token,
                backbone_hidden,
                position,
                true,
                MtpStep::Append,
            )
            .map(|_| ())
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    /// Whether [`Self::mtp_append_rows`] can run on this GPU for this model.
    pub(crate) fn mtp_append_rows_supported(&self, gpu: &Gpu) -> bool {
        self.mtp.is_some() && Qwen4MtpGpu::append_rows_supported(gpu, &self.weights, &self.config)
    }

    /// Map the MTP head's QSA context through `last_position` (the row the
    /// last Append of a run writes) in one go: the coverage each Append's own
    /// `mapped_mtp` would grow to, so those calls then find it mapped. A VMM
    /// grow zero-fills with a blocking memset, which belongs where the device
    /// is idle (before the run's first Append) rather than between launches.
    pub(crate) fn mtp_map_through(
        &mut self,
        gpu: &mut Gpu,
        last_position: usize,
    ) -> Result<(), BundleError> {
        mapped_mtp(self.mtp.as_mut(), gpu, last_position).map(|_| ())
    }

    /// Batched prompt fill: [`MtpStep::Append`] for `tokens` at
    /// `position..position + tokens.len()`, row `i` paired with captured spec
    /// hidden row `hidden_row0 + i`; the same (token p, hidden p) rows as
    /// calling [`Self::mtp_append_token`] per row after `copy_spec_hidden_row_to`.
    ///
    /// `trunk_ids_row` is `Some(r)` when `tokens` are rows `r..` of the spec
    /// forward that just ran, with nothing run since: the head then embeds
    /// from that forward's device token ids when they hold exactly `tokens`.
    /// Otherwise it embeds from `prompt_ids`, device ids that already hold
    /// exactly `tokens` as i32 (the caller's once-per-chunk upload, see
    /// `MtpAppendScratch::upload_prompt_ids`), and uploads its own copy when
    /// that is `None` too.
    pub(crate) fn mtp_append_rows(
        &mut self,
        gpu: &mut Gpu,
        scratch: &mut MtpAppendScratch,
        tokens: &[u32],
        hidden_row0: usize,
        trunk_ids_row: Option<usize>,
        prompt_ids: Option<&GpuTensor>,
        position: usize,
    ) -> Result<(), BundleError> {
        if tokens.is_empty() {
            return Err(BundleError::Forward(
                "Qwen4 MTP batched append has no rows".to_string(),
            ));
        }
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| BundleError::Forward("spec hidden width overflow".to_string()))?;
        let source = self.spec_hidden.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 spec hidden is not allocated".to_string())
        })?;
        let offset = hidden_row0
            .checked_mul(width)
            .ok_or_else(|| BundleError::Forward("spec hidden row offset overflow".to_string()))?;
        let len = tokens
            .len()
            .checked_mul(width)
            .ok_or_else(|| BundleError::Forward("spec hidden row overflow".to_string()))?;
        if offset
            .checked_add(len)
            .is_none_or(|end| end > source.numel())
        {
            return Err(BundleError::Forward(
                "Qwen4 spec hidden rows are outside capture".to_string(),
            ));
        }
        let hidden = source.sub_offset(offset, len);
        // The ids the trunk forward already uploaded stand in for the head's
        // own copy; no forward runs between that upload and this call.
        let trunk_ids = trunk_ids_row.and_then(|row| {
            self.execution
                .as_ref()
                .and_then(|forward| forward.uploaded_token_ids(row, tokens))
        });
        let device_ids = trunk_ids.as_ref().or(prompt_ids);
        let last = position
            .checked_add(tokens.len() - 1)
            .ok_or_else(|| BundleError::Forward("Qwen4 MTP position overflows".to_string()))?;
        mapped_mtp(self.mtp.as_mut(), gpu, last)?
            .append_rows(
                gpu,
                &self.weights,
                &self.config,
                scratch,
                tokens,
                &hidden,
                device_ids,
                position,
            )
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    /// Run one native MTP token from its committed state and copy the
    /// production logits into the caller-owned F32 destination.
    pub fn mtp_forward_token_logits(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        position: usize,
        fresh_qsa_selection: bool,
        logits: &GpuTensor,
    ) -> Result<u32, BundleError> {
        let mtp = mapped_mtp(self.mtp.as_mut(), gpu, position)?;
        mtp.forward_token_with_logits(
            gpu,
            &self.weights,
            &self.config,
            token,
            position,
            fresh_qsa_selection,
            logits,
        )
        .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_snapshot(
        &mut self,
        gpu: &mut Gpu,
    ) -> Result<MtpGpuStateSnapshot, BundleError> {
        self.mtp
            .as_mut()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .snapshot(gpu)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_restore(
        &mut self,
        gpu: &mut Gpu,
        snapshot: MtpGpuStateSnapshot,
    ) -> Result<(), BundleError> {
        self.mtp
            .as_mut()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .restore(gpu, snapshot)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }
    pub(crate) fn mtp_restore_retain(
        &mut self,
        gpu: &mut Gpu,
        snapshot: MtpGpuStateSnapshot,
    ) -> Result<(), BundleError> {
        self.mtp
            .as_mut()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .restore_retain(gpu, snapshot)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_truncate_retain(
        &mut self,
        snapshot: MtpGpuStateSnapshot,
        keep: usize,
    ) -> Result<(), BundleError> {
        let compress = self.config.indexer_compress_ratio;
        self.mtp
            .as_mut()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .truncate_retain(snapshot, keep, compress)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_validate_commit(
        &self,
        snapshot: MtpGpuStateSnapshot,
    ) -> Result<(), BundleError> {
        self.mtp
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .validate_commit(snapshot)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_commit_validated(&mut self, snapshot: MtpGpuStateSnapshot) {
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.commit_validated(snapshot);
        }
    }

    pub(crate) fn mtp_position(&self) -> Result<usize, BundleError> {
        self.mtp
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 MTP resources are not attached".to_string()))
            .map(Qwen4MtpGpu::position)
    }

    /// Run one token through the attached execution owner without exposing a
    /// second bundle owner to callers.
    pub fn forward_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_token(self, gpu, token, logits, top1)
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result
    }

    /// [`Self::forward_token`] of `token`, or (`None`) of the GPU argmax of
    /// `logits` as the previous forward left them; returns the token (see
    /// `Qwen4GpuForward::forward_token_or_argmax`).
    pub fn forward_token_or_argmax(
        &mut self,
        gpu: &mut Gpu,
        token: Option<u32>,
        logits: &GpuTensor,
    ) -> Result<u32, BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_token_or_argmax(self, gpu, token, logits)
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result
    }

    /// Run a token sequence through the shared execution owner.  The forward
    /// owner tiles requests longer than its bounded scratch capacity while
    /// preserving the public all-row logits contract.
    pub fn forward_chunk(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_chunk(self, gpu, tokens, logits, top1, None, Qwen4OutputRows::All)
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result
    }

    /// Run a prompt through the shared execution owner and retain only its
    /// final logits row.  Long prompts are tiled over bounded scratch.
    ///
    /// This is the explicit autoregressive prefill contract; `forward_chunk`
    /// remains the public all-row API for callers that need every row.
    pub fn forward_chunk_final(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_chunk(
                self,
                gpu,
                tokens,
                logits,
                top1,
                None,
                Qwen4OutputRows::Final,
            )
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result
    }

    /// Run `tokens` through the shared execution owner committing model state
    /// only: no logits, argmax or wide-hidden rows. Tiles exactly as
    /// [`Self::forward_chunk_final`], so a prefill split at a capture boundary
    /// keeps the trunk numerics of the unsplit one.
    pub(crate) fn forward_chunk_silent(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
    ) -> Result<(), BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_chunk_silent(self, gpu, tokens, None)
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result
    }

    /// Name the tokens the next prefill forward is followed by, so it warms
    /// their PLE rows while its own chunk runs. Best effort; bytes unchanged.
    pub(crate) fn set_ple_lookahead(&mut self, tokens: &[u32]) {
        if let Some(forward) = self.execution.as_mut() {
            forward.set_ple_lookahead(tokens);
        }
    }

    fn invalidate_ple_epoch(&self) -> Result<(), BundleError> {
        self.ple_rows
            .reset_epoch(PLE_RESET_TIMEOUT)
            .map(|_| ())
            .map_err(BundleError::PleRows)
    }

    /// Quiesce request-local PLE work without advancing the current epoch.
    /// Snapshot and commit preserve that epoch; reset and restore call
    /// `invalidate_ple_epoch` instead so work from the discarded state cannot
    /// publish after the state transition.
    fn quiesce_ple(&self) -> Result<(), BundleError> {
        self.ple_rows
            .quiesce(PLE_RESET_TIMEOUT)
            .map(|_| ())
            .map_err(BundleError::PleRows)?;
        self.ple_rows.resume().map_err(BundleError::PleRows)
    }

    /// Reset every owner for a cold prefill.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        if let Some(session) = self.session.as_mut() {
            session.forget_live();
        }
        self.invalidate_ple_epoch()?;
        self.state.reset(gpu).map_err(BundleError::State)?;
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.reset(gpu)
                .map_err(|error| BundleError::Forward(error.to_string()))?;
        }
        Ok(())
    }

    /// Device bytes the QSA context arenas commit now, `(target, MTP head)`:
    /// mapped pages of VMM owners, full allocations of legacy ones.
    pub fn qsa_context_committed_bytes(&self, gpu: &Gpu) -> Result<(usize, usize), BundleError> {
        let target = self
            .state
            .mapped_context_bytes(gpu)
            .map_err(BundleError::State)?;
        let mtp = match self.mtp.as_ref() {
            Some(mtp) => mtp
                .mapped_context_bytes(gpu)
                .map_err(|error| BundleError::Forward(error.to_string()))?,
            None => 0,
        };
        Ok((target, mtp))
    }

    /// Device bytes the live state may still commit before it holds
    /// `through` tokens (at most the admitted context), `(QSA context arenas
    /// of target and MTP head, gathered prefill attention workspace)`: the
    /// unmapped part of VMM arenas (0 for legacy ones, allocated whole) and
    /// the workspace's extent past what this GPU committed (0 when the route
    /// is off).
    pub fn context_growth_bytes(&self, gpu: &Gpu, through: usize) -> (u64, u64) {
        let through = through.min(self.state.max_seq_len);
        let context = self.state.context_growth_bytes(through).saturating_add(
            self.mtp
                .as_ref()
                .map_or(0, |mtp| mtp.context_growth_bytes(through)),
        );
        let gather = self
            .state
            .qsa
            .first()
            .filter(|qsa| rdna_compute::tensor_ops::qsa_gathered_wmma_enabled(gpu, qsa.format))
            .and_then(|_| {
                rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                    self.config.num_key_value_heads,
                    through,
                )
            })
            .map_or(0, |bytes| {
                (bytes as u64).saturating_sub(gpu.qsa_gather_scratch_bytes() as u64)
            });
        (context, gather)
    }

    /// Before a forward maps context past what is mapped (a VMM growth
    /// boundary, never on the steady path), have the session cache free
    /// snapshots until the growth through `end_position` fits: the cache
    /// never causes a context allocation to fail.
    pub(crate) fn release_session_for_growth(&mut self, gpu: &mut Gpu, end_position: usize) {
        if self.session.is_none() || end_position <= self.state.mapped_context_tokens() {
            return;
        }
        let (context, gather) = self.context_growth_bytes(gpu, end_position);
        if let Some(cache) = self.session.as_mut() {
            cache.release_for(gpu, context.saturating_add(gather));
        }
    }

    /// Attach the session cache (`hipfire_runtime::session_cache`): prefill
    /// then restores and captures whole-chunk snapshots through it.
    pub fn attach_session_cache(&mut self, cache: SessionCache) {
        self.session = Some(SessionDriver::new(cache));
    }

    /// The attached session cache, if any.
    pub fn session_cache(&self) -> Option<&SessionCache> {
        self.session.as_ref().map(SessionDriver::cache)
    }

    /// Drop every session snapshot, pending ones included, releasing their
    /// device buffers, and the live state. No-op without a cache.
    pub fn session_clear(&mut self, gpu: &mut Gpu) {
        if let Some(session) = self.session.as_mut() {
            session.clear(gpu);
        }
    }

    /// Run `f` with the driver taken out, so it can drive `self` as its
    /// [`SessionState`]. `None` without a cache.
    fn with_session<R>(&mut self, f: impl FnOnce(&mut SessionDriver, &mut Self) -> R) -> Option<R> {
        let mut session = self.session.take()?;
        let result = f(&mut session, self);
        self.session = Some(session);
        Some(result)
    }

    /// Start a prefill of `prompt` on `route` that skips `reused` tokens
    /// ([`SessionDriver::begin`]): a live continuation of this conversation's
    /// previous turn, a snapshot restore, or a cold start. Message-end
    /// boundaries of a live continuation are captured when active.
    pub(crate) fn session_begin(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        reused: usize,
        route: SessionRoute,
    ) -> Result<(), BundleError> {
        // A lookahead left by an aborted request names another prompt's rows.
        self.set_ple_lookahead(&[]);
        let capture = self.turn_snapshots_active();
        match self.with_session(|session, bundle| {
            session.begin(gpu, bundle, prompt, route, reused, capture)
        }) {
            // The kept state continues, so request-local PLE work from the
            // previous request must not publish into it.
            Some(Ok(true)) => self.invalidate_ple_epoch(),
            Some(Ok(false)) => Ok(()),
            Some(Err(error)) => Err(BundleError::Forward(error)),
            None if reused == 0 => self.reset(gpu),
            None => Err(BundleError::Forward(format!(
                "Qwen4 session cache is not attached; cannot reuse {reused} tokens"
            ))),
        }
    }

    /// Next absolute position the running prefill must stop at and report.
    pub(crate) fn session_next_boundary(&self) -> Option<usize> {
        self.session.as_ref()?.next_boundary()
    }

    /// Report that every owner consumed exactly `prefix` (the boundary).
    pub(crate) fn session_at_boundary(
        &mut self,
        gpu: &mut Gpu,
        prefix: &[u32],
    ) -> Result<(), BundleError> {
        self.with_session(|session, bundle| session.at_boundary(gpu, bundle, prefix))
            .unwrap_or(Ok(()))
            .map_err(BundleError::Forward)
    }

    /// AR prefill of the full canonical `prompt`, keeping only the final
    /// logits row: restore the `reused`-token snapshot (or reset), then run
    /// the rest in the cold schedule's global chunks, stopping at each
    /// session-cache boundary. A boundary inside the prompt forwards silently
    /// (state only); the final tile alone produces the logits row.
    pub fn prefill_final(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        reused: usize,
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        self.session_begin(gpu, prompt, reused, SessionRoute::Ar)?;
        let mut pos = reused;
        while let Some(boundary) = self.session_next_boundary() {
            self.set_ple_lookahead(&prompt[boundary..]);
            if boundary < prompt.len() {
                self.forward_chunk_silent(gpu, &prompt[pos..boundary])?;
            } else {
                self.forward_chunk_final(gpu, &prompt[pos..boundary], logits, None)?;
            }
            self.session_at_boundary(gpu, &prompt[..boundary])?;
            pos = boundary;
        }
        if pos < prompt.len() {
            self.forward_chunk_final(gpu, &prompt[pos..], logits, None)?;
        }
        Ok(())
    }

    /// Request-state storage formats of the resident (and every lane) state.
    pub fn state_format(&self) -> Qwen4StateFormat {
        self.state_format
    }

    /// Stage the exact AR lane store: `max_lanes` request owners, at most
    /// `row_budget` AR rows per step. Lane states are allocated on first
    /// admission into a slot.
    pub fn stage_lanes(
        &mut self,
        gpu: &mut Gpu,
        max_lanes: usize,
        row_budget: usize,
    ) -> Result<(), BundleError> {
        if self.lanes.is_some() {
            return Err(BundleError::Forward("Qwen4 lanes are already staged".to_string()));
        }
        if self.execution.is_none() {
            return Err(BundleError::Forward(
                "Qwen4 lanes need the attached forward".to_string(),
            ));
        }
        let store = crate::lane::Qwen4LaneStore::new(gpu, self, max_lanes, row_budget)
            .map_err(BundleError::Forward)?;
        self.lanes = Some(store);
        Ok(())
    }

    pub fn lanes(&self) -> Option<&crate::lane::Qwen4LaneStore> {
        self.lanes.as_ref()
    }

    pub fn lanes_mut(&mut self) -> Option<&mut crate::lane::Qwen4LaneStore> {
        self.lanes.as_mut()
    }

    /// The frozen §4.2 executor over the staged lanes.
    pub fn lane_executor(&mut self) -> Result<crate::lane_exec::Qwen4LaneExecutor<'_>, BundleError> {
        if self.lanes.is_none() {
            return Err(BundleError::Forward("Qwen4 lanes are not staged".to_string()));
        }
        Ok(crate::lane_exec::Qwen4LaneExecutor { bundle: self })
    }

    /// Run `f` on the staged store taken out of the bundle, so it may borrow
    /// the bundle (weights, formats) while the store is mutated.
    fn with_lanes<R>(
        &mut self,
        f: impl FnOnce(&mut crate::lane::Qwen4LaneStore, &mut Self) -> Result<R, String>,
    ) -> Result<R, BundleError> {
        let mut store = self
            .lanes
            .take()
            .ok_or_else(|| BundleError::Forward("Qwen4 lanes are not staged".to_string()))?;
        let result = f(&mut store, self);
        self.lanes = Some(store);
        result.map_err(BundleError::Forward)
    }

    /// Admit `epoch` with its canonical `prompt` into a free lane; returns
    /// the lane slot.
    pub fn admit_lane(
        &mut self,
        gpu: &mut Gpu,
        epoch: hipfire_runtime::slot_batch::RequestEpoch,
        prompt: Vec<u32>,
    ) -> Result<usize, BundleError> {
        self.with_lanes(|store, bundle| store.admit(gpu, bundle, epoch, prompt))
    }

    /// Free `epoch`'s lane (its device state stays allocated for reuse).
    pub fn release_lane(
        &mut self,
        epoch: &hipfire_runtime::slot_batch::RequestEpoch,
    ) -> Result<(), BundleError> {
        self.with_lanes(|store, _| store.release(epoch))
    }

    /// Mark `epoch`'s lane untrusted: it runs no further step.
    pub fn poison_lane(&mut self, epoch: &hipfire_runtime::slot_batch::RequestEpoch) {
        if let Some(store) = self.lanes.as_mut() {
            store.poison(epoch);
        }
    }

    /// Feed `token` as `epoch`'s next AR row instead of its last pick (the
    /// singleton's think-budget forced close tokens).
    pub fn force_lane_seed(
        &mut self,
        epoch: &hipfire_runtime::slot_batch::RequestEpoch,
        token: u32,
    ) -> Result<(), BundleError> {
        self.with_lanes(|store, _| store.force_seed(epoch, token))
    }

    /// A fresh lane trunk state: the resident state's formats, QSA backend
    /// and context bound, allocated through the same constructor.
    pub(crate) fn new_lane_state(&self, gpu: &mut Gpu) -> Result<Qwen4State, BundleError> {
        Qwen4State::new_with_backend(
            gpu,
            &self.config,
            self.state.max_seq_len,
            self.state_format,
            self.state.qsa_backend(),
        )
        .map_err(BundleError::State)
    }

    pub(crate) fn lane_vocab(&self) -> usize {
        self.config.vocab_size
    }

    /// The tile [`Self::prefill_final`] runs next on a cold prefill of
    /// `prompt` that has forwarded `prefilled` tokens: up to the next session
    /// boundary (the cache's cold schedule, when one is attached), tiled by
    /// the forward's `max_chunk` from the segment start.
    pub fn lane_prefill_tile_len(&self, prompt: &[u32], prefilled: usize) -> usize {
        let remaining = prompt.len().saturating_sub(prefilled);
        let Some(max_chunk) = self.spec_chunk_rows() else {
            return remaining;
        };
        // The cold schedule's segments start at 0 and at every boundary; a
        // segment is tiled by `max_chunk` from its start.
        let boundaries = if self.session.is_some() {
            SessionState::snapshot_boundaries(self, prompt, 0, prompt.len())
        } else {
            Vec::new()
        };
        let segment_start = boundaries
            .iter()
            .copied()
            .filter(|&b| b <= prefilled)
            .max()
            .unwrap_or(0);
        let segment_end = boundaries
            .iter()
            .copied()
            .find(|&b| b > prefilled)
            .unwrap_or(prompt.len())
            .min(prompt.len());
        let tile_end = segment_start
            + ((prefilled - segment_start) / max_chunk + 1).saturating_mul(max_chunk);
        tile_end.min(segment_end).saturating_sub(prefilled).min(remaining)
    }

    /// One prefill tile of a lane: the singleton chunk program
    /// (`forward_chunk_final` / `forward_chunk_silent`) on `state`. The lane
    /// state is swapped in for the call only; the session cache stays out
    /// (lanes keep none) and the forward runs in lane scope, so it never
    /// records or replays the resident state's retained body. The resident
    /// state is back in place on every return path.
    pub(crate) fn lane_prefill_tile(
        &mut self,
        gpu: &mut Gpu,
        state: &mut Qwen4State,
        tokens: &[u32],
        final_logits: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let session = self.session.take();
        std::mem::swap(&mut self.state, state);
        forward.lane_scope = true;
        // No lookahead names this lane's next rows: never adopt another
        // request's PLE ticket by accident (ids would have to match exactly,
        // which keeps bytes identical anyway, but the prefetch is wasted).
        forward.set_ple_lookahead(&[]);
        let result = match final_logits {
            Some(logits) => forward.forward_chunk(
                self,
                gpu,
                tokens,
                logits,
                None,
                None,
                Qwen4OutputRows::Final,
            ),
            None => forward.forward_chunk_silent(self, gpu, tokens, None),
        }
        .map_err(|error| BundleError::Forward(error.to_string()));
        forward.lane_scope = false;
        std::mem::swap(&mut self.state, state);
        self.session = session;
        self.execution = Some(forward);
        result
    }

    pub fn snapshot(&mut self, gpu: &mut Gpu) -> Result<Qwen4StateSnapshot, BundleError> {
        self.quiesce_ple()?;
        self.state.snapshot(gpu).map_err(BundleError::State)
    }

    pub fn restore(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), BundleError> {
        self.invalidate_ple_epoch()?;
        self.state
            .restore(gpu, snapshot)
            .map_err(BundleError::State)
    }
    pub(crate) fn restore_retain(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), BundleError> {
        self.quiesce_ple()?;
        self.state
            .restore_retain(gpu, snapshot)
            .map_err(BundleError::State)
    }

    pub(crate) fn validate_commit(&self, snapshot: Qwen4StateSnapshot) -> Result<(), BundleError> {
        self.state
            .validate_commit(snapshot)
            .map_err(BundleError::State)
    }

    pub(crate) fn commit_validated(&mut self, snapshot: Qwen4StateSnapshot) {
        self.state.commit_validated(snapshot);
    }

    pub fn commit(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), BundleError> {
        self.quiesce_ple()?;
        self.state.commit(snapshot, gpu).map_err(BundleError::State)
    }

    /// Teardown is deliberately ordered: stop/quiesce PLE reads and release
    /// leases/page-cache resources, free mutable GPU state, free finalized
    /// resident weights, then drain the attached transaction/census.
    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), BundleError> {
        let Qwen4Bundle {
            weights,
            state,
            ple_rows,
            weight_store,
            execution,
            mtp,
            spec_logits,
            spec_top1,
            spec_hidden,
            session,
            penalty_stage,
            lanes,
            ..
        } = self;
        if let Some(mut session) = session {
            session.clear(gpu);
        }
        let ple_result = ple_rows.unload().map(|_| ()).map_err(BundleError::PleRows);
        let execution_result = execution
            .map(|forward| forward.free_gpu(gpu).map_err(BundleError::Hip))
            .unwrap_or(Ok(()));
        let mtp_result = mtp
            .map(|mtp| {
                mtp.free_gpu(gpu)
                    .map_err(|error| BundleError::Forward(error.to_string()))
            })
            .unwrap_or(Ok(()));
        let mut spec_error = None;
        for tensor in [spec_logits, spec_top1, spec_hidden].into_iter().flatten() {
            if let Err(error) = gpu.free_tensor(tensor) {
                spec_error.get_or_insert(error);
            }
        }
        let spec_result = spec_error
            .map_or(Ok(()), |error| Err(BundleError::Hip(error)))
            .and(
                penalty_stage
                    .map(|stage| gpu.free_penalty_table_stage(stage).map_err(BundleError::Hip))
                    .unwrap_or(Ok(())),
            );
        // Lane states are request owners like `state`: free them first.
        let lanes_result = lanes
            .map(|store| store.free_gpu(gpu).map_err(BundleError::Forward))
            .unwrap_or(Ok(()));
        let state_result = lanes_result.and(state.free_gpu(gpu).map_err(BundleError::State));
        let weight_result = weights.free_gpu(gpu).map_err(BundleError::Hip);
        let store_result = weight_store.drain(gpu).map_err(BundleError::Hip);
        first_bundle_error([
            ple_result,
            execution_result,
            mtp_result,
            spec_result,
            state_result,
            weight_result,
            store_result,
        ])
    }
}

impl hipfire_runtime::arch_model::ArchModel for Qwen4Bundle {
    fn dim(&self) -> usize {
        self.config.hidden_size
    }

    fn n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "qwen4"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut hipfire_runtime::llama::KvCache> {
        None
    }

    fn reset_session_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.reset(gpu).map_err(|error| error.to_string())
    }

    fn session_cache_attached(&self) -> bool {
        self.session.is_some()
    }

    fn session_plan(&self, prompt: &[u32], route: SessionRoute) -> usize {
        self.session
            .as_ref()
            .map_or(0, |session| session.plan(self, prompt, route))
    }

    fn session_commit(&mut self) {
        if let Some(session) = self.session.as_mut() {
            session.commit();
        }
    }

    /// Publish as [`Self::session_commit`] and keep the live state when
    /// `consumed` (the host history the client committed) extends this turn's
    /// prompt and the owners sit exactly at its end. The next turn of the
    /// conversation then prefills only the suffix in place (session-exact,
    /// decode lineage); the snapshots just published stay cold-exact because a
    /// live turn captures none unless message-end snapshots are active.
    fn session_commit_live(&mut self, consumed: &[u32]) {
        if let Some(mut session) = self.session.take() {
            session.commit_live(self, consumed);
            self.session = Some(session);
        }
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        if let Err(error) = Qwen4Bundle::free_gpu(*self, gpu) {
            eprintln!("Qwen4 bundle teardown failed: {error}");
        }
    }
}

const PLE_SHARD_NAME_PREFIX: &str =
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";

fn ple_shard_index(name: &str) -> Result<usize, WeightError> {
    let suffix = name
        .strip_prefix(PLE_SHARD_NAME_PREFIX)
        .and_then(|name| name.strip_suffix(".weight"))
        .ok_or_else(|| {
            WeightError::DescriptorMismatch(format!("invalid PLE shard name '{name}'"))
        })?;
    if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(WeightError::DescriptorMismatch(format!(
            "invalid PLE shard name '{name}'"
        )));
    }
    let index = suffix
        .parse::<usize>()
        .map_err(|_| WeightError::DescriptorMismatch(format!("invalid PLE shard name '{name}'")))?;
    if index >= PLE_SHARD_COUNT || index.to_string() != suffix {
        return Err(WeightError::DescriptorMismatch(format!(
            "PLE shard index {index} is outside canonical range in '{name}'"
        )));
    }
    Ok(index)
}

fn ordered_ple_entries<'a>(
    manifest: &'a Qwen4Manifest,
) -> Result<Vec<&'a WeightEntry>, WeightError> {
    let entries = manifest.external_entries().collect::<Vec<_>>();
    if entries.len() != PLE_SHARD_COUNT {
        return Err(WeightError::PleShardCount {
            expected: PLE_SHARD_COUNT,
            actual: entries.len(),
        });
    }
    let mut ordered = vec![None; PLE_SHARD_COUNT];
    for entry in entries {
        let index = ple_shard_index(&entry.name)?;
        if entry.layer != Some(1)
            || entry.logical_shape.as_slice() != [PLE_SHARD_ROWS, PLE_ROW_WIDTH]
        {
            return Err(WeightError::DescriptorMismatch(entry.name.clone()));
        }
        if ordered[index].replace(entry).is_some() {
            return Err(WeightError::DescriptorMismatch(format!(
                "duplicate PLE shard index {index}"
            )));
        }
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, entry)| {
            entry.ok_or_else(|| {
                WeightError::DescriptorMismatch(format!("missing PLE shard index {index}"))
            })
        })
        .collect()
}

fn validate_ple_metadata(metadata: &PleHashMetadata) -> Result<(), BundleError> {
    let physical_rows = (PLE_SHARD_ROWS as u64)
        .checked_mul(PLE_SHARD_COUNT as u64)
        .ok_or_else(|| {
            BundleError::Weights(WeightError::DescriptorMismatch(
                "PLE physical row count overflow".to_string(),
            ))
        })?;
    if metadata.padded_rows() != physical_rows {
        return Err(BundleError::Weights(WeightError::DescriptorMismatch(
            format!(
                "PLE metadata padded rows {} do not match physical rows {physical_rows}",
                metadata.padded_rows()
            ),
        )));
    }
    let valid_rows = (0..PLE_SHARD_COUNT).try_fold(0u64, |sum, shard| {
        sum.checked_add(ple_valid_rows_for_shard(shard) as u64)
            .ok_or_else(|| {
                BundleError::Weights(WeightError::DescriptorMismatch(
                    "PLE valid row count overflow".to_string(),
                ))
            })
    })?;
    if metadata.valid_rows() != valid_rows {
        return Err(BundleError::Weights(WeightError::DescriptorMismatch(
            format!(
                "PLE metadata valid rows {} do not match canonical rows {valid_rows}",
                metadata.valid_rows()
            ),
        )));
    }
    Ok(())
}
fn ple_descriptors(
    transaction: &WeightLoadTransaction,
    manifest: &Qwen4Manifest,
    metadata: &PleHashMetadata,
) -> Result<Vec<SourceRangeDescriptor>, BundleError> {
    validate_ple_metadata(metadata)?;
    let entries = ordered_ple_entries(manifest).map_err(BundleError::Weights)?;
    let mut descriptors: Vec<SourceRangeDescriptor> = Vec::with_capacity(entries.len());
    let mut valid_rows_total = 0u64;
    for (index, entry) in entries.into_iter().enumerate() {
        let descriptor = transaction
            .external_descriptor(&entry.name, entry.layer, 0)
            .ok_or_else(|| {
                BundleError::Weights(WeightError::MissingExternalDescriptor {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                })
            })?
            .clone();
        let (row_bytes, valid_rows) = match entry.residency {
            WeightResidency::ExternalRows {
                row_bytes,
                valid_rows,
            } => (row_bytes, valid_rows),
            WeightResidency::Resident | WeightResidency::HostMapped => {
                return Err(BundleError::Weights(WeightError::DescriptorMismatch(
                    entry.name.clone(),
                )))
            }
        };
        // The manifest's stride is the tier it was declared for; the descriptor
        // side (length for its own declared dtype, shape, valid rows, HFQ index
        // seal, file bounds) was already checked when the transaction was
        // fulfilled. Here the declaration only has to name a tier the reader
        // decodes, so a sealed BF16-PLE artifact stays loadable next to a
        // Q8F16 one.
        if ![
            RowEncoding::Bf16.encoded_row_bytes(PLE_ROW_WIDTH),
            RowEncoding::Q8F16.encoded_row_bytes(PLE_ROW_WIDTH),
        ]
        .contains(&row_bytes)
            || valid_rows != ple_valid_rows_for_shard(index)
        {
            return Err(BundleError::Weights(WeightError::DescriptorMismatch(
                entry.name.clone(),
            )));
        }
        valid_rows_total = valid_rows_total
            .checked_add(valid_rows as u64)
            .ok_or_else(|| {
                BundleError::Weights(WeightError::DescriptorMismatch(
                    "PLE valid row count overflow".to_string(),
                ))
            })?;
        if let Some(first) = descriptors.first() {
            if first.source_identity() != descriptor.source_identity() {
                return Err(BundleError::Weights(WeightError::DescriptorMismatch(
                    format!("PLE shard {index} source identity differs"),
                )));
            }
        }
        // Fulfillment only enforces the index seal for HFQ sources; PLE rows
        // must come from one.
        if descriptor.source_identity().format != SourceFormat::Hfq {
            return Err(BundleError::Weights(WeightError::DescriptorMismatch(
                entry.name.clone(),
            )));
        }
        descriptors.push(descriptor);
    }
    if valid_rows_total != metadata.valid_rows() {
        return Err(BundleError::Weights(WeightError::DescriptorMismatch(
            format!(
                "PLE metadata valid rows {} do not match manifest rows {valid_rows_total}",
                metadata.valid_rows()
            ),
        )));
    }
    Ok(descriptors)
}

/// The attached MTP head with its QSA context mapped through `position`,
/// the row the next step writes. Mapping happens here, outside any capture
/// or record (MTP steps never run under either), never inside the step.
fn mapped_mtp<'a>(
    mtp: Option<&'a mut Qwen4MtpGpu>,
    gpu: &mut Gpu,
    position: usize,
) -> Result<&'a mut Qwen4MtpGpu, BundleError> {
    let mtp = mtp
        .ok_or_else(|| BundleError::Forward("Qwen4 MTP resources are not attached".to_string()))?;
    let required = position
        .checked_add(1)
        .ok_or_else(|| BundleError::Forward("Qwen4 MTP position overflows".to_string()))?;
    mtp.ensure_mapped_capacity(gpu, required)
        .map_err(|error| BundleError::Forward(error.to_string()))?;
    Ok(mtp)
}

fn cleanup_transaction(primary: BundleError, rollback: hip_bridge::HipResult<()>) -> BundleError {
    match rollback {
        Ok(()) => primary,
        Err(error) => BundleError::Rollback {
            cause: primary.to_string(),
            error,
        },
    }
}

fn cleanup_bundle_failure(
    primary: BundleError,
    weights: hip_bridge::HipResult<()>,
    transaction: hip_bridge::HipResult<()>,
) -> BundleError {
    match (weights, transaction) {
        (Ok(()), Ok(())) => primary,
        (Err(weight), Ok(())) => BundleError::Rollback {
            cause: format!("{primary}; resident weight cleanup failed"),
            error: weight,
        },
        (Ok(()), Err(transaction)) => BundleError::Rollback {
            cause: format!("{primary}; transaction cleanup failed"),
            error: transaction,
        },
        (Err(weight), Err(transaction)) => BundleError::Rollback {
            cause: format!("{primary}; resident and transaction cleanup failed: {transaction}"),
            error: weight,
        },
    }
}

fn first_bundle_error(results: [Result<(), BundleError>; 7]) -> Result<(), BundleError> {
    let mut first = None;
    for result in results {
        if let Err(error) = result {
            if first.is_none() {
                first = Some(error);
            }
        }
    }
    first.map_or(Ok(()), Err)
}

#[derive(Debug)]
pub enum BundleError {
    Config(String),
    Weights(WeightError),
    State(StateError),
    PleRows(RowStoreError),
    Forward(String),
    Hip(hip_bridge::HipError),
    Rollback {
        cause: String,
        error: hip_bridge::HipError,
    },
    Transaction(WeightStoreError),
}

impl From<WeightError> for BundleError {
    fn from(value: WeightError) -> Self {
        Self::Weights(value)
    }
}

impl From<WeightStoreError> for BundleError {
    fn from(value: WeightStoreError) -> Self {
        Self::Transaction(value)
    }
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "Qwen4 bundle config: {message}"),
            Self::Weights(error) => write!(f, "Qwen4 bundle weights: {error}"),
            Self::State(error) => write!(f, "Qwen4 bundle state: {error}"),
            Self::Forward(error) => write!(f, "Qwen4 bundle forward: {error}"),
            Self::PleRows(error) => write!(f, "Qwen4 bundle PLE rows: {error}"),
            Self::Hip(error) => write!(f, "Qwen4 bundle HIP teardown: {error}"),
            Self::Rollback { cause, error } => {
                write!(f, "Qwen4 bundle cleanup after {cause}: {error}")
            }
            Self::Transaction(error) => write!(f, "Qwen4 bundle transaction: {error}"),
        }
    }
}

impl std::error::Error for BundleError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ple::PLE_HEAD_COUNT;

    #[test]
    fn rejects_metadata_with_same_padding_but_different_valid_rows() {
        let canonical = PleHashMetadata::qwen4();
        let mut sizes = *canonical.head_vocab_sizes();
        sizes[PLE_HEAD_COUNT - 1] -= 1;
        let mut offsets = [0u64; PLE_HEAD_COUNT];
        for head in 1..PLE_HEAD_COUNT {
            offsets[head] = offsets[head - 1] + sizes[head - 1];
        }
        let metadata = PleHashMetadata::from_stored(
            *canonical.multipliers(),
            sizes,
            offsets,
            canonical.padded_rows(),
        )
        .expect("one valid row removed still rounds to the same physical padding");

        let error = validate_ple_metadata(&metadata).unwrap_err();
        assert!(matches!(
            error,
            BundleError::Weights(WeightError::DescriptorMismatch(message))
                if message.contains("valid rows")
        ));
    }

    const AR: SessionRoute = SessionRoute::Ar;
    const MTP: SessionRoute = SessionRoute::Mtp;

    #[test]
    fn live_position_checks_the_head_and_row_capture() {
        // AR continues at the target and ignores the head.
        assert_eq!(live_position(AR, 3, None, false), Some(3));
        assert_eq!(live_position(AR, 3, Some(1), false), Some(3));
        // MTP needs the head exactly at the target (floor retirement leaves
        // it behind, a missing head cannot continue).
        assert_eq!(live_position(MTP, 3, Some(3), false), Some(3));
        assert_eq!(live_position(MTP, 3, Some(2), false), None);
        assert_eq!(live_position(MTP, 3, Some(4), false), None);
        assert_eq!(live_position(MTP, 3, None, false), None);
        // A verify row capture in flight never continues.
        assert_eq!(live_position(AR, 3, None, true), None);
        assert_eq!(live_position(MTP, 3, Some(3), true), None);
    }
}
