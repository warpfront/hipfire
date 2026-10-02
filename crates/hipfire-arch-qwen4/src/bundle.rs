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
use crate::mtp_gpu::{MtpGpuStateSnapshot, MtpStep, Qwen4MtpGpu};
use crate::ple::PleHashMetadata;
use crate::state::{Qwen4State, Qwen4StateSnapshot, StateError};
use crate::weights::{
    ple_valid_rows_for_shard, Qwen4Manifest, Qwen4Placement, Qwen4Weights, WeightError,
    PLE_ROW_WIDTH, PLE_SHARD_COUNT, PLE_SHARD_ROWS,
};
use hipfire_runtime::external_rows::{RowEncoding, RowStore, RowStoreError};
use hipfire_runtime::model_source::{SourceFormat, SourceRangeDescriptor};
use hipfire_runtime::weight_manifest::{WeightEntry, WeightResidency};
use hipfire_runtime::weight_store::{WeightLoadTransaction, WeightStoreError};
use crate::state::Qwen4StateFormat;
use rdna_compute::{Gpu, GpuTensor};
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
}

impl Qwen4Bundle {
    /// Assemble a complete Single bundle using metadata parsed from the
    /// artifact's canonical `qwen4_ple` object. `state_format` is the request
    /// state's storage ([`crate::resolve_state_format`]).
    pub fn assemble(
        config: Qwen4Config,
        transaction: WeightLoadTransaction,
        placements: &[Qwen4Placement],
        gpu: &mut Gpu,
        max_seq_len: usize,
        metadata: PleHashMetadata,
        state_format: Qwen4StateFormat,
    ) -> Result<Self, BundleError> {
        Self::assemble_with_metadata(
            config,
            transaction,
            placements,
            gpu,
            max_seq_len,
            metadata,
            state_format,
        )
    }

    /// Assemble with validated metadata read from the artifact's exact I64
    /// arrays.  The transaction is consumed so no load-side owner can remain
    /// live after this method publishes the bundle.
    pub fn assemble_with_metadata(
        config: Qwen4Config,
        mut transaction: WeightLoadTransaction,
        placements: &[Qwen4Placement],
        gpu: &mut Gpu,
        max_seq_len: usize,
        metadata: PleHashMetadata,
        state_format: Qwen4StateFormat,
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
        let mut state = match Qwen4State::new(gpu, &config, max_seq_len, state_format) {
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
        // A chunk's PLE prefetch may exceed one row-store staging buffer:
        // RowFetch splits it into consecutive tickets, so the chunk is not
        // capped by the staging capacity.
        let requested = qwen4_prefill_chunk_requested(&gpu.arch, max_chunk);
        // The gathered QSA prefill attention (HIPFIRE_QWEN4_QSA_WMMA_GATHER)
        // converts the cache rows it reads into its own scratch: reserve it
        // for the whole context now, before any capture or record, so a
        // later longer prefill never grows (and frees) it. A no-op when the
        // route is off for this arch and state format.
        if let Some(qsa) = self.state.qsa.first() {
            let reserved = rdna_compute::tensor_ops::reserve_qsa_gathered_wmma_scratch(
                gpu,
                qsa.format,
                self.config.num_key_value_heads,
                qsa.full_capacity,
            )
            .map_err(BundleError::Hip)?;
            if reserved > 0 {
                eprintln!(
                    "  qwen4 QSA gather scratch: {} MiB reserved for {} context tokens",
                    reserved >> 20,
                    qsa.full_capacity
                );
            }
        }
        self.weights
            .requant_from_env(gpu)
            .map_err(BundleError::Forward)?;
        // The largest rung whose chunk-sized resources fit the free device
        // memory beside the kernels' lazily sized workspaces; the smallest
        // rung is attempted regardless and fails at allocation if it must.
        let (free, _) = gpu.hip.get_vram_info().map_err(BundleError::Hip)?;
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
        let mtp = Qwen4MtpGpu::new(gpu, &self.weights, &self.config, max_seq)
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

    /// Start OS readahead of the PLE rows of `tokens[skip..]`, where `tokens`
    /// continue from the current position and `tokens[..skip]` is the chunk
    /// about to run: the next chunk then finds its rows in the page cache
    /// instead of stalling on cold reads (one chunk ahead, so the hints do not
    /// queue ahead of the running chunk's own reads).
    pub(crate) fn ple_readahead(&self, tokens: &[u32], skip: usize) {
        let mut ids = self.state.ple_history.row_ids(&self.ple_metadata, tokens);
        ids.drain(..(skip * crate::ple::PLE_HEAD_COUNT).min(ids.len()));
        self.ple_rows.readahead(ids);
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
    pub(crate) fn spec_chunk_rows(&self) -> Option<usize> {
        self.execution
            .as_ref()
            .map(|forward| forward.scratch.max_chunk)
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

    pub(crate) fn spec_forward_rows_with_output(
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
        gpu.copy_d2d(&source, destination, destination.byte_size())
            .map_err(BundleError::Hip)
    }

    /// Host copy of the first `rows` logit rows the last spec forward wrote.
    pub(crate) fn spec_logits_host(&self, gpu: &Gpu, rows: usize) -> Result<Vec<f32>, BundleError> {
        let spec_logits = self.spec_logits.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 spec logits are not attached".to_string())
        })?;
        let capacity = spec_logits.numel() / self.config.vocab_size;
        if rows > capacity {
            return Err(BundleError::Forward(format!(
                "Qwen4 spec logits hold at most {capacity} rows, {rows} requested"
            )));
        }
        let logits = spec_logits.sub_offset(0, rows * self.config.vocab_size);
        gpu.download_f32(&logits).map_err(BundleError::Hip)
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

    pub(crate) fn mtp_forward_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
        fresh_qsa_selection: bool,
    ) -> Result<u32, BundleError> {
        let mtp = self.mtp.as_mut().ok_or_else(|| {
            BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
        })?;
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

    pub(crate) fn mtp_advance_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
        fresh_qsa_selection: bool,
    ) -> Result<(), BundleError> {
        self.mtp
            .as_mut()
            .ok_or_else(|| BundleError::Forward("Qwen4 MTP resources are not attached".into()))?
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
        self.mtp
            .as_mut()
            .ok_or_else(|| BundleError::Forward("Qwen4 MTP resources are not attached".into()))?
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

    /// [`Self::mtp_append_token`] for `tokens` at `position..`, whose backbone
    /// hidden rows are the spec-hidden capture rows `first_row..` of the last
    /// chunked prefill, in one batched MTP step (at most
    /// [`crate::mtp_gpu::MTP_APPEND_ROWS`] tokens).
    pub(crate) fn mtp_append_rows(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        first_row: usize,
        position: usize,
    ) -> Result<(), BundleError> {
        let width = self.config.hc_count * self.config.hidden_size;
        let source = self.spec_hidden.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 spec hidden is not allocated".to_string())
        })?;
        let (offset, len) = (first_row * width, tokens.len() * width);
        if offset + len > source.numel() {
            return Err(BundleError::Forward(
                "Qwen4 spec hidden rows are outside capture".to_string(),
            ));
        }
        let hidden = source.sub_offset(offset, len);
        self.mtp
            .as_mut()
            .ok_or_else(|| BundleError::Forward("Qwen4 MTP resources are not attached".into()))?
            .append_rows(gpu, &self.weights, &self.config, tokens, &hidden, position)
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
        let mtp = self.mtp.as_mut().ok_or_else(|| {
            BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
        })?;
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

    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        self.invalidate_ple_epoch()?;
        self.state.reset(gpu).map_err(BundleError::State)?;
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.reset(gpu)
                .map_err(|error| BundleError::Forward(error.to_string()))?;
        }
        Ok(())
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
            ..
        } = self;
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
        let spec_result = spec_error.map_or(Ok(()), |error| Err(BundleError::Hip(error)));
        let state_result = state.free_gpu(gpu).map_err(BundleError::State);
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
}
